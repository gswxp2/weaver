
import ctypes
from typing import List, Optional, Union
from torch.distributed import ProcessGroup
import torch.distributed as dist
import torch
import cupy as cp
from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary
from vllm.distributed.parallel_state import get_role
from vllm.distributed.communication_op import sr_tensor_dict, sr_obj
import time

def ptr_to_tensor(device_ptr: int, nbytes: int, shape: tuple, cp_dtype=cp.int32) -> torch.Tensor:
    mem = cp.cuda.UnownedMemory(device_ptr, nbytes, owner=None)
    memptr = cp.cuda.MemoryPointer(mem, offset=0)
    arr = cp.ndarray(shape, dtype=cp_dtype, memptr=memptr)
    return torch.as_tensor(arr, device="cuda")
        
def create_attn_buffer(
        size_in_bytes: int,
        group: Optional[ProcessGroup] = None) -> List[int]:
    """
    Creates a shared buffer and returns a list of pointers
    representing the buffer on all processes in the group.
    """
    lib = CudaRTLibrary()
    pointer = lib.cudaMalloc(size_in_bytes)
    handle = lib.cudaIpcGetMemHandle(pointer)
    world_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)
    handles = [None] * world_size
    dist.all_gather_object(handles, handle, group=group)

    pointers: List[int] = []
    for i, h in enumerate(handles):
        if i == rank:
            pointers.append(pointer.value)  # type: ignore
        else:
            pointers.append(
                lib.cudaIpcOpenMemHandle(h).value)  # type: ignore
    tensors = []
    for i in pointers:
        import cupy as cp
        
        tensor = ptr_to_tensor(i, size_in_bytes, (2,))
        tensors.append(tensor)
    # create a send recv tunnel
    # QKV tunnel
    qkv_bytes = 200 * (4096+1024+1024) * 2
    if get_role() == "RECEIVER":
        pointer = lib.cudaMalloc(qkv_bytes)
        tensor = torch.randn(200, (4096+1024+1024), device="cuda",dtype=torch.bfloat16)
        pointer = tensor.data_ptr()
        handle2 = lib.cudaIpcGetMemHandle(pointer)
        sr_obj(handle2, src=1, dst=0)
    if get_role() == "SENDER":
        handle2 = sr_obj(None, src=1, dst=0)
        pointer = lib.cudaIpcOpenMemHandle(handle2).value
        tensor = ptr_to_tensor(pointer, qkv_bytes, (200,(4096+1024*2)), cp_dtype=cp.float16).view(torch.bfloat16)
    else:
        pass
    tensors.append(tensor)
    return tensors