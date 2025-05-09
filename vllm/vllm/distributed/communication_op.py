from typing import Any, Dict, Optional, Union

import torch
import torch.distributed

from .parallel_state import get_tp_group,get_dis_group


def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    return get_tp_group().all_reduce(input_)


def tensor_model_parallel_all_gather(input_: torch.Tensor,
                                     dim: int = -1) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    return get_tp_group().all_gather(input_, dim)


def tensor_model_parallel_gather(input_: torch.Tensor,
                                 dst: int = 0,
                                 dim: int = -1) -> Optional[torch.Tensor]:
    """Gather the input tensor across model parallel group."""
    return get_tp_group().gather(input_, dst, dim)


def broadcast_tensor_dict(tensor_dict: Optional[Dict[Any, Union[torch.Tensor,
                                                                Any]]] = None,
                          src: int = 0):
    if not torch.distributed.is_initialized():
        return tensor_dict
    return get_tp_group().broadcast_tensor_dict(tensor_dict, src)


def sr_tensor(
    tensor: torch.Tensor, sender: int, receiver: int, stream=None
) -> torch.Tensor:
    if stream is None:
        stream = torch.cuda.current_stream()
    """Send a tensor to destination and receive a tensor from source."""
    if not torch.distributed.is_initialized():
        return tensor
    self_rank = torch.distributed.get_rank()
    if self_rank == sender:
        return get_dis_group().send(tensor, receiver, stream=stream)
    elif self_rank == receiver:
        return get_dis_group().recv(
            None, None, sender, dst_tensor=tensor, stream=stream
        )
    else:
        raise ValueError(f"Rank {self_rank} is neither sender nor receiver.")


def sr_tensor_dict(
    tensor_dict: Optional[Dict[Any, Union[torch.Tensor, Any]]] = None,
    src=0,
    dst=1,
    stream=None,
):
    if stream is None:
        stream = torch.cuda.current_stream()
    self_rank = torch.distributed.get_rank()
    if self_rank == src:
        return get_dis_group().send_tensor_dict(tensor_dict, dst, stream=stream)
    else :
        assert self_rank == dst
        return get_dis_group().recv_tensor_dict(src, stream=stream)


def sr_obj_async(obj, src, dst):
    self_rank = torch.distributed.get_rank()
    if self_rank == src:
        return get_dis_group().send_object(obj, dst)
    elif self_rank == dst:
        return get_dis_group().recv_object_async(src)
    else:
        raise ValueError(f"Rank {self_rank} is neither sender nor receiver.")
    
def sr_obj(obj, src, dst):
    self_rank = torch.distributed.get_rank()
    if self_rank == src:
        return get_dis_group().send_object(obj, dst)
    elif self_rank == dst:
        return get_dis_group().recv_object(src)
    else:
        raise ValueError(f"Rank {self_rank} is neither sender nor receiver.")

