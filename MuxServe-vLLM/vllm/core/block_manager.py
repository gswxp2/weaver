"""A block manager that manages token blocks."""
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from vllm.block import PhysicalTokenBlock
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus
from vllm.utils import Device
from vllm.zmq_tool import ZMQClient


class BlockAllocator:
    """Manages free physical token blocks for a device.

    The allocator maintains a list of free blocks and allocates a block when
    requested. When a block is freed, its reference count is decremented. If
    the reference count becomes zero, the block is added back to the free list.
    """

    def __init__(
        self,
        device: Device,
        block_size: int,
        num_blocks: int,
        group_size: int,
    ) -> None:
        self.device = device
        self.block_size = block_size

        # Initialize the free blocks.
        self.free_blocks: Dict[int, PhysicalTokenBlock] = {}
        for i in range(0, num_blocks, group_size):
            block = PhysicalTokenBlock(device=device,
                                       block_number=i,
                                       block_size=block_size)
            self.free_blocks[i] = block

        self.num_blocks = len(self.free_blocks)

    def get_token_block(self, block_number: int):
        block = self.free_blocks.pop(block_number)
        block.ref_count += 1
        return block

    def allocate(self) -> PhysicalTokenBlock:
        if not self.free_blocks:
            raise ValueError("Out of memory! No free blocks are available.")
        _, block = self.free_blocks.popitem()
        block.ref_count = 1
        return block

    def free(self, block: PhysicalTokenBlock) -> None:
        if block.ref_count == 0:
            raise ValueError(f"Double free! {block} is already freed.")
        block.ref_count -= 1
        if block.ref_count == 0:
            self.free_blocks[block.block_number] = block

    def get_num_free_blocks(self) -> int:
        return len(self.free_blocks)


# Mapping: logical block number -> physical block.
BlockTable = List[PhysicalTokenBlock]


class BlockSpaceManager:
    """Manages the mapping between logical and physical token blocks."""

    def __init__(
        self,
        model_name: str,
        num_layers: int,
        num_heads: int,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        watermark: float = 0.02,
        sliding_window: Optional[int] = None,
        tcp_client: Optional[ZMQClient] = None,
    ) -> None:
        self.model_name = model_name
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.block_size = block_size
        self.num_total_gpu_blocks = num_gpu_blocks
        self.num_total_cpu_blocks = num_cpu_blocks
        self.tcp_client = tcp_client

        self.block_sliding_window = None
        if sliding_window is not None:
            assert sliding_window % block_size == 0, (sliding_window,
                                                      block_size)
            self.block_sliding_window = sliding_window // block_size

        self.watermark = watermark
        assert watermark >= 0.0

        if tcp_client is not None:
            group_size = self.num_heads * self.num_layers
        else:
            group_size = 1
        self.watermark_blocks = int(watermark * num_gpu_blocks)
        self.gpu_allocator = BlockAllocator(Device.GPU, block_size,
                                            num_gpu_blocks, group_size)
        self.cpu_allocator = BlockAllocator(Device.CPU, block_size,
                                            num_cpu_blocks, group_size)
        # Mapping: seq_id -> BlockTable.
        self.block_tables: Dict[int, BlockTable] = {}
        self.layer_block_tables: Dict[int, List[BlockTable]] = {}

        # construct layerwise block table
        self.layerwise_table_np_cache: Dict[int, Tuple(int, np.array)] = {}
        self.layer_offsets = np.arange(self.num_layers, dtype=np.int32).reshape(-1, 1)

    def can_allocate(self, seq_group: SequenceGroup) -> bool:
        # FIXME(woosuk): Here we assume that all sequences in the group share
        # the same prompt. This may not be true for preempted sequences.
        seq = seq_group.get_seqs()[0]
        num_required_blocks = len(seq.logical_token_blocks)
        if self.block_sliding_window is not None:
            num_required_blocks = min(num_required_blocks,
                                      self.block_sliding_window)
        num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()
        # Use watermark to avoid frequent cache eviction.
        return (num_free_gpu_blocks - num_required_blocks >=
                self.watermark_blocks)

    def allocate(self, seq_group: SequenceGroup) -> None:
        # NOTE: Here we assume that all sequences in the group have the same
        # prompt.
        seq = seq_group.get_seqs()[0]

        # Allocate new physical token blocks that will store the prompt tokens.
        block_table: BlockTable = []
        for logical_idx in range(len(seq.logical_token_blocks)):
            if (self.block_sliding_window is not None
                    and logical_idx >= self.block_sliding_window):
                block = block_table[logical_idx % self.block_sliding_window]
            else:
                block = self.gpu_allocator.allocate()
            # Set the reference counts of the token blocks.
            block.ref_count = seq_group.num_seqs()
            block_table.append(block)

        # Assign the block table for each sequence.
        for seq in seq_group.get_seqs():
            self.block_tables[seq.seq_id] = block_table.copy()

    def layerwise_allocate(self, seq_group: SequenceGroup) -> int:
        """
        Return:
            -1: fail
            0: success
            1: find block table
        """
        seq = seq_group.get_seqs()[0]

        # first, check if the prompt is already cached
        request_id = seq_group.request_id
        rank = 0 # assume tensor-model parallelism only
        num_gpu_blocks = len(seq.logical_token_blocks)
        # try to load blocktable
        self.tcp_client.send_pyobj(("blocktable_load", [request_id]))
        self.layerwise_table_np_cache[seq.seq_id] = (-1, None)
        data = self.tcp_client.recv_pyobj()
        if data is not None:
            # set the block table for sequence
            layerwise_block_tables: List[BlockTable] = []
            for token_block_number in data:
                block = self.gpu_allocator.get_token_block(token_block_number)
                layerwise_block_tables.append(block)
            self.layer_block_tables[seq.seq_id] = layerwise_block_tables
            for seq in seq_group.get_seqs():
                seq.in_prefill = False
            return 1
        assert False
        print("want to allocate", num_gpu_blocks)
        print("want to allocate", num_gpu_blocks)
        # try to alloc block table
        self.tcp_client.send_pyobj(
            ("cache_alloc", [request_id, rank, self.model_name, num_gpu_blocks])
        )
        free_block_ids = self.tcp_client.recv_pyobj()

        if free_block_ids is None:
            return -1

        # Allocate new physical token blocks that will store the prompt tokens.
        layerwise_block_tables: List[BlockTable] = []
        for i, block_idx in enumerate(free_block_ids):
            block = self.gpu_allocator.get_token_block(block_idx)
            # Set the reference counts of the token blocks.
            block.ref_count = seq_group.num_seqs()
            layerwise_block_tables.append(block)

        # Assign the block table for each sequence.
        for seq in seq_group.get_seqs():
            self.layer_block_tables[seq.seq_id] = layerwise_block_tables.copy()
        return 0

    def can_append_slot(self, seq_group: SequenceGroup) -> bool:
        # Simple heuristic: If there is at least one free block
        # for each sequence, we can append.
        num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()
        num_seqs = seq_group.num_seqs(status=SequenceStatus.RUNNING)
        return num_seqs <= num_free_gpu_blocks

    def append_slot(self, seq: Sequence) -> Optional[Tuple[int, int]]:
        """Allocate a physical slot for a new token."""
        logical_blocks = seq.logical_token_blocks
        block_table = self.block_tables[seq.seq_id]

        if len(block_table) < len(logical_blocks):
            if (self.block_sliding_window
                    and len(block_table) >= self.block_sliding_window):
                # re-use a block
                block_table.append(block_table[len(block_table) %
                                               self.block_sliding_window])
            else:
                # The sequence has a new logical block.
                # Allocate a new physical block.
                block = self.gpu_allocator.allocate()
                block_table.append(block)
                return None

        # We want to append the token to the last physical block.
        last_block = block_table[-1]
        assert last_block.device == Device.GPU
        if last_block.ref_count == 1:
            # Not shared with other sequences. Appendable.
            return None
        else:
            # The last block is shared with other sequences.
            # Copy on Write: Allocate a new block and copy the tokens.
            new_block = self.gpu_allocator.allocate()
            block_table[-1] = new_block
            self.gpu_allocator.free(last_block)
            return last_block.block_number, new_block.block_number

    def layerwise_append_slot(self, seq: Sequence) -> int:
        logical_blocks = seq.logical_token_blocks
        layerwise_block_tables = self.layer_block_tables[seq.seq_id]

        # set to not in_prefill
        seq.in_prefill = False

        if len(layerwise_block_tables) < len(logical_blocks):
            assert False
            print("want to allocate", num_gpu_blocks)
            print("want to allocate", num_gpu_blocks)
            # FIXME: disable sliding window now
            # allocate a new block table for each layer
            request_id = seq.seq_id
            rank = 0 # assume tensor-model parallelism only
            num_gpu_blocks = 1
            self.tcp_client.send_pyobj(
                ("cache_alloc", [request_id, rank, self.model_name, num_gpu_blocks])
            )
            free_block_ids = self.tcp_client.recv_pyobj()
            if free_block_ids is None:
                return -1

            block = self.gpu_allocator.get_token_block(free_block_ids[0])
            layerwise_block_tables.append(block)

        # We want to append the token to the last physical block.
        last_block = layerwise_block_tables[-1]
        assert last_block.device == Device.GPU
        if last_block.ref_count == 1:
            # Not shared with other sequences. Appendable.
            return 1
        else:
            # FIXME: we disable beam_seam now
            raise ValueError("beam search is not supported now")

    def fork(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        # NOTE: fork does not allocate a new physical block.
        # Thus, it is always safe from OOM.
        src_block_table = self.block_tables[parent_seq.seq_id]
        self.block_tables[child_seq.seq_id] = src_block_table.copy()
        for block in src_block_table:
            block.ref_count += 1

    def _get_physical_blocks(
            self, seq_group: SequenceGroup) -> List[PhysicalTokenBlock]:
        # NOTE: Here, we assume that the physical blocks are only shared by
        # the sequences in the same group.
        blocks: Set[PhysicalTokenBlock] = set()
        for seq in seq_group.get_seqs():
            if seq.is_finished():
                continue
            blocks.update(self.block_tables[seq.seq_id])
        return list(blocks)

    def can_swap_in(self, seq_group: SequenceGroup) -> bool:
        blocks = self._get_physical_blocks(seq_group)
        num_swapped_seqs = seq_group.num_seqs(status=SequenceStatus.SWAPPED)
        num_free_blocks = self.gpu_allocator.get_num_free_blocks()
        # NOTE: Conservatively, we assume that every sequence will allocate
        # at least one free block right after the swap-in.
        # NOTE: This should match the logic in can_append_slot().
        num_required_blocks = len(blocks) + num_swapped_seqs
        return num_free_blocks - num_required_blocks >= self.watermark_blocks

    def swap_in(self, seq_group: SequenceGroup) -> Dict[int, int]:
        # CPU block -> GPU block.
        mapping: Dict[PhysicalTokenBlock, PhysicalTokenBlock] = {}
        for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
            new_block_table: BlockTable = []
            block_table = self.block_tables[seq.seq_id]

            for cpu_block in block_table:
                if cpu_block in mapping:
                    gpu_block = mapping[cpu_block]
                    gpu_block.ref_count += 1
                else:
                    gpu_block = self.gpu_allocator.allocate()
                    mapping[cpu_block] = gpu_block
                new_block_table.append(gpu_block)
                # Free the CPU block swapped in to GPU.
                self.cpu_allocator.free(cpu_block)
            self.block_tables[seq.seq_id] = new_block_table

        block_number_mapping = {
            cpu_block.block_number: gpu_block.block_number
            for cpu_block, gpu_block in mapping.items()
        }
        return block_number_mapping

    def can_swap_out(self, seq_group: SequenceGroup) -> bool:
        blocks = self._get_physical_blocks(seq_group)
        return len(blocks) <= self.cpu_allocator.get_num_free_blocks()

    def swap_out(self, seq_group: SequenceGroup) -> Dict[int, int]:
        # GPU block -> CPU block.
        mapping: Dict[PhysicalTokenBlock, PhysicalTokenBlock] = {}
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            new_block_table: BlockTable = []
            block_table = self.block_tables[seq.seq_id]

            for gpu_block in block_table:
                if gpu_block in mapping:
                    cpu_block = mapping[gpu_block]
                    cpu_block.ref_count += 1
                else:
                    cpu_block = self.cpu_allocator.allocate()
                    mapping[gpu_block] = cpu_block
                new_block_table.append(cpu_block)
                # Free the GPU block swapped out to CPU.
                self.gpu_allocator.free(gpu_block)
            self.block_tables[seq.seq_id] = new_block_table

        block_number_mapping = {
            gpu_block.block_number: cpu_block.block_number
            for gpu_block, cpu_block in mapping.items()
        }
        return block_number_mapping

    def _free_block_table(self, block_table: BlockTable) -> None:
        for block in set(block_table):
            if block.device == Device.GPU:
                self.gpu_allocator.free(block)
            else:
                self.cpu_allocator.free(block)

    def free(self, seq: Sequence) -> None:
        if seq.seq_id not in self.block_tables and seq.seq_id not in self.layer_block_tables:
            # Already freed or haven't been scheduled yet.
            return
        if self.tcp_client is not None:
            block_tables = self.layer_block_tables.pop(seq.seq_id)
            block_ids = [block.block_number for block in block_tables]
            if seq.is_free_cache or not seq.in_prefill:
                rank = 0 # assume tensor-model parallelism only
                self.tcp_client.send_pyobj(
                    ("free_cache", [seq.seq_id, rank, self.model_name, block_ids])
                )
            else:
                self.tcp_client.send_pyobj(
                    ("blocktable_store", [seq.seq_id, block_ids])
                )
            _ = self.tcp_client.recv_pyobj()

            self._free_block_table(block_tables)
        else:
            block_table = self.block_tables[seq.seq_id]
            self._free_block_table(block_table)
            del self.block_tables[seq.seq_id]

    def reset(self) -> None:
        for block_table in self.block_tables.values():
            self._free_block_table(block_table)
        self.block_tables.clear()

    def get_block_table(self, seq: Sequence) -> List[int]:
        block_table = self.block_tables[seq.seq_id]
        return [block.block_number for block in block_table]

    def get_layer_block_table(self, seq: Sequence) -> List[List[int]]:
        layerwise_blks = self.layer_block_tables[seq.seq_id]
        # use numpy array and cache to walk around the python list overhead
        if self.layerwise_table_np_cache[seq.seq_id][0] == len(layerwise_blks):
            return self.layerwise_table_np_cache[seq.seq_id][1]

        block_table = np.array(
            [block.block_number for block in layerwise_blks], dtype=np.int32
        )
        block_table = block_table.reshape(1, -1) // self.num_heads
        block_table = block_table + self.layer_offsets
        # cache the numpy array
        self.layerwise_table_np_cache[seq.seq_id] = (len(layerwise_blks), block_table)
        return block_table

    def get_num_gpu_blocks(self) -> int:
        return self.gpu_allocator.num_blocks

    def get_num_free_gpu_blocks(self) -> int:
        return self.gpu_allocator.get_num_free_blocks()

    def get_num_free_cpu_blocks(self) -> int:
        return self.cpu_allocator.get_num_free_blocks()
