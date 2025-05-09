from vllm.core.block.block_table import BlockTable
from typing import List, Tuple, Dict, Any


class MiniKVManager:
    # class SkewKVAllocator:
    #     def __init__(
    #         self,
    #         master_kv_allocator,
    #     ):
    #         self.master_kv_allocator = master_kv_allocator

    #     def allocate(self, size):
    #         return self.master_kv_allocator.allocate(size)

    def __init__(self, master_allocator, master_kv_cache):
        # print(master_kv_cache[0][0].shape)
        self.block_tables: Dict[int, BlockTable] = {}
        self.seq_lens = {}
        self.block_size = 16
        self.allocator = master_allocator
        self.kv_cache = master_kv_cache
        for i in range(32):
            self.kv_cache[i].fill_(1000.0)

    def allocate_kv_cache(self, seq_id, seq_lens):
        if seq_id in self.block_tables.keys():
            assert seq_lens > self.seq_lens[seq_id]
            self.block_tables[seq_id].append_token_ids(
                [1 for _ in range(seq_lens - self.seq_lens[seq_id])]
            )
        else:

            self.seq_lens[seq_id] = 0
            self.block_tables[seq_id] = BlockTable(16, self.allocator)
            self.block_tables[seq_id].allocate([1 for _ in range(seq_lens)], None)
        self.seq_lens[seq_id] = seq_lens
        
    def free(self, seq_id):
        self.block_tables[seq_id].free()
        del self.block_tables[seq_id]
        del self.seq_lens[seq_id]