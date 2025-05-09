import random
import time
from typing import Tuple, List
import numpy as np
import torch

from vllm import attention_ops

MAX_SEQ_LEN = 256
NUM_BLOCKS = 80000

def create_kv_caches(
    num_blocks: int,
    block_size: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    dtype: torch.dtype,
    seed: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    scale = head_size**-0.5
    x = 16 // torch.tensor([], dtype=dtype).element_size()
    key_cache_shape = (num_blocks, num_heads, head_size // x, block_size, x)
    key_caches = []
    for _ in range(num_layers):
        key_cache = torch.empty(size=key_cache_shape,
                                dtype=dtype,
                                device='cuda')
        key_cache.uniform_(-scale, scale)
        key_caches.append(key_cache)

    value_cache_shape = (num_blocks, num_heads, head_size, block_size)
    value_caches = []
    for _ in range(num_layers):
        value_cache = torch.empty(size=value_cache_shape,
                                  dtype=dtype,
                                  device='cuda')
        value_cache.uniform_(-scale, scale)
        value_caches.append(value_cache)
    return key_caches, value_caches


def run_paged_attention(
    num_seqs: int,
    num_heads: Tuple[int, int],
    head_size: int,
    use_alibi: bool,
    block_size: int,
    dtype: torch.dtype,
    seed: int,
):
    random.seed(seed)
    np.random.seed(seed)
    scale = float(1.0 / (head_size**0.5))
    num_query_heads, num_kv_heads = num_heads
    query = torch.empty(num_seqs,
                        num_query_heads,
                        head_size,
                        dtype=dtype,
                        device="cuda")
    query.uniform_(-scale, scale)

    assert num_query_heads % num_kv_heads == 0
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_mapping = torch.repeat_interleave(
        torch.arange(num_kv_heads, dtype=torch.int32, device="cuda"),
        num_queries_per_kv)
    alibi_slopes = None
    if use_alibi:
        alibi_slopes = torch.randn(num_query_heads,
                                   dtype=torch.float,
                                   device="cuda")

    context_lens = [random.randint(1, MAX_SEQ_LEN) for _ in range(num_seqs)]
    context_lens[-1] = MAX_SEQ_LEN
    max_context_len = max(context_lens)
    context_lens = torch.tensor(context_lens, dtype=torch.int, device="cuda")

    # Create the block tables.
    max_num_blocks_per_seq = (max_context_len + block_size - 1) // block_size
    block_ids = np.random.choice(
        range(NUM_BLOCKS),
        num_seqs * max_num_blocks_per_seq,
        replace=False
    )
    block_ids = block_ids.reshape(num_seqs, max_num_blocks_per_seq)
    block_tables = torch.tensor(block_ids, dtype=torch.int, device="cuda")

    # Create the KV caches.
    key_caches, value_caches = create_kv_caches(NUM_BLOCKS, block_size, 1,
                                                num_kv_heads, head_size, dtype,
                                                seed)
    key_cache, value_cache = key_caches[0], value_caches[0]
    key_cache = key_cache.reshape(-1, key_cache.shape[-3], key_cache.shape[-2], key_cache.shape[-1])
    value_cache = value_cache.reshape(-1, value_cache.shape[-2], value_cache.shape[-1])

    # Call the paged attention kernel.
    def run_kernel():
        output = torch.empty_like(query)
        attention_ops.single_query_cached_kv_attention(
            output,
            query,
            key_cache,
            value_cache,
            head_mapping,
            scale,
            block_tables,
            context_lens,
            block_size,
            max_context_len,
            alibi_slopes,
        )
        return output

    # warmup
    for _ in range(5):
        run_kernel()
    # benchmark
    torch.cuda.synchronize()
    st = time.perf_counter()
    for _ in range(100):
        run_kernel()
    torch.cuda.synchronize()
    ed = time.perf_counter()
    return (ed - st) * 1e3 / 100


def run_headwise_attention(
    num_seqs: int,
    num_heads: Tuple[int, int],
    head_size: int,
    use_alibi: bool,
    block_size: int,
    dtype: torch.dtype,
    seed: int,
):
    random.seed(seed)
    np.random.seed(seed)
    scale = float(1.0 / (head_size**0.5))
    num_query_heads, num_kv_heads = num_heads
    query = torch.empty(num_seqs,
                        num_query_heads,
                        head_size,
                        dtype=dtype,
                        device="cuda")
    query.uniform_(-scale, scale)

    assert num_query_heads % num_kv_heads == 0
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_mapping = torch.repeat_interleave(
        torch.arange(num_kv_heads, dtype=torch.int32, device="cuda"),
        num_queries_per_kv)
    alibi_slopes = None
    if use_alibi:
        alibi_slopes = torch.randn(num_query_heads,
                                   dtype=torch.float,
                                   device="cuda")

    context_lens = [random.randint(1, MAX_SEQ_LEN) for _ in range(num_seqs)]
    context_lens[-1] = MAX_SEQ_LEN
    max_context_len = max(context_lens)
    context_lens = torch.tensor(context_lens, dtype=torch.int, device="cuda")

    # Create the block tables.
    max_num_blocks_per_seq = (max_context_len + block_size - 1) // block_size
    block_ids = np.random.choice(
        range(NUM_BLOCKS * num_kv_heads),
        num_seqs * num_kv_heads * max_num_blocks_per_seq,
        replace=False
    )
    block_ids = block_ids.reshape(num_seqs, num_kv_heads, max_num_blocks_per_seq)
    block_tables = torch.tensor(block_ids, dtype=torch.int, device="cuda")

    # Create the KV caches.
    key_caches, value_caches = create_kv_caches(NUM_BLOCKS, block_size, 1,
                                                num_kv_heads, head_size, dtype,
                                                seed)
    key_cache, value_cache = key_caches[0], value_caches[0]
    key_cache = key_cache.reshape(-1, key_cache.shape[-3], key_cache.shape[-2], key_cache.shape[-1])
    value_cache = value_cache.reshape(-1, value_cache.shape[-2], value_cache.shape[-1])

    # Call the paged attention kernel.
    def run_kernel():
        output = torch.empty_like(query)
        attention_ops.single_query_cached_kv_headwise_attention(
            output,
            query,
            key_cache,
            value_cache,
            head_mapping,
            scale,
            block_tables,
            context_lens,
            block_size,
            max_context_len,
            alibi_slopes,
        )
        return output

    # warmup
    for _ in range(5):
        run_kernel()
    # benchmark
    torch.cuda.synchronize()
    st = time.perf_counter()
    for _ in range(100):
        run_kernel()
    torch.cuda.synchronize()
    ed = time.perf_counter()
    return (ed - st) * 1e3 / 100



if __name__ == "__main__":
    for bs in [1, 2, 4, 8, 16, 32]:
        pg_t = run_paged_attention(bs, (40, 40), 128, False, 32, torch.float16, 0)
        head_t = run_headwise_attention(bs, (40, 40), 128, False, 32, torch.float16, 0)
        diff = head_t / pg_t
        print(f"bs: {bs} seq {MAX_SEQ_LEN} pg_t: {pg_t:.3f} head_t: {head_t:.3f} diff {diff:.3f}")
