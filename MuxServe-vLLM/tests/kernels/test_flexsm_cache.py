import random
import pytest
import torch

from vllm import cache_ops

DTYPES = [torch.float]
NUM_TOKENS = [83]  # Arbitrary values for testing
NUM_LAYERS = [5]  # Arbitrary values for testing
NUM_HEADS = [8]  # Arbitrary values for testing
HEAD_SIZES = [64]
BLOCK_SIZES = [32]
NUM_BLOCKS = [1024]  # Arbitrary values for testing
NUM_MAPPINGS = [256]  # Arbitrary values for testing
SEEDS = [0]

@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_reshape_and_cache(
    kv_cache_factory,
    num_tokens: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    seed: int,
) -> None:
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Create a random slot mapping.
    num_slots = block_size * num_blocks
    slot_mapping = random.sample(range(num_slots), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, dtype=torch.int64, device="cuda")

    qkv = torch.randn(num_tokens,
                      3,
                      num_heads,
                      head_size,
                      dtype=dtype,
                      device="cuda")
    _, key, value = qkv.unbind(dim=1)

    # Create the KV caches.
    key_caches, value_caches = kv_cache_factory(num_blocks, block_size, 1,
                                                num_heads, head_size, dtype,
                                                seed)
    key_cache, value_cache = key_caches[0], value_caches[0]

    # Clone the KV caches.
    cloned_key_cache = key_cache.clone()
    cloned_value_cache = value_cache.clone()

    # Call the reshape_and_cache kernel.
    cache_ops.reshape_and_cache(key, value,
        key_cache.reshape(-1, key_cache.shape[-3], key_cache.shape[-2], key_cache.shape[-1]),
        value_cache.reshape(-1, value_cache.shape[-2], value_cache.shape[-1]),
                                slot_mapping)

    # Run the reference implementation.
    reshaped_key = key.reshape(num_tokens, *key_cache[0, :, :, 0, :].shape)
    block_indicies = torch.div(slot_mapping, block_size, rounding_mode="floor")
    block_indicies = block_indicies.cpu().tolist()
    block_offsets = slot_mapping % block_size
    block_offsets = block_offsets.cpu().tolist()
    for i in range(num_tokens):
        block_idx = block_indicies[i]
        block_offset = block_offsets[i]
        cloned_key_cache[block_idx, :, :, block_offset, :] = reshaped_key[i]
        cloned_value_cache[block_idx, :, :, block_offset] = value[i]

    assert torch.allclose(key_cache, cloned_key_cache)
    assert torch.allclose(value_cache, cloned_value_cache)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_headwise_reshape_and_cache(
    kv_cache_factory,
    num_tokens: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    seed: int,
) -> None:
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    qkv = torch.randn(num_tokens,
                      3,
                      num_heads,
                      head_size,
                      dtype=dtype,
                      device="cuda")
    _, key, value = qkv.unbind(dim=1)

    # Create the KV caches.
    key_caches, value_caches = kv_cache_factory(num_blocks, block_size, 1,
                                                num_heads, head_size, dtype,
                                                seed)
    key_cache, value_cache = key_caches[0], value_caches[0]
    key_cache = key_cache.reshape(-1, key_cache.shape[-3], key_cache.shape[-2], key_cache.shape[-1])
    value_cache = value_cache.reshape(-1, value_cache.shape[-2], value_cache.shape[-1])

    # Clone the KV caches.
    cloned_key_cache = key_cache.clone()
    cloned_value_cache = value_cache.clone()

    # Create a random slot mapping.
    num_slots = block_size * num_heads * num_blocks
    slot_mapping = random.sample(range(num_slots), num_tokens * num_heads * 2)
    slot_mapping = torch.tensor(slot_mapping, dtype=torch.int64, device="cuda")
    slot_mapping = slot_mapping.reshape(num_tokens, 2, num_heads)
    slot_mapping = slot_mapping[:, 1, :]

    # Call the reshape_and_cache kernel.
    cache_ops.headwise_reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)

    # Run the reference implementation.
    reshaped_key = key.reshape(num_tokens, num_heads, *key_cache[0, :, 0, :].shape)
    block_indicies = torch.div(slot_mapping, block_size, rounding_mode="floor")
    block_indicies = block_indicies.cpu().tolist()
    block_offsets = slot_mapping % block_size
    block_offsets = block_offsets.cpu().tolist()
    for i in range(num_tokens):
        for head_idx in range(num_heads):
            block_idx = block_indicies[i][head_idx]
            block_offset = block_offsets[i][head_idx]
            cloned_key_cache[block_idx, :, block_offset, :] = reshaped_key[i, head_idx]
            cloned_value_cache[block_idx, :, block_offset] = value[i, head_idx]

    assert torch.allclose(key_cache, cloned_key_cache)
    assert torch.allclose(value_cache, cloned_value_cache)
