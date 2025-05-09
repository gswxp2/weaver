from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.distributed import (ensure_model_parallel_initialized,
                              init_distributed_environment,
                              set_custom_all_reduce)
import torch
# init_distributed_environment(1, 0,
#                                  "env://", 0)

# ensure_model_parallel_initialized(1,
#                                       1)
embed_tokens = VocabParallelEmbedding(
                128256,
                8192,
                org_num_embeddings=128256,
                quant_config=None)

input_ids = torch.randint(0, 128256, (1, 224), dtype=torch.long)
embed_tokens(input_ids)
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
for _ in range(100):
    embed_tokens(input_ids)
start_event.record()
for _ in range(100):
    embed_tokens(input_ids)
end_event.record()
torch.cuda.synchronize()
print(start_event.elapsed_time(end_event) / 100)
    
