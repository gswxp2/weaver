from vllm.model_executor.layers.activation import SiluAndMul
import torch
batch_size = 16
embedding = 14336 * 2
input_tensor = torch.randn((batch_size, embedding), dtype=torch.bfloat16).cuda()
silu_and_mul = SiluAndMul()
silu_and_mul(input_tensor)
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    for _ in range(100):
        silu_and_mul(input_tensor)
torch.cuda.synchronize()
graph.replay()
start.record()
graph.replay()
end.record()
torch.cuda.synchronize()
print(start.elapsed_time(end) / 100)
nbytes = batch_size * embedding * 3
