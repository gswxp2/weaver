import torch

x_cpu = torch.randn(1024* 512, dtype=torch.bfloat16, device='cpu', pin_memory=True)
x_gpu = torch.randn(1024* 512, dtype=torch.bfloat16, device='cuda')
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
for _ in range(100):
    x_cpu.copy_(x_gpu, non_blocking=True)
torch.cuda.synchronize()
for _ in range(100):
    x_cpu.copy_(x_gpu, non_blocking=True)
start.record()
for _ in range(100):
    x_cpu.copy_(x_gpu, non_blocking=True)
end.record()
end.synchronize()
print(f"Time taken: {start.elapsed_time(end)} ms")