import torch
import time
from tqdm import tqdm
from cuda_extension import run_cuda

H = W = 1024
T_max = 1000
alpha = 0.1
dt = 0.01

T = torch.zeros((H, W), device="cuda")
P = torch.rand((H, W), device="cuda")

# warmup
run_cuda(T, P, alpha, dt, 10)
torch.cuda.synchronize()

start = time.perf_counter()

# run with progress bar
run_cuda(T, P, alpha, dt, T_max)

torch.cuda.synchronize()
end = time.perf_counter()

print(f"CUDA kernel time: {end - start:.4f} seconds")
