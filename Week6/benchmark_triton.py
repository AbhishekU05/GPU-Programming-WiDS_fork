import torch, time, triton
from triton_kernel import heat_kernel

H=W=1024
T_max=1000
alpha=0.1
dt=0.01

T = torch.zeros((H,W), device="cuda")
P = torch.rand((H,W), device="cuda")
Tn = torch.empty_like(T)

BLOCK=1024
grid=lambda meta:(triton.cdiv(H*W, meta["BLOCK"]),)

# warmup
for _ in range(10):
    heat_kernel[grid](T, Tn, P, H, W, alpha, dt, BLOCK=BLOCK)
    T,Tn=Tn,T
torch.cuda.synchronize()

start=time.perf_counter()
for _ in range(T_max):
    heat_kernel[grid](T, Tn, P, H, W, alpha, dt, BLOCK=BLOCK)
    T,Tn=Tn,T
torch.cuda.synchronize()
end=time.perf_counter()

print(f"Triton kernel time: {end-start:.4f} s")
