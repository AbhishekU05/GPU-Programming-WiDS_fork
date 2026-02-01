import torch
import numpy as np

def cpu_step(T, P, alpha, dt):
    Tn = T.copy()
    Tn[1:-1,1:-1] = (
        T[1:-1,1:-1]
        + alpha * dt * (
            T[2:,1:-1] + T[:-2,1:-1]
            + T[1:-1,2:] + T[1:-1,:-2]
            - 4*T[1:-1,1:-1]
        )
        + P[1:-1,1:-1]
    )
    return Tn

H = W = 128
alpha = 0.1
dt = 0.01

T = np.random.rand(H, W).astype(np.float32)
P = np.random.rand(H, W).astype(np.float32)

T_cpu = cpu_step(T, P, alpha, dt)

T_gpu = torch.tensor(T, device="cuda")
P_gpu = torch.tensor(P, device="cuda")

T_ref = T_gpu.clone()
T_ref[1:-1,1:-1] = (
    T_ref[1:-1,1:-1]
    + alpha * dt * (
        T_ref[2:,1:-1] + T_ref[:-2,1:-1]
        + T_ref[1:-1,2:] + T_ref[1:-1,:-2]
        - 4*T_ref[1:-1,1:-1]
    )
    + P_gpu[1:-1,1:-1]
)

err = torch.max(torch.abs(T_ref.cpu() - torch.tensor(T_cpu)))
print("Max error:", err.item())
