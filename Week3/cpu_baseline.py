import numpy as np
import time
from tqdm import tqdm

def step(T, P, alpha, dt):
    H, W = T.shape
    T_new = T.copy()
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            lap = (T[i+1, j] + T[i-1, j] +
                   T[i, j+1] + T[i, j-1] -
                   4 * T[i, j])
            T_new[i, j] = T[i, j] + alpha * dt * lap + P[i, j]
    return T_new

# Grid size
H = W = 1024

# Initial conditions
T = np.zeros((H, W), dtype=np.float32)
P = np.random.rand(H, W).astype(np.float32)

T_max = 1_000
alpha = 0.1
dt = 0.01

start = time.perf_counter()

for _ in tqdm(range(T_max), desc="CPU Heat Simulation"):
    T = step(T, P, alpha, dt)

end = time.perf_counter()

print(f"CPU time: {end - start:.4f} seconds")
