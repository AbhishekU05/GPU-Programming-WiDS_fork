#include <cuda_runtime.h>

__global__ void heat_step(
    const float* T,
    float* T_new,
    const float* P,
    int H, int W,
    float alpha, float dt
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < H - 1 && j > 0 && j < W - 1) {
        int idx = i * W + j;
        float lap =
            T[(i + 1) * W + j] +
            T[(i - 1) * W + j] +
            T[i * W + (j + 1)] +
            T[i * W + (j - 1)] -
            4.0f * T[idx];

        T_new[idx] = T[idx] + alpha * dt * lap + P[idx];
    }
}
