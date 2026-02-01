import torch
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void heat_step(
    const float* T,
    float* Tn,
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

        Tn[idx] = T[idx] + alpha * dt * lap + P[idx];
    }
}

void heat_forward(
    torch::Tensor T,
    torch::Tensor Tn,
    torch::Tensor P,
    float alpha,
    float dt
) {
    int H = T.size(0);
    int W = T.size(1);

    dim3 threads(16, 16);
    dim3 blocks((W + 15) / 16, (H + 15) / 16);

    heat_step<<<blocks, threads>>>(
        T.data_ptr<float>(),
        Tn.data_ptr<float>(),
        P.data_ptr<float>(),
        H, W, alpha, dt
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &heat_forward, "Heat step forward");
}
"""

heat = load_inline(
    name="heat_ext",
    cpp_sources="",
    cuda_sources=cuda_src,
    functions=None,
    extra_cuda_cflags=["-O3"],
    with_cuda=True,
    verbose=False,
)

def run_cuda(T, P, alpha, dt, steps):
    Tn = torch.empty_like(T)
    for _ in range(steps):
        heat.forward(T, Tn, P, alpha, dt)
        T, Tn = Tn, T
    return T
