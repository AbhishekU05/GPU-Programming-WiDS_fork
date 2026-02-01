#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void softmax_optimized(
    const float* input,
    float* output,
    int N
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    input  += row * N;
    output += row * N;

    __shared__ float smax[256];
    __shared__ float ssum[256];

    // 1. Per-thread max
    float local_max = -1e20f;
    for (int i = tid; i < N; i += blockDim.x) {
        local_max = fmaxf(local_max, input[i]);
    }
    smax[tid] = local_max;
    __syncthreads();

    // Reduce max across block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            smax[tid] = fmaxf(smax[tid], smax[tid + stride]);
        __syncthreads();
    }
    float max_val = smax[0];

    // 2. Per-thread sum
    float local_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        local_sum += expf(input[i] - max_val);
    }
    ssum[tid] = local_sum;
    __syncthreads();

    // Reduce sum across block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            ssum[tid] += ssum[tid + stride];
        __syncthreads();
    }
    float sum = ssum[0];

    // 3. Normalize
    for (int i = tid; i < N; i += blockDim.x) {
        output[i] = expf(input[i] - max_val) / sum;
    }
}

extern "C" {

void softmax_forward(
    const float* input,
    float* output,
    int rows,
    int cols
) {
    dim3 grid(rows);
    dim3 block(256);
    softmax_optimized<<<grid, block>>>(input, output, cols);
}

}
