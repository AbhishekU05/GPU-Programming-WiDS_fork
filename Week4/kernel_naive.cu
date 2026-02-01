#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

__global__ void softmax_naive(const float* input, float* output, int N) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    // One block per row
    input += row * N;
    output += row * N;

    // Step 1: find max (numerical stability)
    float max_val = -1e20f;
    for (int i = tid; i < N; i += blockDim.x) {
        max_val = fmaxf(max_val, input[i]);
    }

    // Reduce max across block (naive reduction)
    __shared__ float smax;
    if (tid == 0) smax = max_val;
    __syncthreads();

    atomicMax((int*)&smax, __float_as_int(max_val));
    __syncthreads();

    // Step 2: compute exp and sum
    float sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        sum += expf(input[i] - smax);
    }

    __shared__ float ssum;
    if (tid == 0) ssum = 0.0f;
    __syncthreads();

    atomicAdd(&ssum, sum);
    __syncthreads();

    // Step 3: normalize
    for (int i = tid; i < N; i += blockDim.x) {
        output[i] = expf(input[i] - smax) / ssum;
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
    softmax_naive<<<grid, block>>>(input, output, cols);
}

}
