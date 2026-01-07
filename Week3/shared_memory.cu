#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>

using namespace std;

float rand_num() {
    static mt19937 gen(42);
    static uniform_real_distribution<float> dist(0.0f, 1.0f);
    return dist(gen);
}

__global__ void slidingShared(const float* a, float* b, int n) {
    extern __shared__ float tile[];

    int tx = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tx;

    int radius = 5;
    int lx = tx + radius;

    // center
    if (i < n)
        tile[lx] = a[i];
    else
        tile[lx] = 0.0f;

    // left halo
    if (tx < radius) {
        int li = i - radius;
        tile[lx - radius] = (li >= 0) ? a[li] : 0.0f;
    }

    // right halo
    if (tx >= blockDim.x - radius) {
        int ri = i + radius;
        tile[lx + radius] = (ri < n) ? a[ri] : 0.0f;
    }

    __syncthreads();

    if (i >= radius && i < n - radius) {
        float sum = 0.0f;
        for (int k = -radius; k <= radius; k++) {
            sum += tile[lx + k];
        }
        b[i] = sum;
    }
}

int main() {
    int n = 1 << 24;
    size_t bytes = n * sizeof(float);

    vector<float> h_a(n), h_b(n);
    for (int i = 0; i < n; i++)
        h_a[i] = rand_num();

    float *d_a, *d_b;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);

    int radius = 5;
    size_t sharedBytes = (block.x + 2 * radius) * sizeof(float);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    slidingShared<<<grid, block, sharedBytes>>>(d_a, d_b, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cout << "Shared memory time: " << ms << " ms\n";

    cudaFree(d_a);
    cudaFree(d_b);
    return 0;
}
