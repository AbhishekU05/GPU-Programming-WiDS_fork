#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>

using namespace std;

float rand_num() {
    static mt19937 gen(42);
    static uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
    return dist(gen);
}

__global__ void vecAdd(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
        c[idx] = a[idx] + b[idx];
}

int main() {
    int n = 1 << 24;
    int blockSize = 256;
    size_t bytes = n * sizeof(float);

    vector<float> h_a(n), h_b(n);
    for (int i = 0; i < n; i++) {
        h_a[i] = rand_num();
        h_b[i] = rand_num();
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

    dim3 block(blockSize);
    dim3 grid((n + blockSize - 1) / blockSize);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    vecAdd<<<grid, block>>>(d_a, d_b, d_c, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cout << "Coalesced kernel time: " << ms << " ms\n";

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}
