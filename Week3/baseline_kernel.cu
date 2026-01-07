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

__global__ void slidingGlobal(const float* a, float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= 5 && i < n - 5) {
        float sum = 0.0f;
        for (int k = -5; k <= 5; k++) {
            sum += a[i + k];
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

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    slidingGlobal<<<grid, block>>>(d_a, d_b, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cout << "Global memory time: " << ms << " ms\n";

    cudaFree(d_a);
    cudaFree(d_b);
    return 0;
}
