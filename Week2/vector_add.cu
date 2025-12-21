#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>

float rand_num() {
    static std::mt19937 gen(std::random_device{}());
    static std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
    return dist(gen);
}

__global__ void vecAdd(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

bool verify(const std::vector<float>& a, const std::vector<float>& b, const std::vector<float>& c, int n) {
    for (int i = 0; i < n; i++) {
        float ref = a[i] + b[i];
        if (std::fabs(c[i] - ref) > 1e-6f) {
            std::cout << "Does not match"<< std::endl;
            return false;
        }
    }
    return true;
}

void exp(int n, const std::vector<int>& blockSizes) {
    std::cout << "\nVector size n = " << n << "\n";

    size_t bytes = n * sizeof(float);

    std::vector<float> h_a(n), h_b(n), h_c(n);

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

    for (int blockSize : blockSizes) {
        int gridSize = (n + blockSize - 1) / blockSize;
        long long totalThreads =
            static_cast<long long>(gridSize) * blockSize;

        vecAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
        cudaDeviceSynchronize();

        cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

        bool ok = verify(h_a, h_b, h_c, n);

        std::cout << "  blockDim.x = " << blockSize
                  << " | gridDim.x = " << gridSize
                  << " | threads launched = " << totalThreads
                  << " | status = " << (ok ? "OK" : "FAIL")
                  << "\n";
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

int main() {
    std::vector<int> sizes = {1000, 100000, 10000000};
    std::vector<int> blockSizes = {32, 128, 256, 512};

    for (int n : sizes) {
        exp(n, blockSizes);
    }

    return 0;
}
