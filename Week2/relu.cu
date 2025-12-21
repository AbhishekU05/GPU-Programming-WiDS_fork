#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <random>

float rand_num() {
    static std::mt19937 gen(std::random_device{}());
    static std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
    return dist(gen);
}

__global__ void relu(const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = x[idx] > 0 ? x[idx] : 0.0f;
    }
}

int main() {
    const int N = 1 << 20;
    const size_t bytes = N * sizeof(float);

    float *h_x = new float[N];
    float *h_y = new float[N];
    float *h_ref = new float[N];

    for (int i = 0; i < N; i++) {
        h_x[i] = rand_num();
        h_ref[i] = h_x[i] > 0 ? h_x[i] : 0;
    }

    float *d_x, *d_y;
    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_y, bytes);

    cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    relu<<<gridSize, blockSize>>>(d_x, d_y, N);

    cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost);

    bool flag = true;
    for (int i = 0; i < N; i++) {
        if (std::fabs(h_y[i] - h_ref[i]) > 1e-6f) {
            flag = false;
            std::cerr << "Mismatch at " << i << std::endl;
            break;
        }
    }

    cudaFree(d_x);
    cudaFree(d_y);
    delete[] h_x;
    delete[] h_y;
    delete[] h_ref;

    if (flag) {
        std::cout << "ReLU correct\n";
        return 0;
    }
    else {
        return 1;
    }
}
