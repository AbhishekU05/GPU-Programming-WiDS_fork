#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

__global__ void mulScale(const float* a, const float* b, float* c, float alpha, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = alpha * a[idx] * b[idx];
    }
}

int main() {
    const int N = 1 << 20;
    const float alpha = 7;
    const size_t bytes = N * sizeof(float);

    float *h_a = new float[N];
    float *h_b = new float[N];
    float *h_c = new float[N];
    float *h_ref = new float[N];

    for (int i = 0; i < N; i++) {
        h_a[i] = i * 0.01f;
        h_b[i] = i * 0.02f;
        h_ref[i] = alpha * h_a[i] * h_b[i];
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    mulScale<<<gridSize, blockSize>>>(d_a, d_b, d_c, alpha, N);

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    bool flag = true;
    for (int i = 0; i < N; i++) {
        if (std::fabs(h_c[i] - h_ref[i]) > 1e-6f) {
            flag = false;
            std::cerr << "Mismatch at " << i << std::endl;
            break;
        }
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    delete[] h_ref;

    if (flag) {
        std::cout << "Multiply-and-scale correct\n";
        return 0;
    }
    else {
        return 1;
    }
}
