#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

// Kernel
__global__ void relu(const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = (x[idx] > 0.0f) ? x[idx] : 0.0f;
    }
}

// Main
int main() {
    int n = 1 << 16;
    size_t size = n * sizeof(float);

    // Host memory
    float *h_x   = new float[n];
    float *h_y   = new float[n];
    float *h_ref = new float[n];

    for (int i = 0; i < n; i++) {
        h_x[i]   = (i % 3 == 0) ? -i * 0.1f : i * 0.1f;
        h_ref[i] = (h_x[i] > 0.0f) ? h_x[i] : 0.0f;
    }

    // Device memory
    float *d_x, *d_y;
    cudaMalloc(&d_x, size);
    cudaMalloc(&d_y, size);

    cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize  = (n + blockSize - 1) / blockSize;

    relu<<<gridSize, blockSize>>>(d_x, d_y, n);

    // Error check + sync
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel error: "
                  << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    cudaDeviceSynchronize();

    cudaMemcpy(h_y, d_y, size, cudaMemcpyDeviceToHost);

    // Verification
    bool passed = true;
    for (int i = 0; i < n; i++) {
        if (fabs(h_y[i] - h_ref[i]) > 1e-5) {
            std::cout << "Mismatch at index " << i
                      << " GPU: " << h_y[i]
                      << " CPU: " << h_ref[i] << std::endl;
            passed = false;
            break;
        }
    }

    if (passed)
        std::cout << "ReLU PASSED\n";
    else
        std::cout << "ReLU FAILED\n";

    cudaFree(d_x);
    cudaFree(d_y);
    delete[] h_x;
    delete[] h_y;
    delete[] h_ref;

    return 0;
}
