#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

// Kernel
__global__ void multiplyScale(const float* a, const float* b,
                              float* c, float alpha, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = alpha * a[idx] * b[idx];
    }
}

// Main
int main() {
    int n = 1 << 18;
    float alpha = 2.5f;
    size_t size = n * sizeof(float);

    // Host memory
    float *h_a   = new float[n];
    float *h_b   = new float[n];
    float *h_c   = new float[n];
    float *h_ref = new float[n];

    for (int i = 0; i < n; i++) {
        h_a[i]   = 0.01f * i;
        h_b[i]   = 0.02f * i;
        h_ref[i] = alpha * h_a[i] * h_b[i];
    }

    // Device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    int blockSize = 128;
    int gridSize  = (n + blockSize - 1) / blockSize;

    multiplyScale<<<gridSize, blockSize>>>(d_a, d_b, d_c, alpha, n);

    // Error check + sync
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel error: "
                  << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    cudaDeviceSynchronize();

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Verification
    bool passed = true;
    for (int i = 0; i < n; i++) {
        if (fabs(h_c[i] - h_ref[i]) > 1e-5) {
            std::cout << "Mismatch at index " << i
                      << " GPU: " << h_c[i]
                      << " CPU: " << h_ref[i] << std::endl;
            passed = false;
            break;
        }
    }

    if (passed)
        std::cout << "Multiply & Scale PASSED\n";
    else
        std::cout << "Multiply & Scale FAILED\n";

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    delete[] h_ref;

    return 0;
}
