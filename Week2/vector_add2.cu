//task3
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

// Kernel
__global__ void vectorAdd2(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Main
int main() {
    int n = 10000000;   // Change for Task 2 / Task 3 experiments
    size_t size = n * sizeof(float);

    // Host memory
    float *h_a = new float[n];
    float *h_b = new float[n];
    float *h_c = new float[n];
    float *h_ref = new float[n];

    for (int i = 0; i < n; i++) {
        h_a[i] = 0.5f * i;
        h_b[i] = 0.25f * i;
        h_ref[i] = h_a[i] + h_b[i];
    }

    // Device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    int blockSize = 512;
    int gridSize  = (n + blockSize - 1) / blockSize;

    std::cout << "n = " << n
              << ", blockSize = " << blockSize
              << ", gridSize = " << gridSize
              << ", totalThreads = " << gridSize * blockSize
              << std::endl;

    // CUDA Timing using Events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    vectorAdd2<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    std::cout << "Kernel execution time: "
              << elapsed_ms << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Copy back
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Correctness Check
    bool passed = true;
    for (int i = 0; i < n; i++) {
        if (fabs(h_c[i] - h_ref[i]) > 1e-5) {
            std::cout << "Mismatch at index " << i << std::endl;
            passed = false;
            break;
        }
    }

    if (passed)
        std::cout << "Vector Addition PASSED" << std::endl;
    else
        std::cout << "Vector Addition FAILED " << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    delete[] h_ref;

    return 0;
}
