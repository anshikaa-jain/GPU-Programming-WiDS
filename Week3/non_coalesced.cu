// %%writefile non_coalesced.cu
#include <cuda_runtime.h>
#include <iostream>

#define STRIDE 8

__global__ void nonCoalescedKernel(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int accessIdx = idx * STRIDE;

    if (accessIdx < N) {
        C[accessIdx] = A[accessIdx] + B[accessIdx];
    }
}

int main() {
    int N = 1 << 24;
    size_t size = N * sizeof(float);

    float *hA = new float[N];
    float *hB = new float[N];
    float *hC = new float[N];

    for (int i = 0; i < N; i++) {
        hA[i] = 1.0f;
        hB[i] = 2.0f;
    }

    float *dA, *dB, *dC;
    cudaMalloc(&dA, size);
    cudaMalloc(&dB, size);
    cudaMalloc(&dC, size);

    cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid((N / STRIDE + block.x - 1) / block.x);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    nonCoalescedKernel<<<grid, block>>>(dA, dB, dC, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Non-coalesced kernel time: " << ms << " ms\n";

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    delete[] hA;
    delete[] hB;
    delete[] hC;

    return 0;
}
