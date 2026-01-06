// %%writefile shared_memory.cu
#include <cuda_runtime.h>
#include <iostream>

#define BLOCK_SIZE 256

// Baseline: Global Memory Reduction
__global__ void globalReduce(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        atomicAdd(output, input[idx]);
    }
}

// Optimized: Shared Memory Reduction

__global__ void sharedReduce(const float* input, float* output, int N) {
    __shared__ float sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Load from global memory to shared memory
    if (idx < N)
        sdata[tid] = input[idx];
    else
        sdata[tid] = 0.0f;

    __syncthreads();

    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // One atomic add per block
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

// Host Code
int main() {
    int N = 1 << 20;  // ~1 million elements
    size_t size = N * sizeof(float);

    float* h_input = new float[N];
    for (int i = 0; i < N; i++)
        h_input[i] = 1.0f;

    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, sizeof(float));

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // ---------------- Global Memory Version ----------------
    cudaMemset(d_output, 0, sizeof(float));

    cudaEvent_t start1, stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);

    cudaEventRecord(start1);
    globalReduce<<<grid, block>>>(d_input, d_output, N);
    cudaEventRecord(stop1);

    cudaEventSynchronize(stop1);

    float time_global;
    cudaEventElapsedTime(&time_global, start1, stop1);

    // ---------------- Shared Memory Version ----------------
    cudaMemset(d_output, 0, sizeof(float));

    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);

    cudaEventRecord(start2);
    sharedReduce<<<grid, block>>>(d_input, d_output, N);
    cudaEventRecord(stop2);

    cudaEventSynchronize(stop2);

    float time_shared;
    cudaEventElapsedTime(&time_shared, start2, stop2);

    // ---------------- Results ----------------
    std::cout << "Global memory kernel time:  " << time_global << " ms\n";
    std::cout << "Shared memory kernel time:  " << time_shared << " ms\n";
    std::cout << "Speedup: " << time_global / time_shared << "x\n";

    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;

    return 0;
}
