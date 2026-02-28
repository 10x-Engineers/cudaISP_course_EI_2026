#include <iostream>
#include <cuda_runtime.h>

// Kernel function to add 1 to each element
__global__ void addOneKernel(int* data, int size) {
    // Calculate the global index of the thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check: only operate if index is within array size
    if (idx < size) {
        data[idx] += 1;
    }
}

// Function to print the "bookends" of the array
void printArrayBounds(int* data, int size) {
    std::cout << "First 5: ";
    for (int i = 0; i < 5; i++) std::cout << data[i] << " ";
    
    std::cout << "... Last 5: ";
    for (int i = size - 5; i < size; i++) std::cout << data[i] << " ";
    std::cout << std::endl;
}

int main() {
    const int N = 2048;
    const int sizeInBytes = N * sizeof(int);

    // 1. Setup Host memory
    int h_data[N];
    for (int i = 0; i < N; i++) h_data[i] = i + 1;

    std::cout << "Before Kernel:" << std::endl;
    printArrayBounds(h_data, N);

    // 2. Setup Device memory
    int* d_data;
    cudaMalloc(&d_data, sizeInBytes);
    cudaMemcpy(d_data, h_data, sizeInBytes, cudaMemcpyHostToDevice);

    // 3. Launch Configuration: 3 blocks, 32 threads each
    int threadsPerBlock = 16;
    int blocksPerGrid = 1;
    
    addOneKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, N);

    // Synchronous error detection
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Launch failed: %s\n", cudaGetErrorString(err));
    }

    // Or if you synchronize
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution failed: %s\n", cudaGetErrorString(err));
    }

    // 4. Copy result back and cleanup
    cudaMemcpy(h_data, d_data, sizeInBytes, cudaMemcpyDeviceToHost);

    std::cout << "\nAfter Kernel:" << std::endl;
    printArrayBounds(h_data, N);

    cudaFree(d_data);
    return 0;
}
