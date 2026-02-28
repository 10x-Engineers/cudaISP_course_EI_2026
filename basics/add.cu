#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>

// Kernel for 2D Matrix Addition: A = A + B
__global__ void matrixAddKernel(float* A, float* B, int width, int height) {
    // Calculate global row and column
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Linear index for 1D memory representation
    if (col < width && row < height) {
        int idx = row * width + col;
        A[idx] = A[idx] + B[idx];
    }
}

// Pretty print function: shows 3x3 corners with ellipses
void prettyPrint(float* mat, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        // Only print first 3 and last 3 rows
        if (i < 3 || i >= rows - 3) {
            for (int j = 0; j < cols; j++) {
                // Only print first 3 and last 3 columns
                if (j < 3 || j >= cols - 3) {
                    std::cout << std::setw(6) << mat[i * cols + j] << " ";
                } else if (j == 3) {
                    std::cout << "... ";
                }
            }
            std::cout << std::endl;
        } else if (i == 3) {
            std::cout << "   ...   ...   ...   ...   ...   ..." << std::endl;
        }
    }
}

int main() {
    const int N = 31;
    const size_t size = N * N * sizeof(float);

    // Host allocation
    float *h_A = new float[N * N];
    float *h_B = new float[N * N];

    // Initialize: A with 1s, B with 2s
    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    std::cout << "Matrix A (Initial):" << std::endl;
    prettyPrint(h_A, N, N);
    std::cout << "\nMatrix B (Initial):" << std::endl;
    prettyPrint(h_B, N, N);

    // Device allocation
    float *d_A, *d_B;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Execution Configuration: 3x3 Grid, 16x16 Blocks
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(2, 2);

    matrixAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, N, N);

    // Copy result back
    cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);

    std::cout << "\nResult Matrix A (A = A + B):" << std::endl;
    prettyPrint(h_A, N, N);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    delete[] h_A;
    delete[] h_B;

    return 0;
}
