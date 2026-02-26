#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <cmath>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Constant Memory for Filters
__constant__ float c_f1[25];
__constant__ float c_f2[25];
__constant__ float c_f3[25];
__constant__ float c_f4[25];

// Shared Memory Constants
#define TILE_SIZE 32
#define RADIUS 2
#define SHARED_DIM (TILE_SIZE + 2 * RADIUS)

// --- COMPUTATION KERNELS ---

// Hard-coded convolution helpers using shared memory tile
// Hard-coded convolution helpers using shared memory tile (single expression)
__device__ float convolve_f1(float tile[SHARED_DIM][SHARED_DIM], int ty, int tx) {
    return tile[ty + 0][tx + 2] * -0.125f 
         + tile[ty + 1][tx + 2] * 0.25f
         + tile[ty + 2][tx + 0] * -0.125f 
         + tile[ty + 2][tx + 1] * 0.25f
         + tile[ty + 2][tx + 2] * 0.5f   
         + tile[ty + 2][tx + 3] * 0.25f
         + tile[ty + 2][tx + 4] * -0.125f 
         + tile[ty + 3][tx + 2] * 0.25f
         + tile[ty + 4][tx + 2] * -0.125f;
}

__device__ float convolve_f2(float tile[SHARED_DIM][SHARED_DIM], int ty, int tx) {
    return tile[ty + 0][tx + 2] * 0.0625f 
         + tile[ty + 1][tx + 1] * -0.125f
         + tile[ty + 1][tx + 3] * -0.125f 
         + tile[ty + 2][tx + 0] * -0.125f
         + tile[ty + 2][tx + 1] * 0.5f    
         + tile[ty + 2][tx + 2] * 0.625f
         + tile[ty + 2][tx + 3] * 0.5f    
         + tile[ty + 2][tx + 4] * -0.125f
         + tile[ty + 3][tx + 1] * -0.125f 
         + tile[ty + 3][tx + 3] * -0.125f
         + tile[ty + 4][tx + 2] * 0.0625f;
}

__device__ float convolve_f3(float tile[SHARED_DIM][SHARED_DIM], int ty, int tx) {
    return tile[ty + 2][tx + 0] * 0.0625f 
         + tile[ty + 1][tx + 1] * -0.125f
         + tile[ty + 3][tx + 1] * -0.125f 
         + tile[ty + 0][tx + 2] * -0.125f
         + tile[ty + 1][tx + 2] * 0.5f    
         + tile[ty + 2][tx + 2] * 0.625f
         + tile[ty + 3][tx + 2] * 0.5f    
         + tile[ty + 4][tx + 2] * -0.125f
         + tile[ty + 1][tx + 3] * -0.125f 
         + tile[ty + 3][tx + 3] * -0.125f
         + tile[ty + 2][tx + 4] * 0.0625f;
}

__device__ float convolve_f4(float tile[SHARED_DIM][SHARED_DIM], int ty, int tx) {
    return tile[ty + 0][tx + 2] * -0.1875f 
         + tile[ty + 1][tx + 1] * 0.25f
         + tile[ty + 1][tx + 3] * 0.25f    
         + tile[ty + 2][tx + 0] * -0.1875f
         + tile[ty + 2][tx + 2] * 0.75f    
         + tile[ty + 2][tx + 4] * -0.1875f
         + tile[ty + 3][tx + 1] * 0.25f    
         + tile[ty + 3][tx + 3] * 0.25f
         + tile[ty + 4][tx + 2] * -0.1875f;
}

// Merged Convolution kernel (with pattern-aware computation)
__global__ void mergedConvolveKernel(uint16_t* input, uchar3* output, int width, int height,
                                     int shift_bits, float max_val_inv, float gain, float r_gain, float b_gain) {
    
    __shared__ float tile[SHARED_DIM][SHARED_DIM];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + tx;
    int row = blockIdx.y * TILE_SIZE + ty;
    
    // Load data into shared memory with halo
    for (int i = ty; i < SHARED_DIM; i += TILE_SIZE) {
        for (int j = tx; j < SHARED_DIM; j += TILE_SIZE) {
            int load_row = blockIdx.y * TILE_SIZE + i - RADIUS;
            int load_col = blockIdx.x * TILE_SIZE + j - RADIUS;
            if (load_row >= 0 && load_row < height && load_col >= 0 && load_col < width) {
                tile[i][j] = (float)(input[load_row * width + load_col] >> shift_bits);
            } else {
                tile[i][j] = 0.0f;
            }
        }
    }
    __syncthreads();
    
    if (col < width && row < height) {
        float res_r, res_g, res_b;
        float center = tile[ty + RADIUS][tx + RADIUS];
        bool isEvenRow = (row % 2 == 0);
        bool isEvenCol = (col % 2 == 0);

        if (isEvenRow && isEvenCol) { 
            // Red pixel
            res_r = center; 
            res_g = convolve_f1(tile, ty, tx); 
            res_b = convolve_f4(tile, ty, tx); 
        } 
        else if (isEvenRow && !isEvenCol) { 
            // Green (red row) pixel
            res_r = convolve_f2(tile, ty, tx); 
            res_g = center; 
            res_b = convolve_f3(tile, ty, tx); 
        } 
        else if (!isEvenRow && isEvenCol) { 
            // Green (blue row) pixel
            res_r = convolve_f3(tile, ty, tx); 
            res_g = center; 
            res_b = convolve_f2(tile, ty, tx); 
        } 
        else { 
            // Blue pixel
            res_r = convolve_f4(tile, ty, tx); 
            res_g = convolve_f1(tile, ty, tx); 
            res_b = center; 
        }

        // Apply Gain and Normalize to 8-bit (vectorized store using uchar3)
        float max_val = (float)((1 << 10) - 1); // 10-bit max value = 1023
        uchar3 rgb;
        rgb.x = (uint8_t)(fminf(fmaxf(res_r * gain * r_gain, 0.0f), max_val) * max_val_inv);
        rgb.y = (uint8_t)(fminf(fmaxf(res_g * gain, 0.0f), max_val) * max_val_inv);
        rgb.z = (uint8_t)(fminf(fmaxf(res_b * gain * b_gain, 0.0f), max_val) * max_val_inv);

        output[row * width + col] = rgb;
    }
}

// --- HOST HELPERS ---

uint16_t* load_raw_to_host(const std::string& full_file_path, int width, int height) {
    FILE *file = fopen(full_file_path.c_str(), "rb");
    if (file == NULL) return NULL;
    uint16_t *raw_data = (uint16_t *)malloc(width * height * sizeof(uint16_t));
    fread(raw_data, sizeof(uint16_t), width * height, file);
    fclose(file);
    return raw_data;
}

int main() {
    printf("Start execution\n");
    
    // Initializing parameters
    int width = 3328, height = 2464;
    int bit_depth = 10;
    int shift_bits = 6; // Shift to get 10-bit values from 16-bit input
    float gain = 5.0f, r_gain = 1.2f, b_gain = 1.35f;
    std::string input_path = "file.raw";
    size_t img_size = width * height;
    float total_time = 0.0;

    // Filter Definitions (still needed for constant memory even though we're using hard-coded helpers)
    float h_f1[25] = {0,0,-0.125,0,0, 0,0,0.25,0,0, -0.125,0.25,0.5,0.25,-0.125, 0,0,0.25,0,0, 0,0,-0.125,0,0};
    float h_f2[25] = {0,0,0.0625,0,0, 0,-0.125,0,-0.125,0, -0.125,0.5,0.625,0.5,-0.125, 0,-0.125,0,-0.125,0, 0,0,0.0625,0,0};
    float h_f3[25]; 
    for(int r=0; r<5; r++) for(int c=0; c<5; c++) h_f3[r*5+c] = h_f2[c*5+r];
    float h_f4[25] = {0,0,-0.1875,0,0, 0,0.25,0,0.25,0, -0.1875,0,0.75,0,-0.1875, 0,0.25,0,0.25,0, 0,0,-0.1875,0,0};

    // Device Pointers
    uint16_t *d_raw;
    uchar3 *d_out_img; // Using uchar3 for vectorized store
    uint16_t* h_raw;

    // Initializing memory on GPU
    cudaMalloc(&d_raw, img_size * sizeof(uint16_t));
    cudaMalloc(&d_out_img, img_size * sizeof(uchar3));

    // Copy Filters to constant memory (optional - not used in kernel but kept for consistency)
    cudaMemcpyToSymbol(c_f1, h_f1, 25 * sizeof(float));
    cudaMemcpyToSymbol(c_f2, h_f2, 25 * sizeof(float));
    cudaMemcpyToSymbol(c_f3, h_f3, 25 * sizeof(float));
    cudaMemcpyToSymbol(c_f4, h_f4, 25 * sizeof(float));

    // Load raw image
    h_raw = load_raw_to_host(input_path, width, height);
    if (!h_raw) { printf("Error: Could not load %s\n", input_path.c_str()); return 1; }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Pre-compute inverse max value for normalization
    float max_val_inv = 255.0f / (float)((1 << bit_depth) - 1);
    
    // Benchmarking
    for (int i = 0; i < 100; i++)
    {
        printf("Executing demosaic, run number %d\n", i+1);
            
        // Copy raw image to GPU memory
        cudaMemcpy(d_raw, h_raw, img_size * sizeof(uint16_t), cudaMemcpyHostToDevice);

        // Initializing grid and Block Size
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((int)ceil((float)width / block.x), (int)ceil((float)height / block.y));

        // Start execution time
        cudaEventRecord(start);

        // --- CALL THE CUDA KERNELS ---
        mergedConvolveKernel<<<grid, block>>>(d_raw, d_out_img, width, height, shift_bits, max_val_inv, gain, r_gain, b_gain);
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        // Add the execution time of current iteration to the total time
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        total_time += ms;

        // Reset the output GPU memory
        if (i < 99) {
            cudaMemset(d_out_img, 0, img_size * sizeof(uchar3));
        }
    }

    printf("Demosaic Average Execution Time (Vectorization and Blocksize): %.3f ms\n", total_time/100.0);

    // Copy the output image to CPU memory
    uint8_t* h_out = (uint8_t*)malloc(img_size * 3 * sizeof(uint8_t));
    cudaMemcpy(h_out, d_out_img, img_size * sizeof(uchar3), cudaMemcpyDeviceToHost);
    stbi_write_png("vectorization_and_blocksize.png", width, height, 3, h_out, width * 3);

    // Free all CPU and GPU memory
    free(h_raw); 
    free(h_out);
    cudaFree(d_raw);
    cudaFree(d_out_img);
    cudaEventDestroy(start); 
    cudaEventDestroy(stop);

    printf("Completed\n");
    return 0;
}