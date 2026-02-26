#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <string>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


// Constant Memory for Filters
__constant__ float c_f1[25];
__constant__ float c_f2[25];
__constant__ float c_f3[25];
__constant__ float c_f4[25];

// Shared Memory Constants
#define TILE_SIZE 16
#define RADIUS 2
#define SHARED_DIM (TILE_SIZE + 2 * RADIUS)

// --- MASK KERNELS ---


// --- COMPUTATION KERNELS ---

// Normalization Kernel
__global__ void normalizeKernel(uint16_t* img, int width, int height, int shift_bits) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < width && row < height) {
        img[row * width + col] >>= shift_bits;
    }
}

// Gains and Save Kernel
__global__ void applyGainAndSaveKernel(float* r, float* g, float* b, uint8_t* output, 
                                       float gain, float r_gain, float b_gain, 
                                       int width, int height, int bit_depth) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int idx = row * width + col;
        float max_val = (float)((1 << bit_depth) - 1);

        float pr = fminf(fmaxf(r[idx] * gain * r_gain, 0.0f), max_val);
        float pg = fminf(fmaxf(g[idx] * gain, 0.0f), max_val);
        float pb = fminf(fmaxf(b[idx] * gain * b_gain, 0.0f), max_val);

        output[idx * 3 + 0] = (uint8_t)((pr / max_val) * 255.0f);
        output[idx * 3 + 1] = (uint8_t)((pg / max_val) * 255.0f);
        output[idx * 3 + 2] = (uint8_t)((pb / max_val) * 255.0f);
    }
}

// Hard-coded convolution helpers using shared memory tile
__device__ float convolve_f1(uint16_t tile[SHARED_DIM][SHARED_DIM], int ty, int tx) {
    float sum = 0.0f;
    sum += (float)tile[ty + 0][tx + 2] * -0.125f; 
    sum += (float)tile[ty + 1][tx + 2] * 0.25f;
    sum += (float)tile[ty + 2][tx + 0] * -0.125f; 
    sum += (float)tile[ty + 2][tx + 1] * 0.25f;
    sum += (float)tile[ty + 2][tx + 2] * 0.5f;   
    sum += (float)tile[ty + 2][tx + 3] * 0.25f;
    sum += (float)tile[ty + 2][tx + 4] * -0.125f; 
    sum += (float)tile[ty + 3][tx + 2] * 0.25f;
    sum += (float)tile[ty + 4][tx + 2] * -0.125f;
    return sum;
}

__device__ float convolve_f2(uint16_t tile[SHARED_DIM][SHARED_DIM], int ty, int tx) {
    float sum = 0.0f;
    sum += (float)tile[ty + 0][tx + 2] * 0.0625f; 
    sum += (float)tile[ty + 1][tx + 1] * -0.125f;
    sum += (float)tile[ty + 1][tx + 3] * -0.125f; 
    sum += (float)tile[ty + 2][tx + 0] * -0.125f;
    sum += (float)tile[ty + 2][tx + 1] * 0.5f;    
    sum += (float)tile[ty + 2][tx + 2] * 0.625f;
    sum += (float)tile[ty + 2][tx + 3] * 0.5f;    
    sum += (float)tile[ty + 2][tx + 4] * -0.125f;
    sum += (float)tile[ty + 3][tx + 1] * -0.125f; 
    sum += (float)tile[ty + 3][tx + 3] * -0.125f;
    sum += (float)tile[ty + 4][tx + 2] * 0.0625f;
    return sum;
}

__device__ float convolve_f3(uint16_t tile[SHARED_DIM][SHARED_DIM], int ty, int tx) {
    float sum = 0.0f;
    sum += (float)tile[ty + 2][tx + 0] * 0.0625f; 
    sum += (float)tile[ty + 1][tx + 1] * -0.125f;
    sum += (float)tile[ty + 3][tx + 1] * -0.125f; 
    sum += (float)tile[ty + 0][tx + 2] * -0.125f;
    sum += (float)tile[ty + 1][tx + 2] * 0.5f;    
    sum += (float)tile[ty + 2][tx + 2] * 0.625f;
    sum += (float)tile[ty + 3][tx + 2] * 0.5f;    
    sum += (float)tile[ty + 4][tx + 2] * -0.125f;
    sum += (float)tile[ty + 1][tx + 3] * -0.125f; 
    sum += (float)tile[ty + 3][tx + 3] * -0.125f;
    sum += (float)tile[ty + 2][tx + 4] * 0.0625f;
    return sum;
}

__device__ float convolve_f4(uint16_t tile[SHARED_DIM][SHARED_DIM], int ty, int tx) {
    float sum = 0.0f;
    sum += (float)tile[ty + 0][tx + 2] * -0.1875f; 
    sum += (float)tile[ty + 1][tx + 1] * 0.25f;
    sum += (float)tile[ty + 1][tx + 3] * 0.25f;    
    sum += (float)tile[ty + 2][tx + 0] * -0.1875f;
    sum += (float)tile[ty + 2][tx + 2] * 0.75f;    
    sum += (float)tile[ty + 2][tx + 4] * -0.1875f;
    sum += (float)tile[ty + 3][tx + 1] * 0.25f;    
    sum += (float)tile[ty + 3][tx + 3] * 0.25f;
    sum += (float)tile[ty + 4][tx + 2] * -0.1875f;
    return sum;
}

// Merged Convolution kernel (with pattern-aware computation)
__global__ void mergedConvolveKernel(uint16_t* input, float* out_r, float* out_g, float* out_b, int width, int height) {
    __shared__ uint16_t tile[SHARED_DIM][SHARED_DIM];
    
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
                tile[i][j] = input[load_row * width + load_col];
            } else {
                tile[i][j] = 0;
            }
        }
    }
    __syncthreads();
    
    if (col < width && row < height) {
        float res_r, res_g, res_b;
        float center = (float)tile[ty + RADIUS][tx + RADIUS];
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

        int idx = row * width + col;
        out_r[idx] = res_r; 
        out_g[idx] = res_g; 
        out_b[idx] = res_b;
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
    float gain = 5.0f, r_gain = 1.2f, b_gain = 1.35f;
    std::string input_path = "/content/file.raw";
    size_t img_size = width * height;
    float total_time = 0.0;

    // Filter Definitions
    float h_f1[25] = {0,0,-0.125,0,0, 0,0,0.25,0,0, -0.125,0.25,0.5,0.25,-0.125, 0,0,0.25,0,0, 0,0,-0.125,0,0};
    float h_f2[25] = {0,0,0.0625,0,0, 0,-0.125,0,-0.125,0, -0.125,0.5,0.625,0.5,-0.125, 0,-0.125,0,-0.125,0, 0,0,0.0625,0,0};
    float h_f3[25]; 
    for(int r=0; r<5; r++) for(int c=0; c<5; c++) h_f3[r*5+c] = h_f2[c*5+r];
    float h_f4[25] = {0,0,-0.1875,0,0, 0,0.25,0,0.25,0, -0.1875,0,0.75,0,-0.1875, 0,0.25,0,0.25,0, 0,0,-0.1875,0,0};

    // Device Pointers
    uint16_t *d_raw;
    float *d_mask_r, *d_mask_gr, *d_mask_gb, *d_mask_b, *d_mask_g;
    float *d_r, *d_g, *d_b, *d_i1, *d_i2, *d_i3, *d_i4;
    float *df1, *df2, *df3, *df4;
    uint8_t *d_out_img;
    uint16_t* h_raw;

    // Initializing memory on GPU
    cudaMalloc(&d_raw, img_size * sizeof(uint16_t));
    cudaMalloc(&d_mask_r, img_size * sizeof(float)); 
    cudaMalloc(&d_mask_gr, img_size * sizeof(float));
    cudaMalloc(&d_mask_gb, img_size * sizeof(float)); 
    cudaMalloc(&d_mask_b, img_size * sizeof(float));
    cudaMalloc(&d_mask_g, img_size * sizeof(float));
    cudaMalloc(&d_r, img_size * sizeof(float)); 
    cudaMalloc(&d_g, img_size * sizeof(float)); 
    cudaMalloc(&d_b, img_size * sizeof(float));
    cudaMalloc(&d_i1, img_size * sizeof(float)); 
    cudaMalloc(&d_i2, img_size * sizeof(float)); 
    cudaMalloc(&d_i3, img_size * sizeof(float));
    cudaMalloc(&d_i4, img_size * sizeof(float));
    cudaMalloc(&d_out_img, img_size * 3 * sizeof(uint8_t));
    
    cudaMalloc(&df1, 25 * sizeof(float)); 
    cudaMalloc(&df2, 25 * sizeof(float)); 
    cudaMalloc(&df3, 25 * sizeof(float)); 
    cudaMalloc(&df4, 25 * sizeof(float));

    // Copy Filters to GPU memory
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
    
    // Benchmarking
    for (int i = 0; i<100; i++)
    {
        printf("Executing demosaic, run number %d\n", i+1);
            
        // Copy raw image to GPU memory
        cudaMemcpy(d_raw, h_raw, img_size * sizeof(uint16_t), cudaMemcpyHostToDevice);

        // Initializing grid and Block Size
        dim3 block(16, 16);
        dim3 grid((int)ceil((float)width / block.x), (int)ceil((float)height / block.y));

        // Start execution time
        cudaEventRecord(start);

        // --- CALL THE CUDA KERNELS ---

        normalizeKernel<<<grid, block>>>(d_raw, width, height, 6);
        mergedConvolveKernel<<<grid, block>>>(d_raw, d_r, d_g, d_b, width, height);
        applyGainAndSaveKernel<<<grid, block>>>(d_r, d_g, d_b, d_out_img, gain, r_gain, b_gain, width, height, bit_depth);


        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        // Add the execution time of current iteration to the total time
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        total_time += ms;

        // Reset the GPu memory
        cudaMemset(d_raw, 0, img_size * sizeof(uint16_t));
        cudaMemset(d_mask_r, 0, img_size * sizeof(float));
        cudaMemset(d_mask_gr, 0, img_size * sizeof(float));
        cudaMemset(d_mask_gb, 0, img_size * sizeof(float));
        cudaMemset(d_mask_b, 0, img_size * sizeof(float));
        cudaMemset(d_mask_g, 0, img_size * sizeof(float));
        cudaMemset(d_r, 0, img_size * sizeof(float));
        cudaMemset(d_g, 0, img_size * sizeof(float));
        cudaMemset(d_b, 0, img_size * sizeof(float));
        cudaMemset(d_i1, 0, img_size * sizeof(float));
        cudaMemset(d_i2, 0, img_size * sizeof(float));
        cudaMemset(d_i3, 0, img_size * sizeof(float));
        cudaMemset(d_i4, 0, img_size * sizeof(float));
    }

    printf("Demosaic Average Execution Time: %.3f ms\n", total_time/100.0);

    // Copy the output image to CPU memory
    uint8_t* h_out = (uint8_t*)malloc(img_size * 3 * sizeof(uint8_t));
    cudaMemcpy(h_out, d_out_img, img_size * 3 * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    stbi_write_png("demosaic.png", width, height, 3, h_out, width * 3);

    // Free all CPU and GPU memory
    free(h_raw); free(h_out);
    cudaFree(d_raw); cudaFree(d_mask_r); cudaFree(d_mask_gr); cudaFree(d_mask_gb);
    cudaFree(d_mask_b); cudaFree(d_mask_g); cudaFree(d_r); cudaFree(d_g); cudaFree(d_b);
    cudaFree(d_i1); cudaFree(d_i2); cudaFree(d_i3); cudaFree(d_i4); cudaFree(d_out_img);
    cudaFree(df1); cudaFree(df2); cudaFree(df3); cudaFree(df4);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    printf("Completed\n");
    return 0;
}