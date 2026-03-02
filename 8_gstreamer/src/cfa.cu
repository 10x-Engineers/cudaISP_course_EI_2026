#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <cmath> 
#include "cfa.h"

// Shared Memory Constants - Updated to 32x32
#define TILE_SIZE 32
#define RADIUS 2
#define SHARED_DIM (TILE_SIZE + 2 * RADIUS) // 36

/**
 * OPTIMIZED HARD-CODED CONVOLUTION HELPERS
 */
__device__ __forceinline__ float convolve_f1(uint16_t tile[SHARED_DIM][SHARED_DIM], int ty, int tx) {
    float sum = (float)tile[ty + 2][tx + 2] * 0.5f;
    sum += ((float)tile[ty + 1][tx + 2] + (float)tile[ty + 3][tx + 2] + (float)tile[ty + 2][tx + 1] + (float)tile[ty + 2][tx + 3]) * 0.25f;
    sum -= ((float)tile[ty + 0][tx + 2] + (float)tile[ty + 4][tx + 2] + (float)tile[ty + 2][tx + 0] + (float)tile[ty + 2][tx + 4]) * 0.125f;
    return sum;
}

__device__ __forceinline__ float convolve_f2(uint16_t tile[SHARED_DIM][SHARED_DIM], int ty, int tx) {
    float sum = (float)tile[ty + 2][tx + 2] * 0.625f;
    sum += ((float)tile[ty + 2][tx + 1] + (float)tile[ty + 2][tx + 3]) * 0.5f;
    sum -= ((float)tile[ty + 1][tx + 1] + (float)tile[ty + 1][tx + 3] + (float)tile[ty + 2][tx + 0] + (float)tile[ty + 2][tx + 4] + (float)tile[ty + 3][tx + 1] + (float)tile[ty + 3][tx + 3]) * 0.125f;
    sum += ((float)tile[ty + 0][tx + 2] + (float)tile[ty + 4][tx + 2]) * 0.0625f;
    return sum;
}

__device__ __forceinline__ float convolve_f3(uint16_t tile[SHARED_DIM][SHARED_DIM], int ty, int tx) {
    float sum = (float)tile[ty + 2][tx + 2] * 0.625f;
    sum += ((float)tile[ty + 1][tx + 2] + (float)tile[ty + 3][tx + 2]) * 0.5f;
    sum -= ((float)tile[ty + 1][tx + 1] + (float)tile[ty + 3][tx + 1] + (float)tile[ty + 0][tx + 2] + (float)tile[ty + 4][tx + 2] + (float)tile[ty + 1][tx + 3] + (float)tile[ty + 3][tx + 3]) * 0.125f;
    sum += ((float)tile[ty + 2][tx + 0] + (float)tile[ty + 2][tx + 4]) * 0.0625f;
    return sum;
}

__device__ __forceinline__ float convolve_f4(uint16_t tile[SHARED_DIM][SHARED_DIM], int ty, int tx) {
    float sum = (float)tile[ty + 2][tx + 2] * 0.75f;
    sum += ((float)tile[ty + 1][tx + 1] + (float)tile[ty + 1][tx + 3] + (float)tile[ty + 3][tx + 1] + (float)tile[ty + 3][tx + 3]) * 0.25f;
    sum -= ((float)tile[ty + 0][tx + 2] + (float)tile[ty + 2][tx + 0] + (float)tile[ty + 2][tx + 4] + (float)tile[ty + 4][tx + 2]) * 0.1875f;
    return sum;
}

/**
 * UPDATED DEMOSAIC KERNEL - OUTPUTS RGBX (4 channels with X=0)
 */
__global__ void unifiedDemosaicKernel(const uint16_t* __restrict__ input, uchar4* __restrict__ output, 
                                      int width, int height, 
                                      int shift_bits, float max_val_inv,
                                      float gain, float r_gain, float b_gain) {
    
    __shared__ uint16_t tile[SHARED_DIM][SHARED_DIM];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + tx;
    int row = blockIdx.y * TILE_SIZE + ty;

    // Collaborative Shared Memory Load
    for (int i = ty; i < SHARED_DIM; i += TILE_SIZE) {
        int load_row = min(max(blockIdx.y * TILE_SIZE + i - RADIUS, 0), height - 1);
        for (int j = tx; j < SHARED_DIM; j += TILE_SIZE) {
            int load_col = min(max(blockIdx.x * TILE_SIZE + j - RADIUS, 0), width - 1);
            tile[i][j] = __ldg(&input[load_row * width + load_col]) >> shift_bits;
        }
    }

    __syncthreads();

    if (col < width && row < height) {
        float r, g, b;
        float center_val = (float)tile[ty + RADIUS][tx + RADIUS];

        // Bayer Pattern Logic (RGGB)
        if ((row & 1) == 0) {
            if ((col & 1) == 0) { // Red pixel
                r = center_val;
                g = convolve_f1(tile, ty, tx);
                b = convolve_f4(tile, ty, tx);
            } else { // Gr pixel
                r = convolve_f2(tile, ty, tx);
                g = center_val;
                b = convolve_f3(tile, ty, tx);
            }
        } else {
            if ((col & 1) == 0) { // Gb pixel
                r = convolve_f3(tile, ty, tx);
                g = center_val;
                b = convolve_f2(tile, ty, tx);
            } else { // Blue pixel
                r = convolve_f4(tile, ty, tx);
                g = convolve_f1(tile, ty, tx);
                b = center_val;
            }
        }

        float gr = gain * r_gain;
        float gb = gain * b_gain;

        // Calculate RGB values and pack into uchar4
        uchar4 rgbx;
        rgbx.x = (uint8_t)(fminf(fmaxf(r * gr, 0.0f), 1023.0f) * max_val_inv);
        rgbx.y = (uint8_t)(fminf(fmaxf(g * gain, 0.0f), 1023.0f) * max_val_inv);
        rgbx.z = (uint8_t)(fminf(fmaxf(b * gb, 0.0f), 1023.0f) * max_val_inv);
        rgbx.w = 0; // X channel set to 0

        // Vector store - single 32-bit write
        output[row * width + col] = rgbx;
    }
}

// --- CFA CLASS IMPLEMENTATION ---

// Constructor
cfa::cfa(int img_width, int img_height, int depth, int s_bits, float g, float r_g, float b_g) 
    : width(img_width), height(img_height), bit_depth(depth), 
      shift_bits(s_bits), gain(g), r_gain(r_g), b_gain(b_g) {
    
    // Initialize CUDA memory
    size_t img_size = (size_t)width * height;
    
    cudaError_t err = cudaMalloc(&d_raw, img_size * sizeof(uint16_t));
    if (err != cudaSuccess) {
        printf("CFA ERROR: cudaMalloc for d_raw failed: %s\n", cudaGetErrorString(err));
        d_raw = nullptr;
    }
    
    err = cudaMalloc(&d_out_rgbx, img_size * sizeof(uchar4)); // RGBX: 4 channels as uchar4
    if (err != cudaSuccess) {
        printf("CFA ERROR: cudaMalloc for d_out_rgbx failed: %s\n", cudaGetErrorString(err));
        d_out_rgbx = nullptr;
    }
    
    // Setup grid and block dimensions
    block = dim3(TILE_SIZE, TILE_SIZE);
    grid = dim3((int)std::ceil((float)width / block.x), 
                (int)std::ceil((float)height / block.y));
    
    // Pre-calculate max_val_inv
    max_val_inv = 255.0f / (float)((1 << bit_depth) - 1);
    
    printf("CFA object created: %dx%d, bit depth: %d\n", width, height, bit_depth);
    printf("Output format: RGBX (4 channels, X=0) using uchar4\n");
    printf("Grid: %dx%d, Block: %dx%d\n", grid.x, grid.y, block.x, block.y);
    printf("Memory: d_raw=%p, d_out_rgbx=%p\n", d_raw, d_out_rgbx);
}

// Destructor
cfa::~cfa() {
    // Free GPU memory
    if (d_raw) {
        cudaFree(d_raw);
        d_raw = nullptr;
    }
    if (d_out_rgbx) {
        cudaFree(d_out_rgbx);
        d_out_rgbx = nullptr;
    }
}

// Execute demosaic kernel
void cfa::execute(const uint16_t* gpu_raw_image_ptr) {
    
    // Launch kernel
    unifiedDemosaicKernel<<<grid, block>>>(gpu_raw_image_ptr, d_out_rgbx, width, height, 
                                           shift_bits, max_val_inv, gain, r_gain, b_gain);
    cudaDeviceSynchronize();
}

// Get output device pointer (RGBX format - 4 channels as uchar4)
uchar4* cfa::get_output_ptr() const {
    return d_out_rgbx;
}