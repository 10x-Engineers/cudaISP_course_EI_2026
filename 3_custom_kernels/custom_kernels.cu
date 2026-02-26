#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <string>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// --- MASK KERNELS ---

// Red mask kernel
__global__ void maskRKernel(float* mask, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < width && row < height) {
        mask[row * width + col] = (row % 2 == 0 && col % 2 == 0) ? 1.0f : 0.0f;
    }
}

// Green in red rows mask Kernel
__global__ void maskGrKernel(float* mask, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < width && row < height) {
        mask[row * width + col] = (row % 2 == 0 && col % 2 != 0) ? 1.0f : 0.0f;
    }
}

// Green in blue rows mask kernel
__global__ void maskGbKernel(float* mask, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < width && row < height) {
        mask[row * width + col] = (row % 2 != 0 && col % 2 == 0) ? 1.0f : 0.0f;
    }
}

// Blue mask kernel
__global__ void maskBKernel(float* mask, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < width && row < height) {
        mask[row * width + col] = (row % 2 != 0 && col % 2 != 0) ? 1.0f : 0.0f;
    }
}

// Green Mask kernel
__global__ void maskGCombinedKernel(float* mask, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < width && row < height) {
        bool isGr = (row % 2 == 0 && col % 2 != 0);
        bool isGb = (row % 2 != 0 && col % 2 == 0);
        mask[row * width + col] = (isGr || isGb) ? 1.0f : 0.0f;
    }
}


// --- COMPUTATION KERNELS ---

// Normalization Kernel
__global__ void normalizeKernel(uint16_t* img, int width, int height, int shift_bits) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < width && row < height) {
        img[row * width + col] >>= shift_bits;
    }
}

// Multiply Kernel
__global__ void multiplyKernel(uint16_t* raw, float* mask, float* output, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < width && row < height) {
        int idx = row * width + col;
        output[idx] = (float)raw[idx] * mask[idx];
    }
}

// Where Kenrel
__global__ void whereKernel(float* mask, float* interp, float* channel, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < width && row < height) {
        int idx = row * width + col;
        if (mask[idx] > 0.0f) {
            channel[idx] = interp[idx];
        }
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

// Convolution kernel
__global__ void convolve5x5Kernel(uint16_t* input, float* filter, float* output, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        float sum = 0.0f;
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                int cur_row = row + (i - 2);
                int cur_col = col + (j - 2);
                
                if (cur_row >= 0 && cur_row < height && cur_col >= 0 && cur_col < width) {
                    sum += (float)input[cur_row * width + cur_col] * filter[i * 5 + j];
                }
            }
        }
        output[row * width + col] = sum;
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
    std::string input_path = "file.raw";
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
    float *d_r, *d_g, *d_b, *d_i1, *d_i2, *d_i3;
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
    cudaMalloc(&d_out_img, img_size * 3 * sizeof(uint8_t));
    
    cudaMalloc(&df1, 25 * sizeof(float)); 
    cudaMalloc(&df2, 25 * sizeof(float)); 
    cudaMalloc(&df3, 25 * sizeof(float)); 
    cudaMalloc(&df4, 25 * sizeof(float));

    // Copy Filters to GPU memory
    cudaMemcpy(df1, h_f1, 25 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(df2, h_f2, 25 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(df3, h_f3, 25 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(df4, h_f4, 25 * sizeof(float), cudaMemcpyHostToDevice);

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

        maskRKernel<<<grid, block>>>(d_mask_r, width, height);
        maskGrKernel<<<grid, block>>>(d_mask_gr, width, height);
        maskGbKernel<<<grid, block>>>(d_mask_gb, width, height);
        maskBKernel<<<grid, block>>>(d_mask_b, width, height);
        maskGCombinedKernel<<<grid, block>>>(d_mask_g, width, height);
         
        multiplyKernel<<<grid, block>>>(d_raw, d_mask_r, d_r, width, height);
        multiplyKernel<<<grid, block>>>(d_raw, d_mask_g, d_g, width, height);
        multiplyKernel<<<grid, block>>>(d_raw, d_mask_b, d_b, width, height);
        
        convolve5x5Kernel<<<grid, block>>>(d_raw, df1, d_i1, width, height);
        whereKernel<<<grid, block>>>(d_mask_r, d_i1, d_g, width, height);
        whereKernel<<<grid, block>>>(d_mask_b, d_i1, d_g, width, height);
        
        convolve5x5Kernel<<<grid, block>>>(d_raw, df2, d_i1, width, height);
        convolve5x5Kernel<<<grid, block>>>(d_raw, df3, d_i2, width, height);
        convolve5x5Kernel<<<grid, block>>>(d_raw, df4, d_i3, width, height);
        
        whereKernel<<<grid, block>>>(d_mask_gr, d_i1, d_r, width, height);
        whereKernel<<<grid, block>>>(d_mask_gb, d_i2, d_r, width, height);
        whereKernel<<<grid, block>>>(d_mask_b, d_i3, d_r, width, height);
        whereKernel<<<grid, block>>>(d_mask_gb, d_i1, d_b, width, height);
        whereKernel<<<grid, block>>>(d_mask_gr, d_i2, d_b, width, height);
        whereKernel<<<grid, block>>>(d_mask_r, d_i3, d_b, width, height);

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
    }

    printf("Demosaic Average Execution Time (Custom Kernels): %.3f ms\n", total_time/100.0);

    // Copy the output image to CPU memory
    uint8_t* h_out = (uint8_t*)malloc(img_size * 3 * sizeof(uint8_t));
    cudaMemcpy(h_out, d_out_img, img_size * 3 * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    stbi_write_png("demosaic_custom_kernels.png", width, height, 3, h_out, width * 3);

    // Free all CPU and GPU memory
    free(h_raw); free(h_out);
    cudaFree(d_raw); cudaFree(d_mask_r); cudaFree(d_mask_gr); cudaFree(d_mask_gb);
    cudaFree(d_mask_b); cudaFree(d_mask_g); cudaFree(d_r); cudaFree(d_g); cudaFree(d_b);
    cudaFree(d_i1); cudaFree(d_i2); cudaFree(d_i3); cudaFree(d_out_img);
    cudaFree(df1); cudaFree(df2); cudaFree(df3); cudaFree(df4);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    printf("Completed\n");
    return 0;
}