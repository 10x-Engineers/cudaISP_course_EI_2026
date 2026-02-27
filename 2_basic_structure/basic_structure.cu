// Include All The Necessary Libraries

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <string>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Define All GPU Kernels

// Red mask kernel

// Green in red rows mask Kernel

// Green in blue rows mask kernel

// Blue mask kernel

// Green Mask kernel

// Normalization Kernel

// Multiply Kernel

// Where Kenrel

// Gains and Save Kernel

// Convolution kernel


// Load Raw Image From Disk To Memory

uint16_t* load_raw_to_host(const std::string& full_file_path, int width, int height) {
    FILE *file = fopen(full_file_path.c_str(), "rb");
    if (file == NULL) return NULL;
    uint16_t *raw_data = (uint16_t *)malloc(width * height * sizeof(uint16_t));
    fread(raw_data, sizeof(uint16_t), width * height, file);
    fclose(file);
    return raw_data;
}

// Main Function
int main() {
    printf("Start execution\n");
    
    // Initializing parameters
    int width = 3328, height = 2464;
    int bit_depth = 10;
    float gain = 5.0f, r_gain = 1.2f, b_gain = 1.35f;
    std::string input_path = "/content/file.raw";
    size_t img_size = width * height;
    float total_time = 0.0;

    // Convolitional Filter Definitions
    float h_f1[25] = {0,0,-0.125,0,0, 0,0,0.25,0,0, -0.125,0.25,0.5,0.25,-0.125, 0,0,0.25,0,0, 0,0,-0.125,0,0};
    float h_f2[25] = {0,0,0.0625,0,0, 0,-0.125,0,-0.125,0, -0.125,0.5,0.625,0.5,-0.125, 0,-0.125,0,-0.125,0, 0,0,0.0625,0,0};
    float h_f3[25]; 
    for(int r=0; r<5; r++) for(int c=0; c<5; c++) h_f3[r*5+c] = h_f2[c*5+r];
    float h_f4[25] = {0,0,-0.1875,0,0, 0,0.25,0,0.25,0, -0.1875,0,0.75,0,-0.1875, 0,0.25,0,0.25,0, 0,0,-0.1875,0,0};

    // Host and Device Pointers: We'll be using h_ to save host side pointers and d_ to save device memory location pointers
    uint16_t *d_raw; // Use to store raw image
    float *d_mask_r, *d_mask_gr, *d_mask_gb, *d_mask_b, *d_mask_g;  // Use to store the masks
    float *d_r, *d_g, *d_b; // Use to store the final red green and blue channels 
    float *d_i1, *d_i2, *d_i3, *d_i4; // Use to tore the output of convolutional Kernels
    float *df1, *df2, *df3, *df4; // Use to store the 5x5 convolutional filters
    uint8_t *d_out_img; // Use to store the final output image
    uint16_t* h_raw;   // Use to store the raw image

    // Initializing Memory on GPU
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

    // Copy Convolutional Filters to GPU Memory
    cudaMemcpy(df1, h_f1, 25 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(df2, h_f2, 25 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(df3, h_f3, 25 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(df4, h_f4, 25 * sizeof(float), cudaMemcpyHostToDevice);

    // Load The Raw Image on Host
    h_raw = load_raw_to_host(input_path, width, height);
    if (!h_raw) { printf("Error: Could not load %s\n", input_path.c_str()); return 1; }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Benchmarking
    for (int i = 0; i<100; i++)
    {
        printf("Executing demosaic, run number %d\n", i+1);
            
        // Copy Host Raw Image to GPU Memory location
        cudaMemcpy(d_raw, h_raw, img_size * sizeof(uint16_t), cudaMemcpyHostToDevice);

        // Initializing Grid and Block Size
        dim3 block(16, 16);
        dim3 grid((int)ceil((float)width / block.x), (int)ceil((float)height / block.y));

        // Start Measuring Execution Time
        cudaEventRecord(start);

        // --- CALL THE CUDA KERNELS ---

        // Normalize The Image

        // Demosaic The Image

        // Apply Gains
        
        // Stop Measuring Execution Time
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        // Track Total Execution Time
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        total_time += ms;

        // Reset GPU Memory Locations
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

    // Copy The Final RGB Image from Device To Host
    uint8_t* h_out = (uint8_t*)malloc(img_size * 3 * sizeof(uint8_t));
    cudaMemcpy(h_out, d_out_img, img_size * 3 * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    // Save the Image As Png File
    stbi_write_png("demosaic.png", width, height, 3, h_out, width * 3);

    // Free All CPU and GPU Memory
    free(h_raw); free(h_out);
    cudaFree(d_raw); cudaFree(d_mask_r); cudaFree(d_mask_gr); cudaFree(d_mask_gb);
    cudaFree(d_mask_b); cudaFree(d_mask_g); cudaFree(d_r); cudaFree(d_g); cudaFree(d_b);
    cudaFree(d_i1); cudaFree(d_i2); cudaFree(d_i3); cudaFree(d_i4); cudaFree(d_out_img);
    cudaFree(df1); cudaFree(df2); cudaFree(df3); cudaFree(df4);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    printf("Completed\n");
    return 0;
}