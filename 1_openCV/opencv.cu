#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <cuda_runtime.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

cv::Mat load_raw_to_mat(const std::string& path, int width, int height) {
    FILE* file = fopen(path.c_str(), "rb");
    if (!file) return cv::Mat();
    
    cv::Mat raw(height, width, CV_16UC1);
    fread(raw.data, sizeof(uint16_t), width * height, file);
    fclose(file);
    return raw;
}

void generate_masks(int width, int height, cv::cuda::GpuMat& mR, cv::cuda::GpuMat& mGr, cv::cuda::GpuMat& mGb, cv::cuda::GpuMat& mB, cv::cuda::GpuMat& mG) {
    cv::Mat hR = cv::Mat::zeros(height, width, CV_8UC1);
    cv::Mat hGr = cv::Mat::zeros(height, width, CV_8UC1);
    cv::Mat hGb = cv::Mat::zeros(height, width, CV_8UC1);
    cv::Mat hB = cv::Mat::zeros(height, width, CV_8UC1);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (y % 2 == 0 && x % 2 == 0) hR.at<uint8_t>(y, x) = 255;
            else if (y % 2 == 0 && x % 2 != 0) hGr.at<uint8_t>(y, x) = 255;
            else if (y % 2 != 0 && x % 2 == 0) hGb.at<uint8_t>(y, x) = 255;
            else if (y % 2 != 0 && x % 2 != 0) hB.at<uint8_t>(y, x) = 255;
        }
    }

    mR.upload(hR);
    mGr.upload(hGr);
    mGb.upload(hGb);
    mB.upload(hB);
    cv::cuda::bitwise_or(mGr, mGb, mG);
}

int main() {
    printf("Start execution\n");
    int width = 3328, height = 2464;
    int bit_depth = 10;
    float gain = 5.0f, r_gain = 1.2f, b_gain = 1.35f;
    std::string input_path = "file.raw";

    cv::Mat h_raw_orig = load_raw_to_mat(input_path, width, height);
    if (h_raw_orig.empty()) {
        std::cerr << "Error: Could not load " << input_path << std::endl;
        return 1;
    }

    cv::cuda::GpuMat d_raw_u16, d_raw_f32, d_interp_g, d_interp_f2, d_interp_f3, d_interp_f4;
    cv::cuda::GpuMat d_maskR, d_maskGr, d_maskGb, d_maskB, d_maskG;
    
    cv::cuda::GpuMat d_R(height, width, CV_32F, cv::Scalar(0));
    cv::cuda::GpuMat d_G(height, width, CV_32F, cv::Scalar(0));
    cv::cuda::GpuMat d_B(height, width, CV_32F, cv::Scalar(0));

    d_raw_u16.upload(h_raw_orig);
    generate_masks(width, height, d_maskR, d_maskGr, d_maskGb, d_maskB, d_maskG);

    float k1[25] = {0,0,-0.125f,0,0, 0,0,0.25f,0,0, -0.125f,0.25f,0.5f,0.25f,-0.125f, 0,0,0.25f,0,0, 0,0,-0.125f,0,0};
    float k2[25] = {0,0,0.0625f,0,0, 0,-0.125f,0,-0.125f,0, -0.125f,0.5f,0.625f,0.5f,-0.125f, 0,-0.125f,0,-0.125f,0, 0,0,0.0625f,0,0};
    float k3[25]; 
    for(int i=0; i<5; i++) for(int j=0; j<5; j++) k3[i*5+j] = k2[j*5+i];
    float k4[25] = {0,0,-0.1875f,0,0, 0,0.25f,0,0.25f,0, -0.1875f,0,0.75f,0,-0.1875f, 0,0.25f,0,0.25f,0, 0,0,-0.1875f,0,0};

    auto f1 = cv::cuda::createLinearFilter(CV_32F, CV_32F, cv::Mat(5, 5, CV_32F, k1));
    auto f2 = cv::cuda::createLinearFilter(CV_32F, CV_32F, cv::Mat(5, 5, CV_32F, k2));
    auto f3 = cv::cuda::createLinearFilter(CV_32F, CV_32F, cv::Mat(5, 5, CV_32F, k3));
    auto f4 = cv::cuda::createLinearFilter(CV_32F, CV_32F, cv::Mat(5, 5, CV_32F, k4));

    cv::cuda::GpuMat d_final_bgr;
    // --- Start Timing ---
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for(int i = 0; i< 100;i++)
    {
        printf("Executing demosaic, run number %d\n", i+1);
        // 1. Normalization using divide operator
        d_raw_u16.convertTo(d_raw_f32, CV_32F);
        cv::cuda::divide(d_raw_f32, 64.0f, d_raw_f32);

        // Initial mask copies
        d_raw_f32.copyTo(d_R, d_maskR);
        d_raw_f32.copyTo(d_G, d_maskG);
        d_raw_f32.copyTo(d_B, d_maskB);

        // 2. Optimized Interpolation (4 Convolutions Total)
        
        // G Interpolation (1/4)
        f1->apply(d_raw_f32, d_interp_g);
        d_interp_g.copyTo(d_G, d_maskR); 
        d_interp_g.copyTo(d_G, d_maskB); 

        // R & B shared filters (2/4 and 3/4)
        f2->apply(d_raw_f32, d_interp_f2); // rb_at_g_rbbr
        f3->apply(d_raw_f32, d_interp_f3); // rb_at_g_brrb
        
        // Cross filter (4/4)
        f4->apply(d_raw_f32, d_interp_f4); // rb_at_gr_bbrr

        // Update Red Channel
        d_interp_f2.copyTo(d_R, d_maskGr); // Red at Gr
        d_interp_f3.copyTo(d_R, d_maskGb); // Red at Gb
        d_interp_f4.copyTo(d_R, d_maskB);  // Red at B

        // Update Blue Channel
        d_interp_f2.copyTo(d_B, d_maskGb); // Blue at Gb
        d_interp_f3.copyTo(d_B, d_maskGr); // Blue at Gr
        d_interp_f4.copyTo(d_B, d_maskR);  // Blue at R

        // --- Post-Processing ---
        float max_val = (float)((1 << bit_depth) - 1);
        cv::cuda::multiply(d_R, cv::Scalar(gain * r_gain), d_R);
        cv::cuda::multiply(d_G, cv::Scalar(gain), d_G);
        cv::cuda::multiply(d_B, cv::Scalar(gain * b_gain), d_B);

        std::vector<cv::cuda::GpuMat> channels = {d_R, d_G, d_B}; // Corrected order to BGR for CV_8UC3
        cv::cuda::merge(channels, d_final_bgr);
        d_final_bgr.convertTo(d_final_bgr, CV_8UC3, 255.0 / max_val);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Demosaic Average Execution Time (openCV): %.3f ms\n", ms/100.0);

    cv::Mat h_final;
    d_final_bgr.download(h_final);
    stbi_write_png("demosaic_opencv.png", width, height, 3, h_final.data, width * 3);

    return 0;
}