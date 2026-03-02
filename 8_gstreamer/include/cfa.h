#ifndef CFA_H
#define CFA_H

#include <cuda_runtime.h>

class cfa {
private:
    int width;
    int height;
    int bit_depth;
    int shift_bits;
    float gain;
    float r_gain;
    float b_gain;
    
    // GPU memory pointers
    uint16_t* d_raw;
    uchar4* d_out_rgbx;  // Changed to uchar4* for RGBX (4 channels)
    
    // Grid and block dimensions
    dim3 block;
    dim3 grid;
    float max_val_inv;
    
public:
    // Constructor
    cfa(int img_width, int img_height, int depth = 10, 
        int s_bits = 6, float g = 5.0f, float r_g = 1.2f, float b_g = 1.35f);
    
    // Destructor
    ~cfa();
    
    // Execute demosaic kernel
    void execute(const uint16_t* gpu_raw_image_ptr);
    
    // Get output device pointer (RGBX format - 4 channels as uchar4)
    uchar4* get_output_ptr() const;
    
    // Get image dimensions
    int get_width() const { return width; }
    int get_height() const { return height; }
    
    // Check if memory is allocated
    bool is_memory_allocated() const { return d_raw != nullptr && d_out_rgbx != nullptr; }
    
    // Get input device pointer
    uint16_t* get_input_ptr() const { return d_raw; }
};

#endif // CFA_H