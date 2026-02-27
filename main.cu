// Include All The Necessary Libraries

// Define All GPU Kernels

// 1. Red mask kernel
        // mask_r[0::2, 0::2] = True

// 2. Green in red rows mask Kernel
        // mask_gr[0::2, 1::2] = True

// 3. Green in blue rows mask kernel
        // mask_gb[1::2, 0::2] = True

// 4. Blue mask kernel
        // mask_b[1::2, 1::2] = True

// 5. Green Mask kernel
        // mask_g = mask_gr | mask_gb

// 6. Normalize Image Kernel
        // raw_shifted = np.right_shift(self.img, self.shift_bits)

// 7. Multiply Kernel
        // r_channel = raw_in * mask_r
        // g_channel = raw_in * mask_g
        // b_channel = raw_in * mask_b

// 8. Convolution kernel
        // g_interp = correlate2d(raw_in, self.g_at_r_and_b, **conv_params)
        // rb_at_g_rbbr = correlate2d(raw_in, self.r_at_gr_and_b_at_gb, **conv_params)
        // rb_at_g_brrb = correlate2d(raw_in, self.r_at_gb_and_b_at_gr, **conv_params)
        // rb_at_gr_bbrr = correlate2d(raw_in, self.r_at_b_and_b_at_r, **conv_params)

// 9. np.where Kenrel
        // g_channel = np.where(mask_r, g_interp, g_channel)
        // g_channel = np.where(mask_b, g_interp, g_channel)
        // r_channel = np.where(mask_gr, rb_at_g_rbbr, r_channel)
        // r_channel = np.where(mask_gb, rb_at_g_brrb, r_channel)
        // r_channel = np.where(mask_b, rb_at_gr_bbrr, r_channel)
        // b_channel = np.where(mask_gb, rb_at_g_rbbr, b_channel)
        // b_channel = np.where(mask_gr, rb_at_g_brrb, b_channel)
        // b_channel = np.where(mask_r, rb_at_gr_bbrr, b_channel)

// 10. Gains and Clip Kernel
        // demos_out[:, :, 0] = r_channel * self.gain * self.r_gain
        // demos_out[:, :, 1] = g_channel * self.gain
        // demos_out[:, :, 2] = b_channel * self.gain * self.b_gain
        // max_val = 2**self.bit_depth - 1
        // demos_out = (demos_out / max_val) * 255
        // demos_out = np.clip(demos_out, 0, 255)


// Load/Read Raw Bayer Image From Disk To Memory


// Main Function
int main() {

    // 1. Initialize ISP And Image Parameters (Image Name, Image Width, Image Height, ISP Gains, Image Bit depth etc.)

    // 2. Convolitional Filter Definitions (Define all the 8 filters we have to convole over the raw bayer)

    // 3. Host And Device Pointers ( We'll refer CPU as host and GPU as device) 
    //    We'll be using the prefix (h_) to show host side pointers and the prefix (d_) to save device memory location pointers
    //    Here we'll initialize all the pointers to the memory locations (image pointer, masks pointers, 
    //    convolutional filter pointers, output pointer etc.)

    // 4. Initialize Memory on GPU (We'll allocate the required memory on the pointer locations we initialized above)

    // 5. Copy Convolutional Filters to GPU Memory Defined In Step 2
    
    // 6. Load The Raw Image on Host

    // 7. Copy Host Raw Image to Device Memory Location

    // 8. Initialize Grid and Block Size For Our Kernels

    // 9. Call The Cuda Kernels We Initialized Above

        // 9.1 Normalize The Image

        // 9.2 Demosaic The Image

        // 9.3 Apply Gains And Clip

    // 10. Copy The Final RGB Image from Device To Host
    
    // 11. Save the Image From Memory to Disk As Png File

    // 12. Free All CPU and GPU Memory

    return 0;
}