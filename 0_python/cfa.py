"""
File: cfa.py
Description: Implements CFA interpolation (RGGB) with integrated bit-shifting and 8-bit normalization.
Code / Paper Reference: https://www.ipol.im/pub/art/2011/g_mhcd/article.pdf
Author: 10xEngineers Pvt Ltd (Updated)
------------------------------------------------------------
"""
import time
import numpy as np
from scipy.signal import correlate2d
from PIL import Image
from pathlib import Path

class Demosaic:
    """CFA Interpolation - Demosaicing (RGGB specific with zero padding)"""

    def __init__(self, img, sensor_info, shift_bits=0):
        # Initial image is stored as provided by the loader
        self.img = img.copy()
        self.bit_depth = sensor_info["bit_depth"]
        self.gain = sensor_info["gain"]
        self.r_gain = sensor_info["r_gain"]
        self.b_gain = sensor_info["b_gain"]
        self.shift_bits = shift_bits
        
        # Initialize filter kernels once
        self._initialize_filters()
    
    def _initialize_filters(self):
        """Initialize all 5x5 filter kernels for CFA interpolation."""
        # Green channel at Red & Blue pixel locations
        self.g_at_r_and_b = (
            np.float32(
                [
                    [0, 0, -1, 0, 0],
                    [0, 0, 2, 0, 0],
                    [-1, 2, 4, 2, -1],
                    [0, 0, 2, 0, 0],
                    [0, 0, -1, 0, 0],
                ]
            )
            * 0.125
        )

        # Red at Green in Red row & Blue at Green in Blue row
        self.r_at_gr_and_b_at_gb = (
            np.float32(
                [
                    [0, 0, 0.5, 0, 0],
                    [0, -1, 0, -1, 0],
                    [-1, 4, 5, 4, -1],
                    [0, -1, 0, -1, 0],
                    [0, 0, 0.5, 0, 0],
                ]
            )
            * 0.125
        )

        self.r_at_gb_and_b_at_gr = np.transpose(self.r_at_gr_and_b_at_gb)

        # Red at Blue & Blue at Red pixel locations
        self.r_at_b_and_b_at_r = (
            np.float32(
                [
                    [0, 0, -1.5, 0, 0],
                    [0, 2, 0, 2, 0],
                    [-1.5, 0, 6, 0, -1.5],
                    [0, 2, 0, 2, 0],
                    [0, 0, -1.5, 0, 0],
                ]
            )
            * 0.125
        )

    def apply_cfa(self):
        """
        Demosaicing the given raw image using hardcoded RGGB masks.
        Includes bit shifting at start and uint8 conversion at end.
        """
        # 1. Perform Right Shift before any computation
        raw_shifted = self.img
        if self.shift_bits > 0:
            raw_shifted = np.right_shift(self.img, self.shift_bits)

        raw_in = np.float32(raw_shifted)
        h, w = raw_in.shape

        # Create masks for RGGB Bayer pattern
        mask_r = np.zeros((h, w), dtype=bool)
        mask_gr = np.zeros((h, w), dtype=bool)
        mask_gb = np.zeros((h, w), dtype=bool)
        mask_b = np.zeros((h, w), dtype=bool)

        mask_r[0::2, 0::2] = True
        mask_gr[0::2, 1::2] = True
        mask_gb[1::2, 0::2] = True
        mask_b[1::2, 1::2] = True

        mask_g = mask_gr | mask_gb
        demos_out = np.empty((h, w, 3))

        r_channel = raw_in * mask_r
        g_channel = raw_in * mask_g
        b_channel = raw_in * mask_b

        conv_params = {"mode": "same", "boundary": "fill", "fillvalue": 0}

        # Green Channel Interpolation
        g_interp = correlate2d(raw_in, self.g_at_r_and_b, **conv_params)
        g_channel = np.where(mask_r, g_interp, g_channel)
        g_channel = np.where(mask_b, g_interp, g_channel)

        # Red and Blue Channel Interpolation
        rb_at_g_rbbr = correlate2d(raw_in, self.r_at_gr_and_b_at_gb, **conv_params)
        rb_at_g_brrb = correlate2d(raw_in, self.r_at_gb_and_b_at_gr, **conv_params)
        rb_at_gr_bbrr = correlate2d(raw_in, self.r_at_b_and_b_at_r, **conv_params)

        r_channel = np.where(mask_gr, rb_at_g_rbbr, r_channel)
        r_channel = np.where(mask_gb, rb_at_g_brrb, r_channel)
        r_channel = np.where(mask_b, rb_at_gr_bbrr, r_channel)

        b_channel = np.where(mask_gb, rb_at_g_rbbr, b_channel)
        b_channel = np.where(mask_gr, rb_at_g_brrb, b_channel)
        b_channel = np.where(mask_r, rb_at_gr_bbrr, b_channel)

        # Apply sensor gain
        demos_out[:, :, 0] = r_channel * self.gain * self.r_gain
        demos_out[:, :, 1] = g_channel * self.gain
        demos_out[:, :, 2] = b_channel * self.gain * self.b_gain

        # 2. Convert to uint8 (Normalize by bit depth, clip, and cast)
        max_val = 2**self.bit_depth - 1
        demos_out = (demos_out / max_val) * 255
        demos_out = np.clip(demos_out, 0, 255).astype(np.uint8)
        
        return demos_out

    def execute(self):
        """Applying demosaicing to Bayer image"""
        cfa_out = self.apply_cfa()
        self.img = cfa_out
        return self.img

# --- Helper Functions ---

def load_raw_image(image_path, width, height):
    """Loads a raw image from a path and reshapes it without shifting."""
    path_object = Path(image_path)
    if path_object.suffix.lower() == ".raw":
        raw_data = np.fromfile(str(path_object.resolve()), dtype=np.uint16).reshape((height, width))
        return raw_data
    else:
        raise ValueError("File must be a .raw format")

def save_png_img(demos_img, output_filename):
    """Saves the 8-bit image array as a PNG file."""
    result_image = Image.fromarray(demos_img)
    result_image.save(output_filename)

# --- Main Execution ---

def main():
    print("Start execution")
    
    metadata = {
        "bayer_pattern": "rggb",
        "bit_depth": 10,
        "width": 3328,
        "height": 2464,
        "raw_file": "0_python/file.raw",
        "output_file": "demosaic_python.png",
        "gain": 4.5,
        "r_gain": 1.32,
        "b_gain": 1.4,
        "shift_bits": 6
    }
    
    if not Path(metadata["raw_file"]).exists():
        print(f"Error: {metadata['raw_file']} not found.")
        return

    # 1. Load the raw data (no bit shifting here)
    img = load_raw_image(metadata["raw_file"], metadata["width"], metadata["height"])

    # 2. Demosaic (Shifting and 8-bit conversion handled internally)
    demosaic_instance = Demosaic(img=img, sensor_info=metadata, shift_bits=metadata["shift_bits"])

    start_time = time.perf_counter()
    demos_img = demosaic_instance.execute()
    end_time = time.perf_counter()
    
    print(f"Demosaic Execution Time (python): {end_time - start_time:.4f} seconds")

    # 3. Save the result (already uint8)
    save_png_img(demos_img, metadata["output_file"])

if __name__ == "__main__":
    main()