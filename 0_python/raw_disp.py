import time
import numpy as np
from scipy.signal import correlate2d
from PIL import Image
from pathlib import Path

def main():
    print("Start execution")
    
    metadata = {
        "bayer_pattern": "rggb",
        "bit_depth": 10,
        "width": 3328,
        "height": 2464,
        "raw_file": "0_python/file.raw",
        "output_file": "Raw_Image.png",
        "shift_bits": 6
    }


    if not Path(metadata["raw_file"]).exists():
        print(f"Error: {metadata['raw_file']} not found.")
        return

    # 1. Load the raw data (no bit shifting here)
    img = load_raw_image(metadata["raw_file"], metadata["width"], metadata["height"])

    # 1. Perform Right Shift before any computation
    raw_shifted = img
    if metadata["shift_bits"] > 0:
        raw_shifted = np.right_shift(img, metadata["shift_bits"])

    raw_in = np.float32(raw_shifted)
    h, w = raw_in.shape

    # just show this raw in as a grayscale image for reference
    raw_8bit = (raw_in / (2**metadata["bit_depth"] - 1)) * 255
    raw_8bit = np.clip(raw_8bit, 0, 255).astype(np.uint8)
    save_png_img(raw_8bit, metadata["output_file"])

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

    
if __name__ == "__main__":
    main()