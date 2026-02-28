from pathlib import Path
import numpy as np
from PIL import Image
import time
from cfa import Demosaic

def load_raw_image(image_path, width, height):
    """
    Loads a raw image from a path and reshapes it.
    """
    path_object = Path(image_path)
    raw_path = str(path_object.resolve())
    
    if path_object.suffix.lower() == ".raw":
        raw_data = np.fromfile(raw_path, dtype=np.uint16).reshape((height, width))
        return raw_data
    else:
        raise ValueError("Unsupported file format. Please provide a .raw file.")

def save_png_img(demos_img, output_filename):
    """
    Saves a processed image array directly to a PNG file.
    """
    result_image = Image.fromarray(demos_img)
    result_image.save(output_filename)

def main():
    print("Start execution")
    
    # Configuration
    metadata = {
        "bayer_pattern": "rggb",
        "bit_depth": 10,
        "width": 3328,
        "height": 2464,
        "raw_file": "file.raw",
        "output_file": "demosaic_python.png",
        "gain": 5.0,
        "r_gain": 1.2,
        "b_gain": 1.4,
        "shift_bits": 6
    }
    
    if not Path(metadata["raw_file"]).exists():
        print(f"Error: {metadata['raw_file']} not found.")
        return

    # 1. Load (No shifting here)
    img = load_raw_image(
        metadata["raw_file"], 
        width=metadata["width"], 
        height=metadata["height"]
    )

    # 2. Demosaic (Shifting and normalization happen inside)
    demosaic_instance = Demosaic(
        img=img, 
        sensor_info=metadata
    )

    start_time = time.perf_counter()
    demos_img = demosaic_instance.execute()
    end_time = time.perf_counter()
    
    print(f"Demosaic Execution Time (python): {end_time - start_time:.4f} seconds")

    # 3. Save (No conversion here)
    save_png_img(demos_img, metadata["output_file"])

if __name__ == "__main__":
    main()