#!/bin/bash

# --- Configuration ---
PLUGIN_NAME="cudaisp" # The short name for the GStreamer element
SHARED_LIB="libgst${PLUGIN_NAME}.so" # The resulting library name (libgstcudaisp.so)
CPP_FILE="${PLUGIN_NAME}.cpp" # The main C++ file to compile (cudaisp.cpp)
# List of CUDA source files - only cfa.cu for basic implementation
C_FILES=(src/cfa.cu)
OBJECT_DIR="obj"

# NVCC Flags for compilation: ONLY include -c and -Xcompiler -fPIC
NVCC_COMPILE_FLAGS="-c -Xcompiler -fPIC -std=c++17"
# CXX Flags (for cudaisp.cpp)
CXX_FLAGS="-c -o ${OBJECT_DIR}/${PLUGIN_NAME}.o -fPIC -std=c++17 $(pkg-config --cflags gstreamer-1.0 gstreamer-base-1.0 glib-2.0) -I./include -I/usr/local/cuda/include"

# NEW LINKER FLAGS: Designed for NVCC linking
LINKER_FLAGS="-shared -Xlinker -no-as-needed $(pkg-config --libs gstreamer-1.0 gstreamer-base-1.0 glib-2.0) -lstdc++ -o ${SHARED_LIB}"


# --- Aggressive Cleanup ---
echo "--- Performing Aggressive Cleanup ---"
# Remove old shared library from current dir and the gstreamer cache
rm -f ./${SHARED_LIB}
rm -f ~/.cache/gstreamer-1.0/registry.*
echo "Cleaned GStreamer registry cache."
echo "Removed ${SHARED_LIB} from current directory."


# --- Compilation Steps ---

# 1. Compile CUDA source files (.cu) to object files (.o) using NVCC
echo "--- Compiling CUDA source files (.cu) to object files (.o) using NVCC ---"
mkdir -p ${OBJECT_DIR}
for file in "${C_FILES[@]}"; do
    obj_file="${OBJECT_DIR}/$(basename "${file/.cu/.o}")"
    if [ ! -f "$obj_file" ] || [ "$file" -nt "$obj_file" ]; then
        echo "Compiling $file..."
        # Compile with CUDA architecture for broad compatibility - including older architectures
        /usr/local/cuda/bin/nvcc ${NVCC_COMPILE_FLAGS} \
            -gencode arch=compute_87,code=sm_87 \
            -o ${obj_file} $file -I./include || { echo "Error during NVCC compilation of $file"; exit 1; }
    else
        echo "Skipping $file: ${obj_file} is up to date."
    fi
done

# 2. Compile GStreamer plugin file (cudaisp.cpp) using G++ (Host code compilation)
echo "--- Compiling GStreamer plugin file (${CPP_FILE}) using G++ ---"
g++ ${CXX_FLAGS} ${CPP_FILE} || { echo "Error during G++ compilation of ${CPP_FILE}"; exit 1; }

# 3. Linking object files to create libgstcudaisp.so using NVCC (CUDA linking)
echo "--- Linking object files to create ${SHARED_LIB} using NVCC ---"
OBJECT_FILES="${OBJECT_DIR}/${PLUGIN_NAME}.o"
for file in "${C_FILES[@]}"; do
    OBJECT_FILES="${OBJECT_FILES} ${OBJECT_DIR}/$(basename "${file/.cu/.o}")"
done

# Use nvcc for linking to automatically resolve CUDA runtime symbols
/usr/local/cuda/bin/nvcc ${LINKER_FLAGS} ${OBJECT_FILES} || { echo "Error during NVCC linking"; exit 1; }

echo "--- Compilation Successful! Plugin ${SHARED_LIB} created. ---"
echo ""
echo "--- Installation Instructions ---"
echo "To install the plugin system-wide, copy it to the GStreamer plugins directory:"
echo "sudo cp ${SHARED_LIB} /usr/lib/x86_64-linux-gnu/gstreamer-1.0/"
echo ""
echo "Or to use it locally, set the GST_PLUGIN_PATH environment variable:"
echo "export GST_PLUGIN_PATH=\$GST_PLUGIN_PATH:$(pwd)"
echo ""
echo "Test the plugin with:"
echo "gst-inspect-1.0 ./${SHARED_LIB}"