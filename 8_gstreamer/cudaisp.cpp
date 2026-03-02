#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <glib.h>
#include <iostream>
#include <string>
#include <stdexcept> 
#include <cuda_runtime.h>
#include <cstdint> // For uint8_t, uint16_t, int64_t

// Include the CFA class header
#include "cfa.h"

// Define the standard GStreamer plugin macros
#define PACKAGE "cudaisp" 
#define VERSION "1.0"
#define LICENSE "LGPL"
#define ORIGIN "https://gstreamer.freedesktop.org"
#define DESCRIPTION "Basic CUDA CFA demosaic plugin for Bayer to RGBx conversion"
#define AUTHOR "Talha Tahir <talha.tahir@10xengineers.com>"

// --- 1. Define C-style Structures for GObject (REQUIRED) ---

// GstCudaisp instance structure (inherits from GstBaseTransform)
struct GstCudaisp : public GstBaseTransform {
    guint frame_count;
    gint width;
    gint height;
    
    // --- CFA Class Object ---
    cfa *cfa_processor;
    
    // Buffer sizes
    size_t in_size_bytes;      // Input buffer size in bytes
    size_t out_size_bytes;     // Output buffer size in bytes
};

// GstCudaispClass structure (inherits from GstBaseTransformClass)
struct GstCudaispClass : public GstBaseTransformClass {
    // No new class members needed here
};

// --- 2. Forward Declarations for GObject Boilerplate ---
GType gst_cudaisp_get_type(void); 
static void gst_cudaisp_class_init(GstCudaispClass *klass);
static void gst_cudaisp_init(GstCudaisp *plugin);
static void gst_cudaisp_dispose(GObject *object); 
static void gst_cudaisp_finalize(GObject *object); 

// --- 3. C++ Class Implementation for Logic and Wrappers ---

class GstCudaispImpl {
private:
    GstFlowReturn do_transform(GstCudaisp *self, GstBuffer *inbuf, GstBuffer *outbuf);

public:
    static GstFlowReturn transform_static(GstBaseTransform *trans, GstBuffer *inbuf, GstBuffer *outbuf);
    static gboolean set_caps_static(GstBaseTransform *trans, GstCaps *incaps, GstCaps *outcaps);
    static gboolean get_unit_size_static(GstBaseTransform *trans, GstCaps *caps, gsize *size);
    static GstCaps* transform_caps_static(GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps, GstCaps *filter);
    static GstCaps* fixate_caps_static(GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps, GstCaps *othercaps);
};

// --- 4. GObject/GStreamer Boilerplate (C-style macros for C++) ---
#define GST_TYPE_CUDAISP (gst_cudaisp_get_type())
#define GST_CUDAISP(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_CUDAISP, GstCudaisp))
G_DEFINE_TYPE_WITH_CODE(GstCudaisp, gst_cudaisp, GST_TYPE_BASE_TRANSFORM, {});

// Pad templates
static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE(
    "sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS("video/x-bayer, format=(string)rggb10le, width=(int)[1,MAX], height=(int)[1,MAX], framerate=(fraction)[0,MAX]")
);

static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE(
    "src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS("video/x-raw, format=(string)RGBx, width=(int)[1,MAX], height=(int)[1,MAX], framerate=(fraction)[0,MAX]")
);

// --- 5. Implementations of Core Methods ---

// Cleanup function: releases the CFA object
static void gst_cudaisp_dispose(GObject *object) {
    GstCudaisp *plugin = GST_CUDAISP(object);
    
    if (plugin->cfa_processor) {
        g_print("=== CUDAISP PLUGIN DISPOSE: Deleting CFA object ===\n");
        delete plugin->cfa_processor;
        plugin->cfa_processor = nullptr;
    }
    
    // Always chain up to the parent dispose function
    G_OBJECT_CLASS(gst_cudaisp_parent_class)->dispose(object);
}

// Finalize function: called after dispose
static void gst_cudaisp_finalize(GObject *object) {
    GstCudaisp *plugin = GST_CUDAISP(object);
    
    g_print("=== CUDAISP PLUGIN FINALIZE: Cleanup ===\n");
    
    // Chain up to parent finalize
    G_OBJECT_CLASS(gst_cudaisp_parent_class)->finalize(object);
}

// The core processing function
GstFlowReturn GstCudaispImpl::do_transform(GstCudaisp *self, GstBuffer *inbuf, GstBuffer *outbuf) {
    GstMapInfo in_map = GST_MAP_INFO_INIT, out_map = GST_MAP_INFO_INIT;
    GstFlowReturn flow_status = GST_FLOW_ERROR; 
    cudaError_t cudaStatus = cudaSuccess;

    // g_print("=== CUDAISP PROCESSING FRAME %u ===\n", self->frame_count);

    // Check if CFA processor is properly initialized
    if (!self->cfa_processor) {
        GST_ERROR_OBJECT(self, "CFA processor not initialized");
        return GST_FLOW_ERROR;
    }

    // --- 1. Map input buffer and copy to device (to CFA's internal memory) ---
    if (!gst_buffer_map(inbuf, &in_map, GST_MAP_READ)) {
        GST_ERROR_OBJECT(self, "Failed to map input Bayer buffer");
        return GST_FLOW_ERROR;
    }
    
    uint16_t *h_in_bayer = (uint16_t *)in_map.data;
    // g_print("Input buffer mapped, size: %zu bytes, first few pixels: %d %d %d\n", 
    //         self->in_size_bytes, h_in_bayer[0], h_in_bayer[1], h_in_bayer[2]);
    
    // Get CFA's internal input device pointer
    uint16_t* d_in_bayer = self->cfa_processor->get_input_ptr();
    if (!d_in_bayer) {
        GST_ERROR_OBJECT(self, "CFA processor input pointer is null");
        gst_buffer_unmap(inbuf, &in_map);
        return GST_FLOW_ERROR;
    }
    // g_print("CFA input device pointer: %p\n", d_in_bayer);
    
    // Copy Bayer input from Host to Device (CFA's internal memory)
    cudaStatus = cudaMemcpy(d_in_bayer, h_in_bayer, self->in_size_bytes, cudaMemcpyHostToDevice);
    
    gst_buffer_unmap(inbuf, &in_map); // Unmap input early

    if (cudaStatus != cudaSuccess) {
        GST_ERROR_OBJECT(self, "CUDA Memcpy HtoD failed: %s", cudaGetErrorString(cudaStatus));
        return GST_FLOW_ERROR;
    }
    // g_print("CUDA HtoD copy successful\n");
    
    // --- 2. Execute the CFA demosaic pipeline ---
    // g_print("Executing CFA processor...\n");
    self->cfa_processor->execute(d_in_bayer);
    // g_print("CFA processor execution complete\n");
    
    // --- 3. Get output and copy back to host ---
    uchar4 *d_rgbx_out = self->cfa_processor->get_output_ptr();
    if (!d_rgbx_out) {
        GST_ERROR_OBJECT(self, "CFA processor output pointer is null");
        return GST_FLOW_ERROR;
    }
    
    // Map the GStreamer output buffer
    if (!gst_buffer_map(outbuf, &out_map, GST_MAP_WRITE)) {
        GST_ERROR_OBJECT(self, "Failed to map output RGBx buffer");
        return GST_FLOW_ERROR;
    }
    
    // Copy 4-channel RGBx from Device back to Host
        // Copy 4-channel RGBx (as uchar4) from Device back to Host
    // g_print("Copying output back to host, size: %zu bytes (%zu pixels as uchar4)\n", 
    //         self->out_size_bytes, self->out_size_bytes / sizeof(uchar4));
    cudaStatus = cudaMemcpy(out_map.data, d_rgbx_out, self->out_size_bytes, cudaMemcpyDeviceToHost);
    gst_buffer_unmap(outbuf, &out_map); // Unmap output

    if (cudaStatus == cudaSuccess) {
        self->frame_count++;
        flow_status = GST_FLOW_OK;
        // g_print("Frame %u processed successfully\n", self->frame_count);
        
        // Verify output data (check first few pixels)
        // uint8_t *h_out = (uint8_t*)out_map.data;
        // g_print("Output first few pixels (RGBX): %d %d %d %d, %d %d %d %d\n",
        //         h_out[0], h_out[1], h_out[2], h_out[3],
        //         h_out[4], h_out[5], h_out[6], h_out[7]);
    } else {
        GST_ERROR_OBJECT(self, "CUDA Memcpy DtoH failed: %s", cudaGetErrorString(cudaStatus));
    }

    return flow_status;
}

// Static wrapper functions
GstFlowReturn GstCudaispImpl::transform_static(GstBaseTransform *trans, GstBuffer *inbuf, GstBuffer *outbuf) {
    GstCudaisp *plugin = GST_CUDAISP(trans);
    GstCudaispImpl impl;
    return impl.do_transform(plugin, inbuf, outbuf);
}

gboolean GstCudaispImpl::set_caps_static(GstBaseTransform *trans, GstCaps *incaps, GstCaps *outcaps) {
    GstCudaisp *plugin = GST_CUDAISP(trans);
    GstStructure *in_struct;
    
    g_print("=== CUDAISP SET_CAPS CALLED (C++) ===\n");
    
    in_struct = gst_caps_get_structure(incaps, 0);
    
    if (!gst_structure_get_int(in_struct, "width", &plugin->width) ||
        !gst_structure_get_int(in_struct, "height", &plugin->height)) {
        GST_ERROR_OBJECT(plugin, "Could not get width/height from caps");
        return FALSE;
    }
    
    g_print("Configured for: %dx%d\n", plugin->width, plugin->height);
    
    // Calculate sizes
    plugin->in_size_bytes = (size_t)plugin->width * plugin->height * 2;  // rggb10le: 2 bytes per pixel
    plugin->out_size_bytes = (size_t)plugin->width * plugin->height * 4; // RGBx: 4 bytes per pixel
    
    g_print("Input size: %zu bytes, Output size: %zu bytes\n", 
            plugin->in_size_bytes, plugin->out_size_bytes);
    
    // Create CFA processor with default parameters
    try {
        if (plugin->cfa_processor) {
            delete plugin->cfa_processor;
        }
        plugin->cfa_processor = new cfa(plugin->width, plugin->height, 10, 6, 5.5f, 1.25f, 1.4f);
        g_print("CFA processor successfully created for %dx%d\n", plugin->width, plugin->height);
        
        // Verify that CFA processor allocated memory properly
        if (!plugin->cfa_processor->get_input_ptr()) {
            GST_ERROR_OBJECT(plugin, "CFA processor input pointer is null after creation");
            return FALSE;
        }
        if (!plugin->cfa_processor->get_output_ptr()) {
            GST_ERROR_OBJECT(plugin, "CFA processor output pointer is null after creation");
            return FALSE;
        }
        g_print("CFA processor pointers: input=%p, output=%p\n", 
                plugin->cfa_processor->get_input_ptr(),
                plugin->cfa_processor->get_output_ptr());
    } catch (const std::exception& e) {
        GST_ERROR_OBJECT(plugin, "Failed to initialize CFA processor: %s", e.what());
        plugin->cfa_processor = nullptr;
        return FALSE;
    }
    
    return TRUE;
}

gboolean GstCudaispImpl::get_unit_size_static(GstBaseTransform *trans, GstCaps *caps, gsize *size) {
    GstStructure *structure;
    gint width, height;
    const gchar *name;
    
    structure = gst_caps_get_structure(caps, 0);
    name = gst_structure_get_name(structure);
    
    if (gst_structure_get_int(structure, "width", &width) &&
        gst_structure_get_int(structure, "height", &height)) {
        
        if (g_str_equal(name, "video/x-raw")) {
            // RGBx output: 4 bytes per pixel
            *size = (gsize)width * height * 4;
        } else if (g_str_equal(name, "video/x-bayer")) {
            // Bayer input (rggb10le): 2 bytes per pixel (10-bit packed in 16-bit container)
            *size = (gsize)width * height * 2;
        } else {
            return FALSE;
        }
        return TRUE;
    }
    return FALSE;
}

GstCaps* GstCudaispImpl::transform_caps_static(GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps, GstCaps *filter) {
    GstCaps *result = gst_caps_new_empty();
    GstCaps *tmp = nullptr;
    guint i;
    
    for (i = 0; i < gst_caps_get_size(caps); i++) {
        GstStructure *structure_in = gst_caps_get_structure(caps, i);
        
        if (direction == GST_PAD_SINK) {
            if (gst_structure_has_name(structure_in, "video/x-bayer")) {
                GstStructure *structure_out = gst_structure_copy(structure_in);
                gst_structure_set_name(structure_out, "video/x-raw");
                gst_structure_set(structure_out, "format", G_TYPE_STRING, "RGBx", NULL);
                gst_caps_append_structure(result, structure_out);
            }
        } else {
            if (gst_structure_has_name(structure_in, "video/x-raw")) {
                GstStructure *structure_out = gst_structure_copy(structure_in);
                gst_structure_set_name(structure_out, "video/x-bayer");
                gst_structure_set(structure_out, "format", G_TYPE_STRING, "rggb10le", NULL);
                gst_caps_append_structure(result, structure_out);
            }
        }
    }
    
    if (filter) {
        tmp = gst_caps_intersect_full(result, filter, GST_CAPS_INTERSECT_FIRST);
        gst_caps_unref(result);
        result = tmp;
    }
    return result;
}

GstCaps* GstCudaispImpl::fixate_caps_static(GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps, GstCaps *othercaps) {
    return gst_caps_copy(othercaps);
}

// --- 6. GObject Class and Instance Init Implementations (C functions) ---

static void gst_cudaisp_class_init(GstCudaispClass *klass) {
    GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
    GstBaseTransformClass *transform_class = GST_BASE_TRANSFORM_CLASS(klass);
    GObjectClass *gobject_class = G_OBJECT_CLASS(klass);

    gst_element_class_set_static_metadata(
        element_class,
        "Basic CUDA CFA Plugin",
        "Filter/Video/Converter",
        DESCRIPTION,
        AUTHOR
    );

    // Add pad templates
    gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&sink_template));
    gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&src_template));

    // Override GObject dispose method for cleanup
    gobject_class->dispose = GST_DEBUG_FUNCPTR(gst_cudaisp_dispose); 
    // Override finalize method
    gobject_class->finalize = GST_DEBUG_FUNCPTR(gst_cudaisp_finalize);

    // Override base transform methods
    transform_class->transform = GST_DEBUG_FUNCPTR(GstCudaispImpl::transform_static);
    transform_class->set_caps = GST_DEBUG_FUNCPTR(GstCudaispImpl::set_caps_static);
    transform_class->get_unit_size = GST_DEBUG_FUNCPTR(GstCudaispImpl::get_unit_size_static);
    transform_class->transform_caps = GST_DEBUG_FUNCPTR(GstCudaispImpl::transform_caps_static);
    transform_class->fixate_caps = GST_DEBUG_FUNCPTR(GstCudaispImpl::fixate_caps_static);
}

static void gst_cudaisp_init(GstCudaisp *plugin) {
    g_print("=== CUDAISP PLUGIN INITIALIZED (Basic CFA Version) ===\n");
    plugin->frame_count = 0;
    plugin->width = 0;
    plugin->height = 0;
    
    // Initialize pointers to NULL
    plugin->cfa_processor = nullptr;
    plugin->in_size_bytes = 0;
    plugin->out_size_bytes = 0;
    
    // Note: CFA processor creation is deferred to set_caps when dimensions are known
}

// --- 7. Plugin Entry Point ---

static gboolean plugin_init(GstPlugin *plugin) {
    g_print("=== CUDAISP PLUGIN REGISTRATION (Basic CFA Version) ===\n");
    return gst_element_register(plugin, "cudaisp", GST_RANK_NONE, GST_TYPE_CUDAISP);
}

// GST_PLUGIN_DEFINE macro
GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    cudaisp,
    DESCRIPTION,
    plugin_init,
    VERSION,
    LICENSE,
    PACKAGE,
    ORIGIN
)