#pragma once

#include "ggml.h"

#include <string>
#include <vector>

// disk-backed streaming of MoE routed-expert weights: the model keeps a bounded
// pool of expert slices per expert tensor and streams the rest from the GGUF on
// demand, keyed by the router's expert selection

struct llama_moe_stream_tensor {
    ggml_tensor * pool;       // [ne0, ne1, n_slots], stands in for the full expert tensor
    size_t        offs;       // file offset of expert 0
    size_t        slice_size; // bytes per expert slice
    int64_t       n_expert;
    int64_t       n_slots;
};

struct llama_moe_stream {
    std::vector<llama_moe_stream_tensor> tensors;

    std::string path; // model file holding the expert weights

    bool enabled() const { return !tensors.empty(); }
};
