#pragma once

#include "ggml.h"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

struct llama_file;

// disk-backed streaming of MoE routed-expert weights: the model keeps a bounded
// pool of expert slices per expert tensor and streams the rest from the GGUF on
// demand, keyed by the router's expert selection

struct llama_moe_stream_layer {
    // gate, up, down expert pools; each [ne0, ne1, n_slots], same slot holds the
    // same expert in all three so one remap serves the layer's mul_mat_id calls
    ggml_tensor * pools[3] = { nullptr, nullptr, nullptr };
    size_t        offs [3] = { 0, 0, 0 }; // file offset of expert 0
    size_t        slice[3] = { 0, 0, 0 }; // bytes per expert slice

    int64_t n_expert = 0;
    int64_t n_slots  = 0;

    // residency state, mutated single-threaded by the remap op
    std::vector<int32_t>  slot_expert; // per slot, -1 = free
    std::vector<int32_t>  expert_slot; // per expert, -1 = not resident
    std::vector<uint64_t> slot_used;   // lru stamps
    uint64_t clock = 0;

    struct llama_moe_stream * stream = nullptr;
};

struct llama_moe_stream {
    std::map<int, llama_moe_stream_layer> layers; // by block id

    std::string path; // model file holding the expert weights

    std::unique_ptr<llama_file> file; // lazy-opened on first miss

    std::vector<uint8_t> scratch; // bounce buffer for non-host pools

    llama_moe_stream() = default;
    llama_moe_stream(llama_moe_stream &&);
    llama_moe_stream & operator=(llama_moe_stream &&);
    ~llama_moe_stream();

    bool enabled() const { return !layers.empty(); }

    void add(int il, int role, ggml_tensor * pool, size_t offs, size_t slice, int64_t n_expert, int64_t n_slots);

    // build the op translating expert ids to pool slot ids for layer il,
    // fetching missing slices from disk at eval time; nullptr when the layer
    // is not streamed
    ggml_tensor * remap(ggml_context * ctx, ggml_tensor * ids, int il);
};
