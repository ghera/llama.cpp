#pragma once

#include "ggml.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
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

    int     il       = -1; // block id, for successor lookup at prefetch time
    int64_t n_expert = 0;
    int64_t n_slots  = 0;

    // residency state, mutated single-threaded by the remap op
    std::vector<int32_t>  slot_expert; // per slot, -1 = free
    std::vector<int32_t>  expert_slot; // per expert, -1 = not resident
    std::vector<uint64_t> slot_used;   // lru stamps
    uint64_t clock = 0;

    // per-expert routing frequency (this run + prior runs via the sidecar);
    // pinned experts get a max lru stamp and are never evicted
    std::vector<int64_t> freq;
    std::vector<uint8_t> pinned;

    // the previous batch's last-token routing, used to advise the kernel
    // about the next token's likely reads a layer ahead
    std::vector<int32_t> last_ids;

    struct llama_moe_stream * stream = nullptr;

    // userdata for the per-wave prefill ops, sized n_waves on first use
    struct wave_ud {
        llama_moe_stream_layer * layer;
        int32_t wave;
    };
    std::vector<wave_ud> waves;

    bool zeroed    = false; // reserved zero slots filled for the current graph
    bool clobbered = false; // wave prefill replaced the lru pool contents

    // miss->slot ids of the current remap_overlap node, copied out by
    // await_miss once the io pool signals completion
    std::vector<int32_t> miss_buf;
};

// work queue handing this node's missing slices to the idle ggml threads;
// thread 0 resolves and publishes, every thread fetches
struct llama_moe_stream_sync {
    struct item { int32_t slot, expert, role; };
    std::vector<item> pending; // one fetch per (slot, expert, matrix role)
    std::atomic<uint64_t> seq{0};     // bumped once per node publication
    std::atomic<int32_t>  next{0};    // next pending index to fetch
    std::atomic<int32_t>  done{0};    // fetched count
    std::atomic<int32_t>  arrived{0}; // threads that entered the current node
};

// persistent fetch threads for the overlap decode path: remap publishes the
// miss list and returns, the hits branch runs while these drain it
struct llama_moe_stream_io {
    std::vector<std::thread>   threads;
    std::mutex                 mtx;
    std::condition_variable    cv;
    std::atomic<uint64_t>      seq{0};
    std::atomic<bool>          quit{false};
    llama_moe_stream_layer   * layer = nullptr;
    ~llama_moe_stream_io();
};

struct llama_moe_stream {
    std::map<int, llama_moe_stream_layer> layers; // by block id

    std::string path; // model file holding the expert weights

    std::unique_ptr<llama_file> file; // lazy-opened on first miss

    std::vector<uint8_t> scratch; // bounce buffer for non-host pools

    std::unique_ptr<llama_moe_stream_sync> sync; // parallel fetch queue

    int advise_fd = -1; // dedicated fd for F_RDADVISE readahead hints

    // streaming stats, reported at teardown
    int64_t n_lookup = 0; // expert activations seen by the remap op
    int64_t n_miss   = 0; // slices fetched from disk
    int64_t io_bytes = 0;
    int64_t io_ns    = 0;

    llama_moe_stream() = default;
    llama_moe_stream(llama_moe_stream &&);
    llama_moe_stream & operator=(llama_moe_stream &&);
    ~llama_moe_stream();

    // coarse overlap (LLAMA_MOE_OVERLAP=1): decode remap splits the routed
    // FFN into resident and fetched halves so the resident mul_mat_id runs
    // while the io pool reads the misses; NOT summation-order exact, opt-in
    bool overlap = false;
    std::unique_ptr<llama_moe_stream_io> io;

    bool stats_loaded = false;

    // slot eviction policy: least-frequently-used resident expert by default
    // (wins or ties lru at every measured pool/depth, +21% at 5 GiB/12k);
    // LLAMA_MOE_STREAM_POLICY=lru restores stamps + pinned stick
    bool lfu = true;

    // routing trace sink (LLAMA_MOE_TRACE), qwell-moe-route-trace record
    // format: u32 layer | u32 phase (0 multi-token, 1 decode) | u32 n_ids | ids
    FILE * trace_f = nullptr;

    bool enabled() const { return !layers.empty(); }

    bool streamed(int il) const { return layers.count(il) > 0; }

    std::string stats_path() const { return path + ".moestats"; }

    // load prior routing frequencies from the sidecar and pin the hottest
    // experts per layer; called once at first graph build
    void load_stats();

    void add(int il, int role, ggml_tensor * pool, size_t offs, size_t slice, int64_t n_expert, int64_t n_slots);

    // build the op translating expert ids to pool slot ids for layer il,
    // fetching missing slices from disk at eval time; nullptr when the layer
    // is not streamed
    ggml_tensor * remap(ggml_context * ctx, ggml_tensor * ids, int il);

    // overlap pair: remap_overlap resolves and publishes the miss list to the
    // io pool, returning resident-slot ids immediately (misses point at the
    // reserved zero slots); await_miss joins the pool and yields the fetched
    // half. build the resident branch between the two so it overlaps the reads
    ggml_tensor * remap_overlap(ggml_context * ctx, ggml_tensor * ids, int il);
    ggml_tensor * await_miss(ggml_context * ctx, ggml_tensor * dep, int il);

    // expert-major prefill: each wave bulk-loads a contiguous expert range
    // into pool slots [0, wave_size) with a fixed mapping, once per graph; the
    // routed sum over all waves equals the single-pass result. out-of-wave
    // entries keep mul_mat_id's per-row distinct-ids invariant by pointing at
    // one of n_expert_used reserved zero-filled slots, picked by row position,
    // so their contribution vanishes as x*0 without touching the weights
    int32_t n_slots(int il) const { return (int32_t) layers.at(il).n_slots; }
    int32_t wave_size(int il) const { return (int32_t) (layers.at(il).n_slots - 8); }
    int32_t n_waves(int il) const {
        const auto & l = layers.at(il);
        return (int32_t) ((l.n_expert + l.n_slots - 8 - 1)/(l.n_slots - 8));
    }
    // dep, when given, is consumed as a dataflow-only input: the wave's bulk
    // load mutates pool tensors the scheduler cannot see, so each wave must
    // depend on the previous wave's routed output to not clobber its slots
    ggml_tensor * remap_wave(ggml_context * ctx, ggml_tensor * ids, int il, int32_t wave, ggml_tensor * dep);
};
