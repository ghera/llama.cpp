#include "llama-moe-stream.h"

#include "llama-mmap.h"

#include "ggml-backend.h"

llama_moe_stream::llama_moe_stream(llama_moe_stream &&) = default;
llama_moe_stream & llama_moe_stream::operator=(llama_moe_stream &&) = default;
llama_moe_stream::~llama_moe_stream() = default;

void llama_moe_stream::add(int il, int role, ggml_tensor * pool, size_t offs, size_t slice, int64_t n_expert, int64_t n_slots) {
    auto & layer = layers[il];

    GGML_ASSERT(role >= 0 && role < 3 && layer.pools[role] == nullptr);
    GGML_ASSERT(layer.n_expert == 0 || (layer.n_expert == n_expert && layer.n_slots == n_slots));

    layer.pools[role] = pool;
    layer.offs [role] = offs;
    layer.slice[role] = slice;
    layer.n_expert    = n_expert;
    layer.n_slots     = n_slots;

    if (layer.slot_expert.empty()) {
        layer.slot_expert.assign(n_slots,  -1);
        layer.expert_slot.assign(n_expert, -1);
        layer.slot_used.assign(n_slots, 0);
    }
}

static void llama_moe_stream_load_expert(llama_moe_stream_layer & layer, int32_t slot, int32_t expert) {
    llama_moe_stream * ms = layer.stream;

    if (!ms->file) {
        ms->file.reset(new llama_file(ms->path.c_str(), "rb"));
    }

    for (int r = 0; r < 3; r++) {
        ggml_tensor * pool = layer.pools[r];
        if (!pool) {
            continue;
        }
        GGML_ASSERT(ggml_backend_buffer_is_host(pool->buffer));

        ms->file->seek(layer.offs[r] + (size_t) expert*layer.slice[r], SEEK_SET);
        ms->file->read_raw((char *) pool->data + (size_t) slot*layer.slice[r], layer.slice[r]);
    }
}

static void llama_moe_stream_remap_op(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }

    auto & layer = *(llama_moe_stream_layer *) userdata;

    const int64_t n = ggml_nelements(a);

    const int32_t * ids = (const int32_t *) a->data;
    int32_t       * out = (int32_t       *) dst->data;

    layer.clock++;

    // resolve (and fetch) each unique expert once; stamping slot_used with the
    // current clock also pins this batch's experts against eviction below
    for (int64_t i = 0; i < n; i++) {
        const int32_t expert = ids[i];
        GGML_ASSERT(expert >= 0 && expert < layer.n_expert);

        int32_t slot = layer.expert_slot[expert];
        if (slot < 0) {
            // evict the least recently used slot not touched by this batch
            int32_t best = -1;
            for (int32_t s = 0; s < layer.n_slots; s++) {
                if (layer.slot_used[s] == layer.clock) {
                    continue;
                }
                if (best < 0 || layer.slot_used[s] < layer.slot_used[best]) {
                    best = s;
                }
            }
            if (best < 0) {
                GGML_ABORT("llama_moe_stream: batch needs more than %d expert slots - increase --stream-moe or reduce ubatch size",
                        (int) layer.n_slots);
            }
            if (layer.slot_expert[best] >= 0) {
                layer.expert_slot[layer.slot_expert[best]] = -1;
            }
            llama_moe_stream_load_expert(layer, best, expert);
            layer.slot_expert[best]   = expert;
            layer.expert_slot[expert] = best;
            slot = best;
        }
        layer.slot_used[slot] = layer.clock;

        out[i] = slot;
    }
}

ggml_tensor * llama_moe_stream::remap(ggml_context * ctx, ggml_tensor * ids, int il) {
    auto it = layers.find(il);
    if (it == layers.end()) {
        return nullptr;
    }
    it->second.stream = this;

    return ggml_map_custom1(ctx, ids, llama_moe_stream_remap_op, 1, &it->second);
}
