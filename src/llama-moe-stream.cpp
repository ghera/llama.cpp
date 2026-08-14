#include "llama-moe-stream.h"

#include "llama-impl.h"
#include "llama-mmap.h"

#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <thread>

#include <fcntl.h>
#include <sys/mman.h>

llama_moe_stream::llama_moe_stream(llama_moe_stream &&) = default;
llama_moe_stream & llama_moe_stream::operator=(llama_moe_stream &&) = default;

llama_moe_stream::~llama_moe_stream() {
    if (trace_f) {
        fclose(trace_f);
    }
    if (n_lookup == 0) {
        return;
    }

    fprintf(stderr, "moe-stream: teardown: policy = %s, lookups = %" PRId64 ", misses = %" PRId64 " (%.1f%%), read = %.1f MiB, io = %.1f ms (%.2f GiB/s)\n",
            lfu ? "lfu" : "lru+pins", n_lookup, n_miss, 100.0*n_miss/n_lookup,
            io_bytes/1024.0/1024.0, io_ns/1e6, io_bytes/(io_ns/1e9)/1024.0/1024.0/1024.0);

    // persist cumulative routing frequencies for the next run's pinning
    FILE * f = fopen(stats_path().c_str(), "w");
    if (!f) {
        return;
    }
    for (const auto & kv : layers) {
        for (int32_t e = 0; e < kv.second.n_expert; e++) {
            if (kv.second.freq[e] > 0) {
                fprintf(f, "%d %d %lld\n", kv.first, e, (long long) kv.second.freq[e]);
            }
        }
    }
    fclose(f);
}

// hint the kernel to read a file range ahead of time (macOS async readahead);
// a cheap advisory syscall: when the later pread arrives the bytes are already
// in the page cache, so overlap costs no threads and no staging buffers
static bool llama_moe_stream_advise_enabled(int which) {
    // 1 = decode lookahead, 2 = prefill wave lookahead; both default off,
    // measured harmful when the disk is already bandwidth-bound
    const char * env = getenv(which == 1 ? "LLAMA_MOE_ADVISE_DECODE" : "LLAMA_MOE_ADVISE_PREFILL");
    return env && atoi(env) != 0;
}

static void llama_moe_stream_advise(llama_moe_stream * ms, size_t off, size_t len) {
#if defined(F_RDADVISE)
    if (ms->advise_fd < 0) {
        ms->advise_fd = open(ms->path.c_str(), O_RDONLY);
        if (ms->advise_fd < 0) {
            return;
        }
    }
    struct radvisory ra;
    ra.ra_offset = (off_t) off;
    ra.ra_count  = (int) std::min<size_t>(len, INT_MAX);
    (void) fcntl(ms->advise_fd, F_RDADVISE, &ra);
#else
    GGML_UNUSED(ms); GGML_UNUSED(off); GGML_UNUSED(len);
#endif
}

static void llama_moe_stream_advise_expert(llama_moe_stream * ms, const llama_moe_stream_layer & layer, int32_t expert) {
    for (int r = 0; r < 3; r++) {
        if (layer.pools[r]) {
            llama_moe_stream_advise(ms, layer.offs[r] + (size_t) expert*layer.slice[r], layer.slice[r]);
        }
    }
}

void llama_moe_stream::add(int il, int role, ggml_tensor * pool, size_t offs, size_t slice, int64_t n_expert, int64_t n_slots) {
    auto & layer = layers[il];

    GGML_ASSERT(role >= 0 && role < 3 && layer.pools[role] == nullptr);
    GGML_ASSERT(layer.n_expert == 0 || (layer.n_expert == n_expert && layer.n_slots == n_slots));

    layer.il          = il;
    layer.pools[role] = pool;
    layer.offs [role] = offs;
    layer.slice[role] = slice;
    layer.n_expert    = n_expert;
    layer.n_slots     = n_slots;

    if (layer.slot_expert.empty()) {
        layer.slot_expert.assign(n_slots,  -1);
        layer.expert_slot.assign(n_expert, -1);
        layer.slot_used.assign(n_slots, 0);
        layer.freq.assign(n_expert, 0);
        layer.pinned.assign(n_expert, 0);
    }

    if (!sync) {
        sync.reset(new llama_moe_stream_sync());
    }
}

void llama_moe_stream::load_stats() {
    stats_loaded = true;

    const char * pol = getenv("LLAMA_MOE_STREAM_POLICY");
    lfu = pol && strcmp(pol, "lfu") == 0;
    fprintf(stderr, "moe-stream: eviction policy = %s\n", lfu ? "lfu" : "lru+pins");

    const char * tp = getenv("LLAMA_MOE_TRACE");
    if (tp && !trace_f) {
        trace_f = fopen(tp, "wb");
    }

    FILE * f = fopen(stats_path().c_str(), "r");
    if (!f) {
        return;
    }

    int il, expert;
    long long count;
    while (fscanf(f, "%d %d %lld", &il, &expert, &count) == 3) {
        auto it = layers.find(il);
        if (it != layers.end() && expert >= 0 && expert < it->second.n_expert) {
            it->second.freq[expert] += count;
        }
    }
    fclose(f);

    int64_t n_pinned = 0;

    for (auto & kv : layers) {
        auto & layer = kv.second;

        // pin the hottest experts into at most 60% of the slots, always leaving
        // an lru tail of 64 slots for a full 8-token ubatch expert-union
        std::vector<int32_t> order;
        for (int32_t e = 0; e < layer.n_expert; e++) {
            if (layer.freq[e] > 0) {
                order.push_back(e);
            }
        }
        std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return layer.freq[a] > layer.freq[b]; });

        const int64_t k = std::min<int64_t>(std::min<int64_t>(layer.n_slots*6/10, layer.n_slots - 64), (int64_t) order.size());
        for (int64_t i = 0; i < k; i++) {
            layer.pinned[order[i]] = 1;
        }
        n_pinned += std::max<int64_t>(k, 0);
    }

    if (n_pinned > 0) {
        fprintf(stderr, "moe-stream: pinned %" PRId64 " hot experts from %s\n", n_pinned, stats_path().c_str());
    }

    // wire the pools so memory pressure cannot silently page them out and
    // double the streaming IO; on failure keep whatever locked so far
    // (LLAMA_MOE_MLOCK=0 disables)
    const char * mlock_env = getenv("LLAMA_MOE_MLOCK");
    if (mlock_env && atoi(mlock_env) == 0) {
        return;
    }
    size_t locked = 0;
    bool   failed = false;
    for (auto & kv : layers) {
        for (int r = 0; r < 3 && !failed; r++) {
            ggml_tensor * pool = kv.second.pools[r];
            if (!pool || !pool->data) {
                continue;
            }
            if (mlock(pool->data, ggml_nbytes(pool)) != 0) {
                failed = true;
                break;
            }
            locked += ggml_nbytes(pool);
        }
        if (failed) {
            break;
        }
    }
    fprintf(stderr, "moe-stream: mlocked %.1f MiB of expert pools%s\n",
            locked/1024.0/1024.0, failed ? " (partial, lock limit reached)" : "");
}

// drain the published miss list; runs concurrently on every ggml thread, each
// with its own file handle and bounce buffer
static void llama_moe_stream_fetch(llama_moe_stream_layer & layer) {
    static thread_local std::unique_ptr<llama_file> file;
    static thread_local std::vector<uint8_t> scratch;

    llama_moe_stream * ms = layer.stream;
    auto & sync = *ms->sync;

    if (!file) {
        file.reset(new llama_file(ms->path.c_str(), "rb"));
    }

    for (;;) {
        const int32_t i = sync.next.fetch_add(1);
        if (i >= (int32_t) sync.pending.size()) {
            break;
        }
        const int32_t slot   = sync.pending[i].slot;
        const int32_t expert = sync.pending[i].expert;
        const int32_t r      = sync.pending[i].role;

        ggml_tensor * pool = layer.pools[r];

        file->seek(layer.offs[r] + (size_t) expert*layer.slice[r], SEEK_SET);

        if (ggml_backend_buffer_is_host(pool->buffer)) {
            file->read_raw((char *) pool->data + (size_t) slot*layer.slice[r], layer.slice[r]);
        } else {
            // non-host pool (e.g. Metal unified memory): bounce through host
            // and let the backend place the bytes
            scratch.resize(layer.slice[r]);
            file->read_raw(scratch.data(), layer.slice[r]);
            ggml_backend_tensor_set(pool, scratch.data(), (size_t) slot*layer.slice[r], layer.slice[r]);
        }

        sync.done.fetch_add(1);
    }
}

static void llama_moe_stream_remap_op(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata) {

    auto & layer = *(llama_moe_stream_layer *) userdata;

    {
        // thread 0 publishes this node's miss list only after every thread of
        // the node has checked in, and each worker snapshots seq before
        // checking in: the publication it then waits for is necessarily this
        // node's. the ggml barrier between nodes keeps generations apart.
        auto & sync = *layer.stream->sync;
        if (ith != 0) {
            const uint64_t seq0 = sync.seq.load(std::memory_order_acquire);
            sync.arrived.fetch_add(1);
            while (sync.seq.load(std::memory_order_acquire) == seq0) {
                std::this_thread::yield();
            }
            llama_moe_stream_fetch(layer);
            return;
        }
        sync.arrived.fetch_add(1);
        while (sync.arrived.load(std::memory_order_acquire) < nth) {
            std::this_thread::yield();
        }
        sync.arrived.store(0);
    }

    GGML_ASSERT(a->type == GGML_TYPE_I32);

    // a can be a strided view (ggml_top_k over a full argsort): address rows
    // through nb[1] instead of assuming contiguous data
    const int64_t n_used   = a->ne[0];
    const int64_t n_tokens = ggml_nrows(a);

    // stats checkpoint roughly every 16k activations (~50 tokens on 40x8 routing)
    llama_moe_stream * ms = layer.stream;
    if (ms->trace_f) {
        // per-token records, token-major; phase 1 marks single-token (decode)
        for (int64_t t = 0; t < n_tokens; t++) {
            const int32_t * ids = (const int32_t *) ((const char *) a->data + t*a->nb[1]);
            const uint32_t hdr[3] = { (uint32_t) layer.il, n_tokens > 1 ? 0u : 1u, (uint32_t) n_used };
            fwrite(hdr, 4, 3, ms->trace_f);
            fwrite(ids, 4, n_used, ms->trace_f);
        }
    }
    if (ms->n_lookup/16384 != (ms->n_lookup + n_used*n_tokens)/16384) {
        fprintf(stderr, "moe-stream: lookups = %" PRId64 ", misses = %" PRId64 " (%.1f%%), read = %.1f MiB, io = %.1f ms (%.2f GiB/s)\n",
                ms->n_lookup, ms->n_miss, 100.0*ms->n_miss/ms->n_lookup,
                ms->io_bytes/1024.0/1024.0, ms->io_ns/1e6,
                ms->io_ns > 0 ? ms->io_bytes/(ms->io_ns/1e9)/1024.0/1024.0/1024.0 : 0.0);
    }
    ms->n_lookup += n_used*n_tokens;

    auto & sync = *ms->sync;
    sync.pending.clear();

    if (layer.clobbered) {
        // wave prefill emptied the pool: preload the pinned hot set in one
        // batch instead of paying it back as scattered decode misses
        layer.clobbered = false;
        int32_t slot = 0;
        for (int32_t e = 0; e < layer.n_expert; e++) {
            if (!layer.pinned[e]) {
                continue;
            }
            layer.slot_expert[slot] = e;
            layer.expert_slot[e]    = slot;
            layer.slot_used[slot]   = UINT64_MAX;
            for (int32_t r = 0; r < 3; r++) {
                if (layer.pools[r]) {
                    sync.pending.push_back({slot, e, r});
                    ms->io_bytes += layer.slice[r];
                }
            }
            ms->n_miss++;
            slot++;
        }
    }

    layer.clock++;

    // resolve each unique expert once; stamping slot_used with the
    // current clock also pins this batch's experts against eviction below
    for (int64_t t = 0; t < n_tokens; t++) {
        const int32_t * ids = (const int32_t *) ((const char *) a->data + t*a->nb[1]);
        int32_t       * out = (int32_t       *) ((char       *) dst->data + t*dst->nb[1]);

        for (int64_t i = 0; i < n_used; i++) {
            const int32_t expert = ids[i];
            GGML_ASSERT(expert >= 0 && expert < layer.n_expert);

            layer.freq[expert]++;

            int32_t slot = layer.expert_slot[expert];
            if (slot < 0) {
                // evict a slot not touched by this batch: lru takes the oldest
                // stamp, lfu the least-used resident expert (free slots first;
                // freq already counts this batch, pins have no lfu protection)
                int32_t best = -1;
                uint64_t best_score = 0;
                for (int32_t s = 0; s < layer.n_slots; s++) {
                    if (layer.slot_used[s] == layer.clock) {
                        continue;
                    }
                    const uint64_t score = ms->lfu
                        ? (layer.slot_expert[s] < 0 ? 0 : (uint64_t) layer.freq[layer.slot_expert[s]] + 1)
                        : layer.slot_used[s];
                    if (best < 0 || score < best_score) {
                        best       = s;
                        best_score = score;
                    }
                }
                if (best < 0) {
                    GGML_ABORT("llama_moe_stream: batch needs more than %d expert slots - increase --stream-moe or reduce ubatch size",
                            (int) layer.n_slots);
                }
                if (layer.slot_expert[best] >= 0) {
                    layer.expert_slot[layer.slot_expert[best]] = -1;
                }
                for (int32_t r = 0; r < 3; r++) {
                    if (layer.pools[r]) {
                        sync.pending.push_back({best, expert, r});
                        ms->io_bytes += layer.slice[r];
                    }
                }
                ms->n_miss++;
                layer.slot_expert[best]   = expert;
                layer.expert_slot[expert] = best;
                slot = best;
            }
            layer.slot_used[slot] = layer.pinned[expert] ? UINT64_MAX : layer.clock;

            out[i] = slot;
        }
    }

    // remember the last token's routing: it predicts the next token's ids
    // for this layer well enough to drive kernel readahead
    {
        const int32_t * ids = (const int32_t *) ((const char *) a->data + (n_tokens - 1)*a->nb[1]);
        layer.last_ids.assign(ids, ids + n_used);
    }

    // publish the miss list to the waiting threads, join the fetch, then wait
    // for the pool slices to be in place before mul_mat_id consumes them
    const auto t_start = std::chrono::steady_clock::now();

    sync.next.store(0);
    sync.done.store(0);
    sync.seq.fetch_add(1, std::memory_order_release);

    // while this layer's fetch runs, hint the kernel about the successor
    // layer's likely misses (the previous token's routing there); at the last
    // layer this warms layer 0 for the next token
    if (llama_moe_stream_advise_enabled(1)) {
        auto it = ms->layers.upper_bound(layer.il);
        if (it == ms->layers.end()) {
            it = ms->layers.begin();
        }
        const auto & next = it->second;
        for (const int32_t e : next.last_ids) {
            if (next.expert_slot[e] < 0) {
                llama_moe_stream_advise_expert(ms, next, e);
            }
        }
    }

    llama_moe_stream_fetch(layer);

    while (sync.done.load(std::memory_order_acquire) < (int32_t) sync.pending.size()) {
        std::this_thread::yield();
    }

    if (!sync.pending.empty()) {
        ms->io_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t_start).count();
    }
}

static void llama_moe_stream_wave_op(ggml_tensor * dst, const ggml_tensor * a, const ggml_tensor * b, int ith, int nth, void * userdata) {
    GGML_UNUSED(b); // dataflow-only dependency on the previous wave's output

    auto & ud    = *(llama_moe_stream_layer::wave_ud *) userdata;
    auto & layer = *ud.layer;

    llama_moe_stream * ms = layer.stream;

    {
        // same arrival handshake as the lru remap op (see there)
        auto & sync = *ms->sync;
        if (ith != 0) {
            const uint64_t seq0 = sync.seq.load(std::memory_order_acquire);
            sync.arrived.fetch_add(1);
            while (sync.seq.load(std::memory_order_acquire) == seq0) {
                std::this_thread::yield();
            }
            llama_moe_stream_fetch(layer);
            return;
        }
        sync.arrived.fetch_add(1);
        while (sync.arrived.load(std::memory_order_acquire) < nth) {
            std::this_thread::yield();
        }
        sync.arrived.store(0);
    }

    GGML_ASSERT(a->type == GGML_TYPE_I32);

    const int64_t n_used   = a->ne[0];
    const int64_t n_tokens = ggml_nrows(a);

    GGML_ASSERT(n_used <= 8); // one reserved zero slot per row position

    const int32_t ws = (int32_t) layer.n_slots - 8; // slots n_slots-8.. hold zeros
    const int32_t e0 = ud.wave*ws;
    const int32_t e1 = std::min<int32_t>(e0 + ws, (int32_t) layer.n_expert);

    // the bulk load replaces the pool contents: the lru residency map is stale
    // from here on, the first lru remap rebuilds it
    std::fill(layer.slot_expert.begin(), layer.slot_expert.end(), -1);
    std::fill(layer.expert_slot.begin(), layer.expert_slot.end(), -1);
    std::fill(layer.slot_used.begin(),   layer.slot_used.end(),    0);
    layer.clobbered = true;

    if (ud.wave == 0) {
        layer.zeroed = false;
    }
    if (!layer.zeroed) {
        // fill the reserved slots with zeros; decode may have reused them
        // since the previous prefill graph
        for (int32_t r = 0; r < 3; r++) {
            if (layer.pools[r]) {
                ms->scratch.assign(layer.slice[r], 0);
                for (int32_t f = 0; f < 8; f++) {
                    ggml_backend_tensor_set(layer.pools[r], ms->scratch.data(), (size_t) (ws + f)*layer.slice[r], layer.slice[r]);
                }
            }
        }
        layer.zeroed = true;
    }

    auto & sync = *ms->sync;
    sync.pending.clear();

    for (int32_t e = e0; e < e1; e++) {
        for (int32_t r = 0; r < 3; r++) {
            if (layer.pools[r]) {
                sync.pending.push_back({e - e0, e, r});
                ms->io_bytes += layer.slice[r];
            }
        }
        ms->n_miss++;
    }

    const auto t_start = std::chrono::steady_clock::now();

    sync.next.store(0);
    sync.done.store(0);
    sync.seq.fetch_add(1, std::memory_order_release);

    // hint the kernel about the next wave's byte ranges (or the successor
    // layer's first wave) so its reads overlap with this wave's compute
    if (llama_moe_stream_advise_enabled(2)) {
        const llama_moe_stream_layer * nl = &layer;
        int32_t nw = ud.wave + 1;
        if (nw*ws >= (int32_t) layer.n_expert) {
            auto it = ms->layers.upper_bound(layer.il);
            if (it == ms->layers.end()) {
                it = ms->layers.begin();
            }
            nl = &it->second;
            nw = 0;
        }
        const int32_t ne0 = nw*ws;
        const int32_t ne1 = std::min<int32_t>(ne0 + ws, (int32_t) nl->n_expert);
        for (int r = 0; r < 3; r++) {
            if (nl->pools[r]) {
                llama_moe_stream_advise(ms, nl->offs[r] + (size_t) ne0*nl->slice[r], (size_t) (ne1 - ne0)*nl->slice[r]);
            }
        }
    }

    llama_moe_stream_fetch(layer);

    while (sync.done.load(std::memory_order_acquire) < (int32_t) sync.pending.size()) {
        std::this_thread::yield();
    }

    ms->io_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t_start).count();

    // remap in-wave ids to their fixed slot; out-of-wave entries point at slot
    // 0 and their routing weight is masked to zero by the caller
    for (int64_t t = 0; t < n_tokens; t++) {
        const int32_t * in  = (const int32_t *) ((const char *) a->data + t*a->nb[1]);
        int32_t       * out = (int32_t       *) ((char       *) dst->data + t*dst->nb[1]);

        for (int64_t i = 0; i < n_used; i++) {
            const int32_t expert = in[i];
            GGML_ASSERT(expert >= 0 && expert < layer.n_expert);

            out[i] = expert >= e0 && expert < e1 ? expert - e0 : ws + (int32_t) i;

            if (ud.wave == 0) {
                layer.freq[expert]++;
                ms->n_lookup++;
            }
        }
    }
}

ggml_tensor * llama_moe_stream::remap_wave(ggml_context * ctx, ggml_tensor * ids, int il, int32_t wave, ggml_tensor * dep) {
    auto it = layers.find(il);
    if (it == layers.end()) {
        return nullptr;
    }
    auto & layer = it->second;
    layer.stream = this;

    if (!stats_loaded) {
        load_stats();
    }

    if (layer.waves.empty()) {
        const int32_t n = n_waves(il);
        layer.waves.resize(n);
        for (int32_t w = 0; w < n; w++) {
            layer.waves[w] = { &layer, w };
        }
    }
    GGML_ASSERT(wave >= 0 && wave < (int32_t) layer.waves.size());

    if (!dep) {
        // first wave: no pool state to protect yet, wrap through a dummy dep
        dep = ids;
    }
    return ggml_map_custom2(ctx, ids, dep, llama_moe_stream_wave_op, GGML_N_TASKS_MAX, &layer.waves[wave]);
}

ggml_tensor * llama_moe_stream::remap(ggml_context * ctx, ggml_tensor * ids, int il) {
    auto it = layers.find(il);
    if (it == layers.end()) {
        return nullptr;
    }
    it->second.stream = this;

    if (!stats_loaded) {
        load_stats();
    }

    return ggml_map_custom1(ctx, ids, llama_moe_stream_remap_op, GGML_N_TASKS_MAX, &it->second);
}
