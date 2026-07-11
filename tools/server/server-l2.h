#pragma once

// durable L2 checkpoint cache for single-slot servers: whole-context state
// snapshots keyed by token prefix, so a server restart (or session switch)
// restores in seconds instead of re-prefilling minutes of context.
//
// policy rules follow the QWell/ds4 lineage: identity checked before load,
// pre-evict to budget before store, newest entry protected, failed restores
// quarantined (entry and file dropped). prototype configuration is via env:
//   LLAMA_L2_DIR         cache directory (unset = disabled)
//   LLAMA_L2_BUDGET_MIB  total budget, default 16384
//
// note: the per-sequence state path (llama_state_seq_save_file) writes an
// empty payload for hybrid-recurrent models, so this uses the whole-context
// llama_state_{save,load}_file, which does capture the recurrent state.

#include "common.h"
#include "llama.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

struct server_l2_entry {
    std::string  file;
    llama_tokens tokens;
    size_t       bytes    = 0;
    int64_t      last_hit = 0;
};

struct server_l2 {
    std::string dir;
    size_t      budget = 0;
    int64_t     clock  = 0;
    size_t      n_last_saved = 0;

    std::vector<server_l2_entry> entries;

    static constexpr size_t min_grow = 1024; // tokens between saves / to bother restoring

    bool enabled() const { return !dir.empty(); }

    std::string index_path() const { return dir + "/index.json"; }

    void init() {
        const char * d = getenv("LLAMA_L2_DIR");
        if (!d || !d[0]) {
            return;
        }
        dir = d;
        const char * b = getenv("LLAMA_L2_BUDGET_MIB");
        budget = (b ? (size_t) atoll(b) : 16384) * 1024ull * 1024ull;

        std::ifstream f(index_path());
        if (f.good()) {
            try {
                nlohmann::json j = nlohmann::json::parse(f);
                for (const auto & e : j["entries"]) {
                    server_l2_entry ent;
                    ent.file     = e["file"];
                    ent.bytes    = e["bytes"];
                    ent.last_hit = e["last_hit"];
                    ent.tokens   = e["tokens"].get<llama_tokens>();
                    clock = std::max(clock, ent.last_hit);
                    entries.push_back(std::move(ent));
                }
            } catch (const std::exception & ex) {
                fprintf(stderr, "l2: discarding unreadable index: %s\n", ex.what());
                entries.clear();
            }
        }
        fprintf(stderr, "l2: cache at %s, %zu entries, budget %.0f MiB\n",
                dir.c_str(), entries.size(), budget/1024.0/1024.0);
    }

    void save_index() {
        nlohmann::json j;
        for (const auto & e : entries) {
            j["entries"].push_back({{"file", e.file}, {"bytes", e.bytes}, {"last_hit", e.last_hit}, {"tokens", e.tokens}});
        }
        std::ofstream f(index_path());
        f << j.dump();
    }

    void drop(size_t i) {
        remove((dir + "/" + entries[i].file).c_str());
        entries.erase(entries.begin() + i);
    }

    size_t total_bytes() const {
        size_t t = 0;
        for (const auto & e : entries) {
            t += e.bytes;
        }
        return t;
    }

    static size_t common_prefix(const llama_tokens & a, const llama_tokens & b) {
        size_t i = 0;
        while (i < a.size() && i < b.size() && a[i] == b[i]) {
            i++;
        }
        return i;
    }

    // longest stored strict token-prefix of `prompt` worth restoring over
    // `have` already-cached tokens; -1 when none qualifies
    int lookup(const llama_tokens & prompt, size_t have) {
        int    best   = -1;
        size_t best_n = have + min_grow;
        for (size_t i = 0; i < entries.size(); i++) {
            const auto & t = entries[i].tokens;
            if (t.size() >= best_n && common_prefix(t, prompt) == t.size()) {
                best   = (int) i;
                best_n = t.size();
            }
        }
        return best;
    }

    // whole-context restore; quarantines the entry on failure
    bool restore(llama_context * ctx, int i, llama_tokens & out) {
        auto & e = entries[i];
        out.resize(e.tokens.size() + 8);
        size_t count = 0;
        const bool ok = llama_state_load_file(ctx, (dir + "/" + e.file).c_str(), out.data(), out.size(), &count) &&
                        count == e.tokens.size();
        if (!ok) {
            fprintf(stderr, "l2: restore of %s failed, quarantined\n", e.file.c_str());
            drop(i);
            save_index();
            return false;
        }
        out.resize(count);
        e.last_hit = ++clock;
        n_last_saved = count;
        save_index();
        fprintf(stderr, "l2: restored %zu tokens from %s\n", count, e.file.c_str());
        return true;
    }

    // called at generation end with the slot's full token cache
    void maybe_save(llama_context * ctx, const llama_tokens & tokens) {
        if (tokens.size() < n_last_saved + min_grow) {
            return;
        }
        // pre-evict to make room, assuming the new entry is at most the size
        // of the largest existing one plus growth; never evict the newest
        const size_t est = llama_state_get_size(ctx);
        while (!entries.empty() && total_bytes() + est > budget) {
            size_t oldest = 0;
            for (size_t i = 1; i < entries.size(); i++) {
                if (entries[i].last_hit < entries[oldest].last_hit) {
                    oldest = i;
                }
            }
            drop(oldest);
        }
        char name[64];
        snprintf(name, sizeof(name), "ckpt-%lld-%zu.bin", (long long) ++clock, tokens.size());
        const std::string path = dir + "/" + name;
        if (!llama_state_save_file(ctx, path.c_str(), tokens.data(), tokens.size())) {
            fprintf(stderr, "l2: save to %s failed\n", name);
            remove(path.c_str());
            return;
        }
        std::ifstream fsz(path, std::ios::binary | std::ios::ate);
        const size_t written = (size_t) fsz.tellg();
        server_l2_entry e;
        e.file     = name;
        e.tokens   = tokens;
        e.bytes    = written;
        e.last_hit = clock;
        entries.push_back(std::move(e));
        n_last_saved = tokens.size();
        save_index();
        fprintf(stderr, "l2: saved %zu tokens (%.1f MiB) to %s\n", tokens.size(), written/1024.0/1024.0, name);
    }
};
