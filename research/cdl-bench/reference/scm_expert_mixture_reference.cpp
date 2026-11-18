// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Independent C++20 reference for the SCM2 fixed-share expert-mixture
// diagnostics. This is NOT the production codec and does NOT replace Rust
// compilation/tests. It uses unsigned __int128 to independently reproduce the
// exact-rational expert mixture and posterior update.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

static constexpr uint32_t W = 1u << 24;
static constexpr uint32_t SHARE = W / 256;
static constexpr uint32_t TOTAL = 1u << 15;
static constexpr int N = 256;
static constexpr int MAX_CONTEXT_TOTAL = 1 << 14;
using u128 = unsigned __int128;

struct SplitMix64 {
    uint64_t state;
    explicit SplitMix64(uint64_t s) : state(s) {}
    uint64_t next() {
        state += 0x9E3779B97F4A7C15ULL;
        uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
};

static int64_t clip_q15(int64_t x) {
    return std::max<int64_t>(-32767, std::min<int64_t>(32767, x));
}

static int64_t noise(SplitMix64& rng) {
    return static_cast<int64_t>(rng.next() % 8193ULL) - 4096;
}

struct Config { int g1, g2, g20, resp, g41, g43; };
struct Intervention {
    bool active = false;
    int var = 1;
    size_t start = 0, end = 0;
    int16_t value = 20000;
};

static std::vector<std::array<int16_t, 5>> generate_scm(
    size_t steps, uint64_t seed, Config cfg, Intervention inter
) {
    SplitMix64 rng(seed);
    std::array<int16_t, 5> prev{};
    std::vector<std::array<int16_t, 5>> out;
    out.reserve(steps);

    for (size_t t = 0; t < steps; t++) {
        int64_t n0 = noise(rng), n1 = noise(rng), n2 = noise(rng);
        int64_t n3 = noise(rng), n4 = noise(rng);

        int16_t x0 = static_cast<int16_t>(clip_q15(prev[0] * 3 / 4 + n0));
        if (inter.active && inter.var == 0 && t >= inter.start && t < inter.end)
            x0 = inter.value;

        int16_t x1 = static_cast<int16_t>(clip_q15(
            prev[1] * (100 - cfg.g1) / 100
            + (static_cast<int64_t>(x0) * cfg.g1) / 100 + n1
        ));
        if (inter.active && inter.var == 1 && t >= inter.start && t < inter.end)
            x1 = inter.value;

        int16_t x2 = static_cast<int16_t>(clip_q15(
            prev[2] * 2 / 10
            + (static_cast<int64_t>(x1) * cfg.g2) / 100
            - (static_cast<int64_t>(x0) * cfg.g20) / 100 + n2
        ));
        if (inter.active && inter.var == 2 && t >= inter.start && t < inter.end)
            x2 = inter.value;

        int64_t input = static_cast<int64_t>(x2) + prev[3] * 2 / 10;
        int64_t response = cfg.resp == 0
            ? input
            : (input * 32767) / (32767 + std::llabs(input));
        int16_t x3 = static_cast<int16_t>(clip_q15(response + n3));
        if (inter.active && inter.var == 3 && t >= inter.start && t < inter.end)
            x3 = inter.value;

        int16_t x4 = static_cast<int16_t>(clip_q15(
            prev[4] * 2 / 10
            + (static_cast<int64_t>(x1) * cfg.g41) / 100
            + (static_cast<int64_t>(x3) * cfg.g43) / 100 + n4
        ));
        if (inter.active && inter.var == 4 && t >= inter.start && t < inter.end)
            x4 = inter.value;

        prev = {x0, x1, x2, x3, x4};
        out.push_back(prev);
    }
    return out;
}

struct Cdf {
    std::array<uint32_t, N + 1> v{};
    uint32_t total() const { return v[N]; }
};
using Hist = std::array<uint16_t, N>;
using BaseHist = std::array<std::array<Hist, 2>, 5>;
using LagHist = std::array<std::array<std::array<Hist, 4>, 2>, 5>;
using ParentHist = std::array<std::array<std::array<Hist, 16>, 2>, 5>;

static void hist_update(Hist& h, uint8_t sym) {
    uint32_t total = 0;
    for (uint16_t x : h) total += x;
    if (total >= MAX_CONTEXT_TOTAL) {
        for (auto& x : h) {
            x = static_cast<uint16_t>(
                std::max<uint32_t>(1, (static_cast<uint32_t>(x) + 1) / 2)
            );
        }
    }
    h[sym] = static_cast<uint16_t>(h[sym] + 1);
}

static Cdf cdf_from_hist(const Hist& h) {
    Cdf c;
    for (int i = 0; i < N; i++) c.v[i + 1] = c.v[i] + h[i];
    return c;
}

static int value_bucket(const std::array<uint8_t, 10>& record, int var) {
    int u = record[var * 2] | (static_cast<int>(record[var * 2 + 1]) << 8);
    int value = u >= 32768 ? u - 65536 : u;
    return ((value + 32768) * 4) / 65536;
}

struct MixtureState {
    uint64_t position = 0;
    Hist header{};
    BaseHist base{};
    LagHist lag{};
    ParentHist parent{};
    std::array<uint8_t, 10> current{}, previous{};

    MixtureState() {
        header.fill(1);
        for (auto& v : base) for (auto& lane : v) lane.fill(1);
        for (auto& v : lag) for (auto& lane : v) for (auto& ctx : lane) ctx.fill(1);
        for (auto& v : parent) for (auto& lane : v) for (auto& ctx : lane) ctx.fill(1);
    }

    int parent_context(int var) const {
        auto b = [&](int p) { return value_bucket(current, p); };
        switch (var) {
            case 0: return 0;
            case 1: return b(0);
            case 2: return b(0) | (b(1) << 2);
            case 3: return b(2);
            case 4: return b(1) | (b(3) << 2);
            default: return 0;
        }
    }

    Cdf cdf(int mode) const {
        if (position < 16) return cdf_from_hist(header);
        uint64_t off = position - 16;
        int var = (off / 2) % 5, lane = off % 2;
        if (mode == 0) return cdf_from_hist(base[var][lane]);

        Hist combined{};
        if (mode == 2) {
            int ctx = value_bucket(previous, var);
            for (int i = 0; i < N; i++)
                combined[i] = static_cast<uint16_t>(base[var][lane][i] + lag[var][lane][ctx][i]);
        } else {
            int ctx = parent_context(var);
            for (int i = 0; i < N; i++)
                combined[i] = static_cast<uint16_t>(base[var][lane][i] + parent[var][lane][ctx][i]);
        }
        return cdf_from_hist(combined);
    }

    void observe(uint8_t sym) {
        if (position < 16) {
            hist_update(header, sym);
        } else {
            uint64_t off = position - 16;
            int var = (off / 2) % 5, lane = off % 2;
            int lagctx = value_bucket(previous, var);
            int parentctx = parent_context(var);

            hist_update(base[var][lane], sym);
            hist_update(lag[var][lane][lagctx], sym);
            hist_update(parent[var][lane][parentctx], sym);

            current[var * 2 + lane] = sym;
            if (var == 4 && lane == 1) previous = current;
        }
        position++;
    }
};

static u128 common_denominator(const std::array<Cdf, 3>& cdfs) {
    return static_cast<u128>(cdfs[0].total()) * cdfs[1].total() * cdfs[2].total();
}

static std::array<u128, 3> likelihood_numerators(
    const std::array<Cdf, 3>& cdfs, uint8_t sym
) {
    u128 common = common_denominator(cdfs);
    std::array<u128, 3> out{};
    for (int i = 0; i < 3; i++) {
        uint32_t width = cdfs[i].v[sym + 1] - cdfs[i].v[sym];
        out[i] = static_cast<u128>(width) * (common / cdfs[i].total());
    }
    return out;
}

static std::array<uint32_t, N> mixture_freq(
    const std::array<uint32_t, 3>& weights, const std::array<Cdf, 3>& cdfs
) {
    u128 common = common_denominator(cdfs);
    u128 den = static_cast<u128>(weights[0] + static_cast<uint64_t>(weights[1]) + weights[2]) * common;
    u128 distributable = (static_cast<u128>(1) << 15) - N;

    std::array<uint32_t, N> freq{};
    freq.fill(1);
    std::array<u128, N> rem{};
    u128 assigned = 0;

    for (int s = 0; s < N; s++) {
        u128 num = 0;
        for (int i = 0; i < 3; i++) {
            uint32_t width = cdfs[i].v[s + 1] - cdfs[i].v[s];
            num += static_cast<u128>(weights[i]) * width * (common / cdfs[i].total());
        }
        u128 scaled = num * distributable;
        u128 extra = scaled / den;
        rem[s] = scaled % den;
        freq[s] += static_cast<uint32_t>(extra);
        assigned += extra;
    }

    size_t left = static_cast<size_t>(distributable - assigned);
    std::array<int, N> order{};
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return rem[a] > rem[b] || (rem[a] == rem[b] && a < b);
    });
    for (size_t i = 0; i < left; i++) freq[order[i]]++;
    return freq;
}

static std::array<uint32_t, 3> fixed_share(std::array<uint32_t, 3> posterior) {
    const uint32_t retained_total = W - SHARE;
    const uint32_t share_base = SHARE / 3, share_rem = SHARE % 3;
    std::array<uint32_t, 3> out{share_base, share_base, share_base};
    std::array<uint64_t, 3> rem{};

    for (int i = 0; i < 3; i++) {
        uint64_t scaled = static_cast<uint64_t>(posterior[i]) * retained_total;
        uint64_t retained = scaled / W;
        rem[i] = scaled % W;
        out[i] += static_cast<uint32_t>(retained);
        if (i < static_cast<int>(share_rem)) out[i]++;
    }

    uint64_t sum = static_cast<uint64_t>(out[0]) + out[1] + out[2];
    size_t missing = static_cast<size_t>(W - sum);
    std::array<int, 3> order{0, 1, 2};
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return rem[a] > rem[b] || (rem[a] == rem[b] && a < b);
    });
    for (size_t i = 0; i < missing; i++) out[order[i]]++;
    return out;
}

static std::array<uint32_t, 3> update_weights(
    const std::array<uint32_t, 3>& prior,
    const std::array<Cdf, 3>& cdfs,
    uint8_t sym
) {
    auto likes = likelihood_numerators(cdfs, sym);
    u128 evidence =
        static_cast<u128>(prior[0]) * likes[0]
        + static_cast<u128>(prior[1]) * likes[1]
        + static_cast<u128>(prior[2]) * likes[2];
    u128 distributable = static_cast<u128>(W) - 3;
    std::array<uint32_t, 3> post{1, 1, 1};
    std::array<u128, 3> rem{};
    u128 assigned = 0;

    for (int i = 0; i < 3; i++) {
        u128 scaled = static_cast<u128>(prior[i]) * likes[i] * distributable;
        u128 extra = scaled / evidence;
        rem[i] = scaled % evidence;
        post[i] += static_cast<uint32_t>(extra);
        assigned += extra;
    }

    size_t left = static_cast<size_t>(distributable - assigned);
    std::array<int, 3> order{0, 1, 2};
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return rem[a] > rem[b] || (rem[a] == rem[b] && a < b);
    });
    for (size_t i = 0; i < left; i++) post[order[i]]++;
    return fixed_share(post);
}

struct RowResult {
    int cfg, seed;
    bool treated;
    std::array<double, 4> bits;
    std::array<double, 3> weights, pre, mid, post;
};

static RowResult evaluate(
    const std::vector<uint8_t>& stream, int cfgid, int seedid, bool treated
) {
    MixtureState state;
    std::array<uint32_t, 3> weights{W / 3, W / 3, W - 2 * (W / 3)};
    std::array<double, 4> costs{};
    std::array<double, 3> pre{}, mid{}, post{};

    for (size_t i = 0; i < stream.size(); i++) {
        uint8_t sym = stream[i];
        std::array<Cdf, 3> cdfs{state.cdf(0), state.cdf(2), state.cdf(1)};
        auto mf = mixture_freq(weights, cdfs);

        for (int e = 0; e < 3; e++) {
            uint32_t width = cdfs[e].v[sym + 1] - cdfs[e].v[sym];
            costs[e] += -std::log2(static_cast<double>(width) / cdfs[e].total());
        }
        costs[3] += -std::log2(static_cast<double>(mf[sym]) / TOTAL);

        weights = update_weights(weights, cdfs, sym);
        state.observe(sym);

        size_t n = i + 1;
        if (n == 16 + 256 * 10) for (int j = 0; j < 3; j++) pre[j] = static_cast<double>(weights[j]) / W;
        if (n == 16 + 512 * 10) for (int j = 0; j < 3; j++) mid[j] = static_cast<double>(weights[j]) / W;
        if (n == 16 + 896 * 10) for (int j = 0; j < 3; j++) post[j] = static_cast<double>(weights[j]) / W;
    }

    RowResult r;
    r.cfg = cfgid; r.seed = seedid; r.treated = treated;
    for (int i = 0; i < 4; i++) r.bits[i] = costs[i] / stream.size();
    for (int i = 0; i < 3; i++) {
        r.weights[i] = static_cast<double>(weights[i]) / W;
        r.pre[i] = pre[i]; r.mid[i] = mid[i]; r.post[i] = post[i];
    }
    return r;
}

static std::vector<uint8_t> serialize(
    const std::vector<std::array<int16_t, 5>>& rows, size_t steps
) {
    std::vector<uint8_t> out;
    out.reserve(16 + steps * 10);
    out.insert(out.end(), {'S', 'C', 'M', '2'});
    out.push_back(2); out.push_back(0); out.push_back(0); out.push_back(5);
    uint64_t n = steps;
    for (int i = 0; i < 8; i++) out.push_back(static_cast<uint8_t>((n >> (8 * i)) & 255));
    for (auto& row : rows) for (auto v : row) {
        uint16_t u = static_cast<uint16_t>(v);
        out.push_back(u & 255); out.push_back(u >> 8);
    }
    return out;
}

int main() {
    const size_t steps = 1024;
    std::array<Config, 2> configs{{
        {70, 70, 30, 0, 60, 20},
        {30, 70, 30, 1, 60, 20}
    }};
    std::vector<RowResult> rows;

    for (int ci = 0; ci < 2; ci++) for (int si = 0; si < 10; si++) {
        uint64_t seed = 0xCA55A1ULL + si + 200;
        for (int treat = 0; treat < 2; treat++) {
            Intervention in;
            if (treat) {
                in.active = true; in.var = 1;
                in.start = steps / 4; in.end = steps * 3 / 4; in.value = 20000;
            }
            auto records = generate_scm(steps, seed, configs[ci], in);
            auto stream = serialize(records, steps);
            rows.push_back(evaluate(stream, ci, si, treat));
        }
    }

    std::array<double, 4> overall{}, obs{}, inter{};
    std::array<double, 3> meanw{};
    for (auto& r : rows) {
        for (int i = 0; i < 4; i++) {
            overall[i] += r.bits[i];
            (r.treated ? inter : obs)[i] += r.bits[i];
        }
        for (int i = 0; i < 3; i++) meanw[i] += r.weights[i];
    }
    for (int i = 0; i < 4; i++) {
        overall[i] /= rows.size();
        obs[i] /= rows.size() / 2;
        inter[i] /= rows.size() / 2;
    }
    for (int i = 0; i < 3; i++) meanw[i] /= rows.size();

    std::cout << std::setprecision(10);
    std::cout << "streams=" << rows.size()
              << " bytes_per_stream=" << (16 + steps * 10) << "\n";
    for (int i = 0; i < 4; i++) std::cout << "overall_model_" << i << "=" << overall[i] << "\n";
    for (int i = 0; i < 4; i++) std::cout << "observational_model_" << i << "=" << obs[i] << "\n";
    for (int i = 0; i < 4; i++) std::cout << "interventional_model_" << i << "=" << inter[i] << "\n";
    std::cout << "mean_final_weights=" << meanw[0] << "," << meanw[1] << "," << meanw[2] << "\n";
    std::cout << "rows_csv\n";
    for (auto& r : rows) {
        std::cout << r.cfg << "," << r.seed << "," << (r.treated ? 1 : 0) << ","
                  << r.bits[0] << "," << r.bits[1] << "," << r.bits[2] << "," << r.bits[3] << ","
                  << r.weights[0] << "," << r.weights[1] << "," << r.weights[2] << "\n";
    }
    for (auto& r : rows) if (r.cfg == 0 && r.seed == 0 && r.treated) {
        std::cout << "example_weights_pre=" << r.pre[0] << "," << r.pre[1] << "," << r.pre[2] << "\n";
        std::cout << "example_weights_mid=" << r.mid[0] << "," << r.mid[1] << "," << r.mid[2] << "\n";
        std::cout << "example_weights_post=" << r.post[0] << "," << r.post[1] << "," << r.post[2] << "\n";
    }

    // Match the authored Rust switching-test schedule for an explicit
    // pre-intervention / intervention / recovery check.
    const size_t long_steps = 4096;
    Config cfg{70, 70, 30, 0, 60, 20};
    Intervention in;
    in.active = true; in.var = 1; in.start = 1024; in.end = 3072; in.value = 20000;
    auto records = generate_scm(long_steps, 43, cfg, in);
    auto stream = serialize(records, long_steps);
    MixtureState state;
    std::array<uint32_t, 3> w{W / 3, W / 3, W - 2 * (W / 3)};
    std::array<uint32_t, 3> before{}, during{}, after{};

    for (size_t i = 0; i < stream.size(); i++) {
        auto cdfs = std::array<Cdf, 3>{state.cdf(0), state.cdf(2), state.cdf(1)};
        (void)mixture_freq(w, cdfs);
        w = update_weights(w, cdfs, stream[i]);
        state.observe(stream[i]);
        size_t n = i + 1;
        if (n == 16 + 1024 * 10) before = w;
        if (n == 16 + 2048 * 10) during = w;
        if (n == 16 + (3072 + 4096 / 8) * 10) after = w;
    }
    auto printw = [](const char* label, const std::array<uint32_t, 3>& x) {
        std::cout << label << "="
                  << static_cast<double>(x[0]) / W << ","
                  << static_cast<double>(x[1]) / W << ","
                  << static_cast<double>(x[2]) / W << "\n";
    };
    printw("test_pre", before);
    printw("test_during", during);
    printw("test_after", after);
    return 0;
}
