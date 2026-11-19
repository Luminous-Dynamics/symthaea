// Independent C++20 reference for a deterministic, prefix-only LZV segmentation control.
// WZS-inspired research baseline only; it is not the WZS implementation.
#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct Config {
    std::size_t fast_window{8U};
    std::size_t slow_window{16U};
    std::size_t min_segment{32U};
    std::size_t max_segment{512U};
    std::size_t cooldown{32U};
    std::uint8_t mean_shift{8U};
    std::uint16_t variance_shift{256U};
};

void validate(const Config& c) {
    if (c.fast_window < 2U || c.slow_window < 2U ||
        c.fast_window > 4096U || c.slow_window > 4096U) {
        throw std::runtime_error("window sizes must be in 2..=4096");
    }
    if (c.min_segment == 0U || c.max_segment < c.min_segment || c.max_segment > 1'000'000U) {
        throw std::runtime_error("invalid segment bounds");
    }
    if (c.mean_shift == 0U || c.variance_shift == 0U) {
        throw std::runtime_error("shift thresholds must be positive");
    }
}

struct Stats { std::uint64_t sum{}; std::uint64_t sum_sq{}; std::uint64_t count{}; };
Stats window_stats(const std::deque<std::uint8_t>& history, std::size_t start, std::size_t count) {
    Stats stats{};
    for (std::size_t i = 0U; i < count; ++i) {
        const auto value = static_cast<std::uint64_t>(history[start + i]);
        stats.sum += value;
        stats.sum_sq += value * value;
    }
    stats.count = static_cast<std::uint64_t>(count);
    return stats;
}

bool changed_distribution(const Stats& fast, const Stats& slow, const Config& c) {
    const std::uint64_t wf = fast.count;
    const std::uint64_t ws = slow.count;
    const std::uint64_t mean_left = fast.sum * ws;
    const std::uint64_t mean_right = slow.sum * wf;
    const std::uint64_t mean_delta = mean_left > mean_right ? mean_left - mean_right : mean_right - mean_left;
    const bool mean_shift = mean_delta >= static_cast<std::uint64_t>(c.mean_shift) * wf * ws;

    // Compare population variances exactly as rational integers:
    // variance(W) = (W*sum(x^2) - sum(x)^2) / W^2.
    const std::uint64_t vf_num = wf * fast.sum_sq - fast.sum * fast.sum;
    const std::uint64_t vs_num = ws * slow.sum_sq - slow.sum * slow.sum;
    const std::uint64_t variance_left = vf_num * ws * ws;
    const std::uint64_t variance_right = vs_num * wf * wf;
    const std::uint64_t variance_delta = variance_left > variance_right
        ? variance_left - variance_right : variance_right - variance_left;
    const bool variance_shift = variance_delta >=
        static_cast<std::uint64_t>(c.variance_shift) * wf * wf * ws * ws;
    return mean_shift || variance_shift;
}

std::vector<std::size_t> segment_lzv(const std::vector<std::uint64_t>& bits, const Config& config) {
    validate(config);
    std::vector<std::size_t> boundaries{0U};
    if (bits.empty()) return boundaries;

    const std::size_t history_limit = config.fast_window + config.slow_window;
    std::deque<std::uint8_t> history;
    std::size_t segment_start = 0U;
    std::size_t cooldown_remaining = 0U;
    bool pending_change = false;

    for (std::size_t index = 1U; index < bits.size(); ++index) {
        const std::uint64_t residual = bits[index] ^ bits[index - 1U];
        history.push_back(static_cast<std::uint8_t>(std::countl_zero(residual)));
        if (history.size() > history_limit) history.pop_front();

        const std::size_t next_index = index + 1U;
        const std::size_t current_length = next_index - segment_start;
        const bool forced_max_boundary = current_length >= config.max_segment;
        if (!forced_max_boundary && cooldown_remaining == 0U && history.size() == history_limit) {
            const auto fast = window_stats(history, config.slow_window, config.fast_window);
            const auto slow = window_stats(history, 0U, config.slow_window);
            // Latch an observed prefix-only shift until the minimum segment
            // length is reached; otherwise the short/fast window can adapt
            // before a minimum-length boundary is legally emitted.
            pending_change = pending_change || changed_distribution(fast, slow, config);
        }
        const bool change_boundary = current_length >= config.min_segment && pending_change;

        if (forced_max_boundary || change_boundary) {
            if (next_index < bits.size()) boundaries.push_back(next_index);
            segment_start = next_index;
            pending_change = false;
            // A forced maximum-length split is not evidence of a regime change;
            // do not let it mask a real shift immediately after the split.
            cooldown_remaining = change_boundary ? config.cooldown : 0U;
        } else if (cooldown_remaining > 0U) {
            --cooldown_remaining;
        }
    }
    if (boundaries.back() != bits.size()) boundaries.push_back(bits.size());
    return boundaries;
}

std::vector<std::uint64_t> constant_stream(std::size_t count, std::uint64_t value) {
    return std::vector<std::uint64_t>(count, value);
}
std::vector<std::uint64_t> random_stream(std::size_t count, std::uint64_t seed) {
    std::vector<std::uint64_t> values;
    values.reserve(count);
    std::uint64_t state = seed;
    for (std::size_t i = 0U; i < count; ++i) {
        state ^= state >> 12U;
        state ^= state << 25U;
        state ^= state >> 27U;
        values.push_back(state * 0x2545f4914f6cdd1dULL);
    }
    return values;
}
std::vector<std::uint64_t> abrupt_stream(std::size_t stable_count, std::size_t changed_count, std::uint64_t seed) {
    auto values = constant_stream(stable_count, 0x3ff0000000000000ULL);
    auto random = random_stream(changed_count, seed);
    values.insert(values.end(), random.begin(), random.end());
    return values;
}
void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
void check_boundaries(const std::vector<std::size_t>& boundaries, std::size_t count,
                      const Config& config, const std::string& name) {
    require(!boundaries.empty() && boundaries.front() == 0U, name + " lacks zero start");
    require(boundaries.back() == count, name + " lacks final boundary");
    for (std::size_t i = 1U; i < boundaries.size(); ++i) {
        require(boundaries[i] > boundaries[i - 1U], name + " has empty segment");
        const std::size_t length = boundaries[i] - boundaries[i - 1U];
        require(length <= config.max_segment, name + " exceeds max segment");
        if (i + 1U < boundaries.size()) require(length >= config.min_segment, name + " has short interior segment");
    }
}
void test_constant() {
    Config config{};
    config.min_segment = 32U;
    config.max_segment = 100U;
    const auto values = constant_stream(350U, 0x3ff0000000000000ULL);
    const auto boundaries = segment_lzv(values, config);
    require(boundaries == std::vector<std::size_t>({0U, 100U, 200U, 300U, 350U}),
            "constant stream should split only at max segment");
}
void test_abrupt_shift() {
    Config config{};
    config.min_segment = 32U;
    config.max_segment = 512U;
    const auto values = abrupt_stream(128U, 128U, 0x5eedULL);
    const auto boundaries = segment_lzv(values, config);
    check_boundaries(boundaries, values.size(), config, "abrupt shift");
    const auto it = std::find_if(boundaries.begin() + 1, boundaries.end(), [](std::size_t b) { return b < 256U; });
    require(it != boundaries.end(), "abrupt shift not detected before stream end");
    require(*it >= 128U && *it <= 160U, "abrupt shift detection delay outside 0..32 observations");
}
void test_stationary_random() {
    Config config{};
    config.mean_shift = 12U;
    config.variance_shift = 512U;
    config.max_segment = 256U;
    for (std::uint64_t seed = 0U; seed < 64U; ++seed) {
        const auto values = random_stream(4096U, 0x12345678ULL + seed);
        const auto boundaries = segment_lzv(values, config);
        check_boundaries(boundaries, values.size(), config, "stationary random");
        const std::size_t expected_segments = (values.size() + config.max_segment - 1U) / config.max_segment;
        require(boundaries.size() - 1U <= expected_segments + 2U,
                "stationary random stream over-segmented");
    }
}
void test_deterministic_and_empty() {
    Config config{};
    require(segment_lzv({}, config) == std::vector<std::size_t>({0U}), "empty stream boundaries wrong");
    const auto values = abrupt_stream(120U, 120U, 0x9876ULL);
    require(segment_lzv(values, config) == segment_lzv(values, config), "non-deterministic boundaries");
}

void report(const std::string& name, const std::vector<std::uint64_t>& values, const Config& config) {
    const auto boundaries = segment_lzv(values, config);
    check_boundaries(boundaries, values.size(), config, name);
    std::size_t min_length = values.size();
    std::size_t max_length = 0U;
    std::uint64_t total = 0U;
    for (std::size_t i = 1U; i < boundaries.size(); ++i) {
        const std::size_t length = boundaries[i] - boundaries[i - 1U];
        min_length = std::min(min_length, length);
        max_length = std::max(max_length, length);
        total += static_cast<std::uint64_t>(length);
    }
    const double average = boundaries.size() > 1U
        ? static_cast<double>(total) / static_cast<double>(boundaries.size() - 1U) : 0.0;
    std::cout << name << ": values=" << values.size()
              << " segments=" << (boundaries.size() - 1U)
              << " min_segment=" << (boundaries.size() > 1U ? min_length : 0U)
              << " max_segment=" << max_length
              << " avg_segment=" << average << " boundaries=";
    for (const auto boundary : boundaries) std::cout << boundary << ',';
    std::cout << '\n';
}
} // namespace

int main() {
    try {
        test_constant();
        test_abrupt_shift();
        test_stationary_random();
        test_deterministic_and_empty();
        std::cout << "tests=4 status=PASS\n";
        Config config{};
        report("constant", constant_stream(2048U, 0x3ff0000000000000ULL), config);
        report("stationary-random", random_stream(4096U, 0xabcULL), config);
        report("stable-to-random", abrupt_stream(1024U, 1024U, 0xdefULL), config);
        auto alternating = constant_stream(512U, 0U);
        const auto random = random_stream(512U, 0x111ULL);
        alternating.insert(alternating.end(), random.begin(), random.end());
        alternating.insert(alternating.end(), 512U, 0U);
        report("stable-random-stable", alternating, config);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "LZV reference failure: " << error.what() << '\n';
        return 1;
    }
}
