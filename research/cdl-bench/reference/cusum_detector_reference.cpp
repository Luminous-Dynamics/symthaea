// Independent fixed-point check of the CDL-Bench surprisal CUSUM.
// This does not link the Rust crate or validate stream-level coding.
#include <algorithm>
#include <cstdint>
#include <iostream>

static constexpr uint64_t WARMUP = 64;
static constexpr int64_t DRIFT_Q8 = 64;
static constexpr int64_t THRESHOLD_Q8 = 8 * 256;
static constexpr uint16_t COOLDOWN = 256;
static constexpr uint32_t BASELINE_SHIFT = 6;

uint32_t log2_ratio_q8(uint32_t numerator, uint32_t denominator) {
    if (denominator == 0 || numerator < denominator) return 0;
    uint32_t exponent = 0;
    while (exponent < 31 &&
           (static_cast<uint64_t>(denominator) << (exponent + 1)) <= numerator) {
        ++exponent;
    }
    const unsigned __int128 scaled_den =
        static_cast<unsigned __int128>(denominator) << exponent;
    unsigned __int128 normalized =
        (static_cast<unsigned __int128>(numerator) << 32) / scaled_den;
    uint32_t fraction = 0;
    for (uint32_t bit = 0; bit < 8; ++bit) {
        normalized = (normalized * normalized) >> 32;
        if (normalized >= (static_cast<unsigned __int128>(2) << 32)) {
            normalized >>= 1;
            fraction |= 1u << (7 - bit);
        }
    }
    return exponent * 256 + fraction;
}

struct Cusum {
    uint64_t samples = 0;
    int64_t baseline = 0, fast = 0, sum = 0;
    uint16_t cooldown = 0;
    uint32_t alarms = 0;

    bool observe(uint32_t surprise) {
        const int64_t value = surprise;
        if (samples == 0) {
            baseline = fast = value;
            samples = 1;
            return false;
        }
        fast += (value - fast) / 4;
        if (samples >= WARMUP) {
            sum = std::max<int64_t>(0, sum + value - baseline - DRIFT_Q8);
        }
        baseline += (value - baseline) /
                    (int64_t(1) << BASELINE_SHIFT);
        if (cooldown > 0) --cooldown;
        const bool alarm = samples >= WARMUP && cooldown == 0 &&
                           sum >= THRESHOLD_Q8;
        if (alarm) {
            ++alarms;
            sum = 0;
            baseline = fast;
            cooldown = COOLDOWN;
        }
        ++samples;
        return alarm;
    }
};

int main() {
    for (uint32_t x : {1u, 2u, 4u, 8u, 16u, 100u}) {
        std::cout << "log2_q8(" << x << ")=" << log2_ratio_q8(x, 1) << "\n";
    }

    Cusum stable;
    for (int i = 0; i < 100000; ++i) stable.observe(4 * 256);
    std::cout << "stable_alarms=" << stable.alarms << "\n";

    Cusum shifted;
    for (int i = 0; i < 512; ++i) shifted.observe(4 * 256);
    int delay = -1;
    for (int i = 0; i < 128; ++i) {
        if (shifted.observe(7 * 256)) {
            delay = i + 1;
            break;
        }
    }
    std::cout << "step_shift_detection_delay=" << delay << "\n";
    std::cout << "shift_alarms=" << shifted.alarms << "\n";

    Cusum gradual;
    for (int i = 0; i < 512; ++i) gradual.observe(4 * 256);
    for (int i = 1; i <= 512; ++i) {
        gradual.observe(4 * 256 + static_cast<uint32_t>(i / 16) * 8);
    }
    std::cout << "gradual_shift_alarms=" << gradual.alarms << "\n";
    return (stable.alarms == 0 && delay > 0 && delay <= 8) ? 0 : 1;
}
