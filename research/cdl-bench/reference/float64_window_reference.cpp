// Independent exact bit-window reference for CDL-Bench f64 transform ID 2.
// This is not Rust execution and not a Gorilla/Chimp implementation.
#include <algorithm>
#include <bit>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

struct Window {
    bool valid = false;
    uint32_t leading = 0;
    uint32_t trailing = 0;
};

static void append_u64_varint(uint64_t value, std::vector<uint8_t>& out) {
    while (value >= 0x80ULL) {
        out.push_back(static_cast<uint8_t>((value & 0x7fULL) | 0x80ULL));
        value >>= 7;
    }
    out.push_back(static_cast<uint8_t>(value));
}

static std::vector<uint8_t> encode_xor_varint(const std::vector<uint64_t>& values) {
    std::vector<uint8_t> out;
    uint64_t previous = 0;
    for (uint64_t value : values) {
        append_u64_varint(value ^ previous, out);
        previous = value;
    }
    return out;
}

static std::vector<uint8_t> encode_window(const std::vector<uint64_t>& values) {
    std::vector<uint8_t> out;
    uint64_t previous = 0;
    Window window;
    for (uint64_t value : values) {
        const uint64_t residual = value ^ previous;
        if (residual == 0) {
            out.push_back(0);
        } else {
            const uint32_t actual_leading = static_cast<uint32_t>(std::countl_zero(residual));
            const uint32_t actual_trailing = static_cast<uint32_t>(std::countr_zero(residual));
            const bool reuse = window.valid && actual_leading >= window.leading && actual_trailing >= window.trailing;
            uint32_t leading = actual_leading;
            uint32_t trailing = actual_trailing;
            if (reuse) {
                out.push_back(1);
                leading = window.leading;
                trailing = window.trailing;
            } else {
                out.push_back(2);
                out.push_back(static_cast<uint8_t>(leading));
                out.push_back(static_cast<uint8_t>(trailing));
                window.valid = true;
                window.leading = leading;
                window.trailing = trailing;
            }
            const uint32_t width = 64U - leading - trailing;
            const uint64_t meaningful = residual >> trailing;
            const uint32_t payload_bytes = (width + 7U) / 8U;
            for (uint32_t byte = 0; byte < payload_bytes; ++byte) {
                out.push_back(static_cast<uint8_t>(meaningful >> (byte * 8U)));
            }
        }
        previous = value;
    }
    return out;
}

static std::vector<uint64_t> decode_window(const std::vector<uint8_t>& bytes, size_t count) {
    std::vector<uint64_t> out;
    out.reserve(count);
    size_t cursor = 0;
    uint64_t previous = 0;
    Window window;
    for (size_t item = 0; item < count; ++item) {
        if (cursor >= bytes.size()) throw std::runtime_error("truncated marker");
        const uint8_t marker = bytes[cursor++];
        uint64_t residual = 0;
        if (marker == 0) {
            residual = 0;
        } else if (marker == 1 || marker == 2) {
            if (marker == 2) {
                if (cursor + 2U > bytes.size()) throw std::runtime_error("truncated new window");
                const uint32_t leading = bytes[cursor++];
                const uint32_t trailing = bytes[cursor++];
                if (leading >= 64U || trailing >= 64U || leading + trailing >= 64U)
                    throw std::runtime_error("invalid new window span");
                window.valid = true;
                window.leading = leading;
                window.trailing = trailing;
            } else if (!window.valid) {
                throw std::runtime_error("reuse before a window is defined");
            }
            const uint32_t width = 64U - window.leading - window.trailing;
            const uint32_t payload_bytes = (width + 7U) / 8U;
            const size_t end = cursor + static_cast<size_t>(payload_bytes);
            if (end > bytes.size()) throw std::runtime_error("truncated payload");
            uint64_t meaningful = 0;
            for (uint32_t byte = 0; byte < payload_bytes; ++byte)
                meaningful |= static_cast<uint64_t>(bytes[cursor + byte]) << (byte * 8U);
            cursor = end;
            if ((width % 8U) != 0U) {
                const uint32_t used = width % 8U;
                if ((static_cast<uint32_t>(bytes[end - 1U]) >> used) != 0U)
                    throw std::runtime_error("nonzero unused high bits");
            }
            if (meaningful == 0) throw std::runtime_error("zero residual must use marker zero");
            residual = meaningful << window.trailing;
            const uint32_t actual_leading = static_cast<uint32_t>(std::countl_zero(residual));
            const uint32_t actual_trailing = static_cast<uint32_t>(std::countr_zero(residual));
            if (marker == 2) {
                if ((meaningful & 1ULL) == 0ULL || ((meaningful >> (width - 1U)) & 1ULL) == 0ULL ||
                    actual_leading != window.leading || actual_trailing != window.trailing)
                    throw std::runtime_error("noncanonical new window payload");
            } else if (actual_leading < window.leading || actual_trailing < window.trailing) {
                throw std::runtime_error("reused residual does not fit window");
            }
        } else {
            throw std::runtime_error("unsupported bit-window marker");
        }
        const uint64_t value = previous ^ residual;
        out.push_back(value);
        previous = value;
    }
    if (cursor != bytes.size()) throw std::runtime_error("trailing bytes");
    return out;
}

static void must_reject(const std::vector<uint8_t>& bytes, size_t count, const std::string& label) {
    try {
        (void)decode_window(bytes, count);
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error("malformed case accepted: " + label);
}

int main() {
    std::vector<std::pair<std::string, std::vector<uint64_t>>> fixtures;
    fixtures.push_back({"edge", {0x0000000000000000ULL, 0x8000000000000000ULL,
        0x7ff0000000000000ULL, 0xfff0000000000000ULL, 0x0000000000000001ULL,
        0x0010000000000000ULL, 0x7ff8000000000001ULL, 0xfff8123456789abcULL,
        0x3ff0000000000000ULL}});

    std::vector<uint64_t> smooth;
    smooth.reserve(2048);
    for (uint64_t i = 0; i < 2048ULL; ++i) smooth.push_back(0x3ff0000000000000ULL + i * 17ULL);
    fixtures.push_back({"smooth", std::move(smooth)});

    std::vector<uint64_t> oscillating;
    oscillating.reserve(2048);
    for (uint64_t i = 0; i < 2048ULL; ++i) {
        const uint64_t magnitude = 0x3ff0000000000000ULL + ((i * 7919ULL) & 0x000fffffULL);
        oscillating.push_back(magnitude | ((i % 2ULL) == 0ULL ? 0ULL : (1ULL << 63U)));
    }
    fixtures.push_back({"oscillating", std::move(oscillating)});

    std::mt19937_64 rng(0xF64CULL);
    std::vector<uint64_t> random_bits;
    random_bits.reserve(50000);
    for (size_t i = 0; i < 50000U; ++i) random_bits.push_back(rng());
    fixtures.push_back({"random", std::move(random_bits)});

    std::vector<uint64_t> repeated;
    repeated.reserve(3000);
    repeated.insert(repeated.end(), 1000U, 0ULL);
    repeated.insert(repeated.end(), 1000U, 0x3ff0000000000000ULL);
    repeated.insert(repeated.end(), 1000U, 0ULL);
    fixtures.push_back({"repeated", std::move(repeated)});

    size_t total_samples = 0;
    std::cout << std::fixed << std::setprecision(2);
    for (const auto& fixture : fixtures) {
        const auto& values = fixture.second;
        const auto encoded = encode_window(values);
        const auto decoded = decode_window(encoded, values.size());
        if (decoded != values) throw std::runtime_error("bitwise round-trip failed: " + fixture.first);
        const auto varint = encode_xor_varint(values);
        const double delta = varint.empty() ? 0.0 :
            (static_cast<double>(encoded.size()) / static_cast<double>(varint.size()) - 1.0) * 100.0;
        std::cout << fixture.first << ",samples=" << values.size()
                  << ",raw_bytes=" << (values.size() * 8U)
                  << ",xor_varint_bytes=" << varint.size()
                  << ",xor_window_bytes=" << encoded.size()
                  << ",window_vs_varint_pct=" << delta << "\n";
        total_samples += values.size();
    }

    must_reject({3}, 1U, "bad marker");
    must_reject({1}, 1U, "reuse before window");
    must_reject({2, 64, 0}, 1U, "leading count 64");
    must_reject({2, 0, 64}, 1U, "trailing count 64");
    must_reject({2, 32, 32}, 1U, "empty significant window");
    must_reject({2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 1U, "zero residual with new window");
    must_reject({2, 0, 58, 0xc1}, 1U, "unused high payload bits");

    std::cout << "total_samples=" << total_samples << "\n";
    std::cout << "malformed_cases_rejected=7\n";
    return 0;
}
