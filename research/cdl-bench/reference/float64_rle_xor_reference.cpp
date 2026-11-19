// Independent reference for exact-bit-pattern RLE + XOR-varint coding.
// This does not link to or execute the Rust CDL-Bench crate.
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

void write_varint(std::uint64_t value, std::vector<std::uint8_t>& out) {
    while (value >= 0x80U) {
        out.push_back(static_cast<std::uint8_t>((value & 0x7fU) | 0x80U));
        value >>= 7U;
    }
    out.push_back(static_cast<std::uint8_t>(value));
}

bool read_varint(const std::vector<std::uint8_t>& bytes,
                 std::size_t& cursor,
                 std::uint64_t& value) {
    value = 0;
    for (std::size_t i = 0; i < 10U; ++i) {
        if (cursor >= bytes.size()) return false;
        const std::uint8_t byte = bytes[cursor++];
        if (i == 9U && (byte & 0xfeU) != 0U) return false;
        const auto shift = static_cast<unsigned>(i * 7U);
        value |= static_cast<std::uint64_t>(byte & 0x7fU) << shift;
        if ((byte & 0x80U) == 0U) {
            if (i > 0U && (byte & 0x7fU) == 0U) return false;
            return true;
        }
    }
    return false;
}

std::vector<std::uint8_t> encode(const std::vector<std::uint64_t>& values) {
    std::vector<std::uint8_t> out;
    std::uint64_t previous = 0;
    std::size_t index = 0;
    while (index < values.size()) {
        const std::uint64_t current = values[index];
        std::size_t end = index + 1U;
        while (end < values.size() && values[end] == current) ++end;
        const auto run = static_cast<std::uint64_t>(end - index);
        write_varint(run, out);
        write_varint(current ^ previous, out);
        previous = current;
        index = end;
    }
    return out;
}

bool decode(const std::vector<std::uint8_t>& bytes,
            std::size_t expected_count,
            std::vector<std::uint64_t>& output) {
    output.clear();
    output.reserve(expected_count);
    std::size_t cursor = 0;
    std::uint64_t previous = 0;
    bool has_previous_run = false;

    while (output.size() < expected_count) {
        std::uint64_t run_length = 0;
        if (!read_varint(bytes, cursor, run_length) || run_length == 0U) {
            return false;
        }
        const std::size_t remaining = expected_count - output.size();
        if (run_length > static_cast<std::uint64_t>(remaining)) return false;

        std::uint64_t residual = 0;
        if (!read_varint(bytes, cursor, residual)) return false;
        const std::uint64_t current = previous ^ residual;
        if (has_previous_run && current == previous) return false;

        for (std::uint64_t i = 0; i < run_length; ++i) output.push_back(current);
        previous = current;
        has_previous_run = true;
    }
    return cursor == bytes.size();
}

bool expect_reject(const std::vector<std::uint8_t>& bytes,
                   std::size_t count,
                   const char* label) {
    std::vector<std::uint64_t> decoded;
    if (decode(bytes, count, decoded)) {
        std::cerr << "accepted malformed input: " << label << "\n";
        return false;
    }
    return true;
}

bool round_trip(const char* name, const std::vector<std::uint64_t>& values) {
    const auto encoded = encode(values);
    std::vector<std::uint64_t> decoded;
    if (!decode(encoded, values.size(), decoded) || decoded != values) {
        std::cerr << "round-trip failed: " << name << "\n";
        return false;
    }
    const double raw_bytes = static_cast<double>(values.size()) * 8.0;
    const double ratio = raw_bytes == 0.0 ? 0.0 :
        static_cast<double>(encoded.size()) / raw_bytes;
    std::cout << name << ": values=" << values.size()
              << " raw_bytes=" << static_cast<std::size_t>(raw_bytes)
              << " rle_xor_bytes=" << encoded.size()
              << " ratio=" << ratio << "\n";
    return true;
}

} // namespace

int main() {
    const std::vector<std::uint64_t> edges = {
        0x0000000000000000ULL, 0x0000000000000000ULL,
        0x8000000000000000ULL, 0x8000000000000000ULL,
        0x7ff0000000000000ULL, 0x7ff0000000000000ULL,
        0xfff0000000000000ULL, 0x0000000000000001ULL,
        0x0010000000000000ULL, 0x7ff8000000000001ULL,
        0xfff8123456789abcULL, 0x3ff0000000000000ULL
    };
    std::vector<std::uint64_t> constant(4096U, 0x4045400000000000ULL);

    std::vector<std::uint64_t> plateaus;
    plateaus.reserve(20000U);
    const std::uint64_t plateau_values[] = {
        0x0000000000000000ULL,
        0x4029000000000000ULL,
        0xc00a000000000000ULL,
        0x405e000000000000ULL
    };
    for (std::size_t i = 0; i < 20000U; ++i) {
        const std::size_t group = (i / 256U) % 4U;
        plateaus.push_back(plateau_values[group]);
    }

    std::vector<std::uint64_t> mostly_idle;
    mostly_idle.reserve(50000U);
    for (std::size_t i = 0; i < 50000U; ++i) {
        const std::uint64_t value = (i % 4096U < 3072U)
            ? 0x0000000000000000ULL
            : ((i % 127U == 0U) ? static_cast<std::uint64_t>(i * 31U) : 0x4045400000000000ULL);
        mostly_idle.push_back(value);
    }

    std::mt19937_64 generator(0x5eedc0deULL);
    std::vector<std::uint64_t> mixed;
    mixed.reserve(100000U);
    for (std::size_t i = 0; i < 100000U;) {
        const std::uint64_t value = generator();
        const std::size_t run = 1U + static_cast<std::size_t>(generator() % 19U);
        for (std::size_t j = 0; j < run && i < 100000U; ++j, ++i) {
            mixed.push_back(value);
        }
    }

    if (!round_trip("edge-ieee754", edges) ||
        !round_trip("constant", constant) ||
        !round_trip("plateaus", plateaus) ||
        !round_trip("mostly-idle", mostly_idle) ||
        !round_trip("mixed-random-runs", mixed)) {
        return 1;
    }

    if (!expect_reject({0x00U, 0x00U}, 2U, "zero run") ||
        !expect_reject({0x01U}, 1U, "missing XOR residual") ||
        !expect_reject({0x03U, 0x00U}, 2U, "run exceeds output count") ||
        !expect_reject({0x01U, 0x00U, 0x01U, 0x00U}, 2U, "adjacent duplicate runs") ||
        !expect_reject({0x80U, 0x00U, 0x00U}, 2U, "overlong run varint") ||
        !expect_reject({0x01U, 0x00U, 0x00U}, 1U, "trailing bytes") ||
        !expect_reject({0x01U, 0x00U, 0x80U}, 1U, "truncated trailing varint")) {
        return 1;
    }

    std::cout << "malformed_inputs_rejected=7\n";
    std::cout << "reference_status=PASS\n";
    return 0;
}
