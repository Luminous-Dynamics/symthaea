// Independent C++20 reference for Rabbit's predictive four-branch flag mapping.
// This tests only flag remapping and synchronization, not Rabbit branch payloads or TOE.
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr std::array<std::uint8_t, 4> kCodeBits{0U, 0b10U, 0b110U, 0b111U};
constexpr std::array<std::uint8_t, 4> kCodeLengths{1U, 2U, 3U, 3U};
struct Mapping {
    std::array<std::uint8_t, 4> ranked{0U, 1U, 2U, 3U};
    bool dynamic{true};
};
struct BitWriter {
    std::vector<std::uint8_t> bytes;
    std::size_t bit_count{};
    void write(std::uint8_t value, std::uint8_t width) {
        for (std::uint8_t shift = width; shift > 0U; --shift) {
            const std::uint8_t bit = static_cast<std::uint8_t>((value >> (shift - 1U)) & 1U);
            if ((bit_count & 7U) == 0U) bytes.push_back(0U);
            if (bit != 0U) bytes.back() = static_cast<std::uint8_t>(bytes.back() | (1U << (7U - (bit_count & 7U))));
            ++bit_count;
        }
    }
};
struct BitReader {
    const std::vector<std::uint8_t>& bytes;
    std::size_t bit_count{};
    std::size_t cursor{};
    bool read(std::uint8_t& bit) {
        if (cursor >= bit_count) return false;
        bit = static_cast<std::uint8_t>((bytes[cursor / 8U] >> (7U - (cursor & 7U))) & 1U);
        ++cursor;
        return true;
    }
};
void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
Mapping from_counts(const std::array<std::uint64_t, 4>& counts, bool gate) {
    Mapping mapping{};
    std::sort(mapping.ranked.begin(), mapping.ranked.end(), [&](std::uint8_t left, std::uint8_t right) {
        if (counts[left] != counts[right]) return counts[left] > counts[right];
        return left < right;
    });
    const bool profitable = counts[mapping.ranked[0]] >
        counts[mapping.ranked[2]] + counts[mapping.ranked[3]];
    mapping.dynamic = !gate || profitable;
    return mapping;
}
std::pair<std::uint8_t, std::uint8_t> code_for(const Mapping& mapping, std::uint8_t branch) {
    if (!mapping.dynamic) return {branch, 2U};
    for (std::size_t rank = 0U; rank < mapping.ranked.size(); ++rank) {
        if (mapping.ranked[rank] == branch) return {kCodeBits[rank], kCodeLengths[rank]};
    }
    throw std::runtime_error("invalid branch map");
}
std::uint8_t decode_one(BitReader& reader, const Mapping& mapping) {
    std::uint8_t first = 0U;
    require(reader.read(first), "truncated branch flag");
    if (!mapping.dynamic) {
        std::uint8_t second = 0U;
        require(reader.read(second), "truncated fixed flag");
        return static_cast<std::uint8_t>((first << 1U) | second);
    }
    std::size_t rank = 0U;
    if (first != 0U) {
        std::uint8_t second = 0U;
        require(reader.read(second), "truncated dynamic flag");
        rank = 1U;
        if (second != 0U) {
            std::uint8_t third = 0U;
            require(reader.read(third), "truncated dynamic flag");
            rank = third == 0U ? 2U : 3U;
        }
    }
    return mapping.ranked[rank];
}
struct Encoded { std::vector<std::uint8_t> bytes; std::size_t bits{}; std::uint64_t values{}; };
Encoded encode_decode(const std::vector<std::vector<std::uint8_t>>& blocks, bool gate) {
    Mapping encoder_map{}; // first block default identity ranking, dynamic codes
    Mapping decoder_map{};
    Encoded result{};
    for (const auto& block : blocks) {
        BitWriter writer;
        std::array<std::uint64_t, 4> counts{};
        for (const std::uint8_t branch : block) {
            require(branch < 4U, "fixture has invalid branch");
            const auto [code, length] = code_for(encoder_map, branch);
            writer.write(code, length);
            ++counts[branch];
        }
        result.bits += writer.bit_count;
        result.bytes.insert(result.bytes.end(), writer.bytes.begin(), writer.bytes.end());
        result.values += static_cast<std::uint64_t>(block.size());

        BitReader reader{writer.bytes, writer.bit_count, 0U};
        std::array<std::uint64_t, 4> decoded_counts{};
        std::vector<std::uint8_t> decoded;
        decoded.reserve(block.size());
        while (reader.cursor < reader.bit_count) {
            const std::uint8_t branch = decode_one(reader, decoder_map);
            ++decoded_counts[branch];
            decoded.push_back(branch);
        }
        require(decoded == block, "decoder branch sequence diverged");
        require(decoded_counts == counts, "encoder/decoder branch statistics diverged");
        encoder_map = from_counts(counts, gate);
        decoder_map = from_counts(decoded_counts, gate);
        require(encoder_map.ranked == decoder_map.ranked && encoder_map.dynamic == decoder_map.dynamic,
                "predictive mapping desynchronized");
    }
    return result;
}
std::vector<std::vector<std::uint8_t>> make_blocks(
    const std::vector<std::array<std::uint64_t, 4>>& distributions, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<std::vector<std::uint8_t>> blocks;
    for (const auto& counts : distributions) {
        std::vector<std::uint8_t> block;
        std::uint64_t total = 0U;
        for (const auto count : counts) total += count;
        block.reserve(static_cast<std::size_t>(total));
        for (std::uint8_t branch = 0U; branch < 4U; ++branch)
            for (std::uint64_t i = 0U; i < counts[branch]; ++i) block.push_back(branch);
        std::shuffle(block.begin(), block.end(), rng);
        blocks.push_back(std::move(block));
    }
    return blocks;
}
void test_roundtrip_and_sync() {
    const auto blocks = make_blocks({
        {800U, 100U, 70U, 30U},
        {100U, 800U, 70U, 30U},
        {25U, 25U, 25U, 25U},
        {10U, 20U, 870U, 200U},
        {300U, 300U, 300U, 300U},
    }, 0x5eedU);
    (void)encode_decode(blocks, false);
    (void)encode_decode(blocks, true);
}
void test_skewed_flag_savings() {
    const auto blocks = make_blocks({{800U, 100U, 70U, 30U}}, 0xabcU);
    const auto dynamic = encode_decode(blocks, false);
    const std::uint64_t fixed_bits = dynamic.values * 2U;
    require(dynamic.bits < fixed_bits, "dynamic mapping failed skewed-distribution savings");
    std::cout << "skewed_first_block_flag_bits=" << dynamic.bits
              << " fixed_two_bit_flags=" << fixed_bits << "\n";
}
std::uint64_t flag_cost(const std::array<std::uint64_t, 4>& counts, const Mapping& mapping) {
    std::uint64_t bits = 0U;
    for (std::uint8_t branch = 0U; branch < 4U; ++branch)
        bits += counts[branch] * code_for(mapping, branch).second;
    return bits;
}
void test_misprediction_bound() {
    const auto blocks = make_blocks({{800U, 100U, 70U, 30U}, {30U, 70U, 100U, 800U}}, 0xdefU);
    Mapping predicted{};
    std::uint64_t extra_total = 0U;
    std::uint64_t values_total = 0U;
    for (const auto& block : blocks) {
        std::array<std::uint64_t, 4> counts{};
        for (const auto branch : block) ++counts[branch];
        const Mapping oracle = from_counts(counts, false);
        const std::uint64_t predicted_bits = flag_cost(counts, predicted);
        const std::uint64_t oracle_bits = flag_cost(counts, oracle);
        require(predicted_bits >= oracle_bits, "predicted mapping unexpectedly beats the per-block oracle");
        const std::uint64_t extra = predicted_bits - oracle_bits;
        require(extra <= static_cast<std::uint64_t>(block.size()) * 2U,
                "flag misprediction exceeded the paper's 2 bits/value bound");
        extra_total += extra;
        values_total += static_cast<std::uint64_t>(block.size());
        predicted = from_counts(counts, false);
    }
    std::cout << "misprediction_extra_bits=" << extra_total
              << " maximum_2N_bound=" << values_total * 2U << "\n";
}
void test_gated_uniform_fallback() {
    const auto blocks = make_blocks({
        {800U, 100U, 70U, 30U},
        {250U, 250U, 250U, 250U},
        {250U, 250U, 250U, 250U},
    }, 0x123U);
    const auto gated = encode_decode(blocks, true);
    const auto always = encode_decode(blocks, false);
    require(gated.bits < always.bits, "profitability gate should avoid dynamic-code penalty after a uniform block");
    std::cout << "uniform_following_block_gate_bits=" << gated.bits
              << " always_dynamic_bits=" << always.bits << "\n";
}
} // namespace

int main() {
    try {
        test_roundtrip_and_sync();
        test_skewed_flag_savings();
        test_misprediction_bound();
        test_gated_uniform_fallback();
        std::cout << "tests=4 status=PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Rabbit flag mapping reference failure: " << error.what() << "\n";
        return 1;
    }
}
