// Independent C++20 diagnostic of Rabbit's four-branch predictive prefix mapper.
// This does NOT implement ACTF payload branches or TOE and is not a Rabbit reproduction.
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr std::size_t kBranches = 4U;
constexpr std::size_t kDefaultBlock = 1000U;
using Counts = std::array<std::uint64_t, kBranches>;
using Mapping = std::array<std::uint8_t, kBranches>;

struct Encoded {
    std::vector<std::uint8_t> bytes;
    std::size_t bit_count{};
};

class BitWriter {
public:
    void write_bit(bool bit) {
        if ((bit_count_ & 7U) == 0U) bytes_.push_back(0U);
        if (bit) bytes_.back() |= static_cast<std::uint8_t>(1U << (7U - (bit_count_ & 7U)));
        ++bit_count_;
    }
    void write_bits(std::uint32_t value, std::size_t count) {
        for (std::size_t i = 0U; i < count; ++i)
            write_bit(((value >> (count - i - 1U)) & 1U) != 0U);
    }
    Encoded finish() && { return {std::move(bytes_), bit_count_}; }
private:
    std::vector<std::uint8_t> bytes_;
    std::size_t bit_count_{};
};

class BitReader {
public:
    explicit BitReader(const Encoded& encoded) : encoded_(encoded) {}
    bool read_bit() {
        if (position_ >= encoded_.bit_count) throw std::runtime_error("truncated flag stream");
        const bool value = ((encoded_.bytes[position_ / 8U] >> (7U - (position_ & 7U))) & 1U) != 0U;
        ++position_;
        return value;
    }
    std::size_t position() const { return position_; }
private:
    const Encoded& encoded_;
    std::size_t position_{};
};

Mapping identity_mapping() { return {0U, 1U, 2U, 3U}; }

Mapping rank_by_previous_counts(const Counts& previous) {
    Mapping ranked = identity_mapping();
    std::stable_sort(ranked.begin(), ranked.end(), [&](std::uint8_t left, std::uint8_t right) {
        if (previous[left] != previous[right]) return previous[left] > previous[right];
        return left < right; // deterministic tie-break; paper does not specify it.
    });
    return ranked;
}

std::uint8_t rank_of(const Mapping& ranked, std::uint8_t branch) {
    for (std::uint8_t rank = 0U; rank < kBranches; ++rank)
        if (ranked[rank] == branch) return rank;
    throw std::runtime_error("branch outside 0..=3");
}

void write_dynamic_flag(BitWriter& writer, std::uint8_t rank) {
    switch (rank) {
        case 0U: writer.write_bit(false); break;              // 0
        case 1U: writer.write_bits(0b10U, 2U); break;         // 10
        case 2U: writer.write_bits(0b110U, 3U); break;        // 110
        case 3U: writer.write_bits(0b111U, 3U); break;        // 111
        default: throw std::runtime_error("invalid branch rank");
    }
}

std::uint8_t read_dynamic_flag(BitReader& reader) {
    if (!reader.read_bit()) return 0U;
    if (!reader.read_bit()) return 1U;
    return reader.read_bit() ? 3U : 2U;
}

void write_fixed_flag(BitWriter& writer, std::uint8_t branch) { writer.write_bits(branch, 2U); }
std::uint8_t read_fixed_flag(BitReader& reader) {
    return static_cast<std::uint8_t>((reader.read_bit() ? 2U : 0U) | (reader.read_bit() ? 1U : 0U));
}

bool predicted_dynamic_gain(const Counts& previous) {
    const Mapping ranked = rank_by_previous_counts(previous);
    const std::uint64_t top = previous[ranked[0]];
    const std::uint64_t third = previous[ranked[2]];
    const std::uint64_t fourth = previous[ranked[3]];
    // Rabbit's lemma: dynamic expected flag length < 2 iff x1 > x3 + x4.
    return top > third + fourth;
}

std::uint64_t l1_distance(const Counts& left, const Counts& right) {
    std::uint64_t total = 0U;
    for (std::size_t i = 0U; i < kBranches; ++i)
        total += left[i] > right[i] ? left[i] - right[i] : right[i] - left[i];
    return total;
}

bool stable_histories(const Counts& previous, const Counts& before_previous,
                      std::uint64_t block_size, std::uint64_t threshold_permille) {
    // Normalize total variation by block size. TV = L1 / (2N).
    return l1_distance(previous, before_previous) * 1000U <=
        2U * block_size * threshold_permille;
}

enum class Policy { RabbitPreviousBlock, GuardedPreviousBlock };

bool use_dynamic(Policy policy, std::size_t block_index, const Counts& previous,
                 const Counts& before_previous, std::uint64_t block_size) {
    if (policy == Policy::RabbitPreviousBlock) return true; // first block uses default rank {0,1,2,3}
    if (block_index == 0U) return false; // guarded control uses fixed flags while bootstrapping
    if (!predicted_dynamic_gain(previous)) return false;
    if (block_index == 1U) return true; // one prior block exists; gain condition is available
    return stable_histories(previous, before_previous, block_size, 100U); // TV <= 0.10
}

Encoded encode_flags(const std::vector<std::uint8_t>& branches, std::size_t block_size, Policy policy) {
    if (block_size == 0U) throw std::runtime_error("block size must be positive");
    BitWriter writer;
    Counts previous{};
    Counts before_previous{};
    const std::size_t block_count = branches.empty() ? 0U : (branches.size() - 1U) / block_size + 1U;
    for (std::size_t block = 0U; block < block_count; ++block) {
        const std::size_t start = block * block_size;
        const std::size_t end = std::min(branches.size(), start + block_size);
        const Mapping ranked = block == 0U ? identity_mapping() : rank_by_previous_counts(previous);
        const bool dynamic = use_dynamic(policy, block, previous, before_previous,
                                         static_cast<std::uint64_t>(end - start));
        Counts current{};
        for (std::size_t index = start; index < end; ++index) {
            const std::uint8_t branch = branches[index];
            if (branch >= kBranches) throw std::runtime_error("branch outside 0..=3");
            if (dynamic) write_dynamic_flag(writer, rank_of(ranked, branch));
            else write_fixed_flag(writer, branch);
            ++current[branch];
        }
        before_previous = previous;
        previous = current;
    }
    return std::move(writer).finish();
}

std::vector<std::uint8_t> decode_flags(const Encoded& encoded, std::size_t count,
                                       std::size_t block_size, Policy policy) {
    if (block_size == 0U) throw std::runtime_error("block size must be positive");
    if (encoded.bytes.size() != (encoded.bit_count + 7U) / 8U)
        throw std::runtime_error("noncanonical flag stream byte length");
    BitReader reader(encoded);
    std::vector<std::uint8_t> output;
    output.reserve(count);
    Counts previous{};
    Counts before_previous{};
    const std::size_t block_count = count == 0U ? 0U : (count - 1U) / block_size + 1U;
    for (std::size_t block = 0U; block < block_count; ++block) {
        const std::size_t start = block * block_size;
        const std::size_t end = std::min(count, start + block_size);
        const Mapping ranked = block == 0U ? identity_mapping() : rank_by_previous_counts(previous);
        const bool dynamic = use_dynamic(policy, block, previous, before_previous,
                                         static_cast<std::uint64_t>(end - start));
        Counts current{};
        for (std::size_t index = start; index < end; ++index) {
            const std::uint8_t rank_or_branch = dynamic ? read_dynamic_flag(reader) : read_fixed_flag(reader);
            const std::uint8_t branch = dynamic ? ranked[rank_or_branch] : rank_or_branch;
            output.push_back(branch);
            ++current[branch];
        }
        before_previous = previous;
        previous = current;
    }
    if (reader.position() != encoded.bit_count) throw std::runtime_error("trailing bits in flag stream");
    if ((encoded.bit_count & 7U) != 0U && !encoded.bytes.empty()) {
        const std::uint8_t mask = static_cast<std::uint8_t>((1U << (8U - (encoded.bit_count & 7U))) - 1U);
        if ((encoded.bytes.back() & mask) != 0U) throw std::runtime_error("nonzero padding bits");
    }
    return output;
}

std::size_t fixed_two_bit_cost(const std::vector<std::uint8_t>& branches) { return branches.size() * 2U; }
std::size_t same_block_oracle_cost(const std::vector<std::uint8_t>& branches, std::size_t block_size) {
    std::size_t total = 0U;
    for (std::size_t start = 0U; start < branches.size(); start += block_size) {
        const std::size_t end = std::min(branches.size(), start + block_size);
        Counts counts{};
        for (std::size_t index = start; index < end; ++index) ++counts.at(branches[index]);
        const Mapping ranked = rank_by_previous_counts(counts);
        const std::array<std::size_t, kBranches> lengths = {1U, 2U, 3U, 3U};
        for (std::size_t index = start; index < end; ++index)
            total += lengths[rank_of(ranked, branches[index])];
    }
    return total;
}

std::vector<std::uint8_t> repeated_blocks(const std::vector<std::array<std::uint32_t,4>>& patterns,
                                          std::size_t block_size) {
    std::vector<std::uint8_t> output;
    output.reserve(patterns.size() * block_size);
    for (const auto& pattern : patterns) {
        const std::uint32_t sum = std::accumulate(pattern.begin(), pattern.end(), 0U);
        if (sum != block_size) throw std::runtime_error("branch pattern must sum to block size");
        for (std::size_t branch = 0U; branch < kBranches; ++branch)
            output.insert(output.end(), pattern[branch], static_cast<std::uint8_t>(branch));
    }
    return output;
}

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void test_codewords_and_round_trip() {
    const std::vector<std::uint8_t> all = {0U,1U,2U,3U,0U,1U,2U,3U};
    for (const Policy policy : {Policy::RabbitPreviousBlock, Policy::GuardedPreviousBlock}) {
        const auto encoded = encode_flags(all, 4U, policy);
        require(decode_flags(encoded, all.size(), 4U, policy) == all, "flag mapper round-trip failed");
    }
}
void test_rabbit_gain_lemma() {
    const Counts skew{800U,100U,60U,40U};
    const Counts uniform{250U,250U,250U,250U};
    const Counts borderline{300U,300U,200U,200U};
    require(predicted_dynamic_gain(skew), "skew histogram should meet Rabbit gain condition");
    require(!predicted_dynamic_gain(uniform), "uniform histogram should fail Rabbit gain condition");
    require(!predicted_dynamic_gain(borderline), "borderline histogram should fail Rabbit gain condition");
}
void test_stationary_skew_and_uniform() {
    constexpr std::size_t block = 1000U;
    const auto skew = repeated_blocks({{800U,100U,60U,40U}}, block);
    const auto skew2 = repeated_blocks({{800U,100U,60U,40U},{800U,100U,60U,40U},{800U,100U,60U,40U}}, block);
    const auto rabbit = encode_flags(skew2, block, Policy::RabbitPreviousBlock);
    const auto guarded = encode_flags(skew2, block, Policy::GuardedPreviousBlock);
    require(decode_flags(rabbit, skew2.size(), block, Policy::RabbitPreviousBlock) == skew2, "skew round-trip failed");
    require(decode_flags(guarded, skew2.size(), block, Policy::GuardedPreviousBlock) == skew2, "guarded skew round-trip failed");
    require(rabbit.bit_count < fixed_two_bit_cost(skew2), "skew case should benefit from predicted coding after bootstrap");
    const auto uniform = repeated_blocks({{250U,250U,250U,250U},{250U,250U,250U,250U}}, block);
    const auto uniform_guarded = encode_flags(uniform, block, Policy::GuardedPreviousBlock);
    require(uniform_guarded.bit_count == fixed_two_bit_cost(uniform), "guarded uniform case should fall back to fixed two-bit flags");
    (void)skew;
}
void test_shift_and_alternation() {
    constexpr std::size_t block = 1000U;
    const auto shift = repeated_blocks({{800U,100U,60U,40U},{40U,60U,100U,800U},{40U,60U,100U,800U}}, block);
    for (const Policy policy : {Policy::RabbitPreviousBlock, Policy::GuardedPreviousBlock}) {
        const auto encoded = encode_flags(shift, block, policy);
        require(decode_flags(encoded, shift.size(), block, policy) == shift, "shift round-trip failed");
    }
    const auto alternating = repeated_blocks({{800U,100U,60U,40U},{40U,60U,100U,800U},
                                               {800U,100U,60U,40U},{40U,60U,100U,800U}}, block);
    const auto rabbit = encode_flags(alternating, block, Policy::RabbitPreviousBlock);
    const auto guarded = encode_flags(alternating, block, Policy::GuardedPreviousBlock);
    require(decode_flags(rabbit, alternating.size(), block, Policy::RabbitPreviousBlock) == alternating,
            "alternating Rabbit round-trip failed");
    require(decode_flags(guarded, alternating.size(), block, Policy::GuardedPreviousBlock) == alternating,
            "alternating guarded round-trip failed");
}
void test_determinism_and_bad_symbols() {
    const auto values = repeated_blocks({{700U,100U,100U,100U},{100U,700U,100U,100U}}, 1000U);
    const auto first = encode_flags(values, 1000U, Policy::RabbitPreviousBlock);
    const auto second = encode_flags(values, 1000U, Policy::RabbitPreviousBlock);
    require(first.bit_count == second.bit_count && first.bytes == second.bytes, "encoding is nondeterministic");
    auto invalid = values;
    invalid[0] = 4U;
    bool threw = false;
    try { (void)encode_flags(invalid, 1000U, Policy::RabbitPreviousBlock); }
    catch (const std::runtime_error&) { threw = true; }
    require(threw, "invalid branch accepted");
    Encoded malformed{{}, 8U};
    threw = false;
    try { (void)decode_flags(malformed, 1U, 1U, Policy::RabbitPreviousBlock); }
    catch (const std::runtime_error&) { threw = true; }
    require(threw, "malformed byte length accepted");
}

void report(const std::string& name, const std::vector<std::uint8_t>& branches,
            std::size_t block = kDefaultBlock) {
    const auto rabbit = encode_flags(branches, block, Policy::RabbitPreviousBlock);
    const auto guarded = encode_flags(branches, block, Policy::GuardedPreviousBlock);
    const std::size_t fixed = fixed_two_bit_cost(branches);
    const std::size_t oracle = same_block_oracle_cost(branches, block);
    require(decode_flags(rabbit, branches.size(), block, Policy::RabbitPreviousBlock) == branches,
            name + " Rabbit mapper decode mismatch");
    require(decode_flags(guarded, branches.size(), block, Policy::GuardedPreviousBlock) == branches,
            name + " guarded mapper decode mismatch");
    auto pct = [](std::size_t candidate, std::size_t baseline) {
        if (baseline == 0U) return 0.0;
        return (static_cast<double>(candidate) / static_cast<double>(baseline) - 1.0) * 100.0;
    };
    std::cout << name << ",block_size=" << block << ",symbols=" << branches.size()
              << ",fixed2_bits=" << fixed
              << ",rabbit_predictive_bits=" << rabbit.bit_count
              << ",guarded_bits=" << guarded.bit_count
              << ",same_block_oracle_bits=" << oracle
              << ",rabbit_vs_fixed_pct=" << pct(rabbit.bit_count, fixed)
              << ",guarded_vs_rabbit_pct=" << pct(guarded.bit_count, rabbit.bit_count)
              << '\n';
}
} // namespace

int main() {
    try {
        test_codewords_and_round_trip();
        test_rabbit_gain_lemma();
        test_stationary_skew_and_uniform();
        test_shift_and_alternation();
        test_determinism_and_bad_symbols();
        std::cout << "tests=5 status=PASS\n";
        constexpr std::size_t block = 1000U;
        report("stationary-skew", repeated_blocks({{800U,100U,60U,40U},{800U,100U,60U,40U},
                                                      {800U,100U,60U,40U},{800U,100U,60U,40U}}, block));
        report("uniform", repeated_blocks({{250U,250U,250U,250U},{250U,250U,250U,250U},
                                             {250U,250U,250U,250U},{250U,250U,250U,250U}}, block));
        const auto shift = repeated_blocks({{800U,100U,60U,40U},{40U,60U,100U,800U},
                                            {40U,60U,100U,800U},{40U,60U,100U,800U}}, block);
        report("one-block-shift", shift);
        report("one-block-shift-subblocks", shift, 250U);
        report("one-block-shift-subblocks", shift, 500U);
        const auto alternating = repeated_blocks({{800U,100U,60U,40U},{40U,60U,100U,800U},
                                                  {800U,100U,60U,40U},{40U,60U,100U,800U}}, block);
        report("alternating-dominant", alternating);
        report("alternating-subblocks", alternating, 250U);
        report("alternating-subblocks", alternating, 500U);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "branch mapper reference failure: " << error.what() << '\n';
        return 1;
    }
}
