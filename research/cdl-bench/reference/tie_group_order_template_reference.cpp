// Independent C++20 reference for repeated timestamp tie-group order templates.
// Validates schedule coding only; it does not compile the integrated PRN1 frame.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct BitWriter {
    std::vector<std::uint8_t> bytes;
    std::size_t bits{};
    void put(std::uint32_t value, std::size_t width) {
        for (std::size_t i = 0; i < width; ++i) {
            if ((bits & 7U) == 0U) bytes.push_back(0U);
            if (((value >> (width - i - 1U)) & 1U) != 0U)
                bytes.back() |= static_cast<std::uint8_t>(1U << (7U - (bits & 7U)));
            ++bits;
        }
    }
};
struct BitReader {
    const std::vector<std::uint8_t>& bytes;
    std::size_t bits{};
    std::size_t pos{};
    bool get(std::size_t width, std::uint32_t& value) {
        if (width > 32U || pos > bits || bits - pos < width) return false;
        value = 0U;
        for (std::size_t i = 0; i < width; ++i) {
            value = (value << 1U) | ((bytes[pos / 8U] >> (7U - (pos & 7U))) & 1U);
            ++pos;
        }
        return true;
    }
};
std::size_t id_width(std::size_t key_count) {
    std::size_t width = 0U;
    std::size_t value = key_count > 0U ? key_count - 1U : 0U;
    while (value > 0U) { ++width; value >>= 1U; }
    return width;
}
using Groups = std::vector<std::vector<std::uint32_t>>;
std::vector<std::uint8_t> encode(const Groups& groups, std::size_t key_count) {
    const std::size_t width = id_width(key_count);
    constexpr std::size_t max_templates = 16U;
    std::vector<std::uint8_t> out;
    std::vector<std::vector<std::uint32_t>> templates;
    for (const auto& group : groups) {
        if (group.size() <= 1U) continue;
        for (const auto id : group) if (id >= key_count) throw std::runtime_error("invalid key ID");
        const auto found = std::find(templates.begin(), templates.end(), group);
        if (found != templates.end()) {
            const auto index = static_cast<std::size_t>(found - templates.begin());
            out.push_back(static_cast<std::uint8_t>(index + 1U));
            auto reused = *found;
            templates.erase(found);
            templates.insert(templates.begin(), std::move(reused));
        } else {
            out.push_back(0U); // literal key-ID order
            BitWriter literal;
            for (const auto id : group) literal.put(id, width);
            out.insert(out.end(), literal.bytes.begin(), literal.bytes.end());
            templates.insert(templates.begin(), group);
            if (templates.size() > max_templates) templates.pop_back();
        }
    }
    return out;
}
bool decode(const std::vector<std::uint8_t>& bytes, const std::vector<std::size_t>& group_sizes,
            const std::vector<std::vector<std::size_t>>& expected_key_counts,
            std::size_t key_count, Groups& decoded) {
    decoded.clear();
    if (group_sizes.size() != expected_key_counts.size()) return false;
    const std::size_t width = id_width(key_count);
    std::size_t cursor = 0U;
    constexpr std::size_t max_templates = 16U;
    std::vector<std::vector<std::uint32_t>> templates;
    for (std::size_t group_index = 0U; group_index < group_sizes.size(); ++group_index) {
        const std::size_t count = group_sizes[group_index];
        if (count == 0U || expected_key_counts[group_index].size() != key_count) return false;
        if (count == 1U) {
            std::vector<std::uint32_t> singleton;
            for (std::size_t id = 0U; id < key_count; ++id) {
                if (expected_key_counts[group_index][id] == 1U)
                    singleton.push_back(static_cast<std::uint32_t>(id));
            }
            if (singleton.size() != 1U) return false;
            decoded.push_back(std::move(singleton)); // singleton groups require no schedule
            continue;
        }
        if (cursor >= bytes.size()) return false;
        const std::uint8_t control = bytes[cursor++];
        std::vector<std::uint32_t> ids;
        if (control == 0U) {
            if (width != 0U && count > static_cast<std::size_t>(-1) / width) return false;
            const std::size_t literal_bits = count * width;
            if (literal_bits > static_cast<std::size_t>(-1) - 7U) return false;
            const std::size_t literal_bytes = (literal_bits + 7U) / 8U;
            if (cursor > bytes.size() || literal_bytes > bytes.size() - cursor) return false;
            std::vector<std::uint8_t> literal(bytes.begin() + static_cast<std::ptrdiff_t>(cursor),
                                              bytes.begin() + static_cast<std::ptrdiff_t>(cursor + literal_bytes));
            if ((literal_bits & 7U) != 0U && !literal.empty()) {
                const auto mask = static_cast<std::uint8_t>((1U << (8U - (literal_bits & 7U))) - 1U);
                if ((literal.back() & mask) != 0U) return false;
            }
            BitReader reader{literal, literal_bits, 0U};
            ids.reserve(count);
            for (std::size_t i = 0U; i < count; ++i) {
                std::uint32_t id = 0U;
                if (!reader.get(width, id) || id >= key_count) return false;
                ids.push_back(id);
            }
            if (reader.pos != literal_bits) return false;
            cursor += literal_bytes;
            templates.insert(templates.begin(), ids);
            if (templates.size() > max_templates) templates.pop_back();
        } else {
            const std::size_t template_index = static_cast<std::size_t>(control - 1U);
            if (template_index >= templates.size() || templates[template_index].size() != count) return false;
            ids = templates[template_index];
            auto reused = templates[template_index];
            templates.erase(templates.begin() + static_cast<std::ptrdiff_t>(template_index));
            templates.insert(templates.begin(), std::move(reused));
        }
        std::vector<std::size_t> observed(key_count, 0U);
        for (const auto id : ids) {
            if (id >= key_count) return false;
            ++observed[id];
        }
        if (observed != expected_key_counts[group_index]) return false;
        decoded.push_back(std::move(ids));
    }
    return cursor == bytes.size();
}
void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
void test_repeated_template() {
    Groups groups;
    for (std::size_t i = 0U; i < 64U; ++i) groups.push_back({0U,1U,2U,3U,4U,5U,6U,7U});
    const auto encoded = encode(groups, 8U);
    require(encoded.size() == 1U + 3U + 63U, "repeated templates should be one literal plus 63 reuse controls");
    std::vector<std::size_t> sizes(64U, 8U);
    std::vector<std::vector<std::size_t>> counts(64U, std::vector<std::size_t>(8U, 1U));
    Groups decoded;
    require(decode(encoded, sizes, counts, 8U, decoded), "repeated template decode failed");
    require(decoded == groups, "repeated template mismatch");
}
void test_changed_templates() {
    Groups groups;
    for (std::size_t batch = 0U; batch < 20U; ++batch) {
        std::vector<std::uint32_t> group(16U);
        for (std::size_t i = 0U; i < group.size(); ++i)
            group[i] = static_cast<std::uint32_t>((i + (batch % 2U)) % 16U);
        groups.push_back(std::move(group));
    }
    const auto encoded = encode(groups, 16U);
    std::vector<std::size_t> sizes(groups.size(), 16U);
    std::vector<std::vector<std::size_t>> counts(groups.size(), std::vector<std::size_t>(16U, 1U));
    Groups decoded;
    require(decode(encoded, sizes, counts, 16U, decoded), "changed template decode failed");
    require(decoded == groups, "changed template mismatch");
    require(encoded.size() < groups.size() * 1U + groups.size() * 16U, "template should beat literal schedule");
}
void test_randomized() {
    std::mt19937_64 rng(0x71e5U);
    for (std::size_t trial = 0U; trial < 1000U; ++trial) {
        const std::size_t key_count = 1U + static_cast<std::size_t>(rng() % 32U);
        const std::size_t group_count = 1U + static_cast<std::size_t>(rng() % 80U);
        Groups groups;
        std::vector<std::size_t> sizes;
        std::vector<std::vector<std::size_t>> counts;
        std::vector<std::uint32_t> previous;
        for (std::size_t g = 0U; g < group_count; ++g) {
            const std::size_t count = 1U + static_cast<std::size_t>(rng() % 24U);
            std::vector<std::uint32_t> group;
            std::vector<std::size_t> per_key(key_count, 0U);
            for (std::size_t i = 0U; i < count; ++i) {
                const auto id = static_cast<std::uint32_t>(rng() % key_count);
                group.push_back(id);
                ++per_key[id];
            }
            if (g > 0U && (rng() % 3U) == 0U && previous.size() == count) {
                group = previous;
                std::fill(per_key.begin(), per_key.end(), 0U);
                for (const auto id : group) ++per_key[id];
            }
            sizes.push_back(count);
            counts.push_back(std::move(per_key));
            groups.push_back(group);
            if (count > 1U) previous = group;
        }
        const auto encoded = encode(groups, key_count);
        Groups decoded;
        require(decode(encoded, sizes, counts, key_count, decoded), "random schedule rejected");
        require(decoded == groups, "random schedule mismatch");
    }
}
void report_template_sizes() {
    Groups repeated;
    for (std::size_t i = 0U; i < 64U; ++i) repeated.push_back({0U,1U,2U,3U,4U,5U,6U,7U});
    const auto repeated_bytes = encode(repeated, 8U);
    const std::size_t repeated_fixed = (64U * 8U * id_width(8U) + 7U) / 8U;

    Groups alternating;
    for (std::size_t batch = 0U; batch < 20U; ++batch) {
        std::vector<std::uint32_t> group(16U);
        for (std::size_t i = 0U; i < group.size(); ++i)
            group[i] = static_cast<std::uint32_t>((i + (batch % 2U)) % 16U);
        alternating.push_back(std::move(group));
    }
    const auto alternating_bytes = encode(alternating, 16U);
    const std::size_t alternating_fixed = (20U * 16U * id_width(16U) + 7U) / 8U;
    const double repeated_saving = (1.0 - static_cast<double>(repeated_bytes.size()) / static_cast<double>(repeated_fixed)) * 100.0;
    const double alternating_saving = (1.0 - static_cast<double>(alternating_bytes.size()) / static_cast<double>(alternating_fixed)) * 100.0;
    std::cout << "repeated_tie_groups_bytes=" << repeated_bytes.size()
              << " fixed_schedule_bytes=" << repeated_fixed
              << " saving_pct=" << repeated_saving << "\n";
    std::cout << "alternating_tie_templates_bytes=" << alternating_bytes.size()
              << " fixed_schedule_bytes=" << alternating_fixed
              << " saving_pct=" << alternating_saving << "\n";
}
void test_malformed_and_count_validation() {
    Groups decoded;
    require(!decode({2U}, {2U}, {{1U,1U}}, 2U, decoded), "unknown control accepted");
    require(!decode({1U}, {2U}, {{1U,1U}}, 2U, decoded), "truncated literal accepted");
    require(!decode({0U}, {2U}, {{1U,1U}}, 2U, decoded), "reuse without template accepted");
    require(!decode({1U,0x7fU}, {2U}, {{1U,1U}}, 2U, decoded), "nonzero padding accepted");
    const std::vector<std::uint8_t> valid = {1U,0x10U}; // literal IDs are 0, 1; expected multiset is 0, 2
    require(!decode(valid, {2U}, {{1U,0U,1U,0U}}, 4U, decoded), "invalid key multiplicity accepted");
}
}
int main() {
    try {
        test_repeated_template();
        test_changed_templates();
        test_randomized();
        test_malformed_and_count_validation();
        report_template_sizes();
        std::cout << "repeated_template=PASS changed_template=PASS randomized_groups=1000 malformed_rejections=5 status=PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "tie schedule reference failed: " << e.what() << '\n';
        return 1;
    }
}
