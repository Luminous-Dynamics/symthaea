// Independent C++20 reference for CDL-Bench TRD1 typed telemetry state deltas.
// Transform-level reference only: no Rust linkage and no arithmetic-code frame.
#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {
constexpr std::size_t kFields = 8U;
constexpr std::uint32_t kMaxState = 1'000'000U;
constexpr std::uint32_t kMaxSegment = 1'000'000U;
constexpr std::array<std::uint8_t, kFields> kFieldKinds = {2U, 2U, 3U, 4U, 1U, 0U, 5U, 2U};

struct Record {
    std::uint64_t timestamp{};
    std::uint32_t entity{};
    std::uint32_t series{};
    std::uint8_t present{};
    std::array<std::uint64_t, kFields> values{};
    bool operator==(const Record&) const = default;
};

using Key = std::pair<std::uint32_t, std::uint32_t>;
struct StateEntry { Record record; std::uint64_t last_seen{}; };
using State = std::map<Key, StateEntry>;

void append_u64(std::uint64_t value, std::vector<std::uint8_t>& output) {
    for (unsigned shift = 0U; shift < 64U; shift += 8U)
        output.push_back(static_cast<std::uint8_t>(value >> shift));
}
void append_u32(std::uint32_t value, std::vector<std::uint8_t>& output) {
    for (unsigned shift = 0U; shift < 32U; shift += 8U)
        output.push_back(static_cast<std::uint8_t>(value >> shift));
}
void append_varint(std::uint64_t value, std::vector<std::uint8_t>& output) {
    while (value >= 0x80U) {
        output.push_back(static_cast<std::uint8_t>((value & 0x7fU) | 0x80U));
        value >>= 7U;
    }
    output.push_back(static_cast<std::uint8_t>(value));
}
bool read_varint(const std::vector<std::uint8_t>& input, std::size_t& cursor, std::uint64_t& value) {
    value = 0;
    for (std::size_t i = 0U; i < 10U; ++i) {
        if (cursor >= input.size()) return false;
        const std::uint8_t byte = input[cursor++];
        if (i == 9U && (byte & 0xfeU) != 0U) return false;
        value |= static_cast<std::uint64_t>(byte & 0x7fU) << (i * 7U);
        if ((byte & 0x80U) == 0U) {
            if (i > 0U && (byte & 0x7fU) == 0U) return false;
            return true;
        }
    }
    return false;
}

void evict_oldest(State& states) {
    if (states.empty()) throw std::runtime_error("cannot evict empty state table");
    auto victim = states.begin();
    for (auto current = std::next(states.begin()); current != states.end(); ++current) {
        if (std::tie(current->second.last_seen, current->first) <
            std::tie(victim->second.last_seen, victim->first)) victim = current;
    }
    states.erase(victim);
}

std::uint8_t changed_mask(const Record& previous, const Record& current) {
    std::uint8_t changed = 0U;
    for (std::size_t field = 0U; field < kFields; ++field) {
        const std::uint8_t bit = static_cast<std::uint8_t>(1U << field);
        const bool was_present = (previous.present & bit) != 0U;
        const bool is_present = (current.present & bit) != 0U;
        if (was_present != is_present || (is_present && previous.values[field] != current.values[field]))
            changed = static_cast<std::uint8_t>(changed | bit);
    }
    return changed;
}
bool validate_record(const Record& record) {
    for (std::size_t field = 0U; field < kFields; ++field) {
        const std::uint8_t bit = static_cast<std::uint8_t>(1U << field);
        const bool present = (record.present & bit) != 0U;
        const std::uint64_t value = record.values[field];
        if (!present && value != 0U) return false;
        if (present && field == 2U && value > UINT32_MAX) return false;
        if (present && field == 3U && value > 1U) return false;
    }
    return true;
}

std::vector<std::uint8_t> encode(const std::vector<Record>& records,
                                 std::uint32_t state_capacity,
                                 std::uint32_t segment_records) {
    if (state_capacity == 0U || state_capacity > kMaxState) throw std::runtime_error("bad state capacity");
    if (segment_records == 0U || segment_records > kMaxSegment) throw std::runtime_error("bad segment size");
    std::vector<std::uint8_t> output;
    output.reserve(records.size() * 88U);
    State states;
    std::optional<std::uint64_t> previous_timestamp;
    for (std::size_t index = 0U; index < records.size(); ++index) {
        const Record& record = records[index];
        if (!validate_record(record)) throw std::runtime_error("invalid typed record");
        const std::uint64_t segment_offset = static_cast<std::uint64_t>(index) % segment_records;
        if (segment_offset == 0U) { states.clear(); previous_timestamp.reset(); }
        if (previous_timestamp.has_value() && record.timestamp >= *previous_timestamp) {
            output.push_back(0U);
            append_varint(record.timestamp - *previous_timestamp, output);
        } else {
            output.push_back(1U);
            append_varint(record.timestamp, output);
        }
        previous_timestamp = record.timestamp;
        append_varint(record.entity, output);
        append_varint(record.series, output);

        const Key key{record.entity, record.series};
        const auto found = states.find(key);
        std::uint8_t state_mode = 0U;
        std::uint8_t changed = 0U;
        if (found != states.end()) changed = changed_mask(found->second.record, record);
        else {
            if (states.size() >= state_capacity) evict_oldest(states);
            state_mode = 1U;
            changed = 0xffU;
        }
        output.push_back(state_mode);
        output.push_back(changed);
        output.push_back(record.present);
        for (std::size_t field = 0U; field < kFields; ++field) {
            const std::uint8_t bit = static_cast<std::uint8_t>(1U << field);
            if ((changed & bit) != 0U && (record.present & bit) != 0U)
                append_u64(record.values[field], output);
        }
        states[key] = StateEntry{record, segment_offset};
    }
    return output;
}

bool decode(const std::vector<std::uint8_t>& input, std::size_t count,
            std::uint32_t state_capacity, std::uint32_t segment_records,
            std::vector<Record>& output) {
    output.clear();
    if (state_capacity == 0U || state_capacity > kMaxState ||
        segment_records == 0U || segment_records > kMaxSegment) return false;
    output.reserve(count);
    State states;
    std::optional<std::uint64_t> previous_timestamp;
    std::size_t cursor = 0U;
    for (std::size_t index = 0U; index < count; ++index) {
        const std::uint64_t segment_offset = static_cast<std::uint64_t>(index) % segment_records;
        if (segment_offset == 0U) { states.clear(); previous_timestamp.reset(); }
        if (cursor >= input.size()) return false;
        const std::uint8_t time_mode = input[cursor++];
        if (time_mode > 1U) return false;
        std::uint64_t time_value = 0U;
        if (!read_varint(input, cursor, time_value)) return false;
        std::uint64_t timestamp = 0U;
        if (!previous_timestamp.has_value()) {
            if (time_mode != 1U) return false;
            timestamp = time_value;
        } else if (time_mode == 0U) {
            if (time_value > UINT64_MAX - *previous_timestamp) return false;
            timestamp = *previous_timestamp + time_value;
        } else {
            if (time_value >= *previous_timestamp) return false;
            timestamp = time_value;
        }
        previous_timestamp = timestamp;

        std::uint64_t entity64 = 0U, series64 = 0U;
        if (!read_varint(input, cursor, entity64) || !read_varint(input, cursor, series64)) return false;
        if (entity64 > UINT32_MAX || series64 > UINT32_MAX) return false;
        const auto entity = static_cast<std::uint32_t>(entity64);
        const auto series = static_cast<std::uint32_t>(series64);
        const Key key{entity, series};
        const auto found = states.find(key);
        const bool has_previous = found != states.end();
        if (!has_previous && states.size() >= state_capacity) evict_oldest(states);
        if (input.size() - cursor < 3U) return false;
        const std::uint8_t state_mode = input[cursor++];
        const std::uint8_t changed = input[cursor++];
        const std::uint8_t present = input[cursor++];
        if (state_mode != (has_previous ? 0U : 1U)) return false;
        if (!has_previous && changed != 0xffU) return false;

        Record record{};
        record.timestamp = timestamp;
        record.entity = entity;
        record.series = series;
        record.present = present;
        if (has_previous) record.values = found->second.record.values;
        for (std::size_t field = 0U; field < kFields; ++field) {
            const std::uint8_t bit = static_cast<std::uint8_t>(1U << field);
            if ((changed & bit) != 0U) {
                if ((present & bit) != 0U) {
                    if (input.size() - cursor < 8U) return false;
                    std::uint64_t value = 0U;
                    for (unsigned byte = 0U; byte < 8U; ++byte)
                        value |= static_cast<std::uint64_t>(input[cursor++]) << (byte * 8U);
                    record.values[field] = value;
                } else record.values[field] = 0U;
            } else if (!has_previous ||
                       (((found->second.record.present & bit) != 0U) != ((present & bit) != 0U))) {
                return false;
            }
        }
        if (!validate_record(record)) return false;
        const std::uint8_t expected_changed = has_previous ? changed_mask(found->second.record, record) : 0xffU;
        if (changed != expected_changed) return false;
        states[key] = StateEntry{record, segment_offset};
        output.push_back(record);
    }
    return cursor == input.size();
}

void write_file(const std::filesystem::path& path, const std::vector<std::uint8_t>& bytes) {
    std::ofstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot create " + path.string());
    file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!file) throw std::runtime_error("write failed: " + path.string());
}
std::vector<std::uint8_t> raw_bytes(const std::vector<Record>& records) {
    std::vector<std::uint8_t> raw;
    raw.reserve(kFieldKinds.size() + records.size() * 81U);
    raw.insert(raw.end(), kFieldKinds.begin(), kFieldKinds.end());
    for (const auto& record : records) {
        append_u64(record.timestamp, raw);
        append_u32(record.entity, raw);
        append_u32(record.series, raw);
        raw.push_back(record.present);
        for (const auto value : record.values) append_u64(value, raw);
    }
    return raw;
}
std::uint64_t next_random(std::mt19937_64& rng) { return rng(); }
std::uint64_t value_for(std::size_t field, std::uint64_t random) {
    switch (field) {
        case 0U: return std::bit_cast<std::uint64_t>(static_cast<double>(random % 100'000U) / 1000.0);
        case 1U: return std::bit_cast<std::uint64_t>(static_cast<double>(random % 10'000'000U) / 100.0);
        case 2U: return random % 4096U;
        case 3U: return random & 1U;
        case 4U: return random;
        case 5U: return random;
        case 6U: return std::rotl(random, 17);
        default: return std::bit_cast<std::uint64_t>(static_cast<double>(random % 1'000'000U) / 10'000.0);
    }
}
std::vector<Record> fixture(const std::string& workload, std::uint64_t seed) {
    constexpr std::uint32_t keys = 64U, steps = 1024U;
    std::mt19937_64 rng(seed);
    std::array<Record, keys> states{};
    std::vector<Record> records;
    records.reserve(static_cast<std::size_t>(keys) * steps);
    for (std::uint32_t key = 0U; key < keys; ++key) {
        states[key].entity = key / 16U;
        states[key].series = key % 16U;
        states[key].present = 0xffU;
        for (std::size_t field = 0U; field < kFields; ++field)
            states[key].values[field] = value_for(field, next_random(rng));
    }
    for (std::uint32_t step = 0U; step < steps; ++step) {
        for (std::uint32_t key = 0U; key < keys; ++key) {
            Record record = states[key];
            record.timestamp = (static_cast<std::uint64_t>(step) * keys + key) * 1000U;
            for (std::size_t field = 0U; field < kFields; ++field) {
                const std::uint64_t random = next_random(rng);
                bool update = false;
                if (workload == "sparse") update = (random % 1000U) < 15U;
                else if (workload == "idle") update = (random % 10'000U) < 2U;
                else if (workload == "bursty")
                    update = (step % 256U < 24U) ? (random % 100U < 65U) : (random % 1000U < 5U);
                else if (workload == "noisy") update = true;
                if (update) {
                    const std::uint8_t bit = static_cast<std::uint8_t>(1U << field);
                    if ((std::rotl(random, 11) % 2048U) == 0U)
                        record.present = static_cast<std::uint8_t>(record.present ^ bit);
                    if ((record.present & bit) == 0U) record.values[field] = 0U;
                    else record.values[field] = value_for(field, next_random(rng));
                }
            }
            records.push_back(record);
            states[key] = record;
        }
    }
    return records;
}
void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
void test_roundtrips() {
    std::mt19937_64 rng(0x20261119U);
    for (std::size_t trial = 0U; trial < 200U; ++trial) {
        const std::uint32_t cap = 1U + static_cast<std::uint32_t>(rng() % 12U);
        const std::uint32_t segment = 1U + static_cast<std::uint32_t>(rng() % 50U);
        const std::uint32_t keys = 1U + static_cast<std::uint32_t>(rng() % 25U);
        const std::size_t count = static_cast<std::size_t>(rng() % 500U);
        std::vector<Record> records;
        records.reserve(count);
        std::vector<Record> states(keys);
        std::vector<bool> initialized(keys, false);
        std::int64_t timestamp = 1000;
        for (std::size_t index = 0U; index < count; ++index) {
            const std::uint32_t key = static_cast<std::uint32_t>(rng() % keys);
            timestamp += static_cast<std::int64_t>(rng() % 7U) - 3;
            timestamp = std::clamp<std::int64_t>(timestamp, 0, static_cast<std::int64_t>(UINT32_MAX));
            Record record = initialized[key] ? states[key] : Record{};
            record.timestamp = static_cast<std::uint64_t>(timestamp);
            record.entity = key / 5U;
            record.series = key % 5U;
            if (!initialized[key]) record.present = 0xffU;
            for (std::size_t field = 0U; field < kFields; ++field) {
                const std::uint64_t random = rng();
                const std::uint8_t bit = static_cast<std::uint8_t>(1U << field);
                if (!initialized[key] || (random % 100U) < 5U) {
                    record.present = static_cast<std::uint8_t>(record.present ^ (initialized[key] ? bit : 0U));
                    if (!initialized[key] || (record.present & bit) != 0U) record.values[field] = value_for(field, rng);
                    else record.values[field] = 0U;
                } else if ((record.present & bit) != 0U && (random % 100U) < 8U) {
                    record.values[field] = value_for(field, rng);
                }
            }
            records.push_back(record);
            states[key] = record;
            initialized[key] = true;
        }
        const auto bytes = encode(records, cap, segment);
        std::vector<Record> decoded;
        require(decode(bytes, records.size(), cap, segment, decoded), "random stream decode rejected");
        require(decoded == records, "random stream mismatch");
    }
}
bool reject(const std::vector<std::uint8_t>& bytes, std::size_t count,
            std::uint32_t cap, std::uint32_t segment) {
    std::vector<Record> decoded;
    return !decode(bytes, count, cap, segment, decoded);
}
void test_malformed() {
    require(reject({0U}, 1U, 1U, 8U), "accepted truncated timestamp");
    require(reject({1U, 0x80U, 0x00U}, 1U, 1U, 8U), "accepted noncanonical timestamp");
    require(reject({2U}, 1U, 1U, 8U), "accepted invalid timestamp mode");
    require(reject({1U, 0U, 0U, 0U}, 1U, 1U, 8U), "accepted truncated controls");
    const auto sample = fixture("sparse", 42U);
    const auto encoded = encode(sample, 64U, 4096U);
    auto truncated = encoded;
    truncated.pop_back();
    require(reject(truncated, sample.size(), 64U, 4096U), "accepted truncated final value");
    auto trailing = encoded;
    trailing.push_back(0U);
    require(reject(trailing, sample.size(), 64U, 4096U), "accepted trailing byte");
}
void write_fixture_files(const std::filesystem::path& directory) {
    std::filesystem::create_directories(directory);
    for (const std::string name : {"sparse", "bursty", "noisy", "idle"}) {
        const auto records = fixture(name, 0x7e1e0001U);
        const auto encoded = encode(records, 64U, 4096U);
        std::vector<Record> decoded;
        require(decode(encoded, records.size(), 64U, 4096U, decoded), name + " decode failed");
        require(decoded == records, name + " mismatch");
        const auto raw = raw_bytes(records);
        write_file(directory / (name + "_raw.bin"), raw);
        write_file(directory / (name + "_state_delta.bin"), encoded);
        const double ratio = raw.empty() ? 0.0 : static_cast<double>(encoded.size()) / static_cast<double>(raw.size());
        std::cout << name << ": records=" << records.size() << " raw_bytes=" << raw.size()
                  << " state_delta_bytes=" << encoded.size() << " transform_ratio=" << ratio << "\n";
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        test_roundtrips();
        test_malformed();
        std::cout << "random_roundtrip_trials=200\nmalformed_rejections=6\nreference_status=PASS\n";
        if (argc == 3 && std::string(argv[1]) == "--emit-dir") write_fixture_files(argv[2]);
        else if (argc != 1) {
            std::cerr << "usage: " << argv[0] << " [--emit-dir DIRECTORY]\n";
            return 2;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "reference failure: " << error.what() << "\n";
        return 1;
    }
}
