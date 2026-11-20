// Independent C++20 reference for typed telemetry state-delta event coding.
// This is a transform-level experiment, not the Rust CDL-Bench codec/frame.
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
constexpr std::size_t kFields = 8U;
constexpr std::uint64_t kMaxRecords = 10'000'000ULL;

struct Record {
    std::uint64_t timestamp{};
    std::uint32_t key{};
    std::uint8_t present{};
    std::array<std::uint64_t, kFields> values{};
    bool operator==(const Record&) const = default;
};

void write_u32(std::uint32_t value, std::vector<std::uint8_t>& out) {
    for (unsigned shift = 0; shift < 32U; shift += 8U) {
        out.push_back(static_cast<std::uint8_t>(value >> shift));
    }
}
void write_u64(std::uint64_t value, std::vector<std::uint8_t>& out) {
    for (unsigned shift = 0; shift < 64U; shift += 8U) {
        out.push_back(static_cast<std::uint8_t>(value >> shift));
    }
}

bool read_u64(const std::vector<std::uint8_t>& in, std::size_t& pos, std::uint64_t& value) {
    if (in.size() - pos < 8U) return false;
    value = 0;
    for (unsigned i = 0; i < 8U; ++i) value |= static_cast<std::uint64_t>(in[pos++]) << (i * 8U);
    return true;
}

void write_varint(std::uint64_t value, std::vector<std::uint8_t>& out) {
    while (value >= 0x80U) {
        out.push_back(static_cast<std::uint8_t>((value & 0x7fU) | 0x80U));
        value >>= 7U;
    }
    out.push_back(static_cast<std::uint8_t>(value));
}
bool read_varint(const std::vector<std::uint8_t>& in, std::size_t& pos, std::uint64_t& value) {
    value = 0;
    for (std::size_t i = 0; i < 10U; ++i) {
        if (pos >= in.size()) return false;
        const auto byte = in[pos++];
        if (i == 9U && (byte & 0xfeU) != 0U) return false;
        value |= static_cast<std::uint64_t>(byte & 0x7fU) << (i * 7U);
        if ((byte & 0x80U) == 0U) {
            if (i > 0U && (byte & 0x7fU) == 0U) return false;
            return true;
        }
    }
    return false;
}

bool write_file(const std::string& path, const std::vector<std::uint8_t>& bytes) {
    std::ofstream file(path, std::ios::binary);
    if (!file) return false;
    file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    return static_cast<bool>(file);
}

std::vector<std::uint8_t> encode_raw(const std::vector<Record>& records) {
    std::vector<std::uint8_t> out{'T', 'R', 'R', '0'};
    write_u64(static_cast<std::uint64_t>(records.size()), out);
    for (const auto& record : records) {
        write_u64(record.timestamp, out);
        write_u32(record.key, out);
        out.push_back(record.present);
        for (const auto value : record.values) write_u64(value, out);
    }
    return out;
}

// TRD1: timestamp mode + varint delta/absolute, key varint, changed-field bitmap,
// then (present byte, exact u64 bits if present) for each changed field.
std::vector<std::uint8_t> encode_state_delta(const std::vector<Record>& records) {
    std::vector<std::uint8_t> out{'T', 'R', 'D', '1'};
    write_u64(static_cast<std::uint64_t>(records.size()), out);
    std::map<std::uint32_t, Record> previous_by_key;
    std::uint64_t previous_timestamp = 0;
    bool has_timestamp = false;

    for (const auto& record : records) {
        if (!has_timestamp || record.timestamp < previous_timestamp) {
            out.push_back(1U); // absolute timestamp
            write_varint(record.timestamp, out);
        } else {
            out.push_back(0U); // nonnegative delta from previous stream record
            write_varint(record.timestamp - previous_timestamp, out);
        }
        previous_timestamp = record.timestamp;
        has_timestamp = true;
        write_varint(record.key, out);

        const auto found = previous_by_key.find(record.key);
        std::uint8_t changed = 0;
        if (found == previous_by_key.end()) {
            changed = 0xffU; // first sighting is a full checkpoint
        } else {
            const auto& previous = found->second;
            for (std::size_t field = 0; field < kFields; ++field) {
                const auto bit = static_cast<std::uint8_t>(1U << field);
                const bool now_present = (record.present & bit) != 0U;
                const bool was_present = (previous.present & bit) != 0U;
                if (now_present != was_present || (now_present && record.values[field] != previous.values[field])) {
                    changed = static_cast<std::uint8_t>(changed | bit);
                }
            }
        }
        out.push_back(changed);
        for (std::size_t field = 0; field < kFields; ++field) {
            const auto bit = static_cast<std::uint8_t>(1U << field);
            if ((changed & bit) == 0U) continue;
            const bool present = (record.present & bit) != 0U;
            out.push_back(present ? 1U : 0U);
            if (present) write_u64(record.values[field], out);
        }
        previous_by_key[record.key] = record;
    }
    return out;
}

bool decode_state_delta(const std::vector<std::uint8_t>& in, std::vector<Record>& records) {
    records.clear();
    if (in.size() < 12U || in[0] != 'T' || in[1] != 'R' || in[2] != 'D' || in[3] != '1') return false;
    std::size_t pos = 4U;
    std::uint64_t count = 0;
    if (!read_u64(in, pos, count) || count > kMaxRecords || count > std::numeric_limits<std::size_t>::max()) return false;
    records.reserve(static_cast<std::size_t>(count));
    std::map<std::uint32_t, Record> previous_by_key;
    std::uint64_t previous_timestamp = 0;
    bool has_timestamp = false;

    for (std::uint64_t row = 0; row < count; ++row) {
        if (pos >= in.size()) return false;
        const auto mode = in[pos++];
        if (mode > 1U) return false;
        std::uint64_t time_value = 0;
        if (!read_varint(in, pos, time_value)) return false;
        std::uint64_t timestamp = 0;
        if (mode == 1U) {
            timestamp = time_value;
        } else {
            if (!has_timestamp || time_value > std::numeric_limits<std::uint64_t>::max() - previous_timestamp) return false;
            timestamp = previous_timestamp + time_value;
        }
        previous_timestamp = timestamp;
        has_timestamp = true;

        std::uint64_t key64 = 0;
        if (!read_varint(in, pos, key64) || key64 > std::numeric_limits<std::uint32_t>::max()) return false;
        const auto key = static_cast<std::uint32_t>(key64);
        if (pos >= in.size()) return false;
        const auto changed = in[pos++];
        const auto found = previous_by_key.find(key);
        if (found == previous_by_key.end() && changed != 0xffU) return false;

        Record record{};
        record.timestamp = timestamp;
        record.key = key;
        if (found != previous_by_key.end()) {
            record.present = found->second.present;
            record.values = found->second.values;
        }
        for (std::size_t field = 0; field < kFields; ++field) {
            const auto bit = static_cast<std::uint8_t>(1U << field);
            if ((changed & bit) == 0U) continue;
            if (pos >= in.size()) return false;
            const auto present = in[pos++];
            if (present > 1U) return false;
            if (present == 0U) {
                record.present = static_cast<std::uint8_t>(record.present & static_cast<std::uint8_t>(~bit));
                record.values[field] = 0U;
            } else {
                std::uint64_t value = 0;
                if (!read_u64(in, pos, value)) return false;
                record.present = static_cast<std::uint8_t>(record.present | bit);
                record.values[field] = value;
            }
        }
        previous_by_key[key] = record;
        records.push_back(record);
    }
    return pos == in.size();
}

std::vector<Record> make_fixture(const std::string& mode, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    constexpr std::uint32_t entities = 32U;
    constexpr std::uint32_t steps = 2048U;
    std::vector<Record> records;
    records.reserve(static_cast<std::size_t>(entities) * steps);
    std::array<Record, entities> states{};
    for (std::uint32_t key = 0; key < entities; ++key) {
        states[key].key = key;
        states[key].present = 0xffU;
        for (auto& value : states[key].values) value = rng();
    }

    for (std::uint32_t step = 0; step < steps; ++step) {
        for (std::uint32_t key = 0; key < entities; ++key) {
            auto& state = states[key];
            state.timestamp = static_cast<std::uint64_t>(step) * 1'000'000ULL + key;
            state.key = key;
            for (std::size_t field = 0; field < kFields; ++field) {
                bool update = false;
                if (mode == "sparse") update = (rng() % 1000U) < 15U;
                else if (mode == "bursty") update = ((step % 256U) < 24U) ? ((rng() % 100U) < 65U) : ((rng() % 1000U) < 5U);
                else if (mode == "noisy") update = true;
                else if (mode == "idle") update = (rng() % 10000U) < 2U;
                else throw std::runtime_error("unknown fixture mode");

                if (update) {
                    if ((rng() % 1000U) == 0U) state.present = static_cast<std::uint8_t>(state.present ^ static_cast<std::uint8_t>(1U << field));
                    const auto bit = static_cast<std::uint8_t>(1U << field);
                    if ((state.present & bit) != 0U) {
                        state.values[field] = rng();
                        if ((rng() % 4096U) == 0U) state.values[field] = 0x8000000000000000ULL;
                        if ((rng() % 4096U) == 0U) state.values[field] = 0x7ff8000000000001ULL;
                    } else {
                        state.values[field] = 0U;
                    }
                }
            }
            records.push_back(state);
        }
    }
    return records;
}

bool run_fixture(const std::string& name, const std::vector<Record>& source,
                 const std::filesystem::path& emit_directory) {
    const auto raw = encode_raw(source);
    const auto encoded = encode_state_delta(source);
    std::vector<Record> decoded;
    if (!decode_state_delta(encoded, decoded) || decoded != source) {
        std::cerr << "round-trip failed: " << name << "\n";
        return false;
    }
    if (!emit_directory.empty()) {
        if (!write_file((emit_directory / (name + "_raw.bin")).string(), raw) ||
            !write_file((emit_directory / (name + "_state_delta.bin")).string(), encoded)) {
            std::cerr << "failed to write benchmark artifacts: " << name << "\n";
            return false;
        }
    }
    const double ratio = raw.empty() ? 0.0 : static_cast<double>(encoded.size()) / static_cast<double>(raw.size());
    std::cout << name << ": records=" << source.size()
              << " raw_bytes=" << raw.size()
              << " state_delta_bytes=" << encoded.size()
              << " ratio=" << ratio
              << " reduction_pct=" << (1.0 - ratio) * 100.0 << "\n";
    return true;
}

bool rejected(const std::vector<std::uint8_t>& bytes, const char* label) {
    std::vector<Record> output;
    if (decode_state_delta(bytes, output)) {
        std::cerr << "accepted malformed input: " << label << "\n";
        return false;
    }
    return true;
}
} // namespace

int main(int argc, char** argv) {
    std::filesystem::path emit_directory;
    if (argc == 3 && std::string(argv[1]) == "--emit-dir") {
        emit_directory = argv[2];
        std::error_code ec;
        std::filesystem::create_directories(emit_directory, ec);
        if (ec) {
            std::cerr << "cannot create artifact directory: " << ec.message() << "\n";
            return 2;
        }
    } else if (argc != 1) {
        std::cerr << "usage: " << argv[0] << " [--emit-dir DIRECTORY]\n";
        return 2;
    }

    if (!run_fixture("sparse-updates", make_fixture("sparse", 0x5eedULL), emit_directory) ||
        !run_fixture("bursty-updates", make_fixture("bursty", 0x5eedULL), emit_directory) ||
        !run_fixture("noisy-updates", make_fixture("noisy", 0x5eedULL), emit_directory) ||
        !run_fixture("mostly-idle", make_fixture("idle", 0x5eedULL), emit_directory)) return 1;

    auto one = make_fixture("sparse", 0x1234ULL);
    one.resize(3U);
    auto valid = encode_state_delta(one);
    if (!rejected({valid.begin(), valid.end() - 1}, "truncated last value")) return 1;
    auto truncated_header = valid;
    truncated_header.resize(7U);
    if (!rejected(truncated_header, "truncated header")) return 1;

    auto bad_mode = valid;
    bad_mode[12] = 2U;
    if (!rejected(bad_mode, "unknown timestamp mode")) return 1;
    auto bad_presence = valid;
    bad_presence[16] = 2U;
    if (!rejected(bad_presence, "invalid presence flag")) return 1;
    auto trailing = valid;
    trailing.push_back(0U);
    if (!rejected(trailing, "trailing byte")) return 1;
    std::cout << "malformed_inputs_rejected=5\nreference_status=PASS\n";
    return 0;
}
