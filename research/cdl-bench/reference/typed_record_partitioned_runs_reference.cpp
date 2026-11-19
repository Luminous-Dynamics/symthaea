// Independent C++20 prototype: per-series state/run streams plus an order-restoration schedule.
// Exact order-preserving, transform-level research reference; not a production container.
#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <queue>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {
constexpr std::size_t kFields=8U;
constexpr std::size_t kKeys=64U;
constexpr std::size_t kSteps=1024U;
constexpr std::array<std::uint8_t,kFields> kSchema={2U,2U,3U,4U,1U,0U,5U,2U};
struct Record {
    std::uint64_t timestamp{};
    std::uint32_t entity{};
    std::uint32_t series{};
    std::uint8_t present{};
    std::array<std::uint64_t,kFields> values{};
    bool operator==(const Record&) const=default;
};
using Key=std::pair<std::uint32_t,std::uint32_t>;
Key key_of(const Record& r){return {r.entity,r.series};}
bool same_state(const Record& a,const Record& b){return a.entity==b.entity&&a.series==b.series&&a.present==b.present&&a.values==b.values;}
std::uint8_t changed_mask(const Record& a,const Record& b){
    std::uint8_t mask=0U;
    for(std::size_t i=0U;i<kFields;++i){const auto bit=static_cast<std::uint8_t>(1U<<i);const bool ap=(a.present&bit)!=0U,bp=(b.present&bit)!=0U;if(ap!=bp||(bp&&a.values[i]!=b.values[i]))mask=static_cast<std::uint8_t>(mask|bit);}
    return mask;
}
bool valid_record(const Record& r){
    for(std::size_t i=0U;i<kFields;++i){const auto bit=static_cast<std::uint8_t>(1U<<i);if((r.present&bit)==0U&&r.values[i]!=0U)return false;if(i==2U&&(r.present&bit)!=0U&&r.values[i]>UINT32_MAX)return false;if(i==3U&&(r.present&bit)!=0U&&r.values[i]>1U)return false;}
    return true;
}
void put_u32(std::uint32_t x,std::vector<std::uint8_t>& o){for(unsigned s=0U;s<32U;s+=8U)o.push_back(static_cast<std::uint8_t>(x>>s));}
void put_u64(std::uint64_t x,std::vector<std::uint8_t>& o){for(unsigned s=0U;s<64U;s+=8U)o.push_back(static_cast<std::uint8_t>(x>>s));}
void put_var(std::uint64_t x,std::vector<std::uint8_t>& o){while(x>=128U){o.push_back(static_cast<std::uint8_t>((x&127U)|128U));x>>=7U;}o.push_back(static_cast<std::uint8_t>(x));}
bool get_u32(const std::vector<std::uint8_t>& b,std::size_t& p,std::uint32_t& x){if(p>b.size()||b.size()-p<4U)return false;x=0U;for(unsigned i=0U;i<4U;++i)x|=static_cast<std::uint32_t>(b[p++])<<(i*8U);return true;}
bool get_u64(const std::vector<std::uint8_t>& b,std::size_t& p,std::uint64_t& x){if(p>b.size()||b.size()-p<8U)return false;x=0U;for(unsigned i=0U;i<8U;++i)x|=static_cast<std::uint64_t>(b[p++])<<(i*8U);return true;}
bool get_var(const std::vector<std::uint8_t>& b,std::size_t& p,std::uint64_t& x){x=0U;for(std::size_t i=0U;i<10U;++i){if(p>=b.size())return false;const auto v=b[p++];if(i==9U&&(v&0xfeU)!=0U)return false;x|=static_cast<std::uint64_t>(v&0x7fU)<<(i*7U);if((v&0x80U)==0U)return !(i>0U&&(v&0x7fU)==0U);}return false;}
std::uint32_t crc32(const std::uint8_t* data,std::size_t size){std::uint32_t crc=0xffffffffU;for(std::size_t i=0U;i<size;++i){crc^=data[i];for(unsigned j=0U;j<8U;++j)crc=(crc>>1U)^(0xedb88320U&static_cast<std::uint32_t>(-static_cast<std::int32_t>(crc&1U)));}return ~crc;}
struct BitWriter{std::vector<std::uint8_t> bytes;std::size_t bits{};void put(std::uint32_t x,std::size_t width){for(std::size_t i=0U;i<width;++i){if((bits&7U)==0U)bytes.push_back(0U);if(((x>>(width-1U-i))&1U)!=0U)bytes.back()|=static_cast<std::uint8_t>(1U<<(7U-(bits&7U)));++bits;}}};
struct BitReader{const std::vector<std::uint8_t>& bytes;std::size_t bits{},position{};bool get(std::size_t width,std::uint32_t& x){if(width>32U||position>bits||bits-position<width)return false;x=0U;for(std::size_t i=0U;i<width;++i){x=(x<<1U)|((bytes[position/8U]>>(7U-(position&7U)))&1U);++position;}return true;}};
std::size_t key_bits(std::size_t count){if(count<=1U)return 0U;std::size_t bits=0U;std::size_t max=count-1U;while(max>0U){++bits;max>>=1U;}return bits;}
std::uint64_t value_for(std::size_t field,std::uint64_t r){switch(field){case 0U:return std::bit_cast<std::uint64_t>(static_cast<double>(r%100000U)/1000.0);case 1U:return std::bit_cast<std::uint64_t>(static_cast<double>(r%10000000U)/100.0);case 2U:return r%4096U;case 3U:return r&1U;case 4U:return r;case 5U:return r;case 6U:return std::rotl(r,17);default:return std::bit_cast<std::uint64_t>(static_cast<double>(r%1000000U)/10000.0);}}
std::vector<Record> fixture(const std::string& workload,std::uint64_t seed){
    std::mt19937_64 rng(seed);std::array<Record,kKeys> states{};std::vector<Record> out;out.reserve(kKeys*kSteps);
    for(std::size_t key=0U;key<kKeys;++key){auto& r=states[key];r.entity=static_cast<std::uint32_t>(key/16U);r.series=static_cast<std::uint32_t>(key%16U);r.present=0xffU;for(std::size_t f=0U;f<kFields;++f)r.values[f]=value_for(f,rng());}
    for(std::size_t step=0U;step<kSteps;++step)for(std::size_t key=0U;key<kKeys;++key){auto& r=states[key];r.timestamp=(static_cast<std::uint64_t>(step)*kKeys+key)*1000U;for(std::size_t f=0U;f<kFields;++f){const std::uint64_t random=rng();bool update=false;if(step>0U){if(workload=="sparse")update=random%1000U<15U;else if(workload=="idle")update=random%10000U<2U;else if(workload=="bursty")update=(step%256U<24U)?random%100U<65U:random%1000U<5U;else if(workload=="noisy")update=true;}if(update){const auto bit=static_cast<std::uint8_t>(1U<<f);if(std::rotl(random,11)%2048U==0U)r.present=static_cast<std::uint8_t>(r.present^bit);r.values[f]=(r.present&bit)==0U?0U:value_for(f,rng());}}out.push_back(r);}
    return out;
}
std::vector<std::uint8_t> raw_bytes(const std::vector<Record>& records){std::vector<std::uint8_t> o;o.reserve(8U+records.size()*81U);o.insert(o.end(),kSchema.begin(),kSchema.end());for(const auto& r:records){put_u64(r.timestamp,o);put_u32(r.entity,o);put_u32(r.series,o);o.push_back(r.present);for(auto v:r.values)put_u64(v,o);}return o;}
std::vector<std::uint8_t> global_state_delta(const std::vector<Record>& records){
    std::map<Key,Record> previous;std::vector<std::uint8_t> o;o.reserve(records.size()*40U);bool have_time=false;std::uint64_t last_time=0U;
    for(std::size_t i=0U;i<records.size();++i){const auto& r=records[i];if(!valid_record(r))throw std::runtime_error("invalid record");if(i%4096U==0U){previous.clear();have_time=false;}if(have_time&&r.timestamp>=last_time){o.push_back(0U);put_var(r.timestamp-last_time,o);}else{o.push_back(1U);put_var(r.timestamp,o);}last_time=r.timestamp;have_time=true;put_var(r.entity,o);put_var(r.series,o);const Key key=key_of(r);auto it=previous.find(key);const bool is_new=it==previous.end();const std::uint8_t mask=is_new?static_cast<std::uint8_t>(0xffU):changed_mask(it->second,r);o.push_back(is_new?1U:0U);o.push_back(mask);o.push_back(r.present);for(std::size_t f=0U;f<kFields;++f){const auto bit=static_cast<std::uint8_t>(1U<<f);if((mask&bit)!=0U&&(r.present&bit)!=0U)put_u64(r.values[f],o);}previous[key]=r;}
    return o;
}

std::vector<std::uint8_t> wrap_simple_frame(const char magic[4], std::size_t count,
                                            const std::vector<std::uint8_t>& payload) {
    std::vector<std::uint8_t> frame;
    frame.insert(frame.end(), magic, magic + 4);
    put_u64(static_cast<std::uint64_t>(count), frame);
    put_u64(static_cast<std::uint64_t>(payload.size()), frame);
    frame.insert(frame.end(), payload.begin(), payload.end());
    put_u32(crc32(frame.data(), frame.size()), frame);
    return frame;
}
std::vector<std::uint8_t> framed_raw(const std::vector<Record>& records) {
    const auto payload = raw_bytes(records);
    return wrap_simple_frame("RAW1", records.size(), payload);
}
std::vector<std::uint8_t> framed_global_state_delta(const std::vector<Record>& records) {
    auto payload = global_state_delta(records);
    payload.insert(payload.begin(), kSchema.begin(), kSchema.end());
    return wrap_simple_frame("GSD1", records.size(), payload);
}

std::vector<std::uint8_t> encode_one_series(const std::vector<Record>& rows){
    std::vector<std::uint8_t> o;o.reserve(rows.size()*12U);if(rows.empty())return o;
    for(std::size_t i=0U;i<rows.size();){const auto& r=rows[i];if(!valid_record(r))throw std::runtime_error("invalid series row");
        if(i==0U){o.push_back(1U);put_var(r.timestamp,o);o.push_back(r.present);for(std::size_t f=0U;f<kFields;++f)if((r.present&(1U<<f))!=0U)put_u64(r.values[f],o);++i;continue;}
        const auto& prev=rows[i-1U];const std::uint64_t delta=r.timestamp>=prev.timestamp?r.timestamp-prev.timestamp:0U;
        if(r.timestamp>=prev.timestamp&&same_state(prev,r)){
            std::size_t run=0U;std::uint64_t expected=prev.timestamp;for(std::size_t j=i;j<rows.size();++j){if(!same_state(rows[j-1U],rows[j])||rows[j].timestamp<expected||rows[j].timestamp-expected!=delta)break;expected=rows[j].timestamp;++run;}
            if(run>=2U){o.push_back(2U);put_var(delta,o);put_var(run,o);i+=run;continue;}
        }
        if(r.timestamp>=prev.timestamp){o.push_back(0U);put_var(delta,o);}else{o.push_back(1U);put_var(r.timestamp,o);}
        const auto mask=changed_mask(prev,r);o.push_back(mask);o.push_back(r.present);for(std::size_t f=0U;f<kFields;++f){const auto bit=static_cast<std::uint8_t>(1U<<f);if((mask&bit)!=0U&&(r.present&bit)!=0U)put_u64(r.values[f],o);}++i;
    }
    return o;
}
bool decode_one_series(const std::vector<std::uint8_t>& bytes, std::size_t count,
                       Key key, std::vector<Record>& out) {
    out.clear();
    out.reserve(count);
    std::size_t cursor = 0U;
    std::uint64_t last_time = 0U;

    while (out.size() < count) {
        if (cursor >= bytes.size()) return false;
        const auto mode = bytes[cursor++];

        if (mode == 2U) {
            if (out.empty()) return false;
            std::uint64_t delta = 0U;
            std::uint64_t run = 0U;
            if (!get_var(bytes, cursor, delta) || !get_var(bytes, cursor, run) ||
                run < 2U || run > count - out.size()) return false;

            std::uint64_t timestamp = out.back().timestamp;
            Record repeated = out.back();
            for (std::uint64_t index = 0U; index < run; ++index) {
                if (delta > UINT64_MAX - timestamp) return false;
                timestamp += delta;
                repeated.timestamp = timestamp;
                out.push_back(repeated);
            }
            last_time = timestamp;
            continue;
        }

        if (mode > 1U) return false;
        std::uint64_t time_value = 0U;
        if (!get_var(bytes, cursor, time_value)) return false;

        std::uint64_t timestamp = 0U;
        if (out.empty()) {
            if (mode != 1U) return false;
            timestamp = time_value;
            if (cursor >= bytes.size()) return false;

            const auto present = bytes[cursor++];
            Record record{};
            record.timestamp = timestamp;
            record.entity = key.first;
            record.series = key.second;
            record.present = present;
            for (std::size_t field = 0U; field < kFields; ++field) {
                if ((present & (1U << field)) != 0U) {
                    std::uint64_t value = 0U;
                    if (!get_u64(bytes, cursor, value)) return false;
                    record.values[field] = value;
                }
            }
            if (!valid_record(record)) return false;
            out.push_back(record);
            last_time = timestamp;
            continue;
        }

        if (mode == 0U) {
            if (time_value > UINT64_MAX - last_time) return false;
            timestamp = last_time + time_value;
        } else {
            if (time_value >= last_time) return false;
            timestamp = time_value;
        }
        last_time = timestamp;

        if (bytes.size() - cursor < 2U) return false;
        const auto mask = bytes[cursor++];
        const auto present = bytes[cursor++];
        Record record = out.back();
        record.timestamp = timestamp;
        for (std::size_t field = 0U; field < kFields; ++field) {
            const auto bit = static_cast<std::uint8_t>(1U << field);
            if ((mask & bit) != 0U) {
                if ((present & bit) != 0U) {
                    std::uint64_t value = 0U;
                    if (!get_u64(bytes, cursor, value)) return false;
                    record.values[field] = value;
                } else {
                    record.values[field] = 0U;
                }
            }
        }
        record.present = present;
        if (!valid_record(record) || mask != changed_mask(out.back(), record)) return false;
        out.push_back(record);
    }
    return cursor == bytes.size();
}
std::vector<std::uint8_t> encode_partitioned(const std::vector<Record>& records) {
    std::map<Key, std::size_t> key_to_id;
    std::vector<Key> keys;
    std::vector<std::vector<Record>> per_key;
    bool strictly_increasing = true;
    bool nondecreasing = true;
    for (std::size_t i = 1U; i < records.size(); ++i) {
        if (records[i].timestamp <= records[i - 1U].timestamp) strictly_increasing = false;
        if (records[i].timestamp < records[i - 1U].timestamp) nondecreasing = false;
    }

    for (const auto& record : records) {
        const Key key = key_of(record);
        auto [it, inserted] = key_to_id.emplace(key, keys.size());
        if (inserted) {
            keys.push_back(key);
            per_key.emplace_back();
        }
        per_key[it->second].push_back(record);
    }

    std::vector<std::vector<std::uint8_t>> streams;
    streams.reserve(keys.size());
    for (const auto& rows : per_key) streams.push_back(encode_one_series(rows));

    // 0 = timestamp merge for unique monotonic timestamps.
    // 1 = nondecreasing timestamps with tie-group order templates.
    // 2 = explicit full key-ID schedule for timestamp regressions.
    const std::uint8_t order_mode = strictly_increasing ? 0U : (nondecreasing ? 1U : 2U);
    const std::size_t bits_per_id = order_mode == 0U ? 0U : key_bits(keys.size());
    std::vector<std::uint8_t> order_bytes;
    std::uint64_t order_bits = 0U;

    if (order_mode == 1U) {
        std::vector<std::uint32_t> previous_tie_template;
        for (std::size_t start = 0U; start < records.size();) {
            std::size_t end = start + 1U;
            while (end < records.size() &&
                   records[end].timestamp == records[start].timestamp) ++end;
            if (end - start > 1U) {
                std::vector<std::uint32_t> current_template;
                current_template.reserve(end - start);
                for (std::size_t i = start; i < end; ++i) {
                    const auto found = key_to_id.find(key_of(records[i]));
                    if (found == key_to_id.end()) throw std::runtime_error("missing key dictionary entry");
                    current_template.push_back(static_cast<std::uint32_t>(found->second));
                }
                if (!previous_tie_template.empty() && current_template == previous_tie_template) {
                    order_bytes.push_back(0U); // reuse previous tie-group ID order
                } else {
                    order_bytes.push_back(1U); // literal tie-group ID order follows
                    BitWriter literal;
                    for (const auto id : current_template) literal.put(id, bits_per_id);
                    order_bytes.insert(order_bytes.end(), literal.bytes.begin(), literal.bytes.end());
                    previous_tie_template = std::move(current_template);
                }
            }
            start = end;
        }
        order_bits = static_cast<std::uint64_t>(order_bytes.size()) * 8U;
    } else if (order_mode == 2U) {
        BitWriter full_schedule;
        for (const auto& record : records) {
            const auto found = key_to_id.find(key_of(record));
            if (found == key_to_id.end()) throw std::runtime_error("missing key dictionary entry");
            full_schedule.put(static_cast<std::uint32_t>(found->second), bits_per_id);
        }
        order_bytes = std::move(full_schedule.bytes);
        order_bits = full_schedule.bits;
    }

    std::vector<std::uint8_t> out;
    out.insert(out.end(), {'P', 'R', 'N', '1'});
    put_u64(records.size(), out);
    put_u32(static_cast<std::uint32_t>(keys.size()), out);
    out.push_back(static_cast<std::uint8_t>(kFields));
    out.insert(out.end(), kSchema.begin(), kSchema.end());
    out.push_back(order_mode);
    put_u32(static_cast<std::uint32_t>(bits_per_id), out);
    put_u64(order_bits, out);

    for (std::size_t i = 0U; i < keys.size(); ++i) {
        put_u32(keys[i].first, out);
        put_u32(keys[i].second, out);
        put_u64(per_key[i].size(), out);
        put_u64(streams[i].size(), out);
        out.insert(out.end(), streams[i].begin(), streams[i].end());
    }
    put_u64(order_bytes.size(), out);
    out.insert(out.end(), order_bytes.begin(), order_bytes.end());
    put_u32(crc32(out.data(), out.size()), out);
    return out;
}

bool decode_partitioned(const std::vector<std::uint8_t>& bytes, std::size_t max_records,
                        std::vector<Record>& output) {
    output.clear();
    if (bytes.size() < 4U + 8U + 4U + 1U + 8U + 1U + 4U + 8U + 8U + 4U) return false;
    const std::size_t crc_pos = bytes.size() - 4U;
    std::size_t crc_cursor = crc_pos;
    std::uint32_t stored_crc = 0U;
    if (!get_u32(bytes, crc_cursor, stored_crc) || crc32(bytes.data(), crc_pos) != stored_crc) return false;

    std::size_t cursor = 0U;
    if (bytes[cursor++] != 'P' || bytes[cursor++] != 'R' ||
        bytes[cursor++] != 'N' || bytes[cursor++] != '1') return false;
    std::uint64_t count64 = 0U;
    std::uint32_t key_count = 0U;
    if (!get_u64(bytes, cursor, count64) || count64 > max_records ||
        !get_u32(bytes, cursor, key_count) || key_count > count64 ||
        key_count > 1'000'000U) return false;
    if ((count64 == 0U) != (key_count == 0U)) return false;
    if (cursor >= crc_pos || bytes[cursor++] != kFields) return false;
    if (bytes.size() - cursor < kFields) return false;
    for (const auto kind : kSchema) if (bytes[cursor++] != kind) return false;
    if (cursor >= crc_pos) return false;
    const auto order_mode = bytes[cursor++];
    if (order_mode > 2U) return false;

    std::uint32_t bits = 0U;
    std::uint64_t order_bits = 0U;
    if (!get_u32(bytes, cursor, bits) || !get_u64(bytes, cursor, order_bits)) return false;
    if (order_mode == 0U) {
        if (bits != 0U || order_bits != 0U) return false;
    } else if (order_mode == 1U) {
        if (bits != key_bits(key_count) || (order_bits & 7U) != 0U) return false;
    } else {
        if (bits != key_bits(key_count) ||
            (bits != 0U && count64 > UINT64_MAX / bits) ||
            order_bits != count64 * bits || order_bits > UINT64_MAX - 7U) return false;
    }

    std::vector<Key> keys;
    std::vector<std::vector<Record>> rows;
    keys.reserve(key_count);
    rows.resize(key_count);
    std::uint64_t total_rows = 0U;
    for (std::uint32_t i = 0U; i < key_count; ++i) {
        std::uint32_t entity = 0U, series = 0U;
        std::uint64_t row_count = 0U, child_len = 0U;
        if (!get_u32(bytes, cursor, entity) || !get_u32(bytes, cursor, series) ||
            !get_u64(bytes, cursor, row_count) || !get_u64(bytes, cursor, child_len) ||
            row_count == 0U || row_count > count64 || cursor > crc_pos ||
            child_len > crc_pos - cursor) return false;
        keys.emplace_back(entity, series);
        for (std::uint32_t j = 0U; j < i; ++j) if (keys[j] == keys.back()) return false;
        std::vector<std::uint8_t> child(
            bytes.begin() + static_cast<std::ptrdiff_t>(cursor),
            bytes.begin() + static_cast<std::ptrdiff_t>(cursor + static_cast<std::size_t>(child_len)));
        cursor += static_cast<std::size_t>(child_len);
        if (!decode_one_series(child, static_cast<std::size_t>(row_count), keys.back(), rows[i])) return false;
        if (order_mode != 2U) {
            for (std::size_t j = 1U; j < rows[i].size(); ++j) {
                if (rows[i][j].timestamp < rows[i][j - 1U].timestamp) return false;
                if (order_mode == 0U && rows[i][j].timestamp == rows[i][j - 1U].timestamp) return false;
            }
        }
        if (row_count > count64 - total_rows) return false;
        total_rows += row_count;
    }
    if (total_rows != count64) return false;

    std::uint64_t schedule_len = 0U;
    if (!get_u64(bytes, cursor, schedule_len) || cursor > crc_pos ||
        schedule_len > crc_pos - cursor || cursor + schedule_len != crc_pos) return false;
    std::vector<std::uint8_t> schedule(
        bytes.begin() + static_cast<std::ptrdiff_t>(cursor),
        bytes.begin() + static_cast<std::ptrdiff_t>(cursor + static_cast<std::size_t>(schedule_len)));

    output.reserve(static_cast<std::size_t>(count64));
    if (order_mode == 0U) {
        if (schedule_len != 0U) return false;
        using HeapEntry = std::pair<std::uint64_t, std::size_t>;
        std::priority_queue<HeapEntry, std::vector<HeapEntry>, std::greater<HeapEntry>> heap;
        std::vector<std::size_t> cursors(key_count, 0U);
        for (std::size_t key = 0U; key < rows.size(); ++key) {
            if (!rows[key].empty()) heap.emplace(rows[key][0].timestamp, key);
        }
        bool have_previous = false;
        std::uint64_t previous_timestamp = 0U;
        while (!heap.empty()) {
            const auto [timestamp, key] = heap.top();
            heap.pop();
            if (have_previous && timestamp <= previous_timestamp) return false;
            const auto row_index = cursors[key]++;
            if (row_index >= rows[key].size() || rows[key][row_index].timestamp != timestamp) return false;
            output.push_back(rows[key][row_index]);
            previous_timestamp = timestamp;
            have_previous = true;
            if (cursors[key] < rows[key].size()) heap.emplace(rows[key][cursors[key]].timestamp, key);
        }
        for (std::size_t key = 0U; key < rows.size(); ++key) if (cursors[key] != rows[key].size()) return false;
        return output.size() == count64;
    }

    if (order_mode == 1U) {
        if (schedule_len != order_bits / 8U) return false;
        std::vector<std::size_t> cursors(key_count, 0U);
        std::vector<std::uint32_t> previous_tie_template;
        std::size_t schedule_cursor = 0U;
        while (output.size() < count64) {
            std::uint64_t min_timestamp = 0U;
            bool have_min_timestamp = false;
            for (std::size_t key = 0U; key < rows.size(); ++key) {
                if (cursors[key] < rows[key].size()) {
                    const auto candidate = rows[key][cursors[key]].timestamp;
                    if (!have_min_timestamp || candidate < min_timestamp) {
                        min_timestamp = candidate;
                        have_min_timestamp = true;
                    }
                }
            }
            if (!have_min_timestamp) return false;
            std::size_t tie_count = 0U;
            std::vector<std::size_t> next_count_by_key(key_count, 0U);
            for (std::size_t key = 0U; key < rows.size(); ++key) {
                std::size_t scan = cursors[key];
                while (scan < rows[key].size() && rows[key][scan].timestamp == min_timestamp) {
                    ++next_count_by_key[key];
                    ++tie_count;
                    ++scan;
                }
            }

            std::vector<std::uint32_t> tie_order;
            if (tie_count == 1U) {
                for (std::size_t key = 0U; key < key_count; ++key) {
                    if (next_count_by_key[key] == 1U) {
                        tie_order.push_back(static_cast<std::uint32_t>(key));
                        break;
                    }
                }
            } else {
                if (schedule_cursor >= schedule.size()) return false;
                const auto control = schedule[schedule_cursor++];
                if (control == 0U) {
                    if (previous_tie_template.size() != tie_count) return false;
                    tie_order = previous_tie_template;
                } else if (control == 1U) {
                    if (bits != 0U && tie_count > (SIZE_MAX / bits)) return false;
                    const std::size_t literal_bits = tie_count * bits;
                    if (literal_bits > SIZE_MAX - 7U) return false;
                    const std::size_t literal_bytes = (literal_bits + 7U) / 8U;
                    if (schedule_cursor > schedule.size() || literal_bytes > schedule.size() - schedule_cursor) return false;
                    std::vector<std::uint8_t> literal(
                        schedule.begin() + static_cast<std::ptrdiff_t>(schedule_cursor),
                        schedule.begin() + static_cast<std::ptrdiff_t>(schedule_cursor + literal_bytes));
                    if ((literal_bits & 7U) != 0U && !literal.empty()) {
                        const auto mask = static_cast<std::uint8_t>((1U << (8U - (literal_bits & 7U))) - 1U);
                        if ((literal.back() & mask) != 0U) return false;
                    }
                    BitReader reader{literal, literal_bits, 0U};
                    tie_order.reserve(tie_count);
                    for (std::size_t i = 0U; i < tie_count; ++i) {
                        std::uint32_t id = 0U;
                        if (!reader.get(bits, id) || id >= key_count) return false;
                        tie_order.push_back(id);
                    }
                    if (reader.position != literal_bits) return false;
                    schedule_cursor += literal_bytes;
                    previous_tie_template = tie_order;
                } else {
                    return false;
                }
            }

            if (tie_order.size() != tie_count) return false;
            std::vector<std::size_t> consumed_by_key(key_count, 0U);
            for (const auto id : tie_order) {
                if (id >= key_count || cursors[id] >= rows[id].size() ||
                    rows[id][cursors[id]].timestamp != min_timestamp) return false;
                ++consumed_by_key[id];
                output.push_back(rows[id][cursors[id]++]);
            }
            if (consumed_by_key != next_count_by_key) return false;
        }
        return schedule_cursor == schedule.size();
    }

    // Mode 2: arbitrary arrival order; replay the explicit packed key-ID schedule.
    if (schedule.size() != (order_bits + 7U) / 8U) return false;
    if ((order_bits & 7U) != 0U && !schedule.empty()) {
        const auto mask = static_cast<std::uint8_t>((1U << (8U - (order_bits & 7U))) - 1U);
        if ((schedule.back() & mask) != 0U) return false;
    }
    BitReader reader{schedule, static_cast<std::size_t>(order_bits), 0U};
    std::vector<std::size_t> cursors(key_count, 0U);
    for (std::uint64_t i = 0U; i < count64; ++i) {
        std::uint32_t id = 0U;
        if (!reader.get(bits, id) || id >= key_count || cursors[id] >= rows[id].size()) return false;
        output.push_back(rows[id][cursors[id]++]);
    }
    if (reader.position != reader.bits) return false;
    for (std::size_t i = 0U; i < rows.size(); ++i) if (cursors[i] != rows[i].size()) return false;
    return true;
}
void write_file(const std::filesystem::path& path,const std::vector<std::uint8_t>& bytes){std::ofstream f(path,std::ios::binary);if(!f)throw std::runtime_error("open failed");f.write(reinterpret_cast<const char*>(bytes.data()),static_cast<std::streamsize>(bytes.size()));if(!f)throw std::runtime_error("write failed");}
void require(bool v,const std::string& m){if(!v)throw std::runtime_error(m);}
void test_empty_stream(){const std::vector<Record> records;const auto frame=encode_partitioned(records);std::vector<Record> decoded;require(decode_partitioned(frame,0U,decoded),"empty partitioned stream rejected");require(decoded.empty(),"empty partitioned stream produced records");}
void test_roundtrips(){for(const std::string name:{"sparse","bursty","idle","noisy"}){const auto records=fixture(name,0x7878U);const auto frame=encode_partitioned(records);require(frame.size()>26U&&frame[25]==0U,name+" should use timestamp-merge mode");std::vector<Record> decoded;require(decode_partitioned(frame,records.size(),decoded),name+" partition decoder rejected frame");require(decoded==records,name+" partition order/data mismatch");}}
void test_schedule_fallback(){auto records=fixture("sparse",0x887766U);for(std::size_t i=0U;i<records.size();++i)records[i].timestamp=static_cast<std::uint64_t>(i/kKeys)*1000U;const auto duplicate_frame=encode_partitioned(records);require(duplicate_frame.size()>26U&&duplicate_frame[25]==1U,"duplicate timestamps should use tie-template mode");std::vector<Record> decoded;require(decode_partitioned(duplicate_frame,records.size(),decoded)&&decoded==records,"duplicate timestamp fallback order mismatch");std::uint64_t tie_order_bits=0U;std::size_t offset=30U;require(get_u64(duplicate_frame,offset,tie_order_bits),"tie-template order bit count missing");require(tie_order_bits<records.size()*key_bits(kKeys),"repeated tie templates should compress below full schedule");require(tie_order_bits<5000U,"repeated tie templates should stay compact on identical groups");auto changing=records;for(std::size_t start=0U;start<changing.size();start+=kKeys)if((start/kKeys)%2U==1U)std::reverse(changing.begin()+static_cast<std::ptrdiff_t>(start),changing.begin()+static_cast<std::ptrdiff_t>(start+kKeys));const auto changing_frame=encode_partitioned(changing);require(changing_frame[25]==1U,"nondecreasing changing ties should use template mode");require(decode_partitioned(changing_frame,changing.size(),decoded)&&decoded==changing,"changing tie template order mismatch");auto out_of_order=fixture("bursty",0x778899U);std::swap(out_of_order[1],out_of_order[2]);const auto disorder_frame=encode_partitioned(out_of_order);require(disorder_frame.size()>26U&&disorder_frame[25]==2U,"out-of-order timestamps should use full schedule");require(decode_partitioned(disorder_frame,out_of_order.size(),decoded)&&decoded==out_of_order,"out-of-order fallback mismatch");}
void test_randomized(){std::mt19937_64 rng(0x991122U);for(std::size_t trial=0U;trial<100U;++trial){std::vector<Record> records;std::array<Record,8U> states{};std::array<bool,8U> init{};const std::size_t n=static_cast<std::size_t>(rng()%300U);std::uint64_t time=0U;for(std::size_t i=0U;i<n;++i){const auto k=static_cast<std::size_t>(rng()%8U);time+=rng()%7U;Record r=init[k]?states[k]:Record{};r.timestamp=time;r.entity=static_cast<std::uint32_t>(k/4U);r.series=static_cast<std::uint32_t>(k%4U);if(!init[k])r.present=0xffU;for(std::size_t f=0U;f<kFields;++f){const auto bit=static_cast<std::uint8_t>(1U<<f);if(!init[k]||rng()%20U==0U){if(init[k]&&rng()%8U==0U)r.present=static_cast<std::uint8_t>(r.present^bit);r.values[f]=(r.present&bit)==0U?0U:value_for(f,rng());}}records.push_back(r);states[k]=r;init[k]=true;}const auto frame=encode_partitioned(records);std::vector<Record> decoded;require(decode_partitioned(frame,records.size(),decoded),"random partition decode rejected");require(decoded==records,"random partition mismatch");}}
void test_corruption(){const auto rows=fixture("sparse",0x8877U);auto frame=encode_partitioned(rows);std::vector<Record> out;auto bad=frame;bad[0]^=1U;require(!decode_partitioned(bad,rows.size(),out),"bad magic/CRC accepted");bad=frame;bad.pop_back();require(!decode_partitioned(bad,rows.size(),out),"truncated partition frame accepted");bad=frame;bad.push_back(0U);require(!decode_partitioned(bad,rows.size(),out),"trailing partition bytes accepted");}
void emit(const std::filesystem::path& dir) {
    std::filesystem::create_directories(dir);
    for (const std::string name : {"sparse", "bursty", "idle", "noisy"}) {
        const auto rows = fixture(name, 0x7e1e0001U);
        const auto raw = framed_raw(rows);
        const auto delta = framed_global_state_delta(rows);
        const auto partitioned = encode_partitioned(rows);
        std::vector<Record> decoded;
        require(decode_partitioned(partitioned, rows.size(), decoded) && decoded == rows,
                name + " emission round-trip failed");
        write_file(dir / (name + "_raw.bin"), raw);
        write_file(dir / (name + "_global_delta.bin"), delta);
        write_file(dir / (name + "_partitioned.bin"), partitioned);
        std::cout << name << ",records=" << rows.size()
                  << ",raw_frame=" << raw.size()
                  << ",global_delta_frame=" << delta.size()
                  << ",partitioned_frame=" << partitioned.size() << "\n";
    }
}

int main(int argc,char** argv){try{test_empty_stream();test_roundtrips();test_schedule_fallback();test_randomized();test_corruption();std::cout<<"empty_stream=PASS timestamp_merge=PASS duplicate_tie_template=PASS disorder_full_schedule=PASS fixture_roundtrips=4 randomized_roundtrips=100 corruption_rejections=3 status=PASS\\n";if(argc==3&&std::string(argv[1])=="--emit-dir")emit(argv[2]);else if(argc!=1){std::cerr<<"usage: "<<argv[0]<<" [--emit-dir DIR]\\n";return 2;}return 0;}catch(const std::exception& e){std::cerr<<"partitioned reference failed: "<<e.what()<<"\\n";return 1;}}
