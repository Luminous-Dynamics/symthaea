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
#include <zstd.h>

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
        constexpr std::size_t max_tie_templates = 16U;
        std::vector<std::vector<std::uint32_t>> tie_templates;
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
                const auto found = std::find(tie_templates.begin(), tie_templates.end(), current_template);
                if (found != tie_templates.end()) {
                    const auto template_index = static_cast<std::size_t>(found - tie_templates.begin());
                    order_bytes.push_back(static_cast<std::uint8_t>(template_index + 1U));
                    auto reused = *found;
                    tie_templates.erase(found);
                    tie_templates.insert(tie_templates.begin(), std::move(reused));
                } else {
                    order_bytes.push_back(0U); // literal tie-group ID order follows
                    BitWriter literal;
                    for (const auto id : current_template) literal.put(id, bits_per_id);
                    order_bytes.insert(order_bytes.end(), literal.bytes.begin(), literal.bytes.end());
                    tie_templates.insert(tie_templates.begin(), std::move(current_template));
                    if (tie_templates.size() > max_tie_templates) tie_templates.pop_back();
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
        using HeapEntry = std::pair<std::uint64_t, std::size_t>;
        std::priority_queue<HeapEntry, std::vector<HeapEntry>, std::greater<HeapEntry>> heap;
        std::vector<std::size_t> cursors(key_count, 0U);
        for (std::size_t key = 0U; key < rows.size(); ++key) {
            if (!rows[key].empty()) heap.emplace(rows[key][0].timestamp, key);
        }

        constexpr std::size_t max_tie_templates = 16U;
        std::vector<std::vector<std::uint32_t>> tie_templates;
        std::size_t schedule_cursor = 0U;
        while (output.size() < count64) {
            if (heap.empty()) return false;
            const std::uint64_t min_timestamp = heap.top().first;

            // Consume all rows at this timestamp from the merge frontier. Rows from
            // the same series may repeat at this timestamp; push each successor so
            // the group includes every such row before the next timestamp is chosen.
            std::map<std::size_t, std::size_t> next_count_by_key;
            std::size_t tie_count = 0U;
            while (!heap.empty() && heap.top().first == min_timestamp) {
                const auto [timestamp, key] = heap.top();
                heap.pop();
                if (timestamp != min_timestamp || key >= rows.size()) return false;
                ++next_count_by_key[key];
                ++tie_count;
                const std::size_t next_index = cursors[key] + next_count_by_key[key];
                if (next_index < rows[key].size()) {
                    heap.emplace(rows[key][next_index].timestamp, key);
                }
            }

            std::vector<std::uint32_t> tie_order;
            if (tie_count == 1U) {
                if (next_count_by_key.size() != 1U || next_count_by_key.begin()->second != 1U) return false;
                tie_order.push_back(static_cast<std::uint32_t>(next_count_by_key.begin()->first));
            } else {
                if (schedule_cursor >= schedule.size()) return false;
                const auto control = schedule[schedule_cursor++];
                if (control == 0U) {
                    if (bits != 0U && tie_count > (SIZE_MAX / bits)) return false;
                    const std::size_t literal_bits = tie_count * bits;
                    if (literal_bits > SIZE_MAX - 7U) return false;
                    const std::size_t literal_bytes = (literal_bits + 7U) / 8U;
                    if (schedule_cursor > schedule.size() ||
                        literal_bytes > schedule.size() - schedule_cursor) return false;
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
                    tie_templates.insert(tie_templates.begin(), tie_order);
                    if (tie_templates.size() > max_tie_templates) tie_templates.pop_back();
                } else {
                    const std::size_t template_index = static_cast<std::size_t>(control - 1U);
                    if (template_index >= tie_templates.size() ||
                        tie_templates[template_index].size() != tie_count) return false;
                    tie_order = tie_templates[template_index];
                    auto reused = tie_templates[template_index];
                    tie_templates.erase(tie_templates.begin() + static_cast<std::ptrdiff_t>(template_index));
                    tie_templates.insert(tie_templates.begin(), std::move(reused));
                }
            }

            if (tie_order.size() != tie_count) return false;
            std::map<std::size_t, std::size_t> consumed_by_key;
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

std::vector<std::uint8_t> encode_one_series_columnar(const std::vector<Record>& rows) {
    std::vector<std::uint8_t> out;
    if (rows.empty()) return out;
    for (std::size_t i = 0U; i < rows.size(); ++i) {
        if (!valid_record(rows[i])) throw std::runtime_error("invalid columnar record");
        if (i > 0U && rows[i].timestamp <= rows[i - 1U].timestamp)
            throw std::runtime_error("columnar child timestamps must be strictly increasing");
    }

    put_var(static_cast<std::uint64_t>(rows.size()), out);
    put_var(rows.front().timestamp, out);
    for (std::size_t i = 1U; i < rows.size(); ++i)
        put_var(rows[i].timestamp - rows[i - 1U].timestamp, out);

    const std::size_t bitmap_bytes = (rows.size() + 7U) / 8U;
    for (std::size_t field = 0U; field < kFields; ++field) {
        std::vector<std::uint8_t> present(bitmap_bytes, 0U);
        std::vector<std::uint8_t> changed(bitmap_bytes, 0U);
        std::vector<std::uint8_t> values;
        bool previous_present = false;
        std::uint64_t previous_value = 0U;

        for (std::size_t row = 0U; row < rows.size(); ++row) {
            const auto bit = static_cast<std::uint8_t>(1U << field);
            const bool is_present = (rows[row].present & bit) != 0U;
            const std::uint64_t value = is_present ? rows[row].values[field] : 0U;
            const bool is_changed = row == 0U || is_present != previous_present ||
                                    (is_present && value != previous_value);
            if (is_present)
                present[row / 8U] |= static_cast<std::uint8_t>(1U << (row & 7U));
            if (is_changed)
                changed[row / 8U] |= static_cast<std::uint8_t>(1U << (row & 7U));
            if (is_changed && is_present) put_u64(value, values);
            previous_present = is_present;
            previous_value = value;
        }
        out.insert(out.end(), present.begin(), present.end());
        out.insert(out.end(), changed.begin(), changed.end());
        out.insert(out.end(), values.begin(), values.end());
    }
    return out;
}

bool decode_one_series_columnar(const std::vector<std::uint8_t>& bytes, std::size_t expected_count,
                                Key key, std::vector<Record>& out) {
    out.clear();
    std::size_t cursor = 0U;
    std::uint64_t count64 = 0U;
    if (!get_var(bytes, cursor, count64) || count64 != expected_count ||
        count64 > SIZE_MAX || count64 == 0U) return false;
    const std::size_t count = static_cast<std::size_t>(count64);
    out.resize(count);

    std::uint64_t timestamp = 0U;
    if (!get_var(bytes, cursor, timestamp)) return false;
    out[0].timestamp = timestamp;
    out[0].entity = key.first;
    out[0].series = key.second;
    for (std::size_t row = 1U; row < count; ++row) {
        std::uint64_t delta = 0U;
        if (!get_var(bytes, cursor, delta) || delta == 0U ||
            delta > UINT64_MAX - timestamp) return false;
        timestamp += delta;
        out[row].timestamp = timestamp;
        out[row].entity = key.first;
        out[row].series = key.second;
    }

    const std::size_t bitmap_bytes = (count + 7U) / 8U;
    if (bitmap_bytes > (bytes.size() - cursor) / 2U / kFields) return false;
    for (std::size_t field = 0U; field < kFields; ++field) {
        if (cursor > bytes.size() || bitmap_bytes > bytes.size() - cursor) return false;
        const auto present_begin = bytes.begin() + static_cast<std::ptrdiff_t>(cursor);
        std::vector<std::uint8_t> present(present_begin,
                                          present_begin + static_cast<std::ptrdiff_t>(bitmap_bytes));
        cursor += bitmap_bytes;
        if (cursor > bytes.size() || bitmap_bytes > bytes.size() - cursor) return false;
        const auto changed_begin = bytes.begin() + static_cast<std::ptrdiff_t>(cursor);
        std::vector<std::uint8_t> changed(changed_begin,
                                          changed_begin + static_cast<std::ptrdiff_t>(bitmap_bytes));
        cursor += bitmap_bytes;

        if ((count & 7U) != 0U) {
            const auto unused_mask = static_cast<std::uint8_t>(0xffU << (count & 7U));
            if ((present.back() & unused_mask) != 0U ||
                (changed.back() & unused_mask) != 0U) return false;
        }

        bool previous_present = false;
        std::uint64_t previous_value = 0U;
        for (std::size_t row = 0U; row < count; ++row) {
            const auto mask = static_cast<std::uint8_t>(1U << (row & 7U));
            const bool is_present = (present[row / 8U] & mask) != 0U;
            const bool is_changed = (changed[row / 8U] & mask) != 0U;
            if (!is_changed && (row == 0U || is_present != previous_present)) return false;

            std::uint64_t value = previous_value;
            if (is_changed) {
                value = 0U;
                if (is_present && !get_u64(bytes, cursor, value)) return false;
            }
            const bool expected_changed = row == 0U || is_present != previous_present ||
                                          (is_present && value != previous_value);
            if (is_changed != expected_changed) return false;

            if (is_present) {
                out[row].present = static_cast<std::uint8_t>(out[row].present | (1U << field));
                out[row].values[field] = value;
            } else {
                out[row].values[field] = 0U;
            }
            previous_present = is_present;
            previous_value = value;
        }
    }

    if (cursor != bytes.size()) return false;
    for (const auto& record : out) if (!valid_record(record)) return false;
    return true;
}

std::vector<std::uint8_t> encode_partitioned_columnar(const std::vector<Record>& records) {
    for (std::size_t i = 1U; i < records.size(); ++i) {
        if (records[i].timestamp <= records[i - 1U].timestamp)
            throw std::runtime_error("columnar partition prototype requires strictly increasing unique timestamps");
    }

    std::map<Key, std::size_t> key_to_id;
    std::vector<Key> keys;
    std::vector<std::vector<Record>> per_key;
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
    for (const auto& rows : per_key) streams.push_back(encode_one_series_columnar(rows));

    std::vector<std::uint8_t> out;
    out.insert(out.end(), {'P', 'C', 'L', '1'});
    put_u64(records.size(), out);
    put_u32(static_cast<std::uint32_t>(keys.size()), out);
    out.push_back(static_cast<std::uint8_t>(kFields));
    out.insert(out.end(), kSchema.begin(), kSchema.end());
    for (std::size_t i = 0U; i < keys.size(); ++i) {
        put_u32(keys[i].first, out);
        put_u32(keys[i].second, out);
        put_u64(per_key[i].size(), out);
        put_u64(streams[i].size(), out);
        out.insert(out.end(), streams[i].begin(), streams[i].end());
    }
    put_u32(crc32(out.data(), out.size()), out);
    return out;
}

bool decode_partitioned_columnar(const std::vector<std::uint8_t>& bytes, std::size_t max_records,
                                 std::vector<Record>& output) {
    output.clear();
    if (bytes.size() < 4U + 8U + 4U + 1U + kFields + 4U) return false;
    const std::size_t crc_pos = bytes.size() - 4U;
    std::size_t crc_cursor = crc_pos;
    std::uint32_t stored_crc = 0U;
    if (!get_u32(bytes, crc_cursor, stored_crc) || crc32(bytes.data(), crc_pos) != stored_crc) return false;

    std::size_t cursor = 0U;
    if (bytes[cursor++] != 'P' || bytes[cursor++] != 'C' ||
        bytes[cursor++] != 'L' || bytes[cursor++] != '1') return false;
    std::uint64_t count64 = 0U;
    std::uint32_t key_count = 0U;
    if (!get_u64(bytes, cursor, count64) || count64 > max_records || count64 > SIZE_MAX ||
        !get_u32(bytes, cursor, key_count) || key_count > count64 ||
        key_count > 1'000'000U) return false;
    if ((count64 == 0U) != (key_count == 0U)) return false;
    if (cursor >= crc_pos || bytes[cursor++] != kFields ||
        bytes.size() - cursor < kFields) return false;
    for (const auto kind : kSchema) if (bytes[cursor++] != kind) return false;

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
            row_count == 0U || row_count > count64 || row_count > SIZE_MAX ||
            cursor > crc_pos || child_len > crc_pos - cursor) return false;
        keys.emplace_back(entity, series);
        for (std::uint32_t j = 0U; j < i; ++j) if (keys[j] == keys.back()) return false;
        std::vector<std::uint8_t> child(
            bytes.begin() + static_cast<std::ptrdiff_t>(cursor),
            bytes.begin() + static_cast<std::ptrdiff_t>(cursor + static_cast<std::size_t>(child_len)));
        cursor += static_cast<std::size_t>(child_len);
        if (!decode_one_series_columnar(child, static_cast<std::size_t>(row_count), keys.back(), rows[i])) return false;
        for (std::size_t j = 1U; j < rows[i].size(); ++j) {
            if (rows[i][j].timestamp <= rows[i][j - 1U].timestamp) return false;
        }
        if (row_count > count64 - total_rows) return false;
        total_rows += row_count;
    }
    if (total_rows != count64 || cursor != crc_pos) return false;

    using HeapEntry = std::pair<std::uint64_t, std::size_t>;
    std::priority_queue<HeapEntry, std::vector<HeapEntry>, std::greater<HeapEntry>> heap;
    std::vector<std::size_t> cursors(key_count, 0U);
    for (std::size_t key = 0U; key < rows.size(); ++key) {
        if (!rows[key].empty()) heap.emplace(rows[key][0].timestamp, key);
    }
    output.reserve(static_cast<std::size_t>(count64));
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


std::vector<std::uint8_t> encode_partitioned_adaptive(const std::vector<Record>& records,
                                                       std::size_t* row_children = nullptr,
                                                       std::size_t* columnar_children = nullptr) {
    for (std::size_t i = 1U; i < records.size(); ++i) {
        if (records[i].timestamp <= records[i - 1U].timestamp)
            throw std::runtime_error("adaptive partition prototype requires strictly increasing unique timestamps");
    }
    std::map<Key, std::size_t> key_to_id;
    std::vector<Key> keys;
    std::vector<std::vector<Record>> per_key;
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
    std::vector<std::uint8_t> layouts;
    streams.reserve(keys.size());
    layouts.reserve(keys.size());
    std::size_t row_count = 0U;
    std::size_t columnar_count = 0U;
    for (const auto& rows : per_key) {
        const auto row_stream = encode_one_series(rows);
        const auto columnar_stream = encode_one_series_columnar(rows);
        if (row_stream.size() <= columnar_stream.size()) {
            streams.push_back(row_stream);
            layouts.push_back(0U); // row-state child
            ++row_count;
        } else {
            streams.push_back(columnar_stream);
            layouts.push_back(1U); // field-columnar child
            ++columnar_count;
        }
    }

    if (row_children != nullptr) *row_children = row_count;
    if (columnar_children != nullptr) *columnar_children = columnar_count;

    std::vector<std::uint8_t> out;
    out.insert(out.end(), {'P', 'A', 'X', '1'});
    put_u64(records.size(), out);
    put_u32(static_cast<std::uint32_t>(keys.size()), out);
    out.push_back(static_cast<std::uint8_t>(kFields));
    out.insert(out.end(), kSchema.begin(), kSchema.end());
    for (std::size_t i = 0U; i < keys.size(); ++i) {
        put_u32(keys[i].first, out);
        put_u32(keys[i].second, out);
        put_u64(per_key[i].size(), out);
        put_u64(streams[i].size(), out);
        out.push_back(layouts[i]);
        out.insert(out.end(), streams[i].begin(), streams[i].end());
    }
    put_u32(crc32(out.data(), out.size()), out);
    return out;
}

bool decode_partitioned_adaptive(const std::vector<std::uint8_t>& bytes, std::size_t max_records,
                                 std::vector<Record>& output) {
    output.clear();
    if (bytes.size() < 4U + 8U + 4U + 1U + kFields + 4U) return false;
    const std::size_t crc_pos = bytes.size() - 4U;
    std::size_t crc_cursor = crc_pos;
    std::uint32_t stored_crc = 0U;
    if (!get_u32(bytes, crc_cursor, stored_crc) || crc32(bytes.data(), crc_pos) != stored_crc) return false;

    std::size_t cursor = 0U;
    if (bytes[cursor++] != 'P' || bytes[cursor++] != 'A' ||
        bytes[cursor++] != 'X' || bytes[cursor++] != '1') return false;
    std::uint64_t count64 = 0U;
    std::uint32_t key_count = 0U;
    if (!get_u64(bytes, cursor, count64) || count64 > max_records || count64 > SIZE_MAX ||
        !get_u32(bytes, cursor, key_count) || key_count > count64 ||
        key_count > 1'000'000U) return false;
    if ((count64 == 0U) != (key_count == 0U)) return false;
    if (cursor >= crc_pos || bytes[cursor++] != kFields ||
        bytes.size() - cursor < kFields) return false;
    for (const auto kind : kSchema) if (bytes[cursor++] != kind) return false;

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
            row_count == 0U || row_count > count64 || row_count > SIZE_MAX ||
            cursor >= crc_pos) return false;
        const auto layout = bytes[cursor++];
        if (layout > 1U || cursor > crc_pos || child_len > crc_pos - cursor) return false;
        keys.emplace_back(entity, series);
        for (std::uint32_t j = 0U; j < i; ++j) if (keys[j] == keys.back()) return false;
        std::vector<std::uint8_t> child(
            bytes.begin() + static_cast<std::ptrdiff_t>(cursor),
            bytes.begin() + static_cast<std::ptrdiff_t>(cursor + static_cast<std::size_t>(child_len)));
        cursor += static_cast<std::size_t>(child_len);
        const bool decoded = layout == 0U
            ? decode_one_series(child, static_cast<std::size_t>(row_count), keys.back(), rows[i])
            : decode_one_series_columnar(child, static_cast<std::size_t>(row_count), keys.back(), rows[i]);
        if (!decoded) return false;
        for (std::size_t j = 1U; j < rows[i].size(); ++j) {
            if (rows[i][j].timestamp <= rows[i][j - 1U].timestamp) return false;
        }
        if (row_count > count64 - total_rows) return false;
        total_rows += row_count;
    }
    if (total_rows != count64 || cursor != crc_pos) return false;

    using HeapEntry = std::pair<std::uint64_t, std::size_t>;
    std::priority_queue<HeapEntry, std::vector<HeapEntry>, std::greater<HeapEntry>> heap;
    std::vector<std::size_t> cursors(key_count, 0U);
    for (std::size_t key = 0U; key < rows.size(); ++key) {
        if (!rows[key].empty()) heap.emplace(rows[key][0].timestamp, key);
    }
    output.reserve(static_cast<std::size_t>(count64));
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


std::vector<std::uint8_t> zstd_compress_bytes(const std::vector<std::uint8_t>& input, int level) {
    const std::size_t bound = ZSTD_compressBound(input.size());
    if (ZSTD_isError(bound)) throw std::runtime_error("Zstandard compression bound failed");
    std::vector<std::uint8_t> output(bound);
    const std::size_t written = ZSTD_compress(output.data(), output.size(),
                                              input.data(), input.size(), level);
    if (ZSTD_isError(written))
        throw std::runtime_error(std::string("Zstandard child compression failed: ") + ZSTD_getErrorName(written));
    output.resize(written);
    return output;
}

bool zstd_decompress_bytes(const std::vector<std::uint8_t>& input, std::size_t expected_size,
                           std::vector<std::uint8_t>& output) {
    output.assign(expected_size, 0U);
    const std::size_t written = ZSTD_decompress(output.data(), output.size(), input.data(), input.size());
    if (ZSTD_isError(written) || written != expected_size) {
        output.clear();
        return false;
    }
    return true;
}

std::vector<std::uint8_t> encode_partitioned_zstd_adaptive(const std::vector<Record>& records,
                                                            std::size_t* row_children = nullptr,
                                                            std::size_t* columnar_children = nullptr,
                                                            int zstd_level = 19) {
    if (zstd_level < 1 || zstd_level > ZSTD_maxCLevel())
        throw std::runtime_error("unsupported adaptive child Zstandard level");
    for (std::size_t i = 1U; i < records.size(); ++i) {
        if (records[i].timestamp <= records[i - 1U].timestamp)
            throw std::runtime_error("compressed adaptive frame requires strictly increasing unique timestamps");
    }
    std::map<Key, std::size_t> key_to_id;
    std::vector<Key> keys;
    std::vector<std::vector<Record>> per_key;
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
    std::vector<std::uint8_t> layouts;
    std::vector<std::uint64_t> raw_lengths;
    std::vector<std::uint32_t> raw_checksums;
    streams.reserve(keys.size());
    layouts.reserve(keys.size());
    raw_lengths.reserve(keys.size());
    raw_checksums.reserve(keys.size());
    std::size_t row_count = 0U;
    std::size_t columnar_count = 0U;

    for (const auto& rows : per_key) {
        const auto row_stream = encode_one_series(rows);
        const auto columnar_stream = encode_one_series_columnar(rows);
        const auto row_compressed = zstd_compress_bytes(row_stream, zstd_level);
        const auto columnar_compressed = zstd_compress_bytes(columnar_stream, zstd_level);
        if (row_compressed.size() <= columnar_compressed.size()) {
            streams.push_back(row_compressed);
            layouts.push_back(0U);
            raw_lengths.push_back(static_cast<std::uint64_t>(row_stream.size()));
            raw_checksums.push_back(crc32(row_stream.data(), row_stream.size()));
            ++row_count;
        } else {
            streams.push_back(columnar_compressed);
            layouts.push_back(1U);
            raw_lengths.push_back(static_cast<std::uint64_t>(columnar_stream.size()));
            raw_checksums.push_back(crc32(columnar_stream.data(), columnar_stream.size()));
            ++columnar_count;
        }
    }
    if (row_children != nullptr) *row_children = row_count;
    if (columnar_children != nullptr) *columnar_children = columnar_count;

    std::vector<std::uint8_t> out;
    out.insert(out.end(), {'P', 'Z', 'C', '1'});
    put_u64(records.size(), out);
    put_u32(static_cast<std::uint32_t>(keys.size()), out);
    out.push_back(static_cast<std::uint8_t>(kFields));
    out.insert(out.end(), kSchema.begin(), kSchema.end());
    put_u32(static_cast<std::uint32_t>(zstd_level), out);
    for (std::size_t i = 0U; i < keys.size(); ++i) {
        put_u32(keys[i].first, out);
        put_u32(keys[i].second, out);
        put_u64(per_key[i].size(), out);
        out.push_back(layouts[i]);
        put_u64(raw_lengths[i], out);
        put_u64(streams[i].size(), out);
        put_u32(raw_checksums[i], out);
        out.insert(out.end(), streams[i].begin(), streams[i].end());
    }
    put_u32(crc32(out.data(), out.size()), out);
    return out;
}

bool decode_partitioned_zstd_adaptive(const std::vector<std::uint8_t>& bytes, std::size_t max_records,
                                      std::vector<Record>& output) {
    output.clear();
    if (bytes.size() < 4U + 8U + 4U + 1U + kFields + 4U + 4U) return false;
    const std::size_t crc_pos = bytes.size() - 4U;
    std::size_t crc_cursor = crc_pos;
    std::uint32_t stored_crc = 0U;
    if (!get_u32(bytes, crc_cursor, stored_crc) || crc32(bytes.data(), crc_pos) != stored_crc) return false;

    std::size_t cursor = 0U;
    if (bytes[cursor++] != 'P' || bytes[cursor++] != 'Z' ||
        bytes[cursor++] != 'C' || bytes[cursor++] != '1') return false;
    std::uint64_t count64 = 0U;
    std::uint32_t key_count = 0U;
    if (!get_u64(bytes, cursor, count64) || count64 > max_records || count64 > SIZE_MAX ||
        !get_u32(bytes, cursor, key_count) || key_count > count64 ||
        key_count > 1'000'000U) return false;
    if ((count64 == 0U) != (key_count == 0U)) return false;
    if (cursor >= crc_pos || bytes[cursor++] != kFields || bytes.size() - cursor < kFields) return false;
    for (const auto kind : kSchema) if (bytes[cursor++] != kind) return false;
    std::uint32_t level = 0U;
    if (!get_u32(bytes, cursor, level) || level < 1U ||
        level > static_cast<std::uint32_t>(ZSTD_maxCLevel())) return false;

    std::vector<Key> keys;
    std::vector<std::vector<Record>> rows;
    keys.reserve(key_count);
    rows.resize(key_count);
    std::uint64_t total_rows = 0U;
    for (std::uint32_t i = 0U; i < key_count; ++i) {
        std::uint32_t entity = 0U, series = 0U;
        std::uint64_t row_count = 0U, raw_len = 0U, compressed_len = 0U;
        if (!get_u32(bytes, cursor, entity) || !get_u32(bytes, cursor, series) ||
            !get_u64(bytes, cursor, row_count) || row_count == 0U || row_count > count64 ||
            row_count > SIZE_MAX || cursor >= crc_pos) return false;
        const auto layout = bytes[cursor++];
        std::uint32_t child_crc = 0U;
        if (layout > 1U || !get_u64(bytes, cursor, raw_len) ||
            !get_u64(bytes, cursor, compressed_len) || !get_u32(bytes, cursor, child_crc)) return false;
        if (raw_len == 0U || raw_len > SIZE_MAX || row_count > (UINT64_MAX - 16U) / 128U ||
            raw_len > row_count * 128U + 16U || cursor > crc_pos || compressed_len > crc_pos - cursor) return false;
        keys.emplace_back(entity, series);
        for (std::uint32_t j = 0U; j < i; ++j) if (keys[j] == keys.back()) return false;
        std::vector<std::uint8_t> compressed(
            bytes.begin() + static_cast<std::ptrdiff_t>(cursor),
            bytes.begin() + static_cast<std::ptrdiff_t>(cursor + static_cast<std::size_t>(compressed_len)));
        cursor += static_cast<std::size_t>(compressed_len);
        std::vector<std::uint8_t> child;
        if (!zstd_decompress_bytes(compressed, static_cast<std::size_t>(raw_len), child) ||
            crc32(child.data(), child.size()) != child_crc) return false;
        const bool decoded = layout == 0U
            ? decode_one_series(child, static_cast<std::size_t>(row_count), keys.back(), rows[i])
            : decode_one_series_columnar(child, static_cast<std::size_t>(row_count), keys.back(), rows[i]);
        if (!decoded) return false;
        for (std::size_t j = 1U; j < rows[i].size(); ++j) {
            if (rows[i][j].timestamp <= rows[i][j - 1U].timestamp) return false;
        }
        if (row_count > count64 - total_rows) return false;
        total_rows += row_count;
    }
    if (total_rows != count64 || cursor != crc_pos) return false;

    using HeapEntry = std::pair<std::uint64_t, std::size_t>;
    std::priority_queue<HeapEntry, std::vector<HeapEntry>, std::greater<HeapEntry>> heap;
    std::vector<std::size_t> cursors(key_count, 0U);
    for (std::size_t key = 0U; key < rows.size(); ++key) {
        if (!rows[key].empty()) heap.emplace(rows[key][0].timestamp, key);
    }
    output.reserve(static_cast<std::size_t>(count64));
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

void write_file(const std::filesystem::path& path,const std::vector<std::uint8_t>& bytes){std::ofstream f(path,std::ios::binary);if(!f)throw std::runtime_error("open failed");f.write(reinterpret_cast<const char*>(bytes.data()),static_cast<std::streamsize>(bytes.size()));if(!f)throw std::runtime_error("write failed");}
void require(bool v,const std::string& m){if(!v)throw std::runtime_error(m);}
void test_columnar_empty() {
    const std::vector<Record> records;
    const auto frame = encode_partitioned_columnar(records);
    std::vector<Record> decoded;
    require(decode_partitioned_columnar(frame, 0U, decoded), "empty columnar frame rejected");
    require(decoded.empty(), "empty columnar frame produced records");
}
void test_columnar_roundtrips() {
    for (const std::string name : {"sparse", "bursty", "idle", "noisy"}) {
        const auto records = fixture(name, 0x6677U);
        const auto frame = encode_partitioned_columnar(records);
        std::vector<Record> decoded;
        require(decode_partitioned_columnar(frame, records.size(), decoded),
                name + " columnar frame rejected");
        require(decoded == records, name + " columnar exact reconstruction mismatch");
    }
}
void test_columnar_key_cardinality() {
    std::mt19937_64 rng(0xc011U);
    for (const std::size_t key_count : {1U, 8U, 64U, 1024U, 4096U}) {
        const std::size_t record_count = std::max<std::size_t>(4096U, key_count * 2U);
        std::vector<Record> last(key_count);
        std::vector<bool> initialized(key_count, false);
        std::vector<Record> records;
        records.reserve(record_count);
        for (std::size_t i = 0U; i < record_count; ++i) {
            const std::size_t key = static_cast<std::size_t>(rng() % key_count);
            Record record = initialized[key] ? last[key] : Record{};
            record.timestamp = static_cast<std::uint64_t>(i + 1U);
            record.entity = static_cast<std::uint32_t>(key / 16U);
            record.series = static_cast<std::uint32_t>(key % 16U);
            if (!initialized[key]) record.present = 0xffU;
            for (std::size_t field = 0U; field < kFields; ++field) {
                const auto bit = static_cast<std::uint8_t>(1U << field);
                if (!initialized[key] || rng() % 19U == 0U) {
                    if (initialized[key] && rng() % 11U == 0U)
                        record.present = static_cast<std::uint8_t>(record.present ^ bit);
                    record.values[field] = (record.present & bit) == 0U ? 0U : value_for(field, rng());
                }
            }
            records.push_back(record);
            last[key] = record;
            initialized[key] = true;
        }
        const auto frame = encode_partitioned_columnar(records);
        std::vector<Record> decoded;
        require(decode_partitioned_columnar(frame, records.size(), decoded),
                "columnar high-cardinality decode rejected");
        require(decoded == records, "columnar high-cardinality reconstruction mismatch");
    }
}
void test_columnar_corruption() {
    const auto records = fixture("sparse", 0x77ccU);
    const auto valid = encode_partitioned_columnar(records);
    std::vector<Record> decoded;
    auto corrupt = valid;
    corrupt[0] ^= 1U;
    require(!decode_partitioned_columnar(corrupt, records.size(), decoded),
            "columnar bad magic/CRC accepted");
    corrupt = valid;
    corrupt.pop_back();
    require(!decode_partitioned_columnar(corrupt, records.size(), decoded),
            "columnar truncated frame accepted");
    corrupt = valid;
    corrupt.push_back(0U);
    require(!decode_partitioned_columnar(corrupt, records.size(), decoded),
            "columnar trailing bytes accepted");
}
void test_adaptive_empty() {
    const std::vector<Record> records;
    const auto frame = encode_partitioned_adaptive(records);
    std::vector<Record> decoded;
    require(decode_partitioned_adaptive(frame, 0U, decoded), "empty adaptive frame rejected");
    require(decoded.empty(), "empty adaptive frame produced records");
}
void test_adaptive_roundtrips() {
    for (const std::string name : {"sparse", "bursty", "idle", "noisy"}) {
        const auto records = fixture(name, 0xace5U);
        std::size_t row_children = 0U;
        std::size_t columnar_children = 0U;
        const auto frame = encode_partitioned_adaptive(records, &row_children, &columnar_children);
        require(row_children + columnar_children == kKeys, name + " adaptive child count mismatch");
        std::vector<Record> decoded;
        require(decode_partitioned_adaptive(frame, records.size(), decoded), name + " adaptive frame rejected");
        require(decoded == records, name + " adaptive exact reconstruction mismatch");
    }
}
void test_adaptive_layout_choice() {
    const auto idle = fixture("idle", 0x7e1e0001U);
    const auto noisy = fixture("noisy", 0x7e1e0001U);
    std::size_t idle_rows = 0U, idle_columns = 0U;
    std::size_t noisy_rows = 0U, noisy_columns = 0U;
    const auto idle_frame = encode_partitioned_adaptive(idle, &idle_rows, &idle_columns);
    const auto noisy_frame = encode_partitioned_adaptive(noisy, &noisy_rows, &noisy_columns);
    require(idle_rows + idle_columns == kKeys && noisy_rows + noisy_columns == kKeys,
            "adaptive layout counts do not cover every child");
    require(idle_rows > noisy_rows, "idle data should select more row-state children than noisy data");
    require(noisy_columns > idle_columns, "noisy data should select more columnar children than idle data");
    std::vector<Record> decoded;
    require(decode_partitioned_adaptive(idle_frame, idle.size(), decoded) && decoded == idle,
            "adaptive idle reconstruction mismatch");
    require(decode_partitioned_adaptive(noisy_frame, noisy.size(), decoded) && decoded == noisy,
            "adaptive noisy reconstruction mismatch");
}
void test_adaptive_corruption() {
    const auto rows = fixture("sparse", 0x7788U);
    const auto valid = encode_partitioned_adaptive(rows);
    std::vector<Record> decoded;
    auto bad = valid;
    bad[0] ^= 1U;
    require(!decode_partitioned_adaptive(bad, rows.size(), decoded), "adaptive bad magic/CRC accepted");
    bad = valid;
    bad.pop_back();
    require(!decode_partitioned_adaptive(bad, rows.size(), decoded), "adaptive truncated frame accepted");
    bad = valid;
    std::size_t cursor = 4U + 8U + 4U + 1U + kFields;
    std::uint32_t entity = 0U, series = 0U;
    std::uint64_t count = 0U, length = 0U;
    require(get_u32(bad, cursor, entity) && get_u32(bad, cursor, series) &&
            get_u64(bad, cursor, count) && get_u64(bad, cursor, length),
            "adaptive corruption fixture header could not be parsed");
    bad[cursor] = 2U; // unsupported child layout tag
    const std::size_t crc_position = bad.size() - 4U;
    const auto checksum = crc32(bad.data(), crc_position);
    std::size_t crc_write = crc_position;
    for (unsigned i = 0U; i < 4U; ++i)
        bad[crc_write++] = static_cast<std::uint8_t>(checksum >> (i * 8U));
    require(!decode_partitioned_adaptive(bad, rows.size(), decoded),
            "adaptive unsupported child layout tag accepted");
}
void test_adaptive_zstd_empty() {
    const std::vector<Record> records;
    const auto frame = encode_partitioned_zstd_adaptive(records);
    std::vector<Record> decoded;
    require(decode_partitioned_zstd_adaptive(frame, 0U, decoded), "empty compressed adaptive frame rejected");
    require(decoded.empty(), "empty compressed adaptive frame produced records");
}
void test_adaptive_zstd_roundtrips() {
    for (const std::string name : {"sparse", "bursty", "idle", "noisy"}) {
        const auto records = fixture(name, 0xace5U);
        std::size_t row_children = 0U;
        std::size_t columnar_children = 0U;
        const auto frame = encode_partitioned_zstd_adaptive(records, &row_children, &columnar_children);
        require(row_children + columnar_children == kKeys, name + " compressed adaptive child count mismatch");
        std::vector<Record> decoded;
        require(decode_partitioned_zstd_adaptive(frame, records.size(), decoded),
                name + " compressed adaptive frame rejected");
        require(decoded == records, name + " compressed adaptive exact reconstruction mismatch");
    }
}
void test_adaptive_zstd_layout_choice() {
    const auto idle = fixture("idle", 0x7e1e0001U);
    const auto noisy = fixture("noisy", 0x7e1e0001U);
    std::size_t idle_rows = 0U, idle_columns = 0U;
    std::size_t noisy_rows = 0U, noisy_columns = 0U;
    const auto idle_frame = encode_partitioned_zstd_adaptive(idle, &idle_rows, &idle_columns);
    const auto noisy_frame = encode_partitioned_zstd_adaptive(noisy, &noisy_rows, &noisy_columns);
    require(idle_rows + idle_columns == kKeys && noisy_rows + noisy_columns == kKeys,
            "compressed adaptive layout counts do not cover all children");
    require(idle_rows > noisy_rows, "idle data should choose more row-state children than noisy data");
    require(noisy_columns > idle_columns, "noisy data should choose more columnar children than idle data");
    std::vector<Record> decoded;
    require(decode_partitioned_zstd_adaptive(idle_frame, idle.size(), decoded) && decoded == idle,
            "compressed adaptive idle reconstruction mismatch");
    require(decode_partitioned_zstd_adaptive(noisy_frame, noisy.size(), decoded) && decoded == noisy,
            "compressed adaptive noisy reconstruction mismatch");
}
void test_adaptive_zstd_corruption() {
    const auto rows = fixture("sparse", 0x7788U);
    const auto valid = encode_partitioned_zstd_adaptive(rows);
    std::vector<Record> decoded;
    auto bad = valid;
    bad[0] ^= 1U;
    require(!decode_partitioned_zstd_adaptive(bad, rows.size(), decoded),
            "compressed adaptive bad magic/CRC accepted");
    bad = valid;
    bad.pop_back();
    require(!decode_partitioned_zstd_adaptive(bad, rows.size(), decoded),
            "compressed adaptive truncated frame accepted");
}
void test_empty_stream(){const std::vector<Record> records;const auto frame=encode_partitioned(records);std::vector<Record> decoded;require(decode_partitioned(frame,0U,decoded),"empty partitioned stream rejected");require(decoded.empty(),"empty partitioned stream produced records");}
void test_roundtrips(){for(const std::string name:{"sparse","bursty","idle","noisy"}){const auto records=fixture(name,0x7878U);const auto frame=encode_partitioned(records);require(frame.size()>26U&&frame[25]==0U,name+" should use timestamp-merge mode");std::vector<Record> decoded;require(decode_partitioned(frame,records.size(),decoded),name+" partition decoder rejected frame");require(decoded==records,name+" partition order/data mismatch");}}
void test_schedule_fallback(){auto records=fixture("sparse",0x887766U);for(std::size_t i=0U;i<records.size();++i)records[i].timestamp=static_cast<std::uint64_t>(i/kKeys)*1000U;const auto duplicate_frame=encode_partitioned(records);require(duplicate_frame.size()>26U&&duplicate_frame[25]==1U,"duplicate timestamps should use tie-template mode");std::vector<Record> decoded;require(decode_partitioned(duplicate_frame,records.size(),decoded)&&decoded==records,"duplicate timestamp fallback order mismatch");std::uint64_t tie_order_bits=0U;std::size_t offset=30U;require(get_u64(duplicate_frame,offset,tie_order_bits),"tie-template order bit count missing");require(tie_order_bits<records.size()*key_bits(kKeys),"repeated tie templates should compress below full schedule");require(tie_order_bits<9000U,"repeated tie templates should stay compact across 1,024 identical groups");auto changing=records;for(std::size_t start=0U;start<changing.size();start+=kKeys)if((start/kKeys)%2U==1U)std::reverse(changing.begin()+static_cast<std::ptrdiff_t>(start),changing.begin()+static_cast<std::ptrdiff_t>(start+kKeys));const auto changing_frame=encode_partitioned(changing);require(changing_frame[25]==1U,"nondecreasing changing ties should use template mode");require(decode_partitioned(changing_frame,changing.size(),decoded)&&decoded==changing,"changing tie template order mismatch");std::uint64_t changing_bits=0U;std::size_t changing_offset=30U;require(get_u64(changing_frame,changing_offset,changing_bits),"changing tie order bit count missing");require(changing_bits<10000U,"two recurring tie templates should be cached");auto out_of_order=fixture("bursty",0x778899U);std::swap(out_of_order[1],out_of_order[2]);const auto disorder_frame=encode_partitioned(out_of_order);require(disorder_frame.size()>26U&&disorder_frame[25]==2U,"out-of-order timestamps should use full schedule");require(decode_partitioned(disorder_frame,out_of_order.size(),decoded)&&decoded==out_of_order,"out-of-order fallback mismatch");}
void test_same_series_duplicate_timestamp_rows() {
    const auto base = fixture("sparse", 0x2345U);
    std::vector<Record> records;
    for (std::size_t group = 0U; group < 8U; ++group) {
        for (std::size_t key = 0U; key < kKeys; ++key) {
            Record first = base[group * kKeys + key];
            first.timestamp = static_cast<std::uint64_t>(group) * 1000U;
            Record second = first;
            second.values[0] = value_for(0U, static_cast<std::uint64_t>(group * kKeys + key + 100U));
            records.push_back(first);
            records.push_back(second);
        }
    }
    const auto frame = encode_partitioned(records);
    require(frame[25] == 1U, "same-series duplicate-time rows should use tie-template mode");
    std::vector<Record> decoded;
    require(decode_partitioned(frame, records.size(), decoded), "same-series duplicate-time frame rejected");
    require(decoded == records, "same-series duplicate-time order mismatch");
}

void test_randomized(){std::mt19937_64 rng(0x991122U);for(std::size_t trial=0U;trial<100U;++trial){std::vector<Record> records;std::array<Record,8U> states{};std::array<bool,8U> init{};const std::size_t n=static_cast<std::size_t>(rng()%300U);std::uint64_t time=0U;for(std::size_t i=0U;i<n;++i){const auto k=static_cast<std::size_t>(rng()%8U);time+=rng()%7U;Record r=init[k]?states[k]:Record{};r.timestamp=time;r.entity=static_cast<std::uint32_t>(k/4U);r.series=static_cast<std::uint32_t>(k%4U);if(!init[k])r.present=0xffU;for(std::size_t f=0U;f<kFields;++f){const auto bit=static_cast<std::uint8_t>(1U<<f);if(!init[k]||rng()%20U==0U){if(init[k]&&rng()%8U==0U)r.present=static_cast<std::uint8_t>(r.present^bit);r.values[f]=(r.present&bit)==0U?0U:value_for(f,rng());}}records.push_back(r);states[k]=r;init[k]=true;}const auto frame=encode_partitioned(records);std::vector<Record> decoded;require(decode_partitioned(frame,records.size(),decoded),"random partition decode rejected");require(decoded==records,"random partition mismatch");}}
void test_corruption(){const auto rows=fixture("sparse",0x8877U);auto frame=encode_partitioned(rows);std::vector<Record> out;auto bad=frame;bad[0]^=1U;require(!decode_partitioned(bad,rows.size(),out),"bad magic/CRC accepted");bad=frame;bad.pop_back();require(!decode_partitioned(bad,rows.size(),out),"truncated partition frame accepted");bad=frame;bad.push_back(0U);require(!decode_partitioned(bad,rows.size(),out),"trailing partition bytes accepted");}
void emit(const std::filesystem::path& dir) {
    std::filesystem::create_directories(dir);
    for (const std::string name : {"sparse", "bursty", "idle", "noisy"}) {
        const auto rows = fixture(name, 0x7e1e0001U);
        const auto raw = framed_raw(rows);
        const auto delta = framed_global_state_delta(rows);
        const auto partitioned = encode_partitioned(rows);
        const auto columnar = encode_partitioned_columnar(rows);
        std::size_t raw_row_children = 0U, raw_columnar_children = 0U;
        std::size_t zstd_row_children = 0U, zstd_columnar_children = 0U;
        const auto adaptive = encode_partitioned_adaptive(rows, &raw_row_children, &raw_columnar_children);
        const auto adaptive_zstd = encode_partitioned_zstd_adaptive(rows, &zstd_row_children, &zstd_columnar_children);
        std::vector<Record> decoded;
        require(decode_partitioned(partitioned, rows.size(), decoded) && decoded == rows,
                name + " row-partitioned emission round-trip failed");
        require(decode_partitioned_columnar(columnar, rows.size(), decoded) && decoded == rows,
                name + " field-columnar emission round-trip failed");
        require(decode_partitioned_adaptive(adaptive, rows.size(), decoded) && decoded == rows,
                name + " raw adaptive emission round-trip failed");
        require(decode_partitioned_zstd_adaptive(adaptive_zstd, rows.size(), decoded) && decoded == rows,
                name + " compressed adaptive emission round-trip failed");
        write_file(dir / (name + "_raw.bin"), raw);
        write_file(dir / (name + "_global_delta.bin"), delta);
        write_file(dir / (name + "_partitioned.bin"), partitioned);
        write_file(dir / (name + "_columnar.bin"), columnar);
        write_file(dir / (name + "_adaptive.bin"), adaptive);
        write_file(dir / (name + "_adaptive_zstd.bin"), adaptive_zstd);
        std::cout << name << ",records=" << rows.size()
                  << ",raw_frame=" << raw.size()
                  << ",global_delta_frame=" << delta.size()
                  << ",row_partitioned_frame=" << partitioned.size()
                  << ",columnar_frame=" << columnar.size()
                  << ",raw_adaptive_frame=" << adaptive.size()
                  << ",raw_adaptive_row_children=" << raw_row_children
                  << ",raw_adaptive_columnar_children=" << raw_columnar_children
                  << ",zstd_adaptive_frame=" << adaptive_zstd.size()
                  << ",zstd_adaptive_row_children=" << zstd_row_children
                  << ",zstd_adaptive_columnar_children=" << zstd_columnar_children << "\n";
    }
}

int main(int argc,char** argv){try{test_empty_stream();test_roundtrips();test_schedule_fallback();test_same_series_duplicate_timestamp_rows();test_randomized();test_corruption();test_columnar_empty();test_columnar_roundtrips();test_columnar_key_cardinality();test_columnar_corruption();test_adaptive_empty();test_adaptive_roundtrips();test_adaptive_layout_choice();test_adaptive_corruption();test_adaptive_zstd_empty();test_adaptive_zstd_roundtrips();test_adaptive_zstd_layout_choice();test_adaptive_zstd_corruption();std::cout<<"row_frame_tests=PASS columnar_empty=PASS columnar_fixtures=4 columnar_key_counts=1,8,64,1024,4096 columnar_corruption=3 adaptive_empty=PASS adaptive_fixtures=4 adaptive_layout_choice=PASS adaptive_corruption=3 zstd_adaptive_empty=PASS zstd_adaptive_fixtures=4 zstd_adaptive_layout_choice=PASS zstd_adaptive_corruption=2 randomized_roundtrips=100 status=PASS\\n";if(argc==3&&std::string(argv[1])=="--emit-dir")emit(argv[2]);else if(argc!=1){std::cerr<<"usage: "<<argv[0]<<" [--emit-dir DIR]\\n";return 2;}return 0;}catch(const std::exception& e){std::cerr<<"partitioned reference failed: "<<e.what()<<"\\n";return 1;}}
