// Independent C++20 reference for per-series state-delta plus whole-record heartbeat runs.
// Transform-level only: no Rust linkage, arithmetic coder, or production container.
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
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {
constexpr std::size_t kFields = 8U;
constexpr std::uint32_t kStateCapacity = 64U;
constexpr std::uint32_t kSegmentRecords = 4096U;
struct Record {
    std::uint64_t timestamp{};
    std::uint32_t entity{};
    std::uint32_t series{};
    std::uint8_t present{};
    std::array<std::uint64_t,kFields> values{};
    bool operator==(const Record&) const = default;
};
using Key = std::pair<std::uint32_t,std::uint32_t>;
struct Entry { Record record; std::uint64_t last_seen{}; };
using State = std::map<Key,Entry>;

void append_u64(std::uint64_t value, std::vector<std::uint8_t>& out) {
    for (unsigned shift=0U; shift<64U; shift+=8U) out.push_back(static_cast<std::uint8_t>(value >> shift));
}
void append_u32(std::uint32_t value, std::vector<std::uint8_t>& out) {
    for (unsigned shift=0U; shift<32U; shift+=8U) out.push_back(static_cast<std::uint8_t>(value >> shift));
}
void append_varint(std::uint64_t value, std::vector<std::uint8_t>& out) {
    while (value >= 0x80U) { out.push_back(static_cast<std::uint8_t>((value & 0x7fU) | 0x80U)); value >>= 7U; }
    out.push_back(static_cast<std::uint8_t>(value));
}
bool read_varint(const std::vector<std::uint8_t>& in, std::size_t& cursor, std::uint64_t& value) {
    value=0U;
    for (std::size_t i=0U;i<10U;++i) {
        if (cursor>=in.size()) return false;
        const std::uint8_t byte=in[cursor++];
        if (i==9U && (byte&0xfeU)!=0U) return false;
        value |= static_cast<std::uint64_t>(byte&0x7fU) << (i*7U);
        if ((byte&0x80U)==0U) return !(i>0U && (byte&0x7fU)==0U);
    }
    return false;
}
void evict_oldest(State& states) {
    if (states.empty()) throw std::runtime_error("evict empty state");
    auto victim=states.begin();
    for (auto it=std::next(states.begin());it!=states.end();++it)
        if (std::tie(it->second.last_seen,it->first)<std::tie(victim->second.last_seen,victim->first)) victim=it;
    states.erase(victim);
}
Key key_of(const Record& r) { return {r.entity,r.series}; }
bool same_state(const Record& a,const Record& b) {
    return a.entity==b.entity && a.series==b.series && a.present==b.present && a.values==b.values;
}
std::uint8_t changed_mask(const Record& a,const Record& b) {
    std::uint8_t changed=0U;
    for (std::size_t i=0U;i<kFields;++i) {
        const auto bit=static_cast<std::uint8_t>(1U<<i);
        const bool ap=(a.present&bit)!=0U, bp=(b.present&bit)!=0U;
        if (ap!=bp || (bp && a.values[i]!=b.values[i])) changed=static_cast<std::uint8_t>(changed|bit);
    }
    return changed;
}
bool valid_record(const Record& r) {
    for (std::size_t i=0U;i<kFields;++i) {
        const auto bit=static_cast<std::uint8_t>(1U<<i);
        if ((r.present&bit)==0U && r.values[i]!=0U) return false;
        if (i==2U && (r.present&bit)!=0U && r.values[i]>UINT32_MAX) return false;
        if (i==3U && (r.present&bit)!=0U && r.values[i]>1U) return false;
    }
    return true;
}

std::vector<std::uint8_t> encode_state_delta(const std::vector<Record>& records) {
    std::vector<std::uint8_t> out; out.reserve(records.size()*88U);
    State states; bool have_time=false; std::uint64_t previous_time=0U;
    for (std::size_t i=0U;i<records.size();++i) {
        const Record& r=records[i]; if (!valid_record(r)) throw std::runtime_error("invalid record");
        const std::size_t offset=i%kSegmentRecords;
        if (offset==0U) { states.clear(); have_time=false; }
        if (have_time && r.timestamp>=previous_time) { out.push_back(0U); append_varint(r.timestamp-previous_time,out); }
        else { out.push_back(1U); append_varint(r.timestamp,out); }
        previous_time=r.timestamp; have_time=true;
        append_varint(r.entity,out); append_varint(r.series,out);
        const Key key=key_of(r); const auto found=states.find(key);
        std::uint8_t state_mode=0U, changed=0xffU;
        if (found==states.end()) { if (states.size()>=kStateCapacity) evict_oldest(states); state_mode=1U; }
        else changed=changed_mask(found->second.record,r);
        out.push_back(state_mode); out.push_back(changed); out.push_back(r.present);
        for (std::size_t field=0U;field<kFields;++field) {
            const auto bit=static_cast<std::uint8_t>(1U<<field);
            if ((changed&bit)!=0U && (r.present&bit)!=0U) append_u64(r.values[field],out);
        }
        states[key]=Entry{r,static_cast<std::uint64_t>(offset)};
    }
    return out;
}

// time_mode=2: repeat the preceding same-key state run_count times, with each
// timestamp advanced by the same exact nonnegative delta. Runs never cross a segment.
std::vector<std::uint8_t> encode_record_runs(const std::vector<Record>& records) {
    std::vector<std::uint8_t> out; out.reserve(records.size()*16U);
    State states; bool have_time=false; std::uint64_t previous_time=0U;
    std::size_t index=0U;
    while (index<records.size()) {
        const Record& r=records[index]; if (!valid_record(r)) throw std::runtime_error("invalid record");
        const std::size_t offset=index%kSegmentRecords;
        if (offset==0U) { states.clear(); have_time=false; }
        if (offset>0U && index>0U && have_time && r.timestamp>=previous_time &&
            same_state(records[index-1U],r)) {
            const std::uint64_t delta=r.timestamp-previous_time;
            const std::size_t segment_end=std::min(records.size(),index+(kSegmentRecords-offset));
            std::size_t run_count=0U; std::uint64_t expected_time=previous_time;
            for (std::size_t probe=index;probe<segment_end;++probe) {
                if (!same_state(records[probe-1U],records[probe])) break;
                if (records[probe].timestamp<expected_time || records[probe].timestamp-expected_time!=delta) break;
                expected_time=records[probe].timestamp; ++run_count;
            }
            if (run_count>=2U) {
                out.push_back(2U); append_varint(delta,out); append_varint(run_count,out);
                const Record& last=records[index+run_count-1U]; const Key key=key_of(last);
                if (states.find(key)==states.end()) throw std::runtime_error("run references absent key state");
                states[key]=Entry{last,static_cast<std::uint64_t>((index+run_count-1U)%kSegmentRecords)};
                previous_time=last.timestamp; have_time=true; index+=run_count; continue;
            }
        }
        if (have_time && r.timestamp>=previous_time) { out.push_back(0U); append_varint(r.timestamp-previous_time,out); }
        else { out.push_back(1U); append_varint(r.timestamp,out); }
        previous_time=r.timestamp; have_time=true;
        append_varint(r.entity,out); append_varint(r.series,out);
        const Key key=key_of(r); const auto found=states.find(key);
        std::uint8_t state_mode=0U, changed=0xffU;
        if (found==states.end()) { if (states.size()>=kStateCapacity) evict_oldest(states); state_mode=1U; }
        else changed=changed_mask(found->second.record,r);
        out.push_back(state_mode); out.push_back(changed); out.push_back(r.present);
        for (std::size_t field=0U;field<kFields;++field) {
            const auto bit=static_cast<std::uint8_t>(1U<<field);
            if ((changed&bit)!=0U && (r.present&bit)!=0U) append_u64(r.values[field],out);
        }
        states[key]=Entry{r,static_cast<std::uint64_t>(offset)}; ++index;
    }
    return out;
}

bool decode_record_runs(const std::vector<std::uint8_t>& in,std::size_t count,std::vector<Record>& out) {
    out.clear(); out.reserve(count); State states; bool have_time=false; std::uint64_t previous_time=0U;
    std::size_t cursor=0U,index=0U;
    while (index<count) {
        const std::size_t offset=index%kSegmentRecords;
        if (offset==0U) { states.clear(); have_time=false; }
        if (cursor>=in.size()) return false;
        const std::uint8_t time_mode=in[cursor++];
        if (time_mode==2U) {
            if (offset==0U || out.empty() || !have_time) return false;
            std::uint64_t delta=0U,run_count64=0U;
            if (!read_varint(in,cursor,delta) || !read_varint(in,cursor,run_count64)) return false;
            if (run_count64<2U || run_count64>count-index || run_count64>kSegmentRecords-offset) return false;
            const Record previous=out.back(); const Key key=key_of(previous); const auto found=states.find(key);
            if (found==states.end() || !same_state(found->second.record,previous)) return false;
            std::uint64_t timestamp=out.back().timestamp; Record last=previous;
            for (std::uint64_t j=0U;j<run_count64;++j) {
                if (delta>UINT64_MAX-timestamp) return false;
                timestamp+=delta; last.timestamp=timestamp; out.push_back(last);
            }
            states[key]=Entry{last,static_cast<std::uint64_t>((index+static_cast<std::size_t>(run_count64)-1U)%kSegmentRecords)};
            previous_time=timestamp; have_time=true; index+=static_cast<std::size_t>(run_count64); continue;
        }
        if (time_mode>1U) return false;
        std::uint64_t time_value=0U;
        if (!read_varint(in,cursor,time_value)) return false;
        std::uint64_t timestamp=0U;
        if (!have_time) { if (time_mode!=1U) return false; timestamp=time_value; }
        else if (time_mode==0U) { if (time_value>UINT64_MAX-previous_time) return false; timestamp=previous_time+time_value; }
        else { if (time_value>=previous_time) return false; timestamp=time_value; }
        previous_time=timestamp; have_time=true;
        std::uint64_t entity=0U,series=0U;
        if (!read_varint(in,cursor,entity)||!read_varint(in,cursor,series)||entity>UINT32_MAX||series>UINT32_MAX) return false;
        const Key key{static_cast<std::uint32_t>(entity),static_cast<std::uint32_t>(series)};
        const auto found=states.find(key); const bool has_previous=found!=states.end();
        if (!has_previous && states.size()>=kStateCapacity) evict_oldest(states);
        if (in.size()-cursor<3U) return false;
        const std::uint8_t state_mode=in[cursor++],changed=in[cursor++],present=in[cursor++];
        if (state_mode!=(has_previous?0U:1U) || (!has_previous && changed!=0xffU)) return false;
        Record r{}; r.timestamp=timestamp; r.entity=key.first; r.series=key.second; r.present=present;
        if (has_previous) r.values=found->second.record.values;
        for (std::size_t field=0U;field<kFields;++field) {
            const auto bit=static_cast<std::uint8_t>(1U<<field);
            if ((changed&bit)!=0U) {
                if ((present&bit)!=0U) {
                    if (in.size()-cursor<8U) return false;
                    std::uint64_t value=0U; for (unsigned byte=0U;byte<8U;++byte) value|=static_cast<std::uint64_t>(in[cursor++])<<(byte*8U);
                    r.values[field]=value;
                } else r.values[field]=0U;
            } else if (!has_previous || (((found->second.record.present&bit)!=0U)!=((present&bit)!=0U))) return false;
        }
        if (!valid_record(r)) return false;
        const std::uint8_t expected=has_previous?changed_mask(found->second.record,r):0xffU;
        if (changed!=expected) return false;
        states[key]=Entry{r,static_cast<std::uint64_t>(offset)}; out.push_back(r); ++index;
    }
    return cursor==in.size();
}

std::uint64_t field_value(std::size_t field,std::uint64_t random) {
    switch(field) {
        case 0U: return std::bit_cast<std::uint64_t>(static_cast<double>(random%100000U)/1000.0);
        case 1U: return std::bit_cast<std::uint64_t>(static_cast<double>(random%10000000U)/100.0);
        case 2U: return random%4096U;
        case 3U: return random&1U;
        case 4U: return random;
        case 5U: return random;
        case 6U: return std::rotl(random,17);
        default: return std::bit_cast<std::uint64_t>(static_cast<double>(random%1000000U)/10000.0);
    }
}
std::vector<Record> fixture(const std::string& workload,std::uint64_t seed) {
    constexpr std::uint32_t keys=64U,steps=1024U;
    std::mt19937_64 rng(seed); std::vector<Record> records; records.reserve(static_cast<std::size_t>(keys)*steps);
    for (std::uint32_t key=0U;key<keys;++key) {
        Record current{}; current.entity=key/16U; current.series=key%16U; current.present=0xffU;
        for (std::size_t field=0U;field<kFields;++field) current.values[field]=field_value(field,rng());
        for (std::uint32_t step=0U;step<steps;++step) {
            current.timestamp=(static_cast<std::uint64_t>(step)*keys+key)*1000U;
            for (std::size_t field=0U;field<kFields;++field) {
                const std::uint64_t random=rng(); bool update=false;
                if (step>0U) {
                    if (workload=="sparse") update=random%1000U<15U;
                    else if (workload=="idle") update=random%10000U<2U;
                    else if (workload=="bursty") update=(step%256U<24U)?random%100U<65U:random%1000U<5U;
                    else if (workload=="noisy") update=true;
                }
                if (update) {
                    const auto bit=static_cast<std::uint8_t>(1U<<field);
                    if (std::rotl(random,11)%2048U==0U) current.present=static_cast<std::uint8_t>(current.present^bit);
                    current.values[field]=(current.present&bit)==0U?0U:field_value(field,rng());
                }
            }
            records.push_back(current);
        }
    }
    return records;
}
std::vector<std::uint8_t> raw_bytes(const std::vector<Record>& records) {
    std::vector<std::uint8_t> out; out.reserve(8U+records.size()*81U);
    out.insert(out.end(),{2U,2U,3U,4U,1U,0U,5U,2U});
    for (const auto& r:records) { append_u64(r.timestamp,out); append_u32(r.entity,out); append_u32(r.series,out); out.push_back(r.present); for(auto v:r.values) append_u64(v,out); }
    return out;
}
void write_file(const std::filesystem::path& path,const std::vector<std::uint8_t>& bytes) {
    std::ofstream file(path,std::ios::binary); if(!file) throw std::runtime_error("open output failed");
    file.write(reinterpret_cast<const char*>(bytes.data()),static_cast<std::streamsize>(bytes.size()));
    if(!file) throw std::runtime_error("write output failed");
}
void require(bool condition,const std::string& message) { if(!condition) throw std::runtime_error(message); }
void test_fixtures() {
    for (const std::string name:{"sparse","bursty","noisy","idle"}) {
        const auto records=fixture(name,0x9911U); const auto encoded=encode_record_runs(records);
        std::vector<Record> decoded; require(decode_record_runs(encoded,records.size(),decoded),name+" decode failed");
        require(decoded==records,name+" bitwise record mismatch");
    }
}
void test_random_roundtrips() {
    std::mt19937_64 rng(0x20261119U);
    for (std::size_t trial=0U;trial<200U;++trial) {
        const std::size_t count=static_cast<std::size_t>(rng()%300U); std::vector<Record> records; records.reserve(count);
        std::array<Record,16U> last{}; std::array<bool,16U> initialized{}; std::uint64_t timestamp=1000U;
        for (std::size_t i=0U;i<count;++i) {
            const std::size_t key=static_cast<std::size_t>(rng()%16U); timestamp+=rng()%5U;
            Record r=initialized[key]?last[key]:Record{}; r.timestamp=timestamp; r.entity=static_cast<std::uint32_t>(key/4U); r.series=static_cast<std::uint32_t>(key%4U);
            if(!initialized[key]) r.present=0xffU;
            for(std::size_t field=0U;field<kFields;++field) {
                const auto bit=static_cast<std::uint8_t>(1U<<field);
                if(!initialized[key] || rng()%30U==0U) { if(initialized[key] && rng()%8U==0U) r.present=static_cast<std::uint8_t>(r.present^bit); r.values[field]=(r.present&bit)==0U?0U:field_value(field,rng()); }
            }
            records.push_back(r); last[key]=r; initialized[key]=true;
        }
        const auto encoded=encode_record_runs(records); std::vector<Record> decoded;
        require(decode_record_runs(encoded,records.size(),decoded),"random run decode failed"); require(decoded==records,"random run mismatch");
    }
}
void test_malformed() {
    std::vector<Record> decoded;
    require(!decode_record_runs({2U,0U,2U},2U,decoded),"run without predecessor accepted");
    const auto records=fixture("idle",0x9922U); const auto valid=encode_record_runs(records);
    auto truncated=valid; truncated.pop_back(); require(!decode_record_runs(truncated,records.size(),decoded),"truncated run accepted");
    auto trailing=valid; trailing.push_back(0U); require(!decode_record_runs(trailing,records.size(),decoded),"trailing bytes accepted");
}
void emit(const std::filesystem::path& directory) {
    std::filesystem::create_directories(directory);
    for(const std::string name:{"sparse","bursty","noisy","idle"}) {
        const auto records=fixture(name,0x7e1e0001U); const auto raw=raw_bytes(records);
        const auto delta=encode_state_delta(records); const auto runs=encode_record_runs(records);
        std::vector<Record> decoded; require(decode_record_runs(runs,records.size(),decoded),name+" round-trip failed"); require(decoded==records,name+" mismatch");
        write_file(directory/(name+"_raw.bin"),raw); write_file(directory/(name+"_state_delta.bin"),delta); write_file(directory/(name+"_state_delta_record_runs.bin"),runs);
        std::cout<<name<<": records="<<records.size()<<" raw_bytes="<<raw.size()<<" state_delta_bytes="<<delta.size()<<" record_run_bytes="<<runs.size()<<"\n";
    }
}
}
int main(int argc,char** argv) {
    try {
        test_fixtures(); test_random_roundtrips(); test_malformed();
        std::cout<<"fixture_roundtrips=4 randomized_roundtrips=200 malformed_rejections=3 reference_status=PASS\n";
        if(argc==3 && std::string(argv[1])=="--emit-dir") emit(argv[2]);
        else if(argc!=1) { std::cerr<<"usage: "<<argv[0]<<" [--emit-dir DIR]\n"; return 2; }
        return 0;
    } catch(const std::exception& e) { std::cerr<<"reference failure: "<<e.what()<<"\n"; return 1; }
}
