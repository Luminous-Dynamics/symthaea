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
bool decode_one_series(const std::vector<std::uint8_t>& bytes,std::size_t count,Key key,std::vector<Record>& out){
    out.clear();out.reserve(count);std::size_t p=0U;std::uint64_t last_time=0U;
    while(out.size()<count){
        if(p>=bytes.size()) return false;
        const auto mode=bytes[p++];
        if(mode==2U){
            if(out.empty()) return false;
            std::uint64_t delta=0U,run=0U;
            if(!get_var(bytes,p,delta)||!get_var(bytes,p,run)||run<2U||run>count-out.size()) return false;
            std::uint64_t t=out.back().timestamp;Record r=out.back();
            for(std::uint64_t j=0U;j<run;++j){if(delta>UINT64_MAX-t)return false;t+=delta;r.timestamp=t;out.push_back(r);}
            last_time=t;continue;
        }
        if(mode>1U)return false;
        std::uint64_t tv=0U;if(!get_var(bytes,p,tv))return false;
        std::uint64_t timestamp=0U;
        if(out.empty()){
            if(mode!=1U)return false;timestamp=tv;if(p>=bytes.size())return false;
            const auto present=bytes[p++];Record r{};r.timestamp=timestamp;r.entity=key.first;r.series=key.second;r.present=present;
            for(std::size_t f=0U;f<kFields;++f)if((present&(1U<<f))!=0U){std::uint64_t value=0U;if(!get_u64(bytes,p,value))return false;r.values[f]=value;}
            if(!valid_record(r))return false;out.push_back(r);last_time=timestamp;continue;
        }
        if(mode==0U){if(tv>UINT64_MAX-last_time)return false;timestamp=last_time+tv;}
        else{if(tv>=last_time)return false;timestamp=tv;}
        last_time=timestamp;if(bytes.size()-p<2U)return false;
        const auto mask=bytes[p++];const auto present=bytes[p++];Record r=out.back();r.timestamp=timestamp;
        for(std::size_t f=0U;f<kFields;++f){const auto bit=static_cast<std::uint8_t>(1U<<f);if((mask&bit)!=0U){if((present&bit)!=0U){std::uint64_t value=0U;if(!get_u64(bytes,p,value))return false;r.values[f]=value;}else r.values[f]=0U;}}
        r.present=present;if(!valid_record(r)||mask!=changed_mask(out.back(),r))return false;out.push_back(r);
    }
    return p==bytes.size();
}
std::vector<std::uint8_t> encode_partitioned(const std::vector<Record>& records){
    std::map<Key,std::size_t> key_to_id;std::vector<Key> keys;std::vector<std::vector<Record>> per_key;std::vector<std::size_t> schedule;schedule.reserve(records.size());
    for(const auto& r:records){const Key key=key_of(r);auto [it,inserted]=key_to_id.emplace(key,keys.size());if(inserted){keys.push_back(key);per_key.emplace_back();}const auto id=it->second;per_key[id].push_back(r);schedule.push_back(id);}
    std::vector<std::vector<std::uint8_t>> streams;streams.reserve(keys.size());for(const auto& rows:per_key)streams.push_back(encode_one_series(rows));
    const std::size_t bits_per_id=key_bits(keys.size());BitWriter order;for(const auto id:schedule)order.put(static_cast<std::uint32_t>(id),bits_per_id);
    std::vector<std::uint8_t> out;out.insert(out.end(),{'P','R','N','1'});put_u64(records.size(),out);put_u32(static_cast<std::uint32_t>(keys.size()),out);out.push_back(static_cast<std::uint8_t>(kFields));out.insert(out.end(),kSchema.begin(),kSchema.end());put_u32(static_cast<std::uint32_t>(bits_per_id),out);put_u64(order.bits,out);
    for(std::size_t i=0U;i<keys.size();++i){put_u32(keys[i].first,out);put_u32(keys[i].second,out);put_u64(per_key[i].size(),out);put_u64(streams[i].size(),out);out.insert(out.end(),streams[i].begin(),streams[i].end());}
    put_u64(order.bytes.size(),out);out.insert(out.end(),order.bytes.begin(),order.bytes.end());put_u32(crc32(out.data(),out.size()),out);return out;
}
bool decode_partitioned(const std::vector<std::uint8_t>& bytes,std::size_t max_records,std::vector<Record>& output){
    output.clear();if(bytes.size()<4U+8U+4U+1U+8U+4U+8U+8U+4U)return false;const std::size_t crc_pos=bytes.size()-4U;std::size_t crc_cursor=crc_pos;std::uint32_t stored_crc=0U;if(!get_u32(bytes,crc_cursor,stored_crc)||crc32(bytes.data(),crc_pos)!=stored_crc)return false;
    std::size_t p=0U;if(bytes[p++]!='P'||bytes[p++]!='R'||bytes[p++]!='N'||bytes[p++]!='1')return false;std::uint64_t count64=0U;std::uint32_t key_count=0U;if(!get_u64(bytes,p,count64)||count64>max_records||!get_u32(bytes,p,key_count)||key_count==0U||key_count>count64||key_count>1'000'000U)return false;if(p>=crc_pos||bytes[p++]!=kFields)return false;if(bytes.size()-p<kFields)return false;for(auto kind:kSchema)if(bytes[p++]!=kind)return false;std::uint32_t bits=0U;std::uint64_t order_bits=0U;if(!get_u32(bytes,p,bits)||!get_u64(bytes,p,order_bits)||bits!=key_bits(key_count)||order_bits!=count64*bits)return false;
    std::vector<Key> keys;std::vector<std::vector<Record>> rows;keys.reserve(key_count);rows.resize(key_count);std::uint64_t total_rows=0U;
    for(std::uint32_t i=0U;i<key_count;++i){std::uint32_t entity=0U,series=0U;std::uint64_t n=0U,len=0U;if(!get_u32(bytes,p,entity)||!get_u32(bytes,p,series)||!get_u64(bytes,p,n)||!get_u64(bytes,p,len)||n==0U||n>count64||p>crc_pos||len>crc_pos-p)return false;keys.emplace_back(entity,series);for(std::uint32_t j=0U;j<i;++j)if(keys[j]==keys.back())return false;std::vector<std::uint8_t> child(bytes.begin()+static_cast<std::ptrdiff_t>(p),bytes.begin()+static_cast<std::ptrdiff_t>(p+static_cast<std::size_t>(len)));p+=static_cast<std::size_t>(len);if(!decode_one_series(child,static_cast<std::size_t>(n),keys.back(),rows[i]))return false;total_rows+=n;}
    if(total_rows!=count64)return false;
    std::uint64_t schedule_len=0U;
    if(!get_u64(bytes,p,schedule_len)||p>crc_pos||schedule_len>crc_pos-p||p+schedule_len!=crc_pos)return false;
    std::vector<std::uint8_t> schedule(bytes.begin()+static_cast<std::ptrdiff_t>(p),bytes.begin()+static_cast<std::ptrdiff_t>(p+static_cast<std::size_t>(schedule_len)));
    if(schedule.size()!=(order_bits+7U)/8U)return false;
    if((order_bits&7U)!=0U&&!schedule.empty()){const auto mask=static_cast<std::uint8_t>((1U<<(8U-(order_bits&7U)))-1U);if((schedule.back()&mask)!=0U)return false;}
    BitReader reader{schedule,static_cast<std::size_t>(order_bits),0U};
    std::vector<std::size_t> cursors(key_count,0U);output.reserve(static_cast<std::size_t>(count64));
    for(std::uint64_t i=0U;i<count64;++i){std::uint32_t id=0U;if(!reader.get(bits,id)||id>=key_count||cursors[id]>=rows[id].size())return false;output.push_back(rows[id][cursors[id]++]);}
    if(reader.position!=reader.bits)return false;
    for(std::size_t i=0U;i<rows.size();++i)if(cursors[i]!=rows[i].size())return false;
    return true;
}
void write_file(const std::filesystem::path& path,const std::vector<std::uint8_t>& bytes){std::ofstream f(path,std::ios::binary);if(!f)throw std::runtime_error("open failed");f.write(reinterpret_cast<const char*>(bytes.data()),static_cast<std::streamsize>(bytes.size()));if(!f)throw std::runtime_error("write failed");}
void require(bool v,const std::string& m){if(!v)throw std::runtime_error(m);}
void test_roundtrips(){for(const std::string name:{"sparse","bursty","idle","noisy"}){const auto records=fixture(name,0x7878U);const auto frame=encode_partitioned(records);std::vector<Record> decoded;require(decode_partitioned(frame,records.size(),decoded),name+" partition decoder rejected frame");require(decoded==records,name+" partition order/data mismatch");}}
void test_randomized(){std::mt19937_64 rng(0x991122U);for(std::size_t trial=0U;trial<100U;++trial){std::vector<Record> records;std::array<Record,8U> states{};std::array<bool,8U> init{};const std::size_t n=static_cast<std::size_t>(rng()%300U);std::uint64_t time=0U;for(std::size_t i=0U;i<n;++i){const auto k=static_cast<std::size_t>(rng()%8U);time+=rng()%7U;Record r=init[k]?states[k]:Record{};r.timestamp=time;r.entity=static_cast<std::uint32_t>(k/4U);r.series=static_cast<std::uint32_t>(k%4U);if(!init[k])r.present=0xffU;for(std::size_t f=0U;f<kFields;++f){const auto bit=static_cast<std::uint8_t>(1U<<f);if(!init[k]||rng()%20U==0U){if(init[k]&&rng()%8U==0U)r.present=static_cast<std::uint8_t>(r.present^bit);r.values[f]=(r.present&bit)==0U?0U:value_for(f,rng());}}records.push_back(r);states[k]=r;init[k]=true;}const auto frame=encode_partitioned(records);std::vector<Record> decoded;require(decode_partitioned(frame,records.size(),decoded),"random partition decode rejected");require(decoded==records,"random partition mismatch");}}
void test_corruption(){const auto rows=fixture("sparse",0x8877U);auto frame=encode_partitioned(rows);std::vector<Record> out;auto bad=frame;bad[0]^=1U;require(!decode_partitioned(bad,rows.size(),out),"bad magic/CRC accepted");bad=frame;bad.pop_back();require(!decode_partitioned(bad,rows.size(),out),"truncated partition frame accepted");bad=frame;bad.push_back(0U);require(!decode_partitioned(bad,rows.size(),out),"trailing partition bytes accepted");}
void emit(const std::filesystem::path& dir){std::filesystem::create_directories(dir);for(const std::string name:{"sparse","bursty","idle","noisy"}){const auto rows=fixture(name,0x7e1e0001U);const auto raw=raw_bytes(rows);const auto delta=global_state_delta(rows);const auto partitioned=encode_partitioned(rows);std::vector<Record> decoded;require(decode_partitioned(partitioned,rows.size(),decoded)&&decoded==rows,name+" emission roundtrip failed");write_file(dir/(name+"_raw.bin"),raw);write_file(dir/(name+"_global_delta.bin"),delta);write_file(dir/(name+"_partitioned.bin"),partitioned);std::cout<<name<<",records="<<rows.size()<<",raw="<<raw.size()<<",global_delta="<<delta.size()<<",partitioned="<<partitioned.size()<<"\\n";}}
}
int main(int argc,char** argv){try{test_roundtrips();test_randomized();test_corruption();std::cout<<"fixture_roundtrips=4 randomized_roundtrips=100 corruption_rejections=3 status=PASS\\n";if(argc==3&&std::string(argv[1])=="--emit-dir")emit(argv[2]);else if(argc!=1){std::cerr<<"usage: "<<argv[0]<<" [--emit-dir DIR]\\n";return 2;}return 0;}catch(const std::exception& e){std::cerr<<"partitioned reference failed: "<<e.what()<<"\\n";return 1;}}
