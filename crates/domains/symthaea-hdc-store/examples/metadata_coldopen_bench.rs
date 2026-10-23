// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Synthetic mmap metadata-scan experiment, not a production store format.
//! Compare v2-like interleaved records with metadata-first SoA. On Linux,
//! collect process page-fault deltas and mincore residency snapshots. A cache
//! discard hint is advisory only and never establishes a guaranteed cold run.

use memmap2::Mmap;
use std::{
    env,
    error::Error,
    fs::{self, File, OpenOptions},
    hint::black_box,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::{Instant, SystemTime, UNIX_EPOCH},
};
#[cfg(target_os = "linux")]
use std::os::fd::AsRawFd;

const DATA: usize = 8192;
const V2_STRIDE: usize = 2080;
const V2_HV_OFFSET: usize = 32;
const SOA_META_STRIDE: usize = 16;
const HV_BYTES: usize = 2048;
const META_BYTES_READ: usize = 9;
const LIVE: u8 = 1;
const MAX_VECTORS: usize = 1_048_576;
const IO_BUF: usize = 64 * 1024;

#[derive(Debug)]
struct Config {
    vectors: usize,
    rounds: usize,
    seed: u64,
    csv: Option<PathBuf>,
    keep: bool,
}
impl Default for Config {
    fn default() -> Self {
        Self { vectors: 16_384, rounds: 10, seed: 7, csv: None, keep: false }
    }
}
struct Workspace { path: PathBuf, keep: bool }
impl Drop for Workspace {
    fn drop(&mut self) {
        if !self.keep { let _ = fs::remove_dir_all(&self.path); }
    }
}
#[derive(Clone, Copy)]
struct Faults { minor: u64, major: u64 }
struct Ranges { meta: (usize, usize), payload: (usize, usize) }
struct Sample {
    layout: &'static str,
    round: usize,
    vectors: usize,
    elapsed_ns: u128,
    file_bytes: u64,
    page_size: usize,
    advice_rc: Option<i32>,
    minor: Option<u64>,
    major: Option<u64>,
    meta_resident_before: Option<usize>,
    meta_resident_after: Option<usize>,
    payload_resident_before: Option<usize>,
    payload_resident_after: Option<usize>,
    checksum: u64,
}

fn main() {
    if let Err(e) = run() {
        eprintln!("metadata_coldopen_bench: {e}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let c = parse_args()?;
    if c.vectors == 0 || c.vectors > MAX_VECTORS { return Err(format!("--vectors must be in 1..={MAX_VECTORS}").into()); }
    if c.rounds == 0 || c.rounds > 10_000 { return Err("--rounds must be in 1..=10000".into()); }
    let (page, page_source) = page_size();
    if page == 0 { return Err("invalid OS page size".into()); }

    let stamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let root = env::temp_dir().join(format!("symthaea-coldopen-{}-{stamp}", std::process::id()));
    if root.exists() { return Err(format!("workspace already exists: {}", root.display()).into()); }
    fs::create_dir(&root)?;
    let workspace = Workspace { path: root, keep: c.keep };
    let v2 = workspace.path.join("v2-interleaved.synthetic");
    let soa = workspace.path.join("soa-metadata-first.synthetic");
    if let Some(csv) = c.csv.as_ref() {
        let metadata_path = csv.with_extension("meta.txt");
        if csv.exists() || metadata_path.exists() {
            return Err(format!("refusing to overwrite CSV or metadata sidecar: {}", csv.display()).into());
        }
        if csv.starts_with(&workspace.path) { return Err("--csv must be outside the temporary workspace".into()); }
        if let Some(parent) = csv.parent() {
            if !parent.as_os_str().is_empty() && !parent.is_dir() {
                return Err(format!("CSV parent directory does not exist: {}", parent.display()).into());
            }
        }
    }

    let v2_ranges = write_v2(&v2, c.vectors, c.seed)?;
    let soa_ranges = write_soa(&soa, c.vectors, c.seed, page)?;
    let v2_bytes = fs::metadata(&v2)?.len();
    let soa_bytes = fs::metadata(&soa)?.len();
    let mut rows = Vec::with_capacity(c.rounds * 2);
    let mut expected = None;

    for round in 0..c.rounds {
        let order = if round % 2 == 0 {
            [("v2_interleaved", &v2, &v2_ranges), ("soa_metadata_first", &soa, &soa_ranges)]
        } else {
            [("soa_metadata_first", &soa, &soa_ranges), ("v2_interleaved", &v2, &v2_ranges)]
        };
        for (layout, path, ranges) in order {
            let row = measure(layout, round, path, ranges, c.vectors, page)?;
            if let Some(check) = expected {
                if check != row.checksum { return Err(format!("logical metadata checksum mismatch for {layout}").into()); }
            } else { expected = Some(row.checksum); }
            rows.push(row);
        }
    }

    println!("synthetic_metadata_scan=true; production_format_changed=false");
    println!("vectors={},rounds_per_layout={},seed={}", c.vectors, c.rounds, c.seed);
    println!("os={},arch={},kernel_release={},page_size_bytes={page},page_size_source={page_source}", env::consts::OS, env::consts::ARCH, kernel_release());
    println!("source_sha={}", env::var("SYMTHAEA_SOURCE_SHA").unwrap_or_else(|_| "not-injected".to_string()));
    println!("rustc_version={}", env::var("RUSTC_VERSION").unwrap_or_else(|_| "not-injected".to_string()));
    println!("host_profile={}", env::var("HOST_PROFILE").unwrap_or_else(|_| "not-injected".to_string()));
    println!("storage_profile={}", env::var("STORAGE_PROFILE").unwrap_or_else(|_| "not-injected".to_string()));
    println!("v2_file_bytes={v2_bytes},soa_file_bytes={soa_bytes}");
    println!("metadata_bytes_read_per_scan={}", c.vectors * META_BYTES_READ);
    println!("metadata_checksum={:016x}", expected.unwrap_or(0));
    println!("fault_source={}", if cfg!(target_os = "linux") { "/proc/self/stat" } else { "unavailable" });
    println!("residency_source={}", if cfg!(target_os = "linux") { "mincore snapshots (advisory, can race with reclaim)" } else { "unavailable" });
    println!("cache_hint=POSIX_FADV_DONTNEED where available; advisory only");
    println!("nonclaim=page faults/residency are not physical device I/O, cold-cache guarantees, energy, or NAND amplification");
    for layout in ["v2_interleaved", "soa_metadata_first"] {
        let mut times: Vec<u128> = rows.iter().filter(|r| r.layout == layout).map(|r| r.elapsed_ns).collect();
        times.sort_unstable();
        let selected: Vec<&Sample> = rows.iter().filter(|r| r.layout == layout).collect();
        println!(
            "summary layout={layout},n={},p50_ns={},p95_ns={},mean_major_faults={},mean_minor_faults={},mean_payload_resident_before={},mean_payload_resident_after={}",
            times.len(), percentile(&times,50), percentile(&times,95),
            mean(selected.iter().filter_map(|r| r.major)),
            mean(selected.iter().filter_map(|r| r.minor)),
            mean(selected.iter().filter_map(|r| r.payload_resident_before.map(|v| v as u64))),
            mean(selected.iter().filter_map(|r| r.payload_resident_after.map(|v| v as u64))),
        );
    }
    let csv = render_csv(&rows);
    if let Some(path) = c.csv.as_ref() {
        let metadata_path = path.with_extension("meta.txt");
        fs::write(path, &csv)?;
        let metadata = format!(
            "benchmark=metadata_coldopen_bench\nsource_sha={}\nrustc_version={}\nhost_os={}\nhost_arch={}\nkernel_release={}\npage_size_bytes={}\npage_size_source={}\nhost_profile={}\nstorage_profile={}\nvectors={}\nrounds_per_layout={}\nseed={}\nv2_file_bytes={}\nsoa_file_bytes={}\nmetadata_checksum={:016x}\ncache_hint=POSIX_FADV_DONTNEED_best_effort\nfault_counters={}\nmincore_residency={}\nphysical_device_io=not_measured\nenergy=not_measured\n",
            env::var("SYMTHAEA_SOURCE_SHA").unwrap_or_else(|_| "not-injected".to_string()),
            env::var("RUSTC_VERSION").unwrap_or_else(|_| "not-injected".to_string()),
            env::consts::OS,
            env::consts::ARCH,
            kernel_release(),
            page,
            page_source,
            env::var("HOST_PROFILE").unwrap_or_else(|_| "not-injected".to_string()),
            env::var("STORAGE_PROFILE").unwrap_or_else(|_| "not-injected".to_string()),
            c.vectors,
            c.rounds,
            c.seed,
            v2_bytes,
            soa_bytes,
            expected.unwrap_or(0),
            if cfg!(target_os = "linux") { "proc_self_stat" } else { "unavailable" },
            if cfg!(target_os = "linux") { "mincore" } else { "unavailable" },
        );
        fs::write(&metadata_path, metadata)?;
        eprintln!("raw_samples_csv={}", path.display());
        eprintln!("run_metadata={}", metadata_path.display());
    }
    println!("RAW_SAMPLE_CSV_BEGIN");
    print!("{csv}");
    println!("RAW_SAMPLE_CSV_END");
    if c.keep { println!("synthetic_files_preserved={}", workspace.path.display()); }
    Ok(())
}

fn write_v2(path: &Path, n: usize, seed: u64) -> Result<Ranges, Box<dyn Error>> {
    let file = OpenOptions::new().create_new(true).write(true).open(path)?;
    let mut out = BufWriter::with_capacity(IO_BUF, file);
    zeros(&mut out, DATA)?;
    let mut entry = [0u8; V2_STRIDE];
    for i in 0..n {
        let id = i as u64 + 1;
        entry.fill(0);
        entry[0] = LIVE;
        entry[1..9].copy_from_slice(&id.to_le_bytes());
        for j in 0..HV_BYTES { entry[V2_HV_OFFSET+j] = payload(seed,id,j); }
        out.write_all(&entry)?;
    }
    out.flush()?;
    out.get_ref().sync_all()?;
    let final_metadata_end = DATA
        .checked_add((n - 1).checked_mul(V2_STRIDE).ok_or("v2 metadata offset overflow")?)
        .and_then(|value| value.checked_add(META_BYTES_READ))
        .ok_or("v2 metadata offset overflow")?;
    Ok(Ranges {
        meta: (DATA, final_metadata_end),
        payload: (DATA + V2_HV_OFFSET, DATA + n * V2_STRIDE),
    })
}

fn write_soa(path: &Path, n: usize, seed: u64, page: usize) -> Result<Ranges, Box<dyn Error>> {
    let file = OpenOptions::new().create_new(true).write(true).open(path)?;
    let mut out = BufWriter::with_capacity(IO_BUF, file);
    zeros(&mut out, DATA)?;
    for i in 0..n {
        let mut meta = [0u8; SOA_META_STRIDE];
        meta[0] = LIVE;
        meta[1..9].copy_from_slice(&(i as u64 + 1).to_le_bytes());
        out.write_all(&meta)?;
    }
    let meta_end = DATA.checked_add(n.checked_mul(SOA_META_STRIDE).ok_or("metadata size overflow")?).ok_or("metadata size overflow")?;
    let payload_start = align_up(meta_end, page)?;
    zeros(&mut out, payload_start - meta_end)?;
    let mut hv = [0u8; HV_BYTES];
    for i in 0..n {
        let id = i as u64 + 1;
        for (j,b) in hv.iter_mut().enumerate() { *b = payload(seed,id,j); }
        out.write_all(&hv)?;
    }
    out.flush()?;
    out.get_ref().sync_all()?;
    let payload_end = payload_start.checked_add(n.checked_mul(HV_BYTES).ok_or("payload size overflow")?).ok_or("payload size overflow")?;
    Ok(Ranges { meta: (DATA, meta_end), payload: (payload_start, payload_end) })
}

fn measure(layout: &'static str, round: usize, path: &Path, ranges: &Ranges, n: usize, page: usize) -> Result<Sample, Box<dyn Error>> {
    let file = OpenOptions::new().read(true).open(path)?;
    let advice_rc = discard_hint(&file);
    // SAFETY: the file is open read-only and remains alive for the mapping's lifetime.
    let map = unsafe { Mmap::map(&file)? };
    let before_res = residency(&map, page);
    let before_faults = faults();
    let start = Instant::now();
    let checksum = scan(&map, layout, n)?;
    let elapsed_ns = start.elapsed().as_nanos();
    let after_faults = faults();
    let after_res = residency(&map, page);
    black_box(checksum);
    let (minor,major) = match (before_faults,after_faults) {
        (Some(a),Some(b)) => (Some(b.minor.saturating_sub(a.minor)),Some(b.major.saturating_sub(a.major))),
        _ => (None,None),
    };
    Ok(Sample {
        layout, round, vectors: n, elapsed_ns, file_bytes: file.metadata()?.len(), page_size: page, advice_rc,
        minor, major,
        meta_resident_before: before_res.as_deref().map(|v| resident_in(v,ranges.meta,page)),
        meta_resident_after: after_res.as_deref().map(|v| resident_in(v,ranges.meta,page)),
        payload_resident_before: before_res.as_deref().map(|v| resident_in(v,ranges.payload,page)),
        payload_resident_after: after_res.as_deref().map(|v| resident_in(v,ranges.payload,page)),
        checksum,
    })
}

fn scan(map: &Mmap, layout: &str, n: usize) -> Result<u64, Box<dyn Error>> {
    let stride = if layout == "v2_interleaved" { V2_STRIDE } else { SOA_META_STRIDE };
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for i in 0..n {
        let off = DATA.checked_add(i.checked_mul(stride).ok_or("scan offset overflow")?).ok_or("scan offset overflow")?;
        let status = *map.get(off).ok_or("status outside mapping")?;
        if status != LIVE { return Err(format!("invalid status at row {i}").into()); }
        let end = off.checked_add(META_BYTES_READ).ok_or("ID range overflow")?;
        let id_bytes: [u8;8] = map.get(off+1..end).ok_or("ID outside mapping")?.try_into()?;
        let id = u64::from_le_bytes(id_bytes);
        if id != i as u64 + 1 { return Err(format!("ID mismatch at row {i}").into()); }
        hash ^= status as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        for b in id_bytes { hash ^= b as u64; hash = hash.wrapping_mul(0x0000_0100_0000_01b3); }
    }
    Ok(hash)
}

fn payload(seed: u64, id: u64, index: usize) -> u8 {
    let x = seed.wrapping_add(id.wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .wrapping_add((index as u64).wrapping_mul(0xA076_1D64_78BD_642F));
    (x ^ (x >> 23) ^ (x >> 41)) as u8
}
fn zeros(w: &mut impl Write, mut n: usize) -> Result<(), Box<dyn Error>> {
    const BUF: [u8; IO_BUF] = [0; IO_BUF];
    while n > 0 { let k = n.min(BUF.len()); w.write_all(&BUF[..k])?; n -= k; }
    Ok(())
}
fn align_up(n: usize, a: usize) -> Result<usize, Box<dyn Error>> {
    if a == 0 { return Err("page size is zero".into()); }
    let r = n % a;
    if r == 0 { Ok(n) } else { n.checked_add(a-r).ok_or_else(|| "alignment overflow".into()) }
}
fn page_size() -> (usize, &'static str) {
    #[cfg(target_os = "linux")]
    {
        unsafe extern "C" { fn getpagesize() -> i32; }
        // SAFETY: getpagesize has no arguments.
        let n = unsafe { getpagesize() };
        if n > 0 { return (n as usize, "getpagesize"); }
    }
    (4096, "fallback_4096_assumed")
}
fn kernel_release() -> String {
    #[cfg(target_os = "linux")]
    {
        return fs::read_to_string("/proc/sys/kernel/osrelease")
            .map(|value| value.trim().to_string())
            .unwrap_or_else(|_| "unavailable".to_string());
    }
    #[cfg(not(target_os = "linux"))]
    {
        "unavailable".to_string()
    }
}

fn faults() -> Option<Faults> {
    #[cfg(target_os = "linux")]
    {
        let s = fs::read_to_string("/proc/self/stat").ok()?;
        let end = s.rfind(')')?;
        let f: Vec<&str> = s.get(end+2..)?.split_whitespace().collect();
        Some(Faults { minor: f.get(7)?.parse().ok()?, major: f.get(9)?.parse().ok()? })
    }
    #[cfg(not(target_os = "linux"))]
    { None }
}
fn residency(map: &Mmap, page: usize) -> Option<Vec<u8>> {
    #[cfg(target_os = "linux")]
    {
        use std::ffi::{c_int,c_void};
        unsafe extern "C" { fn mincore(addr: *mut c_void, len: usize, vec: *mut u8) -> c_int; }
        let mut v = vec![0; map.len().div_ceil(page)];
        // SAFETY: mmap base is page-aligned and v has one byte per covered page.
        let rc = unsafe { mincore(map.as_ptr() as *mut c_void, map.len(), v.as_mut_ptr()) };
        if rc == 0 { Some(v) } else { None }
    }
    #[cfg(not(target_os = "linux"))]
    { let _ = (map,page); None }
}
fn discard_hint(file: &File) -> Option<i32> {
    #[cfg(target_os = "linux")]
    {
        use std::ffi::c_int;
        unsafe extern "C" { fn posix_fadvise(fd:c_int,offset:i64,len:i64,advice:c_int)->c_int; }
        // SAFETY: valid open file descriptor; DONTNEED is advisory, not a cold-cache guarantee.
        Some(unsafe { posix_fadvise(file.as_raw_fd(),0,0,4) })
    }
    #[cfg(not(target_os = "linux"))]
    { let _ = file; None }
}
fn resident_in(bitmap: &[u8], range: (usize,usize), page: usize) -> usize {
    if range.1 <= range.0 || page == 0 { return 0; }
    let first = range.0 / page;
    let end = range.1.div_ceil(page).min(bitmap.len());
    bitmap.get(first..end).unwrap_or(&[]).iter().filter(|b| (**b & 1) != 0).count()
}
fn percentile(sorted:&[u128],p:usize)->u128 {
    if sorted.is_empty(){return 0}
    let rank=(sorted.len()*p).div_ceil(100).max(1);
    sorted[rank.saturating_sub(1).min(sorted.len()-1)]
}
fn mean(values: impl Iterator<Item=u64>) -> String {
    let v:Vec<u64>=values.collect();
    if v.is_empty(){"unavailable".into()}else{format!("{:.3}",v.iter().sum::<u64>() as f64/v.len() as f64)}
}
fn render_csv(rows:&[Sample])->String {
    let mut s=String::from("layout,round,elapsed_ns,vectors,logical_metadata_bytes_read,file_bytes,page_size_bytes,eviction_hint_rc,minor_fault_delta,major_fault_delta,metadata_resident_pages_before,metadata_resident_pages_after,payload_resident_pages_before,payload_resident_pages_after,metadata_checksum_fnv1a64\n");
    for r in rows {
        s.push_str(&format!("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:016x}\n",
            r.layout,r.round,r.elapsed_ns,r.vectors,r.vectors * META_BYTES_READ,r.file_bytes,r.page_size,show(r.advice_rc),show(r.minor),show(r.major),
            show(r.meta_resident_before),show(r.meta_resident_after),show(r.payload_resident_before),show(r.payload_resident_after),r.checksum));
    }
    s
}
fn show<T:ToString>(v:Option<T>)->String{v.map(|x|x.to_string()).unwrap_or_default()}
fn parse_args()->Result<Config,Box<dyn Error>>{
    let mut c=Config::default(); let mut a=env::args().skip(1);
    while let Some(k)=a.next(){match k.as_str(){
        "--vectors"=>c.vectors=number("--vectors",a.next())?,
        "--rounds"=>c.rounds=number("--rounds",a.next())?,
        "--seed"=>c.seed=number("--seed",a.next())?,
        "--csv"=>c.csv=Some(PathBuf::from(a.next().ok_or("--csv requires a path")?)),
        "--keep-files"=>c.keep=true,
        "--help"|"-h"=>{println!("Usage: metadata_coldopen_bench [--vectors N] [--rounds N] [--seed N] [--csv PATH] [--keep-files]\nDefaults: vectors=16384, rounds=10, seed=7; use at least 30 rounds for published results.");std::process::exit(0)}
        _=>return Err(format!("unknown argument: {k}").into())
    }}
    Ok(c)
}
fn number<T:std::str::FromStr>(name:&str,v:Option<String>)->Result<T,Box<dyn Error>> where T::Err:Error+'static {
    Ok(v.ok_or_else(||format!("{name} requires a value"))?.parse()?)
}
