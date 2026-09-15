use anyhow::{bail, Context, Result};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs;

const DOMAIN: &[u8] = b"symthaea.spine.000b.subject-manifest.v1\0";

#[derive(Clone)]
struct BoundFile {
    path: String,
    hash: [u8; 32],
}

#[derive(Clone)]
struct Seed {
    name: String,
    value: u64,
}

#[derive(Clone)]
struct Manifest {
    repository: String,
    git_head: String,
    git_tree: String,
    runtime_profile_id: String,
    target_triple: String,
    rustc_version: String,
    cargo_version: String,
    default_features: bool,
    features: Vec<String>,
    bound_files: Vec<BoundFile>,
    protocol_ids: Vec<String>,
    workload_id: String,
    workload_digest: [u8; 32],
    seeds: Vec<Seed>,
    cycle_start: u64,
    cycle_count: u64,
    stopping_rule_id: String,
    manager_capacity: u16,
    application_capacity: u16,
    guard_capacity: u16,
    qualification_policy_digest: [u8; 32],
}

fn put_u8(out: &mut Vec<u8>, v: u8) { out.push(v); }
fn put_u16(out: &mut Vec<u8>, v: u16) { out.extend_from_slice(&v.to_le_bytes()); }
fn put_u64(out: &mut Vec<u8>, v: u64) { out.extend_from_slice(&v.to_le_bytes()); }
fn put_text(out: &mut Vec<u8>, s: &str) -> Result<()> {
    let len: u16 = s.len().try_into().context("string too long for u16")?;
    put_u16(out, len);
    out.extend_from_slice(s.as_bytes());
    Ok(())
}

fn encode(m: &Manifest) -> Result<Vec<u8>> {
    if m.cycle_count == 0 || m.manager_capacity == 0 || m.application_capacity == 0 || m.guard_capacity == 0 {
        bail!("invalid zero campaign bound/capacity");
    }
    let mut out = DOMAIN.to_vec();
    put_text(&mut out, &m.repository)?;
    put_text(&mut out, &m.git_head)?;
    put_text(&mut out, &m.git_tree)?;
    put_u8(&mut out, 1); // clean_worktree_required
    put_text(&mut out, &m.runtime_profile_id)?;
    put_text(&mut out, &m.target_triple)?;
    put_text(&mut out, &m.rustc_version)?;
    put_text(&mut out, &m.cargo_version)?;
    put_u8(&mut out, u8::from(m.default_features));

    let mut features = m.features.clone();
    features.sort();
    features.dedup();
    put_u16(&mut out, features.len().try_into()?);
    for f in features { put_text(&mut out, &f)?; }

    let mut files = m.bound_files.clone();
    files.sort_by(|a, b| a.path.cmp(&b.path));
    put_u16(&mut out, files.len().try_into()?);
    for f in files {
        put_text(&mut out, &f.path)?;
        out.extend_from_slice(&f.hash);
    }

    let mut protocols = m.protocol_ids.clone();
    protocols.sort();
    protocols.dedup();
    put_u16(&mut out, protocols.len().try_into()?);
    for p in protocols { put_text(&mut out, &p)?; }

    put_text(&mut out, &m.workload_id)?;
    out.extend_from_slice(&m.workload_digest);

    let mut seeds = m.seeds.clone();
    seeds.sort_by(|a, b| a.name.cmp(&b.name));
    put_u16(&mut out, seeds.len().try_into()?);
    for s in seeds {
        put_text(&mut out, &s.name)?;
        put_u64(&mut out, s.value);
    }

    put_u64(&mut out, m.cycle_start);
    put_u64(&mut out, m.cycle_count);
    put_text(&mut out, &m.stopping_rule_id)?;
    put_u16(&mut out, m.manager_capacity);
    put_u16(&mut out, m.application_capacity);
    put_u16(&mut out, m.guard_capacity);
    out.extend_from_slice(&m.qualification_policy_digest);
    Ok(out)
}

fn repeated(byte: u8) -> [u8; 32] { [byte; 32] }

fn sample() -> Manifest {
    Manifest {
        repository: "Luminous-Dynamics/symthaea".into(),
        git_head: "11".repeat(20),
        git_tree: "22".repeat(20),
        runtime_profile_id: "spine-runtime-v1".into(),
        target_triple: "x86_64-unknown-linux-gnu".into(),
        rustc_version: "rustc 1.96.0 (sample)".into(),
        cargo_version: "cargo 1.96.0 (sample)".into(),
        default_features: false,
        features: vec!["vision-manifold".into(), "swarm".into()],
        bound_files: vec![
            BoundFile { path: "Cargo.lock".into(), hash: repeated(0x33) },
            BoundFile { path: "rust-toolchain.toml".into(), hash: repeated(0x44) },
            BoundFile { path: "src/cognitive_loop/subsystem_trait.rs".into(), hash: repeated(0x55) },
        ],
        protocol_ids: vec!["C2".into(), "G1".into(), "N1".into(), "N2".into(), "P1R".into(), "R2".into()],
        workload_id: "synthetic-spine-workload-v1".into(),
        workload_digest: repeated(0x66),
        seeds: vec![Seed { name: "genesis".into(), value: 42 }, Seed { name: "workload".into(), value: 7 }],
        cycle_start: 100,
        cycle_count: 256,
        stopping_rule_id: "fixed-cycle-count-v1".into(),
        manager_capacity: 64,
        application_capacity: 32,
        guard_capacity: 32,
        qualification_policy_digest: repeated(0x77),
    }
}

fn to_hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes { s.push_str(&format!("{b:02x}")); }
    s
}

fn sha_hex(bytes: &[u8]) -> String {
    to_hex(&Sha256::digest(bytes))
}

fn check_case(fixture: &Value, name: &str, m: &Manifest) -> Result<()> {
    let bytes = encode(m)?;
    let expected_hex = fixture[name]["bytes_hex"].as_str().context("missing bytes_hex")?;
    let expected_sha = fixture[name]["sha256"].as_str().context("missing sha256")?;
    if to_hex(&bytes) != expected_hex { bail!("{name}: canonical bytes mismatch"); }
    if sha_hex(&bytes) != expected_sha { bail!("{name}: sha256 mismatch"); }
    Ok(())
}

fn main() -> Result<()> {
    let fixture_text = fs::read_to_string("tests/fixtures/spine_000b_subject_manifest_v1_vectors.json")?;
    let fixture: Value = serde_json::from_str(&fixture_text)?;

    let base = sample();
    check_case(&fixture, "base", &base)?;

    let mut reordered = base.clone();
    reordered.features.reverse();
    reordered.bound_files.reverse();
    reordered.protocol_ids.reverse();
    reordered.seeds.reverse();
    check_case(&fixture, "reordered", &reordered)?;

    let mut capacity = base.clone();
    capacity.application_capacity = 33;
    check_case(&fixture, "mutated_capacity", &capacity)?;

    let mut source_hash = base.clone();
    source_hash.bound_files[0].hash = repeated(0x88);
    check_case(&fixture, "mutated_hash", &source_hash)?;

    let base_bytes = encode(&base)?;
    let reordered_bytes = encode(&reordered)?;
    if base_bytes != reordered_bytes { bail!("canonical reordering mismatch"); }
    if base_bytes == encode(&capacity)? { bail!("capacity mutation did not change bytes"); }
    if base_bytes == encode(&source_hash)? { bail!("source-hash mutation did not change bytes"); }

    println!("SPINE-000B-M1 independent Rust vectors: PASS");
    println!("base_sha256={}", sha_hex(&base_bytes));
    println!("canonical_reordering=true");
    println!("authority=measurement-only");
    println!("runtime_evidence_claimed=false");
    println!("causal_load_claimed=false");
    Ok(())
}
