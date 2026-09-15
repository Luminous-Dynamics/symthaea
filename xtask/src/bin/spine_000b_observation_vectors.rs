use anyhow::{bail, Context, Result};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;

const EXECUTION_V2_DOMAIN: &[u8] = b"symthaea.spine.000b.execution-receipt.v2\0";
const GUARD_EVENT_DOMAIN: &[u8] = b"symthaea.spine.000b.guard-witness-event.v1\0";
const GUARD_BUNDLE_DOMAIN: &[u8] = b"symthaea.spine.000b.guard-witness-bundle.v1\0";
const OBS_CYCLE_DOMAIN: &[u8] = b"symthaea.spine.000b.observation-cycle.v1\0";
const OBS_GENESIS_DOMAIN: &[u8] = b"symthaea.spine.000b.observation-genesis.v1\0";
const OBS_CHAIN_DOMAIN: &[u8] = b"symthaea.spine.000b.observation-chain-link.v1\0";

fn push_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_bool(out: &mut Vec<u8>, value: bool) {
    out.push(u8::from(value));
}

fn push_identity(out: &mut Vec<u8>, value: &str) -> Result<()> {
    if value.is_empty() || value.len() > 128 || !value.is_ascii() {
        bail!("invalid identity length/ascii: {value:?}");
    }
    if !value
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b':' | b'-'))
    {
        bail!("invalid identity character: {value:?}");
    }
    push_u16(out, u16::try_from(value.len())?);
    out.extend_from_slice(value.as_bytes());
    Ok(())
}

fn push_path(out: &mut Vec<u8>, value: &str) -> Result<()> {
    if value.is_empty() || value.len() > 512 || !value.is_ascii() || value.starts_with('/') {
        bail!("invalid path: {value:?}");
    }
    if value.contains('\\')
        || !value.bytes().all(|b| {
            b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'/' | b'-')
        })
    {
        bail!("invalid path character: {value:?}");
    }
    if value.split('/').any(|part| part.is_empty() || part == "." || part == "..") {
        bail!("invalid path segment: {value:?}");
    }
    push_u16(out, u16::try_from(value.len())?);
    out.extend_from_slice(value.as_bytes());
    Ok(())
}

fn sha256(domain: &[u8], bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(bytes);
    hasher.finalize().into()
}

fn hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

fn proposal_v2(reserved: u32) -> Vec<u8> {
    let mut out = Vec::new();
    push_u64(&mut out, 0.0_f64.to_bits());
    push_u64(&mut out, 1.0_f64.to_bits());
    push_u64(&mut out, 0.0_f64.to_bits());
    push_u32(&mut out, 0.0_f32.to_bits());
    push_u32(&mut out, 0.0_f32.to_bits());
    push_u32(&mut out, 0);
    push_u32(&mut out, reserved);
    out
}

#[allow(clippy::too_many_arguments)]
fn execution_bytes(
    identity: &str,
    source_path: &str,
    eligible: bool,
    outcome_tag: u8,
    emitted: bool,
    admitted: bool,
    proposal: Option<&[u8]>,
) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    push_u64(&mut out, 42);
    push_identity(&mut out, identity)?;
    push_path(&mut out, source_path)?;
    push_u32(&mut out, 1);
    out.push(1); // NORMAL
    push_bool(&mut out, eligible);
    out.push(outcome_tag);
    push_bool(&mut out, emitted);
    push_bool(&mut out, admitted);
    match proposal {
        None => out.push(0),
        Some(bytes) => {
            out.push(1);
            out.extend_from_slice(bytes);
        }
    }
    push_u16(&mut out, 0); // no overlap refs in golden samples
    Ok(out)
}

fn guard_event_bytes(index: u16, predicate_id: u16, outcome: bool) -> Vec<u8> {
    let mut out = Vec::new();
    push_u64(&mut out, 42);
    push_u16(&mut out, index);
    push_u16(&mut out, predicate_id);
    push_bool(&mut out, outcome);
    out
}

fn guard_bundle_bytes(events: &[(u16, u16, bool)], overflow: bool) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    push_u64(&mut out, 42);
    push_bool(&mut out, overflow);
    push_u16(&mut out, u16::try_from(events.len())?);
    for (expected, &(index, predicate_id, outcome)) in events.iter().enumerate() {
        if index != u16::try_from(expected)? {
            bail!("guard indices must be contiguous");
        }
        let event = guard_event_bytes(index, predicate_id, outcome);
        let digest = sha256(GUARD_EVENT_DOMAIN, &event);
        push_u16(&mut out, index);
        out.extend_from_slice(&digest);
    }
    Ok(out)
}

fn observation_bytes(
    drive_digest: &[u8; 32],
    memory_digest: &[u8; 32],
    guard_bundle_digest: &[u8; 32],
    manager_overflow: bool,
) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    push_u64(&mut out, 42);
    push_u32(&mut out, 2);

    // Canonical execution identity order: drive_manager, memory_manager.
    push_identity(&mut out, "drive_manager")?;
    out.extend_from_slice(drive_digest);
    push_identity(&mut out, "memory_manager")?;
    out.extend_from_slice(memory_digest);

    out.extend_from_slice(&[0x11; 32]); // integration digest
    push_u32(&mut out, 2);
    push_u32(&mut out, 0);
    out.extend_from_slice(&[0x22; 32]);
    push_u32(&mut out, 1);
    out.extend_from_slice(&[0x33; 32]);
    out.extend_from_slice(guard_bundle_digest);

    push_bool(&mut out, manager_overflow);
    push_bool(&mut out, false); // application overflow
    push_bool(&mut out, false); // guard overflow
    push_bool(&mut out, !manager_overflow); // observer_buffers_complete
    Ok(out)
}

fn expect_str(fixture: &Value, key: &str, actual: String) -> Result<()> {
    let expected = fixture
        .get(key)
        .and_then(Value::as_str)
        .with_context(|| format!("fixture key missing/not string: {key}"))?;
    if expected != actual {
        bail!("vector mismatch for {key}\nexpected={expected}\nactual  ={actual}");
    }
    Ok(())
}

fn main() -> Result<()> {
    let fixture_path = Path::new("tests/fixtures/spine_000b_observation_commitment_v2_vectors.json");
    let fixture: Value = serde_json::from_str(
        &fs::read_to_string(fixture_path)
            .with_context(|| format!("read {}", fixture_path.display()))?,
    )?;

    let neutral_prop = proposal_v2(0);
    let reserved_prop = proposal_v2(1);
    let neutral = execution_bytes(
        "drive_manager",
        "src/cognitive_loop/managers/drive_manager.rs",
        true,
        2,
        true,
        false,
        Some(&neutral_prop),
    )?;
    let reserved = execution_bytes(
        "drive_manager",
        "src/cognitive_loop/managers/drive_manager.rs",
        true,
        2,
        true,
        false,
        Some(&reserved_prop),
    )?;
    let memory = execution_bytes(
        "memory_manager",
        "src/cognitive_loop/managers/memory_manager.rs",
        false,
        0,
        false,
        false,
        None,
    )?;

    let neutral_digest = sha256(EXECUTION_V2_DOMAIN, &neutral);
    let reserved_digest = sha256(EXECUTION_V2_DOMAIN, &reserved);
    let memory_digest = sha256(EXECUTION_V2_DOMAIN, &memory);

    let guard_true = guard_event_bytes(0, 1, true);
    let guard_false = guard_event_bytes(0, 1, false);
    let guard_true_digest = sha256(GUARD_EVENT_DOMAIN, &guard_true);
    let guard_false_digest = sha256(GUARD_EVENT_DOMAIN, &guard_false);

    let guard_bundle = guard_bundle_bytes(&[(0, 1, true), (1, 2, true), (2, 3, false)], false)?;
    let guard_bundle_digest = sha256(GUARD_BUNDLE_DOMAIN, &guard_bundle);

    let observation = observation_bytes(
        &neutral_digest,
        &memory_digest,
        &guard_bundle_digest,
        false,
    )?;
    let observation_digest = sha256(OBS_CYCLE_DOMAIN, &observation);
    let overflow_observation = observation_bytes(
        &neutral_digest,
        &memory_digest,
        &guard_bundle_digest,
        true,
    )?;
    let overflow_observation_digest = sha256(OBS_CYCLE_DOMAIN, &overflow_observation);

    let manifest = [0x44_u8; 32];
    let genesis = sha256(OBS_GENESIS_DOMAIN, &manifest);
    let mut link_input = Vec::with_capacity(64);
    link_input.extend_from_slice(&genesis);
    link_input.extend_from_slice(&observation_digest);
    let root1 = sha256(OBS_CHAIN_DOMAIN, &link_input);

    expect_str(&fixture, "neutral_execution_bytes_hex", hex(&neutral))?;
    expect_str(&fixture, "neutral_execution_digest_hex", hex(&neutral_digest))?;
    expect_str(&fixture, "reserved_execution_bytes_hex", hex(&reserved))?;
    expect_str(&fixture, "reserved_execution_digest_hex", hex(&reserved_digest))?;
    expect_str(&fixture, "guard_true_bytes_hex", hex(&guard_true))?;
    expect_str(&fixture, "guard_true_digest_hex", hex(&guard_true_digest))?;
    expect_str(&fixture, "guard_false_bytes_hex", hex(&guard_false))?;
    expect_str(&fixture, "guard_false_digest_hex", hex(&guard_false_digest))?;
    expect_str(&fixture, "guard_bundle_bytes_hex", hex(&guard_bundle))?;
    expect_str(&fixture, "guard_bundle_digest_hex", hex(&guard_bundle_digest))?;
    expect_str(&fixture, "observation_cycle_bytes_hex", hex(&observation))?;
    expect_str(&fixture, "observation_cycle_digest_hex", hex(&observation_digest))?;
    expect_str(
        &fixture,
        "overflow_observation_cycle_bytes_hex",
        hex(&overflow_observation),
    )?;
    expect_str(
        &fixture,
        "overflow_observation_cycle_digest_hex",
        hex(&overflow_observation_digest),
    )?;
    expect_str(&fixture, "observation_genesis_root_hex", hex(&genesis))?;
    expect_str(&fixture, "observation_chain_root_1_hex", hex(&root1))?;

    if neutral == reserved || neutral_digest == reserved_digest {
        bail!("reserved-only control did not change execution identity");
    }
    if guard_true_digest == guard_false_digest {
        bail!("guard TRUE/FALSE control unexpectedly identical");
    }
    if observation_digest == overflow_observation_digest {
        bail!("observer overflow control unexpectedly identical");
    }

    println!("SPINE-000B-C2 independent Rust vector reproduction: PASS");
    println!("authority=measurement-only");
    println!("runtime_evidence_claimed=false");
    println!("causal_load_claimed=false");
    Ok(())
}
