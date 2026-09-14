// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Independent structural verifier for retained validity-capacity evidence.
//!
//! This executable runs no experiment. It verifies the retained v1/v2 artifact
//! pair as a complete lineage before v3 is allowed to summarize it:
//!
//! - every v1 and v2 canonical SHA-256 chain link;
//! - exact record grammar and observation counts for the requested protocol;
//! - exact subject/protocol headers and complete fail-closed footers;
//! - v2 source/bundle/base-verification theorem flags;
//! - v2's base commitment against the *actual retained v1 bytes*, terminal
//!   digest, record count, and capacity/control observation counts.

#[path = "support/evidence_sha256.rs"]
mod evidence_sha256;

use serde_json::{Map, Value, json};
use std::env;
use std::fs;
use std::io;
use std::path::PathBuf;
use symthaea_hdc_ltc::{ValidityCapacityControlPlan, ValidityCapacityPlan};

const V1_CHAIN_DOMAIN: &str = "symthaea:hdc-ltc:validity-capacity:evidence-chain:v1";
const V2_CHAIN_DOMAIN: &str = "symthaea:hdc-ltc:validity-capacity:evidence-chain:v2";

type AnyError = Box<dyn std::error::Error + Send + Sync + 'static>;

#[derive(Debug, Clone, Copy)]
enum Protocol {
    Smoke,
    ResearchV0,
}

impl Protocol {
    fn parse(value: &str) -> Result<Self, io::Error> {
        match value {
            "smoke" => Ok(Self::Smoke),
            "research-v0" => Ok(Self::ResearchV0),
            other => Err(invalid(format!(
                "unknown protocol {other:?}; expected smoke or research-v0"
            ))),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Smoke => "smoke",
            Self::ResearchV0 => "research-v0",
        }
    }

    fn capacity_plan(self) -> ValidityCapacityPlan {
        match self {
            Self::Smoke => ValidityCapacityPlan::smoke(),
            Self::ResearchV0 => ValidityCapacityPlan::research_v0(),
        }
    }

    fn control_plan(self) -> ValidityCapacityControlPlan {
        match self {
            Self::Smoke => ValidityCapacityControlPlan::smoke(),
            Self::ResearchV0 => ValidityCapacityControlPlan::research_v0(),
        }
    }
}

#[derive(Debug)]
struct VerifiedV1 {
    sha256: String,
    terminal_digest: String,
    record_count: u64,
    capacity_count: u64,
    control_count: u64,
}

fn main() -> Result<(), AnyError> {
    let (subject_sha, protocol, v1_path, v2_path) = parse_args()?;
    let capacity_plan = protocol.capacity_plan();
    let control_plan = protocol.control_plan();
    let expected_capacity = checked_product(capacity_plan.cases.len(), capacity_plan.replicate_seeds.len())?;
    let expected_controls = checked_product(control_plan.cases.len(), control_plan.replicate_seeds.len())?;
    let v1_bytes = fs::read(&v1_path)?;
    let v2_bytes = fs::read(&v2_path)?;

    let v1 = verify_v1(
        &v1_bytes,
        &subject_sha,
        protocol,
        expected_capacity,
        expected_controls,
    )?;
    let v2_terminal = verify_v2(
        &v2_bytes,
        &subject_sha,
        protocol,
        expected_capacity,
        &v1,
    )?;

    println!("EVIDENCE_CHAIN_PAIR_VERIFIED=true");
    println!("SUBJECT_SHA={subject_sha}");
    println!("PROTOCOL={}", protocol.label());
    println!("V1_SHA256={}", v1.sha256);
    println!("V1_TERMINAL_RECORD_DIGEST={}", v1.terminal_digest);
    println!("V1_RECORD_COUNT={}", v1.record_count);
    println!("V2_SHA256={}", evidence_sha256::sha256_hex(&v2_bytes));
    println!("V2_TERMINAL_RECORD_DIGEST={v2_terminal}");
    println!("CAPACITY_OBSERVATION_COUNT={expected_capacity}");
    println!("CONTROL_OBSERVATION_COUNT={expected_controls}");
    Ok(())
}

fn verify_v1(
    bytes: &[u8],
    subject_sha: &str,
    protocol: Protocol,
    expected_capacity: u64,
    expected_controls: u64,
) -> Result<VerifiedV1, AnyError> {
    let records = parse_and_verify_chain(bytes, V1_CHAIN_DOMAIN)?;
    let expected_record_count = 4_u64
        .checked_add(expected_capacity)
        .and_then(|value| value.checked_add(expected_controls))
        .ok_or_else(|| other("v1 expected record count overflow"))?;
    if records.len() as u64 != expected_record_count {
        return Err(other(format!(
            "v1 record count mismatch: expected {expected_record_count}, got {}",
            records.len()
        ))
        .into());
    }

    require_kind(&records, 0, "header")?;
    require_kind(&records, 1, "capacity_plan")?;
    require_kind(&records, 2, "control_plan")?;
    for index in 0..expected_capacity as usize {
        require_kind(&records, 3 + index, "capacity_observation")?;
    }
    let control_start = 3 + expected_capacity as usize;
    for index in 0..expected_controls as usize {
        require_kind(&records, control_start + index, "control_observation")?;
    }
    let footer_index = records.len() - 1;
    require_kind(&records, footer_index, "footer")?;

    let header = payload(&records[0])?;
    if header["evidence_version"].as_str() != Some("hls-validity-capacity-evidence-v1")
        || header["chain_domain"].as_str() != Some(V1_CHAIN_DOMAIN)
        || header["subject_commit_sha"].as_str() != Some(subject_sha)
        || header["protocol"].as_str() != Some(protocol.label())
    {
        return Err(other("v1 header does not match requested subject/protocol").into());
    }

    let footer = payload(&records[footer_index])?;
    if footer["complete"].as_bool() != Some(true)
        || footer["subject_unchanged"].as_bool() != Some(true)
        || footer["checkout_clean"].as_bool() != Some(true)
        || footer["primary_score_parity_verified"].as_bool() != Some(true)
        || footer["capacity_observation_count"].as_u64() != Some(expected_capacity)
        || footer["control_observation_count"].as_u64() != Some(expected_controls)
        || footer["total_observation_count"].as_u64()
            != expected_capacity.checked_add(expected_controls)
        || footer["scientific_claim"].as_str() != Some("none")
    {
        return Err(other("v1 footer did not close the declared evidence theorem").into());
    }

    Ok(VerifiedV1 {
        sha256: evidence_sha256::sha256_hex(bytes),
        terminal_digest: required_str(&records[footer_index], "record_digest")?.to_owned(),
        record_count: records.len() as u64,
        capacity_count: expected_capacity,
        control_count: expected_controls,
    })
}

fn verify_v2(
    bytes: &[u8],
    subject_sha: &str,
    protocol: Protocol,
    expected_falsification: u64,
    v1: &VerifiedV1,
) -> Result<String, AnyError> {
    let records = parse_and_verify_chain(bytes, V2_CHAIN_DOMAIN)?;
    let expected_record_count = 3_u64
        .checked_add(expected_falsification)
        .ok_or_else(|| other("v2 expected record count overflow"))?;
    if records.len() as u64 != expected_record_count {
        return Err(other(format!(
            "v2 record count mismatch: expected {expected_record_count}, got {}",
            records.len()
        ))
        .into());
    }

    require_kind(&records, 0, "header")?;
    require_kind(&records, 1, "base_evidence_commitment")?;
    for index in 0..expected_falsification as usize {
        require_kind(&records, 2 + index, "falsification_observation")?;
    }
    let footer_index = records.len() - 1;
    require_kind(&records, footer_index, "footer")?;

    let header = payload(&records[0])?;
    if header["evidence_version"].as_str() != Some("hls-validity-capacity-evidence-v2")
        || header["chain_domain"].as_str() != Some(V2_CHAIN_DOMAIN)
        || header["subject_commit_sha"].as_str() != Some(subject_sha)
        || header["protocol"].as_str() != Some(protocol.label())
        || header["source_bundle_verified"].as_bool() != Some(true)
        || header["new_lineage_required_on_source_or_lockfile_drift"].as_bool() != Some(true)
    {
        return Err(other("v2 header did not establish the expected provenance theorem").into());
    }

    let base = payload(&records[1])?;
    if base["sha256"].as_str() != Some(v1.sha256.as_str())
        || base["terminal_record_digest"].as_str() != Some(v1.terminal_digest.as_str())
        || base["record_count"].as_u64() != Some(v1.record_count)
        || base["capacity_observation_count"].as_u64() != Some(v1.capacity_count)
        || base["control_observation_count"].as_u64() != Some(v1.control_count)
        || base["hash_chain_verified"].as_bool() != Some(true)
        || base["fresh_isolated_build"].as_bool() != Some(true)
    {
        return Err(other("v2 base commitment does not match the retained verified v1 artifact").into());
    }

    let footer = payload(&records[footer_index])?;
    if footer["complete"].as_bool() != Some(true)
        || footer["subject_unchanged"].as_bool() != Some(true)
        || footer["checkout_clean"].as_bool() != Some(true)
        || footer["source_bundle_unchanged"].as_bool() != Some(true)
        || footer["source_bundle_verified"].as_bool() != Some(true)
        || footer["base_v1_hash_chain_verified"].as_bool() != Some(true)
        || footer["base_v1_fresh_isolated_build"].as_bool() != Some(true)
        || footer["falsification_observation_count"].as_u64() != Some(expected_falsification)
        || footer["scientific_claim"].as_str() != Some("none")
    {
        return Err(other("v2 footer did not close the declared evidence theorem").into());
    }

    Ok(required_str(&records[footer_index], "record_digest")?.to_owned())
}

fn parse_and_verify_chain(bytes: &[u8], domain: &str) -> Result<Vec<Value>, AnyError> {
    let text = std::str::from_utf8(bytes)?;
    let mut records = Vec::new();
    let mut expected_sequence = 0_u64;
    let mut expected_previous = "0".repeat(64);

    for line in text.lines().filter(|line| !line.trim().is_empty()) {
        let record: Value = serde_json::from_str(line)?;
        let sequence = required_u64(&record, "sequence")?;
        if sequence != expected_sequence {
            return Err(other(format!(
                "chain sequence mismatch: expected {expected_sequence}, got {sequence}"
            ))
            .into());
        }
        let previous = required_str(&record, "previous_digest")?;
        if previous != expected_previous {
            return Err(other(format!(
                "chain previous-digest mismatch at sequence {sequence}"
            ))
            .into());
        }
        let kind = required_str(&record, "kind")?;
        let payload = record
            .get("payload")
            .cloned()
            .ok_or_else(|| other("record payload missing"))?;
        let recorded_digest = required_str(&record, "record_digest")?;
        let envelope = json!({
            "sequence": sequence,
            "previous_digest": previous,
            "kind": kind,
            "payload": payload,
        });
        let expected_digest = digest_value(domain, &envelope)?;
        if recorded_digest != expected_digest {
            return Err(other(format!(
                "record digest mismatch at sequence {sequence}"
            ))
            .into());
        }

        expected_previous = recorded_digest.to_owned();
        expected_sequence = expected_sequence
            .checked_add(1)
            .ok_or_else(|| other("evidence sequence overflow"))?;
        records.push(record);
    }

    if records.is_empty() {
        return Err(other("evidence chain contained no records").into());
    }
    Ok(records)
}

fn require_kind(records: &[Value], index: usize, expected: &str) -> Result<(), AnyError> {
    let record = records
        .get(index)
        .ok_or_else(|| other(format!("missing record {index}, expected kind {expected}")))?;
    let actual = required_str(record, "kind")?;
    if actual != expected {
        return Err(other(format!(
            "record {index} kind mismatch: expected {expected}, got {actual}"
        ))
        .into());
    }
    Ok(())
}

fn payload(record: &Value) -> Result<&Value, AnyError> {
    record
        .get("payload")
        .ok_or_else(|| other("record payload missing").into())
}

fn digest_value(domain: &str, value: &Value) -> Result<String, AnyError> {
    let canonical = serde_json::to_vec(&canonicalize(value))?;
    let mut bytes = Vec::with_capacity(domain.len() + 1 + canonical.len());
    bytes.extend_from_slice(domain.as_bytes());
    bytes.push(0);
    bytes.extend_from_slice(&canonical);
    Ok(evidence_sha256::sha256_hex(&bytes))
}

fn canonicalize(value: &Value) -> Value {
    match value {
        Value::Array(values) => Value::Array(values.iter().map(canonicalize).collect()),
        Value::Object(map) => {
            let mut keys = map.keys().collect::<Vec<_>>();
            keys.sort();
            let mut sorted = Map::new();
            for key in keys {
                let nested = map
                    .get(key.as_str())
                    .expect("canonicalization key came from the same map");
                sorted.insert(key.to_string(), canonicalize(nested));
            }
            Value::Object(sorted)
        }
        _ => value.clone(),
    }
}

fn checked_product(left: usize, right: usize) -> Result<u64, AnyError> {
    let value = left
        .checked_mul(right)
        .ok_or_else(|| other("observation-count multiplication overflow"))?;
    Ok(u64::try_from(value)?)
}

fn required_str<'a>(value: &'a Value, field: &str) -> Result<&'a str, io::Error> {
    value
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| other(format!("required string field {field:?} missing")))
}

fn required_u64(value: &Value, field: &str) -> Result<u64, io::Error> {
    value
        .get(field)
        .and_then(Value::as_u64)
        .ok_or_else(|| other(format!("required integer field {field:?} missing")))
}

fn parse_args() -> Result<(String, Protocol, PathBuf, PathBuf), io::Error> {
    let mut args = env::args().skip(1);
    let mut subject_sha = None;
    let mut protocol = None;
    let mut v1_path = None;
    let mut v2_path = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--subject-sha" => {
                subject_sha = Some(validate_subject_sha(
                    &args.next().ok_or_else(|| invalid("--subject-sha requires a value"))?,
                )?);
            }
            "--protocol" => {
                protocol = Some(Protocol::parse(
                    &args.next().ok_or_else(|| invalid("--protocol requires a value"))?,
                )?);
            }
            "--v1-evidence" => {
                v1_path = Some(PathBuf::from(
                    args.next().ok_or_else(|| invalid("--v1-evidence requires a value"))?,
                ));
            }
            "--v2-evidence" => {
                v2_path = Some(PathBuf::from(
                    args.next().ok_or_else(|| invalid("--v2-evidence requires a value"))?,
                ));
            }
            "-h" | "--help" => {
                println!(
                    "usage: validity_capacity_evidence_chain_verify --subject-sha <40-hex> \\\n                     --v1-evidence <v1-jsonl> --v2-evidence <v2-jsonl> \\\n                     [--protocol smoke|research-v0]"
                );
                std::process::exit(0);
            }
            other => return Err(invalid(format!("unexpected argument {other:?}"))),
        }
    }

    Ok((
        subject_sha.ok_or_else(|| invalid("--subject-sha is required"))?,
        protocol.unwrap_or(Protocol::Smoke),
        v1_path.ok_or_else(|| invalid("--v1-evidence is required"))?,
        v2_path.ok_or_else(|| invalid("--v2-evidence is required"))?,
    ))
}

fn validate_subject_sha(value: &str) -> Result<String, io::Error> {
    if value.len() != 40 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(invalid(
            "subject SHA must be exactly 40 hexadecimal characters",
        ));
    }
    Ok(value.to_ascii_lowercase())
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn other(message: impl Into<String>) -> io::Error {
    io::Error::other(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_json_is_key_order_independent() {
        let left = json!({"b": 2, "a": {"d": 4, "c": 3}});
        let right = json!({"a": {"c": 3, "d": 4}, "b": 2});
        assert_eq!(
            serde_json::to_vec(&canonicalize(&left)).unwrap(),
            serde_json::to_vec(&canonicalize(&right)).unwrap()
        );
    }

    #[test]
    fn protocol_counts_match_frozen_smoke_surface() {
        let protocol = Protocol::Smoke;
        assert_eq!(
            checked_product(
                protocol.capacity_plan().cases.len(),
                protocol.capacity_plan().replicate_seeds.len()
            )
            .unwrap(),
            4
        );
        assert_eq!(
            checked_product(
                protocol.control_plan().cases.len(),
                protocol.control_plan().replicate_seeds.len()
            )
            .unwrap(),
            4
        );
    }
}
