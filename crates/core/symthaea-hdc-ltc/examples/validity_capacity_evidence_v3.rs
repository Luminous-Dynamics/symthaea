// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Evidence-v3 seed-level summary supplement for validity-memory capacity.
//!
//! V3 does not rerun the capacity experiment. It verifies the retained v2 JSONL
//! hash chain, reconstructs exact per-seed observations from IEEE-754 bit
//! encodings, and derives the preregistered seed-level summary surface. Raw v2
//! records remain authoritative.

#[path = "support/evidence_sha256.rs"]
mod evidence_sha256;

use serde_json::{Map, Value, json};
use std::env;
use std::fs;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use symthaea_hdc_ltc::{
    SeedMetricSummary, ValidityCapacityAxis, ValidityCapacityCase,
    ValidityCapacityFalsificationObservation, ValidityCapacityFalsificationResult,
    ValidityCapacityPlan, ValidityCapacitySeedAggregateObservation,
    aggregate_validity_capacity_by_seed,
};

const V2_CHAIN_DOMAIN: &str = "symthaea:hdc-ltc:validity-capacity:evidence-chain:v2";
const V3_CHAIN_DOMAIN: &str = "symthaea:hdc-ltc:validity-capacity:evidence-chain:v3";
const V3_VERSION: &str = "hls-validity-capacity-evidence-v3";

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

    fn plan(self) -> ValidityCapacityPlan {
        match self {
            Self::Smoke => ValidityCapacityPlan::smoke(),
            Self::ResearchV0 => ValidityCapacityPlan::research_v0(),
        }
    }

    fn evidence_scope(self) -> &'static str {
        match self {
            Self::Smoke => "mechanical_qualification_only",
            Self::ResearchV0 => "exploratory_preregistered_measurement_summary",
        }
    }
}

struct VerifiedV2 {
    sha256: String,
    terminal_record_digest: String,
    observations: ValidityCapacityFalsificationResult,
}

fn main() -> Result<(), AnyError> {
    let (subject_sha, protocol, v2_path) = parse_args()?;
    let repo_root = fs::canonicalize(repository_root()?)?;
    require_clean_exact_subject(&repo_root, &subject_sha)?;

    let verified_v2 = verify_v2_evidence(&v2_path, &subject_sha, protocol)?;
    let plan = protocol.plan();
    let aggregated = aggregate_validity_capacity_by_seed(&plan, &verified_v2.observations)?;
    if aggregated.observations.len() != plan.cases.len() {
        return Err(other(format!(
            "seed aggregation produced {} cases, expected {}",
            aggregated.observations.len(),
            plan.cases.len()
        ))
        .into());
    }

    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    let mut sequence = 0_u64;
    let mut previous_digest = "0".repeat(64);

    emit_record(
        &mut out,
        &mut sequence,
        &mut previous_digest,
        "header",
        json!({
            "evidence_version": V3_VERSION,
            "chain_domain": V3_CHAIN_DOMAIN,
            "subject_commit_sha": subject_sha.clone(),
            "protocol": protocol.label(),
            "evidence_scope": protocol.evidence_scope(),
            "parent_evidence_version": "hls-validity-capacity-evidence-v2",
            "parent_v2_sha256": verified_v2.sha256,
            "parent_v2_terminal_record_digest": verified_v2.terminal_record_digest,
            "parent_v2_hash_chain_verified": true,
            "raw_per_seed_records_remain_authoritative": true,
            "analysis_unit": "preregistered_replicate_seed",
            "query_level_independence_claim": false,
            "p_values_emitted": false,
            "success_threshold_emitted": false,
            "case_count": plan.cases.len(),
            "seed_count_per_case": plan.replicate_seeds.len(),
        }),
    )?;

    for observation in &aggregated.observations {
        emit_record(
            &mut out,
            &mut sequence,
            &mut previous_digest,
            "seed_aggregate_observation",
            aggregate_value(observation),
        )?;
    }

    require_clean_exact_subject(&repo_root, &subject_sha)?;
    emit_record(
        &mut out,
        &mut sequence,
        &mut previous_digest,
        "footer",
        json!({
            "complete": true,
            "subject_unchanged": true,
            "checkout_clean": true,
            "aggregate_observation_count": aggregated.observations.len(),
            "interpretation": "not_performed_by_runner",
            "scientific_claim": "none",
        }),
    )?;

    out.flush()?;
    eprintln!("VALIDITY_CAPACITY_EVIDENCE_V3_TERMINAL_DIGEST={previous_digest}");
    Ok(())
}

fn parse_args() -> Result<(String, Protocol, PathBuf), io::Error> {
    let mut args = env::args().skip(1);
    let mut subject_sha = None;
    let mut protocol = None;
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
            "--v2-evidence" => {
                v2_path = Some(PathBuf::from(
                    args.next().ok_or_else(|| invalid("--v2-evidence requires a value"))?,
                ));
            }
            "-h" | "--help" => {
                println!(
                    "usage: validity_capacity_evidence_v3 --subject-sha <40-hex> \\\n                     --v2-evidence <v2-jsonl> [--protocol smoke|research-v0]"
                );
                std::process::exit(0);
            }
            other => return Err(invalid(format!("unexpected argument {other:?}"))),
        }
    }

    Ok((
        subject_sha.ok_or_else(|| invalid("--subject-sha is required"))?,
        protocol.unwrap_or(Protocol::Smoke),
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

fn verify_v2_evidence(
    path: &Path,
    subject_sha: &str,
    protocol: Protocol,
) -> Result<VerifiedV2, AnyError> {
    let bytes = fs::read(path)?;
    let text = std::str::from_utf8(&bytes)?;
    let mut expected_sequence = 0_u64;
    let mut expected_previous = "0".repeat(64);
    let mut observations = Vec::new();
    let mut terminal_record_digest = None;
    let mut final_kind = None;
    let mut final_payload = None;
    let mut saw_header = false;

    for line in text.lines().filter(|line| !line.trim().is_empty()) {
        let record: Value = serde_json::from_str(line)?;
        let sequence = required_u64(&record, "sequence")?;
        if sequence != expected_sequence {
            return Err(other(format!(
                "v2 sequence mismatch: expected {expected_sequence}, got {sequence}"
            ))
            .into());
        }
        let previous = required_str(&record, "previous_digest")?;
        if previous != expected_previous {
            return Err(other(format!(
                "v2 previous-digest mismatch at sequence {sequence}"
            ))
            .into());
        }
        let kind = required_str(&record, "kind")?;
        let payload = record
            .get("payload")
            .cloned()
            .ok_or_else(|| other("v2 payload missing"))?;
        let recorded_digest = required_str(&record, "record_digest")?;
        let envelope = json!({
            "sequence": sequence,
            "previous_digest": previous,
            "kind": kind,
            "payload": payload.clone(),
        });
        let expected_digest = digest_value(V2_CHAIN_DOMAIN, &envelope)?;
        if recorded_digest != expected_digest {
            return Err(other(format!(
                "v2 record digest mismatch at sequence {sequence}"
            ))
            .into());
        }

        if sequence == 0 {
            if kind != "header"
                || payload["evidence_version"].as_str()
                    != Some("hls-validity-capacity-evidence-v2")
                || payload["chain_domain"].as_str() != Some(V2_CHAIN_DOMAIN)
                || payload["subject_commit_sha"].as_str() != Some(subject_sha)
                || payload["protocol"].as_str() != Some(protocol.label())
            {
                return Err(other("v2 header does not match requested subject/protocol").into());
            }
            saw_header = true;
        } else if kind == "falsification_observation" {
            observations.push(parse_falsification_observation(&payload)?);
        }

        expected_previous = recorded_digest.to_owned();
        terminal_record_digest = Some(recorded_digest.to_owned());
        final_kind = Some(kind.to_owned());
        final_payload = Some(payload);
        expected_sequence = expected_sequence
            .checked_add(1)
            .ok_or_else(|| other("v2 sequence overflow"))?;
    }

    if !saw_header || final_kind.as_deref() != Some("footer") {
        return Err(other("v2 chain is missing a valid header/footer boundary").into());
    }
    let footer = final_payload.ok_or_else(|| other("v2 footer payload missing"))?;
    if footer["complete"].as_bool() != Some(true)
        || footer["subject_unchanged"].as_bool() != Some(true)
        || footer["checkout_clean"].as_bool() != Some(true)
        || footer["scientific_claim"].as_str() != Some("none")
        || footer["falsification_observation_count"].as_u64()
            != Some(observations.len() as u64)
    {
        return Err(other("v2 footer did not close the evidence theorem").into());
    }

    Ok(VerifiedV2 {
        sha256: evidence_sha256::sha256_hex(&bytes),
        terminal_record_digest: terminal_record_digest
            .ok_or_else(|| other("v2 chain contained no records"))?,
        observations: ValidityCapacityFalsificationResult { observations },
    })
}

fn parse_falsification_observation(
    payload: &Value,
) -> Result<ValidityCapacityFalsificationObservation, AnyError> {
    let case_value = payload
        .get("case")
        .ok_or_else(|| other("v2 falsification case missing"))?;
    let case = ValidityCapacityCase {
        axis: parse_axis(required_str(case_value, "axis")?)?,
        dim: required_usize(case_value, "dim")?,
        key_count: required_usize(case_value, "key_count")?,
        candidate_count: required_usize(case_value, "candidate_count")?,
        horizon: required_u64(case_value, "horizon")?,
        span_length: required_u64(case_value, "span_length")?,
    };

    Ok(ValidityCapacityFalsificationObservation {
        case,
        seed: required_u64(payload, "seed")?,
        total_queries: required_u64(payload, "total_queries")?,
        empirical_accuracy: float_bits(&payload["accuracy"]["empirical"])?,
        predicted_accuracy: float_bits(&payload["accuracy"]["predicted_null"])?,
        accuracy_residual: float_bits(&payload["accuracy"]["empirical_minus_null"])?,
        accuracy_integration_refinement_delta: float_bits(
            &payload["accuracy"]["integration_refinement_delta"],
        )?,
        target_mean_bias: float_bits(&payload["target_null_residuals"]["mean_bias"])?,
        target_mean_squared_residual: float_bits(
            &payload["target_null_residuals"]["mean_squared_residual"],
        )?,
        target_mse_ratio_to_null: float_bits(
            &payload["target_null_residuals"]["mse_ratio_to_null"],
        )?,
        vocabulary_distractor_mean_bias: float_bits(
            &payload["vocabulary_distractor"]["mean_bias"],
        )?,
        vocabulary_distractor_mean_squared_residual: float_bits(
            &payload["vocabulary_distractor"]["mean_squared_residual"],
        )?,
        vocabulary_distractor_mse_ratio_to_null: float_bits(
            &payload["vocabulary_distractor"]["mse_ratio_to_null"],
        )?,
        shadow_distractor_mean_bias: float_bits(
            &payload["matched_never_written_shadow"]["mean_bias"],
        )?,
        shadow_distractor_mean_squared_residual: float_bits(
            &payload["matched_never_written_shadow"]["mean_squared_residual"],
        )?,
        shadow_distractor_mse_ratio_to_null: float_bits(
            &payload["matched_never_written_shadow"]["mse_ratio_to_null"],
        )?,
        max_abs_shadow_candidate_similarity: float_bits(
            &payload["matched_never_written_shadow"]["max_abs_shadow_candidate_similarity"],
        )?,
        vocabulary_minus_shadow_mse: float_bits(
            &payload["paired_residuals"]["vocabulary_minus_shadow_mse"],
        )?,
        vocabulary_minus_shadow_mse_ratio: float_bits(
            &payload["paired_residuals"]["vocabulary_minus_shadow_mse_ratio"],
        )?,
        vocabulary_minus_shadow_variance: float_bits(
            &payload["paired_residuals"]["vocabulary_minus_shadow_variance"],
        )?,
        mean_true_margin: float_bits(&payload["signed_true_margin"]["mean"])?,
        smallest_true_margin: float_bits(&payload["signed_true_margin"]["minimum"])?,
    })
}

fn parse_axis(value: &str) -> Result<ValidityCapacityAxis, AnyError> {
    match value {
        "smoke" => Ok(ValidityCapacityAxis::Smoke),
        "dimension" => Ok(ValidityCapacityAxis::Dimension),
        "key_count" => Ok(ValidityCapacityAxis::KeyCount),
        "candidate_count" => Ok(ValidityCapacityAxis::CandidateCount),
        "horizon" => Ok(ValidityCapacityAxis::Horizon),
        "span_length" => Ok(ValidityCapacityAxis::SpanLength),
        unknown => Err(other(format!("unknown v2 capacity axis {unknown:?}")).into()),
    }
}

fn float_bits(value: &Value) -> Result<f64, AnyError> {
    let bits = value
        .get("bits")
        .and_then(Value::as_str)
        .ok_or_else(|| other("float bit encoding missing"))?;
    let hex = bits
        .strip_prefix("0x")
        .ok_or_else(|| other("float bit encoding must start with 0x"))?;
    let raw = u64::from_str_radix(hex, 16)?;
    let decoded = f64::from_bits(raw);
    if !decoded.is_finite() {
        return Err(other("non-finite float in v2 evidence").into());
    }
    Ok(decoded)
}

fn aggregate_value(observation: &ValidityCapacitySeedAggregateObservation) -> Value {
    json!({
        "case": case_value(observation.case),
        "seed_count": observation.seed_count,
        "accuracy_residual": metric_value(observation.accuracy_residual),
        "target_mean_bias": metric_value(observation.target_mean_bias),
        "target_mse_ratio_to_null": metric_value(observation.target_mse_ratio_to_null),
        "vocabulary_distractor_mean_bias": metric_value(observation.vocabulary_distractor_mean_bias),
        "vocabulary_distractor_mse_ratio_to_null": metric_value(observation.vocabulary_distractor_mse_ratio_to_null),
        "shadow_distractor_mean_bias": metric_value(observation.shadow_distractor_mean_bias),
        "shadow_distractor_mse_ratio_to_null": metric_value(observation.shadow_distractor_mse_ratio_to_null),
        "vocabulary_minus_shadow_mse": metric_value(observation.vocabulary_minus_shadow_mse),
        "vocabulary_minus_shadow_variance": metric_value(observation.vocabulary_minus_shadow_variance),
        "mean_true_margin": metric_value(observation.mean_true_margin),
    })
}

fn metric_value(summary: SeedMetricSummary) -> Value {
    json!({
        "seed_count": summary.seed_count,
        "mean": float_value(summary.mean),
        "median": float_value(summary.median),
        "sample_standard_deviation": float_value(summary.sample_standard_deviation),
        "minimum": float_value(summary.minimum),
        "maximum": float_value(summary.maximum),
        "positive_seed_count": summary.positive_seed_count,
        "negative_seed_count": summary.negative_seed_count,
        "zero_seed_count": summary.zero_seed_count,
    })
}

fn case_value(case: ValidityCapacityCase) -> Value {
    json!({
        "axis": match case.axis {
            ValidityCapacityAxis::Smoke => "smoke",
            ValidityCapacityAxis::Dimension => "dimension",
            ValidityCapacityAxis::KeyCount => "key_count",
            ValidityCapacityAxis::CandidateCount => "candidate_count",
            ValidityCapacityAxis::Horizon => "horizon",
            ValidityCapacityAxis::SpanLength => "span_length",
        },
        "dim": case.dim,
        "key_count": case.key_count,
        "candidate_count": case.candidate_count,
        "horizon": case.horizon,
        "span_length": case.span_length,
    })
}

fn float_value(value: f64) -> Value {
    json!({
        "decimal": format!("{value:.17e}"),
        "bits": format!("0x{:016x}", value.to_bits()),
    })
}

fn emit_record<W: Write>(
    out: &mut W,
    sequence: &mut u64,
    previous_digest: &mut String,
    kind: &str,
    payload: Value,
) -> Result<(), AnyError> {
    let envelope = json!({
        "sequence": *sequence,
        "previous_digest": previous_digest.as_str(),
        "kind": kind,
        "payload": payload,
    });
    let digest = digest_value(V3_CHAIN_DOMAIN, &envelope)?;
    let record = json!({
        "sequence": *sequence,
        "previous_digest": previous_digest.as_str(),
        "kind": kind,
        "payload": envelope["payload"].clone(),
        "record_digest": digest,
    });
    out.write_all(&canonical_json_bytes(&record)?)?;
    out.write_all(b"\n")?;
    *previous_digest = required_str(&record, "record_digest")?.to_owned();
    *sequence = sequence
        .checked_add(1)
        .ok_or_else(|| other("v3 evidence sequence overflow"))?;
    Ok(())
}

fn digest_value(domain: &str, value: &Value) -> Result<String, AnyError> {
    let canonical = canonical_json_bytes(value)?;
    let mut bytes = Vec::with_capacity(domain.len() + 1 + canonical.len());
    bytes.extend_from_slice(domain.as_bytes());
    bytes.push(0);
    bytes.extend_from_slice(&canonical);
    Ok(evidence_sha256::sha256_hex(&bytes))
}

fn canonical_json_bytes(value: &Value) -> Result<Vec<u8>, serde_json::Error> {
    serde_json::to_vec(&canonicalize(value))
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

fn required_usize(value: &Value, field: &str) -> Result<usize, AnyError> {
    Ok(usize::try_from(required_u64(value, field)?)?)
}

fn repository_root() -> Result<PathBuf, AnyError> {
    let cwd = env::current_dir()?;
    let output = Command::new("git")
        .current_dir(cwd)
        .args(["rev-parse", "--show-toplevel"])
        .output()?;
    if !output.status.success() {
        return Err(other("git rev-parse --show-toplevel failed").into());
    }
    Ok(PathBuf::from(String::from_utf8(output.stdout)?.trim()))
}

fn require_clean_exact_subject(repo_root: &Path, subject_sha: &str) -> Result<(), AnyError> {
    let head = command_stdout(repo_root, &["rev-parse", "HEAD"])?;
    if head != subject_sha {
        return Err(other(format!(
            "subject mismatch: required {subject_sha}, checkout is {head}"
        ))
        .into());
    }
    let status = command_stdout(
        repo_root,
        &["status", "--porcelain=v1", "--untracked-files=all"],
    )?;
    if !status.is_empty() {
        return Err(other(format!(
            "v3 evidence requires a clean checkout; git status reported:\n{status}"
        ))
        .into());
    }
    Ok(())
}

fn command_stdout(repo_root: &Path, args: &[&str]) -> Result<String, AnyError> {
    let output = Command::new("git")
        .current_dir(repo_root)
        .args(args)
        .output()?;
    if !output.status.success() {
        return Err(other(format!(
            "git {:?} failed: {}",
            args,
            String::from_utf8_lossy(&output.stderr).trim()
        ))
        .into());
    }
    Ok(String::from_utf8(output.stdout)?.trim().to_owned())
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
            canonical_json_bytes(&left).unwrap(),
            canonical_json_bytes(&right).unwrap()
        );
    }

    #[test]
    fn float_bits_round_trip_exactly() {
        for value in [-1.25_f64, -0.0, 0.0, 0.125, 1.0, 1234.5] {
            let encoded = float_value(value);
            assert_eq!(float_bits(&encoded).unwrap().to_bits(), value.to_bits());
        }
    }
}
