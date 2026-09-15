// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Independent retained-artifact verifier for validity-capacity evidence-v3.
//!
//! This executable does not rerun the capacity experiment and does not call the
//! production seed-aggregation implementation. It verifies retained v2/v3 JSONL,
//! reconstructs the frozen per-seed metric vectors from v2 IEEE-754 encodings,
//! independently recomputes each v3 descriptive summary in frozen seed order,
//! and requires exact bit equality with the retained v3 artifact.

#[path = "support/evidence_sha256.rs"]
mod evidence_sha256;

use serde_json::{Map, Value};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::io;
use std::path::PathBuf;
use symthaea_hdc_ltc::{ValidityCapacityAxis, ValidityCapacityCase, ValidityCapacityPlan};

type AnyError = Box<dyn std::error::Error + Send + Sync + 'static>;

const V2_CHAIN_DOMAIN: &str = "symthaea:hdc-ltc:validity-capacity:evidence-chain:v2";
const V3_CHAIN_DOMAIN: &str = "symthaea:hdc-ltc:validity-capacity:evidence-chain:v3";
const V2_VERSION: &str = "hls-validity-capacity-evidence-v2";
const V3_VERSION: &str = "hls-validity-capacity-evidence-v3";

const METRICS: [&str; 10] = [
    "accuracy_residual",
    "target_mean_bias",
    "target_mse_ratio_to_null",
    "vocabulary_distractor_mean_bias",
    "vocabulary_distractor_mse_ratio_to_null",
    "shadow_distractor_mean_bias",
    "shadow_distractor_mse_ratio_to_null",
    "vocabulary_minus_shadow_mse",
    "vocabulary_minus_shadow_variance",
    "mean_true_margin",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
}

#[derive(Debug, Clone)]
struct ChainRecord {
    kind: String,
    payload: Value,
}

#[derive(Debug)]
struct VerifiedChain {
    records: Vec<ChainRecord>,
    terminal_digest: String,
    sha256: String,
}

#[derive(Debug, Clone, Copy)]
struct IndependentSummary {
    seed_count: usize,
    mean: f64,
    median: f64,
    sample_standard_deviation: f64,
    minimum: f64,
    maximum: f64,
    positive_seed_count: usize,
    negative_seed_count: usize,
    zero_seed_count: usize,
}

fn main() -> Result<(), AnyError> {
    let (subject_sha, protocol, v2_path, v3_path) = parse_args()?;
    let plan = protocol.plan();

    let v2_bytes = fs::read(v2_path)?;
    let v3_bytes = fs::read(v3_path)?;
    let v2 = verify_chain(&v2_bytes, V2_CHAIN_DOMAIN)?;
    let v3 = verify_chain(&v3_bytes, V3_CHAIN_DOMAIN)?;

    verify_v2_header_footer(&v2, &subject_sha, protocol, &plan)?;
    verify_v3_header_footer(&v3, &subject_sha, protocol, &plan, &v2)?;

    let raw = collect_v2_metric_vectors(&v2, &plan)?;
    verify_v3_summaries(&v3, &plan, &raw)?;

    println!("VALIDITY_CAPACITY_V3_INDEPENDENT_VERIFY=passed");
    println!("V2_SHA256={}", v2.sha256);
    println!("V3_SHA256={}", v3.sha256);
    println!("V3_SUMMARY_CASE_COUNT={}", plan.cases.len());
    println!("V3_SEED_COUNT_PER_CASE={}", plan.replicate_seeds.len());
    println!("SCIENTIFIC_CLAIM=none");
    Ok(())
}

fn parse_args() -> Result<(String, Protocol, PathBuf, PathBuf), io::Error> {
    let mut args = env::args().skip(1);
    let mut subject_sha = None;
    let mut protocol = None;
    let mut v2_path = None;
    let mut v3_path = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--subject-sha" => {
                let value = args
                    .next()
                    .ok_or_else(|| invalid("--subject-sha requires a value"))?;
                subject_sha = Some(validate_subject_sha(&value)?);
            }
            "--protocol" => {
                let value = args
                    .next()
                    .ok_or_else(|| invalid("--protocol requires a value"))?;
                protocol = Some(Protocol::parse(&value)?);
            }
            "--v2-evidence" => {
                let value = args
                    .next()
                    .ok_or_else(|| invalid("--v2-evidence requires a value"))?;
                v2_path = Some(PathBuf::from(value));
            }
            "--v3-evidence" => {
                let value = args
                    .next()
                    .ok_or_else(|| invalid("--v3-evidence requires a value"))?;
                v3_path = Some(PathBuf::from(value));
            }
            "-h" | "--help" => {
                println!(
                    "usage: validity_capacity_evidence_v3_verify --subject-sha <40-hex> \\\n                     --v2-evidence <v2-jsonl> --v3-evidence <v3-jsonl> \\\n                     [--protocol smoke|research-v0]"
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
        v3_path.ok_or_else(|| invalid("--v3-evidence is required"))?,
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

fn verify_chain(bytes: &[u8], domain: &str) -> Result<VerifiedChain, AnyError> {
    let text = std::str::from_utf8(bytes)?;
    let mut expected_sequence = 0_u64;
    let mut expected_previous = "0".repeat(64);
    let mut terminal_digest = None;
    let mut records = Vec::new();

    for line in text.lines().filter(|line| !line.trim().is_empty()) {
        let record: Value = serde_json::from_str(line)?;
        let sequence = required_u64(&record, "sequence")?;
        if sequence != expected_sequence {
            return Err(other(format!(
                "hash-chain sequence mismatch: expected {expected_sequence}, got {sequence}"
            ))
            .into());
        }
        let previous = required_str(&record, "previous_digest")?;
        if previous != expected_previous {
            return Err(other(format!(
                "hash-chain previous digest mismatch at sequence {sequence}"
            ))
            .into());
        }
        let kind = required_str(&record, "kind")?.to_owned();
        let payload = record
            .get("payload")
            .cloned()
            .ok_or_else(|| other("hash-chain payload missing"))?;
        let recorded_digest = required_str(&record, "record_digest")?;
        let envelope = serde_json::json!({
            "sequence": sequence,
            "previous_digest": previous,
            "kind": kind,
            "payload": payload.clone(),
        });
        let expected_digest = digest_value(domain, &envelope)?;
        if recorded_digest != expected_digest {
            return Err(other(format!(
                "hash-chain record digest mismatch at sequence {sequence}"
            ))
            .into());
        }

        expected_previous = recorded_digest.to_owned();
        terminal_digest = Some(recorded_digest.to_owned());
        records.push(ChainRecord { kind, payload });
        expected_sequence = expected_sequence
            .checked_add(1)
            .ok_or_else(|| other("hash-chain sequence overflow"))?;
    }

    if records.is_empty() {
        return Err(other("evidence chain contained no records").into());
    }

    Ok(VerifiedChain {
        records,
        terminal_digest: terminal_digest.expect("nonempty chain has terminal digest"),
        sha256: evidence_sha256::sha256_hex(bytes),
    })
}

fn verify_v2_header_footer(
    v2: &VerifiedChain,
    subject_sha: &str,
    protocol: Protocol,
    plan: &ValidityCapacityPlan,
) -> Result<(), AnyError> {
    let header = v2
        .records
        .first()
        .ok_or_else(|| other("v2 header missing"))?;
    if header.kind != "header"
        || header.payload["evidence_version"].as_str() != Some(V2_VERSION)
        || header.payload["chain_domain"].as_str() != Some(V2_CHAIN_DOMAIN)
        || header.payload["subject_commit_sha"].as_str() != Some(subject_sha)
        || header.payload["protocol"].as_str() != Some(protocol.label())
    {
        return Err(other("v2 header identity mismatch").into());
    }

    let expected_observations = plan
        .cases
        .len()
        .checked_mul(plan.replicate_seeds.len())
        .ok_or_else(|| other("v2 expected observation count overflow"))?;
    let observed = v2
        .records
        .iter()
        .filter(|record| record.kind == "falsification_observation")
        .count();
    if observed != expected_observations {
        return Err(other(format!(
            "v2 falsification count mismatch: actual={observed}, expected={expected_observations}"
        ))
        .into());
    }

    let footer = v2
        .records
        .last()
        .ok_or_else(|| other("v2 footer missing"))?;
    if footer.kind != "footer"
        || footer.payload["complete"].as_bool() != Some(true)
        || footer.payload["subject_unchanged"].as_bool() != Some(true)
        || footer.payload["checkout_clean"].as_bool() != Some(true)
        || footer.payload["scientific_claim"].as_str() != Some("none")
        || footer.payload["falsification_observation_count"].as_u64()
            != Some(expected_observations as u64)
    {
        return Err(other("v2 footer did not close the expected evidence contract").into());
    }
    Ok(())
}

fn verify_v3_header_footer(
    v3: &VerifiedChain,
    subject_sha: &str,
    protocol: Protocol,
    plan: &ValidityCapacityPlan,
    v2: &VerifiedChain,
) -> Result<(), AnyError> {
    let header = v3
        .records
        .first()
        .ok_or_else(|| other("v3 header missing"))?;
    if header.kind != "header"
        || header.payload["evidence_version"].as_str() != Some(V3_VERSION)
        || header.payload["chain_domain"].as_str() != Some(V3_CHAIN_DOMAIN)
        || header.payload["subject_commit_sha"].as_str() != Some(subject_sha)
        || header.payload["protocol"].as_str() != Some(protocol.label())
        || header.payload["parent_evidence_version"].as_str() != Some(V2_VERSION)
        || header.payload["parent_v2_sha256"].as_str() != Some(v2.sha256.as_str())
        || header.payload["parent_v2_terminal_record_digest"].as_str()
            != Some(v2.terminal_digest.as_str())
        || header.payload["parent_v2_hash_chain_verified"].as_bool() != Some(true)
        || header.payload["raw_per_seed_records_remain_authoritative"].as_bool() != Some(true)
        || header.payload["analysis_unit"].as_str() != Some("preregistered_replicate_seed")
        || header.payload["query_level_independence_claim"].as_bool() != Some(false)
        || header.payload["p_values_emitted"].as_bool() != Some(false)
        || header.payload["success_threshold_emitted"].as_bool() != Some(false)
        || header.payload["case_count"].as_u64() != Some(plan.cases.len() as u64)
        || header.payload["seed_count_per_case"].as_u64()
            != Some(plan.replicate_seeds.len() as u64)
    {
        return Err(other("v3 header identity or parent-linkage mismatch").into());
    }

    let observed = v3
        .records
        .iter()
        .filter(|record| record.kind == "seed_aggregate_observation")
        .count();
    if observed != plan.cases.len() {
        return Err(other(format!(
            "v3 aggregate count mismatch: actual={observed}, expected={}",
            plan.cases.len()
        ))
        .into());
    }

    let footer = v3
        .records
        .last()
        .ok_or_else(|| other("v3 footer missing"))?;
    if footer.kind != "footer"
        || footer.payload["complete"].as_bool() != Some(true)
        || footer.payload["subject_unchanged"].as_bool() != Some(true)
        || footer.payload["checkout_clean"].as_bool() != Some(true)
        || footer.payload["scientific_claim"].as_str() != Some("none")
        || footer.payload["aggregate_observation_count"].as_u64()
            != Some(plan.cases.len() as u64)
    {
        return Err(other("v3 footer did not close the expected evidence contract").into());
    }
    Ok(())
}

fn collect_v2_metric_vectors(
    v2: &VerifiedChain,
    plan: &ValidityCapacityPlan,
) -> Result<HashMap<(ValidityCapacityCase, u64), [f64; 10]>, AnyError> {
    let observations = v2
        .records
        .iter()
        .filter(|record| record.kind == "falsification_observation")
        .collect::<Vec<_>>();

    let mut expected_index = 0_usize;
    let mut result = HashMap::with_capacity(observations.len());
    for &case in &plan.cases {
        for &seed in &plan.replicate_seeds {
            let record = observations
                .get(expected_index)
                .ok_or_else(|| other("v2 observation order ended early"))?;
            let observed_case = parse_case(
                record
                    .payload
                    .get("case")
                    .ok_or_else(|| other("v2 observation case missing"))?,
            )?;
            let observed_seed = required_u64(&record.payload, "seed")?;
            if observed_case != case || observed_seed != seed {
                return Err(other(format!(
                    "v2 ordered case/seed mismatch at index {expected_index}: actual=({observed_case:?},{observed_seed}), expected=({case:?},{seed})"
                ))
                .into());
            }

            let metrics = [
                float_value_at(&record.payload, &["accuracy", "empirical_minus_null"] )?,
                float_value_at(&record.payload, &["target_null_residuals", "mean_bias"] )?,
                float_value_at(&record.payload, &["target_null_residuals", "mse_ratio_to_null"] )?,
                float_value_at(&record.payload, &["vocabulary_distractor", "mean_bias"] )?,
                float_value_at(&record.payload, &["vocabulary_distractor", "mse_ratio_to_null"] )?,
                float_value_at(&record.payload, &["matched_never_written_shadow", "mean_bias"] )?,
                float_value_at(&record.payload, &["matched_never_written_shadow", "mse_ratio_to_null"] )?,
                float_value_at(&record.payload, &["paired_residuals", "vocabulary_minus_shadow_mse"] )?,
                float_value_at(&record.payload, &["paired_residuals", "vocabulary_minus_shadow_variance"] )?,
                float_value_at(&record.payload, &["signed_true_margin", "mean"] )?,
            ];
            if result.insert((case, seed), metrics).is_some() {
                return Err(other(format!(
                    "duplicate v2 case/seed while rebuilding summaries: case={case:?}, seed={seed}"
                ))
                .into());
            }
            expected_index += 1;
        }
    }

    if expected_index != observations.len() {
        return Err(other(format!(
            "v2 observation order has trailing records: consumed={expected_index}, actual={}",
            observations.len()
        ))
        .into());
    }
    Ok(result)
}

fn verify_v3_summaries(
    v3: &VerifiedChain,
    plan: &ValidityCapacityPlan,
    raw: &HashMap<(ValidityCapacityCase, u64), [f64; 10]>,
) -> Result<(), AnyError> {
    let summaries = v3
        .records
        .iter()
        .filter(|record| record.kind == "seed_aggregate_observation")
        .collect::<Vec<_>>();

    for (case_index, &case) in plan.cases.iter().enumerate() {
        let record = summaries
            .get(case_index)
            .ok_or_else(|| other("v3 summary order ended early"))?;
        if record.payload.as_object().map(Map::len) != Some(12) {
            return Err(other(format!(
                "v3 summary payload shape drifted at case index {case_index}"
            ))
            .into());
        }
        let observed_case = parse_case(
            record
                .payload
                .get("case")
                .ok_or_else(|| other("v3 summary case missing"))?,
        )?;
        if observed_case != case {
            return Err(other(format!(
                "v3 case order mismatch at index {case_index}: actual={observed_case:?}, expected={case:?}"
            ))
            .into());
        }
        if required_u64(&record.payload, "seed_count")? != plan.replicate_seeds.len() as u64 {
            return Err(other(format!(
                "v3 top-level seed count mismatch for case={case:?}"
            ))
            .into());
        }

        for (metric_index, metric_name) in METRICS.iter().enumerate() {
            let mut values = Vec::with_capacity(plan.replicate_seeds.len());
            for &seed in &plan.replicate_seeds {
                let metrics = raw.get(&(case, seed)).ok_or_else(|| {
                    other(format!(
                        "missing v2 source metric for case={case:?}, seed={seed}"
                    ))
                })?;
                values.push(metrics[metric_index]);
            }
            let expected = independent_summary(&values);
            let actual = record
                .payload
                .get(*metric_name)
                .ok_or_else(|| other(format!("v3 metric {metric_name:?} missing")))?;
            verify_summary_object(actual, expected, case, metric_name)?;
        }
    }

    if summaries.len() != plan.cases.len() {
        return Err(other("v3 summary order has trailing records").into());
    }
    Ok(())
}

fn independent_summary(values: &[f64]) -> IndependentSummary {
    assert!(!values.is_empty());
    let seed_count = values.len();

    let mut total = 0.0_f64;
    for &value in values {
        total += value;
    }
    let mean = total / seed_count as f64;

    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| left.total_cmp(right));
    let median = if seed_count % 2 == 1 {
        sorted[seed_count / 2]
    } else {
        let right = seed_count / 2;
        0.5 * (sorted[right - 1] + sorted[right])
    };

    let sample_standard_deviation = if seed_count > 1 {
        let mut squared_deviation_sum = 0.0_f64;
        for &value in values {
            let delta = value - mean;
            squared_deviation_sum += delta * delta;
        }
        (squared_deviation_sum / (seed_count - 1) as f64).sqrt()
    } else {
        0.0
    };

    let mut positive_seed_count = 0_usize;
    let mut negative_seed_count = 0_usize;
    let mut zero_seed_count = 0_usize;
    for &value in values {
        if value > 0.0 {
            positive_seed_count += 1;
        } else if value < 0.0 {
            negative_seed_count += 1;
        } else {
            zero_seed_count += 1;
        }
    }

    IndependentSummary {
        seed_count,
        mean,
        median,
        sample_standard_deviation,
        minimum: sorted[0],
        maximum: sorted[seed_count - 1],
        positive_seed_count,
        negative_seed_count,
        zero_seed_count,
    }
}

fn verify_summary_object(
    value: &Value,
    expected: IndependentSummary,
    case: ValidityCapacityCase,
    metric_name: &str,
) -> Result<(), AnyError> {
    if value.as_object().map(Map::len) != Some(9) {
        return Err(other(format!(
            "v3 metric summary shape drifted: case={case:?}, metric={metric_name}"
        ))
        .into());
    }

    require_count(value, "seed_count", expected.seed_count, case, metric_name)?;
    require_float_bits(value, "mean", expected.mean, case, metric_name)?;
    require_float_bits(value, "median", expected.median, case, metric_name)?;
    require_float_bits(
        value,
        "sample_standard_deviation",
        expected.sample_standard_deviation,
        case,
        metric_name,
    )?;
    require_float_bits(value, "minimum", expected.minimum, case, metric_name)?;
    require_float_bits(value, "maximum", expected.maximum, case, metric_name)?;
    require_count(
        value,
        "positive_seed_count",
        expected.positive_seed_count,
        case,
        metric_name,
    )?;
    require_count(
        value,
        "negative_seed_count",
        expected.negative_seed_count,
        case,
        metric_name,
    )?;
    require_count(
        value,
        "zero_seed_count",
        expected.zero_seed_count,
        case,
        metric_name,
    )?;
    Ok(())
}

fn require_count(
    value: &Value,
    field: &str,
    expected: usize,
    case: ValidityCapacityCase,
    metric_name: &str,
) -> Result<(), AnyError> {
    let actual = required_u64(value, field)?;
    if actual != expected as u64 {
        return Err(other(format!(
            "v3 count mismatch: case={case:?}, metric={metric_name}, field={field}, actual={actual}, expected={expected}"
        ))
        .into());
    }
    Ok(())
}

fn require_float_bits(
    value: &Value,
    field: &str,
    expected: f64,
    case: ValidityCapacityCase,
    metric_name: &str,
) -> Result<(), AnyError> {
    let actual = float_value_at(value, &[field])?;
    if actual.to_bits() != expected.to_bits() {
        return Err(other(format!(
            "v3 floating summary mismatch: case={case:?}, metric={metric_name}, field={field}, actual_bits=0x{:016x}, expected_bits=0x{:016x}",
            actual.to_bits(),
            expected.to_bits()
        ))
        .into());
    }
    Ok(())
}

fn float_value_at(root: &Value, path: &[&str]) -> Result<f64, AnyError> {
    let mut value = root;
    for key in path {
        value = value
            .get(*key)
            .ok_or_else(|| other(format!("missing float path component {key:?}")))?;
    }
    decode_float(value)
}

fn decode_float(value: &Value) -> Result<f64, AnyError> {
    if value.as_object().map(Map::len) != Some(2) {
        return Err(other("float encoding must contain exactly decimal and bits").into());
    }
    let bits = required_str(value, "bits")?;
    let hex = bits
        .strip_prefix("0x")
        .ok_or_else(|| other("float bit encoding must start with 0x"))?;
    if hex.len() != 16 || !hex.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(other("float bit encoding must contain exactly 16 hex digits").into());
    }
    let raw = u64::from_str_radix(hex, 16)?;
    let decoded = f64::from_bits(raw);
    if !decoded.is_finite() {
        return Err(other("non-finite float in retained evidence").into());
    }

    let decimal = required_str(value, "decimal")?.parse::<f64>()?;
    if decimal.to_bits() != raw {
        return Err(other(format!(
            "decimal/bit float encoding mismatch: decimal={decimal:?}, bits={bits}"
        ))
        .into());
    }
    Ok(decoded)
}

fn parse_case(value: &Value) -> Result<ValidityCapacityCase, AnyError> {
    Ok(ValidityCapacityCase {
        axis: parse_axis(required_str(value, "axis")?)?,
        dim: required_usize(value, "dim")?,
        key_count: required_usize(value, "key_count")?,
        candidate_count: required_usize(value, "candidate_count")?,
        horizon: required_u64(value, "horizon")?,
        span_length: required_u64(value, "span_length")?,
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
        other => Err(other(format!("unknown validity-capacity axis {other:?}")).into()),
    }
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
                sorted.insert(key.to_owned(), canonicalize(nested));
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
    fn independent_arithmetic_matches_frozen_two_seed_reference_bits() {
        let values = [
            f64::from_bits(0x3fb0_4f07_0cb4_7138),
            f64::from_bits(0xbfaf_61f1_e697_1d90),
        ];
        let summary = independent_summary(&values);
        assert_eq!(summary.mean.to_bits(), 0x3f53_c1c3_2d1c_4e00);
        assert_eq!(summary.median.to_bits(), 0x3f53_c1c3_2d1c_4e00);
        assert_eq!(
            summary.sample_standard_deviation.to_bits(),
            0x3fb6_a09e_667f_3bcd
        );
        assert_eq!(summary.minimum.to_bits(), 0xbfaf_61f1_e697_1d90);
        assert_eq!(summary.maximum.to_bits(), 0x3fb0_4f07_0cb4_7138);
        assert_eq!(summary.positive_seed_count, 1);
        assert_eq!(summary.negative_seed_count, 1);
        assert_eq!(summary.zero_seed_count, 0);
    }

    #[test]
    fn decimal_and_bits_must_encode_the_same_f64() {
        let valid = serde_json::json!({
            "decimal": "1.00000000000000000e0",
            "bits": "0x3ff0000000000000"
        });
        assert_eq!(decode_float(&valid).unwrap().to_bits(), 1.0_f64.to_bits());

        let invalid = serde_json::json!({
            "decimal": "1.00000000000000000e0",
            "bits": "0x3fe0000000000000"
        });
        assert!(decode_float(&invalid).is_err());
    }

    #[test]
    fn single_seed_sample_standard_deviation_is_zero() {
        let summary = independent_summary(&[3.5]);
        assert_eq!(summary.sample_standard_deviation.to_bits(), 0.0_f64.to_bits());
        assert_eq!(summary.median.to_bits(), 3.5_f64.to_bits());
    }
}
