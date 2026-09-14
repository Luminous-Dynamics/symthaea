// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Evidence-v2 wrapper for the validity-memory falsification program.
//!
//! This executable preserves the already-frozen v1 evidence contract rather than
//! replacing it. It:
//!
//! 1. requires an exact subject commit and clean checkout;
//! 2. proves that its own binary was compiled from the same critical source bundle
//!    currently present in that checkout, using compile-time `include_bytes!`;
//! 3. fresh-builds and executes the v1 evidence runner in a new target directory;
//! 4. verifies every record in the returned v1 SHA-256 hash chain;
//! 5. writes that base artifact outside the repository;
//! 6. executes the preregistered threshold-free falsification surface added after
//!    v1 (accuracy null + matched never-written shadow controls);
//! 7. emits a second canonical SHA-256 JSONL chain that commits to the base
//!    artifact and all v2 residual observations;
//! 8. rechecks source provenance, subject identity, and checkout cleanliness before
//!    emitting a completion footer.
//!
//! A successful run establishes evidence-production integrity only. It does not
//! interpret outcomes or claim a historical-HLS architecture victory.

use serde_json::{Map, Value, json};
use std::env;
use std::fs;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use symthaea_hdc_ltc::{
    ValidityCapacityAxis, ValidityCapacityCase, ValidityCapacityFalsificationObservation,
    ValidityCapacityPlan, measure_validity_capacity_falsification_surface,
};

const EVIDENCE_VERSION: &str = "hls-validity-capacity-evidence-v2";
const CHAIN_DOMAIN_V1: &str = "symthaea:hdc-ltc:validity-capacity:evidence-chain:v1";
const CHAIN_DOMAIN_V2: &str = "symthaea:hdc-ltc:validity-capacity:evidence-chain:v2";
const SOURCE_BUNDLE_DOMAIN: &str = "symthaea:hdc-ltc:validity-capacity:source-bundle:v2";

// Paths are repository-relative. The byte slices are captured at compile time.
// A stale binary therefore carries the digest of the source bundle it was actually
// compiled against and will fail when run from a different exact checkout.
const SOURCE_BUNDLE: &[(&str, &[u8])] = &[
    (
        "Cargo.lock",
        include_bytes!("../../../../Cargo.lock"),
    ),
    (
        "crates/core/symthaea-hdc-ltc/Cargo.toml",
        include_bytes!("../Cargo.toml"),
    ),
    (
        "crates/core/symthaea-hdc-ltc/src/lib.rs",
        include_bytes!("../src/lib.rs"),
    ),
    (
        "crates/core/symthaea-hdc-ltc/src/continuous_hv.rs",
        include_bytes!("../src/continuous_hv.rs"),
    ),
    (
        "crates/core/symthaea-hdc-ltc/src/temporal_phasor.rs",
        include_bytes!("../src/temporal_phasor.rs"),
    ),
    (
        "crates/core/symthaea-hdc-ltc/src/validity_interval_memory.rs",
        include_bytes!("../src/validity_interval_memory.rs"),
    ),
    (
        "crates/core/symthaea-hdc-ltc/src/validity_capacity.rs",
        include_bytes!("../src/validity_capacity.rs"),
    ),
    (
        "crates/core/symthaea-hdc-ltc/src/validity_capacity_theory.rs",
        include_bytes!("../src/validity_capacity_theory.rs"),
    ),
    (
        "crates/core/symthaea-hdc-ltc/src/validity_capacity_accuracy_theory.rs",
        include_bytes!("../src/validity_capacity_accuracy_theory.rs"),
    ),
    (
        "crates/core/symthaea-hdc-ltc/src/validity_capacity_controls.rs",
        include_bytes!("../src/validity_capacity_controls.rs"),
    ),
    (
        "crates/core/symthaea-hdc-ltc/src/validity_capacity_score_moments.rs",
        include_bytes!("../src/validity_capacity_score_moments.rs"),
    ),
    (
        "crates/core/symthaea-hdc-ltc/src/validity_capacity_shadow_probe.rs",
        include_bytes!("../src/validity_capacity_shadow_probe.rs"),
    ),
    (
        "crates/core/symthaea-hdc-ltc/src/validity_capacity_falsification.rs",
        include_bytes!("../src/validity_capacity_falsification.rs"),
    ),
    (
        "crates/core/symthaea-hdc-ltc/examples/validity_capacity_evidence.rs",
        include_bytes!("validity_capacity_evidence.rs"),
    ),
    (
        "crates/core/symthaea-hdc-ltc/examples/validity_capacity_evidence_v2.rs",
        include_bytes!("validity_capacity_evidence_v2.rs"),
    ),
];

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

    fn evidence_scope(self) -> &'static str {
        match self {
            Self::Smoke => "mechanical_qualification_only",
            Self::ResearchV0 => "exploratory_preregistered_measurement",
        }
    }

    fn capacity_plan(self) -> ValidityCapacityPlan {
        match self {
            Self::Smoke => ValidityCapacityPlan::smoke(),
            Self::ResearchV0 => ValidityCapacityPlan::research_v0(),
        }
    }
}

#[derive(Debug)]
struct VerifiedBaseEvidence {
    bytes: Vec<u8>,
    sha256: String,
    terminal_record_digest: String,
    record_count: usize,
    capacity_observation_count: usize,
    control_observation_count: usize,
}

fn main() -> Result<(), AnyError> {
    verify_sha256_implementation()?;
    let (subject_sha, protocol, base_output) = parse_args()?;
    let repo_root = repository_root()?;
    let repo_root = fs::canonicalize(repo_root)?;
    let base_output = validate_external_output_path(&repo_root, &base_output)?;

    require_clean_exact_subject(&repo_root, &subject_sha)?;
    let embedded_source_bundle_sha256 = embedded_source_bundle_digest();
    let checkout_source_bundle_sha256 = checkout_source_bundle_digest(&repo_root)?;
    if embedded_source_bundle_sha256 != checkout_source_bundle_sha256 {
        return Err(other(format!(
            "stale-binary/source-bundle mismatch: embedded {embedded_source_bundle_sha256}, checkout {checkout_source_bundle_sha256}"
        ))
        .into());
    }

    let subject_tree = git_stdout(&repo_root, &["rev-parse", "HEAD^{tree}"])?;
    let executable_sha256 = sha256_file(&env::current_exe()?)?;
    let cargo_lock_sha256 = sha256_file(&repo_root.join("Cargo.lock"))?;
    let rustc_verbose = command_stdout("rustc", &["-vV"], Some(&repo_root))?;

    let base = fresh_run_and_verify_v1(&repo_root, &subject_sha, protocol)?;
    fs::write(&base_output, &base.bytes)?;

    // Writing the retained evidence artifact outside the repository must not alter
    // the exact subject. Recheck before any v2 measurement is emitted.
    require_clean_exact_subject(&repo_root, &subject_sha)?;

    let capacity_plan = protocol.capacity_plan();
    let falsification = measure_validity_capacity_falsification_surface(&capacity_plan)?;
    let expected_capacity_observations = capacity_plan
        .cases
        .len()
        .checked_mul(capacity_plan.replicate_seeds.len())
        .ok_or_else(|| other("expected capacity observation count overflow"))?;
    if base.capacity_observation_count != expected_capacity_observations
        || falsification.observations.len() != expected_capacity_observations
    {
        return Err(other(format!(
            "cross-layer observation count mismatch: base={}, falsification={}, expected={expected_capacity_observations}",
            base.capacity_observation_count,
            falsification.observations.len()
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
            "evidence_version": EVIDENCE_VERSION,
            "chain_domain": CHAIN_DOMAIN_V2,
            "protocol": protocol.label(),
            "evidence_scope": protocol.evidence_scope(),
            "claim_status": "measurement_only_no_architecture_victory",
            "subject_commit_sha": subject_sha.clone(),
            "subject_tree_sha": subject_tree,
            "embedded_source_bundle_sha256": embedded_source_bundle_sha256,
            "checkout_source_bundle_sha256": checkout_source_bundle_sha256,
            "source_bundle_verified": true,
            "source_bundle_file_count": SOURCE_BUNDLE.len(),
            "cargo_lock_sha256": cargo_lock_sha256,
            "executable_sha256": executable_sha256,
            "rustc_verbose": rustc_verbose,
            "host_os": env::consts::OS,
            "host_arch": env::consts::ARCH,
            "base_evidence_format": "hls-validity-capacity-evidence-v1",
            "supplement_scope": "accuracy_null_matched_shadow_and_threshold_free_falsification",
            "float_encoding": "decimal_scientific_17_digits_plus_ieee754_bits",
            "new_lineage_required_on_source_or_lockfile_drift": true,
        }),
    )?;

    emit_record(
        &mut out,
        &mut sequence,
        &mut previous_digest,
        "base_evidence_commitment",
        json!({
            "sha256": base.sha256,
            "terminal_record_digest": base.terminal_record_digest,
            "record_count": base.record_count,
            "capacity_observation_count": base.capacity_observation_count,
            "control_observation_count": base.control_observation_count,
            "hash_chain_verified": true,
            "fresh_isolated_build": true,
        }),
    )?;

    for observation in &falsification.observations {
        emit_record(
            &mut out,
            &mut sequence,
            &mut previous_digest,
            "falsification_observation",
            falsification_observation_value(observation),
        )?;
    }

    require_clean_exact_subject(&repo_root, &subject_sha)?;
    let postflight_source_bundle_sha256 = checkout_source_bundle_digest(&repo_root)?;
    if postflight_source_bundle_sha256 != embedded_source_bundle_sha256 {
        return Err(other(format!(
            "source bundle changed during evidence production: pre={embedded_source_bundle_sha256}, post={postflight_source_bundle_sha256}"
        ))
        .into());
    }

    emit_record(
        &mut out,
        &mut sequence,
        &mut previous_digest,
        "footer",
        json!({
            "complete": true,
            "subject_unchanged": true,
            "checkout_clean": true,
            "source_bundle_unchanged": true,
            "source_bundle_verified": true,
            "base_v1_hash_chain_verified": true,
            "base_v1_fresh_isolated_build": true,
            "falsification_observation_count": falsification.observations.len(),
            "interpretation": "not_performed_by_runner",
            "scientific_claim": "none",
        }),
    )?;

    out.flush()?;
    eprintln!("VALIDITY_CAPACITY_EVIDENCE_V2_SHA256={previous_digest}");
    Ok(())
}

fn parse_args() -> Result<(String, Protocol, PathBuf), io::Error> {
    let mut args = env::args().skip(1);
    let mut subject_sha = None;
    let mut protocol = None;
    let mut base_output = None;

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
            "--base-evidence-out" => {
                let value = args
                    .next()
                    .ok_or_else(|| invalid("--base-evidence-out requires a value"))?;
                base_output = Some(PathBuf::from(value));
            }
            "-h" | "--help" => {
                println!(
                    "usage: validity_capacity_evidence_v2 --subject-sha <40-hex> \\\n                     --base-evidence-out <path-outside-repo> [--protocol smoke|research-v0]"
                );
                std::process::exit(0);
            }
            other => return Err(invalid(format!("unexpected argument {other:?}"))),
        }
    }

    Ok((
        subject_sha.ok_or_else(|| invalid("--subject-sha is required"))?,
        protocol.unwrap_or(Protocol::Smoke),
        base_output.ok_or_else(|| invalid("--base-evidence-out is required"))?,
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

fn validate_external_output_path(repo_root: &Path, requested: &Path) -> Result<PathBuf, AnyError> {
    let absolute = if requested.is_absolute() {
        requested.to_path_buf()
    } else {
        env::current_dir()?.join(requested)
    };
    let parent = absolute
        .parent()
        .ok_or_else(|| invalid("base evidence output must have a parent directory"))?;
    fs::create_dir_all(parent)?;
    let canonical_parent = fs::canonicalize(parent)?;
    if canonical_parent.starts_with(repo_root) {
        return Err(invalid("base evidence output must be outside the repository").into());
    }
    let file_name = absolute
        .file_name()
        .ok_or_else(|| invalid("base evidence output must name a file"))?;
    Ok(canonical_parent.join(file_name))
}

fn repository_root() -> Result<PathBuf, AnyError> {
    let cwd = env::current_dir()?;
    Ok(PathBuf::from(git_stdout(
        &cwd,
        &["rev-parse", "--show-toplevel"],
    )?))
}

fn require_clean_exact_subject(repo_root: &Path, subject_sha: &str) -> Result<(), AnyError> {
    let head = git_stdout(repo_root, &["rev-parse", "HEAD"])?;
    if head != subject_sha {
        return Err(other(format!(
            "subject mismatch: required {subject_sha}, checkout is {head}"
        ))
        .into());
    }
    let status = git_stdout(
        repo_root,
        &["status", "--porcelain=v1", "--untracked-files=all"],
    )?;
    if !status.is_empty() {
        return Err(other(format!(
            "evidence run requires a clean checkout; git status reported:\n{status}"
        ))
        .into());
    }
    Ok(())
}

fn fresh_run_and_verify_v1(
    repo_root: &Path,
    subject_sha: &str,
    protocol: Protocol,
) -> Result<VerifiedBaseEvidence, AnyError> {
    let target_dir = env::temp_dir().join(format!(
        "symthaea-validity-evidence-v1-{}-{}",
        std::process::id(),
        &subject_sha[..12]
    ));
    if target_dir.exists() {
        fs::remove_dir_all(&target_dir)?;
    }
    fs::create_dir_all(&target_dir)?;

    let output = Command::new("cargo")
        .current_dir(repo_root)
        .env("CARGO_TARGET_DIR", &target_dir)
        .args([
            "run",
            "--quiet",
            "-p",
            "symthaea-hdc-ltc",
            "--example",
            "validity_capacity_evidence",
            "--",
            "--subject-sha",
            subject_sha,
            "--protocol",
            protocol.label(),
        ])
        .output();

    let cleanup = |path: &Path| {
        if path.exists() {
            let _ = fs::remove_dir_all(path);
        }
    };

    let output = match output {
        Ok(output) => output,
        Err(error) => {
            cleanup(&target_dir);
            return Err(error.into());
        }
    };
    cleanup(&target_dir);

    if !output.status.success() {
        return Err(other(format!(
            "fresh v1 evidence execution failed with status {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ))
        .into());
    }

    verify_v1_chain(output.stdout, subject_sha, protocol)
}

fn verify_v1_chain(
    bytes: Vec<u8>,
    subject_sha: &str,
    protocol: Protocol,
) -> Result<VerifiedBaseEvidence, AnyError> {
    let text = std::str::from_utf8(&bytes)?;
    let mut expected_sequence = 0_u64;
    let mut expected_previous = "0".repeat(64);
    let mut terminal_record_digest = None;
    let mut final_kind = None;
    let mut final_payload = None;
    let mut capacity_observation_count = 0usize;
    let mut control_observation_count = 0usize;
    let mut saw_header = false;

    for line in text.lines().filter(|line| !line.trim().is_empty()) {
        let record: Value = serde_json::from_str(line)?;
        let sequence = record["sequence"]
            .as_u64()
            .ok_or_else(|| other("v1 record sequence missing or invalid"))?;
        if sequence != expected_sequence {
            return Err(other(format!(
                "v1 sequence mismatch: expected {expected_sequence}, got {sequence}"
            ))
            .into());
        }
        let previous = record["previous_digest"]
            .as_str()
            .ok_or_else(|| other("v1 previous_digest missing"))?;
        if previous != expected_previous {
            return Err(other(format!(
                "v1 previous-digest mismatch at sequence {sequence}"
            ))
            .into());
        }
        let kind = record["kind"]
            .as_str()
            .ok_or_else(|| other("v1 kind missing"))?;
        let payload = record["payload"].clone();
        let recorded_digest = record["record_digest"]
            .as_str()
            .ok_or_else(|| other("v1 record_digest missing"))?;
        let envelope = json!({
            "sequence": sequence,
            "previous_digest": previous,
            "kind": kind,
            "payload": payload.clone(),
        });
        let expected_digest = digest_value(CHAIN_DOMAIN_V1, &envelope)?;
        if recorded_digest != expected_digest {
            return Err(other(format!(
                "v1 record digest mismatch at sequence {sequence}"
            ))
            .into());
        }

        if sequence == 0 {
            if kind != "header"
                || payload["subject_commit_sha"].as_str() != Some(subject_sha)
                || payload["protocol"].as_str() != Some(protocol.label())
            {
                return Err(other("v1 header does not match requested subject/protocol").into());
            }
            saw_header = true;
        }
        if kind == "capacity_observation" {
            capacity_observation_count += 1;
        } else if kind == "control_observation" {
            control_observation_count += 1;
        }

        expected_previous = recorded_digest.to_owned();
        terminal_record_digest = Some(recorded_digest.to_owned());
        final_kind = Some(kind.to_owned());
        final_payload = Some(payload);
        expected_sequence = expected_sequence
            .checked_add(1)
            .ok_or_else(|| other("v1 sequence overflow"))?;
    }

    if !saw_header || final_kind.as_deref() != Some("footer") {
        return Err(other("v1 chain is missing a valid header/footer boundary").into());
    }
    let final_payload = final_payload.ok_or_else(|| other("v1 footer payload missing"))?;
    if final_payload["complete"].as_bool() != Some(true)
        || final_payload["subject_unchanged"].as_bool() != Some(true)
        || final_payload["checkout_clean"].as_bool() != Some(true)
        || final_payload["scientific_claim"].as_str() != Some("none")
    {
        return Err(other("v1 footer did not close the evidence theorem").into());
    }

    Ok(VerifiedBaseEvidence {
        sha256: sha256_hex(&bytes),
        bytes,
        terminal_record_digest: terminal_record_digest
            .ok_or_else(|| other("v1 chain contained no records"))?,
        record_count: expected_sequence as usize,
        capacity_observation_count,
        control_observation_count,
    })
}

fn embedded_source_bundle_digest() -> String {
    source_bundle_digest(SOURCE_BUNDLE.iter().copied())
}

fn checkout_source_bundle_digest(repo_root: &Path) -> Result<String, AnyError> {
    let mut owned = Vec::with_capacity(SOURCE_BUNDLE.len());
    for (path, _) in SOURCE_BUNDLE {
        owned.push((*path, fs::read(repo_root.join(path))?));
    }
    Ok(source_bundle_digest(
        owned.iter().map(|(path, bytes)| (*path, bytes.as_slice())),
    ))
}

fn source_bundle_digest<'a>(entries: impl IntoIterator<Item = (&'a str, &'a [u8])>) -> String {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(SOURCE_BUNDLE_DOMAIN.as_bytes());
    bytes.push(0);
    for (path, content) in entries {
        bytes.extend_from_slice(&(path.len() as u64).to_be_bytes());
        bytes.extend_from_slice(path.as_bytes());
        bytes.extend_from_slice(&(content.len() as u64).to_be_bytes());
        bytes.extend_from_slice(content);
    }
    sha256_hex(&bytes)
}

fn falsification_observation_value(observation: &ValidityCapacityFalsificationObservation) -> Value {
    json!({
        "case": capacity_case_value(observation.case),
        "seed": observation.seed,
        "total_queries": observation.total_queries,
        "accuracy": {
            "empirical": float_value(observation.empirical_accuracy),
            "predicted_null": float_value(observation.predicted_accuracy),
            "empirical_minus_null": float_value(observation.accuracy_residual),
            "integration_refinement_delta": float_value(observation.accuracy_integration_refinement_delta),
        },
        "target_null_residuals": {
            "mean_bias": float_value(observation.target_mean_bias),
            "mean_squared_residual": float_value(observation.target_mean_squared_residual),
            "mse_ratio_to_null": float_value(observation.target_mse_ratio_to_null),
        },
        "vocabulary_distractor": {
            "mean_bias": float_value(observation.vocabulary_distractor_mean_bias),
            "mean_squared_residual": float_value(observation.vocabulary_distractor_mean_squared_residual),
            "mse_ratio_to_null": float_value(observation.vocabulary_distractor_mse_ratio_to_null),
        },
        "matched_never_written_shadow": {
            "mean_bias": float_value(observation.shadow_distractor_mean_bias),
            "mean_squared_residual": float_value(observation.shadow_distractor_mean_squared_residual),
            "mse_ratio_to_null": float_value(observation.shadow_distractor_mse_ratio_to_null),
            "max_abs_shadow_candidate_similarity": float_value(observation.max_abs_shadow_candidate_similarity),
        },
        "paired_residuals": {
            "vocabulary_minus_shadow_mse": float_value(observation.vocabulary_minus_shadow_mse),
            "vocabulary_minus_shadow_mse_ratio": float_value(observation.vocabulary_minus_shadow_mse_ratio),
            "vocabulary_minus_shadow_variance": float_value(observation.vocabulary_minus_shadow_variance),
        },
        "signed_true_margin": {
            "mean": float_value(observation.mean_true_margin),
            "minimum": float_value(observation.smallest_true_margin),
        },
    })
}

fn capacity_axis_label(axis: ValidityCapacityAxis) -> &'static str {
    match axis {
        ValidityCapacityAxis::Smoke => "smoke",
        ValidityCapacityAxis::Dimension => "dimension",
        ValidityCapacityAxis::KeyCount => "key_count",
        ValidityCapacityAxis::CandidateCount => "candidate_count",
        ValidityCapacityAxis::Horizon => "horizon",
        ValidityCapacityAxis::SpanLength => "span_length",
    }
}

fn capacity_case_value(case: ValidityCapacityCase) -> Value {
    json!({
        "axis": capacity_axis_label(case.axis),
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
    let digest = digest_value(CHAIN_DOMAIN_V2, &envelope)?;
    let record = json!({
        "sequence": *sequence,
        "previous_digest": previous_digest.as_str(),
        "kind": kind,
        "payload": envelope["payload"].clone(),
        "record_digest": digest,
    });
    out.write_all(&canonical_json_bytes(&record)?)?;
    out.write_all(b"\n")?;
    *previous_digest = record["record_digest"]
        .as_str()
        .ok_or_else(|| other("record digest was not a string"))?
        .to_owned();
    *sequence = sequence
        .checked_add(1)
        .ok_or_else(|| other("v2 evidence sequence overflow"))?;
    Ok(())
}

fn digest_value(domain: &str, value: &Value) -> Result<String, AnyError> {
    let canonical = canonical_json_bytes(value)?;
    let mut bytes = Vec::with_capacity(domain.len() + 1 + canonical.len());
    bytes.extend_from_slice(domain.as_bytes());
    bytes.push(0);
    bytes.extend_from_slice(&canonical);
    Ok(sha256_hex(&bytes))
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

fn git_stdout(repo_root: &Path, args: &[&str]) -> Result<String, AnyError> {
    command_stdout("git", args, Some(repo_root))
}

fn command_stdout(
    program: &str,
    args: &[&str],
    current_dir: Option<&Path>,
) -> Result<String, AnyError> {
    let mut command = Command::new(program);
    command.args(args);
    if let Some(current_dir) = current_dir {
        command.current_dir(current_dir);
    }
    let output = command.output()?;
    if !output.status.success() {
        return Err(other(format!(
            "{program} {:?} failed with status {}: {}",
            args,
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ))
        .into());
    }
    Ok(String::from_utf8(output.stdout)?.trim().to_owned())
}

fn sha256_file(path: &Path) -> Result<String, AnyError> {
    Ok(sha256_hex(&fs::read(path)?))
}

fn verify_sha256_implementation() -> Result<(), AnyError> {
    let empty = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
    let abc = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    if sha256_hex(b"") != empty || sha256_hex(b"abc") != abc {
        return Err(other("internal SHA-256 self-test failed").into());
    }
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = sha256(bytes);
    let mut out = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(out, "{byte:02x}");
    }
    out
}

fn sha256(input: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
        0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
        0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
        0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
        0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];

    let mut h = [
        0x6a09e667_u32,
        0xbb67ae85,
        0x3c6ef372,
        0xa54ff53a,
        0x510e527f,
        0x9b05688c,
        0x1f83d9ab,
        0x5be0cd19,
    ];

    let bit_len = (input.len() as u64).wrapping_mul(8);
    let mut padded = input.to_vec();
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in padded.chunks_exact(64) {
        let mut w = [0_u32; 64];
        for (index, word) in chunk.chunks_exact(4).take(16).enumerate() {
            w[index] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for index in 16..64 {
            let s0 = w[index - 15].rotate_right(7)
                ^ w[index - 15].rotate_right(18)
                ^ (w[index - 15] >> 3);
            let s1 = w[index - 2].rotate_right(17)
                ^ w[index - 2].rotate_right(19)
                ^ (w[index - 2] >> 10);
            w[index] = w[index - 16]
                .wrapping_add(s0)
                .wrapping_add(w[index - 7])
                .wrapping_add(s1);
        }

        let mut a = h[0];
        let mut b = h[1];
        let mut c = h[2];
        let mut d = h[3];
        let mut e = h[4];
        let mut f = h[5];
        let mut g = h[6];
        let mut hh = h[7];

        for index in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choice = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(choice)
                .wrapping_add(K[index])
                .wrapping_add(w[index]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(majority);

            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    let mut out = [0_u8; 32];
    for (index, word) in h.into_iter().enumerate() {
        out[index * 4..index * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
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
    fn embedded_bundle_contains_unique_paths() {
        let mut paths = SOURCE_BUNDLE.iter().map(|(path, _)| *path).collect::<Vec<_>>();
        let original_len = paths.len();
        paths.sort_unstable();
        paths.dedup();
        assert_eq!(paths.len(), original_len);
    }

    #[test]
    fn canonical_json_is_key_order_independent() {
        let first = json!({"b": 2, "a": {"d": 4, "c": 3}});
        let second = json!({"a": {"c": 3, "d": 4}, "b": 2});
        assert_eq!(canonical_json_bytes(&first).unwrap(), canonical_json_bytes(&second).unwrap());
    }

    #[test]
    fn sha256_known_answers_pass() {
        verify_sha256_implementation().unwrap();
    }
}
