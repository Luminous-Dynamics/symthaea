// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Subject-locked JSONL evidence runner for the validity-memory research line.
//!
//! This executable does not interpret outcomes or assert a favorable result. It:
//! - requires an exact 40-hex subject commit and a clean checkout;
//! - commits to the frozen capacity and orthogonal-control plans;
//! - reruns both the primary capacity sweep and the independent score-moment
//!   reconstruction and fails closed on any parity mismatch;
//! - runs the orthogonal semantic-density/write-segmentation controls;
//! - emits canonical, SHA-256 hash-chained JSONL with exact f64 bit patterns;
//! - verifies HEAD and checkout cleanliness again before emitting the footer.
//!
//! Write stdout outside the repository for strict runs, for example:
//!
//! `cargo run -p symthaea-hdc-ltc --example validity_capacity_evidence -- \`
//! `  --subject-sha "$(git rev-parse HEAD)" --protocol research-v0 \`
//! `  > "$RUNNER_TEMP/validity-capacity-evidence.jsonl"`

use serde_json::{Map, Value, json};
use std::env;
use std::fs;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use symthaea_hdc_ltc::{
    ScoreMomentSummary, ValidityCapacityAxis, ValidityCapacityCase, ValidityCapacityControlAxis,
    ValidityCapacityControlCase, ValidityCapacityControlObservation, ValidityCapacityControlPlan,
    ValidityCapacityObservation, ValidityCapacityPlan, ValidityCapacityScoreMomentObservation,
    measure_validity_capacity_score_moments, run_validity_capacity_controls,
    run_validity_capacity_sweep,
};

const EVIDENCE_VERSION: &str = "hls-validity-capacity-evidence-v1";
const CHAIN_DOMAIN: &str = "symthaea:hdc-ltc:validity-capacity:evidence-chain:v1";
const PLAN_DOMAIN: &str = "symthaea:hdc-ltc:validity-capacity:plan:v1";

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
}

fn main() -> Result<(), AnyError> {
    verify_sha256_implementation()?;
    let (subject_sha, protocol) = parse_args()?;
    let repo_root = repository_root()?;

    require_clean_exact_subject(&repo_root, &subject_sha)?;
    let subject_tree = git_stdout(&repo_root, &["rev-parse", "HEAD^{tree}"])?;
    let cargo_lock_digest = sha256_file(&repo_root.join("Cargo.lock"))?;
    let executable_digest = sha256_file(&env::current_exe()?)?;
    let rustc_verbose = command_stdout("rustc", &["-vV"], Some(&repo_root))?;

    let (capacity_plan, control_plan) = match protocol {
        Protocol::Smoke => (
            ValidityCapacityPlan::smoke(),
            ValidityCapacityControlPlan::smoke(),
        ),
        Protocol::ResearchV0 => (
            ValidityCapacityPlan::research_v0(),
            ValidityCapacityControlPlan::research_v0(),
        ),
    };

    let capacity_plan_json = capacity_plan_value(&capacity_plan);
    let control_plan_json = control_plan_value(&control_plan);
    let capacity_plan_digest = digest_value(PLAN_DOMAIN, &capacity_plan_json)?;
    let control_plan_digest = digest_value(PLAN_DOMAIN, &control_plan_json)?;

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
            "chain_domain": CHAIN_DOMAIN,
            "protocol": protocol.label(),
            "evidence_scope": protocol.evidence_scope(),
            "claim_status": "measurement_only_no_architecture_victory",
            "subject_commit_sha": subject_sha.clone(),
            "subject_tree_sha": subject_tree,
            "capacity_plan_sha256": capacity_plan_digest,
            "control_plan_sha256": control_plan_digest,
            "cargo_lock_sha256": cargo_lock_digest,
            "executable_sha256": executable_digest,
            "rustc_verbose": rustc_verbose,
            "host_os": env::consts::OS,
            "host_arch": env::consts::ARCH,
            "float_encoding": "decimal_scientific_17_digits_plus_ieee754_bits",
            "output_format": "canonical_jsonl_hash_chain",
        }),
    )?;

    emit_record(
        &mut out,
        &mut sequence,
        &mut previous_digest,
        "capacity_plan",
        capacity_plan_json,
    )?;
    emit_record(
        &mut out,
        &mut sequence,
        &mut previous_digest,
        "control_plan",
        control_plan_json,
    )?;

    // Deliberately execute both paths. The research artifact is invalid if the
    // primary sweep and the independently reconstructed measurement path diverge.
    let primary = run_validity_capacity_sweep(&capacity_plan)?;
    let score_moments = measure_validity_capacity_score_moments(&capacity_plan)?;
    if primary.observations.len() != score_moments.observations.len() {
        return Err(other(format!(
            "primary/score observation length mismatch: {} != {}",
            primary.observations.len(),
            score_moments.observations.len()
        ))
        .into());
    }

    for (index, (primary_observation, score_observation)) in primary
        .observations
        .iter()
        .zip(&score_moments.observations)
        .enumerate()
    {
        require_capacity_parity(index, primary_observation, score_observation)?;
        emit_record(
            &mut out,
            &mut sequence,
            &mut previous_digest,
            "capacity_observation",
            combined_capacity_observation(primary_observation, score_observation),
        )?;
    }

    let controls = run_validity_capacity_controls(&control_plan)?;
    for observation in &controls.observations {
        emit_record(
            &mut out,
            &mut sequence,
            &mut previous_digest,
            "control_observation",
            control_observation_value(observation),
        )?;
    }

    // Postflight is part of the evidence theorem: result production is not allowed
    // to mutate the exact subject or leave the checkout dirty.
    require_clean_exact_subject(&repo_root, &subject_sha)?;

    let capacity_observation_count = primary.observations.len();
    let control_observation_count = controls.observations.len();
    emit_record(
        &mut out,
        &mut sequence,
        &mut previous_digest,
        "footer",
        json!({
            "complete": true,
            "subject_unchanged": true,
            "checkout_clean": true,
            "primary_score_parity_verified": true,
            "capacity_observation_count": capacity_observation_count,
            "control_observation_count": control_observation_count,
            "total_observation_count": capacity_observation_count + control_observation_count,
            "interpretation": "not_performed_by_runner",
            "scientific_claim": "none",
        }),
    )?;

    out.flush()?;
    eprintln!("VALIDITY_CAPACITY_EVIDENCE_SHA256={previous_digest}");
    Ok(())
}

fn parse_args() -> Result<(String, Protocol), io::Error> {
    let mut args = env::args().skip(1);
    let mut subject_sha = None;
    let mut protocol = None;

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
            "-h" | "--help" => {
                println!(
                    "usage: validity_capacity_evidence --subject-sha <40-hex> \
                     [--protocol smoke|research-v0]"
                );
                std::process::exit(0);
            }
            other => return Err(invalid(format!("unexpected argument {other:?}"))),
        }
    }

    let subject_sha =
        subject_sha.ok_or_else(|| invalid("--subject-sha is required for all evidence runs"))?;
    Ok((subject_sha, protocol.unwrap_or(Protocol::Smoke)))
}

fn validate_subject_sha(value: &str) -> Result<String, io::Error> {
    if value.len() != 40 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(invalid(
            "subject SHA must be exactly 40 hexadecimal characters",
        ));
    }
    Ok(value.to_ascii_lowercase())
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
    let bytes = fs::read(path)
        .map_err(|error| other(format!("failed to read {}: {error}", path.display())))?;
    Ok(sha256_hex(&bytes))
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

fn digest_value(domain: &str, value: &Value) -> Result<String, AnyError> {
    let canonical = canonical_json_bytes(value)?;
    let mut bytes = Vec::with_capacity(domain.len() + 1 + canonical.len());
    bytes.extend_from_slice(domain.as_bytes());
    bytes.push(0);
    bytes.extend_from_slice(&canonical);
    Ok(sha256_hex(&bytes))
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
    let digest = digest_value(CHAIN_DOMAIN, &envelope)?;
    let record = json!({
        "sequence": *sequence,
        "previous_digest": previous_digest.as_str(),
        "kind": kind,
        "payload": envelope["payload"].clone(),
        "record_digest": digest,
    });
    let canonical = canonical_json_bytes(&record)?;
    out.write_all(&canonical)?;
    out.write_all(b"\n")?;
    *previous_digest = record["record_digest"]
        .as_str()
        .ok_or_else(|| other("record digest was not a string"))?
        .to_owned();
    *sequence = (*sequence)
        .checked_add(1)
        .ok_or_else(|| other("evidence sequence overflow"))?;
    Ok(())
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

fn float_value(value: f64) -> Value {
    json!({
        "decimal": format!("{value:.17e}"),
        "bits": format!("0x{:016x}", value.to_bits()),
    })
}

fn ratio_or_infinity(observed: f64, predicted: f64) -> f64 {
    if predicted == 0.0 {
        if observed == 0.0 { 1.0 } else { f64::INFINITY }
    } else {
        observed / predicted
    }
}

fn score_moment_value(summary: ScoreMomentSummary) -> Value {
    json!({
        "count": summary.count,
        "mean": float_value(summary.mean),
        "variance": float_value(summary.variance),
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

fn control_axis_label(axis: ValidityCapacityControlAxis) -> &'static str {
    match axis {
        ValidityCapacityControlAxis::Smoke => "smoke",
        ValidityCapacityControlAxis::SemanticRunLength => "semantic_run_length",
        ValidityCapacityControlAxis::WriteSegmentation => "write_segmentation",
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

fn control_case_value(case: ValidityCapacityControlCase) -> Value {
    json!({
        "axis": control_axis_label(case.axis),
        "dim": case.dim,
        "key_count": case.key_count,
        "candidate_count": case.candidate_count,
        "horizon": case.horizon,
        "semantic_run_length": case.semantic_run_length,
        "write_segment_length": case.write_segment_length,
    })
}

fn capacity_plan_value(plan: &ValidityCapacityPlan) -> Value {
    json!({
        "plan": "validity_capacity",
        "cases": plan.cases.iter().copied().map(capacity_case_value).collect::<Vec<_>>(),
        "replicate_seeds": plan.replicate_seeds.clone(),
    })
}

fn control_plan_value(plan: &ValidityCapacityControlPlan) -> Value {
    json!({
        "plan": "validity_capacity_controls",
        "cases": plan.cases.iter().copied().map(control_case_value).collect::<Vec<_>>(),
        "replicate_seeds": plan.replicate_seeds.clone(),
    })
}

fn require_capacity_parity(
    index: usize,
    primary: &ValidityCapacityObservation,
    score: &ValidityCapacityScoreMomentObservation,
) -> Result<(), AnyError> {
    let exact = primary.case == score.case
        && primary.seed == score.seed
        && primary.correct == score.correct
        && primary.total_queries == score.total_queries
        && primary.accuracy.to_bits() == score.accuracy.to_bits()
        && primary.mean_margin.to_bits() == score.mean_winner_margin.to_bits()
        && primary.smallest_margin.to_bits() == score.smallest_winner_margin.to_bits()
        && primary.spans_written == score.spans_written
        && primary.represented_key_checkpoint_facts == score.represented_key_checkpoint_facts
        && primary.facts_per_dimension.to_bits() == score.facts_per_dimension.to_bits();

    if exact {
        Ok(())
    } else {
        Err(other(format!(
            "primary/score-moment parity failure at observation {index}: \
             primary case={:?} seed={}, score case={:?} seed={}",
            primary.case, primary.seed, score.case, score.seed
        ))
        .into())
    }
}

fn combined_capacity_observation(
    primary: &ValidityCapacityObservation,
    score: &ValidityCapacityScoreMomentObservation,
) -> Value {
    let target_bias = score.target_scores.mean - 1.0;
    let distractor_bias = score.probe_distractor_scores.mean;
    let target_null_mse = score.target_scores.variance + target_bias * target_bias;
    let distractor_null_mse =
        score.probe_distractor_scores.variance + distractor_bias * distractor_bias;

    json!({
        "case": capacity_case_value(primary.case),
        "seed": primary.seed,
        "primary": {
            "correct": primary.correct,
            "total_queries": primary.total_queries,
            "accuracy": float_value(primary.accuracy),
            "mean_winner_margin": float_value(primary.mean_margin),
            "smallest_winner_margin": float_value(primary.smallest_margin),
            "spans_written": primary.spans_written,
            "realized_semantic_changes": primary.realized_semantic_changes,
            "used_candidate_values": primary.used_candidate_values,
            "represented_key_checkpoint_facts": primary.represented_key_checkpoint_facts,
            "facts_per_dimension": float_value(primary.facts_per_dimension),
            "candidate_score_evaluations": primary.candidate_score_evaluations,
            "max_abs_key_similarity": float_value(primary.max_abs_key_similarity),
            "max_abs_candidate_similarity": float_value(primary.max_abs_candidate_similarity),
            "max_abs_key_candidate_similarity": float_value(primary.max_abs_key_candidate_similarity),
            "history_payload_bytes": primary.history_payload_bytes,
            "temporal_axis_payload_bytes": primary.temporal_axis_payload_bytes,
            "codebook_payload_bytes": primary.codebook_payload_bytes,
        },
        "score_moments": {
            "mean_true_margin": float_value(score.mean_true_margin),
            "smallest_true_margin": float_value(score.smallest_true_margin),
            "target_scores": score_moment_value(score.target_scores),
            "probe_distractor_scores": score_moment_value(score.probe_distractor_scores),
        },
        "null_model": {
            "target_mean": float_value(1.0),
            "distractor_mean": float_value(0.0),
            "target_noise_variance": float_value(score.null_target_noise_variance),
            "distractor_noise_variance": float_value(score.null_distractor_noise_variance),
        },
        "null_residuals": {
            "target_mean_bias": float_value(target_bias),
            "distractor_mean_bias": float_value(distractor_bias),
            "target_variance_ratio": float_value(score.target_variance_ratio_to_null),
            "distractor_variance_ratio": float_value(score.distractor_variance_ratio_to_null),
            "target_mean_squared_residual": float_value(target_null_mse),
            "distractor_mean_squared_residual": float_value(distractor_null_mse),
            "target_mse_ratio_to_null": float_value(ratio_or_infinity(
                target_null_mse,
                score.null_target_noise_variance,
            )),
            "distractor_mse_ratio_to_null": float_value(ratio_or_infinity(
                distractor_null_mse,
                score.null_distractor_noise_variance,
            )),
        },
    })
}

fn control_observation_value(observation: &ValidityCapacityControlObservation) -> Value {
    json!({
        "case": control_case_value(observation.case),
        "seed": observation.seed,
        "correct": observation.correct,
        "total_queries": observation.total_queries,
        "accuracy": float_value(observation.accuracy),
        "mean_margin": float_value(observation.mean_margin),
        "smallest_margin": float_value(observation.smallest_margin),
        "spans_written": observation.spans_written,
        "semantic_changes": observation.semantic_changes,
        "represented_key_checkpoint_facts": observation.represented_key_checkpoint_facts,
        "facts_per_dimension": float_value(observation.facts_per_dimension),
        "candidate_score_evaluations": observation.candidate_score_evaluations,
        "null_target_noise_variance": float_value(observation.null_target_noise_variance),
        "null_distractor_noise_variance": float_value(observation.null_distractor_noise_variance),
    })
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn other(message: impl Into<String>) -> io::Error {
    io::Error::other(message.into())
}
