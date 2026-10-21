// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Auditable evaluation for PITB's transfer-selectivity secondary track.
//!
//! This binary analyzes already-adjudicated JSONL outcomes. It does not decide
//! whether a mathematical transfer is valid; it computes metrics against the
//! supplied labels and rejects malformed/inconsistent evaluation inputs.

use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    process::ExitCode,
};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const SCHEMA_VERSION: &str = "pitb.transfer-selectivity-outcome.v0";
const REPORT_SCHEMA_VERSION: &str = "pitb.transfer-selectivity-analysis.v0";
const DEFAULT_BOOTSTRAP_REPLICATES: usize = 10_000;
const DEFAULT_SEED: u64 = 0x5049_5442_5345_4c45;
const MIN_COMPONENTS_PER_CLASS: usize = 20;
const REQUIRED_REVIEWERS: usize = 2;
const REVIEW_REASON_CODES: &[&str] = &[
    "missing_precondition",
    "object_type",
    "domain_mismatch",
    "inequality_direction",
    "predicate_mismatch",
    "quantifier_scope",
    "premise_unavailable",
    "irrelevant_near_neighbour",
    "other",
    "unknown",
];
const TRANSFER_MISMATCH_CODES: &[&str] = REVIEW_REASON_CODES;
const PRECONDITION_OUTCOMES: &[&str] =
    &["satisfied", "violated", "unknown", "not_applicable"];

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Split {
    Train,
    Validation,
    Test,
    ExternalChallenge,
}

impl Split {
    fn as_str(self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Validation => "validation",
            Self::Test => "test",
            Self::ExternalChallenge => "external_challenge",
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Label {
    Compatible,
    Incompatible,
    Partial,
    Unknown,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum AdjudicationStatus {
    Adjudicated,
    Unresolved,
    Pending,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Decision {
    Accept,
    Reject,
    Abstain,
}

#[derive(Debug, Deserialize)]
struct Outcome {
    schema_version: String,
    pair_id: String,
    split: Split,
    component_id: String,
    source: SourceRef,
    target: TargetRef,
    adjudication: Adjudication,
    prediction: Prediction,
}

#[derive(Debug, Deserialize)]
struct SourceRef {
    record_id: String,
    family_id: String,
}

#[derive(Debug, Deserialize)]
struct TargetRef {
    family_id: String,
    obligation_sha256: String,
}

#[derive(Debug, Deserialize)]
struct Adjudication {
    status: AdjudicationStatus,
    label: Label,
    reviewer_decisions: Vec<ReviewerDecision>,
}

#[derive(Debug, Deserialize)]
struct ReviewerDecision {
    reviewer_id: String,
    label: Label,
}

#[derive(Debug, Deserialize)]
struct Prediction {
    decision: Decision,
    predicted_label: Label,
    #[serde(deserialize_with = "deserialize_required_nullable_f64")]
    compatibility_probability: Option<f64>,
}

#[derive(Debug, Default, Clone, Copy, Serialize)]
struct Counts {
    rows: usize,
    adjudicated_compatible: usize,
    adjudicated_incompatible: usize,
    adjudicated_partial: usize,
    unresolved_or_unknown: usize,
    pending: usize,
    accept_decisions: usize,
    reject_decisions: usize,
    abstain_decisions: usize,
    binary_evaluable_pairs: usize,
    compatible_components: usize,
    incompatible_components: usize,
    unique_source_families: usize,
    unique_target_families: usize,
    unique_pair_components: usize,
}

#[derive(Debug, Default, Clone, Copy, Serialize)]
struct CellCounts {
    compatible: usize,
    compatible_accept: usize,
    compatible_reject: usize,
    compatible_abstain: usize,
    incompatible: usize,
    incompatible_accept: usize,
    incompatible_reject: usize,
    incompatible_abstain: usize,
}

impl CellCounts {
    fn add_outcome(&mut self, outcome: &Outcome) {
        match outcome.adjudication.label {
            Label::Compatible
                if outcome.adjudication.status == AdjudicationStatus::Adjudicated =>
            {
                self.compatible += 1;
                match outcome.prediction.decision {
                    Decision::Accept => self.compatible_accept += 1,
                    Decision::Reject => self.compatible_reject += 1,
                    Decision::Abstain => self.compatible_abstain += 1,
                }
            }
            Label::Incompatible
                if outcome.adjudication.status == AdjudicationStatus::Adjudicated =>
            {
                self.incompatible += 1;
                match outcome.prediction.decision {
                    Decision::Accept => self.incompatible_accept += 1,
                    Decision::Reject => self.incompatible_reject += 1,
                    Decision::Abstain => self.incompatible_abstain += 1,
                }
            }
            _ => {}
        }
    }

    fn add_assign(&mut self, other: Self) {
        self.compatible += other.compatible;
        self.compatible_accept += other.compatible_accept;
        self.compatible_reject += other.compatible_reject;
        self.compatible_abstain += other.compatible_abstain;
        self.incompatible += other.incompatible;
        self.incompatible_accept += other.incompatible_accept;
        self.incompatible_reject += other.incompatible_reject;
        self.incompatible_abstain += other.incompatible_abstain;
    }

    fn metric_values(self) -> BTreeMap<&'static str, Option<f64>> {
        let total_binary = self.compatible + self.incompatible;
        let accepted_or_rejected = self.compatible_accept
            + self.compatible_reject
            + self.incompatible_accept
            + self.incompatible_reject;
        let accepted_compatible = self.compatible_accept;
        let rejected_compatible = self.compatible_reject;
        let accepted_incompatible = self.incompatible_accept;
        let false_decisions = rejected_compatible + accepted_incompatible;

        BTreeMap::from([
            (
                "compatible_accept_recall",
                ratio(accepted_compatible, self.compatible),
            ),
            (
                "incompatible_false_accept_rate",
                ratio(accepted_incompatible, self.incompatible),
            ),
            (
                "compatible_false_reject_rate",
                ratio(rejected_compatible, self.compatible),
            ),
            (
                "compatible_abstention_rate",
                ratio(self.compatible_abstain, self.compatible),
            ),
            (
                "incompatible_abstention_rate",
                ratio(self.incompatible_abstain, self.incompatible),
            ),
            (
                "binary_coverage",
                ratio(accepted_or_rejected, total_binary),
            ),
            (
                "selective_risk",
                ratio(false_decisions, accepted_or_rejected),
            ),
        ])
    }
}

#[derive(Debug, Serialize)]
struct Interval {
    lower: f64,
    upper: f64,
}

#[derive(Debug, Serialize)]
struct MetricReport {
    numerator: usize,
    denominator: usize,
    estimate: Option<f64>,
    ci_95: Option<Interval>,
    bootstrap_defined_replicates: usize,
    bootstrap_undefined_replicates: usize,
}

#[derive(Debug, Serialize)]
struct Report {
    schema_version: &'static str,
    input_sha256: String,
    split: &'static str,
    bootstrap_seed: u64,
    bootstrap_replicates: usize,
    counts: Counts,
    metrics: BTreeMap<String, MetricReport>,
    calibration: CalibrationAssessment,
    proposed_gates: ProposedGates,
    limitations: Vec<&'static str>,
}

#[derive(Debug, Serialize)]
struct CalibrationAssessment {
    eligible_binary_pairs: usize,
    probability_scored_pairs: usize,
    probability_coverage: f64,
    complete_binary_coverage: bool,
    scored_components: usize,
    brier_score: Option<f64>,
    brier_score_ci_95: Option<Interval>,
    expected_calibration_error_10_bins: Option<f64>,
    calibration_error_ci_95: Option<Interval>,
    reliability_bins: Vec<ReliabilityBin>,
}

#[derive(Debug, Serialize)]
struct ReliabilityBin {
    bin_index: usize,
    lower_bound: f64,
    upper_bound: f64,
    count: usize,
    mean_predicted_compatibility: f64,
    observed_compatibility_rate: f64,
}

#[derive(Debug, Serialize)]
struct ProposedGates {
    minimum_compatible_components: usize,
    observed_compatible_components: usize,
    compatible_component_floor_met: bool,
    minimum_incompatible_components: usize,
    observed_incompatible_components: usize,
    incompatible_component_floor_met: bool,
    minimum_compatible_recall: f64,
    compatible_recall_target_met: bool,
    pair_selectivity_gate: &'static str,
    finite_mutation_suite_gate: &'static str,
    overall_confirmatory_decision: &'static str,
}

#[derive(Debug, Default, Clone, Copy)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn index(&mut self, upper_exclusive: usize) -> usize {
        debug_assert!(upper_exclusive > 0);
        (self.next_u64() as u128 * upper_exclusive as u128 >> 64) as usize
    }
}

fn main() -> ExitCode {
    match run_from_args() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("PITB transfer-selectivity analysis failed closed: {error:#}");
            ExitCode::FAILURE
        }
    }
}

fn run_from_args() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 || args.len() > 6 || args[1] == "--help" || args[1] == "-h" {
        eprintln!(
            "usage: {} <outcomes.jsonl> <test|external_challenge> <report.json> [bootstrap_replicates] [seed]",
            args.first().map(String::as_str).unwrap_or("analyze_transfer_selectivity")
        );
        if args.len() < 4 && args.get(1).is_some_and(|arg| arg == "--help" || arg == "-h") {
            return Ok(());
        }
        bail!("invalid command-line arguments");
    }

    let expected_split: Split = match args[2].as_str() {
        "test" => Split::Test,
        "external_challenge" => Split::ExternalChallenge,
        other => bail!("confirmatory analysis only accepts test or external_challenge, found {other:?}"),
    };
    let replicates = args
        .get(4)
        .map(|value| value.parse::<usize>().context("parse bootstrap replicate count"))
        .transpose()?
        .unwrap_or(DEFAULT_BOOTSTRAP_REPLICATES);
    if replicates < 1_000 {
        bail!("at least 1000 bootstrap replicates are required for an analysis report");
    }
    let seed = args
        .get(5)
        .map(|value| value.parse::<u64>().context("parse bootstrap seed"))
        .transpose()?
        .unwrap_or(DEFAULT_SEED);

    let input_path = &args[1];
    let output_path = &args[3];
    let bytes = fs::read(input_path).with_context(|| format!("read outcomes {}", input_path))?;
    let input_digest = sha256_hex(&bytes);
    let text = std::str::from_utf8(&bytes).context("outcomes JSONL must be UTF-8")?;
    let outcomes = parse_jsonl(text, expected_split)?;
    let report = analyze(&outcomes, expected_split, input_digest, replicates, seed)?;
    let output = serde_json::to_vec_pretty(&report).context("serialize analysis report")?;
    let mut output = output;
    output.push(b'\n');
    fs::write(output_path, output).with_context(|| format!("write report {output_path}"))?;
    println!(
        "PITB_TRANSFER_SELECTIVITY_ANALYSIS: split={} rows={} binary_pairs={} compatible_components={} incompatible_components={} recall={:?} false_accept={:?} decision={}",
        expected_split.as_str(),
        report.counts.rows,
        report.counts.binary_evaluable_pairs,
        report.counts.compatible_components,
        report.counts.incompatible_components,
        report.metrics["compatible_accept_recall"].estimate,
        report.metrics["incompatible_false_accept_rate"].estimate,
        report.proposed_gates.overall_confirmatory_decision
    );
    Ok(())
}

fn require_exact_keys<'a>(
    value: &'a Value,
    context: &str,
    required: &[&str],
    optional: &[&str],
) -> Result<&'a serde_json::Map<String, Value>> {
    let object = value
        .as_object()
        .with_context(|| format!("{context} must be a JSON object"))?;
    for key in required {
        if !object.contains_key(*key) {
            bail!("{context} is missing required field {key:?}");
        }
    }
    for key in object.keys() {
        if !required.contains(&key.as_str()) && !optional.contains(&key.as_str()) {
            bail!("{context} contains unsupported field {key:?}");
        }
    }
    Ok(object)
}

fn required_nonempty_string<'a>(
    object: &'a serde_json::Map<String, Value>,
    key: &str,
    context: &str,
) -> Result<&'a str> {
    object
        .get(key)
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .with_context(|| format!("{context}.{key} must be a non-empty string"))
}

fn validate_digest_field(
    object: &serde_json::Map<String, Value>,
    key: &str,
    expected_len: usize,
    context: &str,
) -> Result<()> {
    let value = required_nonempty_string(object, key, context)?;
    if value.len() != expected_len
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        bail!(
            "{context}.{key} must be a lowercase hexadecimal digest of {expected_len} characters"
        );
    }
    Ok(())
}

fn validate_string_enum_array(
    values: &[Value],
    allowed: &[&str],
    context: &str,
    require_unique: bool,
) -> Result<()> {
    let mut seen = BTreeSet::new();
    for value in values {
        let value = value
            .as_str()
            .with_context(|| format!("{context} values must be strings"))?;
        if !allowed.contains(&value) {
            bail!("{context} contains unsupported value {value:?}");
        }
        if require_unique && !seen.insert(value) {
            bail!("{context} contains duplicate value {value:?}");
        }
    }
    Ok(())
}

fn validate_outcome_contract(value: &Value, context: &str) -> Result<()> {
    let root = require_exact_keys(
        value,
        context,
        &[
            "schema_version",
            "pair_id",
            "split",
            "component_id",
            "source",
            "target",
            "adjudication",
            "prediction",
            "run",
        ],
        &[],
    )?;
    if required_nonempty_string(root, "schema_version", context)? != SCHEMA_VERSION {
        bail!("{context}.schema_version is not the registered v0 contract");
    }
    for key in ["pair_id", "component_id"] {
        required_nonempty_string(root, key, context)?;
    }

    let source_ctx = format!("{context}.source");
    let source = require_exact_keys(
        root.get("source").context("source field missing")?,
        &source_ctx,
        &[
            "record_id",
            "family_id",
            "source_commit",
            "artifact_sha256",
            "insight_sha256",
        ],
        &[],
    )?;
    for key in ["record_id", "family_id"] {
        required_nonempty_string(source, key, &source_ctx)?;
    }
    validate_digest_field(source, "source_commit", 40, &source_ctx)?;
    validate_digest_field(source, "artifact_sha256", 64, &source_ctx)?;
    validate_digest_field(source, "insight_sha256", 64, &source_ctx)?;

    let target_ctx = format!("{context}.target");
    let target = require_exact_keys(
        root.get("target").context("target field missing")?,
        &target_ctx,
        &[
            "theorem_id",
            "family_id",
            "source_commit",
            "artifact_path",
            "artifact_sha256",
            "statement_sha256",
            "obligation_sha256",
        ],
        &["domain_labels"],
    )?;
    for key in ["theorem_id", "family_id"] {
        required_nonempty_string(target, key, &target_ctx)?;
    }
    validate_digest_field(target, "source_commit", 40, &target_ctx)?;
    for key in ["artifact_sha256", "statement_sha256", "obligation_sha256"] {
        validate_digest_field(target, key, 64, &target_ctx)?;
    }
    let artifact_path = required_nonempty_string(target, "artifact_path", &target_ctx)?;
    let path = Path::new(artifact_path);
    if path.is_absolute() || path.components().any(|component| {
        matches!(component, std::path::Component::ParentDir)
    }) {
        bail!("{target_ctx}.artifact_path must be a safe repository-relative path");
    }
    if let Some(labels) = target.get("domain_labels") {
        let labels = labels
            .as_array()
            .with_context(|| format!("{target_ctx}.domain_labels must be an array"))?;
        if labels.is_empty()
            || labels.iter().any(|label| {
                label.as_str().is_none_or(|value| value.trim().is_empty())
            })
        {
            bail!("{target_ctx}.domain_labels must contain non-empty strings");
        }
    }

    let adjudication_ctx = format!("{context}.adjudication");
    let adjudication = require_exact_keys(
        root.get("adjudication").context("adjudication field missing")?,
        &adjudication_ctx,
        &[
            "status",
            "label",
            "rubric_version",
            "reviewer_decisions",
            "receipt_sha256",
        ],
        &["adjudication_note"],
    )?;
    required_nonempty_string(adjudication, "rubric_version", &adjudication_ctx)?;
    validate_digest_field(adjudication, "receipt_sha256", 64, &adjudication_ctx)?;
    let adjudication_status =
        required_nonempty_string(adjudication, "status", &adjudication_ctx)?;
    let reviewers = adjudication
        .get("reviewer_decisions")
        .and_then(Value::as_array)
        .with_context(|| format!("{adjudication_ctx}.reviewer_decisions must be an array"))?;
    let minimum_reviewers = match adjudication_status {
        "adjudicated" | "unresolved" => REQUIRED_REVIEWERS,
        "pending" => 0,
        other => bail!("{adjudication_ctx}.status has unsupported value {other:?}"),
    };
    if reviewers.len() < minimum_reviewers {
        bail!(
            "{adjudication_ctx}.reviewer_decisions requires at least {minimum_reviewers} independent reviewers for status {adjudication_status:?}"
        );
    }
    for (index, reviewer) in reviewers.iter().enumerate() {
        let reviewer_ctx = format!("{adjudication_ctx}.reviewer_decisions[{index}]");
        let reviewer = require_exact_keys(
            reviewer,
            &reviewer_ctx,
            &["reviewer_id", "label", "reason_codes", "evidence_sha256"],
            &[],
        )?;
        required_nonempty_string(reviewer, "reviewer_id", &reviewer_ctx)?;
        validate_digest_field(reviewer, "evidence_sha256", 64, &reviewer_ctx)?;
        let codes = reviewer
            .get("reason_codes")
            .and_then(Value::as_array)
            .with_context(|| format!("{reviewer_ctx}.reason_codes must be an array"))?;
        if codes.is_empty() {
            bail!("{reviewer_ctx}.reason_codes must contain at least one reason code");
        }
        validate_string_enum_array(
            codes,
            REVIEW_REASON_CODES,
            &format!("{reviewer_ctx}.reason_codes"),
            true,
        )?;
    }

    let prediction_ctx = format!("{context}.prediction");
    let prediction = require_exact_keys(
        root.get("prediction").context("prediction field missing")?,
        &prediction_ctx,
        &[
            "decision",
            "predicted_label",
            "confidence",
            "compatibility_probability",
            "mismatch_codes",
            "precondition_checks",
            "receipt_sha256",
        ],
        &[],
    )?;
    let confidence = prediction
        .get("confidence")
        .and_then(Value::as_f64)
        .with_context(|| format!("{prediction_ctx}.confidence must be numeric"))?;
    if !(0.0..=1.0).contains(&confidence) {
        bail!("{prediction_ctx}.confidence must lie in [0, 1]");
    }
    match prediction.get("compatibility_probability") {
        Some(Value::Null) => {}
        Some(value) => {
            let probability = value.as_f64().with_context(|| {
                format!("{prediction_ctx}.compatibility_probability must be null or numeric")
            })?;
            if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
                bail!("{prediction_ctx}.compatibility_probability must be null or lie in [0, 1]");
            }
        }
        None => bail!(
            "{prediction_ctx}.compatibility_probability is required; use null when unscored"
        ),
    }
    validate_digest_field(prediction, "receipt_sha256", 64, &prediction_ctx)?;
    let mismatch_codes = prediction
        .get("mismatch_codes")
        .and_then(Value::as_array)
        .with_context(|| format!("{prediction_ctx}.mismatch_codes must be an array"))?;
    validate_string_enum_array(
        mismatch_codes,
        TRANSFER_MISMATCH_CODES,
        &format!("{prediction_ctx}.mismatch_codes"),
        true,
    )?;
    let checks = prediction
        .get("precondition_checks")
        .and_then(Value::as_array)
        .with_context(|| format!("{prediction_ctx}.precondition_checks must be an array"))?;
    for (index, check) in checks.iter().enumerate() {
        let check_ctx = format!("{prediction_ctx}.precondition_checks[{index}]");
        let check = require_exact_keys(
            check,
            &check_ctx,
            &["check_id", "outcome", "evidence_sha256"],
            &[],
        )?;
        required_nonempty_string(check, "check_id", &check_ctx)?;
        let outcome = required_nonempty_string(check, "outcome", &check_ctx)?;
        if !PRECONDITION_OUTCOMES.contains(&outcome) {
            bail!("{check_ctx}.outcome has unsupported value {outcome:?}");
        }
        match check.get("evidence_sha256") {
            Some(Value::Null) => {}
            Some(_) => validate_digest_field(check, "evidence_sha256", 64, &check_ctx)?,
            None => bail!("{check_ctx}.evidence_sha256 is required; use null when unavailable"),
        }
    }

    let run_ctx = format!("{context}.run");
    let run = require_exact_keys(
        root.get("run").context("run field missing")?,
        &run_ctx,
        &[
            "model_id",
            "model_version",
            "prompt_sha256",
            "config_sha256",
            "input_sha256",
            "output_sha256",
            "budget",
        ],
        &[
            "actual_model_calls",
            "actual_tokens",
            "actual_wall_ms",
            "actual_cost_microusd",
        ],
    )?;
    for key in ["model_id", "model_version"] {
        required_nonempty_string(run, key, &run_ctx)?;
    }
    for key in ["prompt_sha256", "config_sha256", "input_sha256", "output_sha256"] {
        validate_digest_field(run, key, 64, &run_ctx)?;
    }
    let budget_ctx = format!("{run_ctx}.budget");
    let budget = require_exact_keys(
        run.get("budget").context("run budget missing")?,
        &budget_ctx,
        &["max_model_calls", "max_tokens", "max_wall_ms", "max_cost_microusd"],
        &[],
    )?;
    for key in ["max_model_calls", "max_tokens", "max_wall_ms", "max_cost_microusd"] {
        if budget.get(key).and_then(Value::as_u64).is_none() {
            bail!("{budget_ctx}.{key} must be a non-negative integer");
        }
    }
    for key in ["actual_model_calls", "actual_tokens", "actual_wall_ms", "actual_cost_microusd"] {
        if run.get(key).is_some_and(|value| value.as_u64().is_none()) {
            bail!("{run_ctx}.{key} must be a non-negative integer when present");
        }
    }
    for (actual_key, budget_key) in [
        ("actual_model_calls", "max_model_calls"),
        ("actual_tokens", "max_tokens"),
        ("actual_wall_ms", "max_wall_ms"),
        ("actual_cost_microusd", "max_cost_microusd"),
    ] {
        if let Some(actual) = run.get(actual_key).and_then(Value::as_u64) {
            let limit = budget
                .get(budget_key)
                .and_then(Value::as_u64)
                .context("validated run budget unexpectedly disappeared")?;
            if actual > limit {
                bail!(
                    "{run_ctx}.{actual_key}={actual} exceeds declared {budget_key}={limit}"
                );
            }
        }
    }
    Ok(())
}

fn parse_jsonl(text: &str, expected_split: Split) -> Result<Vec<Outcome>> {
    let mut outcomes = Vec::new();
    let mut pair_ids = BTreeSet::new();
    let mut pair_keys = BTreeSet::new();
    let mut family_components = BTreeMap::<String, String>::new();

    for (line_index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let raw: Value = serde_json::from_str(line)
            .with_context(|| format!("parse outcome JSONL line {}", line_index + 1))?;
        validate_outcome_contract(&raw, &format!("JSONL line {}", line_index + 1))?;
        let outcome: Outcome = serde_json::from_value(raw)
            .with_context(|| format!("deserialize outcome JSONL line {}", line_index + 1))?;
        if outcome.schema_version != SCHEMA_VERSION {
            bail!(
                "line {} has schema_version {:?}, expected {:?}",
                line_index + 1,
                outcome.schema_version,
                SCHEMA_VERSION
            );
        }
        if outcome.split != expected_split {
            bail!(
                "line {} belongs to split {}, but invocation requested {}",
                line_index + 1,
                outcome.split.as_str(),
                expected_split.as_str()
            );
        }
        if outcome.pair_id.trim().is_empty()
            || outcome.component_id.trim().is_empty()
            || outcome.source.record_id.trim().is_empty()
            || outcome.source.family_id.trim().is_empty()
            || outcome.target.family_id.trim().is_empty()
        {
            bail!("line {} contains an empty identity field", line_index + 1);
        }
        if !pair_ids.insert(outcome.pair_id.clone()) {
            bail!("duplicate pair_id {:?}", outcome.pair_id);
        }
        let pair_key = format!(
            "{}\0{}\0{}",
            outcome.source.record_id, outcome.target.family_id, outcome.target.obligation_sha256
        );
        if !pair_keys.insert(pair_key) {
            bail!(
                "duplicate source-insight/target-obligation pair at {}",
                outcome.pair_id
            );
        }

        let mut reviewer_ids = BTreeSet::new();
        for reviewer in &outcome.adjudication.reviewer_decisions {
            if reviewer.reviewer_id.trim().is_empty()
                || !reviewer_ids.insert(&reviewer.reviewer_id)
            {
                bail!(
                    "pair {} has an empty or duplicate reviewer ID",
                    outcome.pair_id
                );
            }
        }
        match outcome.adjudication.status {
            AdjudicationStatus::Adjudicated => {
                if !matches!(
                    outcome.adjudication.label,
                    Label::Compatible | Label::Incompatible | Label::Partial
                ) {
                    bail!("adjudicated pair {} cannot have unknown gold label", outcome.pair_id);
                }
                if reviewer_ids.len() < REQUIRED_REVIEWERS {
                    bail!(
                        "adjudicated pair {} needs at least two independent reviewer IDs",
                        outcome.pair_id
                    );
                }
                if outcome
                    .adjudication
                    .reviewer_decisions
                    .iter()
                    .any(|review| review.label != outcome.adjudication.label)
                {
                    bail!(
                        "adjudicated pair {} has reviewer labels inconsistent with the adjudicated label",
                        outcome.pair_id
                    );
                }
            }
            AdjudicationStatus::Unresolved | AdjudicationStatus::Pending => {
                if outcome.adjudication.label != Label::Unknown {
                    bail!(
                        "unresolved/pending pair {} must retain unknown gold label",
                        outcome.pair_id
                    );
                }
            }
        }
        if outcome
            .prediction
            .compatibility_probability
            .is_some_and(|probability| {
                !probability.is_finite() || !(0.0..=1.0).contains(&probability)
            })
        {
            bail!("pair {} has compatibility probability outside [0, 1]", outcome.pair_id);
        }
        match outcome.prediction.decision {
            Decision::Accept if outcome.prediction.predicted_label != Label::Compatible => {
                bail!("pair {} accepts but does not predict compatible", outcome.pair_id);
            }
            Decision::Reject
                if !matches!(
                    outcome.prediction.predicted_label,
                    Label::Incompatible | Label::Partial
                ) =>
            {
                bail!("pair {} rejects without predicting incompatible/partial", outcome.pair_id);
            }
            _ => {}
        }

        for family_id in [&outcome.source.family_id, &outcome.target.family_id] {
            if let Some(previous) = family_components.get(family_id) {
                if previous != &outcome.component_id {
                    bail!(
                        "family {} appears in components {} and {}; component IDs must encode family-connected components",
                        family_id, previous, outcome.component_id
                    );
                }
            } else {
                family_components.insert(family_id.clone(), outcome.component_id.clone());
            }
        }
        outcomes.push(outcome);
    }
    if outcomes.is_empty() {
        bail!("outcome JSONL contains no records");
    }
    Ok(outcomes)
}

fn analyze(
    outcomes: &[Outcome],
    split: Split,
    input_digest: String,
    bootstrap_replicates: usize,
    seed: u64,
) -> Result<Report> {
    let mut counts = Counts {
        rows: outcomes.len(),
        ..Counts::default()
    };
    let mut source_families = BTreeSet::new();
    let mut target_families = BTreeSet::new();
    let mut all_components = BTreeSet::new();
    let mut class_components = BTreeMap::<String, CellCounts>::new();

    for outcome in outcomes {
        source_families.insert(outcome.source.family_id.clone());
        target_families.insert(outcome.target.family_id.clone());
        all_components.insert(outcome.component_id.clone());
        match outcome.adjudication.status {
            AdjudicationStatus::Adjudicated => match outcome.adjudication.label {
                Label::Compatible => counts.adjudicated_compatible += 1,
                Label::Incompatible => counts.adjudicated_incompatible += 1,
                Label::Partial => counts.adjudicated_partial += 1,
                Label::Unknown => bail!("adjudicated unknown label passed parsing unexpectedly"),
            },
            AdjudicationStatus::Unresolved => counts.unresolved_or_unknown += 1,
            AdjudicationStatus::Pending => counts.pending += 1,
        }
        match outcome.prediction.decision {
            Decision::Accept => counts.accept_decisions += 1,
            Decision::Reject => counts.reject_decisions += 1,
            Decision::Abstain => counts.abstain_decisions += 1,
        }
        class_components
            .entry(outcome.component_id.clone())
            .or_default()
            .add_outcome(outcome);
    }

    counts.unique_source_families = source_families.len();
    counts.unique_target_families = target_families.len();
    counts.unique_pair_components = all_components.len();
    let compatible_component_ids: BTreeSet<&str> = class_components
        .iter()
        .filter(|(_, row)| row.compatible > 0)
        .map(|(id, _)| id.as_str())
        .collect();
    let incompatible_component_ids: BTreeSet<&str> = class_components
        .iter()
        .filter(|(_, row)| row.incompatible > 0)
        .map(|(id, _)| id.as_str())
        .collect();
    counts.compatible_components = compatible_component_ids.len();
    counts.incompatible_components = incompatible_component_ids.len();

    let mut observed = CellCounts::default();
    for row in class_components.values() {
        observed.add_assign(*row);
    }
    counts.binary_evaluable_pairs = observed.compatible + observed.incompatible;
    if counts.binary_evaluable_pairs == 0 {
        bail!("no adjudicated compatible/incompatible outcomes exist in selected split");
    }

    let metric_values = observed.metric_values();
    let mut bootstrap_values: BTreeMap<&'static str, Vec<f64>> = metric_values
        .keys()
        .map(|key| (*key, Vec::with_capacity(bootstrap_replicates)))
        .collect();
    let mut undefined_replicates: BTreeMap<&'static str, usize> =
        metric_values.keys().map(|key| (*key, 0)).collect();
    // Components containing only partial/unknown/pending labels must remain
    // accounted for in the report but cannot contribute zero-valued bootstrap
    // clusters to a binary compatibility metric.
    let clusters: Vec<CellCounts> = class_components
        .values()
        .copied()
        .filter(|row| row.compatible + row.incompatible > 0)
        .collect();
    let mut rng = SplitMix64::new(seed);
    for _ in 0..bootstrap_replicates {
        let mut sample = CellCounts::default();
        for _ in 0..clusters.len() {
            sample.add_assign(clusters[rng.index(clusters.len())]);
        }
        for (name, value) in sample.metric_values() {
            if let Some(value) = value {
                bootstrap_values.get_mut(name).expect("metric registered").push(value);
            } else {
                *undefined_replicates.get_mut(name).expect("metric registered") += 1;
            }
        }
    }

    let mut metrics = BTreeMap::new();
    for (name, estimate) in metric_values {
        let values = bootstrap_values.get_mut(name).expect("metric registered");
        values.sort_by(f64::total_cmp);
        let ci = if values.is_empty() {
            None
        } else {
            Some(Interval {
                lower: quantile(values, 0.025),
                upper: quantile(values, 0.975),
            })
        };
        let (numerator, denominator) = metric_counts(name, observed);
        metrics.insert(
            name.to_string(),
            MetricReport {
                numerator,
                denominator,
                estimate,
                ci_95: ci,
                bootstrap_defined_replicates: values.len(),
                bootstrap_undefined_replicates: undefined_replicates[name],
            },
        );
    }

    let calibration = calibration_assessment(outcomes, bootstrap_replicates, seed);
    let recall = metrics["compatible_accept_recall"].estimate;
    let compatible_floor_met = counts.compatible_components >= MIN_COMPONENTS_PER_CLASS;
    let incompatible_floor_met = counts.incompatible_components >= MIN_COMPONENTS_PER_CLASS;
    let recall_target_met = recall.is_some_and(|value| value >= 0.90);
    let pair_gate_met = compatible_floor_met && incompatible_floor_met && recall_target_met;
    let gates = ProposedGates {
        minimum_compatible_components: MIN_COMPONENTS_PER_CLASS,
        observed_compatible_components: counts.compatible_components,
        compatible_component_floor_met: compatible_floor_met,
        minimum_incompatible_components: MIN_COMPONENTS_PER_CLASS,
        observed_incompatible_components: counts.incompatible_components,
        incompatible_component_floor_met: incompatible_floor_met,
        minimum_compatible_recall: 0.90,
        compatible_recall_target_met: recall_target_met,
        pair_selectivity_gate: if pair_gate_met {
            "pair_track_thresholds_met"
        } else {
            "pair_track_thresholds_not_met"
        },
        finite_mutation_suite_gate: "not_evaluated_by_this_analyzer",
        overall_confirmatory_decision: "not_evaluated_requires_independent_mutation_receipt_and_protocol_audit",
    };

    Ok(Report {
        schema_version: REPORT_SCHEMA_VERSION,
        input_sha256: input_digest,
        split: split.as_str(),
        bootstrap_seed: seed,
        bootstrap_replicates,
        counts,
        metrics,
        calibration,
        proposed_gates: gates,
        limitations: vec![
            "This analyzer computes metrics against supplied adjudication labels; it does not establish that the labels are mathematically correct.",
            "Bootstrap intervals resample declared family-connected components, not individual pairs; incorrect component IDs invalidate the inference.",
            "Unknown, pending, and partial labels are accounted for but excluded from binary compatibility metrics.",
            "Brier score and ten-bin expected calibration error are descriptive and require compatibility_probability coverage; missing probabilities are never imputed.",
            "The finite mutation-control suite must be checked through a separately retained receipt; this analyzer never converts a pass of pair metrics into an overall confirmatory claim.",
            "These outcomes concern transfer selectivity and do not replace the primary verified-proof utility comparison.",
        ],
    })
}

fn calibration_assessment(
    outcomes: &[Outcome],
    bootstrap_replicates: usize,
    seed: u64,
) -> CalibrationAssessment {
    let binary_rows: Vec<(String, Option<f64>, bool)> = outcomes
        .iter()
        .filter(|outcome| outcome.adjudication.status == AdjudicationStatus::Adjudicated)
        .filter_map(|outcome| match outcome.adjudication.label {
            Label::Compatible => Some((
                outcome.component_id.clone(),
                outcome.prediction.compatibility_probability,
                true,
            )),
            Label::Incompatible => Some((
                outcome.component_id.clone(),
                outcome.prediction.compatibility_probability,
                false,
            )),
            Label::Partial | Label::Unknown => None,
        })
        .collect();

    let mut scored_by_component = BTreeMap::<String, Vec<(f64, bool)>>::new();
    for (component_id, probability, truth) in &binary_rows {
        if let Some(probability) = probability {
            scored_by_component
                .entry(component_id.clone())
                .or_default()
                .push((*probability, *truth));
        }
    }
    let scored: Vec<(f64, bool)> = scored_by_component
        .values()
        .flatten()
        .copied()
        .collect();
    let probability_coverage = ratio(scored.len(), binary_rows.len()).unwrap_or(0.0);
    if scored.is_empty() {
        return CalibrationAssessment {
            eligible_binary_pairs: binary_rows.len(),
            probability_scored_pairs: 0,
            probability_coverage,
            complete_binary_coverage: binary_rows.is_empty(),
            scored_components: 0,
            brier_score: None,
            brier_score_ci_95: None,
            expected_calibration_error_10_bins: None,
            calibration_error_ci_95: None,
            reliability_bins: Vec::new(),
        };
    }

    let (brier_score, expected_calibration_error_10_bins) = calibration_scores(&scored);
    let mut bins: Vec<Vec<(f64, bool)>> = (0..10).map(|_| Vec::new()).collect();
    for (probability, truth) in &scored {
        let bin_index = ((*probability * 10.0).floor() as usize).min(9);
        bins[bin_index].push((*probability, *truth));
    }

    let mut reliability_bins = Vec::new();
    for (index, values) in bins.into_iter().enumerate() {
        if values.is_empty() {
            continue;
        }
        let count = values.len();
        let mean_probability = values.iter().map(|(p, _)| *p).sum::<f64>() / count as f64;
        let observed_rate =
            values.iter().filter(|(_, truth)| *truth).count() as f64 / count as f64;
        reliability_bins.push(ReliabilityBin {
            bin_index: index,
            lower_bound: index as f64 / 10.0,
            upper_bound: (index + 1) as f64 / 10.0,
            count,
            mean_predicted_compatibility: mean_probability,
            observed_compatibility_rate: observed_rate,
        });
    }

    let scored_components = scored_by_component.len();
    let (brier_score_ci_95, calibration_error_ci_95) = if scored_components >= 2 {
        let clusters: Vec<&Vec<(f64, bool)>> = scored_by_component.values().collect();
        let mut rng = SplitMix64::new(seed ^ 0x4341_4c49_4252_4154);
        let mut brier_samples = Vec::with_capacity(bootstrap_replicates);
        let mut error_samples = Vec::with_capacity(bootstrap_replicates);
        for _ in 0..bootstrap_replicates {
            let mut sample = Vec::new();
            for _ in 0..clusters.len() {
                sample.extend(clusters[rng.index(clusters.len())].iter().copied());
            }
            let (brier, error) = calibration_scores(&sample);
            brier_samples.push(brier);
            error_samples.push(error);
        }
        brier_samples.sort_by(f64::total_cmp);
        error_samples.sort_by(f64::total_cmp);
        (
            Some(Interval {
                lower: quantile(&brier_samples, 0.025),
                upper: quantile(&brier_samples, 0.975),
            }),
            Some(Interval {
                lower: quantile(&error_samples, 0.025),
                upper: quantile(&error_samples, 0.975),
            }),
        )
    } else {
        (None, None)
    };

    CalibrationAssessment {
        eligible_binary_pairs: binary_rows.len(),
        probability_scored_pairs: scored.len(),
        probability_coverage,
        complete_binary_coverage: scored.len() == binary_rows.len(),
        scored_components,
        brier_score: Some(brier_score),
        brier_score_ci_95,
        expected_calibration_error_10_bins: Some(expected_calibration_error_10_bins),
        calibration_error_ci_95,
        reliability_bins,
    }
}

fn calibration_scores(values: &[(f64, bool)]) -> (f64, f64) {
    debug_assert!(!values.is_empty());
    let brier = values
        .iter()
        .map(|(probability, truth)| {
            let target = if *truth { 1.0 } else { 0.0 };
            (*probability - target).powi(2)
        })
        .sum::<f64>()
        / values.len() as f64;

    let mut bins: Vec<Vec<(f64, bool)>> = (0..10).map(|_| Vec::new()).collect();
    for (probability, truth) in values {
        let bin_index = ((*probability * 10.0).floor() as usize).min(9);
        bins[bin_index].push((*probability, *truth));
    }
    let mut ece = 0.0;
    for bin in bins.into_iter().filter(|bin| !bin.is_empty()) {
        let count = bin.len();
        let mean_probability = bin.iter().map(|(p, _)| *p).sum::<f64>() / count as f64;
        let observed_rate =
            bin.iter().filter(|(_, truth)| *truth).count() as f64 / count as f64;
        ece += count as f64 / values.len() as f64 * (mean_probability - observed_rate).abs();
    }
    (brier, ece)
}

fn metric_counts(name: &str, cells: CellCounts) -> (usize, usize) {
    match name {
        "compatible_accept_recall" => (cells.compatible_accept, cells.compatible),
        "incompatible_false_accept_rate" => (cells.incompatible_accept, cells.incompatible),
        "compatible_false_reject_rate" => (cells.compatible_reject, cells.compatible),
        "compatible_abstention_rate" => (cells.compatible_abstain, cells.compatible),
        "incompatible_abstention_rate" => (cells.incompatible_abstain, cells.incompatible),
        "binary_coverage" => (
            cells.compatible_accept
                + cells.compatible_reject
                + cells.incompatible_accept
                + cells.incompatible_reject,
            cells.compatible + cells.incompatible,
        ),
        "selective_risk" => (
            cells.compatible_reject + cells.incompatible_accept,
            cells.compatible_accept
                + cells.compatible_reject
                + cells.incompatible_accept
                + cells.incompatible_reject,
        ),
        _ => unreachable!("only registered metrics are queried"),
    }
}

fn ratio(numerator: usize, denominator: usize) -> Option<f64> {
    (denominator > 0).then(|| numerator as f64 / denominator as f64)
}

fn quantile(sorted_values: &[f64], q: f64) -> f64 {
    debug_assert!(!sorted_values.is_empty());
    let position = q * (sorted_values.len() - 1) as f64;
    let low = position.floor() as usize;
    let high = position.ceil() as usize;
    let weight = position - low as f64;
    sorted_values[low] + (sorted_values[high] - sorted_values[low]) * weight
}

fn deserialize_required_nullable_f64<'de, D>(
    deserializer: D,
) -> std::result::Result<Option<f64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Option::<f64>::deserialize(deserializer)?;
    if value.is_some_and(|probability| {
        !probability.is_finite() || !(0.0..=1.0).contains(&probability)
    }) {
        return Err(<D::Error as serde::de::Error>::custom(
            "compatibility_probability must be null or a finite value in [0, 1]",
        ));
    }
    Ok(value)
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn outcome(
        pair_id: &str,
        component_id: &str,
        source_family: &str,
        target_family: &str,
        label: Label,
        decision: Decision,
    ) -> Outcome {
        let predicted_label = match decision {
            Decision::Accept => Label::Compatible,
            Decision::Reject => Label::Incompatible,
            Decision::Abstain => Label::Unknown,
        };
        let reviewer_decisions = vec![
            ReviewerDecision { reviewer_id: "reviewer-a".to_string(), label },
            ReviewerDecision { reviewer_id: "reviewer-b".to_string(), label },
        ];
        let json = json!({
            "schema_version": SCHEMA_VERSION,
            "pair_id": pair_id,
            "split": "test",
            "component_id": component_id,
            "source": {"record_id": format!("source-{pair_id}"), "family_id": source_family},
            "target": {
                "family_id": target_family,
                "obligation_sha256": format!("{:064x}", pair_id.len())
            },
            "adjudication": {
                "status": "adjudicated",
                "label": label,
                "reviewer_decisions": reviewer_decisions.iter().map(|review| json!({
                    "reviewer_id": review.reviewer_id,
                    "label": review.label
                })).collect::<Vec<_>>()
            },
            "prediction": {
                "decision": decision,
                "predicted_label": predicted_label,
                "compatibility_probability": match decision {
                    Decision::Accept => Some(0.9),
                    Decision::Reject => Some(0.1),
                    Decision::Abstain => None
                }
            }
        });
        serde_json::from_value(json).expect("valid test fixture")
    }

    fn outcome_contract_fixture() -> Value {
        let digest = "a".repeat(64);
        let commit = "b".repeat(40);
        json!({
            "schema_version": SCHEMA_VERSION,
            "pair_id": "contract-fixture",
            "split": "test",
            "component_id": "component-001",
            "source": {
                "record_id": "source-record",
                "family_id": "source-family",
                "source_commit": commit,
                "artifact_sha256": digest,
                "insight_sha256": digest
            },
            "target": {
                "theorem_id": "Synthetic.Target",
                "family_id": "target-family",
                "source_commit": commit,
                "artifact_path": "lean/Synthetic/Target.lean",
                "artifact_sha256": digest,
                "statement_sha256": digest,
                "obligation_sha256": digest,
                "domain_labels": ["synthetic-only"]
            },
            "adjudication": {
                "status": "adjudicated",
                "label": "compatible",
                "rubric_version": "synthetic-rubric-v0",
                "reviewer_decisions": [
                    {
                        "reviewer_id": "reviewer-a",
                        "label": "compatible",
                        "reason_codes": ["other"],
                        "evidence_sha256": digest
                    },
                    {
                        "reviewer_id": "reviewer-b",
                        "label": "compatible",
                        "reason_codes": ["other"],
                        "evidence_sha256": digest
                    }
                ],
                "receipt_sha256": digest
            },
            "prediction": {
                "decision": "accept",
                "predicted_label": "compatible",
                "confidence": 0.9,
                "compatibility_probability": 0.9,
                "mismatch_codes": [],
                "precondition_checks": [
                    {
                        "check_id": "precondition-001",
                        "outcome": "satisfied",
                        "evidence_sha256": digest
                    }
                ],
                "receipt_sha256": digest
            },
            "run": {
                "model_id": "synthetic-only",
                "model_version": "synthetic-v0",
                "prompt_sha256": digest,
                "config_sha256": digest,
                "input_sha256": digest,
                "output_sha256": digest,
                "budget": {
                    "max_model_calls": 8,
                    "max_tokens": 16000,
                    "max_wall_ms": 300000,
                    "max_cost_microusd": 500000
                },
                "actual_model_calls": 1,
                "actual_tokens": 100,
                "actual_wall_ms": 100,
                "actual_cost_microusd": 1
            }
        })
    }

    #[test]
    fn jsonl_contract_requires_complete_content_addressed_receipts() {
        let valid = outcome_contract_fixture();
        assert!(validate_outcome_contract(&valid, "valid fixture").is_ok());
        let line = serde_json::to_string(&valid).unwrap();
        assert_eq!(parse_jsonl(&line, Split::Test).unwrap().len(), 1);

        let mut missing_run = valid.clone();
        missing_run.as_object_mut().unwrap().remove("run");
        assert!(validate_outcome_contract(&missing_run, "missing run").is_err());

        let mut missing_probability = valid.clone();
        missing_probability["prediction"]
            .as_object_mut()
            .unwrap()
            .remove("compatibility_probability");
        assert!(validate_outcome_contract(&missing_probability, "missing probability").is_err());

        let mut corrupt_digest = valid.clone();
        corrupt_digest["source"]["artifact_sha256"] = json!("not-a-digest");
        assert!(validate_outcome_contract(&corrupt_digest, "corrupt digest").is_err());

        let mut over_budget = valid.clone();
        over_budget["run"]["actual_tokens"] = json!(16001);
        assert!(validate_outcome_contract(&over_budget, "over budget").is_err());

        let mut invalid_reason = valid.clone();
        invalid_reason["adjudication"]["reviewer_decisions"][0]["reason_codes"] =
            json!(["not-a-registered-reason"]);
        assert!(validate_outcome_contract(&invalid_reason, "invalid reason code").is_err());

        let mut invalid_mismatch = valid.clone();
        invalid_mismatch["prediction"]["mismatch_codes"] = json!(["wrong-way"]);
        assert!(validate_outcome_contract(&invalid_mismatch, "invalid mismatch code").is_err());

        let mut invalid_precondition = valid.clone();
        invalid_precondition["prediction"]["precondition_checks"][0]["outcome"] =
            json!("probably-satisfied");
        assert!(validate_outcome_contract(&invalid_precondition, "invalid precondition outcome").is_err());

        let mut extra_field = valid;
        extra_field["unregistered_authority"] = json!(true);
        assert!(validate_outcome_contract(&extra_field, "extra field").is_err());
    }

    #[test]
    fn computes_transfer_selectivity_metrics_and_excludes_partial_rows() {
        let positive = outcome("p", "cp", "sf-p", "tf-p", Label::Compatible, Decision::Accept);
        let negative = outcome("n", "cn", "sf-n", "tf-n", Label::Incompatible, Decision::Reject);
        let report = analyze(&[positive, negative], Split::Test, "a".repeat(64), 1_000, 7)
            .expect("analyze synthetic fixture");
        assert_eq!(report.counts.binary_evaluable_pairs, 2);
        assert_eq!(report.metrics["compatible_accept_recall"].estimate, Some(1.0));
        assert_eq!(report.metrics["incompatible_false_accept_rate"].estimate, Some(0.0));
        assert_eq!(report.metrics["binary_coverage"].estimate, Some(1.0));
        assert_eq!(report.metrics["selective_risk"].estimate, Some(0.0));
        assert_eq!(report.calibration.probability_scored_pairs, 2);
        assert!(report.calibration.complete_binary_coverage);
        assert!(report.calibration.brier_score.is_some());
        assert!((report.calibration.brier_score.unwrap() - 0.01).abs() < 1e-12);
        assert!(report.calibration.brier_score_ci_95.is_some());
        assert!(report.calibration.expected_calibration_error_10_bins.is_some());
        assert!(report.calibration.calibration_error_ci_95.is_some());
        assert_eq!(
            report.proposed_gates.overall_confirmatory_decision,
            "not_evaluated_requires_independent_mutation_receipt_and_protocol_audit"
        );
    }

    #[test]
    fn abstention_is_not_silently_counted_as_accept_or_reject() {
        let positive = outcome("p", "cp", "sf-p", "tf-p", Label::Compatible, Decision::Abstain);
        let negative = outcome("n", "cn", "sf-n", "tf-n", Label::Incompatible, Decision::Abstain);
        let report = analyze(&[positive, negative], Split::Test, "b".repeat(64), 1_000, 8)
            .expect("analyze abstention fixture");
        assert_eq!(report.metrics["compatible_accept_recall"].estimate, Some(0.0));
        assert_eq!(report.metrics["incompatible_false_accept_rate"].estimate, Some(0.0));
        assert_eq!(report.metrics["binary_coverage"].estimate, Some(0.0));
        assert_eq!(report.metrics["selective_risk"].estimate, None);
        assert_eq!(report.calibration.probability_scored_pairs, 0);
        assert_eq!(report.calibration.probability_coverage, 0.0);
    }

    #[test]
    fn deterministic_bootstrap_yields_stable_intervals() {
        let outcomes = vec![
            outcome("p1", "c1", "sf1", "tf1", Label::Compatible, Decision::Accept),
            outcome("n1", "c2", "sf2", "tf2", Label::Incompatible, Decision::Accept),
            outcome("p2", "c3", "sf3", "tf3", Label::Compatible, Decision::Reject),
            outcome("n2", "c4", "sf4", "tf4", Label::Incompatible, Decision::Reject),
        ];
        let first = analyze(&outcomes, Split::Test, "c".repeat(64), 1_000, 42).unwrap();
        let second = analyze(&outcomes, Split::Test, "c".repeat(64), 1_000, 42).unwrap();
        assert_eq!(
            serde_json::to_value(first).unwrap(),
            serde_json::to_value(second).unwrap()
        );
    }

    #[test]
    fn family_component_floor_counts_components_not_pair_rows() {
        let shared = outcome(
            "p",
            "one-component",
            "shared-family",
            "target-p",
            Label::Compatible,
            Decision::Accept,
        );
        let different_edge = outcome(
            "n",
            "one-component",
            "target-p",
            "target-n",
            Label::Incompatible,
            Decision::Reject,
        );
        let report = analyze(
            &[shared, different_edge],
            Split::Test,
            "d".repeat(64),
            1_000,
            9,
        )
        .unwrap();
        assert_eq!(report.counts.compatible_components, 1);
        assert_eq!(report.counts.incompatible_components, 1);
        assert!(!report.proposed_gates.compatible_component_floor_met);
        assert!(!report.proposed_gates.incompatible_component_floor_met);
    }

    #[test]
    fn compatibility_probability_must_be_present_even_when_null_is_allowed() {
        let missing = serde_json::from_value::<Prediction>(json!({
            "decision": "accept",
            "predicted_label": "compatible"
        }));
        assert!(missing.is_err());

        let explicit_null = serde_json::from_value::<Prediction>(json!({
            "decision": "abstain",
            "predicted_label": "unknown",
            "compatibility_probability": null
        }));
        assert!(explicit_null.is_ok());

        let out_of_range = serde_json::from_value::<Prediction>(json!({
            "decision": "accept",
            "predicted_label": "compatible",
            "compatibility_probability": 1.01
        }));
        assert!(out_of_range.is_err());
    }

    #[test]
    fn quantile_uses_linear_interpolation() {
        assert!((quantile(&[0.0, 10.0], 0.25) - 2.5).abs() < f64::EPSILON);
    }
}
