//! Native NewtonBench trial importer and paired scorecard.
//!
//! This crate preserves the upstream RMSLE and judge-derived exact-accuracy
//! fields. It does not run NewtonBench, execute generated code, call an LLM,
//! or treat the symbolic judge as a formal proof.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const MANIFEST_SCHEMA: &str = "symthaea.newtonbench_native_pair_manifest.v1";
pub const REPORT_SCHEMA: &str = "symthaea.newtonbench_native_pair_scorecard.v1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpectedPairV1 {
    pub trial_id: u64,
    pub module_name: String,
    pub noise_level: f64,
    pub equation_difficulty: String,
    pub model_system: String,
    pub law_version: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ManifestV1 {
    pub schema: String,
    pub upstream_repository: String,
    /// For real imports, use the immutable upstream Git commit, not "main".
    pub upstream_revision: String,
    pub model_name: String,
    pub judge_model_name: String,
    pub runner_configuration_sha256: String,
    pub vanilla_backend: String,
    pub tool_backend: String,
    pub expected_pairs: Vec<ExpectedPairV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrialStatus {
    Completed,
    Failed,
    Invalid,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrialObservationV1 {
    pub status: TrialStatus,
    pub rmsle: Option<f64>,
    /// Native trial-level 0/1 score from the upstream symbolic-equivalence LLM judge.
    pub exact_accuracy: Option<f64>,
    pub rounds: Option<u64>,
    pub num_experiments: Option<u64>,
    pub total_tokens: Option<u64>,
    pub source_path: String,
    pub source_sha256: String,
    pub normalized_nonfinite_tokens: usize,
    pub issues: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImportProblemV1 {
    pub arm: String,
    pub source_path: String,
    pub source_sha256: String,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ArmSummaryV1 {
    pub arm: String,
    pub expected_pairs: usize,
    pub observed_pairs: usize,
    pub missing_pairs: usize,
    pub completed_trials: usize,
    pub failed_trials: usize,
    pub invalid_trials: usize,
    pub rmsle_available_completed: usize,
    pub mean_rmsle_completed: Option<f64>,
    pub exact_accuracy_available_completed: usize,
    pub mean_exact_accuracy_completed: Option<f64>,
    pub experiments_available_completed: usize,
    pub rounds_available_completed: usize,
    pub tokens_available_completed: usize,
    /// Sum of available values only; use the availability count to detect partial totals.
    pub sum_reported_experiments_completed: u64,
    pub sum_reported_rounds_completed: u64,
    pub sum_reported_tokens_completed: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PairedTrialV1 {
    pub expected: ExpectedPairV1,
    pub vanilla: Option<TrialObservationV1>,
    pub code_assisted: Option<TrialObservationV1>,
    /// code-assisted RMSLE minus vanilla RMSLE; negative is better.
    pub rmsle_delta_tool_minus_vanilla: Option<f64>,
    /// code-assisted judged exact accuracy minus vanilla judged exact accuracy.
    pub exact_accuracy_delta_tool_minus_vanilla: Option<f64>,
    pub experiments_delta_tool_minus_vanilla: Option<i64>,
    pub rounds_delta_tool_minus_vanilla: Option<i64>,
    pub tokens_delta_tool_minus_vanilla: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ComparisonSummaryV1 {
    pub expected_pairs: usize,
    pub both_observed_pairs: usize,
    pub both_completed_pairs: usize,
    pub rmsle_pair_count: usize,
    pub mean_rmsle_delta_tool_minus_vanilla: Option<f64>,
    pub exact_accuracy_pair_count: usize,
    pub mean_exact_accuracy_delta_tool_minus_vanilla: Option<f64>,
    pub experiment_count_pair_count: usize,
    pub mean_experiments_delta_tool_minus_vanilla: Option<f64>,
    pub rounds_pair_count: usize,
    pub mean_rounds_delta_tool_minus_vanilla: Option<f64>,
    pub tokens_pair_count: usize,
    pub mean_tokens_delta_tool_minus_vanilla: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct StratumSummaryV1 {
    pub module_name: String,
    pub noise_level: f64,
    pub equation_difficulty: String,
    pub model_system: String,
    pub law_version: String,
    pub expected_pairs: usize,
    pub both_observed_pairs: usize,
    pub both_completed_pairs: usize,
    pub rmsle_pair_count: usize,
    pub mean_rmsle_delta_tool_minus_vanilla: Option<f64>,
    pub exact_accuracy_pair_count: usize,
    pub mean_exact_accuracy_delta_tool_minus_vanilla: Option<f64>,
    pub experiment_count_pair_count: usize,
    pub mean_experiments_delta_tool_minus_vanilla: Option<f64>,
    pub rounds_pair_count: usize,
    pub mean_rounds_delta_tool_minus_vanilla: Option<f64>,
    pub tokens_pair_count: usize,
    pub mean_tokens_delta_tool_minus_vanilla: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ReportBodyV1 {
    pub schema: String,
    pub upstream_repository: String,
    pub upstream_revision: String,
    pub model_name: String,
    pub judge_model_name: String,
    pub runner_configuration_sha256: String,
    pub manifest_sha256: String,
    pub expected_pair_count: usize,
    pub vanilla: ArmSummaryV1,
    pub code_assisted: ArmSummaryV1,
    pub comparison: ComparisonSummaryV1,
    /// Secondary aggregate grouped by module and declared task strata.
    /// Do not let a pooled mean hide reversals.
    pub by_stratum: Vec<StratumSummaryV1>,
    pub paired_trials: Vec<PairedTrialV1>,
    pub import_problems: Vec<ImportProblemV1>,
    pub metric_semantics: Vec<String>,
    pub claim_boundary: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ScorecardReportV1 {
    pub body: ReportBodyV1,
    pub receipt_sha256: String,
}

impl ScorecardReportV1 {
    pub fn verify_receipt(&self) -> bool {
        serde_json::to_vec(&self.body)
            .map(|bytes| self.receipt_sha256 == sha256_hex(&bytes))
            .unwrap_or(false)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ArtifactInput {
    /// A normalized relative path within the input arm directory.
    pub path: String,
    /// Exact bytes as read from the source artifact.
    pub raw_bytes: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct StratumKey {
    module_name: String,
    noise_bits: u64,
    equation_difficulty: String,
    model_system: String,
    law_version: String,
}

impl StratumKey {
    fn from_expected(expected: &ExpectedPairV1) -> Result<Self, String> {
        let key = expected.key()?;
        Ok(Self {
            module_name: key.module_name,
            noise_bits: key.noise_bits,
            equation_difficulty: key.equation_difficulty,
            model_system: key.model_system,
            law_version: key.law_version,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct TrialKey {
    trial_id: u64,
    module_name: String,
    noise_bits: u64,
    equation_difficulty: String,
    model_system: String,
    law_version: String,
}

impl ExpectedPairV1 {
    fn key(&self) -> Result<TrialKey, String> {
        if !self.noise_level.is_finite() || self.noise_level < 0.0 {
            return Err(format!("invalid noise_level for trial {}", self.trial_id));
        }
        let noise = if self.noise_level == 0.0 { 0.0 } else { self.noise_level };
        Ok(TrialKey {
            trial_id: self.trial_id,
            module_name: self.module_name.clone(),
            noise_bits: noise.to_bits(),
            equation_difficulty: self.equation_difficulty.clone(),
            model_system: self.model_system.clone(),
            law_version: self.law_version.clone(),
        })
    }
}

fn nonblank(value: &str) -> bool {
    !value.trim().is_empty()
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for byte in digest {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_manifest(manifest: &ManifestV1) -> Result<BTreeMap<TrialKey, ExpectedPairV1>, String> {
    if manifest.schema != MANIFEST_SCHEMA {
        return Err(format!("unsupported manifest schema: {}", manifest.schema));
    }
    for (name, value) in [
        ("upstream_repository", manifest.upstream_repository.as_str()),
        ("upstream_revision", manifest.upstream_revision.as_str()),
        ("model_name", manifest.model_name.as_str()),
        ("judge_model_name", manifest.judge_model_name.as_str()),
        ("vanilla_backend", manifest.vanilla_backend.as_str()),
        ("tool_backend", manifest.tool_backend.as_str()),
    ] {
        if !nonblank(value) {
            return Err(format!("manifest {name} must not be blank"));
        }
    }
    if manifest.vanilla_backend == manifest.tool_backend {
        return Err("vanilla_backend and tool_backend must be distinct".to_owned());
    }
    if !valid_digest(&manifest.runner_configuration_sha256) {
        return Err("runner_configuration_sha256 must be lowercase SHA-256".to_owned());
    }
    if manifest.expected_pairs.is_empty() {
        return Err("manifest must declare at least one expected pair".to_owned());
    }

    let mut pairs = BTreeMap::new();
    for expected in &manifest.expected_pairs {
        for (name, value) in [
            ("module_name", expected.module_name.as_str()),
            ("equation_difficulty", expected.equation_difficulty.as_str()),
            ("model_system", expected.model_system.as_str()),
            ("law_version", expected.law_version.as_str()),
        ] {
            if !nonblank(value) {
                return Err(format!("expected pair {} has blank {name}", expected.trial_id));
            }
        }
        let key = expected.key()?;
        if pairs.insert(key, expected.clone()).is_some() {
            return Err(format!("duplicate expected trial identity: {}", expected.trial_id));
        }
    }
    Ok(pairs)
}

fn is_json_delimiter(byte: u8) -> bool {
    matches!(byte, b':' | b',' | b'[' | b']' | b'{' | b'}' | b' ' | b'\t' | b'\r' | b'\n')
}

/// Python's default json.dump may emit NaN/Infinity tokens, which are not RFC
/// JSON. Normalize only standalone tokens outside JSON strings to null; never
/// rewrite token-like text inside a string.
fn normalize_python_nonfinite_json(input: &str) -> (String, usize) {
    let bytes = input.as_bytes();
    let mut output = Vec::with_capacity(bytes.len());
    let mut index = 0;
    let mut in_string = false;
    let mut escaped = false;
    let mut replaced = 0;

    while index < bytes.len() {
        let byte = bytes[index];
        if in_string {
            output.push(byte);
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
            index += 1;
            continue;
        }

        if byte == b'"' {
            in_string = true;
            output.push(byte);
            index += 1;
            continue;
        }

        let candidates: [&[u8]; 3] = [b"-Infinity", b"Infinity", b"NaN"];
        let mut matched: Option<&[u8]> = None;
        for candidate in candidates {
            if bytes[index..].starts_with(candidate) {
                let before_ok = index == 0 || is_json_delimiter(bytes[index - 1]);
                let after_index = index + candidate.len();
                let after_ok = after_index == bytes.len() || is_json_delimiter(bytes[after_index]);
                if before_ok && after_ok {
                    matched = Some(candidate);
                    break;
                }
            }
        }
        if let Some(token) = matched {
            output.extend_from_slice(b"null");
            index += token.len();
            replaced += 1;
        } else {
            output.push(byte);
            index += 1;
        }
    }

    (String::from_utf8(output).expect("replacement preserves UTF-8"), replaced)
}

fn required_str<'a>(value: &'a Value, field: &str) -> Result<&'a str, String> {
    value
        .get(field)
        .and_then(Value::as_str)
        .filter(|item| nonblank(item))
        .ok_or_else(|| format!("missing or blank field {field}"))
}

fn required_u64(value: &Value, field: &str) -> Result<u64, String> {
    value
        .get(field)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("missing or invalid unsigned integer field {field}"))
}

fn optional_u64(value: &Value, field: &str) -> Option<u64> {
    value.get(field).and_then(Value::as_u64)
}

fn optional_metric(value: Option<&Value>, label: &str, issues: &mut Vec<String>) -> Option<f64> {
    let Some(value) = value else {
        issues.push(format!("{label} missing"));
        return None;
    };
    if value.is_null() {
        issues.push(format!("{label} unavailable or non-finite"));
        return None;
    }
    let Some(number) = value.as_f64() else {
        issues.push(format!("{label} is not numeric"));
        return None;
    };
    if !number.is_finite() || number < 0.0 {
        issues.push(format!("{label} is non-finite or negative"));
        return None;
    }
    Some(number)
}

fn expected_key_from_value(value: &Value) -> Result<TrialKey, String> {
    let trial_id = required_u64(value, "trial_id")?;
    let module_name = required_str(value, "module_name")?.to_owned();
    let noise_level = value
        .get("noise_level")
        .and_then(Value::as_f64)
        .ok_or_else(|| "missing or invalid field noise_level".to_owned())?;
    if !noise_level.is_finite() || noise_level < 0.0 {
        return Err("noise_level must be finite and non-negative".to_owned());
    }
    let noise_level = if noise_level == 0.0 { 0.0 } else { noise_level };
    Ok(TrialKey {
        trial_id,
        module_name,
        noise_bits: noise_level.to_bits(),
        equation_difficulty: required_str(value, "equation_difficulty")?.to_owned(),
        model_system: required_str(value, "model_system")?.to_owned(),
        law_version: required_str(value, "law_version")?.to_owned(),
    })
}

fn parse_artifact(
    arm: &str,
    artifact: &ArtifactInput,
    manifest: &ManifestV1,
) -> Result<(TrialKey, TrialObservationV1), String> {
    let input = std::str::from_utf8(&artifact.raw_bytes)
        .map_err(|error| format!("artifact is not UTF-8: {error}"))?;
    let (normalized, normalized_nonfinite_tokens) = normalize_python_nonfinite_json(input);
    let value: Value = serde_json::from_str(&normalized)
        .map_err(|error| format!("invalid trial JSON: {error}"))?;

    let key = expected_key_from_value(&value)?;
    if required_str(&value, "model_name")? != manifest.model_name {
        return Err(format!("model_name mismatch at {}", artifact.path));
    }
    if required_str(&value, "agent_backend")? != arm {
        return Err(format!(
            "agent_backend mismatch at {}: expected {arm}",
            artifact.path
        ));
    }
    if required_str(&value, "LLM judge")? != manifest.judge_model_name {
        return Err(format!("LLM judge mismatch at {}", artifact.path));
    }

    let status_text = value.get("status").and_then(Value::as_str);
    let (status, status_issue) = match status_text {
        Some("completed") => (TrialStatus::Completed, None),
        Some("failed") => (TrialStatus::Failed, None),
        Some(other) => (
            TrialStatus::Invalid,
            Some(format!("unknown upstream status {other:?}")),
        ),
        None => (TrialStatus::Invalid, Some("upstream status missing".to_owned())),
    };
    let mut issues = Vec::new();
    if let Some(issue) = status_issue {
        issues.push(issue);
    }
    if normalized_nonfinite_tokens > 0 {
        issues.push(format!(concat!(
            "normalized {normalized_nonfinite_tokens} Python-style non-finite JSON token(s); ",
            "null metrics remain unavailable"
        )));
    }

    let evaluation = value.get("evaluation").and_then(Value::as_object);
    let rmsle = optional_metric(
        evaluation.and_then(|evaluation| evaluation.get("rmsle")),
        "evaluation.rmsle",
        &mut issues,
    );
    let exact_raw = evaluation.and_then(|evaluation| evaluation.get("exact_accuracy"));
    let exact_accuracy = match exact_raw {
        Some(metric) if metric.is_null() => {
            issues.push("evaluation.exact_accuracy unavailable".to_owned());
            None
        }
        Some(metric) => match metric.as_f64() {
            Some(number) if number.is_finite() && (number == 0.0 || number == 1.0) => Some(number),
            Some(_) => {
                issues.push(
                    "evaluation.exact_accuracy must be native trial-level 0.0 or 1.0".to_owned(),
                );
                None
            }
            None => {
                issues.push("evaluation.exact_accuracy is not numeric".to_owned());
                None
            }
        },
        None => {
            issues.push("evaluation.exact_accuracy missing".to_owned());
            None
        }
    };

    Ok((
        key,
        TrialObservationV1 {
            status,
            rmsle,
            exact_accuracy,
            rounds: optional_u64(&value, "rounds"),
            num_experiments: optional_u64(&value, "num_experiments"),
            total_tokens: optional_u64(&value, "total_tokens"),
            source_path: artifact.path.clone(),
            source_sha256: sha256_hex(&artifact.raw_bytes),
            normalized_nonfinite_tokens,
            issues,
        },
    ))
}

fn mean(values: impl Iterator<Item = f64>) -> Option<f64> {
    let values = values.collect::<Vec<_>>();
    if values.is_empty() {
        None
    } else {
        Some(values.iter().sum::<f64>() / values.len() as f64)
    }
}

fn sum_optional(values: impl Iterator<Item = Option<u64>>) -> u64 {
    values.flatten().sum()
}

fn summarize_arm(
    arm: &str,
    expected_count: usize,
    rows: &BTreeMap<TrialKey, TrialObservationV1>,
) -> ArmSummaryV1 {
    let all = rows.values().collect::<Vec<_>>();
    let complete = all
        .iter()
        .filter(|trial| trial.status == TrialStatus::Completed)
        .copied()
        .collect::<Vec<_>>();
    let rmsle_values = complete.iter().filter_map(|trial| trial.rmsle).collect::<Vec<_>>();
    let accuracy_values = complete
        .iter()
        .filter_map(|trial| trial.exact_accuracy)
        .collect::<Vec<_>>();
    ArmSummaryV1 {
        arm: arm.to_owned(),
        expected_pairs: expected_count,
        observed_pairs: all.len(),
        missing_pairs: expected_count.saturating_sub(all.len()),
        completed_trials: all.iter().filter(|trial| trial.status == TrialStatus::Completed).count(),
        failed_trials: all.iter().filter(|trial| trial.status == TrialStatus::Failed).count(),
        invalid_trials: all.iter().filter(|trial| trial.status == TrialStatus::Invalid).count(),
        rmsle_available_completed: rmsle_values.len(),
        mean_rmsle_completed: mean(rmsle_values.into_iter()),
        exact_accuracy_available_completed: accuracy_values.len(),
        mean_exact_accuracy_completed: mean(accuracy_values.into_iter()),
        experiments_available_completed: complete
            .iter()
            .filter(|trial| trial.num_experiments.is_some())
            .count(),
        rounds_available_completed: complete
            .iter()
            .filter(|trial| trial.rounds.is_some())
            .count(),
        tokens_available_completed: complete
            .iter()
            .filter(|trial| trial.total_tokens.is_some())
            .count(),
        sum_reported_experiments_completed: sum_optional(
            complete.iter().map(|trial| trial.num_experiments),
        ),
        sum_reported_rounds_completed: sum_optional(complete.iter().map(|trial| trial.rounds)),
        sum_reported_tokens_completed: sum_optional(
            complete.iter().map(|trial| trial.total_tokens),
        ),
    }
}

fn signed_delta(a: Option<u64>, b: Option<u64>) -> Option<i64> {
    let left = i64::try_from(a?).ok()?;
    let right = i64::try_from(b?).ok()?;
    left.checked_sub(right)
}

fn summarize_stratum(key: StratumKey, rows: &[&PairedTrialV1]) -> StratumSummaryV1 {
    let rmsle = rows
        .iter()
        .filter_map(|row| row.rmsle_delta_tool_minus_vanilla)
        .collect::<Vec<_>>();
    let accuracy = rows
        .iter()
        .filter_map(|row| row.exact_accuracy_delta_tool_minus_vanilla)
        .collect::<Vec<_>>();
    let experiments = rows
        .iter()
        .filter_map(|row| row.experiments_delta_tool_minus_vanilla)
        .map(|value| value as f64)
        .collect::<Vec<_>>();
    let rounds = rows
        .iter()
        .filter_map(|row| row.rounds_delta_tool_minus_vanilla)
        .map(|value| value as f64)
        .collect::<Vec<_>>();
    let tokens = rows
        .iter()
        .filter_map(|row| row.tokens_delta_tool_minus_vanilla)
        .map(|value| value as f64)
        .collect::<Vec<_>>();
    StratumSummaryV1 {
        module_name: key.module_name,
        noise_level: f64::from_bits(key.noise_bits),
        equation_difficulty: key.equation_difficulty,
        model_system: key.model_system,
        law_version: key.law_version,
        expected_pairs: rows.len(),
        both_observed_pairs: rows
            .iter()
            .filter(|row| row.vanilla.is_some() && row.code_assisted.is_some())
            .count(),
        both_completed_pairs: rows.iter().filter(|row| matches!(
            (&row.vanilla, &row.code_assisted),
            (Some(vanilla), Some(tool))
                if vanilla.status == TrialStatus::Completed
                    && tool.status == TrialStatus::Completed
        )).count(),
        rmsle_pair_count: rmsle.len(),
        mean_rmsle_delta_tool_minus_vanilla: mean(rmsle.into_iter()),
        exact_accuracy_pair_count: accuracy.len(),
        mean_exact_accuracy_delta_tool_minus_vanilla: mean(accuracy.into_iter()),
        experiment_count_pair_count: experiments.len(),
        mean_experiments_delta_tool_minus_vanilla: mean(experiments.into_iter()),
        rounds_pair_count: rounds.len(),
        mean_rounds_delta_tool_minus_vanilla: mean(rounds.into_iter()),
        tokens_pair_count: tokens.len(),
        mean_tokens_delta_tool_minus_vanilla: mean(tokens.into_iter()),
    }
}

pub fn audit(
    manifest_bytes: &[u8],
    mut vanilla_artifacts: Vec<ArtifactInput>,
    mut tool_artifacts: Vec<ArtifactInput>,
) -> Result<ScorecardReportV1, String> {
    let manifest: ManifestV1 = serde_json::from_slice(manifest_bytes)
        .map_err(|error| format!("cannot parse manifest JSON: {error}"))?;
    let expected_map = validate_manifest(&manifest)?;
    let expected_keys = expected_map.keys().cloned().collect::<BTreeSet<_>>();

    vanilla_artifacts.sort_by(|a, b| a.path.cmp(&b.path));
    tool_artifacts.sort_by(|a, b| a.path.cmp(&b.path));

    let mut import_problems = Vec::new();
    let mut vanilla: BTreeMap<TrialKey, TrialObservationV1> = BTreeMap::new();
    let mut code_assisted: BTreeMap<TrialKey, TrialObservationV1> = BTreeMap::new();

    for (arm, artifacts, table) in [
        (manifest.vanilla_backend.as_str(), &vanilla_artifacts, &mut vanilla),
        (manifest.tool_backend.as_str(), &tool_artifacts, &mut code_assisted),
    ] {
        for artifact in artifacts {
            match parse_artifact(arm, artifact, &manifest) {
                Ok((key, trial)) => {
                    if !expected_keys.contains(&key) {
                        import_problems.push(ImportProblemV1 {
                            arm: arm.to_owned(),
                            source_path: artifact.path.clone(),
                            source_sha256: sha256_hex(&artifact.raw_bytes),
                            reason: "trial identity is not in frozen expected_pairs".to_owned(),
                        });
                        continue;
                    }
                    if table.insert(key.clone(), trial).is_some() {
                        return Err(format!(
                            "duplicate trial identity in arm {arm} for trial_id {} module {}",
                            key.trial_id, key.module_name
                        ));
                    }
                }
                Err(reason) => {
                    import_problems.push(ImportProblemV1 {
                        arm: arm.to_owned(),
                        source_path: artifact.path.clone(),
                        source_sha256: sha256_hex(&artifact.raw_bytes),
                        reason,
                    });
                }
            }
        }
    }

    let mut paired_trials = Vec::with_capacity(manifest.expected_pairs.len());
    let mut rmsle_deltas = Vec::new();
    let mut accuracy_deltas = Vec::new();
    let mut experiment_deltas = Vec::new();
    let mut round_deltas = Vec::new();
    let mut token_deltas = Vec::new();
    let mut both_observed_pairs = 0;
    let mut both_completed_pairs = 0;

    for expected in &manifest.expected_pairs {
        let key = expected.key()?;
        let vanilla_trial = vanilla.get(&key).cloned();
        let tool_trial = code_assisted.get(&key).cloned();
        if vanilla_trial.is_some() && tool_trial.is_some() {
            both_observed_pairs += 1;
        }
        let both_completed = matches!(
            (&vanilla_trial, &tool_trial),
            (Some(v), Some(t))
                if v.status == TrialStatus::Completed
                    && t.status == TrialStatus::Completed
        );
        if both_completed {
            both_completed_pairs += 1;
        }
        let rmsle_delta = match (&vanilla_trial, &tool_trial) {
            (Some(v), Some(t))
                if v.status == TrialStatus::Completed
                    && t.status == TrialStatus::Completed
                    && v.rmsle.is_some()
                    && t.rmsle.is_some() =>
            {
                let delta = t.rmsle.unwrap() - v.rmsle.unwrap();
                rmsle_deltas.push(delta);
                Some(delta)
            }
            _ => None,
        };
        let exact_delta = match (&vanilla_trial, &tool_trial) {
            (Some(v), Some(t))
                if v.status == TrialStatus::Completed
                    && t.status == TrialStatus::Completed
                    && v.exact_accuracy.is_some()
                    && t.exact_accuracy.is_some() =>
            {
                let delta = t.exact_accuracy.unwrap() - v.exact_accuracy.unwrap();
                accuracy_deltas.push(delta);
                Some(delta)
            }
            _ => None,
        };
        let experiments_delta = match (&vanilla_trial, &tool_trial) {
            (Some(v), Some(t))
                if v.status == TrialStatus::Completed && t.status == TrialStatus::Completed =>
            {
                let delta = signed_delta(t.num_experiments, v.num_experiments);
                if let Some(value) = delta {
                    experiment_deltas.push(value as f64);
                }
                delta
            }
            _ => None,
        };
        let rounds_delta = match (&vanilla_trial, &tool_trial) {
            (Some(v), Some(t))
                if v.status == TrialStatus::Completed && t.status == TrialStatus::Completed =>
            {
                let delta = signed_delta(t.rounds, v.rounds);
                if let Some(value) = delta {
                    round_deltas.push(value as f64);
                }
                delta
            }
            _ => None,
        };
        let tokens_delta = match (&vanilla_trial, &tool_trial) {
            (Some(v), Some(t))
                if v.status == TrialStatus::Completed && t.status == TrialStatus::Completed =>
            {
                let delta = signed_delta(t.total_tokens, v.total_tokens);
                if let Some(value) = delta {
                    token_deltas.push(value as f64);
                }
                delta
            }
            _ => None,
        };
        paired_trials.push(PairedTrialV1 {
            expected: expected.clone(),
            vanilla: vanilla_trial,
            code_assisted: tool_trial,
            rmsle_delta_tool_minus_vanilla: rmsle_delta,
            exact_accuracy_delta_tool_minus_vanilla: exact_delta,
            experiments_delta_tool_minus_vanilla: experiments_delta,
            rounds_delta_tool_minus_vanilla: rounds_delta,
            tokens_delta_tool_minus_vanilla: tokens_delta,
        });
    }

    let vanilla_summary = summarize_arm(
        &manifest.vanilla_backend,
        manifest.expected_pairs.len(),
        &vanilla,
    );
    let tool_summary = summarize_arm(
        &manifest.tool_backend,
        manifest.expected_pairs.len(),
        &code_assisted,
    );
    let comparison = ComparisonSummaryV1 {
        expected_pairs: manifest.expected_pairs.len(),
        both_observed_pairs,
        both_completed_pairs,
        rmsle_pair_count: rmsle_deltas.len(),
        mean_rmsle_delta_tool_minus_vanilla: mean(rmsle_deltas.into_iter()),
        exact_accuracy_pair_count: accuracy_deltas.len(),
        mean_exact_accuracy_delta_tool_minus_vanilla: mean(accuracy_deltas.into_iter()),
        experiment_count_pair_count: experiment_deltas.len(),
        mean_experiments_delta_tool_minus_vanilla: mean(experiment_deltas.into_iter()),
        rounds_pair_count: round_deltas.len(),
        mean_rounds_delta_tool_minus_vanilla: mean(round_deltas.into_iter()),
        tokens_pair_count: token_deltas.len(),
        mean_tokens_delta_tool_minus_vanilla: mean(token_deltas.into_iter()),
    };

    let mut grouped: BTreeMap<StratumKey, Vec<&PairedTrialV1>> = BTreeMap::new();
    for paired in &paired_trials {
        grouped
            .entry(StratumKey::from_expected(&paired.expected)?)
            .or_default()
            .push(paired);
    }
    let by_stratum = grouped
        .into_iter()
        .map(|(key, rows)| summarize_stratum(key, &rows))
        .collect();

    let body = ReportBodyV1 {
        schema: REPORT_SCHEMA.to_owned(),
        upstream_repository: manifest.upstream_repository,
        upstream_revision: manifest.upstream_revision,
        model_name: manifest.model_name,
        judge_model_name: manifest.judge_model_name,
        runner_configuration_sha256: manifest.runner_configuration_sha256,
        manifest_sha256: sha256_hex(manifest_bytes),
        expected_pair_count: manifest.expected_pairs.len(),
        vanilla: vanilla_summary,
        code_assisted: tool_summary,
        comparison,
        by_stratum,
        paired_trials,
        import_problems,
        metric_semantics: vec![
            concat!(
                "evaluation.rmsle is preserved as upstream RMSLE; ",
                "it is not converted to NRMSE."
            )
            .to_owned(),
            concat!(
                "Upstream RMSLE masks NaN points and applies absolute values before log1p; ",
                "this importer does not recompute it."
            )
            .to_owned(),
            concat!(
                "evaluation.exact_accuracy is the upstream trial-level output of an LLM ",
                "symbolic-equivalence judge, not a formal proof."
            )
            .to_owned(),
            concat!(
                "This importer trusts the upstream agent_backend label; it does not authenticate ",
                "that code execution or a tool actually occurred."
            )
            .to_owned(),
            concat!(
                "False-confidence and premature-commitment metrics are not provided by this ",
                "native artifact schema and are not inferred."
            )
            .to_owned(),
        ],
        claim_boundary: concat!(
            "Native NewtonBench artifact import and matched scorecard only. No experiment was ",
            "executed by this importer; source-reported backend labels and judge outputs are ",
            "not independently authenticated."
        )
        .to_owned(),
    };
    let receipt_sha256 = sha256_hex(
        &serde_json::to_vec(&body)
            .map_err(|error| format!("cannot serialize native scorecard report: {error}"))?,
    );
    Ok(ScorecardReportV1 { body, receipt_sha256 })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn manifest() -> ManifestV1 {
        ManifestV1 {
            schema: MANIFEST_SCHEMA.to_owned(),
            upstream_repository: "HKUST-KnowComp/NewtonBench".to_owned(),
            upstream_revision: "fixture-only".to_owned(),
            model_name: "fixture-model".to_owned(),
            judge_model_name: "fixture-judge".to_owned(),
            runner_configuration_sha256: "a".repeat(64),
            vanilla_backend: "vanilla_agent".to_owned(),
            tool_backend: "code_assisted_agent".to_owned(),
            expected_pairs: vec![
                ExpectedPairV1 {
                    trial_id: 1,
                    module_name: "m0_gravity".to_owned(),
                    noise_level: 0.0,
                    equation_difficulty: "easy".to_owned(),
                    model_system: "vanilla_equation".to_owned(),
                    law_version: "v0".to_owned(),
                },
                ExpectedPairV1 {
                    trial_id: 2,
                    module_name: "m0_gravity".to_owned(),
                    noise_level: 0.1,
                    equation_difficulty: "hard".to_owned(),
                    model_system: "simple_system".to_owned(),
                    law_version: "v1".to_owned(),
                },
            ],
        }
    }

    fn artifact(path: &str, value: Value) -> ArtifactInput {
        ArtifactInput {
            path: path.to_owned(),
            raw_bytes: serde_json::to_vec(&value).unwrap(),
        }
    }

    fn raw_trial(
        trial_id: u64,
        noise: f64,
        difficulty: &str,
        system: &str,
        law_version: &str,
        backend: &str,
        status: &str,
        rmsle: Value,
        accuracy: f64,
        experiments: u64,
    ) -> Value {
        json!({
            "trial_id": trial_id,
            "module_name": "m0_gravity",
            "noise_level": noise,
            "model_name": "fixture-model",
            "equation_difficulty": difficulty,
            "model_system": system,
            "law_version": law_version,
            "agent_backend": backend,
            "LLM judge": "fixture-judge",
            "status": status,
            "rounds": 4,
            "total_tokens": 1000,
            "num_experiments": experiments,
            "evaluation": {
                "rmsle": rmsle,
                "exact_accuracy": accuracy,
                "symbolic_equivalent": accuracy == 1.0
            }
        })
    }

    fn sample_inputs() -> (Vec<u8>, Vec<ArtifactInput>, Vec<ArtifactInput>) {
        let manifest = manifest();
        let bytes = serde_json::to_vec(&manifest).unwrap();
        let vanilla = vec![
            artifact(
                "trial1.json",
                raw_trial(
                    1,
                    0.0,
                    "easy",
                    "vanilla_equation",
                    "v0",
                    "vanilla_agent",
                    "completed",
                    json!(0.5),
                    0.0,
                    3,
                ),
            ),
            artifact(
                "trial2_fail.json",
                raw_trial(
                    2,
                    0.1,
                    "hard",
                    "simple_system",
                    "v1",
                    "vanilla_agent",
                    "failed",
                    Value::Null,
                    0.0,
                    0,
                ),
            ),
        ];
        let tool = vec![
            artifact(
                "trial1.json",
                raw_trial(
                    1,
                    0.0,
                    "easy",
                    "vanilla_equation",
                    "v0",
                    "code_assisted_agent",
                    "completed",
                    json!(0.2),
                    1.0,
                    2,
                ),
            ),
        ];
        (bytes, vanilla, tool)
    }

    #[test]
    fn preserves_native_metrics_and_pairs_complete_trial() {
        let (manifest_bytes, vanilla, tool) = sample_inputs();
        let report = audit(&manifest_bytes, vanilla, tool).unwrap();
        assert!(report.verify_receipt());
        assert_eq!(report.body.expected_pair_count, 2);
        assert_eq!(report.body.vanilla.observed_pairs, 2);
        assert_eq!(report.body.code_assisted.observed_pairs, 1);
        assert_eq!(report.body.code_assisted.missing_pairs, 1);
        assert_eq!(report.body.vanilla.failed_trials, 1);
        assert_eq!(report.body.comparison.both_observed_pairs, 1);
        assert_eq!(report.body.comparison.rmsle_pair_count, 1);
        assert!(
            (report.body.comparison.mean_rmsle_delta_tool_minus_vanilla.unwrap() + 0.3).abs()
                < 1.0e-12
        );
        assert_eq!(report.body.comparison.exact_accuracy_pair_count, 1);
        assert_eq!(report.body.comparison.mean_exact_accuracy_delta_tool_minus_vanilla, Some(1.0));
        assert_eq!(report.body.comparison.mean_experiments_delta_tool_minus_vanilla, Some(-1.0));
        assert_eq!(report.body.by_stratum.len(), 2);
        let first_stratum = report.body.by_stratum.iter()
            .find(|stratum| stratum.equation_difficulty == "easy")
            .unwrap();
        assert_eq!(first_stratum.expected_pairs, 1);
        assert_eq!(first_stratum.rmsle_pair_count, 1);
        assert_eq!(first_stratum.mean_rmsle_delta_tool_minus_vanilla, Some(-0.3));
        let second_stratum = report.body.by_stratum.iter()
            .find(|stratum| stratum.equation_difficulty == "hard")
            .unwrap();
        assert_eq!(second_stratum.expected_pairs, 1);
        assert_eq!(second_stratum.both_observed_pairs, 0);
        assert_eq!(second_stratum.rmsle_pair_count, 0);
    }

    #[test]
    fn missing_cost_fields_are_counted_not_silently_treated_as_complete_totals() {
        let (m, mut vanilla, tool) = sample_inputs();
        let mut value: Value = serde_json::from_slice(&vanilla[0].raw_bytes).unwrap();
        value.as_object_mut().unwrap().remove("total_tokens");
        vanilla[0].raw_bytes = serde_json::to_vec(&value).unwrap();

        let report = audit(&m, vanilla, tool).unwrap();
        let summary = &report.body.vanilla;
        assert_eq!(summary.completed_trials, 1);
        assert_eq!(summary.tokens_available_completed, 0);
        assert_eq!(summary.sum_reported_tokens_completed, 0);
        assert_eq!(report.body.comparison.tokens_pair_count, 0);
        assert_eq!(report.body.comparison.mean_tokens_delta_tool_minus_vanilla, None);
    }

    #[test]
    fn normalizes_python_nan_failure_without_scoring_it_as_zero() {
        let m = serde_json::to_vec(&manifest()).unwrap();
        let vanilla = vec![ArtifactInput {
            path: "trial2_fail.json".to_owned(),
            raw_bytes: br#"{
                "trial_id": 2,
                "module_name": "m0_gravity",
                "noise_level": 0.1,
                "model_name": "fixture-model",
                "equation_difficulty": "hard",
                "model_system": "simple_system",
                "law_version": "v1",
                "agent_backend": "vanilla_agent",
                "LLM judge": "fixture-judge",
                "status": "failed",
                "rounds": 0,
                "total_tokens": 0,
                "num_experiments": 0,
                "evaluation": {"rmsle": NaN, "exact_accuracy": 0.0}
            }"#
            .to_vec(),
        }];
        let report = audit(&m, vanilla, Vec::new()).unwrap();
        assert_eq!(report.body.vanilla.failed_trials, 1);
        assert_eq!(report.body.vanilla.rmsle_available_completed, 0);
        assert_eq!(report.body.comparison.rmsle_pair_count, 0);
        let trial = report.body.paired_trials.iter().find(|p| p.expected.trial_id == 2).unwrap();
        let observed = trial.vanilla.as_ref().unwrap();
        assert_eq!(observed.rmsle, None);
        assert_eq!(observed.normalized_nonfinite_tokens, 1);
    }

    #[test]
    fn nonfinite_token_normalization_never_rewrites_string_contents() {
        let input = r#"{"text":"NaN Infinity -Infinity","a":NaN,"b":Infinity,"c":-Infinity}"#;
        let (normalized, count) = normalize_python_nonfinite_json(input);
        assert_eq!(count, 3);
        assert!(normalized.contains(r#""text":"NaN Infinity -Infinity""#));
        assert!(normalized.contains(r#""a":null"#));
        assert!(normalized.contains(r#""b":null"#));
        assert!(normalized.contains(r#""c":null"#));
        assert!(serde_json::from_str::<Value>(&normalized).is_ok());
    }

    #[test]
    fn rejects_duplicate_expected_pair_identity() {
        let mut manifest = manifest();
        manifest.expected_pairs.push(manifest.expected_pairs[0].clone());
        assert!(
            validate_manifest(&manifest)
                .unwrap_err()
                .contains("duplicate expected trial identity")
        );
    }

    #[test]
    fn duplicate_observed_trial_in_one_arm_is_rejected() {
        let (m, mut vanilla, tool) = sample_inputs();
        vanilla.push(vanilla[0].clone());
        assert!(audit(&m, vanilla, tool).unwrap_err().contains("duplicate trial identity"));
    }

    #[test]
    fn wrong_backend_is_reported_and_remains_missing() {
        let m = serde_json::to_vec(&manifest()).unwrap();
        let vanilla = vec![artifact(
            "trial1.json",
            raw_trial(
                1,
                0.0,
                "easy",
                "vanilla_equation",
                "v0",
                "code_assisted_agent",
                "completed",
                json!(0.2),
                1.0,
                2,
            ),
        )];
        let report = audit(&m, vanilla, Vec::new()).unwrap();
        assert_eq!(report.body.vanilla.observed_pairs, 0);
        assert_eq!(report.body.vanilla.missing_pairs, 2);
        assert_eq!(report.body.import_problems.len(), 1);
    }

    #[test]
    fn malformed_json_is_retained_as_problem_and_expected_pair_stays_missing() {
        let m = serde_json::to_vec(&manifest()).unwrap();
        let vanilla = vec![ArtifactInput {
            path: "trial1.json".to_owned(),
            raw_bytes: b"{ broken".to_vec(),
        }];
        let report = audit(&m, vanilla, Vec::new()).unwrap();
        assert_eq!(report.body.vanilla.observed_pairs, 0);
        assert_eq!(report.body.vanilla.missing_pairs, 2);
        assert_eq!(report.body.import_problems.len(), 1);
        assert_eq!(report.body.import_problems[0].source_sha256, sha256_hex(b"{ broken"));
    }

    #[test]
    fn report_receipt_detects_tampering() {
        let (m, vanilla, tool) = sample_inputs();
        let mut report = audit(&m, vanilla, tool).unwrap();
        report.body.expected_pair_count += 1;
        assert!(!report.verify_receipt());
    }
}
