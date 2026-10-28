//! Fail-closed audit of paired scientific tool-allocation trial receipts.
//!
//! The auditor consumes externally produced JSONL receipts. It does not execute
//! an LLM, invoke NewtonBench, select tools, authenticate execution, or establish
//! scientific truth.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const MANIFEST_SCHEMA: &str = "symthaea.scientific_tool_allocation_manifest.v1";
pub const TRIAL_SCHEMA: &str = "symthaea.scientific_tool_allocation_trial.v1";
pub const REPORT_SCHEMA: &str = "symthaea.scientific_tool_allocation_audit.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Treatment {
    NoTool,
    AlwaysTool,
    StaticThreshold,
    HeuristicVoc,
    CalibratedVoc,
}

impl Treatment {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NoTool => "no_tool",
            Self::AlwaysTool => "always_tool",
            Self::StaticThreshold => "static_threshold",
            Self::HeuristicVoc => "heuristic_voc",
            Self::CalibratedVoc => "calibrated_voc",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TerminalStatus {
    Completed,
    InfrastructureFailure,
    Invalid,
    NotEvaluable,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskSeedV1 {
    pub task_id: String,
    pub domain: String,
    pub law_complexity: String,
    pub system_complexity: String,
    pub noise_level: f64,
    pub seed: u64,
    pub tool_available: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ManifestV1 {
    pub schema: String,
    pub protocol_id: String,
    pub model_id: String,
    pub model_revision: String,
    pub prompt_sha256: String,
    pub task_manifest_sha256: String,
    pub tool_registry_sha256: String,
    pub treatments: Vec<Treatment>,
    pub task_seeds: Vec<TaskSeedV1>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrialV1 {
    pub schema: String,
    pub protocol_id: String,
    /// SHA-256 of the exact manifest bytes, not a re-serialized manifest.
    pub manifest_sha256: String,
    pub task_id: String,
    pub seed: u64,
    pub treatment: Treatment,
    pub terminal_status: TerminalStatus,
    /// Required only for Completed trials.
    pub law_recovered: Option<bool>,
    /// Independently evaluated NRMSE on a declared held-out set.
    pub heldout_prediction_nrmse: Option<f64>,
    /// Confident unique result that was wrong or unsupported.
    pub false_confident_singleton: Option<bool>,
    /// Commitment before protocol-required discriminating evidence was obtained.
    pub premature_commitment: Option<bool>,
    pub tool_calls: u64,
    pub experiment_queries: u64,
    pub tokens: u64,
    pub elapsed_ms: u64,
    /// Digest of the policy output/receipt, when one exists.
    pub policy_output_sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MissingCellV1 {
    pub task_id: String,
    pub seed: u64,
    pub treatment: Treatment,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TreatmentSummaryV1 {
    pub treatment: Treatment,
    pub assigned_runs: usize,
    pub observed_runs: usize,
    pub missing_runs: usize,
    pub completed_runs: usize,
    pub infrastructure_failures: usize,
    pub invalid_runs: usize,
    pub not_evaluable_runs: usize,
    pub successful_recoveries: usize,
    /// Successful completions / all assigned cells; missing or failed cells stay in denominator.
    pub recovery_rate_all_assigned: f64,
    /// Recovery rate among completed rows; None if none completed.
    pub recovery_rate_completed: Option<f64>,
    pub false_confident_singletons: usize,
    pub false_confidence_denominator_completed: usize,
    pub premature_commitments: usize,
    pub premature_commitment_denominator_completed: usize,
    pub mean_heldout_prediction_nrmse_completed: Option<f64>,
    pub total_tool_calls_observed: u64,
    pub total_experiment_queries_observed: u64,
    pub total_tokens_observed: u64,
    pub total_elapsed_ms_observed: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PairedDeltaV1 {
    pub treatment: Treatment,
    pub expected_pairs: usize,
    pub complete_pairs: usize,
    pub incomplete_pairs: usize,
    /// Mean (treatment recovery - no-tool recovery) on complete pairs.
    pub mean_recovery_delta_vs_no_tool: Option<f64>,
    /// Mean (treatment NRMSE - no-tool NRMSE); negative is better.
    pub mean_heldout_nrmse_delta_vs_no_tool: Option<f64>,
    /// Mean (treatment premature - no-tool premature) on complete pairs.
    pub mean_premature_commitment_delta_vs_no_tool: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ReportBodyV1 {
    pub schema: String,
    pub protocol_id: String,
    pub manifest_sha256: String,
    pub trials_sha256: String,
    pub expected_cell_count: usize,
    pub observed_cell_count: usize,
    pub missing_cell_count: usize,
    pub missing_cells: Vec<MissingCellV1>,
    pub treatment_summaries: Vec<TreatmentSummaryV1>,
    pub paired_deltas_vs_no_tool: Vec<PairedDeltaV1>,
    pub claim_boundary: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AuditReportV1 {
    pub body: ReportBodyV1,
    /// SHA-256 of deterministic compact JSON serialization of body.
    pub receipt_sha256: String,
}

impl AuditReportV1 {
    pub fn verify_receipt(&self) -> bool {
        serde_json::to_vec(&self.body)
            .map(|bytes| self.receipt_sha256 == sha256_hex(&bytes))
            .unwrap_or(false)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct CellKey {
    task_id: String,
    seed: u64,
    treatment: Treatment,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct PairKey {
    task_id: String,
    seed: u64,
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

fn nonblank(value: &str) -> bool {
    !value.trim().is_empty()
}

fn flag(value: bool) -> f64 {
    if value { 1.0 } else { 0.0 }
}

fn validate_manifest(manifest: &ManifestV1) -> Result<(), String> {
    if manifest.schema != MANIFEST_SCHEMA {
        return Err(format!("unsupported manifest schema: {}", manifest.schema));
    }
    for (name, value) in [
        ("protocol_id", manifest.protocol_id.as_str()),
        ("model_id", manifest.model_id.as_str()),
        ("model_revision", manifest.model_revision.as_str()),
    ] {
        if !nonblank(value) {
            return Err(format!("manifest {name} must not be blank"));
        }
    }
    for (name, value) in [
        ("prompt_sha256", manifest.prompt_sha256.as_str()),
        ("task_manifest_sha256", manifest.task_manifest_sha256.as_str()),
        ("tool_registry_sha256", manifest.tool_registry_sha256.as_str()),
    ] {
        if !valid_digest(value) {
            return Err(format!("manifest {name} must be lowercase SHA-256"));
        }
    }
    if manifest.treatments.len() < 2 {
        return Err("manifest must declare no_tool and at least one comparison treatment".to_owned());
    }
    let mut treatments = BTreeSet::new();
    for treatment in &manifest.treatments {
        if !treatments.insert(*treatment) {
            return Err(format!("duplicate declared treatment: {}", treatment.as_str()));
        }
    }
    if !treatments.contains(&Treatment::NoTool) {
        return Err("manifest must include no_tool as the paired control".to_owned());
    }
    if manifest.task_seeds.is_empty() {
        return Err("manifest must declare at least one task/seed".to_owned());
    }
    let mut pairs = BTreeSet::new();
    for task in &manifest.task_seeds {
        if !nonblank(&task.task_id)
            || !nonblank(&task.domain)
            || !nonblank(&task.law_complexity)
            || !nonblank(&task.system_complexity)
        {
            return Err("task/seed metadata must have nonblank IDs and strata".to_owned());
        }
        if !task.noise_level.is_finite() || task.noise_level < 0.0 {
            return Err(format!("invalid noise level for task {}", task.task_id));
        }
        if !pairs.insert(PairKey {
            task_id: task.task_id.clone(),
            seed: task.seed,
        }) {
            return Err(format!(
                "duplicate task/seed assignment: {} seed {}",
                task.task_id, task.seed
            ));
        }
    }
    Ok(())
}

fn validate_trial(
    trial: &TrialV1,
    manifest: &ManifestV1,
    manifest_sha: &str,
    task: &TaskSeedV1,
) -> Result<(), String> {
    if trial.schema != TRIAL_SCHEMA {
        return Err(format!("unsupported trial schema: {}", trial.schema));
    }
    if trial.protocol_id != manifest.protocol_id {
        return Err(format!("protocol ID mismatch for task {}", trial.task_id));
    }
    if trial.manifest_sha256 != manifest_sha {
        return Err(format!("exact manifest digest mismatch for task {}", trial.task_id));
    }
    if !manifest.treatments.contains(&trial.treatment) {
        return Err(format!(
            "undeclared treatment {} for task {}",
            trial.treatment.as_str(),
            trial.task_id
        ));
    }
    if let Some(digest) = &trial.policy_output_sha256 {
        if !valid_digest(digest) {
            return Err(format!("invalid policy output digest for task {}", trial.task_id));
        }
    }
    if trial.treatment == Treatment::NoTool && trial.tool_calls != 0 {
        return Err(format!("no_tool arm called a tool for task {}", trial.task_id));
    }
    if !task.tool_available && trial.tool_calls != 0 {
        return Err(format!(
            "tool call recorded where tools were unavailable for task {}",
            trial.task_id
        ));
    }
    if trial.terminal_status == TerminalStatus::Completed {
        if task.tool_available && trial.treatment == Treatment::AlwaysTool && trial.tool_calls == 0 {
            return Err(format!(
                "always_tool arm made no tool call for task {}",
                trial.task_id
            ));
        }
        if trial.law_recovered.is_none()
            || trial.heldout_prediction_nrmse.is_none()
            || trial.false_confident_singleton.is_none()
            || trial.premature_commitment.is_none()
        {
            return Err(format!(
                "completed trial has missing outcome fields for task {}",
                trial.task_id
            ));
        }
        let error = trial.heldout_prediction_nrmse.unwrap();
        if !error.is_finite() || error < 0.0 {
            return Err(format!("invalid held-out NRMSE for task {}", trial.task_id));
        }
        if trial.policy_output_sha256.is_none() {
            return Err(format!(
                "completed trial lacks policy output digest for task {}",
                trial.task_id
            ));
        }
    } else if trial.law_recovered.is_some()
        || trial.heldout_prediction_nrmse.is_some()
        || trial.false_confident_singleton.is_some()
        || trial.premature_commitment.is_some()
    {
        return Err(format!(
            "non-completed trial contains scored outcomes for task {}",
            trial.task_id
        ));
    }
    Ok(())
}

fn mean(values: impl Iterator<Item = f64>) -> Option<f64> {
    let values = values.collect::<Vec<_>>();
    if values.is_empty() {
        None
    } else {
        Some(values.iter().sum::<f64>() / values.len() as f64)
    }
}

pub fn audit(manifest_bytes: &[u8], trial_bytes: &[u8]) -> Result<AuditReportV1, String> {
    let manifest: ManifestV1 = serde_json::from_slice(manifest_bytes)
        .map_err(|error| format!("cannot parse manifest JSON: {error}"))?;
    validate_manifest(&manifest)?;
    let manifest_sha = sha256_hex(manifest_bytes);
    let trials_text = std::str::from_utf8(trial_bytes)
        .map_err(|error| format!("trial JSONL is not UTF-8: {error}"))?;

    let mut trial_lookup: BTreeMap<CellKey, TrialV1> = BTreeMap::new();
    for (line_index, line) in trials_text.lines().enumerate() {
        if line.trim().is_empty() {
            return Err(format!("blank trial line at {}", line_index + 1));
        }
        let trial: TrialV1 = serde_json::from_str(line)
            .map_err(|error| format!("cannot parse trial line {}: {error}", line_index + 1))?;
        let pair = PairKey {
            task_id: trial.task_id.clone(),
            seed: trial.seed,
        };
        let task = manifest
            .task_seeds
            .iter()
            .find(|task| task.task_id == pair.task_id && task.seed == pair.seed)
            .ok_or_else(|| {
                format!(
                    "trial references undeclared task/seed {} seed {}",
                    pair.task_id, pair.seed
                )
            })?;
        validate_trial(&trial, &manifest, &manifest_sha, task)?;
        let key = CellKey {
            task_id: trial.task_id.clone(),
            seed: trial.seed,
            treatment: trial.treatment,
        };
        if trial_lookup.insert(key, trial).is_some() {
            return Err(format!(
                "duplicate trial cell for task {} seed {}",
                pair.task_id, pair.seed
            ));
        }
    }

    let mut missing_cells = Vec::new();
    for task in &manifest.task_seeds {
        for treatment in &manifest.treatments {
            let key = CellKey {
                task_id: task.task_id.clone(),
                seed: task.seed,
                treatment: *treatment,
            };
            if !trial_lookup.contains_key(&key) {
                missing_cells.push(MissingCellV1 {
                    task_id: task.task_id.clone(),
                    seed: task.seed,
                    treatment: *treatment,
                });
            }
        }
    }

    let assigned_runs = manifest.task_seeds.len();
    let mut treatment_summaries = Vec::with_capacity(manifest.treatments.len());
    for treatment in &manifest.treatments {
        let selected = trial_lookup
            .iter()
            .filter(|(key, _)| key.treatment == *treatment)
            .map(|(_, trial)| trial)
            .collect::<Vec<_>>();
        let completed = selected
            .iter()
            .filter(|trial| trial.terminal_status == TerminalStatus::Completed)
            .collect::<Vec<_>>();
        let successful_recoveries = completed
            .iter()
            .filter(|trial| trial.law_recovered == Some(true))
            .count();
        let false_confident_singletons = completed
            .iter()
            .filter(|trial| trial.false_confident_singleton == Some(true))
            .count();
        let premature_commitments = completed
            .iter()
            .filter(|trial| trial.premature_commitment == Some(true))
            .count();
        let mean_heldout = mean(
            completed
                .iter()
                .map(|trial| trial.heldout_prediction_nrmse.unwrap()),
        );
        treatment_summaries.push(TreatmentSummaryV1 {
            treatment: *treatment,
            assigned_runs,
            observed_runs: selected.len(),
            missing_runs: assigned_runs.saturating_sub(selected.len()),
            completed_runs: completed.len(),
            infrastructure_failures: selected
                .iter()
                .filter(|trial| trial.terminal_status == TerminalStatus::InfrastructureFailure)
                .count(),
            invalid_runs: selected
                .iter()
                .filter(|trial| trial.terminal_status == TerminalStatus::Invalid)
                .count(),
            not_evaluable_runs: selected
                .iter()
                .filter(|trial| trial.terminal_status == TerminalStatus::NotEvaluable)
                .count(),
            successful_recoveries,
            recovery_rate_all_assigned: successful_recoveries as f64 / assigned_runs as f64,
            recovery_rate_completed: if completed.is_empty() {
                None
            } else {
                Some(successful_recoveries as f64 / completed.len() as f64)
            },
            false_confident_singletons,
            false_confidence_denominator_completed: completed.len(),
            premature_commitments,
            premature_commitment_denominator_completed: completed.len(),
            mean_heldout_prediction_nrmse_completed: mean_heldout,
            total_tool_calls_observed: selected.iter().map(|trial| trial.tool_calls).sum(),
            total_experiment_queries_observed: selected.iter().map(|trial| trial.experiment_queries).sum(),
            total_tokens_observed: selected.iter().map(|trial| trial.tokens).sum(),
            total_elapsed_ms_observed: selected.iter().map(|trial| trial.elapsed_ms).sum(),
        });
    }

    let mut paired_deltas = Vec::with_capacity(manifest.treatments.len());
    for treatment in &manifest.treatments {
        let mut recovery_deltas = Vec::new();
        let mut nrmse_deltas = Vec::new();
        let mut premature_deltas = Vec::new();
        for task in &manifest.task_seeds {
            let control_key = CellKey {
                task_id: task.task_id.clone(),
                seed: task.seed,
                treatment: Treatment::NoTool,
            };
            let treatment_key = CellKey {
                task_id: task.task_id.clone(),
                seed: task.seed,
                treatment: *treatment,
            };
            let (Some(control_trial), Some(treatment_trial)) = (
                trial_lookup.get(&control_key),
                trial_lookup.get(&treatment_key),
            ) else {
                continue;
            };
            if control_trial.terminal_status != TerminalStatus::Completed
                || treatment_trial.terminal_status != TerminalStatus::Completed
            {
                continue;
            }
            recovery_deltas.push(
                flag(treatment_trial.law_recovered.unwrap())
                    - flag(control_trial.law_recovered.unwrap()),
            );
            nrmse_deltas.push(
                treatment_trial.heldout_prediction_nrmse.unwrap()
                    - control_trial.heldout_prediction_nrmse.unwrap(),
            );
            premature_deltas.push(
                flag(treatment_trial.premature_commitment.unwrap())
                    - flag(control_trial.premature_commitment.unwrap()),
            );
        }
        let complete_pairs = recovery_deltas.len();
        paired_deltas.push(PairedDeltaV1 {
            treatment: *treatment,
            expected_pairs: assigned_runs,
            complete_pairs,
            incomplete_pairs: assigned_runs.saturating_sub(complete_pairs),
            mean_recovery_delta_vs_no_tool: mean(recovery_deltas.into_iter()),
            mean_heldout_nrmse_delta_vs_no_tool: mean(nrmse_deltas.into_iter()),
            mean_premature_commitment_delta_vs_no_tool: mean(premature_deltas.into_iter()),
        });
    }

    let expected_cell_count = manifest
        .task_seeds
        .len()
        .checked_mul(manifest.treatments.len())
        .ok_or_else(|| "manifest cell count overflows usize".to_owned())?;
    let body = ReportBodyV1 {
        schema: REPORT_SCHEMA.to_owned(),
        protocol_id: manifest.protocol_id,
        manifest_sha256: manifest_sha,
        trials_sha256: sha256_hex(trial_bytes),
        expected_cell_count,
        observed_cell_count: trial_lookup.len(),
        missing_cell_count: missing_cells.len(),
        missing_cells,
        treatment_summaries,
        paired_deltas_vs_no_tool: paired_deltas,
        claim_boundary: "Offline paired-trial audit only. No agent/tool runner was executed by this auditor. Digests bind input bytes and report content but do not authenticate execution or establish scientific truth.".to_owned(),
    };
    let receipt_sha256 = sha256_hex(
        &serde_json::to_vec(&body)
            .map_err(|error| format!("cannot serialize audit report: {error}"))?,
    );
    Ok(AuditReportV1 {
        body,
        receipt_sha256,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(ch: char) -> String {
        std::iter::repeat(ch).take(64).collect()
    }

    fn manifest() -> ManifestV1 {
        ManifestV1 {
            schema: MANIFEST_SCHEMA.to_owned(),
            protocol_id: "sci-tool-alloc-pilot-001".to_owned(),
            model_id: "test-model".to_owned(),
            model_revision: "rev-001".to_owned(),
            prompt_sha256: digest('a'),
            task_manifest_sha256: digest('b'),
            tool_registry_sha256: digest('c'),
            treatments: vec![
                Treatment::NoTool,
                Treatment::AlwaysTool,
                Treatment::StaticThreshold,
                Treatment::HeuristicVoc,
                Treatment::CalibratedVoc,
            ],
            task_seeds: vec![
                TaskSeedV1 {
                    task_id: "law-simple".to_owned(),
                    domain: "mechanics".to_owned(),
                    law_complexity: "easy".to_owned(),
                    system_complexity: "simple".to_owned(),
                    noise_level: 0.0,
                    seed: 11,
                    tool_available: true,
                },
                TaskSeedV1 {
                    task_id: "law-hard".to_owned(),
                    domain: "oscillations".to_owned(),
                    law_complexity: "hard".to_owned(),
                    system_complexity: "complex".to_owned(),
                    noise_level: 0.01,
                    seed: 29,
                    tool_available: false,
                },
            ],
        }
    }

    fn trials_for(manifest: &ManifestV1, manifest_bytes: &[u8]) -> Vec<TrialV1> {
        let manifest_digest = sha256_hex(manifest_bytes);
        let mut trials = Vec::new();
        for task in &manifest.task_seeds {
            for treatment in &manifest.treatments {
                let recovered = task.task_id == "law-simple";
                let tool_calls = if !task.tool_available || *treatment == Treatment::NoTool {
                    0
                } else {
                    match treatment {
                        Treatment::NoTool => 0,
                        Treatment::AlwaysTool => 1,
                        Treatment::StaticThreshold => 1,
                        Treatment::HeuristicVoc => 2,
                        Treatment::CalibratedVoc => 1,
                    }
                };
                let error = match treatment {
                    Treatment::NoTool => 0.5,
                    Treatment::AlwaysTool => 0.2,
                    Treatment::StaticThreshold => 0.35,
                    Treatment::HeuristicVoc => 0.45,
                    Treatment::CalibratedVoc => 0.15,
                };
                let completed = !(task.task_id == "law-hard" && *treatment == Treatment::CalibratedVoc);
                trials.push(TrialV1 {
                    schema: TRIAL_SCHEMA.to_owned(),
                    protocol_id: manifest.protocol_id.clone(),
                    manifest_sha256: manifest_digest.clone(),
                    task_id: task.task_id.clone(),
                    seed: task.seed,
                    treatment: *treatment,
                    terminal_status: if completed {
                        TerminalStatus::Completed
                    } else {
                        TerminalStatus::InfrastructureFailure
                    },
                    law_recovered: completed.then_some(recovered),
                    heldout_prediction_nrmse: completed.then_some(error),
                    false_confident_singleton: completed.then_some(!recovered && *treatment == Treatment::NoTool),
                    premature_commitment: completed.then_some(*treatment == Treatment::NoTool && !recovered),
                    tool_calls,
                    experiment_queries: 12,
                    tokens: 4_000 + tool_calls * 100,
                    elapsed_ms: 500 + tool_calls * 10,
                    policy_output_sha256: completed.then(|| {
                        sha256_hex(format!("{}:{}", task.task_id, treatment.as_str()).as_bytes())
                    }),
                });
            }
        }
        trials
    }

    fn input_bytes(drop_cell: Option<(String, u64, Treatment)>) -> (Vec<u8>, Vec<u8>) {
        let manifest = manifest();
        let manifest_bytes = serde_json::to_vec(&manifest).unwrap();
        let mut trials = trials_for(&manifest, &manifest_bytes);
        if let Some((task_id, seed, treatment)) = drop_cell {
            trials.retain(|trial| {
                !(trial.task_id == task_id && trial.seed == seed && trial.treatment == treatment)
            });
        }
        let trial_bytes = trials
            .iter()
            .map(|trial| serde_json::to_string(trial).unwrap())
            .collect::<Vec<_>>()
            .join("\n")
            .into_bytes();
        (manifest_bytes, trial_bytes)
    }

    #[test]
    fn reports_cells_failures_and_paired_deltas() {
        let (manifest_bytes, trial_bytes) = input_bytes(None);
        let report = audit(&manifest_bytes, &trial_bytes).unwrap();
        assert!(report.verify_receipt());
        assert_eq!(report.body.expected_cell_count, 10);
        assert_eq!(report.body.observed_cell_count, 10);
        assert_eq!(report.body.missing_cell_count, 0);

        let calibrated = report
            .body
            .treatment_summaries
            .iter()
            .find(|summary| summary.treatment == Treatment::CalibratedVoc)
            .unwrap();
        assert_eq!(calibrated.completed_runs, 1);
        assert_eq!(calibrated.infrastructure_failures, 1);
        assert_eq!(calibrated.assigned_runs, 2);
        assert_eq!(calibrated.recovery_rate_all_assigned, 0.5);

        let paired = report
            .body
            .paired_deltas_vs_no_tool
            .iter()
            .find(|summary| summary.treatment == Treatment::AlwaysTool)
            .unwrap();
        assert_eq!(paired.expected_pairs, 2);
        assert_eq!(paired.complete_pairs, 2);
        assert!((paired.mean_heldout_nrmse_delta_vs_no_tool.unwrap() + 0.3).abs() < 1.0e-12);

        let first = serde_json::to_vec(&report).unwrap();
        let second = serde_json::to_vec(&audit(&manifest_bytes, &trial_bytes).unwrap()).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn missing_cell_keeps_manifest_denominator_and_marks_pair_incomplete() {
        let (manifest_bytes, trial_bytes) = input_bytes(Some((
            "law-hard".to_owned(),
            29,
            Treatment::HeuristicVoc,
        )));
        let report = audit(&manifest_bytes, &trial_bytes).unwrap();
        assert_eq!(report.body.expected_cell_count, 10);
        assert_eq!(report.body.observed_cell_count, 9);
        assert_eq!(report.body.missing_cell_count, 1);
        let summary = report
            .body
            .treatment_summaries
            .iter()
            .find(|summary| summary.treatment == Treatment::HeuristicVoc)
            .unwrap();
        assert_eq!(summary.assigned_runs, 2);
        assert_eq!(summary.observed_runs, 1);
        assert_eq!(summary.missing_runs, 1);
        let paired = report
            .body
            .paired_deltas_vs_no_tool
            .iter()
            .find(|summary| summary.treatment == Treatment::HeuristicVoc)
            .unwrap();
        assert_eq!(paired.expected_pairs, 2);
        assert_eq!(paired.complete_pairs, 1);
        assert_eq!(paired.incomplete_pairs, 1);
    }

    #[test]
    fn rejects_duplicate_cells() {
        let (manifest_bytes, trial_bytes) = input_bytes(None);
        let original = String::from_utf8(trial_bytes).unwrap();
        let first = original.lines().next().unwrap();
        let duplicate = format!("{original}\n{first}");
        assert!(
            audit(&manifest_bytes, duplicate.as_bytes())
                .unwrap_err()
                .contains("duplicate trial cell")
        );
    }

    #[test]
    fn rejects_changed_manifest_receipt_binding() {
        let (mut manifest_bytes, trial_bytes) = input_bytes(None);
        manifest_bytes.push(b' ');
        assert!(
            audit(&manifest_bytes, &trial_bytes)
                .unwrap_err()
                .contains("exact manifest digest mismatch")
        );
    }

    #[test]
    fn rejects_no_tool_leakage_and_always_tool_omission() {
        let (manifest_bytes, trial_bytes) = input_bytes(None);
        let mut rows = String::from_utf8(trial_bytes)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<TrialV1>(line).unwrap())
            .collect::<Vec<_>>();
        rows[0].tool_calls = 1;
        let bytes = rows
            .iter()
            .map(|row| serde_json::to_string(row).unwrap())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            audit(&manifest_bytes, bytes.as_bytes())
                .unwrap_err()
                .contains("no_tool arm called a tool")
        );

        let (manifest_bytes, trial_bytes) = input_bytes(None);
        let mut rows = String::from_utf8(trial_bytes)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<TrialV1>(line).unwrap())
            .collect::<Vec<_>>();
        let target = rows
            .iter_mut()
            .find(|row| row.task_id == "law-simple" && row.treatment == Treatment::AlwaysTool)
            .unwrap();
        target.tool_calls = 0;
        let bytes = rows
            .iter()
            .map(|row| serde_json::to_string(row).unwrap())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            audit(&manifest_bytes, bytes.as_bytes())
                .unwrap_err()
                .contains("always_tool arm made no tool call")
        );
    }

    #[test]
    fn rejects_malformed_digests_and_invalid_scores() {
        let (manifest_bytes, trial_bytes) = input_bytes(None);
        let mut rows = String::from_utf8(trial_bytes)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<TrialV1>(line).unwrap())
            .collect::<Vec<_>>();
        rows[0].policy_output_sha256 = Some("not-a-digest".to_owned());
        let bytes = rows
            .iter()
            .map(|row| serde_json::to_string(row).unwrap())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            audit(&manifest_bytes, bytes.as_bytes())
                .unwrap_err()
                .contains("invalid policy output digest")
        );

        let (manifest_bytes, trial_bytes) = input_bytes(None);
        let mut rows = String::from_utf8(trial_bytes)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<TrialV1>(line).unwrap())
            .collect::<Vec<_>>();
        rows[0].heldout_prediction_nrmse = Some(-0.1);
        let bytes = rows
            .iter()
            .map(|row| serde_json::to_string(row).unwrap())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            audit(&manifest_bytes, bytes.as_bytes())
                .unwrap_err()
                .contains("invalid held-out NRMSE")
        );
    }

    #[test]
    fn receipt_detects_report_tampering() {
        let (manifest_bytes, trial_bytes) = input_bytes(None);
        let mut report = audit(&manifest_bytes, &trial_bytes).unwrap();
        report.body.missing_cell_count += 1;
        assert!(!report.verify_receipt());
    }

    #[test]
    fn rejects_invalid_terminal_outcome_semantics() {
        let (manifest_bytes, trial_bytes) = input_bytes(None);
        let mut rows = String::from_utf8(trial_bytes)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<TrialV1>(line).unwrap())
            .collect::<Vec<_>>();
        let target = rows
            .iter_mut()
            .find(|row| row.terminal_status == TerminalStatus::InfrastructureFailure)
            .unwrap();
        target.law_recovered = Some(false);
        let bytes = rows
            .iter()
            .map(|row| serde_json::to_string(row).unwrap())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            audit(&manifest_bytes, bytes.as_bytes())
                .unwrap_err()
                .contains("non-completed trial contains scored outcomes")
        );
    }

    #[test]
    fn manifest_schema_and_pair_ids_are_strict() {
        let mut invalid = manifest();
        invalid.treatments.push(Treatment::NoTool);
        assert!(validate_manifest(&invalid).unwrap_err().contains("duplicate declared treatment"));

        let mut invalid = manifest();
        invalid.task_seeds.push(invalid.task_seeds[0].clone());
        assert!(validate_manifest(&invalid).unwrap_err().contains("duplicate task/seed assignment"));
    }
}
