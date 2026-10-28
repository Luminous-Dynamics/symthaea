// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Fail-closed evaluation and promotion contracts for evolving memory pipelines.
//!
//! This module does not generate or modify a pipeline. It defines evidence rules
//! around candidate lineage, optimizer-visible feedback, and held-out promotion
//! so an outer-loop optimizer cannot quietly tune against test scores.
//!
//! Structural validation cannot prove that a runner told the truth. Receipts
//! must still be independently captured and bound to trusted evaluator outputs.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const PIPELINE_EVOLUTION_SCHEMA: &str = "BROCA-PIPELINE-EVOLUTION-V1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateState {
    Draft,
    DevelopmentQualified,
    Frozen,
    Promoted,
    Rejected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvaluationSplit {
    Development,
    Calibration,
    HeldOut,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReceiptStatus {
    Completed,
    Failed,
    Invalid,
    Skipped,
    NotRun,
}

#[derive(Serialize)]
struct CandidateManifestBody<'a> {
    candidate_id: &'a str,
    parent_candidate_id: &'a Option<String>,
    source_sha256: &'a str,
    retrieval_config_sha256: &'a str,
    lifecycle_policy_sha256: &'a str,
    prompt_sha256: &'a str,
    edit_iteration: u32,
    built_from_receipt_ids: &'a [String],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PipelineCandidate {
    pub candidate_id: String,
    pub parent_candidate_id: Option<String>,
    /// Digest of the canonical candidate manifest, produced before evaluation.
    pub manifest_sha256: String,
    pub source_sha256: String,
    pub retrieval_config_sha256: String,
    pub lifecycle_policy_sha256: String,
    pub prompt_sha256: String,
    pub state: CandidateState,
    pub edit_iteration: u32,
    /// Receipts visible to candidate generation/optimization. Held-out receipts
    /// are forbidden in this list.
    pub built_from_receipt_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaluationReceipt {
    pub receipt_id: String,
    /// Hash of this receipt with this field set to the empty string.
    pub receipt_sha256: String,
    pub candidate_id: String,
    pub candidate_manifest_sha256: String,
    pub benchmark_id: String,
    pub split: EvaluationSplit,
    pub status: ReceiptStatus,
    pub dataset_manifest_sha256: String,
    pub case_set_sha256: String,
    pub answer_model_sha256: String,
    pub reader_prompt_sha256: String,
    pub evaluator_sha256: String,
    pub harness_sha256: String,
    /// Digest of shared evaluation settings. Candidate retrieval parameters are
    /// method-specific and are pinned by candidate_manifest_sha256 instead.
    pub protocol_sha256: String,
    pub paired_group_count: u64,
    pub accuracy: f64,
    /// Full lifecycle cost, not just query latency.
    pub lifecycle_cost_units: f64,
    pub mean_query_latency_ms: f64,
    pub critical_governance_events: u64,
    /// Required on candidate-vs-baseline held-out receipts.
    pub baseline_candidate_id: Option<String>,
    /// Paired task-success lift relative to baseline, in percentage points.
    pub paired_lift_pp: Option<f64>,
    /// Lower endpoint of the preregistered 95% grouped paired interval, pp.
    pub paired_ci95_lower_pp: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PipelineEvolutionTrace {
    pub schema: String,
    pub benchmark_id: String,
    pub root_candidate_id: String,
    pub candidates: Vec<PipelineCandidate>,
    pub receipts: Vec<EvaluationReceipt>,
    /// Explicit receipts supplied to the optimizer. Held-out receipts may be
    /// stored for audit but must never be exposed to candidate generation.
    pub optimizer_visible_receipt_ids: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PromotionPolicy {
    pub min_quality_lift_pp: f64,
    pub non_inferiority_margin_pp: f64,
    pub max_cost_ratio_for_efficiency_win: f64,
    pub min_independent_groups: u64,
}

impl Default for PromotionPolicy {
    fn default() -> Self {
        Self {
            min_quality_lift_pp: 3.0,
            non_inferiority_margin_pp: 1.0,
            max_cost_ratio_for_efficiency_win: 0.75,
            min_independent_groups: 30,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PromotionOutcome {
    QualityWin,
    EfficiencyWin,
    Rejected,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PromotionAssessment {
    pub outcome: PromotionOutcome,
    pub candidate_id: String,
    pub baseline_candidate_id: String,
    pub paired_lift_pp: f64,
    pub paired_ci95_lower_pp: Option<f64>,
    pub lifecycle_cost_ratio: f64,
    pub reason: String,
}


/// Compute the digest binding a candidate's immutable system and derivation
/// inputs. Mutable orchestration state (Draft/Frozen/Promoted) is intentionally
/// excluded so evaluation does not alter the artifact identity.
pub fn candidate_manifest_sha256(candidate: &PipelineCandidate) -> Result<String, String> {
    let body = CandidateManifestBody {
        candidate_id: &candidate.candidate_id,
        parent_candidate_id: &candidate.parent_candidate_id,
        source_sha256: &candidate.source_sha256,
        retrieval_config_sha256: &candidate.retrieval_config_sha256,
        lifecycle_policy_sha256: &candidate.lifecycle_policy_sha256,
        prompt_sha256: &candidate.prompt_sha256,
        edit_iteration: candidate.edit_iteration,
        built_from_receipt_ids: &candidate.built_from_receipt_ids,
    };
    let bytes = serde_json::to_vec(&body)
        .map_err(|error| format!("cannot serialize candidate manifest: {error}"))?;
    Ok(sha256_hex(&bytes))
}


/// Digest the full evaluation receipt after blanking its own digest field.
/// Struct serialization has a fixed field order; the schema version pins this shape.
pub fn evaluation_receipt_sha256(receipt: &EvaluationReceipt) -> Result<String, String> {
    let mut digest_input = receipt.clone();
    digest_input.receipt_sha256.clear();
    let bytes = serde_json::to_vec(&digest_input)
        .map_err(|error| format!("cannot serialize evaluation receipt: {error}"))?;
    Ok(sha256_hex(&bytes))
}

fn seal_receipt(receipt: &mut EvaluationReceipt) {
    receipt.receipt_sha256 =
        evaluation_receipt_sha256(receipt).expect("evaluation receipt serialization is infallible");
}

/// Validate candidate lineage, receipt references, digests and optimizer-visible
/// splits. Held-out receipts may exist in the trace but cannot be optimization
/// feedback or a candidate's build source.
pub fn validate_evolution_trace(trace: &PipelineEvolutionTrace) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    if trace.schema != PIPELINE_EVOLUTION_SCHEMA {
        errors.push("unsupported pipeline evolution schema".to_string());
    }
    if trace.benchmark_id.trim().is_empty() {
        errors.push("benchmark_id must be non-empty".to_string());
    }

    let mut candidates: BTreeMap<&str, &PipelineCandidate> = BTreeMap::new();
    for candidate in &trace.candidates {
        if candidate.candidate_id.trim().is_empty() {
            errors.push("candidate_id must be non-empty".to_string());
            continue;
        }
        if candidates.insert(candidate.candidate_id.as_str(), candidate).is_some() {
            errors.push(format!("duplicate candidate_id: {}", candidate.candidate_id));
        }
        for (name, digest) in [
            ("manifest_sha256", &candidate.manifest_sha256),
            ("source_sha256", &candidate.source_sha256),
            ("retrieval_config_sha256", &candidate.retrieval_config_sha256),
            ("lifecycle_policy_sha256", &candidate.lifecycle_policy_sha256),
            ("prompt_sha256", &candidate.prompt_sha256),
        ] {
            if !is_sha256(digest) {
                errors.push(format!("candidate {} has invalid {name}", candidate.candidate_id));
            }
        }
        match candidate_manifest_sha256(candidate) {
            Ok(expected) if expected != candidate.manifest_sha256 => errors.push(format!(
                "candidate {} immutable manifest digest mismatch",
                candidate.candidate_id
            )),
            Err(error) => errors.push(format!(
                "candidate {} manifest serialization failed: {error}",
                candidate.candidate_id
            )),
            _ => {}
        }
    }

    match candidates.get(trace.root_candidate_id.as_str()) {
        None => errors.push("root_candidate_id does not exist".to_string()),
        Some(root) => {
            if root.parent_candidate_id.is_some() {
                errors.push("root candidate must not have a parent".to_string());
            }
            if root.edit_iteration != 0 {
                errors.push("root candidate edit_iteration must be zero".to_string());
            }
        }
    }

    for candidate in &trace.candidates {
        if candidate.candidate_id == trace.root_candidate_id {
            continue;
        }
        match candidate.parent_candidate_id.as_deref() {
            None => errors.push(format!(
                "non-root candidate {} must name a parent",
                candidate.candidate_id
            )),
            Some(parent) if parent == candidate.candidate_id => errors.push(format!(
                "candidate {} cannot parent itself",
                candidate.candidate_id
            )),
            Some(parent) if !candidates.contains_key(parent) => errors.push(format!(
                "candidate {} references missing parent {parent}",
                candidate.candidate_id
            )),
            Some(parent) => {
                if let Some(parent_candidate) = candidates.get(parent) {
                    if candidate.edit_iteration <= parent_candidate.edit_iteration {
                        errors.push(format!(
                            "candidate {} edit_iteration must exceed its parent's iteration",
                            candidate.candidate_id
                        ));
                    }
                }
            }
        }
    }

    // Detect cycles independently of a claimed root / iteration.
    let mut completed = BTreeSet::new();
    for candidate in &trace.candidates {
        let mut active = BTreeSet::new();
        let mut cursor = Some(candidate.candidate_id.as_str());
        while let Some(id) = cursor {
            if completed.contains(id) {
                break;
            }
            if !active.insert(id) {
                errors.push(format!("candidate parent cycle detected at {id}"));
                break;
            }
            cursor = candidates
                .get(id)
                .and_then(|item| item.parent_candidate_id.as_deref());
        }
        completed.extend(active);
    }

    let mut receipts: BTreeMap<&str, &EvaluationReceipt> = BTreeMap::new();
    for receipt in &trace.receipts {
        if receipt.receipt_id.trim().is_empty() {
            errors.push("receipt_id must be non-empty".to_string());
            continue;
        }
        if !is_sha256(&receipt.receipt_sha256) {
            errors.push(format!("receipt {} has invalid receipt_sha256", receipt.receipt_id));
        } else {
            match evaluation_receipt_sha256(receipt) {
                Ok(expected) if expected != receipt.receipt_sha256 => errors.push(format!(
                    "receipt {} receipt digest mismatch",
                    receipt.receipt_id
                )),
                Err(error) => errors.push(format!(
                    "receipt {} digest serialization failed: {error}",
                    receipt.receipt_id
                )),
                _ => {}
            }
        }
        if receipts.insert(receipt.receipt_id.as_str(), receipt).is_some() {
            errors.push(format!("duplicate receipt_id: {}", receipt.receipt_id));
        }
        if receipt.benchmark_id != trace.benchmark_id {
            errors.push(format!(
                "receipt {} benchmark_id differs from trace",
                receipt.receipt_id
            ));
        }
        match candidates.get(receipt.candidate_id.as_str()) {
            None => errors.push(format!(
                "receipt {} references unknown candidate {}",
                receipt.receipt_id, receipt.candidate_id
            )),
            Some(candidate) if receipt.candidate_manifest_sha256 != candidate.manifest_sha256 => {
                errors.push(format!(
                    "receipt {} candidate manifest digest mismatch",
                    receipt.receipt_id
                ));
            }
            _ => {}
        }
        for (name, digest) in [
            ("candidate_manifest_sha256", &receipt.candidate_manifest_sha256),
            ("dataset_manifest_sha256", &receipt.dataset_manifest_sha256),
            ("case_set_sha256", &receipt.case_set_sha256),
            ("answer_model_sha256", &receipt.answer_model_sha256),
            ("reader_prompt_sha256", &receipt.reader_prompt_sha256),
            ("evaluator_sha256", &receipt.evaluator_sha256),
            ("harness_sha256", &receipt.harness_sha256),
            ("protocol_sha256", &receipt.protocol_sha256),
        ] {
            if !is_sha256(digest) {
                errors.push(format!("receipt {} has invalid {name}", receipt.receipt_id));
            }
        }
        if !receipt.accuracy.is_finite() || !(0.0..=1.0).contains(&receipt.accuracy) {
            errors.push(format!("receipt {} accuracy must be in [0, 1]", receipt.receipt_id));
        }
        if !receipt.lifecycle_cost_units.is_finite() || receipt.lifecycle_cost_units < 0.0 {
            errors.push(format!(
                "receipt {} lifecycle cost must be finite and non-negative",
                receipt.receipt_id
            ));
        }
        if !receipt.mean_query_latency_ms.is_finite() || receipt.mean_query_latency_ms < 0.0 {
            errors.push(format!(
                "receipt {} latency must be finite and non-negative",
                receipt.receipt_id
            ));
        }
        if receipt.split != EvaluationSplit::HeldOut
            && (receipt.paired_lift_pp.is_some()
                || receipt.paired_ci95_lower_pp.is_some()
                || receipt.baseline_candidate_id.is_some())
        {
            errors.push(format!(
                "non-held-out receipt {} must not contain promotion contrast fields",
                receipt.receipt_id
            ));
        }
    }

    let mut visible_ids = BTreeSet::new();
    for receipt_id in &trace.optimizer_visible_receipt_ids {
        if !visible_ids.insert(receipt_id.as_str()) {
            errors.push(format!("duplicate optimizer-visible receipt {receipt_id}"));
            continue;
        }
        match receipts.get(receipt_id.as_str()) {
            None => errors.push(format!("optimizer references missing receipt {receipt_id}")),
            Some(receipt) if receipt.split == EvaluationSplit::HeldOut => errors.push(format!(
                "held-out receipt {receipt_id} is visible to the optimizer"
            )),
            Some(receipt) if receipt.status != ReceiptStatus::Completed => errors.push(format!(
                "optimizer-visible receipt {receipt_id} is not completed"
            )),
            _ => {}
        }
    }

    for candidate in &trace.candidates {
        let mut seen = BTreeSet::new();
        for receipt_id in &candidate.built_from_receipt_ids {
            if !seen.insert(receipt_id.as_str()) {
                errors.push(format!(
                    "candidate {} repeats build receipt {receipt_id}",
                    candidate.candidate_id
                ));
                continue;
            }
            match receipts.get(receipt_id.as_str()) {
                None => errors.push(format!(
                    "candidate {} references missing build receipt {receipt_id}",
                    candidate.candidate_id
                )),
                Some(receipt) if receipt.split == EvaluationSplit::HeldOut => errors.push(format!(
                    "candidate {} was built from held-out receipt {receipt_id}",
                    candidate.candidate_id
                )),
                Some(receipt) if receipt.status != ReceiptStatus::Completed => errors.push(format!(
                    "candidate {} was built from non-completed receipt {receipt_id}",
                    candidate.candidate_id
                )),
                _ => {}
            }
        }
    }

    if errors.is_empty() { Ok(()) } else { Err(errors) }
}

/// Qualify a frozen candidate against a frozen/promoted baseline using one
/// matched held-out evaluation protocol. This is a policy gate, not a scorer.
pub fn assess_promotion(
    trace: &PipelineEvolutionTrace,
    candidate_id: &str,
    baseline_id: &str,
    candidate_receipt_id: &str,
    baseline_receipt_id: &str,
    policy: PromotionPolicy,
) -> Result<PromotionAssessment, Vec<String>> {
    validate_evolution_trace(trace)?;

    let candidates: BTreeMap<&str, &PipelineCandidate> = trace.candidates
        .iter().map(|candidate| (candidate.candidate_id.as_str(), candidate)).collect();
    let receipts: BTreeMap<&str, &EvaluationReceipt> = trace.receipts
        .iter().map(|receipt| (receipt.receipt_id.as_str(), receipt)).collect();

    let mut errors = Vec::new();
    let candidate = candidates.get(candidate_id).copied();
    let baseline = candidates.get(baseline_id).copied();
    let candidate_receipt = receipts.get(candidate_receipt_id).copied();
    let baseline_receipt = receipts.get(baseline_receipt_id).copied();
    let (Some(candidate), Some(baseline), Some(candidate_receipt), Some(baseline_receipt)) =
        (candidate, baseline, candidate_receipt, baseline_receipt)
    else {
        return Err(vec!["candidate, baseline and both receipts must exist".to_string()]);
    };

    if candidate_id == baseline_id {
        errors.push("candidate and baseline must differ".to_string());
    }
    if candidate.state != CandidateState::Frozen {
        errors.push("candidate must be frozen before held-out assessment".to_string());
    }
    if !matches!(baseline.state, CandidateState::Frozen | CandidateState::Promoted) {
        errors.push("baseline must be frozen or previously promoted".to_string());
    }
    if candidate_receipt.candidate_id != candidate_id || baseline_receipt.candidate_id != baseline_id {
        errors.push("receipt candidate IDs do not match requested candidates".to_string());
    }
    if candidate_receipt.status != ReceiptStatus::Completed || baseline_receipt.status != ReceiptStatus::Completed {
        errors.push("both receipts must be completed".to_string());
    }
    if candidate_receipt.split != EvaluationSplit::HeldOut || baseline_receipt.split != EvaluationSplit::HeldOut {
        errors.push("promotion requires held-out receipts".to_string());
    }
    if candidate_receipt.baseline_candidate_id.as_deref() != Some(baseline_id) {
        errors.push("candidate receipt must identify the same baseline candidate".to_string());
    }
    if candidate_receipt.paired_lift_pp.is_none() || candidate_receipt.paired_ci95_lower_pp.is_none() {
        errors.push("candidate receipt requires paired lift and 95% lower confidence bound".to_string());
    }

    let matched = [
        ("benchmark", candidate_receipt.benchmark_id == baseline_receipt.benchmark_id),
        ("dataset manifest", candidate_receipt.dataset_manifest_sha256 == baseline_receipt.dataset_manifest_sha256),
        ("case set", candidate_receipt.case_set_sha256 == baseline_receipt.case_set_sha256),
        ("answer model", candidate_receipt.answer_model_sha256 == baseline_receipt.answer_model_sha256),
        ("reader prompt", candidate_receipt.reader_prompt_sha256 == baseline_receipt.reader_prompt_sha256),
        ("evaluator", candidate_receipt.evaluator_sha256 == baseline_receipt.evaluator_sha256),
        ("harness", candidate_receipt.harness_sha256 == baseline_receipt.harness_sha256),
        ("evaluation protocol", candidate_receipt.protocol_sha256 == baseline_receipt.protocol_sha256),
        ("independent group count", candidate_receipt.paired_group_count == baseline_receipt.paired_group_count),
    ];
    for (label, same) in matched {
        if !same {
            errors.push(format!("candidate and baseline have mismatched {label}"));
        }
    }

    if candidate_receipt.paired_group_count < policy.min_independent_groups
        || baseline_receipt.paired_group_count < policy.min_independent_groups
    {
        errors.push(format!("promotion needs at least {} independent groups", policy.min_independent_groups));
    }
    if candidate_receipt.critical_governance_events != 0 {
        errors.push("candidate has critical governance events".to_string());
    }
    if baseline_receipt.lifecycle_cost_units <= 0.0 {
        errors.push("baseline lifecycle cost must be positive for cost comparison".to_string());
    }
    if !policy.min_quality_lift_pp.is_finite() || policy.min_quality_lift_pp < 0.0
        || !policy.non_inferiority_margin_pp.is_finite() || policy.non_inferiority_margin_pp < 0.0
        || !policy.max_cost_ratio_for_efficiency_win.is_finite()
        || !(0.0..=1.0).contains(&policy.max_cost_ratio_for_efficiency_win)
        || policy.min_independent_groups == 0
    {
        errors.push("promotion policy contains invalid thresholds".to_string());
    }
    if !errors.is_empty() {
        return Err(errors);
    }

    let observed_lift = (candidate_receipt.accuracy - baseline_receipt.accuracy) * 100.0;
    let paired_lift = candidate_receipt.paired_lift_pp.unwrap_or(f64::NAN);
    let ci_lower = candidate_receipt.paired_ci95_lower_pp.unwrap_or(f64::NAN);
    if !paired_lift.is_finite() || !ci_lower.is_finite() || (observed_lift - paired_lift).abs() > 0.11 {
        return Err(vec![
            "paired lift or confidence bound is invalid/inconsistent with aggregate accuracies".to_string(),
        ]);
    }

    let cost_ratio = candidate_receipt.lifecycle_cost_units / baseline_receipt.lifecycle_cost_units;
    if paired_lift >= policy.min_quality_lift_pp && ci_lower > 0.0 {
        return Ok(PromotionAssessment {
            outcome: PromotionOutcome::QualityWin,
            candidate_id: candidate_id.to_string(),
            baseline_candidate_id: baseline_id.to_string(),
            paired_lift_pp: paired_lift,
            paired_ci95_lower_pp: Some(ci_lower),
            lifecycle_cost_ratio: cost_ratio,
            reason: "matched held-out paired quality gain clears the preregistered threshold and lower confidence bound".to_string(),
        });
    }

    // Non-inferiority must be supported by the paired confidence bound, not
    // merely by the point estimate; otherwise a noisy, harmful candidate can
    // pass the cost gate.
    if ci_lower >= -policy.non_inferiority_margin_pp
        && cost_ratio <= policy.max_cost_ratio_for_efficiency_win
    {
        return Ok(PromotionAssessment {
            outcome: PromotionOutcome::EfficiencyWin,
            candidate_id: candidate_id.to_string(),
            baseline_candidate_id: baseline_id.to_string(),
            paired_lift_pp: paired_lift,
            paired_ci95_lower_pp: Some(ci_lower),
            lifecycle_cost_ratio: cost_ratio,
            reason: "held-out quality is non-inferior within the preregistered margin and lifecycle cost is lower".to_string(),
        });
    }

    Ok(PromotionAssessment {
        outcome: PromotionOutcome::Rejected,
        candidate_id: candidate_id.to_string(),
        baseline_candidate_id: baseline_id.to_string(),
        paired_lift_pp: paired_lift,
        paired_ci95_lower_pp: Some(ci_lower),
        lifecycle_cost_ratio: cost_ratio,
        reason: "candidate clears neither the quality-win gate nor the cost/non-inferiority gate".to_string(),
    })
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(text: &str) -> String {
        sha256_hex(text.as_bytes())
    }

    fn candidate(id: &str, parent: Option<&str>, iteration: u32, state: CandidateState) -> PipelineCandidate {
        let mut candidate = PipelineCandidate {
            candidate_id: id.to_string(),
            parent_candidate_id: parent.map(ToString::to_string),
            manifest_sha256: String::new(),
            source_sha256: digest(&format!("{id}:source")),
            retrieval_config_sha256: digest(&format!("{id}:retrieval")),
            lifecycle_policy_sha256: digest(&format!("{id}:lifecycle")),
            prompt_sha256: digest("shared-reader-prompt"),
            state,
            edit_iteration: iteration,
            built_from_receipt_ids: Vec::new(),
        };
        candidate.manifest_sha256 = candidate_manifest_sha256(&candidate).unwrap();
        candidate
    }

    fn receipt(id: &str, candidate: &PipelineCandidate, split: EvaluationSplit, accuracy: f64, cost: f64) -> EvaluationReceipt {
        let mut receipt = EvaluationReceipt {
            receipt_id: id.to_string(),
            receipt_sha256: String::new(),
            candidate_id: candidate.candidate_id.clone(),
            candidate_manifest_sha256: candidate.manifest_sha256.clone(),
            benchmark_id: "LongMemEval-V2/small/web".to_string(),
            split,
            status: ReceiptStatus::Completed,
            dataset_manifest_sha256: digest("dataset"),
            case_set_sha256: digest("case-set"),
            answer_model_sha256: digest("reader-model"),
            reader_prompt_sha256: digest("reader-prompt"),
            evaluator_sha256: digest("evaluator"),
            harness_sha256: digest("official-harness"),
            protocol_sha256: digest("shared-eval-protocol"),
            paired_group_count: 100,
            accuracy,
            lifecycle_cost_units: cost,
            mean_query_latency_ms: 120.0,
            critical_governance_events: 0,
            baseline_candidate_id: None,
            paired_lift_pp: None,
            paired_ci95_lower_pp: None,
        };
        seal_receipt(&mut receipt);
        receipt
    }

    fn matched_trace() -> PipelineEvolutionTrace {
        let root = candidate("baseline", None, 0, CandidateState::Promoted);
        let trial = candidate("trial", Some("baseline"), 1, CandidateState::Frozen);
        let root_dev = receipt("baseline-dev", &root, EvaluationSplit::Development, 0.60, 100.0);
        let trial_dev = receipt("trial-dev", &trial, EvaluationSplit::Development, 0.62, 110.0);
        let root_test = receipt("baseline-heldout", &root, EvaluationSplit::HeldOut, 0.60, 100.0);
        let mut trial_test = receipt("trial-heldout", &trial, EvaluationSplit::HeldOut, 0.64, 72.0);
        trial_test.baseline_candidate_id = Some("baseline".to_string());
        trial_test.paired_lift_pp = Some(4.0);
        trial_test.paired_ci95_lower_pp = Some(1.5);
        seal_receipt(&mut trial_test);
        PipelineEvolutionTrace {
            schema: PIPELINE_EVOLUTION_SCHEMA.to_string(),
            benchmark_id: root_dev.benchmark_id.clone(),
            root_candidate_id: "baseline".to_string(),
            candidates: vec![root, trial],
            receipts: vec![root_dev, trial_dev, root_test, trial_test],
            optimizer_visible_receipt_ids: vec!["baseline-dev".to_string(), "trial-dev".to_string()],
        }
    }

    #[test]
    fn accepts_valid_tree_and_keeps_heldout_out_of_optimizer_view() {
        assert!(validate_evolution_trace(&matched_trace()).is_ok());
    }

    #[test]
    fn rejects_heldout_receipt_visible_to_optimizer() {
        let mut trace = matched_trace();
        trace.optimizer_visible_receipt_ids.push("trial-heldout".to_string());
        assert!(validate_evolution_trace(&trace).unwrap_err().iter().any(|item| item.contains("held-out receipt")));
    }

    #[test]
    fn rejects_candidate_built_from_heldout_feedback() {
        let mut trace = matched_trace();
        trace.candidates[1].built_from_receipt_ids.push("trial-heldout".to_string());
        let updated_manifest = candidate_manifest_sha256(&trace.candidates[1]).unwrap();
        trace.candidates[1].manifest_sha256 = updated_manifest.clone();
        for receipt in trace.receipts.iter_mut().filter(|receipt| receipt.candidate_id == "trial") {
            receipt.candidate_manifest_sha256 = updated_manifest.clone();
            seal_receipt(receipt);
        }
        assert!(validate_evolution_trace(&trace).unwrap_err().iter().any(|item| item.contains("built from held-out")));
    }

    #[test]
    fn rejects_missing_parent_and_parent_cycle() {
        let mut trace = matched_trace();
        trace.candidates[1].parent_candidate_id = Some("does-not-exist".to_string());
        assert!(validate_evolution_trace(&trace).unwrap_err().iter().any(|item| item.contains("missing parent")));
        trace.candidates[1].parent_candidate_id = Some("baseline".to_string());
        trace.candidates[0].parent_candidate_id = Some("trial".to_string());
        assert!(validate_evolution_trace(&trace).unwrap_err().iter().any(|item| item.contains("cycle")));
    }

    #[test]
    fn quality_promotion_requires_matched_heldout_receipts() {
        let trace = matched_trace();
        let result = assess_promotion(&trace, "trial", "baseline", "trial-heldout", "baseline-heldout", PromotionPolicy::default()).unwrap();
        assert_eq!(result.outcome, PromotionOutcome::QualityWin);
        assert_eq!(result.paired_lift_pp, 4.0);
    }

    #[test]
    fn efficiency_promotion_requires_noninferiority_and_cost_reduction() {
        let mut trace = matched_trace();
        let receipt = trace.receipts.iter_mut().find(|r| r.receipt_id == "trial-heldout").unwrap();
        receipt.accuracy = 0.595;
        receipt.paired_lift_pp = Some(-0.5);
        receipt.paired_ci95_lower_pp = Some(-0.8);
        receipt.lifecycle_cost_units = 70.0;
        seal_receipt(receipt);
        let result = assess_promotion(&trace, "trial", "baseline", "trial-heldout", "baseline-heldout", PromotionPolicy::default()).unwrap();
        assert_eq!(result.outcome, PromotionOutcome::EfficiencyWin);
    }


    #[test]
    fn efficiency_route_rejects_point_estimate_noninferiority_without_ci_support() {
        let mut trace = matched_trace();
        let receipt = trace.receipts.iter_mut().find(|r| r.receipt_id == "trial-heldout").unwrap();
        receipt.accuracy = 0.595;
        receipt.paired_lift_pp = Some(-0.5);
        receipt.paired_ci95_lower_pp = Some(-1.2);
        receipt.lifecycle_cost_units = 50.0;
        let result = assess_promotion(
            &trace, "trial", "baseline", "trial-heldout", "baseline-heldout",
            PromotionPolicy::default(),
        ).unwrap();
        assert_eq!(result.outcome, PromotionOutcome::Rejected);
    }

    #[test]
    fn rejects_unmatched_protocol_or_case_set() {
        let mut trace = matched_trace();
        let changed = trace.receipts.iter_mut().find(|r| r.receipt_id == "trial-heldout").unwrap();
        changed.protocol_sha256 = digest("different protocol");
        seal_receipt(changed);
        assert!(assess_promotion(&trace, "trial", "baseline", "trial-heldout", "baseline-heldout", PromotionPolicy::default()).is_err());

        let mut trace = matched_trace();
        let changed = trace.receipts.iter_mut().find(|r| r.receipt_id == "trial-heldout").unwrap();
        changed.case_set_sha256 = digest("different cases");
        seal_receipt(changed);
        assert!(assess_promotion(&trace, "trial", "baseline", "trial-heldout", "baseline-heldout", PromotionPolicy::default()).is_err());
    }

    #[test]
    fn governance_incidents_block_promotion_even_with_high_accuracy() {
        let mut trace = matched_trace();
        let changed = trace.receipts.iter_mut().find(|r| r.receipt_id == "trial-heldout").unwrap();
        changed.critical_governance_events = 1;
        seal_receipt(changed);
        assert!(assess_promotion(&trace, "trial", "baseline", "trial-heldout", "baseline-heldout", PromotionPolicy::default()).is_err());
    }

    #[test]
    fn insufficient_independent_groups_block_promotion() {
        let mut trace = matched_trace();
        let changed = trace.receipts.iter_mut().find(|r| r.receipt_id == "trial-heldout").unwrap();
        changed.paired_group_count = 12;
        seal_receipt(changed);
        assert!(assess_promotion(&trace, "trial", "baseline", "trial-heldout", "baseline-heldout", PromotionPolicy::default()).is_err());
    }



    #[test]
    fn rejects_metric_tampering_without_matching_receipt_digest() {
        let mut trace = matched_trace();
        let receipt = trace.receipts.iter_mut().find(|r| r.receipt_id == "trial-heldout").unwrap();
        receipt.accuracy = 0.99;
        let errors = validate_evolution_trace(&trace).unwrap_err();
        assert!(errors.iter().any(|item| item.contains("receipt digest mismatch")));
    }

    #[test]
    fn invalid_promotion_policy_is_rejected() {
        let trace = matched_trace();
        let policy = PromotionPolicy {
            min_independent_groups: 0,
            ..PromotionPolicy::default()
        };
        assert!(assess_promotion(
            &trace, "trial", "baseline", "trial-heldout", "baseline-heldout", policy
        ).is_err());
    }

    #[test]
    fn edit_iteration_and_receipt_manifest_digest_are_pinned() {
        let mut trace = matched_trace();
        trace.candidates[1].edit_iteration = 0;
        assert!(validate_evolution_trace(&trace).unwrap_err().iter().any(|item| item.contains("immutable manifest digest")));

        let mut trace = matched_trace();
        trace.candidates[1].retrieval_config_sha256 = digest("mutated retrieval config");
        assert!(validate_evolution_trace(&trace).unwrap_err().iter().any(|item| item.contains("immutable manifest digest")));

        let mut trace = matched_trace();
        let changed = trace.receipts.iter_mut().find(|r| r.receipt_id == "trial-heldout").unwrap();
        changed.candidate_manifest_sha256 = digest("mutated candidate");
        seal_receipt(changed);
        assert!(validate_evolution_trace(&trace).unwrap_err().iter().any(|item| item.contains("candidate manifest digest mismatch")));
    }
}
