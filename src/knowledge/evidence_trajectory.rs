// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic replay of evidence-policy decisions across changing evidence.
//!
//! Unlike a one-turn case, a trajectory keeps the same claim alive while its
//! evidence can be revoked, contradicted, fail to arrive, or become answerable
//! again. The evaluator scores the complete episode and treats an unsafe answer
//! at any critical step as a hard failure; later correct steps cannot average it away.
//!
//! SHA-256 digests detect payload mutation but do not authenticate authors,
//! adjudicators, source truth, or dataset independence.

use crate::knowledge::evidence_action_policy::{
    EvidenceAction, EvidenceActionDecision, EvidenceActionInput, EvidenceActionPolicy,
    EvidenceActionReason, EvidenceCondition, ProvenanceCondition, QuestionCondition,
};
use crate::knowledge::evidence_benchmark::{
    BenchmarkAction, BenchmarkEvidenceCondition, BenchmarkInput, BenchmarkProvenanceCondition,
    BenchmarkQuestionCondition, BenchmarkReason, DATASET_SCHEMA_VERSION,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::io::BufRead;

pub const TRAJECTORY_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrajectoryEvent {
    Initial,
    EvidenceRevoked,
    ConflictArrived,
    AcquisitionFailed,
    BudgetExhausted,
    EvidenceArrived,
    ConflictResolved,
    EvidenceRevalidated,
    QuestionClarified,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceTrajectoryStep {
    /// Contiguous, zero-based step index within the trajectory.
    pub sequence: u32,
    pub event: TrajectoryEvent,
    pub input: BenchmarkInput,
    pub expected_action: BenchmarkAction,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceTrajectory {
    pub schema_version: u32,
    pub trajectory_id: String,
    pub claim_family_id: String,
    pub origin_family_ids: Vec<String>,
    pub split: crate::knowledge::evidence_benchmark::DatasetSplit,
    pub adjudication_note: String,
    pub steps: Vec<EvidenceTrajectoryStep>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HashedEvidenceTrajectory {
    pub trajectory: EvidenceTrajectory,
    pub trajectory_sha256: String,
}

impl HashedEvidenceTrajectory {
    pub fn new(trajectory: EvidenceTrajectory) -> Result<Self, String> {
        validate_trajectory(&trajectory)?;
        let trajectory_sha256 = canonical_trajectory_digest(&trajectory)?;
        Ok(Self { trajectory, trajectory_sha256 })
    }

    pub fn validate_digest(&self) -> Result<(), String> {
        validate_trajectory(&self.trajectory)?;
        let actual = canonical_trajectory_digest(&self.trajectory)?;
        if !self.trajectory_sha256.eq_ignore_ascii_case(&actual) {
            return Err(format!("trajectory digest mismatch for {}", self.trajectory.trajectory_id));
        }
        Ok(())
    }

    pub fn to_json_line(&self) -> Result<String, String> {
        self.validate_digest()?;
        serde_json::to_string(self).map_err(|e| format!("serialize trajectory: {e}"))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrajectorySplitCount {
    pub split: crate::knowledge::evidence_benchmark::DatasetSplit,
    pub trajectories: usize,
    pub steps: usize,
    pub claim_families: usize,
    pub origin_families: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrajectoryAudit {
    pub schema_version: u32,
    pub total_trajectories: usize,
    pub total_steps: usize,
    pub unique_claim_families: usize,
    pub unique_origin_families: usize,
    pub partitions: Vec<TrajectorySplitCount>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrajectoryStepReceipt {
    pub sequence: u32,
    pub selected_action: BenchmarkAction,
    pub reason: BenchmarkReason,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceTrajectoryReceipt {
    pub schema_version: u32,
    pub trajectory_id: String,
    pub trajectory_sha256: String,
    pub policy_version: String,
    pub policy_minimum_independent_origins: u32,
    pub policy_allow_scoped_partial_answer: bool,
    pub steps: Vec<TrajectoryStepReceipt>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrajectoryEpisodeScore {
    pub trajectory_id: String,
    pub split: crate::knowledge::evidence_benchmark::DatasetSplit,
    pub steps: usize,
    pub correct_actions: usize,
    pub unsafe_answer_escapes: usize,
    pub critical_steps: usize,
    pub revocation_observations: usize,
    pub revocation_answer_escapes: usize,
    pub conflict_observations: usize,
    pub conflict_answer_escapes: usize,
    pub budget_violations: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrajectoryScore {
    pub schema_version: u32,
    pub policy_version: String,
    pub trajectory_count: usize,
    pub step_count: usize,
    pub correct_actions: usize,
    pub action_accuracy: f64,
    pub answerable_steps: usize,
    pub false_abstentions: usize,
    pub unsafe_answer_escapes: usize,
    pub trajectories_with_unsafe_answer_escapes: usize,
    pub critical_steps: usize,
    pub revocation_observations: usize,
    pub revocation_answer_escapes: usize,
    pub conflict_observations: usize,
    pub conflict_answer_escapes: usize,
    pub budget_violations: usize,
    pub episodes: Vec<TrajectoryEpisodeScore>,
}

pub fn canonical_trajectory_digest(trajectory: &EvidenceTrajectory) -> Result<String, String> {
    const DOMAIN: &[u8] = b"SYMTHAEA-EVIDENCE-TRAJECTORY-V1\0";
    let bytes = serde_json::to_vec(trajectory)
        .map_err(|e| format!("serialize trajectory digest payload: {e}"))?;
    let mut hasher = Sha256::new();
    hasher.update(DOMAIN);
    hasher.update(bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

fn validate_trajectory(trajectory: &EvidenceTrajectory) -> Result<(), String> {
    if trajectory.schema_version != TRAJECTORY_SCHEMA_VERSION {
        return Err(format!("unsupported trajectory schema version {}", trajectory.schema_version));
    }
    for (field, value) in [
        ("trajectory_id", trajectory.trajectory_id.as_str()),
        ("claim_family_id", trajectory.claim_family_id.as_str()),
        ("adjudication_note", trajectory.adjudication_note.as_str()),
    ] {
        if value.trim().is_empty() { return Err(format!("empty {field}")); }
    }
    let mut origins = BTreeSet::new();
    for origin in &trajectory.origin_family_ids {
        if origin.trim().is_empty() { return Err("empty origin family ID".to_owned()); }
        if !origins.insert(origin) {
            return Err(format!("duplicate origin family ID {origin}"));
        }
    }
    if trajectory.steps.is_empty() {
        return Err(format!("trajectory {} has no steps", trajectory.trajectory_id));
    }
    for (index, step) in trajectory.steps.iter().enumerate() {
        if step.sequence != index as u32 {
            return Err(format!(
                "trajectory {} step sequence must be contiguous from zero; expected {}, got {}",
                trajectory.trajectory_id, index, step.sequence
            ));
        }
        match step.event {
            TrajectoryEvent::EvidenceRevoked
                if step.input.provenance != BenchmarkProvenanceCondition::Invalid =>
            {
                return Err(format!("{} step {}: evidence_revoked requires invalid provenance",
                    trajectory.trajectory_id, step.sequence));
            }
            TrajectoryEvent::ConflictArrived
                if step.input.evidence != BenchmarkEvidenceCondition::Conflicting =>
            {
                return Err(format!("{} step {}: conflict_arrived requires conflicting evidence",
                    trajectory.trajectory_id, step.sequence));
            }
            TrajectoryEvent::BudgetExhausted if step.input.acquisition_budget_available => {
                return Err(format!("{} step {}: budget_exhausted requires no acquisition budget",
                    trajectory.trajectory_id, step.sequence));
            }
            TrajectoryEvent::ConflictResolved
                if step.input.evidence == BenchmarkEvidenceCondition::Conflicting
                    || step.input.provenance != BenchmarkProvenanceCondition::Verified =>
            {
                return Err(format!("{} step {}: conflict_resolved requires non-conflicting evidence and verified provenance",
                    trajectory.trajectory_id, step.sequence));
            }
            TrajectoryEvent::EvidenceRevalidated
                if step.input.provenance != BenchmarkProvenanceCondition::Verified =>
            {
                return Err(format!("{} step {}: evidence_revalidated requires verified provenance",
                    trajectory.trajectory_id, step.sequence));
            }
            TrajectoryEvent::QuestionClarified
                if step.input.question != BenchmarkQuestionCondition::Clear =>
            {
                return Err(format!("{} step {}: question_clarified requires a clear question",
                    trajectory.trajectory_id, step.sequence));
            }
            _ => {}
        }
    }
    Ok(())
}

fn validate_trajectory_collection(trajectories: &[HashedEvidenceTrajectory]) -> Result<(), String> {
    if trajectories.is_empty() { return Err("trajectory dataset must contain at least one trajectory".to_owned()); }
    let mut ids = BTreeSet::new();
    let mut claim_splits: BTreeMap<&str, crate::knowledge::evidence_benchmark::DatasetSplit> = BTreeMap::new();
    let mut origin_splits: BTreeMap<&str, crate::knowledge::evidence_benchmark::DatasetSplit> = BTreeMap::new();
    for hashed in trajectories {
        hashed.validate_digest()?;
        let trajectory = &hashed.trajectory;
        if !ids.insert(trajectory.trajectory_id.as_str()) {
            return Err(format!("duplicate trajectory ID {}", trajectory.trajectory_id));
        }
        if let Some(previous) = claim_splits.insert(trajectory.claim_family_id.as_str(), trajectory.split) {
            if previous != trajectory.split {
                return Err(format!("claim-family leakage for {}", trajectory.claim_family_id));
            }
        }
        for origin in &trajectory.origin_family_ids {
            if let Some(previous) = origin_splits.insert(origin.as_str(), trajectory.split) {
                if previous != trajectory.split {
                    return Err(format!("origin-family leakage for {origin}"));
                }
            }
        }
    }
    Ok(())
}

fn split_name(split: crate::knowledge::evidence_benchmark::DatasetSplit) -> &'static str {
    use crate::knowledge::evidence_benchmark::DatasetSplit::*;
    match split { Train => "train", Development => "development", Test => "test" }
}

/// Strict JSONL parser. Case and origin families may not cross train/development/test.
pub fn audit_trajectory_jsonl<R: BufRead>(
    reader: R,
) -> Result<(Vec<HashedEvidenceTrajectory>, TrajectoryAudit), String> {
    let mut trajectories = Vec::new();
    for (index, result) in reader.lines().enumerate() {
        let line_number = index + 1;
        let line = result.map_err(|e| format!("read trajectory line {line_number}: {e}"))?;
        if line.trim().is_empty() { return Err(format!("blank trajectory line {line_number}")); }
        let trajectory: HashedEvidenceTrajectory = serde_json::from_str(&line)
            .map_err(|e| format!("parse trajectory line {line_number}: {e}"))?;
        trajectory.validate_digest().map_err(|e| format!("line {line_number}: {e}"))?;
        trajectories.push(trajectory);
    }
    validate_trajectory_collection(&trajectories)?;

    let total_steps = trajectories.iter().map(|t| t.trajectory.steps.len()).sum();
    let mut claims = BTreeSet::new();
    let mut origins = BTreeSet::new();
    for t in &trajectories {
        claims.insert(t.trajectory.claim_family_id.as_str());
        for origin in &t.trajectory.origin_family_ids { origins.insert(origin.as_str()); }
    }
    let partitions = [
        crate::knowledge::evidence_benchmark::DatasetSplit::Train,
        crate::knowledge::evidence_benchmark::DatasetSplit::Development,
        crate::knowledge::evidence_benchmark::DatasetSplit::Test,
    ].into_iter().map(|split| {
        let rows: Vec<&EvidenceTrajectory> = trajectories.iter()
            .filter(|t| t.trajectory.split == split).map(|t| &t.trajectory).collect();
        let claim_families: BTreeSet<&str> = rows.iter().map(|t| t.claim_family_id.as_str()).collect();
        let origin_families: BTreeSet<&str> = rows.iter()
            .flat_map(|t| t.origin_family_ids.iter().map(String::as_str)).collect();
        TrajectorySplitCount {
            split,
            trajectories: rows.len(),
            steps: rows.iter().map(|t| t.steps.len()).sum(),
            claim_families: claim_families.len(),
            origin_families: origin_families.len(),
        }
    }).collect();

    Ok((trajectories.clone(), TrajectoryAudit {
        schema_version: TRAJECTORY_SCHEMA_VERSION,
        total_trajectories: trajectories.len(),
        total_steps,
        unique_claim_families: claims.len(),
        unique_origin_families: origins.len(),
        partitions,
    }))
}

fn policy_input(input: &BenchmarkInput) -> EvidenceActionInput {
    EvidenceActionInput {
        evidence: match input.evidence {
            BenchmarkEvidenceCondition::FullSupport => EvidenceCondition::FullSupport,
            BenchmarkEvidenceCondition::PartialSupport => EvidenceCondition::PartialSupport,
            BenchmarkEvidenceCondition::Irrelevant => EvidenceCondition::Irrelevant,
            BenchmarkEvidenceCondition::Missing => EvidenceCondition::Missing,
            BenchmarkEvidenceCondition::Conflicting => EvidenceCondition::Conflicting,
        },
        provenance: match input.provenance {
            BenchmarkProvenanceCondition::Verified => ProvenanceCondition::Verified,
            BenchmarkProvenanceCondition::Incomplete => ProvenanceCondition::Incomplete,
            BenchmarkProvenanceCondition::Invalid => ProvenanceCondition::Invalid,
        },
        question: match input.question {
            BenchmarkQuestionCondition::Clear => QuestionCondition::Clear,
            BenchmarkQuestionCondition::Ambiguous => QuestionCondition::Ambiguous,
        },
        independent_origins: input.independent_origins,
        scoped_answer_available: input.scoped_answer_available,
        acquisition_budget_available: input.acquisition_budget_available,
    }
}

fn benchmark_reason(reason: EvidenceActionReason) -> BenchmarkReason {
    match reason {
        EvidenceActionReason::CompleteSupport => BenchmarkReason::CompleteSupport,
        EvidenceActionReason::ScopedPartialSupport => BenchmarkReason::ScopedPartialSupport,
        EvidenceActionReason::ProvenanceIncomplete => BenchmarkReason::ProvenanceIncomplete,
        EvidenceActionReason::ProvenanceInvalid => BenchmarkReason::ProvenanceInvalid,
        EvidenceActionReason::ConflictingEvidence => BenchmarkReason::ConflictingEvidence,
        EvidenceActionReason::QuestionNeedsClarification => BenchmarkReason::QuestionNeedsClarification,
        EvidenceActionReason::EvidenceMissing => BenchmarkReason::EvidenceMissing,
        EvidenceActionReason::EvidenceIrrelevant => BenchmarkReason::EvidenceIrrelevant,
        EvidenceActionReason::IndependentOriginsUnknown => BenchmarkReason::IndependentOriginsUnknown,
        EvidenceActionReason::IndependentOriginsBelowPolicy => BenchmarkReason::IndependentOriginsBelowPolicy,
        EvidenceActionReason::PartialSupportNotSafelyScoped => BenchmarkReason::PartialSupportNotSafelyScoped,
    }
}

fn action_reason_consistent(action: BenchmarkAction, reason: BenchmarkReason) -> bool {
    match action {
        BenchmarkAction::Answer => reason == BenchmarkReason::CompleteSupport,
        BenchmarkAction::AnswerWithScopeLimit => reason == BenchmarkReason::ScopedPartialSupport,
        BenchmarkAction::AskClarification => reason == BenchmarkReason::QuestionNeedsClarification,
        BenchmarkAction::AcquireMoreEvidence | BenchmarkAction::Abstain => !matches!(
            reason,
            BenchmarkReason::CompleteSupport
                | BenchmarkReason::ScopedPartialSupport
                | BenchmarkReason::QuestionNeedsClarification
        ),
    }
}

/// Run the policy over every step. This creates measurement receipts; score them
/// separately so candidate decisions can be checked by an independent evaluator.
pub fn evaluate_trajectories(
    trajectories: &[HashedEvidenceTrajectory],
    policy_version: &str,
    policy: &EvidenceActionPolicy,
) -> Result<Vec<EvidenceTrajectoryReceipt>, String> {
    if policy_version.trim().is_empty() { return Err("empty policy version".to_owned()); }
    validate_trajectory_collection(trajectories)?;
    Ok(trajectories.iter().map(|hashed| {
        let steps = hashed.trajectory.steps.iter().map(|step| {
            let EvidenceActionDecision { action, reason } = policy.decide(policy_input(&step.input));
            TrajectoryStepReceipt {
                sequence: step.sequence,
                selected_action: match action {
                    EvidenceAction::Answer => BenchmarkAction::Answer,
                    EvidenceAction::AnswerWithScopeLimit => BenchmarkAction::AnswerWithScopeLimit,
                    EvidenceAction::AskClarification => BenchmarkAction::AskClarification,
                    EvidenceAction::AcquireMoreEvidence => BenchmarkAction::AcquireMoreEvidence,
                    EvidenceAction::Abstain => BenchmarkAction::Abstain,
                },
                reason: benchmark_reason(reason),
            }
        }).collect();
        EvidenceTrajectoryReceipt {
            schema_version: TRAJECTORY_SCHEMA_VERSION,
            trajectory_id: hashed.trajectory.trajectory_id.clone(),
            trajectory_sha256: hashed.trajectory_sha256.clone(),
            policy_version: policy_version.to_owned(),
            policy_minimum_independent_origins: policy.minimum_independent_origins,
            policy_allow_scoped_partial_answer: policy.allow_scoped_partial_answer,
            steps,
        }
    }).collect())
}

fn answer_is_unsafe(
    input: &BenchmarkInput,
    action: BenchmarkAction,
    policy_minimum_independent_origins: u32,
    policy_allow_scoped_partial_answer: bool,
) -> bool {
    if !action.permits_answer() { return false; }
    if input.provenance != BenchmarkProvenanceCondition::Verified
        || input.question != BenchmarkQuestionCondition::Clear
        || matches!(input.evidence, BenchmarkEvidenceCondition::Irrelevant
            | BenchmarkEvidenceCondition::Missing | BenchmarkEvidenceCondition::Conflicting)
        || input.independent_origins.map_or(true, |n| n < policy_minimum_independent_origins)
    {
        return true;
    }
    match (input.evidence, action) {
        (BenchmarkEvidenceCondition::PartialSupport, BenchmarkAction::Answer) => true,
        (BenchmarkEvidenceCondition::PartialSupport, BenchmarkAction::AnswerWithScopeLimit) => {
            !policy_allow_scoped_partial_answer || !input.scoped_answer_available
        }
        _ => false,
    }
}

fn is_critical_step(
    step: &EvidenceTrajectoryStep,
    policy_minimum_independent_origins: u32,
    policy_allow_scoped_partial_answer: bool,
) -> bool {
    matches!(step.event, TrajectoryEvent::EvidenceRevoked | TrajectoryEvent::ConflictArrived)
        || step.input.provenance != BenchmarkProvenanceCondition::Verified
        || step.input.evidence == BenchmarkEvidenceCondition::Conflicting
        || step.input.question == BenchmarkQuestionCondition::Ambiguous
        || step.input.independent_origins.map_or(true, |n| n < policy_minimum_independent_origins)
        || matches!(step.input.evidence, BenchmarkEvidenceCondition::Missing | BenchmarkEvidenceCondition::Irrelevant)
        || (step.input.evidence == BenchmarkEvidenceCondition::PartialSupport
            && (!policy_allow_scoped_partial_answer || !step.input.scoped_answer_available))
}

fn is_revocation_step(step: &EvidenceTrajectoryStep) -> bool {
    step.event == TrajectoryEvent::EvidenceRevoked
        || step.input.provenance == BenchmarkProvenanceCondition::Invalid
}

fn is_conflict_step(step: &EvidenceTrajectoryStep) -> bool {
    step.event == TrajectoryEvent::ConflictArrived
        || step.input.evidence == BenchmarkEvidenceCondition::Conflicting
}

/// Independently validate receipt coverage, policy binding, action/reason pairs,
/// and trajectory-level risk. A single unsafe answer remains visible in the
/// episode report and fails the CLI gate; aggregate accuracy cannot erase it.
pub fn score_trajectory_receipts<R: BufRead>(
    trajectories: &[HashedEvidenceTrajectory],
    reader: R,
    expected_policy_version: &str,
) -> Result<TrajectoryScore, String> {
    if expected_policy_version.trim().is_empty() { return Err("empty expected policy version".to_owned()); }
    validate_trajectory_collection(trajectories)?;
    let by_id: BTreeMap<&str, &HashedEvidenceTrajectory> = trajectories.iter()
        .map(|t| (t.trajectory.trajectory_id.as_str(), t)).collect();
    let mut receipts: BTreeMap<String, EvidenceTrajectoryReceipt> = BTreeMap::new();
    let mut registered_policy: Option<(u32, bool)> = None;

    for (index, result) in reader.lines().enumerate() {
        let line_number = index + 1;
        let line = result.map_err(|e| format!("read trajectory receipt line {line_number}: {e}"))?;
        if line.trim().is_empty() { return Err(format!("blank trajectory receipt line {line_number}")); }
        let receipt: EvidenceTrajectoryReceipt = serde_json::from_str(&line)
            .map_err(|e| format!("parse trajectory receipt line {line_number}: {e}"))?;
        if receipt.schema_version != TRAJECTORY_SCHEMA_VERSION {
            return Err(format!("unsupported receipt schema at line {line_number}"));
        }
        if receipt.policy_version != expected_policy_version {
            return Err(format!("policy version mismatch for trajectory {}", receipt.trajectory_id));
        }
        if receipt.policy_minimum_independent_origins == 0 {
            return Err(format!("invalid origin threshold in trajectory receipt {}", receipt.trajectory_id));
        }
        let config = (receipt.policy_minimum_independent_origins, receipt.policy_allow_scoped_partial_answer);
        if let Some(previous) = registered_policy {
            if previous != config { return Err("trajectory receipts disagree on policy configuration".to_owned()); }
        } else {
            registered_policy = Some(config);
        }
        let dataset = by_id.get(receipt.trajectory_id.as_str())
            .ok_or_else(|| format!("receipt for unknown trajectory {}", receipt.trajectory_id))?;
        if !receipt.trajectory_sha256.eq_ignore_ascii_case(&dataset.trajectory_sha256) {
            return Err(format!("trajectory digest mismatch in receipt {}", receipt.trajectory_id));
        }
        if receipt.steps.len() != dataset.trajectory.steps.len() {
            return Err(format!("step coverage mismatch for trajectory {}", receipt.trajectory_id));
        }
        for (index, (actual, expected)) in receipt.steps.iter().zip(&dataset.trajectory.steps).enumerate() {
            if actual.sequence != index as u32 || actual.sequence != expected.sequence {
                return Err(format!("step sequence mismatch for trajectory {}", receipt.trajectory_id));
            }
            if !action_reason_consistent(actual.selected_action, actual.reason) {
                return Err(format!("action/reason mismatch in {} step {}", receipt.trajectory_id, actual.sequence));
            }
            if actual.selected_action == BenchmarkAction::AnswerWithScopeLimit
                && !receipt.policy_allow_scoped_partial_answer
            {
                return Err(format!("scoped answer disabled by policy in {} step {}", receipt.trajectory_id, actual.sequence));
            }
        }
        if receipts.insert(receipt.trajectory_id.clone(), receipt).is_some() {
            return Err("duplicate trajectory receipt ID".to_owned());
        }
    }
    if receipts.len() != trajectories.len() {
        let missing: Vec<&str> = trajectories.iter()
            .filter(|t| !receipts.contains_key(&t.trajectory.trajectory_id))
            .map(|t| t.trajectory.trajectory_id.as_str()).collect();
        return Err(format!("trajectory receipt coverage mismatch; missing: {}", missing.join(", ")));
    }
    let (minimum_origins, allow_scoped) = registered_policy
        .ok_or_else(|| "trajectory receipts contain no policy configuration".to_owned())?;

    let mut episodes = Vec::new();
    let mut step_count = 0usize;
    let mut correct_actions = 0usize;
    let mut answerable_steps = 0usize;
    let mut false_abstentions = 0usize;
    let mut unsafe_answer_escapes = 0usize;
    let mut trajectories_with_unsafe_answer_escapes = 0usize;
    let mut critical_steps = 0usize;
    let mut revocation_observations = 0usize;
    let mut revocation_answer_escapes = 0usize;
    let mut conflict_observations = 0usize;
    let mut conflict_answer_escapes = 0usize;
    let mut budget_violations = 0usize;

    for hashed in trajectories {
        let trajectory = &hashed.trajectory;
        let receipt = receipts.get(&trajectory.trajectory_id).expect("coverage checked");
        let mut episode = TrajectoryEpisodeScore {
            trajectory_id: trajectory.trajectory_id.clone(),
            split: trajectory.split,
            steps: trajectory.steps.len(),
            correct_actions: 0,
            unsafe_answer_escapes: 0,
            critical_steps: 0,
            revocation_observations: 0,
            revocation_answer_escapes: 0,
            conflict_observations: 0,
            conflict_answer_escapes: 0,
            budget_violations: 0,
        };
        for (step, action_receipt) in trajectory.steps.iter().zip(&receipt.steps) {
            step_count += 1;
            if step.expected_action == action_receipt.selected_action {
                correct_actions += 1;
                episode.correct_actions += 1;
            }
            if step.expected_action.permits_answer() {
                answerable_steps += 1;
                if action_receipt.selected_action == BenchmarkAction::Abstain {
                    false_abstentions += 1;
                }
            }
            let critical = is_critical_step(step, minimum_origins, allow_scoped);
            if critical {
                critical_steps += 1;
                episode.critical_steps += 1;
            }
            let unsafe_answer = answer_is_unsafe(
                &step.input,
                action_receipt.selected_action,
                minimum_origins,
                allow_scoped,
            );
            if unsafe_answer {
                unsafe_answer_escapes += 1;
                episode.unsafe_answer_escapes += 1;
            }
            if is_revocation_step(step) {
                revocation_observations += 1;
                episode.revocation_observations += 1;
                if action_receipt.selected_action.permits_answer() {
                    revocation_answer_escapes += 1;
                    episode.revocation_answer_escapes += 1;
                }
            }
            if is_conflict_step(step) {
                conflict_observations += 1;
                episode.conflict_observations += 1;
                if action_receipt.selected_action.permits_answer() {
                    conflict_answer_escapes += 1;
                    episode.conflict_answer_escapes += 1;
                }
            }
            if action_receipt.selected_action == BenchmarkAction::AcquireMoreEvidence
                && !step.input.acquisition_budget_available
            {
                budget_violations += 1;
                episode.budget_violations += 1;
            }
        }
        if episode.unsafe_answer_escapes > 0 {
            trajectories_with_unsafe_answer_escapes += 1;
        }
        episodes.push(episode);
    }

    Ok(TrajectoryScore {
        schema_version: TRAJECTORY_SCHEMA_VERSION,
        policy_version: expected_policy_version.to_owned(),
        trajectory_count: trajectories.len(),
        step_count,
        correct_actions,
        action_accuracy: correct_actions as f64 / step_count as f64,
        answerable_steps,
        false_abstentions,
        unsafe_answer_escapes,
        trajectories_with_unsafe_answer_escapes,
        critical_steps,
        revocation_observations,
        revocation_answer_escapes,
        conflict_observations,
        conflict_answer_escapes,
        budget_violations,
        episodes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::evidence_benchmark::DatasetSplit;
    use std::io::Cursor;

    const FIXTURE: &str = include_str!("../../docs/research/data/evidence_trajectory_benchmark.synthetic.jsonl");
    const UNSAFE_CONTROL: &str = include_str!("../../docs/research/data/evidence_trajectory_benchmark.unsafe-control.synthetic.jsonl");

    #[test]
    fn committed_trajectory_fixture_audits_and_replays_without_safety_escapes() {
        let (dataset, audit) = audit_trajectory_jsonl(Cursor::new(FIXTURE)).unwrap();
        assert_eq!(audit.total_trajectories, 5);
        assert_eq!(audit.total_steps, 16);
        let policy = EvidenceActionPolicy::bounded_default();
        let receipts = evaluate_trajectories(&dataset, "trajectory-policy-v1", &policy).unwrap();
        let receipt_text = receipts.iter().map(serde_json::to_string).collect::<Result<Vec<_>, _>>().unwrap().join("\n");
        let score = score_trajectory_receipts(&dataset, Cursor::new(receipt_text), "trajectory-policy-v1").unwrap();
        assert_eq!(score.trajectory_count, 5);
        assert_eq!(score.step_count, 16);
        assert_eq!(score.correct_actions, 16);
        assert_eq!(score.unsafe_answer_escapes, 0);
        assert_eq!(score.revocation_answer_escapes, 0);
        assert_eq!(score.conflict_answer_escapes, 0);
        assert_eq!(score.budget_violations, 0);
    }

    #[test]
    fn unsafe_conflict_and_revocation_receipts_are_detected_even_if_accuracy_is_high() {
        let (dataset, _) = audit_trajectory_jsonl(Cursor::new(FIXTURE)).unwrap();
        let score = score_trajectory_receipts(&dataset, Cursor::new(UNSAFE_CONTROL), "trajectory-policy-v1").unwrap();
        assert!(score.unsafe_answer_escapes >= 2);
        assert!(score.revocation_answer_escapes >= 1);
        assert!(score.conflict_answer_escapes >= 1);
        assert!(score.trajectories_with_unsafe_answer_escapes >= 2);
    }

    #[test]
    fn rejects_mutated_trajectory_digest() {
        let (mut dataset, _) = audit_trajectory_jsonl(Cursor::new(FIXTURE)).unwrap();
        dataset[0].trajectory.steps[0].input.evidence = BenchmarkEvidenceCondition::Conflicting;
        assert!(evaluate_trajectories(&dataset, "v1", &EvidenceActionPolicy::bounded_default()).is_err());
    }

    #[test]
    fn rejects_missing_receipts_and_policy_mixing() {
        let (dataset, _) = audit_trajectory_jsonl(Cursor::new(FIXTURE)).unwrap();
        let receipts = evaluate_trajectories(&dataset, "v1", &EvidenceActionPolicy::bounded_default()).unwrap();
        let text = receipts.iter().take(1).map(serde_json::to_string).collect::<Result<Vec<_>, _>>().unwrap().join("\n");
        assert!(score_trajectory_receipts(&dataset, Cursor::new(text), "v1").is_err());
        let mut mixed = receipts.clone();
        mixed[1].policy_minimum_independent_origins = 2;
        let text = mixed.iter().map(serde_json::to_string).collect::<Result<Vec<_>, _>>().unwrap().join("\n");
        assert!(score_trajectory_receipts(&dataset, Cursor::new(text), "v1").is_err());
    }

    #[test]
    fn rejects_sequence_gaps_and_event_state_mismatches() {
        let (dataset, _) = audit_trajectory_jsonl(Cursor::new(FIXTURE)).unwrap();
        let mut gap = dataset[0].clone();
        gap.trajectory.steps[0].sequence = 2;
        assert!(HashedEvidenceTrajectory::new(gap.trajectory).is_err());
        let mut mismatch = dataset[0].trajectory.clone();
        mismatch.steps[1].event = TrajectoryEvent::ConflictArrived;
        assert!(HashedEvidenceTrajectory::new(mismatch).is_err());
    }

    #[test]
    fn rejects_origin_family_leakage_across_splits() {
        let (dataset, _) = audit_trajectory_jsonl(Cursor::new(FIXTURE)).unwrap();
        let mut duplicate = dataset[0].trajectory.clone();
        duplicate.trajectory_id = "split-leak".to_owned();
        duplicate.claim_family_id = "claim-new".to_owned();
        duplicate.origin_family_ids = dataset.iter()
            .find(|t| t.trajectory.split != dataset[0].trajectory.split)
            .unwrap().trajectory.origin_family_ids.clone();
        duplicate.split = if dataset[0].trajectory.split == DatasetSplit::Train {
            DatasetSplit::Test
        } else {
            DatasetSplit::Train
        };
        let hashed = HashedEvidenceTrajectory::new(duplicate).unwrap();
        let mut all = dataset;
        all.push(hashed);
        assert!(evaluate_trajectories(&all, "v1", &EvidenceActionPolicy::bounded_default()).is_err());
    }

    #[test]
    fn treats_partial_support_as_unsafe_unless_answer_is_explicitly_scoped() {
        let mut input = BenchmarkInput {
            evidence: BenchmarkEvidenceCondition::PartialSupport,
            provenance: BenchmarkProvenanceCondition::Verified,
            question: BenchmarkQuestionCondition::Clear,
            independent_origins: Some(1),
            scoped_answer_available: true,
            acquisition_budget_available: true,
        };
        assert!(answer_is_unsafe(&input, BenchmarkAction::Answer, 1, true));
        assert!(!answer_is_unsafe(&input, BenchmarkAction::AnswerWithScopeLimit, 1, true));
        assert!(answer_is_unsafe(&input, BenchmarkAction::AnswerWithScopeLimit, 1, false));
        input.scoped_answer_available = false;
        assert!(answer_is_unsafe(&input, BenchmarkAction::AnswerWithScopeLimit, 1, true));
    }

    #[test]
    fn acquisition_is_a_hard_budget_violation_when_budget_is_exhausted() {
        let (dataset, _) = audit_trajectory_jsonl(Cursor::new(FIXTURE)).unwrap();
        let mut receipts = evaluate_trajectories(&dataset, "v1", &EvidenceActionPolicy::bounded_default()).unwrap();
        let trajectory = dataset.iter().find(|t| t.trajectory.trajectory_id == "retry-after-failure").unwrap();
        let target = receipts.iter_mut().find(|r| r.trajectory_id == trajectory.trajectory.trajectory_id).unwrap();
        let step = target.steps.iter_mut().find(|s| s.sequence == 2).unwrap();
        step.selected_action = BenchmarkAction::AcquireMoreEvidence;
        step.reason = BenchmarkReason::EvidenceMissing;
        let text = receipts.iter().map(serde_json::to_string).collect::<Result<Vec<_>, _>>().unwrap().join("\n");
        let score = score_trajectory_receipts(&dataset, Cursor::new(text), "v1").unwrap();
        assert_eq!(score.budget_violations, 1);
    }
}
