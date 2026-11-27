// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Dependency-free scoring core for MACE Decision-Sufficient Memory (DSM).
//!
//! Security boundary:
//! - Candidate-facing JSONL contains only candidate episode data.
//! - EvaluatorCaseV1 contains hidden losses and adjudication labels. It must
//!   be constructed and retained only by the evaluator process.
//! - Never serialize EvaluatorCaseV1 into a candidate prompt, request,
//!   artifact, log, or shared fixture.
//!
//! This crate scores already-produced candidate outputs. It does not parse the
//! public JSONL format, generate worlds, run a memory backend, or establish
//! production security. Those are separate adapter/evaluator responsibilities.

pub mod longmemeval_v2;

use std::collections::BTreeSet;

/// Evaluator-owned action loss, expressed in integer micro-loss units.
///
/// Integer losses avoid platform-dependent floating-point tie handling.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActionLossV1 {
    pub action: String,
    pub loss_micros: u64,
}

/// Typed lineage attached to one canonical source event. Digests are identities,
/// not signatures or proof that two roots are statistically independent.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvidenceLineageV1 {
    pub evidence_id: String,
    pub source_family: Option<String>,
    pub source_root_digest_sha256: Option<String>,
}

/// An evaluator-adjudicated contradictory pair. v0.2 has no conflict-resolution
/// receipt, so co-retrieving both sides is reported and blocks sufficiency.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvidenceConflictV1 {
    pub left_evidence_id: String,
    pub right_evidence_id: String,
}

/// Hidden evaluator data. Never expose this structure to candidate systems.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvaluatorCaseV1 {
    pub episode_id: String,
    pub allowed_actions: Vec<String>,
    pub action_losses: Vec<ActionLossV1>,
    pub no_memory_action: String,
    pub oracle_action: String,
    /// Evidence IDs independently adjudicated as admissible at query time.
    pub admissible_evidence_ids: Vec<String>,
    /// Evidence required to achieve the reference evidence objective.
    pub required_evidence_ids: Vec<String>,
    /// Source-family/root metadata resolved from the canonical candidate episode.
    pub evidence_lineage: Vec<EvidenceLineageV1>,
    /// Contradictions adjudicated from the canonical source-event graph.
    pub contradiction_pairs: Vec<EvidenceConflictV1>,
}

/// Candidate-facing output from one memory system for one episode.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CandidateRunV1 {
    pub episode_id: String,
    pub system_id: String,
    /// Digest of the frozen downstream decision policy / reader configuration.
    pub downstream_policy_digest: String,
    /// Fixed seed shared across paired and leave-one-out policy executions.
    pub decision_seed: u64,
    pub chosen_action: String,
    /// Ordered evidence IDs returned by Search / evidence resolution.
    pub retrieved_evidence_ids: Vec<String>,
    /// Evidence IDs actually passed to the downstream decision policy.
    /// Must be a unique subset of retrieved_evidence_ids.
    pub decision_support_evidence_ids: Vec<String>,
}

/// Exact integer ratio. A missing ratio is used when its denominator is zero.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CountRatioV1 {
    pub numerator: u64,
    pub denominator: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RunScoreV1 {
    pub system_id: String,
    pub chosen_action: String,
    pub decision_loss_micros: u64,
    pub oracle_loss_micros: u64,
    pub decision_regret_micros: u64,
    pub decision_optimal: bool,
    pub returned_evidence_count: u64,
    pub admissible_returned_evidence_count: u64,
    pub inadmissible_returned_evidence_count: u64,
    pub decision_support_evidence_count: u64,
    pub admissible_decision_support_evidence_count: u64,
    pub inadmissible_decision_support_evidence_count: u64,
    /// Recall among all IDs returned by retrieval.
    pub retrieved_required_evidence_recall: Option<CountRatioV1>,
    /// Primary required-evidence recall: evidence actually used by the policy.
    pub required_evidence_recall: Option<CountRatioV1>,
    /// Precision among IDs actually used as decision support.
    pub decision_support_evidence_precision: Option<CountRatioV1>,
    /// Distinct source-family and upstream-root counts; these are not attestations.
    pub retrieved_distinct_source_family_count: u64,
    pub retrieved_unknown_source_family_count: u64,
    pub retrieved_distinct_source_root_count: u64,
    pub retrieved_unknown_source_root_count: u64,
    pub decision_support_distinct_source_family_count: u64,
    pub decision_support_unknown_source_family_count: u64,
    pub decision_support_distinct_source_root_count: u64,
    pub decision_support_unknown_source_root_count: u64,
    pub unresolved_retrieved_contradiction_pair_count: u64,
    pub unresolved_decision_support_contradiction_pair_count: u64,
    pub decision_sufficient: bool,
    /// Precision of the full Search result, retained for backward-compatible reporting.
    pub admissible_evidence_precision: Option<CountRatioV1>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvidenceAblationV1 {
    pub removed_evidence_id: String,
    /// Same system and episode, re-run with exactly one evidence ID withheld.
    pub output_without_evidence: CandidateRunV1,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvidenceInfluenceV1 {
    pub evidence_id: String,
    pub full_decision_loss_micros: u64,
    pub loss_without_evidence_micros: u64,
    /// Positive means retaining the item lowered loss; negative means the item
    /// increased loss under the frozen downstream decision policy.
    pub contribution_micros: i128,
    pub removal_changed_action: bool,
    pub removal_helped: bool,
    pub removal_harmed: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PairedScoreV1 {
    pub episode_id: String,
    pub downstream_policy_digest: String,
    pub decision_seed: u64,
    pub memory: RunScoreV1,
    pub no_memory: RunScoreV1,
    /// Positive means memory lowered decision loss; negative means memory
    /// increased decision loss relative to the paired no-memory control.
    pub memory_gain_micros: i128,
    pub memory_induced_harm: bool,
    pub memory_changed_action: bool,
    pub changed_action_helped: bool,
    pub changed_action_harmed: bool,
}

/// One completed paired episode, assigned to its independent world cluster.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClusteredPairedScoreV1 {
    pub cluster_id: String,
    pub score: PairedScoreV1,
}

/// Percentile interval for a cluster-bootstrap statistic.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConfidenceIntervalV1 {
    pub lower: f64,
    pub upper: f64,
}

/// Episode-weighted point estimates with cluster-resampled uncertainty.
#[derive(Clone, Debug, PartialEq)]
pub struct ClusteredSummaryV1 {
    pub episode_count: u64,
    pub cluster_count: u64,
    pub mean_memory_gain_micros: f64,
    pub memory_induced_harm_rate: CountRatioV1,
    pub memory_improvement_rate: CountRatioV1,
    pub mean_gain_confidence_95: ConfidenceIntervalV1,
    pub harm_rate_confidence_95: ConfidenceIntervalV1,
    pub bootstrap_replicates: u64,
    pub bootstrap_seed: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PhaseTimingsV1 {
    pub add_ingest_ns: u64,
    pub index_update_ns: u64,
    pub retrieval_ns: u64,
    pub evidence_verification_ns: u64,
    pub policy_application_ns: u64,
    pub maintenance_ns: u64,
    pub unattributed_ns: u64,
    pub wall_ns: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ResourceMeasurementStatusV1 {
    Measured,
    Estimated,
    #[default]
    Unknown,
    NotApplicable,
}

/// A byte quantity that cannot silently turn an unreported value into zero.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ResourceQuantityV1 {
    pub value: Option<u64>,
    pub status: ResourceMeasurementStatusV1,
}

impl ResourceQuantityV1 {
    fn is_consistent(self) -> bool {
        match self.status {
            ResourceMeasurementStatusV1::Measured | ResourceMeasurementStatusV1::Estimated => {
                self.value.is_some()
            }
            ResourceMeasurementStatusV1::Unknown | ResourceMeasurementStatusV1::NotApplicable => {
                self.value.is_none()
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CostReceiptV1 {
    pub phases: PhaseTimingsV1,
    /// Monotonic, measured resource cost in an implementation-defined unit.
    pub measured_cost_microunits: Option<u64>,
    /// Separately identified estimate; never substitute this for measurement.
    pub estimated_cost_microunits: Option<u64>,
    pub retained_bytes: ResourceQuantityV1,
    pub index_bytes: ResourceQuantityV1,
    pub network_bytes: ResourceQuantityV1,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CostReceiptErrorV1 {
    PhaseSumOverflow,
    PhasesDoNotReconcile,
    ResourceStatusValueMismatch,
}

impl CostReceiptV1 {
    /// Ensure non-overlapping measured phases, including residual, equal wall time.
    pub fn validate(&self) -> Result<(), CostReceiptErrorV1> {
        if !self.retained_bytes.is_consistent()
            || !self.index_bytes.is_consistent()
            || !self.network_bytes.is_consistent()
        {
            return Err(CostReceiptErrorV1::ResourceStatusValueMismatch);
        }
        let p = self.phases;
        let sum = [
            p.add_ingest_ns,
            p.index_update_ns,
            p.retrieval_ns,
            p.evidence_verification_ns,
            p.policy_application_ns,
            p.maintenance_ns,
            p.unattributed_ns,
        ]
        .into_iter()
        .try_fold(0_u128, |acc, value| {
            acc.checked_add(u128::from(value))
        })
        .ok_or(CostReceiptErrorV1::PhaseSumOverflow)?;

        if sum != u128::from(p.wall_ns) {
            return Err(CostReceiptErrorV1::PhasesDoNotReconcile);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EvaluationErrorV1 {
    EmptyEpisodeId,
    EmptySystemId,
    DuplicateAllowedAction,
    DuplicateLossAction,
    LossActionSetMismatch,
    InvalidNoMemoryAction,
    InvalidOracleAction,
    OracleActionIsNotOptimal,
    DuplicateAdmissibleEvidenceId,
    DuplicateRequiredEvidenceId,
    RequiredEvidenceIsNotAdmissible,
    DuplicateEvidenceLineageId,
    InvalidEvidenceLineage,
    MissingEvidenceLineage,
    DuplicateContradictionPair,
    InvalidContradictionPair,
    UnknownRetrievedEvidence,
    DecisionSupportEvidenceNotRetrieved,
    EpisodeMismatch,
    SameSystemUsedForBothArms,
    EmptyDownstreamPolicyDigest,
    InvalidDownstreamPolicyDigest,
    DownstreamPolicyMismatch,
    DecisionSeedMismatch,
    InvalidCandidateAction,
    EmptyActionId,
    EmptyEvidenceId,
    DuplicateReturnedEvidenceId,
    DuplicateDecisionSupportEvidenceId,
    NoMemoryArmContainsEvidence,
    NoMemoryArmActionMismatch,
    AblationTargetMissing,
    DuplicateAblationTarget,
    AblationSetIncomplete,
    AblationEvidenceSetMismatch,
    AblationSelectedSetMismatch,
    AblationSystemMismatch,
    NoClusteredEpisodes,
    EmptyClusterId,
    DuplicateEpisodeId,
    InsufficientIndependentClusters,
    InvalidBootstrapReplicates,
    AggregateOverflow,
    InvalidPairedScore,
    MissingActionLoss,
}

fn contains(values: &[String], target: &str) -> bool {
    values.iter().any(|value| value == target)
}

fn is_lowercase_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

#[derive(Clone, Copy, Debug, Default)]
struct SourceLineageCountsV1 {
    distinct_families: u64,
    unknown_families: u64,
    distinct_roots: u64,
    unknown_roots: u64,
}

fn source_lineage_counts(
    case: &EvaluatorCaseV1,
    evidence_ids: &[String],
) -> SourceLineageCountsV1 {
    let mut families = BTreeSet::new();
    let mut roots = BTreeSet::new();
    let mut unknown_families = 0_u64;
    let mut unknown_roots = 0_u64;

    for evidence_id in evidence_ids {
        if let Some(lineage) = case
            .evidence_lineage
            .iter()
            .find(|lineage| lineage.evidence_id == *evidence_id)
        {
            if let Some(family) = &lineage.source_family {
                families.insert(family.as_str());
            } else {
                unknown_families += 1;
            }
            if let Some(root) = &lineage.source_root_digest_sha256 {
                roots.insert(root.as_str());
            } else {
                unknown_roots += 1;
            }
        }
    }

    SourceLineageCountsV1 {
        distinct_families: families.len() as u64,
        unknown_families,
        distinct_roots: roots.len() as u64,
        unknown_roots,
    }
}

fn unresolved_conflict_count(
    case: &EvaluatorCaseV1,
    evidence_ids: &[String],
) -> u64 {
    case.contradiction_pairs
        .iter()
        .filter(|pair| {
            contains(evidence_ids, &pair.left_evidence_id)
                && contains(evidence_ids, &pair.right_evidence_id)
        })
        .count() as u64
}

fn unique(values: &[String]) -> bool {
    let mut seen = BTreeSet::new();
    values.iter().all(|value| seen.insert(value.as_str()))
}

fn action_loss(case: &EvaluatorCaseV1, action: &str) -> Option<u64> {
    case.action_losses
        .iter()
        .find(|entry| entry.action == action)
        .map(|entry| entry.loss_micros)
}

/// Validate evaluator-only ground truth before scoring any candidate output.
pub fn validate_case(case: &EvaluatorCaseV1) -> Result<(), EvaluationErrorV1> {
    if case.episode_id.trim().is_empty() {
        return Err(EvaluationErrorV1::EmptyEpisodeId);
    }
    if case.allowed_actions.is_empty() {
        return Err(EvaluationErrorV1::LossActionSetMismatch);
    }
    if !unique(&case.allowed_actions) {
        return Err(EvaluationErrorV1::DuplicateAllowedAction);
    }
    if case.allowed_actions.iter().any(|action| action.trim().is_empty())
        || case.action_losses.iter().any(|entry| entry.action.trim().is_empty())
    {
        return Err(EvaluationErrorV1::EmptyActionId);
    }

    let loss_actions: Vec<String> = case
        .action_losses
        .iter()
        .map(|entry| entry.action.clone())
        .collect();
    if !unique(&loss_actions) {
        return Err(EvaluationErrorV1::DuplicateLossAction);
    }

    let allowed: BTreeSet<&str> = case.allowed_actions.iter().map(String::as_str).collect();
    let losses: BTreeSet<&str> = case
        .action_losses
        .iter()
        .map(|entry| entry.action.as_str())
        .collect();
    if allowed != losses {
        return Err(EvaluationErrorV1::LossActionSetMismatch);
    }

    let mut lineage_ids = BTreeSet::new();
    for lineage in &case.evidence_lineage {
        if lineage.evidence_id.trim().is_empty()
            || !lineage_ids.insert(lineage.evidence_id.as_str())
        {
            return Err(EvaluationErrorV1::DuplicateEvidenceLineageId);
        }
        if lineage.source_family.as_ref().is_some_and(|family| family.trim().is_empty())
            || lineage.source_root_digest_sha256.as_ref().is_some_and(|root| !is_lowercase_sha256(root))
            || (lineage.source_root_digest_sha256.is_some() && lineage.source_family.is_none())
        {
            return Err(EvaluationErrorV1::InvalidEvidenceLineage);
        }
    }
    if case.admissible_evidence_ids.iter().chain(case.required_evidence_ids.iter())
        .any(|evidence_id| !lineage_ids.contains(evidence_id.as_str()))
    {
        return Err(EvaluationErrorV1::MissingEvidenceLineage);
    }
    let mut conflict_pairs = BTreeSet::new();
    for pair in &case.contradiction_pairs {
        if pair.left_evidence_id.trim().is_empty()
            || pair.right_evidence_id.trim().is_empty()
            || pair.left_evidence_id == pair.right_evidence_id
            || !lineage_ids.contains(pair.left_evidence_id.as_str())
            || !lineage_ids.contains(pair.right_evidence_id.as_str())
        {
            return Err(EvaluationErrorV1::InvalidContradictionPair);
        }
        let canonical = if pair.left_evidence_id < pair.right_evidence_id {
            (pair.left_evidence_id.as_str(), pair.right_evidence_id.as_str())
        } else {
            (pair.right_evidence_id.as_str(), pair.left_evidence_id.as_str())
        };
        if !conflict_pairs.insert(canonical) {
            return Err(EvaluationErrorV1::DuplicateContradictionPair);
        }
    }

    if !contains(&case.allowed_actions, &case.no_memory_action) {
        return Err(EvaluationErrorV1::InvalidNoMemoryAction);
    }
    if !contains(&case.allowed_actions, &case.oracle_action) {
        return Err(EvaluationErrorV1::InvalidOracleAction);
    }

    let oracle_loss =
        action_loss(case, &case.oracle_action).ok_or(EvaluationErrorV1::MissingActionLoss)?;
    let minimum_loss = case
        .action_losses
        .iter()
        .map(|entry| entry.loss_micros)
        .min()
        .ok_or(EvaluationErrorV1::LossActionSetMismatch)?;
    if oracle_loss != minimum_loss {
        return Err(EvaluationErrorV1::OracleActionIsNotOptimal);
    }

    if !unique(&case.admissible_evidence_ids) {
        return Err(EvaluationErrorV1::DuplicateAdmissibleEvidenceId);
    }
    if !unique(&case.required_evidence_ids) {
        return Err(EvaluationErrorV1::DuplicateRequiredEvidenceId);
    }
    if case.admissible_evidence_ids.iter().any(|id| id.trim().is_empty())
        || case.required_evidence_ids.iter().any(|id| id.trim().is_empty())
    {
        return Err(EvaluationErrorV1::EmptyEvidenceId);
    }
    if case
        .required_evidence_ids
        .iter()
        .any(|id| !contains(&case.admissible_evidence_ids, id))
    {
        return Err(EvaluationErrorV1::RequiredEvidenceIsNotAdmissible);
    }
    Ok(())
}

fn validate_candidate(
    case: &EvaluatorCaseV1,
    candidate: &CandidateRunV1,
) -> Result<(), EvaluationErrorV1> {
    if candidate.episode_id.trim().is_empty() || candidate.episode_id != case.episode_id {
        return Err(EvaluationErrorV1::EpisodeMismatch);
    }
    if candidate.system_id.trim().is_empty() {
        return Err(EvaluationErrorV1::EmptySystemId);
    }
    if candidate.downstream_policy_digest.is_empty() {
        return Err(EvaluationErrorV1::EmptyDownstreamPolicyDigest);
    }
    if candidate.downstream_policy_digest.len() != 64
        || !candidate
            .downstream_policy_digest
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(EvaluationErrorV1::InvalidDownstreamPolicyDigest);
    }
    if !contains(&case.allowed_actions, &candidate.chosen_action) {
        return Err(EvaluationErrorV1::InvalidCandidateAction);
    }
    if !unique(&candidate.retrieved_evidence_ids) {
        return Err(EvaluationErrorV1::DuplicateReturnedEvidenceId);
    }
    if candidate
        .retrieved_evidence_ids
        .iter()
        .any(|id| id.trim().is_empty())
    {
        return Err(EvaluationErrorV1::EmptyEvidenceId);
    }
    if !unique(&candidate.decision_support_evidence_ids) {
        return Err(EvaluationErrorV1::DuplicateDecisionSupportEvidenceId);
    }
    if candidate.decision_support_evidence_ids.iter().any(|id| id.trim().is_empty()) {
        return Err(EvaluationErrorV1::EmptyEvidenceId);
    }
    if candidate.decision_support_evidence_ids.iter()
        .any(|id| !contains(&candidate.retrieved_evidence_ids, id))
    {
        return Err(EvaluationErrorV1::DecisionSupportEvidenceNotRetrieved);
    }
    if candidate.retrieved_evidence_ids.iter()
        .any(|id| !case.evidence_lineage.iter().any(|lineage| lineage.evidence_id == *id))
    {
        return Err(EvaluationErrorV1::UnknownRetrievedEvidence);
    }
    Ok(())
}

fn score_run(
    case: &EvaluatorCaseV1,
    candidate: &CandidateRunV1,
) -> Result<RunScoreV1, EvaluationErrorV1> {
    validate_candidate(case, candidate)?;

    let decision_loss_micros = action_loss(case, &candidate.chosen_action)
        .ok_or(EvaluationErrorV1::MissingActionLoss)?;
    let oracle_loss_micros =
        action_loss(case, &case.oracle_action).ok_or(EvaluationErrorV1::MissingActionLoss)?;
    let admissible: BTreeSet<&str> = case
        .admissible_evidence_ids
        .iter()
        .map(String::as_str)
        .collect();
    let required: BTreeSet<&str> = case
        .required_evidence_ids
        .iter()
        .map(String::as_str)
        .collect();

    let returned = candidate.retrieved_evidence_ids.len() as u64;
    let admissible_returned = candidate
        .retrieved_evidence_ids
        .iter()
        .filter(|id| admissible.contains(id.as_str()))
        .count() as u64;
    let retrieved_required = candidate
        .retrieved_evidence_ids
        .iter()
        .filter(|id| required.contains(id.as_str()))
        .count() as u64;
    let selected = candidate.decision_support_evidence_ids.len() as u64;
    let admissible_selected = candidate
        .decision_support_evidence_ids
        .iter()
        .filter(|id| admissible.contains(id.as_str()))
        .count() as u64;
    let selected_required = candidate
        .decision_support_evidence_ids
        .iter()
        .filter(|id| required.contains(id.as_str()))
        .count() as u64;
    let retrieved_lineage = source_lineage_counts(case, &candidate.retrieved_evidence_ids);
    let selected_lineage = source_lineage_counts(case, &candidate.decision_support_evidence_ids);
    let retrieved_conflicts = unresolved_conflict_count(case, &candidate.retrieved_evidence_ids);
    let selected_conflicts = unresolved_conflict_count(case, &candidate.decision_support_evidence_ids);
    let decision_optimal = decision_loss_micros == oracle_loss_micros;
    let all_required_selected = required.is_empty() || selected_required == required.len() as u64;
    let decision_sufficient = decision_optimal
        && all_required_selected
        && admissible_returned == returned
        && retrieved_conflicts == 0;

    Ok(RunScoreV1 {
        system_id: candidate.system_id.clone(),
        chosen_action: candidate.chosen_action.clone(),
        decision_loss_micros,
        oracle_loss_micros,
        decision_regret_micros: decision_loss_micros - oracle_loss_micros,
        decision_optimal,
        returned_evidence_count: returned,
        admissible_returned_evidence_count: admissible_returned,
        inadmissible_returned_evidence_count: returned - admissible_returned,
        decision_support_evidence_count: selected,
        admissible_decision_support_evidence_count: admissible_selected,
        inadmissible_decision_support_evidence_count: selected - admissible_selected,
        retrieved_required_evidence_recall: (!required.is_empty()).then_some(CountRatioV1 {
            numerator: retrieved_required,
            denominator: required.len() as u64,
        }),
        required_evidence_recall: (!required.is_empty()).then_some(CountRatioV1 {
            numerator: selected_required,
            denominator: required.len() as u64,
        }),
        decision_support_evidence_precision: (selected > 0).then_some(CountRatioV1 {
            numerator: admissible_selected,
            denominator: selected,
        }),
        retrieved_distinct_source_family_count: retrieved_lineage.distinct_families,
        retrieved_unknown_source_family_count: retrieved_lineage.unknown_families,
        retrieved_distinct_source_root_count: retrieved_lineage.distinct_roots,
        retrieved_unknown_source_root_count: retrieved_lineage.unknown_roots,
        decision_support_distinct_source_family_count: selected_lineage.distinct_families,
        decision_support_unknown_source_family_count: selected_lineage.unknown_families,
        decision_support_distinct_source_root_count: selected_lineage.distinct_roots,
        decision_support_unknown_source_root_count: selected_lineage.unknown_roots,
        unresolved_retrieved_contradiction_pair_count: retrieved_conflicts,
        unresolved_decision_support_contradiction_pair_count: selected_conflicts,
        decision_sufficient,
        admissible_evidence_precision: (returned > 0).then_some(CountRatioV1 {
            numerator: admissible_returned,
            denominator: returned,
        }),
    })
}

/// Score memory and no-memory arms on the exact same evaluator episode.
///
/// The no-memory arm is constrained to the declared no-memory action and must
/// not return evidence. Candidate outputs are validated but never used to
/// derive evaluator labels, losses, admissible IDs or the oracle action.
pub fn score_paired(
    case: &EvaluatorCaseV1,
    memory: &CandidateRunV1,
    no_memory: &CandidateRunV1,
) -> Result<PairedScoreV1, EvaluationErrorV1> {
    validate_case(case)?;
    validate_candidate(case, memory)?;
    validate_candidate(case, no_memory)?;

    if memory.system_id == no_memory.system_id {
        return Err(EvaluationErrorV1::SameSystemUsedForBothArms);
    }
    if memory.downstream_policy_digest != no_memory.downstream_policy_digest {
        return Err(EvaluationErrorV1::DownstreamPolicyMismatch);
    }
    if memory.decision_seed != no_memory.decision_seed {
        return Err(EvaluationErrorV1::DecisionSeedMismatch);
    }
    if no_memory.chosen_action != case.no_memory_action {
        return Err(EvaluationErrorV1::NoMemoryArmActionMismatch);
    }
    if !no_memory.retrieved_evidence_ids.is_empty()
        || !no_memory.decision_support_evidence_ids.is_empty()
    {
        return Err(EvaluationErrorV1::NoMemoryArmContainsEvidence);
    }

    let memory_score = score_run(case, memory)?;
    let no_memory_score = score_run(case, no_memory)?;
    let memory_gain_micros =
        i128::from(no_memory_score.decision_loss_micros)
            - i128::from(memory_score.decision_loss_micros);
    let memory_changed_action = memory.chosen_action != no_memory.chosen_action;

    Ok(PairedScoreV1 {
        episode_id: case.episode_id.clone(),
        downstream_policy_digest: memory.downstream_policy_digest.clone(),
        decision_seed: memory.decision_seed,
        memory: memory_score,
        no_memory: no_memory_score,
        memory_gain_micros,
        memory_induced_harm: memory_gain_micros < 0,
        memory_changed_action,
        changed_action_helped: memory_changed_action && memory_gain_micros > 0,
        changed_action_harmed: memory_changed_action && memory_gain_micros < 0,
    })
}

/// Attribute the decision effect of each returned memory item by rerunning
/// the same deterministic downstream policy with exactly that one item
/// removed. Positive contribution means the item helped; negative means it
/// harmed. This is a leave-one-out intervention, not a Shapley value: item
/// interactions require additional subset interventions and are reported
/// separately by future benchmark versions.
pub fn score_leave_one_out(
    case: &EvaluatorCaseV1,
    full: &CandidateRunV1,
    ablations: &[EvidenceAblationV1],
) -> Result<Vec<EvidenceInfluenceV1>, EvaluationErrorV1> {
    validate_case(case)?;
    validate_candidate(case, full)?;

    if ablations.len() != full.retrieved_evidence_ids.len() {
        return Err(EvaluationErrorV1::AblationSetIncomplete);
    }

    let full_score = score_run(case, full)?;
    let mut seen = BTreeSet::new();
    let mut results = Vec::with_capacity(ablations.len());

    for ablation in ablations {
        if !contains(&full.retrieved_evidence_ids, &ablation.removed_evidence_id) {
            return Err(EvaluationErrorV1::AblationTargetMissing);
        }
        if !seen.insert(ablation.removed_evidence_id.as_str()) {
            return Err(EvaluationErrorV1::DuplicateAblationTarget);
        }
        if ablation.output_without_evidence.system_id != full.system_id {
            return Err(EvaluationErrorV1::AblationSystemMismatch);
        }
        if ablation.output_without_evidence.downstream_policy_digest
            != full.downstream_policy_digest
        {
            return Err(EvaluationErrorV1::DownstreamPolicyMismatch);
        }
        if ablation.output_without_evidence.decision_seed != full.decision_seed {
            return Err(EvaluationErrorV1::DecisionSeedMismatch);
        }
        validate_candidate(case, &ablation.output_without_evidence)?;

        let expected_evidence: Vec<String> = full
            .retrieved_evidence_ids
            .iter()
            .filter(|id| *id != &ablation.removed_evidence_id)
            .cloned()
            .collect();
        if ablation.output_without_evidence.retrieved_evidence_ids != expected_evidence {
            return Err(EvaluationErrorV1::AblationEvidenceSetMismatch);
        }
        let expected_support: Vec<String> = full
            .decision_support_evidence_ids
            .iter()
            .filter(|id| *id != &ablation.removed_evidence_id)
            .cloned()
            .collect();
        if ablation.output_without_evidence.decision_support_evidence_ids != expected_support {
            return Err(EvaluationErrorV1::AblationSelectedSetMismatch);
        }

        let without_score = score_run(case, &ablation.output_without_evidence)?;
        let contribution_micros = i128::from(without_score.decision_loss_micros)
            - i128::from(full_score.decision_loss_micros);
        let removal_changed_action = without_score.chosen_action != full_score.chosen_action;

        results.push(EvidenceInfluenceV1 {
            evidence_id: ablation.removed_evidence_id.clone(),
            full_decision_loss_micros: full_score.decision_loss_micros,
            loss_without_evidence_micros: without_score.decision_loss_micros,
            contribution_micros,
            removal_changed_action,
            removal_helped: removal_changed_action && contribution_micros < 0,
            removal_harmed: removal_changed_action && contribution_micros > 0,
        });
    }

    if seen.len() != full.retrieved_evidence_ids.len() {
        return Err(EvaluationErrorV1::AblationSetIncomplete);
    }
    Ok(results)
}

const MIN_BOOTSTRAP_REPLICATES: usize = 100;

/// Small deterministic SplitMix64 generator; used only to make cluster
/// bootstrap resampling exactly reproducible from the run manifest seed.
struct SplitMix64V1 {
    state: u64,
}

impl SplitMix64V1 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    fn index(&mut self, upper: usize) -> usize {
        let bound = u64::try_from(upper).unwrap_or(u64::MAX);
        let zone = u64::MAX - (u64::MAX % bound);
        loop {
            let value = self.next_u64();
            if value < zone {
                return usize::try_from(value % bound).unwrap_or(0);
            }
        }
    }
}

fn valid_ratio_v1(
    ratio: Option<CountRatioV1>,
    expected_denominator: Option<u64>,
) -> bool {
    match ratio {
        Some(value) => {
            value.denominator > 0
                && value.numerator <= value.denominator
                && expected_denominator.map_or(true, |expected| value.denominator == expected)
        }
        None => expected_denominator.map_or(true, |expected| expected == 0),
    }
}

fn valid_run_score_v1(score: &RunScoreV1) -> bool {
    let retrieved_partition = score
        .admissible_returned_evidence_count
        .checked_add(score.inadmissible_returned_evidence_count);
    let selected_partition = score
        .admissible_decision_support_evidence_count
        .checked_add(score.inadmissible_decision_support_evidence_count);
    let retrieved_lineage = score
        .retrieved_distinct_source_family_count
        .checked_add(score.retrieved_unknown_source_family_count)
        .zip(score.retrieved_distinct_source_root_count
            .checked_add(score.retrieved_unknown_source_root_count));
    let selected_lineage = score
        .decision_support_distinct_source_family_count
        .checked_add(score.decision_support_unknown_source_family_count)
        .zip(score.decision_support_distinct_source_root_count
            .checked_add(score.decision_support_unknown_source_root_count));
    let recall_shapes_match = match (
        score.retrieved_required_evidence_recall,
        score.required_evidence_recall,
    ) {
        (Some(retrieved), Some(selected)) => {
            retrieved.denominator == selected.denominator
                && selected.numerator <= retrieved.numerator
        }
        (None, None) => true,
        _ => false,
    };
    let required_is_complete = score.required_evidence_recall
        .map_or(true, |ratio| ratio.numerator == ratio.denominator)
        && score.retrieved_required_evidence_recall
            .map_or(true, |ratio| ratio.numerator == ratio.denominator);

    if score.decision_regret_micros != score.decision_loss_micros.saturating_sub(score.oracle_loss_micros)
        || score.decision_optimal != (score.decision_regret_micros == 0)
        || retrieved_partition != Some(score.returned_evidence_count)
        || selected_partition != Some(score.decision_support_evidence_count)
        || score.decision_support_evidence_count > score.returned_evidence_count
        || score.admissible_decision_support_evidence_count
            > score.admissible_returned_evidence_count
        || score.inadmissible_decision_support_evidence_count
            > score.inadmissible_returned_evidence_count
        || score.decision_support_distinct_source_family_count
            > score.retrieved_distinct_source_family_count
        || score.decision_support_unknown_source_family_count
            > score.retrieved_unknown_source_family_count
        || score.decision_support_distinct_source_root_count
            > score.retrieved_distinct_source_root_count
        || score.decision_support_unknown_source_root_count
            > score.retrieved_unknown_source_root_count
        || !valid_ratio_v1(score.admissible_evidence_precision, Some(score.returned_evidence_count))
        || !valid_ratio_v1(
            score.decision_support_evidence_precision,
            Some(score.decision_support_evidence_count),
        )
        || !valid_ratio_v1(score.retrieved_required_evidence_recall, None)
        || !valid_ratio_v1(score.required_evidence_recall, None)
        || !recall_shapes_match
        || retrieved_lineage.is_none_or(|(families, roots)| families > score.returned_evidence_count || roots > score.returned_evidence_count)
        || selected_lineage.is_none_or(|(families, roots)| families > score.decision_support_evidence_count || roots > score.decision_support_evidence_count)
        || score.unresolved_decision_support_contradiction_pair_count
            > score.unresolved_retrieved_contradiction_pair_count
    {
        return false;
    }

    !score.decision_sufficient
        || (score.decision_optimal
            && score.inadmissible_returned_evidence_count == 0
            && score.inadmissible_decision_support_evidence_count == 0
            && score.unresolved_retrieved_contradiction_pair_count == 0
            && required_is_complete)
}

fn validate_paired_score(score: &PairedScoreV1) -> Result<(), EvaluationErrorV1> {
    if !valid_run_score_v1(&score.memory) || !valid_run_score_v1(&score.no_memory) {
        return Err(EvaluationErrorV1::InvalidPairedScore);
    }
    if score.episode_id.trim().is_empty()
        || score.downstream_policy_digest.len() != 64
        || !score
            .downstream_policy_digest
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        || score.memory.system_id == score.no_memory.system_id
        || score.memory.oracle_loss_micros != score.no_memory.oracle_loss_micros
        || score.memory.decision_loss_micros.checked_sub(score.memory.oracle_loss_micros)
            != Some(score.memory.decision_regret_micros)
        || score.no_memory.decision_loss_micros.checked_sub(score.no_memory.oracle_loss_micros)
            != Some(score.no_memory.decision_regret_micros)
        || score.memory.decision_optimal != (score.memory.decision_regret_micros == 0)
        || score.no_memory.decision_optimal != (score.no_memory.decision_regret_micros == 0)
        || score.no_memory.returned_evidence_count != 0
        || score.no_memory.admissible_returned_evidence_count != 0
        || score.no_memory.inadmissible_returned_evidence_count != 0
        || score.no_memory.decision_support_evidence_count != 0
        || score.no_memory.admissible_decision_support_evidence_count != 0
        || score.no_memory.inadmissible_decision_support_evidence_count != 0
        || score.no_memory.decision_support_distinct_source_family_count != 0
        || score.no_memory.decision_support_unknown_source_family_count != 0
        || score.no_memory.decision_support_distinct_source_root_count != 0
        || score.no_memory.decision_support_unknown_source_root_count != 0
        || score.no_memory.unresolved_retrieved_contradiction_pair_count != 0
        || score.no_memory.unresolved_decision_support_contradiction_pair_count != 0
        || score.memory.admissible_returned_evidence_count
            .checked_add(score.memory.inadmissible_returned_evidence_count)
            != Some(score.memory.returned_evidence_count)
    {
        return Err(EvaluationErrorV1::InvalidPairedScore);
    }

    let expected_gain = i128::from(score.no_memory.decision_loss_micros)
        - i128::from(score.memory.decision_loss_micros);
    let changed_action = score.memory.chosen_action != score.no_memory.chosen_action;
    if score.memory_gain_micros != expected_gain
        || score.memory_induced_harm != (expected_gain < 0)
        || score.memory_changed_action != changed_action
        || score.changed_action_helped != (changed_action && expected_gain > 0)
        || score.changed_action_harmed != (changed_action && expected_gain < 0)
    {
        return Err(EvaluationErrorV1::InvalidPairedScore);
    }
    Ok(())
}

fn nearest_rank(sorted: &[f64], probability: f64) -> f64 {
    let rank = (probability * sorted.len() as f64).ceil() as usize;
    let index = rank.clamp(1, sorted.len()) - 1;
    sorted[index]
}

/// Summarize paired memory effects using a cluster bootstrap.
///
/// Each bootstrap draw samples independent world clusters with replacement;
/// all episodes from each selected cluster move together. The point estimate
/// is episode-weighted. The bootstrap interval therefore preserves within-world
/// dependence instead of falsely treating related questions as independent.
///
/// The returned intervals are percentile 95% intervals, not posterior
/// probabilities. They describe the sampled benchmark worlds and do not
/// establish generalization beyond the frozen scenario distribution.
pub fn summarize_clustered_pairs(
    rows: &[ClusteredPairedScoreV1],
    bootstrap_replicates: usize,
    bootstrap_seed: u64,
) -> Result<ClusteredSummaryV1, EvaluationErrorV1> {
    if rows.is_empty() {
        return Err(EvaluationErrorV1::NoClusteredEpisodes);
    }
    if bootstrap_replicates < MIN_BOOTSTRAP_REPLICATES {
        return Err(EvaluationErrorV1::InvalidBootstrapReplicates);
    }

    let mut seen_episodes = BTreeSet::new();
    let mut clusters = std::collections::BTreeMap::<String, Vec<i128>>::new();
    let mut total_gain = 0_i128;
    let mut harm_count = 0_u64;
    let mut improvement_count = 0_u64;

    for row in rows {
        validate_paired_score(&row.score)?;
        if row.cluster_id.trim().is_empty() {
            return Err(EvaluationErrorV1::EmptyClusterId);
        }
        if !seen_episodes.insert(row.score.episode_id.as_str()) {
            return Err(EvaluationErrorV1::DuplicateEpisodeId);
        }
        total_gain = total_gain
            .checked_add(row.score.memory_gain_micros)
            .ok_or(EvaluationErrorV1::AggregateOverflow)?;
        if row.score.memory_gain_micros < 0 {
            harm_count = harm_count
                .checked_add(1)
                .ok_or(EvaluationErrorV1::AggregateOverflow)?;
        }
        if row.score.memory_gain_micros > 0 {
            improvement_count = improvement_count
                .checked_add(1)
                .ok_or(EvaluationErrorV1::AggregateOverflow)?;
        }
        clusters
            .entry(row.cluster_id.clone())
            .or_default()
            .push(row.score.memory_gain_micros);
    }

    if clusters.len() < 2 {
        return Err(EvaluationErrorV1::InsufficientIndependentClusters);
    }

    let episode_count =
        u64::try_from(rows.len()).map_err(|_| EvaluationErrorV1::AggregateOverflow)?;
    let cluster_count =
        u64::try_from(clusters.len()).map_err(|_| EvaluationErrorV1::AggregateOverflow)?;
    let mean_memory_gain_micros = total_gain as f64 / episode_count as f64;

    let cluster_values: Vec<&Vec<i128>> = clusters.values().collect();
    let mut rng = SplitMix64V1::new(bootstrap_seed);
    let mut gain_samples = Vec::with_capacity(bootstrap_replicates);
    let mut harm_samples = Vec::with_capacity(bootstrap_replicates);

    for _ in 0..bootstrap_replicates {
        let mut sample_gain = 0_i128;
        let mut sample_harms = 0_u64;
        let mut sample_episodes = 0_u64;

        // Draw K independent clusters, where K is the observed cluster count.
        for _ in 0..cluster_values.len() {
            let selected = cluster_values[rng.index(cluster_values.len())];
            for gain in selected {
                sample_gain = sample_gain
                    .checked_add(*gain)
                    .ok_or(EvaluationErrorV1::AggregateOverflow)?;
                sample_episodes = sample_episodes
                    .checked_add(1)
                    .ok_or(EvaluationErrorV1::AggregateOverflow)?;
                if *gain < 0 {
                    sample_harms = sample_harms
                        .checked_add(1)
                        .ok_or(EvaluationErrorV1::AggregateOverflow)?;
                }
            }
        }

        if sample_episodes == 0 {
            return Err(EvaluationErrorV1::NoClusteredEpisodes);
        }
        gain_samples.push(sample_gain as f64 / sample_episodes as f64);
        harm_samples.push(sample_harms as f64 / sample_episodes as f64);
    }

    gain_samples.sort_by(f64::total_cmp);
    harm_samples.sort_by(f64::total_cmp);

    Ok(ClusteredSummaryV1 {
        episode_count,
        cluster_count,
        mean_memory_gain_micros,
        memory_induced_harm_rate: CountRatioV1 {
            numerator: harm_count,
            denominator: episode_count,
        },
        memory_improvement_rate: CountRatioV1 {
            numerator: improvement_count,
            denominator: episode_count,
        },
        mean_gain_confidence_95: ConfidenceIntervalV1 {
            lower: nearest_rank(&gain_samples, 0.025),
            upper: nearest_rank(&gain_samples, 0.975),
        },
        harm_rate_confidence_95: ConfidenceIntervalV1 {
            lower: nearest_rank(&harm_samples, 0.025),
            upper: nearest_rank(&harm_samples, 0.975),
        },
        bootstrap_replicates: u64::try_from(bootstrap_replicates)
            .map_err(|_| EvaluationErrorV1::AggregateOverflow)?,
        bootstrap_seed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Public known-answer fixture only. Confirmatory evaluator answer keys and
    // held-out truth must live outside the candidate-accessible repository.
    fn case() -> EvaluatorCaseV1 {
        EvaluatorCaseV1 {
            episode_id: "known-answer/0001".to_owned(),
            allowed_actions: vec![
                "approve".to_owned(),
                "hold".to_owned(),
                "abstain".to_owned(),
            ],
            action_losses: vec![
                ActionLossV1 {
                    action: "approve".to_owned(),
                    loss_micros: 9_000_000,
                },
                ActionLossV1 {
                    action: "hold".to_owned(),
                    loss_micros: 0,
                },
                ActionLossV1 {
                    action: "abstain".to_owned(),
                    loss_micros: 2_000_000,
                },
            ],
            no_memory_action: "abstain".to_owned(),
            oracle_action: "hold".to_owned(),
            admissible_evidence_ids: vec!["signed-current".to_owned()],
            required_evidence_ids: vec!["signed-current".to_owned()],
            evidence_lineage: vec![
                EvidenceLineageV1 {
                    evidence_id: "signed-current".to_owned(),
                    source_family: Some("supplier-registry".to_owned()),
                    source_root_digest_sha256: Some("a".repeat(64)),
                },
                EvidenceLineageV1 {
                    evidence_id: "expired-account".to_owned(),
                    source_family: Some("supplier-registry".to_owned()),
                    source_root_digest_sha256: Some("a".repeat(64)),
                },
                EvidenceLineageV1 {
                    evidence_id: "chat-assertion".to_owned(),
                    source_family: Some("team-chat".to_owned()),
                    source_root_digest_sha256: None,
                },
                EvidenceLineageV1 {
                    evidence_id: "signed-release-gate".to_owned(),
                    source_family: Some("release-gate".to_owned()),
                    source_root_digest_sha256: Some("b".repeat(64)),
                },
            ],
            contradiction_pairs: vec![EvidenceConflictV1 {
                left_evidence_id: "chat-assertion".to_owned(),
                right_evidence_id: "signed-release-gate".to_owned(),
            }],
        }
    }

    fn run(system: &str, action: &str, evidence: &[&str]) -> CandidateRunV1 {
        CandidateRunV1 {
            episode_id: "known-answer/0001".to_owned(),
            system_id: system.to_owned(),
            downstream_policy_digest: "a".repeat(64),
            decision_seed: 4101,
            chosen_action: action.to_owned(),
            retrieved_evidence_ids: evidence.iter().map(|id| (*id).to_owned()).collect(),
            decision_support_evidence_ids: evidence.iter().map(|id| (*id).to_owned()).collect(),
        }
    }

    #[test]
    fn qualified_memory_reduces_regret_against_paired_no_memory() {
        let result = score_paired(
            &case(),
            &run("mace-qualified", "hold", &["signed-current"]),
            &run("no-memory", "abstain", &[]),
        )
        .expect("valid paired evaluation");

        assert_eq!(result.memory.decision_regret_micros, 0);
        assert_eq!(result.no_memory.decision_regret_micros, 2_000_000);
        assert_eq!(result.memory_gain_micros, 2_000_000);
        assert!(!result.memory_induced_harm);
        assert!(result.changed_action_helped);
        assert_eq!(
            result.memory.admissible_evidence_precision,
            Some(CountRatioV1 {
                numerator: 1,
                denominator: 1,
            })
        );
    }

    #[test]
    fn paired_validation_rejects_tampered_evidence_partitions() {
        let mut paired = score_paired(
            &case(),
            &run("mace", "hold", &["signed-current"]),
            &run("no-memory", "abstain", &[]),
        ).expect("valid paired result");
        paired.memory.inadmissible_returned_evidence_count += 1;
        assert_eq!(validate_paired_score(&paired), Err(EvaluationErrorV1::InvalidPairedScore));
    }

    #[test]
    fn paired_validation_rejects_decision_support_metrics_exceeding_retrieval() {
        let mut paired = score_paired(
            &case(),
            &run("mace", "hold", &["signed-current", "expired-account"]),
            &run("no-memory", "abstain", &[]),
        ).expect("valid paired result");
        paired.memory.decision_support_distinct_source_root_count += 1;
        assert_eq!(validate_paired_score(&paired), Err(EvaluationErrorV1::InvalidPairedScore));
    }

    #[test]
    fn paired_validation_rejects_falsely_sufficient_retrieved_context() {
        let mut memory = run("mace", "hold", &["signed-current", "expired-account"]);
        memory.decision_support_evidence_ids = vec!["signed-current".to_owned()];
        let mut paired = score_paired(
            &case(),
            &memory,
            &run("no-memory", "abstain", &[]),
        ).expect("candidate score remains structurally valid");
        assert!(!paired.memory.decision_sufficient);
        paired.memory.decision_sufficient = true;
        assert_eq!(validate_paired_score(&paired), Err(EvaluationErrorV1::InvalidPairedScore));
    }

    #[test]
    fn retrieval_and_decision_support_are_scored_separately() {
        let mut candidate = run("mace", "hold", &["signed-current", "expired-account"]);
        candidate.decision_support_evidence_ids = vec!["signed-current".to_owned()];
        let score = score_run(&case(), &candidate).expect("candidate IDs resolve");

        assert_eq!(score.returned_evidence_count, 2);
        assert_eq!(score.inadmissible_returned_evidence_count, 1);
        assert_eq!(score.decision_support_evidence_count, 1);
        assert_eq!(score.admissible_decision_support_evidence_count, 1);
        assert_eq!(score.inadmissible_decision_support_evidence_count, 0);
        assert_eq!(score.retrieved_required_evidence_recall, Some(CountRatioV1 { numerator: 1, denominator: 1 }));
        assert_eq!(score.required_evidence_recall, Some(CountRatioV1 { numerator: 1, denominator: 1 }));
        assert_eq!(score.admissible_evidence_precision, Some(CountRatioV1 { numerator: 1, denominator: 2 }));
        assert_eq!(score.decision_support_evidence_precision, Some(CountRatioV1 { numerator: 1, denominator: 1 }));
        assert!(!score.decision_sufficient);
    }

    #[test]
    fn decision_support_must_be_a_subset_of_retrieved_evidence() {
        let mut candidate = run("mace", "hold", &["signed-current"]);
        candidate.decision_support_evidence_ids = vec!["expired-account".to_owned()];
        assert_eq!(
            score_run(&case(), &candidate),
            Err(EvaluationErrorV1::DecisionSupportEvidenceNotRetrieved)
        );
    }

    #[test]
    fn source_family_and_root_counts_deduplicate_repeated_lineage() {
        let candidate = run(
            "mace",
            "hold",
            &["signed-current", "expired-account", "chat-assertion"],
        );
        let score = score_run(&case(), &candidate).expect("candidate IDs resolve");
        assert_eq!(score.retrieved_distinct_source_family_count, 2);
        assert_eq!(score.retrieved_distinct_source_root_count, 1);
        assert_eq!(score.retrieved_unknown_source_family_count, 0);
        assert_eq!(score.retrieved_unknown_source_root_count, 1);
        assert_eq!(score.decision_support_distinct_source_root_count, 1);
    }

    #[test]
    fn retrieved_conflict_blocks_sufficiency_even_when_not_selected_as_support() {
        let mut candidate = run(
            "mace",
            "hold",
            &["signed-current", "chat-assertion", "signed-release-gate"],
        );
        candidate.decision_support_evidence_ids = vec!["signed-current".to_owned()];
        let score = score_run(&case(), &candidate).expect("candidate IDs resolve");
        assert_eq!(score.unresolved_retrieved_contradiction_pair_count, 1);
        assert_eq!(score.unresolved_decision_support_contradiction_pair_count, 0);
        assert!(!score.decision_sufficient);
    }

    #[test]
    fn invalid_source_root_digest_is_rejected() {
        let mut evaluator_case = case();
        evaluator_case.evidence_lineage[0].source_root_digest_sha256 = Some("not-a-sha256".to_owned());
        assert_eq!(
            validate_case(&evaluator_case),
            Err(EvaluationErrorV1::InvalidEvidenceLineage)
        );
    }

    #[test]
    fn stale_or_unknown_evidence_is_counted_as_inadmissible() {
        let result = score_paired(
            &case(),
            &run("mace-stale", "approve", &["expired-account"]),
            &run("no-memory", "abstain", &[]),
        )
        .expect("candidate errors are scored, not silently trusted");

        assert_eq!(result.memory.inadmissible_returned_evidence_count, 1);
        assert_eq!(
            result.memory.admissible_evidence_precision,
            Some(CountRatioV1 {
                numerator: 0,
                denominator: 1,
            })
        );
        assert_eq!(result.memory_gain_micros, -7_000_000);
        assert!(result.memory_induced_harm);
        assert!(result.changed_action_harmed);
    }

    #[test]
    fn leave_one_out_attributes_helpful_evidence() {
        let full = run("mace", "hold", &["signed-current"]);
        let without = CandidateRunV1 {
            episode_id: full.episode_id.clone(),
            system_id: full.system_id.clone(),
            downstream_policy_digest: full.downstream_policy_digest.clone(),
            decision_seed: full.decision_seed,
            chosen_action: "abstain".to_owned(),
            retrieved_evidence_ids: vec![],
            decision_support_evidence_ids: vec![],
        };
        let result = score_leave_one_out(
            &case(),
            &full,
            &[EvidenceAblationV1 {
                removed_evidence_id: "signed-current".to_owned(),
                output_without_evidence: without,
            }],
        )
        .expect("valid leave-one-out intervention");

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].contribution_micros, 2_000_000);
        assert!(result[0].removal_changed_action);
        assert!(result[0].removal_harmed);
    }

    #[test]
    fn leave_one_out_rejects_an_ablation_that_keeps_the_removed_item() {
        let full = run("mace", "hold", &["signed-current"]);
        let invalid = score_leave_one_out(
            &case(),
            &full,
            &[EvidenceAblationV1 {
                removed_evidence_id: "signed-current".to_owned(),
                output_without_evidence: full.clone(),
            }],
        );
        assert_eq!(
            invalid,
            Err(EvaluationErrorV1::AblationEvidenceSetMismatch)
        );
    }

    #[test]
    fn candidate_cannot_change_episode_identity_between_arms() {
        let mut memory = run("mace", "hold", &["signed-current"]);
        memory.episode_id = "different-world".to_owned();
        let error = score_paired(&case(), &memory, &run("no-memory", "abstain", &[]));
        assert_eq!(error, Err(EvaluationErrorV1::EpisodeMismatch));
    }

    #[test]
    fn no_memory_control_must_not_return_evidence() {
        let error = score_paired(
            &case(),
            &run("mace", "hold", &["signed-current"]),
            &run("no-memory", "abstain", &["signed-current"]),
        );
        assert_eq!(error, Err(EvaluationErrorV1::NoMemoryArmContainsEvidence));
    }

    #[test]
    fn paired_arms_require_identical_downstream_policy_digest() {
        let memory = run("mace", "hold", &["signed-current"]);
        let mut baseline = run("no-memory", "abstain", &[]);
        baseline.downstream_policy_digest = "c".repeat(64);
        assert_eq!(
            score_paired(&case(), &memory, &baseline),
            Err(EvaluationErrorV1::DownstreamPolicyMismatch)
        );
    }

    #[test]
    fn paired_arms_require_identical_decision_seed() {
        let memory = run("mace", "hold", &["signed-current"]);
        let mut baseline = run("no-memory", "abstain", &[]);
        baseline.decision_seed = 4102;
        assert_eq!(
            score_paired(&case(), &memory, &baseline),
            Err(EvaluationErrorV1::DecisionSeedMismatch)
        );
    }

    #[test]
    fn leave_one_out_requires_identical_downstream_policy_digest() {
        let full = run("mace", "hold", &["signed-current"]);
        let mut without = CandidateRunV1 {
            episode_id: full.episode_id.clone(),
            system_id: full.system_id.clone(),
            downstream_policy_digest: full.downstream_policy_digest.clone(),
            decision_seed: full.decision_seed,
            chosen_action: "abstain".to_owned(),
            retrieved_evidence_ids: vec![],
            decision_support_evidence_ids: vec![],
        };
        without.downstream_policy_digest = "c".repeat(64);
        assert_eq!(
            score_leave_one_out(
                &case(),
                &full,
                &[EvidenceAblationV1 {
                    removed_evidence_id: "signed-current".to_owned(),
                    output_without_evidence: without,
                }],
            ),
            Err(EvaluationErrorV1::DownstreamPolicyMismatch)
        );
    }

    fn clustered_row(episode: &str, cluster: &str, gain: i8) -> ClusteredPairedScoreV1 {
        let mut evaluator_case = case();
        evaluator_case.episode_id = episode.to_owned();
        let (memory_action, memory_evidence) = if gain < 0 {
            ("approve", vec!["expired-account"])
        } else {
            ("hold", vec!["signed-current"])
        };
        let memory = CandidateRunV1 {
            episode_id: episode.to_owned(),
            system_id: "mace".to_owned(),
            downstream_policy_digest: "b".repeat(64),
            decision_seed: 77,
            chosen_action: memory_action.to_owned(),
            retrieved_evidence_ids: memory_evidence.iter().map(|id| (*id).to_owned()).collect(),
            decision_support_evidence_ids: memory_evidence.iter().map(|id| (*id).to_owned()).collect(),
        };
        let baseline = CandidateRunV1 {
            episode_id: episode.to_owned(),
            system_id: "no-memory".to_owned(),
            downstream_policy_digest: "frozen-reader-sha256:aggregate-v1".to_owned(),
            decision_seed: 77,
            chosen_action: "abstain".to_owned(),
            retrieved_evidence_ids: vec![],
            decision_support_evidence_ids: vec![],
        };
        // The candidate evidence may intentionally be inadmissible; the
        // aggregate function scores the paired result, not its correctness.
        let score = score_paired(&evaluator_case, &memory, &baseline)
            .expect("valid paired known-answer case");
        ClusteredPairedScoreV1 {
            cluster_id: cluster.to_owned(),
            score,
        }
    }

    #[test]
    fn cluster_bootstrap_resamples_whole_world_clusters_deterministically() {
        let rows = vec![
            clustered_row("episode-a1", "world-a", 1),
            clustered_row("episode-a2", "world-a", 1),
            clustered_row("episode-b1", "world-b", -1),
        ];
        let first = summarize_clustered_pairs(&rows, 1_000, 0xD5_01)
            .expect("valid cluster bootstrap");
        let second = summarize_clustered_pairs(&rows, 1_000, 0xD5_01)
            .expect("same-seed bootstrap replay");

        assert_eq!(first, second);
        assert_eq!(first.episode_count, 3);
        assert_eq!(first.cluster_count, 2);
        assert_eq!(
            first.memory_induced_harm_rate,
            CountRatioV1 {
                numerator: 1,
                denominator: 3,
            }
        );
        assert_eq!(
            first.memory_improvement_rate,
            CountRatioV1 {
                numerator: 2,
                denominator: 3,
            }
        );
    }

    #[test]
    fn cluster_bootstrap_rejects_pseudoreplication_from_one_world() {
        let rows = vec![
            clustered_row("episode-a1", "world-a", 1),
            clustered_row("episode-a2", "world-a", 1),
        ];
        assert_eq!(
            summarize_clustered_pairs(&rows, 1_000, 1),
            Err(EvaluationErrorV1::InsufficientIndependentClusters)
        );
    }

    #[test]
    fn cluster_bootstrap_rejects_duplicate_episode_identity() {
        let rows = vec![
            clustered_row("episode-a1", "world-a", 1),
            clustered_row("episode-a1", "world-b", -1),
        ];
        assert_eq!(
            summarize_clustered_pairs(&rows, 1_000, 1),
            Err(EvaluationErrorV1::DuplicateEpisodeId)
        );
    }

    #[test]
    fn cluster_bootstrap_rejects_too_few_replicates() {
        let rows = vec![
            clustered_row("episode-a1", "world-a", 1),
            clustered_row("episode-b1", "world-b", -1),
        ];
        assert_eq!(
            summarize_clustered_pairs(&rows, 99, 1),
            Err(EvaluationErrorV1::InvalidBootstrapReplicates)
        );
    }

    #[test]
    fn cluster_bootstrap_rejects_a_forged_paired_gain() {
        let mut row = clustered_row("episode-a1", "world-a", 1);
        row.score.memory_gain_micros += 1;
        let rows = vec![
            row,
            clustered_row("episode-b1", "world-b", -1),
        ];
        assert_eq!(
            summarize_clustered_pairs(&rows, 1_000, 1),
            Err(EvaluationErrorV1::InvalidPairedScore)
        );
    }

    #[test]
    fn candidate_policy_digest_must_be_a_lowercase_sha256_hex_string() {
        let mut candidate = run("mace", "hold", &["signed-current"]);
        candidate.downstream_policy_digest = "not-a-digest".to_owned();
        assert_eq!(
            score_paired(&case(), &candidate, &run("no-memory", "abstain", &[])),
            Err(EvaluationErrorV1::InvalidDownstreamPolicyDigest)
        );
    }

    #[test]
    fn oracle_action_must_be_a_minimum_loss_action() {
        let mut invalid = case();
        invalid.oracle_action = "approve".to_owned();
        assert_eq!(
            validate_case(&invalid),
            Err(EvaluationErrorV1::OracleActionIsNotOptimal)
        );
    }

    #[test]
    fn duplicate_candidate_evidence_ids_are_rejected() {
        let result = score_paired(
            &case(),
            &run("mace", "hold", &["signed-current", "signed-current"]),
            &run("no-memory", "abstain", &[]),
        );
        assert_eq!(result, Err(EvaluationErrorV1::DuplicateReturnedEvidenceId));
    }

    #[test]
    fn empty_retrieval_has_no_precision_value_not_a_perfect_precision_claim() {
        let score = score_run(&case(), &run("empty", "abstain", &[]))
            .expect("empty retrieval is a valid candidate run");
        assert_eq!(score.admissible_evidence_precision, None);
        assert_eq!(
            score.required_evidence_recall,
            Some(CountRatioV1 {
                numerator: 0,
                denominator: 1,
            })
        );
    }

    #[test]
    fn phase_cost_receipt_rejects_overstated_phase_time() {
        let receipt = CostReceiptV1 {
            phases: PhaseTimingsV1 {
                add_ingest_ns: 30,
                index_update_ns: 30,
                retrieval_ns: 30,
                evidence_verification_ns: 30,
                policy_application_ns: 30,
                maintenance_ns: 0,
                unattributed_ns: 0,
                wall_ns: 100,
            },
            measured_cost_microunits: Some(10),
            estimated_cost_microunits: None,
            retained_bytes: ResourceQuantityV1 {
                value: Some(100),
                status: ResourceMeasurementStatusV1::Measured,
            },
            index_bytes: ResourceQuantityV1 {
                value: Some(40),
                status: ResourceMeasurementStatusV1::Measured,
            },
            network_bytes: ResourceQuantityV1 {
                value: Some(20),
                status: ResourceMeasurementStatusV1::Measured,
            },
        };
        assert_eq!(
            receipt.validate(),
            Err(CostReceiptErrorV1::PhasesDoNotReconcile)
        );
    }

    #[test]
    fn byte_cost_receipts_distinguish_measured_zero_from_unknown_and_not_applicable() {
        let receipt = CostReceiptV1 {
            phases: PhaseTimingsV1 {
                add_ingest_ns: 0,
                index_update_ns: 0,
                retrieval_ns: 0,
                evidence_verification_ns: 0,
                policy_application_ns: 0,
                maintenance_ns: 0,
                unattributed_ns: 0,
                wall_ns: 0,
            },
            measured_cost_microunits: Some(0),
            estimated_cost_microunits: None,
            retained_bytes: ResourceQuantityV1 {
                value: Some(0),
                status: ResourceMeasurementStatusV1::Measured,
            },
            index_bytes: ResourceQuantityV1 {
                value: Some(64),
                status: ResourceMeasurementStatusV1::Estimated,
            },
            network_bytes: ResourceQuantityV1 {
                value: None,
                status: ResourceMeasurementStatusV1::Unknown,
            },
        };
        assert_eq!(receipt.validate(), Ok(()));
        let not_applicable = ResourceQuantityV1 {
            value: None,
            status: ResourceMeasurementStatusV1::NotApplicable,
        };
        assert!(not_applicable.is_consistent());
        let forged_zero = ResourceQuantityV1 {
            value: Some(0),
            status: ResourceMeasurementStatusV1::Unknown,
        };
        assert!(!forged_zero.is_consistent());
    }

    #[test]
    fn phase_cost_receipt_accepts_reconciled_wall_time() {
        let receipt = CostReceiptV1 {
            phases: PhaseTimingsV1 {
                add_ingest_ns: 20,
                index_update_ns: 10,
                retrieval_ns: 15,
                evidence_verification_ns: 20,
                policy_application_ns: 10,
                maintenance_ns: 5,
                unattributed_ns: 20,
                wall_ns: 100,
            },
            measured_cost_microunits: None,
            estimated_cost_microunits: Some(12),
            retained_bytes: ResourceQuantityV1 {
                value: Some(100),
                status: ResourceMeasurementStatusV1::Measured,
            },
            index_bytes: ResourceQuantityV1 {
                value: Some(40),
                status: ResourceMeasurementStatusV1::Measured,
            },
            network_bytes: ResourceQuantityV1 {
                value: Some(20),
                status: ResourceMeasurementStatusV1::Measured,
            },
        };
        assert_eq!(receipt.validate(), Ok(()));
    }
}
