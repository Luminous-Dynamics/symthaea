//! Synthetic-only, evidence-bounded candidate ranking for organ-design research.
//!
//! Q1 compares declared objective vectors; it does not model biological organs,
//! infer unmeasured physiology, or authorize experiments or clinical decisions.
//! Hard constraints and evidence admission are evaluated before Pareto ranking.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write as _;

pub const CONTRACT_VERSION: &str = "organ-design-q1-v0.1.0";

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum EvidenceTier {
    AnalyticReference,
    SyntheticSimulation,
    PhysicalBench,
    TissueSpecific,
    Translational,
}

impl EvidenceTier {
    pub fn code(self) -> &'static str {
        match self {
            Self::AnalyticReference => "analytic_reference",
            Self::SyntheticSimulation => "synthetic_simulation",
            Self::PhysicalBench => "physical_bench",
            Self::TissueSpecific => "tissue_specific",
            Self::Translational => "translational",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvaluationMode {
    CandidateSelection,
    HeldOutEvaluation,
}

impl EvaluationMode {
    fn code(self) -> &'static str {
        match self {
            Self::CandidateSelection => "candidate_selection",
            Self::HeldOutEvaluation => "held_out_evaluation",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DataSplit {
    Training,
    Validation,
    HeldOut,
}

impl DataSplit {
    fn code(self) -> &'static str {
        match self {
            Self::Training => "training",
            Self::Validation => "validation",
            Self::HeldOut => "held_out",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObjectiveDirection {
    Minimize,
    Maximize,
}

impl ObjectiveDirection {
    fn code(self) -> &'static str {
        match self {
            Self::Minimize => "minimize",
            Self::Maximize => "maximize",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvidenceStatus {
    Current,
    Stale,
    Contradictory,
    Unverified,
}

impl EvidenceStatus {
    fn code(self) -> &'static str {
        match self {
            Self::Current => "current",
            Self::Stale => "stale",
            Self::Contradictory => "contradictory",
            Self::Unverified => "unverified",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObjectiveDefinition {
    pub id: String,
    pub unit: String,
    pub direction: ObjectiveDirection,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvidenceRequirement {
    pub source_id: String,
    pub sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvaluationProfile {
    pub id: String,
    pub purpose: String,
    pub mode: EvaluationMode,
    pub data_split: DataSplit,
    pub objectives: Vec<ObjectiveDefinition>,
    pub required_hard_constraints: Vec<String>,
    pub required_assumptions: Vec<String>,
    pub required_evidence: Vec<EvidenceRequirement>,
    pub baseline_candidate_id: String,
    pub ablation_candidate_ids: Vec<String>,
    /// Q1 is explicitly capped at synthetic evidence. A future physical or tissue
    /// profile requires a separately reviewed contract/version, not a config flag.
    pub result_evidence_tier: EvidenceTier,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Metric {
    pub objective_id: String,
    pub value: f64,
    pub unit: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HardConstraintResult {
    pub id: String,
    pub satisfied: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AssumptionResolution {
    pub id: String,
    pub resolved: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvidenceObservation {
    pub source_id: String,
    pub sha256: String,
    pub status: EvidenceStatus,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Candidate {
    pub id: String,
    pub profile_id: String,
    pub data_split: DataSplit,
    pub metrics: Vec<Metric>,
    pub hard_constraints: Vec<HardConstraintResult>,
    pub assumptions: Vec<AssumptionResolution>,
    pub evidence: Vec<EvidenceObservation>,
    /// Diagnostic annotation only. It is never an objective, tie-breaker, or
    /// evidence-tier promotion mechanism.
    pub internal_confidence: Option<f64>,
    /// An asserted ceiling above the profile's tier is rejected, not clamped.
    pub requested_claim_ceiling: EvidenceTier,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum CandidateRejectionReason {
    ProfileMismatch,
    SplitMismatch,
    ClaimCeilingExceeded,
    InvalidConfidence,
    MissingMetric(String),
    DuplicateMetric(String),
    UnknownMetric(String),
    WrongMetricUnit(String),
    NonFiniteMetric(String),
    MissingHardConstraint(String),
    DuplicateHardConstraint(String),
    UnknownHardConstraint(String),
    HardConstraintFailed(String),
    MissingAssumption(String),
    DuplicateAssumption(String),
    UnknownAssumption(String),
    AssumptionUnresolved(String),
    MissingEvidence(String),
    DuplicateEvidence(String),
    UnknownEvidence(String),
    EvidenceDigestMismatch(String),
    EvidenceStale(String),
    EvidenceContradictory(String),
    EvidenceUnverified(String),
}

impl CandidateRejectionReason {
    pub fn code(&self) -> &'static str {
        match self {
            Self::ProfileMismatch => "profile_mismatch",
            Self::SplitMismatch => "split_mismatch",
            Self::ClaimCeilingExceeded => "claim_ceiling_exceeded",
            Self::InvalidConfidence => "invalid_confidence",
            Self::MissingMetric(_) => "missing_metric",
            Self::DuplicateMetric(_) => "duplicate_metric",
            Self::UnknownMetric(_) => "unknown_metric",
            Self::WrongMetricUnit(_) => "wrong_metric_unit",
            Self::NonFiniteMetric(_) => "non_finite_metric",
            Self::MissingHardConstraint(_) => "missing_hard_constraint",
            Self::DuplicateHardConstraint(_) => "duplicate_hard_constraint",
            Self::UnknownHardConstraint(_) => "unknown_hard_constraint",
            Self::HardConstraintFailed(_) => "hard_constraint_failed",
            Self::MissingAssumption(_) => "missing_assumption",
            Self::DuplicateAssumption(_) => "duplicate_assumption",
            Self::UnknownAssumption(_) => "unknown_assumption",
            Self::AssumptionUnresolved(_) => "assumption_unresolved",
            Self::MissingEvidence(_) => "missing_evidence",
            Self::DuplicateEvidence(_) => "duplicate_evidence",
            Self::UnknownEvidence(_) => "unknown_evidence",
            Self::EvidenceDigestMismatch(_) => "evidence_digest_mismatch",
            Self::EvidenceStale(_) => "evidence_stale",
            Self::EvidenceContradictory(_) => "evidence_contradictory",
            Self::EvidenceUnverified(_) => "evidence_unverified",
        }
    }

    fn subject(&self) -> &str {
        match self {
            Self::MissingMetric(value)
            | Self::DuplicateMetric(value)
            | Self::UnknownMetric(value)
            | Self::WrongMetricUnit(value)
            | Self::NonFiniteMetric(value)
            | Self::MissingHardConstraint(value)
            | Self::DuplicateHardConstraint(value)
            | Self::UnknownHardConstraint(value)
            | Self::HardConstraintFailed(value)
            | Self::MissingAssumption(value)
            | Self::DuplicateAssumption(value)
            | Self::UnknownAssumption(value)
            | Self::AssumptionUnresolved(value)
            | Self::MissingEvidence(value)
            | Self::DuplicateEvidence(value)
            | Self::UnknownEvidence(value)
            | Self::EvidenceDigestMismatch(value)
            | Self::EvidenceStale(value)
            | Self::EvidenceContradictory(value)
            | Self::EvidenceUnverified(value) => value,
            Self::ProfileMismatch => "profile_id",
            Self::SplitMismatch => "data_split",
            Self::ClaimCeilingExceeded => "requested_claim_ceiling",
            Self::InvalidConfidence => "internal_confidence",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RankingError {
    EmptyProfileId,
    EmptyPurpose,
    NoObjectives,
    DuplicateObjective(String),
    EmptyObjectiveUnit(String),
    NoHardConstraints,
    DuplicateHardConstraint(String),
    NoAssumptions,
    DuplicateAssumption(String),
    NoEvidenceRequirements,
    DuplicateEvidenceRequirement(String),
    InvalidEvidenceDigest(String),
    EmptyBaselineId,
    DuplicateAblation(String),
    BaselineAlsoListedAsAblation,
    InvalidSplitForMode,
    Q1MustRemainSyntheticOnly,
    EmptyCandidateId,
    DuplicateCandidateId(String),
    NoCandidates,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReceiptStatus {
    Ready,
    Blocked,
}

impl ReceiptStatus {
    pub fn code(self) -> &'static str {
        match self {
            Self::Ready => "READY",
            Self::Blocked => "BLOCKED",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum BlockingReason {
    MissingBaseline(String),
    BaselineRejected(String),
    MissingAblation(String),
    AblationRejected(String),
    NoAdmittedCandidates,
}

impl BlockingReason {
    pub fn code(&self) -> &'static str {
        match self {
            Self::MissingBaseline(_) => "missing_baseline",
            Self::BaselineRejected(_) => "baseline_rejected",
            Self::MissingAblation(_) => "missing_ablation",
            Self::AblationRejected(_) => "ablation_rejected",
            Self::NoAdmittedCandidates => "no_admitted_candidates",
        }
    }

    fn subject(&self) -> &str {
        match self {
            Self::MissingBaseline(id)
            | Self::BaselineRejected(id)
            | Self::MissingAblation(id)
            | Self::AblationRejected(id) => id,
            Self::NoAdmittedCandidates => "candidate_set",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CandidateDecision {
    ParetoRanked {
        candidate_id: String,
        pareto_front: usize,
        claim_ceiling: EvidenceTier,
    },
    EvaluatedOnly {
        candidate_id: String,
        claim_ceiling: EvidenceTier,
    },
    Rejected {
        candidate_id: String,
        reasons: Vec<CandidateRejectionReason>,
    },
}

impl CandidateDecision {
    pub fn candidate_id(&self) -> &str {
        match self {
            Self::ParetoRanked { candidate_id, .. }
            | Self::EvaluatedOnly { candidate_id, .. }
            | Self::Rejected { candidate_id, .. } => candidate_id,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RankingReceipt {
    pub contract_version: &'static str,
    pub profile_id: String,
    pub status: ReceiptStatus,
    pub mode: EvaluationMode,
    pub selection_permitted: bool,
    pub claim_ceiling: EvidenceTier,
    pub decisions: Vec<CandidateDecision>,
    pub blocking_reasons: Vec<BlockingReason>,
    /// Collision-resistant, length-prefixed canonical material. Hash this exact
    /// UTF-8 string with an external SHA-256 tool for a digest; no hash crate is
    /// pulled into the runtime reference contract.
    pub identity_material: String,
}

pub fn evaluate_candidates(
    profile: &EvaluationProfile,
    candidates: &[Candidate],
) -> Result<RankingReceipt, RankingError> {
    validate_profile(profile)?;
    if candidates.is_empty() {
        return Err(RankingError::NoCandidates);
    }

    let mut candidate_ids = HashSet::new();
    for candidate in candidates {
        if candidate.id.trim().is_empty() {
            return Err(RankingError::EmptyCandidateId);
        }
        if !candidate_ids.insert(candidate.id.as_str()) {
            return Err(RankingError::DuplicateCandidateId(candidate.id.clone()));
        }
    }

    let mut sorted_candidates: Vec<&Candidate> = candidates.iter().collect();
    sorted_candidates.sort_by(|left, right| left.id.cmp(&right.id));

    let mut reasons_by_id: BTreeMap<String, Vec<CandidateRejectionReason>> = BTreeMap::new();
    let mut admitted: Vec<&Candidate> = Vec::new();
    for candidate in &sorted_candidates {
        let mut reasons = validate_candidate(profile, candidate);
        reasons.sort();
        reasons.dedup();
        if reasons.is_empty() {
            admitted.push(candidate);
        } else {
            reasons_by_id.insert(candidate.id.clone(), reasons);
        }
    }

    let mut blocking_reasons = Vec::new();
    let admitted_ids: HashSet<&str> = admitted.iter().map(|candidate| candidate.id.as_str()).collect();
    if !candidate_ids.contains(profile.baseline_candidate_id.as_str()) {
        blocking_reasons.push(BlockingReason::MissingBaseline(
            profile.baseline_candidate_id.clone(),
        ));
    } else if !admitted_ids.contains(profile.baseline_candidate_id.as_str()) {
        blocking_reasons.push(BlockingReason::BaselineRejected(
            profile.baseline_candidate_id.clone(),
        ));
    }
    for ablation_id in &profile.ablation_candidate_ids {
        if !candidate_ids.contains(ablation_id.as_str()) {
            blocking_reasons.push(BlockingReason::MissingAblation(ablation_id.clone()));
        } else if !admitted_ids.contains(ablation_id.as_str()) {
            blocking_reasons.push(BlockingReason::AblationRejected(ablation_id.clone()));
        }
    }
    if admitted.is_empty() {
        blocking_reasons.push(BlockingReason::NoAdmittedCandidates);
    }
    blocking_reasons.sort();
    blocking_reasons.dedup();

    let mut ranked_fronts: HashMap<String, usize> = HashMap::new();
    if profile.mode == EvaluationMode::CandidateSelection {
        ranked_fronts = pareto_fronts(profile, &admitted);
    }

    let mut decisions = Vec::with_capacity(candidates.len());
    for candidate in &sorted_candidates {
        if let Some(reasons) = reasons_by_id.get(candidate.id.as_str()) {
            decisions.push(CandidateDecision::Rejected {
                candidate_id: candidate.id.clone(),
                reasons: reasons.clone(),
            });
        } else if profile.mode == EvaluationMode::HeldOutEvaluation {
            decisions.push(CandidateDecision::EvaluatedOnly {
                candidate_id: candidate.id.clone(),
                claim_ceiling: profile.result_evidence_tier,
            });
        } else {
            // Every admitted selection candidate receives a front from the
            // finite-set Pareto peeling routine.
            let pareto_front = *ranked_fronts
                .get(candidate.id.as_str())
                .expect("admitted candidate must be assigned a Pareto front");
            decisions.push(CandidateDecision::ParetoRanked {
                candidate_id: candidate.id.clone(),
                pareto_front,
                claim_ceiling: profile.result_evidence_tier,
            });
        }
    }

    let status = if blocking_reasons.is_empty() {
        ReceiptStatus::Ready
    } else {
        ReceiptStatus::Blocked
    };
    let mut receipt = RankingReceipt {
        contract_version: CONTRACT_VERSION,
        profile_id: profile.id.clone(),
        status,
        mode: profile.mode,
        selection_permitted: profile.mode == EvaluationMode::CandidateSelection,
        claim_ceiling: profile.result_evidence_tier,
        decisions,
        blocking_reasons,
        identity_material: String::new(),
    };
    receipt.identity_material = canonical_identity_material(profile, &sorted_candidates, &receipt);
    Ok(receipt)
}

fn validate_profile(profile: &EvaluationProfile) -> Result<(), RankingError> {
    if profile.id.trim().is_empty() {
        return Err(RankingError::EmptyProfileId);
    }
    if profile.purpose.trim().is_empty() {
        return Err(RankingError::EmptyPurpose);
    }
    if profile.objectives.is_empty() {
        return Err(RankingError::NoObjectives);
    }
    let mut objectives = HashSet::new();
    for objective in &profile.objectives {
        if objective.id.trim().is_empty() || !objectives.insert(objective.id.as_str()) {
            return Err(RankingError::DuplicateObjective(objective.id.clone()));
        }
        if objective.unit.trim().is_empty() {
            return Err(RankingError::EmptyObjectiveUnit(objective.id.clone()));
        }
    }
    if profile.required_hard_constraints.is_empty() {
        return Err(RankingError::NoHardConstraints);
    }
    unique_nonempty_ids(
        &profile.required_hard_constraints,
        |id| RankingError::DuplicateHardConstraint(id.to_owned()),
    )?;
    if profile.required_assumptions.is_empty() {
        return Err(RankingError::NoAssumptions);
    }
    unique_nonempty_ids(&profile.required_assumptions, |id| {
        RankingError::DuplicateAssumption(id.to_owned())
    })?;
    if profile.required_evidence.is_empty() {
        return Err(RankingError::NoEvidenceRequirements);
    }
    let mut evidence_ids = HashSet::new();
    for requirement in &profile.required_evidence {
        if requirement.source_id.trim().is_empty()
            || !evidence_ids.insert(requirement.source_id.as_str())
        {
            return Err(RankingError::DuplicateEvidenceRequirement(
                requirement.source_id.clone(),
            ));
        }
        if !is_sha256(&requirement.sha256) {
            return Err(RankingError::InvalidEvidenceDigest(
                requirement.source_id.clone(),
            ));
        }
    }
    if profile.baseline_candidate_id.trim().is_empty() {
        return Err(RankingError::EmptyBaselineId);
    }
    let mut ablations = HashSet::new();
    for id in &profile.ablation_candidate_ids {
        if id.trim().is_empty() || !ablations.insert(id.as_str()) {
            return Err(RankingError::DuplicateAblation(id.clone()));
        }
        if id == &profile.baseline_candidate_id {
            return Err(RankingError::BaselineAlsoListedAsAblation);
        }
    }
    match profile.mode {
        EvaluationMode::CandidateSelection if profile.data_split != DataSplit::Training => {
            return Err(RankingError::InvalidSplitForMode);
        }
        EvaluationMode::HeldOutEvaluation if profile.data_split != DataSplit::HeldOut => {
            return Err(RankingError::InvalidSplitForMode);
        }
        _ => {}
    }
    if profile.result_evidence_tier != EvidenceTier::SyntheticSimulation {
        return Err(RankingError::Q1MustRemainSyntheticOnly);
    }
    Ok(())
}

fn unique_nonempty_ids(
    ids: &[String],
    duplicate_error: impl Fn(&str) -> RankingError,
) -> Result<(), RankingError> {
    let mut seen = HashSet::new();
    for id in ids {
        if id.trim().is_empty() || !seen.insert(id.as_str()) {
            return Err(duplicate_error(id));
        }
    }
    Ok(())
}

fn validate_candidate(
    profile: &EvaluationProfile,
    candidate: &Candidate,
) -> Vec<CandidateRejectionReason> {
    let mut reasons = Vec::new();
    if candidate.profile_id != profile.id {
        reasons.push(CandidateRejectionReason::ProfileMismatch);
    }
    if candidate.data_split != profile.data_split {
        reasons.push(CandidateRejectionReason::SplitMismatch);
    }
    if candidate.requested_claim_ceiling > profile.result_evidence_tier {
        reasons.push(CandidateRejectionReason::ClaimCeilingExceeded);
    }
    if candidate
        .internal_confidence
        .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
    {
        reasons.push(CandidateRejectionReason::InvalidConfidence);
    }

    let objective_by_id: HashMap<&str, &ObjectiveDefinition> = profile
        .objectives
        .iter()
        .map(|objective| (objective.id.as_str(), objective))
        .collect();
    let mut seen_metrics: HashSet<&str> = HashSet::new();
    for metric in &candidate.metrics {
        let id = metric.objective_id.as_str();
        if !seen_metrics.insert(id) {
            reasons.push(CandidateRejectionReason::DuplicateMetric(id.to_owned()));
            continue;
        }
        let Some(objective) = objective_by_id.get(id) else {
            reasons.push(CandidateRejectionReason::UnknownMetric(id.to_owned()));
            continue;
        };
        if metric.unit.trim().is_empty() || metric.unit != objective.unit {
            reasons.push(CandidateRejectionReason::WrongMetricUnit(id.to_owned()));
        }
        if !metric.value.is_finite() {
            reasons.push(CandidateRejectionReason::NonFiniteMetric(id.to_owned()));
        }
    }
    for objective in &profile.objectives {
        if !seen_metrics.contains(objective.id.as_str()) {
            reasons.push(CandidateRejectionReason::MissingMetric(objective.id.clone()));
        }
    }

    let required_constraints: HashSet<&str> = profile
        .required_hard_constraints
        .iter()
        .map(String::as_str)
        .collect();
    let mut seen_constraints = HashSet::new();
    for constraint in &candidate.hard_constraints {
        if !seen_constraints.insert(constraint.id.as_str()) {
            reasons.push(CandidateRejectionReason::DuplicateHardConstraint(
                constraint.id.clone(),
            ));
        } else if !required_constraints.contains(constraint.id.as_str()) {
            reasons.push(CandidateRejectionReason::UnknownHardConstraint(
                constraint.id.clone(),
            ));
        } else if !constraint.satisfied {
            reasons.push(CandidateRejectionReason::HardConstraintFailed(
                constraint.id.clone(),
            ));
        }
    }
    for constraint_id in &profile.required_hard_constraints {
        if !seen_constraints.contains(constraint_id.as_str()) {
            reasons.push(CandidateRejectionReason::MissingHardConstraint(
                constraint_id.clone(),
            ));
        }
    }

    let required_assumptions: HashSet<&str> = profile
        .required_assumptions
        .iter()
        .map(String::as_str)
        .collect();
    let mut seen_assumptions = HashSet::new();
    for assumption in &candidate.assumptions {
        if !seen_assumptions.insert(assumption.id.as_str()) {
            reasons.push(CandidateRejectionReason::DuplicateAssumption(
                assumption.id.clone(),
            ));
        } else if !required_assumptions.contains(assumption.id.as_str()) {
            reasons.push(CandidateRejectionReason::UnknownAssumption(
                assumption.id.clone(),
            ));
        } else if !assumption.resolved {
            reasons.push(CandidateRejectionReason::AssumptionUnresolved(
                assumption.id.clone(),
            ));
        }
    }
    for assumption_id in &profile.required_assumptions {
        if !seen_assumptions.contains(assumption_id.as_str()) {
            reasons.push(CandidateRejectionReason::MissingAssumption(
                assumption_id.clone(),
            ));
        }
    }

    let required_evidence: HashMap<&str, &EvidenceRequirement> = profile
        .required_evidence
        .iter()
        .map(|requirement| (requirement.source_id.as_str(), requirement))
        .collect();
    let mut seen_evidence = HashSet::new();
    for evidence in &candidate.evidence {
        if !seen_evidence.insert(evidence.source_id.as_str()) {
            reasons.push(CandidateRejectionReason::DuplicateEvidence(
                evidence.source_id.clone(),
            ));
            continue;
        }
        let Some(requirement) = required_evidence.get(evidence.source_id.as_str()) else {
            reasons.push(CandidateRejectionReason::UnknownEvidence(
                evidence.source_id.clone(),
            ));
            continue;
        };
        if evidence.sha256 != requirement.sha256 {
            reasons.push(CandidateRejectionReason::EvidenceDigestMismatch(
                evidence.source_id.clone(),
            ));
        }
        match evidence.status {
            EvidenceStatus::Current => {}
            EvidenceStatus::Stale => reasons.push(CandidateRejectionReason::EvidenceStale(
                evidence.source_id.clone(),
            )),
            EvidenceStatus::Contradictory => {
                reasons.push(CandidateRejectionReason::EvidenceContradictory(
                    evidence.source_id.clone(),
                ))
            }
            EvidenceStatus::Unverified => reasons.push(
                CandidateRejectionReason::EvidenceUnverified(evidence.source_id.clone()),
            ),
        }
    }
    for requirement in &profile.required_evidence {
        if !seen_evidence.contains(requirement.source_id.as_str()) {
            reasons.push(CandidateRejectionReason::MissingEvidence(
                requirement.source_id.clone(),
            ));
        }
    }

    reasons.sort();
    reasons.dedup();
    reasons
}

fn pareto_fronts(
    profile: &EvaluationProfile,
    candidates: &[&Candidate],
) -> HashMap<String, usize> {
    let mut remaining: Vec<&Candidate> = candidates.to_vec();
    remaining.sort_by(|left, right| left.id.cmp(&right.id));
    let mut output = HashMap::new();
    let mut front_index = 0usize;

    while !remaining.is_empty() {
        let current_front: Vec<&Candidate> = remaining
            .iter()
            .copied()
            .filter(|candidate| {
                !remaining.iter().any(|other| {
                    other.id != candidate.id && dominates(profile, other, candidate)
                })
            })
            .collect();

        // For a finite strict Pareto order, a non-empty set always has at least
        // one non-dominated element. An empty front indicates a contract bug.
        assert!(!current_front.is_empty(), "Pareto peeling made no progress");
        let current_ids: HashSet<&str> = current_front
            .iter()
            .map(|candidate| candidate.id.as_str())
            .collect();
        for candidate in &current_front {
            output.insert(candidate.id.clone(), front_index);
        }
        remaining.retain(|candidate| !current_ids.contains(candidate.id.as_str()));
        front_index += 1;
    }
    output
}

fn dominates(
    profile: &EvaluationProfile,
    left: &Candidate,
    right: &Candidate,
) -> bool {
    let mut strictly_better = false;
    for objective in &profile.objectives {
        let left_value = left
            .metrics
            .iter()
            .find(|metric| metric.objective_id == objective.id)
            .expect("admitted candidate has every declared metric")
            .value;
        let right_value = right
            .metrics
            .iter()
            .find(|metric| metric.objective_id == objective.id)
            .expect("admitted candidate has every declared metric")
            .value;

        match objective.direction {
            ObjectiveDirection::Minimize => {
                if left_value > right_value {
                    return false;
                }
                if left_value < right_value {
                    strictly_better = true;
                }
            }
            ObjectiveDirection::Maximize => {
                if left_value < right_value {
                    return false;
                }
                if left_value > right_value {
                    strictly_better = true;
                }
            }
        }
    }
    strictly_better
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn canonical_identity_material(
    profile: &EvaluationProfile,
    candidates: &[&Candidate],
    receipt: &RankingReceipt,
) -> String {
    let mut out = String::new();
    put(&mut out, "contract_version", CONTRACT_VERSION);
    put(&mut out, "profile_id", &profile.id);
    put(&mut out, "purpose", &profile.purpose);
    put(&mut out, "mode", profile.mode.code());
    put(&mut out, "data_split", profile.data_split.code());
    put(&mut out, "profile_result_tier", profile.result_evidence_tier.code());
    put(&mut out, "baseline_candidate_id", &profile.baseline_candidate_id);
    put_list(&mut out, "ablation_candidate_ids", &sorted_strings(&profile.ablation_candidate_ids));

    let mut objectives: Vec<&ObjectiveDefinition> = profile.objectives.iter().collect();
    objectives.sort_by(|left, right| left.id.cmp(&right.id));
    for objective in objectives {
        let fields = vec![
            objective.id.clone(),
            objective.unit.clone(),
            objective.direction.code().to_owned(),
        ];
        put_record(&mut out, "objective", &fields);
    }
    put_list(
        &mut out,
        "required_hard_constraints",
        &sorted_strings(&profile.required_hard_constraints),
    );
    put_list(
        &mut out,
        "required_assumptions",
        &sorted_strings(&profile.required_assumptions),
    );
    let mut required_evidence: Vec<&EvidenceRequirement> =
        profile.required_evidence.iter().collect();
    required_evidence.sort_by(|left, right| left.source_id.cmp(&right.source_id));
    for item in required_evidence {
        put_record(
            &mut out,
            "required_evidence",
            &[item.source_id.clone(), item.sha256.clone()],
        );
    }

    for candidate in candidates {
        put(&mut out, "candidate_id", &candidate.id);
        put(&mut out, "candidate_profile_id", &candidate.profile_id);
        put(&mut out, "candidate_split", candidate.data_split.code());
        put(
            &mut out,
            "candidate_internal_confidence_bits",
            &candidate
                .internal_confidence
                .map(|value| format!("{:016x}", value.to_bits()))
                .unwrap_or_else(|| "none".to_owned()),
        );
        put(
            &mut out,
            "candidate_requested_claim_ceiling",
            candidate.requested_claim_ceiling.code(),
        );

        let mut metrics: Vec<&Metric> = candidate.metrics.iter().collect();
        metrics.sort_by(|left, right| left.objective_id.cmp(&right.objective_id));
        for metric in metrics {
            put_record(
                &mut out,
                "candidate_metric",
                &[
                    metric.objective_id.clone(),
                    metric.unit.clone(),
                    format!("{:016x}", metric.value.to_bits()),
                ],
            );
        }

        let mut constraints: Vec<&HardConstraintResult> =
            candidate.hard_constraints.iter().collect();
        constraints.sort_by(|left, right| left.id.cmp(&right.id));
        for constraint in constraints {
            put_record(
                &mut out,
                "candidate_constraint",
                &[
                    constraint.id.clone(),
                    if constraint.satisfied { "true" } else { "false" }.to_owned(),
                ],
            );
        }
        let mut assumptions: Vec<&AssumptionResolution> = candidate.assumptions.iter().collect();
        assumptions.sort_by(|left, right| left.id.cmp(&right.id));
        for assumption in assumptions {
            put_record(
                &mut out,
                "candidate_assumption",
                &[
                    assumption.id.clone(),
                    if assumption.resolved { "resolved" } else { "unresolved" }.to_owned(),
                ],
            );
        }
        let mut evidence: Vec<&EvidenceObservation> = candidate.evidence.iter().collect();
        evidence.sort_by(|left, right| left.source_id.cmp(&right.source_id));
        for item in evidence {
            put_record(
                &mut out,
                "candidate_evidence",
                &[
                    item.source_id.clone(),
                    item.sha256.clone(),
                    item.status.code().to_owned(),
                ],
            );
        }
    }

    put(&mut out, "receipt_status", receipt.status.code());
    put(
        &mut out,
        "selection_permitted",
        if receipt.selection_permitted { "true" } else { "false" },
    );
    put(&mut out, "receipt_claim_ceiling", receipt.claim_ceiling.code());
    for decision in &receipt.decisions {
        match decision {
            CandidateDecision::ParetoRanked {
                candidate_id,
                pareto_front,
                claim_ceiling,
            } => put_record(
                &mut out,
                "decision_pareto",
                &[
                    candidate_id.clone(),
                    pareto_front.to_string(),
                    claim_ceiling.code().to_owned(),
                ],
            ),
            CandidateDecision::EvaluatedOnly {
                candidate_id,
                claim_ceiling,
            } => put_record(
                &mut out,
                "decision_evaluated_only",
                &[candidate_id.clone(), claim_ceiling.code().to_owned()],
            ),
            CandidateDecision::Rejected {
                candidate_id,
                reasons,
            } => {
                put(&mut out, "decision_rejected_candidate", candidate_id);
                for reason in reasons {
                    put_record(
                        &mut out,
                        "decision_rejection_reason",
                        &[reason.code().to_owned(), reason.subject().to_owned()],
                    );
                }
            }
        }
    }
    for reason in &receipt.blocking_reasons {
        put_record(
            &mut out,
            "blocking_reason",
            &[reason.code().to_owned(), reason.subject().to_owned()],
        );
    }
    out
}

fn sorted_strings(values: &[String]) -> Vec<String> {
    let mut values = values.to_vec();
    values.sort();
    values
}

fn put(out: &mut String, name: &str, value: &str) {
    // Length-prefix both fields; embedded delimiters/newlines cannot create an
    // ambiguous canonical identity representation.
    let _ = write!(out, "{}:{}{}:{};", name.len(), name, value.len(), value);
}

fn put_list(out: &mut String, name: &str, values: &[String]) {
    put(out, name, &values.len().to_string());
    for value in values {
        put(out, "item", value);
    }
}

fn put_record(out: &mut String, name: &str, fields: &[String]) {
    put(out, "record_type", name);
    put(out, "record_field_count", &fields.len().to_string());
    for field in fields {
        put(out, "field", field);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const HOSTILE_CASES: &str = include_str!("../hostile-cases.v0.json");
    const HOSTILE_CASE_IDS: &[&str] = &[
        "identical-inputs-replay-deterministic",
        "objective-profile-changes-identity",
        "hard-constraint-cannot-be-traded-away",
        "missing-or-wrong-units-fail-closed",
        "unresolved-assumption-fails-closed",
        "stale-or-contradictory-evidence-fails-closed",
        "confidence-cannot-promote-synthetic-evidence",
        "synthetic-cannot-claim-physical-validation",
        "heldout-cannot-be-used-for-selection",
        "baseline-and-ablations-are-mandatory",
        "pareto-incomparables-are-preserved",
        "duplicate-identities-are-rejected",
    ];

    fn profile() -> EvaluationProfile {
        EvaluationProfile {
            id: "vascular-q1-synthetic-v0".into(),
            purpose: "rank candidate synthetic vascular graphs under a fixed proxy profile".into(),
            mode: EvaluationMode::CandidateSelection,
            data_split: DataSplit::Training,
            objectives: vec![
                ObjectiveDefinition {
                    id: "coverage_proxy".into(),
                    unit: "fraction".into(),
                    direction: ObjectiveDirection::Maximize,
                },
                ObjectiveDefinition {
                    id: "material_cost_proxy".into(),
                    unit: "arb_unit".into(),
                    direction: ObjectiveDirection::Minimize,
                },
                ObjectiveDefinition {
                    id: "pressure_drop_proxy".into(),
                    unit: "Pa".into(),
                    direction: ObjectiveDirection::Minimize,
                },
            ],
            required_hard_constraints: vec![
                "mass_balance".into(),
                "manufacturable_geometry".into(),
            ],
            required_assumptions: vec!["rigid_tube_proxy".into()],
            required_evidence: vec![EvidenceRequirement {
                source_id: "q0-fixture-corpus".into(),
                sha256: "a".repeat(64),
            }],
            baseline_candidate_id: "baseline-simple-tree".into(),
            ablation_candidate_ids: vec!["ablation-naive-heuristic".into()],
            result_evidence_tier: EvidenceTier::SyntheticSimulation,
        }
    }

    fn candidate(
        id: &str,
        coverage: f64,
        cost: f64,
        pressure_drop: f64,
    ) -> Candidate {
        Candidate {
            id: id.into(),
            profile_id: "vascular-q1-synthetic-v0".into(),
            data_split: DataSplit::Training,
            metrics: vec![
                Metric {
                    objective_id: "coverage_proxy".into(),
                    value: coverage,
                    unit: "fraction".into(),
                },
                Metric {
                    objective_id: "material_cost_proxy".into(),
                    value: cost,
                    unit: "arb_unit".into(),
                },
                Metric {
                    objective_id: "pressure_drop_proxy".into(),
                    value: pressure_drop,
                    unit: "Pa".into(),
                },
            ],
            hard_constraints: vec![
                HardConstraintResult { id: "mass_balance".into(), satisfied: true },
                HardConstraintResult { id: "manufacturable_geometry".into(), satisfied: true },
            ],
            assumptions: vec![AssumptionResolution { id: "rigid_tube_proxy".into(), resolved: true }],
            evidence: vec![EvidenceObservation {
                source_id: "q0-fixture-corpus".into(),
                sha256: "a".repeat(64),
                status: EvidenceStatus::Current,
            }],
            internal_confidence: Some(0.5),
            requested_claim_ceiling: EvidenceTier::SyntheticSimulation,
        }
    }

    fn standard_candidates() -> Vec<Candidate> {
        vec![
            candidate("baseline-simple-tree", 0.65, 7.0, 90.0),
            candidate("ablation-naive-heuristic", 0.60, 8.0, 95.0),
            candidate("design-a", 0.90, 11.0, 80.0),
            candidate("design-b", 0.85, 6.0, 110.0),
        ]
    }

    fn decision<'a>(receipt: &'a RankingReceipt, id: &str) -> &'a CandidateDecision {
        receipt
            .decisions
            .iter()
            .find(|decision| decision.candidate_id() == id)
            .unwrap_or_else(|| panic!("receipt omits candidate {id}"))
    }

    #[test]
    fn hostile_case_catalog_is_complete_and_unique() {
        let root: Value = serde_json::from_str(HOSTILE_CASES).expect("hostile case catalog must be valid JSON");
        assert_eq!(root["schema_version"].as_str(), Some("0.1.0"));
        assert_eq!(root["status"].as_str(), Some("FIXTURE_SPECIFICATION_NOT_EXECUTION_RECEIPT"));
        let cases = root["cases"].as_array().expect("cases must be an array");
        assert_eq!(cases.len(), HOSTILE_CASE_IDS.len());
        let ids: Vec<&str> = cases
            .iter()
            .map(|case| case["id"].as_str().expect("case ID must be a string"))
            .collect();
        assert_eq!(ids.as_slice(), HOSTILE_CASE_IDS);
        assert_eq!(ids.iter().collect::<HashSet<_>>().len(), ids.len());
    }

    #[test]
    fn identical_inputs_replay_deterministically_under_candidate_and_metric_permutations() {
        let p = profile();
        let candidates = standard_candidates();
        let first = evaluate_candidates(&p, &candidates).unwrap();
        let mut permuted = candidates.clone();
        permuted.reverse();
        for candidate in &mut permuted {
            candidate.metrics.reverse();
            candidate.hard_constraints.reverse();
            candidate.assumptions.reverse();
            candidate.evidence.reverse();
        }
        let second = evaluate_candidates(&p, &permuted).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.status, ReceiptStatus::Ready);
        assert!(first.selection_permitted);
    }

    #[test]
    fn changing_the_objective_profile_changes_identity_material() {
        let candidates = standard_candidates();
        let mut first_profile = profile();
        let first = evaluate_candidates(&first_profile, &candidates).unwrap();
        first_profile.objectives[1].direction = ObjectiveDirection::Maximize;
        let second = evaluate_candidates(&first_profile, &candidates).unwrap();
        assert_ne!(first.identity_material, second.identity_material);
    }

    #[test]
    fn hard_constraint_violator_cannot_win_through_better_objectives() {
        let p = profile();
        let mut candidates = standard_candidates();
        let mut unsafe_candidate = candidate("unsafe-fast", 1.0, 0.0, 0.0);
        unsafe_candidate.hard_constraints[0].satisfied = false;
        candidates.push(unsafe_candidate);
        let receipt = evaluate_candidates(&p, &candidates).unwrap();
        match decision(&receipt, "unsafe-fast") {
            CandidateDecision::Rejected { reasons, .. } => {
                assert!(reasons.contains(&CandidateRejectionReason::HardConstraintFailed("mass_balance".into())));
            }
            other => panic!("hard-constraint violator was admitted: {other:?}"),
        }
        assert!(!receipt.decisions.iter().any(|item| {
            matches!(item, CandidateDecision::ParetoRanked { candidate_id, .. } if candidate_id == "unsafe-fast")
        }));
    }

    #[test]
    fn missing_or_wrong_units_fail_closed() {
        let p = profile();
        let mut candidates = standard_candidates();
        candidates[2].metrics[0].unit.clear();
        candidates[3].metrics[1].unit = "kg".into();
        let receipt = evaluate_candidates(&p, &candidates).unwrap();
        assert!(matches!(decision(&receipt, "design-a"), CandidateDecision::Rejected { reasons, .. }
            if reasons.contains(&CandidateRejectionReason::WrongMetricUnit("coverage_proxy".into()))));
        assert!(matches!(decision(&receipt, "design-b"), CandidateDecision::Rejected { reasons, .. }
            if reasons.contains(&CandidateRejectionReason::WrongMetricUnit("material_cost_proxy".into()))));
    }

    #[test]
    fn unresolved_assumption_fails_closed() {
        let p = profile();
        let mut candidates = standard_candidates();
        candidates[2].assumptions[0].resolved = false;
        let receipt = evaluate_candidates(&p, &candidates).unwrap();
        assert!(matches!(decision(&receipt, "design-a"), CandidateDecision::Rejected { reasons, .. }
            if reasons.contains(&CandidateRejectionReason::AssumptionUnresolved("rigid_tube_proxy".into()))));
    }

    #[test]
    fn stale_and_contradictory_evidence_are_rejected_without_averaging() {
        let p = profile();
        let mut candidates = standard_candidates();
        candidates[2].evidence[0].status = EvidenceStatus::Stale;
        candidates[3].evidence[0].status = EvidenceStatus::Contradictory;
        let receipt = evaluate_candidates(&p, &candidates).unwrap();
        assert!(matches!(decision(&receipt, "design-a"), CandidateDecision::Rejected { reasons, .. }
            if reasons.contains(&CandidateRejectionReason::EvidenceStale("q0-fixture-corpus".into()))));
        assert!(matches!(decision(&receipt, "design-b"), CandidateDecision::Rejected { reasons, .. }
            if reasons.contains(&CandidateRejectionReason::EvidenceContradictory("q0-fixture-corpus".into()))));
    }

    #[test]
    fn internal_confidence_never_changes_objective_ranking_or_claim_ceiling() {
        let p = profile();
        let mut candidates = standard_candidates();
        let first = evaluate_candidates(&p, &candidates).unwrap();
        candidates.iter_mut().find(|item| item.id == "design-a").unwrap().internal_confidence = Some(1.0);
        candidates.iter_mut().find(|item| item.id == "design-b").unwrap().internal_confidence = Some(0.01);
        let second = evaluate_candidates(&p, &candidates).unwrap();
        assert_eq!(
            front(&first, "design-a"),
            front(&second, "design-a")
        );
        assert_eq!(
            front(&first, "design-b"),
            front(&second, "design-b")
        );
        assert_eq!(second.claim_ceiling, EvidenceTier::SyntheticSimulation);
    }

    #[test]
    fn synthetic_evidence_cannot_claim_physical_validation() {
        let p = profile();
        let mut candidates = standard_candidates();
        candidates[2].requested_claim_ceiling = EvidenceTier::PhysicalBench;
        let receipt = evaluate_candidates(&p, &candidates).unwrap();
        assert!(matches!(decision(&receipt, "design-a"), CandidateDecision::Rejected { reasons, .. }
            if reasons.contains(&CandidateRejectionReason::ClaimCeilingExceeded)));
        assert_eq!(receipt.claim_ceiling, EvidenceTier::SyntheticSimulation);
    }

    #[test]
    fn heldout_evaluation_is_diagnostic_only_and_cannot_select_candidates() {
        let mut p = profile();
        p.mode = EvaluationMode::HeldOutEvaluation;
        p.data_split = DataSplit::HeldOut;
        let mut candidates = standard_candidates();
        for candidate in &mut candidates {
            candidate.data_split = DataSplit::HeldOut;
        }
        let receipt = evaluate_candidates(&p, &candidates).unwrap();
        assert_eq!(receipt.status, ReceiptStatus::Ready);
        assert!(!receipt.selection_permitted);
        assert!(receipt.decisions.iter().all(|item| {
            matches!(item, CandidateDecision::EvaluatedOnly { .. })
        }));
    }

    #[test]
    fn profile_requires_baseline_and_each_ablation_to_be_admitted() {
        let p = profile();
        let candidates = vec![
            candidate("baseline-simple-tree", 0.65, 7.0, 90.0),
            candidate("design-a", 0.90, 11.0, 80.0),
        ];
        let receipt = evaluate_candidates(&p, &candidates).unwrap();
        assert_eq!(receipt.status, ReceiptStatus::Blocked);
        assert!(receipt.blocking_reasons.contains(&BlockingReason::MissingAblation("ablation-naive-heuristic".into())));
    }

    #[test]
    fn pareto_incomparables_are_preserved_instead_of_scalarized() {
        let p = profile();
        let receipt = evaluate_candidates(&p, &standard_candidates()).unwrap();
        assert_eq!(front(&receipt, "design-a"), Some(0));
        assert_eq!(front(&receipt, "design-b"), Some(0));
        assert_ne!(receipt.decisions.iter().find(|d| d.candidate_id() == "design-a"),
                   receipt.decisions.iter().find(|d| d.candidate_id() == "design-b"));
        assert_eq!(front(&receipt, "ablation-naive-heuristic"), Some(1));
    }

    #[test]
    fn duplicate_candidate_ids_are_rejected_before_ranking() {
        let p = profile();
        let mut candidates = standard_candidates();
        candidates.push(candidates[0].clone());
        assert_eq!(
            evaluate_candidates(&p, &candidates),
            Err(RankingError::DuplicateCandidateId("baseline-simple-tree".into()))
        );
    }

    fn front(receipt: &RankingReceipt, id: &str) -> Option<usize> {
        match decision(receipt, id) {
            CandidateDecision::ParetoRanked { pareto_front, .. } => Some(*pareto_front),
            _ => None,
        }
    }
}
