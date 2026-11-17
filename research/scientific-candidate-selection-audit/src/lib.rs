//! Evaluator-blind scientific candidate selection and post-freeze audit.
//!
//! The selector API receives only a SelectorPacket. EvaluatorOutcome records
//! are a separate type and are accepted only by audit_submission, after the
//! SelectionTrace has been frozen. This is an API/object-graph boundary, not
//! an OS process sandbox: production runners must also enforce separate
//! evaluator custody and deny the selector access to held-out artifacts.
//!
//! This crate does not implement a Pareto optimizer or candidate generator.
//! It validates the selector's evidence prefix and computes evaluator-only
//! dominance/regret from frozen outcome records.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

const SHA256_HEX_LEN: usize = 64;
const METRIC_NUMERIC_FIT: u8 = 0b001;
const METRIC_SCIENTIFIC_VALIDITY: u8 = 0b010;
const METRIC_STRUCTURE_RECOVERY: u8 = 0b100;
const KNOWN_METRIC_BITS: u8 =
    METRIC_NUMERIC_FIT | METRIC_SCIENTIFIC_VALIDITY | METRIC_STRUCTURE_RECOVERY;

/// Exact score dimensions declared by the pinned benchmark task.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetricMask(u8);

impl MetricMask {
    pub const NUMERIC_FIT: Self = Self(METRIC_NUMERIC_FIT);
    pub const SCIENTIFIC_VALIDITY: Self = Self(METRIC_SCIENTIFIC_VALIDITY);
    pub const STRUCTURE_RECOVERY: Self = Self(METRIC_STRUCTURE_RECOVERY);
    pub const REAL_FIT_AND_VALIDITY: Self = Self(METRIC_NUMERIC_FIT | METRIC_SCIENTIFIC_VALIDITY);
    pub const PARALLEL_FIT_AND_STRUCTURE: Self =
        Self(METRIC_NUMERIC_FIT | METRIC_STRUCTURE_RECOVERY);
    pub const ALL: Self = Self(KNOWN_METRIC_BITS);

    pub fn new(bits: u8) -> Result<Self, SelectionAuditError> {
        if bits == 0 || bits & !KNOWN_METRIC_BITS != 0 {
            return Err(SelectionAuditError::InvalidMetricMask(bits));
        }
        Ok(Self(bits))
    }

    pub fn bits(self) -> u8 {
        self.0
    }

    fn contains(self, bit: u8) -> bool {
        self.0 & bit != 0
    }
}

/// Evidence available to the live selector before submission is frozen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceStage {
    Training,
    Development,
    PredeclaredValidation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PermittedEvidence {
    pub evidence_id: String,
    pub sha256: String,
    pub stage: EvidenceStage,
}

/// Pre-submission admissibility check. This is not the hidden evaluator's
/// scientific-validity score.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DevelopmentEligibility {
    Eligible,
    Ineligible,
    Unresolved,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DevelopmentMetrics {
    /// Task-native development metric, greater is better.
    pub predictive_score: Option<f64>,
    /// A development-only stability statistic, greater is better.
    pub stability_score: Option<f64>,
    /// Simplicity/complexity descriptor; lower is usually preferable, but
    /// its semantics belong to the declared selector policy.
    pub complexity: Option<f64>,
}

/// A selector-visible candidate projection. It deliberately contains no
/// held-out metrics, evaluator rubric, hidden structure, reference law,
/// oracle ranking, or post-hoc validity label.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectorCandidate {
    pub candidate_id: String,
    pub canonical_model_sha256: Option<String>,
    pub state: CandidateState,
    pub eligibility: DevelopmentEligibility,
    pub development: DevelopmentMetrics,
    pub evidence_ids: Vec<String>,
}

/// Retained generation/execution attempt state. Failed attempts remain in
/// the candidate denominator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandidateState {
    Ready,
    GenerationFailed,
    ExecutionFailed,
    TimedOut,
    Invalid,
    Rejected,
    Incomplete,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SelectorPacket {
    pub task_id: String,
    /// SHA-256 supplied by the upstream canonical pool-manifest producer.
    /// This crate validates its syntax and trace echo, but does not recompute
    /// it: production integration must independently verify the canonical
    /// manifest-to-packet binding before constructing SelectorPacket.
    pub candidate_pool_sha256: String,
    pub metric_mask: MetricMask,
    /// One task-native cost meter. Other resources remain separate and are
    /// not silently converted into this unit.
    pub cost_unit_id: String,
    pub budget_limit_cost_units: f64,
    pub permitted_evidence: Vec<PermittedEvidence>,
    /// Complete generation/execution attempt rows. They remain in the audit
    /// denominator, but exact-artifact aliases are not separate selector choices.
    pub candidates: Vec<SelectorCandidate>,
}

/// Canonical selector-visible projection. It contains one row per unique
/// ready artifact digest; non-ready attempts and duplicate aliases are not
/// exposed as separate candidates.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectorView {
    pub task_id: String,
    pub candidate_pool_sha256: String,
    pub metric_mask: MetricMask,
    pub cost_unit_id: String,
    pub budget_limit_cost_units: f64,
    pub permitted_evidence: Vec<PermittedEvidence>,
    pub candidates: Vec<SelectorCandidate>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelectionDecision {
    Select { candidate_id: String },
    Abstain { reason: String },
}

/// What a policy returns. It receives only a canonical SelectorView, never
/// evaluator outcomes or the raw generation-attempt list.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectionObservation {
    pub decision: SelectionDecision,
    pub consumed_evidence_ids: Vec<String>,
    pub selector_cost_units: f64,
}

pub trait CandidateSelectionPolicy {
    fn policy_id(&self) -> &str;
    fn select(&self, packet: &SelectorView) -> Result<SelectionObservation, SelectionAuditError>;
}

#[derive(Debug, Clone, PartialEq)]
pub struct SelectionTrace {
    pub task_id: String,
    pub candidate_pool_sha256: String,
    pub cost_unit_id: String,
    pub policy_id: String,
    pub decision: SelectionDecision,
    pub consumed_evidence_ids: Vec<String>,
    pub selector_cost_units: f64,
}

/// Run a policy against the selector-only packet and immediately validate
/// the resulting trace. No evaluator data is accepted by this function.
pub fn run_selector<P: CandidateSelectionPolicy>(
    policy: &P,
    packet: &SelectorPacket,
) -> Result<SelectionTrace, SelectionAuditError> {
    let view = build_selector_view(packet)?;
    let observation = policy.select(&view)?;
    let trace = SelectionTrace {
        task_id: packet.task_id.clone(),
        candidate_pool_sha256: packet.candidate_pool_sha256.clone(),
        cost_unit_id: packet.cost_unit_id.clone(),
        policy_id: policy.policy_id().to_owned(),
        decision: observation.decision,
        consumed_evidence_ids: observation.consumed_evidence_ids,
        selector_cost_units: observation.selector_cost_units,
    };
    validate_selection_trace(packet, &trace)?;
    Ok(trace)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvaluatorStatus {
    Scored,
    Failed,
    TimedOut,
    Invalid,
    Incomplete,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvaluatorAdmissibility {
    Qualified,
    Unqualified,
    NotAssessed,
}

/// Evaluator-owned score vector. These fields are not present in
/// SelectorPacket or CandidateSelectionPolicy::select.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct EvaluatorMetrics {
    pub numeric_fit: Option<f64>,
    pub scientific_validity: Option<f64>,
    pub structure_recovery: Option<f64>,
}

impl EvaluatorMetrics {
    fn matches_mask(self, mask: MetricMask) -> bool {
        self.numeric_fit.is_some() == mask.contains(METRIC_NUMERIC_FIT)
            && self.scientific_validity.is_some() == mask.contains(METRIC_SCIENTIFIC_VALIDITY)
            && self.structure_recovery.is_some() == mask.contains(METRIC_STRUCTURE_RECOVERY)
    }

    fn values(self, mask: MetricMask) -> Vec<f64> {
        let mut values = Vec::with_capacity(3);
        if mask.contains(METRIC_NUMERIC_FIT) {
            values.push(self.numeric_fit.unwrap_or(f64::NAN));
        }
        if mask.contains(METRIC_SCIENTIFIC_VALIDITY) {
            values.push(self.scientific_validity.unwrap_or(f64::NAN));
        }
        if mask.contains(METRIC_STRUCTURE_RECOVERY) {
            values.push(self.structure_recovery.unwrap_or(f64::NAN));
        }
        values
    }

    fn maxima(left: Self, right: Self) -> Self {
        Self {
            numeric_fit: max_option(left.numeric_fit, right.numeric_fit),
            scientific_validity: max_option(left.scientific_validity, right.scientific_validity),
            structure_recovery: max_option(left.structure_recovery, right.structure_recovery),
        }
    }

    fn regret(best: Self, submitted: Self) -> Self {
        Self {
            numeric_fit: difference(best.numeric_fit, submitted.numeric_fit),
            scientific_validity: difference(
                best.scientific_validity,
                submitted.scientific_validity,
            ),
            structure_recovery: difference(best.structure_recovery, submitted.structure_recovery),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EvaluatorOutcome {
    pub candidate_id: String,
    pub status: EvaluatorStatus,
    pub admissibility: EvaluatorAdmissibility,
    pub metrics: EvaluatorMetrics,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MetricRegret {
    /// Independent per-axis upper bounds; these need not be achieved by one
    /// jointly available candidate and must never be scalarized implicitly.
    pub best_valid_by_metric: EvaluatorMetrics,
    pub submitted: Option<EvaluatorMetrics>,
    pub regret_by_metric: Option<EvaluatorMetrics>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CandidateSelectionAudit {
    pub task_id: String,
    pub candidate_pool_sha256: String,
    pub policy_id: String,
    pub cost_unit_id: String,
    pub selection_cost_units: f64,
    pub budget_limit_cost_units: f64,
    pub attempted_candidates: usize,
    pub ready_attempts: usize,
    pub generation_failed_attempts: usize,
    pub execution_failed_attempts: usize,
    pub timed_out_attempts: usize,
    pub invalid_attempts: usize,
    pub rejected_attempts: usize,
    pub incomplete_attempts: usize,
    pub distinct_model_artifact_count: usize,
    pub selectable_candidate_count: usize,
    pub exact_duplicate_candidate_attempts: usize,
    pub scored_candidates: usize,
    pub qualified_candidates: usize,
    pub unqualified_candidates: usize,
    pub not_assessed_candidates: usize,
    pub failed_candidates: usize,
    pub timed_out_candidates: usize,
    pub invalid_candidates: usize,
    pub incomplete_candidates: usize,
    pub missing_evaluator_outcomes: usize,
    pub identical_model_artifact_groups: Vec<Vec<String>>,
    pub submitted_candidate_id: Option<String>,
    pub submitted_status: Option<EvaluatorStatus>,
    pub submitted_admissibility: Option<EvaluatorAdmissibility>,
    pub submitted_metrics: Option<EvaluatorMetrics>,
    pub submitted_is_strictly_dominated: Option<bool>,
    pub submitted_dominator_ids: Vec<String>,
    pub qualified_pareto_frontier_ids: Vec<String>,
    pub metric_regret: MetricRegret,
    pub abstention_reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelectionAuditError {
    EmptyTaskId,
    EmptyPolicyId,
    EmptyCandidatePool,
    InvalidCandidatePoolDigest,
    InvalidArtifactDigest { evidence_id: String },
    InvalidModelDigest { candidate_id: String },
    MissingModelDigest { candidate_id: String },
    ConflictingDuplicateArtifact { digest: String },
    DuplicateCandidateId(String),
    EmptyCandidateId,
    EmptyEvidenceId,
    DuplicateEvidenceId(String),
    UnknownEvidenceId(String),
    DuplicateConsumedEvidenceId(String),
    TraceTaskMismatch,
    TracePoolDigestMismatch,
    InvalidCost,
    InvalidCostUnitId,
    InvalidBudget,
    TraceCostUnitMismatch,
    BudgetExceeded,
    EmptyAbstentionReason,
    UnknownSelectedCandidate(String),
    SelectedCandidateUnavailable(String),
    SelectedCandidateIneligible(String),
    UnknownEvaluatorCandidate(String),
    DuplicateEvaluatorOutcome(String),
    MissingMetric { candidate_id: String },
    UnexpectedMetric { candidate_id: String },
    NonFiniteMetric { candidate_id: String },
    MetricOutOfRange { candidate_id: String },
    InvalidMetricMask(u8),
    MetricsOnUnscoredOutcome { candidate_id: String },
    QualifiedUnscoredOutcome { candidate_id: String },
}

impl fmt::Display for SelectionAuditError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyTaskId => write!(f, "task ID must not be empty"),
            Self::EmptyPolicyId => write!(f, "policy ID must not be empty"),
            Self::EmptyCandidatePool => write!(f, "candidate pool must not be empty"),
            Self::InvalidCandidatePoolDigest => {
                write!(
                    f,
                    "candidate-pool digest must be 64 lowercase hexadecimal characters"
                )
            }
            Self::InvalidArtifactDigest { evidence_id } => {
                write!(f, "evidence {evidence_id} has an invalid SHA-256 digest")
            }
            Self::InvalidModelDigest { candidate_id } => {
                write!(
                    f,
                    "candidate {candidate_id} has an invalid model SHA-256 digest"
                )
            }
            Self::MissingModelDigest { candidate_id } => write!(
                f,
                "ready candidate {candidate_id} is missing its canonical model artifact digest"
            ),
            Self::ConflictingDuplicateArtifact { digest } => write!(
                f,
                "exact duplicate artifact {digest} has conflicting development metrics, eligibility or evidence references"
            ),
            Self::DuplicateCandidateId(id) => write!(f, "duplicate candidate ID: {id}"),
            Self::EmptyCandidateId => write!(f, "candidate ID must not be empty"),
            Self::EmptyEvidenceId => write!(f, "evidence ID must not be empty"),
            Self::DuplicateEvidenceId(id) => write!(f, "duplicate evidence ID: {id}"),
            Self::UnknownEvidenceId(id) => {
                write!(f, "selector trace references unavailable evidence: {id}")
            }
            Self::DuplicateConsumedEvidenceId(id) => {
                write!(f, "selector trace repeats evidence ID: {id}")
            }
            Self::TraceTaskMismatch => write!(f, "selector trace task ID does not match packet"),
            Self::TracePoolDigestMismatch => {
                write!(
                    f,
                    "selector trace candidate-pool digest does not match frozen pool"
                )
            }
            Self::InvalidCost => write!(f, "all-in selector cost must be finite and nonnegative"),
            Self::InvalidCostUnitId => write!(f, "task-native cost unit ID must not be empty"),
            Self::InvalidBudget => {
                write!(f, "task-native cost budget must be finite and nonnegative")
            }
            Self::TraceCostUnitMismatch => {
                write!(f, "selector trace cost unit does not match task contract")
            }
            Self::BudgetExceeded => write!(f, "selector cost exceeds the task-native hard budget"),
            Self::EmptyAbstentionReason => write!(f, "abstention reason must not be empty"),
            Self::UnknownSelectedCandidate(id) => {
                write!(f, "selected candidate is not in the frozen pool: {id}")
            }
            Self::SelectedCandidateUnavailable(id) => {
                write!(f, "selected candidate {id} is not in Ready state")
            }
            Self::SelectedCandidateIneligible(id) => {
                write!(f, "selected candidate {id} is not development-eligible")
            }
            Self::UnknownEvaluatorCandidate(id) => {
                write!(f, "evaluator outcome references unknown candidate: {id}")
            }
            Self::DuplicateEvaluatorOutcome(id) => {
                write!(f, "duplicate evaluator outcome for candidate: {id}")
            }
            Self::MissingMetric { candidate_id } => {
                write!(f, "candidate {candidate_id} is missing a declared metric")
            }
            Self::UnexpectedMetric { candidate_id } => {
                write!(f, "candidate {candidate_id} has an undeclared metric")
            }
            Self::NonFiniteMetric { candidate_id } => {
                write!(
                    f,
                    "candidate {candidate_id} has a non-finite evaluator metric"
                )
            }
            Self::MetricOutOfRange { candidate_id } => {
                write!(
                    f,
                    "candidate {candidate_id} evaluator metric is outside [0, 100]"
                )
            }
            Self::InvalidMetricMask(bits) => {
                write!(f, "invalid evaluator metric mask: {bits:#05b}")
            }
            Self::MetricsOnUnscoredOutcome { candidate_id } => {
                write!(
                    f,
                    "unscored outcome for {candidate_id} contains evaluator metrics"
                )
            }
            Self::QualifiedUnscoredOutcome { candidate_id } => {
                write!(f, "unscored outcome for {candidate_id} cannot be Qualified")
            }
        }
    }
}
impl Error for SelectionAuditError {}

fn valid_sha256(value: &str) -> bool {
    value.len() == SHA256_HEX_LEN
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn max_option(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (left, right) {
        (Some(a), Some(b)) => Some(a.max(b)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

fn difference(best: Option<f64>, submitted: Option<f64>) -> Option<f64> {
    match (best, submitted) {
        (Some(best), Some(submitted)) => Some((best - submitted).max(0.0)),
        _ => None,
    }
}

fn all_metrics_absent(metrics: EvaluatorMetrics) -> bool {
    metrics.numeric_fit.is_none()
        && metrics.scientific_validity.is_none()
        && metrics.structure_recovery.is_none()
}

fn validate_metric_values(
    candidate_id: &str,
    metrics: EvaluatorMetrics,
    mask: MetricMask,
) -> Result<(), SelectionAuditError> {
    if !metrics.matches_mask(mask) {
        let values = metrics.values(mask);
        if values.iter().any(|value| !value.is_finite()) {
            return Err(SelectionAuditError::MissingMetric {
                candidate_id: candidate_id.to_owned(),
            });
        }
        return Err(SelectionAuditError::UnexpectedMetric {
            candidate_id: candidate_id.to_owned(),
        });
    }
    for value in metrics.values(mask) {
        if !value.is_finite() {
            return Err(SelectionAuditError::NonFiniteMetric {
                candidate_id: candidate_id.to_owned(),
            });
        }
        if !(0.0..=100.0).contains(&value) {
            return Err(SelectionAuditError::MetricOutOfRange {
                candidate_id: candidate_id.to_owned(),
            });
        }
    }
    Ok(())
}

pub fn validate_selector_packet(packet: &SelectorPacket) -> Result<(), SelectionAuditError> {
    if packet.task_id.trim().is_empty() {
        return Err(SelectionAuditError::EmptyTaskId);
    }
    if !valid_sha256(&packet.candidate_pool_sha256) {
        return Err(SelectionAuditError::InvalidCandidatePoolDigest);
    }
    if packet.candidates.is_empty() {
        return Err(SelectionAuditError::EmptyCandidatePool);
    }
    if packet.cost_unit_id.trim().is_empty() {
        return Err(SelectionAuditError::InvalidCostUnitId);
    }
    if !packet.budget_limit_cost_units.is_finite() || packet.budget_limit_cost_units < 0.0 {
        return Err(SelectionAuditError::InvalidBudget);
    }

    let mut candidate_ids = BTreeSet::new();
    for candidate in &packet.candidates {
        if candidate.candidate_id.trim().is_empty() {
            return Err(SelectionAuditError::EmptyCandidateId);
        }
        if !candidate_ids.insert(candidate.candidate_id.as_str()) {
            return Err(SelectionAuditError::DuplicateCandidateId(
                candidate.candidate_id.clone(),
            ));
        }
        if let Some(digest) = &candidate.canonical_model_sha256
            && !valid_sha256(digest)
        {
            return Err(SelectionAuditError::InvalidModelDigest {
                candidate_id: candidate.candidate_id.clone(),
            });
        }
        if candidate.state == CandidateState::Ready && candidate.canonical_model_sha256.is_none() {
            return Err(SelectionAuditError::MissingModelDigest {
                candidate_id: candidate.candidate_id.clone(),
            });
        }
        for value in [
            candidate.development.predictive_score,
            candidate.development.stability_score,
            candidate.development.complexity,
        ]
        .into_iter()
        .flatten()
        {
            if !value.is_finite() {
                return Err(SelectionAuditError::NonFiniteMetric {
                    candidate_id: candidate.candidate_id.clone(),
                });
            }
        }
    }

    let mut evidence_ids = BTreeSet::new();
    for evidence in &packet.permitted_evidence {
        if evidence.evidence_id.trim().is_empty() {
            return Err(SelectionAuditError::EmptyEvidenceId);
        }
        if !evidence_ids.insert(evidence.evidence_id.as_str()) {
            return Err(SelectionAuditError::DuplicateEvidenceId(
                evidence.evidence_id.clone(),
            ));
        }
        if !valid_sha256(&evidence.sha256) {
            return Err(SelectionAuditError::InvalidArtifactDigest {
                evidence_id: evidence.evidence_id.clone(),
            });
        }
    }

    for candidate in &packet.candidates {
        let mut local = BTreeSet::new();
        for evidence_id in &candidate.evidence_ids {
            if !local.insert(evidence_id.as_str()) {
                return Err(SelectionAuditError::DuplicateConsumedEvidenceId(
                    evidence_id.clone(),
                ));
            }
            if !evidence_ids.contains(evidence_id.as_str()) {
                return Err(SelectionAuditError::UnknownEvidenceId(evidence_id.clone()));
            }
        }
    }
    Ok(())
}

/// Construct the selector-visible projection from the complete attempt
/// packet. Exact SHA-256 duplicates collapse to one canonical candidate ID.
/// Duplicate rows must have identical development metrics, eligibility and
/// evidence references; disagreement fails closed rather than choosing the
/// most favorable duplicate. Non-ready attempts remain evaluator-visible but
/// are not selector choices.
pub fn build_selector_view(packet: &SelectorPacket) -> Result<SelectorView, SelectionAuditError> {
    validate_selector_packet(packet)?;
    let mut by_digest: BTreeMap<String, SelectorCandidate> = BTreeMap::new();
    for candidate in packet
        .candidates
        .iter()
        .filter(|candidate| candidate.state == CandidateState::Ready)
    {
        let digest = candidate
            .canonical_model_sha256
            .as_ref()
            .expect("ready candidate digest was validated");
        match by_digest.get_mut(digest) {
            Some(existing) => {
                let mut left_evidence = existing.evidence_ids.clone();
                let mut right_evidence = candidate.evidence_ids.clone();
                left_evidence.sort();
                right_evidence.sort();
                if existing.eligibility != candidate.eligibility
                    || existing.development != candidate.development
                    || left_evidence != right_evidence
                {
                    return Err(SelectionAuditError::ConflictingDuplicateArtifact {
                        digest: digest.clone(),
                    });
                }
                if candidate.candidate_id < existing.candidate_id {
                    existing.candidate_id = candidate.candidate_id.clone();
                    existing.evidence_ids = right_evidence;
                }
            }
            None => {
                let mut canonical = candidate.clone();
                canonical.evidence_ids.sort();
                by_digest.insert(digest.clone(), canonical);
            }
        }
    }
    let mut candidates = by_digest.into_values().collect::<Vec<_>>();
    candidates.sort_by(|left, right| left.candidate_id.cmp(&right.candidate_id));
    Ok(SelectorView {
        task_id: packet.task_id.clone(),
        candidate_pool_sha256: packet.candidate_pool_sha256.clone(),
        metric_mask: packet.metric_mask,
        cost_unit_id: packet.cost_unit_id.clone(),
        budget_limit_cost_units: packet.budget_limit_cost_units,
        permitted_evidence: packet.permitted_evidence.clone(),
        candidates,
    })
}

pub fn validate_selection_trace(
    packet: &SelectorPacket,
    trace: &SelectionTrace,
) -> Result<(), SelectionAuditError> {
    let view = build_selector_view(packet)?;
    if trace.task_id != packet.task_id {
        return Err(SelectionAuditError::TraceTaskMismatch);
    }
    if trace.candidate_pool_sha256 != packet.candidate_pool_sha256 {
        return Err(SelectionAuditError::TracePoolDigestMismatch);
    }
    if trace.policy_id.trim().is_empty() {
        return Err(SelectionAuditError::EmptyPolicyId);
    }
    if trace.cost_unit_id != packet.cost_unit_id {
        return Err(SelectionAuditError::TraceCostUnitMismatch);
    }
    if !trace.selector_cost_units.is_finite() || trace.selector_cost_units < 0.0 {
        return Err(SelectionAuditError::InvalidCost);
    }
    if trace.selector_cost_units > packet.budget_limit_cost_units {
        return Err(SelectionAuditError::BudgetExceeded);
    }

    let allowed = packet
        .permitted_evidence
        .iter()
        .map(|evidence| evidence.evidence_id.as_str())
        .collect::<BTreeSet<_>>();
    let mut consumed = BTreeSet::new();
    for evidence_id in &trace.consumed_evidence_ids {
        if !consumed.insert(evidence_id.as_str()) {
            return Err(SelectionAuditError::DuplicateConsumedEvidenceId(
                evidence_id.clone(),
            ));
        }
        if !allowed.contains(evidence_id.as_str()) {
            return Err(SelectionAuditError::UnknownEvidenceId(evidence_id.clone()));
        }
    }

    match &trace.decision {
        SelectionDecision::Select { candidate_id } => {
            let candidate = view
                .candidates
                .iter()
                .find(|candidate| candidate.candidate_id == *candidate_id)
                .ok_or_else(|| {
                    SelectionAuditError::UnknownSelectedCandidate(candidate_id.clone())
                })?;
            if candidate.state != CandidateState::Ready {
                return Err(SelectionAuditError::SelectedCandidateUnavailable(
                    candidate_id.clone(),
                ));
            }
            if candidate.eligibility != DevelopmentEligibility::Eligible {
                return Err(SelectionAuditError::SelectedCandidateIneligible(
                    candidate_id.clone(),
                ));
            }
        }
        SelectionDecision::Abstain { reason } => {
            if reason.trim().is_empty() {
                return Err(SelectionAuditError::EmptyAbstentionReason);
            }
        }
    }
    Ok(())
}

fn admissibility_rank(value: EvaluatorAdmissibility) -> u8 {
    match value {
        EvaluatorAdmissibility::NotAssessed => 0,
        EvaluatorAdmissibility::Unqualified => 1,
        EvaluatorAdmissibility::Qualified => 2,
    }
}

fn dominates(left: &EvaluatorOutcome, right: &EvaluatorOutcome, mask: MetricMask) -> bool {
    if left.status != EvaluatorStatus::Scored
        || right.status != EvaluatorStatus::Scored
        || !left.metrics.matches_mask(mask)
        || !right.metrics.matches_mask(mask)
    {
        return false;
    }
    let left_rank = admissibility_rank(left.admissibility);
    let right_rank = admissibility_rank(right.admissibility);
    if left_rank < right_rank {
        return false;
    }
    let left_values = left.metrics.values(mask);
    let right_values = right.metrics.values(mask);
    let mut strictly_better = left_rank > right_rank;
    for (a, b) in left_values.into_iter().zip(right_values) {
        if a < b {
            return false;
        }
        if a > b {
            strictly_better = true;
        }
    }
    strictly_better
}

fn identical_model_artifact_groups(candidates: &[SelectorCandidate]) -> Vec<Vec<String>> {
    let mut by_digest: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for candidate in candidates {
        if let Some(digest) = &candidate.canonical_model_sha256 {
            by_digest
                .entry(digest)
                .or_default()
                .push(candidate.candidate_id.as_str());
        }
    }
    by_digest
        .into_values()
        .filter(|ids| ids.len() > 1)
        .map(|mut ids| {
            ids.sort_unstable();
            ids.into_iter().map(str::to_owned).collect()
        })
        .collect()
}

/// Validate a frozen selector trace, then score it with evaluator-only
/// outcomes. This is the only function that accepts both public and
/// evaluator-owned records; call it only after the selection trace is frozen.
pub fn audit_submission(
    packet: &SelectorPacket,
    trace: &SelectionTrace,
    outcomes: &[EvaluatorOutcome],
) -> Result<CandidateSelectionAudit, SelectionAuditError> {
    let view = build_selector_view(packet)?;
    validate_selection_trace(packet, trace)?;

    // Evaluator outcomes are scored once per canonical selectable artifact,
    // not once per duplicate generation alias.
    let selectable_ids = view
        .candidates
        .iter()
        .map(|candidate| candidate.candidate_id.as_str())
        .collect::<BTreeSet<_>>();
    let mut by_id: BTreeMap<&str, &EvaluatorOutcome> = BTreeMap::new();
    for outcome in outcomes {
        if !selectable_ids.contains(outcome.candidate_id.as_str()) {
            return Err(SelectionAuditError::UnknownEvaluatorCandidate(
                outcome.candidate_id.clone(),
            ));
        }
        if by_id
            .insert(outcome.candidate_id.as_str(), outcome)
            .is_some()
        {
            return Err(SelectionAuditError::DuplicateEvaluatorOutcome(
                outcome.candidate_id.clone(),
            ));
        }
        match outcome.status {
            EvaluatorStatus::Scored => {
                validate_metric_values(&outcome.candidate_id, outcome.metrics, packet.metric_mask)?;
            }
            _ => {
                if !all_metrics_absent(outcome.metrics) {
                    return Err(SelectionAuditError::MetricsOnUnscoredOutcome {
                        candidate_id: outcome.candidate_id.clone(),
                    });
                }
                if outcome.admissibility == EvaluatorAdmissibility::Qualified {
                    return Err(SelectionAuditError::QualifiedUnscoredOutcome {
                        candidate_id: outcome.candidate_id.clone(),
                    });
                }
            }
        }
    }

    let mut scored = 0;
    let mut qualified = 0;
    let mut unqualified = 0;
    let mut not_assessed = 0;
    let mut failed = 0;
    let mut timed_out = 0;
    let mut invalid = 0;
    let mut incomplete = 0;
    let mut best_valid = EvaluatorMetrics::default();
    let mut has_valid = false;

    for outcome in outcomes {
        match outcome.status {
            EvaluatorStatus::Scored => {
                scored += 1;
                match outcome.admissibility {
                    EvaluatorAdmissibility::Qualified => {
                        qualified += 1;
                        best_valid = if has_valid {
                            EvaluatorMetrics::maxima(best_valid, outcome.metrics)
                        } else {
                            outcome.metrics
                        };
                        has_valid = true;
                    }
                    EvaluatorAdmissibility::Unqualified => unqualified += 1,
                    EvaluatorAdmissibility::NotAssessed => not_assessed += 1,
                }
            }
            EvaluatorStatus::Failed => failed += 1,
            EvaluatorStatus::TimedOut => timed_out += 1,
            EvaluatorStatus::Invalid => invalid += 1,
            EvaluatorStatus::Incomplete => incomplete += 1,
        }
    }
    let missing = view.candidates.len().saturating_sub(outcomes.len());

    let mut frontier = Vec::new();
    for candidate in &view.candidates {
        let Some(outcome) = by_id.get(candidate.candidate_id.as_str()) else {
            continue;
        };
        if outcome.status != EvaluatorStatus::Scored
            || outcome.admissibility != EvaluatorAdmissibility::Qualified
        {
            continue;
        }
        let is_dominated = view.candidates.iter().any(|other| {
            if other.candidate_id == candidate.candidate_id {
                return false;
            }
            let Some(other_outcome) = by_id.get(other.candidate_id.as_str()) else {
                return false;
            };
            other_outcome.status == EvaluatorStatus::Scored
                && other_outcome.admissibility == EvaluatorAdmissibility::Qualified
                && dominates(other_outcome, outcome, packet.metric_mask)
        });
        if !is_dominated {
            frontier.push(candidate.candidate_id.clone());
        }
    }
    frontier.sort();

    let (submitted_id, abstention_reason) = match &trace.decision {
        SelectionDecision::Select { candidate_id } => (Some(candidate_id.clone()), None),
        SelectionDecision::Abstain { reason } => (None, Some(reason.clone())),
    };
    let submitted_outcome = submitted_id
        .as_deref()
        .and_then(|id| by_id.get(id).copied());

    let mut dominators = Vec::new();
    if let Some(submitted) = submitted_outcome
        && submitted.status == EvaluatorStatus::Scored
    {
        for candidate in &view.candidates {
            if Some(candidate.candidate_id.as_str()) == submitted_id.as_deref() {
                continue;
            }
            if let Some(other) = by_id.get(candidate.candidate_id.as_str())
                && other.status == EvaluatorStatus::Scored
                && dominates(other, submitted, packet.metric_mask)
            {
                dominators.push(candidate.candidate_id.clone());
            }
        }
    }
    dominators.sort();

    let submitted_metrics = submitted_outcome
        .filter(|outcome| outcome.status == EvaluatorStatus::Scored)
        .map(|outcome| outcome.metrics);
    let regret = MetricRegret {
        best_valid_by_metric: best_valid,
        submitted: submitted_metrics,
        regret_by_metric: if has_valid {
            submitted_metrics.map(|metrics| EvaluatorMetrics::regret(best_valid, metrics))
        } else {
            None
        },
    };

    let identical_groups = identical_model_artifact_groups(&packet.candidates);
    Ok(CandidateSelectionAudit {
        task_id: packet.task_id.clone(),
        candidate_pool_sha256: packet.candidate_pool_sha256.clone(),
        policy_id: trace.policy_id.clone(),
        cost_unit_id: trace.cost_unit_id.clone(),
        selection_cost_units: trace.selector_cost_units,
        budget_limit_cost_units: packet.budget_limit_cost_units,
        attempted_candidates: packet.candidates.len(),
        ready_attempts: packet
            .candidates
            .iter()
            .filter(|candidate| candidate.state == CandidateState::Ready)
            .count(),
        generation_failed_attempts: packet
            .candidates
            .iter()
            .filter(|candidate| candidate.state == CandidateState::GenerationFailed)
            .count(),
        execution_failed_attempts: packet
            .candidates
            .iter()
            .filter(|candidate| candidate.state == CandidateState::ExecutionFailed)
            .count(),
        timed_out_attempts: packet
            .candidates
            .iter()
            .filter(|candidate| candidate.state == CandidateState::TimedOut)
            .count(),
        invalid_attempts: packet
            .candidates
            .iter()
            .filter(|candidate| candidate.state == CandidateState::Invalid)
            .count(),
        rejected_attempts: packet
            .candidates
            .iter()
            .filter(|candidate| candidate.state == CandidateState::Rejected)
            .count(),
        incomplete_attempts: packet
            .candidates
            .iter()
            .filter(|candidate| candidate.state == CandidateState::Incomplete)
            .count(),
        distinct_model_artifact_count: packet
            .candidates
            .iter()
            .filter_map(|candidate| candidate.canonical_model_sha256.as_deref())
            .collect::<BTreeSet<_>>()
            .len(),
        selectable_candidate_count: view.candidates.len(),
        exact_duplicate_candidate_attempts: identical_groups
            .iter()
            .map(|group| group.len().saturating_sub(1))
            .sum(),
        scored_candidates: scored,
        qualified_candidates: qualified,
        unqualified_candidates: unqualified,
        not_assessed_candidates: not_assessed,
        failed_candidates: failed,
        timed_out_candidates: timed_out,
        invalid_candidates: invalid,
        incomplete_candidates: incomplete,
        missing_evaluator_outcomes: missing,
        identical_model_artifact_groups: identical_groups,
        submitted_candidate_id: submitted_id,
        submitted_status: submitted_outcome.map(|outcome| outcome.status),
        submitted_admissibility: submitted_outcome.map(|outcome| outcome.admissibility),
        submitted_metrics,
        submitted_is_strictly_dominated: submitted_outcome
            .filter(|outcome| outcome.status == EvaluatorStatus::Scored)
            .map(|_| !dominators.is_empty()),
        submitted_dominator_ids: dominators,
        qualified_pareto_frontier_ids: frontier,
        metric_regret: regret,
        abstention_reason,
    })
}

fn json_escape(input: &str) -> String {
    let mut output = String::with_capacity(input.len() + 2);
    output.push('"');
    for character in input.chars() {
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            c if c.is_control() => {
                use std::fmt::Write;
                let _ = write!(output, "\\u{:04x}", c as u32);
            }
            c => output.push(c),
        }
    }
    output.push('"');
    output
}

fn json_number(value: Option<f64>) -> String {
    value
        .map(|number| format!("{number:.8}"))
        .unwrap_or_else(|| "null".to_owned())
}

fn metrics_json(metrics: Option<EvaluatorMetrics>) -> String {
    match metrics {
        None => "null".to_owned(),
        Some(metrics) => format!(
            concat!(
                "{{\"numeric_fit\":{},",
                "\"scientific_validity\":{},",
                "\"structure_recovery\":{}}}"
            ),
            json_number(metrics.numeric_fit),
            json_number(metrics.scientific_validity),
            json_number(metrics.structure_recovery),
        ),
    }
}

impl CandidateSelectionAudit {
    /// Deterministic JSON projection for a machine-readable evaluator receipt.
    pub fn to_json(&self) -> String {
        let ids = |values: &[String]| {
            format!(
                "[{}]",
                values
                    .iter()
                    .map(|value| json_escape(value))
                    .collect::<Vec<_>>()
                    .join(",")
            )
        };
        let duplicate_groups = format!(
            "[{}]",
            self.identical_model_artifact_groups
                .iter()
                .map(|group| ids(group))
                .collect::<Vec<_>>()
                .join(",")
        );
        let status = self
            .submitted_status
            .map(|value| json_escape(&format!("{value:?}")))
            .unwrap_or_else(|| "null".to_owned());
        let admissibility = self
            .submitted_admissibility
            .map(|value| json_escape(&format!("{value:?}")))
            .unwrap_or_else(|| "null".to_owned());
        let dominated = self
            .submitted_is_strictly_dominated
            .map(|value| value.to_string())
            .unwrap_or_else(|| "null".to_owned());
        let regret = &self.metric_regret;
        format!(
            concat!(
                "{{",
                "\"schema\":\"scientific_candidate_selection_audit_v0\",",
                "\"task_id\":{},",
                "\"candidate_pool_sha256\":{},",
                "\"policy_id\":{},",
                "\"cost_unit_id\":{},",
                "\"selection_cost_units\":{:.8},",
                "\"budget_limit_cost_units\":{:.8},",
                "\"attempted_candidates\":{},",
                "\"ready_attempts\":{},",
                "\"generation_failed_attempts\":{},",
                "\"execution_failed_attempts\":{},",
                "\"timed_out_attempts\":{},",
                "\"invalid_attempts\":{},",
                "\"rejected_attempts\":{},",
                "\"incomplete_attempts\":{},",
                "\"distinct_model_artifact_count\":{},",
                "\"selectable_candidate_count\":{},",
                "\"exact_duplicate_candidate_attempts\":{},",
                "\"scored_candidates\":{},",
                "\"qualified_candidates\":{},",
                "\"unqualified_candidates\":{},",
                "\"not_assessed_candidates\":{},",
                "\"failed_candidates\":{},",
                "\"timed_out_candidates\":{},",
                "\"invalid_candidates\":{},",
                "\"incomplete_candidates\":{},",
                "\"missing_evaluator_outcomes\":{},",
                "\"identical_model_artifact_groups\":{},",
                "\"submitted_candidate_id\":{},",
                "\"submitted_status\":{},",
                "\"submitted_admissibility\":{},",
                "\"submitted_metrics\":{},",
                "\"submitted_is_strictly_dominated\":{},",
                "\"submitted_dominator_ids\":{},",
                "\"qualified_pareto_frontier_ids\":{},",
                "\"best_valid_by_metric\":{},",
                "\"regret_by_metric\":{},",
                "\"abstention_reason\":{}",
                "}}"
            ),
            json_escape(&self.task_id),
            json_escape(&self.candidate_pool_sha256),
            json_escape(&self.policy_id),
            json_escape(&self.cost_unit_id),
            self.selection_cost_units,
            self.budget_limit_cost_units,
            self.attempted_candidates,
            self.ready_attempts,
            self.generation_failed_attempts,
            self.execution_failed_attempts,
            self.timed_out_attempts,
            self.invalid_attempts,
            self.rejected_attempts,
            self.incomplete_attempts,
            self.distinct_model_artifact_count,
            self.selectable_candidate_count,
            self.exact_duplicate_candidate_attempts,
            self.scored_candidates,
            self.qualified_candidates,
            self.unqualified_candidates,
            self.not_assessed_candidates,
            self.failed_candidates,
            self.timed_out_candidates,
            self.invalid_candidates,
            self.incomplete_candidates,
            self.missing_evaluator_outcomes,
            duplicate_groups,
            self.submitted_candidate_id
                .as_deref()
                .map(json_escape)
                .unwrap_or_else(|| "null".to_owned()),
            status,
            admissibility,
            metrics_json(self.submitted_metrics),
            dominated,
            ids(&self.submitted_dominator_ids),
            ids(&self.qualified_pareto_frontier_ids),
            metrics_json(Some(regret.best_valid_by_metric)),
            metrics_json(regret.regret_by_metric),
            self.abstention_reason
                .as_deref()
                .map(json_escape)
                .unwrap_or_else(|| "null".to_owned()),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const POOL_SHA: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const MODEL_A: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const MODEL_B: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const MODEL_C: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const EVIDENCE_SHA: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";

    fn candidate(id: &str, digest: &str) -> SelectorCandidate {
        SelectorCandidate {
            candidate_id: id.to_owned(),
            canonical_model_sha256: Some(digest.to_owned()),
            state: CandidateState::Ready,
            eligibility: DevelopmentEligibility::Eligible,
            development: DevelopmentMetrics {
                predictive_score: Some(0.8),
                stability_score: Some(0.9),
                complexity: Some(4.0),
            },
            evidence_ids: vec!["dev-fold-1".to_owned()],
        }
    }

    fn packet() -> SelectorPacket {
        SelectorPacket {
            task_id: "scilaws-real-fixture".to_owned(),
            candidate_pool_sha256: POOL_SHA.to_owned(),
            metric_mask: MetricMask::REAL_FIT_AND_VALIDITY,
            cost_unit_id: "fixture_cost_units".to_owned(),
            budget_limit_cost_units: 20.0,
            permitted_evidence: vec![PermittedEvidence {
                evidence_id: "dev-fold-1".to_owned(),
                sha256: EVIDENCE_SHA.to_owned(),
                stage: EvidenceStage::Development,
            }],
            candidates: vec![
                candidate("a", MODEL_A),
                candidate("b", MODEL_B),
                candidate("c", MODEL_C),
            ],
        }
    }

    fn trace(decision: SelectionDecision) -> SelectionTrace {
        SelectionTrace {
            task_id: "scilaws-real-fixture".to_owned(),
            candidate_pool_sha256: POOL_SHA.to_owned(),
            cost_unit_id: "fixture_cost_units".to_owned(),
            policy_id: "test-policy-v1".to_owned(),
            decision,
            consumed_evidence_ids: vec!["dev-fold-1".to_owned()],
            selector_cost_units: 12.5,
        }
    }

    fn real_outcome(
        id: &str,
        numeric_fit: f64,
        scientific_validity: f64,
        admissibility: EvaluatorAdmissibility,
    ) -> EvaluatorOutcome {
        EvaluatorOutcome {
            candidate_id: id.to_owned(),
            status: EvaluatorStatus::Scored,
            admissibility,
            metrics: EvaluatorMetrics {
                numeric_fit: Some(numeric_fit),
                scientific_validity: Some(scientific_validity),
                structure_recovery: None,
            },
        }
    }

    struct FixedPolicy {
        selected: String,
    }

    impl CandidateSelectionPolicy for FixedPolicy {
        fn policy_id(&self) -> &str {
            "fixed-test-policy-v1"
        }

        fn select(
            &self,
            packet: &SelectorView,
        ) -> Result<SelectionObservation, SelectionAuditError> {
            assert!(
                packet
                    .candidates
                    .iter()
                    .any(|c| c.candidate_id == self.selected)
            );
            Ok(SelectionObservation {
                decision: SelectionDecision::Select {
                    candidate_id: self.selected.clone(),
                },
                consumed_evidence_ids: vec!["dev-fold-1".to_owned()],
                selector_cost_units: 1.0,
            })
        }
    }

    #[test]
    fn selector_interface_receives_only_development_packet() {
        let packet = packet();
        let policy = FixedPolicy {
            selected: "a".to_owned(),
        };
        let trace = run_selector(&policy, &packet).unwrap();
        assert_eq!(
            trace.decision,
            SelectionDecision::Select {
                candidate_id: "a".to_owned()
            }
        );
        assert_eq!(trace.consumed_evidence_ids, vec!["dev-fold-1"]);
    }

    #[test]
    fn exact_duplicate_artifacts_are_not_separate_selector_choices() {
        let original = packet();
        let mut duplicated = packet();
        duplicated.candidates.push(candidate("a-copy", MODEL_A));

        let original_view = build_selector_view(&original).unwrap();
        let duplicate_view = build_selector_view(&duplicated).unwrap();
        assert_eq!(original_view.candidates.len(), 3);
        assert_eq!(duplicate_view.candidates.len(), 3);
        assert_eq!(original_view.candidates, duplicate_view.candidates);
        assert_eq!(
            duplicate_view
                .candidates
                .iter()
                .filter(|candidate| candidate.canonical_model_sha256.as_deref() == Some(MODEL_A))
                .map(|candidate| candidate.candidate_id.as_str())
                .collect::<Vec<_>>(),
            vec!["a"]
        );

        let policy = FixedPolicy {
            selected: "a".to_owned(),
        };
        let original_trace = run_selector(&policy, &original).unwrap();
        let duplicate_trace = run_selector(&policy, &duplicated).unwrap();
        assert_eq!(original_trace.decision, duplicate_trace.decision);
    }

    #[test]
    fn selector_cannot_choose_a_noncanonical_duplicate_alias() {
        let mut packet = packet();
        packet.candidates.push(candidate("a-copy", MODEL_A));
        let mut frozen = trace(SelectionDecision::Select {
            candidate_id: "a-copy".to_owned(),
        });
        assert_eq!(
            validate_selection_trace(&packet, &frozen),
            Err(SelectionAuditError::UnknownSelectedCandidate(
                "a-copy".to_owned()
            ))
        );
        frozen.decision = SelectionDecision::Select {
            candidate_id: "a".to_owned(),
        };
        validate_selection_trace(&packet, &frozen).unwrap();
    }

    #[test]
    fn conflicting_duplicate_artifact_evidence_fails_closed() {
        let mut packet = packet();
        let mut duplicate = candidate("a-copy", MODEL_A);
        duplicate.development.predictive_score = Some(0.81);
        packet.candidates.push(duplicate);
        assert_eq!(
            build_selector_view(&packet),
            Err(SelectionAuditError::ConflictingDuplicateArtifact {
                digest: MODEL_A.to_owned()
            })
        );
    }

    #[test]
    fn evaluator_artifact_cannot_be_smuggled_through_trace() {
        let packet = packet();
        let mut frozen = trace(SelectionDecision::Select {
            candidate_id: "a".to_owned(),
        });
        frozen
            .consumed_evidence_ids
            .push("heldout-test-sv".to_owned());
        assert_eq!(
            validate_selection_trace(&packet, &frozen),
            Err(SelectionAuditError::UnknownEvidenceId(
                "heldout-test-sv".to_owned()
            ))
        );
    }

    #[test]
    fn pool_and_task_identity_are_bound_to_selection_trace() {
        let packet = packet();
        let mut frozen = trace(SelectionDecision::Select {
            candidate_id: "a".to_owned(),
        });
        frozen.candidate_pool_sha256 = MODEL_A.to_owned();
        assert_eq!(
            validate_selection_trace(&packet, &frozen),
            Err(SelectionAuditError::TracePoolDigestMismatch)
        );
    }

    #[test]
    fn cost_unit_and_hard_budget_are_bound_to_the_task_contract() {
        let packet = packet();
        let mut frozen = trace(SelectionDecision::Select {
            candidate_id: "a".to_owned(),
        });
        frozen.selector_cost_units = 21.0;
        assert_eq!(
            validate_selection_trace(&packet, &frozen),
            Err(SelectionAuditError::BudgetExceeded)
        );
        frozen.selector_cost_units = 2.0;
        frozen.cost_unit_id = "unrelated_cost_unit".to_owned();
        assert_eq!(
            validate_selection_trace(&packet, &frozen),
            Err(SelectionAuditError::TraceCostUnitMismatch)
        );
    }

    #[test]
    fn strict_dominance_and_per_metric_regret_are_reported_separately() {
        let packet = packet();
        let frozen = trace(SelectionDecision::Select {
            candidate_id: "a".to_owned(),
        });
        let outcomes = vec![
            real_outcome("a", 70.0, 85.0, EvaluatorAdmissibility::Qualified),
            real_outcome("b", 75.0, 90.0, EvaluatorAdmissibility::Qualified),
            real_outcome("c", 80.0, 82.0, EvaluatorAdmissibility::Qualified),
        ];
        let report = audit_submission(&packet, &frozen, &outcomes).unwrap();
        assert_eq!(report.submitted_is_strictly_dominated, Some(true));
        assert_eq!(report.submitted_dominator_ids, vec!["b"]);
        assert_eq!(report.qualified_pareto_frontier_ids, vec!["b", "c"]);
        assert_eq!(
            report.metric_regret.regret_by_metric.unwrap().numeric_fit,
            Some(10.0)
        );
        assert_eq!(
            report
                .metric_regret
                .regret_by_metric
                .unwrap()
                .scientific_validity,
            Some(5.0)
        );
        // The per-metric maxima need not come from one candidate.
        assert_eq!(
            report.metric_regret.best_valid_by_metric.numeric_fit,
            Some(80.0)
        );
        assert_eq!(
            report
                .metric_regret
                .best_valid_by_metric
                .scientific_validity,
            Some(90.0)
        );
    }

    #[test]
    fn incomparable_candidates_remain_on_the_frontier() {
        let packet = packet();
        let frozen = trace(SelectionDecision::Select {
            candidate_id: "b".to_owned(),
        });
        let outcomes = vec![
            real_outcome("a", 80.0, 70.0, EvaluatorAdmissibility::Qualified),
            real_outcome("b", 70.0, 80.0, EvaluatorAdmissibility::Qualified),
            real_outcome("c", 65.0, 60.0, EvaluatorAdmissibility::Unqualified),
        ];
        let report = audit_submission(&packet, &frozen, &outcomes).unwrap();
        assert_eq!(report.qualified_pareto_frontier_ids, vec!["a", "b"]);
        assert_eq!(report.submitted_is_strictly_dominated, Some(false));
    }

    #[test]
    fn unqualified_candidate_cannot_dominate_a_qualified_submission() {
        let packet = packet();
        let frozen = trace(SelectionDecision::Select {
            candidate_id: "a".to_owned(),
        });
        let outcomes = vec![
            real_outcome("a", 70.0, 80.0, EvaluatorAdmissibility::Qualified),
            real_outcome("b", 99.0, 99.0, EvaluatorAdmissibility::Unqualified),
            real_outcome("c", 65.0, 75.0, EvaluatorAdmissibility::Unqualified),
        ];
        let report = audit_submission(&packet, &frozen, &outcomes).unwrap();
        assert_eq!(report.submitted_is_strictly_dominated, Some(false));
        assert!(report.submitted_dominator_ids.is_empty());
        assert_eq!(report.qualified_pareto_frontier_ids, vec!["a"]);
    }

    #[test]
    fn evaluator_failures_and_missing_outcomes_stay_in_attempt_denominator() {
        let mut packet = packet();
        packet.candidates.push(candidate("d", MODEL_C));
        let frozen = trace(SelectionDecision::Select {
            candidate_id: "a".to_owned(),
        });
        let outcomes = vec![
            real_outcome("a", 70.0, 85.0, EvaluatorAdmissibility::Qualified),
            EvaluatorOutcome {
                candidate_id: "b".to_owned(),
                status: EvaluatorStatus::Failed,
                admissibility: EvaluatorAdmissibility::NotAssessed,
                metrics: EvaluatorMetrics::default(),
            },
            EvaluatorOutcome {
                candidate_id: "c".to_owned(),
                status: EvaluatorStatus::TimedOut,
                admissibility: EvaluatorAdmissibility::NotAssessed,
                metrics: EvaluatorMetrics::default(),
            },
        ];
        let report = audit_submission(&packet, &frozen, &outcomes).unwrap();
        assert_eq!(report.attempted_candidates, 4);
        assert_eq!(report.scored_candidates, 1);
        assert_eq!(report.failed_candidates, 1);
        assert_eq!(report.timed_out_candidates, 1);
        assert_eq!(report.missing_evaluator_outcomes, 1);
    }

    #[test]
    fn ready_candidates_require_exact_artifact_identity_but_failed_attempts_may_not_have_one() {
        let mut packet = packet();
        packet.candidates[0].canonical_model_sha256 = None;
        assert_eq!(
            validate_selector_packet(&packet),
            Err(SelectionAuditError::MissingModelDigest {
                candidate_id: "a".to_owned()
            })
        );

        packet.candidates[0].state = CandidateState::GenerationFailed;
        packet.candidates[0].eligibility = DevelopmentEligibility::Unresolved;
        validate_selector_packet(&packet).unwrap();
    }

    #[test]
    fn semantic_duplicates_are_retained_and_reported_as_a_group() {
        let mut packet = packet();
        packet.candidates.push(candidate("a-copy", MODEL_A));
        let frozen = trace(SelectionDecision::Select {
            candidate_id: "a".to_owned(),
        });
        let outcomes = vec![
            real_outcome("a", 70.0, 80.0, EvaluatorAdmissibility::Qualified),
            real_outcome("b", 72.0, 82.0, EvaluatorAdmissibility::Qualified),
            real_outcome("c", 68.0, 83.0, EvaluatorAdmissibility::Qualified),
        ];
        let report = audit_submission(&packet, &frozen, &outcomes).unwrap();
        assert_eq!(report.attempted_candidates, 4);
        assert_eq!(report.ready_attempts, 4);
        assert_eq!(report.distinct_model_artifact_count, 3);
        assert_eq!(report.selectable_candidate_count, 3);
        assert_eq!(report.exact_duplicate_candidate_attempts, 1);
        assert_eq!(report.scored_candidates, 3);
        assert_eq!(report.missing_evaluator_outcomes, 0);
        assert_eq!(
            report.identical_model_artifact_groups,
            vec![vec!["a".to_owned(), "a-copy".to_owned()]]
        );

        let mut reordered = packet.clone();
        reordered.candidates.reverse();
        let replay = audit_submission(&reordered, &frozen, &outcomes).unwrap();
        assert_eq!(
            replay.identical_model_artifact_groups,
            report.identical_model_artifact_groups
        );
    }

    #[test]
    fn failed_generation_attempts_remain_in_denominator_but_not_selector_view() {
        let mut packet = packet();
        let mut failed = candidate("failed-attempt", MODEL_A);
        failed.state = CandidateState::GenerationFailed;
        failed.canonical_model_sha256 = None;
        failed.eligibility = DevelopmentEligibility::Unresolved;
        packet.candidates.push(failed);

        let view = build_selector_view(&packet).unwrap();
        assert_eq!(view.candidates.len(), 3);
        let frozen = trace(SelectionDecision::Select {
            candidate_id: "a".to_owned(),
        });
        let report = audit_submission(
            &packet,
            &frozen,
            &[
                real_outcome("a", 70.0, 80.0, EvaluatorAdmissibility::Qualified),
                real_outcome("b", 75.0, 82.0, EvaluatorAdmissibility::Qualified),
                real_outcome("c", 68.0, 84.0, EvaluatorAdmissibility::Qualified),
            ],
        )
        .unwrap();
        assert_eq!(report.attempted_candidates, 4);
        assert_eq!(report.ready_attempts, 3);
        assert_eq!(report.generation_failed_attempts, 1);
        assert_eq!(report.selectable_candidate_count, 3);
        assert_eq!(report.missing_evaluator_outcomes, 0);
    }

    #[test]
    fn evaluator_outcome_for_duplicate_alias_is_rejected() {
        let mut packet = packet();
        packet.candidates.push(candidate("a-copy", MODEL_A));
        let frozen = trace(SelectionDecision::Select {
            candidate_id: "a".to_owned(),
        });
        let outcomes = vec![
            real_outcome("a", 70.0, 80.0, EvaluatorAdmissibility::Qualified),
            real_outcome("a-copy", 70.0, 80.0, EvaluatorAdmissibility::Qualified),
        ];
        assert_eq!(
            audit_submission(&packet, &frozen, &outcomes),
            Err(SelectionAuditError::UnknownEvaluatorCandidate(
                "a-copy".to_owned()
            ))
        );
    }

    #[test]
    fn no_admissible_candidate_supports_explicit_abstention() {
        let mut packet = packet();
        for candidate in &mut packet.candidates {
            candidate.eligibility = DevelopmentEligibility::Unresolved;
        }
        let frozen = trace(SelectionDecision::Abstain {
            reason: "no candidate passed declared pre-submission admissibility".to_owned(),
        });
        validate_selection_trace(&packet, &frozen).unwrap();
        let outcomes = vec![
            real_outcome("a", 70.0, 80.0, EvaluatorAdmissibility::Unqualified),
            real_outcome("b", 65.0, 82.0, EvaluatorAdmissibility::Unqualified),
            real_outcome("c", 60.0, 84.0, EvaluatorAdmissibility::NotAssessed),
        ];
        let report = audit_submission(&packet, &frozen, &outcomes).unwrap();
        assert!(report.submitted_candidate_id.is_none());
        assert_eq!(
            report.abstention_reason.as_deref(),
            Some("no candidate passed declared pre-submission admissibility")
        );
        assert!(report.metric_regret.regret_by_metric.is_none());
    }

    #[test]
    fn task_metric_profile_is_exact_and_missing_dimensions_fail_closed() {
        let mut packet = packet();
        packet.metric_mask = MetricMask::PARALLEL_FIT_AND_STRUCTURE;
        let frozen = trace(SelectionDecision::Select {
            candidate_id: "a".to_owned(),
        });
        let bad = vec![real_outcome(
            "a",
            70.0,
            80.0,
            EvaluatorAdmissibility::Qualified,
        )];
        assert!(matches!(
            audit_submission(&packet, &frozen, &bad),
            Err(SelectionAuditError::MissingMetric { .. })
                | Err(SelectionAuditError::UnexpectedMetric { .. })
        ));
    }

    #[test]
    fn duplicate_and_unknown_outcomes_fail_closed() {
        let packet = packet();
        let frozen = trace(SelectionDecision::Select {
            candidate_id: "a".to_owned(),
        });
        let duplicate = vec![
            real_outcome("a", 70.0, 80.0, EvaluatorAdmissibility::Qualified),
            real_outcome("a", 71.0, 81.0, EvaluatorAdmissibility::Qualified),
        ];
        assert_eq!(
            audit_submission(&packet, &frozen, &duplicate),
            Err(SelectionAuditError::DuplicateEvaluatorOutcome(
                "a".to_owned()
            ))
        );
        let unknown = vec![real_outcome(
            "outside-pool",
            70.0,
            80.0,
            EvaluatorAdmissibility::Qualified,
        )];
        assert_eq!(
            audit_submission(&packet, &frozen, &unknown),
            Err(SelectionAuditError::UnknownEvaluatorCandidate(
                "outside-pool".to_owned()
            ))
        );
    }

    #[test]
    fn malformed_scores_and_unavailable_candidate_selection_fail_closed() {
        let packet = packet();
        let frozen = trace(SelectionDecision::Select {
            candidate_id: "a".to_owned(),
        });
        let bad = vec![real_outcome(
            "a",
            f64::NAN,
            80.0,
            EvaluatorAdmissibility::Qualified,
        )];
        assert!(matches!(
            audit_submission(&packet, &frozen, &bad),
            Err(SelectionAuditError::NonFiniteMetric { .. })
        ));

        let mut bad_digest = packet.clone();
        bad_digest.candidates[0].canonical_model_sha256 = Some("not-a-sha256".to_owned());
        assert!(matches!(
            validate_selector_packet(&bad_digest),
            Err(SelectionAuditError::InvalidModelDigest { .. })
        ));

        let mut not_ready = packet.clone();
        not_ready.candidates[0].state = CandidateState::TimedOut;
        assert_eq!(
            validate_selection_trace(&not_ready, &frozen),
            Err(SelectionAuditError::SelectedCandidateUnavailable(
                "a".to_owned()
            ))
        );

        let mut ineligible = packet.clone();
        ineligible.candidates[0].eligibility = DevelopmentEligibility::Unresolved;
        assert_eq!(
            validate_selection_trace(&ineligible, &frozen),
            Err(SelectionAuditError::SelectedCandidateIneligible(
                "a".to_owned()
            ))
        );
    }

    #[test]
    fn json_receipt_preserves_axes_and_attempt_denominator() {
        let packet = packet();
        let frozen = trace(SelectionDecision::Select {
            candidate_id: "a".to_owned(),
        });
        let outcomes = vec![
            real_outcome("a", 70.0, 85.0, EvaluatorAdmissibility::Qualified),
            real_outcome("b", 75.0, 90.0, EvaluatorAdmissibility::Qualified),
            real_outcome("c", 80.0, 82.0, EvaluatorAdmissibility::Qualified),
        ];
        let json = audit_submission(&packet, &frozen, &outcomes)
            .unwrap()
            .to_json();
        assert!(json.contains("\"attempted_candidates\":3"));
        assert!(json.contains("\"submitted_is_strictly_dominated\":true"));
        assert!(json.contains("\"numeric_fit\":10.00000000"));
        assert!(json.contains("\"scientific_validity\":5.00000000"));
        assert!(!json.contains("hidden_test_predictions"));
    }
}
