//! Evidence-first alternatives assessment.
//!
//! This crate deliberately avoids a single "green score". Burden dimensions
//! remain separate, hard constraints fail closed, Pareto dominance is
//! conservative over uncertainty intervals, and qualification cannot exceed
//! what the linked evidence demonstrates.
//!
//! Intended composition:
//!
//! FunctionalRequirement -> CandidatePathway[] -> EvidenceBundle[]
//! -> ConstraintEvaluation -> ParetoFrontier -> QualificationState
//! -> AssessmentReceipt
//!
//! No type in this crate authorizes physical execution.

#![deny(unsafe_code)]
#![warn(missing_docs)]

use blake3::Hasher;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Reusable adversarial benchmark scenarios.
pub mod corpus;

/// Serialized assessment schema version.
pub const SCHEMA_VERSION: u16 = 1;
/// Assessment algorithm version.
pub const ALGORITHM_VERSION: &str = "pareto-interval-v1";

/// A burden dimension. Lower values are better for every dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Dimension {
    /// Intrinsic human/ecological hazard burden.
    Hazard,
    /// Human/ecological exposure burden.
    Exposure,
    /// Climate/carbon burden.
    Carbon,
    /// Water burden.
    Water,
    /// Energy burden.
    Energy,
    /// Critical/virgin-material burden.
    CriticalMaterial,
    /// Waste/disposal burden.
    Waste,
    /// End-of-life/circularity burden; lower means more circular.
    CircularityBurden,
    /// Worker safety burden.
    WorkerSafety,
    /// Economic cost burden.
    Cost,
    /// Manufacturing difficulty/capability burden.
    Manufacturability,
    /// Supply-chain fragility burden.
    SupplyChainFragility,
}

impl Dimension {
    /// All dimensions in deterministic order.
    pub const ALL: [Self; 12] = [
        Self::Hazard,
        Self::Exposure,
        Self::Carbon,
        Self::Water,
        Self::Energy,
        Self::CriticalMaterial,
        Self::Waste,
        Self::CircularityBurden,
        Self::WorkerSafety,
        Self::Cost,
        Self::Manufacturability,
        Self::SupplyChainFragility,
    ];
}

/// An uncertainty interval for a burden value.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Interval {
    /// Lower plausible bound.
    pub lower: f64,
    /// Upper plausible bound.
    pub upper: f64,
}

impl Interval {
    /// Construct a validated interval.
    pub fn new(lower: f64, upper: f64) -> Result<Self, AssessmentError> {
        if !lower.is_finite() || !upper.is_finite() {
            return Err(AssessmentError::NonFinite);
        }
        if lower > upper {
            return Err(AssessmentError::InvalidInterval { lower, upper });
        }
        Ok(Self { lower, upper })
    }

    /// Construct a point estimate.
    pub fn point(value: f64) -> Result<Self, AssessmentError> {
        Self::new(value, value)
    }

    /// Midpoint of the interval.
    pub fn midpoint(self) -> f64 {
        (self.lower + self.upper) / 2.0
    }

    /// Width of the uncertainty interval.
    pub fn width(self) -> f64 {
        self.upper - self.lower
    }

    fn clearly_better_than(self, other: Self) -> bool {
        self.upper < other.lower
    }

    fn clearly_no_worse_than(self, other: Self) -> bool {
        self.upper <= other.lower
    }

    fn clearly_worse_than(self, other: Self) -> bool {
        self.lower > other.upper
    }
}

/// The type of evidence behind an assertion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceKind {
    /// Direct physical or operational observation.
    Observed,
    /// Value reported by an external source.
    Reported,
    /// Simulation/model output.
    Simulated,
    /// Deterministically derived value.
    Derived,
    /// Explicit lifecycle-assessment evidence covering the declared scope.
    LifecycleAssessed,
    /// Unverified conjecture.
    Hypothesis,
    /// Observation from manufacturing-scale operation.
    ManufacturingObserved,
    /// Observation from field deployment.
    FieldObserved,
    /// Repeated operational monitoring.
    ContinuouslyMonitored,
}

/// Whether evidence supports or contradicts the linked assertion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceStance {
    /// Evidence supports the assertion.
    Supports,
    /// Evidence contradicts the assertion.
    Contradicts,
}

/// Provenance-aware evidence metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvidenceRecord {
    /// Stable evidence identifier.
    pub id: String,
    /// Evidence classification.
    pub kind: EvidenceKind,
    /// Support or contradiction.
    pub stance: EvidenceStance,
    /// Caller-supplied confidence in [0, 1].
    pub confidence: f64,
    /// Opaque source identifier for independence checks.
    pub source_id: String,
    /// Human-readable scope: functional unit, geography, process, etc.
    pub scope: String,
    /// Optional unit for the associated quantity.
    pub unit: Option<String>,
    /// Optional source timestamp/version label.
    pub as_of: Option<String>,
}

impl EvidenceRecord {
    /// Validate identity and confidence.
    pub fn validate(&self) -> Result<(), AssessmentError> {
        if !self.confidence.is_finite() || !(0.0..=1.0).contains(&self.confidence) {
            return Err(AssessmentError::InvalidConfidence(self.confidence));
        }
        if self.id.is_empty() || self.source_id.is_empty() || self.scope.is_empty() {
            return Err(AssessmentError::EmptyEvidenceIdentity);
        }
        Ok(())
    }
}

/// A burden estimate linked to the evidence that supports or contradicts it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BurdenEstimate {
    /// Plausible burden interval.
    pub interval: Interval,
    /// Unit of measure used for cross-candidate comparison.
    pub unit: String,
    /// Scope in which this value is comparable: geography, functional unit,
    /// lifecycle boundary, process boundary, time basis, etc.
    pub scope: String,
    /// Evidence IDs that specifically bear on this dimension.
    pub evidence_ids: Vec<String>,
}

/// A bound on a functional requirement.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum RequirementBound {
    /// Performance must be at least this value.
    AtLeast(f64),
    /// Performance must be at most this value.
    AtMost(f64),
    /// Performance must lie in this inclusive range.
    Between { min: f64, max: f64 },
}

impl RequirementBound {
    fn check(self, value: Option<f64>) -> ConstraintStatus {
        let Some(value) = value else {
            return ConstraintStatus::Unresolved;
        };
        if !value.is_finite() {
            return ConstraintStatus::Unresolved;
        }
        let pass = match self {
            Self::AtLeast(min) => value >= min,
            Self::AtMost(max) => value <= max,
            Self::Between { min, max } => value >= min && value <= max,
        };
        if pass {
            ConstraintStatus::Pass
        } else {
            ConstraintStatus::Fail
        }
    }
}

/// The explicit comparison scale for one burden dimension.
///
/// The engine never infers a comparison cohort's scale from the candidates.
/// This prevents a mutually inconsistent set of candidate units/scopes from
/// silently becoming its own reference frame.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComparisonScale {
    /// Unit shared by all comparable candidates for the dimension.
    pub unit: String,
    /// Functional-unit / lifecycle / geography / temporal scope identifier.
    pub scope: String,
}

impl ComparisonScale {
    /// Validate that the comparison scale is explicit and non-empty.
    pub fn validate(&self) -> Result<(), AssessmentError> {
        if self.unit.is_empty() || self.scope.is_empty() {
            return Err(AssessmentError::EmptyBurdenScale);
        }
        Ok(())
    }
}

/// The function that must be satisfied independently of the incumbent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionalRequirement {
    /// Stable requirement identifier.
    pub id: String,
    /// Human-readable description.
    pub description: String,
    /// Named performance constraints.
    pub constraints: BTreeMap<String, RequirementBound>,
    /// Explicit comparison scales for every burden dimension.
    pub comparison_scales: BTreeMap<Dimension, ComparisonScale>,
}

impl FunctionalRequirement {
    /// Validate identity, numeric bounds, and explicit comparison scales.
    pub fn validate(&self) -> Result<(), AssessmentError> {
        if self.id.is_empty() || self.description.is_empty() {
            return Err(AssessmentError::EmptyRequirementIdentity);
        }
        for bound in self.constraints.values() {
            match bound {
                RequirementBound::AtLeast(v) | RequirementBound::AtMost(v) => {
                    if !v.is_finite() {
                        return Err(AssessmentError::NonFinite);
                    }
                }
                RequirementBound::Between { min, max } => {
                    if !min.is_finite() || !max.is_finite() || min > max {
                        return Err(AssessmentError::InvalidRequirementRange {
                            min: *min,
                            max: *max,
                        });
                    }
                }
            }
        }
        for dimension in Dimension::ALL {
            let Some(scale) = self.comparison_scales.get(&dimension) else {
                return Err(AssessmentError::MissingComparisonScale(dimension));
            };
            scale.validate()?;
        }
        Ok(())
    }
}

/// How a candidate changes the incumbent pathway.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PathwayKind {
    /// Material drop-in or close substitution.
    MaterialSubstitution,
    /// Different process providing the same function.
    ProcessSubstitution,
    /// Product/system redesign.
    ProductRedesign,
    /// Elimination of the material or process.
    Elimination,
    /// Sourcing or logistics redesign.
    SupplyChainRedesign,
    /// Reuse/remanufacturing pathway.
    ReuseRemanufacture,
}

/// One candidate solution pathway.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidatePathway {
    /// Stable candidate identifier.
    pub id: String,
    /// Human-readable candidate name.
    pub name: String,
    /// Candidate pathway class.
    pub kind: PathwayKind,
    /// Performance values keyed by requirement metric.
    pub performance: BTreeMap<String, f64>,
    /// Burden estimates by dimension.
    pub burdens: BTreeMap<Dimension, BurdenEstimate>,
    /// Candidate-level evidence bundle.
    pub evidence: Vec<EvidenceRecord>,
}

impl CandidatePathway {
    /// Validate the candidate, burden intervals, evidence references, and
    /// evidence records.
    pub fn validate(&self) -> Result<(), AssessmentError> {
        if self.id.is_empty() || self.name.is_empty() {
            return Err(AssessmentError::EmptyCandidateIdentity);
        }
        if self.burdens.is_empty() {
            return Err(AssessmentError::NoBurdenData);
        }
        for value in self.performance.values() {
            if !value.is_finite() {
                return Err(AssessmentError::NonFinite);
            }
        }
        let evidence_ids = self
            .evidence
            .iter()
            .map(|e| e.id.as_str())
            .collect::<BTreeSet<_>>();
        for estimate in self.burdens.values() {
            Interval::new(estimate.interval.lower, estimate.interval.upper)?;
            if estimate.unit.is_empty() || estimate.scope.is_empty() {
                return Err(AssessmentError::EmptyBurdenScale);
            }
            for evidence_id in &estimate.evidence_ids {
                let Some(evidence) = self.evidence.iter().find(|e| e.id == *evidence_id) else {
                    if !evidence_ids.contains(evidence_id.as_str()) {
                        return Err(AssessmentError::MissingEvidenceReference(
                            evidence_id.clone(),
                        ));
                    }
                    continue;
                };
                if evidence.scope != estimate.scope {
                    return Err(AssessmentError::EvidenceScopeMismatch {
                        evidence_id: evidence.id.clone(),
                        burden_scope: estimate.scope.clone(),
                        evidence_scope: evidence.scope.clone(),
                    });
                }
                if let Some(evidence_unit) = &evidence.unit
                    && evidence_unit != &estimate.unit
                {
                    return Err(AssessmentError::EvidenceUnitMismatch {
                        evidence_id: evidence.id.clone(),
                        burden_unit: estimate.unit.clone(),
                        evidence_unit: evidence_unit.clone(),
                    });
                }
            }
        }
        for evidence in &self.evidence {
            evidence.validate()?;
        }
        Ok(())
    }

    fn linked_evidence<'a>(
        &'a self,
        estimate: &'a BurdenEstimate,
    ) -> impl Iterator<Item = &'a EvidenceRecord> {
        let ids = estimate
            .evidence_ids
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        self.evidence.iter().filter(move |e| ids.contains(e.id.as_str()))
    }

    fn dimension_has_conflict(&self, estimate: &BurdenEstimate) -> bool {
        let support = self
            .linked_evidence(estimate)
            .any(|e| e.stance == EvidenceStance::Supports);
        let contradict = self
            .linked_evidence(estimate)
            .any(|e| e.stance == EvidenceStance::Contradicts);
        support && contradict
    }

    fn has_conflict(&self) -> bool {
        self.burdens
            .values()
            .any(|estimate| self.dimension_has_conflict(estimate))
    }

    fn qualification_ceiling(&self) -> QualificationState {
        if self.burdens.is_empty()
            || self.burdens.values().all(|estimate| {
                let linked = self.linked_evidence(estimate).collect::<Vec<_>>();
                linked.is_empty()
                    || linked
                        .iter()
                        .all(|e| e.kind == EvidenceKind::Hypothesis)
            })
        {
            return QualificationState::Hypothesis;
        }

        if self.has_conflict() {
            return QualificationState::ComputationallyPlausible;
        }

        let any_simulation = self.burdens.values().any(|estimate| {
            self.linked_evidence(estimate)
                .any(|e| e.kind == EvidenceKind::Simulated)
        });
        let any_supported_measurement = self.burdens.values().any(|estimate| {
            self.linked_evidence(estimate).any(|e| {
                matches!(
                    e.kind,
                    EvidenceKind::Observed | EvidenceKind::Reported | EvidenceKind::Derived
                ) && e.stance == EvidenceStance::Supports
                    && e.confidence >= 0.7
            })
        });
        let independent_sources = self
            .burdens
            .values()
            .flat_map(|estimate| self.linked_evidence(estimate))
            .filter(|e| {
                matches!(
                    e.kind,
                    EvidenceKind::Observed
                        | EvidenceKind::Reported
                        | EvidenceKind::Derived
                        | EvidenceKind::LifecycleAssessed
                        | EvidenceKind::ManufacturingObserved
                        | EvidenceKind::FieldObserved
                        | EvidenceKind::ContinuouslyMonitored
                ) && e.stance == EvidenceStance::Supports
                    && e.confidence >= 0.7
            })
            .map(|e| e.source_id.as_str())
            .collect::<BTreeSet<_>>()
            .len();
        let has_all_dimension_evidence = Dimension::ALL.iter().all(|dimension| {
            self.burdens
                .get(dimension)
                .map(|estimate| {
                    self.linked_evidence(estimate).any(|e| {
                        matches!(
                            e.kind,
                            EvidenceKind::Observed
                                | EvidenceKind::Reported
                                | EvidenceKind::Derived
                                | EvidenceKind::LifecycleAssessed
                                | EvidenceKind::ManufacturingObserved
                                | EvidenceKind::FieldObserved
                                | EvidenceKind::ContinuouslyMonitored
                        ) && e.stance == EvidenceStance::Supports
                            && e.confidence >= 0.7
                    })
                })
                .unwrap_or(false)
        });
        let has_lifecycle_assessment = self.burdens.values().any(|estimate| {
            self.linked_evidence(estimate).any(|e| {
                e.kind == EvidenceKind::LifecycleAssessed
                    && e.stance == EvidenceStance::Supports
                    && e.confidence >= 0.7
            })
        });
        let all_dimensions_field_observed = Dimension::ALL.iter().all(|dimension| {
            self.burdens
                .get(dimension)
                .map(|estimate| {
                    self.linked_evidence(estimate).any(|e| {
                        e.kind == EvidenceKind::FieldObserved
                            && e.stance == EvidenceStance::Supports
                            && e.confidence >= 0.7
                    })
                })
                .unwrap_or(false)
        });
        let all_dimensions_monitored = Dimension::ALL.iter().all(|dimension| {
            self.burdens
                .get(dimension)
                .map(|estimate| {
                    self.linked_evidence(estimate).any(|e| {
                        e.kind == EvidenceKind::ContinuouslyMonitored
                            && e.stance == EvidenceStance::Supports
                            && e.confidence >= 0.7
                    })
                })
                .unwrap_or(false)
        });
        let has_manufacturing_observation = self.burdens.values().any(|estimate| {
            self.linked_evidence(estimate).any(|e| {
                e.kind == EvidenceKind::ManufacturingObserved
                    && e.stance == EvidenceStance::Supports
                    && e.confidence >= 0.7
            })
        });

        if all_dimensions_monitored {
            QualificationState::ContinuouslyMonitored
        } else if all_dimensions_field_observed {
            QualificationState::FieldQualified
        } else if any_supported_measurement
            && independent_sources >= 2
            && has_all_dimension_evidence
            && has_manufacturing_observation
        {
            QualificationState::ManufacturingQualified
        } else if any_supported_measurement
            && independent_sources >= 2
            && has_all_dimension_evidence
            && has_lifecycle_assessment
        {
            QualificationState::LifecycleQualified
        } else if any_supported_measurement {
            QualificationState::EvidenceSupported
        } else if any_simulation {
            QualificationState::ComputationallyPlausible
        } else {
            QualificationState::Hypothesis
        }
    }
}

/// Qualification ceiling derived only from supplied evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum QualificationState {
    /// Only a proposed hypothesis exists.
    Hypothesis,
    /// Computational evidence exists.
    ComputationallyPlausible,
    /// At least one substantive empirical/reporting claim is supported.
    EvidenceSupported,
    /// Multiple independent supported sources exist.
    LifecycleQualified,
    /// Multiple independent sources plus full dimension coverage exist.
    ManufacturingQualified,
    /// Field deployment has been observed.
    FieldQualified,
    /// Post-deployment monitoring is active.
    ContinuouslyMonitored,
}

/// Functional constraint status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConstraintStatus {
    /// Requirement is satisfied.
    Pass,
    /// Requirement is violated.
    Fail,
    /// No trustworthy value is available.
    Unresolved,
}

/// Result for one named functional constraint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConstraintEvaluation {
    /// Metric name.
    pub metric: String,
    /// Required bound.
    pub requirement: RequirementBound,
    /// Evaluation status.
    pub status: ConstraintStatus,
}

/// Assessment of one candidate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateAssessment {
    /// Candidate identifier.
    pub candidate_id: String,
    /// Per-constraint outcomes.
    pub constraints: Vec<ConstraintEvaluation>,
    /// Burdens with dimension-specific evidence linkage.
    pub burdens: BTreeMap<Dimension, BurdenEstimate>,
    /// Conservative qualification ceiling.
    pub qualification: QualificationState,
    /// Whether linked evidence contains explicit contradiction.
    pub evidence_conflict: bool,
    /// Whether blocked from Pareto comparison.
    pub frontier_blocked: bool,
    /// Number of directly observed/field-observed/monitored items per dimension.
    pub observed_evidence_count: BTreeMap<Dimension, usize>,
}

/// Why a candidate is blocked from the frontier.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FrontierBlocker {
    /// A hard functional constraint failed.
    ConstraintFailed(String),
    /// A hard functional constraint is unresolved.
    ConstraintUnresolved(String),
    /// A burden dimension is missing.
    MissingDimension(Dimension),
    /// Candidate and comparison cohort use different units or scopes.
    IncompatibleScale {
        /// Dimension whose comparison scale differs.
        dimension: Dimension,
        /// Expected comparison unit.
        expected_unit: String,
        /// Candidate unit.
        actual_unit: String,
        /// Expected comparison scope.
        expected_scope: String,
        /// Candidate scope.
        actual_scope: String,
    },
}

/// Candidate-versus-incumbent burden transfer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BurdenTransfer {
    /// Candidate identifier.
    pub candidate_id: String,
    /// Dimensions where the candidate is clearly better.
    pub clearly_better: Vec<Dimension>,
    /// Dimensions where the candidate is clearly worse.
    pub clearly_worse: Vec<Dimension>,
}

impl BurdenTransfer {
    /// True when benefits and harms move across different dimensions.
    pub fn is_regrettable_substitution(&self) -> bool {
        !self.clearly_better.is_empty() && !self.clearly_worse.is_empty()
    }
}

/// Conservative next-measurement target.
///
/// This is explicitly a heuristic rather than a formal expected-value-of-
—information calculation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeasurementPriority {
    /// Dimension to investigate next.
    pub dimension: Dimension,
    /// Number of frontier candidates lacking direct observed evidence for this dimension.
    pub unresolved_candidate_count: usize,
    /// Number of candidates on the current frontier.
    pub frontier_candidate_count: usize,
    /// Rationale.
    pub rationale: String,
}

/// Complete deterministic assessment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssessmentResult {
    /// Schema version.
    pub schema_version: u16,
    /// Algorithm version.
    pub algorithm_version: String,
    /// Functional requirement.
    pub requirement: FunctionalRequirement,
    /// Candidate assessments.
    pub candidates: Vec<CandidateAssessment>,
    /// Candidate IDs on the conservative Pareto frontier.
    ///
    /// Frontier membership is a comparison result, not a recommendation or
    /// authorization to deploy a candidate.
    pub pareto_frontier: Vec<String>,
    /// Candidate comparisons against the incumbent.
    pub burden_transfers: Vec<BurdenTransfer>,
    /// Candidate blockers.
    pub frontier_blockers: BTreeMap<String, Vec<FrontierBlocker>>,
    /// Heuristic next-measurement target.
    pub next_measurement: Option<MeasurementPriority>,
    /// Deterministic receipt.
    pub receipt: AssessmentReceipt,
}

/// Deterministic integrity receipt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssessmentReceipt {
    /// Schema version included in the hash.
    pub schema_version: u16,
    /// Algorithm version included in the hash.
    pub algorithm_version: String,
    /// BLAKE3 digest over the receipt-free canonical payload.
    pub payload_hash: String,
}

/// Assessment construction errors.
#[derive(Debug, Clone, PartialEq)]
pub enum AssessmentError {
    /// Encountered a non-finite number.
    NonFinite,
    /// Interval bounds were inverted.
    InvalidInterval { lower: f64, upper: f64 },
    /// Confidence was outside [0, 1].
    InvalidConfidence(f64),
    /// Evidence identity metadata is incomplete.
    EmptyEvidenceIdentity,
    /// Requirement identity is incomplete.
    EmptyRequirementIdentity,
    /// Candidate identity is incomplete.
    EmptyCandidateIdentity,
    /// Candidate has no burden dimensions.
    NoBurdenData,
    /// A burden estimate lacks a comparable unit or scope.
    EmptyBurdenScale,
    /// The requirement does not declare a comparison scale for a dimension.
    MissingComparisonScale(Dimension),
    /// Linked evidence uses a different scope from the burden estimate.
    EvidenceScopeMismatch {
        /// Evidence identifier.
        evidence_id: String,
        /// Burden comparison scope.
        burden_scope: String,
        /// Evidence scope.
        evidence_scope: String,
    },
    /// Linked evidence uses a different unit from the burden estimate.
    EvidenceUnitMismatch {
        /// Evidence identifier.
        evidence_id: String,
        /// Burden comparison unit.
        burden_unit: String,
        /// Evidence unit.
        evidence_unit: String,
    },
    /// Requirement range is invalid.
    InvalidRequirementRange { min: f64, max: f64 },
    /// A burden references unknown evidence.
    MissingEvidenceReference(String),
    /// Requested incumbent does not exist.
    MissingIncumbent(String),
    /// Two candidates have the same stable identifier.
    DuplicateCandidateId(String),
}

impl std::fmt::Display for AssessmentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFinite => write!(f, "non-finite numeric value"),
            Self::InvalidInterval { lower, upper } => {
                write!(f, "invalid interval [{lower}, {upper}]")
            }
            Self::InvalidConfidence(value) => write!(f, "invalid confidence {value}"),
            Self::EmptyEvidenceIdentity => write!(f, "evidence identity is incomplete"),
            Self::EmptyRequirementIdentity => write!(f, "requirement identity is incomplete"),
            Self::EmptyCandidateIdentity => write!(f, "candidate identity is incomplete"),
            Self::NoBurdenData => write!(f, "candidate has no burden data"),
            Self::EmptyBurdenScale => write!(f, "burden unit/scope is empty"),
            Self::MissingComparisonScale(dimension) => {
                write!(f, "missing comparison scale for {dimension:?}")
            }
            Self::EvidenceScopeMismatch {
                evidence_id,
                burden_scope,
                evidence_scope,
            } => write!(
                f,
                "evidence {evidence_id} scope {evidence_scope} does not match burden scope {burden_scope}"
            ),
            Self::EvidenceUnitMismatch {
                evidence_id,
                burden_unit,
                evidence_unit,
            } => write!(
                f,
                "evidence {evidence_id} unit {evidence_unit} does not match burden unit {burden_unit}"
            ),
            Self::InvalidRequirementRange { min, max } => {
                write!(f, "invalid requirement range [{min}, {max}]")
            }
            Self::MissingEvidenceReference(id) => {
                write!(f, "missing evidence reference {id}")
            }
            Self::MissingIncumbent(id) => write!(f, "incumbent {id} not found"),
            Self::DuplicateCandidateId(id) => write!(f, "duplicate candidate id {id}"),
        }
    }
}

impl std::error::Error for AssessmentError {}

/// Evidence-first alternatives assessment engine.
#[derive(Debug, Default, Clone, Copy)]
pub struct AlternativesEngine;

impl AlternativesEngine {
    /// Evaluate candidates against a functional requirement.
    pub fn assess(
        &self,
        requirement: &FunctionalRequirement,
        candidates: &[CandidatePathway],
        incumbent_id: Option<&str>,
    ) -> Result<AssessmentResult, AssessmentError> {
        requirement.validate()?;
        if let Some(id) = incumbent_id {
            if !candidates.iter().any(|candidate| candidate.id == id) {
                return Err(AssessmentError::MissingIncumbent(id.to_string()));
            }
        }

        let mut normalized_candidates = candidates.to_vec();
        normalized_candidates.sort_by(|a, b| a.id.cmp(&b.id));
        if normalized_candidates
            .windows(2)
            .any(|pair| pair[0].id == pair[1].id)
        {
            return Err(AssessmentError::DuplicateCandidateId(
                normalized_candidates
                    .first()
                    .map(|candidate| candidate.id.clone())
                    .unwrap_or_default(),
            ));
        }
        for candidate in &mut normalized_candidates {
            candidate
                .evidence
                .sort_by(|a, b| a.id.cmp(&b.id));
            for estimate in candidate.burdens.values_mut() {
                estimate.evidence_ids.sort();
            }
            candidate.validate()?;
        }

        let mut assessments = Vec::with_capacity(normalized_candidates.len());
        let mut blockers = BTreeMap::new();
        let expected_scales = requirement
            .comparison_scales
            .iter()
            .map(|(dimension, scale)| (dimension, (scale.unit.clone(), scale.scope.clone())))
            .collect::<BTreeMap<_, _>>();

        for candidate in &normalized_candidates {
            let constraints = requirement
                .constraints
                .iter()
                .map(|(metric, bound)| ConstraintEvaluation {
                    metric: metric.clone(),
                    requirement: *bound,
                    status: bound.check(candidate.performance.get(metric).copied()),
                })
                .collect::<Vec<_>>();

            let mut candidate_blockers = Vec::new();
            for evaluation in &constraints {
                match evaluation.status {
                    ConstraintStatus::Fail => candidate_blockers
                        .push(FrontierBlocker::ConstraintFailed(evaluation.metric.clone())),
                    ConstraintStatus::Unresolved => candidate_blockers
                        .push(FrontierBlocker::ConstraintUnresolved(evaluation.metric.clone())),
                    ConstraintStatus::Pass => {}
                }
            }
            for dimension in Dimension::ALL {
                match (candidate.burdens.get(&dimension), expected_scales.get(&dimension)) {
                    (None, _) => candidate_blockers.push(FrontierBlocker::MissingDimension(dimension)),
                    (Some(estimate), Some((expected_unit, expected_scope)))
                        if estimate.unit != *expected_unit || estimate.scope != *expected_scope =>
                    {
                        candidate_blockers.push(FrontierBlocker::IncompatibleScale {
                            dimension,
                            expected_unit: expected_unit.clone(),
                            actual_unit: estimate.unit.clone(),
                            expected_scope: expected_scope.clone(),
                            actual_scope: estimate.scope.clone(),
                        });
                    }
                    _ => {}
                }
            }

            let frontier_blocked = !candidate_blockers.is_empty();
            if frontier_blocked {
                blockers.insert(candidate.id.clone(), candidate_blockers);
            }

            let observed_evidence_count = Dimension::ALL
                .into_iter()
                .map(|dimension| {
                    let count = candidate
                        .burdens
                        .get(&dimension)
                        .map(|estimate| {
                            candidate
                                .linked_evidence(estimate)
                                .filter(|e| {
                                    matches!(
                                        e.kind,
                                        EvidenceKind::Observed
                                            | EvidenceKind::FieldObserved
                                            | EvidenceKind::ContinuouslyMonitored
                                    )
                                })
                                .count()
                        })
                        .unwrap_or(0);
                    (dimension, count)
                })
                .collect();

            assessments.push(CandidateAssessment {
                candidate_id: candidate.id.clone(),
                constraints,
                burdens: candidate.burdens.clone(),
                qualification: candidate.qualification_ceiling(),
                evidence_conflict: candidate.has_conflict(),
                frontier_blocked,
                observed_evidence_count,
            });
        }

        let eligible = assessments
            .iter()
            .filter(|assessment| !assessment.frontier_blocked)
            .collect::<Vec<_>>();

        let mut frontier = Vec::new();
        for candidate in &eligible {
            let dominated = eligible.iter().any(|other| {
                other.candidate_id != candidate.candidate_id
                    && Self::dominates(&other.burdens, &candidate.burdens)
            });
            if !dominated {
                frontier.push(candidate.candidate_id.clone());
            }
        }
        frontier.sort();

        let burden_transfers = if let Some(incumbent_id) = incumbent_id {
            let incumbent = normalized_candidates
                .iter()
                .find(|candidate| candidate.id == incumbent_id)
                .expect("validated incumbent exists");
            normalized_candidates
                .iter()
                .filter(|candidate| candidate.id != incumbent_id)
                .map(|candidate| Self::burden_transfer(candidate, incumbent))
                .collect()
        } else {
            Vec::new()
        };

        let next_measurement = Self::next_measurement(&frontier, &assessments);

        let mut result = AssessmentResult {
            schema_version: SCHEMA_VERSION,
            algorithm_version: ALGORITHM_VERSION.to_string(),
            requirement: requirement.clone(),
            candidates: assessments,
            pareto_frontier: frontier,
            burden_transfers,
            frontier_blockers: blockers,
            next_measurement,
            receipt: AssessmentReceipt {
                schema_version: SCHEMA_VERSION,
                algorithm_version: ALGORITHM_VERSION.to_string(),
                payload_hash: String::new(),
            },
        };

        result.receipt.payload_hash = canonical_payload_hash(&result)?;
        Ok(result)
    }

    /// Conservative interval Pareto dominance.
    pub fn dominates(
        a: &BTreeMap<Dimension, BurdenEstimate>,
        b: &BTreeMap<Dimension, BurdenEstimate>,
    ) -> bool {
        if !Dimension::ALL
            .iter()
            .all(|dimension| a.contains_key(dimension) && b.contains_key(dimension))
        {
            return false;
        }

        let mut strict = false;
        for dimension in Dimension::ALL {
            let a_estimate = &a[&dimension];
            let b_estimate = &b[&dimension];
            if a_estimate.unit != b_estimate.unit || a_estimate.scope != b_estimate.scope {
                return false;
            }
            let ai = a_estimate.interval;
            let bi = b_estimate.interval;
            if !ai.clearly_no_worse_than(bi) {
                return false;
            }
            if ai.clearly_better_than(bi) {
                strict = true;
            }
        }
        strict
    }

    fn burden_transfer(
        candidate: &CandidatePathway,
        incumbent: &CandidatePathway,
    ) -> BurdenTransfer {
        let mut clearly_better = Vec::new();
        let mut clearly_worse = Vec::new();
        for dimension in Dimension::ALL {
            let Some(candidate_interval) = candidate.burdens.get(&dimension).map(|b| b.interval)
            else {
                continue;
            };
            let Some(incumbent_interval) = incumbent.burdens.get(&dimension).map(|b| b.interval)
            else {
                continue;
            };
            let candidate_scale = candidate.burdens.get(&dimension).unwrap();
            let incumbent_scale = incumbent.burdens.get(&dimension).unwrap();
            if candidate_scale.unit != incumbent_scale.unit
                || candidate_scale.scope != incumbent_scale.scope
            {
                continue;
            }
            if candidate_interval.clearly_better_than(incumbent_interval) {
                clearly_better.push(dimension);
            } else if candidate_interval.clearly_worse_than(incumbent_interval) {
                clearly_worse.push(dimension);
            }
        }
        BurdenTransfer {
            candidate_id: candidate.id.clone(),
            clearly_better,
            clearly_worse,
        }
    }

    fn next_measurement(
        frontier: &[String],
        assessments: &[CandidateAssessment],
    ) -> Option<MeasurementPriority> {
        if frontier.is_empty() {
            return None;
        }

        let frontier_assessments = assessments
            .iter()
            .filter(|candidate| frontier.contains(&candidate.candidate_id))
            .collect::<Vec<_>>();

        let mut ranked = Dimension::ALL
            .iter()
            .filter_map(|dimension| {
                let unresolved_count = frontier_assessments
                    .iter()
                    .filter(|candidate| {
                        candidate.observed_evidence_count[dimension] == 0
                            || candidate.evidence_conflict
                    })
                    .count();
                (unresolved_count > 0).then_some((*dimension, unresolved_count))
            })
            .collect::<Vec<_>>();

        ranked.sort_by(|a, b| {
            b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0))
        });

        ranked.first().map(|(dimension, unresolved_count)| MeasurementPriority {
            dimension: *dimension,
            unresolved_candidate_count: *unresolved_count,
            frontier_candidate_count: frontier_assessments.len(),
            rationale: "heuristic: largest count of unresolved frontier candidates for one dimension; no cross-dimension unit scalarization".to_string(),
        })
    }
}

fn canonical_payload_hash(result: &AssessmentResult) -> Result<String, AssessmentError> {
    let mut payload = result.clone();
    payload.receipt.payload_hash.clear();
    let bytes = serde_json::to_vec(&payload).map_err(|_| AssessmentError::NonFinite)?;
    let mut hasher = Hasher::new();
    hasher.update(&bytes);
    Ok(hasher.finalize().to_hex().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_burdens(base: f64, evidence_ids: &[&str]) -> BTreeMap<Dimension, BurdenEstimate> {
        Dimension::ALL
            .into_iter()
            .map(|dimension| {
                (
                    dimension,
                    BurdenEstimate {
                        interval: Interval::point(base).unwrap(),
                        unit: "unit".into(),
                        scope: "synthetic-global-v1".into(),
                        evidence_ids: evidence_ids.iter().map(|id| (*id).to_string()).collect(),
                    },
                )
            })
            .collect()
    }

    fn evidence(
        id: &str,
        source_id: &str,
        kind: EvidenceKind,
        stance: EvidenceStance,
        confidence: f64,
    ) -> EvidenceRecord {
        EvidenceRecord {
            id: id.into(),
            kind,
            stance,
            confidence,
            source_id: source_id.into(),
            scope: "synthetic functional unit".into(),
            unit: Some("unit".into()),
            as_of: Some("fixture-v1".into()),
        }
    }

    fn fixture_requirement() -> FunctionalRequirement {
        FunctionalRequirement {
            id: "seal-v1".into(),
            description: "Provide a durable chemical-resistant seal.".into(),
            constraints: BTreeMap::from([
                ("service_life_years".into(), RequirementBound::AtLeast(10.0)),
                ("throughput_per_hour".into(), RequirementBound::AtLeast(100.0)),
            ]),
        }
    }

    fn candidate(
        id: &str,
        kind: PathwayKind,
        hazard: f64,
        water: f64,
        evidence: Vec<EvidenceRecord>,
    ) -> CandidatePathway {
        let evidence_ids = evidence.iter().map(|e| e.id.as_str()).collect::<Vec<_>>();
        let mut performance = BTreeMap::new();
        performance.insert("service_life_years".into(), 12.0);
        performance.insert("throughput_per_hour".into(), 120.0);
        let mut burdens = all_burdens(5.0, &evidence_ids);
        burdens.insert(
            Dimension::Hazard,
            BurdenEstimate {
                interval: Interval::point(hazard).unwrap(),
                unit: "unit".into(),
                scope: "synthetic-global-v1".into(),
                evidence_ids: evidence_ids.iter().map(|id| (*id).to_string()).collect(),
            },
        );
        burdens.insert(
            Dimension::Water,
            BurdenEstimate {
                interval: Interval::point(water).unwrap(),
                unit: "unit".into(),
                scope: "synthetic-global-v1".into(),
                evidence_ids: evidence_ids.iter().map(|id| (*id).to_string()).collect(),
            },
        );
        CandidatePathway {
            id: id.into(),
            name: id.into(),
            kind,
            performance,
            burdens,
            evidence,
        }
    }

    #[test]
    fn regrettable_substitution_is_not_scalarized() {
        let incumbent = candidate(
            "incumbent",
            PathwayKind::MaterialSubstitution,
            10.0,
            10.0,
            vec![evidence("i1", "source-a", EvidenceKind::Observed, EvidenceStance::Supports, 0.9)],
        );
        let direct = candidate(
            "direct",
            PathwayKind::MaterialSubstitution,
            3.0,
            30.0,
            vec![
                evidence("d1", "source-b", EvidenceKind::Observed, EvidenceStance::Supports, 0.9),
                evidence("d2", "source-c", EvidenceKind::Reported, EvidenceStance::Supports, 0.9),
                evidence("d3", "source-lca", EvidenceKind::LifecycleAssessed, EvidenceStance::Supports, 0.9),
            ],
        );
        let process = candidate(
            "process",
            PathwayKind::ProcessSubstitution,
            4.0,
            4.0,
            vec![
                evidence("p1", "source-d", EvidenceKind::Observed, EvidenceStance::Supports, 0.9),
                evidence("p2", "source-e", EvidenceKind::Reported, EvidenceStance::Supports, 0.9),
            ],
        );
        let result = AlternativesEngine
            .assess(
                &fixture_requirement(),
                &[incumbent, direct, process],
                Some("incumbent"),
            )
            .unwrap();

        let transfer = result
            .burden_transfers
            .iter()
            .find(|transfer| transfer.candidate_id == "direct")
            .unwrap();
        assert!(transfer.is_regrettable_substitution());
        assert!(transfer.clearly_better.contains(&Dimension::Hazard));
        assert!(transfer.clearly_worse.contains(&Dimension::Water));

        assert_eq!(
            result
                .candidates
                .iter()
                .find(|candidate| candidate.candidate_id == "direct")
                .unwrap()
                .qualification,
            QualificationState::LifecycleQualified
        );
    }

    #[test]
    fn missing_evidence_lowers_qualification_and_missing_constraints_block_frontier() {
        let unknown = candidate("unknown", PathwayKind::Elimination, 1.0, 1.0, vec![]);
        let mut blocked = candidate(
            "blocked",
            PathwayKind::ProcessSubstitution,
            2.0,
            2.0,
            vec![evidence("b1", "b", EvidenceKind::Simulated, EvidenceStance::Supports, 0.8)],
        );
        blocked.performance.remove("throughput_per_hour");

        let result = AlternativesEngine
            .assess(&fixture_requirement(), &[unknown, blocked], None)
            .unwrap();

        assert_eq!(
            result
                .candidates
                .iter()
                .find(|candidate| candidate.candidate_id == "unknown")
                .unwrap()
                .qualification,
            QualificationState::Hypothesis
        );
        assert!(!result.pareto_frontier.contains(&"unknown".into()));
        assert_eq!(
            result.frontier_blockers["blocked"][0],
            FrontierBlocker::ConstraintUnresolved("throughput_per_hour".into())
        );
    }

    #[test]
    fn conflicting_sources_remain_visible_and_cap_qualification() {
        let candidate = candidate(
            "conflict",
            PathwayKind::ProcessSubstitution,
            3.0,
            3.0,
            vec![
                evidence("s1", "source-a", EvidenceKind::Observed, EvidenceStance::Supports, 0.95),
                evidence("s2", "source-b", EvidenceKind::Observed, EvidenceStance::Contradicts, 0.95),
            ],
        );

        let result = AlternativesEngine::assess(&fixture_requirement(), &[candidate], None).unwrap();
        let assessment = &result.candidates[0];
        assert!(assessment.evidence_conflict);
        assert_eq!(
            assessment.qualification,
            QualificationState::ComputationallyPlausible
        );
    }

    #[test]
    fn changing_functional_requirement_changes_frontier() {
        let a = candidate(
            "a",
            PathwayKind::ProcessSubstitution,
            2.0,
            8.0,
            vec![
                evidence("a1", "a1", EvidenceKind::Observed, EvidenceStance::Supports, 0.9),
                evidence("a2", "a2", EvidenceKind::Observed, EvidenceStance::Supports, 0.9),
            ],
        );
        let b = candidate(
            "b",
            PathwayKind::ProductRedesign,
            4.0,
            4.0,
            vec![
                evidence("b1", "b1", EvidenceKind::Observed, EvidenceStance::Supports, 0.9),
                evidence("b2", "b2", EvidenceKind::Observed, EvidenceStance::Supports, 0.9),
            ],
        );

        let requirement_a = fixture_requirement();
        let mut requirement_b = fixture_requirement();
        requirement_b
            .constraints
            .insert("throughput_per_hour".into(), RequirementBound::AtLeast(125.0));

        let baseline = AlternativesEngine.assess(&requirement_a, &[a.clone(), b.clone()], None).unwrap();
        let constrained = AlternativesEngine.assess(&requirement_b, &[a, b], None).unwrap();

        assert_ne!(baseline.pareto_frontier, constrained.pareto_frontier);
        assert!(constrained.frontier_blockers.contains_key("a"));
        assert!(constrained.frontier_blockers.contains_key("b"));
    }

    #[test]
    fn receipt_is_deterministic() {
        let c = candidate(
            "c",
            PathwayKind::ProcessSubstitution,
            3.0,
            3.0,
            vec![
                evidence("c1", "s1", EvidenceKind::Observed, EvidenceStance::Supports, 0.9),
                evidence("c2", "s2", EvidenceKind::Observed, EvidenceStance::Supports, 0.9),
            ],
        );
        let requirement = fixture_requirement();
        let first = AlternativesEngine.assess(&requirement, &[c.clone()], None).unwrap();
        let second = AlternativesEngine.assess(&requirement, &[c], None).unwrap();
        assert_eq!(first.receipt, second.receipt);
        assert!(!first.receipt.payload_hash.is_empty());
    }

    #[test]
    fn heuristic_measurement_target_is_exposed() {
        let mut c = candidate(
            "c",
            PathwayKind::ProcessSubstitution,
            2.0,
            8.0,
            vec![evidence("c1", "source", EvidenceKind::Observed, EvidenceStance::Supports, 0.9)],
        );
        let estimate = c.burdens.get(&Dimension::Water).unwrap().clone();
        c.burdens.insert(
            Dimension::Water,
            BurdenEstimate {
                interval: Interval::new(estimate.interval.lower, estimate.interval.upper + 100.0)
                    .unwrap(),
                unit: estimate.unit,
                scope: estimate.scope,
                evidence_ids: estimate.evidence_ids,
            },
        );

        let result = AlternativesEngine.assess(&fixture_requirement(), &[c], None).unwrap();
        assert_eq!(
            result.next_measurement.as_ref().unwrap().dimension,
            Dimension::Water
        );
        assert_eq!(
            result
                .next_measurement
                .as_ref()
                .unwrap()
                .unresolved_candidate_count,
            1
        );
    }

    #[test]
    #[test]
    fn single_field_observation_cannot_promote_entire_candidate() {
        let mut c = candidate(
            "field",
            PathwayKind::ProcessSubstitution,
            1.0,
            1.0,
            vec![
                evidence("f1", "field-source", EvidenceKind::FieldObserved, EvidenceStance::Supports, 0.95),
                evidence("f2", "source-a", EvidenceKind::Observed, EvidenceStance::Supports, 0.9),
                evidence("f3", "source-b", EvidenceKind::Observed, EvidenceStance::Supports, 0.9),
            ],
        );
        c.burdens.values_mut().skip(1).for_each(|estimate| {
            estimate.evidence_ids = vec!["f2".into(), "f3".into()];
        });

        let result = AlternativesEngine.assess(&fixture_requirement(), &[c], None).unwrap();
        assert_eq!(
            result.candidates[0].qualification,
            QualificationState::LifecycleQualified
        );
    }

    #[test]
    fn assessment_is_order_independent() {
        let a = candidate(
            "a",
            PathwayKind::ProcessSubstitution,
            2.0,
            8.0,
            vec![
                evidence("a2", "source-2", EvidenceKind::Observed, EvidenceStance::Supports, 0.9),
                evidence("a1", "source-1", EvidenceKind::Observed, EvidenceStance::Supports, 0.9),
            ],
        );
        let b = candidate(
            "b",
            PathwayKind::ProductRedesign,
            3.0,
            4.0,
            vec![
                evidence("b2", "source-4", EvidenceKind::Observed, EvidenceStance::Supports, 0.9),
                evidence("b1", "source-3", EvidenceKind::Observed, EvidenceStance::Supports, 0.9),
            ],
        );

        let first = AlternativesEngine
            .assess(&fixture_requirement(), &[a.clone(), b.clone()], None)
            .unwrap();
        let second = AlternativesEngine
            .assess(&fixture_requirement(), &[b, a], None)
            .unwrap();

        assert_eq!(first, second);
        assert_eq!(first.pareto_frontier, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn evidence_scope_mismatch_is_rejected() {
        let mut c = candidate(
            "scope-mismatch",
            PathwayKind::ProcessSubstitution,
            2.0,
            2.0,
            vec![evidence("x1", "source", EvidenceKind::Observed, EvidenceStance::Supports, 0.9)],
        );
        c.burdens.get_mut(&Dimension::Water).unwrap().scope = "EU".into();

        let error = AlternativesEngine
            .assess(&fixture_requirement(), &[c], None)
            .unwrap_err();

        assert!(matches!(error, AssessmentError::EvidenceScopeMismatch { .. }));
    }

    #[test]
    fn evidence_unit_mismatch_is_rejected() {
        let mut c = candidate(
            "unit-mismatch",
            PathwayKind::ProcessSubstitution,
            2.0,
            2.0,
            vec![evidence("x1", "source", EvidenceKind::Observed, EvidenceStance::Supports, 0.9)],
        );
        c.burdens.get_mut(&Dimension::Water).unwrap().unit = "litre".into();

        let error = AlternativesEngine
            .assess(&fixture_requirement(), &[c], None)
            .unwrap_err();

        assert!(matches!(error, AssessmentError::EvidenceUnitMismatch { .. }));
    }

    #[test]
    fn lifecycle_qualification_requires_explicit_lifecycle_evidence() {
        let c = candidate(
            "reported-only",
            PathwayKind::MaterialSubstitution,
            2.0,
            2.0,
            vec![
                evidence("x1", "source-a", EvidenceKind::Observed, EvidenceStance::Supports, 0.9),
                evidence("x2", "source-b", EvidenceKind::Reported, EvidenceStance::Supports, 0.9),
            ],
        );
        let result = AlternativesEngine.assess(&fixture_requirement(), &[c], None).unwrap();

        assert_eq!(
            result.candidates[0].qualification,
            QualificationState::EvidenceSupported
        );
    }

    #[test]
    fn missing_dimension_blocks_frontier() {
        let mut c = candidate(
            "missing",
            PathwayKind::Elimination,
            1.0,
            1.0,
            vec![],
        );
        c.burdens.remove(&Dimension::Carbon);
        let result = AlternativesEngine.assess(&fixture_requirement(), &[c], None).unwrap();
        assert!(matches!(
            &result.frontier_blockers["missing"][0],
            FrontierBlocker::MissingDimension(Dimension::Carbon)
        ));
    }
}
