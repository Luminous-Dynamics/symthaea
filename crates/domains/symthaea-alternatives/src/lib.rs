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
pub const SCHEMA_VERSION: u16 = 9;
/// Assessment algorithm version.
pub const ALGORITHM_VERSION: &str = "pareto-interval-evidence-time-envelope-derivation-source-admission-subject-freshness-v15";

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
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
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

/// Explicit decision-profile freshness semantics for evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceFreshnessPolicy {
    /// Stable identity of the freshness policy profile.
    pub policy_id: String,
    /// Policy revision.
    pub policy_revision: String,
    /// Digest of the exact policy semantics.
    pub policy_digest: String,
    /// Maximum evidence age by evidence kind in seconds.
    ///
    /// Kinds absent from this map are not freshness-bounded by this policy.
    pub max_age_seconds_by_kind: BTreeMap<EvidenceKind, u64>,
}

impl EvidenceFreshnessPolicy {
    /// Validate the explicit freshness-policy identity and rules.
    pub fn validate(&self) -> Result<(), AssessmentError> {
        if self.policy_id.is_empty()
            || self.policy_revision.is_empty()
            || self.policy_digest.is_empty()
        {
            return Err(AssessmentError::EmptyFreshnessPolicyIdentity);
        }
        if self.max_age_seconds_by_kind.is_empty() {
            return Err(AssessmentError::EmptyFreshnessPolicy);
        }
        Ok(())
    }

    fn max_age_for(&self, kind: EvidenceKind) -> Option<u64> {
        self.max_age_seconds_by_kind.get(&kind).copied()
    }
}

/// Reproducible provenance for a simulated or derived evidence record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DerivationRecord {
    /// Stable identifier for the model, solver, transformation, or reasoning procedure.
    pub method_id: String,
    /// Version of the derivation method.
    pub method_version: String,
    /// Stable references to the inputs consumed by the derivation.
    pub input_refs: Vec<String>,
    /// Optional digest of the derivation configuration or source artifact.
    pub configuration_hash: Option<String>,
}

impl DerivationRecord {
    /// Validate derivation identity and input references.
    pub fn validate(&self) -> Result<(), AssessmentError> {
        if self.method_id.is_empty() || self.method_version.is_empty() || self.input_refs.is_empty()
        {
            return Err(AssessmentError::EmptyDerivationIdentity);
        }
        if self.input_refs.iter().any(|input| input.is_empty()) {
            return Err(AssessmentError::EmptyDerivationInput);
        }
        if self
            .configuration_hash
            .as_ref()
            .is_some_and(String::is_empty)
        {
            return Err(AssessmentError::EmptyDerivationConfigurationHash);
        }
        Ok(())
    }
}

/// Whether evidence supports or contradicts the linked assertion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceStance {
    /// Evidence supports the assertion.
    Supports,
    /// Evidence contradicts the assertion.
    Contradicts,
}

/// Canonical identity for the provenance source of an evidence record.
///
/// This is an identity contract, not an authenticity proof. External
/// admission/attestation must establish that the declared authority actually
/// controls the referenced source.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceSourceIdentity {
    /// Stable authority/organization identity used for authority-diversity accounting.
    pub authority_id: String,
    /// Stable identifier for the referenced artifact, dataset, report, or observation stream.
    pub artifact_id: String,
    /// Digest of the referenced artifact or canonical source payload.
    pub artifact_digest: String,
    /// Optional issuer key fingerprint for future cryptographic attestation.
    pub issuer_key_fingerprint: Option<String>,
    /// Optional externally qualified source-authority admission reference.
    pub admission: Option<SourceAdmissionRef>,
}

/// Reference to an externally qualified source-authority admission.
///
/// This structure carries the exact policy/admission identity used by an
/// external authority boundary such as Mycelix. It does not perform or imply
/// cryptographic verification inside this crate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceAdmissionRef {
    /// Stable identifier for the source-admissibility policy.
    pub policy_id: String,
    /// Policy revision.
    pub policy_revision: String,
    /// Digest of the exact source-admissibility policy.
    pub policy_digest: String,
    /// Stable identity of the admission record.
    pub admission_id: String,
    /// Authority epoch/generation under which the admission was issued.
    pub authority_epoch: String,
    /// Optional canonical fault-domain identity supplied by the authority policy.
    pub fault_domain_id: Option<String>,
    /// Optional Unix timestamp from which the admission is valid.
    pub valid_from_epoch_seconds: Option<i64>,
    /// Optional Unix timestamp through which the admission is valid.
    pub valid_until_epoch_seconds: Option<i64>,
}

impl SourceAdmissionRef {
    /// Validate the externally supplied admission reference structurally.
    pub fn validate(&self) -> Result<(), AssessmentError> {
        if self.policy_id.is_empty()
            || self.policy_revision.is_empty()
            || self.policy_digest.is_empty()
            || self.admission_id.is_empty()
            || self.authority_epoch.is_empty()
        {
            return Err(AssessmentError::EmptySourceAdmissionReference);
        }
        if self
            .fault_domain_id
            .as_ref()
            .is_some_and(String::is_empty)
        {
            return Err(AssessmentError::EmptySourceAdmissionReference);
        }
        if let (Some(from), Some(until)) = (
            self.valid_from_epoch_seconds,
            self.valid_until_epoch_seconds,
        ) && from > until
        {
            return Err(AssessmentError::InvalidSourceAdmissionValidity { from, until });
        }
        Ok(())
    }
}

impl EvidenceSourceIdentity {
    /// Validate the canonical source identity fields.
    pub fn validate(&self) -> Result<(), AssessmentError> {
        if self.authority_id.is_empty()
            || self.artifact_id.is_empty()
            || self.artifact_digest.is_empty()
        {
            return Err(AssessmentError::EmptySourceIdentity);
        }
        if self
            .issuer_key_fingerprint
            .as_ref()
            .is_some_and(String::is_empty)
        {
            return Err(AssessmentError::EmptySourceIdentity);
        }
        if let Some(admission) = &self.admission {
            admission.validate()?;
        }
        Ok(())
    }

    /// Derive the stable identity used when counting distinct authority groups.
    ///
    /// This is structural source diversity, not proof of epistemic or organizational independence.
    pub fn authority_group_id(&self) -> String {
        let bytes = serde_json::to_vec(&self.authority_id)
            .expect("source authority identity is serializable");
        let mut hasher = Hasher::new();
        hasher.update(&bytes);
        hasher.finalize().to_hex().to_string()
    }
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
    /// Canonical provenance identity used for source-diversity accounting.
    pub source: EvidenceSourceIdentity,
    /// Human-readable scope: functional unit, geography, process, etc.
    pub scope: String,
    /// Optional unit for the associated quantity.
    pub unit: Option<String>,
    /// Optional source timestamp/version label.
    pub as_of: Option<String>,
    /// Optional Unix timestamp representing when the observation or measurement occurred.
    ///
    /// This is distinct from validity windows and the human/source version label in as_of.
    pub observed_at_epoch_seconds: Option<i64>,
    /// Optional Unix timestamp from which this evidence is valid.
    pub valid_from_epoch_seconds: Option<i64>,
    /// Optional Unix timestamp through which this evidence is valid.
    pub valid_until_epoch_seconds: Option<i64>,
    /// Reproducible derivation provenance for simulated/derived evidence.
    pub derivation: Option<DerivationRecord>,
}

impl EvidenceRecord {
    /// Validate identity and confidence.
    pub fn validate(&self) -> Result<(), AssessmentError> {
        if !self.confidence.is_finite() || !(0.0..=1.0).contains(&self.confidence) {
            return Err(AssessmentError::InvalidConfidence(self.confidence));
        }
        if self.id.is_empty() || self.scope.is_empty() {
            return Err(AssessmentError::EmptyEvidenceIdentity);
        }
        self.source.validate()?;
        if let (Some(from), Some(until)) = (
            self.valid_from_epoch_seconds,
            self.valid_until_epoch_seconds,
        ) && from > until
        {
            return Err(AssessmentError::InvalidEvidenceValidity { from, until });
        }
        if matches!(self.kind, EvidenceKind::Simulated | EvidenceKind::Derived) {
            let Some(derivation) = &self.derivation else {
                return Err(AssessmentError::MissingDerivationMetadata(self.kind));
            };
            derivation.validate()?;
        } else if let Some(derivation) = &self.derivation {
            derivation.validate()?;
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
    fn check(self, interval: Option<Interval>) -> ConstraintStatus {
        let Some(interval) = interval else {
            return ConstraintStatus::Unresolved;
        };
        match self {
            Self::AtLeast(min) if interval.lower >= min => ConstraintStatus::Pass,
            Self::AtLeast(min) if interval.upper < min => ConstraintStatus::Fail,
            Self::AtMost(max) if interval.upper <= max => ConstraintStatus::Pass,
            Self::AtMost(max) if interval.lower > max => ConstraintStatus::Fail,
            Self::Between { min, max } if interval.lower >= min && interval.upper <= max => {
                ConstraintStatus::Pass
            }
            Self::Between { max, .. } if interval.lower > max => ConstraintStatus::Fail,
            Self::Between { min, .. } if interval.upper < min => ConstraintStatus::Fail,
            _ => ConstraintStatus::Unresolved,
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

/// Required operating range for one physical/environmental condition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OperatingRequirement {
    /// Required interval that the candidate must cover.
    pub interval: Interval,
    /// Unit for the operating condition.
    pub unit: String,
    /// Functional/geographic/system scope for the condition.
    pub scope: String,
}

impl OperatingRequirement {
    /// Validate the required range and comparison metadata.
    pub fn validate(&self) -> Result<(), AssessmentError> {
        Interval::new(self.interval.lower, self.interval.upper)?;
        if self.unit.is_empty() || self.scope.is_empty() {
            return Err(AssessmentError::EmptyOperatingScale);
        }
        Ok(())
    }
}

/// Immutable identity of the product/component/design context being assessed.
///
/// This binds an assessment to an exact externally identified subject without
/// making that identity authoritative inside Symthaea.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssessmentSubjectRef {
    /// Stable identifier assigned by the owning system.
    pub subject_id: String,
    /// Versioned identity of the subject/profile namespace.
    pub profile_id: String,
    /// Revision of the subject/profile namespace.
    pub profile_revision: String,
    /// Digest of the exact BOM/design/product/routing context being assessed.
    pub subject_digest: String,
}

impl AssessmentSubjectRef {
    /// Validate the immutable subject identity fields.
    pub fn validate(&self) -> Result<(), AssessmentError> {
        if self.subject_id.is_empty()
            || self.profile_id.is_empty()
            || self.profile_revision.is_empty()
            || self.subject_digest.is_empty()
        {
            return Err(AssessmentError::EmptyAssessmentSubject);
        }
        Ok(())
    }
}

/// The function that must be satisfied independently of the incumbent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionalRequirement {
    /// Stable requirement identifier.
    pub id: String,
    /// Exact product/component/design context to which this requirement applies.
    pub subject: AssessmentSubjectRef,
    /// Human-readable description.
    pub description: String,
    /// Named performance constraints.
    pub constraints: BTreeMap<String, RequirementBound>,
    /// Explicit comparison scales for every burden dimension.
    pub comparison_scales: BTreeMap<Dimension, ComparisonScale>,
    /// Explicit comparison scales for every constrained performance metric.
    pub performance_scales: BTreeMap<String, ComparisonScale>,
    /// Required operating envelope keyed by condition (for example temperature or pressure).
    pub operating_envelope: BTreeMap<String, OperatingRequirement>,
}

impl FunctionalRequirement {
    /// Validate identity, numeric bounds, and explicit comparison scales.
    pub fn validate(&self) -> Result<(), AssessmentError> {
        if self.id.is_empty() || self.description.is_empty() {
            return Err(AssessmentError::EmptyRequirementIdentity);
        }
        self.subject.validate()?;
        if self.constraints.is_empty() {
            return Err(AssessmentError::EmptyFunctionalConstraints);
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
        for metric in self.constraints.keys() {
            let Some(scale) = self.performance_scales.get(metric) else {
                return Err(AssessmentError::MissingPerformanceScale(metric.clone()));
            };
            scale.validate()?;
        }
        for (condition, requirement) in &self.operating_envelope {
            if condition.is_empty() {
                return Err(AssessmentError::EmptyOperatingCondition);
            }
            requirement.validate()?;
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

/// Evidence-linked functional performance measurement.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerformanceEstimate {
    /// Plausible performance interval.
    pub interval: Interval,
    /// Unit declared by the functional requirement.
    pub unit: String,
    /// Scope in which this performance value applies.
    pub scope: String,
    /// Evidence IDs supporting or contradicting the value.
    pub evidence_ids: Vec<String>,
}

impl PerformanceEstimate {
    /// Validate the performance value and scale metadata.
    pub fn validate(&self) -> Result<(), AssessmentError> {
        Interval::new(self.interval.lower, self.interval.upper)?;
        if self.unit.is_empty() || self.scope.is_empty() {
            return Err(AssessmentError::EmptyPerformanceScale);
        }
        Ok(())
    }
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
    /// Evidence-linked performance values keyed by requirement metric.
    pub performance: BTreeMap<String, PerformanceEstimate>,
    /// Burden estimates by dimension.
    pub burdens: BTreeMap<Dimension, BurdenEstimate>,
    /// Evidence-linked operating capabilities keyed by condition.
    pub operating_capabilities: BTreeMap<String, PerformanceEstimate>,
    /// Candidate-level evidence bundle.
    pub evidence: Vec<EvidenceRecord>,
}

impl CandidatePathway {
    /// Validate the candidate, performance values, burden intervals, evidence
    /// references, and evidence records.
    pub fn validate(&self) -> Result<(), AssessmentError> {
        if self.id.is_empty() || self.name.is_empty() {
            return Err(AssessmentError::EmptyCandidateIdentity);
        }
        if self.burdens.is_empty() {
            return Err(AssessmentError::NoBurdenData);
        }
        for performance in self.performance.values() {
            performance.validate()?;
        }
        for capability in self.operating_capabilities.values() {
            capability.validate()?;
        }
        let evidence_ids = self
            .evidence
            .iter()
            .map(|e| e.id.as_str())
            .collect::<BTreeSet<_>>();
        if evidence_ids.len() != self.evidence.len() {
            let duplicate = self
                .evidence
                .iter()
                .find(|evidence| {
                    self.evidence
                        .iter()
                        .filter(|other| other.id == evidence.id)
                        .count()
                        > 1
                })
                .map(|evidence| evidence.id.clone())
                .unwrap_or_default();
            return Err(AssessmentError::DuplicateEvidenceId(duplicate));
        }
        for performance in self.performance.values() {
            for evidence_id in &performance.evidence_ids {
                let Some(evidence) = self.evidence.iter().find(|e| e.id == *evidence_id) else {
                    return Err(AssessmentError::MissingEvidenceReference(
                        evidence_id.clone(),
                    ));
                };
                if evidence.scope != performance.scope {
                    return Err(AssessmentError::PerformanceEvidenceScopeMismatch {
                        evidence_id: evidence.id.clone(),
                        performance_scope: performance.scope.clone(),
                        evidence_scope: evidence.scope.clone(),
                    });
                }
                if let Some(evidence_unit) = &evidence.unit
                    && evidence_unit != &performance.unit
                {
                    return Err(AssessmentError::PerformanceEvidenceUnitMismatch {
                        evidence_id: evidence.id.clone(),
                        performance_unit: performance.unit.clone(),
                        evidence_unit: evidence_unit.clone(),
                    });
                }
            }
        }
        for capability in self.operating_capabilities.values() {
            for evidence_id in &capability.evidence_ids {
                let Some(evidence) = self.evidence.iter().find(|e| e.id == *evidence_id) else {
                    return Err(AssessmentError::MissingEvidenceReference(
                        evidence_id.clone(),
                    ));
                };
                if evidence.scope != capability.scope {
                    return Err(AssessmentError::PerformanceEvidenceScopeMismatch {
                        evidence_id: evidence.id.clone(),
                        performance_scope: capability.scope.clone(),
                        evidence_scope: evidence.scope.clone(),
                    });
                }
                if let Some(evidence_unit) = &evidence.unit
                    && evidence_unit != &capability.unit
                {
                    return Err(AssessmentError::PerformanceEvidenceUnitMismatch {
                        evidence_id: evidence.id.clone(),
                        performance_unit: capability.unit.clone(),
                        evidence_unit: evidence_unit.clone(),
                    });
                }
            }
        }
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

    fn evidence_is_valid_at(evidence: &EvidenceRecord, as_of: Option<i64>) -> bool {
        match as_of {
            Some(timestamp) => {
                evidence
                    .valid_from_epoch_seconds
                    .is_none_or(|from| from <= timestamp)
                    && evidence
                        .valid_until_epoch_seconds
                        .is_none_or(|until| timestamp <= until)
            }
            None => {
                evidence.valid_from_epoch_seconds.is_none()
                    && evidence.valid_until_epoch_seconds.is_none()
            }
        }
    }

    fn evidence_is_usable_at(
        evidence: &EvidenceRecord,
        as_of: Option<i64>,
        freshness_policy: Option<&EvidenceFreshnessPolicy>,
    ) -> bool {
        if !Self::evidence_is_valid_at(evidence, as_of) {
            return false;
        }

        let Some(policy) = freshness_policy else {
            return true;
        };
        let Some(max_age_seconds) = policy.max_age_for(evidence.kind) else {
            return true;
        };
        let Some(assessed_at) = as_of else {
            return false;
        };
        let Some(observed_at) = evidence.observed_at_epoch_seconds else {
            return false;
        };

        let age = i128::from(assessed_at) - i128::from(observed_at);
        age >= 0 && age <= i128::from(max_age_seconds)
    }

    fn linked_evidence_at<'a>(
        &'a self,
        ids: &'a [String],
        as_of: Option<i64>,
        freshness_policy: Option<&EvidenceFreshnessPolicy>,
    ) -> impl Iterator<Item = &'a EvidenceRecord> {
        self.evidence.iter().filter(move |e| {
            ids.iter().any(|id| id == &e.id)
                && Self::evidence_is_usable_at(e, as_of, freshness_policy)
        })
    }

    fn dimension_has_conflict_at(
        &self,
        estimate: &BurdenEstimate,
        as_of: Option<i64>,
        freshness_policy: Option<&EvidenceFreshnessPolicy>,
    ) -> bool {
        let support = self
            .linked_evidence_at(&estimate.evidence_ids, as_of, freshness_policy)
            .any(|e| e.stance == EvidenceStance::Supports);
        let contradict = self
            .linked_evidence_at(&estimate.evidence_ids, as_of)
            .any(|e| e.stance == EvidenceStance::Contradicts);
        support && contradict
    }

    fn has_conflict_at(
        &self,
        as_of: Option<i64>,
        freshness_policy: Option<&EvidenceFreshnessPolicy>,
    ) -> bool {
        self.burdens
            .values()
            .any(|estimate| self.dimension_has_conflict_at(estimate, as_of, freshness_policy))
    }

    fn performance_evidence_is_supported_at(
        &self,
        metric: &str,
        as_of: Option<i64>,
        freshness_policy: Option<&EvidenceFreshnessPolicy>,
    ) -> bool {
        self.performance
            .get(metric)
            .map(|estimate| {
                self.linked_evidence_at(&estimate.evidence_ids, as_of, freshness_policy)
                    .any(|e| {
                        matches!(
                            e.kind,
                            EvidenceKind::Observed
                                | EvidenceKind::Reported
                                | EvidenceKind::Derived
                                | EvidenceKind::ManufacturingObserved
                                | EvidenceKind::FieldObserved
                                | EvidenceKind::ContinuouslyMonitored
                        ) && e.stance == EvidenceStance::Supports
                            && e.confidence >= 0.7
                    })
            })
            .unwrap_or(false)
    }

    fn operating_evidence_is_supported_at(
        &self,
        condition: &str,
        as_of: Option<i64>,
        freshness_policy: Option<&EvidenceFreshnessPolicy>,
    ) -> bool {
        self.operating_capabilities
            .get(condition)
            .map(|estimate| {
                self.linked_evidence_at(&estimate.evidence_ids, as_of, freshness_policy)
                    .any(|e| {
                        matches!(
                            e.kind,
                            EvidenceKind::Observed
                                | EvidenceKind::Reported
                                | EvidenceKind::Derived
                                | EvidenceKind::ManufacturingObserved
                                | EvidenceKind::FieldObserved
                                | EvidenceKind::ContinuouslyMonitored
                        ) && e.stance == EvidenceStance::Supports
                            && e.confidence >= 0.7
                    })
            })
            .unwrap_or(false)
    }

    fn operating_envelope_is_supported(
        &self,
        requirement: &FunctionalRequirement,
        as_of: Option<i64>,
        freshness_policy: Option<&EvidenceFreshnessPolicy>,
    ) -> bool {
        requirement.operating_envelope.iter().all(|(condition, required)| {
            let Some(capability) = self.operating_capabilities.get(condition) else {
                return false;
            };
            capability.unit == required.unit
                && capability.scope == required.scope
                && capability.interval.lower <= required.interval.lower
                && capability.interval.upper >= required.interval.upper
                && self.operating_evidence_is_supported_at(condition, as_of, freshness_policy)
        })
    }

    fn performance_is_supported(
        &self,
        requirement: &FunctionalRequirement,
        as_of: Option<i64>,
        freshness_policy: Option<&EvidenceFreshnessPolicy>,
    ) -> bool {
        requirement.constraints.keys().all(|metric| {
            let Some(estimate) = self.performance.get(metric) else {
                return false;
            };
            let Some(scale) = requirement.performance_scales.get(metric) else {
                return false;
            };
            estimate.unit == scale.unit
                && estimate.scope == scale.scope
                && self.performance_evidence_is_supported_at(metric, as_of, freshness_policy)
        })
    }

    fn qualification_ceiling(
        &self,
        requirement: &FunctionalRequirement,
        as_of: Option<i64>,
        freshness_policy: Option<&EvidenceFreshnessPolicy>,
    ) -> QualificationState {
        if !self.performance_is_supported(requirement, as_of, freshness_policy)
            || !self.operating_envelope_is_supported(requirement, as_of, freshness_policy)
            || self.burdens.is_empty()
            || self.burdens.values().all(|estimate| {
                let linked = self
                    .linked_evidence_at(&estimate.evidence_ids, as_of)
                    .collect::<Vec<_>>();
                linked.is_empty()
                    || linked
                        .iter()
                        .all(|e| e.kind == EvidenceKind::Hypothesis)
            })
        {
            return QualificationState::Hypothesis;
        }

        if self.has_conflict_at(as_of, freshness_policy) {
            return QualificationState::ComputationallyPlausible;
        }

        let any_simulation = self.burdens.values().any(|estimate| {
            self.linked_evidence_at(&estimate.evidence_ids, as_of)
                .any(|e| e.kind == EvidenceKind::Simulated)
        });
        let any_supported_measurement = self.burdens.values().any(|estimate| {
            self.linked_evidence_at(&estimate.evidence_ids, as_of).any(|e| {
                matches!(
                    e.kind,
                    EvidenceKind::Observed | EvidenceKind::Reported | EvidenceKind::Derived
                ) && e.stance == EvidenceStance::Supports
                    && e.confidence >= 0.7
            })
        });
        let distinct_authority_sources = self
            .burdens
            .values()
            .flat_map(|estimate| self.linked_evidence_at(&estimate.evidence_ids, as_of))
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
            .map(|e| e.source.authority_group_id())
            .collect::<BTreeSet<_>>()
            .len();
        let has_all_dimension_evidence = Dimension::ALL.iter().all(|dimension| {
            self.burdens
                .get(dimension)
                .map(|estimate| {
                    self.linked_evidence_at(&estimate.evidence_ids, as_of).any(|e| {
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
            self.linked_evidence_at(&estimate.evidence_ids, as_of).any(|e| {
                e.kind == EvidenceKind::LifecycleAssessed
                    && e.stance == EvidenceStance::Supports
                    && e.confidence >= 0.7
            })
        });
        let field_distinct_authority_sources = self
            .burdens
            .values()
            .flat_map(|estimate| self.linked_evidence_at(&estimate.evidence_ids, as_of))
            .filter(|e| {
                e.kind == EvidenceKind::FieldObserved
                    && e.stance == EvidenceStance::Supports
                    && e.confidence >= 0.7
            })
            .map(|e| e.source.authority_group_id())
            .collect::<BTreeSet<_>>()
            .len();
        let monitoring_distinct_authority_sources = self
            .burdens
            .values()
            .flat_map(|estimate| self.linked_evidence_at(&estimate.evidence_ids, as_of))
            .filter(|e| {
                e.kind == EvidenceKind::ContinuouslyMonitored
                    && e.stance == EvidenceStance::Supports
                    && e.confidence >= 0.7
            })
            .map(|e| e.source.authority_group_id())
            .collect::<BTreeSet<_>>()
            .len();
        let all_dimensions_field_observed = Dimension::ALL.iter().all(|dimension| {
            self.burdens
                .get(dimension)
                .map(|estimate| {
                    self.linked_evidence_at(&estimate.evidence_ids, as_of).any(|e| {
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
                    self.linked_evidence_at(&estimate.evidence_ids, as_of).any(|e| {
                        e.kind == EvidenceKind::ContinuouslyMonitored
                            && e.stance == EvidenceStance::Supports
                            && e.confidence >= 0.7
                    })
                })
                .unwrap_or(false)
        });
        let has_manufacturing_observation = self.burdens.values().any(|estimate| {
            self.linked_evidence_at(&estimate.evidence_ids, as_of).any(|e| {
                e.kind == EvidenceKind::ManufacturingObserved
                    && e.stance == EvidenceStance::Supports
                    && e.confidence >= 0.7
            })
        });

        if all_dimensions_monitored && monitoring_distinct_authority_sources >= 2 {
            QualificationState::ContinuouslyMonitored
        } else if all_dimensions_field_observed && field_distinct_authority_sources >= 2 {
            QualificationState::FieldQualified
        } else if any_supported_measurement
            && distinct_authority_sources >= 2
            && has_all_dimension_evidence
            && has_manufacturing_observation
        {
            QualificationState::ManufacturingQualified
        } else if any_supported_measurement
            && distinct_authority_sources >= 2
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
    /// Multiple distinct authority groups provide supported evidence.
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
    /// Evidence-linked functional performance estimates.
    pub performance: BTreeMap<String, PerformanceEstimate>,
    /// Evidence-linked operating capabilities.
    pub operating_capabilities: BTreeMap<String, PerformanceEstimate>,
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
        /// Candidate comparison scope.
        actual_scope: String,
    },
    /// A burden has explicit evidence references, but none are valid at assessment time.
    EvidenceUnavailable(Dimension),
    /// A required operating condition is not covered by the candidate.
    OperatingConditionUnresolved(String),
    /// A required operating condition lies outside the candidate capability.
    OperatingConditionFailed(String),
    /// Functional performance uses a different unit or scope from the requirement.
    PerformanceIncompatibleScale {
        /// Functional requirement metric.
        metric: String,
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

/// A set of functional requirements that must all be satisfied by one pathway.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionalRequirementSet {
    /// Requirements keyed by their stable IDs.
    pub requirements: BTreeMap<String, FunctionalRequirement>,
}

impl FunctionalRequirementSet {
    /// Construct a requirement set from an ordered map.
    pub fn new(requirements: BTreeMap<String, FunctionalRequirement>) -> Result<Self, AssessmentError> {
        if requirements.is_empty() {
            return Err(AssessmentError::EmptyRequirementSet);
        }
        for (id, requirement) in &requirements {
            requirement.validate()?;
            if id != &requirement.id {
                return Err(AssessmentError::RequirementSetKeyMismatch {
                    key: id.clone(),
                    requirement_id: requirement.id.clone(),
                });
            }
        }
        Ok(Self { requirements })
    }

    /// Validate every requirement and its map identity.
    pub fn validate(&self) -> Result<(), AssessmentError> {
        if self.requirements.is_empty() {
            return Err(AssessmentError::EmptyRequirementSet);
        }
        for (id, requirement) in &self.requirements {
            requirement.validate()?;
            if id != &requirement.id {
                return Err(AssessmentError::RequirementSetKeyMismatch {
                    key: id.clone(),
                    requirement_id: requirement.id.clone(),
                });
            }
        }
        Ok(())
    }
}

/// Why a pathway is excluded from joint requirement-set eligibility.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequirementSetBlocker {
    /// Requirement that produced the blocker.
    pub requirement_id: String,
    /// Underlying fail-closed assessment blocker.
    pub blocker: FrontierBlocker,
}

/// Joint assessment across multiple functions of one system.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RequirementSetAssessment {
    /// Version of the requirement-set result schema.
    pub schema_version: u16,
    /// Version of the joint-gating algorithm.
    pub algorithm_version: String,
    /// Individual requirement assessments.
    pub assessments: BTreeMap<String, AssessmentResult>,
    /// Candidate IDs that satisfy every requirement without unresolved blockers.
    pub jointly_eligible_candidate_ids: Vec<String>,
    /// Qualification ceiling limited by the least-qualified requirement assessment.
    pub joint_qualification: BTreeMap<String, QualificationState>,
    /// All blockers grouped by candidate across the requirement set.
    pub blockers: BTreeMap<String, Vec<RequirementSetBlocker>>,
    /// Deterministic receipt over the complete joint assessment payload.
    pub receipt: AssessmentReceipt,
}

/// Schema version for multi-requirement assessment results.
pub const REQUIREMENT_SET_SCHEMA_VERSION: u16 = 1;
/// Algorithm version for multi-requirement intersection gating.
pub const REQUIREMENT_SET_ALGORITHM_VERSION: &str = "multi-requirement-intersection-v1";

/// Complete deterministic assessment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssessmentResult {
    /// Schema version.
    pub schema_version: u16,
    /// Algorithm version.
    pub algorithm_version: String,
    /// Functional requirement.
    pub requirement: FunctionalRequirement,
    /// Optional Unix timestamp at which time-bounded evidence was evaluated.
    pub assessed_at_epoch_seconds: Option<i64>,
    /// Exact evidence-freshness policy applied during the assessment.
    pub freshness_policy: Option<EvidenceFreshnessPolicy>,
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
    /// Evidence validity bounds are inverted.
    InvalidEvidenceValidity { from: i64, until: i64 },
    /// An operating condition lacks a unit or scope.
    EmptyOperatingScale,
    /// An operating condition has an empty identity.
    EmptyOperatingCondition,
    /// A functional requirement contains no performance constraints.
    EmptyFunctionalConstraints,
    /// A performance estimate lacks a comparable unit or scope.
    EmptyPerformanceScale,
    /// The requirement does not declare a comparison scale for a dimension.
    MissingComparisonScale(Dimension),
    /// The requirement does not declare a comparison scale for a performance metric.
    MissingPerformanceScale(String),
    /// Linked evidence uses a different scope from a performance estimate.
    PerformanceEvidenceScopeMismatch {
        /// Evidence identifier.
        evidence_id: String,
        /// Performance comparison scope.
        performance_scope: String,
        /// Evidence scope.
        evidence_scope: String,
    },
    /// Linked evidence uses a different unit from a performance estimate.
    PerformanceEvidenceUnitMismatch {
        /// Evidence identifier.
        evidence_id: String,
        /// Performance comparison unit.
        performance_unit: String,
        /// Evidence unit.
        evidence_unit: String,
    },
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
    /// Two evidence records within one candidate have the same stable identifier.
    DuplicateEvidenceId(String),
    /// Evidence provenance source identity is incomplete.
    EmptySourceIdentity,
    /// Assessment subject identity is incomplete.
    EmptyAssessmentSubject,
    /// External source admission reference is incomplete.
    EmptySourceAdmissionReference,
    /// Freshness policy identity is incomplete.
    EmptyFreshnessPolicyIdentity,
    /// Freshness policy contains no rules.
    EmptyFreshnessPolicy,
    /// A freshness policy was supplied without an assessment timestamp.
    FreshnessPolicyRequiresAssessmentTimestamp,
    /// External source admission validity bounds are inverted.
    InvalidSourceAdmissionValidity { from: i64, until: i64 },
    /// A requirement set contains no requirements.
    EmptyRequirementSet,
    /// A requirement-set map key does not match the requirement's stable ID.
    RequirementSetKeyMismatch { key: String, requirement_id: String },
    /// Simulated or derived evidence lacks reproducible derivation provenance.
    MissingDerivationMetadata(EvidenceKind),
    /// Derivation metadata has incomplete identity.
    EmptyDerivationIdentity,
    /// Derivation metadata contains an empty input reference.
    EmptyDerivationInput,
    /// Derivation metadata contains an empty configuration hash.
    EmptyDerivationConfigurationHash,
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
            Self::InvalidEvidenceValidity { from, until } => {
                write!(f, "evidence validity [{from}, {until}] is inverted")
            }
            Self::EmptyPerformanceScale => write!(f, "performance unit/scope is empty"),
            Self::EmptyFunctionalConstraints => {
                write!(f, "functional requirement has no performance constraints")
            }
            Self::MissingComparisonScale(dimension) => {
                write!(f, "missing comparison scale for {dimension:?}")
            }
            Self::MissingPerformanceScale(metric) => {
                write!(f, "missing performance scale for {metric}")
            }
            Self::EmptyOperatingScale => {
                write!(f, "operating condition unit/scope is empty")
            }
            Self::EmptyOperatingCondition => {
                write!(f, "operating condition identity is empty")
            }
            Self::PerformanceEvidenceScopeMismatch {
                evidence_id,
                performance_scope,
                evidence_scope,
            } => write!(
                f,
                "evidence {evidence_id} scope {evidence_scope} does not match performance scope {performance_scope}"
            ),
            Self::PerformanceEvidenceUnitMismatch {
                evidence_id,
                performance_unit,
                evidence_unit,
            } => write!(
                f,
                "evidence {evidence_id} unit {evidence_unit} does not match performance unit {performance_unit}"
            ),
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
            Self::DuplicateEvidenceId(id) => write!(f, "duplicate evidence id {id}"),
            Self::EmptySourceIdentity => write!(f, "evidence source identity is incomplete"),
            Self::EmptyAssessmentSubject => write!(f, "assessment subject identity is incomplete"),
            Self::EmptySourceAdmissionReference => {
                write!(f, "source admission reference is incomplete")
            }
            Self::EmptyFreshnessPolicyIdentity => {
                write!(f, "freshness policy identity is incomplete")
            }
            Self::EmptyFreshnessPolicy => write!(f, "freshness policy has no rules"),
            Self::FreshnessPolicyRequiresAssessmentTimestamp => {
                write!(f, "freshness policy requires an assessment timestamp")
            }
            Self::InvalidSourceAdmissionValidity { from, until } => {
                write!(f, "source admission validity [{from}, {until}] is inverted")
            },
            Self::EmptyRequirementSet => write!(f, "requirement set is empty"),
            Self::RequirementSetKeyMismatch { key, requirement_id } => write!(
                f,
                "requirement set key {key} does not match requirement id {requirement_id}"
            ),
            Self::MissingDerivationMetadata(kind) => {
                write!(f, "evidence kind {kind:?} requires derivation metadata")
            }
            Self::EmptyDerivationIdentity => write!(f, "derivation identity is incomplete"),
            Self::EmptyDerivationInput => write!(f, "derivation input reference is empty"),
            Self::EmptyDerivationConfigurationHash => {
                write!(f, "derivation configuration hash is empty")
            }
        }
    }
}

impl std::error::Error for AssessmentError {}

/// Evidence-first alternatives assessment engine.
#[derive(Debug, Default, Clone, Copy)]
pub struct AlternativesEngine;

impl AlternativesEngine {
    /// Evaluate one candidate set against every requirement in a requirement set.
    ///
    /// Joint eligibility is the intersection of the individual requirement
    /// eligibility sets. Pareto frontiers remain per-requirement because
    /// requirements may legitimately use different functional/lifecycle scopes.
    pub fn assess_requirement_set(
        &self,
        requirement_set: &FunctionalRequirementSet,
        candidates: &[CandidatePathway],
        incumbent_id: Option<&str>,
        assessed_at_epoch_seconds: Option<i64>,
    ) -> Result<RequirementSetAssessment, AssessmentError> {
        self.assess_requirement_set_with_freshness(
            requirement_set,
            candidates,
            incumbent_id,
            assessed_at_epoch_seconds,
            None,
        )
    }

    /// Evaluate a requirement set with an explicit evidence-freshness policy.
    pub fn assess_requirement_set_with_freshness(
        &self,
        requirement_set: &FunctionalRequirementSet,
        candidates: &[CandidatePathway],
        incumbent_id: Option<&str>,
        assessed_at_epoch_seconds: Option<i64>,
        freshness_policy: Option<&EvidenceFreshnessPolicy>,
    ) -> Result<RequirementSetAssessment, AssessmentError> {
        requirement_set.validate()?;
        if freshness_policy.is_some() && assessed_at_epoch_seconds.is_none() {
            return Err(AssessmentError::FreshnessPolicyRequiresAssessmentTimestamp);
        }
        if let Some(policy) = freshness_policy {
            policy.validate()?;
        }
        let mut assessments = BTreeMap::new();
        for (requirement_id, requirement) in &requirement_set.requirements {
            let assessment = self.assess_at_with_freshness(
                requirement,
                candidates,
                incumbent_id,
                assessed_at_epoch_seconds,
                freshness_policy,
            )?;
            assessments.insert(requirement_id.clone(), assessment);
        }

        let mut candidate_ids = candidates
            .iter()
            .map(|candidate| candidate.id.clone())
            .collect::<Vec<_>>();
        candidate_ids.sort();
        candidate_ids.dedup();

        let mut jointly_eligible_candidate_ids = candidate_ids
            .iter()
            .filter(|candidate_id| {
                assessments
                    .values()
                    .all(|assessment| !assessment.frontier_blockers.contains_key(*candidate_id))
            })
            .cloned()
            .collect::<Vec<_>>();
        jointly_eligible_candidate_ids.sort();

        let mut blockers = BTreeMap::<String, Vec<RequirementSetBlocker>>::new();
        for (requirement_id, assessment) in &assessments {
            for (candidate_id, candidate_blockers) in &assessment.frontier_blockers {
                blockers
                    .entry(candidate_id.clone())
                    .or_default()
                    .extend(candidate_blockers.iter().cloned().map(|blocker| {
                        RequirementSetBlocker {
                            requirement_id: requirement_id.clone(),
                            blocker,
                        }
                    }));
            }
        }

        let mut joint_qualification = BTreeMap::new();
        for candidate_id in &candidate_ids {
            let minimum = assessments
                .values()
                .filter_map(|assessment| {
                    assessment
                        .candidates
                        .iter()
                        .find(|candidate| candidate.candidate_id == *candidate_id)
                        .map(|candidate| candidate.qualification)
                })
                .min()
                .unwrap_or(QualificationState::Hypothesis);
            joint_qualification.insert(candidate_id.clone(), minimum);
        }

        let mut result = RequirementSetAssessment {
            schema_version: REQUIREMENT_SET_SCHEMA_VERSION,
            algorithm_version: REQUIREMENT_SET_ALGORITHM_VERSION.into(),
            assessments,
            jointly_eligible_candidate_ids,
            joint_qualification,
            blockers,
            receipt: AssessmentReceipt {
                schema_version: REQUIREMENT_SET_SCHEMA_VERSION,
                algorithm_version: REQUIREMENT_SET_ALGORITHM_VERSION.into(),
                payload_hash: String::new(),
            },
        };
        let mut payload = result.clone();
        payload.receipt.payload_hash.clear();
        let bytes = serde_json::to_vec(&payload).map_err(|_| AssessmentError::NonFinite)?;
        let mut hasher = Hasher::new();
        hasher.update(&bytes);
        result.receipt.payload_hash = hasher.finalize().to_hex().to_string();
        Ok(result)
    }

    /// Evaluate candidates against a functional requirement.
    pub fn assess(
        &self,
        requirement: &FunctionalRequirement,
        candidates: &[CandidatePathway],
        incumbent_id: Option<&str>,
    ) -> Result<AssessmentResult, AssessmentError> {
        self.assess_at(requirement, candidates, incumbent_id, None)
    }

    /// Evaluate candidates at an explicit Unix timestamp.
    ///
    /// Time-bounded evidence is used only when valid at the supplied timestamp.
    /// An assessment without a timestamp conservatively excludes any evidence
    /// with an explicit validity window.
    pub fn assess_at(
        &self,
        requirement: &FunctionalRequirement,
        candidates: &[CandidatePathway],
        incumbent_id: Option<&str>,
        assessed_at_epoch_seconds: Option<i64>,
    ) -> Result<AssessmentResult, AssessmentError> {
        self.assess_at_with_freshness(
            requirement,
            candidates,
            incumbent_id,
            assessed_at_epoch_seconds,
            None,
        )
    }

    /// Evaluate candidates at an explicit timestamp with an explicit freshness policy.
    pub fn assess_at_with_freshness(
        &self,
        requirement: &FunctionalRequirement,
        candidates: &[CandidatePathway],
        incumbent_id: Option<&str>,
        assessed_at_epoch_seconds: Option<i64>,
        freshness_policy: Option<&EvidenceFreshnessPolicy>,
    ) -> Result<AssessmentResult, AssessmentError> {
        requirement.validate()?;
        if freshness_policy.is_some() && assessed_at_epoch_seconds.is_none() {
            return Err(AssessmentError::FreshnessPolicyRequiresAssessmentTimestamp);
        }
        if let Some(policy) = freshness_policy {
            policy.validate()?;
        };
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
            for estimate in candidate.performance.values_mut() {
                estimate.evidence_ids.sort();
            }
            for estimate in candidate.operating_capabilities.values_mut() {
                estimate.evidence_ids.sort();
            }
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
                .map(|(metric, bound)| {
                    let status = match (
                        candidate.performance.get(metric),
                        requirement.performance_scales.get(metric),
                    ) {
                        (Some(estimate), Some(scale))
                            if estimate.unit == scale.unit
                                && estimate.scope == scale.scope
                                && candidate.performance_evidence_is_supported_at(metric, assessed_at_epoch_seconds, freshness_policy) =>
                        {
                            bound.check(Some(estimate.interval))
                        }
                        _ => ConstraintStatus::Unresolved,
                    };
                    ConstraintEvaluation {
                        metric: metric.clone(),
                        requirement: *bound,
                        status,
                    }
                })
                .collect::<Vec<_>>();

            let mut candidate_blockers = Vec::new();
            for (metric, scale) in &requirement.performance_scales {
                if let Some(estimate) = candidate.performance.get(metric)
                    && (estimate.unit != scale.unit || estimate.scope != scale.scope)
                {
                    candidate_blockers.push(FrontierBlocker::PerformanceIncompatibleScale {
                        metric: metric.clone(),
                        expected_unit: scale.unit.clone(),
                        actual_unit: estimate.unit.clone(),
                        expected_scope: scale.scope.clone(),
                        actual_scope: estimate.scope.clone(),
                    });
                }
            }
            for (condition, required) in &requirement.operating_envelope {
                match candidate.operating_capabilities.get(condition) {
                    None => candidate_blockers.push(FrontierBlocker::OperatingConditionUnresolved(
                        condition.clone(),
                    )),
                    Some(capability)
                        if capability.unit != required.unit || capability.scope != required.scope =>
                    {
                        candidate_blockers.push(FrontierBlocker::PerformanceIncompatibleScale {
                            metric: format!("operating:{condition}"),
                            expected_unit: required.unit.clone(),
                            actual_unit: capability.unit.clone(),
                            expected_scope: required.scope.clone(),
                            actual_scope: capability.scope.clone(),
                        });
                    }
                    Some(capability)
                        if capability.interval.lower > required.interval.lower
                            || capability.interval.upper < required.interval.upper =>
                    {
                        candidate_blockers.push(FrontierBlocker::OperatingConditionFailed(
                            condition.clone(),
                        ));
                    }
                    Some(capability)
                        if !candidate.operating_evidence_is_supported_at(
                            condition,
                            assessed_at_epoch_seconds,
                            freshness_policy,
                        ) =>
                    {
                        candidate_blockers.push(FrontierBlocker::OperatingConditionUnresolved(
                            condition.clone(),
                        ));
                    }
                    Some(_) => {}
                }
            }
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
                    (Some(estimate), Some(_))
                        if !estimate.evidence_ids.is_empty()
                            && candidate
                                .linked_evidence_at(
                                    &estimate.evidence_ids,
                                    assessed_at_epoch_seconds,
                                )
                                .next()
                                .is_none() =>
                    {
                        candidate_blockers.push(FrontierBlocker::EvidenceUnavailable(dimension));
                    }
                    _ => {}
                }
            }

            let frontier_blocked = !candidate_blockers.is_empty();
d::ContinuouslyMonitored
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
                performance: candidate.performance.clone(),
                operating_capabilities: candidate.operating_capabilities.clone(),
                constraints,
                burdens: candidate.burdens.clone(),
                qualification: candidate.qualification_ceiling(
                    requirement,
                    assessed_at_epoch_seconds,
                    freshness_policy,
                ),
                evidence_conflict: candidate.has_conflict_at(
                    assessed_at_epoch_seconds,
                    freshness_policy,
                ),
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
            assessed_at_epoch_seconds,
            freshness_policy: freshness_policy.cloned(),
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
            source: EvidenceSourceIdentity {
                authority_id: source_id.into(),
                artifact_id: format!("artifact:{id}"),
                artifact_digest: format!("fixture-digest:{id}"),
                issuer_key_fingerprint: None,
                admission: None,
            },
            scope: "synthetic functional unit".into(),
            unit: Some("unit".into()),
            as_of: Some("fixture-v1".into()),
            valid_from_epoch_seconds: None,
            valid_until_epoch_seconds: None,
            derivation: matches!(kind, EvidenceKind::Simulated | EvidenceKind::Derived).then(
                || DerivationRecord {
                    method_id: "synthetic-fixture".into(),
                    method_version: "fixture-v1".into(),
                    input_refs: vec!["fixture-input".into()],
                    configuration_hash: Some("fixture-config-v1".into()),
                },
            ),
        }
    }

    fn fixture_requirement() -> FunctionalRequirement {
        FunctionalRequirement {
            id: "seal-v1".into(),
            subject: AssessmentSubjectRef {
                subject_id: "fixture-product".into(),
                profile_id: "fixture-product-profile".into(),
                profile_revision: "v1".into(),
                subject_digest: "fixture-product-digest".into(),
            },
            description: "Provide a durable chemical-resistant seal.".into(),
            constraints: BTreeMap::from([
                ("service_life_years".into(), RequirementBound::AtLeast(10.0)),
                ("throughput_per_hour".into(), RequirementBound::AtLeast(100.0)),
            ]),
            comparison_scales: Dimension::ALL
                .into_iter()
                .map(|dimension| {
                    (
                        dimension,
                        ComparisonScale {
                            unit: "unit".into(),
                            scope: "synthetic functional unit".into(),
                        },
                    )
                })
                .collect(),
            performance_scales: BTreeMap::from([
                (
                    "service_life_years".into(),
                    ComparisonScale {
                        unit: "unit".into(),
                        scope: "synthetic functional unit".into(),
                    },
                ),
                (
                    "throughput_per_hour".into(),
                    ComparisonScale {
                        unit: "unit".into(),
                        scope: "synthetic functional unit".into(),
                    },
                ),
            ]),
            operating_envelope: BTreeMap::from([
                (
                    "temperature".into(),
                    OperatingRequirement {
                        interval: Interval::new(-20.0, 80.0).unwrap(),
                        unit: "unit".into(),
                        scope: "synthetic functional unit".into(),
                    },
                ),
                (
                    "pressure".into(),
                    OperatingRequirement {
                        interval: Interval::new(0.5, 10.0).unwrap(),
                        unit: "unit".into(),
                        scope: "synthetic functional unit".into(),
                    },
                ),
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
        performance.insert(
            "service_life_years".into(),
            PerformanceEstimate {
                interval: Interval::point(12.0).unwrap(),
                unit: "unit".into(),
                scope: "synthetic functional unit".into(),
                evidence_ids: evidence_ids.iter().map(|id| (*id).to_string()).collect(),
            },
        );
        performance.insert(
            "throughput_per_hour".into(),
            PerformanceEstimate {
                interval: Interval::point(120.0).unwrap(),
                unit: "unit".into(),
                scope: "synthetic functional unit".into(),
                evidence_ids: evidence_ids.iter().map(|id| (*id).to_string()).collect(),
            },
        );

        let operating_capabilities = BTreeMap::from([
            (
                "temperature".into(),
                PerformanceEstimate {
                    interval: Interval::new(-40.0, 120.0).unwrap(),
                    unit: "unit".into(),
                    scope: "synthetic functional unit".into(),
                    evidence_ids: evidence_ids.iter().map(|id| (*id).to_string()).collect(),
                },
            ),
            (
                "pressure".into(),
                PerformanceEstimate {
                    interval: Interval::new(0.1, 20.0).unwrap(),
                    unit: "unit".into(),
                    scope: "synthetic functional unit".into(),
                    evidence_ids: evidence_ids.iter().map(|id| (*id).to_string()).collect(),
                },
            ),
        ]);
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
            operating_capabilities,
            evidence,
        }
    }

    #[test]
    fn requirement_set_intersects_functional_eligibility() {
        let candidate = candidate(
            "multi-function",
            PathwayKind::ProductRedesign,
            2.0,
            2.0,
            vec![evidence(
                "m1",
                "source",
                EvidenceKind::Observed,
                EvidenceStance::Supports,
                0.9,
            )],
        );
        let mut safe = fixture_requirement();
        safe.id = "safety-function".into();
        let mut incompatible = fixture_requirement();
        incompatible.id = "throughput-function".into();
        incompatible
            .constraints
            .insert("throughput_per_hour".into(), RequirementBound::AtLeast(125.0));
        let requirements = FunctionalRequirementSet::new(BTreeMap::from([
            (safe.id.clone(), safe),
            (incompatible.id.clone(), incompatible),
        ]))
        .unwrap();

        let result = AlternativesEngine
            .assess_requirement_set(&requirements, &[candidate], None, None)
            .unwrap();

        assert!(result.assessments["safety-function"].frontier_blockers.is_empty());
        assert!(result.assessments["throughput-function"].frontier_blockers.contains_key("multi-function"));
        assert!(!result.jointly_eligible_candidate_ids.contains(&"multi-function".into()));
        assert!(result.blockers["multi-function"]
            .iter()
            .any(|blocker| blocker.requirement_id == "throughput-function"));
    }

    #[test]
    fn requirement_set_receipt_is_order_independent() {
        let a = candidate(
            "a",
            PathwayKind::ProcessSubstitution,
            2.0,
            2.0,
            vec![evidence("a1", "a", EvidenceKind::Observed, EvidenceStance::Supports, 0.9)],
        );
        let b = candidate(
            "b",
            PathwayKind::ProcessSubstitution,
            3.0,
            3.0,
            vec![evidence("b1", "b", EvidenceKind::Observed, EvidenceStance::Supports, 0.9)],
        );
        let requirements = FunctionalRequirementSet::new(BTreeMap::from([
            ("seal-v1".into(), fixture_requirement()),
        ]))
        .unwrap();

        let first = AlternativesEngine
            .assess_requirement_set(&requirements, &[a.clone(), b.clone()], None, None)
            .unwrap();
        let second = AlternativesEngine
            .assess_requirement_set(&requirements, &[b, a], None, None)
            .unwrap();

        assert_eq!(first, second);
        assert!(!first.receipt.payload_hash.is_empty());
    }

    #[test]
    fn overlapping_performance_interval_cannot_satisfy_requirement() {
        let mut c = candidate(
            "uncertain-performance",
            PathwayKind::ProcessSubstitution,
            2.0,
            2.0,
            vec![evidence(
                "p1",
                "source",
                EvidenceKind::Observed,
                EvidenceStance::Supports,
                0.9,
            )],
        );
        c.performance.get_mut("service_life_years").unwrap().interval =
            Interval::new(8.0, 12.0).unwrap();

        let result = AlternativesEngine
            .assess(&fixture_requirement(), &[c], None)
            .unwrap();

        assert_eq!(
            result.candidates[0].constraints[0].status,
            ConstraintStatus::Unresolved
        );
        assert!(result.frontier_blockers.contains_key("uncertain-performance"));
    }

    #[test]
    fn one_field_source_cannot_promote_field_qualification() {
        let mut c = candidate(
            "single-field-source",
            PathwayKind::ProcessSubstitution,
            1.0,
            1.0,
            vec![evidence(
                "field",
                "field-source",
                EvidenceKind::FieldObserved,
                EvidenceStance::Supports,
                0.95,
            )],
        );
        for estimate in c.burdens.values_mut() {
            estimate.evidence_ids = vec!["field".into()];
        }

        let result = AlternativesEngine
            .assess(&fixture_requirement(), &[c], None)
            .unwrap();

        assert_ne!(
            result.candidates[0].qualification,
            QualificationState::FieldQualified
        );
    }

    #[test]
    fn missing_operating_capability_blocks_frontier() {
        let mut c = candidate(
            "missing-envelope",
            PathwayKind::ProcessSubstitution,
            2.0,
            2.0,
            vec![evidence(
                "o1",
                "source",
                EvidenceKind::Observed,
                EvidenceStance::Supports,
                0.9,
            )],
        );
        c.operating_capabilities.remove("pressure");

        let result = AlternativesEngine
            .assess(&fixture_requirement(), &[c], None)
            .unwrap();

        assert!(result.frontier_blockers["missing-envelope"]
            .iter()
            .any(|b| matches!(
                b,
                FrontierBlocker::OperatingConditionUnresolved(name)
                    if name == "pressure"
            )));
    }

    #[test]
    fn insufficient_operating_capability_blocks_frontier() {
        let mut c = candidate(
            "narrow-envelope",
            PathwayKind::ProcessSubstitution,
            2.0,
            2.0,
            vec![evidence(
                "o1",
                "source",
                EvidenceKind::Observed,
                EvidenceStance::Supports,
                0.9,
            )],
        );
        c.operating_capabilities
            .get_mut("temperature")
            .unwrap()
            .interval = Interval::new(0.0, 60.0).unwrap();

        let result = AlternativesEngine
            .assess(&fixture_requirement(), &[c], None)
            .unwrap();

        assert!(result.frontier_blockers["narrow-envelope"]
            .iter()
            .any(|b| matches!(
                b,
                FrontierBlocker::OperatingConditionFailed(name)
                    if name == "temperature"
            )));
    }

    #[test]
    fn expired_evidence_cannot_satisfy_functional_constraint() {
        let mut c = candidate(
            "expired",
            PathwayKind::ProcessSubstitution,
            2.0,
            2.0,
            vec![evidence(
                "e1",
                "source",
                EvidenceKind::Observed,
                EvidenceStance::Supports,
                0.9,
            )],
        );
        c.evidence[0].valid_until_epoch_seconds = Some(100);

        let current = AlternativesEngine
            .assess_at(&fixture_requirement(), &[c.clone()], None, Some(200))
            .unwrap();
        assert!(current.frontier_blockers.contains_key("expired"));
        assert_eq!(
            current.candidates[0].qualification,
            QualificationState::Hypothesis
        );

        let valid = AlternativesEngine
            .assess_at(&fixture_requirement(), &[c.clone()], None, Some(50))
            .unwrap();
        assert!(!valid.frontier_blockers.contains_key("expired"));
        assert!(
            valid
                .candidates[0]
                .constraints
                .iter()
                .all(|c| c.status == ConstraintStatus::Pass)
        );
        let timeless = AlternativesEngine
            .assess(&fixture_requirement(), &[c], None)
            .unwrap();
        assert!(timeless.frontier_blockers.contains_key("expired"));
    }

    #[test]
    fn stale_burden_evidence_blocks_burden_comparison() {
        let mut c = candidate(
            "stale-burden",
            PathwayKind::ProcessSubstitution,
            2.0,
            2.0,
            vec![
                evidence(
                    "current",
                    "source-current",
                    EvidenceKind::Observed,
                    EvidenceStance::Supports,
                    0.9,
                ),
                evidence(
                    "stale",
                    "source-stale",
                    EvidenceKind::Observed,
                    EvidenceStance::Supports,
                    0.9,
                ),
            ],
        );
        c.evidence
            .iter_mut()
            .find(|e| e.id == "stale")
            .unwrap()
            .valid_until_epoch_seconds = Some(100);
        for estimate in c.burdens.values_mut() {
            estimate.evidence_ids = vec!["stale".into()];
        }
        for estimate in c.performance.values_mut() {
            estimate.evidence_ids = vec!["current".into()];
        }

        let result = AlternativesEngine
            .assess_at(&fixture_requirement(), &[c], None, Some(200))
            .unwrap();

        assert!(result.frontier_blockers["stale-burden"]
            .iter()
            .any(|blocker| matches!(blocker, FrontierBlocker::EvidenceUnavailable(_))));
    }

    #[test]
    fn inverted_evidence_validity_window_is_rejected() {
        let mut c = candidate(
            "invalid-validity",
            PathwayKind::ProcessSubstitution,
            2.0,
            2.0,
            vec![evidence(
                "e1",
                "source",
                EvidenceKind::Observed,
                EvidenceStance::Supports,
                0.9,
            )],
        );
        c.evidence[0].valid_from_epoch_seconds = Some(200);
        c.evidence[0].valid_until_epoch_seconds = Some(100);

        let error = AlternativesEngine
            .assess(&fixture_requirement(), &[c], None)
            .unwrap_err();
        assert!(matches!(
            error,
            AssessmentError::InvalidEvidenceValidity {
                from: 200,
                until: 100
            }
        ));
    }

    #[test]
    fn missing_performance_evidence_blocks_functional_constraint() {
        let mut c = candidate(
            "unverified-performance",
            PathwayKind::ProcessSubstitution,
            2.0,
            2.0,
            vec![],
        );
        for estimate in c.performance.values_mut() {
            estimate.evidence_ids.clear();
        }

        let result = AlternativesEngine
            .assess(&fixture_requirement(), &[c], None)
            .unwrap();

        assert_eq!(
            result.candidates[0].qualification,
            QualificationState::Hypothesis
        );
        assert!(matches!(
            result.frontier_blockers["unverified-performance"][0],
            FrontierBlocker::ConstraintUnresolved(_)
        ));
    }

    #[test]
    fn incompatible_performance_scale_blocks_candidate() {
        let mut c = candidate(
            "performance-scale-drift",
            PathwayKind::ProcessSubstitution,
            2.0,
            2.0,
            vec![evidence("p1", "source", EvidenceKind::Observed, EvidenceStance::Supports, 0.9)],
        );
        c.performance.get_mut("throughput_per_hour").unwrap().unit = "other-unit".into();

        let result = AlternativesEngine
            .assess(&fixture_requirement(), &[c], None)
            .unwrap();

        assert!(result.frontier_blockers["performance-scale-drift"]
            .iter()
            .any(|blocker| matches!(
                blocker,
                FrontierBlocker::PerformanceIncompatibleScale {
                    metric,
                    ..
                } if metric == "throughput_per_hour"
            )));
        assert!(!result.pareto_frontier.contains(&"performance-scale-drift".into()));
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
            QualificationState::EvidenceSupported
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
    fn changing_assessment_subject_changes_receipt_identity() {
        let requirement = fixture_requirement();
        let candidate = candidate(
            "subject-bound",
            PathwayKind::ProcessSubstitution,
            2.0,
            2.0,
            vec![evidence("s1", "source", EvidenceKind::Observed, EvidenceStance::Supports, 0.9)],
        );
        let baseline = AlternativesEngine
            .assess(&requirement, &[candidate.clone()], None)
            .unwrap()
            .receipt
            .payload_hash;

        let mut changed_requirement = requirement;
        changed_requirement.subject.subject_digest = "different-design-digest".into();
        let changed = AlternativesEngine
            .assess(&changed_requirement, &[candidate], None)
            .unwrap()
            .receipt
            .payload_hash;

        assert_ne!(baseline, changed);
    }

    #[test]
    fn source_admission_changes_assessment_identity() {
        let mut evidence = evidence(
            "admitted",
            "authority",
            EvidenceKind::Observed,
            EvidenceStance::Supports,
            0.9,
        );
        let mut c = candidate(
            "admitted-candidate",
            PathwayKind::ProcessSubstitution,
            2.0,
            2.0,
            vec![evidence.clone()],
        );
        let baseline = AlternativesEngine
            .assess(&fixture_requirement(), &[c.clone()], None)
            .unwrap()
            .receipt
            .payload_hash;

        evidence.source.admission = Some(SourceAdmissionRef {
            policy_id: "policy".into(),
            policy_revision: "r1".into(),
            policy_digest: "policy-digest".into(),
            admission_id: "admission".into(),
            authority_epoch: "epoch-1".into(),
            fault_domain_id: Some("domain-a".into()),
            valid_from_epoch_seconds: Some(100),
            valid_until_epoch_seconds: Some(200),
        });
        c.evidence[0] = evidence;

        let admitted = AlternativesEngine
            .assess(&fixture_requirement(), &[c], None)
            .unwrap()
            .receipt
            .payload_hash;
        assert_ne!(baseline, admitted);
    }

    #[test]
    fn source_admission_reference_validates_without_claiming_authenticity() {
        let mut source = EvidenceSourceIdentity {
            authority_id: "authority".into(),
            artifact_id: "artifact".into(),
            artifact_digest: "digest".into(),
            issuer_key_fingerprint: None,
            admission: Some(SourceAdmissionRef {
                policy_id: "policy".into(),
                policy_revision: "r1".into(),
                policy_digest: "policy-digest".into(),
                admission_id: "admission".into(),
                authority_epoch: "epoch-1".into(),
                fault_domain_id: Some("domain-a".into()),
                valid_from_epoch_seconds: Some(100),
                valid_until_epoch_seconds: Some(200),
            }),
        };
        assert!(source.validate().is_ok());
        source.admission.as_mut().unwrap().valid_until_epoch_seconds = Some(50);
        assert!(matches!(
            source.validate().unwrap_err(),
            AssessmentError::InvalidSourceAdmissionValidity { from: 100, until: 50 }
        ));
    }

    #[test]
    fn duplicate_evidence_id_fails_closed() {
        let mut c = candidate(
            "duplicate-evidence",
            PathwayKind::ProcessSubstitution,
            2.0,
            2.0,
            vec![
                evidence("same", "source-a", EvidenceKind::Observed, EvidenceStance::Supports, 0.9),
                evidence("same", "source-b", EvidenceKind::Reported, EvidenceStance::Supports, 0.9),
            ],
        );

        let error = AlternativesEngine.assess(&fixture_requirement(), &[c.clone()], None).unwrap_err();
        assert_eq!(error, AssessmentError::DuplicateEvidenceId("same".into()));

        c.evidence.reverse();
        let error = AlternativesEngine.assess(&fixture_requirement(), &[c], None).unwrap_err();
        assert_eq!(error, AssessmentError::DuplicateEvidenceId("same".into()));
    }

    #[test]
    fn source_diversity_is_based_on_authority_identity() {
        let a = EvidenceSourceIdentity {
            authority_id: "authority-a".into(),
            artifact_id: "artifact-1".into(),
            artifact_digest: "digest-1".into(),
            issuer_key_fingerprint: None,
            admission: None,
        };
        let b = EvidenceSourceIdentity {
            authority_id: "authority-a".into(),
            artifact_id: "artifact-2".into(),
            artifact_digest: "digest-2".into(),
            issuer_key_fingerprint: None,
            admission: None,
        };
        let c = EvidenceSourceIdentity {
            authority_id: "authority-b".into(),
            artifact_id: "artifact-3".into(),
            artifact_digest: "digest-3".into(),
            issuer_key_fingerprint: None,
            admission: None,
        };

        assert_eq!(a.authority_group_id(), b.authority_group_id());
        assert_ne!(a.authority_group_id(), c.authority_group_id());

        let mut rotated_key = a.clone();
        rotated_key.issuer_key_fingerprint = Some("new-key".into());
        assert_eq!(a.authority_group_id(), rotated_key.authority_group_id());
    }

    #[test]
    fn simulated_evidence_requires_derivation_metadata() {
        let mut evidence = evidence(
            "simulated",
            "model",
            EvidenceKind::Simulated,
            EvidenceStance::Supports,
            0.8,
        );
        evidence.derivation = None;

        assert!(matches!(
            evidence.validate().unwrap_err(),
            AssessmentError::MissingDerivationMetadata(EvidenceKind::Simulated)
        ));
    }

    #[test]
    fn derived_evidence_rejects_empty_derivation_inputs() {
        let mut evidence = evidence(
            "derived",
            "model",
            EvidenceKind::Derived,
            EvidenceStance::Supports,
            0.8,
        );
        evidence.derivation.as_mut().unwrap().input_refs.clear();

        assert!(matches!(
            evidence.validate().unwrap_err(),
            AssessmentError::EmptyDerivationIdentity
        ));
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
    fn requirement_must_declare_every_comparison_scale() {
        let mut requirement = fixture_requirement();
        requirement.comparison_scales.remove(&Dimension::Water);

        let error = AlternativesEngine
            .assess(&requirement, &[], None)
            .unwrap_err();

        assert!(matches!(
            error,
            AssessmentError::MissingComparisonScale(Dimension::Water)
        ));
    }

    #[test]
    fn candidate_cannot_define_its_own_comparison_scale() {
        let mut c = candidate(
            "cohort-authority",
            PathwayKind::ProcessSubstitution,
            2.0,
            2.0,
            vec![evidence(
                "x1",
                "source",
                EvidenceKind::Observed,
                EvidenceStance::Supports,
                0.9,
            )],
        );
        let water = c.burdens.get_mut(&Dimension::Water).unwrap();
        water.unit = "candidate-defined-unit".into();
        water.evidence_ids.clear();

        let result = AlternativesEngine::assess(&fixture_requirement(), &[c], None).unwrap();

        assert!(matches!(
            result.frontier_blockers["cohort-authority"][0],
            FrontierBlocker::IncompatibleScale {
                dimension: Dimension::Water,
                ..
            }
        ));
    }

    #[test]
    fn overlapping_intervals_remain_incomparable() {
        let a = Interval::new(1.0, 3.0).unwrap();
        let b = Interval::new(2.0, 4.0).unwrap();

        assert!(!a.clearly_no_worse_than(&b));
        assert!(!b.clearly_no_worse_than(&a));
        assert!(!a.clearly_better_than(&b));
        assert!(!b.clearly_better_than(&a));
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
