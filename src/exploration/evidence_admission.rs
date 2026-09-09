// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evidence-aware admission for generativity Pareto comparison.
//!
//! Reported confidence and evidence support remain separate. This module does not infer
//! confidence from evidence counts. Instead, it can require every compared dimension to
//! name evidence and to have an explicit minimum number of independently verified
//! evidence-plane runs before the candidate reaches the Pareto archive.

#![deny(unsafe_code)]

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use super::archive::{
    ArchiveEntry, ArchiveInsertOutcome, ArchiveValidationError, GenerativityParetoArchive,
    ParetoPolicy,
};
use super::generativity::{GenerativityEstimate, GenerativityVector};
use super::persisted_evidence::VerifiedGenerativityBundle;

pub const MAX_EVIDENCE_IDS_PER_DIMENSION: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GenerativityDimension {
    ImmediateUtility,
    EpistemicGain,
    OptionValue,
    Diversity,
    CapabilityGain,
    Diffusion,
    CommonsGain,
    Regeneration,
    DependencyRisk,
    ConcentrationRisk,
    IrreversibilityRisk,
}

impl GenerativityDimension {
    pub const ALL: [Self; 11] = [
        Self::ImmediateUtility,
        Self::EpistemicGain,
        Self::OptionValue,
        Self::Diversity,
        Self::CapabilityGain,
        Self::Diffusion,
        Self::CommonsGain,
        Self::Regeneration,
        Self::DependencyRisk,
        Self::ConcentrationRisk,
        Self::IrreversibilityRisk,
    ];

    pub fn estimate(self, vector: &GenerativityVector) -> GenerativityEstimate {
        match self {
            Self::ImmediateUtility => vector.immediate_utility,
            Self::EpistemicGain => vector.epistemic_gain,
            Self::OptionValue => vector.option_value,
            Self::Diversity => vector.diversity,
            Self::CapabilityGain => vector.capability_gain,
            Self::Diffusion => vector.diffusion,
            Self::CommonsGain => vector.commons_gain,
            Self::Regeneration => vector.regeneration,
            Self::DependencyRisk => vector.dependency_risk,
            Self::ConcentrationRisk => vector.concentration_risk,
            Self::IrreversibilityRisk => vector.irreversibility_risk,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DimensionEvidenceBinding {
    pub dimension: GenerativityDimension,
    pub evidence_ids: Vec<String>,
}

/// Evidence support policy. It never rewrites a dimension's reported confidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceAdmissionPolicy {
    pub min_qualified_runs_per_dimension: usize,
    pub max_evidence_ids_per_dimension: usize,
}

impl Default for EvidenceAdmissionPolicy {
    fn default() -> Self {
        Self {
            min_qualified_runs_per_dimension: 1,
            max_evidence_ids_per_dimension: 16,
        }
    }
}

impl EvidenceAdmissionPolicy {
    pub fn validate(&self) -> Result<(), EvidenceAdmissionError> {
        if self.max_evidence_ids_per_dimension == 0
            || self.max_evidence_ids_per_dimension > MAX_EVIDENCE_IDS_PER_DIMENSION
        {
            return Err(EvidenceAdmissionError::InvalidMaxEvidenceIds {
                value: self.max_evidence_ids_per_dimension,
            });
        }
        if self.min_qualified_runs_per_dimension > self.max_evidence_ids_per_dimension {
            return Err(EvidenceAdmissionError::ImpossibleQualifiedFloor {
                minimum: self.min_qualified_runs_per_dimension,
                max_ids: self.max_evidence_ids_per_dimension,
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DimensionSupport {
    pub dimension: GenerativityDimension,
    pub reported_confidence: f64,
    pub evidence_ids: Vec<String>,
    pub qualified_evidence_plane_runs: usize,
    pub violated_evidence_plane_runs: usize,
    pub generic_evidence_items: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EvidenceSupportReport {
    pub assessment_subject_id: String,
    pub dimensions: Vec<DimensionSupport>,
    /// Evidence IDs reused across dimensions. Reuse is visible, not counted as replication.
    pub shared_evidence_ids: Vec<String>,
}

/// Constructor-qualified candidate. Intentionally not deserializable.
#[derive(Debug, Clone, Serialize)]
pub struct EvidenceBackedArchiveCandidate {
    entry: ArchiveEntry,
    support: EvidenceSupportReport,
}

impl EvidenceBackedArchiveCandidate {
    pub fn entry(&self) -> &ArchiveEntry {
        &self.entry
    }

    pub fn support(&self) -> &EvidenceSupportReport {
        &self.support
    }
}

pub fn qualify_archive_candidate(
    entry_id: impl Into<String>,
    niche: impl Into<String>,
    bundle: &VerifiedGenerativityBundle,
    bindings: &[DimensionEvidenceBinding],
    policy: EvidenceAdmissionPolicy,
) -> Result<EvidenceBackedArchiveCandidate, EvidenceAdmissionError> {
    policy.validate()?;
    let assessment = bundle.assessment();
    assessment.validate().map_err(ArchiveValidationError::from)?;

    let mut evidence_by_id = HashMap::new();
    for evidence in &assessment.evidence {
        if evidence_by_id
            .insert(evidence.evidence_id.as_str(), evidence)
            .is_some()
        {
            return Err(EvidenceAdmissionError::DuplicateAssessmentEvidenceId(
                evidence.evidence_id.clone(),
            ));
        }
    }

    let mut qualified_ids = HashSet::new();
    let mut violated_ids = HashSet::new();
    for capsule in bundle.capsules() {
        let id = format!("evidence-plane:{}", capsule.envelope().run_id());
        if capsule.envelope().integrity_satisfied() {
            qualified_ids.insert(id);
        } else {
            violated_ids.insert(id);
        }
    }

    let mut binding_by_dimension = HashMap::new();
    for binding in bindings {
        if binding_by_dimension
            .insert(binding.dimension, binding)
            .is_some()
        {
            return Err(EvidenceAdmissionError::DuplicateDimensionBinding(
                binding.dimension,
            ));
        }
    }

    let mut evidence_use_count: HashMap<&str, usize> = HashMap::new();
    let mut dimensions = Vec::with_capacity(GenerativityDimension::ALL.len());

    for dimension in GenerativityDimension::ALL {
        let binding = binding_by_dimension
            .get(&dimension)
            .ok_or(EvidenceAdmissionError::MissingDimensionBinding(dimension))?;
        if binding.evidence_ids.len() > policy.max_evidence_ids_per_dimension {
            return Err(EvidenceAdmissionError::TooManyEvidenceIds {
                dimension,
                count: binding.evidence_ids.len(),
                max: policy.max_evidence_ids_per_dimension,
            });
        }

        let mut local_ids = HashSet::new();
        let mut qualified = 0usize;
        let mut violated = 0usize;
        let mut generic = 0usize;

        for evidence_id in &binding.evidence_ids {
            if !local_ids.insert(evidence_id.as_str()) {
                return Err(EvidenceAdmissionError::DuplicateDimensionEvidenceId {
                    dimension,
                    evidence_id: evidence_id.clone(),
                });
            }
            if !evidence_by_id.contains_key(evidence_id.as_str()) {
                return Err(EvidenceAdmissionError::UnknownEvidenceId {
                    dimension,
                    evidence_id: evidence_id.clone(),
                });
            }

            *evidence_use_count.entry(evidence_id.as_str()).or_insert(0) += 1;
            if qualified_ids.contains(evidence_id) {
                qualified += 1;
            } else if violated_ids.contains(evidence_id) {
                violated += 1;
            } else {
                generic += 1;
            }
        }

        if qualified < policy.min_qualified_runs_per_dimension {
            return Err(EvidenceAdmissionError::InsufficientQualifiedSupport {
                dimension,
                required: policy.min_qualified_runs_per_dimension,
                observed: qualified,
            });
        }

        dimensions.push(DimensionSupport {
            dimension,
            reported_confidence: dimension.estimate(&assessment.vector).confidence,
            evidence_ids: binding.evidence_ids.clone(),
            qualified_evidence_plane_runs: qualified,
            violated_evidence_plane_runs: violated,
            generic_evidence_items: generic,
        });
    }

    let mut shared_evidence_ids = evidence_use_count
        .into_iter()
        .filter_map(|(id, count)| (count > 1).then(|| id.to_string()))
        .collect::<Vec<_>>();
    shared_evidence_ids.sort();

    let entry = ArchiveEntry::new(entry_id, niche, assessment.vector.clone());
    entry.validate()?;

    Ok(EvidenceBackedArchiveCandidate {
        entry,
        support: EvidenceSupportReport {
            assessment_subject_id: assessment.subject_id.clone(),
            dimensions,
            shared_evidence_ids,
        },
    })
}

/// Runtime-derived archive. Intentionally serializable but not deserializable; persisted
/// candidates must be rebuilt from verified bundles before they can be admitted again.
#[derive(Debug, Clone, Serialize)]
pub struct EvidenceAwareGenerativityArchive {
    inner: GenerativityParetoArchive,
    admission_policy: EvidenceAdmissionPolicy,
    support_by_entry_id: HashMap<String, EvidenceSupportReport>,
}

impl EvidenceAwareGenerativityArchive {
    pub fn new(
        pareto_policy: ParetoPolicy,
        admission_policy: EvidenceAdmissionPolicy,
    ) -> Result<Self, EvidenceAdmissionError> {
        admission_policy.validate()?;
        Ok(Self {
            inner: GenerativityParetoArchive::new(pareto_policy)?,
            admission_policy,
            support_by_entry_id: HashMap::new(),
        })
    }

    pub fn entries(&self) -> &[ArchiveEntry] {
        self.inner.entries()
    }

    pub fn support_for(&self, entry_id: &str) -> Option<&EvidenceSupportReport> {
        self.support_by_entry_id.get(entry_id)
    }

    pub fn qualify_and_insert(
        &mut self,
        entry_id: impl Into<String>,
        niche: impl Into<String>,
        bundle: &VerifiedGenerativityBundle,
        bindings: &[DimensionEvidenceBinding],
    ) -> Result<ArchiveInsertOutcome, EvidenceAdmissionError> {
        let candidate = qualify_archive_candidate(
            entry_id,
            niche,
            bundle,
            bindings,
            self.admission_policy,
        )?;
        self.insert(candidate)
    }

    pub fn insert(
        &mut self,
        candidate: EvidenceBackedArchiveCandidate,
    ) -> Result<ArchiveInsertOutcome, EvidenceAdmissionError> {
        let entry_id = candidate.entry.entry_id.clone();
        let outcome = self.inner.insert(candidate.entry)?;
        if let ArchiveInsertOutcome::Inserted { removed_entry_ids } = &outcome {
            for removed in removed_entry_ids {
                self.support_by_entry_id.remove(removed);
            }
            self.support_by_entry_id.insert(entry_id, candidate.support);
        }
        Ok(outcome)
    }
}

impl Default for EvidenceAwareGenerativityArchive {
    fn default() -> Self {
        Self::new(ParetoPolicy::default(), EvidenceAdmissionPolicy::default())
            .expect("default evidence admission policy is valid")
    }
}

#[derive(Debug)]
pub enum EvidenceAdmissionError {
    InvalidMaxEvidenceIds { value: usize },
    ImpossibleQualifiedFloor { minimum: usize, max_ids: usize },
    DuplicateAssessmentEvidenceId(String),
    DuplicateDimensionBinding(GenerativityDimension),
    MissingDimensionBinding(GenerativityDimension),
    TooManyEvidenceIds {
        dimension: GenerativityDimension,
        count: usize,
        max: usize,
    },
    DuplicateDimensionEvidenceId {
        dimension: GenerativityDimension,
        evidence_id: String,
    },
    UnknownEvidenceId {
        dimension: GenerativityDimension,
        evidence_id: String,
    },
    InsufficientQualifiedSupport {
        dimension: GenerativityDimension,
        required: usize,
        observed: usize,
    },
    Archive(ArchiveValidationError),
}

impl From<ArchiveValidationError> for EvidenceAdmissionError {
    fn from(value: ArchiveValidationError) -> Self {
        Self::Archive(value)
    }
}

impl std::fmt::Display for EvidenceAdmissionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidMaxEvidenceIds { value } => write!(
                f,
                "max evidence IDs per dimension must be within 1..={MAX_EVIDENCE_IDS_PER_DIMENSION}, got {value}"
            ),
            Self::ImpossibleQualifiedFloor { minimum, max_ids } => write!(
                f,
                "minimum qualified run count {minimum} exceeds max evidence IDs {max_ids}"
            ),
            Self::DuplicateAssessmentEvidenceId(id) => {
                write!(f, "assessment contains duplicate evidence ID: {id}")
            }
            Self::DuplicateDimensionBinding(dimension) => {
                write!(f, "duplicate dimension evidence binding: {dimension:?}")
            }
            Self::MissingDimensionBinding(dimension) => {
                write!(f, "missing dimension evidence binding: {dimension:?}")
            }
            Self::TooManyEvidenceIds {
                dimension,
                count,
                max,
            } => write!(
                f,
                "dimension {dimension:?} has {count} evidence IDs; maximum is {max}"
            ),
            Self::DuplicateDimensionEvidenceId {
                dimension,
                evidence_id,
            } => write!(
                f,
                "dimension {dimension:?} repeats evidence ID {evidence_id}"
            ),
            Self::UnknownEvidenceId {
                dimension,
                evidence_id,
            } => write!(
                f,
                "dimension {dimension:?} references unknown evidence ID {evidence_id}"
            ),
            Self::InsufficientQualifiedSupport {
                dimension,
                required,
                observed,
            } => write!(
                f,
                "dimension {dimension:?} requires {required} qualified run(s), observed {observed}"
            ),
            Self::Archive(error) => write!(f, "archive validation failed: {error}"),
        }
    }
}

impl std::error::Error for EvidenceAdmissionError {}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use symthaea_evidence_plane::{EvidenceCounters, Expectation, RunEvidence, RunId};

    use super::*;
    use crate::exploration::evidence_binding::EvidencePlaneEnvelope;
    use crate::exploration::generativity::{
        GenerativityAssessment, GenerativityEvidence,
    };
    use crate::exploration::persisted_evidence::{
        PersistedEvidenceCapsule, PersistedGenerativityBundle,
    };

    fn vector(positive: f64, risk: f64, confidence: f64) -> GenerativityVector {
        let e = |value| GenerativityEstimate::new(value, confidence).unwrap();
        GenerativityVector {
            immediate_utility: e(positive),
            epistemic_gain: e(positive),
            option_value: e(positive),
            diversity: e(positive),
            capability_gain: e(positive),
            diffusion: e(positive),
            commons_gain: e(positive),
            regeneration: e(positive),
            dependency_risk: e(risk),
            concentration_risk: e(risk),
            irreversibility_risk: e(risk),
        }
    }

    fn run(run_id: &str, calls: f64) -> RunEvidence {
        let mut declared = BTreeMap::new();
        declared.insert("mechanism_calls".into(), Expectation::MustBePositive);
        let mut measured = EvidenceCounters::new();
        measured.record("mechanism_calls", calls);
        RunEvidence::new(RunId::new(run_id), &("mode", "active"), declared, measured)
    }

    fn verified_bundle(
        subject: &str,
        positive: f64,
        risk: f64,
        confidence: f64,
    ) -> VerifiedGenerativityBundle {
        let run = run(&format!("run:{subject}"), 4.0);
        let envelope = EvidencePlaneEnvelope::from_run(&run).unwrap();
        let capsule = PersistedEvidenceCapsule::from_run(run, vec![]).unwrap();
        let mut assessment = GenerativityAssessment::new(
            subject,
            "test context",
            vector(positive, risk, confidence),
        );
        envelope.bind_qualified(&mut assessment, None).unwrap();
        PersistedGenerativityBundle::new(assessment, vec![capsule])
            .verify()
            .unwrap()
    }

    fn all_bindings(evidence_id: &str) -> Vec<DimensionEvidenceBinding> {
        GenerativityDimension::ALL
            .iter()
            .map(|dimension| DimensionEvidenceBinding {
                dimension: *dimension,
                evidence_ids: vec![evidence_id.to_string()],
            })
            .collect()
    }

    #[test]
    fn reported_confidence_without_dimension_support_is_rejected() {
        let bundle = verified_bundle("alpha", 0.8, 0.2, 0.9);
        assert!(matches!(
            qualify_archive_candidate(
                "alpha-entry",
                "low-water",
                &bundle,
                &[],
                EvidenceAdmissionPolicy::default(),
            ),
            Err(EvidenceAdmissionError::MissingDimensionBinding(_))
        ));
    }

    #[test]
    fn shared_evidence_is_visible_and_does_not_raise_confidence() {
        let bundle = verified_bundle("alpha", 0.8, 0.2, 0.9);
        let bindings = all_bindings("evidence-plane:run:alpha");
        let candidate = qualify_archive_candidate(
            "alpha-entry",
            "low-water",
            &bundle,
            &bindings,
            EvidenceAdmissionPolicy::default(),
        )
        .unwrap();
        assert_eq!(candidate.support().shared_evidence_ids.len(), 1);
        assert!(candidate.support().dimensions.iter().all(|support| {
            support.qualified_evidence_plane_runs == 1
                && (support.reported_confidence - 0.9).abs() < f64::EPSILON
        }));
    }

    #[test]
    fn generic_evidence_does_not_satisfy_qualified_run_floor() {
        let run = run("run:generic", 4.0);
        let envelope = EvidencePlaneEnvelope::from_run(&run).unwrap();
        let capsule = PersistedEvidenceCapsule::from_run(run, vec![]).unwrap();
        let mut assessment = GenerativityAssessment::new(
            "generic",
            "test context",
            vector(0.8, 0.2, 0.9),
        );
        assessment.evidence.push(GenerativityEvidence {
            evidence_id: "literature:1".into(),
            kind: "peer_review".into(),
            reference: Some("doi:example".into()),
            note: None,
        });
        envelope.bind_qualified(&mut assessment, None).unwrap();
        let bundle = PersistedGenerativityBundle::new(assessment, vec![capsule])
            .verify()
            .unwrap();
        let bindings = all_bindings("literature:1");
        assert!(matches!(
            qualify_archive_candidate(
                "generic-entry",
                "research",
                &bundle,
                &bindings,
                EvidenceAdmissionPolicy::default(),
            ),
            Err(EvidenceAdmissionError::InsufficientQualifiedSupport { .. })
        ));
    }

    #[test]
    fn duplicate_generic_evidence_ids_are_rejected() {
        let run = run("run:duplicate", 4.0);
        let envelope = EvidencePlaneEnvelope::from_run(&run).unwrap();
        let capsule = PersistedEvidenceCapsule::from_run(run, vec![]).unwrap();
        let mut assessment = GenerativityAssessment::new(
            "duplicate",
            "test context",
            vector(0.8, 0.2, 0.9),
        );
        for kind in ["simulation", "measurement"] {
            assessment.evidence.push(GenerativityEvidence {
                evidence_id: "duplicate-id".into(),
                kind: kind.into(),
                reference: None,
                note: None,
            });
        }
        envelope.bind_qualified(&mut assessment, None).unwrap();
        let bundle = PersistedGenerativityBundle::new(assessment, vec![capsule])
            .verify()
            .unwrap();
        assert!(matches!(
            qualify_archive_candidate(
                "duplicate-entry",
                "research",
                &bundle,
                &all_bindings("evidence-plane:run:duplicate"),
                EvidenceAdmissionPolicy::default(),
            ),
            Err(EvidenceAdmissionError::DuplicateAssessmentEvidenceId(id))
                if id == "duplicate-id"
        ));
    }

    #[test]
    fn support_metadata_tracks_pareto_pruning() {
        let worse = verified_bundle("worse", 0.6, 0.4, 0.9);
        let better = verified_bundle("better", 0.8, 0.2, 0.9);
        let mut archive = EvidenceAwareGenerativityArchive::default();
        archive
            .qualify_and_insert(
                "worse-entry",
                "repairable",
                &worse,
                &all_bindings("evidence-plane:run:worse"),
            )
            .unwrap();
        let outcome = archive
            .qualify_and_insert(
                "better-entry",
                "repairable",
                &better,
                &all_bindings("evidence-plane:run:better"),
            )
            .unwrap();
        assert!(matches!(outcome, ArchiveInsertOutcome::Inserted { .. }));
        assert!(archive.support_for("worse-entry").is_none());
        assert!(archive.support_for("better-entry").is_some());
    }

    #[test]
    fn verified_support_does_not_override_low_reported_confidence() {
        let uncertain = verified_bundle("uncertain", 0.9, 0.1, 0.2);
        let mut archive = EvidenceAwareGenerativityArchive::default();
        assert_eq!(
            archive
                .qualify_and_insert(
                    "uncertain-entry",
                    "research",
                    &uncertain,
                    &all_bindings("evidence-plane:run:uncertain"),
                )
                .unwrap(),
            ArchiveInsertOutcome::RejectedInsufficientEvidence
        );
        assert!(archive.entries().is_empty());
    }
}
