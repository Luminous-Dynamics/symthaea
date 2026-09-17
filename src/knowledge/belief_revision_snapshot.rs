// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Stable, read-only schemas for belief-revision policy and decision state.
//!
//! EKM-034 deliberately leaves policy/decision internals opaque in the V1 wire
//! envelope. This module defines the explicit semantic target without widening
//! mutation authority:
//!
//! - future policies can originate from [`BeliefRevisionPolicySchemaV1`], which
//!   deterministically builds the existing private [`BeliefRevisionPolicy`];
//! - existing decisions can be projected losslessly into
//!   [`BeliefRevisionDecisionSnapshotV1`] through public read-only APIs.
//!
//! None of the snapshot types can apply a belief revision or authorize one.

use super::belief_revision_gate::{
    BeliefRevisionDecision, BeliefRevisionFailure, BeliefRevisionPolicy,
    BeliefRevisionPolicyError,
};
use super::claim_evidence::{ClaimId, EvidenceId};
use super::epistemic_vector::UncertaintyDimension;
use super::knowledge_weight_routing::{
    KnowledgeWeightDimension, KnowledgeWeightRoutingFailure, KnowledgeWeightSource,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BeliefRevisionSnapshotVersion {
    V1,
}

/// Explicit V1 policy schema.
///
/// The schema is intended to become the source of future policy objects rather
/// than attempting to infer private policy fields after a decision was made.
#[derive(Debug, Clone, PartialEq)]
pub struct BeliefRevisionPolicySchemaV1 {
    pub max_abs_delta: f32,
    pub min_declared_provenance_roots: usize,
    pub require_calibration: bool,
    pub min_calibration_samples: u64,
    pub max_calibration_ece: f64,
    pub require_uncertainty_assessment: bool,
    pub require_current_uncertainty: bool,
    pub block_strengthen_with_unresolved_contradictions: bool,
    pub require_intervention_for_causal_strengthen: bool,
    /// Canonically ordered by uncertainty-dimension tag.
    pub strengthen_uncertainty_caps: Vec<(UncertaintyDimension, f64)>,
}

impl BeliefRevisionPolicySchemaV1 {
    pub fn new(
        max_abs_delta: f32,
        min_declared_provenance_roots: usize,
        require_calibration: bool,
        min_calibration_samples: u64,
        max_calibration_ece: f64,
    ) -> Result<Self, BeliefRevisionPolicyError> {
        // Delegate the numeric contract to the existing authority-free policy
        // constructor so this schema cannot develop a second validation rule.
        BeliefRevisionPolicy::new(
            max_abs_delta,
            min_declared_provenance_roots,
            require_calibration,
            min_calibration_samples,
            max_calibration_ece,
        )?;
        Ok(Self {
            max_abs_delta,
            min_declared_provenance_roots,
            require_calibration,
            min_calibration_samples,
            max_calibration_ece,
            require_uncertainty_assessment: false,
            require_current_uncertainty: false,
            block_strengthen_with_unresolved_contradictions: false,
            require_intervention_for_causal_strengthen: false,
            strengthen_uncertainty_caps: Vec::new(),
        })
    }

    pub fn with_uncertainty_requirements(
        mut self,
        require_assessment: bool,
        require_current: bool,
    ) -> Self {
        self.require_uncertainty_assessment = require_assessment;
        self.require_current_uncertainty = require_current;
        self
    }

    pub fn block_strengthen_with_unresolved_contradictions(mut self, block: bool) -> Self {
        self.block_strengthen_with_unresolved_contradictions = block;
        self
    }

    pub fn require_intervention_for_causal_strengthen(mut self, require: bool) -> Self {
        self.require_intervention_for_causal_strengthen = require;
        self
    }

    pub fn with_strengthen_uncertainty_cap(
        mut self,
        dimension: UncertaintyDimension,
        maximum: f64,
    ) -> Result<Self, BeliefRevisionPolicyError> {
        // Reuse the existing policy validator without retaining the temporary.
        BeliefRevisionPolicy::new(
            self.max_abs_delta,
            self.min_declared_provenance_roots,
            self.require_calibration,
            self.min_calibration_samples,
            self.max_calibration_ece,
        )?
        .with_strengthen_uncertainty_cap(dimension, maximum)?;

        if let Some(existing) = self
            .strengthen_uncertainty_caps
            .iter_mut()
            .find(|(candidate, _)| *candidate == dimension)
        {
            existing.1 = maximum;
        } else {
            self.strengthen_uncertainty_caps.push((dimension, maximum));
        }
        self.strengthen_uncertainty_caps
            .sort_by_key(|(dimension, _)| uncertainty_dimension_tag(*dimension));
        Ok(self)
    }

    /// Deterministically construct the existing private policy type.
    ///
    /// This is policy construction, not decision or mutation authority.
    pub fn build_policy(&self) -> Result<BeliefRevisionPolicy, BeliefRevisionPolicyError> {
        let mut policy = BeliefRevisionPolicy::new(
            self.max_abs_delta,
            self.min_declared_provenance_roots,
            self.require_calibration,
            self.min_calibration_samples,
            self.max_calibration_ece,
        )?
        .with_uncertainty_requirements(
            self.require_uncertainty_assessment,
            self.require_current_uncertainty,
        )
        .block_strengthen_with_unresolved_contradictions(
            self.block_strengthen_with_unresolved_contradictions,
        )
        .require_intervention_for_causal_strengthen(
            self.require_intervention_for_causal_strengthen,
        );
        for (dimension, maximum) in &self.strengthen_uncertainty_caps {
            policy = policy.with_strengthen_uncertainty_cap(*dimension, *maximum)?;
        }
        Ok(policy)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KnowledgeWeightRoutingFailureSnapshotV1 {
    SourceCannotUpdateDimension {
        source: KnowledgeWeightSource,
        dimension: KnowledgeWeightDimension,
    },
    EpistemicUpdateHasNoEvidenceBasis,
}

impl From<&KnowledgeWeightRoutingFailure> for KnowledgeWeightRoutingFailureSnapshotV1 {
    fn from(value: &KnowledgeWeightRoutingFailure) -> Self {
        match value {
            KnowledgeWeightRoutingFailure::SourceCannotUpdateDimension { source, dimension } => {
                Self::SourceCannotUpdateDimension {
                    source: *source,
                    dimension: *dimension,
                }
            }
            KnowledgeWeightRoutingFailure::EpistemicUpdateHasNoEvidenceBasis => {
                Self::EpistemicUpdateHasNoEvidenceBasis
            }
        }
    }
}

/// Stable typed projection of all EKM-024 decision failure semantics.
#[derive(Debug, Clone, PartialEq)]
pub enum BeliefRevisionFailureSnapshotV1 {
    WeightRoutingDenied(Vec<KnowledgeWeightRoutingFailureSnapshotV1>),
    UnknownClaim(ClaimId),
    UnknownEvidence(EvidenceId),
    EvidenceForDifferentClaim {
        evidence_id: EvidenceId,
        expected_claim: ClaimId,
        actual_claim: ClaimId,
    },
    PositiveDeltaLacksSupportingEvidence,
    PositiveDeltaIncludesContradictingEvidence,
    NegativeDeltaLacksContradictingEvidence,
    NegativeDeltaIncludesSupportingEvidence,
    DeclaredProvenanceRootsBelowMinimum {
        required: usize,
        actual: usize,
    },
    DeltaExceedsPolicy {
        maximum: f32,
        actual: f32,
    },
    CalibrationMissing,
    CalibrationSamplesBelowMinimum {
        required: u64,
        actual: u64,
    },
    CalibrationEceAboveMaximum {
        maximum: f64,
        actual: f64,
    },
    UncertaintyAssessmentMissing,
    UncertaintyAssessmentForDifferentClaim {
        expected_claim: ClaimId,
        actual_claim: ClaimId,
    },
    UncertaintyAssessmentPredatesEvidence {
        assessment_cycle: u64,
        latest_evidence_cycle: u64,
    },
    RequiredUncertaintyDimensionUnassessed(UncertaintyDimension),
    UncertaintyAboveMaximum {
        dimension: UncertaintyDimension,
        maximum: f64,
        actual: f64,
    },
    UnresolvedContradictionBlocksStrengthen {
        count: usize,
    },
    CausalStrengthenLacksInterventionalBasis,
}

impl From<&BeliefRevisionFailure> for BeliefRevisionFailureSnapshotV1 {
    fn from(value: &BeliefRevisionFailure) -> Self {
        match value {
            BeliefRevisionFailure::WeightRoutingDenied(failures) => Self::WeightRoutingDenied(
                failures.iter().map(Into::into).collect(),
            ),
            BeliefRevisionFailure::UnknownClaim(id) => Self::UnknownClaim(*id),
            BeliefRevisionFailure::UnknownEvidence(id) => Self::UnknownEvidence(*id),
            BeliefRevisionFailure::EvidenceForDifferentClaim {
                evidence_id,
                expected_claim,
                actual_claim,
            } => Self::EvidenceForDifferentClaim {
                evidence_id: *evidence_id,
                expected_claim: *expected_claim,
                actual_claim: *actual_claim,
            },
            BeliefRevisionFailure::PositiveDeltaLacksSupportingEvidence => {
                Self::PositiveDeltaLacksSupportingEvidence
            }
            BeliefRevisionFailure::PositiveDeltaIncludesContradictingEvidence => {
                Self::PositiveDeltaIncludesContradictingEvidence
            }
            BeliefRevisionFailure::NegativeDeltaLacksContradictingEvidence => {
                Self::NegativeDeltaLacksContradictingEvidence
            }
            BeliefRevisionFailure::NegativeDeltaIncludesSupportingEvidence => {
                Self::NegativeDeltaIncludesSupportingEvidence
            }
            BeliefRevisionFailure::DeclaredProvenanceRootsBelowMinimum { required, actual } => {
                Self::DeclaredProvenanceRootsBelowMinimum {
                    required: *required,
                    actual: *actual,
                }
            }
            BeliefRevisionFailure::DeltaExceedsPolicy { maximum, actual } => {
                Self::DeltaExceedsPolicy {
                    maximum: *maximum,
                    actual: *actual,
                }
            }
            BeliefRevisionFailure::CalibrationMissing => Self::CalibrationMissing,
            BeliefRevisionFailure::CalibrationSamplesBelowMinimum { required, actual } => {
                Self::CalibrationSamplesBelowMinimum {
                    required: *required,
                    actual: *actual,
                }
            }
            BeliefRevisionFailure::CalibrationEceAboveMaximum { maximum, actual } => {
                Self::CalibrationEceAboveMaximum {
                    maximum: *maximum,
                    actual: *actual,
                }
            }
            BeliefRevisionFailure::UncertaintyAssessmentMissing => {
                Self::UncertaintyAssessmentMissing
            }
            BeliefRevisionFailure::UncertaintyAssessmentForDifferentClaim {
                expected_claim,
                actual_claim,
            } => Self::UncertaintyAssessmentForDifferentClaim {
                expected_claim: *expected_claim,
                actual_claim: *actual_claim,
            },
            BeliefRevisionFailure::UncertaintyAssessmentPredatesEvidence {
                assessment_cycle,
                latest_evidence_cycle,
            } => Self::UncertaintyAssessmentPredatesEvidence {
                assessment_cycle: *assessment_cycle,
                latest_evidence_cycle: *latest_evidence_cycle,
            },
            BeliefRevisionFailure::RequiredUncertaintyDimensionUnassessed(dimension) => {
                Self::RequiredUncertaintyDimensionUnassessed(*dimension)
            }
            BeliefRevisionFailure::UncertaintyAboveMaximum {
                dimension,
                maximum,
                actual,
            } => Self::UncertaintyAboveMaximum {
                dimension: *dimension,
                maximum: *maximum,
                actual: *actual,
            },
            BeliefRevisionFailure::UnresolvedContradictionBlocksStrengthen { count } => {
                Self::UnresolvedContradictionBlocksStrengthen { count: *count }
            }
            BeliefRevisionFailure::CausalStrengthenLacksInterventionalBasis => {
                Self::CausalStrengthenLacksInterventionalBasis
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BeliefRevisionDecisionSnapshotV1 {
    pub version: BeliefRevisionSnapshotVersion,
    pub eligible: bool,
    pub declared_provenance_root_count: usize,
    pub failures: Vec<BeliefRevisionFailureSnapshotV1>,
}

impl BeliefRevisionDecisionSnapshotV1 {
    pub fn capture(decision: &BeliefRevisionDecision) -> Self {
        Self {
            version: BeliefRevisionSnapshotVersion::V1,
            eligible: decision.eligible(),
            declared_provenance_root_count: decision.declared_provenance_root_count(),
            failures: decision.failures().iter().map(Into::into).collect(),
        }
    }
}

pub fn uncertainty_dimension_tag(dimension: UncertaintyDimension) -> u8 {
    match dimension {
        UncertaintyDimension::Epistemic => 1,
        UncertaintyDimension::Aleatoric => 2,
        UncertaintyDimension::Ontological => 3,
        UncertaintyDimension::DistributionShift => 4,
    }
}

pub fn knowledge_weight_dimension_tag(dimension: KnowledgeWeightDimension) -> u8 {
    match dimension {
        KnowledgeWeightDimension::EpistemicSupport => 1,
        KnowledgeWeightDimension::Accessibility => 2,
        KnowledgeWeightDimension::Retention => 3,
        KnowledgeWeightDimension::Consolidation => 4,
    }
}

pub fn knowledge_weight_source_tag(source: KnowledgeWeightSource) -> u8 {
    match source {
        KnowledgeWeightSource::AdmittedEvidence => 1,
        KnowledgeWeightSource::Retrieval => 2,
        KnowledgeWeightSource::SimilarityMatch => 3,
        KnowledgeWeightSource::DreamReplay => 4,
        KnowledgeWeightSource::CausalConsolidation => 5,
        KnowledgeWeightSource::MemoryDecay => 6,
        KnowledgeWeightSource::TaskRelevance => 7,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::{
        BeliefRevisionGate, CalibrationSnapshot, ClaimKind, EpistemicLedger, EvidenceKind,
        EvidencePolarity, EpistemicRevisionProposal,
    };

    #[test]
    fn schema_builds_deterministically_and_canonicalizes_caps() {
        let schema = BeliefRevisionPolicySchemaV1::new(0.2, 2, true, 50, 0.1)
            .unwrap()
            .with_uncertainty_requirements(true, true)
            .block_strengthen_with_unresolved_contradictions(true)
            .require_intervention_for_causal_strengthen(true)
            .with_strengthen_uncertainty_cap(UncertaintyDimension::DistributionShift, 0.3)
            .unwrap()
            .with_strengthen_uncertainty_cap(UncertaintyDimension::Epistemic, 0.2)
            .unwrap()
            .with_strengthen_uncertainty_cap(UncertaintyDimension::Epistemic, 0.25)
            .unwrap();
        assert_eq!(
            schema.strengthen_uncertainty_caps,
            vec![
                (UncertaintyDimension::Epistemic, 0.25),
                (UncertaintyDimension::DistributionShift, 0.3),
            ]
        );
        assert_eq!(schema.build_policy().unwrap(), schema.build_policy().unwrap());
    }

    #[test]
    fn decision_snapshot_preserves_typed_failure_payloads() {
        let mut ledger = EpistemicLedger::new();
        let provenance = ledger
            .add_provenance("report", None, None, 1, vec![])
            .unwrap();
        let claim = ledger.add_claim("A causes B", ClaimKind::Causal, None, None, 1);
        let report = ledger
            .add_evidence(
                claim,
                EvidenceKind::Report,
                EvidencePolarity::Supports,
                provenance,
                2,
                None,
                None,
            )
            .unwrap();
        let policy = BeliefRevisionPolicySchemaV1::new(0.2, 1, true, 10, 0.2)
            .unwrap()
            .require_intervention_for_causal_strengthen(true)
            .build_policy()
            .unwrap();
        let proposal = EpistemicRevisionProposal::new(claim, 0.1, vec![report], "report")
            .unwrap();
        let decision = BeliefRevisionGate::evaluate(
            &ledger,
            &proposal,
            &policy,
            Some(CalibrationSnapshot::new(0, 0.0).unwrap()),
            None,
        );
        let snapshot = BeliefRevisionDecisionSnapshotV1::capture(&decision);
        assert!(!snapshot.eligible);
        assert!(snapshot.failures.contains(
            &BeliefRevisionFailureSnapshotV1::CalibrationSamplesBelowMinimum {
                required: 10,
                actual: 0,
            }
        ));
        assert!(snapshot.failures.contains(
            &BeliefRevisionFailureSnapshotV1::CausalStrengthenLacksInterventionalBasis
        ));
    }

    #[test]
    fn stable_tags_are_explicit() {
        assert_eq!(uncertainty_dimension_tag(UncertaintyDimension::Epistemic), 1);
        assert_eq!(knowledge_weight_dimension_tag(KnowledgeWeightDimension::EpistemicSupport), 1);
        assert_eq!(knowledge_weight_source_tag(KnowledgeWeightSource::AdmittedEvidence), 1);
    }
}
