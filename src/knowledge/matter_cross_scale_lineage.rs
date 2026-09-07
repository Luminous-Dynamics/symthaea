// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Structured lineage envelopes for cross-scale Matter claims.
//!
//! `matter_cross_scale` decides whether a path is admissible and caps derived
//! E/N/M authority. This module preserves the *exact required claims* that led to
//! that decision so downstream code cannot lose source statements, solver
//! limitations, uncertainty, external evidence references, or falsifiers merely
//! because the final derived claim is concise.
//!
//! The envelope is a lineage object, not a stronger epistemic credential. It
//! snapshots caller-provided Matter claims and never upgrades their authority.

use std::collections::BTreeSet;
use std::fmt;

use symthaea_epistemic_types::{MatterClaim, MatterEvidenceRef, MatterScale};

use super::matter_cross_scale::{
    assess_cross_scale_evidence, CrossScaleEvidenceAssessment, CrossScaleEvidenceError,
    MatterScaleTransition,
};

/// Why one exact Matter claim is required by a cross-scale lineage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CrossScaleDependencyRole {
    Source,
    TransitionEvidence { transition_id: String },
}

/// Exact snapshot of a required Matter claim at lineage-construction time.
///
/// Cloning the canonical claim deliberately retains statement, validation stage,
/// E/N/M coordinate, provenance, solver profile and limitations, uncertainty,
/// evidence references, falsifiers, and claim limitations without inventing a
/// second partial representation.
#[derive(Debug, Clone, PartialEq)]
pub struct CrossScaleDependencySnapshot {
    pub role: CrossScaleDependencyRole,
    pub claim: MatterClaim,
}

/// Structured limitation inherited from a required dependency.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CrossScaleConstraintKind {
    ClaimLimitation,
    SolverLimitation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CrossScaleInheritedConstraint {
    pub source_claim_id: String,
    pub role: CrossScaleDependencyRole,
    pub kind: CrossScaleConstraintKind,
    pub text: String,
}

/// Structured falsifier inherited from a required dependency.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CrossScaleInheritedFalsifier {
    pub source_claim_id: String,
    pub role: CrossScaleDependencyRole,
    pub text: String,
}

/// External/canonical evidence reference inherited from a required dependency.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CrossScaleInheritedEvidence {
    pub source_claim_id: String,
    pub role: CrossScaleDependencyRole,
    pub evidence: MatterEvidenceRef,
}

/// Assessment plus exact dependency snapshots for one cross-scale path.
#[derive(Debug, Clone, PartialEq)]
pub struct CrossScaleLineageEnvelope {
    pub assessment: CrossScaleEvidenceAssessment,
    pub dependencies: Vec<CrossScaleDependencySnapshot>,
}

impl CrossScaleLineageEnvelope {
    /// Build an assessment and its lineage atomically from the same borrowed
    /// claims/transitions. This avoids reconstructing lineage later from IDs alone.
    pub fn assess(
        source_claims: &[&MatterClaim],
        transitions: &[MatterScaleTransition<'_>],
        target_scale: MatterScale,
    ) -> Result<Self, CrossScaleLineageError> {
        let assessment = assess_cross_scale_evidence(source_claims, transitions, target_scale)?;

        let mut dependencies = Vec::new();
        for claim in source_claims {
            dependencies.push(CrossScaleDependencySnapshot {
                role: CrossScaleDependencyRole::Source,
                claim: (*claim).clone(),
            });
        }
        for transition in transitions {
            if let Some(evidence) = transition.evidence() {
                dependencies.push(CrossScaleDependencySnapshot {
                    role: CrossScaleDependencyRole::TransitionEvidence {
                        transition_id: transition.transition_id().to_string(),
                    },
                    claim: evidence.clone(),
                });
            }
        }

        let envelope = Self {
            assessment,
            dependencies,
        };
        envelope.validate_dependency_set()?;
        Ok(envelope)
    }

    /// Every inherited claim/solver limitation with its originating claim and
    /// dependency role preserved.
    pub fn inherited_constraints(&self) -> Vec<CrossScaleInheritedConstraint> {
        let mut inherited = Vec::new();
        for dependency in &self.dependencies {
            for limitation in &dependency.claim.limitations {
                inherited.push(CrossScaleInheritedConstraint {
                    source_claim_id: dependency.claim.claim_id.clone(),
                    role: dependency.role.clone(),
                    kind: CrossScaleConstraintKind::ClaimLimitation,
                    text: limitation.clone(),
                });
            }
            if let Some(solver) = &dependency.claim.solver {
                for limitation in &solver.limitations {
                    inherited.push(CrossScaleInheritedConstraint {
                        source_claim_id: dependency.claim.claim_id.clone(),
                        role: dependency.role.clone(),
                        kind: CrossScaleConstraintKind::SolverLimitation,
                        text: limitation.clone(),
                    });
                }
            }
        }
        inherited
    }

    /// Every inherited falsifier with source identity retained.
    pub fn inherited_falsifiers(&self) -> Vec<CrossScaleInheritedFalsifier> {
        let mut inherited = Vec::new();
        for dependency in &self.dependencies {
            for falsifier in &dependency.claim.falsifiers {
                inherited.push(CrossScaleInheritedFalsifier {
                    source_claim_id: dependency.claim.claim_id.clone(),
                    role: dependency.role.clone(),
                    text: falsifier.clone(),
                });
            }
        }
        inherited
    }

    /// Every canonical evidence reference carried by required dependency claims.
    pub fn inherited_evidence(&self) -> Vec<CrossScaleInheritedEvidence> {
        let mut inherited = Vec::new();
        for dependency in &self.dependencies {
            for evidence in &dependency.claim.evidence {
                inherited.push(CrossScaleInheritedEvidence {
                    source_claim_id: dependency.claim.claim_id.clone(),
                    role: dependency.role.clone(),
                    evidence: evidence.clone(),
                });
            }
        }
        inherited
    }

    /// Admit a derived computational claim while retaining the complete lineage
    /// envelope as part of the resulting object.
    pub fn admit_derived_computational_claim(
        &self,
        candidate: MatterClaim,
    ) -> Result<AdmittedCrossScaleMatterClaim, CrossScaleLineageError> {
        if candidate.claim_id.trim().is_empty() {
            return Err(CrossScaleLineageError::EmptyDerivedClaimId);
        }
        if candidate.statement.trim().is_empty() {
            return Err(CrossScaleLineageError::EmptyDerivedStatement);
        }
        if self
            .dependencies
            .iter()
            .any(|dependency| dependency.claim.claim_id == candidate.claim_id)
        {
            return Err(CrossScaleLineageError::DerivedClaimReusesDependencyId(
                candidate.claim_id.clone(),
            ));
        }
        self.assessment
            .enforce_derived_computational_claim(&candidate)?;
        Ok(AdmittedCrossScaleMatterClaim {
            claim: candidate,
            lineage: self.clone(),
        })
    }

    fn validate_dependency_set(&self) -> Result<(), CrossScaleLineageError> {
        let expected: BTreeSet<String> = self
            .assessment
            .source_claim_ids
            .iter()
            .chain(self.assessment.transition_evidence_claim_ids.iter())
            .cloned()
            .collect();
        let actual: BTreeSet<String> = self
            .dependencies
            .iter()
            .map(|dependency| dependency.claim.claim_id.clone())
            .collect();
        if expected != actual || actual.len() != self.dependencies.len() {
            return Err(CrossScaleLineageError::DependencySetMismatch);
        }
        Ok(())
    }
}

/// A derived Matter claim that cannot be separated from the exact dependency
/// snapshots used to admit it without explicitly discarding this wrapper.
#[derive(Debug, Clone, PartialEq)]
pub struct AdmittedCrossScaleMatterClaim {
    pub claim: MatterClaim,
    pub lineage: CrossScaleLineageEnvelope,
}

impl AdmittedCrossScaleMatterClaim {
    pub fn inherited_constraints(&self) -> Vec<CrossScaleInheritedConstraint> {
        self.lineage.inherited_constraints()
    }

    pub fn inherited_falsifiers(&self) -> Vec<CrossScaleInheritedFalsifier> {
        self.lineage.inherited_falsifiers()
    }

    pub fn inherited_evidence(&self) -> Vec<CrossScaleInheritedEvidence> {
        self.lineage.inherited_evidence()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CrossScaleLineageError {
    CrossScale(CrossScaleEvidenceError),
    DependencySetMismatch,
    EmptyDerivedClaimId,
    EmptyDerivedStatement,
    DerivedClaimReusesDependencyId(String),
}

impl From<CrossScaleEvidenceError> for CrossScaleLineageError {
    fn from(value: CrossScaleEvidenceError) -> Self {
        Self::CrossScale(value)
    }
}

impl fmt::Display for CrossScaleLineageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CrossScale(error) => write!(f, "{error}"),
            Self::DependencySetMismatch => write!(
                f,
                "cross-scale lineage snapshots do not exactly match the assessment's required claim IDs"
            ),
            Self::EmptyDerivedClaimId => write!(f, "derived cross-scale claim id must not be empty"),
            Self::EmptyDerivedStatement => {
                write!(f, "derived cross-scale claim statement must not be empty")
            }
            Self::DerivedClaimReusesDependencyId(id) => write!(
                f,
                "derived cross-scale claim `{id}` reuses a required dependency claim id"
            ),
        }
    }
}

impl std::error::Error for CrossScaleLineageError {}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_epistemic_types::{
        EmpiricalLevel, EpistemicContext, EpistemicCoordinate, MaterialityLevel,
        MatterSolverCapability, MatterSolverProfile, MatterUncertainty, MatterValidationStage,
        NormativeLevel,
    };

    fn solver(name: &str, limitation: &str) -> MatterSolverProfile {
        MatterSolverProfile {
            name: name.to_string(),
            version: "1".to_string(),
            method: format!("{name} fixture method"),
            capabilities: vec![MatterSolverCapability::CoupledMultiphysics],
            limitations: vec![limitation.to_string()],
        }
    }

    fn claim(
        id: &str,
        scale: MatterScale,
        solver_name: &str,
        solver_limitation: &str,
        ood: bool,
    ) -> MatterClaim {
        let mut claim = MatterClaim::local_computational(
            id,
            format!("statement for {id}"),
            scale,
            MatterValidationStage::HighFidelitySimulated,
            solver(solver_name, solver_limitation),
        )
        .unwrap()
        .with_limitation(format!("claim limitation for {id}"))
        .with_falsifier(format!("falsifier for {id}"))
        .with_evidence(MatterEvidenceRef {
            evidence_id: format!("evidence:{id}"),
            run_id: Some(format!("run:{id}")),
            config_hash: Some(format!("config:{id}")),
            dataset_ids: vec![format!("dataset:{id}")],
        });
        if ood {
            claim = claim.with_uncertainty(
                MatterUncertainty::new(0.125, "eV", "fixture distance", true).unwrap(),
            );
        }
        claim
    }

    #[test]
    fn envelope_snapshots_complete_dependency_claims() {
        let source = claim(
            "nuclear-source",
            MatterScale::Nuclear,
            "nuclear-solver",
            "nuclear solver limitation",
            false,
        );
        let transition_evidence = claim(
            "electronic-transition",
            MatterScale::AtomicElectronic,
            "relativistic-solver",
            "relativistic solver limitation",
            true,
        );
        let transition = MatterScaleTransition::supported_out_of_domain(
            "nuclear->electronic",
            MatterScale::Nuclear,
            MatterScale::AtomicElectronic,
            &transition_evidence,
        );

        let envelope = CrossScaleLineageEnvelope::assess(
            &[&source],
            &[transition],
            MatterScale::AtomicElectronic,
        )
        .unwrap();

        assert_eq!(envelope.dependencies.len(), 2);
        assert_eq!(envelope.dependencies[0].claim, source);
        assert_eq!(envelope.dependencies[1].claim, transition_evidence);
        assert!(matches!(
            envelope.dependencies[0].role,
            CrossScaleDependencyRole::Source
        ));
        assert!(matches!(
            &envelope.dependencies[1].role,
            CrossScaleDependencyRole::TransitionEvidence { transition_id }
                if transition_id == "nuclear->electronic"
        ));
    }

    #[test]
    fn constraints_falsifiers_and_evidence_keep_source_identity() {
        let source = claim(
            "nuclear-source",
            MatterScale::Nuclear,
            "nuclear-solver",
            "nuclear solver limitation",
            false,
        );
        let transition_evidence = claim(
            "electronic-transition",
            MatterScale::AtomicElectronic,
            "relativistic-solver",
            "relativistic solver limitation",
            false,
        );
        let transition = MatterScaleTransition::supported_in_domain(
            "nuclear->electronic",
            MatterScale::Nuclear,
            MatterScale::AtomicElectronic,
            &transition_evidence,
        );
        let envelope = CrossScaleLineageEnvelope::assess(
            &[&source],
            &[transition],
            MatterScale::AtomicElectronic,
        )
        .unwrap();

        let constraints = envelope.inherited_constraints();
        assert!(constraints.iter().any(|item| {
            item.source_claim_id == "nuclear-source"
                && item.kind == CrossScaleConstraintKind::ClaimLimitation
                && item.text == "claim limitation for nuclear-source"
        }));
        assert!(constraints.iter().any(|item| {
            item.source_claim_id == "electronic-transition"
                && item.kind == CrossScaleConstraintKind::SolverLimitation
                && item.text == "relativistic solver limitation"
        }));

        let falsifiers = envelope.inherited_falsifiers();
        assert!(falsifiers.iter().any(|item| {
            item.source_claim_id == "nuclear-source"
                && item.text == "falsifier for nuclear-source"
        }));

        let evidence = envelope.inherited_evidence();
        assert!(evidence.iter().any(|item| {
            item.source_claim_id == "electronic-transition"
                && item.evidence.evidence_id == "evidence:electronic-transition"
        }));
    }

    #[test]
    fn exact_uncertainty_value_survives_dependency_snapshot() {
        let source = claim(
            "nuclear-source",
            MatterScale::Nuclear,
            "nuclear-solver",
            "nuclear solver limitation",
            false,
        );
        let transition_evidence = claim(
            "electronic-transition",
            MatterScale::AtomicElectronic,
            "relativistic-solver",
            "relativistic solver limitation",
            true,
        );
        let transition = MatterScaleTransition::supported_out_of_domain(
            "nuclear->electronic",
            MatterScale::Nuclear,
            MatterScale::AtomicElectronic,
            &transition_evidence,
        );
        let envelope = CrossScaleLineageEnvelope::assess(
            &[&source],
            &[transition],
            MatterScale::AtomicElectronic,
        )
        .unwrap();

        let snapshot = envelope
            .dependencies
            .iter()
            .find(|dependency| dependency.claim.claim_id == "electronic-transition")
            .unwrap();
        assert_eq!(
            snapshot.claim.uncertainty.as_ref().unwrap().value.to_bits(),
            0.125_f64.to_bits()
        );
        assert!(snapshot.claim.uncertainty.as_ref().unwrap().out_of_domain);
    }

    #[test]
    fn admitted_claim_retains_lineage_and_ood_requirement() {
        let source = claim(
            "nuclear-source",
            MatterScale::Nuclear,
            "nuclear-solver",
            "nuclear solver limitation",
            false,
        );
        let transition_evidence = claim(
            "electronic-transition",
            MatterScale::AtomicElectronic,
            "relativistic-solver",
            "relativistic solver limitation",
            true,
        );
        let transition = MatterScaleTransition::supported_out_of_domain(
            "nuclear->electronic",
            MatterScale::Nuclear,
            MatterScale::AtomicElectronic,
            &transition_evidence,
        );
        let envelope = CrossScaleLineageEnvelope::assess(
            &[&source],
            &[transition],
            MatterScale::AtomicElectronic,
        )
        .unwrap();

        let mut candidate = claim(
            "derived-electronic",
            MatterScale::AtomicElectronic,
            "derived-solver",
            "derived solver limitation",
            true,
        );
        candidate.epistemic = EpistemicCoordinate::new(
            EmpiricalLevel::E1Testimonial,
            NormativeLevel::N0Personal,
            MaterialityLevel::M0Ephemeral,
            EpistemicContext::Scientific,
        );

        let admitted = envelope
            .admit_derived_computational_claim(candidate)
            .unwrap();
        assert_eq!(admitted.lineage.dependencies.len(), 2);
        assert_eq!(admitted.claim.claim_id, "derived-electronic");
        assert!(!admitted.inherited_constraints().is_empty());
        assert!(!admitted.inherited_falsifiers().is_empty());
        assert!(!admitted.inherited_evidence().is_empty());
    }

    #[test]
    fn derived_claim_cannot_reuse_dependency_identity() {
        let source = claim(
            "nuclear-source",
            MatterScale::Nuclear,
            "nuclear-solver",
            "nuclear solver limitation",
            false,
        );
        let transition_evidence = claim(
            "electronic-transition",
            MatterScale::AtomicElectronic,
            "relativistic-solver",
            "relativistic solver limitation",
            false,
        );
        let transition = MatterScaleTransition::supported_in_domain(
            "nuclear->electronic",
            MatterScale::Nuclear,
            MatterScale::AtomicElectronic,
            &transition_evidence,
        );
        let envelope = CrossScaleLineageEnvelope::assess(
            &[&source],
            &[transition],
            MatterScale::AtomicElectronic,
        )
        .unwrap();

        let candidate = claim(
            "electronic-transition",
            MatterScale::AtomicElectronic,
            "derived-solver",
            "derived solver limitation",
            false,
        );
        assert!(matches!(
            envelope.admit_derived_computational_claim(candidate),
            Err(CrossScaleLineageError::DerivedClaimReusesDependencyId(id))
                if id == "electronic-transition"
        ));
    }
}
