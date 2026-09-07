// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cross-scale evidence propagation for Matter Observatory claims.
//!
//! A derived multiscale claim must not gain epistemic authority merely because
//! several calculations were chained together. Every required source claim and
//! every supported scale-transition claim contributes an E/N/M upper bound; the
//! derived claim is capped componentwise by the weakest required dependency.
//!
//! Validation stages are deliberately not ranked or averaged. Missing or
//! unsupported transitions refuse composition. An out-of-domain transition is
//! admissible only when its evidence explicitly carries an OOD uncertainty marker,
//! and that marker must propagate to any derived computational claim.

use std::cmp::min;
use std::collections::BTreeSet;
use std::fmt;

use symthaea_epistemic_types::{
    EmpiricalLevel, EpistemicContext, EpistemicCoordinate, MaterialityLevel, MatterClaim,
    MatterScale, MatterValidationStage, NormativeLevel,
};

/// Scientific support state for one required transition between matter scales.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatterTransitionSupport {
    SupportedInDomain,
    SupportedOutOfDomain,
    Unsupported,
    MissingEvidence,
}

/// One ordered transition in a cross-scale Matter evidence path.
///
/// Fields are private so callers cannot construct a nominally supported transition
/// without supplying an evidence claim.
#[derive(Debug, Clone, Copy)]
pub struct MatterScaleTransition<'a> {
    transition_id: &'a str,
    from_scale: MatterScale,
    to_scale: MatterScale,
    support: MatterTransitionSupport,
    evidence: Option<&'a MatterClaim>,
}

impl<'a> MatterScaleTransition<'a> {
    pub fn supported_in_domain(
        transition_id: &'a str,
        from_scale: MatterScale,
        to_scale: MatterScale,
        evidence: &'a MatterClaim,
    ) -> Self {
        Self {
            transition_id,
            from_scale,
            to_scale,
            support: MatterTransitionSupport::SupportedInDomain,
            evidence: Some(evidence),
        }
    }

    pub fn supported_out_of_domain(
        transition_id: &'a str,
        from_scale: MatterScale,
        to_scale: MatterScale,
        evidence: &'a MatterClaim,
    ) -> Self {
        Self {
            transition_id,
            from_scale,
            to_scale,
            support: MatterTransitionSupport::SupportedOutOfDomain,
            evidence: Some(evidence),
        }
    }

    pub fn unsupported(
        transition_id: &'a str,
        from_scale: MatterScale,
        to_scale: MatterScale,
    ) -> Self {
        Self {
            transition_id,
            from_scale,
            to_scale,
            support: MatterTransitionSupport::Unsupported,
            evidence: None,
        }
    }

    pub fn missing_evidence(
        transition_id: &'a str,
        from_scale: MatterScale,
        to_scale: MatterScale,
    ) -> Self {
        Self {
            transition_id,
            from_scale,
            to_scale,
            support: MatterTransitionSupport::MissingEvidence,
            evidence: None,
        }
    }

    pub fn transition_id(&self) -> &str {
        self.transition_id
    }

    pub fn from_scale(&self) -> MatterScale {
        self.from_scale
    }

    pub fn to_scale(&self) -> MatterScale {
        self.to_scale
    }

    pub fn support(&self) -> MatterTransitionSupport {
        self.support
    }

    pub fn evidence(&self) -> Option<&MatterClaim> {
        self.evidence
    }
}

/// Componentwise upper bound inherited from every required evidence dependency.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatterAuthorityCeiling {
    pub coordinate: EpistemicCoordinate,
}

impl MatterAuthorityCeiling {
    fn from_claim(claim: &MatterClaim) -> Result<Self, CrossScaleEvidenceError> {
        require_scientific_context(claim)?;
        Ok(Self {
            coordinate: claim.epistemic,
        })
    }

    fn include(&mut self, claim: &MatterClaim) -> Result<(), CrossScaleEvidenceError> {
        require_scientific_context(claim)?;
        self.coordinate.empirical = min(self.coordinate.empirical, claim.epistemic.empirical);
        self.coordinate.normative = min(self.coordinate.normative, claim.epistemic.normative);
        self.coordinate.materiality = min(self.coordinate.materiality, claim.epistemic.materiality);
        Ok(())
    }

    pub fn permits(&self, coordinate: EpistemicCoordinate) -> bool {
        coordinate.context == EpistemicContext::Scientific
            && coordinate.empirical <= self.coordinate.empirical
            && coordinate.normative <= self.coordinate.normative
            && coordinate.materiality <= self.coordinate.materiality
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossScaleDecision {
    AdmissibleInDomain,
    AdmissibleWithOutOfDomainEvidence,
    Refused,
}

/// Result of evaluating one ordered cross-scale evidence path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CrossScaleEvidenceAssessment {
    pub target_scale: MatterScale,
    pub authority_ceiling: MatterAuthorityCeiling,
    pub decision: CrossScaleDecision,
    pub source_claim_ids: Vec<String>,
    pub transition_ids: Vec<String>,
    pub transition_evidence_claim_ids: Vec<String>,
    pub out_of_domain_transition_ids: Vec<String>,
    pub blocking_transition_ids: Vec<String>,
}

impl CrossScaleEvidenceAssessment {
    pub fn has_out_of_domain_evidence(&self) -> bool {
        !self.out_of_domain_transition_ids.is_empty()
    }

    pub fn is_admissible(&self) -> bool {
        self.decision != CrossScaleDecision::Refused
    }

    /// Enforce this assessment against a derived *computational* Matter claim.
    ///
    /// This does not create the claim. It proves only that the candidate stays
    /// within the path's authority ceiling, targets the adjudicated scale, does
    /// not mint an observation/replication stage, and propagates OOD status.
    pub fn enforce_derived_computational_claim(
        &self,
        candidate: &MatterClaim,
    ) -> Result<(), CrossScaleEvidenceError> {
        if !self.is_admissible() {
            return Err(CrossScaleEvidenceError::CompositionRefused);
        }
        if candidate.scale != self.target_scale {
            return Err(CrossScaleEvidenceError::DerivedTargetScaleMismatch {
                expected: self.target_scale,
                actual: candidate.scale,
            });
        }
        require_scientific_context(candidate)?;
        if !candidate.validation_stage.is_local_computational() {
            return Err(CrossScaleEvidenceError::DerivedClaimCannotMintObservation);
        }
        if !self.authority_ceiling.permits(candidate.epistemic) {
            return Err(CrossScaleEvidenceError::AuthorityEscalation);
        }
        if self.has_out_of_domain_evidence()
            && !candidate
                .uncertainty
                .as_ref()
                .is_some_and(|uncertainty| uncertainty.out_of_domain)
        {
            return Err(CrossScaleEvidenceError::OutOfDomainStatusNotPropagated);
        }
        Ok(())
    }
}

/// Evaluate an ordered cross-scale evidence path.
///
/// The first transition must originate at the scale of at least one source claim;
/// transitions must form a contiguous path; the final transition must end at
/// `target_scale`. Supplemental source claims at other scales are allowed, but all
/// required claims remain part of the authority ceiling.
pub fn assess_cross_scale_evidence(
    source_claims: &[&MatterClaim],
    transitions: &[MatterScaleTransition<'_>],
    target_scale: MatterScale,
) -> Result<CrossScaleEvidenceAssessment, CrossScaleEvidenceError> {
    if source_claims.is_empty() {
        return Err(CrossScaleEvidenceError::EmptySourceClaims);
    }
    if transitions.is_empty() {
        return Err(CrossScaleEvidenceError::EmptyTransitionPath);
    }

    let mut dependency_ids = BTreeSet::new();
    let mut source_claim_ids = Vec::with_capacity(source_claims.len());
    for claim in source_claims {
        validate_claim_identity(claim)?;
        require_scientific_context(claim)?;
        if !dependency_ids.insert(claim.claim_id.as_str()) {
            return Err(CrossScaleEvidenceError::DuplicateDependencyClaimId(
                claim.claim_id.clone(),
            ));
        }
        source_claim_ids.push(claim.claim_id.clone());
    }

    if !source_claims
        .iter()
        .any(|claim| claim.scale == transitions[0].from_scale)
    {
        return Err(CrossScaleEvidenceError::PathDoesNotStartFromSourceScale {
            first_scale: transitions[0].from_scale,
        });
    }

    let mut transition_ids_seen = BTreeSet::new();
    let mut transition_ids = Vec::with_capacity(transitions.len());
    let mut transition_evidence_claim_ids = Vec::new();
    let mut out_of_domain_transition_ids = Vec::new();
    let mut blocking_transition_ids = Vec::new();

    let mut ceiling = MatterAuthorityCeiling::from_claim(source_claims[0])?;
    for claim in &source_claims[1..] {
        ceiling.include(claim)?;
    }

    for (index, transition) in transitions.iter().enumerate() {
        if transition.transition_id.trim().is_empty() {
            return Err(CrossScaleEvidenceError::EmptyTransitionId);
        }
        if !transition_ids_seen.insert(transition.transition_id) {
            return Err(CrossScaleEvidenceError::DuplicateTransitionId(
                transition.transition_id.to_string(),
            ));
        }
        if transition.from_scale == transition.to_scale {
            return Err(CrossScaleEvidenceError::DegenerateScaleTransition {
                transition_id: transition.transition_id.to_string(),
            });
        }
        if index > 0 && transitions[index - 1].to_scale != transition.from_scale {
            return Err(CrossScaleEvidenceError::NonContiguousScalePath {
                previous_to: transitions[index - 1].to_scale,
                next_from: transition.from_scale,
            });
        }
        transition_ids.push(transition.transition_id.to_string());

        match transition.support {
            MatterTransitionSupport::Unsupported | MatterTransitionSupport::MissingEvidence => {
                blocking_transition_ids.push(transition.transition_id.to_string());
            }
            MatterTransitionSupport::SupportedInDomain
            | MatterTransitionSupport::SupportedOutOfDomain => {
                let evidence = transition
                    .evidence
                    .ok_or_else(|| CrossScaleEvidenceError::SupportedTransitionMissingEvidence {
                        transition_id: transition.transition_id.to_string(),
                    })?;
                validate_claim_identity(evidence)?;
                require_scientific_context(evidence)?;
                if evidence.scale != transition.to_scale && evidence.scale != MatterScale::Multiscale {
                    return Err(CrossScaleEvidenceError::TransitionEvidenceScaleMismatch {
                        transition_id: transition.transition_id.to_string(),
                        expected: transition.to_scale,
                        actual: evidence.scale,
                    });
                }
                if evidence.validation_stage == MatterValidationStage::Hypothesis {
                    return Err(CrossScaleEvidenceError::HypothesisCannotSupportTransition {
                        transition_id: transition.transition_id.to_string(),
                    });
                }
                if !dependency_ids.insert(evidence.claim_id.as_str()) {
                    return Err(CrossScaleEvidenceError::DuplicateDependencyClaimId(
                        evidence.claim_id.clone(),
                    ));
                }

                let evidence_is_ood = evidence
                    .uncertainty
                    .as_ref()
                    .is_some_and(|uncertainty| uncertainty.out_of_domain);
                match transition.support {
                    MatterTransitionSupport::SupportedInDomain if evidence_is_ood => {
                        return Err(CrossScaleEvidenceError::InDomainTransitionCarriesOodEvidence {
                            transition_id: transition.transition_id.to_string(),
                        });
                    }
                    MatterTransitionSupport::SupportedOutOfDomain if !evidence_is_ood => {
                        return Err(CrossScaleEvidenceError::OutOfDomainTransitionMissingMarker {
                            transition_id: transition.transition_id.to_string(),
                        });
                    }
                    MatterTransitionSupport::SupportedOutOfDomain => {
                        out_of_domain_transition_ids.push(transition.transition_id.to_string());
                    }
                    MatterTransitionSupport::SupportedInDomain => {}
                    MatterTransitionSupport::Unsupported | MatterTransitionSupport::MissingEvidence => {
                        unreachable!("blocking transitions are handled before evidence admission")
                    }
                }

                ceiling.include(evidence)?;
                transition_evidence_claim_ids.push(evidence.claim_id.clone());
            }
        }
    }

    let final_scale = transitions
        .last()
        .expect("non-empty transition path checked above")
        .to_scale;
    if final_scale != target_scale {
        return Err(CrossScaleEvidenceError::PathDoesNotReachTargetScale {
            final_scale,
            target_scale,
        });
    }

    let decision = if !blocking_transition_ids.is_empty() {
        CrossScaleDecision::Refused
    } else if !out_of_domain_transition_ids.is_empty() {
        CrossScaleDecision::AdmissibleWithOutOfDomainEvidence
    } else {
        CrossScaleDecision::AdmissibleInDomain
    };

    Ok(CrossScaleEvidenceAssessment {
        target_scale,
        authority_ceiling: ceiling,
        decision,
        source_claim_ids,
        transition_ids,
        transition_evidence_claim_ids,
        out_of_domain_transition_ids,
        blocking_transition_ids,
    })
}

fn validate_claim_identity(claim: &MatterClaim) -> Result<(), CrossScaleEvidenceError> {
    if claim.claim_id.trim().is_empty() {
        return Err(CrossScaleEvidenceError::EmptyDependencyClaimId);
    }
    Ok(())
}

fn require_scientific_context(claim: &MatterClaim) -> Result<(), CrossScaleEvidenceError> {
    if claim.epistemic.context != EpistemicContext::Scientific {
        return Err(CrossScaleEvidenceError::NonScientificMatterClaim {
            claim_id: claim.claim_id.clone(),
            context: claim.epistemic.context,
        });
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CrossScaleEvidenceError {
    EmptySourceClaims,
    EmptyTransitionPath,
    EmptyTransitionId,
    EmptyDependencyClaimId,
    DuplicateDependencyClaimId(String),
    DuplicateTransitionId(String),
    DegenerateScaleTransition {
        transition_id: String,
    },
    PathDoesNotStartFromSourceScale {
        first_scale: MatterScale,
    },
    NonContiguousScalePath {
        previous_to: MatterScale,
        next_from: MatterScale,
    },
    PathDoesNotReachTargetScale {
        final_scale: MatterScale,
        target_scale: MatterScale,
    },
    SupportedTransitionMissingEvidence {
        transition_id: String,
    },
    TransitionEvidenceScaleMismatch {
        transition_id: String,
        expected: MatterScale,
        actual: MatterScale,
    },
    HypothesisCannotSupportTransition {
        transition_id: String,
    },
    InDomainTransitionCarriesOodEvidence {
        transition_id: String,
    },
    OutOfDomainTransitionMissingMarker {
        transition_id: String,
    },
    NonScientificMatterClaim {
        claim_id: String,
        context: EpistemicContext,
    },
    CompositionRefused,
    DerivedTargetScaleMismatch {
        expected: MatterScale,
        actual: MatterScale,
    },
    DerivedClaimCannotMintObservation,
    AuthorityEscalation,
    OutOfDomainStatusNotPropagated,
}

impl fmt::Display for CrossScaleEvidenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptySourceClaims => write!(f, "cross-scale evidence requires at least one source claim"),
            Self::EmptyTransitionPath => write!(f, "cross-scale evidence requires at least one scale transition"),
            Self::EmptyTransitionId => write!(f, "cross-scale transition id must not be empty"),
            Self::EmptyDependencyClaimId => write!(f, "cross-scale dependency claim id must not be empty"),
            Self::DuplicateDependencyClaimId(id) => {
                write!(f, "cross-scale dependency claim `{id}` is reused in more than one required role")
            }
            Self::DuplicateTransitionId(id) => {
                write!(f, "cross-scale transition id `{id}` occurs more than once")
            }
            Self::DegenerateScaleTransition { transition_id } => write!(
                f,
                "cross-scale transition `{transition_id}` must connect two distinct matter scales"
            ),
            Self::PathDoesNotStartFromSourceScale { first_scale } => write!(
                f,
                "first cross-scale transition begins at {first_scale:?}, but no source claim addresses that scale"
            ),
            Self::NonContiguousScalePath {
                previous_to,
                next_from,
            } => write!(
                f,
                "cross-scale path is discontinuous: previous transition ends at {previous_to:?}, next begins at {next_from:?}"
            ),
            Self::PathDoesNotReachTargetScale {
                final_scale,
                target_scale,
            } => write!(
                f,
                "cross-scale path ends at {final_scale:?}, expected target {target_scale:?}"
            ),
            Self::SupportedTransitionMissingEvidence { transition_id } => write!(
                f,
                "supported transition `{transition_id}` has no evidence claim"
            ),
            Self::TransitionEvidenceScaleMismatch {
                transition_id,
                expected,
                actual,
            } => write!(
                f,
                "transition `{transition_id}` evidence addresses {actual:?}, expected {expected:?} or Multiscale"
            ),
            Self::HypothesisCannotSupportTransition { transition_id } => write!(
                f,
                "transition `{transition_id}` cannot be supported by a hypothesis-only claim"
            ),
            Self::InDomainTransitionCarriesOodEvidence { transition_id } => write!(
                f,
                "transition `{transition_id}` is labeled in-domain but its evidence is marked out-of-domain"
            ),
            Self::OutOfDomainTransitionMissingMarker { transition_id } => write!(
                f,
                "transition `{transition_id}` is labeled out-of-domain but its evidence does not carry an OOD uncertainty marker"
            ),
            Self::NonScientificMatterClaim { claim_id, context } => write!(
                f,
                "Matter claim `{claim_id}` uses {context:?} epistemic context; cross-scale scientific calculus requires Scientific"
            ),
            Self::CompositionRefused => write!(
                f,
                "cross-scale composition is refused because at least one required transition is unsupported or missing evidence"
            ),
            Self::DerivedTargetScaleMismatch { expected, actual } => write!(
                f,
                "derived claim addresses {actual:?}, expected cross-scale target {expected:?}"
            ),
            Self::DerivedClaimCannotMintObservation => write!(
                f,
                "a derived computational cross-scale claim cannot mint experimental observation or independent replication"
            ),
            Self::AuthorityEscalation => write!(
                f,
                "derived claim exceeds the weakest required dependency on at least one E/N/M axis"
            ),
            Self::OutOfDomainStatusNotPropagated => write!(
                f,
                "derived claim must retain an out-of-domain uncertainty marker from its required transition evidence"
            ),
        }
    }
}

impl std::error::Error for CrossScaleEvidenceError {}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_epistemic_types::{
        MatterClaimError, MatterSolverCapability, MatterSolverProfile, MatterUncertainty,
    };

    fn solver() -> MatterSolverProfile {
        MatterSolverProfile {
            name: "cross-scale-fixture".to_string(),
            version: "1".to_string(),
            method: "fixture physics".to_string(),
            capabilities: vec![MatterSolverCapability::CoupledMultiphysics],
            limitations: vec!["fixture only".to_string()],
        }
    }

    fn claim(
        id: &str,
        scale: MatterScale,
        stage: MatterValidationStage,
        empirical: EmpiricalLevel,
        normative: NormativeLevel,
        materiality: MaterialityLevel,
        ood: bool,
    ) -> MatterClaim {
        let mut claim = MatterClaim::local_computational(id, id, scale, stage, solver()).unwrap();
        claim.epistemic = EpistemicCoordinate::new(
            empirical,
            normative,
            materiality,
            EpistemicContext::Scientific,
        );
        if ood {
            claim = claim.with_uncertainty(
                MatterUncertainty::new(1.0, "arb", "fixture OOD diagnostic", true).unwrap(),
            );
        }
        claim
    }

    fn high_source() -> MatterClaim {
        claim(
            "nuclear-source",
            MatterScale::Nuclear,
            MatterValidationStage::ReferenceBenchmarked,
            EmpiricalLevel::E4PubliclyReproducible,
            NormativeLevel::N2Network,
            MaterialityLevel::M2Persistent,
            false,
        )
    }

    fn weak_transition() -> MatterClaim {
        claim(
            "nuclear-to-electronic",
            MatterScale::AtomicElectronic,
            MatterValidationStage::HighFidelitySimulated,
            EmpiricalLevel::E1Testimonial,
            NormativeLevel::N0Personal,
            MaterialityLevel::M0Ephemeral,
            false,
        )
    }

    #[test]
    fn weakest_transition_caps_all_axes() {
        let source = high_source();
        let transition_evidence = weak_transition();
        let transition = MatterScaleTransition::supported_in_domain(
            "nuclear->electronic",
            MatterScale::Nuclear,
            MatterScale::AtomicElectronic,
            &transition_evidence,
        );
        let assessment = assess_cross_scale_evidence(
            &[&source],
            &[transition],
            MatterScale::AtomicElectronic,
        )
        .unwrap();

        assert_eq!(
            assessment.authority_ceiling.coordinate.empirical,
            EmpiricalLevel::E1Testimonial
        );
        assert_eq!(
            assessment.authority_ceiling.coordinate.normative,
            NormativeLevel::N0Personal
        );
        assert_eq!(
            assessment.authority_ceiling.coordinate.materiality,
            MaterialityLevel::M0Ephemeral
        );
        assert_eq!(assessment.decision, CrossScaleDecision::AdmissibleInDomain);
    }

    #[test]
    fn candidate_cannot_exceed_weakest_dependency() {
        let source = high_source();
        let transition_evidence = weak_transition();
        let assessment = assess_cross_scale_evidence(
            &[&source],
            &[MatterScaleTransition::supported_in_domain(
                "nuclear->electronic",
                MatterScale::Nuclear,
                MatterScale::AtomicElectronic,
                &transition_evidence,
            )],
            MatterScale::AtomicElectronic,
        )
        .unwrap();

        let candidate = claim(
            "too-strong",
            MatterScale::AtomicElectronic,
            MatterValidationStage::HighFidelitySimulated,
            EmpiricalLevel::E2PrivatelyVerifiable,
            NormativeLevel::N0Personal,
            MaterialityLevel::M0Ephemeral,
            false,
        );
        assert_eq!(
            assessment
                .enforce_derived_computational_claim(&candidate)
                .unwrap_err(),
            CrossScaleEvidenceError::AuthorityEscalation
        );
    }

    #[test]
    fn unsupported_transition_refuses_composition() {
        let source = high_source();
        let assessment = assess_cross_scale_evidence(
            &[&source],
            &[MatterScaleTransition::unsupported(
                "nuclear->electronic",
                MatterScale::Nuclear,
                MatterScale::AtomicElectronic,
            )],
            MatterScale::AtomicElectronic,
        )
        .unwrap();
        assert_eq!(assessment.decision, CrossScaleDecision::Refused);

        let candidate = weak_transition();
        assert_eq!(
            assessment
                .enforce_derived_computational_claim(&candidate)
                .unwrap_err(),
            CrossScaleEvidenceError::CompositionRefused
        );
    }

    #[test]
    fn ood_transition_requires_ood_evidence_marker() {
        let source = high_source();
        let transition_evidence = weak_transition();
        let result = assess_cross_scale_evidence(
            &[&source],
            &[MatterScaleTransition::supported_out_of_domain(
                "nuclear->electronic",
                MatterScale::Nuclear,
                MatterScale::AtomicElectronic,
                &transition_evidence,
            )],
            MatterScale::AtomicElectronic,
        );
        assert!(matches!(
            result,
            Err(CrossScaleEvidenceError::OutOfDomainTransitionMissingMarker { .. })
        ));
    }

    #[test]
    fn ood_status_must_propagate_to_derived_claim() {
        let source = high_source();
        let transition_evidence = claim(
            "nuclear-to-electronic-ood",
            MatterScale::AtomicElectronic,
            MatterValidationStage::HighFidelitySimulated,
            EmpiricalLevel::E1Testimonial,
            NormativeLevel::N0Personal,
            MaterialityLevel::M0Ephemeral,
            true,
        );
        let assessment = assess_cross_scale_evidence(
            &[&source],
            &[MatterScaleTransition::supported_out_of_domain(
                "nuclear->electronic",
                MatterScale::Nuclear,
                MatterScale::AtomicElectronic,
                &transition_evidence,
            )],
            MatterScale::AtomicElectronic,
        )
        .unwrap();
        assert_eq!(
            assessment.decision,
            CrossScaleDecision::AdmissibleWithOutOfDomainEvidence
        );

        let candidate_without_ood = weak_transition();
        assert_eq!(
            assessment
                .enforce_derived_computational_claim(&candidate_without_ood)
                .unwrap_err(),
            CrossScaleEvidenceError::OutOfDomainStatusNotPropagated
        );

        let candidate_with_ood = claim(
            "derived-ood",
            MatterScale::AtomicElectronic,
            MatterValidationStage::HighFidelitySimulated,
            EmpiricalLevel::E1Testimonial,
            NormativeLevel::N0Personal,
            MaterialityLevel::M0Ephemeral,
            true,
        );
        assessment
            .enforce_derived_computational_claim(&candidate_with_ood)
            .unwrap();
    }

    #[test]
    fn in_domain_transition_rejects_ood_evidence() {
        let source = high_source();
        let transition_evidence = claim(
            "unexpected-ood",
            MatterScale::AtomicElectronic,
            MatterValidationStage::HighFidelitySimulated,
            EmpiricalLevel::E1Testimonial,
            NormativeLevel::N0Personal,
            MaterialityLevel::M0Ephemeral,
            true,
        );
        let result = assess_cross_scale_evidence(
            &[&source],
            &[MatterScaleTransition::supported_in_domain(
                "nuclear->electronic",
                MatterScale::Nuclear,
                MatterScale::AtomicElectronic,
                &transition_evidence,
            )],
            MatterScale::AtomicElectronic,
        );
        assert!(matches!(
            result,
            Err(CrossScaleEvidenceError::InDomainTransitionCarriesOodEvidence { .. })
        ));
    }

    #[test]
    fn scale_path_must_be_contiguous() {
        let source = high_source();
        let electronic = weak_transition();
        let material = claim(
            "material-evidence",
            MatterScale::Crystal,
            MatterValidationStage::PhysicsSimulated,
            EmpiricalLevel::E1Testimonial,
            NormativeLevel::N0Personal,
            MaterialityLevel::M0Ephemeral,
            false,
        );
        let result = assess_cross_scale_evidence(
            &[&source],
            &[
                MatterScaleTransition::supported_in_domain(
                    "nuclear->electronic",
                    MatterScale::Nuclear,
                    MatterScale::AtomicElectronic,
                    &electronic,
                ),
                MatterScaleTransition::supported_in_domain(
                    "molecular->crystal",
                    MatterScale::Molecular,
                    MatterScale::Crystal,
                    &material,
                ),
            ],
            MatterScale::Crystal,
        );
        assert!(matches!(
            result,
            Err(CrossScaleEvidenceError::NonContiguousScalePath { .. })
        ));
    }

    #[test]
    fn derived_computation_cannot_inherit_experimental_stage() {
        let source = high_source();
        let transition_evidence = weak_transition();
        let assessment = assess_cross_scale_evidence(
            &[&source],
            &[MatterScaleTransition::supported_in_domain(
                "nuclear->electronic",
                MatterScale::Nuclear,
                MatterScale::AtomicElectronic,
                &transition_evidence,
            )],
            MatterScale::AtomicElectronic,
        )
        .unwrap();

        let mut candidate = weak_transition();
        candidate.validation_stage = MatterValidationStage::ExperimentallyObserved;
        assert_eq!(
            assessment
                .enforce_derived_computational_claim(&candidate)
                .unwrap_err(),
            CrossScaleEvidenceError::DerivedClaimCannotMintObservation
        );
    }

    #[test]
    fn local_constructor_still_rejects_experimental_stage_independently() {
        assert_eq!(
            MatterClaim::local_computational(
                "bad",
                "bad",
                MatterScale::Crystal,
                MatterValidationStage::ExperimentallyObserved,
                solver(),
            )
            .unwrap_err(),
            MatterClaimError::ObservationRequired
        );
    }
}
