// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Static EUREKA claim constitution.
//!
//! This module deliberately contains **no experiment runner**. It freezes the
//! initial operational-understanding claim families, their minimum evidence
//! classes, shared-dependency hints, and anti-leakage invariants before
//! EUREKA target experiments are implemented.
//!
//! Parent program: <https://github.com/Luminous-Dynamics/symthaea/issues/2045>
//! Constitution issue: <https://github.com/Luminous-Dynamics/symthaea/issues/2046>

/// Independently-scored EUREKA construct families.
///
/// There is intentionally no aggregate `UnderstandingScore`: strength in one
/// family must not average away failure or missing evidence in another.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnderstandingClaimFamily {
    RelationalRepresentation,
    PredictiveModeling,
    CausalIntervention,
    CounterfactualReasoning,
    CompositionalGeneralization,
    StructuralTransfer,
    ConceptFormation,
    MetacognitiveMonitoring,
    MetacognitiveControl,
    ActiveInquiry,
    ScientificModelRevision,
}

impl UnderstandingClaimFamily {
    pub const ALL: [Self; 11] = [
        Self::RelationalRepresentation,
        Self::PredictiveModeling,
        Self::CausalIntervention,
        Self::CounterfactualReasoning,
        Self::CompositionalGeneralization,
        Self::StructuralTransfer,
        Self::ConceptFormation,
        Self::MetacognitiveMonitoring,
        Self::MetacognitiveControl,
        Self::ActiveInquiry,
        Self::ScientificModelRevision,
    ];

    /// Stable, machine-readable identity for evidence manifests.
    pub const fn stable_id(self) -> &'static str {
        match self {
            Self::RelationalRepresentation => "EUREKA.RELATIONAL_REPRESENTATION.v1",
            Self::PredictiveModeling => "EUREKA.PREDICTIVE_MODELING.v1",
            Self::CausalIntervention => "EUREKA.CAUSAL_INTERVENTION.v1",
            Self::CounterfactualReasoning => "EUREKA.COUNTERFACTUAL_REASONING.v1",
            Self::CompositionalGeneralization => "EUREKA.COMPOSITIONAL_GENERALIZATION.v1",
            Self::StructuralTransfer => "EUREKA.STRUCTURAL_TRANSFER.v1",
            Self::ConceptFormation => "EUREKA.CONCEPT_FORMATION.v1",
            Self::MetacognitiveMonitoring => "EUREKA.METACOGNITIVE_MONITORING.v1",
            Self::MetacognitiveControl => "EUREKA.METACOGNITIVE_CONTROL.v1",
            Self::ActiveInquiry => "EUREKA.ACTIVE_INQUIRY.v1",
            Self::ScientificModelRevision => "EUREKA.SCIENTIFIC_MODEL_REVISION.v1",
        }
    }
}

/// Protocol/evidence maturity is separate from the scientific result.
///
/// A highly mature protocol may produce a null or negative result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum UnderstandingMaturity {
    Specified,
    Implemented,
    Executed,
    CausallyQualified,
    Replicated,
    IndependentlyReplicated,
}

/// Scientific direction/result is separate from protocol maturity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScientificDisposition {
    Supported,
    Mixed,
    Null,
    Negative,
    Inconclusive,
}

/// Generic evidence classes used by the static EUREKA claim constitution.
///
/// These describe evidence *roles*, not authority. Domain-specific runners
/// remain responsible for proving that concrete artifacts actually satisfy
/// the declared role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EvidenceClass {
    ProspectiveCommitment,
    HeldOutOutcome,
    FreshEvidence,
    ShortcutBaseline,
    PositiveControl,
    TargetedIntervention,
    MatchedSham,
    SelectiveRescue,
    CrossDomainTransfer,
    MultiSeedReplication,
    IndependentEvaluation,
}

/// Cross-cutting rules that every EUREKA runner must preserve.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProtocolInvariant {
    HiddenTruthInaccessibleToTarget,
    NoPostOutcomeProspectiveCredit,
    EvidenceTransformDoesNotCreateFreshEvidence,
    HeldOutCorpusSeparated,
    SharedEvidenceIsNotIndependentReplication,
    MissingOutcomeCannotSupportPositiveClaim,
    SelfDescriptionIsNonAuthoritative,
    PredictionIsNotCausalEvidence,
    SimulationIsNotPhysicalTruth,
    UnderstandingDoesNotGrantAuthority,
}

impl ProtocolInvariant {
    pub const ALL: [Self; 10] = [
        Self::HiddenTruthInaccessibleToTarget,
        Self::NoPostOutcomeProspectiveCredit,
        Self::EvidenceTransformDoesNotCreateFreshEvidence,
        Self::HeldOutCorpusSeparated,
        Self::SharedEvidenceIsNotIndependentReplication,
        Self::MissingOutcomeCannotSupportPositiveClaim,
        Self::SelfDescriptionIsNonAuthoritative,
        Self::PredictionIsNotCausalEvidence,
        Self::SimulationIsNotPhysicalTruth,
        Self::UnderstandingDoesNotGrantAuthority,
    ];
}

/// One static construct definition.
///
/// `shared_dependency_group` is deliberately descriptive: sharing a group
/// warns evaluators not to count downstream results as independent
/// replications. It does not imply that all claims in a group are equivalent.
#[derive(Debug, Clone, Copy)]
pub struct UnderstandingClaimSpec {
    pub family: UnderstandingClaimFamily,
    pub stable_id: &'static str,
    pub proposition: &'static str,
    pub minimum_evidence: &'static [EvidenceClass],
    pub shared_dependency_group: Option<&'static str>,
    pub explicit_non_claims: &'static [&'static str],
}

impl UnderstandingClaimSpec {
    pub fn requires(&self, evidence: EvidenceClass) -> bool {
        self.minimum_evidence.contains(&evidence)
    }
}

/// Initial EUREKA v1 construct constitution.
///
/// These are intentionally stronger than "a module exists" or "the system
/// emitted a plausible answer" and intentionally weaker than metaphysical,
/// consciousness, alignment, truth-authority, or action-authority claims.
pub const EUREKA_CLAIM_SPECS: [UnderstandingClaimSpec; 11] = [
    UnderstandingClaimSpec {
        family: UnderstandingClaimFamily::RelationalRepresentation,
        stable_id: UnderstandingClaimFamily::RelationalRepresentation.stable_id(),
        proposition: "Relations represented by the target system support held-out relational consequences beyond shortcut controls.",
        minimum_evidence: &[
            EvidenceClass::HeldOutOutcome,
            EvidenceClass::ShortcutBaseline,
            EvidenceClass::PositiveControl,
            EvidenceClass::MultiSeedReplication,
        ],
        shared_dependency_group: Some("representation"),
        explicit_non_claims: &["ontology truth", "causal understanding", "consciousness"],
    },
    UnderstandingClaimSpec {
        family: UnderstandingClaimFamily::PredictiveModeling,
        stable_id: UnderstandingClaimFamily::PredictiveModeling.stable_id(),
        proposition: "Prospectively committed world-state consequences predict held-out outcomes better than copy and simple transition baselines.",
        minimum_evidence: &[
            EvidenceClass::ProspectiveCommitment,
            EvidenceClass::HeldOutOutcome,
            EvidenceClass::ShortcutBaseline,
            EvidenceClass::MultiSeedReplication,
        ],
        shared_dependency_group: Some("world-model"),
        explicit_non_claims: &["causal mechanism identification", "physical truth", "authority"],
    },
    UnderstandingClaimSpec {
        family: UnderstandingClaimFamily::CausalIntervention,
        stable_id: UnderstandingClaimFamily::CausalIntervention.stable_id(),
        proposition: "Changing an independently verified intervention while controlling relevant inputs changes predicted and observed outcomes in the preregistered direction.",
        minimum_evidence: &[
            EvidenceClass::ProspectiveCommitment,
            EvidenceClass::HeldOutOutcome,
            EvidenceClass::TargetedIntervention,
            EvidenceClass::MatchedSham,
            EvidenceClass::PositiveControl,
            EvidenceClass::MultiSeedReplication,
        ],
        shared_dependency_group: Some("causal-model"),
        explicit_non_claims: &["universal causal discovery", "mechanism truth outside the tested envelope"],
    },
    UnderstandingClaimSpec {
        family: UnderstandingClaimFamily::CounterfactualReasoning,
        stable_id: UnderstandingClaimFamily::CounterfactualReasoning.stable_id(),
        proposition: "Counterfactual outcomes preserve the frozen actual-history context except where the declared intervention and its causal descendants require change.",
        minimum_evidence: &[
            EvidenceClass::ProspectiveCommitment,
            EvidenceClass::HeldOutOutcome,
            EvidenceClass::ShortcutBaseline,
            EvidenceClass::PositiveControl,
            EvidenceClass::MultiSeedReplication,
        ],
        shared_dependency_group: Some("causal-model"),
        explicit_non_claims: &["observed alternate world", "counterfactual truth without model assumptions"],
    },
    UnderstandingClaimSpec {
        family: UnderstandingClaimFamily::CompositionalGeneralization,
        stable_id: UnderstandingClaimFamily::CompositionalGeneralization.stable_id(),
        proposition: "Known primitives and relations support held-out systematic, productive, or substitutive compositions beyond memorization controls.",
        minimum_evidence: &[
            EvidenceClass::HeldOutOutcome,
            EvidenceClass::ShortcutBaseline,
            EvidenceClass::PositiveControl,
            EvidenceClass::MultiSeedReplication,
        ],
        shared_dependency_group: Some("representation"),
        explicit_non_claims: &["general intelligence", "unbounded systematicity"],
    },
    UnderstandingClaimSpec {
        family: UnderstandingClaimFamily::StructuralTransfer,
        stable_id: UnderstandingClaimFamily::StructuralTransfer.stable_id(),
        proposition: "A learned relational or causal structure improves held-out performance in a surface-divergent domain without privileged correspondence labels.",
        minimum_evidence: &[
            EvidenceClass::HeldOutOutcome,
            EvidenceClass::ShortcutBaseline,
            EvidenceClass::CrossDomainTransfer,
            EvidenceClass::MultiSeedReplication,
        ],
        shared_dependency_group: Some("abstraction-transfer"),
        explicit_non_claims: &["surface identity matching", "universal transfer"],
    },
    UnderstandingClaimSpec {
        family: UnderstandingClaimFamily::ConceptFormation,
        stable_id: UnderstandingClaimFamily::ConceptFormation.stable_id(),
        proposition: "A target-generated abstraction not supplied as the evaluator answer improves prospective held-out prediction, intervention, transfer, compression, or sample efficiency against preregistered latent controls.",
        minimum_evidence: &[
            EvidenceClass::ProspectiveCommitment,
            EvidenceClass::HeldOutOutcome,
            EvidenceClass::ShortcutBaseline,
            EvidenceClass::CrossDomainTransfer,
            EvidenceClass::MultiSeedReplication,
        ],
        shared_dependency_group: Some("abstraction-transfer"),
        explicit_non_claims: &["truth from compression", "concept validity from naming alone"],
    },
    UnderstandingClaimSpec {
        family: UnderstandingClaimFamily::MetacognitiveMonitoring,
        stable_id: UnderstandingClaimFamily::MetacognitiveMonitoring.stable_id(),
        proposition: "Frozen pre-answer competence forecasts predict independent task success, error, or uncertainty better than trivial confidence baselines.",
        minimum_evidence: &[
            EvidenceClass::ProspectiveCommitment,
            EvidenceClass::HeldOutOutcome,
            EvidenceClass::ShortcutBaseline,
            EvidenceClass::MultiSeedReplication,
        ],
        shared_dependency_group: Some("metacognition"),
        explicit_non_claims: &["metacognitive control", "self-knowledge from retrospective self-report"],
    },
    UnderstandingClaimSpec {
        family: UnderstandingClaimFamily::MetacognitiveControl,
        stable_id: UnderstandingClaimFamily::MetacognitiveControl.stable_id(),
        proposition: "Selective change to higher-order competence information changes inquiry, abstention, learning, belief, or action policy under matched first-order evidence.",
        minimum_evidence: &[
            EvidenceClass::HeldOutOutcome,
            EvidenceClass::TargetedIntervention,
            EvidenceClass::MatchedSham,
            EvidenceClass::SelectiveRescue,
            EvidenceClass::MultiSeedReplication,
        ],
        shared_dependency_group: Some("metacognition"),
        explicit_non_claims: &["confidence calibration alone", "obedience", "authority"],
    },
    UnderstandingClaimSpec {
        family: UnderstandingClaimFamily::ActiveInquiry,
        stable_id: UnderstandingClaimFamily::ActiveInquiry.stable_id(),
        proposition: "Prospectively selected information-gathering actions produce greater realized information gain per declared cost than preregistered exploration controls.",
        minimum_evidence: &[
            EvidenceClass::ProspectiveCommitment,
            EvidenceClass::FreshEvidence,
            EvidenceClass::HeldOutOutcome,
            EvidenceClass::ShortcutBaseline,
            EvidenceClass::MultiSeedReplication,
        ],
        shared_dependency_group: Some("epistemic-control"),
        explicit_non_claims: &["predicted information gain as evidence", "permission to act"],
    },
    UnderstandingClaimSpec {
        family: UnderstandingClaimFamily::ScientificModelRevision,
        stable_id: UnderstandingClaimFamily::ScientificModelRevision.stable_id(),
        proposition: "Fresh contradictory or regime-shift evidence causes bounded model revision that improves later held-out prediction or intervention performance without rewriting prior evidence.",
        minimum_evidence: &[
            EvidenceClass::ProspectiveCommitment,
            EvidenceClass::FreshEvidence,
            EvidenceClass::HeldOutOutcome,
            EvidenceClass::ShortcutBaseline,
            EvidenceClass::MultiSeedReplication,
        ],
        shared_dependency_group: Some("epistemic-control"),
        explicit_non_claims: &["automatic truth convergence", "self-certification", "promotion authority"],
    },
];

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn every_claim_family_has_exactly_one_v1_spec() {
        assert_eq!(EUREKA_CLAIM_SPECS.len(), UnderstandingClaimFamily::ALL.len());

        let mut seen = HashSet::new();
        for spec in EUREKA_CLAIM_SPECS {
            assert!(seen.insert(spec.family), "duplicate family: {:?}", spec.family);
        }

        for family in UnderstandingClaimFamily::ALL {
            assert!(seen.contains(&family), "missing family: {family:?}");
        }
    }

    #[test]
    fn stable_ids_are_unique_and_match_family_identity() {
        let mut ids = HashSet::new();
        for spec in EUREKA_CLAIM_SPECS {
            assert_eq!(spec.stable_id, spec.family.stable_id());
            assert!(ids.insert(spec.stable_id), "duplicate id: {}", spec.stable_id);
        }
    }

    #[test]
    fn every_claim_has_a_nonempty_proposition_evidence_and_nonclaim_boundary() {
        for spec in EUREKA_CLAIM_SPECS {
            assert!(!spec.proposition.trim().is_empty(), "empty proposition: {}", spec.stable_id);
            assert!(!spec.minimum_evidence.is_empty(), "no evidence roles: {}", spec.stable_id);
            assert!(!spec.explicit_non_claims.is_empty(), "no non-claims: {}", spec.stable_id);
        }
    }

    #[test]
    fn causal_intervention_requires_target_sham_and_positive_control() {
        let spec = spec(UnderstandingClaimFamily::CausalIntervention);
        assert!(spec.requires(EvidenceClass::TargetedIntervention));
        assert!(spec.requires(EvidenceClass::MatchedSham));
        assert!(spec.requires(EvidenceClass::PositiveControl));
        assert!(spec.requires(EvidenceClass::ProspectiveCommitment));
    }

    #[test]
    fn structural_transfer_requires_cross_domain_evidence() {
        let spec = spec(UnderstandingClaimFamily::StructuralTransfer);
        assert!(spec.requires(EvidenceClass::CrossDomainTransfer));
        assert!(spec.requires(EvidenceClass::ShortcutBaseline));
    }

    #[test]
    fn concept_formation_requires_prospective_and_transfer_evidence() {
        let spec = spec(UnderstandingClaimFamily::ConceptFormation);
        assert!(spec.requires(EvidenceClass::ProspectiveCommitment));
        assert!(spec.requires(EvidenceClass::CrossDomainTransfer));
        assert!(spec.requires(EvidenceClass::ShortcutBaseline));
    }

    #[test]
    fn metacognitive_monitoring_is_prospective_but_control_requires_causal_manipulation() {
        let monitoring = spec(UnderstandingClaimFamily::MetacognitiveMonitoring);
        assert!(monitoring.requires(EvidenceClass::ProspectiveCommitment));
        assert!(!monitoring.requires(EvidenceClass::TargetedIntervention));

        let control = spec(UnderstandingClaimFamily::MetacognitiveControl);
        assert!(control.requires(EvidenceClass::TargetedIntervention));
        assert!(control.requires(EvidenceClass::MatchedSham));
        assert!(control.requires(EvidenceClass::SelectiveRescue));
    }

    #[test]
    fn active_inquiry_requires_fresh_realized_evidence() {
        let spec = spec(UnderstandingClaimFamily::ActiveInquiry);
        assert!(spec.requires(EvidenceClass::ProspectiveCommitment));
        assert!(spec.requires(EvidenceClass::FreshEvidence));
        assert!(spec.requires(EvidenceClass::HeldOutOutcome));
    }

    #[test]
    fn protocol_invariant_set_is_unique() {
        let mut seen = HashSet::new();
        for invariant in ProtocolInvariant::ALL {
            assert!(seen.insert(invariant), "duplicate invariant: {invariant:?}");
        }
    }

    fn spec(family: UnderstandingClaimFamily) -> &'static UnderstandingClaimSpec {
        EUREKA_CLAIM_SPECS
            .iter()
            .find(|spec| spec.family == family)
            .expect("all v1 families must have a spec")
    }
}
