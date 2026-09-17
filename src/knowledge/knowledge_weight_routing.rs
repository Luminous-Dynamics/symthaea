// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Passive authority routing for knowledge weights.
//!
//! Legacy `TemporalFact::confidence` currently mixes evidence-like support with
//! retrieval recency, replay/consolidation, contradiction handling, and direct
//! caller adjustment. This module introduces typed dimensions and a conservative
//! routing gate so those mechanisms can be separated before any live migration.
//!
//! The gate is proposal-only: it never changes a fact, ledger, confidence value,
//! world model, causal graph, or action-selection state.

use super::claim_evidence::EvidenceId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KnowledgeWeightDimension {
    /// Evidence-grounded support for treating a proposition as more or less credible.
    EpistemicSupport,
    /// Ease/priority of retrieval for current cognition.
    Accessibility,
    /// Resistance to eviction/forgetting in memory storage.
    Retention,
    /// Strength of internally rehearsed/consolidated representation.
    Consolidation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KnowledgeWeightSource {
    /// A separately admitted EKM evidence record or evidence set.
    AdmittedEvidence,
    /// Ordinary retrieval/search/access.
    Retrieval,
    /// HDC/vector similarity match or near-duplicate exposure.
    SimilarityMatch,
    /// Dream/replay/rehearsal activity generated internally.
    DreamReplay,
    /// Generic consolidation applied because a representation is marked causal.
    CausalConsolidation,
    /// Storage/forgetting dynamics unrelated to new external evidence.
    MemoryDecay,
    /// Current task/context relevance.
    TaskRelevance,
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct BoundedWeight(f32);

impl BoundedWeight {
    pub fn new(value: f32) -> Result<Self, KnowledgeWeightError> {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(KnowledgeWeightError::WeightOutOfRange(value));
        }
        Ok(Self(value))
    }

    pub fn get(self) -> f32 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct SignedWeightDelta(f32);

impl SignedWeightDelta {
    pub fn new(value: f32) -> Result<Self, KnowledgeWeightError> {
        if !value.is_finite() || !(-1.0..=1.0).contains(&value) {
            return Err(KnowledgeWeightError::DeltaOutOfRange(value));
        }
        Ok(Self(value))
    }

    pub fn get(self) -> f32 {
        self.0
    }
}

/// Multidimensional replacement target for the overloaded legacy confidence field.
///
/// `None` means not represented/assessed yet; it is not silently interpreted as zero.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct KnowledgeWeightVector {
    pub epistemic_support: Option<BoundedWeight>,
    pub accessibility: Option<BoundedWeight>,
    pub retention: Option<BoundedWeight>,
    pub consolidation: Option<BoundedWeight>,
}

impl KnowledgeWeightVector {
    pub fn get(&self, dimension: KnowledgeWeightDimension) -> Option<BoundedWeight> {
        match dimension {
            KnowledgeWeightDimension::EpistemicSupport => self.epistemic_support,
            KnowledgeWeightDimension::Accessibility => self.accessibility,
            KnowledgeWeightDimension::Retention => self.retention,
            KnowledgeWeightDimension::Consolidation => self.consolidation,
        }
    }

    pub fn assessed_dimension_count(&self) -> usize {
        [
            self.epistemic_support,
            self.accessibility,
            self.retention,
            self.consolidation,
        ]
        .into_iter()
        .flatten()
        .count()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct KnowledgeWeightUpdateProposal {
    pub dimension: KnowledgeWeightDimension,
    pub source: KnowledgeWeightSource,
    pub delta: SignedWeightDelta,
    /// Evidence IDs that justify an epistemic-support proposal.
    ///
    /// Non-epistemic proposals may retain evidence IDs for audit context, but only
    /// `AdmittedEvidence` may target `EpistemicSupport`.
    pub basis_evidence_ids: Vec<EvidenceId>,
    pub rationale: String,
}

impl KnowledgeWeightUpdateProposal {
    pub fn new(
        dimension: KnowledgeWeightDimension,
        source: KnowledgeWeightSource,
        delta: f32,
        basis_evidence_ids: Vec<EvidenceId>,
        rationale: impl Into<String>,
    ) -> Result<Self, KnowledgeWeightError> {
        let rationale = rationale.into();
        if rationale.trim().is_empty() {
            return Err(KnowledgeWeightError::EmptyRationale);
        }
        Ok(Self {
            dimension,
            source,
            delta: SignedWeightDelta::new(delta)?,
            basis_evidence_ids,
            rationale,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KnowledgeWeightRoutingFailure {
    SourceCannotUpdateDimension {
        source: KnowledgeWeightSource,
        dimension: KnowledgeWeightDimension,
    },
    EpistemicUpdateHasNoEvidenceBasis,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KnowledgeWeightRoutingDecision {
    eligible: bool,
    failures: Vec<KnowledgeWeightRoutingFailure>,
}

impl KnowledgeWeightRoutingDecision {
    pub fn eligible(&self) -> bool {
        self.eligible
    }

    pub fn failures(&self) -> &[KnowledgeWeightRoutingFailure] {
        &self.failures
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KnowledgeWeightAuthorityRouter;

impl KnowledgeWeightAuthorityRouter {
    pub fn evaluate(proposal: &KnowledgeWeightUpdateProposal) -> KnowledgeWeightRoutingDecision {
        let mut failures = Vec::new();

        if !source_may_target(proposal.source, proposal.dimension) {
            failures.push(KnowledgeWeightRoutingFailure::SourceCannotUpdateDimension {
                source: proposal.source,
                dimension: proposal.dimension,
            });
        }

        if proposal.dimension == KnowledgeWeightDimension::EpistemicSupport
            && proposal.basis_evidence_ids.is_empty()
        {
            failures.push(KnowledgeWeightRoutingFailure::EpistemicUpdateHasNoEvidenceBasis);
        }

        KnowledgeWeightRoutingDecision {
            eligible: failures.is_empty(),
            failures,
        }
    }
}

fn source_may_target(
    source: KnowledgeWeightSource,
    dimension: KnowledgeWeightDimension,
) -> bool {
    match source {
        KnowledgeWeightSource::AdmittedEvidence => {
            dimension == KnowledgeWeightDimension::EpistemicSupport
        }
        KnowledgeWeightSource::Retrieval => dimension == KnowledgeWeightDimension::Accessibility,
        KnowledgeWeightSource::SimilarityMatch => {
            dimension == KnowledgeWeightDimension::Accessibility
        }
        KnowledgeWeightSource::DreamReplay => matches!(
            dimension,
            KnowledgeWeightDimension::Retention | KnowledgeWeightDimension::Consolidation
        ),
        KnowledgeWeightSource::CausalConsolidation => matches!(
            dimension,
            KnowledgeWeightDimension::Retention | KnowledgeWeightDimension::Consolidation
        ),
        KnowledgeWeightSource::MemoryDecay => dimension == KnowledgeWeightDimension::Retention,
        KnowledgeWeightSource::TaskRelevance => {
            dimension == KnowledgeWeightDimension::Accessibility
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum KnowledgeWeightError {
    WeightOutOfRange(f32),
    DeltaOutOfRange(f32),
    EmptyRationale,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::EvidenceId;

    fn proposal(
        dimension: KnowledgeWeightDimension,
        source: KnowledgeWeightSource,
        basis: Vec<EvidenceId>,
    ) -> KnowledgeWeightUpdateProposal {
        KnowledgeWeightUpdateProposal::new(
            dimension,
            source,
            0.1,
            basis,
            "characterization proposal",
        )
        .unwrap()
    }

    #[test]
    fn replay_cannot_propose_epistemic_support_change() {
        let proposal = proposal(
            KnowledgeWeightDimension::EpistemicSupport,
            KnowledgeWeightSource::DreamReplay,
            vec![EvidenceId(1)],
        );
        let decision = KnowledgeWeightAuthorityRouter::evaluate(&proposal);
        assert!(!decision.eligible());
        assert!(decision.failures().contains(
            &KnowledgeWeightRoutingFailure::SourceCannotUpdateDimension {
                source: KnowledgeWeightSource::DreamReplay,
                dimension: KnowledgeWeightDimension::EpistemicSupport,
            }
        ));
    }

    #[test]
    fn retrieval_only_routes_to_accessibility() {
        let allowed = proposal(
            KnowledgeWeightDimension::Accessibility,
            KnowledgeWeightSource::Retrieval,
            vec![],
        );
        assert!(KnowledgeWeightAuthorityRouter::evaluate(&allowed).eligible());

        let denied = proposal(
            KnowledgeWeightDimension::Retention,
            KnowledgeWeightSource::Retrieval,
            vec![],
        );
        assert!(!KnowledgeWeightAuthorityRouter::evaluate(&denied).eligible());
    }

    #[test]
    fn similarity_does_not_count_as_epistemic_corroboration() {
        let proposal = proposal(
            KnowledgeWeightDimension::EpistemicSupport,
            KnowledgeWeightSource::SimilarityMatch,
            vec![],
        );
        assert!(!KnowledgeWeightAuthorityRouter::evaluate(&proposal).eligible());
    }

    #[test]
    fn admitted_evidence_can_propose_epistemic_update_only_with_basis() {
        let with_basis = proposal(
            KnowledgeWeightDimension::EpistemicSupport,
            KnowledgeWeightSource::AdmittedEvidence,
            vec![EvidenceId(7)],
        );
        assert!(KnowledgeWeightAuthorityRouter::evaluate(&with_basis).eligible());

        let without_basis = proposal(
            KnowledgeWeightDimension::EpistemicSupport,
            KnowledgeWeightSource::AdmittedEvidence,
            vec![],
        );
        let decision = KnowledgeWeightAuthorityRouter::evaluate(&without_basis);
        assert!(!decision.eligible());
        assert!(decision
            .failures()
            .contains(&KnowledgeWeightRoutingFailure::EpistemicUpdateHasNoEvidenceBasis));
    }

    #[test]
    fn dream_and_causal_consolidation_route_to_memory_dimensions_only() {
        for source in [
            KnowledgeWeightSource::DreamReplay,
            KnowledgeWeightSource::CausalConsolidation,
        ] {
            for dimension in [
                KnowledgeWeightDimension::Retention,
                KnowledgeWeightDimension::Consolidation,
            ] {
                assert!(KnowledgeWeightAuthorityRouter::evaluate(&proposal(
                    dimension,
                    source,
                    vec![],
                ))
                .eligible());
            }
        }
    }

    #[test]
    fn unassessed_weight_is_distinct_from_zero() {
        let vector = KnowledgeWeightVector::default();
        assert_eq!(vector.epistemic_support, None);
        let zero = BoundedWeight::new(0.0).unwrap();
        assert_ne!(vector.epistemic_support, Some(zero));
        assert_eq!(vector.assessed_dimension_count(), 0);
    }

    #[test]
    fn invalid_weights_and_deltas_fail_closed() {
        assert!(BoundedWeight::new(-0.1).is_err());
        assert!(BoundedWeight::new(1.1).is_err());
        assert!(BoundedWeight::new(f32::NAN).is_err());
        assert!(SignedWeightDelta::new(-1.1).is_err());
        assert!(SignedWeightDelta::new(1.1).is_err());
        assert!(SignedWeightDelta::new(f32::INFINITY).is_err());
    }
}
