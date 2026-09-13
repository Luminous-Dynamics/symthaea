// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Measurement-only shadow integration for the existing live meta-reasoner.
//!
//! The historical `MetaCognitiveReasoner` remains behavior-authoritative in this wrapper. Every
//! successful historical call is mirrored into the canonical V2 reasoning kernel using the same
//! production-shaped candidate set. Canonical success/failure and selection agreement are recorded
//! only as telemetry and never change the returned historical result or learning/plasticity.

use super::reasoning_context_competition::{ContextHypothesis, ContextResolution};
use super::reasoning_kernel_v2::{CanonicalReasoningInputV2, CanonicalReasoningKernelV2};
use crate::consciousness::meta_reasoning::{
    MetaCognitiveReasoner, MetaReasoningConfig, MetaReasoningResult,
};
use crate::consciousness::primitive_evolution::{CandidatePrimitive, EvolutionConfig};
use crate::consciousness::primitive_reasoning::ReasoningChain;
use anyhow::Result;
use std::collections::HashSet;

pub const LIVE_META_SHADOW_VERSION: &str = "rq-006-live-meta-shadow-v1";
const SHADOW_SUBJECT_ID: &str = "cognitive-loop-meta-reasoner";
const SHADOW_EVIDENCE_REF: &str = "live-input";
const SHADOW_CONTEXT_SOURCE: &str = "legacy-meta-context-reflection-shadow-v1";
const SHADOW_HISTORY_CAPACITY: usize = 100;
const OBJECTIVE_SPREAD_EPSILON: f64 = 1.0e-12;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ShadowMetaStats {
    pub attempts: u64,
    pub committed: u64,
    pub rejected: u64,
    pub resolved_contexts: u64,
    pub ambiguous_contexts: u64,
    pub informative_commits: u64,
    pub uninformative_commits: u64,
    pub selection_agreements: u64,
    pub selection_disagreements: u64,
    pub informative_agreements: u64,
    pub informative_disagreements: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ObjectiveSpread {
    pub integration_proxy: f64,
    pub harmonic_alignment: f64,
    pub epistemic_grounding: f64,
}

impl ObjectiveSpread {
    pub fn informative(self) -> bool {
        self.integration_proxy > OBJECTIVE_SPREAD_EPSILON
            || self.harmonic_alignment > OBJECTIVE_SPREAD_EPSILON
            || self.epistemic_grounding > OBJECTIVE_SPREAD_EPSILON
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ShadowMetaObservation {
    pub shadow_version: String,
    pub attempt: u64,
    pub canonical_sequence: u64,
    pub committed: bool,
    pub primary_context: Option<crate::consciousness::context_aware_evolution::ReasoningContext>,
    pub active_contexts: usize,
    pub legacy_selected_candidate: String,
    pub canonical_selected_candidate: Option<String>,
    pub selection_agreement: Option<bool>,
    /// `true` only when at least one canonical objective dimension actually varies across the
    /// admitted candidate set. Agreement/disagreement on an all-identical set is tie behavior,
    /// not evidence that one selector made a better substantive choice.
    pub selection_informative: Option<bool>,
    pub objective_spread: Option<ObjectiveSpread>,
    pub legacy_meta_confidence: f64,
    pub legacy_context_confidence: f64,
    pub decision_commitment: Option<String>,
    pub rejection: Option<String>,
}

/// Drop-in wrapper used by the cognitive loop while the canonical path remains measurement-only.
pub struct ShadowQualifiedMetaReasoner {
    legacy: MetaCognitiveReasoner,
    shadow: CanonicalReasoningKernelV2,
    stats: ShadowMetaStats,
    last_observation: Option<ShadowMetaObservation>,
}

impl ShadowQualifiedMetaReasoner {
    pub fn new(evolution_config: EvolutionConfig, meta_config: MetaReasoningConfig) -> Result<Self> {
        Ok(Self {
            legacy: MetaCognitiveReasoner::new(evolution_config, meta_config)?,
            shadow: CanonicalReasoningKernelV2::new(SHADOW_HISTORY_CAPACITY)
                .map_err(|err| anyhow::anyhow!("canonical meta shadow init failed: {err}"))?,
            stats: ShadowMetaStats::default(),
            last_observation: None,
        })
    }

    /// Preserve the historical result exactly while mirroring the same call into canonical V2.
    pub fn meta_reason(
        &mut self,
        query: &str,
        primitives: Vec<CandidatePrimitive>,
        chain: &mut ReasoningChain,
    ) -> Result<MetaReasoningResult> {
        let shadow_candidates = primitives.clone();
        let legacy_result = self.legacy.meta_reason(query, primitives, chain)?;
        self.observe_shadow(&legacy_result, shadow_candidates);
        Ok(legacy_result)
    }

    /// Compatibility view for existing typed accessors.
    pub fn legacy(&self) -> &MetaCognitiveReasoner {
        &self.legacy
    }

    pub fn shadow_stats(&self) -> &ShadowMetaStats {
        &self.stats
    }

    pub fn last_shadow_observation(&self) -> Option<&ShadowMetaObservation> {
        self.last_observation.as_ref()
    }

    pub fn canonical_next_sequence(&self) -> u64 {
        self.shadow
            .subject_state(SHADOW_SUBJECT_ID)
            .map(|state| state.next_sequence())
            .unwrap_or(0)
    }

    fn observe_shadow(
        &mut self,
        legacy_result: &MetaReasoningResult,
        candidates: Vec<CandidatePrimitive>,
    ) {
        let attempt = self.stats.attempts;
        self.stats.attempts = self.stats.attempts.saturating_add(1);
        let sequence = self.canonical_next_sequence();
        let legacy_selected_candidate = legacy_result.optimization_result.primitive.name.clone();
        let legacy_meta_confidence = legacy_result.meta_confidence;
        let legacy_context_confidence = legacy_result.context_reflection.confidence;

        let hypotheses = context_hypotheses(legacy_result);
        let strategy_support = legacy_result
            .optimization_result
            .tradeoff_point
            .weighted_fitness(&legacy_result.optimization_result.weights);
        let input = CanonicalReasoningInputV2 {
            subject_id: SHADOW_SUBJECT_ID.into(),
            episode_id: format!("live-meta-shadow-attempt-{attempt}"),
            sequence,
            context_hypotheses: hypotheses,
            strategy_support,
            abstained: false,
            evidence_items: 1,
            weak_assumptions_flagged: 0,
            candidates,
        };

        match self.shadow.reason(input) {
            Ok(decision) => {
                self.stats.committed = self.stats.committed.saturating_add(1);
                match decision.context_selection.assessment.resolution {
                    ContextResolution::Resolved => {
                        self.stats.resolved_contexts = self.stats.resolved_contexts.saturating_add(1)
                    }
                    ContextResolution::Ambiguous => {
                        self.stats.ambiguous_contexts =
                            self.stats.ambiguous_contexts.saturating_add(1)
                    }
                }

                let spread = objective_spread(&decision.context_selection.evaluations);
                let selection_informative = spread.informative();
                if selection_informative {
                    self.stats.informative_commits = self.stats.informative_commits.saturating_add(1);
                } else {
                    self.stats.uninformative_commits =
                        self.stats.uninformative_commits.saturating_add(1);
                }

                let canonical_selected_candidate = decision.selected_candidate.name.clone();
                let selection_agreement = canonical_selected_candidate == legacy_selected_candidate;
                if selection_agreement {
                    self.stats.selection_agreements =
                        self.stats.selection_agreements.saturating_add(1);
                    if selection_informative {
                        self.stats.informative_agreements =
                            self.stats.informative_agreements.saturating_add(1);
                    }
                } else {
                    self.stats.selection_disagreements =
                        self.stats.selection_disagreements.saturating_add(1);
                    if selection_informative {
                        self.stats.informative_disagreements =
                            self.stats.informative_disagreements.saturating_add(1);
                    }
                }

                tracing::debug!(
                    target: "symthaea::reasoning_shadow",
                    attempt,
                    canonical_sequence = decision.sequence,
                    active_contexts = decision.context_selection.assessment.active_contexts.len(),
                    selection_informative,
                    selection_agreement,
                    legacy_selected = %legacy_selected_candidate,
                    canonical_selected = %canonical_selected_candidate,
                    committed = self.stats.committed,
                    rejected = self.stats.rejected,
                    informative_agreements = self.stats.informative_agreements,
                    informative_disagreements = self.stats.informative_disagreements,
                    "canonical V2 live meta shadow committed"
                );

                self.last_observation = Some(ShadowMetaObservation {
                    shadow_version: LIVE_META_SHADOW_VERSION.into(),
                    attempt,
                    canonical_sequence: decision.sequence,
                    committed: true,
                    primary_context: Some(decision.context_selection.assessment.primary_context),
                    active_contexts: decision.context_selection.assessment.active_contexts.len(),
                    legacy_selected_candidate,
                    canonical_selected_candidate: Some(canonical_selected_candidate),
                    selection_agreement: Some(selection_agreement),
                    selection_informative: Some(selection_informative),
                    objective_spread: Some(spread),
                    legacy_meta_confidence,
                    legacy_context_confidence,
                    decision_commitment: Some(decision.decision_commitment.clone()),
                    rejection: None,
                });
            }
            Err(err) => {
                self.stats.rejected = self.stats.rejected.saturating_add(1);
                tracing::debug!(
                    target: "symthaea::reasoning_shadow",
                    attempt,
                    canonical_sequence = sequence,
                    legacy_selected = %legacy_selected_candidate,
                    legacy_meta_confidence,
                    legacy_context_confidence,
                    committed = self.stats.committed,
                    rejected = self.stats.rejected,
                    error = %err,
                    "canonical V2 live meta shadow rejected measurement"
                );
                self.last_observation = Some(ShadowMetaObservation {
                    shadow_version: LIVE_META_SHADOW_VERSION.into(),
                    attempt,
                    canonical_sequence: sequence,
                    committed: false,
                    primary_context: None,
                    active_contexts: 0,
                    legacy_selected_candidate,
                    canonical_selected_candidate: None,
                    selection_agreement: None,
                    selection_informative: None,
                    objective_spread: None,
                    legacy_meta_confidence,
                    legacy_context_confidence,
                    decision_commitment: None,
                    rejection: Some(err.to_string()),
                });
            }
        }
    }
}

fn objective_spread(
    evaluations: &[super::reasoning_context_competition::RobustCandidateEvaluation],
) -> ObjectiveSpread {
    if evaluations.is_empty() {
        return ObjectiveSpread {
            integration_proxy: 0.0,
            harmonic_alignment: 0.0,
            epistemic_grounding: 0.0,
        };
    }

    let first = evaluations[0].vector;
    let mut min_integration = first.integration_proxy;
    let mut max_integration = first.integration_proxy;
    let mut min_harmonic = first.harmonic_alignment;
    let mut max_harmonic = first.harmonic_alignment;
    let mut min_epistemic = first.epistemic_grounding;
    let mut max_epistemic = first.epistemic_grounding;

    for evaluation in &evaluations[1..] {
        let vector = evaluation.vector;
        min_integration = min_integration.min(vector.integration_proxy);
        max_integration = max_integration.max(vector.integration_proxy);
        min_harmonic = min_harmonic.min(vector.harmonic_alignment);
        max_harmonic = max_harmonic.max(vector.harmonic_alignment);
        min_epistemic = min_epistemic.min(vector.epistemic_grounding);
        max_epistemic = max_epistemic.max(vector.epistemic_grounding);
    }

    ObjectiveSpread {
        integration_proxy: max_integration - min_integration,
        harmonic_alignment: max_harmonic - min_harmonic,
        epistemic_grounding: max_epistemic - min_epistemic,
    }
}

fn context_hypotheses(result: &MetaReasoningResult) -> Vec<ContextHypothesis> {
    let reflection = &result.context_reflection;
    let mut seen = HashSet::new();
    let mut hypotheses = Vec::new();

    seen.insert(reflection.detected_context);
    hypotheses.push(ContextHypothesis {
        context: reflection.detected_context,
        support: reflection.confidence,
        source: SHADOW_CONTEXT_SOURCE.into(),
        evidence_refs: vec![SHADOW_EVIDENCE_REF.into()],
    });

    for (context, support) in &reflection.alternative_contexts {
        if seen.insert(*context) {
            hypotheses.push(ContextHypothesis {
                context: *context,
                support: *support,
                source: SHADOW_CONTEXT_SOURCE.into(),
                evidence_refs: vec![SHADOW_EVIDENCE_REF.into()],
            });
        }
    }
    hypotheses
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consciousness::epistemic_tiers::EpistemicCoordinate;
    use crate::hdc::BinaryHV;
    use symthaea_core::hdc::primitive_system::PrimitiveTier;

    fn candidate(name: &str) -> CandidatePrimitive {
        candidate_with_scores(name, 0.5, 0.6, EpistemicCoordinate::axiom())
    }

    fn candidate_with_scores(
        name: &str,
        fitness: f64,
        harmonic_alignment: f64,
        epistemic_coordinate: EpistemicCoordinate,
    ) -> CandidatePrimitive {
        CandidatePrimitive {
            name: name.into(),
            tier: PrimitiveTier::Physical,
            definition: format!("fixture-{name}"),
            fitness,
            encoding: BinaryHV::random(name.len() as u64 + 1400),
            epistemic_coordinate,
            harmonic_alignment,
        }
    }

    fn reasoner() -> ShadowQualifiedMetaReasoner {
        ShadowQualifiedMetaReasoner::new(
            EvolutionConfig::default(),
            MetaReasoningConfig::default(),
        )
        .unwrap()
    }

    #[test]
    fn weak_context_shadow_rejection_does_not_reject_legacy_result() {
        let mut reasoner = reasoner();
        let mut chain = ReasoningChain::new(BinaryHV::random(1500));
        let result = reasoner.meta_reason(
            "ordinary problem with no strong context marker",
            vec![candidate("a")],
            &mut chain,
        );
        assert!(result.is_ok());
        assert_eq!(reasoner.shadow_stats().attempts, 1);
        assert_eq!(reasoner.shadow_stats().rejected, 1);
        assert_eq!(reasoner.canonical_next_sequence(), 0);
        let observation = reasoner.last_shadow_observation().unwrap();
        assert!(!observation.committed);
        assert!(observation.selection_agreement.is_none());
        assert!(observation.selection_informative.is_none());
    }

    #[test]
    fn identical_objective_vectors_are_classified_uninformative() {
        let mut reasoner = reasoner();
        let mut chain = ReasoningChain::new(BinaryHV::random(1501));
        reasoner
            .meta_reason(
                "evidence experiment research theory scientific",
                vec![candidate("a"), candidate("b")],
                &mut chain,
            )
            .unwrap();
        let observation = reasoner.last_shadow_observation().unwrap();
        assert_eq!(observation.selection_informative, Some(false));
        assert_eq!(reasoner.shadow_stats().uninformative_commits, 1);
        assert_eq!(reasoner.shadow_stats().informative_commits, 0);
    }

    #[test]
    fn objective_spread_marks_substantive_comparison() {
        let mut reasoner = reasoner();
        let mut chain = ReasoningChain::new(BinaryHV::random(1502));
        reasoner
            .meta_reason(
                "evidence experiment research theory scientific",
                vec![
                    candidate_with_scores("weak", 0.5, 0.2, EpistemicCoordinate::null()),
                    candidate_with_scores("strong", 0.5, 0.8, EpistemicCoordinate::axiom()),
                ],
                &mut chain,
            )
            .unwrap();
        let observation = reasoner.last_shadow_observation().unwrap();
        assert_eq!(observation.selection_informative, Some(true));
        let spread = observation.objective_spread.unwrap();
        assert!(spread.harmonic_alignment > 0.0);
        assert!(spread.epistemic_grounding > 0.0);
        assert_eq!(reasoner.shadow_stats().informative_commits, 1);
        assert_eq!(
            reasoner.shadow_stats().informative_agreements
                + reasoner.shadow_stats().informative_disagreements,
            1
        );
    }

    #[test]
    fn strong_scientific_context_commits_shadow_longitudinally() {
        let mut reasoner = reasoner();
        let mut chain = ReasoningChain::new(BinaryHV::random(1503));
        let first = reasoner.meta_reason(
            "evidence experiment research theory scientific",
            vec![candidate("a"), candidate("b")],
            &mut chain,
        );
        assert!(first.is_ok());
        assert_eq!(reasoner.shadow_stats().committed, 1);
        assert_eq!(reasoner.canonical_next_sequence(), 1);
        let observation = reasoner.last_shadow_observation().unwrap();
        assert!(observation.committed);
        assert!(observation.decision_commitment.is_some());
        assert!(observation.canonical_selected_candidate.is_some());
        assert!(observation.selection_agreement.is_some());

        let mut second_chain = ReasoningChain::new(BinaryHV::random(1504));
        let second = reasoner.meta_reason(
            "evidence experiment research theory scientific",
            vec![candidate("a"), candidate("b")],
            &mut second_chain,
        );
        assert!(second.is_ok());
        assert_eq!(reasoner.shadow_stats().committed, 2);
        assert_eq!(reasoner.canonical_next_sequence(), 2);
        assert_eq!(
            reasoner.shadow_stats().selection_agreements
                + reasoner.shadow_stats().selection_disagreements,
            2
        );
    }

    #[test]
    fn mixed_strong_safety_scientific_context_can_remain_ambiguous() {
        let mut reasoner = reasoner();
        let mut chain = ReasoningChain::new(BinaryHV::random(1505));
        let result = reasoner.meta_reason(
            "safety harm dangerous evidence experiment research theory",
            vec![candidate("a"), candidate("b")],
            &mut chain,
        );
        assert!(result.is_ok());
        let observation = reasoner.last_shadow_observation().unwrap();
        assert!(observation.committed);
        assert!(observation.active_contexts >= 2);
        assert_eq!(reasoner.shadow_stats().ambiguous_contexts, 1);
    }
}
