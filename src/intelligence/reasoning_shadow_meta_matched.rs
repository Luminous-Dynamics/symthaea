// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Compatibility-preserving matched IntegrationProxy shadow layer.
//!
//! The existing `reasoning_shadow_meta::ShadowQualifiedMetaReasoner` remains responsible for
//! historical behavior, canonical V2 shadowing, and the all-unknown V3 baseline. This wrapper adds
//! a second, measurement-only V3 pass using the matched counterfactual IntegrationProxy operator
//! panel. The two source classes remain explicit: the baseline says what V3 knows from live
//! identity and activation alone; the matched pass says what changes after bounded evidence from
//! the same frozen input and preregistered operator panel is introduced.

use super::reasoning_active_primitive_evidence::ActivePrimitiveEvidenceReport;
use super::reasoning_context_competition::{ContextCompetitionPolicy, ContextHypothesis};
use super::reasoning_evidence_seeking::{EvidenceSeekingOutcome, EvidenceSeekingPlanReport};
use super::reasoning_matched_integration_probe::{
    plan_active_primitive_with_matched_integration, MatchedIntegrationProbeReport,
};
use super::reasoning_shadow_meta::{
    ShadowMetaObservation, ShadowMetaStats, ShadowQualifiedMetaReasoner as BaselineShadowReasoner,
    V3ShadowObservation, V3ShadowOutcomeKind,
};
use crate::consciousness::meta_reasoning::{
    MetaCognitiveReasoner, MetaReasoningConfig, MetaReasoningResult,
};
use crate::consciousness::primitive_evolution::{CandidatePrimitive, EvolutionConfig};
use crate::consciousness::primitive_reasoning::ReasoningChain;
use crate::consciousness::ActivePrimitive;
use crate::hdc::BinaryHV;
use anyhow::Result;
use std::collections::HashSet;

pub const MATCHED_META_SHADOW_VERSION: &str = "rq-006y-live-matched-shadow-v2";
const MATCHED_CONTEXT_SOURCE: &str = "legacy-meta-context-reflection-matched-shadow-v2";
const MATCHED_EVIDENCE_REF: &str = "live-pre-meta-input";

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MatchedShadowStats {
    pub attempts: u64,
    pub successes: u64,
    pub errors: u64,
    pub outcome_changes: u64,
    pub top_request_changes: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MatchedV3ShadowObservation {
    pub shadow_version: String,
    pub attempt: u64,
    pub baseline_outcome: Option<V3ShadowOutcomeKind>,
    pub matched_outcome: V3ShadowOutcomeKind,
    pub baseline_request_count: Option<usize>,
    pub matched_request_count: usize,
    pub baseline_top_request_id: Option<String>,
    pub matched_top_request_id: Option<String>,
    pub baseline_selected_candidate: Option<String>,
    pub matched_selected_candidate: Option<String>,
    pub active_profiles: usize,
    pub matched_probe_profiles: usize,
    pub matched_probe_input_digest: Option<String>,
    /// Global minimum lower bound across candidate operator envelopes.
    pub integration_min: Option<f64>,
    /// Global maximum upper bound across candidate operator envelopes.
    pub integration_max: Option<f64>,
    /// Overall span across the bounded matched evidence. This is not a confidence interval.
    pub integration_spread: Option<f64>,
    pub evidence_changed_outcome: Option<bool>,
    pub evidence_changed_top_request: Option<bool>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct OutcomeSummary {
    kind: V3ShadowOutcomeKind,
    request_count: usize,
    top_request_id: Option<String>,
    selected_candidate: Option<String>,
}

/// Public compatibility wrapper. Historical behavior still comes entirely from `inner`.
pub struct MatchedShadowQualifiedMetaReasoner {
    inner: BaselineShadowReasoner,
    matched_stats: MatchedShadowStats,
    last_matched_v3_observation: Option<MatchedV3ShadowObservation>,
}

impl MatchedShadowQualifiedMetaReasoner {
    pub fn new(
        evolution_config: EvolutionConfig,
        meta_config: MetaReasoningConfig,
    ) -> Result<Self> {
        Ok(Self {
            inner: BaselineShadowReasoner::new(evolution_config, meta_config)?,
            matched_stats: MatchedShadowStats::default(),
            last_matched_v3_observation: None,
        })
    }

    /// Compatibility path: exactly delegates historical + V2 behavior to the existing wrapper.
    pub fn meta_reason(
        &mut self,
        query: &str,
        primitives: Vec<CandidatePrimitive>,
        chain: &mut ReasoningChain,
    ) -> Result<MetaReasoningResult> {
        self.inner.meta_reason(query, primitives, chain)
    }

    /// Run the established historical/V2/V3-baseline path first. Then, and only then, perform the
    /// matched counterfactual IntegrationProxy operator panel against the frozen pre-meta input.
    /// Failure of the second pass can only affect matched-shadow telemetry; the historical result
    /// is returned unchanged.
    pub fn meta_reason_with_active_evidence(
        &mut self,
        query: &str,
        primitives: Vec<CandidatePrimitive>,
        active_primitives: &[ActivePrimitive],
        chain: &mut ReasoningChain,
    ) -> Result<MetaReasoningResult> {
        self.meta_reason_with_active_evidence_using_probe(
            query,
            primitives,
            active_primitives,
            chain,
            |hypotheses, policy, active, candidate_ids, frozen_input| {
                plan_active_primitive_with_matched_integration(
                    hypotheses,
                    policy,
                    active,
                    candidate_ids,
                    frozen_input,
                )
                .map_err(|err| err.to_string())
            },
        )
    }

    fn meta_reason_with_active_evidence_using_probe<F>(
        &mut self,
        query: &str,
        primitives: Vec<CandidatePrimitive>,
        active_primitives: &[ActivePrimitive],
        chain: &mut ReasoningChain,
        probe: F,
    ) -> Result<MetaReasoningResult>
    where
        F: FnOnce(
            &[ContextHypothesis],
            ContextCompetitionPolicy,
            &[ActivePrimitive],
            &[String],
            BinaryHV,
        ) -> std::result::Result<
            (
                ActivePrimitiveEvidenceReport,
                MatchedIntegrationProbeReport,
                EvidenceSeekingPlanReport,
            ),
            String,
        >,
    {
        let frozen_input = chain.question;
        let candidate_ids = primitives
            .iter()
            .map(|primitive| primitive.name.clone())
            .collect::<Vec<_>>();

        // This call remains behavior-authoritative. Its own V3 observation is the all-unknown
        // baseline against which the bounded matched probe is compared.
        let legacy_result = self.inner.meta_reason_with_active_evidence(
            query,
            primitives,
            active_primitives,
            chain,
        )?;
        let baseline = self.inner.last_v3_shadow_observation().cloned();

        let attempt = self.matched_stats.attempts;
        self.matched_stats.attempts = self.matched_stats.attempts.saturating_add(1);
        let hypotheses = context_hypotheses(&legacy_result);

        match probe(
            &hypotheses,
            ContextCompetitionPolicy::development_v1(),
            active_primitives,
            &candidate_ids,
            frozen_input,
        ) {
            Ok((active_report, probe_report, plan)) => {
                self.matched_stats.successes = self.matched_stats.successes.saturating_add(1);
                let matched = summarize_outcome(&plan.outcome);
                let baseline_summary = baseline.as_ref().and_then(summarize_baseline);
                let changed_outcome = baseline_summary
                    .as_ref()
                    .map(|summary| summary.kind != matched.kind);
                let changed_top_request = baseline_summary
                    .as_ref()
                    .map(|summary| summary.top_request_id != matched.top_request_id);

                if changed_outcome == Some(true) {
                    self.matched_stats.outcome_changes =
                        self.matched_stats.outcome_changes.saturating_add(1);
                }
                if changed_top_request == Some(true) {
                    self.matched_stats.top_request_changes =
                        self.matched_stats.top_request_changes.saturating_add(1);
                }

                tracing::debug!(
                    target: "symthaea::reasoning_shadow_v3_matched",
                    attempt,
                    matched_outcome = ?matched.kind,
                    baseline_outcome = ?baseline_summary.as_ref().map(|summary| summary.kind),
                    active_profiles = active_report.profiles.len(),
                    probe_profiles = probe_report.profiles.len(),
                    probe_input = %probe_report.input_digest,
                    integration_min = probe_report.integration_min,
                    integration_max = probe_report.integration_max,
                    integration_spread = probe_report.integration_spread,
                    changed_outcome = changed_outcome.unwrap_or(false),
                    changed_top_request = changed_top_request.unwrap_or(false),
                    "matched bounded IntegrationProxy shadow observed"
                );

                self.last_matched_v3_observation = Some(MatchedV3ShadowObservation {
                    shadow_version: MATCHED_META_SHADOW_VERSION.into(),
                    attempt,
                    baseline_outcome: baseline_summary.as_ref().map(|summary| summary.kind),
                    matched_outcome: matched.kind,
                    baseline_request_count: baseline_summary
                        .as_ref()
                        .map(|summary| summary.request_count),
                    matched_request_count: matched.request_count,
                    baseline_top_request_id: baseline_summary
                        .as_ref()
                        .and_then(|summary| summary.top_request_id.clone()),
                    matched_top_request_id: matched.top_request_id,
                    baseline_selected_candidate: baseline_summary
                        .as_ref()
                        .and_then(|summary| summary.selected_candidate.clone()),
                    matched_selected_candidate: matched.selected_candidate,
                    active_profiles: active_report.profiles.len(),
                    matched_probe_profiles: probe_report.profiles.len(),
                    matched_probe_input_digest: Some(probe_report.input_digest),
                    integration_min: Some(probe_report.integration_min),
                    integration_max: Some(probe_report.integration_max),
                    integration_spread: Some(probe_report.integration_spread),
                    evidence_changed_outcome: changed_outcome,
                    evidence_changed_top_request: changed_top_request,
                    error: None,
                });
            }
            Err(err) => {
                self.matched_stats.errors = self.matched_stats.errors.saturating_add(1);
                let baseline_summary = baseline.as_ref().and_then(summarize_baseline);
                tracing::debug!(
                    target: "symthaea::reasoning_shadow_v3_matched",
                    attempt,
                    error = %err,
                    "matched IntegrationProxy shadow failed without affecting historical behavior"
                );
                self.last_matched_v3_observation = Some(MatchedV3ShadowObservation {
                    shadow_version: MATCHED_META_SHADOW_VERSION.into(),
                    attempt,
                    baseline_outcome: baseline_summary.as_ref().map(|summary| summary.kind),
                    matched_outcome: V3ShadowOutcomeKind::Error,
                    baseline_request_count: baseline_summary
                        .as_ref()
                        .map(|summary| summary.request_count),
                    matched_request_count: 0,
                    baseline_top_request_id: baseline_summary
                        .as_ref()
                        .and_then(|summary| summary.top_request_id.clone()),
                    matched_top_request_id: None,
                    baseline_selected_candidate: baseline_summary
                        .as_ref()
                        .and_then(|summary| summary.selected_candidate.clone()),
                    matched_selected_candidate: None,
                    active_profiles: 0,
                    matched_probe_profiles: 0,
                    matched_probe_input_digest: None,
                    integration_min: None,
                    integration_max: None,
                    integration_spread: None,
                    evidence_changed_outcome: None,
                    evidence_changed_top_request: None,
                    error: Some(err),
                });
            }
        }

        Ok(legacy_result)
    }

    /// Existing accessors retain their original semantics and types.
    pub fn legacy(&self) -> &MetaCognitiveReasoner {
        self.inner.legacy()
    }

    pub fn shadow_stats(&self) -> &ShadowMetaStats {
        self.inner.shadow_stats()
    }

    pub fn last_shadow_observation(&self) -> Option<&ShadowMetaObservation> {
        self.inner.last_shadow_observation()
    }

    pub fn last_v3_shadow_observation(&self) -> Option<&V3ShadowObservation> {
        self.inner.last_v3_shadow_observation()
    }

    pub fn canonical_next_sequence(&self) -> u64 {
        self.inner.canonical_next_sequence()
    }

    /// New matched-evidence telemetry is explicitly separate from baseline V3 telemetry.
    pub fn matched_shadow_stats(&self) -> &MatchedShadowStats {
        &self.matched_stats
    }

    pub fn last_matched_v3_shadow_observation(&self) -> Option<&MatchedV3ShadowObservation> {
        self.last_matched_v3_observation.as_ref()
    }
}

fn summarize_baseline(observation: &V3ShadowObservation) -> Option<OutcomeSummary> {
    if observation.outcome == V3ShadowOutcomeKind::Error {
        return None;
    }
    Some(OutcomeSummary {
        kind: observation.outcome,
        request_count: observation.request_count,
        top_request_id: observation.top_request_id.clone(),
        selected_candidate: observation.selected_candidate.clone(),
    })
}

fn summarize_outcome(outcome: &EvidenceSeekingOutcome) -> OutcomeSummary {
    match outcome {
        EvidenceSeekingOutcome::Selected { candidate_id, .. } => OutcomeSummary {
            kind: V3ShadowOutcomeKind::Selected,
            request_count: 0,
            top_request_id: None,
            selected_candidate: Some(candidate_id.clone()),
        },
        EvidenceSeekingOutcome::NeedEvidence { requests, .. } => OutcomeSummary {
            kind: V3ShadowOutcomeKind::NeedEvidence,
            request_count: requests.len(),
            top_request_id: requests.first().map(|request| request.request_id.clone()),
            selected_candidate: None,
        },
        EvidenceSeekingOutcome::Abstained { .. } => OutcomeSummary {
            kind: V3ShadowOutcomeKind::Abstained,
            request_count: 0,
            top_request_id: None,
            selected_candidate: None,
        },
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
        source: MATCHED_CONTEXT_SOURCE.into(),
        evidence_refs: vec![MATCHED_EVIDENCE_REF.into()],
    });

    for (context, support) in &reflection.alternative_contexts {
        if seen.insert(*context) {
            hypotheses.push(ContextHypothesis {
                context: *context,
                support: *support,
                source: MATCHED_CONTEXT_SOURCE.into(),
                evidence_refs: vec![MATCHED_EVIDENCE_REF.into()],
            });
        }
    }
    hypotheses
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consciousness::epistemic_tiers::EpistemicCoordinate;
    use crate::consciousness::ActivationReason;
    use symthaea_core::hdc::primitive_system::{PrimitiveSystem, PrimitiveTier};

    fn candidate(name: &str) -> CandidatePrimitive {
        CandidatePrimitive {
            name: name.into(),
            tier: PrimitiveTier::Physical,
            definition: format!("fixture-{name}"),
            fitness: 0.5,
            encoding: BinaryHV::random(name.len() as u64 + 2100),
            epistemic_coordinate: EpistemicCoordinate::axiom(),
            harmonic_alignment: 0.6,
        }
    }

    fn active(name: &str, activation: f64) -> ActivePrimitive {
        ActivePrimitive {
            primitive: PrimitiveSystem::global()
                .get(name)
                .unwrap_or_else(|| panic!("fixture primitive `{name}` must exist"))
                .clone(),
            activation,
            activation_reason: ActivationReason::BottomUp {
                input_similarity: activation,
            },
            duration: 2,
        }
    }

    fn reasoner() -> MatchedShadowQualifiedMetaReasoner {
        MatchedShadowQualifiedMetaReasoner::new(
            EvolutionConfig::default(),
            MetaReasoningConfig::default(),
        )
        .unwrap()
    }

    #[test]
    fn matched_shadow_preserves_baseline_accessors_and_adds_probe_telemetry() {
        let mut reasoner = reasoner();
        let input = BinaryHV::random(2200);
        let mut chain = ReasoningChain::new(input);
        let actives = [active("NSM_KNOW", 0.8), active("NSM_DO", 0.6)];
        let result = reasoner.meta_reason_with_active_evidence(
            "evidence experiment research theory scientific",
            vec![candidate("NSM_KNOW"), candidate("NSM_DO")],
            &actives,
            &mut chain,
        );
        assert!(result.is_ok());
        assert_eq!(chain.question, input);
        assert_eq!(reasoner.shadow_stats().v3_attempts, 1);
        assert!(reasoner.last_v3_shadow_observation().is_some());
        assert_eq!(reasoner.matched_shadow_stats().attempts, 1);
        assert_eq!(reasoner.matched_shadow_stats().successes, 1);

        let observation = reasoner.last_matched_v3_shadow_observation().unwrap();
        assert_eq!(
            observation.baseline_outcome,
            Some(V3ShadowOutcomeKind::NeedEvidence)
        );
        assert_eq!(observation.baseline_request_count, Some(6));
        assert_eq!(observation.matched_probe_profiles, 2);
        assert_eq!(observation.active_profiles, 2);
        // Bounded IntegrationProxy evidence can itself require refinement, so the matched pass may
        // still contain up to the original six requests. The important theorem is that it never
        // fabricates completeness by discarding operator uncertainty.
        assert!(observation.matched_request_count <= 6);
        assert!(observation.matched_probe_input_digest.is_some());
        assert!(observation.integration_min.is_some());
        assert!(observation.integration_max.is_some());
        assert!(observation.integration_spread.is_some());
        assert!(observation.evidence_changed_top_request.is_some());
    }

    #[test]
    fn injected_matched_probe_failure_cannot_reject_legacy_result() {
        let mut reasoner = reasoner();
        let input = BinaryHV::random(2201);
        let mut chain = ReasoningChain::new(input);
        let actives = [active("NSM_KNOW", 0.8), active("NSM_DO", 0.6)];
        let result = reasoner.meta_reason_with_active_evidence_using_probe(
            "evidence experiment research theory scientific",
            vec![candidate("NSM_KNOW"), candidate("NSM_DO")],
            &actives,
            &mut chain,
            |_hypotheses, _policy, _active, _candidate_ids, _frozen_input| {
                Err("injected matched-probe failure".into())
            },
        );
        assert!(result.is_ok());
        assert_eq!(chain.question, input);
        assert_eq!(reasoner.shadow_stats().v3_attempts, 1);
        assert_eq!(reasoner.matched_shadow_stats().attempts, 1);
        assert_eq!(reasoner.matched_shadow_stats().errors, 1);
        let observation = reasoner.last_matched_v3_shadow_observation().unwrap();
        assert_eq!(observation.matched_outcome, V3ShadowOutcomeKind::Error);
        assert_eq!(
            observation.baseline_outcome,
            Some(V3ShadowOutcomeKind::NeedEvidence)
        );
        assert!(observation
            .error
            .as_deref()
            .unwrap_or_default()
            .contains("injected matched-probe failure"));
    }

    #[test]
    fn matched_probe_input_is_bound_to_pre_meta_question() {
        let actives = [active("NSM_KNOW", 0.8), active("NSM_DO", 0.6)];

        let mut first = reasoner();
        let mut first_chain = ReasoningChain::new(BinaryHV::random(2202));
        first
            .meta_reason_with_active_evidence(
                "evidence experiment research theory scientific",
                vec![candidate("NSM_KNOW"), candidate("NSM_DO")],
                &actives,
                &mut first_chain,
            )
            .unwrap();
        let first_digest = first
            .last_matched_v3_shadow_observation()
            .and_then(|observation| observation.matched_probe_input_digest.clone())
            .unwrap();

        let mut second = reasoner();
        let mut second_chain = ReasoningChain::new(BinaryHV::random(2203));
        second
            .meta_reason_with_active_evidence(
                "evidence experiment research theory scientific",
                vec![candidate("NSM_KNOW"), candidate("NSM_DO")],
                &actives,
                &mut second_chain,
            )
            .unwrap();
        let second_digest = second
            .last_matched_v3_shadow_observation()
            .and_then(|observation| observation.matched_probe_input_digest.clone())
            .unwrap();

        assert_ne!(first_digest, second_digest);
    }
}
