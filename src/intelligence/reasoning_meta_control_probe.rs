// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! RQ-006G development probe for meta-reasoning control flow.
//!
//! This probe records externally inspectable state transitions only. It does not persist hidden
//! chain-of-thought. Its purpose is to freeze the current subject behavior before repairing the
//! dominant-objective identity mismatch and current-episode confidence/state-commit defects.

use crate::consciousness::epistemic_tiers::EpistemicCoordinate;
use crate::consciousness::meta_reasoning::{
    MetaCognitiveReasoner, MetaReasoningConfig,
};
use crate::consciousness::primitive_evolution::{CandidatePrimitive, EvolutionConfig};
use crate::consciousness::primitive_reasoning::ReasoningChain;
use crate::hdc::BinaryHV;
use crate::hdc::primitive_system::PrimitiveTier;
use serde::{Deserialize, Serialize};

pub const META_CONTROL_PROBE_VERSION: &str = "rq-006g-meta-control-v1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetaControlObservation {
    pub episode: usize,
    pub query_class: String,
    pub returned_context_confidence: f64,
    pub returned_meta_confidence: f64,
    pub strategy_adjustment_requested: bool,
    pub state_context_confidence_after: f64,
    pub current_context_committed: bool,
    pub decision_history_len_after: usize,
    pub meta_insights_len: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetaControlProbeReport {
    pub probe_version: String,
    pub persistent_general_episodes: Vec<MetaControlObservation>,
    pub context_switch_episode: MetaControlObservation,
    pub fresh_general_episodes: Vec<MetaControlObservation>,
}

/// Run a public development probe over the current default `MetaCognitiveReasoner` control flow.
///
/// The repeated persistent condition asks whether state and meta-learning can accumulate when the
/// reasoner object itself survives. The fresh condition models call sites that reconstruct the
/// reasoner every episode. A final literal safety query has high *current* context confidence so a
/// mismatch between returned reflection and committed state is directly observable.
pub fn run_meta_control_probe() -> anyhow::Result<MetaControlProbeReport> {
    let mut persistent = MetaCognitiveReasoner::new(
        EvolutionConfig::default(),
        MetaReasoningConfig::default(),
    )?;

    let mut persistent_general_episodes = Vec::new();
    for episode in 0..5 {
        persistent_general_episodes.push(run_one(
            &mut persistent,
            episode,
            "general-low-keyword",
            "Consider this situation carefully.",
        )?);
    }

    let context_switch_episode = run_one(
        &mut persistent,
        5,
        "literal-safety-high-keyword",
        "Assess the safety risk, harm, and dangerous consequences.",
    )?;

    let mut fresh_general_episodes = Vec::new();
    for episode in 0..5 {
        let mut fresh = MetaCognitiveReasoner::new(
            EvolutionConfig::default(),
            MetaReasoningConfig::default(),
        )?;
        fresh_general_episodes.push(run_one(
            &mut fresh,
            episode,
            "general-low-keyword",
            "Consider this situation carefully.",
        )?);
    }

    Ok(MetaControlProbeReport {
        probe_version: META_CONTROL_PROBE_VERSION.into(),
        persistent_general_episodes,
        context_switch_episode,
        fresh_general_episodes,
    })
}

fn run_one(
    reasoner: &mut MetaCognitiveReasoner,
    episode: usize,
    query_class: &str,
    query: &str,
) -> anyhow::Result<MetaControlObservation> {
    let mut chain = ReasoningChain::new(BinaryHV::random(10_000 + episode as u64));
    let result = reasoner.meta_reason(query, vec![probe_primitive()], &mut chain)?;
    let state = reasoner.state();
    let current_context_committed =
        state.context_reflection.detected_context == result.context_reflection.detected_context
            && (state.context_reflection.confidence - result.context_reflection.confidence).abs()
                <= f64::EPSILON;

    Ok(MetaControlObservation {
        episode,
        query_class: query_class.into(),
        returned_context_confidence: result.context_reflection.confidence,
        returned_meta_confidence: result.meta_confidence,
        strategy_adjustment_requested: result.strategy_reflection.adjust_strategy,
        state_context_confidence_after: state.context_reflection.confidence,
        current_context_committed,
        decision_history_len_after: state.decision_history.len(),
        meta_insights_len: result.meta_insights.len(),
    })
}

fn probe_primitive() -> CandidatePrimitive {
    CandidatePrimitive {
        name: "rq006g-probe".into(),
        tier: PrimitiveTier::Physical,
        definition: "fixed development-probe primitive".into(),
        fitness: 0.9,
        encoding: BinaryHV::random(6_006),
        epistemic_coordinate: EpistemicCoordinate::default(),
        harmonic_alignment: 0.8,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_records_both_persistent_and_fresh_conditions() {
        let report = run_meta_control_probe().expect("probe should execute");
        assert_eq!(report.persistent_general_episodes.len(), 5);
        assert_eq!(report.fresh_general_episodes.len(), 5);
        for observation in report
            .persistent_general_episodes
            .iter()
            .chain(report.fresh_general_episodes.iter())
            .chain(std::iter::once(&report.context_switch_episode))
        {
            assert!(observation.returned_context_confidence.is_finite());
            assert!((0.0..=1.0).contains(&observation.returned_context_confidence));
            assert!(observation.returned_meta_confidence.is_finite());
            assert!((0.0..=1.0).contains(&observation.returned_meta_confidence));
        }
    }

    #[test]
    fn current_subject_baseline_exposes_short_circuited_meta_learning() {
        let report = run_meta_control_probe().expect("probe should execute");
        assert!(
            report
                .persistent_general_episodes
                .iter()
                .all(|observation| observation.strategy_adjustment_requested),
            "current default path is expected to request strategy adjustment every episode"
        );
        assert!(
            report
                .persistent_general_episodes
                .iter()
                .all(|observation| observation.meta_insights_len == 0),
            "current default early-return path should expose unreachable meta-learning"
        );
        assert_eq!(
            report
                .persistent_general_episodes
                .last()
                .expect("persistent observations")
                .decision_history_len_after,
            5,
            "decision history can accumulate even while meta-learning is skipped"
        );
    }

    #[test]
    fn current_subject_baseline_exposes_uncommitted_high_confidence_context() {
        let report = run_meta_control_probe().expect("probe should execute");
        let switched = &report.context_switch_episode;
        assert!(switched.returned_context_confidence > 0.7);
        assert!(
            !switched.current_context_committed,
            "current adjustment early return should leave prior context in state"
        );
        assert!(
            switched.state_context_confidence_after < switched.returned_context_confidence,
            "stored state should visibly lag the returned current reflection in the baseline"
        );
    }

    #[test]
    fn fresh_subject_baseline_restarts_decision_history() {
        let report = run_meta_control_probe().expect("probe should execute");
        assert!(
            report
                .fresh_general_episodes
                .iter()
                .all(|observation| observation.decision_history_len_after == 1),
            "fresh reasoners should expose the per-episode reset boundary"
        );
    }
}
