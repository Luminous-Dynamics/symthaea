// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! MCE narrative boundary for non-lived narrative material.
//!
//! The Master Consciousness Equation's `NarrativeCoherence` distinguishes
//! autobiographical integration from prospective future-simulation depth. Neither
//! evidence class alone nor the fact that content was generated establishes that it
//! is prospective. Temporal class is therefore explicit.
//!
//! Retrospective counterfactuals (the current dream engine's alternative-past
//! wisdom), exact recollections, and atemporal generated material remain neutral to
//! MCE narrative scores. Only explicitly prospective non-lived scenarios may enter
//! `future_scenarios`. Generated material never enters autobiographical episodes.

use std::collections::HashSet;

use super::epistemic_world::WorldEvidenceKind;
use symthaea_consciousness_equation::NarrativeCoherence;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MceNarrativeTemporalClass {
    /// Recollection or alternative to an already elapsed event.
    Retrospective,
    /// Explicit possible future scenario.
    Prospective,
    /// Generated/derived material with no asserted temporal direction.
    Atemporal,
}

#[derive(Debug, Clone, Copy)]
pub struct MceDerivedNarrativeInput<'a> {
    pub evidence_kind: WorldEvidenceKind,
    pub temporal_class: MceNarrativeTemporalClass,
    pub source_digest: &'a str,
    pub description: &'a str,
    pub horizon_steps: usize,
    pub probability: f64,
    pub desirability: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MceDerivedNarrativeOutcome {
    /// An explicitly prospective generated/predicted item entered future simulation.
    FutureScenarioAdded,
    /// Exact replay is grounded recollection, not another lived/future event.
    RecollectionIgnored,
    /// Derived material is non-prospective and therefore neutral to MCE narrative score.
    NonProspectiveDerivedIgnored,
    /// This exact source was already routed during this runtime.
    DuplicateIgnored,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MceDerivedNarrativeError {
    EmptySourceDigest,
    EmptyDescription,
    RecordedMustUseLivedPath,
    NonFiniteProbability,
    ProbabilityOutOfRange,
    NonFiniteDesirability,
    DesirabilityOutOfRange,
}

/// Runtime-only deduplication/router for non-lived narrative material.
///
/// This type intentionally does not derive serde traits. A restart begins a fresh
/// runtime routing cache; durable narrative provenance requires a separate explicit
/// persistence design rather than silently restoring generated material as authority.
#[derive(Debug, Clone, Default)]
pub struct MceDerivedNarrativeRouter {
    routed_sources: HashSet<String>,
}

impl MceDerivedNarrativeRouter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn routed_count(&self) -> usize {
        self.routed_sources.len()
    }

    pub fn contains_source(&self, source_digest: &str) -> bool {
        self.routed_sources.contains(source_digest)
    }

    pub fn route(
        &mut self,
        narrative: &mut NarrativeCoherence,
        input: MceDerivedNarrativeInput<'_>,
    ) -> Result<MceDerivedNarrativeOutcome, MceDerivedNarrativeError> {
        validate_input(input)?;

        if self.contains_source(input.source_digest) {
            return Ok(MceDerivedNarrativeOutcome::DuplicateIgnored);
        }

        let outcome = match input.evidence_kind {
            WorldEvidenceKind::Recorded => {
                return Err(MceDerivedNarrativeError::RecordedMustUseLivedPath);
            }
            WorldEvidenceKind::ReplayDerived => MceDerivedNarrativeOutcome::RecollectionIgnored,
            WorldEvidenceKind::Interpolated
            | WorldEvidenceKind::ModelPredicted
            | WorldEvidenceKind::Counterfactual
            | WorldEvidenceKind::Extrapolated
            | WorldEvidenceKind::AdversarialGenerated => {
                if input.temporal_class != MceNarrativeTemporalClass::Prospective {
                    MceDerivedNarrativeOutcome::NonProspectiveDerivedIgnored
                } else {
                    narrative.add_future_scenario(
                        input.description.to_string(),
                        input.horizon_steps,
                        input.probability,
                        input.desirability,
                    );
                    MceDerivedNarrativeOutcome::FutureScenarioAdded
                }
            }
        };

        self.routed_sources.insert(input.source_digest.to_string());
        Ok(outcome)
    }
}

fn validate_input(input: MceDerivedNarrativeInput<'_>) -> Result<(), MceDerivedNarrativeError> {
    if input.source_digest.trim().is_empty() {
        return Err(MceDerivedNarrativeError::EmptySourceDigest);
    }
    if input.description.trim().is_empty() {
        return Err(MceDerivedNarrativeError::EmptyDescription);
    }
    if !input.probability.is_finite() {
        return Err(MceDerivedNarrativeError::NonFiniteProbability);
    }
    if !(0.0..=1.0).contains(&input.probability) {
        return Err(MceDerivedNarrativeError::ProbabilityOutOfRange);
    }
    if !input.desirability.is_finite() {
        return Err(MceDerivedNarrativeError::NonFiniteDesirability);
    }
    if !(-1.0..=1.0).contains(&input.desirability) {
        return Err(MceDerivedNarrativeError::DesirabilityOutOfRange);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counterfactual<'a>(digest: &'a str) -> MceDerivedNarrativeInput<'a> {
        MceDerivedNarrativeInput {
            evidence_kind: WorldEvidenceKind::Counterfactual,
            temporal_class: MceNarrativeTemporalClass::Retrospective,
            source_digest: digest,
            description: "dream-generated alternative past",
            horizon_steps: 1,
            probability: 0.5,
            desirability: 0.4,
        }
    }

    #[test]
    fn retrospective_counterfactual_is_neutral_to_mce_scores() {
        let mut narrative = NarrativeCoherence::new();
        let mut router = MceDerivedNarrativeRouter::new();
        let episodes_before = narrative.episode_count();
        let scenarios_before = narrative.scenario_count();

        assert_eq!(
            router.route(&mut narrative, counterfactual("blake3:dream-1")),
            Ok(MceDerivedNarrativeOutcome::NonProspectiveDerivedIgnored)
        );
        assert_eq!(narrative.episode_count(), episodes_before);
        assert_eq!(narrative.scenario_count(), scenarios_before);
    }

    #[test]
    fn explicitly_prospective_prediction_uses_future_scenario_path() {
        let mut narrative = NarrativeCoherence::new();
        let mut router = MceDerivedNarrativeRouter::new();
        let mut input = counterfactual("blake3:future-1");
        input.evidence_kind = WorldEvidenceKind::ModelPredicted;
        input.temporal_class = MceNarrativeTemporalClass::Prospective;
        input.description = "predicted future state";

        assert_eq!(
            router.route(&mut narrative, input),
            Ok(MceDerivedNarrativeOutcome::FutureScenarioAdded)
        );
        assert_eq!(narrative.episode_count(), 0);
        assert_eq!(narrative.scenario_count(), 1);
    }

    #[test]
    fn duplicate_source_cannot_reapply_mce_effect() {
        let mut narrative = NarrativeCoherence::new();
        let mut router = MceDerivedNarrativeRouter::new();
        let mut input = counterfactual("blake3:same-future");
        input.evidence_kind = WorldEvidenceKind::ModelPredicted;
        input.temporal_class = MceNarrativeTemporalClass::Prospective;

        router.route(&mut narrative, input).unwrap();
        let scenarios_after_first = narrative.scenario_count();
        assert_eq!(
            router.route(&mut narrative, input),
            Ok(MceDerivedNarrativeOutcome::DuplicateIgnored)
        );
        assert_eq!(narrative.scenario_count(), scenarios_after_first);
    }

    #[test]
    fn replay_is_recollection_not_new_episode_or_scenario() {
        let mut narrative = NarrativeCoherence::new();
        let mut router = MceDerivedNarrativeRouter::new();
        let mut input = counterfactual("blake3:replay");
        input.evidence_kind = WorldEvidenceKind::ReplayDerived;

        assert_eq!(
            router.route(&mut narrative, input),
            Ok(MceDerivedNarrativeOutcome::RecollectionIgnored)
        );
        assert_eq!(narrative.episode_count(), 0);
        assert_eq!(narrative.scenario_count(), 0);
    }

    #[test]
    fn recorded_observation_cannot_enter_through_generated_router() {
        let mut narrative = NarrativeCoherence::new();
        let mut router = MceDerivedNarrativeRouter::new();
        let mut input = counterfactual("blake3:recorded");
        input.evidence_kind = WorldEvidenceKind::Recorded;

        assert_eq!(
            router.route(&mut narrative, input),
            Err(MceDerivedNarrativeError::RecordedMustUseLivedPath)
        );
        assert_eq!(narrative.episode_count(), 0);
        assert_eq!(narrative.scenario_count(), 0);
    }
}
