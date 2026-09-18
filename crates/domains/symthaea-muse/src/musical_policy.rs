// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Explicit policy evaluation, separate from the learned outcome model.
//!
//! The outcome model answers "what will this intervention do?". This module
//! answers "which predicted valid outcome best serves the artist and formal
//! obligation?". Prediction accuracy is retained as evidence after measurement
//! but is never rewarded as musical value during candidate selection.

use crate::adaptive_prediction::{InterventionCalibrationEvidence, OutcomeUncertainty};
use crate::cognitive_bridge::{
    MusicalOutcomeError, ObservedMusicalOutcome, PredictedMusicalOutcome, SymbolicAction,
};
use serde::{Deserialize, Serialize};
use symthaea_music_theory::TheoryValidationReport;

pub const MUSICAL_POLICY_VERSION: &str = "musical-policy-v1";
pub const EFFECT_COVERAGE_VERSION: &str = "effect-coverage-v1";
const EFFECT_REQUEST_EPSILON: f32 = 1.0e-6;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutcomeChannel {
    Tension,
    Density,
    Familiarity,
    TonalDisplacement,
}

impl OutcomeChannel {
    const ALL: [Self; 4] = [
        Self::Tension,
        Self::Density,
        Self::Familiarity,
        Self::TonalDisplacement,
    ];

    fn desired(self, outcome: PredictedMusicalOutcome) -> f32 {
        match self {
            Self::Tension => outcome.tension_delta,
            Self::Density => outcome.density_delta,
            Self::Familiarity => outcome.familiarity_delta,
            Self::TonalDisplacement => outcome.tonal_displacement_delta,
        }
    }

    fn observed(self, outcome: ObservedMusicalOutcome) -> f32 {
        match self {
            Self::Tension => outcome.tension_delta,
            Self::Density => outcome.density_delta,
            Self::Familiarity => outcome.familiarity_delta,
            Self::TonalDisplacement => outcome.tonal_displacement_delta,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChannelEffectCoverageV1 {
    pub channel: OutcomeChannel,
    pub requested_delta: f32,
    pub requested: bool,
    pub measured_alternatives: usize,
    pub observed_min: Option<f32>,
    pub observed_max: Option<f32>,
    pub observed_span: Option<f32>,
    /// Maximum measured movement in the sign requested by `requested_delta`.
    /// A candidate family that only moves in the opposite direction therefore
    /// cannot qualify the requested capability merely because its raw span is large.
    pub directional_reach: Option<f32>,
    pub supported: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EffectCoverageDisposition {
    CognitiveEligible,
    NoRequestedEffect,
    ShadowFallbackInvalidRequest,
    ShadowFallbackInsufficientAlternatives,
    ShadowFallbackUnsupportedDominant,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EffectCoverageV1 {
    pub coverage_version: String,
    pub minimum_effect_span: f32,
    pub dominant_channel: Option<OutcomeChannel>,
    /// Candidates that can actually compete at the musical-policy utility tier:
    /// theory/Preserve-valid and tied on every lexicographic gate that precedes
    /// `outcome_utility` (target verification rank and overdue obligations).
    pub formal_frontier_ids: Vec<String>,
    pub measured_frontier_alternatives: usize,
    pub requested_channels: usize,
    pub supported_requested_channels: usize,
    pub requested_coverage_fraction: f32,
    pub channels: Vec<ChannelEffectCoverageV1>,
    pub disposition: EffectCoverageDisposition,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct OutcomeChannelWeights {
    pub tension: f32,
    pub density: f32,
    pub familiarity: f32,
    pub tonal_displacement: f32,
}

impl OutcomeChannelWeights {
    fn normalized(self) -> Self {
        let sum = self.tension.max(0.0)
            + self.density.max(0.0)
            + self.familiarity.max(0.0)
            + self.tonal_displacement.max(0.0);
        if sum <= f32::EPSILON {
            return Self {
                tension: 0.25,
                density: 0.25,
                familiarity: 0.25,
                tonal_displacement: 0.25,
            };
        }
        Self {
            tension: self.tension.max(0.0) / sum,
            density: self.density.max(0.0) / sum,
            familiarity: self.familiarity.max(0.0) / sum,
            tonal_displacement: self.tonal_displacement.max(0.0) / sum,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MusicalPolicyPreference {
    pub policy_version: String,
    pub action: SymbolicAction,
    pub desired_outcome: PredictedMusicalOutcome,
    pub channel_weights: OutcomeChannelWeights,
    pub uncertainty_penalty_weight: f32,
    pub motif_identity_weight: f32,
}

impl MusicalPolicyPreference {
    /// Frozen initial utility targets. These are preferences, not predictions;
    /// learning an outcome model never mutates them.
    pub fn for_action(action: SymbolicAction) -> Self {
        let (desired_outcome, channel_weights, motif_identity_weight) = match action {
            SymbolicAction::ReturnOpeningMaterial => (
                PredictedMusicalOutcome {
                    tension_delta: -0.2,
                    density_delta: 0.0,
                    familiarity_delta: 0.5,
                    tonal_displacement_delta: -0.35,
                },
                OutcomeChannelWeights {
                    tension: 0.20,
                    density: 0.10,
                    familiarity: 0.40,
                    tonal_displacement: 0.30,
                },
                0.30,
            ),
            SymbolicAction::StrengthenCadence => (
                PredictedMusicalOutcome {
                    tension_delta: -0.35,
                    density_delta: -0.05,
                    familiarity_delta: 0.2,
                    tonal_displacement_delta: -0.2,
                },
                OutcomeChannelWeights {
                    tension: 0.40,
                    density: 0.10,
                    familiarity: 0.15,
                    tonal_displacement: 0.35,
                },
                0.10,
            ),
            _ => (
                crate::cognitive_bridge::default_predicted_outcome(action),
                OutcomeChannelWeights {
                    tension: 0.25,
                    density: 0.25,
                    familiarity: 0.25,
                    tonal_displacement: 0.25,
                },
                0.10,
            ),
        };
        Self {
            policy_version: MUSICAL_POLICY_VERSION.into(),
            action,
            desired_outcome,
            channel_weights,
            uncertainty_penalty_weight: 0.15,
            motif_identity_weight,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolicyCandidateEvidence {
    pub alternative_id: String,
    pub theory_validation: TheoryValidationReport,
    pub preserved_invariants: bool,
    pub overdue_obligations_remaining: usize,
    pub unresolved_obligations_remaining: usize,
    pub obligation_pressure_remaining: f32,
    pub target_obligation_verified: Option<bool>,
    pub motif_return_similarity: Option<f32>,
    pub prediction: InterventionCalibrationEvidence,
    /// Measured outcome is retained only to audit the world model. It is not
    /// consulted by the policy ordering.
    pub observed_outcome: Option<ObservedMusicalOutcome>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolicyCandidateAssessment {
    pub alternative_id: String,
    pub eligible: bool,
    pub theory_validation: TheoryValidationReport,
    pub preserved_invariants: bool,
    pub overdue_obligations_remaining: usize,
    pub unresolved_obligations_remaining: usize,
    pub obligation_pressure_remaining: f32,
    pub target_obligation_verified: Option<bool>,
    pub motif_return_similarity: Option<f32>,
    pub predicted_outcome: PredictedMusicalOutcome,
    pub outcome_utility: f32,
    pub uncertainty_penalty: f32,
    pub prediction_error: Option<MusicalOutcomeError>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MusicalPolicySelection {
    pub policy: MusicalPolicyPreference,
    pub recommended_id: Option<String>,
    pub rationale: Vec<String>,
    pub assessments: Vec<PolicyCandidateAssessment>,
}

fn effect_target_is_finite(outcome: PredictedMusicalOutcome) -> bool {
    outcome.tension_delta.is_finite()
        && outcome.density_delta.is_finite()
        && outcome.familiarity_delta.is_finite()
        && outcome.tonal_displacement_delta.is_finite()
}

/// Measure whether the candidate family can *actually express* the effect a
/// policy target requests, using measured symbolic outcomes rather than model
/// predictions.
///
/// Coverage is evaluated only over the formal frontier: candidates that passed
/// canonical theory + Preserve validation and are tied on all lexicographic
/// gates that occur before musical utility (target-verification rank and
/// overdue-obligation count). A musically invalid or formally dominated
/// alternative therefore cannot be used to claim cognitive capability.
///
/// A requested channel qualifies only if the frontier has at least two measured
/// alternatives, spans `minimum_effect_span`, and reaches at least that far in
/// the requested sign. The dominant requested channel must qualify before a
/// future cognition-enabled product path should receive authority.
pub fn assess_effect_coverage(
    policy: &MusicalPolicyPreference,
    candidates: &[PolicyCandidateEvidence],
    minimum_effect_span: f32,
) -> EffectCoverageV1 {
    if !minimum_effect_span.is_finite() || !effect_target_is_finite(policy.desired_outcome) {
        return EffectCoverageV1 {
            coverage_version: EFFECT_COVERAGE_VERSION.into(),
            minimum_effect_span: if minimum_effect_span.is_finite() {
                minimum_effect_span.max(0.0)
            } else {
                0.0
            },
            dominant_channel: None,
            formal_frontier_ids: Vec::new(),
            measured_frontier_alternatives: 0,
            requested_channels: 0,
            supported_requested_channels: 0,
            requested_coverage_fraction: 0.0,
            channels: Vec::new(),
            disposition: EffectCoverageDisposition::ShadowFallbackInvalidRequest,
        };
    }

    let minimum_effect_span = minimum_effect_span.max(0.0);
    let mut eligible: Vec<&PolicyCandidateEvidence> = candidates
        .iter()
        .filter(|candidate| candidate.theory_validation.valid && candidate.preserved_invariants)
        .collect();

    let best_target_rank = eligible
        .iter()
        .map(|candidate| target_rank(candidate.target_obligation_verified))
        .min();
    if let Some(rank) = best_target_rank {
        eligible.retain(|candidate| target_rank(candidate.target_obligation_verified) == rank);
    }
    let best_overdue = eligible
        .iter()
        .map(|candidate| candidate.overdue_obligations_remaining)
        .min();
    if let Some(overdue) = best_overdue {
        eligible.retain(|candidate| candidate.overdue_obligations_remaining == overdue);
    }

    let mut formal_frontier_ids: Vec<_> = eligible
        .iter()
        .map(|candidate| candidate.alternative_id.clone())
        .collect();
    formal_frontier_ids.sort();

    let observed: Vec<_> = eligible
        .iter()
        .filter_map(|candidate| candidate.observed_outcome)
        .collect();
    let desired = policy.desired_outcome;
    let dominant_channel = OutcomeChannel::ALL
        .into_iter()
        .filter(|channel| channel.desired(desired).abs() > EFFECT_REQUEST_EPSILON)
        .max_by(|left, right| {
            left.desired(desired)
                .abs()
                .total_cmp(&right.desired(desired).abs())
        });

    let mut channels = Vec::with_capacity(OutcomeChannel::ALL.len());
    for channel in OutcomeChannel::ALL {
        let requested_delta = channel.desired(desired);
        let requested = requested_delta.abs() > EFFECT_REQUEST_EPSILON;
        let values: Vec<f32> = observed
            .iter()
            .map(|outcome| channel.observed(*outcome))
            .filter(|value| value.is_finite())
            .collect();
        let (observed_min, observed_max, observed_span) = if values.is_empty() {
            (None, None, None)
        } else {
            let min = values.iter().copied().fold(f32::INFINITY, f32::min);
            let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            (Some(min), Some(max), Some((max - min).max(0.0)))
        };
        let directional_reach = match (requested, observed_min, observed_max) {
            (true, Some(_), Some(max)) if requested_delta > 0.0 => Some(max.max(0.0)),
            (true, Some(min), Some(_)) if requested_delta < 0.0 => Some((-min).max(0.0)),
            (true, _, _) => None,
            (false, _, _) => Some(0.0),
        };
        let supported = requested
            && values.len() >= 2
            && observed_span.is_some_and(|span| span >= minimum_effect_span)
            && directional_reach.is_some_and(|reach| reach >= minimum_effect_span);
        channels.push(ChannelEffectCoverageV1 {
            channel,
            requested_delta,
            requested,
            measured_alternatives: values.len(),
            observed_min,
            observed_max,
            observed_span,
            directional_reach,
            supported,
        });
    }

    let requested_channels = channels.iter().filter(|channel| channel.requested).count();
    let supported_requested_channels = channels
        .iter()
        .filter(|channel| channel.requested && channel.supported)
        .count();
    let requested_coverage_fraction = if requested_channels == 0 {
        1.0
    } else {
        supported_requested_channels as f32 / requested_channels as f32
    };
    let dominant_supported = dominant_channel.is_some_and(|dominant| {
        channels
            .iter()
            .find(|channel| channel.channel == dominant)
            .is_some_and(|channel| channel.supported)
    });
    let disposition = if dominant_channel.is_none() {
        EffectCoverageDisposition::NoRequestedEffect
    } else if observed.len() < 2 {
        EffectCoverageDisposition::ShadowFallbackInsufficientAlternatives
    } else if dominant_supported {
        EffectCoverageDisposition::CognitiveEligible
    } else {
        EffectCoverageDisposition::ShadowFallbackUnsupportedDominant
    };

    EffectCoverageV1 {
        coverage_version: EFFECT_COVERAGE_VERSION.into(),
        minimum_effect_span,
        dominant_channel,
        formal_frontier_ids,
        measured_frontier_alternatives: observed.len(),
        requested_channels,
        supported_requested_channels,
        requested_coverage_fraction,
        channels,
        disposition,
    }
}

pub fn select_by_musical_policy(
    policy: MusicalPolicyPreference,
    candidates: &[PolicyCandidateEvidence],
) -> MusicalPolicySelection {
    let weights = policy.channel_weights.normalized();
    let mut assessments: Vec<_> = candidates
        .iter()
        .map(|candidate| {
            let prediction = candidate.prediction.calibrated;
            let distance = weighted_distance(prediction, policy.desired_outcome, weights);
            let uncertainty_penalty =
                weighted_uncertainty(candidate.prediction.uncertainty, weights)
                    * policy.uncertainty_penalty_weight.max(0.0);
            let motif_bonus = candidate
                .motif_return_similarity
                .unwrap_or(0.0)
                .clamp(0.0, 1.0)
                * policy.motif_identity_weight.max(0.0);
            let outcome_utility = motif_bonus - distance - uncertainty_penalty;
            PolicyCandidateAssessment {
                alternative_id: candidate.alternative_id.clone(),
                eligible: candidate.theory_validation.valid && candidate.preserved_invariants,
                theory_validation: candidate.theory_validation.clone(),
                preserved_invariants: candidate.preserved_invariants,
                overdue_obligations_remaining: candidate.overdue_obligations_remaining,
                unresolved_obligations_remaining: candidate.unresolved_obligations_remaining,
                obligation_pressure_remaining: candidate
                    .obligation_pressure_remaining
                    .clamp(0.0, 1.0),
                target_obligation_verified: candidate.target_obligation_verified,
                motif_return_similarity: candidate
                    .motif_return_similarity
                    .map(|value| value.clamp(0.0, 1.0)),
                predicted_outcome: prediction,
                outcome_utility,
                uncertainty_penalty,
                prediction_error: candidate
                    .observed_outcome
                    .map(|observed| prediction.error(observed)),
            }
        })
        .collect();

    assessments.sort_by(|left, right| {
        right
            .eligible
            .cmp(&left.eligible)
            .then_with(|| {
                target_rank(left.target_obligation_verified)
                    .cmp(&target_rank(right.target_obligation_verified))
            })
            .then_with(|| {
                left.overdue_obligations_remaining
                    .cmp(&right.overdue_obligations_remaining)
            })
            .then_with(|| right.outcome_utility.total_cmp(&left.outcome_utility))
            .then_with(|| {
                left.obligation_pressure_remaining
                    .total_cmp(&right.obligation_pressure_remaining)
            })
            .then_with(|| {
                left.unresolved_obligations_remaining
                    .cmp(&right.unresolved_obligations_remaining)
            })
            .then_with(|| left.alternative_id.cmp(&right.alternative_id))
    });

    let recommended = assessments.iter().find(|assessment| assessment.eligible);
    let rationale = if let Some(selected) = recommended {
        vec![
            format!(
                "selected {} after canonical theory and Preserve-contract checks",
                selected.alternative_id
            ),
            format!(
                "policy utility {:.3} from predicted musical effects; uncertainty penalty {:.3}",
                selected.outcome_utility, selected.uncertainty_penalty
            ),
            "prediction accuracy is retained separately and did not participate in selection"
                .to_owned(),
        ]
    } else {
        vec!["no candidate passed canonical theory and Preserve-contract checks".into()]
    };

    MusicalPolicySelection {
        policy,
        recommended_id: recommended.map(|assessment| assessment.alternative_id.clone()),
        rationale,
        assessments,
    }
}

fn weighted_distance(
    actual: PredictedMusicalOutcome,
    desired: PredictedMusicalOutcome,
    weights: OutcomeChannelWeights,
) -> f32 {
    (actual.tension_delta - desired.tension_delta).abs() * weights.tension
        + (actual.density_delta - desired.density_delta).abs() * weights.density
        + (actual.familiarity_delta - desired.familiarity_delta).abs() * weights.familiarity
        + (actual.tonal_displacement_delta - desired.tonal_displacement_delta).abs()
            * weights.tonal_displacement
}

fn weighted_uncertainty(uncertainty: OutcomeUncertainty, weights: OutcomeChannelWeights) -> f32 {
    uncertainty.tension.abs() * weights.tension
        + uncertainty.density.abs() * weights.density
        + uncertainty.familiarity.abs() * weights.familiarity
        + uncertainty.tonal_displacement.abs() * weights.tonal_displacement
}

fn target_rank(value: Option<bool>) -> u8 {
    match value {
        Some(true) => 0,
        None => 1,
        Some(false) => 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adaptive_prediction::{
        InterventionPredictionContext, PredictionContext, PredictionEvidenceSource, TextureBand,
    };
    use crate::cognitive_bridge::CognitiveSection;
    use crate::intervention::{InterventionDescriptor, InterventionStrategy, ObligationClass};
    use symthaea_music_theory::{ScoreValidationConfig, validate_score};

    fn prediction(
        id: &str,
        predicted: PredictedMusicalOutcome,
        observed: ObservedMusicalOutcome,
    ) -> PolicyCandidateEvidence {
        let base = PredictionContext::new(
            SymbolicAction::ReturnOpeningMaterial,
            CognitiveSection::Recapitulation,
            "Sonata",
            "Sonata",
            4,
            TextureBand::Chamber,
        );
        let descriptor = InterventionDescriptor::new(
            SymbolicAction::ReturnOpeningMaterial,
            InterventionStrategy::Literal,
            CognitiveSection::Exposition,
            CognitiveSection::Recapitulation,
            ObligationClass::ReturnMotif,
            0,
            1.0,
            0.5,
            0.5,
            0.5,
            0.5,
            8,
            80,
        );
        let empty = symthaea_music_theory::Score::new(
            symthaea_music_theory::Key::major(symthaea_music_theory::PitchClass::C),
            120.0,
            4,
        );
        let mut report = validate_score(&empty, &ScoreValidationConfig::default());
        report.valid = true;
        report.issues.clear();
        PolicyCandidateEvidence {
            alternative_id: id.into(),
            theory_validation: report,
            preserved_invariants: true,
            overdue_obligations_remaining: 0,
            unresolved_obligations_remaining: 0,
            obligation_pressure_remaining: 0.0,
            target_obligation_verified: Some(true),
            motif_return_similarity: Some(1.0),
            prediction: InterventionCalibrationEvidence {
                model_version: "test".into(),
                context: InterventionPredictionContext::new(base, descriptor),
                source: PredictionEvidenceSource::HandAuthoredPrior,
                intervention_context_samples: 0,
                strategy_fallback_samples: 0,
                action_fallback_samples: 0,
                intervention_context_moments: None,
                strategy_fallback_moments: None,
                action_fallback_moments: None,
                prior: predicted,
                calibrated: predicted,
                uncertainty: OutcomeUncertainty::default(),
            },
            observed_outcome: Some(observed),
        }
    }

    fn outcome(density: f32) -> ObservedMusicalOutcome {
        ObservedMusicalOutcome {
            tension_delta: 0.0,
            density_delta: density,
            familiarity_delta: 0.0,
            tonal_displacement_delta: 0.0,
        }
    }

    fn density_policy() -> MusicalPolicyPreference {
        let mut policy = MusicalPolicyPreference::for_action(SymbolicAction::ReturnOpeningMaterial);
        policy.desired_outcome = PredictedMusicalOutcome {
            tension_delta: 0.0,
            density_delta: 0.3,
            familiarity_delta: 0.0,
            tonal_displacement_delta: 0.0,
        };
        policy
    }

    #[test]
    fn effect_coverage_fails_closed_on_nonfinite_requests() {
        let finite_target = density_policy().desired_outcome;
        let mut policy = density_policy();
        policy.desired_outcome.density_delta = f32::NAN;
        let candidates = vec![
            prediction("a", finite_target, outcome(0.0)),
            prediction("b", finite_target, outcome(0.2)),
        ];
        let coverage = assess_effect_coverage(&policy, &candidates, 0.05);
        assert_eq!(
            coverage.disposition,
            EffectCoverageDisposition::ShadowFallbackInvalidRequest
        );

        let coverage = assess_effect_coverage(&density_policy(), &candidates, f32::INFINITY);
        assert_eq!(
            coverage.disposition,
            EffectCoverageDisposition::ShadowFallbackInvalidRequest
        );
    }

    #[test]
    fn effect_coverage_rejects_a_zero_span_dominant_channel() {
        let policy = density_policy();
        let candidates = vec![
            prediction("a", policy.desired_outcome, outcome(0.0)),
            prediction("b", policy.desired_outcome, outcome(0.0)),
        ];
        let coverage = assess_effect_coverage(&policy, &candidates, 0.05);
        assert_eq!(coverage.dominant_channel, Some(OutcomeChannel::Density));
        assert_eq!(
            coverage.disposition,
            EffectCoverageDisposition::ShadowFallbackUnsupportedDominant
        );
        let density = coverage
            .channels
            .iter()
            .find(|channel| channel.channel == OutcomeChannel::Density)
            .unwrap();
        assert_eq!(density.observed_span, Some(0.0));
        assert_eq!(density.directional_reach, Some(0.0));
        assert!(!density.supported);
    }

    #[test]
    fn effect_coverage_requires_variation_and_reach_in_the_requested_direction() {
        let policy = density_policy();
        let supported = vec![
            prediction("baseline", policy.desired_outcome, outcome(0.0)),
            prediction("denser", policy.desired_outcome, outcome(0.2)),
        ];
        let coverage = assess_effect_coverage(&policy, &supported, 0.05);
        assert_eq!(
            coverage.disposition,
            EffectCoverageDisposition::CognitiveEligible
        );

        let wrong_direction = vec![
            prediction("baseline", policy.desired_outcome, outcome(0.0)),
            prediction("thinner", policy.desired_outcome, outcome(-0.2)),
        ];
        let coverage = assess_effect_coverage(&policy, &wrong_direction, 0.05);
        assert_eq!(
            coverage.disposition,
            EffectCoverageDisposition::ShadowFallbackUnsupportedDominant
        );
        let density = coverage
            .channels
            .iter()
            .find(|channel| channel.channel == OutcomeChannel::Density)
            .unwrap();
        assert_eq!(density.observed_span, Some(0.2));
        assert_eq!(density.directional_reach, Some(0.0));
    }

    #[test]
    fn formally_dominated_effects_do_not_prove_capability() {
        let policy = density_policy();
        let mut baseline = prediction("baseline", policy.desired_outcome, outcome(0.0));
        let mut peer = prediction("peer", policy.desired_outcome, outcome(0.0));
        let mut denser_but_overdue =
            prediction("denser-overdue", policy.desired_outcome, outcome(0.3));
        baseline.overdue_obligations_remaining = 0;
        peer.overdue_obligations_remaining = 0;
        denser_but_overdue.overdue_obligations_remaining = 1;
        let coverage = assess_effect_coverage(&policy, &[baseline, peer, denser_but_overdue], 0.05);
        assert_eq!(coverage.formal_frontier_ids, vec!["baseline", "peer"]);
        assert_eq!(
            coverage.disposition,
            EffectCoverageDisposition::ShadowFallbackUnsupportedDominant
        );
    }

    #[test]
    fn invalid_effectful_candidates_do_not_prove_capability() {
        let policy = density_policy();
        let baseline = prediction("baseline", policy.desired_outcome, outcome(0.0));
        let peer = prediction("peer", policy.desired_outcome, outcome(0.0));
        let mut invalid = prediction("invalid-denser", policy.desired_outcome, outcome(0.3));
        invalid.theory_validation.valid = false;
        let coverage = assess_effect_coverage(&policy, &[baseline, peer, invalid], 0.05);
        assert_eq!(
            coverage.disposition,
            EffectCoverageDisposition::ShadowFallbackUnsupportedDominant
        );
        assert!(!coverage.formal_frontier_ids.contains(&"invalid-denser".to_owned()));
    }

    #[test]
    fn effect_coverage_requires_two_measured_frontier_alternatives() {
        let policy = density_policy();
        let only = prediction("only", policy.desired_outcome, outcome(0.3));
        let coverage = assess_effect_coverage(&policy, &[only], 0.05);
        assert_eq!(
            coverage.disposition,
            EffectCoverageDisposition::ShadowFallbackInsufficientAlternatives
        );
    }

    #[test]
    fn prediction_accuracy_is_not_mistaken_for_musical_utility() {
        let desirable = PredictedMusicalOutcome {
            tension_delta: -0.2,
            density_delta: 0.0,
            familiarity_delta: 0.5,
            tonal_displacement_delta: -0.35,
        };
        let undesirable = PredictedMusicalOutcome {
            tension_delta: 0.4,
            density_delta: 0.4,
            familiarity_delta: -0.4,
            tonal_displacement_delta: 0.5,
        };
        let candidates = vec![
            prediction(
                "desirable-but-imperfectly-predicted",
                desirable,
                ObservedMusicalOutcome {
                    tension_delta: 0.2,
                    density_delta: 0.2,
                    familiarity_delta: 0.1,
                    tonal_displacement_delta: 0.1,
                },
            ),
            prediction(
                "undesirable-but-perfectly-predicted",
                undesirable,
                ObservedMusicalOutcome {
                    tension_delta: 0.4,
                    density_delta: 0.4,
                    familiarity_delta: -0.4,
                    tonal_displacement_delta: 0.5,
                },
            ),
        ];
        let selection = select_by_musical_policy(
            MusicalPolicyPreference::for_action(SymbolicAction::ReturnOpeningMaterial),
            &candidates,
        );
        assert_eq!(
            selection.recommended_id.as_deref(),
            Some("desirable-but-imperfectly-predicted")
        );
        let accurate = selection
            .assessments
            .iter()
            .find(|item| item.alternative_id == "undesirable-but-perfectly-predicted")
            .unwrap();
        assert_eq!(accurate.prediction_error.unwrap().mean_absolute_error, 0.0);
    }
}
