// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Transparent accounting for requested and observed musical intervention effects.
//!
//! This module deliberately does **not** compute a musical-quality score. It
//! records whether each requested symbolic outcome channel moved in the
//! requested direction and separately records unrequested movement. That keeps
//! "the intervention achieved its target" distinct from "the intervention had
//! no collateral effects" and from any later listener-quality judgment.
//!
//! The evidence is descriptive. An observed delta after an intervention is not,
//! by itself, a general causal claim about music or perception.

use crate::cognitive_bridge::{ObservedMusicalOutcome, PredictedMusicalOutcome};
use crate::musical_policy::OutcomeChannel;
use serde::{Deserialize, Serialize};

pub const INTERVENTION_EFFECT_EVIDENCE_VERSION: &str = "intervention-effect-evidence-v1";

const CHANNELS: [OutcomeChannel; 4] = [
    OutcomeChannel::Tension,
    OutcomeChannel::Density,
    OutcomeChannel::Familiarity,
    OutcomeChannel::TonalDisplacement,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InterventionEffectEvidenceDisposition {
    Valid,
    InvalidRequest,
    InvalidObservation,
    InvalidTolerance,
}

/// The role a channel plays in the requested intervention target.
///
/// Only one channel is marked `Dominant`: the largest requested absolute
/// delta, with the fixed channel order used as a deterministic tie-break.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RequestRoleV1 {
    Dominant,
    Secondary,
    Unrequested,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EffectRelationV1 {
    /// A requested channel moved beyond tolerance in the requested sign.
    RequestedAligned,
    /// A requested channel moved beyond tolerance in the opposite sign.
    RequestedOpposed,
    /// A requested channel did not move beyond tolerance.
    RequestedUnchanged,
    /// An unrequested channel stayed within tolerance.
    UnrequestedPreserved,
    /// An unrequested channel moved beyond tolerance. This is retained as
    /// collateral movement, not automatically labelled beneficial or harmful.
    UnrequestedChanged,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChannelInterventionEffectV1 {
    pub channel: OutcomeChannel,
    pub request_role: RequestRoleV1,
    pub requested_delta: f32,
    pub observed_delta: f32,
    pub moved: bool,
    pub relation: EffectRelationV1,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InterventionEffectEvidenceV1 {
    pub evidence_version: String,
    pub movement_epsilon: f32,
    pub disposition: InterventionEffectEvidenceDisposition,
    pub dominant_requested_channel: Option<OutcomeChannel>,
    pub requested_channels: usize,
    pub aligned_requested_channels: usize,
    pub opposed_requested_channels: usize,
    pub unchanged_requested_channels: usize,
    pub preserved_unrequested_channels: usize,
    pub changed_unrequested_channels: usize,
    pub channels: Vec<ChannelInterventionEffectV1>,
}

impl InterventionEffectEvidenceV1 {
    fn invalid(disposition: InterventionEffectEvidenceDisposition) -> Self {
        Self {
            evidence_version: INTERVENTION_EFFECT_EVIDENCE_VERSION.into(),
            movement_epsilon: 0.0,
            disposition,
            dominant_requested_channel: None,
            requested_channels: 0,
            aligned_requested_channels: 0,
            opposed_requested_channels: 0,
            unchanged_requested_channels: 0,
            preserved_unrequested_channels: 0,
            changed_unrequested_channels: 0,
            channels: Vec::new(),
        }
    }
}

/// Describe the relationship between a requested symbolic effect and one
/// measured outcome without converting the result into a scalar utility.
///
/// `movement_epsilon` defines both when a target channel counts as requested
/// and when an observed channel counts as having moved. Invalid requests,
/// observations, or tolerances fail closed and emit no per-channel evidence,
/// preventing NaN/inf values from entering serialized evidence.
pub fn account_intervention_effects(
    requested: PredictedMusicalOutcome,
    observed: ObservedMusicalOutcome,
    movement_epsilon: f32,
) -> InterventionEffectEvidenceV1 {
    if !movement_epsilon.is_finite() || movement_epsilon < 0.0 {
        return InterventionEffectEvidenceV1::invalid(
            InterventionEffectEvidenceDisposition::InvalidTolerance,
        );
    }
    if !predicted_is_finite(requested) {
        return InterventionEffectEvidenceV1::invalid(
            InterventionEffectEvidenceDisposition::InvalidRequest,
        );
    }
    if !observed_is_finite(observed) {
        return InterventionEffectEvidenceV1::invalid(
            InterventionEffectEvidenceDisposition::InvalidObservation,
        );
    }

    let dominant_requested_channel = dominant_requested_channel(requested, movement_epsilon);
    let mut channels = Vec::with_capacity(CHANNELS.len());
    for channel in CHANNELS {
        let requested_delta = requested_value(channel, requested);
        let observed_delta = observed_value(channel, observed);
        let is_requested = requested_delta.abs() > movement_epsilon;
        let request_role = if dominant_requested_channel == Some(channel) {
            RequestRoleV1::Dominant
        } else if is_requested {
            RequestRoleV1::Secondary
        } else {
            RequestRoleV1::Unrequested
        };
        let moved = observed_delta.abs() > movement_epsilon;
        let relation = if is_requested {
            if !moved {
                EffectRelationV1::RequestedUnchanged
            } else if requested_delta.is_sign_positive() == observed_delta.is_sign_positive() {
                EffectRelationV1::RequestedAligned
            } else {
                EffectRelationV1::RequestedOpposed
            }
        } else if moved {
            EffectRelationV1::UnrequestedChanged
        } else {
            EffectRelationV1::UnrequestedPreserved
        };
        channels.push(ChannelInterventionEffectV1 {
            channel,
            request_role,
            requested_delta,
            observed_delta,
            moved,
            relation,
        });
    }

    InterventionEffectEvidenceV1 {
        evidence_version: INTERVENTION_EFFECT_EVIDENCE_VERSION.into(),
        movement_epsilon,
        disposition: InterventionEffectEvidenceDisposition::Valid,
        dominant_requested_channel,
        requested_channels: count_relation(
            &channels,
            &[
                EffectRelationV1::RequestedAligned,
                EffectRelationV1::RequestedOpposed,
                EffectRelationV1::RequestedUnchanged,
            ],
        ),
        aligned_requested_channels: count_relation(
            &channels,
            &[EffectRelationV1::RequestedAligned],
        ),
        opposed_requested_channels: count_relation(
            &channels,
            &[EffectRelationV1::RequestedOpposed],
        ),
        unchanged_requested_channels: count_relation(
            &channels,
            &[EffectRelationV1::RequestedUnchanged],
        ),
        preserved_unrequested_channels: count_relation(
            &channels,
            &[EffectRelationV1::UnrequestedPreserved],
        ),
        changed_unrequested_channels: count_relation(
            &channels,
            &[EffectRelationV1::UnrequestedChanged],
        ),
        channels,
    }
}

fn count_relation(channels: &[ChannelInterventionEffectV1], relations: &[EffectRelationV1]) -> usize {
    channels
        .iter()
        .filter(|channel| relations.contains(&channel.relation))
        .count()
}

fn dominant_requested_channel(
    requested: PredictedMusicalOutcome,
    movement_epsilon: f32,
) -> Option<OutcomeChannel> {
    let mut best: Option<(OutcomeChannel, f32)> = None;
    for channel in CHANNELS {
        let magnitude = requested_value(channel, requested).abs();
        if magnitude <= movement_epsilon {
            continue;
        }
        if best.is_none_or(|(_, best_magnitude)| magnitude > best_magnitude) {
            best = Some((channel, magnitude));
        }
    }
    best.map(|(channel, _)| channel)
}

fn requested_value(channel: OutcomeChannel, outcome: PredictedMusicalOutcome) -> f32 {
    match channel {
        OutcomeChannel::Tension => outcome.tension_delta,
        OutcomeChannel::Density => outcome.density_delta,
        OutcomeChannel::Familiarity => outcome.familiarity_delta,
        OutcomeChannel::TonalDisplacement => outcome.tonal_displacement_delta,
    }
}

fn observed_value(channel: OutcomeChannel, outcome: ObservedMusicalOutcome) -> f32 {
    match channel {
        OutcomeChannel::Tension => outcome.tension_delta,
        OutcomeChannel::Density => outcome.density_delta,
        OutcomeChannel::Familiarity => outcome.familiarity_delta,
        OutcomeChannel::TonalDisplacement => outcome.tonal_displacement_delta,
    }
}

fn predicted_is_finite(outcome: PredictedMusicalOutcome) -> bool {
    outcome.tension_delta.is_finite()
        && outcome.density_delta.is_finite()
        && outcome.familiarity_delta.is_finite()
        && outcome.tonal_displacement_delta.is_finite()
}

fn observed_is_finite(outcome: ObservedMusicalOutcome) -> bool {
    outcome.tension_delta.is_finite()
        && outcome.density_delta.is_finite()
        && outcome.familiarity_delta.is_finite()
        && outcome.tonal_displacement_delta.is_finite()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn requested_density() -> PredictedMusicalOutcome {
        PredictedMusicalOutcome {
            tension_delta: 0.0,
            density_delta: 0.30,
            familiarity_delta: 0.0,
            tonal_displacement_delta: 0.0,
        }
    }

    fn channel<'a>(
        evidence: &'a InterventionEffectEvidenceV1,
        wanted: OutcomeChannel,
    ) -> &'a ChannelInterventionEffectV1 {
        evidence
            .channels
            .iter()
            .find(|channel| channel.channel == wanted)
            .unwrap()
    }

    #[test]
    fn requested_effect_and_collateral_movement_remain_separate() {
        let evidence = account_intervention_effects(
            requested_density(),
            ObservedMusicalOutcome {
                tension_delta: -0.03,
                density_delta: 0.18,
                familiarity_delta: 0.0,
                tonal_displacement_delta: 0.0,
            },
            0.01,
        );

        assert_eq!(
            evidence.disposition,
            InterventionEffectEvidenceDisposition::Valid
        );
        assert_eq!(evidence.dominant_requested_channel, Some(OutcomeChannel::Density));
        assert_eq!(evidence.requested_channels, 1);
        assert_eq!(evidence.aligned_requested_channels, 1);
        assert_eq!(evidence.opposed_requested_channels, 0);
        assert_eq!(evidence.changed_unrequested_channels, 1);
        assert_eq!(evidence.preserved_unrequested_channels, 2);
        assert_eq!(
            channel(&evidence, OutcomeChannel::Density).request_role,
            RequestRoleV1::Dominant
        );
        assert_eq!(
            channel(&evidence, OutcomeChannel::Density).relation,
            EffectRelationV1::RequestedAligned
        );
        assert_eq!(
            channel(&evidence, OutcomeChannel::Tension).request_role,
            RequestRoleV1::Unrequested
        );
        assert_eq!(
            channel(&evidence, OutcomeChannel::Tension).relation,
            EffectRelationV1::UnrequestedChanged
        );
        assert_eq!(
            channel(&evidence, OutcomeChannel::Familiarity).relation,
            EffectRelationV1::UnrequestedPreserved
        );
    }

    #[test]
    fn dominant_and_secondary_requests_are_explicit_per_channel() {
        let evidence = account_intervention_effects(
            PredictedMusicalOutcome {
                tension_delta: 0.15,
                density_delta: 0.35,
                familiarity_delta: 0.0,
                tonal_displacement_delta: 0.0,
            },
            ObservedMusicalOutcome {
                tension_delta: -0.02,
                density_delta: 0.10,
                familiarity_delta: 0.0,
                tonal_displacement_delta: 0.0,
            },
            0.01,
        );
        assert_eq!(
            channel(&evidence, OutcomeChannel::Density).request_role,
            RequestRoleV1::Dominant
        );
        assert_eq!(
            channel(&evidence, OutcomeChannel::Tension).request_role,
            RequestRoleV1::Secondary
        );
        assert_eq!(
            channel(&evidence, OutcomeChannel::Familiarity).request_role,
            RequestRoleV1::Unrequested
        );
        assert_eq!(
            channel(&evidence, OutcomeChannel::TonalDisplacement).request_role,
            RequestRoleV1::Unrequested
        );
        assert_eq!(
            channel(&evidence, OutcomeChannel::Tension).relation,
            EffectRelationV1::RequestedOpposed
        );
    }

    #[test]
    fn opposite_movement_is_not_mistaken_for_requested_support() {
        let evidence = account_intervention_effects(
            requested_density(),
            ObservedMusicalOutcome {
                tension_delta: 0.0,
                density_delta: -0.20,
                familiarity_delta: 0.0,
                tonal_displacement_delta: 0.0,
            },
            0.01,
        );
        assert_eq!(evidence.aligned_requested_channels, 0);
        assert_eq!(evidence.opposed_requested_channels, 1);
        assert_eq!(
            channel(&evidence, OutcomeChannel::Density).relation,
            EffectRelationV1::RequestedOpposed
        );
    }

    #[test]
    fn no_requested_effect_can_still_record_unrequested_change() {
        let evidence = account_intervention_effects(
            PredictedMusicalOutcome {
                tension_delta: 0.0,
                density_delta: 0.0,
                familiarity_delta: 0.0,
                tonal_displacement_delta: 0.0,
            },
            ObservedMusicalOutcome {
                tension_delta: 0.11,
                density_delta: 0.0,
                familiarity_delta: 0.0,
                tonal_displacement_delta: 0.0,
            },
            0.01,
        );
        assert_eq!(evidence.dominant_requested_channel, None);
        assert_eq!(evidence.requested_channels, 0);
        assert_eq!(evidence.changed_unrequested_channels, 1);
        assert!(
            evidence
                .channels
                .iter()
                .all(|channel| channel.request_role == RequestRoleV1::Unrequested)
        );
    }

    #[test]
    fn non_finite_inputs_fail_closed_without_nan_evidence() {
        let invalid_request = account_intervention_effects(
            PredictedMusicalOutcome {
                density_delta: f32::NAN,
                ..requested_density()
            },
            ObservedMusicalOutcome {
                tension_delta: 0.0,
                density_delta: 0.1,
                familiarity_delta: 0.0,
                tonal_displacement_delta: 0.0,
            },
            0.01,
        );
        assert_eq!(
            invalid_request.disposition,
            InterventionEffectEvidenceDisposition::InvalidRequest
        );
        assert!(invalid_request.channels.is_empty());

        let invalid_observation = account_intervention_effects(
            requested_density(),
            ObservedMusicalOutcome {
                tension_delta: f32::INFINITY,
                density_delta: 0.1,
                familiarity_delta: 0.0,
                tonal_displacement_delta: 0.0,
            },
            0.01,
        );
        assert_eq!(
            invalid_observation.disposition,
            InterventionEffectEvidenceDisposition::InvalidObservation
        );
        assert!(invalid_observation.channels.is_empty());

        let invalid_tolerance = account_intervention_effects(
            requested_density(),
            ObservedMusicalOutcome {
                tension_delta: 0.0,
                density_delta: 0.1,
                familiarity_delta: 0.0,
                tonal_displacement_delta: 0.0,
            },
            f32::NAN,
        );
        assert_eq!(
            invalid_tolerance.disposition,
            InterventionEffectEvidenceDisposition::InvalidTolerance
        );
        assert!(invalid_tolerance.channels.is_empty());
    }
}
