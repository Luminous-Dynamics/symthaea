// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Aggregate repeated intervention-effect evidence without collapsing it into
//! a scalar quality score.
//!
//! A panel is valid only when every record describes the same requested target,
//! tolerance, channel layout, and dominant/secondary/unrequested roles. This
//! prevents a mixed experiment from being summarized as if it were one stable
//! intervention question.

use crate::evidence_digest::intervention_effect_evidence::{
    EffectRelationV1, INTERVENTION_EFFECT_EVIDENCE_VERSION, InterventionEffectEvidenceDisposition,
    InterventionEffectEvidenceV1, RequestRoleV1,
};
use crate::musical_policy::OutcomeChannel;
use serde::{Deserialize, Serialize};

pub const INTERVENTION_EFFECT_PANEL_VERSION: &str = "intervention-effect-panel-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct EffectRelationCountsV1 {
    pub requested_aligned: usize,
    pub requested_opposed: usize,
    pub requested_unchanged: usize,
    pub unrequested_preserved: usize,
    pub unrequested_changed: usize,
}

impl EffectRelationCountsV1 {
    fn observe(&mut self, relation: EffectRelationV1) {
        match relation {
            EffectRelationV1::RequestedAligned => self.requested_aligned += 1,
            EffectRelationV1::RequestedOpposed => self.requested_opposed += 1,
            EffectRelationV1::RequestedUnchanged => self.requested_unchanged += 1,
            EffectRelationV1::UnrequestedPreserved => self.unrequested_preserved += 1,
            EffectRelationV1::UnrequestedChanged => self.unrequested_changed += 1,
        }
    }

    pub fn total(self) -> usize {
        self.requested_aligned
            + self.requested_opposed
            + self.requested_unchanged
            + self.unrequested_preserved
            + self.unrequested_changed
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChannelEffectPanelV1 {
    pub channel: OutcomeChannel,
    pub request_role: RequestRoleV1,
    pub requested_delta: f32,
    pub samples: usize,
    pub observed_min: f32,
    pub observed_max: f32,
    pub observed_mean: f32,
    pub relation_counts: EffectRelationCountsV1,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InterventionEffectPanelV1 {
    pub panel_version: String,
    pub source_evidence_version: String,
    pub sample_count: usize,
    pub movement_epsilon: f32,
    pub dominant_requested_channel: Option<OutcomeChannel>,
    pub channels: Vec<ChannelEffectPanelV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum InterventionEffectPanelError {
    EmptyPanel,
    InvalidSample(usize),
    EvidenceVersionDrift(usize),
    MovementEpsilonDrift(usize),
    DominantRequestDrift(usize),
    ChannelLayoutDrift(usize),
    RequestDefinitionDrift {
        sample_index: usize,
        channel: OutcomeChannel,
    },
}

/// Summarize repeated measurements of one exact requested intervention target.
///
/// The output is descriptive: observed min/max/mean and categorical relation
/// counts per channel. It does not rank the intervention or convert effects into
/// one score.
pub fn summarize_intervention_effect_panel(
    records: &[InterventionEffectEvidenceV1],
) -> Result<InterventionEffectPanelV1, InterventionEffectPanelError> {
    let Some(first) = records.first() else {
        return Err(InterventionEffectPanelError::EmptyPanel);
    };
    validate_record(first, 0)?;

    for (index, record) in records.iter().enumerate().skip(1) {
        validate_record(record, index)?;
        if record.evidence_version != first.evidence_version {
            return Err(InterventionEffectPanelError::EvidenceVersionDrift(index));
        }
        if record.movement_epsilon.to_bits() != first.movement_epsilon.to_bits() {
            return Err(InterventionEffectPanelError::MovementEpsilonDrift(index));
        }
        if record.dominant_requested_channel != first.dominant_requested_channel {
            return Err(InterventionEffectPanelError::DominantRequestDrift(index));
        }
        if record.channels.len() != first.channels.len() {
            return Err(InterventionEffectPanelError::ChannelLayoutDrift(index));
        }
        for (template, candidate) in first.channels.iter().zip(&record.channels) {
            if template.channel != candidate.channel {
                return Err(InterventionEffectPanelError::ChannelLayoutDrift(index));
            }
            if template.request_role != candidate.request_role
                || template.requested_delta.to_bits() != candidate.requested_delta.to_bits()
            {
                return Err(InterventionEffectPanelError::RequestDefinitionDrift {
                    sample_index: index,
                    channel: template.channel,
                });
            }
        }
    }

    let mut channels = Vec::with_capacity(first.channels.len());
    for (channel_index, template) in first.channels.iter().enumerate() {
        let mut observed_min = f32::INFINITY;
        let mut observed_max = f32::NEG_INFINITY;
        let mut observed_sum = 0.0_f64;
        let mut relation_counts = EffectRelationCountsV1::default();

        for record in records {
            let sample = &record.channels[channel_index];
            observed_min = observed_min.min(sample.observed_delta);
            observed_max = observed_max.max(sample.observed_delta);
            observed_sum += sample.observed_delta as f64;
            relation_counts.observe(sample.relation);
        }

        channels.push(ChannelEffectPanelV1 {
            channel: template.channel,
            request_role: template.request_role,
            requested_delta: template.requested_delta,
            samples: records.len(),
            observed_min,
            observed_max,
            observed_mean: (observed_sum / records.len() as f64) as f32,
            relation_counts,
        });
    }

    Ok(InterventionEffectPanelV1 {
        panel_version: INTERVENTION_EFFECT_PANEL_VERSION.into(),
        source_evidence_version: first.evidence_version.clone(),
        sample_count: records.len(),
        movement_epsilon: first.movement_epsilon,
        dominant_requested_channel: first.dominant_requested_channel,
        channels,
    })
}

fn validate_record(
    record: &InterventionEffectEvidenceV1,
    index: usize,
) -> Result<(), InterventionEffectPanelError> {
    if record.disposition != InterventionEffectEvidenceDisposition::Valid {
        return Err(InterventionEffectPanelError::InvalidSample(index));
    }
    if record.evidence_version != INTERVENTION_EFFECT_EVIDENCE_VERSION {
        return Err(InterventionEffectPanelError::EvidenceVersionDrift(index));
    }
    if !record.movement_epsilon.is_finite() || record.movement_epsilon < 0.0 {
        return Err(InterventionEffectPanelError::InvalidSample(index));
    }
    if record.channels.is_empty()
        || record
            .channels
            .iter()
            .any(|channel| !channel.requested_delta.is_finite() || !channel.observed_delta.is_finite())
    {
        return Err(InterventionEffectPanelError::InvalidSample(index));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cognitive_bridge::{ObservedMusicalOutcome, PredictedMusicalOutcome};
    use crate::evidence_digest::intervention_effect_evidence::account_intervention_effects;

    fn target() -> PredictedMusicalOutcome {
        PredictedMusicalOutcome {
            tension_delta: 0.15,
            density_delta: 0.35,
            familiarity_delta: 0.0,
            tonal_displacement_delta: 0.0,
        }
    }

    fn record(tension: f32, density: f32) -> InterventionEffectEvidenceV1 {
        account_intervention_effects(
            target(),
            ObservedMusicalOutcome {
                tension_delta: tension,
                density_delta: density,
                familiarity_delta: 0.0,
                tonal_displacement_delta: 0.0,
            },
            0.01,
        )
    }

    fn channel(panel: &InterventionEffectPanelV1, wanted: OutcomeChannel) -> &ChannelEffectPanelV1 {
        panel
            .channels
            .iter()
            .find(|channel| channel.channel == wanted)
            .unwrap()
    }

    #[test]
    fn panel_retains_quantitative_range_and_relation_counts() {
        let panel = summarize_intervention_effect_panel(&[
            record(-0.02, 0.10),
            record(0.03, 0.20),
        ])
        .unwrap();

        assert_eq!(panel.sample_count, 2);
        assert_eq!(panel.dominant_requested_channel, Some(OutcomeChannel::Density));

        let density = channel(&panel, OutcomeChannel::Density);
        assert_eq!(density.request_role, RequestRoleV1::Dominant);
        assert_eq!(density.observed_min, 0.10);
        assert_eq!(density.observed_max, 0.20);
        assert!((density.observed_mean - 0.15).abs() <= f32::EPSILON);
        assert_eq!(density.relation_counts.requested_aligned, 2);
        assert_eq!(density.relation_counts.total(), 2);

        let tension = channel(&panel, OutcomeChannel::Tension);
        assert_eq!(tension.request_role, RequestRoleV1::Secondary);
        assert_eq!(tension.relation_counts.requested_aligned, 1);
        assert_eq!(tension.relation_counts.requested_opposed, 1);
        assert_eq!(tension.relation_counts.total(), 2);

        let familiarity = channel(&panel, OutcomeChannel::Familiarity);
        assert_eq!(familiarity.request_role, RequestRoleV1::Unrequested);
        assert_eq!(familiarity.relation_counts.unrequested_preserved, 2);
    }

    #[test]
    fn panel_rejects_request_definition_drift() {
        let first = record(0.01, 0.10);
        let mut second = record(0.02, 0.12);
        let density = second
            .channels
            .iter_mut()
            .find(|channel| channel.channel == OutcomeChannel::Density)
            .unwrap();
        density.request_role = RequestRoleV1::Secondary;

        assert_eq!(
            summarize_intervention_effect_panel(&[first, second]),
            Err(InterventionEffectPanelError::RequestDefinitionDrift {
                sample_index: 1,
                channel: OutcomeChannel::Density,
            })
        );
    }

    #[test]
    fn panel_rejects_invalid_samples() {
        let valid = record(0.01, 0.10);
        let invalid = account_intervention_effects(
            PredictedMusicalOutcome {
                tension_delta: f32::NAN,
                ..target()
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
            summarize_intervention_effect_panel(&[valid, invalid]),
            Err(InterventionEffectPanelError::InvalidSample(1))
        );
    }
}
