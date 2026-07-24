// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Stable descriptors for parameterized symbolic interventions.
//!
//! An action name such as `ReturnOpeningMaterial` is too coarse for outcome
//! learning: a literal return, an octave displacement, and a rhythmic
//! augmentation have different effects. These descriptors retain the exact
//! strategy and bounded baseline features without storing notes in the model.

use crate::cognitive_bridge::{CognitiveSection, SymbolicAction};
use serde::{Deserialize, Serialize};

pub const INTERVENTION_DESCRIPTOR_VERSION: &str = "intervention-descriptor-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InterventionStrategy {
    KeepCurrent,
    Literal,
    RegisterShift,
    ContourPreserving,
    Inversion,
    Augmentation,
    Diminution,
    Fragmentation,
    Restoration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ObligationClass {
    None,
    ReturnMotif,
    ReachKey,
    RestoreIdentity,
    ResolveAlteredDegree,
    Cadence,
    Climax,
    VoiceEntry,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InterventionDescriptor {
    pub descriptor_version: String,
    pub action: SymbolicAction,
    pub strategy: InterventionStrategy,
    pub source_section: CognitiveSection,
    pub target_section: CognitiveSection,
    pub target_obligation: ObligationClass,
    pub pitch_shift_semitones: i8,
    /// Rhythmic scale in thousandths: 1000 = unchanged, 2000 = augmented.
    pub rhythm_scale_milli: u16,
    /// Intended transformation strength in thousandths.
    pub transformation_strength_milli: u16,
    /// Baseline thematic identity bucket in [0, 20].
    pub baseline_motif_similarity_bucket: u8,
    /// Baseline tension bucket in [0, 20].
    pub baseline_tension_bucket: u8,
    /// Baseline density bucket in [0, 20].
    pub baseline_density_bucket: u8,
    pub affected_note_count: u32,
    /// Fraction of the complete score affected, in thousandths.
    pub affected_score_fraction_milli: u16,
}

impl InterventionDescriptor {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        action: SymbolicAction,
        strategy: InterventionStrategy,
        source_section: CognitiveSection,
        target_section: CognitiveSection,
        target_obligation: ObligationClass,
        pitch_shift_semitones: i8,
        rhythm_scale: f32,
        transformation_strength: f32,
        baseline_motif_similarity: f32,
        baseline_tension: f32,
        baseline_density: f32,
        affected_note_count: usize,
        score_note_count: usize,
    ) -> Self {
        Self {
            descriptor_version: INTERVENTION_DESCRIPTOR_VERSION.into(),
            action,
            strategy,
            source_section,
            target_section,
            target_obligation,
            pitch_shift_semitones,
            rhythm_scale_milli: milli(rhythm_scale, 4.0),
            transformation_strength_milli: milli(transformation_strength, 1.0),
            baseline_motif_similarity_bucket: bucket(baseline_motif_similarity),
            baseline_tension_bucket: bucket(baseline_tension),
            baseline_density_bucket: bucket(baseline_density),
            affected_note_count: affected_note_count.min(u32::MAX as usize) as u32,
            affected_score_fraction_milli: if score_note_count == 0 {
                0
            } else {
                ((affected_note_count as f64 / score_note_count as f64) * 1000.0)
                    .round()
                    .clamp(0.0, 1000.0) as u16
            },
        }
    }

    pub fn is_compatible(&self) -> bool {
        self.descriptor_version == INTERVENTION_DESCRIPTOR_VERSION
            && self.rhythm_scale_milli <= 4000
            && self.transformation_strength_milli <= 1000
            && self.baseline_motif_similarity_bucket <= 20
            && self.baseline_tension_bucket <= 20
            && self.baseline_density_bucket <= 20
            && self.affected_score_fraction_milli <= 1000
    }
}

fn bucket(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 20.0).round() as u8
}

fn milli(value: f32, max: f32) -> u16 {
    (value.clamp(0.0, max) * 1000.0).round() as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_quantizes_context_without_hiding_strategy() {
        let descriptor = InterventionDescriptor::new(
            SymbolicAction::ReturnOpeningMaterial,
            InterventionStrategy::RegisterShift,
            CognitiveSection::Exposition,
            CognitiveSection::Recapitulation,
            ObligationClass::ReturnMotif,
            12,
            1.0,
            0.6,
            0.94,
            0.41,
            0.22,
            8,
            80,
        );
        assert_eq!(descriptor.pitch_shift_semitones, 12);
        assert_eq!(descriptor.baseline_motif_similarity_bucket, 19);
        assert_eq!(descriptor.affected_score_fraction_milli, 100);
        assert!(descriptor.is_compatible());
    }
}
