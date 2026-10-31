// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evidence-bounded inventory of instrument models currently implemented by Muse.
//!
//! The module exposes facts already represented by Instrument methods. It
//! deliberately does not guess playable ranges, sample coverage, articulation
//! fidelity or SoundFont controller support. Those are renderer/asset-specific
//! and remain unqualified until measured against exact assets and builds.

use crate::instruments::Instrument;
use serde::{Deserialize, Serialize};

/// Broad acoustic/electronic family. This is a taxonomy for orchestration and
/// reporting, not a claim that every member has identical playing technique.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InstrumentFamily {
    Keyboard,
    BowedStrings,
    PluckedStrings,
    BassStrings,
    Woodwinds,
    Brass,
    FreeReed,
    StruckIdiophones,
    PluckedIdiophones,
    MembranePercussion,
    Voice,
    Electronic,
}

/// Native Muse synthesis path. Separate from the General MIDI program used
/// when a score is exported to an external renderer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeSynthesisPath {
    Additive,
    FrequencyModulation,
    KarplusStrong,
}

/// What current source code establishes, not external perceptual validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QualificationStatus {
    /// Behavior is directly implemented in current Muse source.
    CodeDeclared,
    /// No instrument/renderer qualification has been supplied yet.
    NotQualified,
    /// Coverage is not known from the instrument model alone.
    Unknown,
}

/// Semantic status of an exported GM program mapping. A distinct number does
/// not mean an external SoundFont contains a faithful sample patch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GmProgramMapping {
    DedicatedProgramNumber,
    NearestFamilyApproximation,
    IntentionalSharedProgram,
}

/// Attack component already exposed by Instrument::attack_noise.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AttackTransientProfile {
    pub amplitude: f32,
    pub duration_seconds: f32,
}

/// Vibrato parameterization already exposed by Instrument::vibrato.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VibratoProfile {
    pub rate_hz: f32,
    pub depth_ratio: f32,
    pub onset_delay_seconds: f32,
}

/// Conservative projection of current in-code instrument behavior.
///
/// playable_midi_range is intentionally None for every v1 record. The current
/// enum/model does not establish manufacturer- or technique-specific playable
/// ranges or tessituras. A future range profile must carry a source, evidence
/// class and renderer/technique scope.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct InstrumentCapabilityProfile {
    pub instrument: Instrument,
    pub family: InstrumentFamily,
    pub native_synthesis: NativeSynthesisPath,
    pub model_behavior_status: QualificationStatus,
    pub sustains_while_held: bool,
    pub attack_transient: Option<AttackTransientProfile>,
    pub vibrato: Option<VibratoProfile>,
    pub gm_program: u8,
    pub gm_mapping: GmProgramMapping,
    pub playable_midi_range: Option<(u8, u8)>,
    pub range_status: QualificationStatus,
    pub articulation_status: QualificationStatus,
    pub renderer_controller_status: QualificationStatus,
    /// Runtime asset coverage is unknown until the optional VCSL/VSCO2 banks
    /// are indexed and probed. A code-declared path is not proof that every
    /// pitch, velocity layer or articulation is installed and playable.
    pub sample_coverage_status: QualificationStatus,
}

/// Canonical enumeration of the 42 variants in the current Instrument enum.
/// The coverage test makes additions require an explicit inventory update.
pub const INSTRUMENT_INVENTORY: [Instrument; 42] = [
    Instrument::Piano,
    Instrument::PianoPP,
    Instrument::Violin,
    Instrument::Cello,
    Instrument::Flute,
    Instrument::Organ,
    Instrument::ElectricPiano,
    Instrument::Bell,
    Instrument::AcousticGuitar,
    Instrument::Harp,
    Instrument::Pad,
    Instrument::Clarinet,
    Instrument::Trumpet,
    Instrument::Saxophone,
    Instrument::Marimba,
    Instrument::Sitar,
    Instrument::Kalimba,
    Instrument::SawLead,
    Instrument::Koto,
    Instrument::Oud,
    Instrument::Ney,
    Instrument::UprightBass,
    Instrument::Viola,
    Instrument::Oboe,
    Instrument::FrenchHorn,
    Instrument::Bassoon,
    Instrument::Accordion,
    Instrument::Mandolin,
    Instrument::Banjo,
    Instrument::Vibraphone,
    Instrument::Celesta,
    Instrument::ChoirAah,
    Instrument::StringEnsemble,
    Instrument::ElectricBass,
    Instrument::Timpani,
    Instrument::MusicBox,
    Instrument::Harpsichord,
    Instrument::Tuba,
    Instrument::Trombone,
    Instrument::Duduk,
    Instrument::ChoirOoh,
    Instrument::Xylophone,
];

/// Classify an instrument using the current Muse enum vocabulary.
pub const fn family(instrument: Instrument) -> InstrumentFamily {
    use Instrument as I;
    match instrument {
        I::Piano | I::PianoPP | I::ElectricPiano | I::Organ | I::Harpsichord => {
            InstrumentFamily::Keyboard
        }
        I::Violin | I::Viola | I::Cello | I::StringEnsemble => {
            InstrumentFamily::BowedStrings
        }
        I::AcousticGuitar
        | I::Harp
        | I::Sitar
        | I::Koto
        | I::Oud
        | I::Mandolin
        | I::Banjo => InstrumentFamily::PluckedStrings,
        I::UprightBass | I::ElectricBass => InstrumentFamily::BassStrings,
        I::Flute | I::Clarinet | I::Oboe | I::Bassoon | I::Ney | I::Duduk => {
            InstrumentFamily::Woodwinds
        }
        I::Trumpet | I::FrenchHorn | I::Trombone | I::Tuba => InstrumentFamily::Brass,
        I::Accordion => InstrumentFamily::FreeReed,
        I::Bell | I::Marimba | I::Vibraphone | I::Celesta | I::Xylophone => {
            InstrumentFamily::StruckIdiophones
        }
        I::Kalimba | I::MusicBox => InstrumentFamily::PluckedIdiophones,
        I::Timpani => InstrumentFamily::MembranePercussion,
        I::ChoirAah | I::ChoirOoh => InstrumentFamily::Voice,
        I::Pad | I::SawLead => InstrumentFamily::Electronic,
    }
}

fn gm_mapping(instrument: Instrument) -> GmProgramMapping {
    match instrument {
        Instrument::PianoPP => GmProgramMapping::IntentionalSharedProgram,
        Instrument::Oud | Instrument::Ney | Instrument::Mandolin | Instrument::Duduk => {
            GmProgramMapping::NearestFamilyApproximation
        }
        _ => GmProgramMapping::DedicatedProgramNumber,
    }
}

/// Build one profile from the instrument's authoritative current methods.
/// This records implementation facts only; it does not perform acoustic,
/// sample-bank or external-renderer qualification.
pub fn capability_profile(instrument: Instrument) -> InstrumentCapabilityProfile {
    let native_synthesis = if instrument.uses_karplus_strong() {
        NativeSynthesisPath::KarplusStrong
    } else if instrument.uses_fm() {
        NativeSynthesisPath::FrequencyModulation
    } else {
        NativeSynthesisPath::Additive
    };

    let (attack_amplitude, attack_duration) = instrument.attack_noise();
    let attack_transient = if attack_amplitude > 0.0 && attack_duration > 0.0 {
        Some(AttackTransientProfile {
            amplitude: attack_amplitude,
            duration_seconds: attack_duration,
        })
    } else {
        None
    };

    let vibrato = instrument.vibrato().map(|(rate_hz, depth_ratio, onset_delay_seconds)| {
        VibratoProfile {
            rate_hz,
            depth_ratio,
            onset_delay_seconds,
        }
    });

    InstrumentCapabilityProfile {
        instrument,
        family: family(instrument),
        native_synthesis,
        model_behavior_status: QualificationStatus::CodeDeclared,
        sustains_while_held: instrument.sustains(),
        attack_transient,
        vibrato,
        gm_program: instrument.gm_program(),
        gm_mapping: gm_mapping(instrument),
        playable_midi_range: None,
        range_status: QualificationStatus::NotQualified,
        articulation_status: QualificationStatus::NotQualified,
        renderer_controller_status: QualificationStatus::NotQualified,
        sample_coverage_status: QualificationStatus::Unknown,
    }
}

/// Return profiles for the full current instrument vocabulary in a stable
/// order. The resulting values are suitable for JSON serialization and audits.
pub fn inventory() -> Vec<InstrumentCapabilityProfile> {
    INSTRUMENT_INVENTORY
        .iter()
        .copied()
        .map(capability_profile)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn inventory_contains_42_unique_variants() {
        assert_eq!(INSTRUMENT_INVENTORY.len(), 42);
        let unique: HashSet<_> = INSTRUMENT_INVENTORY.iter().copied().collect();
        assert_eq!(unique.len(), INSTRUMENT_INVENTORY.len());
    }

    #[test]
    fn profile_projection_matches_authoritative_instrument_methods() {
        for instrument in INSTRUMENT_INVENTORY {
            let profile = capability_profile(instrument);
            assert_eq!(profile.family, family(instrument));
            assert_eq!(profile.model_behavior_status, QualificationStatus::CodeDeclared);
            assert_eq!(profile.sustains_while_held, instrument.sustains());
            assert_eq!(profile.gm_program, instrument.gm_program());
            assert_eq!(
                profile.native_synthesis,
                if instrument.uses_karplus_strong() {
                    NativeSynthesisPath::KarplusStrong
                } else if instrument.uses_fm() {
                    NativeSynthesisPath::FrequencyModulation
                } else {
                    NativeSynthesisPath::Additive
                }
            );

            let (attack_amplitude, attack_duration) = instrument.attack_noise();
            assert_eq!(
                profile.attack_transient.is_some(),
                attack_amplitude > 0.0 && attack_duration > 0.0
            );

            let vibrato = instrument.vibrato();
            assert_eq!(profile.vibrato.is_some(), vibrato.is_some());
            if let (Some(actual), Some(expected)) = (profile.vibrato, vibrato) {
                assert_eq!(actual.rate_hz, expected.0);
                assert_eq!(actual.depth_ratio, expected.1);
                assert_eq!(actual.onset_delay_seconds, expected.2);
            }
        }
    }

    #[test]
    fn unqualified_playability_and_renderer_claims_remain_unknown() {
        for profile in inventory() {
            assert_eq!(profile.playable_midi_range, None);
            assert_eq!(profile.range_status, QualificationStatus::NotQualified);
            assert_eq!(profile.articulation_status, QualificationStatus::NotQualified);
            assert_eq!(
                profile.renderer_controller_status,
                QualificationStatus::NotQualified
            );
            assert_eq!(profile.sample_coverage_status, QualificationStatus::Unknown);
        }
    }

    #[test]
    fn documented_gm_approximations_and_shared_program_are_explicit() {
        assert_eq!(
            capability_profile(Instrument::PianoPP).gm_mapping,
            GmProgramMapping::IntentionalSharedProgram
        );
        for instrument in [
            Instrument::Oud,
            Instrument::Ney,
            Instrument::Mandolin,
            Instrument::Duduk,
        ] {
            assert_eq!(
                capability_profile(instrument).gm_mapping,
                GmProgramMapping::NearestFamilyApproximation
            );
        }
    }

    #[test]
    fn all_families_are_non_empty() {
        let families: HashSet<_> = inventory().into_iter().map(|p| p.family).collect();
        assert_eq!(families.len(), 12);
    }
}
