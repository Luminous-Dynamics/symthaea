// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Non-breaking, machine-readable identity map for Muse's current instrument
//! timbre enum.
//!
//! This is an ontology *index*, not a playable-range or technique certificate.
//! In particular, a GM program, enum variant, partial table, or available
//! synthesis path does not prove that every register, articulation, setup, or
//! renderer combination is supported. Those claims belong in future
//! evidence-backed capability cells.

use crate::instruments::Instrument;

/// Version of the descriptor schema and stable-ID mapping.
pub const INSTRUMENT_ONTOLOGY_VERSION: u16 = 1;
/// Number of variants in the current public `Instrument` enum at this schema
/// version. Tests deliberately fail when the enum and this list diverge.
pub const CURRENT_INSTRUMENT_VARIANT_COUNT: usize = 42;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum InstrumentEntityKind {
    /// A named physical instrument identity.
    PhysicalInstrument,
    /// A timbral/dynamic/setup variant of another instrument identity.
    TimbreVariant,
    /// A section or ensemble realization, not a solo physical instrument.
    EnsembleConfiguration,
    /// A voice realization distinguished by vowel/phonation.
    VocalRealization,
    /// An explicitly electronic source or synthesis category.
    ElectronicSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum InstrumentFamily {
    Keyboard,
    BowedStrings,
    PluckedStrings,
    Woodwind,
    Brass,
    FreeReed,
    Idiophone,
    Membranophone,
    Voice,
    Electronic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PrimaryRealizationPath {
    AdditivePartials,
    KarplusStrong,
    FrequencyModulation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PartialTableEvidence {
    /// The source file explicitly identifies this table as measured.
    Measured,
    /// The source file documents derivation from another table.
    Derived,
    /// A physically motivated or idealized approximation, not a measured
    /// instrument-specific partial table.
    Approximation,
    /// The partial table is shared/fallback data; the primary synthesis path
    /// may be FM or Karplus-Strong and must not be inferred from this table.
    SharedFallback,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityQualification {
    /// This initial identity map does not yet qualify register, technique,
    /// articulation, setup, dynamic, or renderer-specific capability.
    NotEvaluated,
}

/// First-pass classification of one existing `Instrument` enum member.
///
/// `gm_program` is an export mapping only. It is not an identity assertion,
/// timbral-fidelity claim, range certificate, or evidence that a renderer
/// supports every technique.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct InstrumentDescriptor {
    pub id: &'static str,
    pub instrument: Instrument,
    pub entity_kind: InstrumentEntityKind,
    pub family: InstrumentFamily,
    pub primary_realization: PrimaryRealizationPath,
    pub partial_table_evidence: PartialTableEvidence,
    pub gm_program: u8,
    pub capability_qualification: CapabilityQualification,
}

macro_rules! descriptor {
    ($variant:ident, $id:literal, $kind:ident, $family:ident, $partials:ident) => {{
        let instrument = Instrument::$variant;
        InstrumentDescriptor {
            id: $id,
            instrument,
            entity_kind: InstrumentEntityKind::$kind,
            family: InstrumentFamily::$family,
            primary_realization: if instrument.uses_karplus_strong() {
                PrimaryRealizationPath::KarplusStrong
            } else if instrument.uses_fm() {
                PrimaryRealizationPath::FrequencyModulation
            } else {
                PrimaryRealizationPath::AdditivePartials
            },
            partial_table_evidence: PartialTableEvidence::$partials,
            gm_program: instrument.gm_program(),
            capability_qualification: CapabilityQualification::NotEvaluated,
        }
    }};
}

/// The complete, deterministic descriptor set for the current 42 variants.
///
/// Stable IDs are independent of display names and General MIDI program
/// numbers. Additions require an explicit schema/count update and a coverage
/// test change; do not silently infer descriptors from Rust variant names.
pub fn instrument_descriptors() -> [InstrumentDescriptor; CURRENT_INSTRUMENT_VARIANT_COUNT] {
    [
        descriptor!(Piano, "piano", PhysicalInstrument, Keyboard, Measured),
        descriptor!(PianoPP, "piano.pp", TimbreVariant, Keyboard, Measured),
        descriptor!(Violin, "violin", PhysicalInstrument, BowedStrings, Measured),
        descriptor!(Cello, "cello", PhysicalInstrument, BowedStrings, Measured),
        descriptor!(Flute, "flute", PhysicalInstrument, Woodwind, Measured),
        descriptor!(Organ, "organ", PhysicalInstrument, Keyboard, Measured),
        descriptor!(ElectricPiano, "electric_piano", PhysicalInstrument, Keyboard, SharedFallback),
        descriptor!(Bell, "bell", PhysicalInstrument, Idiophone, SharedFallback),
        descriptor!(AcousticGuitar, "acoustic_guitar", PhysicalInstrument, PluckedStrings, SharedFallback),
        descriptor!(Harp, "harp", PhysicalInstrument, PluckedStrings, SharedFallback),
        descriptor!(Pad, "pad", ElectronicSource, Electronic, SharedFallback),
        descriptor!(Clarinet, "clarinet", PhysicalInstrument, Woodwind, Measured),
        descriptor!(Trumpet, "trumpet", PhysicalInstrument, Brass, Measured),
        descriptor!(Saxophone, "saxophone", PhysicalInstrument, Woodwind, Measured),
        descriptor!(Marimba, "marimba", PhysicalInstrument, Idiophone, SharedFallback),
        descriptor!(Sitar, "sitar", PhysicalInstrument, PluckedStrings, Measured),
        descriptor!(Kalimba, "kalimba", PhysicalInstrument, Idiophone, SharedFallback),
        descriptor!(SawLead, "saw_lead", ElectronicSource, Electronic, Approximation),
        descriptor!(Koto, "koto", PhysicalInstrument, PluckedStrings, SharedFallback),
        descriptor!(Oud, "oud", PhysicalInstrument, PluckedStrings, SharedFallback),
        descriptor!(Ney, "ney", PhysicalInstrument, Woodwind, SharedFallback),
        descriptor!(UprightBass, "upright_bass", PhysicalInstrument, PluckedStrings, SharedFallback),
        descriptor!(Viola, "viola", PhysicalInstrument, BowedStrings, Derived),
        descriptor!(Oboe, "oboe", PhysicalInstrument, Woodwind, Approximation),
        descriptor!(FrenchHorn, "french_horn", PhysicalInstrument, Brass, Approximation),
        descriptor!(Bassoon, "bassoon", PhysicalInstrument, Woodwind, Approximation),
        descriptor!(Accordion, "accordion", PhysicalInstrument, FreeReed, Derived),
        descriptor!(Mandolin, "mandolin", PhysicalInstrument, PluckedStrings, SharedFallback),
        descriptor!(Banjo, "banjo", PhysicalInstrument, PluckedStrings, SharedFallback),
        descriptor!(Vibraphone, "vibraphone", PhysicalInstrument, Idiophone, SharedFallback),
        descriptor!(Celesta, "celesta", PhysicalInstrument, Idiophone, SharedFallback),
        descriptor!(ChoirAah, "choir.aah", VocalRealization, Voice, Approximation),
        descriptor!(StringEnsemble, "strings.ensemble", EnsembleConfiguration, BowedStrings, Derived),
        descriptor!(ElectricBass, "electric_bass", PhysicalInstrument, PluckedStrings, Derived),
        descriptor!(Timpani, "timpani", PhysicalInstrument, Membranophone, Approximation),
        descriptor!(MusicBox, "music_box", PhysicalInstrument, Idiophone, SharedFallback),
        descriptor!(Harpsichord, "harpsichord", PhysicalInstrument, Keyboard, Approximation),
        descriptor!(Tuba, "tuba", PhysicalInstrument, Brass, Approximation),
        descriptor!(Trombone, "trombone", PhysicalInstrument, Brass, Derived),
        descriptor!(Duduk, "duduk", PhysicalInstrument, Woodwind, Derived),
        descriptor!(ChoirOoh, "choir.ooh", VocalRealization, Voice, Derived),
        descriptor!(Xylophone, "xylophone", PhysicalInstrument, Idiophone, Approximation),
    ]
}

/// Find the current descriptor for an enum variant. Returns `None` rather
/// than silently manufacturing an identity for a future unmapped variant.
pub fn descriptor_for(instrument: Instrument) -> Option<InstrumentDescriptor> {
    instrument_descriptors()
        .into_iter()
        .find(|entry| entry.instrument == instrument)
}

/// Validate catalogue identity invariants. This does not validate instrument
/// performance capability; that requires evidence-backed capability cells.
pub fn validate_instrument_catalogue() -> Result<(), Vec<String>> {
    let entries = instrument_descriptors();
    let mut issues = Vec::new();

    for (index, entry) in entries.iter().enumerate() {
        if entry.id.trim().is_empty() {
            issues.push(format!("descriptor at index {index} has an empty stable ID"));
        }
        if entries[..index].iter().any(|earlier| earlier.id == entry.id) {
            issues.push(format!("duplicate instrument stable ID: {}", entry.id));
        }
        if entries[..index]
            .iter()
            .any(|earlier| earlier.instrument == entry.instrument)
        {
            issues.push(format!("duplicate enum mapping for stable ID: {}", entry.id));
        }
        let expected_path = if entry.instrument.uses_karplus_strong() {
            PrimaryRealizationPath::KarplusStrong
        } else if entry.instrument.uses_fm() {
            PrimaryRealizationPath::FrequencyModulation
        } else {
            PrimaryRealizationPath::AdditivePartials
        };
        if entry.primary_realization != expected_path {
            issues.push(format!("realization path mismatch for {}", entry.id));
        }
        if entry.gm_program != entry.instrument.gm_program() {
            issues.push(format!("GM export mapping mismatch for {}", entry.id));
        }
        if entry.capability_qualification != CapabilityQualification::NotEvaluated {
            issues.push(format!(
                "identity-only catalogue must not promote capability for {}",
                entry.id
            ));
        }
    }

    if issues.is_empty() { Ok(()) } else { Err(issues) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_current_instrument_variants_have_one_stable_descriptor() {
        let entries = instrument_descriptors();
        assert_eq!(entries.len(), CURRENT_INSTRUMENT_VARIANT_COUNT);

        let expected = [
            Instrument::Piano, Instrument::PianoPP, Instrument::Violin, Instrument::Cello,
            Instrument::Flute, Instrument::Organ, Instrument::ElectricPiano, Instrument::Bell,
            Instrument::AcousticGuitar, Instrument::Harp, Instrument::Pad, Instrument::Clarinet,
            Instrument::Trumpet, Instrument::Saxophone, Instrument::Marimba, Instrument::Sitar,
            Instrument::Kalimba, Instrument::SawLead, Instrument::Koto, Instrument::Oud,
            Instrument::Ney, Instrument::UprightBass, Instrument::Viola, Instrument::Oboe,
            Instrument::FrenchHorn, Instrument::Bassoon, Instrument::Accordion,
            Instrument::Mandolin, Instrument::Banjo, Instrument::Vibraphone, Instrument::Celesta,
            Instrument::ChoirAah, Instrument::StringEnsemble, Instrument::ElectricBass,
            Instrument::Timpani, Instrument::MusicBox, Instrument::Harpsichord, Instrument::Tuba,
            Instrument::Trombone, Instrument::Duduk, Instrument::ChoirOoh, Instrument::Xylophone,
        ];

        for instrument in expected {
            assert_eq!(
                entries.iter().filter(|entry| entry.instrument == instrument).count(),
                1,
                "each enum variant must map exactly once: {instrument:?}"
            );
            assert!(descriptor_for(instrument).is_some());
        }
    }

    #[test]
    fn stable_ids_are_unique_and_catalogue_invariants_hold() {
        validate_instrument_catalogue().expect("instrument identity map must be internally valid");
    }

    #[test]
    fn ontology_does_not_promote_shared_presets_to_physical_instruments() {
        let piano_pp = descriptor_for(Instrument::PianoPP).unwrap();
        assert_eq!(piano_pp.entity_kind, InstrumentEntityKind::TimbreVariant);

        let strings = descriptor_for(Instrument::StringEnsemble).unwrap();
        assert_eq!(strings.entity_kind, InstrumentEntityKind::EnsembleConfiguration);

        for instrument in [Instrument::ChoirAah, Instrument::ChoirOoh] {
            assert_eq!(
                descriptor_for(instrument).unwrap().entity_kind,
                InstrumentEntityKind::VocalRealization
            );
        }

        for instrument in [Instrument::Pad, Instrument::SawLead] {
            assert_eq!(
                descriptor_for(instrument).unwrap().entity_kind,
                InstrumentEntityKind::ElectronicSource
            );
        }
    }

    #[test]
    fn gm_program_and_synthesis_path_are_only_mappings_not_qualification() {
        let oud = descriptor_for(Instrument::Oud).unwrap();
        assert_eq!(oud.gm_program, Instrument::Oud.gm_program());
        assert_eq!(oud.primary_realization, PrimaryRealizationPath::KarplusStrong);
        assert_eq!(oud.capability_qualification, CapabilityQualification::NotEvaluated);

        let xylophone = descriptor_for(Instrument::Xylophone).unwrap();
        assert_eq!(xylophone.primary_realization, PrimaryRealizationPath::FrequencyModulation);
        assert_eq!(xylophone.partial_table_evidence, PartialTableEvidence::Approximation);
        assert_eq!(xylophone.capability_qualification, CapabilityQualification::NotEvaluated);
    }
}
