// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Source-native instrument identity catalog for pinned VCSL paths.
//!
//! This catalog is intentionally separate from Muse's procedural Instrument
//! enum, named-style renderer selection, and local playable-bank status.
//! Every record is only catalogued from an upstream directory entry. No local
//! assets are implied, no renderer alias is assigned, and no path counts as
//! decoded or playable merely because the source directory exists.

use serde::{Deserialize, Serialize};

pub const VCSL_TREE_REVISION_V1: &str = "c1ea7bcc3c7309650ab0da9d15c9cd1fbc4a4c7e";
pub const VCSL_LICENSE_V1: &str = "CC0-1.0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceFamilyV1 {
    Chordophone,
    EdgeBlownAerophone,
    FreeReedAerophone,
    Membranophone,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PitchSemanticsV1 {
    PitchedSource,
    /// The source path identifies a drum/sample set; it is not a verified MIDI key map.
    PercussionDrumIdentifierOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SampleRoleV1 {
    Sustain,
    ShortArticulation,
    ReleaseEvent,
    AuxiliaryEvent,
    PercussionHit,
    PercussionRoll,
    PercussionMallet,
    PercussionStick,
    TunedDrumVariant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventLaneV1 {
    NoteOn,
    Release,
    Auxiliary,
    Percussion,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlayabilityEvidenceV1 {
    CataloguedOnly,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceInstrumentIdentityV1 {
    /// Stable ID derived from the normalized full source-relative path.
    pub record_id: String,
    /// Stable model/instrument ID shared across its articulations.
    pub source_instrument_id: String,
    pub display_name: String,
    pub family: SourceFamilyV1,
    pub pitch_semantics: PitchSemanticsV1,
    pub source_path: String,
    /// Source-native directory label (kept verbatim except canonical ID form).
    pub articulation: String,
    pub sample_role: SampleRoleV1,
    pub event_lane: EventLaneV1,
    pub library: String,
    pub source_tree_revision: String,
    pub license_identifier: String,
    /// None means no procedural substitute is declared by this catalog.
    pub renderer_substitute_id: Option<String>,
    /// This record verifies a path in a pinned source tree, not downloaded audio.
    pub playability_evidence: PlayabilityEvidenceV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceInstrumentIdentityCatalogV1 {
    pub schema: String,
    pub library: String,
    pub source_tree_revision: String,
    pub license_identifier: String,
    pub identity_record_count: usize,
    pub identities: Vec<SourceInstrumentIdentityV1>,
}

#[derive(Debug, Clone, Copy)]
struct IdentitySpec {
    record_id: &'static str,
    source_instrument_id: &'static str,
    display_name: &'static str,
    family: SourceFamilyV1,
    pitch_semantics: PitchSemanticsV1,
    source_path: &'static str,
    articulation: &'static str,
    sample_role: SampleRoleV1,
    event_lane: EventLaneV1,
}

const IDENTITY_SPECS: &[IdentitySpec] = &[
    IdentitySpec { record_id: "vcsl.path.chordophones.zithers.dan-tranh.normal", source_instrument_id: "vcsl.dan-tranh", display_name: "Dan Tranh", family: SourceFamilyV1::Chordophone, pitch_semantics: PitchSemanticsV1::PitchedSource, source_path: "Chordophones/Zithers/Dan Tranh/Normal", articulation: "normal", sample_role: SampleRoleV1::Sustain, event_lane: EventLaneV1::NoteOn },
    IdentitySpec { record_id: "vcsl.path.chordophones.zithers.dan-tranh.tremolo", source_instrument_id: "vcsl.dan-tranh", display_name: "Dan Tranh", family: SourceFamilyV1::Chordophone, pitch_semantics: PitchSemanticsV1::PitchedSource, source_path: "Chordophones/Zithers/Dan Tranh/Tremolo", articulation: "tremolo", sample_role: SampleRoleV1::Sustain, event_lane: EventLaneV1::NoteOn },
    IdentitySpec { record_id: "vcsl.path.chordophones.zithers.dan-tranh.vibrato", source_instrument_id: "vcsl.dan-tranh", display_name: "Dan Tranh", family: SourceFamilyV1::Chordophone, pitch_semantics: PitchSemanticsV1::PitchedSource, source_path: "Chordophones/Zithers/Dan Tranh/Vibrato", articulation: "vibrato", sample_role: SampleRoleV1::Sustain, event_lane: EventLaneV1::NoteOn },
    IdentitySpec { record_id: "vcsl.path.chordophones.zithers.dan-tranh.gliss", source_instrument_id: "vcsl.dan-tranh", display_name: "Dan Tranh", family: SourceFamilyV1::Chordophone, pitch_semantics: PitchSemanticsV1::PitchedSource, source_path: "Chordophones/Zithers/Dan Tranh/Gliss", articulation: "glissando", sample_role: SampleRoleV1::AuxiliaryEvent, event_lane: EventLaneV1::Auxiliary },
    IdentitySpec { record_id: "vcsl.path.chordophones.zithers.dan-tranh.fx", source_instrument_id: "vcsl.dan-tranh", display_name: "Dan Tranh", family: SourceFamilyV1::Chordophone, pitch_semantics: PitchSemanticsV1::PitchedSource, source_path: "Chordophones/Zithers/Dan Tranh/FX", articulation: "effect", sample_role: SampleRoleV1::AuxiliaryEvent, event_lane: EventLaneV1::Auxiliary },
    IdentitySpec { record_id: "vcsl.path.aerophones.edge-blown-aerophones.ocarina-small.sustain", source_instrument_id: "vcsl.ocarina-small", display_name: "Ocarina (Small)", family: SourceFamilyV1::EdgeBlownAerophone, pitch_semantics: PitchSemanticsV1::PitchedSource, source_path: "Aerophones/Edge-blown Aerophones/Ocarina, Small/Sustain", articulation: "sustain", sample_role: SampleRoleV1::Sustain, event_lane: EventLaneV1::NoteOn },
    IdentitySpec { record_id: "vcsl.path.aerophones.edge-blown-aerophones.ocarina-small.staccato", source_instrument_id: "vcsl.ocarina-small", display_name: "Ocarina (Small)", family: SourceFamilyV1::EdgeBlownAerophone, pitch_semantics: PitchSemanticsV1::PitchedSource, source_path: "Aerophones/Edge-blown Aerophones/Ocarina, Small/Staccato", articulation: "staccato", sample_role: SampleRoleV1::ShortArticulation, event_lane: EventLaneV1::NoteOn },
    IdentitySpec { record_id: "vcsl.path.aerophones.edge-blown-aerophones.ocarina-typical.sustains.sus", source_instrument_id: "vcsl.ocarina-typical", display_name: "Ocarina (Typical)", family: SourceFamilyV1::EdgeBlownAerophone, pitch_semantics: PitchSemanticsV1::PitchedSource, source_path: "Aerophones/Edge-blown Aerophones/Ocarina, Typical/Sustains/Sus", articulation: "sustain", sample_role: SampleRoleV1::Sustain, event_lane: EventLaneV1::NoteOn },
    IdentitySpec { record_id: "vcsl.path.aerophones.edge-blown-aerophones.ocarina-typical.sustains.susvib", source_instrument_id: "vcsl.ocarina-typical", display_name: "Ocarina (Typical)", family: SourceFamilyV1::EdgeBlownAerophone, pitch_semantics: PitchSemanticsV1::PitchedSource, source_path: "Aerophones/Edge-blown Aerophones/Ocarina, Typical/Sustains/SusVib", articulation: "sustain_vibrato", sample_role: SampleRoleV1::Sustain, event_lane: EventLaneV1::NoteOn },
    IdentitySpec { record_id: "vcsl.path.aerophones.edge-blown-aerophones.ocarina-typical.releases.sus", source_instrument_id: "vcsl.ocarina-typical", display_name: "Ocarina (Typical)", family: SourceFamilyV1::EdgeBlownAerophone, pitch_semantics: PitchSemanticsV1::PitchedSource, source_path: "Aerophones/Edge-blown Aerophones/Ocarina, Typical/Releases/Sus", articulation: "release_sustain", sample_role: SampleRoleV1::ReleaseEvent, event_lane: EventLaneV1::Release },
    IdentitySpec { record_id: "vcsl.path.aerophones.edge-blown-aerophones.ocarina-typical.releases.susvib", source_instrument_id: "vcsl.ocarina-typical", display_name: "Ocarina (Typical)", family: SourceFamilyV1::EdgeBlownAerophone, pitch_semantics: PitchSemanticsV1::PitchedSource, source_path: "Aerophones/Edge-blown Aerophones/Ocarina, Typical/Releases/SusVib", articulation: "release_sustain_vibrato", sample_role: SampleRoleV1::ReleaseEvent, event_lane: EventLaneV1::Release },
    IdentitySpec { record_id: "vcsl.path.aerophones.free-aerophones.harmonica-hohner-special20-c.sustains.normal", source_instrument_id: "vcsl.harmonica-special20-c", display_name: "Harmonica Hohner Special 20 (C)", family: SourceFamilyV1::FreeReedAerophone, pitch_semantics: PitchSemanticsV1::PitchedSource, source_path: "Aerophones/Free Aerophones/Harmonica-Hohner-Special20-C/Sustains/Normal", articulation: "normal", sample_role: SampleRoleV1::Sustain, event_lane: EventLaneV1::NoteOn },
    IdentitySpec { record_id: "vcsl.path.aerophones.free-aerophones.harmonica-hohner-special20-c.sustains.soft", source_instrument_id: "vcsl.harmonica-special20-c", display_name: "Harmonica Hohner Special 20 (C)", family: SourceFamilyV1::FreeReedAerophone, pitch_semantics: PitchSemanticsV1::PitchedSource, source_path: "Aerophones/Free Aerophones/Harmonica-Hohner-Special20-C/Sustains/Soft", articulation: "soft", sample_role: SampleRoleV1::Sustain, event_lane: EventLaneV1::NoteOn },
    IdentitySpec { record_id: "vcsl.path.aerophones.free-aerophones.harmonica-hohner-special20-c.sustains.vib", source_instrument_id: "vcsl.harmonica-special20-c", display_name: "Harmonica Hohner Special 20 (C)", family: SourceFamilyV1::FreeReedAerophone, pitch_semantics: PitchSemanticsV1::PitchedSource, source_path: "Aerophones/Free Aerophones/Harmonica-Hohner-Special20-C/Sustains/Vib", articulation: "vibrato", sample_role: SampleRoleV1::Sustain, event_lane: EventLaneV1::NoteOn },
    IdentitySpec { record_id: "vcsl.path.aerophones.free-aerophones.harmonica-hohner-special20-c.releases.normal", source_instrument_id: "vcsl.harmonica-special20-c", display_name: "Harmonica Hohner Special 20 (C)", family: SourceFamilyV1::FreeReedAerophone, pitch_semantics: PitchSemanticsV1::PitchedSource, source_path: "Aerophones/Free Aerophones/Harmonica-Hohner-Special20-C/Releases/Normal", articulation: "release_normal", sample_role: SampleRoleV1::ReleaseEvent, event_lane: EventLaneV1::Release },
    IdentitySpec { record_id: "vcsl.path.aerophones.free-aerophones.harmonica-hohner-special20-c.releases.soft", source_instrument_id: "vcsl.harmonica-special20-c", display_name: "Harmonica Hohner Special 20 (C)", family: SourceFamilyV1::FreeReedAerophone, pitch_semantics: PitchSemanticsV1::PitchedSource, source_path: "Aerophones/Free Aerophones/Harmonica-Hohner-Special20-C/Releases/Soft", articulation: "release_soft", sample_role: SampleRoleV1::ReleaseEvent, event_lane: EventLaneV1::Release },
    IdentitySpec { record_id: "vcsl.path.aerophones.free-aerophones.harmonica-hohner-special20-c.releases.vib", source_instrument_id: "vcsl.harmonica-special20-c", display_name: "Harmonica Hohner Special 20 (C)", family: SourceFamilyV1::FreeReedAerophone, pitch_semantics: PitchSemanticsV1::PitchedSource, source_path: "Aerophones/Free Aerophones/Harmonica-Hohner-Special20-C/Releases/Vib", articulation: "release_vibrato", sample_role: SampleRoleV1::ReleaseEvent, event_lane: EventLaneV1::Release },
    IdentitySpec { record_id: "vcsl.path.aerophones.free-aerophones.harmonica-hohner-special20-f.sustains.accented", source_instrument_id: "vcsl.harmonica-special20-f", display_name: "Harmonica Hohner Special 20 (F)", family: SourceFamilyV1::FreeReedAerophone, pitch_semantics: PitchSemanticsV1::PitchedSource, source_path: "Aerophones/Free Aerophones/Harmonica-Hohner-Special20-F/Sustains/Accented", articulation: "accented", sample_role: SampleRoleV1::Sustain, event_lane: EventLaneV1::NoteOn },
    IdentitySpec { record_id: "vcsl.path.aerophones.free-aerophones.harmonica-hohner-special20-f.sustains.handvib", source_instrument_id: "vcsl.harmonica-special20-f", display_name: "Harmonica Hohner Special 20 (F)", family: SourceFamilyV1::FreeReedAerophone, pitch_semantics: PitchSemanticsV1::PitchedSource, source_path: "Aerophones/Free Aerophones/Harmonica-Hohner-Special20-F/Sustains/HandVib", articulation: "hand_vibrato", sample_role: SampleRoleV1::Sustain, event_lane: EventLaneV1::NoteOn },
    IdentitySpec { record_id: "vcsl.path.aerophones.free-aerophones.harmonica-hohner-special20-f.sustains.normal", source_instrument_id: "vcsl.harmonica-special20-f", display_name: "Harmonica Hohner Special 20 (F)", family: SourceFamilyV1::FreeReedAerophone, pitch_semantics: PitchSemanticsV1::PitchedSource, source_path: "Aerophones/Free Aerophones/Harmonica-Hohner-Special20-F/Sustains/Normal", articulation: "normal", sample_role: SampleRoleV1::Sustain, event_lane: EventLaneV1::NoteOn },
    IdentitySpec { record_id: "vcsl.path.aerophones.free-aerophones.harmonica-hohner-special20-f.sustains.stac", source_instrument_id: "vcsl.harmonica-special20-f", display_name: "Harmonica Hohner Special 20 (F)", family: SourceFamilyV1::FreeReedAerophone, pitch_semantics: PitchSemanticsV1::PitchedSource, source_path: "Aerophones/Free Aerophones/Harmonica-Hohner-Special20-F/Sustains/Stac", articulation: "staccato", sample_role: SampleRoleV1::ShortArticulation, event_lane: EventLaneV1::NoteOn },
    IdentitySpec { record_id: "vcsl.path.aerophones.free-aerophones.harmonica-hohner-special20-f.sustains.vib", source_instrument_id: "vcsl.harmonica-special20-f", display_name: "Harmonica Hohner Special 20 (F)", family: SourceFamilyV1::FreeReedAerophone, pitch_semantics: PitchSemanticsV1::PitchedSource, source_path: "Aerophones/Free Aerophones/Harmonica-Hohner-Special20-F/Sustains/Vib", articulation: "vibrato", sample_role: SampleRoleV1::Sustain, event_lane: EventLaneV1::NoteOn },
    IdentitySpec { record_id: "vcsl.path.aerophones.free-aerophones.harmonica-hohner-special20-f.releases.accented", source_instrument_id: "vcsl.harmonica-special20-f", display_name: "Harmonica Hohner Special 20 (F)", family: SourceFamilyV1::FreeReedAerophone, pitch_semantics: PitchSemanticsV1::PitchedSource, source_path: "Aerophones/Free Aerophones/Harmonica-Hohner-Special20-F/Releases/Accented", articulation: "release_accented", sample_role: SampleRoleV1::ReleaseEvent, event_lane: EventLaneV1::Release },
    IdentitySpec { record_id: "vcsl.path.aerophones.free-aerophones.harmonica-hohner-special20-f.releases.handvib", source_instrument_id: "vcsl.harmonica-special20-f", display_name: "Harmonica Hohner Special 20 (F)", family: SourceFamilyV1::FreeReedAerophone, pitch_semantics: PitchSemanticsV1::PitchedSource, source_path: "Aerophones/Free Aerophones/Harmonica-Hohner-Special20-F/Releases/HandVib", articulation: "release_hand_vibrato", sample_role: SampleRoleV1::ReleaseEvent, event_lane: EventLaneV1::Release },
    IdentitySpec { record_id: "vcsl.path.aerophones.free-aerophones.harmonica-hohner-special20-f.releases.normal", source_instrument_id: "vcsl.harmonica-special20-f", display_name: "Harmonica Hohner Special 20 (F)", family: SourceFamilyV1::FreeReedAerophone, pitch_semantics: PitchSemanticsV1::PitchedSource, source_path: "Aerophones/Free Aerophones/Harmonica-Hohner-Special20-F/Releases/Normal", articulation: "release_normal", sample_role: SampleRoleV1::ReleaseEvent, event_lane: EventLaneV1::Release },
    IdentitySpec { record_id: "vcsl.path.aerophones.free-aerophones.harmonica-hohner-special20-f.releases.vib", source_instrument_id: "vcsl.harmonica-special20-f", display_name: "Harmonica Hohner Special 20 (F)", family: SourceFamilyV1::FreeReedAerophone, pitch_semantics: PitchSemanticsV1::PitchedSource, source_path: "Aerophones/Free Aerophones/Harmonica-Hohner-Special20-F/Releases/Vib", articulation: "release_vibrato", sample_role: SampleRoleV1::ReleaseEvent, event_lane: EventLaneV1::Release },
    IdentitySpec { record_id: "vcsl.path.aerophones.free-aerophones.harmonica-hohner-super64.sustains.accented", source_instrument_id: "vcsl.harmonica-super64", display_name: "Harmonica Hohner Super 64", family: SourceFamilyV1::FreeReedAerophone, pitch_semantics: PitchSemanticsV1::PitchedSource, source_path: "Aerophones/Free Aerophones/Harmonica-Hohner-Super64/Sustains/Accented", articulation: "accented", sample_role: SampleRoleV1::Sustain, event_lane: EventLaneV1::NoteOn },
    IdentitySpec { record_id: "vcsl.path.aerophones.free-aerophones.harmonica-hohner-super64.sustains.normal", source_instrument_id: "vcsl.harmonica-super64", display_name: "Harmonica Hohner Super 64", family: SourceFamilyV1::FreeReedAerophone, pitch_semantics: PitchSemanticsV1::PitchedSource, source_path: "Aerophones/Free Aerophones/Harmonica-Hohner-Super64/Sustains/Normal", articulation: "normal", sample_role: SampleRoleV1::Sustain, event_lane: EventLaneV1::NoteOn },
    IdentitySpec { record_id: "vcsl.path.aerophones.free-aerophones.harmonica-hohner-super64.sustains.vib", source_instrument_id: "vcsl.harmonica-super64", display_name: "Harmonica Hohner Super 64", family: SourceFamilyV1::FreeReedAerophone, pitch_semantics: PitchSemanticsV1::PitchedSource, source_path: "Aerophones/Free Aerophones/Harmonica-Hohner-Super64/Sustains/Vib", articulation: "vibrato", sample_role: SampleRoleV1::Sustain, event_lane: EventLaneV1::NoteOn },
    IdentitySpec { record_id: "vcsl.path.aerophones.free-aerophones.harmonica-hohner-super64.releases.accented", source_instrument_id: "vcsl.harmonica-super64", display_name: "Harmonica Hohner Super 64", family: SourceFamilyV1::FreeReedAerophone, pitch_semantics: PitchSemanticsV1::PitchedSource, source_path: "Aerophones/Free Aerophones/Harmonica-Hohner-Super64/Releases/Accented", articulation: "release_accented", sample_role: SampleRoleV1::ReleaseEvent, event_lane: EventLaneV1::Release },
    IdentitySpec { record_id: "vcsl.path.aerophones.free-aerophones.harmonica-hohner-super64.releases.normal", source_instrument_id: "vcsl.harmonica-super64", display_name: "Harmonica Hohner Super 64", family: SourceFamilyV1::FreeReedAerophone, pitch_semantics: PitchSemanticsV1::PitchedSource, source_path: "Aerophones/Free Aerophones/Harmonica-Hohner-Super64/Releases/Normal", articulation: "release_normal", sample_role: SampleRoleV1::ReleaseEvent, event_lane: EventLaneV1::Release },
    IdentitySpec { record_id: "vcsl.path.aerophones.free-aerophones.harmonica-hohner-super64.releases.vib", source_instrument_id: "vcsl.harmonica-super64", display_name: "Harmonica Hohner Super 64", family: SourceFamilyV1::FreeReedAerophone, pitch_semantics: PitchSemanticsV1::PitchedSource, source_path: "Aerophones/Free Aerophones/Harmonica-Hohner-Super64/Releases/Vib", articulation: "release_vibrato", sample_role: SampleRoleV1::ReleaseEvent, event_lane: EventLaneV1::Release },
    IdentitySpec { record_id: "vcsl.path.aerophones.edge-blown-aerophones.baroque-soprano-recorder.sustain", source_instrument_id: "vcsl.baroque-soprano-recorder", display_name: "Baroque Soprano Recorder", family: SourceFamilyV1::EdgeBlownAerophone, pitch_semantics: PitchSemanticsV1::PitchedSource, source_path: "Aerophones/Edge-blown Aerophones/Baroque Soprano Recorder/Sustain", articulation: "sustain", sample_role: SampleRoleV1::Sustain, event_lane: EventLaneV1::NoteOn },
    IdentitySpec { record_id: "vcsl.path.aerophones.edge-blown-aerophones.baroque-soprano-recorder.staccato", source_instrument_id: "vcsl.baroque-soprano-recorder", display_name: "Baroque Soprano Recorder", family: SourceFamilyV1::EdgeBlownAerophone, pitch_semantics: PitchSemanticsV1::PitchedSource, source_path: "Aerophones/Edge-blown Aerophones/Baroque Soprano Recorder/Staccato", articulation: "staccato", sample_role: SampleRoleV1::ShortArticulation, event_lane: EventLaneV1::NoteOn },
    IdentitySpec { record_id: "vcsl.path.aerophones.edge-blown-aerophones.baroque-alto-recorder.sustain", source_instrument_id: "vcsl.baroque-alto-recorder", display_name: "Baroque Alto Recorder", family: SourceFamilyV1::EdgeBlownAerophone, pitch_semantics: PitchSemanticsV1::PitchedSource, source_path: "Aerophones/Edge-blown Aerophones/Baroque Alto Recorder/Sustain", articulation: "sustain", sample_role: SampleRoleV1::Sustain, event_lane: EventLaneV1::NoteOn },
    IdentitySpec { record_id: "vcsl.path.aerophones.edge-blown-aerophones.baroque-alto-recorder.staccato", source_instrument_id: "vcsl.baroque-alto-recorder", display_name: "Baroque Alto Recorder", family: SourceFamilyV1::EdgeBlownAerophone, pitch_semantics: PitchSemanticsV1::PitchedSource, source_path: "Aerophones/Edge-blown Aerophones/Baroque Alto Recorder/Staccato", articulation: "staccato", sample_role: SampleRoleV1::ShortArticulation, event_lane: EventLaneV1::NoteOn },
    IdentitySpec { record_id: "vcsl.path.aerophones.edge-blown-aerophones.baroque-alto-recorder.susvib", source_instrument_id: "vcsl.baroque-alto-recorder", display_name: "Baroque Alto Recorder", family: SourceFamilyV1::EdgeBlownAerophone, pitch_semantics: PitchSemanticsV1::PitchedSource, source_path: "Aerophones/Edge-blown Aerophones/Baroque Alto Recorder/SusVib", articulation: "sustain_vibrato", sample_role: SampleRoleV1::Sustain, event_lane: EventLaneV1::NoteOn },
    IdentitySpec { record_id: "vcsl.path.aerophones.edge-blown-aerophones.baroque-bass-recorder.sustain", source_instrument_id: "vcsl.baroque-bass-recorder", display_name: "Baroque Bass Recorder", family: SourceFamilyV1::EdgeBlownAerophone, pitch_semantics: PitchSemanticsV1::PitchedSource, source_path: "Aerophones/Edge-blown Aerophones/Baroque Bass Recorder/Sustain", articulation: "sustain", sample_role: SampleRoleV1::Sustain, event_lane: EventLaneV1::NoteOn },
    IdentitySpec { record_id: "vcsl.path.aerophones.edge-blown-aerophones.baroque-bass-recorder.staccato", source_instrument_id: "vcsl.baroque-bass-recorder", display_name: "Baroque Bass Recorder", family: SourceFamilyV1::EdgeBlownAerophone, pitch_semantics: PitchSemanticsV1::PitchedSource, source_path: "Aerophones/Edge-blown Aerophones/Baroque Bass Recorder/Staccato", articulation: "staccato", sample_role: SampleRoleV1::ShortArticulation, event_lane: EventLaneV1::NoteOn },
    IdentitySpec { record_id: "vcsl.path.aerophones.edge-blown-aerophones.baroque-bass-recorder.susvib", source_instrument_id: "vcsl.baroque-bass-recorder", display_name: "Baroque Bass Recorder", family: SourceFamilyV1::EdgeBlownAerophone, pitch_semantics: PitchSemanticsV1::PitchedSource, source_path: "Aerophones/Edge-blown Aerophones/Baroque Bass Recorder/SusVib", articulation: "sustain_vibrato", sample_role: SampleRoleV1::Sustain, event_lane: EventLaneV1::NoteOn },
    IdentitySpec { record_id: "vcsl.path.aerophones.edge-blown-aerophones.baroque-tenor-recorder.sustain", source_instrument_id: "vcsl.baroque-tenor-recorder", display_name: "Baroque Tenor Recorder", family: SourceFamilyV1::EdgeBlownAerophone, pitch_semantics: PitchSemanticsV1::PitchedSource, source_path: "Aerophones/Edge-blown Aerophones/Baroque Tenor Recorder/Sustain", articulation: "sustain", sample_role: SampleRoleV1::Sustain, event_lane: EventLaneV1::NoteOn },
    IdentitySpec { record_id: "vcsl.path.aerophones.edge-blown-aerophones.baroque-tenor-recorder.staccato", source_instrument_id: "vcsl.baroque-tenor-recorder", display_name: "Baroque Tenor Recorder", family: SourceFamilyV1::EdgeBlownAerophone, pitch_semantics: PitchSemanticsV1::PitchedSource, source_path: "Aerophones/Edge-blown Aerophones/Baroque Tenor Recorder/Staccato", articulation: "staccato", sample_role: SampleRoleV1::ShortArticulation, event_lane: EventLaneV1::NoteOn },
    IdentitySpec { record_id: "vcsl.path.aerophones.edge-blown-aerophones.baroque-tenor-recorder.susvib", source_instrument_id: "vcsl.baroque-tenor-recorder", display_name: "Baroque Tenor Recorder", family: SourceFamilyV1::EdgeBlownAerophone, pitch_semantics: PitchSemanticsV1::PitchedSource, source_path: "Aerophones/Edge-blown Aerophones/Baroque Tenor Recorder/SusVib", articulation: "sustain_vibrato", sample_role: SampleRoleV1::Sustain, event_lane: EventLaneV1::NoteOn },
    IdentitySpec { record_id: "vcsl.path.membranophones.struck-membranophones.timpani-1.hit", source_instrument_id: "vcsl.timpani-1", display_name: "Timpani 1", family: SourceFamilyV1::Membranophone, pitch_semantics: PitchSemanticsV1::PercussionDrumIdentifierOnly, source_path: "Membranophones/Struck Membranophones/Timpani 1/Hit", articulation: "hit", sample_role: SampleRoleV1::PercussionHit, event_lane: EventLaneV1::Percussion },
    IdentitySpec { record_id: "vcsl.path.membranophones.struck-membranophones.timpani-1.roll", source_instrument_id: "vcsl.timpani-1", display_name: "Timpani 1", family: SourceFamilyV1::Membranophone, pitch_semantics: PitchSemanticsV1::PercussionDrumIdentifierOnly, source_path: "Membranophones/Struck Membranophones/Timpani 1/Roll", articulation: "roll", sample_role: SampleRoleV1::PercussionRoll, event_lane: EventLaneV1::Percussion },
    IdentitySpec { record_id: "vcsl.path.membranophones.struck-membranophones.timpani-2.hit", source_instrument_id: "vcsl.timpani-2", display_name: "Timpani 2", family: SourceFamilyV1::Membranophone, pitch_semantics: PitchSemanticsV1::PercussionDrumIdentifierOnly, source_path: "Membranophones/Struck Membranophones/Timpani 2/Hit", articulation: "hit", sample_role: SampleRoleV1::PercussionHit, event_lane: EventLaneV1::Percussion },
    IdentitySpec { record_id: "vcsl.path.membranophones.struck-membranophones.tom-1.mallet", source_instrument_id: "vcsl.tom-1", display_name: "Tom 1", family: SourceFamilyV1::Membranophone, pitch_semantics: PitchSemanticsV1::PercussionDrumIdentifierOnly, source_path: "Membranophones/Struck Membranophones/Tom 1/Mallet", articulation: "mallet", sample_role: SampleRoleV1::PercussionMallet, event_lane: EventLaneV1::Percussion },
    IdentitySpec { record_id: "vcsl.path.membranophones.struck-membranophones.tom-1.stick", source_instrument_id: "vcsl.tom-1", display_name: "Tom 1", family: SourceFamilyV1::Membranophone, pitch_semantics: PitchSemanticsV1::PercussionDrumIdentifierOnly, source_path: "Membranophones/Struck Membranophones/Tom 1/Stick", articulation: "stick", sample_role: SampleRoleV1::PercussionStick, event_lane: EventLaneV1::Percussion },
    IdentitySpec { record_id: "vcsl.path.membranophones.struck-membranophones.tom-2.mallet", source_instrument_id: "vcsl.tom-2", display_name: "Tom 2", family: SourceFamilyV1::Membranophone, pitch_semantics: PitchSemanticsV1::PercussionDrumIdentifierOnly, source_path: "Membranophones/Struck Membranophones/Tom 2/Mallet", articulation: "mallet", sample_role: SampleRoleV1::PercussionMallet, event_lane: EventLaneV1::Percussion },
    IdentitySpec { record_id: "vcsl.path.membranophones.struck-membranophones.tom-2.stick", source_instrument_id: "vcsl.tom-2", display_name: "Tom 2", family: SourceFamilyV1::Membranophone, pitch_semantics: PitchSemanticsV1::PercussionDrumIdentifierOnly, source_path: "Membranophones/Struck Membranophones/Tom 2/Stick", articulation: "stick", sample_role: SampleRoleV1::PercussionStick, event_lane: EventLaneV1::Percussion },
    IdentitySpec { record_id: "vcsl.path.membranophones.struck-membranophones.legacy-toms.tenor-higher", source_instrument_id: "vcsl.legacy-toms", display_name: "Legacy Toms", family: SourceFamilyV1::Membranophone, pitch_semantics: PitchSemanticsV1::PercussionDrumIdentifierOnly, source_path: "Membranophones/Struck Membranophones/Legacy Toms/tenor_higher", articulation: "tenor_higher", sample_role: SampleRoleV1::TunedDrumVariant, event_lane: EventLaneV1::Percussion },
    IdentitySpec { record_id: "vcsl.path.membranophones.struck-membranophones.legacy-toms.tenor-lower", source_instrument_id: "vcsl.legacy-toms", display_name: "Legacy Toms", family: SourceFamilyV1::Membranophone, pitch_semantics: PitchSemanticsV1::PercussionDrumIdentifierOnly, source_path: "Membranophones/Struck Membranophones/Legacy Toms/tenor_lower", articulation: "tenor_lower", sample_role: SampleRoleV1::TunedDrumVariant, event_lane: EventLaneV1::Percussion },
];

/// Return a serializable catalog bound to the source tree revision.
pub fn vcsl_source_identity_catalog_v1() -> SourceInstrumentIdentityCatalogV1 {
    let identities: Vec<_> = IDENTITY_SPECS
        .iter()
        .map(|spec| SourceInstrumentIdentityV1 {
            record_id: spec.record_id.to_owned(),
            source_instrument_id: spec.source_instrument_id.to_owned(),
            display_name: spec.display_name.to_owned(),
            family: spec.family,
            pitch_semantics: spec.pitch_semantics,
            source_path: spec.source_path.to_owned(),
            articulation: spec.articulation.to_owned(),
            sample_role: spec.sample_role,
            event_lane: spec.event_lane,
            library: "VCSL".to_owned(),
            source_tree_revision: VCSL_TREE_REVISION_V1.to_owned(),
            license_identifier: VCSL_LICENSE_V1.to_owned(),
            renderer_substitute_id: None,
            playability_evidence: PlayabilityEvidenceV1::CataloguedOnly,
        })
        .collect();
    SourceInstrumentIdentityCatalogV1 {
        schema: "muse-source-instrument-identity-catalog-v1".to_owned(),
        library: "VCSL".to_owned(),
        source_tree_revision: VCSL_TREE_REVISION_V1.to_owned(),
        license_identifier: VCSL_LICENSE_V1.to_owned(),
        identity_record_count: identities.len(),
        identities,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn catalog_uses_stable_unique_record_ids_and_pinned_provenance() {
        let catalog = vcsl_source_identity_catalog_v1();
        assert_eq!(catalog.schema, "muse-source-instrument-identity-catalog-v1");
        assert_eq!(catalog.source_tree_revision, VCSL_TREE_REVISION_V1);
        assert_eq!(catalog.license_identifier, "CC0-1.0");
        assert_eq!(catalog.identity_record_count, catalog.identities.len());
        assert!(catalog.identity_record_count >= 50);

        let ids: BTreeSet<_> = catalog.identities.iter().map(|entry| &entry.record_id).collect();
        assert_eq!(ids.len(), catalog.identities.len(), "record ids must be unique");
        for record in &catalog.identities {
            assert_eq!(record.library, "VCSL");
            assert_eq!(record.source_tree_revision, VCSL_TREE_REVISION_V1);
            assert_eq!(record.license_identifier, VCSL_LICENSE_V1);
            assert!(record.source_path.starts_with("Aerophones/")
                || record.source_path.starts_with("Chordophones/")
                || record.source_path.starts_with("Membranophones/"));
            assert!(!record.source_path.starts_with('/'));
            assert!(!record.source_path.split('/').any(|part| part == ".."));
            assert_eq!(record.playability_evidence, PlayabilityEvidenceV1::CataloguedOnly);
            assert_eq!(record.renderer_substitute_id, None);
        }
    }

    #[test]
    fn release_and_percussion_lanes_cannot_be_misread_as_note_on_banks() {
        let catalog = vcsl_source_identity_catalog_v1();
        for record in &catalog.identities {
            match record.sample_role {
                SampleRoleV1::ReleaseEvent => assert_eq!(record.event_lane, EventLaneV1::Release),
                SampleRoleV1::PercussionHit
                | SampleRoleV1::PercussionRoll
                | SampleRoleV1::PercussionMallet
                | SampleRoleV1::PercussionStick
                | SampleRoleV1::TunedDrumVariant => {
                    assert_eq!(record.event_lane, EventLaneV1::Percussion);
                    assert_eq!(record.pitch_semantics, PitchSemanticsV1::PercussionDrumIdentifierOnly);
                }
                SampleRoleV1::AuxiliaryEvent => assert_eq!(record.event_lane, EventLaneV1::Auxiliary),
                SampleRoleV1::Sustain | SampleRoleV1::ShortArticulation => {
                    assert_eq!(record.event_lane, EventLaneV1::NoteOn);
                    assert_eq!(record.pitch_semantics, PitchSemanticsV1::PitchedSource);
                }
            }
        }
    }

    #[test]
    fn serialization_round_trips_without_collapsing_source_models() {
        let original = vcsl_source_identity_catalog_v1();
        let encoded = serde_json::to_vec(&original).unwrap();
        let decoded: SourceInstrumentIdentityCatalogV1 = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(original, decoded);

        let ids: BTreeSet<_> = decoded.identities.iter()
            .map(|record| record.source_instrument_id.as_str()).collect();
        for required in [
            "vcsl.dan-tranh",
            "vcsl.ocarina-small",
            "vcsl.ocarina-typical",
            "vcsl.harmonica-special20-c",
            "vcsl.harmonica-special20-f",
            "vcsl.harmonica-super64",
            "vcsl.baroque-soprano-recorder",
            "vcsl.baroque-alto-recorder",
            "vcsl.baroque-tenor-recorder",
            "vcsl.baroque-bass-recorder",
        ] {
            assert!(ids.contains(required), "missing source-native identity {required}");
        }
    }
}
