// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Explicit, source-bound MIDI measure maps.
//!
//! Standard MIDI File time-signature metadata is not a complete measure-boundary
//! sequence. This adapter therefore requires a separate sidecar whose source
//! digest and PPQ division match the exact MIDI bytes. It converts source ticks
//! to exact rational quarter-note beats and never infers bars from FF 58,
//! marker events, note density or the legacy Score::meter field.

use crate::barline_eval::{MeasureKind, MeasureMap, MeasureMapStatus, MeasureSpan, RationalBeat};
use crate::evidence_digest::{canonical_json_sha256, sha256_hex};
use midly::{Smf, Timing};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const MIDI_MEASURE_SIDECAR_SCHEMA_V1: &str = "muse-midi-measure-sidecar-v1";
pub const MIDI_MEASURE_CANONICALIZATION_V1: &str =
    "midi-explicit-ppq-ticks-to-rational-quarter-beats-v1";

/// A single explicit measure span in source MIDI ticks.
///
/// Start is inclusive and end is exclusive. Tick coordinates are relative to
/// the beginning of the track's notated score-clock, not wall-clock time.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MidiMeasureSpanSidecarV1 {
    pub start_tick: u64,
    pub end_tick: u64,
    pub kind: MeasureKind,
    pub measure_number: Option<String>,
    pub source_measure_id: Option<String>,
    pub display_text: Option<String>,
    pub implicit: bool,
    pub non_controlling: bool,
}

/// One part's explicitly annotated measure lattice, sourced from one MIDI track.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MidiPartMeasureSidecarV1 {
    pub part_id: String,
    pub source_track_index: usize,
    pub measures: Vec<MidiMeasureSpanSidecarV1>,
}

/// Sidecar JSON envelope. Its digest is calculated from the exact sidecar bytes
/// at ingestion and retained in the resulting receipt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MidiMeasureSidecarV1 {
    pub schema: String,
    pub midi_sha256: String,
    pub ticks_per_quarter: u16,
    pub score_clock_id: String,
    pub parts: Vec<MidiPartMeasureSidecarV1>,
}

/// Source-native locator carried beside a normalized MeasureMap.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MidiMeasureLocatorV1 {
    pub measure_index: usize,
    pub start_tick: u64,
    pub end_tick: u64,
    pub measure_number: Option<String>,
    pub source_measure_id: Option<String>,
    pub display_text: Option<String>,
    pub implicit: bool,
    pub non_controlling: bool,
}

/// One part's explicit, PPQ-normalized map.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MidiPartMeasureMapV1 {
    pub part_id: String,
    pub source_track_index: usize,
    pub measure_locators: Vec<MidiMeasureLocatorV1>,
    pub non_controlling_measure_indices: Vec<usize>,
    pub map: MeasureMap,
}

/// Verified receipt binding the normalized maps to the original MIDI and exact
/// sidecar inputs. Digests identify content; they do not establish authority.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MidiMeasureSidecarReceiptV1 {
    pub schema: String,
    pub midi_sha256: String,
    pub sidecar_sha256: String,
    pub score_clock_id: String,
    pub ticks_per_quarter: u16,
    pub part_maps: Vec<MidiPartMeasureMapV1>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MidiMeasureSidecarError {
    InvalidMidi(String),
    InvalidSidecarJson(String),
    UnsupportedSidecarSchema { schema: String },
    EmptyScoreClockId,
    EmptyMidiSourceDigest,
    SourceDigestMismatch,
    SidecarScoreClockMismatch,
    PpqMismatch { expected: u16, actual: u16 },
    SmpteTimingUnsupported,
    InvalidPpq,
    NoTracks,
    NoParts,
    EmptyPartId,
    DuplicatePartId { part_id: String },
    DuplicateTrackIndex { track_index: usize },
    TrackIndexOutOfRange { track_index: usize, track_count: usize },
    EmptyMeasureList { part_id: String },
    TickOutOfRange { tick: u64 },
    EmptyOrReversedMeasure { part_id: String, measure_index: usize },
    NonContiguousMeasures { part_id: String, left_index: usize },
    ArithmeticOverflow,
    SerializationFailed,
}

/// Parse a source-bound MIDI sidecar into exact per-part measure maps.
///
/// `score_clock_id` is an explicit caller-selected identity for the notation
/// coordinate frame. The sidecar must declare the same value.
pub fn parse_midi_measure_sidecar(
    midi_bytes: &[u8],
    sidecar_bytes: &[u8],
    score_clock_id: &str,
) -> Result<MidiMeasureSidecarReceiptV1, MidiMeasureSidecarError> {
    if score_clock_id.trim().is_empty() {
        return Err(MidiMeasureSidecarError::EmptyScoreClockId);
    }
    let smf = Smf::parse(midi_bytes)
        .map_err(|error| MidiMeasureSidecarError::InvalidMidi(error.to_string()))?;
    if smf.tracks.is_empty() {
        return Err(MidiMeasureSidecarError::NoTracks);
    }
    let ppq = match smf.header.timing {
        Timing::Metrical(value) => {
            let ppq = value.as_int();
            if ppq == 0 {
                return Err(MidiMeasureSidecarError::InvalidPpq);
            }
            ppq
        }
        Timing::Timecode(_, _) => return Err(MidiMeasureSidecarError::SmpteTimingUnsupported),
    };

    let sidecar: MidiMeasureSidecarV1 = serde_json::from_slice(sidecar_bytes)
        .map_err(|error| MidiMeasureSidecarError::InvalidSidecarJson(error.to_string()))?;
    if sidecar.schema != MIDI_MEASURE_SIDECAR_SCHEMA_V1 {
        return Err(MidiMeasureSidecarError::UnsupportedSidecarSchema {
            schema: sidecar.schema,
        });
    }
    if sidecar.midi_sha256.trim().is_empty() {
        return Err(MidiMeasureSidecarError::EmptyMidiSourceDigest);
    }
    if sidecar.midi_sha256 != sha256_hex(midi_bytes) {
        return Err(MidiMeasureSidecarError::SourceDigestMismatch);
    }
    if sidecar.score_clock_id != score_clock_id {
        return Err(MidiMeasureSidecarError::SidecarScoreClockMismatch);
    }
    if sidecar.ticks_per_quarter != ppq {
        return Err(MidiMeasureSidecarError::PpqMismatch {
            expected: ppq,
            actual: sidecar.ticks_per_quarter,
        });
    }
    if sidecar.parts.is_empty() {
        return Err(MidiMeasureSidecarError::NoParts);
    }

    let sidecar_sha256 = sha256_hex(sidecar_bytes);
    let mut part_ids = BTreeSet::new();
    let mut track_indices = BTreeSet::new();
    let mut part_maps = Vec::with_capacity(sidecar.parts.len());

    for part in sidecar.parts {
        if part.part_id.trim().is_empty() {
            return Err(MidiMeasureSidecarError::EmptyPartId);
        }
        if !part_ids.insert(part.part_id.clone()) {
            return Err(MidiMeasureSidecarError::DuplicatePartId {
                part_id: part.part_id,
            });
        }
        if !track_indices.insert(part.source_track_index) {
            return Err(MidiMeasureSidecarError::DuplicateTrackIndex {
                track_index: part.source_track_index,
            });
        }
        if part.source_track_index >= smf.tracks.len() {
            return Err(MidiMeasureSidecarError::TrackIndexOutOfRange {
                track_index: part.source_track_index,
                track_count: smf.tracks.len(),
            });
        }
        if part.measures.is_empty() {
            return Err(MidiMeasureSidecarError::EmptyMeasureList {
                part_id: part.part_id,
            });
        }

        let mut spans = Vec::with_capacity(part.measures.len());
        let mut locators = Vec::with_capacity(part.measures.len());
        let mut non_controlling_indices = Vec::new();
        let mut previous_end_tick = 0_u64;
        for (measure_index, measure) in part.measures.iter().enumerate() {
            if measure.start_tick > i64::MAX as u64 || measure.end_tick > i64::MAX as u64 {
                return Err(MidiMeasureSidecarError::TickOutOfRange {
                    tick: measure.start_tick.max(measure.end_tick),
                });
            }
            if measure.start_tick >= measure.end_tick {
                return Err(MidiMeasureSidecarError::EmptyOrReversedMeasure {
                    part_id: part.part_id.clone(),
                    measure_index,
                });
            }
            if measure_index == 0 {
                if measure.start_tick != 0 {
                    return Err(MidiMeasureSidecarError::NonContiguousMeasures {
                        part_id: part.part_id.clone(),
                        left_index: measure_index,
                    });
                }
            } else if measure.start_tick != previous_end_tick {
                return Err(MidiMeasureSidecarError::NonContiguousMeasures {
                    part_id: part.part_id.clone(),
                    left_index: measure_index - 1,
                });
            }
            let start = RationalBeat::new(measure.start_tick as i64, i64::from(ppq))
                .map_err(|_| MidiMeasureSidecarError::ArithmeticOverflow)?;
            let end = RationalBeat::new(measure.end_tick as i64, i64::from(ppq))
                .map_err(|_| MidiMeasureSidecarError::ArithmeticOverflow)?;
            spans.push(MeasureSpan {
                start,
                end,
                kind: measure.kind,
            });
            if measure.non_controlling {
                non_controlling_indices.push(measure_index);
            }
            locators.push(MidiMeasureLocatorV1 {
                measure_index,
                start_tick: measure.start_tick,
                end_tick: measure.end_tick,
                measure_number: measure.measure_number.clone(),
                source_measure_id: measure.source_measure_id.clone(),
                display_text: measure.display_text.clone(),
                implicit: measure.implicit,
                non_controlling: measure.non_controlling,
            });
            previous_end_tick = measure.end_tick;
        }

        part_maps.push(MidiPartMeasureMapV1 {
            part_id: part.part_id.clone(),
            source_track_index: part.source_track_index,
            measure_locators: locators,
            non_controlling_measure_indices: non_controlling_indices,
            map: MeasureMap {
                source_id: format!(
                    "midi-sha256:{}/sidecar-sha256:{}#track={}",
                    sidecar.midi_sha256, sidecar_sha256, part.source_track_index
                ),
                score_clock_id: score_clock_id.to_string(),
                canonicalization_version: MIDI_MEASURE_CANONICALIZATION_V1.into(),
                status: MeasureMapStatus::Supported,
                measures: spans,
            },
        });
    }

    Ok(MidiMeasureSidecarReceiptV1 {
        schema: MIDI_MEASURE_SIDECAR_SCHEMA_V1.into(),
        midi_sha256: sidecar.midi_sha256,
        sidecar_sha256,
        score_clock_id: score_clock_id.to_string(),
        ticks_per_quarter: ppq,
        part_maps,
    })
}

/// Canonical content commitment to a normalized MIDI sidecar receipt.
pub fn midi_measure_sidecar_commitment(
    receipt: &MidiMeasureSidecarReceiptV1,
) -> Result<String, MidiMeasureSidecarError> {
    canonical_json_sha256(receipt).map_err(|_| MidiMeasureSidecarError::SerializationFailed)
}

/// Reparse the exact MIDI and sidecar bytes and compare every normalized map
/// and source locator. Any changed input, span or flag returns false.
pub fn validate_midi_measure_sidecar_replay(
    midi_bytes: &[u8],
    sidecar_bytes: &[u8],
    score_clock_id: &str,
    supplied: &MidiMeasureSidecarReceiptV1,
) -> Result<bool, MidiMeasureSidecarError> {
    let replayed = parse_midi_measure_sidecar(midi_bytes, sidecar_bytes, score_clock_id)?;
    Ok(replayed == *supplied)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn midi_file(ppq: u16, track_count: u16) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"MThd");
        bytes.extend_from_slice(&6_u32.to_be_bytes());
        bytes.extend_from_slice(&1_u16.to_be_bytes());
        bytes.extend_from_slice(&track_count.to_be_bytes());
        bytes.extend_from_slice(&ppq.to_be_bytes());
        for _ in 0..track_count {
            bytes.extend_from_slice(b"MTrk");
            bytes.extend_from_slice(&4_u32.to_be_bytes());
            bytes.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        }
        bytes
    }

    fn measure(start_tick: u64, end_tick: u64) -> MidiMeasureSpanSidecarV1 {
        MidiMeasureSpanSidecarV1 {
            start_tick,
            end_tick,
            kind: MeasureKind::Regular,
            measure_number: None,
            source_measure_id: None,
            display_text: None,
            implicit: false,
            non_controlling: false,
        }
    }

    fn sidecar(midi: &[u8], ppq: u16, parts: Vec<MidiPartMeasureSidecarV1>) -> Vec<u8> {
        serde_json::to_vec(&MidiMeasureSidecarV1 {
            schema: MIDI_MEASURE_SIDECAR_SCHEMA_V1.into(),
            midi_sha256: sha256_hex(midi),
            ticks_per_quarter: ppq,
            score_clock_id: "midi-test-clock".into(),
            parts,
        })
        .unwrap()
    }

    #[test]
    fn ppq_96_480_960_1920_maps_share_identical_rational_spans() {
        let mut maps = Vec::new();
        for ppq in [96_u16, 480, 960, 1920] {
            let midi = midi_file(ppq, 1);
            let json = sidecar(
                &midi,
                ppq,
                vec![MidiPartMeasureSidecarV1 {
                    part_id: "Piano".into(),
                    source_track_index: 0,
                    measures: vec![measure(0, u64::from(ppq) * 4)],
                }],
            );
            let receipt =
                parse_midi_measure_sidecar(&midi, &json, "midi-test-clock").unwrap();
            maps.push(receipt.part_maps[0].map.measures.clone());
        }
        for map in maps.iter().skip(1) {
            assert_eq!(map, &maps[0]);
        }
        assert_eq!(maps[0][0].start, RationalBeat::new(0, 1).unwrap());
        assert_eq!(maps[0][0].end, RationalBeat::new(4, 1).unwrap());
    }

    #[test]
    fn source_digest_mismatch_is_rejected() {
        let midi = midi_file(480, 1);
        let mut envelope: MidiMeasureSidecarV1 = serde_json::from_slice(&sidecar(
            &midi,
            480,
            vec![MidiPartMeasureSidecarV1 {
                part_id: "Piano".into(),
                source_track_index: 0,
                measures: vec![measure(0, 1920)],
            }],
        ))
        .unwrap();
        envelope.midi_sha256 = "0".repeat(64);
        let json = serde_json::to_vec(&envelope).unwrap();
        assert_eq!(
            parse_midi_measure_sidecar(&midi, &json, "midi-test-clock"),
            Err(MidiMeasureSidecarError::SourceDigestMismatch)
        );
    }

    #[test]
    fn ppq_mismatch_is_rejected() {
        let midi = midi_file(480, 1);
        let json = sidecar(
            &midi,
            960,
            vec![MidiPartMeasureSidecarV1 {
                part_id: "Piano".into(),
                source_track_index: 0,
                measures: vec![measure(0, 3840)],
            }],
        );
        assert_eq!(
            parse_midi_measure_sidecar(&midi, &json, "midi-test-clock"),
            Err(MidiMeasureSidecarError::PpqMismatch {
                expected: 480,
                actual: 960
            })
        );
    }

    #[test]
    fn smpte_timing_is_rejected_for_score_measure_maps() {
        let mut midi = midi_file(480, 1);
        midi[12] = 0xE7;
        midi[13] = 0x28;
        let json = sidecar(
            &midi,
            480,
            vec![MidiPartMeasureSidecarV1 {
                part_id: "Piano".into(),
                source_track_index: 0,
                measures: vec![measure(0, 1920)],
            }],
        );
        assert_eq!(
            parse_midi_measure_sidecar(&midi, &json, "midi-test-clock"),
            Err(MidiMeasureSidecarError::SmpteTimingUnsupported)
        );
    }

    #[test]
    fn nonexistent_track_and_duplicate_part_or_track_are_rejected() {
        let midi = midi_file(480, 1);
        let nonexistent = sidecar(
            &midi,
            480,
            vec![MidiPartMeasureSidecarV1 {
                part_id: "Piano".into(),
                source_track_index: 1,
                measures: vec![measure(0, 1920)],
            }],
        );
        assert!(matches!(
            parse_midi_measure_sidecar(&midi, &nonexistent, "midi-test-clock"),
            Err(MidiMeasureSidecarError::TrackIndexOutOfRange { .. })
        ));

        let duplicate_tracks = sidecar(
            &midi,
            480,
            vec![
                MidiPartMeasureSidecarV1 {
                    part_id: "Piano".into(),
                    source_track_index: 0,
                    measures: vec![measure(0, 1920)],
                },
                MidiPartMeasureSidecarV1 {
                    part_id: "Strings".into(),
                    source_track_index: 0,
                    measures: vec![measure(0, 1920)],
                },
            ],
        );
        assert!(matches!(
            parse_midi_measure_sidecar(&midi, &duplicate_tracks, "midi-test-clock"),
            Err(MidiMeasureSidecarError::DuplicateTrackIndex { .. })
        ));
    }

    #[test]
    fn duplicate_part_ids_and_empty_clock_identity_are_rejected() {
        let midi = midi_file(480, 1);
        let duplicate_parts = sidecar(
            &midi,
            480,
            vec![
                MidiPartMeasureSidecarV1 {
                    part_id: "Piano".into(),
                    source_track_index: 0,
                    measures: vec![measure(0, 1920)],
                },
                MidiPartMeasureSidecarV1 {
                    part_id: "Piano".into(),
                    source_track_index: 0,
                    measures: vec![measure(0, 1920)],
                },
            ],
        );
        assert!(matches!(
            parse_midi_measure_sidecar(&midi, &duplicate_parts, "midi-test-clock"),
            Err(MidiMeasureSidecarError::DuplicateTrackIndex { .. })
                | Err(MidiMeasureSidecarError::DuplicatePartId { .. })
        ));
        let json = sidecar(
            &midi,
            480,
            vec![MidiPartMeasureSidecarV1 {
                part_id: "Piano".into(),
                source_track_index: 0,
                measures: vec![measure(0, 1920)],
            }],
        );
        assert_eq!(
            parse_midi_measure_sidecar(&midi, &json, " "),
            Err(MidiMeasureSidecarError::EmptyScoreClockId)
        );
    }

    #[test]
    fn gaps_overlaps_and_zero_length_measures_are_rejected() {
        let midi = midi_file(480, 1);
        for measures in [
            vec![measure(0, 960), measure(961, 1920)],
            vec![measure(0, 960), measure(959, 1920)],
            vec![measure(0, 0)],
        ] {
            let json = sidecar(
                &midi,
                480,
                vec![MidiPartMeasureSidecarV1 {
                    part_id: "Piano".into(),
                    source_track_index: 0,
                    measures,
                }],
            );
            assert!(matches!(
                parse_midi_measure_sidecar(&midi, &json, "midi-test-clock"),
                Err(MidiMeasureSidecarError::NonContiguousMeasures { .. })
                    | Err(MidiMeasureSidecarError::EmptyOrReversedMeasure { .. })
            ));
        }
    }

    #[test]
    fn polymetric_tracks_keep_independent_lattices_and_shared_clock() {
        let midi = midi_file(480, 2);
        let json = sidecar(
            &midi,
            480,
            vec![
                MidiPartMeasureSidecarV1 {
                    part_id: "Waltz".into(),
                    source_track_index: 0,
                    measures: vec![
                        measure(0, 1440),
                        measure(1440, 2880),
                    ],
                },
                MidiPartMeasureSidecarV1 {
                    part_id: "Ostinato".into(),
                    source_track_index: 1,
                    measures: vec![
                        measure(0, 960),
                        MidiMeasureSpanSidecarV1 {
                            non_controlling: true,
                            ..measure(960, 1920)
                        },
                        MidiMeasureSpanSidecarV1 {
                            non_controlling: true,
                            ..measure(1920, 2880)
                        },
                    ],
                },
            ],
        );
        let receipt =
            parse_midi_measure_sidecar(&midi, &json, "midi-test-clock").unwrap();
        assert_eq!(receipt.part_maps.len(), 2);
        assert_eq!(
            receipt.part_maps[0].map.score_clock_id,
            receipt.part_maps[1].map.score_clock_id
        );
        assert_eq!(
            receipt.part_maps[0].map.measures.len(),
            2
        );
        assert_eq!(
            receipt.part_maps[1].non_controlling_measure_indices,
            vec![1, 2]
        );
        assert_eq!(
            receipt.part_maps[0].map.measures[0].end,
            RationalBeat::new(3, 1).unwrap()
        );
        assert_eq!(
            receipt.part_maps[1].map.measures[0].end,
            RationalBeat::new(2, 1).unwrap()
        );
    }

    #[test]
    fn source_locators_and_classification_flags_survive_sidecar_conversion() {
        let midi = midi_file(480, 1);
        let mut pickup = measure(0, 480);
        pickup.kind = MeasureKind::Pickup;
        pickup.implicit = true;
        pickup.measure_number = Some("Pickup".into());
        pickup.source_measure_id = Some("m0".into());
        let mut irregular = measure(480, 2400);
        irregular.kind = MeasureKind::Irregular;
        irregular.non_controlling = true;
        irregular.measure_number = Some("A".into());
        irregular.display_text = Some("A section".into());
        let json = sidecar(
            &midi,
            480,
            vec![MidiPartMeasureSidecarV1 {
                part_id: "Piano".into(),
                source_track_index: 0,
                measures: vec![pickup, irregular],
            }],
        );
        let receipt =
            parse_midi_measure_sidecar(&midi, &json, "midi-test-clock").unwrap();
        assert_eq!(receipt.part_maps[0].map.measures[0].kind, MeasureKind::Pickup);
        assert_eq!(receipt.part_maps[0].measure_locators[0].measure_number.as_deref(), Some("Pickup"));
        assert!(receipt.part_maps[0].measure_locators[0].implicit);
        assert_eq!(receipt.part_maps[0].measure_locators[1].display_text.as_deref(), Some("A section"));
        assert_eq!(receipt.part_maps[0].non_controlling_measure_indices, vec![1]);
    }

    #[test]
    fn replay_rejects_changed_sidecar_or_measure_map() {
        let midi = midi_file(480, 1);
        let json = sidecar(
            &midi,
            480,
            vec![MidiPartMeasureSidecarV1 {
                part_id: "Piano".into(),
                source_track_index: 0,
                measures: vec![measure(0, 1920)],
            }],
        );
        let receipt =
            parse_midi_measure_sidecar(&midi, &json, "midi-test-clock").unwrap();
        assert!(validate_midi_measure_sidecar_replay(
            &midi,
            &json,
            "midi-test-clock",
            &receipt
        ).unwrap());
        assert_eq!(
            midi_measure_sidecar_commitment(&receipt).unwrap().len(),
            64
        );

        let mut tampered = receipt.clone();
        tampered.part_maps[0].map.measures[0].end =
            RationalBeat::new(5, 1).unwrap();
        assert!(!validate_midi_measure_sidecar_replay(
            &midi,
            &json,
            "midi-test-clock",
            &tampered
        ).unwrap());
    }
}
