// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! PartId-aware, timeline-bound Standard MIDI realization and receipt generation.
//!
//! This child module intentionally reuses the parent MIDI encoder, while keeping
//! the new score-native realization contract separate from legacy exporters.

use super::{event_ordering_key, encode_track, MidiEvent, MidiEventType, Track};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use symthaea_music_theory::orchestration::InstrumentCapabilityV1;
use symthaea_music_theory::orchestration_timeline::{
    validate_orchestration_timeline, OrchestrationTimelineIssueV1,
    OrchestrationTimelineV1, RationalBeatV1, ORCHESTRATION_SCORE_DIGEST_SCHEMA_V1,
};
use symthaea_music_theory::rhythm::Duration;
use symthaea_music_theory::score::{Score, ScoreNote};

pub const PART_AWARE_MIDI_EXPORT_SCHEMA_V1: &str =
    "muse-part-aware-midi-export-v1";
pub const PART_AWARE_MIDI_TIMELINE_DIGEST_SCHEMA_V1: &str =
    "muse-orchestration-timeline-serde-json-sha256-v1";
pub const MIDI_TICKS_PER_QUARTER_V1: u16 = 960;

/// Explicit caller-supplied mapping from one concrete instrument instance
/// to a General MIDI Level 1 program number (0-based).
///
/// mapping_reference identifies the external mapping decision or record.
/// This exporter checks that the mapping is complete and unambiguous; it
/// does not authenticate that reference or certify acoustic patch quality.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MidiProgramBindingV1 {
    pub instrument_id: String,
    pub program: u8,
    pub mapping_reference: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PartAwareMidiTrackReceiptV1 {
    pub part_id: u16,
    /// Zero-based Standard MIDI File track index. Track zero is the
    /// tempo/time-signature track.
    pub track_index: u16,
    /// Stable channel for this persistent part across instrument handoffs.
    pub channel: u8,
    pub instrument_ids: Vec<String>,
    pub mapping_references: Vec<String>,
    pub note_count: usize,
}

/// One source score note and its exact encoded MIDI note-on/note-off pair.
///
/// score_note_ordinal is meaningful only together with score_sha256.
/// onset_tick and offset_tick use nearest-ties-to-even quantization at
/// MIDI_TICKS_PER_QUARTER_V1. The original rational beat values are kept
/// so the quantization can be independently recalculated.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PartAwareMidiEventMapV1 {
    pub score_sha256: String,
    pub score_note_ordinal: usize,
    pub part_id: u16,
    pub segment_id: String,
    pub instrument_id: String,
    pub track_index: u16,
    pub channel: u8,
    pub midi_pitch: u8,
    pub source_velocity: f32,
    pub midi_velocity: u8,
    pub source_onset: RationalBeatV1,
    pub source_duration: RationalBeatV1,
    pub onset_tick: u64,
    pub offset_tick: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PartAwareMidiProgramChangeReceiptV1 {
    pub part_id: u16,
    pub segment_id: String,
    pub instrument_id: String,
    pub track_index: u16,
    pub channel: u8,
    pub program: u8,
    pub mapping_reference: String,
    pub source_start: RationalBeatV1,
    pub start_tick: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PartAwareMidiExportReceiptV1 {
    pub schema: String,
    pub score_digest_schema: String,
    pub score_sha256: String,
    pub timeline_digest_schema: String,
    pub timeline_sha256: String,
    pub midi_sha256: String,
    pub ticks_per_quarter: u16,
    pub part_count: usize,
    pub note_count: usize,
    pub track_count: usize,
    pub tracks: Vec<PartAwareMidiTrackReceiptV1>,
    pub program_changes: Vec<PartAwareMidiProgramChangeReceiptV1>,
    /// Entries retain input Score.notes order; ordinals are digest-scoped.
    pub event_map: Vec<PartAwareMidiEventMapV1>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PartAwareMidiExportErrorV1 {
    TimelineInvalid {
        issues: Vec<OrchestrationTimelineIssueV1>,
    },
    InvalidTempoOrMeter,
    TooManyParts {
        actual: usize,
        supported: usize,
    },
    InvalidProgramBinding {
        detail: String,
    },
    MissingProgramBinding {
        instrument_id: String,
    },
    UnusedProgramBinding {
        instrument_id: String,
    },
    InvalidRationalTime {
        score_note_ordinal: usize,
    },
    MidiTickOverflow {
        score_note_ordinal: usize,
    },
    MidiDeltaExceedsStandardMaximum {
        track_index: u16,
        delta_ticks: u64,
    },
    UnrepresentableNoteDuration {
        score_note_ordinal: usize,
    },
    InvalidVelocity {
        score_note_ordinal: usize,
    },
    MissingOrAmbiguousSegment {
        score_note_ordinal: usize,
    },
    ProgramHandoffQuantizationCollision {
        part_id: u16,
        tick: u64,
    },
    OverlappingSamePitchNotes {
        part_id: u16,
        pitch: u8,
        first_score_note_ordinal: usize,
        second_score_note_ordinal: usize,
    },
    ScoreSerialization(String),
    ScoreDigestMismatch {
        expected: String,
        actual: String,
    },
    TimelineSerialization(String),
    MidiWrite(String),
}

/// A ready-to-write MIDI artifact plus the receipt that binds every
/// encoded note pair back to the exact source score and timeline.
#[derive(Debug, Clone, PartialEq)]
pub struct PartAwareMidiArtifactV1 {
    pub midi_bytes: Vec<u8>,
    pub receipt: PartAwareMidiExportReceiptV1,
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn domain_separated_sha256(schema: &str, bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(schema.as_bytes());
    hasher.update([0]);
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn cmp_rational(
    left_num: i128,
    left_den: i128,
    right_num: i128,
    right_den: i128,
) -> Option<Ordering> {
    if left_den <= 0 || right_den <= 0 {
        return None;
    }
    let left = left_num.checked_mul(right_den)?;
    let right = right_num.checked_mul(left_den)?;
    Some(left.cmp(&right))
}

fn rational_to_tick(numerator: i128, denominator: i128) -> Option<u64> {
    if numerator < 0 || denominator <= 0 {
        return None;
    }
    let scaled = numerator.checked_mul(i128::from(MIDI_TICKS_PER_QUARTER_V1))?;
    let quotient = scaled / denominator;
    let remainder = scaled % denominator;
    let twice_remainder = remainder.checked_mul(2)?;
    let round_up = twice_remainder > denominator
        || (twice_remainder == denominator && quotient % 2 != 0);
    let rounded = quotient.checked_add(if round_up { 1_i128 } else { 0_i128 })?;
    u64::try_from(rounded).ok()
}

fn rational_beat_to_tick(beat: RationalBeatV1) -> Option<u64> {
    rational_to_tick(
        i128::from(beat.numerator),
        i128::from(beat.denominator),
    )
}

fn duration_pair(value: Duration) -> (i128, i128) {
    (i128::from(value.num()), i128::from(value.den()))
}

fn note_end_pair(note: &ScoreNote) -> Option<(i128, i128)> {
    let (on_num, on_den) = duration_pair(note.onset);
    let (duration_num, duration_den) = duration_pair(note.duration);
    if on_num < 0 || duration_num <= 0 || on_den <= 0 || duration_den <= 0 {
        return None;
    }
    let left = on_num.checked_mul(duration_den)?;
    let right = duration_num.checked_mul(on_den)?;
    let numerator = left.checked_add(right)?;
    let denominator = on_den.checked_mul(duration_den)?;
    (denominator > 0).then_some((numerator, denominator))
}

fn note_fits_segment(note: &ScoreNote, segment: &symthaea_music_theory::orchestration_timeline::OrchestrationSegmentV1) -> Option<bool> {
    let (on_num, on_den) = duration_pair(note.onset);
    let (end_num, end_den) = note_end_pair(note)?;
    let onset_after_start = cmp_rational(
        on_num,
        on_den,
        i128::from(segment.start.numerator),
        i128::from(segment.start.denominator),
    )?;
    let end_before_segment_end = cmp_rational(
        end_num,
        end_den,
        i128::from(segment.end.numerator),
        i128::from(segment.end.denominator),
    )?;
    Some(
        onset_after_start != Ordering::Less
            && end_before_segment_end != Ordering::Greater,
    )
}

fn midi_velocity(value: f32, ordinal: usize) -> Result<u8, PartAwareMidiExportErrorV1> {
    if !value.is_finite() || value <= 0.0 || value > 1.0 {
        return Err(PartAwareMidiExportErrorV1::InvalidVelocity {
            score_note_ordinal: ordinal,
        });
    }
    Ok((value * 127.0).round().clamp(1.0, 127.0) as u8)
}

/// Encode a Score using persistent PartId and the exact time-indexed
/// orchestration plan. No VoiceRole grouping or pitch/register heuristics
/// are used. The expected digest must come from the trusted score-demand
/// boundary; the timeline validator checks the supplied score against it.
///
/// The caller supplies the explicit MIDI program mapping. Capabilities
/// passed here are the inputs to the timeline validator; this function
/// does not independently authenticate their evidence provenance.
pub fn encode_score_midi_part_aware_v1(
    score: &Score,
    timeline: &OrchestrationTimelineV1,
    instruments: &[InstrumentCapabilityV1],
    program_bindings: &[MidiProgramBindingV1],
    expected_score_sha256: &str,
) -> Result<PartAwareMidiArtifactV1, PartAwareMidiExportErrorV1> {
    // Recompute the exact score-demand digest from the score being
    // serialized. The timeline validator checks agreement with the
    // caller-supplied expected digest; this extra check prevents a stale
    // trusted digest from being paired with a changed Score.
    let serialized_score = serde_json::to_vec(score).map_err(|error| {
        PartAwareMidiExportErrorV1::ScoreSerialization(error.to_string())
    })?;
    let actual_score_sha256 = sha256_hex(&serialized_score);
    if actual_score_sha256 != expected_score_sha256 {
        return Err(PartAwareMidiExportErrorV1::ScoreDigestMismatch {
            expected: expected_score_sha256.to_owned(),
            actual: actual_score_sha256,
        });
    }

    let timeline_report = validate_orchestration_timeline(
        score,
        timeline,
        instruments,
        expected_score_sha256,
    );
    if !timeline_report.valid {
        return Err(PartAwareMidiExportErrorV1::TimelineInvalid {
            issues: timeline_report.issues,
        });
    }
    if !score.tempo_bpm.is_finite()
        || score.tempo_bpm <= 0.0
        || score.meter == 0
    {
        return Err(PartAwareMidiExportErrorV1::InvalidTempoOrMeter);
    }

    let used_instruments: BTreeSet<String> = timeline
        .segments
        .iter()
        .map(|segment| segment.instrument_id.clone())
        .collect();
    let mut binding_by_instrument = BTreeMap::<String, &MidiProgramBindingV1>::new();
    for binding in program_bindings {
        if binding.instrument_id.trim().is_empty()
            || binding.mapping_reference.trim().is_empty()
            || binding.program > 127
        {
            return Err(PartAwareMidiExportErrorV1::InvalidProgramBinding {
                detail: "instrument ID, mapping reference and 0..=127 program are required".into(),
            });
        }
        if binding_by_instrument
            .insert(binding.instrument_id.clone(), binding)
            .is_some()
        {
            return Err(PartAwareMidiExportErrorV1::InvalidProgramBinding {
                detail: format!("duplicate program binding for {}", binding.instrument_id),
            });
        }
    }
    for instrument_id in &used_instruments {
        if !binding_by_instrument.contains_key(instrument_id) {
            return Err(PartAwareMidiExportErrorV1::MissingProgramBinding {
                instrument_id: instrument_id.clone(),
            });
        }
    }
    for instrument_id in binding_by_instrument.keys() {
        if !used_instruments.contains(instrument_id) {
            return Err(PartAwareMidiExportErrorV1::UnusedProgramBinding {
                instrument_id: instrument_id.clone(),
            });
        }
    }

    // General MIDI convention reserves channel 10 (zero-based channel 9)
    // for percussion. We allocate one stable channel per PartId so
    // simultaneous same-pitch notes in independent parts remain distinct.
    const CHANNELS_V1: [u8; 15] =
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15];
    let part_ids: Vec<u16> = score
        .parts()
        .into_iter()
        .map(|part| part.0)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    if part_ids.len() > CHANNELS_V1.len() {
        return Err(PartAwareMidiExportErrorV1::TooManyParts {
            actual: part_ids.len(),
            supported: CHANNELS_V1.len(),
        });
    }
    let channel_by_part: BTreeMap<u16, u8> = part_ids
        .iter()
        .enumerate()
        .map(|(index, part_id)| (*part_id, CHANNELS_V1[index]))
        .collect();
    let track_index_by_part: BTreeMap<u16, u16> = part_ids
        .iter()
        .enumerate()
        .map(|(index, part_id)| (*part_id, (index + 1) as u16))
        .collect();

    let usec_per_beat = (60_000_000.0_f64 / f64::from(score.tempo_bpm)).round();
    if !usec_per_beat.is_finite() || usec_per_beat < 1.0 || usec_per_beat > f64::from(u32::MAX) {
        return Err(PartAwareMidiExportErrorV1::InvalidTempoOrMeter);
    }

    let tempo_track = Track {
        name: "Tempo".into(),
        channel: 0,
        events: vec![
            MidiEvent {
                tick: 0,
                channel: 0,
                event_type: MidiEventType::Tempo(usec_per_beat as u32),
            },
            MidiEvent {
                tick: 0,
                channel: 0,
                event_type: MidiEventType::TimeSignature(score.meter, 4),
            },
        ],
    };

    let mut events_by_part = BTreeMap::<u16, Vec<MidiEvent>>::new();
    for part_id in &part_ids {
        events_by_part.insert(*part_id, Vec::new());
    }

    // Program changes are emitted at every timeline-segment boundary.
    // A continuing PartId keeps its MIDI channel while its instrument
    // instance/patch changes explicitly at the handoff.
    let mut program_at_tick =
        BTreeMap::<(u16, u64), (u8, String)>::new();
    let mut program_change_receipts = Vec::with_capacity(timeline.segments.len());
    for segment in &timeline.segments {
        let tick = rational_beat_to_tick(segment.start).ok_or(
            PartAwareMidiExportErrorV1::InvalidRationalTime {
                score_note_ordinal: usize::MAX,
            },
        )?;
        let binding = binding_by_instrument
            .get(&segment.instrument_id)
            .expect("all timeline instruments have validated program bindings");
        let channel = *channel_by_part
            .get(&segment.part_id)
            .expect("validated score part has a MIDI channel");
        if let Some((previous_program, previous_instrument_id)) =
            program_at_tick.insert(
                (segment.part_id, tick),
                (binding.program, segment.instrument_id.clone()),
            )
        {
            // Even identical GM program numbers cannot distinguish two
            // different concrete instrument instances on the same channel.
            if previous_program != binding.program
                || previous_instrument_id != segment.instrument_id
            {
                return Err(
                    PartAwareMidiExportErrorV1::ProgramHandoffQuantizationCollision {
                        part_id: segment.part_id,
                        tick,
                    },
                );
            }
        }
        events_by_part
            .get_mut(&segment.part_id)
            .expect("validated score part has an event stream")
            .push(MidiEvent {
                tick,
                channel,
                event_type: MidiEventType::ProgramChange(binding.program),
            });
        program_change_receipts.push(PartAwareMidiProgramChangeReceiptV1 {
            part_id: segment.part_id,
            segment_id: segment.id.clone(),
            instrument_id: segment.instrument_id.clone(),
            track_index: *track_index_by_part
                .get(&segment.part_id)
                .expect("validated score part has a MIDI track"),
            channel,
            program: binding.program,
            mapping_reference: binding.mapping_reference.clone(),
            source_start: segment.start,
            start_tick: tick,
        });
    }

    let mut event_map = Vec::with_capacity(score.notes.len());
    let mut same_pitch_intervals =
        BTreeMap::<(u16, u8), Vec<(i128, i128, i128, i128, usize)>>::new();

    for (ordinal, note) in score.notes.iter().enumerate() {
        let mut matched_segment = None;
        for segment in timeline
            .segments
            .iter()
            .filter(|segment| segment.part_id == note.part.0)
        {
            if note_fits_segment(note, segment).ok_or(
                PartAwareMidiExportErrorV1::InvalidRationalTime {
                    score_note_ordinal: ordinal,
                },
            )? {
                if matched_segment.is_some() {
                    return Err(
                        PartAwareMidiExportErrorV1::MissingOrAmbiguousSegment {
                            score_note_ordinal: ordinal,
                        },
                    );
                }
                matched_segment = Some(segment);
            }
        }
        let segment = matched_segment.ok_or(
            PartAwareMidiExportErrorV1::MissingOrAmbiguousSegment {
                score_note_ordinal: ordinal,
            },
        )?;

        let (on_num, on_den) = duration_pair(note.onset);
        let (end_num, end_den) = note_end_pair(note).ok_or(
            PartAwareMidiExportErrorV1::InvalidRationalTime {
                score_note_ordinal: ordinal,
            },
        )?;
        let onset_tick = rational_to_tick(on_num, on_den).ok_or(
            PartAwareMidiExportErrorV1::MidiTickOverflow {
                score_note_ordinal: ordinal,
            },
        )?;
        let offset_tick = rational_to_tick(end_num, end_den).ok_or(
            PartAwareMidiExportErrorV1::MidiTickOverflow {
                score_note_ordinal: ordinal,
            },
        )?;
        if offset_tick <= onset_tick {
            return Err(PartAwareMidiExportErrorV1::UnrepresentableNoteDuration {
                score_note_ordinal: ordinal,
            });
        }

        let velocity = midi_velocity(note.velocity, ordinal)?;
        let channel = *channel_by_part
            .get(&note.part.0)
            .expect("validated score part has a MIDI channel");
        let track_index = *track_index_by_part
            .get(&note.part.0)
            .expect("validated score part has a MIDI track");

        same_pitch_intervals
            .entry((note.part.0, note.pitch.midi()))
            .or_default()
            .push((on_num, on_den, end_num, end_den, ordinal));

        let events = events_by_part
            .get_mut(&note.part.0)
            .expect("validated score part has an event stream");
        events.push(MidiEvent {
            tick: onset_tick,
            channel,
            event_type: MidiEventType::NoteOn {
                note: note.pitch.midi(),
                velocity,
            },
        });
        events.push(MidiEvent {
            tick: offset_tick,
            channel,
            event_type: MidiEventType::NoteOff {
                note: note.pitch.midi(),
            },
        });

        event_map.push(PartAwareMidiEventMapV1 {
            score_sha256: expected_score_sha256.to_owned(),
            score_note_ordinal: ordinal,
            part_id: note.part.0,
            segment_id: segment.id.clone(),
            instrument_id: segment.instrument_id.clone(),
            track_index,
            channel,
            midi_pitch: note.pitch.midi(),
            source_velocity: note.velocity,
            midi_velocity: velocity,
            source_onset: RationalBeatV1 {
                numerator: note.onset.num(),
                denominator: note.onset.den(),
            },
            source_duration: RationalBeatV1 {
                numerator: note.duration.num(),
                denominator: note.duration.den(),
            },
            onset_tick,
            offset_tick,
        });
    }

    // Standard MIDI NoteOff addresses channel+pitch, not a unique source
    // event. Compare exact rational source intervals, not quantized ticks:
    // two source overlaps must not become an apparently clean retrigger
    // merely because both boundaries round to the same MIDI tick.
    for ((part_id, pitch), intervals) in &mut same_pitch_intervals {
        intervals.sort_by(|left, right| {
            cmp_rational(left.0, left.1, right.0, right.1)
                .expect("validated score rational times are inside the bounded domain")
                .then_with(|| left.4.cmp(&right.4))
        });
        let mut previous_max_end: Option<(i128, i128, usize)> = None;
        for (onset_num, onset_den, end_num, end_den, ordinal) in
            intervals.iter().copied()
        {
            if let Some((previous_end_num, previous_end_den, previous_ordinal)) =
                previous_max_end
            {
                let onset_vs_previous_end = cmp_rational(
                    onset_num,
                    onset_den,
                    previous_end_num,
                    previous_end_den,
                )
                .ok_or(PartAwareMidiExportErrorV1::InvalidRationalTime {
                    score_note_ordinal: ordinal,
                })?;
                if onset_vs_previous_end == Ordering::Less {
                    return Err(
                        PartAwareMidiExportErrorV1::OverlappingSamePitchNotes {
                            part_id: *part_id,
                            pitch: *pitch,
                            first_score_note_ordinal: previous_ordinal,
                            second_score_note_ordinal: ordinal,
                        },
                    );
                }
                let end_vs_previous_end = cmp_rational(
                    end_num,
                    end_den,
                    previous_end_num,
                    previous_end_den,
                )
                .ok_or(PartAwareMidiExportErrorV1::InvalidRationalTime {
                    score_note_ordinal: ordinal,
                })?;
                if end_vs_previous_end == Ordering::Greater {
                    previous_max_end = Some((end_num, end_den, ordinal));
                }
            } else {
                previous_max_end = Some((end_num, end_den, ordinal));
            }
        }
    }

    let mut tracks = Vec::with_capacity(part_ids.len() + 1);
    for part_id in &part_ids {
        let channel = channel_by_part[part_id];
        let mut events = events_by_part.remove(part_id).unwrap_or_default();
        events.sort_by_key(event_ordering_key);
        let mut previous_tick = 0_u64;
        for event in &events {
            let delta = event.tick.saturating_sub(previous_tick);
            if delta > 0x0fff_ffff {
                return Err(
                    PartAwareMidiExportErrorV1::MidiDeltaExceedsStandardMaximum {
                        track_index: (tracks.len() + 1) as u16,
                        delta_ticks: delta,
                    },
                );
            }
            previous_tick = event.tick;
        }
        tracks.push(Track {
            name: format!("Part-{}", part_id),
            channel,
            events,
        });
    }
    let track_receipts: Vec<PartAwareMidiTrackReceiptV1> = part_ids
        .iter()
        .enumerate()
        .map(|(part_index, part_id)| {
            let instrument_ids: Vec<String> = timeline
                .segments
                .iter()
                .filter(|segment| segment.part_id == *part_id)
                .map(|segment| segment.instrument_id.clone())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
            let mapping_references: Vec<String> = instrument_ids
                .iter()
                .filter_map(|instrument_id| {
                    binding_by_instrument
                        .get(instrument_id)
                        .map(|binding| binding.mapping_reference.clone())
                })
                .collect();
            PartAwareMidiTrackReceiptV1 {
                part_id: *part_id,
                track_index: (part_index + 1) as u16,
                channel: channel_by_part[part_id],
                instrument_ids,
                mapping_references,
                note_count: event_map
                    .iter()
                    .filter(|mapping| mapping.part_id == *part_id)
                    .count(),
            }
        })
        .collect();

    // Tempo track precedes all persistent-part tracks in deterministic order.
    let mut smf_tracks = Vec::with_capacity(tracks.len() + 1);
    smf_tracks.push(tempo_track);
    smf_tracks.extend(tracks);
    let midi_bytes = encode_smf_type_one(&smf_tracks);
    let midi_sha256 = sha256_hex(&midi_bytes);

    let timeline_bytes = serde_json::to_vec(timeline).map_err(|error| {
        PartAwareMidiExportErrorV1::TimelineSerialization(error.to_string())
    })?;
    let timeline_sha256 = domain_separated_sha256(
        PART_AWARE_MIDI_TIMELINE_DIGEST_SCHEMA_V1,
        &timeline_bytes,
    );

    program_change_receipts.sort_by_key(|item| {
        (item.part_id, item.start_tick, item.segment_id.clone())
    });
    let receipt = PartAwareMidiExportReceiptV1 {
        schema: PART_AWARE_MIDI_EXPORT_SCHEMA_V1.into(),
        score_digest_schema: ORCHESTRATION_SCORE_DIGEST_SCHEMA_V1.into(),
        score_sha256: expected_score_sha256.to_owned(),
        timeline_digest_schema: PART_AWARE_MIDI_TIMELINE_DIGEST_SCHEMA_V1.into(),
        timeline_sha256,
        midi_sha256,
        ticks_per_quarter: MIDI_TICKS_PER_QUARTER_V1,
        part_count: part_ids.len(),
        note_count: event_map.len(),
        track_count: smf_tracks.len(),
        tracks: track_receipts,
        program_changes: program_change_receipts,
        event_map,
    };

    Ok(PartAwareMidiArtifactV1 {
        midi_bytes,
        receipt,
    })
}

/// Write the part-aware MIDI bytes to disk and return the same content
/// receipt. The file hash is over the exact bytes written.
pub fn export_score_midi_part_aware_v1(
    score: &Score,
    timeline: &OrchestrationTimelineV1,
    instruments: &[InstrumentCapabilityV1],
    program_bindings: &[MidiProgramBindingV1],
    expected_score_sha256: &str,
    output_path: &std::path::Path,
) -> Result<PartAwareMidiExportReceiptV1, PartAwareMidiExportErrorV1> {
    let artifact = encode_score_midi_part_aware_v1(
        score,
        timeline,
        instruments,
        program_bindings,
        expected_score_sha256,
    )?;
    std::fs::write(output_path, &artifact.midi_bytes)
        .map_err(|error| PartAwareMidiExportErrorV1::MidiWrite(error.to_string()))?;
    let written = std::fs::read(output_path)
        .map_err(|error| PartAwareMidiExportErrorV1::MidiWrite(error.to_string()))?;
    let written_sha256 = sha256_hex(&written);
    if written_sha256 != artifact.receipt.midi_sha256 {
        return Err(PartAwareMidiExportErrorV1::MidiWrite(
            "written MIDI bytes do not match the encoded artifact digest".into(),
        ));
    }
    Ok(artifact.receipt)
}

fn encode_smf_type_one(tracks: &[Track]) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"MThd");
    bytes.extend_from_slice(&6_u32.to_be_bytes());
    bytes.extend_from_slice(&1_u16.to_be_bytes());
    bytes.extend_from_slice(&(tracks.len() as u16).to_be_bytes());
    bytes.extend_from_slice(&MIDI_TICKS_PER_QUARTER_V1.to_be_bytes());

    for track in tracks {
        let track_bytes = encode_track(track);
        bytes.extend_from_slice(b"MTrk");
        bytes.extend_from_slice(&(track_bytes.len() as u32).to_be_bytes());
        bytes.extend_from_slice(&track_bytes);
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_music_theory::harmony::Key;
    use symthaea_music_theory::pitch::{Pitch, PitchClass};
    use symthaea_music_theory::score::{
        Emphasis, PartId, Score, ScoreNote, VoiceRole,
    };
    use symthaea_music_theory::orchestration::{
        InstrumentKindV1, SoundingRangeV1,
    };
    use symthaea_music_theory::orchestration_timeline::{
        HandoffKindV1, OrchestrationSegmentV1, SegmentHandoffV1,
        ORCHESTRATION_TIMELINE_SCHEMA_V1,
    };

    fn score_digest(score: &Score) -> String {
        let bytes = serde_json::to_vec(score).expect("test score serializes");
        sha256_hex(&bytes)
    }

    fn note(part: u16, midi: u8, onset: Duration, duration: Duration) -> ScoreNote {
        ScoreNote {
            part: PartId(part),
            pitch: Pitch::from_midi(midi),
            onset,
            duration,
            velocity: 0.72,
            role: VoiceRole::Melody,
            emphasis: Emphasis::Normal,
            section_intensity: 1.0,
        }
    }

    fn cap(id: &str, kind: InstrumentKindV1, low: u8, high: u8, polyphony: u8) -> InstrumentCapabilityV1 {
        InstrumentCapabilityV1 {
            id: id.into(),
            kind,
            sounding_range: SoundingRangeV1 {
                lowest_midi: low,
                highest_midi: high,
            },
            max_polyphony: polyphony,
        }
    }

    fn segment(
        id: &str,
        part_id: u16,
        instrument_id: &str,
        start_num: i64,
        start_den: i64,
        end_num: i64,
        end_den: i64,
        handoff: Option<SegmentHandoffV1>,
    ) -> OrchestrationSegmentV1 {
        OrchestrationSegmentV1 {
            id: id.into(),
            part_id,
            instrument_id: instrument_id.into(),
            start: RationalBeatV1 {
                numerator: start_num,
                denominator: start_den,
            },
            end: RationalBeatV1 {
                numerator: end_num,
                denominator: end_den,
            },
            musical_function: "authored_test_part".into(),
            handoff,
        }
    }

    fn score(notes: Vec<ScoreNote>) -> Score {
        let mut score = Score::new(Key::major(PitchClass::C), 120.0, 4);
        for item in notes {
            score.push(item);
        }
        score
    }

    fn binding(id: &str, program: u8) -> MidiProgramBindingV1 {
        MidiProgramBindingV1 {
            instrument_id: id.into(),
            program,
            mapping_reference: format!("test-map:{id}:{program}"),
        }
    }

    fn plan(score_sha256: &str, segments: Vec<OrchestrationSegmentV1>) -> OrchestrationTimelineV1 {
        OrchestrationTimelineV1 {
            schema: ORCHESTRATION_TIMELINE_SCHEMA_V1.into(),
            score_digest_schema: ORCHESTRATION_SCORE_DIGEST_SCHEMA_V1.into(),
            score_sha256: score_sha256.into(),
            segments,
        }
    }

    fn assert_parsed_midi_matches_receipt(artifact: &PartAwareMidiArtifactV1) {
        let smf = midly::Smf::parse(&artifact.midi_bytes)
            .expect("part-aware exporter must emit a parseable Standard MIDI File");
        assert_eq!(smf.tracks.len(), artifact.receipt.track_count);
        assert!(matches!(
            smf.header.timing,
            midly::Timing::Metrical(ticks)
                if ticks.as_int() == MIDI_TICKS_PER_QUARTER_V1
        ));

        let mut actual_note_ons = Vec::<(u16, u8, u8, u8, u64)>::new();
        let mut actual_note_offs = Vec::<(u16, u8, u8, u64)>::new();
        let mut actual_program_changes = Vec::<(u16, u8, u8, u64)>::new();
        for (track_index, track) in smf.tracks.iter().enumerate() {
            let mut absolute_tick = 0_u64;
            for event in track {
                absolute_tick += u64::from(event.delta.as_int());
                let midly::TrackEventKind::Midi { channel, message } = &event.kind else {
                    continue;
                };
                let channel = channel.as_int();
                match message {
                    midly::MidiMessage::NoteOn { key, vel } if vel.as_int() > 0 => {
                        actual_note_ons.push((
                            track_index as u16,
                            channel,
                            key.as_int(),
                            vel.as_int(),
                            absolute_tick,
                        ));
                    }
                    midly::MidiMessage::NoteOff { key, .. } => {
                        actual_note_offs.push((
                            track_index as u16,
                            channel,
                            key.as_int(),
                            absolute_tick,
                        ));
                    }
                    midly::MidiMessage::NoteOn { key, vel } if vel.as_int() == 0 => {
                        actual_note_offs.push((
                            track_index as u16,
                            channel,
                            key.as_int(),
                            absolute_tick,
                        ));
                    }
                    midly::MidiMessage::ProgramChange { program } => {
                        actual_program_changes.push((
                            track_index as u16,
                            channel,
                            program.as_int(),
                            absolute_tick,
                        ));
                    }
                    _ => {}
                }
            }
        }

        let mut expected_note_ons: Vec<_> = artifact
            .receipt
            .event_map
            .iter()
            .map(|event| {
                (
                    event.track_index,
                    event.channel,
                    event.midi_pitch,
                    event.midi_velocity,
                    event.onset_tick,
                )
            })
            .collect();
        let mut expected_note_offs: Vec<_> = artifact
            .receipt
            .event_map
            .iter()
            .map(|event| {
                (
                    event.track_index,
                    event.channel,
                    event.midi_pitch,
                    event.offset_tick,
                )
            })
            .collect();
        let mut expected_program_changes: Vec<_> = artifact
            .receipt
            .program_changes
            .iter()
            .map(|event| {
                (
                    event.track_index,
                    event.channel,
                    event.program,
                    event.start_tick,
                )
            })
            .collect();

        actual_note_ons.sort_unstable();
        actual_note_offs.sort_unstable();
        actual_program_changes.sort_unstable();
        expected_note_ons.sort_unstable();
        expected_note_offs.sort_unstable();
        expected_program_changes.sort_unstable();

        assert_eq!(actual_note_ons, expected_note_ons);
        assert_eq!(actual_note_offs, expected_note_offs);
        assert_eq!(actual_program_changes, expected_program_changes);
    }

    #[test]
    fn changed_score_is_rejected_when_caller_reuses_a_stale_digest() {
        let source = score(vec![note(1, 60, Duration::zero(), Duration::quarter())]);
        let digest = score_digest(&source);
        let timeline = plan(
            &digest,
            vec![segment("p1", 1, "cello-1", 0, 1, 2, 1, None)],
        );
        let instruments = vec![cap("cello-1", InstrumentKindV1::Cello, 36, 84, 1)];
        let bindings = vec![binding("cello-1", 42)];
        let mut changed_score = source.clone();
        changed_score.notes[0].velocity = 0.93;

        let result = encode_score_midi_part_aware_v1(
            &changed_score,
            &timeline,
            &instruments,
            &bindings,
            &digest,
        );
        assert!(matches!(
            result,
            Err(PartAwareMidiExportErrorV1::ScoreDigestMismatch { .. })
        ));
    }

    #[test]
    fn different_instruments_cannot_share_a_quantized_handoff_tick() {
        let source = score(vec![note(
            7,
            60,
            Duration::new(1, 10_000),
            Duration::new(1, 2),
        )]);
        let digest = score_digest(&source);
        // These adjacent rational boundaries differ but both quantize to tick 0.
        let timeline = plan(
            &digest,
            vec![
                segment("opening", 7, "cello-1", 0, 1, 1, 10_000, None),
                segment(
                    "response",
                    7,
                    "flute-1",
                    1,
                    10_000,
                    2,
                    1,
                    Some(SegmentHandoffV1 {
                        from_segment_id: "opening".into(),
                        kind: HandoffKindV1::Reattack,
                    }),
                ),
            ],
        );
        let instruments = vec![
            cap("cello-1", InstrumentKindV1::Cello, 36, 84, 1),
            cap("flute-1", InstrumentKindV1::Flute, 60, 96, 1),
        ];
        let bindings = vec![binding("cello-1", 42), binding("flute-1", 42)];

        let result = encode_score_midi_part_aware_v1(
            &source,
            &timeline,
            &instruments,
            &bindings,
            &digest,
        );
        assert!(matches!(
            result,
            Err(PartAwareMidiExportErrorV1::ProgramHandoffQuantizationCollision { .. })
        ));
    }

    #[test]
    fn independent_parts_with_the_same_voice_role_get_separate_midi_tracks() {
        let source = score(vec![
            note(1, 60, Duration::zero(), Duration::quarter()),
            note(2, 64, Duration::zero(), Duration::quarter()),
        ]);
        let digest = score_digest(&source);
        let timeline = plan(
            &digest,
            vec![
                segment("part-1", 1, "cello-1", 0, 1, 2, 1, None),
                segment("part-2", 2, "horn-1", 0, 1, 2, 1, None),
            ],
        );
        let instruments = vec![
            cap("cello-1", InstrumentKindV1::Cello, 36, 84, 1),
            cap("horn-1", InstrumentKindV1::FrenchHorn, 48, 91, 1),
        ];
        let bindings = vec![binding("cello-1", 42), binding("horn-1", 60)];

        let artifact = encode_score_midi_part_aware_v1(
            &source,
            &timeline,
            &instruments,
            &bindings,
            &digest,
        )
        .expect("valid part-aware score should encode");

        assert_eq!(&artifact.midi_bytes[0..4], b"MThd");
        assert_eq!(u16::from_be_bytes([artifact.midi_bytes[10], artifact.midi_bytes[11]]), 3);
        assert_eq!(artifact.receipt.part_count, 2);
        assert_eq!(artifact.receipt.track_count, 3);
        assert_eq!(artifact.receipt.event_map.len(), 2);
        assert_ne!(
            artifact.receipt.event_map[0].track_index,
            artifact.receipt.event_map[1].track_index
        );
        assert_ne!(
            artifact.receipt.event_map[0].channel,
            artifact.receipt.event_map[1].channel
        );
        assert!(artifact
            .receipt
            .event_map
            .iter()
            .all(|mapping| mapping.score_sha256 == digest));
        assert_parsed_midi_matches_receipt(&artifact);
    }

    #[test]
    fn instrument_handoff_keeps_part_channel_and_changes_program_at_boundary() {
        let source = score(vec![
            note(7, 60, Duration::zero(), Duration::quarter()),
            note(7, 62, Duration::new(2, 1), Duration::quarter()),
        ]);
        let digest = score_digest(&source);
        let timeline = plan(
            &digest,
            vec![
                segment("opening", 7, "cello-1", 0, 1, 2, 1, None),
                segment(
                    "response",
                    7,
                    "flute-1",
                    2,
                    1,
                    4,
                    1,
                    Some(SegmentHandoffV1 {
                        from_segment_id: "opening".into(),
                        kind: HandoffKindV1::Reattack,
                    }),
                ),
            ],
        );
        let instruments = vec![
            cap("cello-1", InstrumentKindV1::Cello, 36, 84, 1),
            cap("flute-1", InstrumentKindV1::Flute, 60, 96, 1),
        ];
        let bindings = vec![binding("cello-1", 42), binding("flute-1", 73)];

        let artifact = encode_score_midi_part_aware_v1(
            &source,
            &timeline,
            &instruments,
            &bindings,
            &digest,
        )
        .expect("valid handoff should encode");

        assert_eq!(artifact.receipt.event_map[0].part_id, artifact.receipt.event_map[1].part_id);
        assert_eq!(artifact.receipt.event_map[0].channel, artifact.receipt.event_map[1].channel);
        assert_ne!(artifact.receipt.event_map[0].instrument_id, artifact.receipt.event_map[1].instrument_id);
        assert_eq!(artifact.receipt.event_map[1].onset_tick, 2 * u64::from(MIDI_TICKS_PER_QUARTER_V1));
        assert_eq!(artifact.receipt.tracks[0].instrument_ids.len(), 2);
        assert_eq!(artifact.receipt.program_changes.len(), 2);
        assert_eq!(artifact.receipt.program_changes[0].program, 42);
        assert_eq!(artifact.receipt.program_changes[1].program, 73);
        assert_eq!(artifact.receipt.program_changes[1].segment_id, "response");
        assert_parsed_midi_matches_receipt(&artifact);
    }

    #[test]
    fn missing_program_binding_is_rejected() {
        let source = score(vec![note(1, 60, Duration::zero(), Duration::quarter())]);
        let digest = score_digest(&source);
        let timeline = plan(
            &digest,
            vec![segment("p1", 1, "cello-1", 0, 1, 2, 1, None)],
        );
        let instruments = vec![cap("cello-1", InstrumentKindV1::Cello, 36, 84, 1)];

        let result = encode_score_midi_part_aware_v1(
            &source,
            &timeline,
            &instruments,
            &[],
            &digest,
        );
        assert!(matches!(
            result,
            Err(PartAwareMidiExportErrorV1::MissingProgramBinding { .. })
        ));
    }

    #[test]
    fn missing_part_identity_is_rejected_instead_of_grouped_by_role() {
        let mut source = score(vec![note(1, 60, Duration::zero(), Duration::quarter())]);
        source.notes[0].part = PartId::UNASSIGNED;
        let digest = score_digest(&source);
        let timeline = plan(
            &digest,
            vec![segment("p1", 1, "cello-1", 0, 1, 2, 1, None)],
        );
        let instruments = vec![cap("cello-1", InstrumentKindV1::Cello, 36, 84, 1)];
        let bindings = vec![binding("cello-1", 42)];

        let result = encode_score_midi_part_aware_v1(
            &source,
            &timeline,
            &instruments,
            &bindings,
            &digest,
        );
        assert!(matches!(
            result,
            Err(PartAwareMidiExportErrorV1::TimelineInvalid { .. })
        ));
    }

    #[test]
    fn same_pitch_overlap_is_rejected_instead_of_silently_clipped() {
        let source = score(vec![
            note(1, 60, Duration::zero(), Duration::new(2, 1)),
            note(1, 60, Duration::quarter(), Duration::new(2, 1)),
        ]);
        let digest = score_digest(&source);
        let timeline = plan(
            &digest,
            vec![segment("p1", 1, "cello-1", 0, 1, 4, 1, None)],
        );
        let instruments = vec![cap("cello-1", InstrumentKindV1::Cello, 36, 84, 2)];
        let bindings = vec![binding("cello-1", 42)];

        let result = encode_score_midi_part_aware_v1(
            &source,
            &timeline,
            &instruments,
            &bindings,
            &digest,
        );
        assert!(matches!(
            result,
            Err(PartAwareMidiExportErrorV1::OverlappingSamePitchNotes { .. })
        ));
    }

    #[test]
    fn rational_tick_quantization_uses_nearest_ties_to_even() {
        assert_eq!(rational_to_tick(1, 1_920), Some(0));
        assert_eq!(rational_to_tick(3, 1_920), Some(2));
        assert_eq!(rational_to_tick(5, 1_920), Some(2));
        assert_eq!(rational_to_tick(7, 1_920), Some(4));
    }

    #[test]
    fn source_overlap_is_rejected_even_when_tick_quantization_hides_it() {
        let source = score(vec![
            note(1, 60, Duration::zero(), Duration::new(1, 2)),
            note(
                1,
                60,
                Duration::new(959, 1_920),
                Duration::new(1, 2),
            ),
        ]);
        let digest = score_digest(&source);
        let timeline = plan(
            &digest,
            vec![segment("p1", 1, "cello-1", 0, 1, 2, 1, None)],
        );
        let instruments = vec![cap("cello-1", InstrumentKindV1::Cello, 36, 84, 2)];
        let bindings = vec![binding("cello-1", 42)];

        // The second onset is 479.5 MIDI ticks and rounds to tick 480,
        // which is also the first note's offset. The source events still
        // overlap by 1/1920 beat and must not be silently normalized.
        assert_eq!(
            rational_to_tick(959, 1_920),
            Some(480)
        );
        let result = encode_score_midi_part_aware_v1(
            &source,
            &timeline,
            &instruments,
            &bindings,
            &digest,
        );
        assert!(matches!(
            result,
            Err(PartAwareMidiExportErrorV1::OverlappingSamePitchNotes { .. })
        ));
    }

    #[test]
    fn sub_tick_duration_is_rejected_instead_of_dropped_or_stretched() {
        let source = score(vec![note(
            1,
            60,
            Duration::zero(),
            Duration::new(1, 10_000),
        )]);
        let digest = score_digest(&source);
        let timeline = plan(
            &digest,
            vec![segment("p1", 1, "cello-1", 0, 1, 1, 1, None)],
        );
        let instruments = vec![cap("cello-1", InstrumentKindV1::Cello, 36, 84, 1)];
        let bindings = vec![binding("cello-1", 42)];

        let result = encode_score_midi_part_aware_v1(
            &source,
            &timeline,
            &instruments,
            &bindings,
            &digest,
        );
        assert!(matches!(
            result,
            Err(PartAwareMidiExportErrorV1::UnrepresentableNoteDuration { .. })
        ));
    }

    #[test]
    fn midi_delta_beyond_standard_vlq_limit_is_rejected() {
        let source = score(vec![note(
            1,
            60,
            Duration::new(0x1000_0000, 1),
            Duration::quarter(),
        )]);
        let digest = score_digest(&source);
        let timeline = plan(
            &digest,
            vec![segment("p1", 1, "cello-1", 0, 1, 1_000_000_000, 1, None)],
        );
        let instruments = vec![cap("cello-1", InstrumentKindV1::Cello, 36, 84, 1)];
        let bindings = vec![binding("cello-1", 42)];

        let result = encode_score_midi_part_aware_v1(
            &source,
            &timeline,
            &instruments,
            &bindings,
            &digest,
        );
        assert!(matches!(
            result,
            Err(PartAwareMidiExportErrorV1::MidiDeltaExceedsStandardMaximum { .. })
        ));
    }

    #[test]
    fn event_mapping_and_midi_hash_are_deterministic_for_frozen_inputs() {
        let source = score(vec![
            note(1, 60, Duration::zero(), Duration::quarter()),
            note(2, 67, Duration::zero(), Duration::quarter()),
        ]);
        let digest = score_digest(&source);
        let timeline = plan(
            &digest,
            vec![
                segment("p1", 1, "cello-1", 0, 1, 2, 1, None),
                segment("p2", 2, "horn-1", 0, 1, 2, 1, None),
            ],
        );
        let instruments = vec![
            cap("cello-1", InstrumentKindV1::Cello, 36, 84, 1),
            cap("horn-1", InstrumentKindV1::FrenchHorn, 48, 91, 1),
        ];
        let bindings = vec![binding("cello-1", 42), binding("horn-1", 60)];

        let first = encode_score_midi_part_aware_v1(
            &source, &timeline, &instruments, &bindings, &digest,
        )
        .expect("first encoding");
        let second = encode_score_midi_part_aware_v1(
            &source, &timeline, &instruments, &bindings, &digest,
        )
        .expect("second encoding");
        assert_eq!(first.midi_bytes, second.midi_bytes);
        assert_eq!(first.receipt, second.receipt);
        assert_eq!(first.receipt.midi_sha256, sha256_hex(&first.midi_bytes));
    }
}
