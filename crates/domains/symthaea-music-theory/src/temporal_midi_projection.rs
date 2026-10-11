// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Pure projection of work-scale temporal authority into Standard MIDI File
//! timing/meta-event semantics.
//!
//! This module does not write bytes. It freezes the semantic loss boundary
//! before an exporter serializes anything: exact rational beat offsets must be
//! representable at the chosen PPQ, rational BPM is rounded to MIDI's integer
//! microseconds-per-quarter with an exact error receipt, and additive meter
//! grouping is retained in a marker because the standard time-signature meta
//! event cannot encode it.

use crate::rhythm::Duration;
use crate::temporal_map::TempoV1;
use crate::temporal_score::{TemporalScoreErrorV1, TemporalScoreV1};
use serde::{Deserialize, Serialize};

pub const MIDI_TEMPORAL_PROJECTION_VERSION: &str = "melothaea-midi-temporal-projection-v1";
pub const DEFAULT_MIDI_TICKS_PER_QUARTER: u16 = 480;
const MIDI_MAX_TEMPO_USEC: u32 = 0x00ff_ffff;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MidiTemporalProjectionPolicyV1 {
    pub ticks_per_quarter: u16,
    /// Standard MIDI cannot represent additive grouping, so V1 can retain it
    /// as a text marker at each meter change.
    pub emit_grouping_marker: bool,
}

impl Default for MidiTemporalProjectionPolicyV1 {
    fn default() -> Self {
        Self {
            ticks_per_quarter: DEFAULT_MIDI_TICKS_PER_QUARTER,
            emit_grouping_marker: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MidiTemporalMetaKindV1 {
    Tempo {
        microseconds_per_quarter: u32,
    },
    TimeSignature {
        numerator: u8,
        /// Standard MIDI stores the denominator as log2(denominator).
        denominator_power: u8,
    },
    Marker {
        text: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MidiTemporalMetaEventV1 {
    pub tick: u64,
    pub kind: MidiTemporalMetaKindV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TempoMidiProjectionReceiptV1 {
    pub at: Duration,
    pub source_tempo: TempoV1,
    pub microseconds_per_quarter: u32,
    /// Exact signed projection error in microseconds:
    /// `error_numerator / error_denominator`.
    /// Negative means the rounded MIDI quarter is slightly shorter.
    pub error_numerator: i64,
    pub error_denominator: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeterMidiProjectionReceiptV1 {
    pub at: Duration,
    pub numerator: u8,
    pub denominator: u8,
    pub grouping: Vec<u8>,
    /// Always false for Standard MIDI's time-signature event: grouping is not
    /// part of that event's representation.
    pub standard_event_preserves_grouping: bool,
    pub grouping_marker_emitted: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MidiTemporalProjectionV1 {
    pub version: String,
    pub ticks_per_quarter: u16,
    pub events: Vec<MidiTemporalMetaEventV1>,
    pub tempo_receipts: Vec<TempoMidiProjectionReceiptV1>,
    pub meter_receipts: Vec<MeterMidiProjectionReceiptV1>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MidiTemporalProjectionErrorV1 {
    InvalidTemporalScore(TemporalScoreErrorV1),
    ZeroTicksPerQuarter,
    NegativeBeat,
    NonRepresentableBeat {
        numerator: i64,
        denominator: i64,
        ticks_per_quarter: u16,
    },
    TickOverflow,
    TempoArithmeticOverflow,
    MidiTempoOutOfRange {
        microseconds_per_quarter: u64,
    },
}

pub fn project_temporal_score_to_midi(
    temporal_score: &TemporalScoreV1,
    policy: MidiTemporalProjectionPolicyV1,
) -> Result<MidiTemporalProjectionV1, MidiTemporalProjectionErrorV1> {
    temporal_score
        .validate()
        .map_err(MidiTemporalProjectionErrorV1::InvalidTemporalScore)?;
    if policy.ticks_per_quarter == 0 {
        return Err(MidiTemporalProjectionErrorV1::ZeroTicksPerQuarter);
    }

    let mut events = Vec::new();
    let mut tempo_receipts = Vec::new();
    let mut meter_receipts = Vec::new();

    for point in &temporal_score.temporal_map.points {
        let tick = duration_to_ticks_exact(point.at, policy.ticks_per_quarter)?;

        if let Some(meter) = &point.meter {
            let denominator_power = meter.denominator().trailing_zeros() as u8;
            events.push(MidiTemporalMetaEventV1 {
                tick,
                kind: MidiTemporalMetaKindV1::TimeSignature {
                    numerator: meter.numerator(),
                    denominator_power,
                },
            });

            let marker_emitted = policy.emit_grouping_marker;
            if marker_emitted {
                let grouping = meter
                    .grouping()
                    .iter()
                    .map(u8::to_string)
                    .collect::<Vec<_>>()
                    .join("+");
                events.push(MidiTemporalMetaEventV1 {
                    tick,
                    kind: MidiTemporalMetaKindV1::Marker {
                        text: format!(
                            "melothaea:meter-grouping={}/{}:{}",
                            meter.numerator(),
                            meter.denominator(),
                            grouping
                        ),
                    },
                });
            }
            meter_receipts.push(MeterMidiProjectionReceiptV1 {
                at: point.at,
                numerator: meter.numerator(),
                denominator: meter.denominator(),
                grouping: meter.grouping().to_vec(),
                standard_event_preserves_grouping: false,
                grouping_marker_emitted: marker_emitted,
            });
        }

        if let Some(tempo) = point.tempo {
            let receipt = project_tempo(point.at, tempo)?;
            events.push(MidiTemporalMetaEventV1 {
                tick,
                kind: MidiTemporalMetaKindV1::Tempo {
                    microseconds_per_quarter: receipt.microseconds_per_quarter,
                },
            });
            tempo_receipts.push(receipt);
        }
    }

    // Points are already strictly increasing. Within one point, retain the
    // canonical TimeSignature -> grouping Marker -> Tempo order above.
    Ok(MidiTemporalProjectionV1 {
        version: MIDI_TEMPORAL_PROJECTION_VERSION.into(),
        ticks_per_quarter: policy.ticks_per_quarter,
        events,
        tempo_receipts,
        meter_receipts,
    })
}

fn duration_to_ticks_exact(
    duration: Duration,
    ticks_per_quarter: u16,
) -> Result<u64, MidiTemporalProjectionErrorV1> {
    if duration.num() < 0 {
        return Err(MidiTemporalProjectionErrorV1::NegativeBeat);
    }
    let scaled = i128::from(duration.num())
        .checked_mul(i128::from(ticks_per_quarter))
        .ok_or(MidiTemporalProjectionErrorV1::TickOverflow)?;
    let denominator = i128::from(duration.den());
    if scaled % denominator != 0 {
        return Err(MidiTemporalProjectionErrorV1::NonRepresentableBeat {
            numerator: duration.num(),
            denominator: duration.den(),
            ticks_per_quarter,
        });
    }
    u64::try_from(scaled / denominator).map_err(|_| MidiTemporalProjectionErrorV1::TickOverflow)
}

fn project_tempo(
    at: Duration,
    tempo: TempoV1,
) -> Result<TempoMidiProjectionReceiptV1, MidiTemporalProjectionErrorV1> {
    let numerator = u128::from(60_000_000u32)
        .checked_mul(u128::from(tempo.denominator()))
        .ok_or(MidiTemporalProjectionErrorV1::TempoArithmeticOverflow)?;
    let denominator = u128::from(tempo.numerator_bpm());
    let rounded = numerator
        .checked_add(denominator / 2)
        .ok_or(MidiTemporalProjectionErrorV1::TempoArithmeticOverflow)?
        / denominator;
    let rounded_u64 = u64::try_from(rounded)
        .map_err(|_| MidiTemporalProjectionErrorV1::TempoArithmeticOverflow)?;
    if rounded_u64 == 0 || rounded_u64 > u64::from(MIDI_MAX_TEMPO_USEC) {
        return Err(MidiTemporalProjectionErrorV1::MidiTempoOutOfRange {
            microseconds_per_quarter: rounded_u64,
        });
    }

    let rounded_scaled = i128::from(rounded_u64)
        .checked_mul(i128::from(tempo.numerator_bpm()))
        .ok_or(MidiTemporalProjectionErrorV1::TempoArithmeticOverflow)?;
    let source_scaled = i128::from(60_000_000u32)
        .checked_mul(i128::from(tempo.denominator()))
        .ok_or(MidiTemporalProjectionErrorV1::TempoArithmeticOverflow)?;
    let error = rounded_scaled - source_scaled;

    Ok(TempoMidiProjectionReceiptV1 {
        at,
        source_tempo: tempo,
        microseconds_per_quarter: rounded_u64 as u32,
        error_numerator: i64::try_from(error)
            .map_err(|_| MidiTemporalProjectionErrorV1::TempoArithmeticOverflow)?,
        error_denominator: tempo.numerator_bpm(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Emphasis, Key, PartId, Pitch, PitchClass, Score, ScoreNote, TemporalMapV1,
        TimeSignature, VoiceRole,
    };

    fn score(total_beats: i64, tempo: f32, meter: u8) -> Score {
        let mut score = Score::new(Key::major(PitchClass::C), tempo, meter);
        score.push(ScoreNote {
            part: PartId::UNASSIGNED,
            pitch: Pitch::from_midi(60),
            onset: Duration::zero(),
            duration: Duration::new(total_beats, 1),
            velocity: 0.7,
            role: VoiceRole::Melody,
            emphasis: Emphasis::Normal,
            section_intensity: 1.0,
        });
        score
    }

    #[test]
    fn exact_common_tempo_has_zero_projection_error() {
        let score = score(8, 120.0, 4);
        let map = TemporalMapV1::new(
            TempoV1::integer(120).unwrap(),
            TimeSignature::new(4, 4).unwrap(),
        );
        let bound = TemporalScoreV1::bind(score, map).unwrap();
        let projected = project_temporal_score_to_midi(&bound, Default::default()).unwrap();
        assert_eq!(projected.tempo_receipts[0].microseconds_per_quarter, 500_000);
        assert_eq!(projected.tempo_receipts[0].error_numerator, 0);
    }

    #[test]
    fn fractional_tempo_rounding_is_retained_as_exact_error() {
        let score = score(8, 121.5, 4);
        let map = TemporalMapV1::new(
            TempoV1::new(243, 2).unwrap(),
            TimeSignature::new(4, 4).unwrap(),
        );
        let bound = TemporalScoreV1::bind(score, map).unwrap();
        let projected = project_temporal_score_to_midi(&bound, Default::default()).unwrap();
        let receipt = &projected.tempo_receipts[0];
        assert_eq!(receipt.microseconds_per_quarter, 493_827);
        assert_eq!(receipt.error_numerator, -39);
        assert_eq!(receipt.error_denominator, 243);
    }

    #[test]
    fn additive_meter_grouping_survives_as_marker_and_receipt() {
        let score = score(16, 120.0, 4);
        let mut map = TemporalMapV1::new(
            TempoV1::integer(120).unwrap(),
            TimeSignature::new(4, 4).unwrap(),
        );
        map.push_change(
            Duration::new(8, 1),
            Some(TimeSignature::with_grouping(7, 8, vec![2, 2, 3]).unwrap()),
            None,
        )
        .unwrap();
        let bound = TemporalScoreV1::bind(score, map).unwrap();
        let projected = project_temporal_score_to_midi(&bound, Default::default()).unwrap();

        assert!(projected.events.iter().any(|event| {
            event.tick == 8 * u64::from(DEFAULT_MIDI_TICKS_PER_QUARTER)
                && matches!(
                    &event.kind,
                    MidiTemporalMetaKindV1::TimeSignature {
                        numerator: 7,
                        denominator_power: 3
                    }
                )
        }));
        assert!(projected.events.iter().any(|event| {
            matches!(
                &event.kind,
                MidiTemporalMetaKindV1::Marker { text }
                    if text == "melothaea:meter-grouping=7/8:2+2+3"
            )
        }));
        let receipt = projected.meter_receipts.last().unwrap();
        assert_eq!(receipt.grouping, vec![2, 2, 3]);
        assert!(!receipt.standard_event_preserves_grouping);
        assert!(receipt.grouping_marker_emitted);
    }

    #[test]
    fn denominator_power_supports_beyond_legacy_hardcoded_cases() {
        let score = score(8, 120.0, 4);
        let mut map = TemporalMapV1::new(
            TempoV1::integer(120).unwrap(),
            TimeSignature::new(4, 4).unwrap(),
        );
        map.push_change(
            Duration::new(4, 1),
            Some(TimeSignature::new(5, 32).unwrap()),
            None,
        )
        .unwrap();
        let bound = TemporalScoreV1::bind(score, map).unwrap();
        let projected = project_temporal_score_to_midi(&bound, Default::default()).unwrap();
        assert!(projected.events.iter().any(|event| matches!(
            event.kind,
            MidiTemporalMetaKindV1::TimeSignature {
                numerator: 5,
                denominator_power: 5
            }
        )));
    }

    #[test]
    fn exact_tick_grid_accepts_triplets_but_refuses_unrepresentable_sevenths() {
        assert_eq!(
            duration_to_ticks_exact(Duration::new(1, 3), 480).unwrap(),
            160
        );
        assert_eq!(
            duration_to_ticks_exact(Duration::new(1, 7), 480),
            Err(MidiTemporalProjectionErrorV1::NonRepresentableBeat {
                numerator: 1,
                denominator: 7,
                ticks_per_quarter: 480,
            })
        );
    }

    #[test]
    fn policy_can_refuse_nonstandard_grouping_marker_without_hiding_loss() {
        let score = score(8, 120.0, 4);
        let map = TemporalMapV1::new(
            TempoV1::integer(120).unwrap(),
            TimeSignature::new(4, 4).unwrap(),
        );
        let bound = TemporalScoreV1::bind(score, map).unwrap();
        let projected = project_temporal_score_to_midi(
            &bound,
            MidiTemporalProjectionPolicyV1 {
                ticks_per_quarter: 480,
                emit_grouping_marker: false,
            },
        )
        .unwrap();
        assert!(!projected.meter_receipts[0].standard_event_preserves_grouping);
        assert!(!projected.meter_receipts[0].grouping_marker_emitted);
    }

    #[test]
    fn very_slow_tempo_outside_midi_24_bit_field_fails_closed() {
        let score = score(8, 1.0, 4);
        let map = TemporalMapV1::new(
            TempoV1::integer(1).unwrap(),
            TimeSignature::new(4, 4).unwrap(),
        );
        let bound = TemporalScoreV1::bind(score, map).unwrap();
        assert_eq!(
            project_temporal_score_to_midi(&bound, Default::default()),
            Err(MidiTemporalProjectionErrorV1::MidiTempoOutOfRange {
                microseconds_per_quarter: 60_000_000,
            })
        );
    }
}
