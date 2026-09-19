// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Source-local resolution of comparison bar/beat anchors.
//!
//! `ComparisonSession` deliberately stores a musical coordinate rather than a
//! shared wall-clock timestamp. This module is the next authority boundary: it
//! resolves that coordinate against one subject's real composition bundle using
//! its meter and tempo maps. It never falls back to normalized duration or equal
//! seconds across subjects.

use std::fmt;

use symthaea_muse_protocol::{ListenCompositionBundle, MeterPoint, TempoPoint};

use crate::comparison::MusicalComparisonAnchor;

const EPS_BEATS: f64 = 1.0e-7;
const EPS_SECONDS: f64 = 1.0e-4;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResolvedComparisonAnchor {
    pub bar_index: u32,
    pub beat_offset: f64,
    /// Absolute quarter-note beats from the beginning of this subject.
    pub absolute_beats: f64,
    pub seconds: f64,
    pub meter_numerator: u8,
    pub meter_denominator: u8,
    pub tempo_bpm: f32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ComparisonAnchorResolutionError {
    InvalidDuration,
    MissingInitialTempo,
    InvalidTempoPoint,
    TempoMapNotStrictlyIncreasing,
    TempoMapDiscontinuous,
    MissingInitialMeter,
    InvalidMeterPoint,
    MeterMapNotStrictlyIncreasing,
    MeterChangeMidBar,
    BeatOutsideBar,
    AnchorOutsidePiece,
    ResolvedSecondsOutsidePiece,
}

impl fmt::Display for ComparisonAnchorResolutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDuration => write!(f, "comparison timeline has an invalid duration"),
            Self::MissingInitialTempo => {
                write!(f, "comparison timeline has no tempo point at beat zero")
            }
            Self::InvalidTempoPoint => write!(f, "comparison timeline contains an invalid tempo point"),
            Self::TempoMapNotStrictlyIncreasing => {
                write!(f, "comparison tempo map is not strictly increasing")
            }
            Self::TempoMapDiscontinuous => {
                write!(f, "comparison tempo map seconds disagree with its BPM segments")
            }
            Self::MissingInitialMeter => {
                write!(f, "comparison timeline has no meter point at beat zero")
            }
            Self::InvalidMeterPoint => write!(f, "comparison timeline contains an invalid meter point"),
            Self::MeterMapNotStrictlyIncreasing => {
                write!(f, "comparison meter map is not strictly increasing")
            }
            Self::MeterChangeMidBar => {
                write!(f, "comparison meter change does not land on a bar boundary")
            }
            Self::BeatOutsideBar => {
                write!(f, "comparison beat offset lies outside the target bar")
            }
            Self::AnchorOutsidePiece => write!(f, "comparison anchor lies outside the piece"),
            Self::ResolvedSecondsOutsidePiece => {
                write!(f, "comparison anchor resolves outside the rendered duration")
            }
        }
    }
}

impl std::error::Error for ComparisonAnchorResolutionError {}

/// Resolve one semantic comparison anchor into this subject's local transport
/// time using only the authoritative timeline carried by its composition bundle.
pub fn resolve_musical_anchor(
    bundle: &ListenCompositionBundle,
    anchor: MusicalComparisonAnchor,
) -> Result<ResolvedComparisonAnchor, ComparisonAnchorResolutionError> {
    resolve_from_maps(
        bundle.duration_beats,
        bundle.duration_seconds,
        &bundle.tempo_map,
        &bundle.meter_map,
        anchor,
    )
}

fn resolve_from_maps(
    duration_beats: f64,
    duration_seconds: f64,
    tempo_map: &[TempoPoint],
    meter_map: &[MeterPoint],
    anchor: MusicalComparisonAnchor,
) -> Result<ResolvedComparisonAnchor, ComparisonAnchorResolutionError> {
    if !duration_beats.is_finite()
        || duration_beats <= 0.0
        || !duration_seconds.is_finite()
        || duration_seconds <= 0.0
    {
        return Err(ComparisonAnchorResolutionError::InvalidDuration);
    }
    // `MusicalComparisonAnchor` has public fields, so callers are not forced to
    // use its constructor. Revalidate at the authority boundary rather than
    // assuming construction history.
    if !anchor.beat_offset.is_finite() || anchor.beat_offset < 0.0 {
        return Err(ComparisonAnchorResolutionError::BeatOutsideBar);
    }

    validate_tempo_map(tempo_map)?;
    validate_meter_map(meter_map)?;

    let (absolute_beats, meter) = resolve_bar_beat(duration_beats, meter_map, anchor)?;
    let (seconds, tempo_bpm) = beat_to_seconds(tempo_map, absolute_beats)?;

    if !seconds.is_finite() || seconds < 0.0 || seconds > duration_seconds + EPS_SECONDS {
        return Err(ComparisonAnchorResolutionError::ResolvedSecondsOutsidePiece);
    }

    Ok(ResolvedComparisonAnchor {
        bar_index: anchor.bar_index,
        beat_offset: anchor.beat_offset,
        absolute_beats,
        seconds,
        meter_numerator: meter.numerator,
        meter_denominator: meter.denominator,
        tempo_bpm,
    })
}

fn validate_tempo_map(
    tempo_map: &[TempoPoint],
) -> Result<(), ComparisonAnchorResolutionError> {
    let Some(first) = tempo_map.first() else {
        return Err(ComparisonAnchorResolutionError::MissingInitialTempo);
    };
    if !near_zero(first.at.beats) || !near_zero(first.at.seconds) {
        return Err(ComparisonAnchorResolutionError::MissingInitialTempo);
    }

    for point in tempo_map {
        if !point.at.beats.is_finite()
            || point.at.beats < 0.0
            || !point.at.seconds.is_finite()
            || point.at.seconds < 0.0
            || !point.bpm.is_finite()
            || point.bpm <= 0.0
        {
            return Err(ComparisonAnchorResolutionError::InvalidTempoPoint);
        }
    }

    for pair in tempo_map.windows(2) {
        let current = &pair[0];
        let next = &pair[1];
        if next.at.beats <= current.at.beats || next.at.seconds <= current.at.seconds {
            return Err(ComparisonAnchorResolutionError::TempoMapNotStrictlyIncreasing);
        }
        let expected_seconds = current.at.seconds
            + (next.at.beats - current.at.beats) * 60.0 / f64::from(current.bpm);
        if (expected_seconds - next.at.seconds).abs() > EPS_SECONDS {
            return Err(ComparisonAnchorResolutionError::TempoMapDiscontinuous);
        }
    }
    Ok(())
}

fn validate_meter_map(
    meter_map: &[MeterPoint],
) -> Result<(), ComparisonAnchorResolutionError> {
    let Some(first) = meter_map.first() else {
        return Err(ComparisonAnchorResolutionError::MissingInitialMeter);
    };
    if !near_zero(first.at.beats) {
        return Err(ComparisonAnchorResolutionError::MissingInitialMeter);
    }

    for point in meter_map {
        if !point.at.beats.is_finite()
            || point.at.beats < 0.0
            || point.numerator == 0
            || point.denominator == 0
        {
            return Err(ComparisonAnchorResolutionError::InvalidMeterPoint);
        }
    }

    for pair in meter_map.windows(2) {
        let current = &pair[0];
        let next = &pair[1];
        if next.at.beats <= current.at.beats {
            return Err(ComparisonAnchorResolutionError::MeterMapNotStrictlyIncreasing);
        }
        let bar_length = meter_bar_length_quarter_beats(current)?;
        let bars = (next.at.beats - current.at.beats) / bar_length;
        if (bars - bars.round()).abs() > EPS_BEATS {
            return Err(ComparisonAnchorResolutionError::MeterChangeMidBar);
        }
    }
    Ok(())
}

fn resolve_bar_beat(
    duration_beats: f64,
    meter_map: &[MeterPoint],
    anchor: MusicalComparisonAnchor,
) -> Result<(f64, &MeterPoint), ComparisonAnchorResolutionError> {
    let mut remaining_bar = u64::from(anchor.bar_index);

    for (index, meter) in meter_map.iter().enumerate() {
        let bar_length = meter_bar_length_quarter_beats(meter)?;
        let next_start = meter_map.get(index + 1).map(|next| next.at.beats);

        if let Some(next_start) = next_start {
            let segment_bars = ((next_start - meter.at.beats) / bar_length).round() as u64;
            if remaining_bar >= segment_bars {
                remaining_bar -= segment_bars;
                continue;
            }
        }

        if anchor.beat_offset >= f64::from(meter.numerator) {
            return Err(ComparisonAnchorResolutionError::BeatOutsideBar);
        }
        // `beat_offset` counts denominator-defined meter units: in 6/8, an
        // offset of 3 means three eighth-note units, i.e. 1.5 quarter beats.
        let meter_unit_quarter_beats = 4.0 / f64::from(meter.denominator);
        let absolute_beats = meter.at.beats
            + remaining_bar as f64 * bar_length
            + anchor.beat_offset * meter_unit_quarter_beats;

        // The endpoint itself is not a meaningful playback comparison anchor:
        // it would seek directly to Ended and differs from the named bar/beat.
        if absolute_beats < 0.0 || absolute_beats >= duration_beats - EPS_BEATS {
            return Err(ComparisonAnchorResolutionError::AnchorOutsidePiece);
        }
        return Ok((absolute_beats, meter));
    }

    Err(ComparisonAnchorResolutionError::AnchorOutsidePiece)
}

fn beat_to_seconds(
    tempo_map: &[TempoPoint],
    absolute_beats: f64,
) -> Result<(f64, f32), ComparisonAnchorResolutionError> {
    let tempo = tempo_map
        .iter()
        .rev()
        .find(|point| point.at.beats <= absolute_beats + EPS_BEATS)
        .ok_or(ComparisonAnchorResolutionError::MissingInitialTempo)?;
    let seconds = tempo.at.seconds
        + (absolute_beats - tempo.at.beats) * 60.0 / f64::from(tempo.bpm);
    Ok((seconds, tempo.bpm))
}

fn meter_bar_length_quarter_beats(
    meter: &MeterPoint,
) -> Result<f64, ComparisonAnchorResolutionError> {
    if meter.numerator == 0 || meter.denominator == 0 {
        return Err(ComparisonAnchorResolutionError::InvalidMeterPoint);
    }
    Ok(f64::from(meter.numerator) * 4.0 / f64::from(meter.denominator))
}

fn near_zero(value: f64) -> bool {
    value.abs() <= EPS_BEATS
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_muse_protocol::MusicalTime;

    fn time(beats: f64, seconds: f64) -> MusicalTime {
        MusicalTime {
            tick: (beats * 960.0).round() as u64,
            beats,
            seconds,
        }
    }

    fn tempo(beats: f64, seconds: f64, bpm: f32) -> TempoPoint {
        TempoPoint {
            at: time(beats, seconds),
            bpm,
        }
    }

    fn meter(beats: f64, numerator: u8, denominator: u8) -> MeterPoint {
        MeterPoint {
            at: time(beats, 0.0),
            numerator,
            denominator,
        }
    }

    #[test]
    fn constant_four_four_resolves_bar_and_beat_to_local_seconds() {
        let resolved = resolve_from_maps(
            16.0,
            8.0,
            &[tempo(0.0, 0.0, 120.0)],
            &[meter(0.0, 4, 4)],
            MusicalComparisonAnchor::new(2, 1.0).unwrap(),
        )
        .unwrap();

        assert!((resolved.absolute_beats - 9.0).abs() < EPS_BEATS);
        assert!((resolved.seconds - 4.5).abs() < EPS_SECONDS);
        assert_eq!((resolved.meter_numerator, resolved.meter_denominator), (4, 4));
        assert_eq!(resolved.tempo_bpm, 120.0);
    }

    #[test]
    fn meter_and_tempo_change_resolve_same_musical_anchor_source_locally() {
        // Two bars of 4/4 (8 quarter beats), then 3/4 at 60 BPM.
        let resolved = resolve_from_maps(
            14.0,
            10.0,
            &[tempo(0.0, 0.0, 120.0), tempo(8.0, 4.0, 60.0)],
            &[meter(0.0, 4, 4), meter(8.0, 3, 4)],
            MusicalComparisonAnchor::new(2, 1.0).unwrap(),
        )
        .unwrap();

        assert!((resolved.absolute_beats - 9.0).abs() < EPS_BEATS);
        assert!((resolved.seconds - 5.0).abs() < EPS_SECONDS);
        assert_eq!((resolved.meter_numerator, resolved.meter_denominator), (3, 4));
        assert_eq!(resolved.tempo_bpm, 60.0);
    }

    #[test]
    fn denominator_controls_meter_beat_units() {
        // 6/8 is three quarter-note beats per bar. Beat offset 3 means three
        // eighth-note meter units = 1.5 quarter-note beats into bar 1.
        let resolved = resolve_from_maps(
            9.0,
            4.5,
            &[tempo(0.0, 0.0, 120.0)],
            &[meter(0.0, 6, 8)],
            MusicalComparisonAnchor::new(1, 3.0).unwrap(),
        )
        .unwrap();

        assert!((resolved.absolute_beats - 4.5).abs() < EPS_BEATS);
        assert!((resolved.seconds - 2.25).abs() < EPS_SECONDS);
    }

    #[test]
    fn meter_change_mid_bar_fails_closed() {
        let error = resolve_from_maps(
            12.0,
            6.0,
            &[tempo(0.0, 0.0, 120.0)],
            &[meter(0.0, 4, 4), meter(3.0, 3, 4)],
            MusicalComparisonAnchor::new(1, 0.0).unwrap(),
        )
        .unwrap_err();
        assert_eq!(error, ComparisonAnchorResolutionError::MeterChangeMidBar);
    }

    #[test]
    fn beat_offset_must_stay_inside_target_bar() {
        let error = resolve_from_maps(
            8.0,
            4.0,
            &[tempo(0.0, 0.0, 120.0)],
            &[meter(0.0, 4, 4)],
            MusicalComparisonAnchor::new(0, 4.0).unwrap(),
        )
        .unwrap_err();
        assert_eq!(error, ComparisonAnchorResolutionError::BeatOutsideBar);
    }

    #[test]
    fn direct_struct_construction_cannot_bypass_anchor_validation() {
        let error = resolve_from_maps(
            8.0,
            4.0,
            &[tempo(0.0, 0.0, 120.0)],
            &[meter(0.0, 4, 4)],
            MusicalComparisonAnchor {
                bar_index: 1,
                beat_offset: -0.5,
            },
        )
        .unwrap_err();
        assert_eq!(error, ComparisonAnchorResolutionError::BeatOutsideBar);
    }

    #[test]
    fn discontinuous_tempo_map_fails_closed() {
        let error = resolve_from_maps(
            16.0,
            8.0,
            // At 120 BPM, beat 8 must occur at 4.0s, not 3.0s.
            &[tempo(0.0, 0.0, 120.0), tempo(8.0, 3.0, 60.0)],
            &[meter(0.0, 4, 4)],
            MusicalComparisonAnchor::new(2, 0.0).unwrap(),
        )
        .unwrap_err();
        assert_eq!(error, ComparisonAnchorResolutionError::TempoMapDiscontinuous);
    }

    #[test]
    fn out_of_order_tempo_map_fails_closed() {
        let error = resolve_from_maps(
            16.0,
            8.0,
            &[
                tempo(0.0, 0.0, 120.0),
                tempo(8.0, 4.0, 90.0),
                tempo(7.0, 5.0, 80.0),
            ],
            &[meter(0.0, 4, 4)],
            MusicalComparisonAnchor::new(1, 0.0).unwrap(),
        )
        .unwrap_err();
        assert_eq!(
            error,
            ComparisonAnchorResolutionError::TempoMapNotStrictlyIncreasing
        );
    }

    #[test]
    fn anchor_at_or_beyond_piece_end_is_rejected_not_clamped() {
        let error = resolve_from_maps(
            8.0,
            4.0,
            &[tempo(0.0, 0.0, 120.0)],
            &[meter(0.0, 4, 4)],
            MusicalComparisonAnchor::new(2, 0.0).unwrap(),
        )
        .unwrap_err();
        assert_eq!(error, ComparisonAnchorResolutionError::AnchorOutsidePiece);
    }
}
