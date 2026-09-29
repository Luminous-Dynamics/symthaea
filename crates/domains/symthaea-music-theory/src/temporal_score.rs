// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Backward-compatible binding between a legacy symbolic score and work-scale
//! temporal authority.
//!
//! `Score` remains unchanged in V1. Instead, callers that need mixed meter or
//! tempo changes bind the existing score to a validated [`TemporalMapV1`].
//! This avoids silently changing serialized score identity for unrelated
//! experiments while still making the temporal map authoritative for work-scale
//! consumers such as MIDI export, long-form planning, and performance timing.

use crate::meter::TimeSignature;
use crate::rhythm::Duration;
use crate::score::Score;
use crate::temporal_map::{TempoV1, TemporalMapErrorV1, TemporalMapV1};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

pub const TEMPORAL_SCORE_VERSION: &str = "melothaea-temporal-score-v1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalScoreV1 {
    pub version: String,
    pub score: Score,
    pub temporal_map: TemporalMapV1,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TemporalScoreErrorV1 {
    WrongVersion { found: String },
    InvalidTemporalMap(TemporalMapErrorV1),
    LegacyMeterInvalid,
    LegacyTempoInvalid,
    OpeningMeterMismatch,
    OpeningTempoMismatch,
    NegativeScoreLength,
    TemporalPointAfterScore { point_index: usize },
}

impl TemporalScoreV1 {
    pub fn bind(score: Score, temporal_map: TemporalMapV1) -> Result<Self, TemporalScoreErrorV1> {
        let bound = Self {
            version: TEMPORAL_SCORE_VERSION.into(),
            score,
            temporal_map,
        };
        bound.validate()?;
        Ok(bound)
    }

    /// Validate both the temporal map and its compatibility with the legacy
    /// opening metadata retained on `Score`.
    ///
    /// Tempo compatibility is checked at the exact stored f32 boundary: the
    /// rational authoritative tempo, when projected to f32, must equal the
    /// legacy opening tempo bit-for-bit. The rational map remains authoritative.
    pub fn validate(&self) -> Result<(), TemporalScoreErrorV1> {
        if self.version != TEMPORAL_SCORE_VERSION {
            return Err(TemporalScoreErrorV1::WrongVersion {
                found: self.version.clone(),
            });
        }
        self.temporal_map
            .validate()
            .map_err(TemporalScoreErrorV1::InvalidTemporalMap)?;

        if self.score.meter == 0 {
            return Err(TemporalScoreErrorV1::LegacyMeterInvalid);
        }
        if !self.score.tempo_bpm.is_finite() || self.score.tempo_bpm <= 0.0 {
            return Err(TemporalScoreErrorV1::LegacyTempoInvalid);
        }
        if self.score.total_beats.num() < 0 {
            return Err(TemporalScoreErrorV1::NegativeScoreLength);
        }

        let opening_meter = self
            .temporal_map
            .meter_at(Duration::zero())
            .map_err(TemporalScoreErrorV1::InvalidTemporalMap)?;
        if opening_meter != self.score.time_signature() {
            return Err(TemporalScoreErrorV1::OpeningMeterMismatch);
        }

        let opening_tempo = self
            .temporal_map
            .tempo_at(Duration::zero())
            .map_err(TemporalScoreErrorV1::InvalidTemporalMap)?;
        if (opening_tempo.bpm() as f32).to_bits() != self.score.tempo_bpm.to_bits() {
            return Err(TemporalScoreErrorV1::OpeningTempoMismatch);
        }

        for (index, point) in self.temporal_map.points.iter().enumerate() {
            if compare_duration(point.at, self.score.total_beats) == Ordering::Greater {
                return Err(TemporalScoreErrorV1::TemporalPointAfterScore { point_index: index });
            }
        }
        Ok(())
    }

    pub fn tempo_at(&self, at: Duration) -> Result<TempoV1, TemporalScoreErrorV1> {
        self.validate()?;
        self.temporal_map
            .tempo_at(at)
            .map_err(TemporalScoreErrorV1::InvalidTemporalMap)
    }

    pub fn meter_at(&self, at: Duration) -> Result<TimeSignature, TemporalScoreErrorV1> {
        self.validate()?;
        self.temporal_map
            .meter_at(at)
            .map_err(TemporalScoreErrorV1::InvalidTemporalMap)
    }

    /// Authoritative work duration under the temporal map.
    pub fn seconds(&self) -> Result<f64, TemporalScoreErrorV1> {
        self.validate()?;
        self.temporal_map
            .elapsed_seconds(self.score.total_beats)
            .map_err(TemporalScoreErrorV1::InvalidTemporalMap)
    }

    /// Legacy fixed-tempo duration retained only for migration diagnostics.
    pub fn legacy_seconds(&self) -> Result<f64, TemporalScoreErrorV1> {
        self.validate()?;
        Ok(self.score.seconds())
    }
}

fn compare_duration(left: Duration, right: Duration) -> Ordering {
    (i128::from(left.num()) * i128::from(right.den()))
        .cmp(&(i128::from(right.num()) * i128::from(left.den())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Emphasis, Key, PartId, Pitch, PitchClass, ScoreNote, VoiceRole};

    fn score(total_beats: i64) -> Score {
        let mut score = Score::new(Key::major(PitchClass::C), 120.0, 4);
        if total_beats > 0 {
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
        }
        score
    }

    fn initial_map() -> TemporalMapV1 {
        TemporalMapV1::new(
            TempoV1::integer(120).unwrap(),
            TimeSignature::new(4, 4).unwrap(),
        )
    }

    #[test]
    fn legacy_fixed_tempo_score_binds_without_changing_score_identity() {
        let score = score(8);
        let original = score.clone();
        let bound = TemporalScoreV1::bind(score, initial_map()).unwrap();
        assert_eq!(bound.score, original);
        assert!((bound.seconds().unwrap() - 4.0).abs() < 1e-12);
        assert!((bound.legacy_seconds().unwrap() - 4.0).abs() < 1e-12);
    }

    #[test]
    fn temporal_authority_changes_duration_without_rewriting_notes() {
        let score = score(12);
        let original_notes = score.notes.clone();
        let mut map = initial_map();
        map.push_change(
            Duration::new(8, 1),
            None,
            Some(TempoV1::integer(60).unwrap()),
        )
        .unwrap();
        let bound = TemporalScoreV1::bind(score, map).unwrap();
        assert_eq!(bound.score.notes, original_notes);
        assert!((bound.legacy_seconds().unwrap() - 6.0).abs() < 1e-12);
        assert!((bound.seconds().unwrap() - 8.0).abs() < 1e-12);
    }

    #[test]
    fn grouped_meter_changes_are_queryable_without_touching_legacy_opening_meter() {
        let score = score(16);
        let mut map = initial_map();
        let seven = TimeSignature::with_grouping(7, 8, vec![2, 2, 3]).unwrap();
        map.push_change(Duration::new(8, 1), Some(seven.clone()), None)
            .unwrap();
        let bound = TemporalScoreV1::bind(score, map).unwrap();
        assert_eq!(bound.score.meter, 4);
        assert_eq!(bound.meter_at(Duration::new(7, 1)).unwrap(), TimeSignature::new(4, 4).unwrap());
        assert_eq!(bound.meter_at(Duration::new(8, 1)).unwrap(), seven);
    }

    #[test]
    fn opening_meter_must_match_legacy_score_metadata() {
        let score = score(8);
        let map = TemporalMapV1::new(
            TempoV1::integer(120).unwrap(),
            TimeSignature::new(6, 8).unwrap(),
        );
        assert_eq!(
            TemporalScoreV1::bind(score, map),
            Err(TemporalScoreErrorV1::OpeningMeterMismatch)
        );
    }

    #[test]
    fn opening_tempo_must_project_exactly_to_legacy_f32() {
        let score = score(8);
        let map = TemporalMapV1::new(
            TempoV1::integer(121).unwrap(),
            TimeSignature::new(4, 4).unwrap(),
        );
        assert_eq!(
            TemporalScoreV1::bind(score, map),
            Err(TemporalScoreErrorV1::OpeningTempoMismatch)
        );
    }

    #[test]
    fn changes_after_the_score_end_fail_closed() {
        let score = score(8);
        let mut map = initial_map();
        map.push_change(
            Duration::new(9, 1),
            None,
            Some(TempoV1::integer(100).unwrap()),
        )
        .unwrap();
        assert_eq!(
            TemporalScoreV1::bind(score, map),
            Err(TemporalScoreErrorV1::TemporalPointAfterScore { point_index: 1 })
        );
    }

    #[test]
    fn change_exactly_at_score_end_is_valid_metadata_boundary() {
        let score = score(8);
        let mut map = initial_map();
        map.push_change(
            Duration::new(8, 1),
            Some(TimeSignature::new(3, 4).unwrap()),
            None,
        )
        .unwrap();
        assert!(TemporalScoreV1::bind(score, map).is_ok());
    }
}
