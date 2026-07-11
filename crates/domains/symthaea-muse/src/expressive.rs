// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Learned expressive performance: velocity shaping and articulation fitted
//! on real human performances (MAESTRO v3, Disklavier-captured virtuoso
//! piano), applied when rendering theory Scores.
//!
//! ## What is (and is not) modeled — and why
//!
//! MAESTRO MIDI is *performance capture*: timing drifts continuously with the
//! player's rubato, so onsets quantized against a nominal beat grid measure
//! tempo drift, not expression — grid-based micro-timing would need real beat
//! tracking (its own project). This model therefore learns only the two
//! expressive dimensions that are **grid-free**:
//!
//! - **Velocity deviation**: how far a note's loudness sits above/below its
//!   local context (rolling mean over ±[`LOCAL_WINDOW`] notes). Structural
//!   dynamics (phrase arcs, climaxes) stay the composer's job — the model
//!   adds the note-to-note accent texture human players layer on top.
//! - **Articulation**: sounded duration as a fraction of the inter-onset
//!   interval (staccato ≈ 0.5, legato ≈ 1.0+, overlapping ≈ >1). Ratios are
//!   robust to slow tempo drift because numerator and denominator drift
//!   together.
//!
//! Every context feature is computable from a symbolic [`Score`]
//! (`symthaea-music-theory`) at render time — training uses ONLY information
//! the renderer will actually have, so there is no train/apply mismatch.
//!
//! Trained by `src/bin/train_performance_model.rs` (ridge least squares,
//! file-level split, honest held-out metrics vs baselines); weights +
//! provenance embedded from `data/performance_model_weights.json`. Absent or
//! unparseable weights fall back LOUDLY to a neutral model (zero deviation,
//! default articulation) — never silently to garbage.

use crate::midi_trainer::ExtractedMelody;
use serde::{Deserialize, Serialize};

/// Feature vector length (excluding the bias the trainer appends).
pub const N_FEATURES: usize = 10;

/// Rolling window half-width (in notes) for the local velocity mean.
pub const LOCAL_WINDOW: usize = 8;

/// Fallback articulation when the model is neutral: mostly-legato, matching
/// the hand-tuned humanizer's prior behavior.
pub const NEUTRAL_ARTICULATION: f32 = 0.95;

/// One training example: score-derivable context → performed expression.
#[derive(Debug, Clone)]
pub struct PerformancePair {
    pub features: [f32; N_FEATURES],
    /// Velocity deviation from the local rolling mean, in normalized
    /// velocity units (≈ ±0.5 extreme).
    pub velocity_dev: f32,
    /// Sounded duration / inter-onset interval, clamped to [0.05, 2.0].
    pub articulation: f32,
}

/// Context features for one note. All inputs are available from a symbolic
/// score at render time (pitches, quantized IOIs in beats, position).
///
/// `ioi_prev`/`ioi` are inter-onset intervals in beats (prev note→this note,
/// this note→next note).
pub fn features_from_context(
    prev_midi: u8,
    cur_midi: u8,
    next_midi: u8,
    ioi_prev_beats: f32,
    ioi_beats: f32,
    position_in_piece: f32,
) -> [f32; N_FEATURES] {
    let iv_prev = (cur_midi as f32 - prev_midi as f32).clamp(-12.0, 12.0) / 12.0;
    let iv_next = (next_midi as f32 - cur_midi as f32).clamp(-12.0, 12.0) / 12.0;
    let is_peak = (cur_midi > prev_midi && cur_midi > next_midi) as u8 as f32;
    let is_valley = (cur_midi < prev_midi && cur_midi < next_midi) as u8 as f32;
    let rhythm_change = if ioi_prev_beats > 1e-4 && ioi_beats > 1e-4 {
        (ioi_beats / ioi_prev_beats).log2().clamp(-2.0, 2.0) / 2.0
    } else {
        0.0
    };
    let speed = if ioi_beats > 1e-4 {
        ioi_beats.log2().clamp(-3.0, 2.0) / 3.0
    } else {
        0.0
    };
    let direction_change =
        (iv_prev != 0.0 && iv_next != 0.0 && (iv_prev > 0.0) != (iv_next > 0.0)) as u8 as f32;
    let large_leap = (iv_prev.abs() * 12.0 >= 5.0) as u8 as f32;
    [
        iv_prev,
        iv_next,
        is_peak,
        is_valley,
        rhythm_change,
        speed,
        cur_midi as f32 / 127.0,
        position_in_piece.clamp(0.0, 1.0),
        direction_change,
        large_leap,
    ]
}

/// Extract training pairs from one performance. Uses the skyline melody line
/// (same reduction as the melody predictor) — this is deliberately a model of
/// *melodic-line* expression, which is what the renderer applies it to.
pub fn extract_pairs(melody: &ExtractedMelody) -> Vec<PerformancePair> {
    let notes = &melody.notes;
    if notes.len() < 2 * LOCAL_WINDOW + 3 {
        return Vec::new();
    }
    let tpb = melody.ticks_per_beat.max(1) as f32;
    let onset = |i: usize| notes[i].onset_tick as f32 / tpb;
    let velocities: Vec<f32> = notes.iter().map(|n| n.velocity as f32 / 127.0).collect();
    let total_beats = onset(notes.len() - 1).max(1.0);

    let mut pairs = Vec::new();
    for i in 1..notes.len() - 1 {
        let ioi_prev = onset(i) - onset(i - 1);
        let ioi = onset(i + 1) - onset(i);
        // Skip section breaks / grace-note pileups / capture glitches: the
        // articulation ratio is meaningless across a pause and unstable when
        // onsets nearly coincide.
        if !(0.02..=8.0).contains(&ioi_prev) || !(0.02..=8.0).contains(&ioi) {
            continue;
        }
        let lo = i.saturating_sub(LOCAL_WINDOW);
        let hi = (i + LOCAL_WINDOW + 1).min(notes.len());
        let local_mean = velocities[lo..hi].iter().sum::<f32>() / (hi - lo) as f32;
        let velocity_dev = velocities[i] - local_mean;
        let duration_beats = notes[i].duration_ticks as f32 / tpb;
        let articulation = (duration_beats / ioi).clamp(0.05, 2.0);
        pairs.push(PerformancePair {
            features: features_from_context(
                notes[i - 1].pitch,
                notes[i].pitch,
                notes[i + 1].pitch,
                ioi_prev,
                ioi,
                onset(i) / total_beats,
            ),
            velocity_dev,
            articulation,
        });
    }
    pairs
}

/// Serialized weights + provenance (see the trainer bin).
#[derive(Debug, Serialize, Deserialize)]
pub struct PerformanceWeights {
    /// Velocity-deviation head: N_FEATURES weights + bias (last).
    pub w_velocity: Vec<f32>,
    /// Articulation head: N_FEATURES weights + bias (last).
    pub w_articulation: Vec<f32>,
    pub provenance: serde_json::Value,
}

/// The runtime model: two linear heads over the shared feature vector.
#[derive(Debug, Clone)]
pub struct ExpressiveModel {
    w_velocity: [f32; N_FEATURES + 1],
    w_articulation: [f32; N_FEATURES + 1],
}

const EMBEDDED_WEIGHTS: &str = include_str!("../data/performance_model_weights.json");

impl ExpressiveModel {
    /// Neutral model: zero velocity deviation, [`NEUTRAL_ARTICULATION`] —
    /// behaviorally the pre-model renderer.
    pub fn neutral() -> Self {
        let mut w_articulation = [0.0f32; N_FEATURES + 1];
        w_articulation[N_FEATURES] = NEUTRAL_ARTICULATION; // bias-only
        ExpressiveModel {
            w_velocity: [0.0; N_FEATURES + 1],
            w_articulation,
        }
    }

    /// Load the MAESTRO-trained weights embedded at compile time. Falls back
    /// to [`Self::neutral`] with a LOUD stderr warning if they fail to parse
    /// (a guard test keeps that from ever shipping silently).
    pub fn from_embedded() -> Self {
        match Self::try_from_embedded() {
            Ok(m) => m,
            Err(e) => {
                eprintln!(
                    "[muse::expressive] EMBEDDED WEIGHTS FAILED TO PARSE ({e}) — \
                     running with the NEUTRAL model (no learned expression)"
                );
                Self::neutral()
            }
        }
    }

    /// Fallible embedded-weights loader (exposed so the guard test can make
    /// a parse failure a hard error rather than a silent fallback).
    pub fn try_from_embedded() -> Result<Self, String> {
        let parsed: PerformanceWeights =
            serde_json::from_str(EMBEDDED_WEIGHTS).map_err(|e| e.to_string())?;
        let to_arr = |v: &[f32]| -> Result<[f32; N_FEATURES + 1], String> {
            v.try_into()
                .map_err(|_| format!("expected {} weights, got {}", N_FEATURES + 1, v.len()))
        };
        Ok(ExpressiveModel {
            w_velocity: to_arr(&parsed.w_velocity)?,
            w_articulation: to_arr(&parsed.w_articulation)?,
        })
    }

    /// Predict `(velocity_deviation, articulation)` for one note's context.
    /// Outputs are clamped to musically-sane ranges: the model textures a
    /// performance, it must never be able to blow one up.
    pub fn predict(&self, features: &[f32; N_FEATURES]) -> (f32, f32) {
        let dot = |w: &[f32; N_FEATURES + 1]| -> f32 {
            w[..N_FEATURES]
                .iter()
                .zip(features)
                .map(|(a, b)| a * b)
                .sum::<f32>()
                + w[N_FEATURES]
        };
        let velocity_dev = dot(&self.w_velocity).clamp(-0.35, 0.35);
        let articulation = dot(&self.w_articulation).clamp(0.3, 1.3);
        (velocity_dev, articulation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::midi_trainer::MidiNote;

    fn synthetic_melody() -> ExtractedMelody {
        // Quarter notes at 480 tpb, an up-down contour, alternating loud/soft.
        let notes: Vec<MidiNote> = (0..40)
            .map(|i| MidiNote {
                pitch: 60 + (if i % 4 < 2 { i % 4 } else { 4 - i % 4 }) as u8 * 2,
                onset_tick: i as u64 * 480,
                duration_ticks: 400, // articulation 400/480 ≈ 0.833
                velocity: if i % 2 == 0 { 90 } else { 60 },
            })
            .collect();
        ExtractedMelody {
            notes,
            ticks_per_beat: 480,
            tempo_bpm: 120.0,
            key: 0,
            minor: false,
            source: "synthetic".into(),
        }
    }

    #[test]
    fn extraction_computes_grid_free_targets() {
        let pairs = extract_pairs(&synthetic_melody());
        assert!(!pairs.is_empty());
        for p in &pairs {
            // Articulation: 400 ticks sounded / 480 ticks IOI.
            assert!(
                (p.articulation - 400.0 / 480.0).abs() < 1e-4,
                "articulation {}",
                p.articulation
            );
            // Alternating 90/60 around a ~75 mean → deviation ≈ ±15/127.
            assert!(
                (p.velocity_dev.abs() - 15.0 / 127.0).abs() < 0.03,
                "velocity_dev {}",
                p.velocity_dev
            );
            assert!(p.features.iter().all(|f| f.is_finite()));
        }
    }

    #[test]
    fn extraction_skips_across_pauses() {
        let mut m = synthetic_melody();
        // Insert a long gap: the pair straddling it must be dropped, not
        // produce a nonsense articulation against a 20-beat "IOI".
        let n = m.notes.len();
        for note in &mut m.notes[n / 2..] {
            note.onset_tick += 480 * 20;
        }
        let pairs = extract_pairs(&m);
        assert!(
            pairs
                .iter()
                .all(|p| p.articulation >= 0.05 && p.articulation <= 2.0)
        );
        // One fewer pair than the continuous version (the straddler).
        assert!(pairs.len() < extract_pairs(&synthetic_melody()).len());
    }

    #[test]
    fn peak_and_valley_flags_fire_correctly() {
        let f = features_from_context(60, 67, 64, 1.0, 1.0, 0.5);
        assert_eq!(f[2], 1.0, "67 above both neighbors = peak");
        assert_eq!(f[3], 0.0);
        let f = features_from_context(67, 60, 64, 1.0, 1.0, 0.5);
        assert_eq!(f[3], 1.0, "60 below both neighbors = valley");
    }

    #[test]
    fn neutral_model_is_behaviorally_inert() {
        let m = ExpressiveModel::neutral();
        let f = features_from_context(60, 64, 67, 1.0, 0.5, 0.3);
        let (dev, art) = m.predict(&f);
        assert_eq!(dev, 0.0);
        assert!((art - NEUTRAL_ARTICULATION).abs() < 1e-6);
    }

    #[test]
    fn predictions_are_always_bounded() {
        // Even a hostile weight vector can't push outputs past the clamps.
        let m = ExpressiveModel {
            w_velocity: [100.0; N_FEATURES + 1],
            w_articulation: [-100.0; N_FEATURES + 1],
        };
        let f = features_from_context(0, 127, 0, 8.0, 0.02, 1.0);
        let (dev, art) = m.predict(&f);
        assert!((-0.35..=0.35).contains(&dev));
        assert!((0.3..=1.3).contains(&art));
    }

    #[test]
    fn embedded_weights_parse_or_the_build_is_dishonest() {
        // The guard the provenance discipline requires: if this file ships,
        // it parses. (A missing/placeholder file must fail HERE, visibly,
        // not degrade to neutral in production.)
        let model = ExpressiveModel::try_from_embedded()
            .expect("data/performance_model_weights.json must parse");
        // And the weights must not be the all-zero placeholder.
        assert!(
            model.w_velocity.iter().any(|w| *w != 0.0),
            "embedded weights look like a placeholder — retrain or fix"
        );
    }
}
