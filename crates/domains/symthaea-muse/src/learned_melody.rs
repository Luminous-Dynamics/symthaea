// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Linear melody predictor with weights trained in-repo on MAESTRO v3.0.0.
//!
//! PROVENANCE: weights are fit by `src/bin/train_melody_predictor.rs`
//! (ridge least squares on `midi_trainer` features, file-level held-out
//! split, no augmentation) and shipped in
//! `data/melody_predictor_weights.json`, whose embedded `provenance` block
//! records the dataset, split, and honest held-out metrics of the run that
//! produced it. Read that block rather than trusting any number quoted here.
//!
//! HISTORY: an earlier header claimed "trained on 65M pairs, direction
//! accuracy 93.2%" over hand-tuned constants no code had fit. During the
//! 2026-07-07 training run the origin of that mythical number surfaced:
//! `melody_to_training_pairs` leaked the target interval into the context
//! window (off-by-one), and any direction-accuracy evaluation through that
//! leak reads ~93-100%. The leak is fixed (see the regression test in
//! `midi_trainer`); honest melodic-direction accuracy is far lower — see the
//! provenance JSON. The old constants remain only as a parse-failure
//! fallback (`FALLBACK_*`).

use std::sync::OnceLock;

/// Trained weights loaded from the committed artifact.
/// Layout: [8 intervals | 8 durations | beat_pos | phrase_pos | valence | arousal]
struct Weights {
    interval_weights: [f32; 20],
    interval_bias: f32,
    duration_weights: [f32; 20],
    duration_bias: f32,
}

#[derive(serde::Deserialize)]
struct TrainedWeightsFile {
    interval_weights: Vec<f32>,
    interval_bias: f32,
    duration_weights: Vec<f32>,
    duration_bias: f32,
}

/// The committed trained artifact (see module header for provenance).
const TRAINED_JSON: &str = include_str!("../data/melody_predictor_weights.json");

fn weights() -> &'static Weights {
    static WEIGHTS: OnceLock<Weights> = OnceLock::new();
    WEIGHTS.get_or_init(
        || match serde_json::from_str::<TrainedWeightsFile>(TRAINED_JSON) {
            Ok(t) if t.interval_weights.len() == 20 && t.duration_weights.len() == 20 => Weights {
                interval_weights: t.interval_weights.try_into().unwrap(),
                interval_bias: t.interval_bias,
                duration_weights: t.duration_weights.try_into().unwrap(),
                duration_bias: t.duration_bias,
            },
            _ => {
                eprintln!(
                    "[symthaea-muse] WARNING: melody_predictor_weights.json failed to \
                     parse; MelodyPredictor falling back to hand-tuned constants"
                );
                Weights {
                    interval_weights: FALLBACK_INTERVAL_WEIGHTS,
                    interval_bias: FALLBACK_INTERVAL_BIAS,
                    duration_weights: FALLBACK_DURATION_WEIGHTS,
                    duration_bias: FALLBACK_DURATION_BIAS,
                }
            }
        },
    )
}

/// Hand-tuned fallback weights (pre-training; used only if the committed
/// JSON fails to parse). Provenance unknown — see module header HISTORY.
#[rustfmt::skip]
const FALLBACK_INTERVAL_WEIGHTS: [f32; 20] = [
    0.0012, -0.0018, 0.0065, -0.0025, 0.0008, -0.0015, -0.0839, 0.1854,
    0.0003, -0.0005, 0.0002, -0.0008, 0.0004, -0.0003, 0.0011, 0.0029,
    0.0015, -0.0022, 0.0018, 0.0008,
];
const FALLBACK_INTERVAL_BIAS: f32 = -0.002;
#[rustfmt::skip]
const FALLBACK_DURATION_WEIGHTS: [f32; 20] = [
    -0.0005, 0.0003, -0.0002, 0.0001, -0.0004, 0.0002, 0.0008, -0.0012,
    0.0015, -0.0008, 0.0010, -0.0005, 0.0012, -0.0006, 0.0280, 0.0850,
    0.0030, 0.0120, -0.0005, -0.0180,
];
const FALLBACK_DURATION_BIAS: f32 = 0.25;

/// The fallback as a const struct, selectable for A/B comparison.
const FALLBACK_WEIGHTS: Weights = Weights {
    interval_weights: FALLBACK_INTERVAL_WEIGHTS,
    interval_bias: FALLBACK_INTERVAL_BIAS,
    duration_weights: FALLBACK_DURATION_WEIGHTS,
    duration_bias: FALLBACK_DURATION_BIAS,
};

/// Context window for the learned predictor.
pub struct MelodyPredictor {
    /// Recent intervals (semitones), newest last.
    interval_history: Vec<f32>,
    /// Recent durations (in beats), newest last.
    duration_history: Vec<f32>,
    /// Use the pre-training hand-tuned constants instead of the trained
    /// artifact. ONLY for A/B evaluation harnesses — production paths must
    /// use the trained weights.
    use_fallback: bool,
}

impl MelodyPredictor {
    pub fn new() -> Self {
        Self {
            interval_history: Vec::with_capacity(8),
            duration_history: Vec::with_capacity(8),
            use_fallback: false,
        }
    }

    /// A/B-testing constructor: predict with the OLD hand-tuned constants.
    /// Exists so listening tests can compare trained vs pre-training output;
    /// never use in a production path.
    pub fn with_fallback_weights() -> Self {
        Self {
            use_fallback: true,
            ..Self::new()
        }
    }

    /// Record a note (update context).
    pub fn record(&mut self, interval: f32, duration_beats: f32) {
        self.interval_history.push(interval);
        if self.interval_history.len() > 8 {
            self.interval_history.remove(0);
        }
        self.duration_history.push(duration_beats);
        if self.duration_history.len() > 8 {
            self.duration_history.remove(0);
        }
    }

    /// Predict the next interval (in semitones) and duration (in beats).
    ///
    /// Uses the trained linear model: y = tanh(Wx + b)
    pub fn predict(
        &self,
        beat_position: f32,
        phrase_position: f32,
        valence: f32,
        arousal: f32,
    ) -> (f32, f32) {
        let features = self.build_features(beat_position, phrase_position, valence, arousal);
        let w = if self.use_fallback {
            &FALLBACK_WEIGHTS
        } else {
            weights()
        };

        // Interval prediction. The trainer fits against this exact decode
        // (tanh(5·raw)·12), so the 5x factor is part of the trained model,
        // not an ad-hoc amplifier.
        let mut interval_raw = w.interval_bias;
        for (i, &f) in features.iter().enumerate() {
            interval_raw += w.interval_weights[i] * f;
        }
        let mut predicted_interval = (interval_raw * 5.0).tanh() * 12.0; // [-12, 12] semitones

        // Prevent stagnation: if interval is near zero, nudge it to at least ±1 semitone
        if predicted_interval.abs() < 0.5 {
            predicted_interval = if predicted_interval >= 0.0 { 1.5 } else { -1.5 };
        }

        // Duration prediction (trainer fits against this decode too)
        let mut duration_raw = w.duration_bias;
        for (i, &f) in features.iter().enumerate() {
            duration_raw += w.duration_weights[i] * f;
        }
        let predicted_duration = (duration_raw.tanh() * 0.5 + 0.5) * 4.0; // [0, 4] beats

        (predicted_interval, predicted_duration.max(0.1))
    }

    /// Build the 20D feature vector from current context.
    fn build_features(
        &self,
        beat_position: f32,
        phrase_position: f32,
        valence: f32,
        arousal: f32,
    ) -> [f32; 20] {
        let mut features = [0.0f32; 20];

        // Pad interval history to 8 (oldest first)
        let _iv_offset = 8 - self.interval_history.len().min(8);
        for (i, &iv) in self.interval_history.iter().rev().take(8).enumerate() {
            features[7 - i] = (iv / 12.0).clamp(-1.0, 1.0);
        }

        // Pad duration history to 8
        let dur_offset = 8;
        for (i, &dur) in self.duration_history.iter().rev().take(8).enumerate() {
            features[dur_offset + 7 - i] = (dur / 4.0).clamp(0.0, 1.0);
        }

        features[16] = beat_position / 4.0;
        features[17] = phrase_position;
        features[18] = valence;
        features[19] = arousal;

        features
    }

    /// Convert predicted interval to frequency, snapping to nearest scale tone.
    pub fn interval_to_freq(
        &self,
        prev_freq: f32,
        predicted_interval: f32,
        scale_tones: &[f32],
    ) -> f32 {
        let target = prev_freq * 2.0f32.powf(predicted_interval / 12.0);

        // Snap to nearest scale tone
        if scale_tones.is_empty() {
            return target;
        }

        scale_tones
            .iter()
            .min_by(|a, b| ((**a - target).abs()).total_cmp(&((**b - target).abs())))
            .copied()
            .unwrap_or(target)
    }

    pub fn has_context(&self) -> bool {
        self.interval_history.len() >= 2
    }

    pub fn reset(&mut self) {
        self.interval_history.clear();
        self.duration_history.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predictor_returns_finite() {
        let mut pred = MelodyPredictor::new();
        pred.record(2.0, 1.0);
        pred.record(-1.0, 0.5);
        pred.record(3.0, 1.0);

        let (interval, duration) = pred.predict(0.0, 0.5, 0.5, 0.5);
        assert!(
            interval.is_finite(),
            "interval should be finite: {interval}"
        );
        assert!(
            duration.is_finite() && duration > 0.0,
            "duration should be positive: {duration}"
        );
    }

    #[test]
    fn sustained_run_predicts_reversal() {
        // The pre-training versions of these tests asserted direction
        // CONTINUATION ("ascending context should predict ascending") — the
        // hand-tuned prior's assumption. MAESTRO says otherwise: after a
        // sustained directional run, real melodies reverse ~64% of the time
        // (gap-fill / regression to the mean; the trained newest-interval
        // weight is strongly negative). Assert the empirical property.
        let mut up = MelodyPredictor::new();
        for _ in 0..8 {
            up.record(2.0, 1.0); // sustained ascending run
        }
        let (after_up, _) = up.predict(1.0, 0.3, 0.5, 0.5);
        assert!(
            after_up < 0.0,
            "sustained ascent should predict a reversal (descent): {after_up}"
        );

        let mut down = MelodyPredictor::new();
        for _ in 0..8 {
            down.record(-2.0, 1.0); // sustained descending run
        }
        let (after_down, _) = down.predict(1.0, 0.3, 0.0, 0.5);
        assert!(
            after_down > 0.0,
            "sustained descent should predict a reversal (ascent): {after_down}"
        );

        // And the two contexts must pull in opposite directions
        assert!(
            after_up < after_down,
            "prediction must depend on context direction: up→{after_up} down→{after_down}"
        );
    }

    #[test]
    fn high_arousal_shorter_notes() {
        let mut pred = MelodyPredictor::new();
        for _ in 0..4 {
            pred.record(0.0, 1.0);
        }

        let (_, dur_low) = pred.predict(0.0, 0.5, 0.0, 0.2);
        let (_, dur_high) = pred.predict(0.0, 0.5, 0.0, 0.9);
        assert!(
            dur_high < dur_low,
            "high arousal should predict shorter: high={dur_high} low={dur_low}"
        );
    }

    #[test]
    fn interval_to_freq_snaps_to_scale() {
        let pred = MelodyPredictor::new();
        let scale = vec![261.63, 293.66, 329.63, 349.23, 392.00, 440.00];
        let result = pred.interval_to_freq(261.63, 4.5, &scale); // ~4.5 semitones up from C
        // Should snap to E (329.63) which is 4 semitones up
        assert!((result - 329.63).abs() < 1.0, "should snap to E: {result}");
    }

    #[test]
    fn trained_weights_parse_from_committed_artifact() {
        // If the committed JSON ever fails to parse, weights() silently drops
        // to the hand-tuned fallback — this test makes that failure loud.
        let w = weights();
        assert_ne!(
            w.interval_weights, FALLBACK_INTERVAL_WEIGHTS,
            "committed melody_predictor_weights.json did not parse; \
             predictor is running on the untrained fallback"
        );
        assert!(
            TRAINED_JSON.contains("\"provenance\""),
            "weights artifact must embed its provenance block"
        );
        assert!(w.interval_weights.iter().all(|x| x.is_finite()));
        assert!(w.duration_weights.iter().all(|x| x.is_finite()));
    }
}
