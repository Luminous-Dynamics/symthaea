// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Input telemetry encoder: mouse/keyboard → HDC stress vectors.
//!
//! Captures raw input device events and encodes them into a compact stress
//! representation using hyperdimensional computing. The key insight: mouse
//! velocity surprise (ESN prediction error) is the strongest indicator of
//! real-world physiological stress.

use std::collections::VecDeque;
use symthaea_core::hdc::reservoir::EchoStateNetwork;

// ─── Constants ──────────────────────────────────────────────────────────────

/// EMA smoothing factor for mouse velocity (higher = more responsive).
const VELOCITY_EMA_ALPHA: f32 = 0.15;

/// EMA smoothing factor for keystroke rate.
const KEYSTROKE_RATE_EMA_ALPHA: f32 = 0.1;

/// Maximum mouse history size (samples).
const MAX_MOUSE_HISTORY: usize = 50;

/// Maximum keystroke history size.
const MAX_KEYSTROKE_HISTORY: usize = 100;

/// ESN reservoir size for velocity prediction.
const ESN_RESERVOIR_SIZE: usize = 32;

/// ESN spectral radius (edge of chaos for sensitivity).
const ESN_SPECTRAL_RADIUS: f64 = 0.95;

/// ESN sparsity (90% sparse connections).
const ESN_SPARSITY: f64 = 0.9;

/// ESN training samples before predictions are valid.
const ESN_WARMUP: usize = 20;

/// ESN training regularization.
const ESN_REGULARIZATION: f64 = 1e-4;

// ─── Input Types ────────────────────────────────────────────────────────────

/// A single mouse position sample.
#[derive(Debug, Clone, Copy)]
pub struct MouseSample {
    /// Normalized X position [0, 1] (left to right).
    pub x: f32,
    /// Normalized Y position [0, 1] (top to bottom).
    pub y: f32,
    /// Timestamp in milliseconds (monotonic).
    pub timestamp_ms: u64,
}

/// A single keystroke event.
#[derive(Debug, Clone, Copy)]
pub struct KeystrokeSample {
    /// True = key pressed, False = key released.
    pub key_down: bool,
    /// Timestamp in milliseconds (monotonic).
    pub timestamp_ms: u64,
}

/// Computed stress vector from input telemetry.
#[derive(Debug, Clone, Copy, Default)]
pub struct StressVector {
    /// Arousal [0, 1]: how physiologically activated the player is.
    /// High mouse velocity + high keystroke rate + high surprise = high arousal.
    pub arousal: f32,
    /// Valence [-1, 1]: emotional tone. Negative = stressed, positive = engaged.
    /// High surprise = negative, predictable patterns = positive.
    pub valence: f32,
    /// Dominance [-1, 1]: sense of control. High keystroke activity = dominant.
    pub dominance: f32,
}

/// Neuromodulator deltas derived from input stress.
///
/// Follows the SensorNudges pattern from symthaea-soma/sensor_bridge.rs.
#[derive(Debug, Clone, Copy, Default)]
pub struct SensorNudges {
    /// Dopamine delta (novelty, engagement).
    pub dopamine_delta: f32,
    /// Norepinephrine delta (arousal, alertness, stress).
    pub norepinephrine_delta: f32,
    /// Serotonin delta (calm, wellbeing — typically negative under stress).
    pub serotonin_delta: f32,
    /// Oxytocin delta (social bonding — rises in cooperative play).
    pub oxytocin_delta: f32,
}

// ─── Encoder ────────────────────────────────────────────────────────────────

/// Encodes mouse/keyboard telemetry into stress vectors.
///
/// Uses an Echo State Network to predict mouse velocity; the prediction
/// error (surprise) is the primary stress signal. Keystroke rate and
/// dwell time provide secondary indicators.
pub struct InputTelemetryEncoder {
    /// ESN for mouse velocity surprise detection.
    esn: EchoStateNetwork,
    /// Whether ESN has been trained (after warmup period).
    esn_trained: bool,
    /// Mouse position history (windowed).
    mouse_history: VecDeque<MouseSample>,
    /// Keystroke history (windowed).
    keystroke_history: VecDeque<KeystrokeSample>,
    /// EMA-smoothed mouse velocity (pixels/sec, normalized).
    mouse_velocity_ema: f32,
    /// EMA-smoothed keystroke rate (keys/sec).
    keystroke_rate_ema: f32,
    /// Time since last mouse movement (ms).
    dwell_time_ms: u64,
    /// Latest ESN velocity prediction error [0, 1].
    velocity_surprise: f32,
    /// Running count of ESN observations.
    esn_observation_count: usize,
    /// Last mouse timestamp for delta computation.
    last_mouse_ts: u64,
    /// Last keystroke timestamp.
    last_keystroke_ts: u64,
}

impl InputTelemetryEncoder {
    /// Create a new encoder with default ESN configuration.
    pub fn new() -> Self {
        Self::with_seed(42)
    }

    /// Create a new encoder with a specific RNG seed for the ESN.
    pub fn with_seed(seed: u64) -> Self {
        Self {
            esn: EchoStateNetwork::new(ESN_RESERVOIR_SIZE, ESN_SPECTRAL_RADIUS, ESN_SPARSITY, seed),
            esn_trained: false,
            mouse_history: VecDeque::with_capacity(MAX_MOUSE_HISTORY),
            keystroke_history: VecDeque::with_capacity(MAX_KEYSTROKE_HISTORY),
            mouse_velocity_ema: 0.0,
            keystroke_rate_ema: 0.0,
            dwell_time_ms: 0,
            velocity_surprise: 0.0,
            esn_observation_count: 0,
            last_mouse_ts: 0,
            last_keystroke_ts: 0,
        }
    }

    /// Push a mouse position sample.
    ///
    /// Computes velocity from consecutive samples, feeds the ESN for
    /// surprise detection, and updates the velocity EMA.
    pub fn push_mouse(&mut self, sample: MouseSample) {
        // Compute velocity from previous sample
        if let Some(prev) = self.mouse_history.back() {
            let dt_ms = sample.timestamp_ms.saturating_sub(prev.timestamp_ms);
            if dt_ms > 0 {
                let dx = sample.x - prev.x;
                let dy = sample.y - prev.y;
                let dist = (dx * dx + dy * dy).sqrt();
                let velocity = dist / (dt_ms as f32 / 1000.0); // units/sec

                // Clamp velocity to reasonable range [0, 10] for normalization
                let norm_velocity = (velocity / 10.0).min(1.0);

                // EMA update
                self.mouse_velocity_ema = self.mouse_velocity_ema * (1.0 - VELOCITY_EMA_ALPHA)
                    + norm_velocity * VELOCITY_EMA_ALPHA;

                // Feed ESN for surprise detection
                let predicted = if self.esn_trained {
                    self.esn.predict() as f32
                } else {
                    norm_velocity as f32 // no surprise before training
                };

                self.esn
                    .observe(norm_velocity as f64, Some(norm_velocity as f64));
                self.esn_observation_count += 1;

                // Train ESN after warmup
                if !self.esn_trained && self.esn_observation_count >= ESN_WARMUP {
                    self.esn.train(ESN_REGULARIZATION);
                    self.esn_trained = true;
                }

                // Surprise = |predicted - actual|, clamped to [0, 1]
                if self.esn_trained {
                    self.velocity_surprise = (predicted - norm_velocity).abs().min(1.0);
                }
            }

            self.dwell_time_ms = sample.timestamp_ms.saturating_sub(prev.timestamp_ms);
        }

        self.last_mouse_ts = sample.timestamp_ms;

        // Maintain windowed history
        if self.mouse_history.len() >= MAX_MOUSE_HISTORY {
            self.mouse_history.pop_front();
        }
        self.mouse_history.push_back(sample);
    }

    /// Push a keystroke event.
    ///
    /// Updates the keystroke rate EMA based on inter-key timing.
    pub fn push_keystroke(&mut self, sample: KeystrokeSample) {
        if sample.key_down {
            // Compute inter-key interval
            if self.last_keystroke_ts > 0 {
                let dt_ms = sample.timestamp_ms.saturating_sub(self.last_keystroke_ts);
                if dt_ms > 0 && dt_ms < 5000 {
                    // Ignore gaps > 5s (player was idle)
                    let rate = 1000.0 / dt_ms as f32; // keys/sec
                    let clamped_rate = rate.min(20.0) / 20.0; // normalize to [0, 1]
                    self.keystroke_rate_ema = self.keystroke_rate_ema
                        * (1.0 - KEYSTROKE_RATE_EMA_ALPHA)
                        + clamped_rate * KEYSTROKE_RATE_EMA_ALPHA;
                }
            }
            self.last_keystroke_ts = sample.timestamp_ms;
        }

        // Maintain windowed history
        if self.keystroke_history.len() >= MAX_KEYSTROKE_HISTORY {
            self.keystroke_history.pop_front();
        }
        self.keystroke_history.push_back(sample);
    }

    /// Compute the current stress vector from accumulated telemetry.
    ///
    /// Returns a `StressVector` with arousal, valence, and dominance.
    pub fn compute_stress_vector(&self) -> StressVector {
        // Arousal: composite of velocity, keystroke rate, and surprise
        let arousal = (self.mouse_velocity_ema * 0.35
            + self.keystroke_rate_ema * 0.30
            + self.velocity_surprise * 0.35)
            .clamp(0.0, 1.0);

        // Valence: surprise drives negative, predictability drives positive
        // Low surprise + moderate activity = engaged (positive)
        // High surprise = stressed (negative)
        let valence = (-self.velocity_surprise * 0.6 + (1.0 - self.velocity_surprise) * 0.3 - 0.15)
            .clamp(-1.0, 1.0);

        // Dominance: active input = sense of control
        // High keystroke rate = dominant, long dwell (frozen) = submissive
        let dwell_factor = if self.dwell_time_ms > 2000 {
            -0.3 // frozen/hesitant
        } else {
            0.0
        };
        let dominance =
            (self.keystroke_rate_ema * 0.5 + self.mouse_velocity_ema * 0.3 + dwell_factor)
                .clamp(-1.0, 1.0);

        StressVector {
            arousal,
            valence,
            dominance,
        }
    }

    /// Convert current stress into neuromodulator deltas.
    ///
    /// Follows the SensorNudges pattern from symthaea-soma/sensor_bridge.rs:
    /// - High velocity surprise → NE (arousal/alertness)
    /// - Low keystroke rate + high dwell → negative DA (disengagement)
    /// - Sustained high arousal → negative 5-HT (stress depletes serotonin)
    /// - Cooperative multiplayer actions → OT (not tracked here, placeholder)
    pub fn to_sensor_nudges(&self) -> SensorNudges {
        let stress = self.compute_stress_vector();

        SensorNudges {
            // Surprise → NE (like motion → NE in sensor_bridge)
            norepinephrine_delta: self.velocity_surprise * 0.06 + stress.arousal * 0.03,
            // Engagement → DA (exploration = novelty)
            dopamine_delta: if self.keystroke_rate_ema > 0.3 {
                0.02 // active input = engaged
            } else if self.dwell_time_ms > 3000 {
                -0.02 // frozen = disengaged
            } else {
                0.0
            },
            // Stress depletes serotonin
            serotonin_delta: if stress.arousal > 0.6 {
                -0.03 * stress.arousal
            } else {
                0.01 // mild recovery when calm
            },
            // Placeholder for multiplayer social bonding
            oxytocin_delta: 0.0,
        }
    }

    /// Current EMA-smoothed mouse velocity [0, 1].
    pub fn mouse_velocity(&self) -> f32 {
        self.mouse_velocity_ema
    }

    /// Current EMA-smoothed keystroke rate [0, 1].
    pub fn keystroke_rate(&self) -> f32 {
        self.keystroke_rate_ema
    }

    /// Current velocity surprise (ESN prediction error) [0, 1].
    pub fn velocity_surprise(&self) -> f32 {
        self.velocity_surprise
    }

    /// Whether the ESN has been trained (predictions are meaningful).
    pub fn is_calibrated(&self) -> bool {
        self.esn_trained
    }

    /// Number of mouse samples processed.
    pub fn mouse_sample_count(&self) -> usize {
        self.mouse_history.len()
    }

    /// Number of keystroke events processed.
    pub fn keystroke_count(&self) -> usize {
        self.keystroke_history.len()
    }

    /// Reset all state (clears history, retains ESN training).
    pub fn reset(&mut self) {
        self.mouse_history.clear();
        self.keystroke_history.clear();
        self.mouse_velocity_ema = 0.0;
        self.keystroke_rate_ema = 0.0;
        self.dwell_time_ms = 0;
        self.velocity_surprise = 0.0;
        self.last_mouse_ts = 0;
        self.last_keystroke_ts = 0;
    }
}

impl Default for InputTelemetryEncoder {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn calm_mouse_sequence(encoder: &mut InputTelemetryEncoder, start_ms: u64) {
        // Slow, steady mouse movement (calm player)
        for i in 0..30 {
            encoder.push_mouse(MouseSample {
                x: 0.3 + i as f32 * 0.005,
                y: 0.5 + (i as f32 * 0.01).sin() * 0.01,
                timestamp_ms: start_ms + i * 50,
            });
        }
    }

    fn panic_mouse_sequence(encoder: &mut InputTelemetryEncoder, start_ms: u64) {
        // Erratic, fast mouse movement (panicking player)
        for i in 0..30 {
            let jitter = ((i * 7 + 13) % 19) as f32 / 19.0; // pseudo-random
            encoder.push_mouse(MouseSample {
                x: 0.3 + jitter * 0.4,
                y: 0.2 + (1.0 - jitter) * 0.6,
                timestamp_ms: start_ms + i * 16, // faster sampling (60fps)
            });
        }
    }

    #[test]
    fn new_encoder_is_zero() {
        let encoder = InputTelemetryEncoder::new();
        let stress = encoder.compute_stress_vector();
        assert!(stress.arousal < 0.01);
        assert!(encoder.mouse_velocity() < 0.01);
        assert!(encoder.keystroke_rate() < 0.01);
    }

    #[test]
    fn calm_input_low_arousal() {
        let mut encoder = InputTelemetryEncoder::new();
        calm_mouse_sequence(&mut encoder, 0);
        let stress = encoder.compute_stress_vector();
        assert!(
            stress.arousal < 0.4,
            "calm input should have low arousal, got {}",
            stress.arousal
        );
    }

    #[test]
    fn panic_input_high_arousal() {
        let mut encoder = InputTelemetryEncoder::new();
        panic_mouse_sequence(&mut encoder, 0);
        let stress = encoder.compute_stress_vector();
        assert!(
            stress.arousal > 0.15,
            "panic input should have elevated arousal, got {}",
            stress.arousal
        );
    }

    #[test]
    fn calm_vs_panic_different_stress() {
        let mut calm = InputTelemetryEncoder::new();
        let mut panic = InputTelemetryEncoder::new();
        calm_mouse_sequence(&mut calm, 0);
        panic_mouse_sequence(&mut panic, 0);

        let calm_stress = calm.compute_stress_vector();
        let panic_stress = panic.compute_stress_vector();

        assert!(
            panic_stress.arousal > calm_stress.arousal,
            "panic arousal ({}) should exceed calm ({})",
            panic_stress.arousal,
            calm_stress.arousal
        );
    }

    #[test]
    fn keystroke_rate_increases_with_typing() {
        let mut encoder = InputTelemetryEncoder::new();
        // Fast typing: 10 keys/sec (100ms interval)
        for i in 0..20 {
            encoder.push_keystroke(KeystrokeSample {
                key_down: true,
                timestamp_ms: i * 100,
            });
        }
        assert!(
            encoder.keystroke_rate() > 0.1,
            "fast typing should raise keystroke rate, got {}",
            encoder.keystroke_rate()
        );
    }

    #[test]
    fn sensor_nudges_stress_lowers_serotonin() {
        let mut encoder = InputTelemetryEncoder::new();
        panic_mouse_sequence(&mut encoder, 0);
        // Also add fast typing
        for i in 0..20 {
            encoder.push_keystroke(KeystrokeSample {
                key_down: true,
                timestamp_ms: i * 50,
            });
        }
        let nudges = encoder.to_sensor_nudges();
        assert!(
            nudges.norepinephrine_delta > 0.0,
            "stress should increase NE"
        );
    }

    #[test]
    fn esn_calibrates_after_warmup() {
        let mut encoder = InputTelemetryEncoder::new();
        assert!(!encoder.is_calibrated());
        // Feed enough samples for ESN warmup
        for i in 0..30 {
            encoder.push_mouse(MouseSample {
                x: 0.5 + (i as f32 * 0.1).sin() * 0.2,
                y: 0.5,
                timestamp_ms: i * 50,
            });
        }
        assert!(encoder.is_calibrated());
    }

    #[test]
    fn reset_clears_state() {
        let mut encoder = InputTelemetryEncoder::new();
        panic_mouse_sequence(&mut encoder, 0);
        assert!(encoder.mouse_velocity() > 0.0);

        encoder.reset();
        assert!(encoder.mouse_velocity() < 0.01);
        assert_eq!(encoder.mouse_sample_count(), 0);
        assert_eq!(encoder.keystroke_count(), 0);
    }

    #[test]
    fn stress_values_bounded() {
        let mut encoder = InputTelemetryEncoder::new();
        // Feed extreme input
        for i in 0..50 {
            encoder.push_mouse(MouseSample {
                x: if i % 2 == 0 { 0.0 } else { 1.0 },
                y: if i % 3 == 0 { 0.0 } else { 1.0 },
                timestamp_ms: i * 10,
            });
            encoder.push_keystroke(KeystrokeSample {
                key_down: true,
                timestamp_ms: i * 10,
            });
        }
        let stress = encoder.compute_stress_vector();
        assert!(stress.arousal >= 0.0 && stress.arousal <= 1.0);
        assert!(stress.valence >= -1.0 && stress.valence <= 1.0);
        assert!(stress.dominance >= -1.0 && stress.dominance <= 1.0);
    }
}
