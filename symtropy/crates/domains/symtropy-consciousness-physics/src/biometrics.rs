// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Biometric-to-Φ bridge: maps real physiological signals to [`ConsciousnessInputs`].
//!
//! Eight biometric channels each map to one component of the consciousness equation:
//!
//! | Channel | Maps to | Scientific basis |
//! |---------|---------|-----------------|
//! | HRV RMSSD (ms) | `synchrony` | Vagal tone predicts attentional coherence (Thayer & Lane 2009) |
//! | EEG gamma coherence (0–1) | `phi` | Long-range gamma binding ≈ IIT integration proxy (Tononi 2008) |
//! | GSR / EDA (μS) | `attention` | Yerkes-Dodson inverted-U arousal curve (Critchley 2002) |
//! | Respiration rate (bpm) | `embodiment` | Slow breath = 4E embodied grounding (Varela et al. 1991) |
//! | Heart rate (bpm) | `broadcast` | Cardiac-neural entrainment of global workspace (Park et al. 2014) |
//! | EEG alpha power (μV²/Hz) | `recurrence` | Alpha = default-mode recurrent processing (Dehaene & Changeux 2011) |
//! | EEG theta power (μV²/Hz) | `working_memory` | Frontal theta indexes WM load (Jensen & Tesche 2002) |
//! | EEG beta power (μV²/Hz) | `knowledge` | Beta tracks active knowledge integration (Engel & Fries 2010) |
//!
//! # Design
//!
//! All channels are optional — the bridge degrades gracefully to a neutral-0.5
//! contribution for any missing sensor. A rolling [`BiometricHistory`] window
//! smooths sensor noise before the values reach the consciousness equation.
//! Per-signal adaptive normalization calibrates to the player's individual
//! physiological range across the session.
//!
//! # Usage (from Bevy)
//!
//! ```ignore
//! // Push a sample each tick (or at sensor update rate):
//! history.push(BiometricSample { hrv_rmssd: Some(45.0), ..Default::default() });
//! // Convert to consciousness inputs:
//! let inputs = history.to_consciousness_inputs();
//! field.update_entity(player_handle, &inputs, player_pos);
//! ```

use std::collections::VecDeque;
use symthaea_consciousness_equation::ConsciousnessInputs;

// ────────────────────────────────────────────────────────────
//  Raw sample
// ────────────────────────────────────────────────────────────

/// A single biometric reading at one moment in time.
///
/// All fields are `Option<f64>` — the bridge works with any subset of sensors.
/// Missing channels produce a neutral 0.5 consciousness contribution.
#[derive(Debug, Clone, Default)]
pub struct BiometricSample {
    /// Heart Rate Variability (RMSSD) in milliseconds. Typical range: 10–100 ms.
    ///
    /// RMSSD is the root-mean-square of successive RR-interval differences.
    /// Higher = stronger parasympathetic tone = higher neural synchrony.
    /// Maps to: `synchrony`.
    pub hrv_rmssd: Option<f64>,

    /// EEG gamma-band coherence (30–80 Hz), dimensionless 0.0–1.0.
    ///
    /// Measures how correlated distant scalp regions are in the gamma band —
    /// a proxy for long-range neural integration (IIT-style binding).
    /// Maps to: `phi`.
    pub eeg_gamma_coherence: Option<f64>,

    /// Galvanic Skin Response / EDA in microsiemens (μS). Typical: 1–20 μS.
    ///
    /// Reflects sympathetic arousal. Optimal attentional engagement at ~5–10 μS
    /// (Yerkes-Dodson). Too low = under-aroused; too high = stressed.
    /// Maps to: `attention` (inverted-U curve).
    pub gsr_microsiemens: Option<f64>,

    /// Respiration rate in breaths per minute. Normal: 4–20 bpm.
    ///
    /// Coherent/diaphragmatic breathing (~5–6 bpm) maximizes cardiac-respiratory
    /// coupling and embodied grounding (Varela et al. 1991 — 4E cognition).
    /// Maps to: `embodiment` (inverted — slower = more grounded).
    pub respiration_bpm: Option<f64>,

    /// Heart rate in beats per minute. Typical: 50–100 bpm.
    ///
    /// Cardiac rhythms entrain neural oscillations. Resting/low HR correlates
    /// with optimal global broadcast (Park et al. 2014).
    /// Maps to: `broadcast` (inverted — lower HR = better entrainment).
    pub heart_rate_bpm: Option<f64>,

    /// EEG alpha power spectral density (8–12 Hz) in μV²/Hz. Typical: 0–100.
    ///
    /// Eyes-open alpha suppression during engagement; alpha rebound during
    /// default-mode / recurrent processing (Dehaene & Changeux 2011).
    /// Maps to: `recurrence`.
    pub eeg_alpha_power: Option<f64>,

    /// EEG theta power spectral density (4–8 Hz) in μV²/Hz. Typical: 0–50.
    ///
    /// Frontal midline theta peaks with working memory load (Jensen & Tesche 2002).
    /// Maps to: `working_memory`.
    pub eeg_theta_power: Option<f64>,

    /// EEG beta power spectral density (12–30 Hz) in μV²/Hz. Typical: 0–30.
    ///
    /// Active cognition, sensorimotor engagement, knowledge retrieval
    /// (Engel & Fries 2010).
    /// Maps to: `knowledge`.
    pub eeg_beta_power: Option<f64>,
}

// ────────────────────────────────────────────────────────────
//  Adaptive per-channel normalization
// ────────────────────────────────────────────────────────────

/// Adaptive normalization for a single biometric channel.
///
/// Calibrates to the player's individual physiological range during the session.
/// Soft min/max drift slowly toward observed extremes, then recover toward the
/// center when the signal stays in the middle — models homeostatic range tracking.
#[derive(Debug, Clone)]
pub struct ChannelNorm {
    /// Soft minimum (EMA-tracked, drifts toward observed lows).
    pub soft_min: f64,
    /// Soft maximum (EMA-tracked, drifts toward observed highs).
    pub soft_max: f64,
    /// EMA alpha for slow drift (default 0.005 ≈ 200-sample window at 1 Hz).
    pub alpha: f64,
}

impl ChannelNorm {
    /// Create with physiologically realistic initial range.
    pub fn new(initial_min: f64, initial_max: f64) -> Self {
        Self {
            soft_min: initial_min,
            soft_max: initial_max,
            alpha: 0.005,
        }
    }

    /// Update with a new sample and return the normalized value in [0.0, 1.0].
    pub fn update(&mut self, value: f64) -> f64 {
        // Expand range if new extremes observed
        if value < self.soft_min {
            self.soft_min = self.soft_min * (1.0 - self.alpha) + value * self.alpha;
        } else if value > self.soft_max {
            self.soft_max = self.soft_max * (1.0 - self.alpha) + value * self.alpha;
        }
        self.normalize(value)
    }

    /// Normalize a value against current range without updating state.
    pub fn normalize(&self, value: f64) -> f64 {
        let range = (self.soft_max - self.soft_min).max(1e-6);
        ((value - self.soft_min) / range).clamp(0.0, 1.0)
    }
}

/// Normalization state for all eight biometric channels.
#[derive(Debug, Clone)]
pub struct BiometricNormParams {
    pub hrv: ChannelNorm,
    pub gamma: ChannelNorm,
    pub gsr: ChannelNorm,
    pub respiration: ChannelNorm,
    pub heart_rate: ChannelNorm,
    pub alpha: ChannelNorm,
    pub theta: ChannelNorm,
    pub beta: ChannelNorm,
}

impl Default for BiometricNormParams {
    fn default() -> Self {
        Self {
            hrv: ChannelNorm::new(10.0, 100.0),        // ms RMSSD
            gamma: ChannelNorm::new(0.0, 1.0),         // dimensionless coherence
            gsr: ChannelNorm::new(0.5, 25.0),          // μS
            respiration: ChannelNorm::new(4.0, 24.0),  // bpm
            heart_rate: ChannelNorm::new(45.0, 110.0), // bpm
            alpha: ChannelNorm::new(0.5, 80.0),        // μV²/Hz
            theta: ChannelNorm::new(0.5, 40.0),        // μV²/Hz
            beta: ChannelNorm::new(0.5, 25.0),         // μV²/Hz
        }
    }
}

// ────────────────────────────────────────────────────────────
//  Rolling history + conversion
// ────────────────────────────────────────────────────────────

/// Rolling buffer of biometric samples with adaptive normalization.
///
/// Temporal smoothing prevents game mechanics from thrashing on sensor noise.
/// The default 30-sample window (~30 s @ 1 Hz) is long enough to dampen noise
/// but short enough to track genuine attentional shifts.
#[derive(Debug, Clone)]
pub struct BiometricHistory {
    samples: VecDeque<BiometricSample>,
    window: usize,
    /// Adaptive normalization state (updated on each `to_consciousness_inputs()` call).
    pub norm: BiometricNormParams,
}

impl Default for BiometricHistory {
    fn default() -> Self {
        Self::new(30)
    }
}

impl BiometricHistory {
    /// Create with the given smoothing window (number of samples).
    pub fn new(window: usize) -> Self {
        Self {
            samples: VecDeque::with_capacity(window),
            window: window.max(1),
            norm: BiometricNormParams::default(),
        }
    }

    /// Push a new sample, evicting the oldest if the window is full.
    pub fn push(&mut self, sample: BiometricSample) {
        if self.samples.len() >= self.window {
            self.samples.pop_front();
        }
        self.samples.push_back(sample);
    }

    /// Number of samples currently buffered.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// True if no samples have been pushed yet.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Reset both normalization state and sample buffer (e.g., new player session).
    pub fn reset(&mut self) {
        self.norm = BiometricNormParams::default();
        self.samples.clear();
    }

    /// Convert the current window to [`ConsciousnessInputs`].
    ///
    /// Averages raw values over the window, normalizes adaptively, then applies
    /// any channel-specific transforms (Yerkes-Dodson for GSR, inversion for HR
    /// and respiration). Missing channels default to 0.5 (neutral contribution).
    ///
    /// This call updates the adaptive normalization state.
    pub fn to_consciousness_inputs(&mut self) -> ConsciousnessInputs {
        if self.samples.is_empty() {
            return neutral_inputs();
        }

        // Accumulate per-channel sums and presence counts
        let mut hrv_sum = 0.0f64;
        let mut hrv_n = 0u32;
        let mut gam_sum = 0.0f64;
        let mut gam_n = 0u32;
        let mut gsr_sum = 0.0f64;
        let mut gsr_n = 0u32;
        let mut resp_sum = 0.0f64;
        let mut resp_n = 0u32;
        let mut hr_sum = 0.0f64;
        let mut hr_n = 0u32;
        let mut alp_sum = 0.0f64;
        let mut alp_n = 0u32;
        let mut the_sum = 0.0f64;
        let mut the_n = 0u32;
        let mut bet_sum = 0.0f64;
        let mut bet_n = 0u32;

        for s in &self.samples {
            if let Some(v) = s.hrv_rmssd {
                hrv_sum += v;
                hrv_n += 1;
            }
            if let Some(v) = s.eeg_gamma_coherence {
                gam_sum += v;
                gam_n += 1;
            }
            if let Some(v) = s.gsr_microsiemens {
                gsr_sum += v;
                gsr_n += 1;
            }
            if let Some(v) = s.respiration_bpm {
                resp_sum += v;
                resp_n += 1;
            }
            if let Some(v) = s.heart_rate_bpm {
                hr_sum += v;
                hr_n += 1;
            }
            if let Some(v) = s.eeg_alpha_power {
                alp_sum += v;
                alp_n += 1;
            }
            if let Some(v) = s.eeg_theta_power {
                the_sum += v;
                the_n += 1;
            }
            if let Some(v) = s.eeg_beta_power {
                bet_sum += v;
                bet_n += 1;
            }
        }

        // Majority-presence threshold: channel must be present in >50% of the window
        let n = self.samples.len() as u32;
        let thresh = n / 2;

        // phi ← gamma coherence (direct integration proxy, no inversion)
        let phi = if gam_n > thresh {
            self.norm.gamma.update(gam_sum / gam_n as f64)
        } else {
            0.5
        };

        // synchrony ← HRV RMSSD (higher = more synchrony)
        let synchrony = if hrv_n > thresh {
            self.norm.hrv.update(hrv_sum / hrv_n as f64)
        } else {
            0.5
        };

        // attention ← GSR (Yerkes-Dodson inverted-U: peak at mid-arousal)
        let attention = if gsr_n > thresh {
            let norm = self.norm.gsr.update(gsr_sum / gsr_n as f64);
            // Inverted-U: f(x) = 1 − 4(x − 0.5)²  →  1.0 at x=0.5, 0.0 at x=0 or x=1
            (1.0 - 4.0 * (norm - 0.5).powi(2)).clamp(0.0, 1.0)
        } else {
            0.5
        };

        // embodiment ← respiration (invert: slower rate = more grounded)
        let embodiment = if resp_n > thresh {
            let norm = self.norm.respiration.update(resp_sum / resp_n as f64);
            1.0 - norm
        } else {
            0.5
        };

        // broadcast ← heart rate (invert: lower rate = better cardiac-neural entrainment)
        let broadcast = if hr_n > thresh {
            let norm = self.norm.heart_rate.update(hr_sum / hr_n as f64);
            1.0 - norm
        } else {
            0.5
        };

        // recurrence ← alpha power (higher alpha = more recurrent/default-mode)
        let recurrence = if alp_n > thresh {
            self.norm.alpha.update(alp_sum / alp_n as f64)
        } else {
            0.5
        };

        // working_memory ← theta power
        let working_memory = if the_n > thresh {
            self.norm.theta.update(the_sum / the_n as f64)
        } else {
            0.5
        };

        // knowledge ← beta power
        let knowledge = if bet_n > thresh {
            self.norm.beta.update(bet_sum / bet_n as f64)
        } else {
            0.5
        };

        ConsciousnessInputs {
            phi,
            broadcast,
            working_memory,
            attention,
            recurrence,
            embodiment,
            knowledge,
            synchrony,
        }
    }
}

// ────────────────────────────────────────────────────────────
//  Helpers
// ────────────────────────────────────────────────────────────

/// Neutral inputs: all channels at 0.5 (no sensor data).
pub fn neutral_inputs() -> ConsciousnessInputs {
    ConsciousnessInputs {
        phi: 0.5,
        broadcast: 0.5,
        working_memory: 0.5,
        attention: 0.5,
        recurrence: 0.5,
        embodiment: 0.5,
        knowledge: 0.5,
        synchrony: 0.5,
    }
}

// ────────────────────────────────────────────────────────────
//  Tests
// ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_full_sample(v: f64) -> BiometricSample {
        BiometricSample {
            hrv_rmssd: Some(v * 100.0), // 0→0 ms, 1→100 ms
            eeg_gamma_coherence: Some(v),
            gsr_microsiemens: Some(v * 20.0),
            respiration_bpm: Some(4.0 + v * 20.0),
            heart_rate_bpm: Some(45.0 + v * 65.0),
            eeg_alpha_power: Some(v * 80.0),
            eeg_theta_power: Some(v * 40.0),
            eeg_beta_power: Some(v * 25.0),
        }
    }

    #[test]
    fn empty_history_returns_neutral() {
        let mut h = BiometricHistory::default();
        let inputs = h.to_consciousness_inputs();
        assert!((inputs.phi - 0.5).abs() < 1e-9);
        assert!((inputs.synchrony - 0.5).abs() < 1e-9);
    }

    #[test]
    fn single_full_sample_produces_valid_range() {
        let mut h = BiometricHistory::new(1);
        h.push(make_full_sample(0.8));
        let inputs = h.to_consciousness_inputs();
        // All values should be within [0, 1]
        assert!((0.0..=1.0).contains(&inputs.phi));
        assert!((0.0..=1.0).contains(&inputs.synchrony));
        assert!((0.0..=1.0).contains(&inputs.attention));
        assert!((0.0..=1.0).contains(&inputs.embodiment));
        assert!((0.0..=1.0).contains(&inputs.broadcast));
        assert!((0.0..=1.0).contains(&inputs.recurrence));
        assert!((0.0..=1.0).contains(&inputs.working_memory));
        assert!((0.0..=1.0).contains(&inputs.knowledge));
    }

    #[test]
    fn high_hrv_produces_high_synchrony() {
        let mut h = BiometricHistory::new(5);
        // Fill window with high-HRV samples (80ms — near top of default range)
        for _ in 0..5 {
            h.push(BiometricSample {
                hrv_rmssd: Some(85.0),
                ..Default::default()
            });
        }
        let inputs = h.to_consciousness_inputs();
        assert!(
            inputs.synchrony > 0.6,
            "high HRV should yield high synchrony, got {}",
            inputs.synchrony
        );
    }

    #[test]
    fn low_hrv_produces_low_synchrony() {
        let mut h = BiometricHistory::new(5);
        for _ in 0..5 {
            h.push(BiometricSample {
                hrv_rmssd: Some(12.0),
                ..Default::default()
            });
        }
        let inputs = h.to_consciousness_inputs();
        assert!(
            inputs.synchrony < 0.4,
            "low HRV should yield low synchrony, got {}",
            inputs.synchrony
        );
    }

    #[test]
    fn slow_respiration_produces_high_embodiment() {
        let mut h = BiometricHistory::new(5);
        // 5 bpm — coherent breathing (near bottom of range)
        for _ in 0..5 {
            h.push(BiometricSample {
                respiration_bpm: Some(5.0),
                ..Default::default()
            });
        }
        let inputs = h.to_consciousness_inputs();
        assert!(
            inputs.embodiment > 0.6,
            "slow breathing should yield high embodiment, got {}",
            inputs.embodiment
        );
    }

    #[test]
    fn fast_respiration_produces_low_embodiment() {
        let mut h = BiometricHistory::new(5);
        // 22 bpm — near top of range
        for _ in 0..5 {
            h.push(BiometricSample {
                respiration_bpm: Some(22.0),
                ..Default::default()
            });
        }
        let inputs = h.to_consciousness_inputs();
        assert!(
            inputs.embodiment < 0.4,
            "fast breathing should yield low embodiment, got {}",
            inputs.embodiment
        );
    }

    #[test]
    fn gsr_inverted_u_peaks_at_mid_arousal() {
        let mut low = BiometricHistory::new(5);
        let mut mid = BiometricHistory::new(5);
        let mut high = BiometricHistory::new(5);

        // GSR default range: 0.5–25 μS. Mid ≈ 12 μS.
        for _ in 0..5 {
            low.push(BiometricSample {
                gsr_microsiemens: Some(1.0),
                ..Default::default()
            });
            mid.push(BiometricSample {
                gsr_microsiemens: Some(12.5),
                ..Default::default()
            });
            high.push(BiometricSample {
                gsr_microsiemens: Some(24.0),
                ..Default::default()
            });
        }
        let attn_low = low.to_consciousness_inputs().attention;
        let attn_mid = mid.to_consciousness_inputs().attention;
        let attn_high = high.to_consciousness_inputs().attention;
        assert!(
            attn_mid > attn_low,
            "mid GSR should yield more attention than low: {attn_mid} vs {attn_low}"
        );
        assert!(
            attn_mid > attn_high,
            "mid GSR should yield more attention than high: {attn_mid} vs {attn_high}"
        );
    }

    #[test]
    fn missing_channels_return_neutral() {
        let mut h = BiometricHistory::new(5);
        // Only HRV set; all others missing
        for _ in 0..5 {
            h.push(BiometricSample {
                hrv_rmssd: Some(50.0),
                ..Default::default()
            });
        }
        let inputs = h.to_consciousness_inputs();
        assert!((inputs.phi - 0.5).abs() < 1e-9, "missing gamma → phi=0.5");
        assert!(
            (inputs.attention - 0.5).abs() < 1e-9,
            "missing GSR → attention=0.5"
        );
        assert!(
            (inputs.embodiment - 0.5).abs() < 1e-9,
            "missing respiration → embodiment=0.5"
        );
        assert!(
            (inputs.broadcast - 0.5).abs() < 1e-9,
            "missing HR → broadcast=0.5"
        );
        assert!(
            (inputs.recurrence - 0.5).abs() < 1e-9,
            "missing alpha → recurrence=0.5"
        );
        assert!(
            (inputs.working_memory - 0.5).abs() < 1e-9,
            "missing theta → working_memory=0.5"
        );
        assert!(
            (inputs.knowledge - 0.5).abs() < 1e-9,
            "missing beta → knowledge=0.5"
        );
    }

    #[test]
    fn window_evicts_old_samples() {
        let mut h = BiometricHistory::new(3);
        // Push 5 samples: first 3 should be evicted
        for i in 0..5u32 {
            h.push(BiometricSample {
                hrv_rmssd: Some(i as f64 * 10.0),
                ..Default::default()
            });
        }
        assert_eq!(h.len(), 3, "window should hold at most 3 samples");
    }

    #[test]
    fn reset_clears_history_and_norm() {
        let mut h = BiometricHistory::new(10);
        for _ in 0..10 {
            h.push(make_full_sample(0.9));
        }
        h.reset();
        assert!(h.is_empty());
        // After reset, norm should be back to defaults (check HRV range)
        assert!((h.norm.hrv.soft_min - 10.0).abs() < 1.0);
        assert!((h.norm.hrv.soft_max - 100.0).abs() < 1.0);
    }

    #[test]
    fn channel_norm_update_expands_range() {
        let mut norm = ChannelNorm::new(10.0, 50.0);
        // Push a value below the soft_min
        let v = norm.update(5.0);
        assert!(
            v < 0.1,
            "value below soft_min should normalize near 0, got {v}"
        );
        assert!(norm.soft_min < 10.0, "soft_min should have drifted down");
    }

    #[test]
    fn channel_norm_normalize_clamps_to_unit_interval() {
        let norm = ChannelNorm::new(0.0, 100.0);
        assert_eq!(norm.normalize(-10.0), 0.0);
        assert_eq!(norm.normalize(150.0), 1.0);
        assert!((norm.normalize(50.0) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn neutral_inputs_are_all_half() {
        let n = neutral_inputs();
        assert_eq!(n.phi, 0.5);
        assert_eq!(n.synchrony, 0.5);
        assert_eq!(n.attention, 0.5);
        assert_eq!(n.embodiment, 0.5);
        assert_eq!(n.broadcast, 0.5);
        assert_eq!(n.recurrence, 0.5);
        assert_eq!(n.working_memory, 0.5);
        assert_eq!(n.knowledge, 0.5);
    }
}
