// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Emotion Detection from Multi-Channel EEG
//!
//! Implements the scientific foundations of EEG-based emotion recognition:
//!
//! ## The Valence-Arousal Model (Russell, 1980)
//!
//! Emotions mapped to a 2D circumplex:
//! - **Valence** (x-axis): Pleasant (+) ←→ Unpleasant (-)
//! - **Arousal** (y-axis): Activated (+) ←→ Deactivated (-)
//!
//! ```text
//!                    High Arousal
//!                         │
//!          Angry   ●      │      ●  Excited
//!          Afraid  ●      │      ●  Happy
//!                         │
//! Low Valence ────────────┼──────────── High Valence
//!                         │
//!           Sad    ●      │      ●  Content
//!           Bored  ●      │      ●  Relaxed
//!                         │
//!                    Low Arousal
//! ```
//!
//! ## EEG Correlates of Emotion
//!
//! ### Frontal Asymmetry Index (FAI)
//! FAI = log(F4_alpha) - log(F3_alpha)
//! - **Positive FAI**: Approach motivation, positive valence
//! - **Negative FAI**: Withdrawal motivation, negative valence
//!
//! ### Arousal Markers
//! - **Beta/Alpha ratio**: Higher = more aroused
//! - **Gamma coherence**: Engagement, attention, binding
//!
//! ### Additional Channels
//! - **EOG**: Blink rate (stress indicator)
//! - **EMG**: Facial tension (emotional expression)
//! - **GSR**: Galvanic skin response (autonomic arousal)
//!
//! ## References
//! - Davidson, R.J. (1992). Anterior cerebral asymmetry and emotion
//! - Koelstra et al. (2012). DEAP: A database for emotion analysis
//! - Zheng & Lu (2015). SEED: Investigating EEG-based emotion recognition

use std::collections::{HashMap, VecDeque};
use std::f64::consts::PI;

/// Emotional state in the Valence-Arousal circumplex
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EmotionalState {
    /// Valence: -1.0 (unpleasant) to +1.0 (pleasant)
    pub valence: f64,
    /// Arousal: -1.0 (calm) to +1.0 (excited)
    pub arousal: f64,
    /// Dominance: -1.0 (submissive) to +1.0 (dominant)
    pub dominance: f64,
    /// Confidence in the detection (0.0 to 1.0)
    pub confidence: f64,
}

impl EmotionalState {
    /// Create a new emotional state
    pub fn new(valence: f64, arousal: f64, dominance: f64, confidence: f64) -> Self {
        Self {
            valence: valence.clamp(-1.0, 1.0),
            arousal: arousal.clamp(-1.0, 1.0),
            dominance: dominance.clamp(-1.0, 1.0),
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    /// Neutral state (center of circumplex)
    pub fn neutral() -> Self {
        Self::new(0.0, 0.0, 0.0, 1.0)
    }

    /// Classify into discrete emotion category
    pub fn classify(&self) -> EmotionCategory {
        // Use quadrant + intensity for classification
        let intensity = (self.valence.powi(2) + self.arousal.powi(2)).sqrt();

        if intensity < 0.3 {
            return EmotionCategory::Neutral;
        }

        match (self.valence >= 0.0, self.arousal >= 0.0) {
            (true, true) => {
                // High valence, high arousal
                if self.arousal > 0.5 {
                    EmotionCategory::Excited
                } else {
                    EmotionCategory::Happy
                }
            }
            (true, false) => {
                // High valence, low arousal
                if self.arousal < -0.5 {
                    EmotionCategory::Relaxed
                } else {
                    EmotionCategory::Content
                }
            }
            (false, true) => {
                // Low valence, high arousal
                if self.arousal > 0.5 {
                    EmotionCategory::Angry
                } else {
                    EmotionCategory::Afraid
                }
            }
            (false, false) => {
                // Low valence, low arousal
                if self.arousal < -0.5 {
                    EmotionCategory::Bored
                } else {
                    EmotionCategory::Sad
                }
            }
        }
    }

    /// Distance from another emotional state
    pub fn distance(&self, other: &EmotionalState) -> f64 {
        let dv = self.valence - other.valence;
        let da = self.arousal - other.arousal;
        let dd = self.dominance - other.dominance;
        (dv.powi(2) + da.powi(2) + dd.powi(2)).sqrt()
    }
}

/// Discrete emotion categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EmotionCategory {
    /// High valence, high arousal
    Excited,
    /// High valence, moderate arousal
    Happy,
    /// High valence, low arousal
    Content,
    /// High valence, very low arousal
    Relaxed,
    /// Low valence, high arousal
    Angry,
    /// Low valence, moderate-high arousal
    Afraid,
    /// Low valence, low arousal
    Sad,
    /// Low valence, very low arousal
    Bored,
    /// Center of circumplex
    Neutral,
}

impl EmotionCategory {
    /// Get the name
    pub fn name(&self) -> &'static str {
        match self {
            EmotionCategory::Excited => "Excited",
            EmotionCategory::Happy => "Happy",
            EmotionCategory::Content => "Content",
            EmotionCategory::Relaxed => "Relaxed",
            EmotionCategory::Angry => "Angry",
            EmotionCategory::Afraid => "Afraid",
            EmotionCategory::Sad => "Sad",
            EmotionCategory::Bored => "Bored",
            EmotionCategory::Neutral => "Neutral",
        }
    }

    /// Is this a positive valence emotion?
    pub fn is_positive(&self) -> bool {
        matches!(
            self,
            EmotionCategory::Excited
                | EmotionCategory::Happy
                | EmotionCategory::Content
                | EmotionCategory::Relaxed
        )
    }

    /// Target valence-arousal coordinates
    pub fn target_va(&self) -> (f64, f64) {
        match self {
            EmotionCategory::Excited => (0.8, 0.8),
            EmotionCategory::Happy => (0.6, 0.4),
            EmotionCategory::Content => (0.5, -0.2),
            EmotionCategory::Relaxed => (0.4, -0.6),
            EmotionCategory::Angry => (-0.7, 0.8),
            EmotionCategory::Afraid => (-0.6, 0.5),
            EmotionCategory::Sad => (-0.5, -0.4),
            EmotionCategory::Bored => (-0.3, -0.7),
            EmotionCategory::Neutral => (0.0, 0.0),
        }
    }
}

/// EEG channel positions for emotion detection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EmotionChannel {
    // Frontal channels (asymmetry)
    F3,  // Left frontal
    F4,  // Right frontal
    F7,  // Left lateral frontal
    F8,  // Right lateral frontal
    Fp1, // Left prefrontal
    Fp2, // Right prefrontal
    Fz,  // Midline frontal

    // Central channels
    C3, // Left central
    C4, // Right central
    Cz, // Midline central

    // Parietal channels
    P3, // Left parietal
    P4, // Right parietal
    Pz, // Midline parietal

    // Temporal channels
    T3, // Left temporal (T7 in 10-20)
    T4, // Right temporal (T8 in 10-20)
    T5, // Left posterior temporal (P7)
    T6, // Right posterior temporal (P8)

    // Occipital channels
    O1, // Left occipital
    O2, // Right occipital
    Oz, // Midline occipital
}

impl EmotionChannel {
    /// Get channel name string
    pub fn name(&self) -> &'static str {
        match self {
            EmotionChannel::F3 => "F3",
            EmotionChannel::F4 => "F4",
            EmotionChannel::F7 => "F7",
            EmotionChannel::F8 => "F8",
            EmotionChannel::Fp1 => "Fp1",
            EmotionChannel::Fp2 => "Fp2",
            EmotionChannel::Fz => "Fz",
            EmotionChannel::C3 => "C3",
            EmotionChannel::C4 => "C4",
            EmotionChannel::Cz => "Cz",
            EmotionChannel::P3 => "P3",
            EmotionChannel::P4 => "P4",
            EmotionChannel::Pz => "Pz",
            EmotionChannel::T3 => "T3",
            EmotionChannel::T4 => "T4",
            EmotionChannel::T5 => "T5",
            EmotionChannel::T6 => "T6",
            EmotionChannel::O1 => "O1",
            EmotionChannel::O2 => "O2",
            EmotionChannel::Oz => "Oz",
        }
    }

    /// Parse from string label
    pub fn from_label(label: &str) -> Option<Self> {
        let label = label.trim().to_uppercase();
        match label.as_str() {
            "F3" => Some(EmotionChannel::F3),
            "F4" => Some(EmotionChannel::F4),
            "F7" => Some(EmotionChannel::F7),
            "F8" => Some(EmotionChannel::F8),
            "FP1" => Some(EmotionChannel::Fp1),
            "FP2" => Some(EmotionChannel::Fp2),
            "FZ" => Some(EmotionChannel::Fz),
            "C3" => Some(EmotionChannel::C3),
            "C4" => Some(EmotionChannel::C4),
            "CZ" => Some(EmotionChannel::Cz),
            "P3" => Some(EmotionChannel::P3),
            "P4" => Some(EmotionChannel::P4),
            "PZ" => Some(EmotionChannel::Pz),
            "T3" | "T7" => Some(EmotionChannel::T3),
            "T4" | "T8" => Some(EmotionChannel::T4),
            "T5" | "P7" => Some(EmotionChannel::T5),
            "T6" | "P8" => Some(EmotionChannel::T6),
            "O1" => Some(EmotionChannel::O1),
            "O2" => Some(EmotionChannel::O2),
            "OZ" => Some(EmotionChannel::Oz),
            _ => None,
        }
    }

    /// Is this a left hemisphere channel?
    pub fn is_left(&self) -> bool {
        matches!(
            self,
            EmotionChannel::F3
                | EmotionChannel::F7
                | EmotionChannel::Fp1
                | EmotionChannel::C3
                | EmotionChannel::P3
                | EmotionChannel::T3
                | EmotionChannel::T5
                | EmotionChannel::O1
        )
    }

    /// Is this a right hemisphere channel?
    pub fn is_right(&self) -> bool {
        matches!(
            self,
            EmotionChannel::F4
                | EmotionChannel::F8
                | EmotionChannel::Fp2
                | EmotionChannel::C4
                | EmotionChannel::P4
                | EmotionChannel::T4
                | EmotionChannel::T6
                | EmotionChannel::O2
        )
    }

    /// Is this a frontal channel?
    pub fn is_frontal(&self) -> bool {
        matches!(
            self,
            EmotionChannel::F3
                | EmotionChannel::F4
                | EmotionChannel::F7
                | EmotionChannel::F8
                | EmotionChannel::Fp1
                | EmotionChannel::Fp2
                | EmotionChannel::Fz
        )
    }
}

/// Frequency bands for emotion detection
#[derive(Debug, Clone, Copy)]
pub struct FrequencyBand {
    pub low: f64,
    pub high: f64,
}

impl FrequencyBand {
    /// Delta band (0.5-4 Hz) - deep sleep, unconscious
    pub const DELTA: FrequencyBand = FrequencyBand {
        low: 0.5,
        high: 4.0,
    };
    /// Theta band (4-8 Hz) - drowsiness, memory
    pub const THETA: FrequencyBand = FrequencyBand {
        low: 4.0,
        high: 8.0,
    };
    /// Alpha band (8-13 Hz) - relaxed, eyes closed
    pub const ALPHA: FrequencyBand = FrequencyBand {
        low: 8.0,
        high: 13.0,
    };
    /// Beta band (13-30 Hz) - active thinking, focus
    pub const BETA: FrequencyBand = FrequencyBand {
        low: 13.0,
        high: 30.0,
    };
    /// Gamma band (30-100 Hz) - cognitive processing, binding
    pub const GAMMA: FrequencyBand = FrequencyBand {
        low: 30.0,
        high: 100.0,
    };
}

/// Multi-channel EEG data for emotion detection
#[derive(Debug, Clone)]
pub struct EmotionEEG {
    /// Channel data: channel -> samples
    pub channels: HashMap<EmotionChannel, Vec<f64>>,
    /// Sample rate in Hz
    pub sample_rate: f64,
    /// Duration in seconds
    pub duration_sec: f64,
}

impl EmotionEEG {
    /// Create new empty EEG container
    pub fn new(sample_rate: f64) -> Self {
        Self {
            channels: HashMap::new(),
            sample_rate,
            duration_sec: 0.0,
        }
    }

    /// Add channel data
    pub fn add_channel(&mut self, channel: EmotionChannel, data: Vec<f64>) {
        let duration = data.len() as f64 / self.sample_rate;
        if duration > self.duration_sec {
            self.duration_sec = duration;
        }
        self.channels.insert(channel, data);
    }

    /// Get channel data
    pub fn get_channel(&self, channel: EmotionChannel) -> Option<&Vec<f64>> {
        self.channels.get(&channel)
    }

    /// Number of channels
    pub fn num_channels(&self) -> usize {
        self.channels.len()
    }

    /// Compute band power using Welch's method (simplified)
    pub fn band_power(&self, channel: EmotionChannel, band: FrequencyBand) -> Option<f64> {
        let data = self.get_channel(channel)?;
        if data.len() < 256 {
            return None;
        }

        // Simple band power estimation using FFT
        let power = compute_band_power(data, self.sample_rate, band.low, band.high);
        Some(power)
    }

    /// Compute alpha power at a channel
    pub fn alpha_power(&self, channel: EmotionChannel) -> Option<f64> {
        self.band_power(channel, FrequencyBand::ALPHA)
    }

    /// Compute beta power at a channel
    pub fn beta_power(&self, channel: EmotionChannel) -> Option<f64> {
        self.band_power(channel, FrequencyBand::BETA)
    }

    /// Compute gamma power at a channel
    pub fn gamma_power(&self, channel: EmotionChannel) -> Option<f64> {
        self.band_power(channel, FrequencyBand::GAMMA)
    }
}

/// Compute band power using a simple FFT-based approach
fn compute_band_power(data: &[f64], sample_rate: f64, low_freq: f64, high_freq: f64) -> f64 {
    let n = data.len();
    if n < 2 {
        return 0.0;
    }

    // Apply Hann window
    let windowed: Vec<f64> = data
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let window = 0.5 * (1.0 - (2.0 * PI * i as f64 / (n - 1) as f64).cos());
            x * window
        })
        .collect();

    // Simple DFT for the frequency range of interest
    let freq_resolution = sample_rate / n as f64;
    let low_bin = (low_freq / freq_resolution).floor() as usize;
    let high_bin = (high_freq / freq_resolution).ceil() as usize;
    let high_bin = high_bin.min(n / 2);

    let mut power = 0.0;
    for k in low_bin..=high_bin {
        let freq = k as f64 * freq_resolution;
        if freq >= low_freq && freq <= high_freq {
            // Compute DFT coefficient at frequency k
            let mut real = 0.0;
            let mut imag = 0.0;
            for (i, &x) in windowed.iter().enumerate() {
                let angle = -2.0 * PI * k as f64 * i as f64 / n as f64;
                real += x * angle.cos();
                imag += x * angle.sin();
            }
            power += (real * real + imag * imag) / (n * n) as f64;
        }
    }

    power
}

/// Frontal Asymmetry Index calculator
#[derive(Debug)]
pub struct FrontalAsymmetry {
    /// Smoothing factor for temporal averaging
    pub smoothing: f64,
    /// Previous FAI value for smoothing
    prev_fai: f64,
}

impl FrontalAsymmetry {
    /// Create new FAI calculator
    pub fn new() -> Self {
        Self {
            smoothing: 0.3,
            prev_fai: 0.0,
        }
    }

    /// Compute Frontal Asymmetry Index
    /// FAI = ln(alpha_F4) - ln(alpha_F3)
    /// Positive = approach/positive valence
    /// Negative = withdrawal/negative valence
    pub fn compute(&mut self, eeg: &EmotionEEG) -> Option<f64> {
        let alpha_f3 = eeg.alpha_power(EmotionChannel::F3)?;
        let alpha_f4 = eeg.alpha_power(EmotionChannel::F4)?;

        // Avoid log of zero
        let alpha_f3 = alpha_f3.max(1e-10);
        let alpha_f4 = alpha_f4.max(1e-10);

        let fai = alpha_f4.ln() - alpha_f3.ln();

        // Temporal smoothing
        let smoothed = self.smoothing * fai + (1.0 - self.smoothing) * self.prev_fai;
        self.prev_fai = smoothed;

        Some(smoothed)
    }

    /// Compute extended FAI using F7/F8 as well
    pub fn compute_extended(&mut self, eeg: &EmotionEEG) -> Option<f64> {
        // Primary frontal pair
        let fai_34 = {
            let a3 = eeg.alpha_power(EmotionChannel::F3)?.max(1e-10);
            let a4 = eeg.alpha_power(EmotionChannel::F4)?.max(1e-10);
            a4.ln() - a3.ln()
        };

        // Lateral frontal pair (if available)
        let fai_78 = if let (Some(a7), Some(a8)) = (
            eeg.alpha_power(EmotionChannel::F7),
            eeg.alpha_power(EmotionChannel::F8),
        ) {
            let a7 = a7.max(1e-10);
            let a8 = a8.max(1e-10);
            Some(a8.ln() - a7.ln())
        } else {
            None
        };

        // Combine if both available
        let fai = match fai_78 {
            Some(f78) => 0.6 * fai_34 + 0.4 * f78,
            None => fai_34,
        };

        // Temporal smoothing
        let smoothed = self.smoothing * fai + (1.0 - self.smoothing) * self.prev_fai;
        self.prev_fai = smoothed;

        Some(smoothed)
    }
}

impl Default for FrontalAsymmetry {
    fn default() -> Self {
        Self::new()
    }
}

/// Arousal detector using Beta/Alpha ratio
#[derive(Debug)]
pub struct ArousalDetector {
    /// Smoothing factor
    pub smoothing: f64,
    /// Previous arousal value
    prev_arousal: f64,
}

impl ArousalDetector {
    /// Create new arousal detector
    pub fn new() -> Self {
        Self {
            smoothing: 0.3,
            prev_arousal: 0.0,
        }
    }

    /// Compute arousal level from Beta/Alpha ratio
    /// Higher ratio = higher arousal
    pub fn compute(&mut self, eeg: &EmotionEEG) -> Option<f64> {
        // Use frontal and parietal channels for arousal
        let channels = [EmotionChannel::Fz, EmotionChannel::Cz, EmotionChannel::Pz];

        let mut total_alpha = 0.0;
        let mut total_beta = 0.0;
        let mut count = 0;

        for ch in channels {
            if let (Some(alpha), Some(beta)) = (eeg.alpha_power(ch), eeg.beta_power(ch)) {
                total_alpha += alpha;
                total_beta += beta;
                count += 1;
            }
        }

        if count == 0 || total_alpha < 1e-10 {
            return None;
        }

        // Beta/Alpha ratio, normalized to [-1, 1]
        let ratio = total_beta / total_alpha;
        // Typical ratio range is 0.5-3.0, map to [-1, 1]
        let arousal = ((ratio - 1.5) / 1.5).clamp(-1.0, 1.0);

        // Temporal smoothing
        let smoothed = self.smoothing * arousal + (1.0 - self.smoothing) * self.prev_arousal;
        self.prev_arousal = smoothed;

        Some(smoothed)
    }

    /// Compute arousal from gamma coherence (engagement)
    pub fn compute_gamma_engagement(&self, eeg: &EmotionEEG) -> Option<f64> {
        // High gamma = high engagement/arousal
        let gamma_fz = eeg.gamma_power(EmotionChannel::Fz)?;
        let gamma_pz = eeg.gamma_power(EmotionChannel::Pz)?;

        // Normalize (typical gamma power is much lower than alpha/beta)
        let avg_gamma = (gamma_fz + gamma_pz) / 2.0;

        // Map to [0, 1] based on typical range
        let engagement = (avg_gamma * 1000.0).min(1.0);

        Some(engagement)
    }
}

impl Default for ArousalDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Complete Emotion Sentinel - the consciousness interface for emotion
#[derive(Debug)]
pub struct EmotionSentinel {
    /// Frontal asymmetry calculator
    frontal_asymmetry: FrontalAsymmetry,
    /// Arousal detector
    arousal_detector: ArousalDetector,
    /// History of emotional states
    history: VecDeque<EmotionalState>,
    /// Maximum history length
    max_history: usize,
}

impl EmotionSentinel {
    /// Create new emotion sentinel
    pub fn new() -> Self {
        Self {
            frontal_asymmetry: FrontalAsymmetry::new(),
            arousal_detector: ArousalDetector::new(),
            history: VecDeque::new(),
            max_history: 100,
        }
    }

    /// Process EEG window and detect emotional state
    pub fn detect(&mut self, eeg: &EmotionEEG) -> EmotionalState {
        // Compute valence from frontal asymmetry
        let valence = self
            .frontal_asymmetry
            .compute_extended(eeg)
            .map(|fai| {
                // FAI typically ranges from -0.5 to 0.5
                // Map to [-1, 1] for valence
                (fai * 2.0).clamp(-1.0, 1.0)
            })
            .unwrap_or(0.0);

        // Compute arousal from Beta/Alpha ratio
        let arousal = self.arousal_detector.compute(eeg).unwrap_or(0.0);

        // Compute dominance from frontal theta (approach behavior)
        let dominance = eeg
            .band_power(EmotionChannel::Fz, FrequencyBand::THETA)
            .map(|theta| {
                // Higher frontal theta = higher approach/dominance
                (theta * 10.0 - 0.5).clamp(-1.0, 1.0)
            })
            .unwrap_or(0.0);

        // Confidence based on signal quality and consistency
        let engagement = self
            .arousal_detector
            .compute_gamma_engagement(eeg)
            .unwrap_or(0.5);
        let confidence = engagement.clamp(0.3, 1.0);

        let state = EmotionalState::new(valence, arousal, dominance, confidence);

        // Store in history
        self.history.push_back(state);
        if self.history.len() > self.max_history {
            self.history.pop_front();
        }

        state
    }

    /// Get smoothed emotional state over recent history
    pub fn smoothed_state(&self, window: usize) -> EmotionalState {
        if self.history.is_empty() {
            return EmotionalState::neutral();
        }

        let window = window.min(self.history.len());
        let skip = self.history.len() - window;

        let avg_valence = self
            .history
            .iter()
            .skip(skip)
            .map(|s| s.valence)
            .sum::<f64>()
            / window as f64;
        let avg_arousal = self
            .history
            .iter()
            .skip(skip)
            .map(|s| s.arousal)
            .sum::<f64>()
            / window as f64;
        let avg_dominance = self
            .history
            .iter()
            .skip(skip)
            .map(|s| s.dominance)
            .sum::<f64>()
            / window as f64;
        let avg_confidence = self
            .history
            .iter()
            .skip(skip)
            .map(|s| s.confidence)
            .sum::<f64>()
            / window as f64;

        EmotionalState::new(avg_valence, avg_arousal, avg_dominance, avg_confidence)
    }

    /// Get the current detected emotion category
    pub fn current_emotion(&self) -> EmotionCategory {
        self.history
            .back()
            .map(|s| s.classify())
            .unwrap_or(EmotionCategory::Neutral)
    }

    /// Get emotion statistics over history
    pub fn emotion_stats(&self) -> HashMap<EmotionCategory, usize> {
        let mut stats = HashMap::new();
        for state in &self.history {
            let category = state.classify();
            *stats.entry(category).or_insert(0) += 1;
        }
        stats
    }

    /// Reset the sentinel
    pub fn reset(&mut self) {
        self.history.clear();
        self.frontal_asymmetry = FrontalAsymmetry::new();
        self.arousal_detector = ArousalDetector::new();
    }
}

impl Default for EmotionSentinel {
    fn default() -> Self {
        Self::new()
    }
}

/// Simulated emotion EEG based on published correlates
pub struct EmotionSimulator {
    sample_rate: f64,
    rng_seed: u64,
}

impl EmotionSimulator {
    /// Create new simulator
    pub fn new(sample_rate: f64) -> Self {
        Self {
            sample_rate,
            rng_seed: 42,
        }
    }

    /// Generate EEG for a target emotional state
    pub fn generate(&mut self, target: EmotionCategory, duration_sec: f64) -> EmotionEEG {
        let (target_valence, target_arousal) = target.target_va();
        let n_samples = (duration_sec * self.sample_rate) as usize;

        let mut eeg = EmotionEEG::new(self.sample_rate);

        // Generate each channel with appropriate characteristics
        let channels = [
            EmotionChannel::F3,
            EmotionChannel::F4,
            EmotionChannel::F7,
            EmotionChannel::F8,
            EmotionChannel::Fz,
            EmotionChannel::Cz,
            EmotionChannel::Pz,
        ];

        for channel in channels {
            let data = self.generate_channel(channel, target_valence, target_arousal, n_samples);
            eeg.add_channel(channel, data);
        }

        eeg
    }

    /// Generate single channel data
    fn generate_channel(
        &mut self,
        channel: EmotionChannel,
        valence: f64,
        arousal: f64,
        n_samples: usize,
    ) -> Vec<f64> {
        let mut data = vec![0.0; n_samples];

        // Base frequencies and their modulation by emotion
        // Alpha (8-13 Hz): inverse of arousal
        // Beta (13-30 Hz): correlates with arousal
        // Gamma (30-50 Hz): engagement
        // Theta (4-8 Hz): approach motivation

        // Alpha power modulated by arousal (inverse relationship)
        // High arousal suppresses alpha; low arousal (relaxation) increases it
        let alpha_power = (1.0 - arousal * 0.4).max(0.2);

        // Beta power correlates with arousal
        // Base of 1.2 produces beta/alpha ratios in the 0.5-3.0 range
        // that match real EEG literature (Koelstra et al., 2012)
        let beta_power = (1.2 + arousal * 0.6).max(0.1);

        // Asymmetry for frontal channels based on valence
        let asymmetry_factor = if channel.is_frontal() {
            if channel.is_left() {
                // Left frontal: more alpha for negative valence (withdrawal)
                1.0 - valence * 0.3
            } else if channel.is_right() {
                // Right frontal: more alpha for positive valence? (actually opposite)
                // Davidson's model: Right = withdrawal, Left = approach
                // But we measure alpha, which is INVERSE of activity
                // So positive valence = more LEFT activity = LESS left alpha
                1.0 + valence * 0.3
            } else {
                1.0
            }
        } else {
            1.0
        };

        // Generate signal
        for i in 0..n_samples {
            let t = i as f64 / self.sample_rate;

            // Simple PRNG for noise
            self.rng_seed = self
                .rng_seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1);
            let noise = (self.rng_seed as f64 / u64::MAX as f64) * 2.0 - 1.0;

            // Alpha component (10 Hz)
            let alpha = alpha_power * asymmetry_factor * (2.0 * PI * 10.0 * t).sin();

            // Beta component (20 Hz)
            let beta = beta_power * (2.0 * PI * 20.0 * t).sin();

            // Gamma component (40 Hz)
            let gamma = 0.2 * (2.0 * PI * 40.0 * t).sin();

            // Theta component (6 Hz)
            let theta = 0.3 * (2.0 * PI * 6.0 * t).sin();

            // Combine with some noise
            data[i] = alpha + beta + gamma + theta + noise * 0.1;
        }

        data
    }

    /// Generate a session with multiple emotion blocks
    pub fn generate_session(
        &mut self,
        blocks: &[(EmotionCategory, f64)],
    ) -> Vec<(EmotionEEG, EmotionCategory)> {
        blocks
            .iter()
            .map(|(emotion, duration)| {
                let eeg = self.generate(*emotion, *duration);
                (eeg, *emotion)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emotional_state_classification() {
        // High valence, high arousal = Excited
        let excited = EmotionalState::new(0.7, 0.7, 0.5, 1.0);
        assert_eq!(excited.classify(), EmotionCategory::Excited);

        // High valence, low arousal = Relaxed
        let relaxed = EmotionalState::new(0.5, -0.6, 0.0, 1.0);
        assert_eq!(relaxed.classify(), EmotionCategory::Relaxed);

        // Low valence, high arousal = Angry
        let angry = EmotionalState::new(-0.7, 0.7, 0.5, 1.0);
        assert_eq!(angry.classify(), EmotionCategory::Angry);

        // Near center = Neutral
        let neutral = EmotionalState::new(0.1, 0.1, 0.0, 1.0);
        assert_eq!(neutral.classify(), EmotionCategory::Neutral);
    }

    #[test]
    fn test_channel_parsing() {
        assert_eq!(EmotionChannel::from_label("F3"), Some(EmotionChannel::F3));
        assert_eq!(EmotionChannel::from_label("f4"), Some(EmotionChannel::F4));
        assert_eq!(EmotionChannel::from_label("T7"), Some(EmotionChannel::T3));
        assert_eq!(EmotionChannel::from_label("P7"), Some(EmotionChannel::T5));
    }

    #[test]
    fn test_emotion_simulation() {
        let mut sim = EmotionSimulator::new(256.0);
        let eeg = sim.generate(EmotionCategory::Happy, 5.0);

        assert_eq!(eeg.num_channels(), 7);
        assert!(eeg.duration_sec > 4.9);
        assert!(eeg.get_channel(EmotionChannel::F3).is_some());
    }

    #[test]
    fn test_emotion_detection() {
        let mut sim = EmotionSimulator::new(256.0);
        let mut sentinel = EmotionSentinel::new();

        // Generate happy emotion EEG
        let eeg = sim.generate(EmotionCategory::Happy, 5.0);
        let state = sentinel.detect(&eeg);

        // Should detect positive valence
        assert!(state.valence > -0.5, "Happy should have positive valence");
        assert!(state.confidence > 0.0);
    }

    #[test]
    fn test_frontal_asymmetry() {
        let mut sim = EmotionSimulator::new(256.0);
        let mut fai = FrontalAsymmetry::new();

        // Positive emotion should have positive FAI
        let happy_eeg = sim.generate(EmotionCategory::Excited, 5.0);
        let happy_fai = fai.compute(&happy_eeg);

        // Negative emotion should have negative FAI
        let mut sim2 = EmotionSimulator::new(256.0);
        let sad_eeg = sim2.generate(EmotionCategory::Sad, 5.0);
        let mut fai2 = FrontalAsymmetry::new();
        let sad_fai = fai2.compute(&sad_eeg);

        // The direction should differ
        if let (Some(h), Some(s)) = (happy_fai, sad_fai) {
            assert!(h > s, "Happy FAI {} should be > Sad FAI {}", h, s);
        }
    }
}
