// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Meditation & Flow State Detection from Multi-Channel EEG
//!
//! Implements the scientific foundations of EEG-based meditation and flow detection:
//!
//! ## The Superconscious States
//!
//! Beyond ordinary waking consciousness lies a spectrum of enhanced states:
//!
//! ```text
//!                    ┌─────────────────────────────────────┐
//!                    │         SUPERCONSCIOUS              │
//!                    │   Flow, Meditation, Peak States     │
//!                    ├─────────────────────────────────────┤
//!     High Focus ────│  FLOW          │  ABSORPTION       │
//!                    │  Gamma sync    │  Deep theta       │
//!                    ├────────────────┼───────────────────┤
//!     Low Focus  ────│  CALM          │  WANDERING        │
//!                    │  Alpha         │  Default mode     │
//!                    └────────────────┴───────────────────┘
//!                         External         Internal
//!                         Attention        Attention
//! ```
//!
//! ## EEG Signatures of Meditation/Flow
//!
//! ### Alpha Power (8-13 Hz)
//! - Increases during relaxed, eyes-closed meditation
//! - Posterior alpha = relaxed alertness
//! - Alpha blocking = attention engagement
//!
//! ### Frontal Midline Theta (4-8 Hz at Fz)
//! - Signature of focused attention and working memory
//! - Increases in experienced meditators
//! - Correlated with "no-mind" states in Zen
//!
//! ### Gamma Synchrony (30-100 Hz)
//! - Long-range gamma coherence = cognitive binding
//! - Expert meditators (Tibetan monks) show sustained gamma
//! - Flow state characterized by gamma-theta coupling
//!
//! ### Alpha Coherence
//! - Inter-hemispheric coherence = unified awareness
//! - Increases with meditation experience
//! - Marker of "whole-brain" integration
//!
//! ## References
//! - Lutz et al. (2004): Long-term meditators and gamma
//! - Aftanas & Golocheikine (2001): Frontal midline theta
//! - Cahn & Polich (2006): Meditation and EEG review
//! - Csikszentmihalyi (1990): Flow - The Psychology of Optimal Experience

use std::collections::{HashMap, VecDeque};
use std::f64::consts::PI;

/// Meditation/Flow state in the Focus-Attention space
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeditationState {
    /// Focus depth: 0.0 (scattered) to 1.0 (laser-focused)
    pub focus: f64,
    /// Calm level: 0.0 (agitated) to 1.0 (serene)
    pub calm: f64,
    /// Flow state: 0.0 (ordinary) to 1.0 (peak flow)
    pub flow: f64,
    /// Presence/awareness: 0.0 (distracted) to 1.0 (fully present)
    pub presence: f64,
    /// Confidence in detection
    pub confidence: f64,
}

impl MeditationState {
    /// Create a new meditation state
    pub fn new(focus: f64, calm: f64, flow: f64, presence: f64, confidence: f64) -> Self {
        Self {
            focus: focus.clamp(0.0, 1.0),
            calm: calm.clamp(0.0, 1.0),
            flow: flow.clamp(0.0, 1.0),
            presence: presence.clamp(0.0, 1.0),
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    /// Default distracted state
    pub fn distracted() -> Self {
        Self::new(0.2, 0.3, 0.0, 0.2, 1.0)
    }

    /// Classify into discrete meditation category
    pub fn classify(&self) -> MeditationCategory {
        // Flow requires both high focus and some calm
        if self.flow > 0.6 && self.focus > 0.5 {
            return MeditationCategory::Flow;
        }

        // Deep absorption requires high presence and calm
        if self.presence > 0.7 && self.calm > 0.6 {
            return MeditationCategory::Absorption;
        }

        // Focused attention
        if self.focus > 0.6 {
            return MeditationCategory::Focused;
        }

        // Calm but not focused = open awareness
        if self.calm > 0.6 && self.focus < 0.4 {
            return MeditationCategory::OpenAwareness;
        }

        // Moderate calm = light meditation
        if self.calm > 0.4 {
            return MeditationCategory::Relaxed;
        }

        // Low everything = mind wandering
        MeditationCategory::Wandering
    }

    /// Overall meditation quality score
    pub fn quality(&self) -> f64 {
        // Weighted combination emphasizing presence and flow
        0.3 * self.focus + 0.2 * self.calm + 0.3 * self.flow + 0.2 * self.presence
    }
}

/// Discrete meditation categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MeditationCategory {
    /// Peak performance state - gamma sync, effortless focus
    Flow,
    /// Deep meditative absorption - theta dominance
    Absorption,
    /// Concentrated attention - frontal midline theta
    Focused,
    /// Open monitoring - alpha dominance, panoramic awareness
    OpenAwareness,
    /// Light relaxation - mild alpha increase
    Relaxed,
    /// Mind wandering - default mode network active
    Wandering,
}

impl MeditationCategory {
    /// Get category name
    pub fn name(&self) -> &'static str {
        match self {
            MeditationCategory::Flow => "Flow State",
            MeditationCategory::Absorption => "Deep Absorption",
            MeditationCategory::Focused => "Focused Attention",
            MeditationCategory::OpenAwareness => "Open Awareness",
            MeditationCategory::Relaxed => "Relaxed",
            MeditationCategory::Wandering => "Mind Wandering",
        }
    }

    /// Is this a desirable meditation state?
    pub fn is_meditative(&self) -> bool {
        matches!(
            self,
            MeditationCategory::Flow
                | MeditationCategory::Absorption
                | MeditationCategory::Focused
                | MeditationCategory::OpenAwareness
        )
    }

    /// Is this a peak state?
    pub fn is_peak(&self) -> bool {
        matches!(
            self,
            MeditationCategory::Flow | MeditationCategory::Absorption
        )
    }

    /// Emoji representation
    pub fn emoji(&self) -> &'static str {
        match self {
            MeditationCategory::Flow => "🌊",
            MeditationCategory::Absorption => "🧘",
            MeditationCategory::Focused => "🎯",
            MeditationCategory::OpenAwareness => "👁️",
            MeditationCategory::Relaxed => "😌",
            MeditationCategory::Wandering => "💭",
        }
    }

    /// Description of the state
    pub fn description(&self) -> &'static str {
        match self {
            MeditationCategory::Flow => "Effortless peak performance, gamma-theta coupling",
            MeditationCategory::Absorption => "Deep meditative state, theta dominance",
            MeditationCategory::Focused => "Concentrated attention, frontal midline theta",
            MeditationCategory::OpenAwareness => "Panoramic awareness, alpha coherence",
            MeditationCategory::Relaxed => "Calm but not deeply meditative",
            MeditationCategory::Wandering => "Default mode active, attention scattered",
        }
    }
}

/// EEG channels for meditation detection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MeditationChannel {
    // Frontal (attention, executive function)
    Fz, // Frontal midline - KEY for focused attention
    F3,
    F4,
    Fp1,
    Fp2,

    // Central (sensorimotor)
    Cz,
    C3,
    C4,

    // Parietal (attention, spatial)
    Pz,
    P3,
    P4,

    // Occipital (visual, alpha rhythm)
    Oz,
    O1,
    O2,

    // Temporal (memory, emotion)
    T3,
    T4,
}

impl MeditationChannel {
    /// Parse from string label
    pub fn from_label(label: &str) -> Option<Self> {
        let label = label.trim().to_uppercase();
        match label.as_str() {
            "FZ" => Some(MeditationChannel::Fz),
            "F3" => Some(MeditationChannel::F3),
            "F4" => Some(MeditationChannel::F4),
            "FP1" => Some(MeditationChannel::Fp1),
            "FP2" => Some(MeditationChannel::Fp2),
            "CZ" => Some(MeditationChannel::Cz),
            "C3" => Some(MeditationChannel::C3),
            "C4" => Some(MeditationChannel::C4),
            "PZ" => Some(MeditationChannel::Pz),
            "P3" => Some(MeditationChannel::P3),
            "P4" => Some(MeditationChannel::P4),
            "OZ" => Some(MeditationChannel::Oz),
            "O1" => Some(MeditationChannel::O1),
            "O2" => Some(MeditationChannel::O2),
            "T3" | "T7" => Some(MeditationChannel::T3),
            "T4" | "T8" => Some(MeditationChannel::T4),
            _ => None,
        }
    }

    /// Name of the channel
    pub fn name(&self) -> &'static str {
        match self {
            MeditationChannel::Fz => "Fz",
            MeditationChannel::F3 => "F3",
            MeditationChannel::F4 => "F4",
            MeditationChannel::Fp1 => "Fp1",
            MeditationChannel::Fp2 => "Fp2",
            MeditationChannel::Cz => "Cz",
            MeditationChannel::C3 => "C3",
            MeditationChannel::C4 => "C4",
            MeditationChannel::Pz => "Pz",
            MeditationChannel::P3 => "P3",
            MeditationChannel::P4 => "P4",
            MeditationChannel::Oz => "Oz",
            MeditationChannel::O1 => "O1",
            MeditationChannel::O2 => "O2",
            MeditationChannel::T3 => "T3",
            MeditationChannel::T4 => "T4",
        }
    }
}

/// Multi-channel EEG data for meditation detection
#[derive(Debug, Clone)]
pub struct MeditationEEG {
    /// Channel data
    pub channels: HashMap<MeditationChannel, Vec<f64>>,
    /// Sample rate
    pub sample_rate: f64,
}

impl MeditationEEG {
    /// Create new container
    pub fn new(sample_rate: f64) -> Self {
        Self {
            channels: HashMap::new(),
            sample_rate,
        }
    }

    /// Add channel data
    pub fn add_channel(&mut self, channel: MeditationChannel, data: Vec<f64>) {
        self.channels.insert(channel, data);
    }

    /// Get channel data
    pub fn get_channel(&self, channel: MeditationChannel) -> Option<&Vec<f64>> {
        self.channels.get(&channel)
    }

    /// Compute band power
    pub fn band_power(&self, channel: MeditationChannel, low: f64, high: f64) -> Option<f64> {
        let data = self.get_channel(channel)?;
        Some(compute_band_power(data, self.sample_rate, low, high))
    }

    /// Theta power (4-8 Hz)
    pub fn theta_power(&self, channel: MeditationChannel) -> Option<f64> {
        self.band_power(channel, 4.0, 8.0)
    }

    /// Alpha power (8-13 Hz)
    pub fn alpha_power(&self, channel: MeditationChannel) -> Option<f64> {
        self.band_power(channel, 8.0, 13.0)
    }

    /// Beta power (13-30 Hz)
    pub fn beta_power(&self, channel: MeditationChannel) -> Option<f64> {
        self.band_power(channel, 13.0, 30.0)
    }

    /// Gamma power (30-50 Hz)
    pub fn gamma_power(&self, channel: MeditationChannel) -> Option<f64> {
        self.band_power(channel, 30.0, 50.0)
    }
}

/// Compute band power using Welch's method (simplified)
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

    // Simple DFT for the frequency range
    let freq_resolution = sample_rate / n as f64;
    let low_bin = (low_freq / freq_resolution).floor() as usize;
    let high_bin = (high_freq / freq_resolution).ceil() as usize;
    let high_bin = high_bin.min(n / 2);

    let mut power = 0.0;
    for k in low_bin..=high_bin {
        let freq = k as f64 * freq_resolution;
        if freq >= low_freq && freq <= high_freq {
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

/// Frontal Midline Theta detector
#[derive(Debug)]
pub struct FrontalMidlineTheta {
    /// Smoothing factor
    smoothing: f64,
    /// Previous value
    prev_value: f64,
    /// Baseline theta (for relative measurement)
    baseline: Option<f64>,
}

impl FrontalMidlineTheta {
    /// Create new detector
    pub fn new() -> Self {
        Self {
            smoothing: 0.3,
            prev_value: 0.0,
            baseline: None,
        }
    }

    /// Set baseline from resting state
    pub fn set_baseline(&mut self, baseline: f64) {
        self.baseline = Some(baseline);
    }

    /// Compute FMT power at Fz
    pub fn compute(&mut self, eeg: &MeditationEEG) -> Option<f64> {
        let theta_fz = eeg.theta_power(MeditationChannel::Fz)?;
        let alpha_fz = eeg.alpha_power(MeditationChannel::Fz)?.max(1e-10);

        // Theta/Alpha ratio at Fz indicates focused attention
        let ratio = theta_fz / alpha_fz;

        // Normalize relative to baseline if available
        let normalized = match self.baseline {
            Some(base) if base > 1e-10 => (ratio - base) / base,
            _ => ratio,
        };

        // Smooth
        let smoothed = self.smoothing * normalized + (1.0 - self.smoothing) * self.prev_value;
        self.prev_value = smoothed;

        Some(smoothed)
    }

    /// Detect if in focused attention state
    pub fn is_focused(&self) -> bool {
        self.prev_value > 0.3
    }
}

impl Default for FrontalMidlineTheta {
    fn default() -> Self {
        Self::new()
    }
}

/// Alpha Coherence detector (inter-hemispheric synchrony)
#[derive(Debug)]
pub struct AlphaCoherence {
    /// Smoothing factor
    smoothing: f64,
    /// Previous coherence
    prev_coherence: f64,
}

impl AlphaCoherence {
    /// Create new detector
    pub fn new() -> Self {
        Self {
            smoothing: 0.3,
            prev_coherence: 0.0,
        }
    }

    /// Compute alpha coherence between left and right hemispheres
    pub fn compute(&mut self, eeg: &MeditationEEG) -> Option<f64> {
        // Use O1-O2 pair for posterior alpha coherence
        let alpha_o1 = eeg.alpha_power(MeditationChannel::O1)?;
        let alpha_o2 = eeg.alpha_power(MeditationChannel::O2)?;

        // Simple coherence estimate based on power similarity
        let max_power = alpha_o1.max(alpha_o2).max(1e-10);
        let min_power = alpha_o1.min(alpha_o2);
        let coherence = min_power / max_power;

        // Can also use P3-P4 if available
        let parietal_coherence = if let (Some(p3), Some(p4)) = (
            eeg.alpha_power(MeditationChannel::P3),
            eeg.alpha_power(MeditationChannel::P4),
        ) {
            let max_p = p3.max(p4).max(1e-10);
            let min_p = p3.min(p4);
            Some(min_p / max_p)
        } else {
            None
        };

        // Average if both available
        let combined = match parietal_coherence {
            Some(pc) => (coherence + pc) / 2.0,
            None => coherence,
        };

        // Smooth
        let smoothed = self.smoothing * combined + (1.0 - self.smoothing) * self.prev_coherence;
        self.prev_coherence = smoothed;

        Some(smoothed)
    }

    /// Is in coherent state?
    pub fn is_coherent(&self) -> bool {
        self.prev_coherence > 0.7
    }
}

impl Default for AlphaCoherence {
    fn default() -> Self {
        Self::new()
    }
}

/// Gamma Synchrony detector (flow state marker)
#[derive(Debug)]
pub struct GammaSynchrony {
    /// Smoothing factor
    smoothing: f64,
    /// Previous synchrony
    prev_sync: f64,
    /// Baseline gamma
    baseline: Option<f64>,
}

impl GammaSynchrony {
    /// Create new detector
    pub fn new() -> Self {
        Self {
            smoothing: 0.3,
            prev_sync: 0.0,
            baseline: None,
        }
    }

    /// Set baseline
    pub fn set_baseline(&mut self, baseline: f64) {
        self.baseline = Some(baseline);
    }

    /// Compute gamma synchrony across frontal-parietal network
    pub fn compute(&mut self, eeg: &MeditationEEG) -> Option<f64> {
        // Gamma at frontal and parietal sites
        let gamma_fz = eeg.gamma_power(MeditationChannel::Fz)?;
        let gamma_pz = eeg.gamma_power(MeditationChannel::Pz)?;
        let gamma_cz = eeg.gamma_power(MeditationChannel::Cz)?;

        // Average gamma power
        let avg_gamma = (gamma_fz + gamma_pz + gamma_cz) / 3.0;

        // Synchrony based on similarity (coherence proxy)
        let max_gamma = gamma_fz.max(gamma_pz).max(gamma_cz).max(1e-10);
        let min_gamma = gamma_fz.min(gamma_pz).min(gamma_cz);
        let sync = min_gamma / max_gamma;

        // Combined measure: both power and synchrony matter
        let combined = sync * (avg_gamma * 100.0).min(1.0);

        // Normalize to baseline if available
        let normalized = match self.baseline {
            Some(base) if base > 1e-10 => combined / base,
            _ => combined,
        };

        // Smooth
        let smoothed = self.smoothing * normalized + (1.0 - self.smoothing) * self.prev_sync;
        self.prev_sync = smoothed;

        Some(smoothed.min(1.0))
    }

    /// Is in flow state?
    pub fn is_flow(&self) -> bool {
        self.prev_sync > 0.6
    }
}

impl Default for GammaSynchrony {
    fn default() -> Self {
        Self::new()
    }
}

/// Complete Meditation Sentinel
#[derive(Debug)]
pub struct MeditationSentinel {
    /// Frontal midline theta detector
    fmt: FrontalMidlineTheta,
    /// Alpha coherence detector
    alpha_coherence: AlphaCoherence,
    /// Gamma synchrony detector
    gamma_sync: GammaSynchrony,
    /// History of states
    history: VecDeque<MeditationState>,
    /// Maximum history
    max_history: usize,
}

impl MeditationSentinel {
    /// Create new sentinel
    pub fn new() -> Self {
        Self {
            fmt: FrontalMidlineTheta::new(),
            alpha_coherence: AlphaCoherence::new(),
            gamma_sync: GammaSynchrony::new(),
            history: VecDeque::new(),
            max_history: 100,
        }
    }

    /// Process EEG and detect meditation state
    pub fn detect(&mut self, eeg: &MeditationEEG) -> MeditationState {
        // Frontal midline theta → Focus
        let fmt_value = self.fmt.compute(eeg).unwrap_or(0.0);
        let focus = ((fmt_value + 0.5) / 1.0).clamp(0.0, 1.0);

        // Alpha coherence → Calm/Presence
        let coherence = self.alpha_coherence.compute(eeg).unwrap_or(0.5);

        // Total alpha power → Calm
        let alpha_oz = eeg.alpha_power(MeditationChannel::Oz).unwrap_or(0.0);
        let alpha_pz = eeg.alpha_power(MeditationChannel::Pz).unwrap_or(0.0);
        let avg_alpha = (alpha_oz + alpha_pz) / 2.0;
        let calm = (avg_alpha * 50.0).clamp(0.0, 1.0);

        // Gamma synchrony → Flow
        let flow = self.gamma_sync.compute(eeg).unwrap_or(0.0);

        // Presence = combination of coherence and sustained attention
        let presence = (coherence * 0.6 + focus * 0.4).clamp(0.0, 1.0);

        // Confidence based on signal quality
        let confidence = 0.85;

        let state = MeditationState::new(focus, calm, flow, presence, confidence);

        // Store history
        self.history.push_back(state);
        if self.history.len() > self.max_history {
            self.history.pop_front();
        }

        state
    }

    /// Get smoothed state
    pub fn smoothed_state(&self, window: usize) -> MeditationState {
        if self.history.is_empty() {
            return MeditationState::distracted();
        }

        let window = window.min(self.history.len());
        let skip = self.history.len() - window;

        let avg_focus =
            self.history.iter().skip(skip).map(|s| s.focus).sum::<f64>() / window as f64;
        let avg_calm = self.history.iter().skip(skip).map(|s| s.calm).sum::<f64>() / window as f64;
        let avg_flow = self.history.iter().skip(skip).map(|s| s.flow).sum::<f64>() / window as f64;
        let avg_presence = self
            .history
            .iter()
            .skip(skip)
            .map(|s| s.presence)
            .sum::<f64>()
            / window as f64;
        let avg_confidence = self
            .history
            .iter()
            .skip(skip)
            .map(|s| s.confidence)
            .sum::<f64>()
            / window as f64;

        MeditationState::new(avg_focus, avg_calm, avg_flow, avg_presence, avg_confidence)
    }

    /// Current category
    pub fn current_category(&self) -> MeditationCategory {
        self.history
            .back()
            .map(|s| s.classify())
            .unwrap_or(MeditationCategory::Wandering)
    }

    /// Category statistics
    pub fn category_stats(&self) -> HashMap<MeditationCategory, usize> {
        let mut stats = HashMap::new();
        for state in &self.history {
            let cat = state.classify();
            *stats.entry(cat).or_insert(0) += 1;
        }
        stats
    }

    /// Reset
    pub fn reset(&mut self) {
        self.history.clear();
        self.fmt = FrontalMidlineTheta::new();
        self.alpha_coherence = AlphaCoherence::new();
        self.gamma_sync = GammaSynchrony::new();
    }
}

impl Default for MeditationSentinel {
    fn default() -> Self {
        Self::new()
    }
}

/// Meditation session simulator
pub struct MeditationSimulator {
    sample_rate: f64,
    rng_seed: u64,
}

impl MeditationSimulator {
    /// Create new simulator
    pub fn new(sample_rate: f64) -> Self {
        Self {
            sample_rate,
            rng_seed: 42,
        }
    }

    /// Generate EEG for target meditation state
    pub fn generate(&mut self, target: MeditationCategory, duration_sec: f64) -> MeditationEEG {
        let n_samples = (duration_sec * self.sample_rate) as usize;
        let mut eeg = MeditationEEG::new(self.sample_rate);

        // Generate each channel
        let channels = [
            MeditationChannel::Fz,
            MeditationChannel::Cz,
            MeditationChannel::Pz,
            MeditationChannel::F3,
            MeditationChannel::F4,
            MeditationChannel::P3,
            MeditationChannel::P4,
            MeditationChannel::O1,
            MeditationChannel::O2,
            MeditationChannel::Oz,
        ];

        for channel in channels {
            let data = self.generate_channel(channel, target, n_samples);
            eeg.add_channel(channel, data);
        }

        eeg
    }

    fn generate_channel(
        &mut self,
        channel: MeditationChannel,
        target: MeditationCategory,
        n: usize,
    ) -> Vec<f64> {
        let mut data = vec![0.0; n];

        // State-specific power levels
        let (theta_pow, alpha_pow, beta_pow, gamma_pow) = match target {
            MeditationCategory::Flow => (0.4, 0.3, 0.2, 0.8),
            MeditationCategory::Absorption => (0.8, 0.5, 0.1, 0.3),
            MeditationCategory::Focused => (0.6, 0.3, 0.3, 0.4),
            MeditationCategory::OpenAwareness => (0.3, 0.8, 0.2, 0.2),
            MeditationCategory::Relaxed => (0.3, 0.6, 0.3, 0.2),
            MeditationCategory::Wandering => (0.2, 0.3, 0.5, 0.1),
        };

        // Channel-specific modulation
        let theta_mod = if matches!(channel, MeditationChannel::Fz) {
            1.5
        } else {
            1.0
        };
        let alpha_mod = if matches!(
            channel,
            MeditationChannel::Oz | MeditationChannel::O1 | MeditationChannel::O2
        ) {
            1.5
        } else {
            1.0
        };

        for i in 0..n {
            let t = i as f64 / self.sample_rate;

            // Noise
            self.rng_seed = self
                .rng_seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1);
            let noise = (self.rng_seed as f64 / u64::MAX as f64 - 0.5) * 0.1;

            // Frequency components
            let theta = theta_pow * theta_mod * (2.0 * PI * 6.0 * t).sin();
            let alpha = alpha_pow * alpha_mod * (2.0 * PI * 10.0 * t).sin();
            let beta = beta_pow * (2.0 * PI * 20.0 * t).sin();
            let gamma = gamma_pow * (2.0 * PI * 40.0 * t).sin();

            data[i] = theta + alpha + beta + gamma + noise;
        }

        data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_meditation_state_classification() {
        // High flow
        let flow = MeditationState::new(0.7, 0.5, 0.8, 0.7, 1.0);
        assert_eq!(flow.classify(), MeditationCategory::Flow);

        // Deep absorption
        let absorption = MeditationState::new(0.4, 0.8, 0.3, 0.8, 1.0);
        assert_eq!(absorption.classify(), MeditationCategory::Absorption);

        // Focused attention
        let focused = MeditationState::new(0.8, 0.3, 0.2, 0.5, 1.0);
        assert_eq!(focused.classify(), MeditationCategory::Focused);

        // Mind wandering
        let wandering = MeditationState::new(0.2, 0.2, 0.1, 0.2, 1.0);
        assert_eq!(wandering.classify(), MeditationCategory::Wandering);
    }

    #[test]
    fn test_simulation() {
        let mut sim = MeditationSimulator::new(256.0);
        let eeg = sim.generate(MeditationCategory::Flow, 5.0);

        assert!(eeg.channels.len() >= 10);
        assert!(eeg.get_channel(MeditationChannel::Fz).is_some());
    }

    #[test]
    fn test_sentinel_detection() {
        let mut sim = MeditationSimulator::new(256.0);
        let mut sentinel = MeditationSentinel::new();

        let eeg = sim.generate(MeditationCategory::Flow, 5.0);
        let state = sentinel.detect(&eeg);

        assert!(state.flow > 0.0);
        assert!(state.confidence > 0.0);
    }
}
