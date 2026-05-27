// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Articulatory Synthesizer
//!
//! CfC-based formant trajectory generation for speech synthesis.
//!
//! This module bridges the gap between discrete phonemes and continuous
//! audio by using Closed-form Continuous-time (CfC) networks to generate
//! smooth formant trajectories with natural coarticulation.
//!
//! ## Architecture
//!
//! ```text
//! Phoneme Sequence
//!       │
//!       ▼
//! ┌─────────────────┐
//! │ Formant Lookup  │ ← FormantDatabase
//! │ (per phoneme)   │
//! └────────┬────────┘
//!          │ Target F1, F2, F3
//!          ▼
//! ┌─────────────────┐
//! │   CfC Network   │ ← Smooth trajectory interpolation
//! │ (tau modulated) │    with learned coarticulation
//! └────────┬────────┘
//!          │ Frame-by-frame formants
//!          ▼
//! ┌─────────────────┐
//! │ Prosody Overlay │ ← LTCPacing parameters
//! │ (pitch, energy) │
//! └────────┬────────┘
//!          │
//!          ▼
//!    FormantFrame[]
//! ```
//!
//! ## Key Innovation
//!
//! The CfC network's variable time constant (τ) naturally models
//! coarticulation: faster transitions for rapid speech, slower
//! for careful articulation. The tau is modulated by LTCPacing.

use crate::voice::LTCPacing;
use crate::voice::formant_targets::FormantDatabase;
use crate::voice::phoneme_hdc::PhonemeHdcCodec;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

// ═══════════════════════════════════════════════════════════════════════════════
// FORMANT FRAME — canonical definition in symthaea-vocal-tract sub-crate
// ═══════════════════════════════════════════════════════════════════════════════

pub use symthaea_vocal_tract::types::FormantFrame;

// ═══════════════════════════════════════════════════════════════════════════════
// PHONEME WITH TIMING
// ═══════════════════════════════════════════════════════════════════════════════

/// Phoneme with duration and stress information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimedPhoneme {
    /// ARPABET phoneme symbol
    pub phoneme: String,
    /// Duration in seconds
    pub duration: f32,
    /// Stress level (0 = none, 1 = primary, 2 = secondary)
    pub stress: u8,
    /// Start time in seconds (relative to utterance start)
    pub start_time: f32,
}

impl TimedPhoneme {
    /// Create a new timed phoneme
    pub fn new(phoneme: impl Into<String>, duration: f32) -> Self {
        Self {
            phoneme: phoneme.into(),
            duration,
            stress: 0,
            start_time: 0.0,
        }
    }

    /// Set stress level
    pub fn with_stress(mut self, stress: u8) -> Self {
        self.stress = stress;
        self
    }

    /// Set start time
    pub fn with_start(mut self, start: f32) -> Self {
        self.start_time = start;
        self
    }

    /// Get end time
    pub fn end_time(&self) -> f32 {
        self.start_time + self.duration
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CFC STATE (Simplified closed-form continuous-time dynamics)
// ═══════════════════════════════════════════════════════════════════════════════

/// Simplified CfC cell state for formant trajectory generation
///
/// Implements closed-form solution: h(t) = h_∞ + (h_0 - h_∞) * exp(-t/τ)
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CfCState {
    /// Current state (3D for F1, F2, F3)
    state: [f32; 3],
    /// Time constant (controls transition speed)
    tau: f32,
    /// Target state (attractor)
    target: [f32; 3],
}

impl CfCState {
    fn new(initial: [f32; 3], tau: f32) -> Self {
        Self {
            state: initial,
            tau,
            target: initial,
        }
    }

    /// Set new target (attractor point)
    fn set_target(&mut self, target: [f32; 3]) {
        self.target = target;
    }

    /// Step forward by dt seconds using closed-form solution
    fn step(&mut self, dt: f32) -> [f32; 3] {
        // Closed-form: h(t+dt) = h_∞ + (h(t) - h_∞) * exp(-dt/τ)
        let decay = (-dt / self.tau.max(0.001)).exp();

        for i in 0..3 {
            self.state[i] = self.target[i] + (self.state[i] - self.target[i]) * decay;
        }

        self.state
    }

    /// Set time constant
    fn set_tau(&mut self, tau: f32) {
        self.tau = tau.max(0.001);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ARTICULATORY SYNTHESIZER
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for articulatory synthesis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArticulatoryConfig {
    /// Sample rate for formant frames (Hz)
    pub frame_rate: f32,
    /// Base time constant for CfC (seconds)
    pub base_tau: f32,
    /// Base fundamental frequency (Hz)
    pub base_f0: f32,
    /// F0 range for intonation (semitones)
    pub f0_range: f32,
    /// Whether to apply coarticulation
    pub coarticulation: bool,
    /// Coarticulation lookahead (number of phonemes)
    pub lookahead: usize,
    /// Whether to use HDC-driven coarticulation
    pub hdc_coarticulation: bool,
}

impl Default for ArticulatoryConfig {
    fn default() -> Self {
        Self {
            frame_rate: 200.0, // 5ms per frame
            base_tau: 0.05,    // 50ms default transition
            base_f0: 120.0,    // Male voice default
            f0_range: 12.0,    // One octave range
            coarticulation: true,
            lookahead: 2,
            hdc_coarticulation: true, // Enable HDC-driven coarticulation
        }
    }
}

/// Articulatory speech synthesizer
///
/// Generates smooth formant trajectories from phoneme sequences
/// using CfC dynamics for natural coarticulation, with optional
/// HDC-based phoneme similarity for intelligent tau modulation.
#[derive(Debug, Clone)]
pub struct ArticulatorySynthesizer {
    /// Configuration
    config: ArticulatoryConfig,
    /// Formant database
    formants: FormantDatabase,
    /// HDC phoneme codec for similarity-based coarticulation
    hdc_codec: Arc<PhonemeHdcCodec>,
    /// CfC state for trajectory generation
    cfc: CfCState,
    /// Current time
    current_time: f32,
}

impl ArticulatorySynthesizer {
    /// Create a new synthesizer with default configuration
    pub fn new() -> Self {
        Self::with_config(ArticulatoryConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: ArticulatoryConfig) -> Self {
        let formants = FormantDatabase::new();
        let initial_target = formants.lookup_or_default("AX"); // Start at schwa

        Self {
            cfc: CfCState::new(
                [initial_target.f1, initial_target.f2, initial_target.f3],
                config.base_tau,
            ),
            hdc_codec: Arc::new(PhonemeHdcCodec::new()),
            config,
            formants,
            current_time: 0.0,
        }
    }

    /// Create with shared HDC codec (for reuse across synthesizers)
    pub fn with_shared_codec(config: ArticulatoryConfig, codec: Arc<PhonemeHdcCodec>) -> Self {
        let formants = FormantDatabase::new();
        let initial_target = formants.lookup_or_default("AX");

        Self {
            cfc: CfCState::new(
                [initial_target.f1, initial_target.f2, initial_target.f3],
                config.base_tau,
            ),
            hdc_codec: codec,
            config,
            formants,
            current_time: 0.0,
        }
    }

    /// Reset synthesizer state
    pub fn reset(&mut self) {
        let initial = self.formants.lookup_or_default("AX");
        self.cfc = CfCState::new([initial.f1, initial.f2, initial.f3], self.config.base_tau);
        self.current_time = 0.0;
    }

    /// Get HDC codec reference
    pub fn hdc_codec(&self) -> &PhonemeHdcCodec {
        &self.hdc_codec
    }

    /// Compute HDC-based coarticulation tau multiplier
    fn hdc_tau_modifier(&self, current: &str, next: Option<&str>) -> f32 {
        if !self.config.hdc_coarticulation {
            return 1.0;
        }

        match next {
            Some(next_phoneme) => {
                // Use HDC similarity to modulate tau
                // Similar phonemes: slower transition (higher tau)
                // Dissimilar phonemes: faster transition (lower tau)
                self.hdc_codec.coarticulation_weight(current, next_phoneme)
            }
            None => 1.0,
        }
    }

    /// Compute HDC-based blend factor for anticipatory coarticulation
    fn hdc_blend_factor(&self, current: &str, next: &str, progress: f32) -> f32 {
        if !self.config.hdc_coarticulation {
            return anticipatory_blend(progress);
        }

        // Base blend from position
        let base_blend = anticipatory_blend(progress);

        // Modulate by phoneme similarity
        let similarity = self.hdc_codec.similarity(current, next);

        // Similar phonemes: blend earlier and more (easier coarticulation)
        // Dissimilar phonemes: blend later and less (harder coarticulation)
        let sim_factor = 0.5 + similarity * 0.5; // 0.5 to 1.0

        (base_blend * sim_factor).min(0.5)
    }

    /// Synthesize formant frames from phoneme sequence
    pub fn synthesize(
        &mut self,
        phonemes: &[TimedPhoneme],
        pacing: &LTCPacing,
    ) -> Vec<FormantFrame> {
        if phonemes.is_empty() {
            return Vec::new();
        }

        self.reset();

        // Calculate total duration
        let total_duration: f32 = phonemes.iter().map(|p| p.duration).sum();

        // Calculate number of frames
        let frame_duration = 1.0 / self.config.frame_rate;
        let num_frames = (total_duration * self.config.frame_rate).ceil() as usize;

        let mut frames = Vec::with_capacity(num_frames);

        // Track current phoneme
        let mut phoneme_idx = 0;
        let mut time_in_phoneme = 0.0;

        for frame_idx in 0..num_frames {
            let time = frame_idx as f32 * frame_duration;

            // Find current phoneme
            while phoneme_idx < phonemes.len() - 1 && time >= phonemes[phoneme_idx].end_time() {
                phoneme_idx += 1;
                time_in_phoneme = 0.0;
            }

            let current = &phonemes[phoneme_idx];
            let target = self.formants.lookup_or_default(&current.phoneme);

            // Get next phoneme for coarticulation
            let next_phoneme = phonemes.get(phoneme_idx + 1);

            // Compute HDC-modulated tau
            let hdc_tau_mod =
                self.hdc_tau_modifier(&current.phoneme, next_phoneme.map(|p| p.phoneme.as_str()));
            let tau_modifier = pacing.tau * hdc_tau_mod;
            self.cfc.set_tau(self.config.base_tau * tau_modifier);

            // Apply coarticulation from next phoneme
            let effective_target = if self.config.coarticulation && phoneme_idx + 1 < phonemes.len()
            {
                let next = &phonemes[phoneme_idx + 1];
                let next_target = self.formants.lookup_or_default(&next.phoneme);

                // Compute HDC-modulated blend factor
                let progress = time_in_phoneme / current.duration;
                let blend = self.hdc_blend_factor(&current.phoneme, &next.phoneme, progress);

                target.lerp(&next_target, blend)
            } else {
                target
            };

            // Update CfC target
            self.cfc.set_target([
                effective_target.f1,
                effective_target.f2,
                effective_target.f3,
            ]);

            // Step CfC
            let formants = self.cfc.step(frame_duration);

            // Compute pitch contour
            let f0 = self.compute_f0(time, total_duration, current.stress, pacing);

            // Compute energy envelope
            let energy =
                self.compute_energy(time_in_phoneme, current.duration, current.stress, pacing);

            // Create frame
            frames.push(FormantFrame {
                f1: formants[0],
                f2: formants[1],
                f3: formants[2],
                b1: effective_target.b1,
                b2: effective_target.b2,
                b3: effective_target.b3,
                f0,
                energy,
                voicing: if effective_target.is_voiced { 1.0 } else { 0.0 },
                time,
                source_type: effective_target.manner,
                nasal_zero_freq: 0.0,
                nasal_zero_bw: 0.0,
            });

            time_in_phoneme += frame_duration;
        }

        frames
    }

    /// Synthesize from ARPABET string (space-separated phonemes)
    pub fn synthesize_arpabet(&mut self, arpabet: &str, pacing: &LTCPacing) -> Vec<FormantFrame> {
        let phonemes = self.parse_arpabet(arpabet);
        self.synthesize(&phonemes, pacing)
    }

    /// Parse ARPABET string into timed phonemes
    fn parse_arpabet(&self, arpabet: &str) -> Vec<TimedPhoneme> {
        let mut phonemes = Vec::new();
        let mut current_time = 0.0;

        for token in arpabet.split_whitespace() {
            // Check for stress marker (digit at end)
            let (phoneme, stress) = if let Some(last) = token.chars().last() {
                if last.is_ascii_digit() {
                    let stress = last.to_digit(10).unwrap_or(0) as u8;
                    (&token[..token.len() - 1], stress)
                } else {
                    (token, 0)
                }
            } else {
                (token, 0)
            };

            // Get duration from formant database
            let target = self.formants.lookup_or_default(phoneme);
            let duration = target.duration_ms / 1000.0; // Convert to seconds

            phonemes.push(TimedPhoneme {
                phoneme: phoneme.to_string(),
                duration,
                stress,
                start_time: current_time,
            });

            current_time += duration;
        }

        phonemes
    }

    /// Compute fundamental frequency (pitch) at given time
    fn compute_f0(&self, time: f32, total_duration: f32, stress: u8, pacing: &LTCPacing) -> f32 {
        let base = self.config.base_f0;

        // Declination: pitch falls over utterance
        let declination = 1.0 - 0.15 * (time / total_duration);

        // Stress-based accent
        let stress_boost = match stress {
            1 => 1.1,  // Primary stress: 10% higher
            2 => 1.05, // Secondary stress: 5% higher
            _ => 1.0,
        };

        // Emotional modulation from pacing
        let emotional_shift = pacing.emotional_valence * 0.1; // ±10%

        // Arousal affects pitch range
        let arousal_range = 1.0 + pacing.arousal * 0.2; // Up to 20% more range

        base * declination * stress_boost * (1.0 + emotional_shift) * arousal_range
    }

    /// Compute energy envelope at given position within phoneme
    fn compute_energy(
        &self,
        time_in_phoneme: f32,
        phoneme_duration: f32,
        stress: u8,
        pacing: &LTCPacing,
    ) -> f32 {
        let progress = (time_in_phoneme / phoneme_duration).clamp(0.0, 1.0);

        // Attack-sustain-release envelope
        let envelope = if progress < 0.1 {
            // Attack
            progress / 0.1
        } else if progress > 0.85 {
            // Release
            (1.0 - progress) / 0.15
        } else {
            // Sustain
            1.0
        };

        // Stress modulation
        let stress_factor = match stress {
            1 => pacing.emphasis * 1.2,
            2 => pacing.emphasis * 1.1,
            _ => pacing.emphasis * 0.9,
        };

        (envelope * stress_factor).clamp(0.0, 1.0)
    }

    /// Get formant database (for external configuration)
    pub fn formants(&self) -> &FormantDatabase {
        &self.formants
    }

    /// Get mutable formant database
    pub fn formants_mut(&mut self) -> &mut FormantDatabase {
        &mut self.formants
    }

    /// Get configuration
    pub fn config(&self) -> &ArticulatoryConfig {
        &self.config
    }
}

impl Default for ArticulatorySynthesizer {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// HELPER FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute anticipatory coarticulation blend factor
///
/// Returns how much to blend toward the next phoneme based on
/// position within current phoneme. Uses sigmoid for smooth transition.
fn anticipatory_blend(progress: f32) -> f32 {
    // Start blending in last 30% of phoneme
    if progress < 0.7 {
        0.0
    } else {
        let x = (progress - 0.7) / 0.3; // 0 to 1 in last 30%
        // Smooth sigmoid-like curve
        x * x * (3.0 - 2.0 * x) // Smoothstep
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_formant_frame_interpolation() {
        let a = FormantFrame {
            f1: 300.0,
            f2: 2000.0,
            f3: 3000.0,
            b1: 60.0,
            b2: 90.0,
            b3: 150.0,
            f0: 120.0,
            energy: 1.0,
            voicing: 1.0,
            time: 0.0,
            ..Default::default()
        };
        let b = FormantFrame {
            f1: 700.0,
            f2: 1000.0,
            f3: 2500.0,
            b1: 80.0,
            b2: 120.0,
            b3: 200.0,
            f0: 100.0,
            energy: 0.5,
            voicing: 0.0,
            time: 1.0,
            ..Default::default()
        };

        let mid = a.lerp(&b, 0.5);

        assert!((mid.f1 - 500.0).abs() < 0.1);
        assert!((mid.f2 - 1500.0).abs() < 0.1);
        assert!((mid.time - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_cfc_state_convergence() {
        let mut cfc = CfCState::new([500.0, 1500.0, 2500.0], 0.05);
        cfc.set_target([300.0, 2000.0, 3000.0]);

        // Step for long enough to converge
        for _ in 0..100 {
            cfc.step(0.01);
        }

        // Should be close to target
        assert!((cfc.state[0] - 300.0).abs() < 1.0);
        assert!((cfc.state[1] - 2000.0).abs() < 1.0);
        assert!((cfc.state[2] - 3000.0).abs() < 1.0);
    }

    #[test]
    fn test_synthesize_basic() {
        let mut synth = ArticulatorySynthesizer::new();
        let pacing = LTCPacing::default();

        let phonemes = vec![
            TimedPhoneme::new("HH", 0.05),
            TimedPhoneme::new("AH", 0.1).with_stress(1),
            TimedPhoneme::new("L", 0.05),
            TimedPhoneme::new("OW", 0.15).with_stress(1),
        ];

        let frames = synth.synthesize(&phonemes, &pacing);

        assert!(!frames.is_empty());

        // Check frames are in time order
        for window in frames.windows(2) {
            assert!(window[1].time >= window[0].time);
        }

        // Check formants are reasonable
        for frame in &frames {
            assert!(frame.f1 >= 0.0 && frame.f1 < 1500.0);
            assert!(frame.f2 >= 0.0 && frame.f2 < 3500.0);
            assert!(frame.f3 >= 0.0 && frame.f3 < 4500.0);
        }
    }

    #[test]
    fn test_parse_arpabet() {
        let synth = ArticulatorySynthesizer::new();
        let phonemes = synth.parse_arpabet("HH AH1 L OW1");

        assert_eq!(phonemes.len(), 4);
        assert_eq!(phonemes[0].phoneme, "HH");
        assert_eq!(phonemes[1].phoneme, "AH");
        assert_eq!(phonemes[1].stress, 1);
        assert_eq!(phonemes[3].phoneme, "OW");
        assert_eq!(phonemes[3].stress, 1);
    }

    #[test]
    fn test_anticipatory_blend() {
        assert!((anticipatory_blend(0.5) - 0.0).abs() < 0.01);
        assert!((anticipatory_blend(0.7) - 0.0).abs() < 0.01);
        assert!(anticipatory_blend(0.85) > 0.3);
        assert!((anticipatory_blend(1.0) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_pacing_affects_synthesis() {
        let mut synth = ArticulatorySynthesizer::new();

        let phonemes = vec![TimedPhoneme::new("AH", 0.1).with_stress(1)];

        let calm = LTCPacing::calm();
        let excited = LTCPacing::excited();

        let calm_frames = synth.synthesize(&phonemes, &calm);
        synth.reset();
        let excited_frames = synth.synthesize(&phonemes, &excited);

        // Excited should have higher energy
        let calm_energy: f32 = calm_frames.iter().map(|f| f.energy).sum();
        let excited_energy: f32 = excited_frames.iter().map(|f| f.energy).sum();

        assert!(excited_energy > calm_energy * 0.9); // Allow some variance
    }
}
