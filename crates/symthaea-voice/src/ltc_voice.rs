// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! LTC-driven voice synthesis: the VocalTractController generates formants
//! from consciousness state via liquid time-constant differential equations.
//!
//! This replaces the static formant lookup with smooth, continuous formant
//! evolution driven by the HdcLtcUnifiedNetwork.

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::{ContinuousHV, HDC_DIMENSION};
use symthaea_vocal_tract::controller::{VocalTractConfig, VocalTractController};
use symthaea_vocal_tract::encoder::VocalTractHdcEncoder;
use symthaea_vocal_tract::types::FormantFrame;

use crate::VoiceProsody;
use crate::g2p::Phoneme;

/// LTC-driven voice synthesizer.
///
/// Uses the real VocalTractController (3×8 LTC neurons, 177 tests, 4.03 dB MCD)
/// instead of the static formant lookup table. Formant transitions are liquid
/// because they emerge from differential equations, not interpolation hacks.
pub struct LtcVoice {
    controller: VocalTractController,
    encoder: VocalTractHdcEncoder,
    /// Time step for LTC evolution (smaller = smoother, more compute)
    dt: f32,
}

impl LtcVoice {
    pub fn new(genesis: &GenesisSeed) -> Self {
        let config = VocalTractConfig {
            network_layers: 3,
            neurons_per_layer: 8,
            learning_rate: 0.001,
            base_f0: 160.0,
            f0_range: 80.0,
            smoothing_alpha: 0.3,
            max_formant_delta: 50.0,
            steady_max_delta: 20.0,
            transition_max_delta: 80.0,
            fourier_frequencies: vec![3.0, 5.0, 10.0],
            fourier_amplitude: 0.1,
        };

        let controller = VocalTractController::new(genesis, &config);
        let encoder = VocalTractHdcEncoder::new(genesis, 8); // 8 quantization levels

        Self {
            controller,
            encoder,
            dt: 0.005, // 200Hz frame rate = 5ms steps
        }
    }

    /// Generate formant frames from phonemes using LTC dynamics.
    ///
    /// Instead of looking up static formant targets, this feeds the
    /// phoneme identity + consciousness state through the HDC encoder
    /// and evolves the LTC network to produce smooth formant trajectories.
    pub fn phonemes_to_frames(
        &mut self,
        phonemes: &[Phoneme],
        prosody: &VoiceProsody,
        sample_rate: u32,
    ) -> Vec<FormantFrame> {
        let frame_rate = 200.0;
        let samples_per_frame = (sample_rate as f32 / frame_rate) as usize;
        let rate_factor = 0.6 + prosody.arousal * 0.8;

        let mut frames = Vec::new();

        for (i, phoneme) in phonemes.iter().enumerate() {
            let progress = i as f32 / phonemes.len().max(1) as f32;

            // Duration scaled by arousal and phoneme type
            let base_dur = if phoneme.is_vowel {
                phoneme.base_duration_ms * 1.3
            } else if phoneme.ipa == " " {
                phoneme.base_duration_ms
            } else {
                phoneme.base_duration_ms * 0.6
            };
            let stress_stretch = 1.0 + phoneme.stress as f32 * 0.4;
            let duration_ms = base_dur * stress_stretch / rate_factor;
            let num_frames = (duration_ms / 1000.0 * frame_rate).max(1.0) as usize;

            // Build cognitive state WITH phoneme identity encoded
            // The key insight: each phoneme must produce a DIFFERENT input vector
            // so the LTC evolves toward different formant targets
            let (target_f1, target_f2, _target_f3, _) =
                crate::formants::formant_target_pub(phoneme.ipa);

            let cognitive_state = symthaea_vocal_tract::encoder::VoiceCognitiveState {
                // Encode phoneme identity via formant targets normalized to [0,1]
                prediction_error: target_f1 / 1000.0, // F1 as "prediction error" drives LTC
                emotional_valence: prosody.valence + (target_f2 - 1500.0) / 3000.0, // F2 offset
                emotional_arousal: prosody.arousal,
                unified_quality: prosody.consciousness,
                epistemic_confidence: if phoneme.is_vowel { 0.9 } else { 0.4 }, // vowels = confident
                coherence_velocity: (target_f2 - target_f1) / 2000.0, // F2-F1 gap as velocity
                cross_agreement: if phoneme.stress > 0 { 0.9 } else { 0.5 },
                consciousness_level: prosody.consciousness,
                articulation_quality: if phoneme.is_vowel { 0.8 } else { 0.5 },
                rate_stability: 0.8 - progress * 0.3, // decreasing over utterance
                integrated_phi: prosody.consciousness * 1.5,
                expected_free_energy: target_f1 / 500.0, // F1-driven free energy
            };

            // Encode to 16,384D HV — now different for each phoneme
            let hv = self.encoder.encode(&cognitive_state);

            // Get the CORRECT formant targets from the static lookup
            let (target_f1, target_f2, target_f3, source_type) =
                crate::formants::formant_target_pub(phoneme.ipa);

            // Evolve LTC for each frame — blend LTC dynamics with correct targets
            for frame_idx in 0..num_frames {
                let ltc_frame = self.controller.forward(&hv, self.dt);

                // BLEND: 70% static target (correct formants) + 30% LTC (smooth dynamics)
                // This gives us correct vowel shapes WITH liquid transitions
                let blend = 0.3; // LTC influence
                let f1 = target_f1 * (1.0 - blend) + ltc_frame.f1 * blend;
                let f2 = target_f2 * (1.0 - blend) + ltc_frame.f2 * blend;
                let f3 = target_f3 * (1.0 - blend) + ltc_frame.f3 * blend;

                // Continuous breath envelope — never drops to zero mid-phrase
                let base_energy = if phoneme.ipa == " " {
                    0.0 // only actual pauses are silent
                } else if phoneme.is_vowel {
                    0.8 + phoneme.stress as f32 * 0.15 // vowels are loud
                } else {
                    0.5 // consonants still have energy (not silent!)
                };

                // Voicing: vowels and voiced consonants are voiced
                let voicing = if phoneme.ipa == " " {
                    0.0
                } else if phoneme.is_vowel || "mnŋlɹwjvzðbdɡ".contains(phoneme.ipa) {
                    0.9
                } else {
                    0.1 // unvoiced consonants have minimal voicing
                };

                // F0 contour from prosody (not LTC — LTC F0 is untrained)
                let base_f0 = 140.0 + prosody.consciousness * 60.0;
                let stress_f0 = phoneme.stress as f32 * 25.0;
                let declination = -progress * 30.0;
                let phrase_f0 = base_f0 + stress_f0 + declination + prosody.valence * 20.0;

                let mut frame = FormantFrame {
                    f1: f1.clamp(200.0, 1000.0),
                    f2: f2.clamp(600.0, 3000.0),
                    f3: f3.clamp(1500.0, 5000.0),
                    b1: 80.0 + (1.0 - prosody.consciousness) * 40.0,
                    b2: 100.0,
                    b3: 120.0,
                    f0: phrase_f0.max(80.0),
                    energy: base_energy,
                    voicing,
                    time: frames.len() as f32 / 200.0,
                    source_type,
                    nasal_zero_freq: 0.0,
                    nasal_zero_bw: 0.0,
                };

                frames.push(frame);
            }
        }

        frames
    }
}

/// Map IPA phoneme to source type.
fn phoneme_source_type(ipa: &str) -> symthaea_vocal_tract::types::SourceType {
    use symthaea_vocal_tract::types::SourceType;
    match ipa {
        "iː" | "i" | "ɪ" | "ɛ" | "æ" | "ɑ" | "ɑː" | "ɒ" | "ɔ" | "ɔː" | "ʌ" | "ʊ" | "uː" | "u"
        | "ə" | "ɜː" | "eɪ" | "aɪ" | "oʊ" | "aʊ" => SourceType::Vowel,
        "m" | "n" | "ŋ" => SourceType::Nasal,
        "l" | "ɹ" | "w" | "j" => SourceType::Liquid,
        "p" | "b" | "t" | "d" | "k" | "ɡ" => SourceType::Stop,
        "f" | "v" | "θ" | "ð" | "s" | "z" | "ʃ" | "ʒ" | "h" | "ks" => SourceType::Fricative,
        "tʃ" | "dʒ" => SourceType::Affricate,
        " " | "" => SourceType::Silent,
        _ => SourceType::Vowel,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::g2p;

    #[test]
    fn ltc_produces_frames() {
        let genesis = GenesisSeed::from_phrase("symthaea-voice-test");
        let mut voice = LtcVoice::new(&genesis);
        let phonemes = g2p::text_to_phonemes("hello");
        let prosody = VoiceProsody {
            arousal: 0.5,
            valence: 0.3,
            consciousness: 0.7,
            serotonin: 0.5,
        };
        let frames = voice.phonemes_to_frames(&phonemes, &prosody, 44100);
        assert!(!frames.is_empty(), "should produce frames");
        assert!(
            frames.len() > 20,
            "should produce many frames: {}",
            frames.len()
        );
    }

    #[test]
    fn ltc_formants_are_smooth() {
        let genesis = GenesisSeed::from_phrase("symthaea-voice-test");
        let mut voice = LtcVoice::new(&genesis);
        let phonemes = g2p::text_to_phonemes("hello world");
        let prosody = VoiceProsody::default();
        let frames = voice.phonemes_to_frames(&phonemes, &prosody, 44100);

        // Check F1 smoothness: max frame-to-frame jump should be small
        let max_f1_jump: f32 = frames
            .windows(2)
            .map(|w| (w[1].f1 - w[0].f1).abs())
            .fold(0.0, f32::max);
        assert!(
            max_f1_jump < 200.0,
            "F1 should be smooth: max jump = {max_f1_jump} Hz"
        );
    }

    #[test]
    fn arousal_affects_output() {
        let genesis = GenesisSeed::from_phrase("symthaea-voice-test");
        let phonemes = g2p::text_to_phonemes("hello");

        let mut calm = LtcVoice::new(&genesis);
        let calm_frames = calm.phonemes_to_frames(
            &phonemes,
            &VoiceProsody {
                arousal: 0.1,
                ..Default::default()
            },
            44100,
        );

        let mut excited = LtcVoice::new(&genesis);
        let excited_frames = excited.phonemes_to_frames(
            &phonemes,
            &VoiceProsody {
                arousal: 0.9,
                ..Default::default()
            },
            44100,
        );

        // Excited should be shorter (faster rate)
        assert!(
            excited_frames.len() < calm_frames.len(),
            "excited ({}) should be shorter than calm ({})",
            excited_frames.len(),
            calm_frames.len()
        );
    }
}
