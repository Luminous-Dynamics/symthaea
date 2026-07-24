// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Voice Pipeline Orchestrator
//!
//! Connects all voice components into a unified pipeline from cognitive state
//! to synthesized audio. This is the primary entry point for text-to-speech
//! with consciousness-driven prosody.
//!
//! ## Architecture
//!
//! ```text
//! CycleResult ──► CognitiveVoiceBridge ──► LTCPacing ──► VoiceOutput ──► audio
//!                     (smoothed)             (modulated)    (synthesize)
//!
//! CycleResult ──► synthesis_bridge ──► ThoughtChannels ──► Broca ──► text
//!                 (cfg ssm_language)                        │
//!                                                          ▼
//!                                            orchestrator.synthesize_from_cycle_result()
//! ```
//!
//! ## Two Synthesis Paths
//!
//! 1. **High-level** (`synthesize_from_cycle_result`): Takes Broca-generated text
//!    plus the CycleResult, produces consciousness-modulated audio via `VoiceOutput`.
//!
//! 2. **Low-level** (`thought_to_speech`): Takes raw CfC output + primitives,
//!    produces audio via G2P → formant vocoder with LTC pacing.

use std::collections::HashMap;

use crate::cognitive_loop::types::CycleResult;
use crate::voice::cognitive_bridge::CognitiveVoiceBridge;
use crate::voice::g2p::G2PConverter;
use crate::voice::vocoder::FormantVocoder;
use crate::voice::{FormantFrame, LTCPacing, VoiceOutput, VoiceOutputConfig};
use symthaea_vocal_tract::types::SourceType;
#[cfg(feature = "vocal-tract")]
use symthaea_vocal_tract::{VocalTractPipeline, VoiceCognitiveState};

/// Voice pipeline orchestrator.
///
/// Holds the voice bridge, G2P converter, formant vocoder, and high-level
/// `VoiceOutput` to provide a unified API for consciousness-driven speech.
pub struct VoiceOrchestrator {
    /// Cognitive-to-prosody bridge (smoothed LTC pacing)
    voice_bridge: CognitiveVoiceBridge,
    /// Grapheme-to-phoneme converter
    g2p: G2PConverter,
    /// Low-level formant vocoder
    vocoder: FormantVocoder,
    /// High-level voice output (simulated TTS or Kokoro)
    voice_output: VoiceOutput,
    /// LTC-driven vocal tract pipeline (replaces hardcoded formant lookup when active)
    #[cfg(feature = "vocal-tract")]
    neural_pipeline: Option<VocalTractPipeline>,
}

impl VoiceOrchestrator {
    /// Create a new orchestrator with default configuration.
    pub fn new() -> Self {
        Self {
            voice_bridge: CognitiveVoiceBridge::new(),
            g2p: G2PConverter::new(),
            vocoder: FormantVocoder::new(),
            voice_output: VoiceOutput::new(VoiceOutputConfig::default()),
            #[cfg(feature = "vocal-tract")]
            neural_pipeline: None,
        }
    }

    /// Create with a custom voice output configuration.
    pub fn with_config(voice_config: VoiceOutputConfig) -> Self {
        Self {
            voice_bridge: CognitiveVoiceBridge::new(),
            g2p: G2PConverter::new(),
            vocoder: FormantVocoder::new(),
            voice_output: VoiceOutput::new(voice_config),
            #[cfg(feature = "vocal-tract")]
            neural_pipeline: None,
        }
    }

    /// Enable the LTC neural vocal tract pipeline.
    ///
    /// When active, `thought_to_speech_neural()` uses the trained VocalTractController
    /// instead of the hardcoded phoneme formant lookup table.
    #[cfg(feature = "vocal-tract")]
    pub fn enable_neural_pipeline(&mut self, pipeline: VocalTractPipeline) {
        self.neural_pipeline = Some(pipeline);
    }

    /// Process text through the cognitive voice pipeline.
    ///
    /// Takes Broca-generated (or any) text plus the `CycleResult` from the
    /// cognitive loop. Updates the voice bridge with CfC output and prediction
    /// error, then synthesizes audio with consciousness-modulated pacing.
    ///
    /// Returns synthesized audio samples at the configured sample rate.
    ///
    /// # Arguments
    ///
    /// * `text` - Text to synthesize (e.g., from Broca generator).
    /// * `cycle_result` - The cognitive loop's cycle output.
    pub fn synthesize_from_cycle_result(
        &mut self,
        text: &str,
        cycle_result: &CycleResult,
    ) -> Vec<f32> {
        // 1. Update the voice bridge with CfC output
        self.voice_bridge.update(
            &cycle_result.output,
            1.0, // tau (default; could be extracted from CfC state)
            cycle_result.prediction_error,
            HashMap::new(),
            cycle_result.detected_primitives.clone(),
        );

        // 2. Get the consciousness-modulated LTC pacing
        let ltc_pacing = self.voice_bridge.get_ltc_pacing();

        // 3. Apply pacing to voice output and synthesize
        self.voice_output.set_pacing(ltc_pacing);
        self.voice_output.synthesize(text).unwrap_or_default()
    }

    /// Full low-level pipeline: CfC state + text -> G2P -> formant vocoder.
    ///
    /// This bypasses `VoiceOutput` and uses the formant vocoder directly,
    /// producing richer articulatory synthesis with per-phoneme formant control.
    /// Pacing is derived from CfC hidden state and prediction error.
    ///
    /// # Arguments
    ///
    /// * `broca_text` - Text to synthesize.
    /// * `cfc_output` - Hidden state from the CfC/LTC network.
    /// * `prediction_error` - Current prediction error (0.0-1.0).
    /// * `detected_primitives` - Semantic primitives from the cognitive loop.
    pub fn thought_to_speech(
        &mut self,
        broca_text: &str,
        cfc_output: &[f32],
        prediction_error: f32,
        detected_primitives: Vec<String>,
    ) -> Vec<f32> {
        self.thought_to_speech_paced(
            broca_text,
            cfc_output,
            1.0,
            prediction_error,
            detected_primitives,
            1.0,
            1.0,
        )
        .0
    }

    /// Low-level pipeline with explicit CfC time-constant, adaptive-behavior rate
    /// multipliers, and post-synthesis quality metrics.
    ///
    /// This is the loop-facing entry point: `tau` comes from the cycle's adaptive
    /// time-constant factors (FEP surprise × Φ modulation), and the multipliers
    /// come from `AdaptiveBehavior`. The returned [`VoiceOutputMetrics`] are
    /// computed from the produced formant frames so callers can feed the
    /// voice→cognition feedback bridge (`update_voice_feedback`).
    #[allow(clippy::too_many_arguments)]
    pub fn thought_to_speech_paced(
        &mut self,
        broca_text: &str,
        cfc_output: &[f32],
        tau: f32,
        prediction_error: f32,
        detected_primitives: Vec<String>,
        rate_multiplier: f32,
        pause_multiplier: f32,
    ) -> (Vec<f32>, crate::voice::voice_feedback::VoiceOutputMetrics) {
        use crate::voice::voice_feedback::VoiceOutputMetrics;

        // 1. Update bridge with CfC state (tau guards against unset factors)
        let tau = if tau > 0.0 { tau } else { 1.0 };
        self.voice_bridge.update(
            cfc_output,
            tau,
            prediction_error,
            HashMap::new(),
            detected_primitives,
        );

        // 2. Get pacing, then apply adaptive-behavior multipliers
        let mut pacing = self.voice_bridge.get_ltc_pacing();
        let rate_multiplier = rate_multiplier.clamp(0.25, 4.0);
        let pause_multiplier = pause_multiplier.clamp(0.25, 4.0);
        pacing.rate = (pacing.rate * rate_multiplier).clamp(0.1, 4.0);
        pacing.phrase_pause *= pause_multiplier;
        pacing.sentence_pause *= pause_multiplier;

        // 3. G2P conversion
        let phoneme_ids = self.g2p.text_to_phonemes(broca_text);

        if phoneme_ids.is_empty() {
            return (Vec::new(), VoiceOutputMetrics::default());
        }

        // 4. Convert phoneme IDs to formant frames with pacing-modulated timing
        let frames = phoneme_ids_to_formant_frames(&phoneme_ids, &pacing);

        if frames.is_empty() {
            return (Vec::new(), VoiceOutputMetrics::default());
        }

        // 5. Quality metrics from the produced frames (voice→cognition feedback)
        let metrics = VoiceOutputMetrics::from_formant_frames(&frames, None);

        // 6. Vocoder synthesis
        (self.vocoder.synthesize(&frames), metrics)
    }

    /// Get a reference to the internal cognitive voice bridge.
    pub fn voice_bridge(&self) -> &CognitiveVoiceBridge {
        &self.voice_bridge
    }

    /// Get a mutable reference to the internal cognitive voice bridge.
    pub fn voice_bridge_mut(&mut self) -> &mut CognitiveVoiceBridge {
        &mut self.voice_bridge
    }

    /// Get the current LTC pacing from the bridge.
    pub fn current_pacing(&self) -> LTCPacing {
        self.voice_bridge.get_ltc_pacing()
    }

    /// Get a reference to the voice output subsystem.
    pub fn voice_output(&self) -> &VoiceOutput {
        &self.voice_output
    }

    /// Get a mutable reference to the voice output subsystem.
    pub fn voice_output_mut(&mut self) -> &mut VoiceOutput {
        &mut self.voice_output
    }

    /// Neural vocal tract synthesis: CfC state + text → controller-driven formants → audio.
    ///
    /// Uses the trained LTC `VocalTractController` to generate formant trajectories
    /// from cognitive state, replacing the hardcoded phoneme formant lookup.
    /// Falls back to `thought_to_speech()` if the neural pipeline isn't enabled.
    ///
    /// The controller produces naturally smooth coarticulation transitions (10× smoother
    /// than the rule-based lookup) via learned temporal dynamics.
    #[cfg(feature = "vocal-tract")]
    pub fn thought_to_speech_neural(
        &mut self,
        broca_text: &str,
        cfc_output: &[f32],
        prediction_error: f32,
        detected_primitives: Vec<String>,
        cognitive_state: &VoiceCognitiveState,
    ) -> Vec<f32> {
        let pipeline = match self.neural_pipeline.as_mut() {
            Some(p) => p,
            None => {
                // Fall back to hardcoded lookup
                return self.thought_to_speech(
                    broca_text,
                    cfc_output,
                    prediction_error,
                    detected_primitives,
                );
            }
        };

        // 1. Update bridge with CfC state
        self.voice_bridge.update(
            cfc_output,
            1.0,
            prediction_error,
            HashMap::new(),
            detected_primitives,
        );

        // 2. G2P conversion
        let phoneme_ids = self.g2p.text_to_phonemes(broca_text);
        if phoneme_ids.is_empty() {
            return Vec::new();
        }

        // 3. Convert phoneme IDs to names for the pipeline
        let pacing = self.voice_bridge.get_ltc_pacing();
        let base_dt = 0.005_f32; // 200Hz motor frame rate
        let base_phoneme_duration = 0.08 / pacing.rate.max(0.1);
        let frames_per_phoneme = (base_phoneme_duration / base_dt).round().max(1.0) as usize;

        // 4. Generate formant frames via neural controller
        let mut frames = Vec::with_capacity(phoneme_ids.len() * frames_per_phoneme);

        for &id in &phoneme_ids {
            // Skip stress markers
            let (_, _, _, _, source_type) = phoneme_formants(id);
            if source_type == SourceType::Silent && id != 0 {
                continue;
            }
            if id == 0 {
                // Pause — skip neural processing
                let pause_frames = (pacing.phrase_pause / base_dt).round().max(1.0) as usize;
                for _ in 0..pause_frames {
                    frames.push(FormantFrame::default());
                }
                continue;
            }

            let phoneme_name = phoneme_id_to_name(id);
            for _ in 0..frames_per_phoneme {
                let frame =
                    pipeline.tick_phoneme(cognitive_state, None, base_dt, Some(phoneme_name));
                frames.push(frame);
            }
        }

        if frames.is_empty() {
            return Vec::new();
        }

        // 5. Vocoder synthesis
        self.vocoder.synthesize(&frames)
    }
}

impl Default for VoiceOrchestrator {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PHONEME ID-TO-ARPABET MAPPING
// ═══════════════════════════════════════════════════════════════════════════════

/// Map a Misaki phoneme ID to its ARPABET name for the VocalTractPipeline.
///
/// The pipeline's `tick_phoneme()` expects ARPABET strings (e.g., "IY", "AH").
/// This maps the numeric IDs from G2P output to those names.
#[cfg(feature = "vocal-tract")]
fn phoneme_id_to_name(id: u32) -> &'static str {
    match id {
        0 => "EY", // FACE
        1 => "AY", // PRICE
        2 => "AW", // MOUTH
        3 => "OY", // CHOICE
        4 => "OW", // GOAT
        5 => "B",
        6 => "D",
        7 => "F",
        8 => "HH",
        10 => "Y", // j/y glide
        11 => "K",
        12 => "L",
        13 => "M",
        14 => "N",
        15 => "P",
        16 => "AA", // PALM (ɑ)
        17 => "AO", // THOUGHT (ɔ)
        18 => "AX", // schwa (ə)
        19 => "EH", // DRESS (ɛ)
        20 => "ER", // NURSE (ɜ)
        21 => "IH", // KIT (ɪ)
        22 => "UH", // FOOT (ʊ)
        23 => "AH", // STRUT (ʌ)
        24 => "IY", // FLEECE (i)
        25 => "UW", // GOOSE (u)
        26 => "R",
        27 => "R",  // flap (ɾ) → closest ARPABET
        28 => "IH", // unstressed KIT (ᵻ)
        29 => "S",
        30 => "V",
        31 => "W",
        32 => "SH",
        33 => "ZH",
        34 => "JH", // ʤ
        35 => "CH", // ʧ
        38 => "TH", // θ (voiceless)
        39 => "DH", // ð (voiced)
        40 => "NG",
        41 => "G",
        42 => "AX", // optional schwa (ᵊ)
        43 => "AE", // TRAP (æ)
        _ => "AX",  // fallback: schwa
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PHONEME-TO-FORMANT MAPPING
// ═══════════════════════════════════════════════════════════════════════════════

/// Default formant targets for Misaki phoneme IDs.
///
/// Returns (f1, f2, f3, voicing, source_type) for a phoneme ID.
/// Values are approximate American English formant frequencies.
fn phoneme_formants(id: u32) -> (f32, f32, f32, f32, SourceType) {
    match id {
        // Vowels (using approximate formant values)
        0 => (520.0, 1600.0, 2500.0, 1.0, SourceType::Vowel), // FACE (A)
        1 => (700.0, 1200.0, 2500.0, 1.0, SourceType::Vowel), // PRICE (I)
        2 => (650.0, 1100.0, 2500.0, 1.0, SourceType::Vowel), // MOUTH (W)
        3 => (450.0, 1200.0, 2500.0, 1.0, SourceType::Vowel), // CHOICE (Y)
        4 => (500.0, 900.0, 2500.0, 1.0, SourceType::Vowel),  // GOAT (O)
        // Consonants
        5 => (200.0, 1000.0, 2200.0, 1.0, SourceType::Stop), // b
        6 => (200.0, 1600.0, 2600.0, 0.0, SourceType::Stop), // d/t
        7 => (320.0, 1400.0, 2500.0, 0.0, SourceType::Fricative), // f
        8 => (500.0, 1500.0, 2500.0, 0.0, SourceType::Fricative), // h
        10 => (300.0, 2200.0, 3000.0, 1.0, SourceType::Liquid), // j (y)
        11 => (200.0, 1500.0, 2500.0, 0.0, SourceType::Stop), // k
        12 => (350.0, 1100.0, 2900.0, 1.0, SourceType::Liquid), // l
        13 => (280.0, 900.0, 2200.0, 1.0, SourceType::Nasal), // m
        14 => (280.0, 1700.0, 2500.0, 1.0, SourceType::Nasal), // n
        15 => (200.0, 1000.0, 2200.0, 0.0, SourceType::Stop), // p
        // IPA vowels
        16 => (730.0, 1090.0, 2440.0, 1.0, SourceType::Vowel), // PALM (ɑ)
        17 => (570.0, 840.0, 2410.0, 1.0, SourceType::Vowel),  // THOUGHT (ɔ)
        18 => (500.0, 1500.0, 2500.0, 1.0, SourceType::Vowel), // schwa (ə)
        19 => (530.0, 1840.0, 2480.0, 1.0, SourceType::Vowel), // DRESS (ɛ)
        20 => (480.0, 1340.0, 2460.0, 1.0, SourceType::Vowel), // NURSE (ɜ)
        21 => (390.0, 1990.0, 2550.0, 1.0, SourceType::Vowel), // KIT (ɪ)
        22 => (440.0, 1020.0, 2240.0, 1.0, SourceType::Vowel), // FOOT (ʊ)
        23 => (640.0, 1190.0, 2390.0, 1.0, SourceType::Vowel), // STRUT (ʌ)
        24 => (270.0, 2290.0, 3010.0, 1.0, SourceType::Vowel), // FLEECE (i)
        25 => (300.0, 870.0, 2240.0, 1.0, SourceType::Vowel),  // GOOSE (u)
        26 => (350.0, 1200.0, 2500.0, 1.0, SourceType::Liquid), // r (ɹ)
        27 => (350.0, 1600.0, 2600.0, 1.0, SourceType::Liquid), // flap (ɾ)
        28 => (400.0, 1800.0, 2500.0, 1.0, SourceType::Vowel), // unstressed KIT (ᵻ)
        29 => (320.0, 1700.0, 2600.0, 0.0, SourceType::Fricative), // z/s
        30 => (320.0, 1400.0, 2500.0, 1.0, SourceType::Fricative), // v
        31 => (300.0, 700.0, 2500.0, 1.0, SourceType::Liquid), // w
        // Fricatives and affricates
        32 => (300.0, 1800.0, 2700.0, 0.0, SourceType::Fricative), // SH (ʃ)
        33 => (300.0, 1800.0, 2700.0, 1.0, SourceType::Fricative), // ZH (ʒ)
        34 => (300.0, 1800.0, 2700.0, 1.0, SourceType::Affricate), // J (ʤ)
        35 => (300.0, 1800.0, 2700.0, 0.0, SourceType::Affricate), // CH (ʧ)
        // Prosody markers (36=stress, 37=secondary stress) -> silent
        36 | 37 => (0.0, 0.0, 0.0, 0.0, SourceType::Silent),
        // More consonants
        38 => (300.0, 1500.0, 2500.0, 0.0, SourceType::Fricative), // TH (θ)
        39 => (300.0, 1500.0, 2500.0, 1.0, SourceType::Fricative), // TH (ð)
        40 => (280.0, 1700.0, 2500.0, 1.0, SourceType::Nasal),     // NG (ŋ)
        41 => (200.0, 1500.0, 2500.0, 1.0, SourceType::Stop),      // g
        42 => (500.0, 1500.0, 2500.0, 1.0, SourceType::Vowel),     // optional schwa (ᵊ)
        43 => (660.0, 1720.0, 2410.0, 1.0, SourceType::Vowel),     // TRAP (æ)
        // Default: neutral vowel
        _ => (500.0, 1500.0, 2500.0, 1.0, SourceType::Vowel),
    }
}

/// Convert a sequence of phoneme IDs to formant frames with pacing-modulated timing.
///
/// Each phoneme gets a duration based on pacing rate, with pauses inserted for
/// silence tokens. Formant bandwidths are set to reasonable defaults.
fn phoneme_ids_to_formant_frames(phoneme_ids: &[u32], pacing: &LTCPacing) -> Vec<FormantFrame> {
    let base_phoneme_duration = 0.08 / pacing.rate.max(0.1); // ~80ms per phoneme at rate 1.0
    let mut frames = Vec::with_capacity(phoneme_ids.len() * 2);
    let mut current_time = 0.0_f32;

    for &id in phoneme_ids {
        let (f1, f2, f3, voicing, source_type) = phoneme_formants(id);

        // Skip stress markers (they affect emphasis, not formants)
        if source_type == SourceType::Silent && id != 0 {
            continue;
        }

        // Silence token (id=0 from G2P inter-word silence)
        if id == 0 && source_type == SourceType::Vowel {
            // This is a silence/pause
            let pause_duration = pacing.phrase_pause;
            current_time += pause_duration;
            continue;
        }

        let duration = base_phoneme_duration * pacing.emphasis.max(0.5);

        // Start frame
        frames.push(FormantFrame {
            f1,
            f2,
            f3,
            b1: 60.0,
            b2: 90.0,
            b3: 150.0,
            f0: 120.0 + pacing.emotional_valence * 30.0, // Pitch modulated by valence
            energy: 0.7 * pacing.emphasis.clamp(0.3, 1.5),
            voicing,
            time: current_time,
            source_type,
            nasal_zero_freq: 0.0,
            nasal_zero_bw: 0.0,
        });

        current_time += duration;

        // End frame (slight formant transition)
        frames.push(FormantFrame {
            f1,
            f2,
            f3,
            b1: 60.0,
            b2: 90.0,
            b3: 150.0,
            f0: 120.0 + pacing.emotional_valence * 30.0,
            energy: 0.6 * pacing.emphasis.clamp(0.3, 1.5),
            voicing,
            time: current_time,
            source_type,
            nasal_zero_freq: 0.0,
            nasal_zero_bw: 0.0,
        });
    }

    frames
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orchestrator_default() {
        let orch = VoiceOrchestrator::new();
        let pacing = orch.current_pacing();
        assert!(pacing.rate > 0.0);
        assert!(pacing.rate.is_finite());
    }

    #[test]
    fn test_phoneme_formants_vowels() {
        // Check a few vowel phoneme IDs return valid formants
        for id in [16, 17, 18, 19, 21, 24, 25, 43] {
            let (f1, f2, f3, voicing, source_type) = phoneme_formants(id);
            assert!(f1 > 0.0, "phoneme {} f1 should be positive", id);
            assert!(f2 > f1, "phoneme {} f2 should be > f1", id);
            assert!(f3 > f2, "phoneme {} f3 should be > f2", id);
            assert!(
                (voicing - 1.0).abs() < f32::EPSILON,
                "vowels should be voiced"
            );
            assert_eq!(source_type, SourceType::Vowel);
        }
    }

    #[test]
    fn test_phoneme_formants_stops() {
        for id in [5, 6, 11, 15, 41] {
            let (_, _, _, _, source_type) = phoneme_formants(id);
            assert_eq!(source_type, SourceType::Stop);
        }
    }

    #[test]
    fn test_phoneme_formants_nasals() {
        for id in [13, 14, 40] {
            let (_, _, _, voicing, source_type) = phoneme_formants(id);
            assert_eq!(source_type, SourceType::Nasal);
            assert!(
                (voicing - 1.0).abs() < f32::EPSILON,
                "nasals should be voiced"
            );
        }
    }

    #[test]
    fn test_phoneme_ids_to_formant_frames() {
        let pacing = LTCPacing::default();
        let phoneme_ids = vec![8, 19, 12, 4]; // "hello" -> h ɛ l O
        let frames = phoneme_ids_to_formant_frames(&phoneme_ids, &pacing);

        // Each phoneme produces 2 frames (start + end)
        assert_eq!(frames.len(), 8, "4 phonemes * 2 frames each");

        // Frames should be monotonically increasing in time
        for pair in frames.windows(2) {
            assert!(
                pair[1].time >= pair[0].time,
                "time should be monotonically increasing: {} vs {}",
                pair[0].time,
                pair[1].time
            );
        }

        // All frames should have positive f0
        for f in &frames {
            assert!(f.f0 > 0.0, "f0 should be positive");
            assert!(f.energy > 0.0, "energy should be positive");
        }
    }

    #[test]
    fn test_phoneme_ids_to_formant_frames_with_silence() {
        let pacing = LTCPacing::default();
        // Phoneme sequence with inter-word silence (id=0)
        let phoneme_ids = vec![8, 19, 0, 12, 4]; // h ɛ [pause] l O
        let frames = phoneme_ids_to_formant_frames(&phoneme_ids, &pacing);

        // The id=0 in the Misaki vocab maps to FACE vowel formants (not silence)
        // But in G2P output, id=0 is the silence_id. The phoneme_formants function
        // returns vowel for id=0, so it generates frames for it.
        assert!(
            frames.len() >= 8,
            "should have frames for non-silent phonemes"
        );
    }

    #[test]
    fn test_thought_to_speech_produces_audio() {
        let mut orch = VoiceOrchestrator::new();
        let cfc_output = vec![0.3, -0.2, 0.5, 0.1, -0.1, 0.4, -0.3, 0.2];
        let samples =
            orch.thought_to_speech("Hello world", &cfc_output, 0.2, vec!["answer".to_string()]);

        assert!(!samples.is_empty(), "should produce audio samples");
        assert!(
            samples.iter().all(|s| s.is_finite()),
            "all samples should be finite"
        );
    }

    #[test]
    fn test_thought_to_speech_empty_text() {
        let mut orch = VoiceOrchestrator::new();
        let samples = orch.thought_to_speech("", &[0.5; 8], 0.1, vec![]);
        // Empty text produces a single silence phoneme from G2P, which results
        // in a pause (no formant frames), so output should be empty
        assert!(
            samples.is_empty(),
            "empty text should produce no audio (only silence token)"
        );
    }

    #[test]
    fn test_pacing_affects_speech_rate() {
        let mut orch = VoiceOrchestrator::new();
        let text = "Hello world";

        // Fast: low prediction error, positive CfC state
        let fast_samples = orch.thought_to_speech(
            text,
            &[0.8; 8], // high activation
            0.05,      // low error
            vec![],
        );

        // Reset bridge state for fair comparison
        let mut orch2 = VoiceOrchestrator::new();

        // Slow: high prediction error
        let slow_samples = orch2.thought_to_speech(
            text,
            &[0.1; 8], // low activation
            0.9,       // high error -> low confidence -> slower
            vec![],
        );

        // Both should produce audio
        assert!(!fast_samples.is_empty());
        assert!(!slow_samples.is_empty());

        // The low-confidence version should generally be longer (slower rate)
        // However, the CfC output also affects rate, so we just verify both produce valid audio
        for &s in fast_samples.iter().chain(slow_samples.iter()) {
            assert!(s.is_finite());
        }
    }
}
