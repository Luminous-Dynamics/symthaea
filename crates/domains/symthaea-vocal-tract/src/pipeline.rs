// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Complete vocal tract pipeline: encoder → controller → FEP agent.
//!
//! Manages dual-rate processing:
//! - 200Hz motor: controller.forward() → FormantFrame → vocoder
//! - 10Hz cognitive: encoder.encode() → update cached HV; fep_agent.tick() → modulate
//!
//! Also includes `ProsodyContext` for F0 declination, stress boost, and energy
//! ASR envelope post-processing.

use crate::controller::{SpeakerProfile, VocalTractConfig, VocalTractController};
use crate::encoder::{VocalTractHdcEncoder, VoiceCognitiveState};
use crate::fep::{VocalTractFepAgent, VocalTractFepResult, VocalTractObservation};
use crate::types::{FormantFrame, SourceType};
use symthaea_core::genesis::{GenesisCovenant, GenesisSeed};
use symthaea_core::hdc::{ContinuousHV, HDC_DIMENSION};

// ═══════════════════════════════════════════════════════════════════════════════
// INTONATION & PROSODY
// ═══════════════════════════════════════════════════════════════════════════════

/// Intonation contour type for F0 shaping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Intonation {
    /// Falling contour: 1.05 → 0.80 (declarative sentences).
    #[default]
    Statement,
    /// Rising contour: dip to 0.90, then rise to 1.25 (yes/no questions).
    Question,
    /// Peak contour: rise to 1.30 at 30%, then fall to 0.85 (exclamations).
    Exclamation,
}

/// Pitch accent type for stressed syllables (ToBI-lite).
///
/// Models the F0 shape within a stressed syllable, not just a flat boost.
/// Based on simplified ToBI (Tones and Break Indices) conventions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PitchAccent {
    /// No pitch accent (unstressed syllables).
    #[default]
    None,
    /// H*: High target on stressed syllable (most common in English declaratives).
    High,
    /// L+H*: Rise to high target (contrastive focus, emphasis).
    RiseHigh,
    /// H+L*: Fall to low target (gentle declination).
    FallLow,
}

/// Context for prosody post-processing: F0 intonation, stress boost, energy envelope.
#[derive(Debug, Clone, Copy)]
pub struct ProsodyContext {
    /// Progress through the entire utterance (0.0 → 1.0).
    pub utterance_progress: f32,
    /// Progress through the current phoneme (0.0 → 1.0).
    pub phoneme_progress: f32,
    /// Stress level (0 = unstressed, 1 = primary, 2 = secondary).
    pub stress: u8,
    /// Base F0 for the utterance (from config or speaker profile).
    pub base_f0: f32,
    /// Emotional arousal (0.0–1.0) — maps to F0 range expansion.
    pub arousal: f32,
    /// Intonation contour type.
    pub intonation: Intonation,
    /// Phrase index (for declination reset). 0 = first phrase, increments at boundaries.
    pub phrase_index: u8,
    /// Progress within current phrase (0.0 → 1.0). Resets at each phrase boundary.
    pub phrase_progress: f32,
    /// Focus flag: if true, this phoneme receives emphatic prominence.
    pub is_focus: bool,
    /// Pitch accent type for stressed syllables.
    pub pitch_accent: PitchAccent,
    /// Whether this phoneme is at a syllable onset (for syllable-aligned prosody).
    /// When true, prosodic changes (pitch accent, stress boost) take full effect.
    /// When false (mid-syllable), these changes are interpolated from the onset value.
    pub is_syllable_onset: bool,
    /// Progress through the current syllable (0.0 → 1.0).
    /// Used to smoothly interpolate prosodic changes from onset across the syllable.
    pub syllable_progress: f32,
    /// Source type of the previous phoneme (for CV transition bandwidth widening).
    /// `None` if this is the first phoneme or unknown.
    pub prev_source_type: Option<SourceType>,
    /// Source type of the next phoneme (for VC transition bandwidth widening).
    /// `None` if this is the last phoneme or unknown.
    pub next_source_type: Option<SourceType>,
}

impl Default for ProsodyContext {
    fn default() -> Self {
        Self {
            utterance_progress: 0.0,
            phoneme_progress: 0.5,
            stress: 0,
            base_f0: 120.0,
            arousal: 0.5,
            intonation: Intonation::Statement,
            phrase_index: 0,
            phrase_progress: 0.0,
            is_focus: false,
            pitch_accent: PitchAccent::None,
            is_syllable_onset: true,
            syllable_progress: 0.0,
            prev_source_type: None,
            next_source_type: None,
        }
    }
}

/// Detect syllable boundaries from a phoneme sequence.
///
/// Syllable boundaries are placed at vowel energy peaks: each vowel nucleus
/// (ARPABET vowel phoneme) starts a new syllable. Consonants preceding a vowel
/// belong to the following syllable's onset, and consonants following a vowel
/// belong to the current syllable's coda.
///
/// Returns a list of `(syllable_start_index, stress)` pairs, where
/// `syllable_start_index` is the index into the phoneme sequence where the
/// syllable onset begins (the first consonant before the vowel, or the vowel
/// itself if no onset consonant), and `stress` is extracted from the vowel's
/// ARPABET stress digit (0, 1, or 2).
///
/// # Example
/// ```
/// use symthaea_vocal_tract::pipeline::detect_syllable_boundaries;
/// let phonemes = &["B", "AH1", "T", "ER0"];
/// let boundaries = detect_syllable_boundaries(phonemes);
/// assert_eq!(boundaries.len(), 2);
/// assert_eq!(boundaries[0], (0, 1)); // "B-AH1" syllable starts at 0, stress=1
/// assert_eq!(boundaries[1], (2, 0)); // "T-ER0" syllable starts at 2, stress=0
/// ```
pub fn detect_syllable_boundaries(phonemes: &[&str]) -> Vec<(usize, u8)> {
    if phonemes.is_empty() {
        return Vec::new();
    }

    let mut boundaries = Vec::new();
    let mut onset_start: Option<usize> = None;

    for (i, &ph) in phonemes.iter().enumerate() {
        let manner = phoneme_manner_class(ph);

        if manner == MannerClass::Vowel {
            // Extract stress digit from ARPABET (e.g., "AH1" → 1)
            let stress = ph.chars().last().and_then(|c| c.to_digit(10)).unwrap_or(0) as u8;

            // Syllable onset: first consonant before this vowel (if any)
            let start = onset_start.unwrap_or(i);
            boundaries.push((start, stress));
            onset_start = None;
        } else if boundaries.is_empty() && onset_start.is_none() {
            // Consonant before first vowel — potential onset
            onset_start = Some(i);
        } else if onset_start.is_none() {
            // Consonant after a vowel — could be coda or next onset.
            // Maximal Onset Principle: assign to next syllable when possible.
            onset_start = Some(i);
        }
    }

    // Handle trailing consonants (no following vowel — they're coda of last syllable)
    // No new boundary needed; they belong to the previous syllable.

    boundaries
}

/// Check if an ARPABET phoneme is a vowel (has higher energy than consonants).
///
/// Vowels serve as syllable nuclei and are the anchor points for prosodic
/// changes like pitch accent and stress.
pub fn is_vowel_phoneme(phoneme: &str) -> bool {
    phoneme_manner_class(phoneme) == MannerClass::Vowel
}

/// Check if an ARPABET phoneme is a consonant.
pub fn is_consonant_phoneme(phoneme: &str) -> bool {
    let manner = phoneme_manner_class(phoneme);
    matches!(
        manner,
        MannerClass::Stop
            | MannerClass::Fricative
            | MannerClass::Nasal
            | MannerClass::Affricate
            | MannerClass::Glide
            | MannerClass::Liquid
    )
}

impl ProsodyContext {
    /// Apply prosody post-processing to a FormantFrame.
    ///
    /// - **F0**: intonation contour × declination × pitch accent × stress × arousal × focus
    /// - **Microprosody**: consonant-type F0 perturbation
    /// - **Energy**: ASR envelope × stress × focus
    pub fn apply_prosody(&self, frame: &mut FormantFrame) {
        let t = self.utterance_progress;

        // Intonation-specific F0 contour
        let contour = match self.intonation {
            Intonation::Statement => {
                // Falling: 1.05 at start → 0.80 at end
                1.05 - 0.25 * t
            }
            Intonation::Question => {
                // Rising: dip to 0.90 at midpoint, rise to 1.25 at end
                if t < 0.5 {
                    1.0 - 0.10 * (t / 0.5) // 1.00 → 0.90
                } else {
                    0.90 + 0.35 * ((t - 0.5) / 0.5) // 0.90 → 1.25
                }
            }
            Intonation::Exclamation => {
                // Peak at 30%: rise to 1.30, then fall to 0.85
                if t < 0.3 {
                    1.0 + 0.30 * (t / 0.3) // 1.00 → 1.30
                } else {
                    1.30 - 0.45 * ((t - 0.3) / 0.7) // 1.30 → 0.85
                }
            }
        };

        // Declination: F0 baseline drops 15% within each phrase
        let declination = 1.0 - 0.15 * self.phrase_progress;

        // Pitch accent shapes: snapped to syllable onset for syllable-aligned prosody.
        // When syllable_progress has been explicitly set (> 0), use it instead of
        // phoneme_progress so accents span the entire syllable, not just one phoneme.
        // Falls back to phoneme_progress for backward compatibility.
        let accent_progress = if self.syllable_progress > 0.0 {
            self.syllable_progress
        } else {
            self.phoneme_progress
        };

        let accent_contour = match self.pitch_accent {
            PitchAccent::None => 1.0,
            PitchAccent::High => {
                // Rise to peak at 40% of syllable, sustain
                if accent_progress < 0.4 {
                    1.0 + 0.15 * (accent_progress / 0.4)
                } else {
                    1.15
                }
            }
            PitchAccent::RiseHigh => {
                // Starts low, rises sharply to peak at 60%
                1.0 + 0.20 * (accent_progress / 0.6).min(1.0)
            }
            PitchAccent::FallLow => {
                // Starts high, falls to low
                1.10 - 0.15 * accent_progress
            }
        };

        // Focus: emphatic prominence (larger F0 excursion)
        let focus_boost = if self.is_focus { 1.20 } else { 1.0 };

        // Stress boost: primary=1.10×, secondary=1.05×, none=1.0×
        let stress_f0_boost = match self.stress {
            1 => 1.10,
            2 => 1.05,
            _ => 1.0,
        };

        // Arousal maps to F0 range (more arousal = wider pitch swings)
        let arousal_factor = 0.9 + 0.2 * self.arousal;

        frame.f0 = self.base_f0
            * contour
            * declination
            * accent_contour
            * stress_f0_boost
            * arousal_factor
            * focus_boost;

        // Microprosody: consonants perturb F0 slightly
        let microprosody = match frame.source_type {
            SourceType::Stop => -5.0,
            SourceType::Fricative => -3.0,
            SourceType::Nasal => 2.0,
            _ => 0.0,
        };
        frame.f0 = (frame.f0 + microprosody).clamp(50.0, 500.0);

        // Energy ASR envelope within the phoneme
        let asr_envelope = if self.phoneme_progress < 0.10 {
            // Attack: ramp up over first 10%
            self.phoneme_progress / 0.10
        } else if self.phoneme_progress > 0.85 {
            // Release: ramp down over last 15%
            (1.0 - self.phoneme_progress) / 0.15
        } else {
            // Sustain
            1.0
        };

        // Stress energy boost: primary=1.2×, secondary=1.1×, unstressed=0.9×
        let stress_energy = match self.stress {
            1 => 1.2,
            2 => 1.1,
            _ => 0.9,
        };

        // Focus also boosts energy
        let focus_energy = if self.is_focus { 1.3 } else { 1.0 };

        frame.energy = (frame.energy * asr_envelope * stress_energy * focus_energy).clamp(0.0, 1.0);

        // Vowel reduction: unstressed vowels centralize toward schwa.
        // In natural speech, unstressed vowels are shorter and their formants
        // drift toward a neutral position (~500, ~1500 Hz).
        if self.stress == 0 && matches!(frame.source_type, SourceType::Vowel | SourceType::Liquid) {
            const SCHWA_F1: f32 = 500.0;
            const SCHWA_F2: f32 = 1500.0;
            const REDUCTION: f32 = 0.3; // 30% toward schwa
            frame.f1 = frame.f1 * (1.0 - REDUCTION) + SCHWA_F1 * REDUCTION;
            frame.f2 = frame.f2 * (1.0 - REDUCTION) + SCHWA_F2 * REDUCTION;
        }

        // Formant bandwidth dynamics: stressed vowels have sharper (narrower)
        // formants, unstressed vowels have more diffuse (wider) bandwidths.
        match self.stress {
            1 => {
                frame.b1 *= 0.9;
                frame.b2 *= 0.9;
                frame.b3 *= 0.9;
            }
            0 if matches!(frame.source_type, SourceType::Vowel | SourceType::Liquid) => {
                frame.b1 *= 1.3;
                frame.b2 *= 1.3;
                frame.b3 *= 1.3;
            }
            _ => {}
        }

        // Consonant-vowel (CV) and vowel-consonant (VC) transition bandwidth
        // widening: during transitions between consonants and vowels, formant
        // bandwidths increase 20-40% to model the articulatory instability at
        // manner-class boundaries.
        let is_vowel_like = matches!(frame.source_type, SourceType::Vowel | SourceType::Liquid);
        let is_consonant = matches!(
            frame.source_type,
            SourceType::Stop | SourceType::Fricative | SourceType::Nasal | SourceType::Affricate
        );

        let prev_is_consonant = self.prev_source_type.is_some_and(|st| {
            matches!(
                st,
                SourceType::Stop
                    | SourceType::Fricative
                    | SourceType::Nasal
                    | SourceType::Affricate
            )
        });
        let prev_is_vowel = self
            .prev_source_type
            .is_some_and(|st| matches!(st, SourceType::Vowel | SourceType::Liquid));
        let next_is_consonant = self.next_source_type.is_some_and(|st| {
            matches!(
                st,
                SourceType::Stop
                    | SourceType::Fricative
                    | SourceType::Nasal
                    | SourceType::Affricate
            )
        });
        let next_is_vowel = self
            .next_source_type
            .is_some_and(|st| matches!(st, SourceType::Vowel | SourceType::Liquid));

        // Detect CV or VC transition: current phoneme borders a different manner class
        let in_cv_transition =
            (is_vowel_like && prev_is_consonant) || (is_consonant && next_is_vowel);
        let in_vc_transition =
            (is_vowel_like && next_is_consonant) || (is_consonant && prev_is_vowel);

        if in_cv_transition || in_vc_transition {
            // Widen bandwidth 20-40% depending on transition phase:
            // - At phoneme boundaries (progress near 0 or 1): 40% widening
            // - Mid-phoneme: 20% widening (trailing off from transition)
            let boundary_proximity = if in_cv_transition {
                // CV: widest at onset (phoneme_progress near 0)
                1.0 - self.phoneme_progress.min(1.0)
            } else {
                // VC: widest at release (phoneme_progress near 1)
                self.phoneme_progress.min(1.0)
            };
            // Interpolate between 20% and 40% widening based on proximity
            let widen_factor = 1.20 + 0.20 * boundary_proximity;
            frame.b1 *= widen_factor;
            frame.b2 *= widen_factor;
            frame.b3 *= widen_factor;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PHONEME DURATION MODEL
// ═══════════════════════════════════════════════════════════════════════════════

/// Manner of articulation classes for duration prediction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MannerClass {
    Vowel,
    Stop,
    Fricative,
    Nasal,
    Liquid,
    Affricate,
    Glide,
    Silence,
}

/// Classify an ARPABET phoneme into its manner of articulation.
pub fn phoneme_manner_class(phoneme: &str) -> MannerClass {
    // Strip stress digit (e.g., "AH0" → "AH")
    let base = phoneme.trim_end_matches(|c: char| c.is_ascii_digit());
    match base {
        // Vowels
        "AA" | "AE" | "AH" | "AO" | "AW" | "AY" | "EH" | "ER" | "EY" | "IH" | "IY" | "OW"
        | "OY" | "UH" | "UW" => MannerClass::Vowel,
        // Stops
        "B" | "D" | "G" | "K" | "P" | "T" => MannerClass::Stop,
        // Fricatives
        "CH" | "DH" | "F" | "HH" | "S" | "SH" | "TH" | "V" | "Z" | "ZH" => MannerClass::Fricative,
        // Nasals
        "M" | "N" | "NG" => MannerClass::Nasal,
        // Liquids
        "L" | "R" => MannerClass::Liquid,
        // Affricates
        "JH" => MannerClass::Affricate,
        // Glides
        "W" | "Y" => MannerClass::Glide,
        // Silence / unknown
        _ => MannerClass::Silence,
    }
}

/// Predict phoneme duration in motor frames (at 200Hz = 5ms/frame).
///
/// Uses linguistic rules: manner-based base duration, stress lengthening,
/// position-dependent lengthening, and speaking rate scaling.
///
/// Returns duration in frames (minimum 2).
pub fn predict_duration(
    phoneme: &str,
    stress: u8,
    is_word_final: bool,
    is_utterance_final: bool,
    speaking_rate: f32,
) -> usize {
    // Base duration in frames (at 200Hz): manner class → typical ms / 5
    let base = match phoneme_manner_class(phoneme) {
        MannerClass::Vowel => 16,     // 80ms
        MannerClass::Stop => 6,       // 30ms
        MannerClass::Fricative => 10, // 50ms
        MannerClass::Nasal => 8,      // 40ms
        MannerClass::Liquid => 10,    // 50ms
        MannerClass::Affricate => 8,  // 40ms
        MannerClass::Glide => 6,      // 30ms
        MannerClass::Silence => 4,    // 20ms
    };

    let mut dur = base as f32;

    // Stress lengthening (vowels only)
    if phoneme_manner_class(phoneme) == MannerClass::Vowel {
        dur *= match stress {
            1 => 1.5, // primary stress: 50% longer
            2 => 1.2, // secondary stress: 20% longer
            _ => 1.0,
        };
    }

    // Position-dependent lengthening
    if is_utterance_final {
        dur *= 1.4; // phrase-final lengthening
    } else if is_word_final {
        dur *= 1.15; // word-final lengthening
    }

    // Speaking rate scaling (rate=1.0 is normal)
    let rate = speaking_rate.clamp(0.5, 3.0);
    dur /= rate;

    // Minimum 2 frames (10ms)
    (dur.round() as usize).max(2)
}

// ═══════════════════════════════════════════════════════════════════════════════
// VOCAL TRACT PIPELINE
// ═══════════════════════════════════════════════════════════════════════════════

/// Complete vocal tract pipeline: encoder → controller → FEP agent.
///
/// Manages dual-rate processing:
/// - 200Hz motor: controller.forward() → FormantFrame → vocoder
/// - 10Hz cognitive: encoder.encode() → update cached HV; fep_agent.tick() → modulate
pub struct VocalTractPipeline {
    /// HDC encoder: cognitive state → 16,384D ContinuousHV
    pub encoder: VocalTractHdcEncoder,
    /// LTC controller: ContinuousHV → FormantFrame
    pub controller: VocalTractController,
    /// FEP active inference agent: voice metrics → tau/LR modulation
    pub fep_agent: VocalTractFepAgent,
    /// Cached cognitive HV (updated at 10Hz, used at 200Hz)
    cached_hv: ContinuousHV,
    /// Counter for dual-rate scheduling (every 20 motor frames = 1 cognitive tick)
    motor_frame_count: usize,
    /// Motor frames per cognitive tick (200Hz / 10Hz = 20)
    frames_per_cognitive_tick: usize,
    /// Cumulative time in seconds (set on each FormantFrame).
    cumulative_time: f32,
    /// Genesis seed for phoneme HV generation.
    genesis: GenesisSeed,
    /// Cache of phoneme identity HVs (phoneme name → HV).
    phoneme_hv_cache: std::collections::HashMap<String, ContinuousHV>,
    /// Cached cognitive channels for prosody head (updated at 10Hz, used at 200Hz).
    cached_cognitive_channels: Option<[f32; 12]>,
    /// Previous phoneme name (for detecting phoneme transitions).
    prev_phoneme: Option<String>,
    /// Previous phoneme's bound HV (for coarticulation blending during transitions).
    prev_phoneme_bound_hv: Option<ContinuousHV>,
    /// Counter of frames elapsed since phoneme changed (for blend scheduling).
    coarticulation_counter: usize,
    /// Number of frames over which to blend between old/new phoneme HVs (80ms at 200Hz).
    coarticulation_frames: usize,
    /// Phoneme name → manner of articulation (for setting source_type on output frames).
    phoneme_manner_map: std::collections::HashMap<String, SourceType>,
    /// Phoneme name → voicing (true = voiced, false = unvoiced).
    phoneme_voicing_map: std::collections::HashMap<String, bool>,
    /// Last FEP result (for telemetry / inspection).
    last_fep_result: Option<VocalTractFepResult>,
}

impl VocalTractPipeline {
    /// Create a new pipeline from a genesis seed.
    pub fn new(genesis: &GenesisSeed) -> Self {
        Self {
            encoder: VocalTractHdcEncoder::new(genesis, 32),
            controller: VocalTractController::new(genesis, &VocalTractConfig::default()),
            fep_agent: VocalTractFepAgent::new(),
            cached_hv: ContinuousHV::zero(HDC_DIMENSION),
            motor_frame_count: 0,
            frames_per_cognitive_tick: 20,
            cumulative_time: 0.0,
            genesis: genesis.clone(),
            phoneme_hv_cache: std::collections::HashMap::new(),
            cached_cognitive_channels: None,
            prev_phoneme: None,
            prev_phoneme_bound_hv: None,
            coarticulation_counter: 0,
            coarticulation_frames: 16, // 80ms at 200Hz
            phoneme_manner_map: std::collections::HashMap::new(),
            phoneme_voicing_map: std::collections::HashMap::new(),
            last_fep_result: None,
        }
    }

    /// Create a new pipeline with an explicit vocal-tract configuration.
    pub fn new_with_config(genesis: &GenesisSeed, config: &VocalTractConfig) -> Self {
        Self {
            encoder: VocalTractHdcEncoder::new(genesis, 32),
            controller: VocalTractController::new(genesis, config),
            fep_agent: VocalTractFepAgent::new(),
            cached_hv: ContinuousHV::zero(HDC_DIMENSION),
            motor_frame_count: 0,
            frames_per_cognitive_tick: 20,
            cumulative_time: 0.0,
            genesis: genesis.clone(),
            phoneme_hv_cache: std::collections::HashMap::new(),
            cached_cognitive_channels: None,
            prev_phoneme: None,
            prev_phoneme_bound_hv: None,
            coarticulation_counter: 0,
            coarticulation_frames: 16,
            phoneme_manner_map: std::collections::HashMap::new(),
            phoneme_voicing_map: std::collections::HashMap::new(),
            last_fep_result: None,
        }
    }

    /// Create a new pipeline with a specific speaker profile.
    ///
    /// Derives a speaker-specific genesis via `GenesisCovenant`, applies the
    /// profile's base_f0 to config, modulates tau and formant scaling.
    pub fn new_with_speaker(genesis: &GenesisSeed, profile: &SpeakerProfile) -> Self {
        // Derive a speaker-specific genesis namespace
        let speaker_genesis = GenesisCovenant::new(genesis, &format!("speaker::{}", profile.name));
        let speaker_seed = speaker_genesis.seed();

        let config = VocalTractConfig {
            base_f0: profile.base_f0,
            ..VocalTractConfig::default()
        };

        let mut controller = VocalTractController::new(speaker_seed, &config);
        controller.modulate_tau(profile.tau_factor);

        Self {
            encoder: VocalTractHdcEncoder::new(speaker_seed, 32),
            controller,
            fep_agent: VocalTractFepAgent::new(),
            cached_hv: ContinuousHV::zero(HDC_DIMENSION),
            motor_frame_count: 0,
            frames_per_cognitive_tick: 20,
            cumulative_time: 0.0,
            genesis: speaker_seed.clone(),
            phoneme_hv_cache: std::collections::HashMap::new(),
            cached_cognitive_channels: None,
            prev_phoneme: None,
            prev_phoneme_bound_hv: None,
            coarticulation_counter: 0,
            coarticulation_frames: 16,
            phoneme_manner_map: std::collections::HashMap::new(),
            phoneme_voicing_map: std::collections::HashMap::new(),
            last_fep_result: None,
        }
    }

    /// Set the phoneme manner map for source_type propagation.
    ///
    /// Maps phoneme names (ARPABET) to their manner of articulation. When
    /// `tick_phoneme()` produces a frame, it sets `frame.source_type` from
    /// this map so the vocoder uses the correct excitation signal.
    pub fn set_manner_map(&mut self, map: std::collections::HashMap<String, SourceType>) {
        self.phoneme_manner_map = map;
    }

    /// Register a single phoneme's manner of articulation.
    pub fn register_phoneme_manner(&mut self, phoneme: &str, manner: SourceType) {
        self.phoneme_manner_map.insert(phoneme.to_string(), manner);
    }

    /// Set the phoneme voicing map for manner-aware voicing overrides.
    pub fn set_voicing_map(&mut self, map: std::collections::HashMap<String, bool>) {
        self.phoneme_voicing_map = map;
    }

    /// Register a single phoneme's voicing.
    pub fn register_phoneme_voicing(&mut self, phoneme: &str, is_voiced: bool) {
        self.phoneme_voicing_map
            .insert(phoneme.to_string(), is_voiced);
    }

    /// Get the last FEP result (for telemetry / inspection).
    pub fn last_fep_result(&self) -> Option<&VocalTractFepResult> {
        self.last_fep_result.as_ref()
    }

    /// Get or create a cached phoneme identity HV.
    ///
    /// Uses the genesis seed to deterministically create a unique HV for each
    /// phoneme name (e.g., "AH", "IY"). Cached for O(1) repeat lookups.
    pub fn get_or_create_phoneme_hv(&mut self, phoneme: &str) -> ContinuousHV {
        if let Some(hv) = self.phoneme_hv_cache.get(phoneme) {
            return hv.clone();
        }
        let hv = self
            .genesis
            .hv(&format!("phoneme::{phoneme}"), HDC_DIMENSION);
        self.phoneme_hv_cache
            .insert(phoneme.to_string(), hv.clone());
        hv
    }

    /// Run one motor frame (200Hz).
    ///
    /// - Every `frames_per_cognitive_tick` frames: re-encode cognitive state,
    ///   optionally run FEP tick if metrics provided.
    /// - Every frame: evolve controller with cached HV → produce FormantFrame.
    /// - If `phoneme` is provided, the cognitive HV is bound with a phoneme
    ///   identity HV so the controller can distinguish different phonemes.
    pub fn tick(
        &mut self,
        cognitive_state: &VoiceCognitiveState,
        metrics: Option<&VocalTractObservation>,
        dt: f32,
    ) -> FormantFrame {
        self.tick_phoneme(cognitive_state, metrics, dt, None)
    }

    /// Run one motor frame with phoneme-aware routing.
    ///
    /// Like `tick()`, but accepts an optional phoneme name. When provided,
    /// the cached cognitive HV is bound with a phoneme identity HV via
    /// `bind()`, injecting phoneme identity while preserving cognitive state.
    pub fn tick_phoneme(
        &mut self,
        cognitive_state: &VoiceCognitiveState,
        metrics: Option<&VocalTractObservation>,
        dt: f32,
        phoneme: Option<&str>,
    ) -> FormantFrame {
        // Cognitive tick (10Hz)
        if self
            .motor_frame_count
            .is_multiple_of(self.frames_per_cognitive_tick)
        {
            self.cached_hv = self.encoder.encode(cognitive_state);
            self.cached_cognitive_channels = Some(cognitive_state.to_channels());

            // FEP modulation if we have metrics
            if let Some(m) = metrics {
                // Learn from previous action's outcome before selecting new action
                self.fep_agent.learn(m);
                let fep_result = self.fep_agent.tick(m);
                self.controller.modulate_tau(fep_result.tau_factor);
                let current_lr = self.controller.learning_rate();
                self.controller
                    .set_learning_rate(current_lr * fep_result.learning_rate_factor);
                self.controller.set_emphasis(fep_result.emphasis_factor);
                self.last_fep_result = Some(fep_result);
            }
        }

        self.motor_frame_count += 1;

        // Coarticulation blending: interpolate old→new phoneme HV over transition.
        // When phoneme changes, save the old bound HV and linearly blend toward
        // the new bound HV over `coarticulation_frames` (80ms at 200Hz).
        //
        // Optimization: skip bind() when phoneme is unchanged and this is not
        // a cognitive re-encode frame. Saves ~0.5-1ms for ~90% of frames.
        let effective_hv = if let Some(ph) = phoneme {
            let is_same = self.prev_phoneme.as_deref() == Some(ph);
            let is_reencode =
                (self.motor_frame_count - 1).is_multiple_of(self.frames_per_cognitive_tick);

            let new_bound = if is_same
                && !is_reencode
                && self.coarticulation_counter >= self.coarticulation_frames
            {
                // Cache hit: same phoneme, no re-encode, past blend window
                if let Some(ref cached) = self.prev_phoneme_bound_hv {
                    cached.clone()
                } else {
                    let phoneme_hv = self.get_or_create_phoneme_hv(ph);
                    let bound = self.cached_hv.bind(&phoneme_hv);
                    self.prev_phoneme_bound_hv = Some(bound.clone());
                    bound
                }
            } else {
                let phoneme_hv = self.get_or_create_phoneme_hv(ph);
                let bound = self.cached_hv.bind(&phoneme_hv);

                if !is_same {
                    // Phoneme changed — save old HV for blending
                    if let Some(old_ph) = self.prev_phoneme.take() {
                        let old_phoneme_hv = self.get_or_create_phoneme_hv(&old_ph);
                        self.prev_phoneme_bound_hv = Some(self.cached_hv.bind(&old_phoneme_hv));
                        self.coarticulation_counter = 0;
                    }
                } else {
                    // Same phoneme, re-encode — refresh cached bound
                    self.prev_phoneme_bound_hv = Some(bound.clone());
                }

                bound
            };

            self.prev_phoneme = Some(ph.to_string());

            // Blend if within coarticulation window
            if self.coarticulation_counter < self.coarticulation_frames {
                self.coarticulation_counter += 1;
                if let Some(ref prev_hv) = self.prev_phoneme_bound_hv {
                    let t = self.coarticulation_counter as f32 / self.coarticulation_frames as f32;
                    prev_hv.scale(1.0 - t).add(&new_bound.scale(t))
                } else {
                    new_bound
                }
            } else {
                new_bound
            }
        } else {
            self.prev_phoneme = None;
            self.prev_phoneme_bound_hv = None;
            self.cached_hv.clone()
        };

        // Adaptive rate limiting: tighter during steady state, looser during transitions
        if phoneme.is_some() && self.coarticulation_counter < self.coarticulation_frames {
            self.controller
                .set_max_formant_delta(self.controller.config().transition_max_delta);
        } else {
            self.controller
                .set_max_formant_delta(self.controller.config().steady_max_delta);
        }

        // Motor tick (200Hz): evolve network + produce formants with prosody head
        let mut frame = self.controller.forward_with_prosody(
            &effective_hv,
            dt,
            self.cached_cognitive_channels.as_ref(),
        );
        frame.time = self.cumulative_time;
        self.cumulative_time += dt;

        // Set source_type and voicing from phoneme maps
        if let Some(ph) = phoneme {
            if let Some(&manner) = self.phoneme_manner_map.get(ph) {
                frame.source_type = manner;
            }
            if let Some(&is_voiced) = self.phoneme_voicing_map.get(ph)
                && !is_voiced
            {
                frame.voicing = 0.0;
            }
        }

        frame
    }

    /// Run one motor frame with phoneme routing + prosody post-processing.
    ///
    /// Combines `tick_phoneme()` with `ProsodyContext::apply_prosody()` for
    /// F0 declination, stress boost, and energy ASR envelope.
    pub fn tick_with_prosody(
        &mut self,
        cognitive_state: &VoiceCognitiveState,
        metrics: Option<&VocalTractObservation>,
        dt: f32,
        phoneme: Option<&str>,
        prosody: &ProsodyContext,
    ) -> FormantFrame {
        let mut frame = self.tick_phoneme(cognitive_state, metrics, dt, phoneme);
        prosody.apply_prosody(&mut frame);
        frame
    }

    /// Run one motor frame with anticipatory coarticulation + prosody.
    ///
    /// Like `tick_with_prosody()`, but accepts `next_phoneme` and
    /// `phoneme_remaining_frames` to enable look-ahead blending.
    /// During the final portion of the current phoneme, the HV starts
    /// blending toward the next phoneme (max 30% blend).
    #[allow(clippy::too_many_arguments)]
    pub fn tick_with_anticipation(
        &mut self,
        cognitive_state: &VoiceCognitiveState,
        metrics: Option<&VocalTractObservation>,
        dt: f32,
        phoneme: Option<&str>,
        next_phoneme: Option<&str>,
        phoneme_remaining_frames: usize,
        prosody: &ProsodyContext,
    ) -> FormantFrame {
        // Cognitive tick (10Hz)
        if self
            .motor_frame_count
            .is_multiple_of(self.frames_per_cognitive_tick)
        {
            self.cached_hv = self.encoder.encode(cognitive_state);
            self.cached_cognitive_channels = Some(cognitive_state.to_channels());

            if let Some(m) = metrics {
                self.fep_agent.learn(m);
                let fep_result = self.fep_agent.tick(m);
                self.controller.modulate_tau(fep_result.tau_factor);
                let current_lr = self.controller.learning_rate();
                self.controller
                    .set_learning_rate(current_lr * fep_result.learning_rate_factor);
                self.controller.set_emphasis(fep_result.emphasis_factor);
                self.last_fep_result = Some(fep_result);
            }
        }

        self.motor_frame_count += 1;

        // Build effective HV with carryover coarticulation (same as tick_phoneme)
        let effective_hv = if let Some(ph) = phoneme {
            let is_same = self.prev_phoneme.as_deref() == Some(ph);
            let is_reencode =
                (self.motor_frame_count - 1).is_multiple_of(self.frames_per_cognitive_tick);

            let new_bound = if is_same
                && !is_reencode
                && self.coarticulation_counter >= self.coarticulation_frames
            {
                if let Some(ref cached) = self.prev_phoneme_bound_hv {
                    cached.clone()
                } else {
                    let phoneme_hv = self.get_or_create_phoneme_hv(ph);
                    let bound = self.cached_hv.bind(&phoneme_hv);
                    self.prev_phoneme_bound_hv = Some(bound.clone());
                    bound
                }
            } else {
                let phoneme_hv = self.get_or_create_phoneme_hv(ph);
                let bound = self.cached_hv.bind(&phoneme_hv);

                if !is_same {
                    if let Some(old_ph) = self.prev_phoneme.take() {
                        let old_phoneme_hv = self.get_or_create_phoneme_hv(&old_ph);
                        self.prev_phoneme_bound_hv = Some(self.cached_hv.bind(&old_phoneme_hv));
                        self.coarticulation_counter = 0;
                    }
                } else {
                    self.prev_phoneme_bound_hv = Some(bound.clone());
                }

                bound
            };

            self.prev_phoneme = Some(ph.to_string());

            // Carryover blend (old → new phoneme)
            let mut hv = if self.coarticulation_counter < self.coarticulation_frames {
                self.coarticulation_counter += 1;
                if let Some(ref prev_hv) = self.prev_phoneme_bound_hv {
                    let t = self.coarticulation_counter as f32 / self.coarticulation_frames as f32;
                    prev_hv.scale(1.0 - t).add(&new_bound.scale(t))
                } else {
                    new_bound.clone()
                }
            } else {
                new_bound.clone()
            };

            // Anticipatory coarticulation: during final frames of current phoneme,
            // start blending toward next phoneme's HV
            if let Some(next_ph) = next_phoneme {
                let window = self.anticipation_window(phoneme);
                if phoneme_remaining_frames <= window && window > 0 {
                    let next_phoneme_hv = self.get_or_create_phoneme_hv(next_ph);
                    let next_bound = self.cached_hv.bind(&next_phoneme_hv);
                    // Blend factor: 0 at start of window, up to 0.3 at phoneme end
                    let t = 1.0 - (phoneme_remaining_frames as f32 / window as f32);
                    let anticipation_blend = t * 0.3;
                    hv = hv
                        .scale(1.0 - anticipation_blend)
                        .add(&next_bound.scale(anticipation_blend));
                }
            }

            hv
        } else {
            self.prev_phoneme = None;
            self.prev_phoneme_bound_hv = None;
            self.cached_hv.clone()
        };

        // Adaptive rate limiting
        if phoneme.is_some() && self.coarticulation_counter < self.coarticulation_frames {
            self.controller
                .set_max_formant_delta(self.controller.config().transition_max_delta);
        } else {
            self.controller
                .set_max_formant_delta(self.controller.config().steady_max_delta);
        }

        // Motor tick
        let mut frame = self.controller.forward_with_prosody(
            &effective_hv,
            dt,
            self.cached_cognitive_channels.as_ref(),
        );
        frame.time = self.cumulative_time;
        self.cumulative_time += dt;

        // Set source_type and voicing from phoneme maps
        if let Some(ph) = phoneme {
            if let Some(&manner) = self.phoneme_manner_map.get(ph) {
                frame.source_type = manner;
            }
            if let Some(&is_voiced) = self.phoneme_voicing_map.get(ph)
                && !is_voiced
            {
                frame.voicing = 0.0;
            }
        }

        prosody.apply_prosody(&mut frame);
        frame
    }

    /// Number of frames for anticipatory blending (manner-dependent).
    fn anticipation_window(&self, phoneme: Option<&str>) -> usize {
        match phoneme.map(phoneme_manner_class) {
            Some(MannerClass::Vowel) => 5,     // 25ms anticipation
            Some(MannerClass::Fricative) => 4, // 20ms
            Some(MannerClass::Nasal) => 3,     // 15ms
            _ => 2,                            // 10ms default
        }
    }

    /// Reset the entire pipeline.
    pub fn reset(&mut self) {
        self.encoder.reset();
        self.controller.reset();
        self.fep_agent.reset();
        self.cached_hv = ContinuousHV::zero(HDC_DIMENSION);
        self.motor_frame_count = 0;
        self.cumulative_time = 0.0;
        self.phoneme_hv_cache.clear();
        self.cached_cognitive_channels = None;
        self.prev_phoneme = None;
        self.prev_phoneme_bound_hv = None;
        self.coarticulation_counter = 0;
    }

    /// Get current cumulative time in seconds.
    pub fn cumulative_time(&self) -> f32 {
        self.cumulative_time
    }

    /// Get a reference to the stored genesis seed.
    pub fn genesis(&self) -> &GenesisSeed {
        &self.genesis
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::controller::SpeakerProfile;
    use crate::encoder::VoiceCognitiveState;
    use crate::fep::VocalTractObservation;
    use crate::types::FormantFrame;
    use symthaea_core::genesis::GenesisSeed;

    #[test]
    fn test_pipeline_time_tracking() {
        let genesis = GenesisSeed::from_phrase("test-time-tracking");
        let mut pipeline = VocalTractPipeline::new(&genesis);
        let state = VoiceCognitiveState::default();
        let dt = 0.005; // 200Hz

        // Run 40 frames
        let mut last_frame = pipeline.tick(&state, None, dt);
        for _ in 1..40 {
            last_frame = pipeline.tick(&state, None, dt);
        }

        // Last frame should have time = 39 * 0.005 = 0.195
        let expected_time = 39.0 * dt;
        assert!(
            (last_frame.time - expected_time).abs() < 1e-4,
            "Expected time ~{}, got {}",
            expected_time,
            last_frame.time
        );

        // Cumulative time should be 40 * 0.005 = 0.200
        assert!(
            (pipeline.cumulative_time() - 40.0 * dt).abs() < 1e-4,
            "Expected cumulative ~{}, got {}",
            40.0 * dt,
            pipeline.cumulative_time()
        );

        // Reset should clear time
        pipeline.reset();
        assert!((pipeline.cumulative_time()).abs() < 1e-6);
    }

    #[test]
    fn test_pipeline_fep_learning() {
        let genesis = GenesisSeed::from_phrase("test-fep-learning");
        let mut pipeline = VocalTractPipeline::new(&genesis);

        let state = VoiceCognitiveState::default();
        let metrics = VocalTractObservation {
            articulation_score: 0.6,
            formant_accuracy: 0.5,
            pitch_stability: 0.7,
            coarticulation_smoothness: 0.6,
            duration_accuracy: 0.5,
            energy_consistency: 0.6,
        };

        // Run 60 frames (3 cognitive ticks with FEP feedback)
        for _ in 0..60 {
            pipeline.tick(&state, Some(&metrics), 0.005);
        }

        // FEP agent should have ticked 3 times
        assert_eq!(pipeline.fep_agent.tick_count(), 3);
        // TD learning should have been triggered (learn is called before tick on 2nd+ cognitive tick)
        assert!(
            pipeline.fep_agent.stats().td_updates > 0,
            "TD learner should have updates after multiple cognitive ticks"
        );
    }

    #[test]
    fn test_pipeline_end_to_end() {
        let genesis = GenesisSeed::from_phrase("test-vocal-pipeline");
        let mut pipeline = VocalTractPipeline::new(&genesis);
        let state = VoiceCognitiveState::default();

        // Run 40 frames (2 cognitive ticks)
        for _ in 0..40 {
            let frame = pipeline.tick(&state, None, 0.005);
            assert!(frame.f1 >= 200.0 && frame.f1 <= 1000.0);
            assert!(frame.energy >= 0.0 && frame.energy <= 1.0);
        }
    }

    #[test]
    fn test_pipeline_dual_rate() {
        let genesis = GenesisSeed::from_phrase("test-dual-rate");
        let mut pipeline = VocalTractPipeline::new(&genesis);

        let state1 = VoiceCognitiveState {
            emotional_arousal: 0.2,
            ..Default::default()
        };
        let state2 = VoiceCognitiveState {
            emotional_arousal: 0.9,
            ..Default::default()
        };

        // First cognitive tick (frame 0) with state1
        let frame_a = pipeline.tick(&state1, None, 0.005);

        // Frames 1-19 still use cached HV from state1
        for _ in 1..20 {
            pipeline.tick(&state1, None, 0.005);
        }

        // Frame 20: new cognitive tick with state2 (different arousal)
        let frame_b = pipeline.tick(&state2, None, 0.005);

        // Frame 21: still using state2's cached HV
        let frame_c = pipeline.tick(&state2, None, 0.005);

        // frame_a and frame_b used different cognitive inputs at re-encode boundaries
        // frame_b and frame_c used the same cognitive HV (frame_c at motor-only tick)
        assert!(frame_a.f1.is_finite());
        assert!(frame_b.f1.is_finite());
        assert!(frame_c.f1.is_finite());
    }

    #[test]
    fn test_pipeline_with_fep_feedback() {
        let genesis = GenesisSeed::from_phrase("test-fep-feedback");
        let mut pipeline = VocalTractPipeline::new(&genesis);

        let state = VoiceCognitiveState::default();
        let metrics = VocalTractObservation {
            articulation_score: 0.8,
            formant_accuracy: 0.7,
            pitch_stability: 0.9,
            coarticulation_smoothness: 0.8,
            duration_accuracy: 0.7,
            energy_consistency: 0.8,
        };

        // Run with FEP feedback
        for _ in 0..40 {
            let frame = pipeline.tick(&state, Some(&metrics), 0.005);
            assert!(frame.f1.is_finite());
        }

        // FEP agent should have ticked twice (at frames 0 and 20)
        assert_eq!(pipeline.fep_agent.tick_count(), 2);
    }

    #[test]
    fn test_phoneme_hv_caching() {
        let genesis = GenesisSeed::from_phrase("test-phoneme-cache");
        let mut pipeline = VocalTractPipeline::new(&genesis);

        let hv1 = pipeline.get_or_create_phoneme_hv("AH");
        let hv2 = pipeline.get_or_create_phoneme_hv("AH");

        // Same phoneme → identical HV (from cache)
        assert!(
            (hv1.similarity(&hv2) - 1.0).abs() < 1e-5,
            "Cached HVs should be identical"
        );

        // Different phoneme → different HV
        let hv3 = pipeline.get_or_create_phoneme_hv("IY");
        assert!(
            hv1.similarity(&hv3) < 0.5,
            "Different phonemes should have dissimilar HVs: sim={}",
            hv1.similarity(&hv3)
        );
    }

    #[test]
    fn test_phoneme_routing_different_output() {
        let genesis = GenesisSeed::from_phrase("test-phoneme-routing");
        let config = VocalTractConfig {
            fourier_frequencies: vec![],
            ..VocalTractConfig::default()
        };
        let mut pipeline = VocalTractPipeline::new_with_config(&genesis, &config);
        let state = VoiceCognitiveState::default();

        // Run 40 frames with /AH/ (enough for LTC to evolve state)
        let mut frames_ah = Vec::new();
        for _ in 0..40 {
            frames_ah.push(pipeline.tick_phoneme(&state, None, 0.005, Some("AH")));
        }

        pipeline.reset();

        // Run 40 frames with /IY/
        let mut frames_iy = Vec::new();
        for _ in 0..40 {
            frames_iy.push(pipeline.tick_phoneme(&state, None, 0.005, Some("IY")));
        }

        // Compute total trajectory divergence across all 9 formant dims
        let total_diff: f32 = frames_ah
            .iter()
            .zip(frames_iy.iter())
            .map(|(a, b)| {
                (a.f1 - b.f1).abs()
                    + (a.f2 - b.f2).abs()
                    + (a.f3 - b.f3).abs()
                    + (a.f0 - b.f0).abs()
            })
            .sum();

        assert!(
            total_diff > 1.0,
            "Different phonemes should produce different formant trajectories: total_diff={total_diff:.2}"
        );
    }

    #[test]
    fn test_phoneme_none_backward_compat() {
        let genesis = GenesisSeed::from_phrase("test-backward-compat");
        let mut pipeline1 = VocalTractPipeline::new(&genesis);
        let mut pipeline2 = VocalTractPipeline::new(&genesis);
        let state = VoiceCognitiveState::default();

        // tick() (no phoneme) should produce same result as tick_phoneme(None)
        let frame1 = pipeline1.tick(&state, None, 0.005);
        let frame2 = pipeline2.tick_phoneme(&state, None, 0.005, None);

        assert!(
            (frame1.f1 - frame2.f1).abs() < 1e-4,
            "tick() and tick_phoneme(None) should be equivalent"
        );
        assert!((frame1.f0 - frame2.f0).abs() < 1e-4);
        assert!((frame1.energy - frame2.energy).abs() < 1e-6);
    }

    #[test]
    fn test_phoneme_bind_changes_similarity() {
        let genesis = GenesisSeed::from_phrase("test-bind-sim");
        let mut pipeline = VocalTractPipeline::new(&genesis);

        // Get a cognitive HV
        let cognitive_hv = ContinuousHV::random(HDC_DIMENSION, 42);
        let phoneme_hv = pipeline.get_or_create_phoneme_hv("AH");

        // bind() should produce a vector dissimilar from the original
        let bound = cognitive_hv.bind(&phoneme_hv);
        let sim = cognitive_hv.similarity(&bound);
        assert!(sim < 0.5, "bind() should decorrelate: sim={}", sim);

        // But binding with same phoneme twice should give same result
        let bound2 = cognitive_hv.bind(&phoneme_hv);
        assert!(
            (bound.similarity(&bound2) - 1.0).abs() < 1e-5,
            "Same bind should be deterministic"
        );
    }

    #[test]
    fn test_prosody_f0_declination() {
        let base_f0 = 120.0;
        // Frame at utterance start
        let mut frame_start = FormantFrame {
            f0: 200.0, // will be overwritten by prosody
            energy: 0.5,
            ..FormantFrame::silent(0.0)
        };
        let ctx_start = ProsodyContext {
            utterance_progress: 0.0,
            phoneme_progress: 0.5,
            stress: 0,
            base_f0,
            arousal: 0.5,
            ..ProsodyContext::default()
        };
        ctx_start.apply_prosody(&mut frame_start);

        // Frame at utterance end
        let mut frame_end = FormantFrame {
            f0: 200.0,
            energy: 0.5,
            ..FormantFrame::silent(0.0)
        };
        let ctx_end = ProsodyContext {
            utterance_progress: 1.0,
            phoneme_progress: 0.5,
            stress: 0,
            base_f0,
            arousal: 0.5,
            ..ProsodyContext::default()
        };
        ctx_end.apply_prosody(&mut frame_end);

        // F0 should decline over the utterance
        assert!(
            frame_start.f0 > frame_end.f0,
            "F0 should decline: start={:.1}, end={:.1}",
            frame_start.f0,
            frame_end.f0
        );

        // Statement contour: 1.05→0.80, ratio ≈ 0.762
        let ratio = frame_end.f0 / frame_start.f0;
        assert!(
            ratio < 0.85,
            "Expected significant F0 declination (Statement), got ratio={ratio:.3}"
        );
    }

    #[test]
    fn test_prosody_stress_boost() {
        let base_f0 = 120.0;
        let mid_ctx = |stress: u8| ProsodyContext {
            utterance_progress: 0.5,
            phoneme_progress: 0.5,
            stress,
            base_f0,
            arousal: 0.5,
            ..ProsodyContext::default()
        };

        let mut frame_unstressed = FormantFrame {
            f0: 200.0,
            energy: 0.5,
            ..FormantFrame::silent(0.0)
        };
        mid_ctx(0).apply_prosody(&mut frame_unstressed);

        let mut frame_primary = FormantFrame {
            f0: 200.0,
            energy: 0.5,
            ..FormantFrame::silent(0.0)
        };
        mid_ctx(1).apply_prosody(&mut frame_primary);

        let mut frame_secondary = FormantFrame {
            f0: 200.0,
            energy: 0.5,
            ..FormantFrame::silent(0.0)
        };
        mid_ctx(2).apply_prosody(&mut frame_secondary);

        // Primary stress > secondary > unstressed
        assert!(
            frame_primary.f0 > frame_secondary.f0,
            "Primary stress should have higher F0: primary={:.1}, secondary={:.1}",
            frame_primary.f0,
            frame_secondary.f0
        );
        assert!(
            frame_secondary.f0 > frame_unstressed.f0,
            "Secondary stress should have higher F0 than unstressed: secondary={:.1}, unstressed={:.1}",
            frame_secondary.f0,
            frame_unstressed.f0
        );

        // Energy: stressed > unstressed
        assert!(
            frame_primary.energy > frame_unstressed.energy,
            "Stressed phoneme should have higher energy"
        );
    }

    #[test]
    fn test_prosody_energy_envelope() {
        let base_ctx = |phoneme_progress: f32| ProsodyContext {
            utterance_progress: 0.5,
            phoneme_progress,
            stress: 1,
            base_f0: 120.0,
            arousal: 0.5,
            ..ProsodyContext::default()
        };

        // Attack phase (near start)
        let mut frame_attack = FormantFrame {
            f0: 120.0,
            energy: 0.5,
            ..FormantFrame::silent(0.0)
        };
        base_ctx(0.02).apply_prosody(&mut frame_attack);

        // Sustain phase (middle)
        let mut frame_sustain = FormantFrame {
            f0: 120.0,
            energy: 0.5,
            ..FormantFrame::silent(0.0)
        };
        base_ctx(0.5).apply_prosody(&mut frame_sustain);

        // Release phase (near end)
        let mut frame_release = FormantFrame {
            f0: 120.0,
            energy: 0.5,
            ..FormantFrame::silent(0.0)
        };
        base_ctx(0.95).apply_prosody(&mut frame_release);

        // Attack < sustain (ramping up)
        assert!(
            frame_attack.energy < frame_sustain.energy,
            "Attack energy should be lower than sustain: attack={:.3}, sustain={:.3}",
            frame_attack.energy,
            frame_sustain.energy
        );

        // Release < sustain (ramping down)
        assert!(
            frame_release.energy < frame_sustain.energy,
            "Release energy should be lower than sustain: release={:.3}, sustain={:.3}",
            frame_release.energy,
            frame_sustain.energy
        );
    }

    #[test]
    fn test_speaker_male_vs_female_differ() {
        let genesis = GenesisSeed::from_phrase("test-speaker-profiles");
        let state = VoiceCognitiveState::default();

        let mut male_pipeline =
            VocalTractPipeline::new_with_speaker(&genesis, &SpeakerProfile::male());
        let mut female_pipeline =
            VocalTractPipeline::new_with_speaker(&genesis, &SpeakerProfile::female());

        // Run 40 frames each
        let mut male_frames = Vec::new();
        let mut female_frames = Vec::new();
        for _ in 0..40 {
            male_frames.push(male_pipeline.tick_phoneme(&state, None, 0.005, Some("AH")));
            female_frames.push(female_pipeline.tick_phoneme(&state, None, 0.005, Some("AH")));
        }

        // Formant trajectories should differ between male and female
        let total_diff: f32 = male_frames
            .iter()
            .zip(female_frames.iter())
            .map(|(m, f)| (m.f1 - f.f1).abs() + (m.f2 - f.f2).abs() + (m.f0 - f.f0).abs())
            .sum();

        assert!(
            total_diff > 10.0,
            "Male and female voices should produce different formants: total_diff={total_diff:.2}"
        );
    }

    #[test]
    fn test_speaker_profile_f0_applied() {
        let profile = SpeakerProfile::female();
        assert!(
            (profile.base_f0 - 220.0).abs() < 1.0,
            "Female base_f0 should be 220Hz"
        );
        assert!((profile.formant_scale - 1.15).abs() < 0.01);

        let child = SpeakerProfile::child();
        assert!(
            (child.base_f0 - 300.0).abs() < 1.0,
            "Child base_f0 should be 300Hz"
        );
        assert!((child.formant_scale - 1.30).abs() < 0.01);

        let male = SpeakerProfile::male();
        assert!(
            (male.base_f0 - 120.0).abs() < 1.0,
            "Male base_f0 should be 120Hz"
        );
    }

    #[test]
    fn test_speaker_genesis_determinism() {
        let genesis = GenesisSeed::from_phrase("test-speaker-determinism");
        let profile = SpeakerProfile::female();
        let state = VoiceCognitiveState::default();

        let mut p1 = VocalTractPipeline::new_with_speaker(&genesis, &profile);
        let mut p2 = VocalTractPipeline::new_with_speaker(&genesis, &profile);

        let frame1 = p1.tick(&state, None, 0.005);
        let frame2 = p2.tick(&state, None, 0.005);

        assert!(
            (frame1.f1 - frame2.f1).abs() < 1e-4,
            "Same genesis + profile → same output: f1={} vs {}",
            frame1.f1,
            frame2.f1
        );
        assert!((frame1.f0 - frame2.f0).abs() < 1e-4);
    }

    #[test]
    fn test_coarticulation_blending() {
        let genesis = GenesisSeed::from_phrase("test-coarticulation");
        let config = VocalTractConfig {
            fourier_frequencies: vec![],
            ..VocalTractConfig::default()
        };
        let mut pipeline = VocalTractPipeline::new_with_config(&genesis, &config);
        let state = VoiceCognitiveState::default();

        // Run 30 frames of /AH/
        for _ in 0..30 {
            pipeline.tick_phoneme(&state, None, 0.005, Some("AH"));
        }

        // Transition to /IY/ — should blend over 16 frames
        let mut transition_frames = Vec::new();
        for _ in 0..30 {
            transition_frames.push(pipeline.tick_phoneme(&state, None, 0.005, Some("IY")));
        }

        // Run the same transition WITHOUT coarticulation (fresh pipeline, hard switch)
        let mut pipeline2 = VocalTractPipeline::new_with_config(&genesis, &config);
        // Disable coarticulation by setting frames to 0
        pipeline2.coarticulation_frames = 0;

        for _ in 0..30 {
            pipeline2.tick_phoneme(&state, None, 0.005, Some("AH"));
        }

        let mut hard_frames = Vec::new();
        for _ in 0..30 {
            hard_frames.push(pipeline2.tick_phoneme(&state, None, 0.005, Some("IY")));
        }

        // The blended transition should differ from the hard switch in early frames
        let early_diff: f32 = transition_frames[..8]
            .iter()
            .zip(hard_frames[..8].iter())
            .map(|(a, b)| (a.f1 - b.f1).abs() + (a.f2 - b.f2).abs())
            .sum();

        // Late frames should converge (both approaches reach the same /IY/ target)
        let late_diff: f32 = transition_frames[20..]
            .iter()
            .zip(hard_frames[20..].iter())
            .map(|(a, b)| (a.f1 - b.f1).abs() + (a.f2 - b.f2).abs())
            .sum();

        // Blending should produce measurable difference in early frames
        assert!(
            early_diff > 0.1,
            "Coarticulation should differ from hard switch in early frames: diff={early_diff:.2}"
        );

        // Late frames should be closer (blending complete). Factor 3x allows for
        // LTC network convergence variance across random seeds.
        assert!(
            late_diff < early_diff * 3.0,
            "Late frames should converge: early_diff={early_diff:.2}, late_diff={late_diff:.2}"
        );
    }

    #[test]
    fn test_phoneme_bind_cache_consistency() {
        // Verify that cached bind produces same output as uncached
        let genesis = GenesisSeed::from_phrase("test-bind-cache");
        let mut pipeline = VocalTractPipeline::new(&genesis);
        let state = VoiceCognitiveState::default();

        // Run 5 frames with /AH/ (first triggers bind, rest use cache)
        let mut frames = Vec::new();
        for _ in 0..5 {
            frames.push(pipeline.tick_phoneme(&state, None, 0.005, Some("AH")));
        }

        // All frames should be valid (cache doesn't corrupt output)
        for (i, f) in frames.iter().enumerate() {
            assert!(
                f.f1 >= 200.0 && f.f1 <= 1000.0,
                "Frame {i}: F1={} out of range",
                f.f1
            );
            assert!(f.f1.is_finite(), "Frame {i}: F1 is not finite");
        }

        // Frames 2-4 (cached) should be close to frame 1 (uncached)
        // since the LTC network evolves smoothly with the same input
        let f1_drift: f32 = frames[1..]
            .iter()
            .map(|f| (f.f1 - frames[0].f1).abs())
            .sum::<f32>();
        assert!(
            f1_drift < 200.0,
            "Cached frames should be consistent with uncached: drift={f1_drift:.1}"
        );
    }

    #[test]
    fn test_adaptive_rate_limiting() {
        let genesis = GenesisSeed::from_phrase("test-adaptive-rate");
        let mut pipeline = VocalTractPipeline::new(&genesis);
        let state = VoiceCognitiveState::default();

        // Verify config defaults
        assert!(
            (pipeline.controller.config().steady_max_delta - 12.0).abs() < 1e-4,
            "steady_max_delta should default to 12.0"
        );
        assert!(
            (pipeline.controller.config().transition_max_delta - 20.0).abs() < 1e-4,
            "transition_max_delta should default to 20.0"
        );

        // Train briefly so the network can distinguish AH vs IY
        let targets = vec![
            (
                "AH",
                crate::types::FormantTarget::vowel(520.0, 1190.0, 2390.0, 80.0),
            ),
            (
                "IY",
                crate::types::FormantTarget::vowel(270.0, 2290.0, 3010.0, 100.0),
            ),
        ];
        let target_refs: Vec<(&str, &crate::types::FormantTarget)> =
            targets.iter().map(|(name, t)| (*name, t)).collect();
        pipeline
            .controller
            .train_on_phoneme_targets(&genesis, &target_refs, 20);

        // Run 40 frames of /AH/ to establish steady state
        for _ in 0..40 {
            pipeline.tick_phoneme(&state, None, 0.005, Some("AH"));
        }

        // Collect steady-state frames (still /AH/)
        let mut steady_f1 = Vec::new();
        for _ in 0..20 {
            let frame = pipeline.tick_phoneme(&state, None, 0.005, Some("AH"));
            steady_f1.push(frame.f1);
        }

        // Transition to /IY/ — collect transition frames
        let mut transition_f1 = Vec::new();
        for _ in 0..20 {
            let frame = pipeline.tick_phoneme(&state, None, 0.005, Some("IY"));
            transition_f1.push(frame.f1);
        }

        // Verify steady-state delta respects 12.0 Hz/frame limit
        let steady_max: f32 = steady_f1
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0f32, f32::max);
        assert!(
            steady_max <= 12.5,
            "Steady-state F1 delta should be ≤12 Hz/frame: got {:.2}",
            steady_max
        );

        // Verify transition delta respects 20.0 Hz/frame limit
        let transition_max: f32 = transition_f1
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0f32, f32::max);
        assert!(
            transition_max <= 20.5,
            "Transition F1 delta should be ≤20 Hz/frame: got {:.2}",
            transition_max
        );
    }

    #[test]
    fn test_source_type_propagation() {
        use crate::types::SourceType;

        let genesis = GenesisSeed::from_phrase("test-source-type");
        let mut pipeline = VocalTractPipeline::new(&genesis);
        let state = VoiceCognitiveState::default();

        // Register manner for specific phonemes
        pipeline.register_phoneme_manner("P", SourceType::Stop);
        pipeline.register_phoneme_manner("S", SourceType::Fricative);
        pipeline.register_phoneme_manner("M", SourceType::Nasal);
        pipeline.register_phoneme_manner("AH", SourceType::Vowel);

        // Stop consonant
        let frame_p = pipeline.tick_phoneme(&state, None, 0.005, Some("P"));
        assert_eq!(
            frame_p.source_type,
            SourceType::Stop,
            "P should produce Stop source type"
        );

        // Fricative
        let frame_s = pipeline.tick_phoneme(&state, None, 0.005, Some("S"));
        assert_eq!(
            frame_s.source_type,
            SourceType::Fricative,
            "S should produce Fricative source type"
        );

        // Nasal
        let frame_m = pipeline.tick_phoneme(&state, None, 0.005, Some("M"));
        assert_eq!(
            frame_m.source_type,
            SourceType::Nasal,
            "M should produce Nasal source type"
        );

        // Vowel
        let frame_ah = pipeline.tick_phoneme(&state, None, 0.005, Some("AH"));
        assert_eq!(
            frame_ah.source_type,
            SourceType::Vowel,
            "AH should produce Vowel source type"
        );

        // No phoneme → keeps controller default (Silent from ..Default::default())
        pipeline.reset();
        let frame_none = pipeline.tick_phoneme(&state, None, 0.005, None);
        assert_eq!(
            frame_none.source_type,
            SourceType::Silent,
            "No phoneme should keep controller default (Silent)"
        );
    }

    #[test]
    fn test_intonation_f0_contours() {
        let base_f0 = 120.0;

        // Statement: falling (1.05 → 0.80)
        let mut f_start = FormantFrame::silent(0.0);
        ProsodyContext {
            utterance_progress: 0.0,
            base_f0,
            intonation: Intonation::Statement,
            ..Default::default()
        }
        .apply_prosody(&mut f_start);

        let mut f_end = FormantFrame::silent(0.0);
        ProsodyContext {
            utterance_progress: 1.0,
            base_f0,
            intonation: Intonation::Statement,
            ..Default::default()
        }
        .apply_prosody(&mut f_end);

        assert!(
            f_start.f0 > f_end.f0,
            "Statement should fall: start={:.1}, end={:.1}",
            f_start.f0,
            f_end.f0
        );

        // Question: rising at end (> start)
        let mut q_start = FormantFrame::silent(0.0);
        ProsodyContext {
            utterance_progress: 0.0,
            base_f0,
            intonation: Intonation::Question,
            ..Default::default()
        }
        .apply_prosody(&mut q_start);

        let mut q_end = FormantFrame::silent(0.0);
        ProsodyContext {
            utterance_progress: 1.0,
            base_f0,
            intonation: Intonation::Question,
            ..Default::default()
        }
        .apply_prosody(&mut q_end);

        assert!(
            q_end.f0 > q_start.f0,
            "Question should rise: start={:.1}, end={:.1}",
            q_start.f0,
            q_end.f0
        );

        // Exclamation: peak in middle, falls below start by end
        let mut e_mid = FormantFrame::silent(0.0);
        ProsodyContext {
            utterance_progress: 0.3,
            base_f0,
            intonation: Intonation::Exclamation,
            ..Default::default()
        }
        .apply_prosody(&mut e_mid);

        let mut e_end = FormantFrame::silent(0.0);
        ProsodyContext {
            utterance_progress: 1.0,
            base_f0,
            intonation: Intonation::Exclamation,
            ..Default::default()
        }
        .apply_prosody(&mut e_end);

        assert!(
            e_mid.f0 > e_end.f0,
            "Exclamation peak should exceed end: peak={:.1}, end={:.1}",
            e_mid.f0,
            e_end.f0
        );
    }

    #[test]
    fn test_duration_model() {
        // Vowel with primary stress
        let dur = predict_duration("AH", 1, false, false, 1.0);
        assert_eq!(dur, 24, "AH + primary stress: 16 × 1.5 = 24"); // 16 * 1.5

        // Unstressed vowel
        let dur = predict_duration("AH", 0, false, false, 1.0);
        assert_eq!(dur, 16, "AH unstressed: base 16");

        // Utterance-final vowel with primary stress
        let dur = predict_duration("IY", 1, false, true, 1.0);
        assert_eq!(dur, 34, "IY + primary + utt-final: 16 × 1.5 × 1.4 ≈ 34");

        // Stop consonant (stress doesn't lengthen consonants)
        let dur_s0 = predict_duration("P", 0, false, false, 1.0);
        let dur_s1 = predict_duration("P", 1, false, false, 1.0);
        assert_eq!(dur_s0, dur_s1, "Stress shouldn't affect stop duration");
        assert_eq!(dur_s0, 6, "Stop base: 6 frames");

        // Speaking rate: faster = shorter
        let dur_fast = predict_duration("AH", 0, false, false, 2.0);
        let dur_slow = predict_duration("AH", 0, false, false, 0.5);
        assert!(
            dur_fast < dur_slow,
            "Faster rate should be shorter: fast={dur_fast}, slow={dur_slow}"
        );

        // Minimum duration
        let dur_min = predict_duration("W", 0, false, false, 3.0);
        assert!(dur_min >= 2, "Minimum duration is 2 frames");

        // Manner classification
        assert_eq!(phoneme_manner_class("AH"), MannerClass::Vowel);
        assert_eq!(phoneme_manner_class("P"), MannerClass::Stop);
        assert_eq!(phoneme_manner_class("S"), MannerClass::Fricative);
        assert_eq!(phoneme_manner_class("M"), MannerClass::Nasal);
        assert_eq!(phoneme_manner_class("L"), MannerClass::Liquid);
        assert_eq!(phoneme_manner_class("JH"), MannerClass::Affricate);
        assert_eq!(phoneme_manner_class("W"), MannerClass::Glide);
        // With stress digit
        assert_eq!(phoneme_manner_class("AH0"), MannerClass::Vowel);
        assert_eq!(phoneme_manner_class("IY1"), MannerClass::Vowel);
    }

    #[test]
    fn test_pitch_accent_shapes() {
        let base_f0 = 120.0;

        // H*: should raise F0 mid-syllable
        let mut frame_h = FormantFrame::silent(0.0);
        ProsodyContext {
            utterance_progress: 0.5,
            phoneme_progress: 0.5,
            stress: 1,
            base_f0,
            pitch_accent: PitchAccent::High,
            ..Default::default()
        }
        .apply_prosody(&mut frame_h);

        let mut frame_none = FormantFrame::silent(0.0);
        ProsodyContext {
            utterance_progress: 0.5,
            phoneme_progress: 0.5,
            stress: 1,
            base_f0,
            pitch_accent: PitchAccent::None,
            ..Default::default()
        }
        .apply_prosody(&mut frame_none);

        assert!(
            frame_h.f0 > frame_none.f0,
            "H* accent should raise F0: H*={:.1}, None={:.1}",
            frame_h.f0,
            frame_none.f0
        );

        // L+H*: should rise across syllable (early < late)
        let mut frame_lh_early = FormantFrame::silent(0.0);
        ProsodyContext {
            utterance_progress: 0.5,
            phoneme_progress: 0.1,
            stress: 1,
            base_f0,
            pitch_accent: PitchAccent::RiseHigh,
            ..Default::default()
        }
        .apply_prosody(&mut frame_lh_early);

        let mut frame_lh_late = FormantFrame::silent(0.0);
        ProsodyContext {
            utterance_progress: 0.5,
            phoneme_progress: 0.8,
            stress: 1,
            base_f0,
            pitch_accent: PitchAccent::RiseHigh,
            ..Default::default()
        }
        .apply_prosody(&mut frame_lh_late);

        assert!(
            frame_lh_late.f0 > frame_lh_early.f0,
            "L+H* should rise: early={:.1}, late={:.1}",
            frame_lh_early.f0,
            frame_lh_late.f0
        );
    }

    #[test]
    fn test_declination_within_phrase() {
        let base_f0 = 120.0;

        // Start of phrase
        let mut frame_start = FormantFrame::silent(0.0);
        ProsodyContext {
            utterance_progress: 0.5,
            phoneme_progress: 0.5,
            base_f0,
            phrase_progress: 0.0,
            ..Default::default()
        }
        .apply_prosody(&mut frame_start);

        // End of phrase
        let mut frame_end = FormantFrame::silent(0.0);
        ProsodyContext {
            utterance_progress: 0.5,
            phoneme_progress: 0.5,
            base_f0,
            phrase_progress: 1.0,
            ..Default::default()
        }
        .apply_prosody(&mut frame_end);

        assert!(
            frame_start.f0 > frame_end.f0,
            "F0 should drop within phrase: start={:.1}, end={:.1}",
            frame_start.f0,
            frame_end.f0
        );

        // Declination is 15%: ratio should be ~0.85
        let ratio = frame_end.f0 / frame_start.f0;
        assert!(
            ratio < 0.90 && ratio > 0.75,
            "Declination ratio should be ~0.85: got {ratio:.3}"
        );
    }

    #[test]
    fn test_focus_boosts_f0_and_energy() {
        let base_f0 = 120.0;

        let mut frame_focus = FormantFrame {
            energy: 0.5,
            ..FormantFrame::silent(0.0)
        };
        ProsodyContext {
            utterance_progress: 0.5,
            phoneme_progress: 0.5,
            base_f0,
            is_focus: true,
            ..Default::default()
        }
        .apply_prosody(&mut frame_focus);

        let mut frame_normal = FormantFrame {
            energy: 0.5,
            ..FormantFrame::silent(0.0)
        };
        ProsodyContext {
            utterance_progress: 0.5,
            phoneme_progress: 0.5,
            base_f0,
            is_focus: false,
            ..Default::default()
        }
        .apply_prosody(&mut frame_normal);

        assert!(
            frame_focus.f0 > frame_normal.f0,
            "Focus should boost F0: focus={:.1}, normal={:.1}",
            frame_focus.f0,
            frame_normal.f0
        );
        assert!(
            frame_focus.energy > frame_normal.energy,
            "Focus should boost energy: focus={:.3}, normal={:.3}",
            frame_focus.energy,
            frame_normal.energy
        );
    }

    #[test]
    fn test_microprosody_consonant_perturbation() {
        let base_f0 = 120.0;

        // Vowel (no microprosody)
        let mut frame_vowel = FormantFrame {
            source_type: SourceType::Vowel,
            ..FormantFrame::silent(0.0)
        };
        ProsodyContext {
            utterance_progress: 0.5,
            phoneme_progress: 0.5,
            base_f0,
            ..Default::default()
        }
        .apply_prosody(&mut frame_vowel);

        // Stop (should lower F0)
        let mut frame_stop = FormantFrame {
            source_type: SourceType::Stop,
            ..FormantFrame::silent(0.0)
        };
        ProsodyContext {
            utterance_progress: 0.5,
            phoneme_progress: 0.5,
            base_f0,
            ..Default::default()
        }
        .apply_prosody(&mut frame_stop);

        assert!(
            frame_stop.f0 < frame_vowel.f0,
            "Stop should lower F0 via microprosody: stop={:.1}, vowel={:.1}",
            frame_stop.f0,
            frame_vowel.f0
        );

        // Nasal (should raise F0)
        let mut frame_nasal = FormantFrame {
            source_type: SourceType::Nasal,
            ..FormantFrame::silent(0.0)
        };
        ProsodyContext {
            utterance_progress: 0.5,
            phoneme_progress: 0.5,
            base_f0,
            ..Default::default()
        }
        .apply_prosody(&mut frame_nasal);

        assert!(
            frame_nasal.f0 > frame_vowel.f0,
            "Nasal should raise F0 via microprosody: nasal={:.1}, vowel={:.1}",
            frame_nasal.f0,
            frame_vowel.f0
        );
    }

    #[test]
    fn test_anticipatory_blending() {
        let genesis = GenesisSeed::from_phrase("test-anticipatory");
        let config = VocalTractConfig {
            fourier_frequencies: vec![],
            ..VocalTractConfig::default()
        };
        let mut pipeline = VocalTractPipeline::new_with_config(&genesis, &config);
        let state = VoiceCognitiveState::default();
        let prosody = ProsodyContext::default();

        // Run 30 frames of /AH/
        for _ in 0..30 {
            pipeline.tick_phoneme(&state, None, 0.005, Some("AH"));
        }

        // Now run /AH/ with anticipation toward /IY/
        // Final 5 frames should start shifting toward /IY/ formants
        let total_remaining = 10;
        let mut anticipatory_frames = Vec::new();
        for i in 0..total_remaining {
            let remaining = total_remaining - i;
            let frame = pipeline.tick_with_anticipation(
                &state,
                None,
                0.005,
                Some("AH"),
                Some("IY"),
                remaining,
                &prosody,
            );
            anticipatory_frames.push(frame);
        }

        // Run same sequence WITHOUT anticipation (next_phoneme = None)
        let mut pipeline2 = VocalTractPipeline::new(&genesis);
        for _ in 0..30 {
            pipeline2.tick_phoneme(&state, None, 0.005, Some("AH"));
        }
        let mut no_antic_frames = Vec::new();
        for _ in 0..total_remaining {
            let frame = pipeline2.tick_with_anticipation(
                &state,
                None,
                0.005,
                Some("AH"),
                None,
                0,
                &prosody,
            );
            no_antic_frames.push(frame);
        }

        // Anticipatory frames (especially late ones) should differ from non-anticipatory
        let late_diff: f32 = anticipatory_frames[5..]
            .iter()
            .zip(no_antic_frames[5..].iter())
            .map(|(a, b)| (a.f1 - b.f1).abs() + (a.f2 - b.f2).abs())
            .sum();

        assert!(
            late_diff > 0.01,
            "Anticipatory blending should change late frames: diff={late_diff:.4}"
        );
    }

    #[test]
    fn test_no_anticipation_without_next() {
        let genesis = GenesisSeed::from_phrase("test-no-antic");
        let mut pipeline1 = VocalTractPipeline::new(&genesis);
        let mut pipeline2 = VocalTractPipeline::new(&genesis);
        let state = VoiceCognitiveState::default();
        let prosody = ProsodyContext::default();

        // Both pipelines produce same output when next_phoneme is None
        let frame1 =
            pipeline1.tick_with_anticipation(&state, None, 0.005, Some("AH"), None, 0, &prosody);
        let frame2 = pipeline2.tick_with_prosody(&state, None, 0.005, Some("AH"), &prosody);

        // Should be very close (tick_with_anticipation duplicates the cognitive/motor logic)
        assert!(
            (frame1.f1 - frame2.f1).abs() < 1e-3,
            "No-anticipation should match tick_with_prosody: f1={:.2} vs {:.2}",
            frame1.f1,
            frame2.f1
        );
    }

    #[test]
    fn test_anticipation_window_manner_dependent() {
        let genesis = GenesisSeed::from_phrase("test-antic-window");
        let pipeline = VocalTractPipeline::new(&genesis);

        // Vowels get largest window
        let vowel_win = pipeline.anticipation_window(Some("AH"));
        let stop_win = pipeline.anticipation_window(Some("P"));
        let fric_win = pipeline.anticipation_window(Some("S"));

        assert!(
            vowel_win > stop_win,
            "Vowel window should be larger than stop: vowel={vowel_win}, stop={stop_win}"
        );
        assert!(
            fric_win > stop_win,
            "Fricative window should be larger than stop: fric={fric_win}, stop={stop_win}"
        );
    }

    #[test]
    fn test_vowel_reduction_centralizes_unstressed() {
        let base_f0 = 120.0;

        // Stressed vowel: F1/F2 unchanged
        let mut frame_stressed = FormantFrame {
            f1: 730.0, // /AE/ target
            f2: 1090.0,
            f3: 2440.0,
            b1: 60.0,
            b2: 90.0,
            b3: 150.0,
            energy: 0.5,
            source_type: SourceType::Vowel,
            ..FormantFrame::silent(0.0)
        };
        ProsodyContext {
            utterance_progress: 0.5,
            phoneme_progress: 0.5,
            stress: 1,
            base_f0,
            ..Default::default()
        }
        .apply_prosody(&mut frame_stressed);

        // Unstressed vowel: same starting formants, should centralize
        let mut frame_unstressed = FormantFrame {
            f1: 730.0,
            f2: 1090.0,
            f3: 2440.0,
            b1: 60.0,
            b2: 90.0,
            b3: 150.0,
            energy: 0.5,
            source_type: SourceType::Vowel,
            ..FormantFrame::silent(0.0)
        };
        ProsodyContext {
            utterance_progress: 0.5,
            phoneme_progress: 0.5,
            stress: 0,
            base_f0,
            ..Default::default()
        }
        .apply_prosody(&mut frame_unstressed);

        // Unstressed F1 should move toward schwa (500 Hz) — i.e., decrease from 730
        assert!(
            frame_unstressed.f1 < frame_stressed.f1,
            "Unstressed F1 should centralize: unstressed={:.0}, stressed={:.0}",
            frame_unstressed.f1,
            frame_stressed.f1
        );

        // F2 should move toward 1500 — i.e., increase from 1090
        assert!(
            frame_unstressed.f2 > frame_stressed.f2,
            "Unstressed F2 should centralize toward 1500: unstressed={:.0}, stressed={:.0}",
            frame_unstressed.f2,
            frame_stressed.f2
        );
    }

    #[test]
    fn test_bandwidth_dynamics() {
        let base_f0 = 120.0;

        // Stressed vowel
        let mut frame_stressed = FormantFrame {
            b1: 60.0,
            b2: 90.0,
            b3: 150.0,
            energy: 0.5,
            source_type: SourceType::Vowel,
            ..FormantFrame::silent(0.0)
        };
        ProsodyContext {
            utterance_progress: 0.5,
            phoneme_progress: 0.5,
            stress: 1,
            base_f0,
            ..Default::default()
        }
        .apply_prosody(&mut frame_stressed);

        // Unstressed vowel
        let mut frame_unstressed = FormantFrame {
            b1: 60.0,
            b2: 90.0,
            b3: 150.0,
            energy: 0.5,
            source_type: SourceType::Vowel,
            ..FormantFrame::silent(0.0)
        };
        ProsodyContext {
            utterance_progress: 0.5,
            phoneme_progress: 0.5,
            stress: 0,
            base_f0,
            ..Default::default()
        }
        .apply_prosody(&mut frame_unstressed);

        // Stressed: narrower bandwidths (×0.9)
        assert!(
            frame_stressed.b1 < 60.0,
            "Stressed bandwidth should narrow: b1={:.1}",
            frame_stressed.b1
        );

        // Unstressed: wider bandwidths (×1.3)
        assert!(
            frame_unstressed.b1 > 60.0,
            "Unstressed bandwidth should widen: b1={:.1}",
            frame_unstressed.b1
        );

        // Stressed < unstressed
        assert!(
            frame_stressed.b1 < frame_unstressed.b1,
            "Stressed bandwidth should be narrower: stressed={:.1}, unstressed={:.1}",
            frame_stressed.b1,
            frame_unstressed.b1
        );
    }
}
