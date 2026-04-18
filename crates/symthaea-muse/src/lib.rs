// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # symthaea-muse
//!
//! Consciousness-driven music synthesis for Symthaea. Maps cognitive state —
//! Eight Harmonies, neuromodulators, temporal dynamics — into melody, rhythm,
//! and timbre, producing PCM audio.
//!
//! # Architecture
//!
//! ```text
//! MusicalState → pitch (Harmonies → intervals) + rhythm (arousal/NE → tempo)
//!              → melody (autoregressive note generation) → synth (additive) → PCM
//! ```
//!
//! # Harmony-to-Interval Mapping
//!
//! | Harmony | Interval | Character |
//! |---------|----------|-----------|
//! | ResonantCoherence | Unison/Octave | Consonance |
//! | PanSentientFlourishing | Major 3rd | Warmth |
//! | IntegralWisdom | Perfect 5th | Stability |
//! | InfinitePlay | Minor 7th | Tension |
//! | UniversalInterconnectedness | Perfect 4th | Openness |
//! | SacredReciprocity | Major 6th | Reciprocal warmth |
//! | EvolutionaryProgression | Ascending seq | Growth |
//! | SacredStillness | Drone/pedal | Contemplation |

#![deny(unsafe_code)]

pub mod choreography;
pub mod critic;
pub mod export;
pub mod fingerprint;
pub mod form;
#[cfg(feature = "muse-live")]
pub mod live_output;
pub mod mel_extractor;
pub mod melody;
pub mod midi;
pub mod midi_loader;
pub mod neural_melody;
pub mod notation;
pub mod training_pairs;
pub mod hdc_mel_decoder;
pub mod pitch;
pub mod rhythm;
pub mod stream;
pub mod streaming;
pub mod structure;
pub mod synth;
pub mod training;
pub mod voice;

use serde::{Deserialize, Serialize};

// ─── Core Types ──────────────────────────────────────────────────────────────

/// Cognitive state for music generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MusicalState {
    pub harmony_activations: [f32; 8],
    pub dopamine: f32,
    pub serotonin: f32,
    pub noradrenaline: f32,
    pub arousal: f32,
    pub valence: f32,
    pub consciousness_level: f32,
    pub prediction_error: f32,
}

impl Default for MusicalState {
    fn default() -> Self {
        Self {
            harmony_activations: [0.3; 8],
            dopamine: 0.5,
            serotonin: 0.5,
            noradrenaline: 0.3,
            arousal: 0.4,
            valence: 0.0,
            consciousness_level: 0.5,
            prediction_error: 0.1,
        }
    }
}

/// A single musical frame (one time-step of synthesis parameters).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MusicalFrame {
    /// Fundamental frequency in Hz (20-4000 range).
    pub pitch_hz: f32,
    /// Note velocity/loudness [0, 1].
    pub velocity: f32,
    /// Harmonic partial amplitudes (variable length, one per partial).
    pub timbre: Vec<f32>,
    /// Attack probability [0, 1] — new note trigger.
    pub onset: f32,
    /// Note sustain signal [0, 1].
    pub sustain: f32,
    /// FM modulation depth [0, 1] (CfC-controlled).
    pub fm_depth: f32,
    /// FM carrier:modulator ratio [0, 1] (CfC-controlled).
    pub fm_ratio: f32,
    /// Stereo pan position [-1, 1] (CfC-controlled).
    pub pan: f32,
    /// Reverb send amount [0, 1] (CfC-controlled).
    pub reverb_send: f32,
    /// Vibrato depth [0, 1] (CfC-controlled).
    pub vibrato_depth: f32,
}

/// Melody generation mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MelodyMode {
    /// Original rand-based melody: fast, deterministic with seed.
    Classic,
    /// CfC neural melody: temporal coherence creates phrases, motifs, contours.
    Neural,
}

impl Default for MelodyMode {
    fn default() -> Self {
        Self::Classic
    }
}

/// Audio output format for compositions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputFormat {
    /// Mono 16-bit PCM (legacy, compact).
    Mono16,
    /// Mono 32-bit float [-1.0, 1.0].
    MonoF32,
    /// Stereo 32-bit float with per-voice panning.
    StereoF32,
}

impl Default for OutputFormat {
    fn default() -> Self {
        Self::StereoF32
    }
}

/// Reverb configuration.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ReverbConfig {
    /// Room size [0, 1] — controls feedback coefficient.
    pub room_size: f32,
    /// Damping [0, 1] — high-frequency absorption.
    pub damping: f32,
    /// Stereo width [0, 1] — decorrelation between L/R.
    pub width: f32,
}

impl Default for ReverbConfig {
    fn default() -> Self {
        Self {
            room_size: 0.5,
            damping: 0.5,
            width: 0.8,
        }
    }
}

/// Configuration for music generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MuseConfig {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Duration in seconds.
    pub duration_secs: f32,
    /// Base tempo in BPM (modulated by arousal/NE).
    pub base_tempo_bpm: f32,
    /// Number of notes to generate.
    pub max_notes: usize,
    /// Melody generation mode.
    pub melody_mode: MelodyMode,
    /// Output audio format.
    pub output_format: OutputFormat,
    /// Number of harmonic partials for additive synthesis (1-16).
    pub num_partials: usize,
    /// Enable PolyBLEP anti-aliasing for alias-free oscillators.
    pub enable_antialiasing: bool,
    /// Reverb configuration.
    pub reverb: ReverbConfig,
    /// CfC network layer sizes for neural melody mode.
    pub cfc_layer_sizes: Vec<usize>,
    /// Maximum FM modulation depth in radians (default 3.0, horror uses 8.0).
    pub max_fm_depth: f32,
    /// Enable sub-bass oscillator one octave below fundamental.
    pub enable_sub_bass: bool,
    /// Unison detune amount (0.0 = off, 0.01 = heavy chorus).
    pub unison_detune: f32,
    /// Filtered white noise mix level (0.0 = off, 0.2 = heavy).
    pub noise_mix: f32,
}

impl Default for MuseConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44100,
            duration_secs: 8.0,
            base_tempo_bpm: 80.0,
            max_notes: 32,
            melody_mode: MelodyMode::Classic,
            output_format: OutputFormat::StereoF32,
            num_partials: 8,
            enable_antialiasing: true,
            reverb: ReverbConfig::default(),
            cfc_layer_sizes: vec![16, 16, 8],
            max_fm_depth: 3.0,
            enable_sub_bass: false,
            unison_detune: 0.0,
            noise_mix: 0.0,
        }
    }
}

impl MuseConfig {
    /// Configuration for horror/high-tension audio.
    ///
    /// Deep FM modulation (8 rad), sub-bass, detuned unison, noise texture,
    /// large reverb with low damping. Designed to make players sweat.
    pub fn horror() -> Self {
        Self {
            max_fm_depth: 8.0,
            enable_sub_bass: true,
            unison_detune: 0.005,
            noise_mix: 0.1,
            num_partials: 12,
            reverb: ReverbConfig {
                room_size: 0.85,
                damping: 0.3,
                width: 1.0,
            },
            ..Default::default()
        }
    }

    /// Configuration for the Lunar Elite's sterile aesthetic.
    ///
    /// Pure sine tones, minimal partials, tight reverb, no sub-bass.
    /// Quantization-locked perfection. Cold and expensive.
    pub fn elite_sterile() -> Self {
        Self {
            max_fm_depth: 0.5,
            enable_sub_bass: false,
            unison_detune: 0.0,
            noise_mix: 0.0,
            num_partials: 2,
            reverb: ReverbConfig {
                room_size: 0.2,
                damping: 0.8,
                width: 0.3,
            },
            ..Default::default()
        }
    }
}

/// Audio sample data in various formats.
#[derive(Debug, Clone)]
pub enum AudioData {
    /// Mono 16-bit PCM samples.
    I16(Vec<i16>),
    /// Mono 32-bit float samples [-1.0, 1.0].
    F32(Vec<f32>),
    /// Stereo 32-bit float samples [left, right].
    StereoF32(Vec<[f32; 2]>),
}

impl AudioData {
    /// Number of sample frames (mono samples or stereo pairs).
    pub fn len(&self) -> usize {
        match self {
            Self::I16(v) => v.len(),
            Self::F32(v) => v.len(),
            Self::StereoF32(v) => v.len(),
        }
    }

    /// True if the audio data is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A generated musical composition.
#[derive(Debug, Clone)]
pub struct Composition {
    /// Audio sample data.
    pub audio: AudioData,
    /// Sample rate.
    pub sample_rate: u32,
    /// Generated note sequence.
    pub notes: Vec<Note>,
    /// Duration in seconds.
    pub duration_secs: f32,
    /// Musical section type (derived from cognitive state).
    pub section: structure::SectionType,
}

impl Composition {
    /// Export as MusicXML 4.0 notation.
    pub fn to_musicxml(&self, tempo_bpm: f32) -> String {
        notation::to_musicxml(self, tempo_bpm)
    }

    /// Export as SVG staff notation.
    pub fn to_score_svg(&self, tempo_bpm: f32) -> String {
        notation::to_score_svg(self, tempo_bpm)
    }
}

/// A discrete musical note.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Note {
    /// Frequency in Hz.
    pub frequency: f32,
    /// Start time in seconds.
    pub start_time: f32,
    /// Duration in seconds.
    pub duration: f32,
    /// Velocity [0, 1].
    pub velocity: f32,
}

// ─── Top-Level API ───────────────────────────────────────────────────────────

pub use arc::compose_with_arc;

/// Generate a musical composition from a cognitive state.
///
/// Pipeline: plan song form → generate per-section melodies (with key shifts
/// and density modulation) → arrange polyphonic voices → synthesize audio.
pub fn compose(config: &MuseConfig, state: &MusicalState, seed: u64) -> Composition {
    // 1. Plan song form — multi-section structure from Eight Harmonies
    let song_form = form::plan_form(state, config.duration_secs);
    let overall_section = structure::determine_section(state);

    // 2. Determine base rhythm
    let tempo = rhythm::compute_tempo(config, state);
    let beat_duration = 60.0 / tempo;
    let base_scale = pitch::build_scale(state);

    // HIGH-AROUSAL GRACE: when arousal > 0.5, cap note density so the
    // synthesis engine can handle the output. Intensity comes from harmonic
    // tension and register, not density. This is a honest acknowledgement
    // of synthesis limits — dense polyphony sounds harsh/arcade-like.
    let arousal_density_cap = if state.arousal > 0.5 {
        // Linear reduction: arousal 0.5 → 1.0x, arousal 1.0 → 0.35x
        1.0 - (state.arousal - 0.5) * 1.3
    } else {
        1.0
    };

    // 3. Generate melody per section with key shifts and density modulation
    let base_notes_per_section = config
        .max_notes
        .max(2)
        .checked_div(song_form.sections.len().max(1))
        .unwrap_or(config.max_notes)
        .max(2);
    let notes_per_section = ((base_notes_per_section as f32) * arousal_density_cap)
        .round()
        .max(2.0) as usize;

    let mut all_notes: Vec<Note> = Vec::new();

    for (sec_idx, section) in song_form.sections.iter().enumerate() {
        let density = structure::density_multiplier(section.section_type);
        let sec_max =
            ((notes_per_section as f32) * density * section.energy_level.max(0.3)).round() as usize;

        let sec_config = MuseConfig {
            max_notes: sec_max.max(1),
            duration_secs: section.duration,
            ..config.clone()
        };

        // Apply key shift: transpose the scale
        let key_ratio = 2.0_f32.powf(section.key_shift as f32 / 12.0);
        let sec_scale: Vec<f32> = base_scale.iter().map(|&f| f * key_ratio).collect();

        let sec_seed = seed.wrapping_add(sec_idx as u64);
        let mut sec_notes = match config.melody_mode {
            MelodyMode::Classic => {
                melody::generate_melody(&sec_config, state, &sec_scale, beat_duration, sec_seed)
            }
            MelodyMode::Neural => {
                use symthaea_core::genesis::GenesisSeed;
                let genesis = GenesisSeed::from_phrase(&format!("muse-neural-{sec_seed}"));
                let mut neural = neural_melody::NeuralMelody::new(&genesis, config);
                // Load trained projections if available
                let proj_path = std::path::Path::new("data/midi-training/melody_projections.json");
                neural.load_trained_projections(proj_path);
                neural.generate(&sec_config, state, &sec_scale, beat_duration)
            }
        };

        // Offset note times to this section's position
        for note in &mut sec_notes {
            note.start_time += section.start_time;
        }
        all_notes.extend(sec_notes);
    }

    // 3.5. Generate chord accompaniment — bass + harmony voices from progression
    let progression = instruments::select_progression(state);
    let chord_notes = generate_chord_accompaniment(
        &progression, &base_scale, tempo, config.duration_secs, state, seed,
    );
    all_notes.extend(chord_notes);

    // 4. Arrange voices (polyphony gated by consciousness level)
    let arrangement = voice::arrange(&all_notes, state);

    // 5. Synthesize audio
    let total_samples = (config.duration_secs * config.sample_rate as f32) as usize;
    let audio = synth::render_arrangement(
        &arrangement,
        config.sample_rate,
        total_samples,
        state,
        config,
    );

    Composition {
        audio,
        sample_rate: config.sample_rate,
        notes: all_notes,
        duration_secs: config.duration_secs,
        section: overall_section,
    }
}

/// Generate bass + harmony chord notes from a progression.
///
/// Creates two layers:
/// - **Bass**: chord root, one octave below the melody register, playing on
///   beat 1 of each chord change (and beat 3 for walking bass at high arousal).
/// - **Harmony pad**: chord tones sustained through each chord change, at
///   reduced velocity for background warmth.
///
/// These notes are mixed with the melody before arrangement, so `voice::arrange()`
/// can assign them to Bass and Harmony voice roles.
fn generate_chord_accompaniment(
    progression: &[instruments::ProgressionChord],
    scale: &[f32],
    tempo: f32,
    duration_secs: f32,
    state: &MusicalState,
    _seed: u64,
) -> Vec<Note> {
    if progression.is_empty() || scale.is_empty() {
        return Vec::new();
    }

    let beat_dur = 60.0 / tempo;
    let mut notes = Vec::new();
    let mut time = 0.0f32;

    // Root frequency: use the lowest scale degree as reference
    let root_freq = scale.iter().copied().fold(f32::MAX, f32::min).max(60.0);

    // Bass register: one octave below root
    let bass_octave = root_freq * 0.5;

    // Harmony register: at root level (melody is typically above)
    let harmony_octave = root_freq;

    // Gesture shapes velocity and articulation
    let gesture = emotional_gestures::gesture_for_emotion(
        emotional_gestures::detect_emotion(state),
    );
    let bass_vel_base = (0.5 * gesture.velocity_scale).clamp(0.2, 0.8);
    let harmony_vel_base = (0.3 * gesture.velocity_scale).clamp(0.15, 0.5);
    // Chord counter for dynamic crescendo/decrescendo across progression
    let mut chord_idx = 0usize;

    // Cycle through progression for the full duration
    let total_prog_beats: f32 = progression.iter().map(|c| c.duration_beats).sum();
    if total_prog_beats < 0.1 {
        return Vec::new();
    }

    while time < duration_secs {
        for chord in progression {
            if time >= duration_secs {
                break;
            }

            let chord_dur_secs = chord.duration_beats * beat_dur;
            let root_ratio = 2.0f32.powf(chord.root_semitones as f32 / 12.0);

            // Dynamic shaping: crescendo through first half, decrescendo through second
            let prog_len = progression.len().max(1);
            let prog_t = (chord_idx % prog_len) as f32 / prog_len as f32;
            let dyn_curve = 1.0 - (prog_t - 0.4).abs() * 1.5; // peaks at 40%
            let dyn_factor = 0.8 + dyn_curve.clamp(0.0, 1.0) * 0.4; // 0.8-1.2

            let bass_vel = (bass_vel_base * dyn_factor).clamp(0.15, 0.9);
            let harmony_vel = (harmony_vel_base * dyn_factor).clamp(0.10, 0.6);

            // Bass note: root of the chord, one octave below
            let bass_freq = bass_octave * root_ratio;
            let bass_dur = chord_dur_secs * (1.0 - gesture.staccato * 0.5);
            notes.push(Note {
                frequency: bass_freq,
                start_time: time,
                duration: bass_dur.max(0.1),
                velocity: bass_vel,
            });

            // Walking bass: add 5th on beat 3 at higher arousal
            if state.arousal > 0.4 && chord.duration_beats >= 4.0 {
                let fifth_ratio = 2.0f32.powf(7.0 / 12.0);
                notes.push(Note {
                    frequency: bass_freq * fifth_ratio,
                    start_time: time + beat_dur * 2.0,
                    duration: beat_dur * 1.5,
                    velocity: bass_vel * 0.75,
                });
            }

            // Harmony pad: chord tones sustained through chord, on the beat
            let ratios = chord.chord_type.ratios();
            for (i, &ratio) in ratios.iter().enumerate() {
                if i == 0 { continue; } // root is in bass
                let harm_freq = harmony_octave * root_ratio * ratio;
                let harm_dur = chord_dur_secs * (1.0 - gesture.staccato * 0.4);
                notes.push(Note {
                    frequency: harm_freq,
                    start_time: time, // on the beat, not staggered
                    duration: harm_dur.max(0.1),
                    velocity: harmony_vel * (1.0 - i as f32 * 0.05),
                });
            }

            time += chord_dur_secs;
            chord_idx += 1;
        }
    }

    notes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compose_produces_audio() {
        let config = MuseConfig {
            duration_secs: 2.0,
            max_notes: 8,
            ..Default::default()
        };
        let state = MusicalState::default();
        let comp = compose(&config, &state, 42);
        assert!(!comp.audio.is_empty());
        assert!(!comp.notes.is_empty());
        assert_eq!(comp.sample_rate, 44100);
    }

    #[test]
    fn deterministic_with_same_seed() {
        let config = MuseConfig {
            duration_secs: 1.0,
            max_notes: 4,
            ..Default::default()
        };
        let state = MusicalState::default();
        let c1 = compose(&config, &state, 42);
        let c2 = compose(&config, &state, 42);
        assert_eq!(c1.audio.len(), c2.audio.len());
        assert_eq!(c1.notes.len(), c2.notes.len());
    }

    #[test]
    fn different_states_different_music() {
        let config = MuseConfig {
            duration_secs: 1.0,
            max_notes: 4,
            ..Default::default()
        };
        let calm = MusicalState {
            arousal: 0.1,
            valence: 0.5,
            ..Default::default()
        };
        let excited = MusicalState {
            arousal: 0.9,
            valence: -0.5,
            noradrenaline: 0.8,
            ..Default::default()
        };
        let c1 = compose(&config, &calm, 42);
        let c2 = compose(&config, &excited, 42);
        assert_ne!(c1.notes.len(), c2.notes.len());
    }

    #[test]
    fn no_nan_or_inf_samples() {
        let config = MuseConfig {
            duration_secs: 2.0,
            max_notes: 16,
            ..Default::default()
        };
        let state = MusicalState {
            harmony_activations: [0.8; 8],
            consciousness_level: 0.9,
            ..Default::default()
        };
        let comp = compose(&config, &state, 42);
        match &comp.audio {
            AudioData::StereoF32(samples) => {
                for s in samples {
                    assert!(s[0].is_finite(), "left NaN/Inf");
                    assert!(s[1].is_finite(), "right NaN/Inf");
                }
            }
            AudioData::F32(samples) => {
                for &s in samples {
                    assert!(s.is_finite(), "NaN/Inf");
                }
            }
            AudioData::I16(samples) => {
                for &s in samples {
                    assert!(s >= i16::MIN && s <= i16::MAX);
                }
            }
        }
    }

    #[test]
    fn neural_mode_produces_audio() {
        let config = MuseConfig {
            duration_secs: 2.0,
            max_notes: 8,
            melody_mode: MelodyMode::Neural,
            ..Default::default()
        };
        let state = MusicalState::default();
        let comp = compose(&config, &state, 42);
        assert!(!comp.audio.is_empty());
        assert!(!comp.notes.is_empty());
    }

    #[test]
    fn neural_differs_from_classic() {
        let state = MusicalState::default();
        let classic = compose(
            &MuseConfig {
                duration_secs: 2.0,
                max_notes: 8,
                melody_mode: MelodyMode::Classic,
                ..Default::default()
            },
            &state,
            42,
        );
        let neural = compose(
            &MuseConfig {
                duration_secs: 2.0,
                max_notes: 8,
                melody_mode: MelodyMode::Neural,
                ..Default::default()
            },
            &state,
            42,
        );
        let freqs_c: Vec<i32> = classic
            .notes
            .iter()
            .map(|n| (n.frequency * 10.0) as i32)
            .collect();
        let freqs_n: Vec<i32> = neural
            .notes
            .iter()
            .map(|n| (n.frequency * 10.0) as i32)
            .collect();
        assert_ne!(freqs_c, freqs_n, "neural and classic should differ");
    }

    #[test]
    fn structure_modulates_density() {
        let ambient_state = MusicalState {
            harmony_activations: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8],
            arousal: 0.1,
            ..Default::default()
        };
        let climactic_state = MusicalState {
            harmony_activations: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0],
            arousal: 0.7,
            ..Default::default()
        };
        let config = MuseConfig {
            duration_secs: 4.0,
            max_notes: 16,
            ..Default::default()
        };
        let ambient = compose(&config, &ambient_state, 42);
        let climactic = compose(&config, &climactic_state, 42);
        assert!(
            ambient.notes.len() <= climactic.notes.len(),
            "ambient {} should have <= climactic {} notes",
            ambient.notes.len(),
            climactic.notes.len()
        );
        assert_eq!(ambient.section, structure::SectionType::Ambient);
        assert_eq!(climactic.section, structure::SectionType::Climactic);
    }

    #[test]
    fn stereo_output_format() {
        let config = MuseConfig {
            duration_secs: 1.0,
            max_notes: 4,
            output_format: OutputFormat::StereoF32,
            ..Default::default()
        };
        let state = MusicalState::default();
        let comp = compose(&config, &state, 42);
        assert!(matches!(comp.audio, AudioData::StereoF32(_)));
    }

    #[test]
    fn mono16_backward_compat() {
        let config = MuseConfig {
            duration_secs: 1.0,
            max_notes: 4,
            output_format: OutputFormat::Mono16,
            ..Default::default()
        };
        let state = MusicalState::default();
        let comp = compose(&config, &state, 42);
        assert!(matches!(comp.audio, AudioData::I16(_)));
    }
}

pub mod arc;
pub mod creative_bench;
pub mod narrative_bridge;
pub mod ablation;
pub mod aesthetic_listener;
pub mod ambient_drone;
pub mod audio_feedback;
pub mod auto_master;
pub mod binaural;
pub mod collaborative;
pub mod composer_mind;
pub mod consciousness_reverb;
pub mod creative_agency;
pub mod density_regulator;
pub mod dramatic;
pub mod emotional_gestures;
pub mod genre_presets;
pub mod instruments;
pub mod learned_melody;
pub mod melodic_grammar;
pub mod midi_export;
pub mod midi_trainer;
pub mod mixing;
pub mod motif_memory;
pub mod musical_inference;
pub mod param_tuner;
pub mod percussion;
pub mod performance;
pub mod phi_optimizer;
pub mod production;
pub mod rhythm_engine;
pub mod sample_player;
pub mod sidechain;
pub mod similarity_monitor;
pub mod spectral_vocoder;
pub mod state_smoother;
pub mod substrate_timbre;
pub mod synesthesia;
pub mod taste_bench;
pub mod taste_melody;
pub mod taste_space;
pub mod temporal_hierarchy;
pub mod timbre_space;
#[cfg(feature = "voice")]
pub mod voice_bridge;
pub mod voice_leader;
pub mod wake_protocol;
pub mod wavetable;
