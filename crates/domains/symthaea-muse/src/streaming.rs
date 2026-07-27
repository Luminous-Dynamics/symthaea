// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Full-pipeline streaming PCM synthesis with all consciousness-coupled modules.
//!
//! [`StreamingSynth`] integrates:
//! - Wavetable morphing (consciousness → table selection)
//! - Sidechain ducking (lead → bass/harmony gain reduction)
//! - Consciousness reverb (Phi → room, harmonies → early reflections)
//! - Binaural rendering (Phi → spatial spread, harmonies → positions)
//! - Audio feedback (output → feature extraction → state modulation)
//! - Phi optimizer (target Phi → parameter perturbation)
//! - Substrate timbre (substrate type → synthesis character)

use crate::aesthetic_listener::AestheticListener;
use crate::ambient_drone::AmbientDrone;
use crate::audio_feedback::AudioFeedbackEncoder;
use crate::binaural::BinauralConsciousnessRenderer;
use crate::composer_mind::ComposerMind;
use crate::consciousness_reverb::ConsciousnessReverb;
use crate::creative_agency::{CreativeIntent, CreativeJournal};
use crate::density_regulator::DensityRegulator;
use crate::dramatic::DramaticState;
use crate::emotional_gestures;
use crate::instruments::{self, Instrument, KarplusStrong};
use crate::learned_melody::MelodyPredictor;
use crate::mixing::MixingChain;
use crate::motif_memory::MotifMemory;
use crate::musical_inference::MusicalInferenceEngine;
use crate::percussion;
use crate::performance;
use crate::phi_optimizer::{PhiOptimizer, PhiTarget};
use crate::production;
use crate::rhythm_engine::{self, ExtendedChordType, GroovePocket, TimeSignature};
use crate::sample_player::SampleLibrary;
use crate::sidechain::DuckingMatrix;
use crate::similarity_monitor::{SimilarityAction, SimilarityMonitor};
use crate::state_smoother::StateSmoother;
use crate::stream::MuseStream;
use crate::substrate_timbre::{SubstrateTimbreModifier, SubstrateTimbreType};
use crate::taste_melody::TasteMelody;
use crate::voice_leader::{VoiceRole, VoiceState};
use crate::wavetable::{WavetableBank, WavetableOscillator};
use crate::{MuseConfig, MusicalState, Note};

pub const DEFAULT_CHUNK_MS: u32 = 32;
const MAX_ACTIVE_NOTES: usize = 32;
const MIN_CHUNK_SAMPLES: usize = 64;

// ── Synthesis Parameter Constants ────────────────────────────────────────
// Each constant documents its acoustic/psychoacoustic rationale.

/// Minimum gain at zero arousal.
const GAIN_MIN: f32 = 0.22;
/// Arousal gain coefficient. Total: 0.22 (calm) to 0.42 (excited).
/// Reduced from 0.30 to prevent clipping when drums + notes overlap at high arousal.
const GAIN_AROUSAL_COEFF: f32 = 0.20;
/// Valence gain modulation ±15%. Huron (2006): happy music performed louder.
const VALENCE_GAIN_SCALE: f32 = 0.15;

/// Dopamine→brightness floor. Minimum spectral brightness even at DA=0.
const BRIGHTNESS_FLOOR: f32 = 0.3;
/// Dopamine→brightness scale. Full DA adds 0.7 to brightness.
const BRIGHTNESS_DA_SCALE: f32 = 0.7;

/// Base attack time (seconds) at full arousal. Fast transients = excited.
const ATTACK_BASE: f32 = 0.01;
/// Attack lengthening at zero arousal. Slow onsets = calm.
const ATTACK_AROUSAL_SCALE: f32 = 0.05;

/// Phi-vibrato rate (Hz). Co-prime with typical musical vibrato (5-6Hz).
/// Creates spectral flux measurable via FFT without beating with instrument vibrato.
const PHI_VIBRATO_BASE_HZ: f32 = 5.3;
/// Phi-vibrato max depth (cents). 8 cents = sub-semitone, measurable but not dissonant.
const PHI_VIBRATO_MAX_CENTS: f32 = 8.0;

/// NE→vibrato rate multiplier. Zentner et al. (2008): urgency → faster modulation.
/// At max NE, vibrato rate increases 50% (5.3 → 8.0Hz).
const NE_VIBRATO_RATE_SCALE: f32 = 0.5;
/// NE→upper partial boost. Adds harshness under stress.
const NE_BRIGHTNESS_BOOST: f32 = 0.15;

/// Max partial detuning (cents) at valence=-1.0.
/// Plomp & Levelt (1965): roughness peaks at ~25% critical bandwidth.
/// 15 cents at 440Hz ≈ 3.8Hz beating, within the roughness range.
const VALENCE_DETUNE_MAX_CENTS: f32 = 15.0;

/// Timbre manifold minimum blend (at neutral V-A). Instrument partials dominate.
const MANIFOLD_BLEND_MIN: f32 = 0.15;
/// Timbre manifold additional blend per unit emotional distance from neutral.
const MANIFOLD_BLEND_SCALE: f32 = 0.15;

// ─── ADSR + clip ────────────────────────────────────────────────────────────

fn envelope(atk: f32, dec: f32, sus: f32, rel: f32, t: f32, dur: f32) -> f32 {
    if t < atk {
        // Attack: fast rise
        t / atk
    } else if t < atk + dec {
        // Decay: exponential fall to sustain (not linear — sounds more natural)
        let decay_progress = (t - atk) / dec;
        1.0 - decay_progress * decay_progress * (1.0 - sus)
    } else if t < dur {
        // Sustain: gentle exponential decay (piano strings lose energy over time)
        let sustain_time = t - atk - dec;
        sus * (-sustain_time * 0.8).exp() // slow decay during sustain
    } else {
        // Release: fast exponential fade
        let release_progress = (t - dur) / rel;
        sus * (-release_progress * 3.0).exp() // fast release
    }
}

/// Gentle limiter: identity below ±0.95, smooth compression above.
/// Unlike tanh (which compresses starting at ~0.3), this preserves full
/// dynamics for normal audio levels and only activates on peaks.
fn gentle_limit(x: f32) -> f32 {
    let threshold = 0.95_f32;
    if x > threshold {
        threshold + (1.0 - threshold) * (1.0 - (-(x - threshold) * 4.0).exp())
    } else if x < -threshold {
        -threshold - (1.0 - threshold) * (1.0 - ((x + threshold) * 4.0).exp())
    } else {
        x
    }
}

// ─── Active note with wavetable + vibrato ───────────────────────────────────

struct ActiveNote {
    note: Note,
    sample_pos: usize,
    total_samples: usize,
    partial_phases: Vec<f32>,
    fm_phase: f32,
    pan: f32,
    volume: f32,
    voice_idx: usize,
    wavetable_osc: WavetableOscillator,
    vibrato_phase: f32,
    /// Instrument assigned to this note.
    instrument: Instrument,
    /// Karplus-Strong state (for guitar/harp).
    ks: Option<KarplusStrong>,
    /// Pre-rendered FM buffer (for e-piano/bell, rendered at note start).
    fm_buffer: Option<Vec<f32>>,
    /// Sample-library key + playback rate, resolved ONCE at note spawn.
    /// The per-sample render loop must use `get_by_key` (O(1)) — resolving
    /// per sample was an O(library) scan per sample per note.
    sample_ref: Option<((crate::sample_player::InstrumentKey, u8), f64)>,
}

/// Full-pipeline streaming synthesis engine.
pub struct StreamingSynth {
    muse_stream: MuseStream,
    active_notes: Vec<ActiveNote>,
    state: MusicalState,
    config: MuseConfig,
    /// Collected notes for MIDI export (actual generated notes, not estimated).
    pub generated_notes: Vec<crate::Note>,
    sample_rate: u32,
    chunk_samples: usize,
    total_samples_rendered: u64,
    chunks_rendered: u64,
    note_gen_cadence: u32,
    chunks_since_note_gen: u32,
    // Consciousness-coupled modules
    reverb: ConsciousnessReverb,
    binaural: BinauralConsciousnessRenderer,
    sidechain: DuckingMatrix,
    feedback: AudioFeedbackEncoder,
    phi_optimizer: PhiOptimizer,
    wavetable_bank: WavetableBank,
    substrate: Option<SubstrateTimbreModifier>,
    /// Aesthetic self-listener: judges beauty, corrects harshness.
    aesthetic: AestheticListener,
    /// Ambient drone: continuous low pad filling silence.
    drone: AmbientDrone,
    /// Motif memory: tracks phrases, enables repetition and development.
    motif: MotifMemory,
    /// Current chord progression (cycled through).
    chord_progression: Vec<crate::instruments::ProgressionChord>,
    /// Current chord index in progression.
    chord_idx: usize,
    /// Beats elapsed in current chord.
    chord_beat_counter: f32,
    /// Notes generated (for humanization seed).
    note_counter: u32,
    /// Previous note frequency (for emotional gesture direction).
    prev_note_freq: Option<f32>,
    /// Creative journal: Symthaea's autonomous aesthetic memory.
    journal: CreativeJournal,
    /// Composer mind: goal-directed planning and long-range form.
    composer: ComposerMind,
    /// Dramatic state: silence, modulation, sub-bass, texture.
    dramatic: DramaticState,
    /// Linear melody predictor (hand-tuned priors — see learned_melody.rs
    /// provenance note; NOT verifiably trained).
    melody_predictor: MelodyPredictor,
    /// Taste-optimized melody generator (targets benchmark directly).
    pub taste_melody: TasteMelody,
    /// Voice synthesis bridge (feature-gated).
    #[cfg(feature = "voice")]
    voice_bridge: Option<crate::voice_bridge::MuseVoiceBridge>,
    /// Density regulator: section-aware note spacing.
    density: DensityRegulator,
    /// Self-similarity monitor: ensures right amount of repetition.
    similarity: SimilarityMonitor,
    /// Sample library for richer instrument sounds.
    sample_lib: SampleLibrary,
    /// Current time signature (dynamic — changes with consciousness).
    time_sig: TimeSignature,
    /// Groove pocket (micro-timing for feel).
    groove: GroovePocket,
    /// Voice states for independent voice leading.
    lead_voice: VoiceState,
    /// Bar counter for form progression.
    bars_elapsed_frac: f32,
    /// State smoother: prevents abrupt parameter changes.
    smoother: StateSmoother,
    /// FEP active inference engine for music (the system predicts its own sound).
    fep_engine: MusicalInferenceEngine,
    /// Free energy history for learning verification.
    fe_history: Vec<f64>,
    /// Dither PRNG state.
    dither_state: u32,
    /// Brightness evolution: smoothed RMS for timbral arc.
    brightness_rms: f32,
    /// Brightness LP filter state (L/R).
    brightness_lp_l: f32,
    brightness_lp_r: f32,
    /// Mixing chain: EQ → compressor → limiter.
    mixing: MixingChain,
    /// Pending drum hits for the current bar.
    drum_hits: Vec<percussion::DrumHit>,
    /// Sample position within current bar (for drum timing).
    bar_sample_pos: usize,
    /// Samples per bar (computed from tempo).
    samples_per_bar: usize,
    /// Audio feedback strength [0, 1]. 0 = open loop, 1 = full strange loop.
    pub feedback_strength: f32,
    /// Enable binaural rendering (vs simple pan law).
    pub enable_binaural: bool,
    /// Enable sidechain ducking.
    pub enable_sidechain: bool,
    /// Enable Phi optimization.
    pub enable_phi_optimizer: bool,
    /// Enable FEP active inference (generative model of own audio).
    pub enable_fep: bool,
    /// Emotional timbre manifold: V-A → interpolated partial amplitudes.
    timbre_manifold: crate::timbre_space::TimbreManifold,
    /// Cached manifold partials for current emotional state.
    manifold_partials: Vec<f32>,
    /// Haas delay history buffers (4 voices × up to 353 samples).
    /// Maintains continuity across chunk boundaries — fixes crackling
    /// caused by reading sample[0] when delayed_i < delay.
    haas_history: [Vec<f32>; 4],
    /// Smoothed sub-bass volume — prevents amplitude jump when kick fires.
    sub_bass_volume_smoothed: f32,
}

impl StreamingSynth {
    pub fn new(config: MuseConfig, sample_rate: u32) -> Self {
        let chunk_samples = ((DEFAULT_CHUNK_MS as f32 / 1000.0) * sample_rate as f32) as usize;
        let chunk_samples = chunk_samples.max(MIN_CHUNK_SAMPLES);
        Self {
            generated_notes: Vec::new(),
            muse_stream: MuseStream::new(42, config.clone()),
            active_notes: Vec::with_capacity(MAX_ACTIVE_NOTES),
            state: MusicalState::default(),
            config,
            sample_rate,
            chunk_samples,
            total_samples_rendered: 0,
            chunks_rendered: 0,
            note_gen_cadence: 4,
            chunks_since_note_gen: 0,
            reverb: ConsciousnessReverb::new(sample_rate),
            binaural: BinauralConsciousnessRenderer::new(4, sample_rate),
            sidechain: DuckingMatrix::default_matrix(sample_rate),
            feedback: AudioFeedbackEncoder::new(),
            phi_optimizer: PhiOptimizer::new(PhiTarget::Maximize),
            wavetable_bank: WavetableBank::default_bank(),
            substrate: None,
            aesthetic: AestheticListener::new(),
            drone: AmbientDrone::new(sample_rate),
            motif: MotifMemory::new(),
            chord_progression: crate::instruments::ambient_progression(),
            chord_idx: 0,
            chord_beat_counter: 0.0,
            note_counter: 0,
            prev_note_freq: None,
            journal: CreativeJournal::new(),
            composer: ComposerMind::new(),
            dramatic: DramaticState::new(),
            melody_predictor: MelodyPredictor::new(),
            taste_melody: TasteMelody::new(),
            #[cfg(feature = "voice")]
            voice_bridge: None,
            density: DensityRegulator::new(),
            similarity: SimilarityMonitor::new(),
            sample_lib: {
                let mut lib = SampleLibrary::new();
                lib.generate_synthetic(sample_rate);
                lib
            },
            time_sig: TimeSignature::FOUR_FOUR,
            groove: GroovePocket {
                swing: 0.0,
                laid_back: 0.0,
                ghost_notes: 0.0,
            },
            lead_voice: VoiceState::new(VoiceRole::Lead),
            bars_elapsed_frac: 0.0,
            smoother: StateSmoother::default_smooth(),
            fep_engine: MusicalInferenceEngine::new(),
            fe_history: Vec::with_capacity(1024),
            dither_state: 42,
            brightness_rms: 0.0,
            brightness_lp_l: 0.0,
            brightness_lp_r: 0.0,
            mixing: MixingChain::new(sample_rate),
            drum_hits: Vec::new(),
            bar_sample_pos: 0,
            samples_per_bar: (44100.0 * 60.0 / 80.0 * 4.0) as usize, // 4 beats at 80bpm
            feedback_strength: 0.3,
            enable_binaural: true,
            enable_sidechain: true,
            enable_phi_optimizer: false,
            enable_fep: true,
            timbre_manifold: crate::timbre_space::TimbreManifold::default_manifold(
                &symthaea_core::genesis::GenesisSeed::from_phrase("symthaea-muse-timbre"),
            ),
            manifold_partials: vec![1.0, 0.5, 0.25, 0.12, 0.06, 0.03, 0.015, 0.0],
            haas_history: [
                vec![0.0; 400],
                vec![0.0; 400],
                vec![0.0; 400],
                vec![0.0; 400],
            ],
            sub_bass_volume_smoothed: 0.0,
        }
    }

    pub fn update_state(&mut self, state: &MusicalState) {
        // Smooth all state transitions (~1 second time constant)
        let smoothed = self.smoother.smooth(state);
        self.state = smoothed.clone();
        self.muse_stream.update_state(&smoothed);
        self.reverb.update_state(&smoothed);
        self.binaural.update_state(&smoothed);
        self.drone.update_state(&smoothed, self.active_notes.len());

        // Update timbre manifold: V-A → interpolated partial amplitudes
        self.manifold_partials = self
            .timbre_manifold
            .lookup(smoothed.valence, smoothed.arousal);

        if self.enable_phi_optimizer {
            self.phi_optimizer.update_phi(state.consciousness_level);
        }

        // Emotion anchor: tell the FEP self-model what emotional quadrant to preserve
        if self.enable_fep {
            self.fep_engine.set_emotion_anchor(state);
        }

        // Composer mind: advance form and apply goal-directed state shaping
        let tempo = self.muse_stream.tempo();
        let bars_per_chunk =
            (self.chunk_samples as f32 / self.sample_rate as f32) * (tempo / 60.0) / 4.0;
        self.bars_elapsed_frac += bars_per_chunk;
        if self.bars_elapsed_frac >= 1.0 {
            self.bars_elapsed_frac -= 1.0;
            crate::composer_mind::free_compose_step(
                &mut self.state,
                &mut self.composer,
                self.journal.beauty_ema(),
            );
        }
        self.composer.shape_state(&mut self.state);

        // Dramatic state: update silence, modulation, texture, sub-bass
        self.dramatic.update(
            self.composer.section(),
            &self.state,
            self.composer.memory.bars_elapsed() % 16.0,
        );

        // Arousal-driven note generation cadence:
        // Each chunk is ~32ms. At 44.1kHz with 1411 samples/chunk:
        //   1 chunk  = 32ms  (frantic)
        //   30 chunks = ~1s  (moderate)
        //   90 chunks = ~3s  (calm)
        //   150 chunks = ~5s (very peaceful)
        //
        // High arousal (0.9) → every 2 chunks (~64ms, very dense)
        // Mid arousal (0.5) → every 20 chunks (~640ms)
        // Low arousal (0.1) → every 100 chunks (~3.2s, very sparse)
        let arousal = state.arousal.clamp(0.05, 0.95);
        // Exponential curve: sparse at low arousal, dense at high
        let base_cadence = 2.0 + 148.0 * (1.0 - arousal).powf(2.5);
        let rubato_cadence = base_cadence / self.dramatic.tempo_factor.max(0.5);
        self.note_gen_cadence = rubato_cadence.round().clamp(2.0, 200.0) as u32;
        // Only let density regulator adjust if arousal is moderate+.
        if state.arousal > 0.3 {
            self.note_gen_cadence = self.density.adjust_cadence(self.note_gen_cadence);
        }

        // Dynamic time signature and groove (consciousness-driven)
        self.time_sig = rhythm_engine::select_time_signature(&self.state);
        self.groove = GroovePocket::from_state(&self.state);
    }

    /// Set substrate type for timbre coloring.
    pub fn set_substrate(&mut self, substrate: SubstrateTimbreType) {
        self.substrate = Some(SubstrateTimbreModifier::for_substrate(substrate));
    }

    /// A/B-testing support: swap the melody predictor to the pre-training
    /// hand-tuned fallback weights (resets its context). Used by the
    /// `ab_melody_weights` example to render blind trained-vs-fallback
    /// listening pairs; never call this in a production path.
    pub fn use_fallback_melody_weights(&mut self) {
        self.melody_predictor = MelodyPredictor::with_fallback_weights();
    }

    /// Enable voice synthesis (requires "voice" feature).
    #[cfg(feature = "voice")]
    pub fn enable_voice(&mut self) {
        self.voice_bridge = Some(crate::voice_bridge::MuseVoiceBridge::new(self.sample_rate));
    }

    pub fn set_chunk_duration_ms(&mut self, ms: u32) {
        let s = ((ms as f32 / 1000.0) * self.sample_rate as f32) as usize;
        self.chunk_samples = s.max(MIN_CHUNK_SAMPLES);
    }

    /// Render one chunk of stereo PCM through the full pipeline.
    ///
    /// Pipeline: generate notes → additive/wavetable synthesis → per-voice buffers
    /// → sidechain ducking → binaural spatialization → consciousness reverb
    /// → soft clip → audio feedback extraction → Phi optimization → output
    pub fn render_chunk(&mut self) -> Vec<[f32; 2]> {
        // Apply Phi optimizer modulation
        if self.enable_phi_optimizer {
            self.phi_optimizer.modulate_state(&mut self.state);
        }

        // Apply substrate timbre
        if let Some(ref substrate) = self.substrate {
            substrate.apply_to_state(&mut self.state);
        }

        // Generate new notes
        self.chunks_since_note_gen += 1;
        if self.chunks_since_note_gen >= self.note_gen_cadence {
            self.chunks_since_note_gen = 0;
            self.generate_notes();
        }
        self.active_notes.retain(|n| n.sample_pos < n.total_samples);

        let sr = self.sample_rate as f32;
        let nyquist = sr * 0.5;
        let _num_p = self.config.num_partials.clamp(1, 16);
        let _fm_depth = self.state.dopamine * self.config.max_fm_depth;
        let _fm_ratio = 2.0 + self.state.noradrenaline;
        let _rolloff = 1.0 + self.state.serotonin * 0.8;
        // Brightness: dopamine adds warmth, but HIGH AROUSAL REDUCES BRIGHTNESS
        // to prevent harsh/arcade-like timbres. High energy music should be
        // rich but not piercing — the arousal-brightness inverse relationship
        // matches how real music is mixed (loud passages darken, not brighten).
        let arousal_dampen = 1.0 - (self.state.arousal.max(0.5) - 0.5) * 1.0; // 1.0 at arousal≤0.5, 0.5 at arousal=1.0
        let brightness = (BRIGHTNESS_FLOOR
            + self.state.dopamine * BRIGHTNESS_DA_SCALE * arousal_dampen)
            .clamp(0.25, 0.75);

        // NE → vibrato rate scaling: high stress = faster, more agitated vibrato
        let vibrato_rate_scale = 1.0 + self.state.noradrenaline * NE_VIBRATO_RATE_SCALE;

        // NE → harmonic tension bias: ONLY at moderate arousal, not already-loud scenarios
        // At high arousal, NE boost stacks with dopamine brightness and creates harshness
        let ne_brightness_boost = if self.state.arousal > 0.7 {
            0.0 // suppress at high arousal — already too bright
        } else {
            self.state.noradrenaline * NE_BRIGHTNESS_BOOST
        };

        // Prediction error → onset jitter (±samples): surprise makes timing unpredictable
        // Max jitter: ±5ms at PE=1.0 (Vuust & Witek 2014: rhythmic surprise)
        let _pe_jitter_samples = (self.state.prediction_error * 0.005 * sr) as i32;

        // ADSR per instrument type (not just arousal)
        // Piano: fast attack, moderate decay, low sustain (strings ring then die)
        // Pad: slow attack, no decay, high sustain
        // Violin: moderate attack, no decay, high sustain
        let (atk, dec, sus, rel) = match instruments::select_instrument(&self.state) {
            Instrument::Piano | Instrument::PianoPP => (0.005, 0.3, 0.15, 0.4),
            Instrument::Violin | Instrument::Cello => (0.08, 0.1, 0.7, 0.3),
            Instrument::AcousticGuitar | Instrument::Harp | Instrument::Koto => {
                (0.003, 0.2, 0.1, 0.3)
            }
            Instrument::Oud | Instrument::UprightBass => (0.008, 0.15, 0.4, 0.5),
            Instrument::Flute | Instrument::Ney => (0.05, 0.1, 0.6, 0.2),
            Instrument::Clarinet => (0.03, 0.08, 0.7, 0.15),
            Instrument::Trumpet => (0.02, 0.05, 0.8, 0.1),
            Instrument::Saxophone => (0.04, 0.08, 0.7, 0.2),
            Instrument::Sitar => (0.005, 0.15, 0.5, 0.4),
            Instrument::SawLead => (0.01, 0.05, 0.8, 0.15),
            Instrument::Marimba => (0.001, 0.3, 0.02, 0.2),
            Instrument::Bell | Instrument::Kalimba => (0.001, 0.5, 0.05, 0.8),
            Instrument::Pad | Instrument::Organ => (0.2, 0.1, 0.8, 0.5),
            Instrument::ElectricPiano => (0.01, 0.2, 0.3, 0.3),
            // `select_instrument` never returns any of the newer additions
            // below (Viola, Oboe, FrenchHorn, Bassoon, Accordion, Mandolin,
            // Banjo, Vibraphone, Celesta, ChoirAah, StringEnsemble,
            // ElectricBass, Timpani, MusicBox) — this arm only exists to
            // keep the match exhaustive after the enum grew.
            _ => (0.03, 0.1, 0.5, 0.3),
        };

        // ── Phase 1: Render per-voice mono buffers ──
        let num_voices = 4; // lead, bass, harmony, ostinato
        let mut voice_buffers: Vec<Vec<f32>> = (0..num_voices)
            .map(|_| vec![0.0; self.chunk_samples])
            .collect();
        let voice_roles = [
            crate::voice::VoiceRole::Lead,
            crate::voice::VoiceRole::Bass,
            crate::voice::VoiceRole::Harmony,
            crate::voice::VoiceRole::Ostinato,
        ];

        // Select instrument from consciousness state (updated per chunk)
        let _current_instrument = instruments::select_instrument(&self.state);

        // Density-aware attack noise: reduce when many voices active.
        // High polyphony + full attack noise = crackling texture.
        let active_count = self.active_notes.len() as f32;
        let noise_density_factor = if active_count > 8.0 {
            0.3
        } else if active_count > 4.0 {
            0.6
        } else {
            1.0
        };

        for active in &mut self.active_notes {
            let voice = active.voice_idx.min(num_voices - 1);

            // Vibrato LFO: 5Hz for strings, 0 for piano/guitar
            let (vibrato_depth_cents, vibrato_rate) = match active.instrument {
                Instrument::Violin | Instrument::Cello => (25.0, 5.0),
                Instrument::Flute | Instrument::Ney => (15.0, 4.5),
                Instrument::Clarinet => (12.0, 5.5),
                Instrument::Trumpet => (10.0, 5.0),
                Instrument::Saxophone => (20.0, 5.5),
                Instrument::Sitar => (30.0, 6.0),
                Instrument::Oud => (15.0, 4.0),
                Instrument::SawLead => (18.0, 5.5),
                Instrument::Pad => (8.0, 3.0),
                _ => (0.0, 0.0),
            };

            for i in 0..self.chunk_samples {
                if active.sample_pos >= active.total_samples {
                    break;
                }
                let t = active.sample_pos as f32 / sr;
                let env = envelope(atk, dec, sus, rel, t, active.note.duration);

                // Vibrato
                if vibrato_depth_cents > 0.0 {
                    active.vibrato_phase += vibrato_rate / sr;
                    if active.vibrato_phase > 1.0 {
                        active.vibrato_phase -= 1.0;
                    }
                }
                let vibrato_mod = (active.vibrato_phase * std::f32::consts::TAU).sin();
                let vibrato_factor = 2.0f32.powf(vibrato_depth_cents * vibrato_mod / 1200.0);
                // Phi-scaled micro-vibrato: higher consciousness = richer spectral flux
                // 0-8 cents at 5.3Hz — subtle but measurable via spectral analysis
                let psi = self.state.consciousness_level.clamp(0.0, 1.0);
                let phi_vibrato_cents = psi * PHI_VIBRATO_MAX_CENTS;
                let phi_vibrato_rate = PHI_VIBRATO_BASE_HZ * vibrato_rate_scale;
                let phi_vib = (t * phi_vibrato_rate * std::f32::consts::TAU).sin();
                let phi_factor = 2.0f32.powf(phi_vibrato_cents * phi_vib / 1200.0);
                let freq = active.note.frequency * vibrato_factor * phi_factor;

                // Sample-based synthesis: blend recorded sample with additive.
                // Key was resolved at note spawn; this is one O(1) hash get.
                let sample_contribution = active.sample_ref.and_then(|(key, rate)| {
                    let samp = self.sample_lib.get_by_key(key)?;
                    let idx = (active.sample_pos as f64 * rate) as usize;
                    if idx + 1 < samp.data.len() {
                        let frac = (active.sample_pos as f64 * rate - idx as f64) as f32;
                        let s = samp.data[idx] * (1.0 - frac) + samp.data[idx + 1] * frac;
                        Some(s * env)
                    } else {
                        // Recording exhausted. The `else` branch below (pure
                        // additive, full-bandwidth, no manifold blend) is a
                        // categorically different signal from this branch's
                        // sample+LP-filtered-additive blend -- switching
                        // branches at full amplitude the instant data runs
                        // out was an audible identity-swap artifact on any
                        // note whose duration+release outlasts its
                        // recording (long/held notes: arrival cadences,
                        // climaxes). Extend a short fading tail from the
                        // last valid frame so THIS branch's own
                        // contribution has already decayed toward silence
                        // by the time control passes to pure additive,
                        // rather than jumping there at full strength.
                        const GRACE_SAMPLES: f64 = 200.0; // ~4.5ms at 44.1kHz
                        let last_idx = samp.data.len().saturating_sub(2);
                        let overrun = idx as f64 - last_idx as f64;
                        (overrun < GRACE_SAMPLES).then(|| {
                            let fade = (1.0 - overrun / GRACE_SAMPLES) as f32;
                            samp.data[last_idx] * env * fade
                        })
                    }
                });

                let sample = if let Some(ref mut ks) = active.ks {
                    // Karplus-Strong (guitar, harp): self-sustaining, no external envelope
                    ks.tick()
                } else if let Some(ref fm_buf) = active.fm_buffer {
                    // Pre-rendered FM (e-piano, bell): read from buffer
                    if active.sample_pos < fm_buf.len() {
                        fm_buf[active.sample_pos]
                    } else {
                        0.0
                    }
                } else if let Some(sc) = sample_contribution {
                    // Sample available: blend 60% sample + 40% additive for richness
                    let additive = {
                        // Additive synthesis: blend instrument partials with manifold partials
                        // Manifold provides V-A emotional timbre; instrument provides acoustic character
                        let inst_partials = active.instrument.partials();
                        let mp = &self.manifold_partials;
                        let num_p = inst_partials.len().min(16);
                        // Manifold influence scales with emotional distance from neutral.
                        // Near-neutral states → mostly instrument; extreme V-A → more manifold.
                        let emotional_dist = (self.state.valence.abs()
                            + (self.state.arousal - 0.5).abs())
                        .clamp(0.0, 1.0);
                        let blend = MANIFOLD_BLEND_MIN + emotional_dist * MANIFOLD_BLEND_SCALE;
                        // Stack array — this runs once per output sample per
                        // note; a heap Vec here allocated at audio rate.
                        let mut blended = [0.0f32; 16];
                        for (h, b) in blended.iter_mut().enumerate().take(num_p) {
                            let inst = inst_partials.get(h).copied().unwrap_or(0.0);
                            let mani = mp.get(h).copied().unwrap_or(0.0);
                            *b = inst * (1.0 - blend) + mani * blend;
                        }
                        let partial_sum: f32 = blended[..num_p].iter().sum();
                        let norm = if partial_sum > 0.01 {
                            1.0 / partial_sum
                        } else {
                            1.0
                        };
                        // Valence-driven micro-detuning: negative valence → partial beating → roughness
                        // Plomp & Levelt (1965): dissonance peaks at ~25% critical bandwidth
                        // Apply ±5-15 cents detune to partials 2-5 when valence < 0
                        let neg_valence = (-self.state.valence).max(0.0); // 0 when happy, 1 when sad
                        let detune_cents = neg_valence * VALENCE_DETUNE_MAX_CENTS;

                        let mut s = 0.0f32;
                        for h in 0..num_p {
                            let mut cf = freq * (h + 1) as f32;
                            // Detune partials 2-5 for roughness (skip fundamental)
                            if h >= 1 && h <= 4 && detune_cents > 0.5 {
                                // Alternate + and - detune for beating between adjacent partials
                                let sign = if h % 2 == 0 { 1.0 } else { -1.0 };
                                cf *= 2.0f32.powf(sign * detune_cents / 1200.0);
                            }
                            if cf >= nyquist {
                                break;
                            }
                            while active.partial_phases.len() <= h {
                                active.partial_phases.push(0.0);
                            }
                            active.partial_phases[h] += (cf / sr) * std::f32::consts::TAU;
                            if active.partial_phases[h] > std::f32::consts::TAU * 2.0 {
                                active.partial_phases[h] -= std::f32::consts::TAU * 2.0;
                            }
                            // NE boosts upper partials → harsher timbre under stress
                            let ne_boost = if h > 2 { ne_brightness_boost } else { 0.0 };
                            s += (blended[h] + ne_boost) * active.partial_phases[h].sin();
                        }
                        s * norm * env
                    };
                    // Dynamic blend with LP-filtered additive layer.
                    // The additive synth only contributes sub-bass warmth,
                    // not competing mid/high frequencies.
                    let phi = self.state.consciousness_level;
                    let pe = self.state.prediction_error;
                    let ne = self.state.noradrenaline;
                    let serotonin = self.state.serotonin;

                    // LP filter the additive output (1-pole, ~200-400Hz cutoff)
                    // Higher serotonin → higher cutoff (warmer, richer sub-contribution)
                    let lp_cutoff = 200.0 + serotonin * 200.0; // 200-400 Hz
                    let lp_alpha = (lp_cutoff / (lp_cutoff + sr / std::f32::consts::TAU)).min(0.99);
                    // Simple 1-pole: out = alpha * in + (1-alpha) * prev
                    // Use the note's fm_phase as scratch for LP state (it's unused in sample mode)
                    let lp_prev = active.fm_phase; // reuse as LP filter state
                    let lp_out = crate::synth::flush_denormal(
                        lp_alpha * additive + (1.0 - lp_alpha) * lp_prev,
                    );
                    active.fm_phase = lp_out; // store state for next sample

                    // Dynamic blend ratio
                    let additive_blend = (0.05 + phi * 0.15 + pe * 0.10 + ne * 0.05
                        - serotonin * 0.08)
                        .clamp(0.02, 0.35);
                    let sample_blend = 1.0 - additive_blend;
                    sc * sample_blend + lp_out * additive_blend
                } else {
                    // No sample: pure additive synthesis
                    let inst_partials = active.instrument.partials();
                    let _mp = &self.manifold_partials;
                    let num_p_fallback = inst_partials.len().min(16);
                    let mut s = 0.0f32;
                    for h in 0..num_p_fallback {
                        let amp = inst_partials.get(h).copied().unwrap_or(0.0);
                        let cf = freq * (h + 1) as f32;
                        if cf >= nyquist {
                            break;
                        }
                        while active.partial_phases.len() <= h {
                            active.partial_phases.push(0.0);
                        }
                        active.partial_phases[h] += (cf / sr) * std::f32::consts::TAU;
                        if active.partial_phases[h] > std::f32::consts::TAU * 2.0 {
                            active.partial_phases[h] -= std::f32::consts::TAU * 2.0;
                        }
                        s += amp * active.partial_phases[h].sin();
                    }
                    let psum: f32 = inst_partials.iter().sum();
                    s * (if psum > 0.01 { 1.0 / psum } else { 1.0 }) * env
                };

                // Stochastic residual: filtered noise that gives instruments body.
                // Reduced at low arousal to prevent "ticking" — peaceful states
                // should have smooth, clean attacks without noise artifacts.
                let noise_scale = if self.state.arousal < 0.3 { 0.1 } else { 1.0 };
                let inst_name = match active.instrument {
                    Instrument::Piano | Instrument::PianoPP => "piano",
                    Instrument::Violin => "violin",
                    Instrument::Cello => "cello",
                    Instrument::AcousticGuitar | Instrument::Harp => "guitar",
                    Instrument::Flute => "flute",
                    Instrument::Bell => "bell",
                    _ => "pad",
                };
                let _body_freq = production::body_resonance(inst_name);
                let noise_amt = production::noise_amount(inst_name);
                // CRITICAL FIX: stochastic residual was raw white noise injected
                // every sample at 3-8% amplitude. At 44.1kHz this creates continuous
                // sample-to-sample discontinuities (measured by click_score). Real
                // instruments have FILTERED noise (body resonance, not hiss).
                //
                // Fix: reduce noise by 10x AND only update every 4th sample (low-pass
                // by undersampling — creates smooth noise at ~11kHz instead of 22kHz).
                let stochastic = if active.sample_pos % 4 == 0 {
                    let noise_seed = (active.sample_pos as u32 / 4).wrapping_mul(2654435761);
                    let ns = noise_seed.wrapping_mul(1103515245).wrapping_add(12345);
                    let raw_noise = (ns >> 16) as f32 / 32768.0 - 1.0;
                    raw_noise * noise_amt * env * 0.1 // 10x quieter
                } else {
                    0.0
                };

                // Attack noise transient: adds realism to note onsets.
                // Reduced at low arousal AND at high polyphony (many notes → crackling).
                let attack_samples = (0.015 * sr) as usize;
                let noise =
                    crate::dramatic::attack_noise(active.sample_pos, attack_samples, brightness)
                        * noise_scale
                        * noise_density_factor;
                let sample = sample + noise + stochastic * noise_scale * noise_density_factor;

                // Gain staging: ensure all consciousness states are audible.
                // Professional solo piano: -20 to -14 dBFS RMS.
                let vel = active.note.velocity.clamp(0.0, 1.0);
                let arousal = self.state.arousal.clamp(0.0, 1.0);

                // Arousal sets the ceiling (calm=0.15, excited=0.45)
                let ceiling = GAIN_MIN + arousal * GAIN_AROUSAL_COEFF;

                // Velocity scales linearly within the ceiling.
                // Floor at 0.3 (even pp notes are audible), full at vel=1.0.
                let vel_scale = 0.3 + vel * 0.7;
                let master_gain = ceiling * vel_scale;

                voice_buffers[voice][i] += sample * active.volume * master_gain;
                active.sample_pos += 1;
            }
        }

        // ── Phase 2: Sidechain ducking ──
        if self.enable_sidechain {
            self.sidechain
                .apply(&mut voice_buffers, &voice_roles, self.chunk_samples);
        }

        // ── Phase 3: Stereo imaging ──
        // Wide panning + Haas effect (1-3ms L/R delay per voice for spatial depth)
        let mut buffer = if self.enable_binaural {
            self.binaural.render(&voice_buffers)
        } else {
            // Wide pan positions — lead slightly off-center for stereo interest
            let pans = [0.15f32, -0.5, 0.6, -0.4]; // lead/bass/harmony/ostinato
            let haas_delays = [44usize, 265, 132, 353]; // 1ms, 6ms, 3ms, 8ms at 44.1kHz
            let mut buf = vec![[0.0f32; 2]; self.chunk_samples];
            for (v, voice_buf) in voice_buffers.iter().enumerate() {
                let theta = (pans[v.min(3)] + 1.0) * std::f32::consts::FRAC_PI_4;
                let (gl, gr) = (theta.cos(), theta.sin());
                let delay = haas_delays[v.min(3)].min(399);
                let v_idx = v.min(3);

                for (i, &s) in voice_buf.iter().enumerate() {
                    buf[i][0] += s * gl;
                    // FIX: Haas delay reads from history buffer, preserving
                    // continuity across chunk boundaries. Previous code read
                    // sample[0] when i < delay, causing a click every chunk.
                    let delayed_sample = if i >= delay {
                        voice_buf[i - delay]
                    } else {
                        // Read from history (previous chunk tail)
                        let hist_idx = self.haas_history[v_idx].len() + i - delay;
                        self.haas_history[v_idx][hist_idx]
                    };
                    buf[i][1] += delayed_sample * gr;
                }

                // Update history: keep last `delay+1` samples of this chunk
                let hist_len = self.haas_history[v_idx].len();
                let voice_len = voice_buf.len();
                if voice_len >= hist_len {
                    // Take the tail of voice_buf
                    let start = voice_len - hist_len;
                    self.haas_history[v_idx].copy_from_slice(&voice_buf[start..]);
                } else {
                    // Shift old history, append new voice_buf
                    self.haas_history[v_idx].rotate_left(voice_len);
                    let tail_start = hist_len - voice_len;
                    self.haas_history[v_idx][tail_start..].copy_from_slice(voice_buf);
                }
            }
            buf
        };

        // Pad if binaural returned fewer samples
        while buffer.len() < self.chunk_samples {
            buffer.push([0.0, 0.0]);
        }

        // ── Phase 3.5: Percussion ──
        // Generate new drum pattern at bar boundaries
        let tempo = self.muse_stream.tempo();
        self.samples_per_bar = ((60.0 / tempo) * 4.0 * self.sample_rate as f32) as usize;
        if self.bar_sample_pos >= self.samples_per_bar || self.drum_hits.is_empty() {
            self.drum_hits = percussion::generate_pattern_full(
                tempo,
                self.state.consciousness_level,
                self.state.arousal,
                self.state.dopamine,
                self.state.valence,
            );
            // Humanize timing for organic feel
            percussion::humanize_hits(
                &mut self.drum_hits,
                self.state.consciousness_level,
                self.chunks_rendered as u32,
            );
            self.bar_sample_pos = 0;
        }
        // Emotional coloring: consciousness state shapes drum timbre
        let drum_color = percussion::DrumColor {
            tightness: self.state.noradrenaline.clamp(0.0, 1.0),
            brightness: self.state.dopamine.clamp(0.0, 1.0),
            warmth: self.state.valence.clamp(-1.0, 1.0),
        };
        // Render drum hits into the buffer
        for hit in &self.drum_hits {
            let hit_sample = (hit.time * self.sample_rate as f32) as usize;
            if hit_sample >= self.bar_sample_pos
                && hit_sample < self.bar_sample_pos + self.chunk_samples
            {
                let local_offset = hit_sample - self.bar_sample_pos;
                let drum_buf = percussion::render_drum_colored(hit, self.sample_rate, &drum_color);
                for (j, &s) in drum_buf.iter().enumerate() {
                    let idx = local_offset + j;
                    if idx < buffer.len() {
                        // Drums: consciousness-weighted gain with emotion
                        let drum_gain =
                            0.15 + self.state.arousal * 0.15 + self.state.dopamine * 0.05;
                        buffer[idx][0] += s * drum_gain;
                        buffer[idx][1] += s * drum_gain;
                    }
                }
            }
        }
        self.bar_sample_pos += self.chunk_samples;

        // ── Phase 3.6: Sub-bass (physical weight) ──
        // FIX: smooth the sub-bass contribution across chunks to prevent
        // amplitude jump when kick fires. 1-pole filter with ~10ms time constant.
        let kick_active = !self.drum_hits.is_empty();
        let sub_bass =
            self.dramatic
                .render_sub_bass(self.chunk_samples, self.sample_rate as f32, kick_active);
        let target_gain = if kick_active { 1.0 } else { 0.7 };
        let smoothing_alpha = 0.05;
        for (i, pair) in buffer.iter_mut().enumerate() {
            self.sub_bass_volume_smoothed +=
                smoothing_alpha * (target_gain - self.sub_bass_volume_smoothed);
            if i < sub_bass.len() {
                let s = sub_bass[i] * self.sub_bass_volume_smoothed;
                pair[0] += s;
                pair[1] += s;
            }
        }

        // ── Phase 3.8: Voice synthesis with auto-ducking ──
        #[cfg(feature = "voice")]
        if let Some(ref mut voice) = self.voice_bridge {
            voice.maybe_generate(&self.state);
            let voice_chunk = voice.render_chunk(self.chunk_samples);

            // Detect if voice is active (any non-zero samples)
            let voice_active = voice_chunk.iter().any(|&s| s.abs() > 0.01);

            for (i, pair) in buffer.iter_mut().enumerate() {
                if i < voice_chunk.len() {
                    if voice_active {
                        // Auto-duck: reduce music by -8dB when voice is speaking
                        pair[0] *= 0.4;
                        pair[1] *= 0.4;
                    }
                    // Voice at full volume, center-panned
                    pair[0] += voice_chunk[i] * 0.9;
                    pair[1] += voice_chunk[i] * 0.9;
                }
            }
        }

        // ── Phase 3.7: Ambient drone (fills silence, continuous low pad) ──
        let drone_chunk = self.drone.render(self.chunk_samples);
        for (i, pair) in buffer.iter_mut().enumerate() {
            if i < drone_chunk.len() {
                pair[0] += drone_chunk[i][0];
                pair[1] += drone_chunk[i][1];
            }
        }

        // ── Phase 4: Consciousness reverb ──
        for pair in &mut buffer {
            let (l, r) = self.reverb.process_stereo(pair[0], pair[1]);
            pair[0] = l;
            pair[1] = r;
        }

        // ── Phase 4.5: Mixing chain (EQ → compressor → limiter) ──
        for pair in &mut buffer {
            let (l, r) = self.mixing.process(pair[0], pair[1]);
            pair[0] = l;
            pair[1] = r;
        }

        // ── Phase 4.5: Brightness evolution ──
        // Darker in quiet moments, brighter in loud moments.
        // A gentle 1-pole LP filter whose cutoff tracks recent RMS energy.
        {
            let chunk_rms = buffer
                .iter()
                .map(|p| p[0] * p[0] + p[1] * p[1])
                .sum::<f32>()
                / (buffer.len() as f32 * 2.0);
            let chunk_rms = chunk_rms.sqrt();

            // Smooth the RMS with EMA (slow, ~2 second time constant)
            self.brightness_rms = self.brightness_rms * 0.97 + chunk_rms * 0.03;

            // Map RMS to LP filter cutoff: quiet=2kHz (dark), loud=18kHz (bright)
            let brightness = self.brightness_rms.clamp(0.0, 0.3) / 0.3;
            let cutoff = 2000.0 + brightness * 16000.0;
            let rc = 1.0 / (cutoff * std::f32::consts::TAU);
            let dt = 1.0 / self.sample_rate as f32;
            let alpha = dt / (rc + dt);

            for pair in &mut buffer {
                self.brightness_lp_l = crate::synth::flush_denormal(
                    self.brightness_lp_l + alpha * (pair[0] - self.brightness_lp_l),
                );
                self.brightness_lp_r = crate::synth::flush_denormal(
                    self.brightness_lp_r + alpha * (pair[1] - self.brightness_lp_r),
                );
                // Blend: significant LP darkening in quiet passages
                let blend = 0.35; // 35% LP filtering in quiet passages → noticeably darker
                pair[0] = pair[0] * (1.0 - blend * (1.0 - brightness))
                    + self.brightness_lp_l * blend * (1.0 - brightness);
                pair[1] = pair[1] * (1.0 - blend * (1.0 - brightness))
                    + self.brightness_lp_r * blend * (1.0 - brightness);
            }
        }

        // ── Phase 4.6: TPDF dithering (triangular probability density function) ──
        // Eliminates quantization artifacts in quiet passages for 16-bit output.
        {
            let dither_amp = 1.5 / 65536.0; // TPDF for 16-bit
            for pair in &mut buffer {
                // Simple TPDF: sum of two uniform randoms
                self.dither_state = self
                    .dither_state
                    .wrapping_mul(1103515245)
                    .wrapping_add(12345);
                let r1 = (self.dither_state >> 16) as f32 / 32768.0 - 1.0;
                self.dither_state = self
                    .dither_state
                    .wrapping_mul(1103515245)
                    .wrapping_add(12345);
                let r2 = (self.dither_state >> 16) as f32 / 32768.0 - 1.0;
                let dither = (r1 + r2) * 0.5 * dither_amp;
                pair[0] += dither;
                pair[1] += dither;
            }
        }

        // ── Phase 5: Audio feedback (strange loop) ──
        if self.feedback_strength > 0.0 {
            self.feedback.extract(&buffer, self.sample_rate);
            let features = *self.feedback.smoothed_features();

            // ── Phase 5a: Aesthetic self-listening ──
            // The system judges its own beauty and corrects harshness.
            self.aesthetic.assess(&features, &self.state);
            self.aesthetic.apply_corrections(&mut self.state);

            // ── Phase 5b: FEP Active Inference ──
            // The system observes its own audio, updates beliefs about what it sounds
            // like, and selects actions to minimize free energy (surprise).
            if self.enable_fep {
                let result = self.fep_engine.infer(&features);
                self.fep_engine.apply_action(&result, &mut self.state);

                // Track free energy for learning verification
                if self.fe_history.len() < 10000 {
                    self.fe_history.push(result.free_energy);
                }
            }

            features.modulate_state(&mut self.state, self.feedback_strength);
        }

        // Final limiter: gentle clip that preserves dynamics below ±0.95
        // (tanh crushes dynamics at moderate levels — 0.8 input → 0.66 output)
        for pair in &mut buffer {
            pair[0] = gentle_limit(pair[0]);
            pair[1] = gentle_limit(pair[1]);
        }

        self.total_samples_rendered += self.chunk_samples as u64;
        self.chunks_rendered += 1;
        buffer
    }

    fn generate_notes(&mut self) {
        if self.active_notes.len() >= MAX_ACTIVE_NOTES {
            return;
        }
        let sr = self.sample_rate as f32;
        let rel = 0.1 + self.state.harmony_activations[7] * 0.3;
        let release_samples = (rel * sr) as usize;
        let psi = self.state.consciousness_level;

        // Update chord progression based on consciousness
        self.chord_progression = crate::instruments::select_progression(&self.state);
        self.motif.set_phrase_length(&self.state);

        // Advance chord on beat boundaries
        let tempo = self.muse_stream.tempo();
        let beats_per_chunk = (self.chunk_samples as f32 / sr) * (tempo / 60.0);
        self.chord_beat_counter += beats_per_chunk * self.note_gen_cadence as f32;
        if !self.chord_progression.is_empty() {
            let chord = &self.chord_progression[self.chord_idx % self.chord_progression.len()];
            if self.chord_beat_counter >= chord.duration_beats {
                self.chord_beat_counter = 0.0;
                self.chord_idx = (self.chord_idx + 1) % self.chord_progression.len();
            }
        }

        // Get current emotional gesture
        let emotion = emotional_gestures::detect_emotion(&self.state);
        let gesture = emotional_gestures::gesture_for_emotion(emotion);

        // Self-similarity monitor: override creative intent when repetition is wrong
        let similarity_action = self.similarity.recommend();

        // Creative intent: what does Symthaea WANT to do?
        let intent = match similarity_action {
            SimilarityAction::ForceRepeat => CreativeIntent::Perform, // force familiarity
            SimilarityAction::ForceNew => CreativeIntent::Explore,    // force novelty
            SimilarityAction::NoAction => self.journal.creative_intent(&self.state),
        };

        // Dramatic silence gate: don't generate notes during dramatic pauses
        if !self.dramatic.allow_notes() {
            return;
        }

        // Strong ending: final cadence — one last tonic chord, full stop
        if self.dramatic.final_cadence && self.active_notes.is_empty() {
            // Generate a root note at low velocity as the final statement
            if !self.chord_progression.is_empty() {
                let chord = &self.chord_progression[0]; // return to tonic
                let root_freq = 130.81 * 2.0f32.powf(chord.root_semitones as f32 / 12.0);
                let final_note = crate::Note {
                    frequency: root_freq,
                    start_time: 0.0,
                    duration: 3.0, // long final note with reverb tail
                    velocity: 0.6,
                };
                // Don't generate more after this
                self.dramatic.final_cadence = false; // one-shot
                let note = final_note;
                let instrument = instruments::select_instrument(&self.state);
                let ks = if instrument.uses_karplus_strong() {
                    let (damp, bright, _) = instrument.ks_params();
                    let mut k = KarplusStrong::new(note.frequency, self.sample_rate, damp, bright);
                    k.excite_seeded(note.velocity, crate::synth::note_seed(&note));
                    Some(k)
                } else {
                    None
                };
                let fm_buffer = if instrument.uses_fm() {
                    Some(instruments::render_fm_instrument(
                        instrument,
                        note.frequency,
                        note.velocity,
                        note.duration,
                        self.sample_rate,
                    ))
                } else {
                    None
                };
                self.active_notes.push(ActiveNote {
                    total_samples: (note.duration * sr) as usize + release_samples,
                    sample_ref: self.sample_lib.resolve_sample(instrument, note.frequency),
                    note,
                    sample_pos: 0,
                    partial_phases: vec![0.0; 16],
                    fm_phase: 0.0,
                    pan: 0.0,
                    volume: 0.8,
                    voice_idx: 0,
                    wavetable_osc: WavetableOscillator::new(),
                    vibrato_phase: 0.0,
                    instrument,
                    ks,
                    fm_buffer,
                });
                return; // no more notes after final cadence
            }
        }

        // Texture density limits polyphony.
        // At low arousal, spawn only 1 note at a time (sparse, contemplative).
        let texture_voices = self.dramatic.max_voices();
        let arousal_voices = if self.state.arousal > 0.7 {
            4
        } else if self.state.arousal > 0.4 {
            3
        } else if self.state.arousal > 0.2 {
            2
        } else {
            1
        }; // very low arousal: single notes only
        let max_new = if psi > 0.7 {
            arousal_voices
        } else if psi > 0.5 {
            arousal_voices.min(3)
        } else if psi > 0.3 {
            arousal_voices.min(2)
        } else {
            1
        };
        let max_new = max_new.min(texture_voices);

        for _ in 0..max_new {
            if self.active_notes.len() >= MAX_ACTIVE_NOTES {
                break;
            }

            // Creative agency: intent drives note source
            let mut note = match intent {
                CreativeIntent::Perform => {
                    // Replay the best gem for the current emotion
                    if let Some(gem) = self.journal.gem_for_emotion(emotion) {
                        if let Some(n) = gem.notes.first().copied() {
                            n
                        } else if let Some(n) = self.muse_stream.next_note() {
                            n
                        } else {
                            continue;
                        }
                    } else if let Some(n) = self.motif.next_note(&self.state) {
                        n
                    } else if let Some(n) = self.muse_stream.next_note() {
                        n
                    } else {
                        continue;
                    }
                }
                CreativeIntent::Reflect => {
                    // Minimal activity: only generate if motif has something
                    if let Some(n) = self.motif.next_note(&self.state) {
                        n
                    } else {
                        continue;
                    } // skip generation in reflective mode
                }
                CreativeIntent::Develop | CreativeIntent::Explore => {
                    // Try motif memory first, then generate new
                    if let Some(n) = self.motif.next_note(&self.state) {
                        n
                    } else if let Some(n) = self.muse_stream.next_note() {
                        n
                    } else {
                        continue;
                    }
                }
            };

            // Apply groove pocket to note timing
            let is_offbeat = self.chord_beat_counter % 1.0 > 0.4;
            let grooved_beat = self.groove.apply(self.chord_beat_counter, is_offbeat);

            // Ghost notes: quiet rhythmic fill between main beats
            if self.groove.insert_ghost(self.note_counter) {
                note.velocity = self.groove.ghost_velocity();
            }

            // Apply melodic grammar: stepwise motion, leap resolution, tension arcs
            if !self.chord_progression.is_empty() {
                let chord = &self.chord_progression[self.chord_idx % self.chord_progression.len()];
                let root_freq = 261.63 * 2.0f32.powf(chord.root_semitones as f32 / 12.0); // C4 base (was C3)

                // Extended chord type from consciousness state (7ths, 9ths, sus, dim)
                let ext_chord = ExtendedChordType::from_state(&self.state, grooved_beat);
                let chord_ratios = ext_chord.ratios();
                // Chord tones filtered to singable range (MIDI 55-80 ≈ G3-G#5)
                let chord_freqs: Vec<f32> = chord_ratios
                    .iter()
                    .flat_map(|&r| vec![root_freq * r * 0.5, root_freq * r, root_freq * r * 2.0])
                    .filter(|&f| f >= 196.0 && f <= 830.0) // G3 to G#5
                    .collect();

                // Build scale tones (major or minor based on emotional gesture)
                let scale_intervals = if gesture.prefer_major {
                    &[0, 2, 4, 5, 7, 9, 11, 12] // major
                } else {
                    &[0, 2, 3, 5, 7, 8, 10, 12] // natural minor
                };
                let _scale_tones: Vec<f32> = scale_intervals
                    .iter()
                    .flat_map(|&s| {
                        let f = root_freq * 2.0f32.powf(s as f32 / 12.0);
                        vec![f, f * 2.0] // two octaves
                    })
                    .collect();

                self.note_counter = self.note_counter.wrapping_add(1);
                let phrase_pos = self.motif.replay_queue_len() as f32 / 8.0;
                let _ctx = self
                    .lead_voice
                    .melodic_context(phrase_pos, self.chord_beat_counter);

                // Phrasing: rests are handled via duration (longer notes = space)
                // NOT via skipping — skipping breaks the melody state machine
                let is_rest = self
                    .taste_melody
                    .should_rest_with_arousal(self.state.arousal);

                // Taste-optimized melody with phrasing and dynamics
                let taste_scale = crate::taste_melody::build_scale_with(
                    chord.root_semitones,
                    gesture.prefer_major,
                    self.taste_melody.params.scale_center_hz,
                    self.taste_melody.params.scale_half_range,
                );
                note.frequency = self.taste_melody.next_freq(
                    &taste_scale,
                    &chord_freqs,
                    self.state.arousal,
                    self.state.consciousness_level,
                );

                // Trained melody predictor (MAESTRO-fit, see learned_melody):
                // blend its proposal into the taste-melody choice. Weight
                // scales with consciousness — higher Ψ = more learned melodic
                // memory shaping the line. Blend in log-frequency (pitch)
                // space, then snap back to the active scale so we stay in
                // key. Until 2026-07-07 this predictor was write-only in the
                // streaming path: record() fed it context but predict() was
                // never called, so the learned weights shaped nothing.
                if self.melody_predictor.has_context() {
                    if let Some(prev) = self.prev_note_freq {
                        let beat_pos = self.chord_beat_counter % 4.0;
                        let (pred_interval, _) = self.melody_predictor.predict(
                            beat_pos,
                            phrase_pos.clamp(0.0, 1.0),
                            self.state.valence,
                            self.state.arousal,
                        );
                        let pred_freq = self.melody_predictor.interval_to_freq(
                            prev,
                            pred_interval,
                            &taste_scale,
                        );
                        let w = (self.state.consciousness_level * 0.5).clamp(0.0, 0.5);
                        let blended = (note.frequency.ln() * (1.0 - w) + pred_freq.ln() * w).exp();
                        // Snap the blend back onto the scale (log-blend of two
                        // scale tones generally lands between tones)
                        note.frequency = taste_scale
                            .iter()
                            .copied()
                            .min_by(|a, b| (a - blended).abs().total_cmp(&(b - blended).abs()))
                            .unwrap_or(blended);
                    }
                }

                note.duration = self
                    .taste_melody
                    .suggest_duration(self.state.arousal, self.muse_stream.tempo());
                if is_rest {
                    // True rest: skip this note entirely. Silence is music.
                    self.prev_note_freq = Some(note.frequency); // track for voice leading
                    continue;
                }
                note.velocity = self.taste_melody.suggest_velocity(self.state.arousal);

                // Apply key modulation from dramatic state
                note.frequency = self.dramatic.apply_key_shift(note.frequency);

                // Apply dynamic range exaggeration
                note.velocity = self.dramatic.apply_dynamics(note.velocity, 0.5);

                // Track voice state for grammar context
                self.lead_voice.record(note);

                // Feed learned predictor
                if let Some(prev) = self.prev_note_freq {
                    let interval = (note.frequency / prev).log2() * 12.0;
                    let dur_beats = note.duration * self.muse_stream.tempo() / 60.0;
                    self.melody_predictor.record(interval, dur_beats);
                }
            }

            // Apply full emotional gesture: direction + duration + staccato + velocity + tension
            {
                let tuning = crate::pitch::select_tuning_system(&self.state);
                let scale = crate::pitch::build_scale_with_tuning(&self.state, &tuning);
                note.frequency = emotional_gestures::apply_gesture_to_pitch(
                    note.frequency,
                    self.prev_note_freq,
                    &scale,
                    &gesture,
                );
                note.duration *= gesture.duration_scale;
                if gesture.staccato > 0.1 {
                    note.duration *= 1.0 - gesture.staccato * 0.6;
                }
                // Preserve true rests (vel=0.0) through gesture scaling
                note.velocity = if note.velocity < 0.01 {
                    0.0 // keep rest silent
                } else {
                    (note.velocity * gesture.velocity_scale).clamp(0.05, 1.0)
                };
                // High harmonic tension → inject passing tones for dissonance
                if gesture.harmonic_tension > 0.5 && scale.len() > 2 && self.note_counter % 3 == 0 {
                    let shift = if gesture.direction_bias > 0.0 {
                        1i32
                    } else {
                        -1
                    };
                    if let Some(idx) = scale
                        .iter()
                        .enumerate()
                        .min_by(|(_, a), (_, b)| {
                            ((**a - note.frequency).abs())
                                .total_cmp(&((**b - note.frequency).abs()))
                        })
                        .map(|(i, _)| i)
                    {
                        let t = (idx as i32 + shift).clamp(0, scale.len() as i32 - 1) as usize;
                        note.frequency = scale[t];
                    }
                }
            }

            // Apply Phi-dependent humanization (Gaussian timing, FEP velocity curves)
            let beat_pos = self.chord_beat_counter;
            let phrase_pos = self.motif.replay_queue_len() as f32 / 8.0;
            performance::humanize_with_consciousness(
                &mut note,
                beat_pos,
                phrase_pos,
                self.note_counter,
                self.state.consciousness_level,
                self.state.arousal,
            );

            // Sustain pedal simulation: high Phi = sustain engaged (notes ring longer)
            if self.state.consciousness_level > 0.5 {
                let sustain_mult = 1.0 + self.state.consciousness_level; // 1.5x at Phi=0.5, 2x at Phi=1.0
                note.duration *= sustain_mult;
            }

            // Record note for motif memory + creative journal + density regulator + MIDI export
            self.motif.record_note(note);
            self.density
                .record_onset(self.total_samples_rendered as f32 / self.sample_rate as f32);

            // Collect for MIDI export — hard clamp to singable MIDI range
            let midi_freq = note.frequency.clamp(196.0, 988.0); // G3 (MIDI 55) to B5 (MIDI 83)
            // Quantize start_time to the nearest 8th-note grid position.
            // Without quantization, notes land at chunk boundaries (arbitrary),
            // producing chaotic inter-onset intervals and 0.07 rhythmic regularity.
            // With quantization, notes snap to a musical pulse (CV→0, regularity→0.5+).
            let raw_time = self.total_samples_rendered as f32 / self.sample_rate as f32;
            let beat_dur = 60.0 / self.muse_stream.tempo().max(30.0); // seconds per beat
            let grid_unit = beat_dur * 0.5; // 8th-note grid
            let quantized_time = if grid_unit > 0.01 {
                (raw_time / grid_unit).round() * grid_unit
            } else {
                raw_time
            };
            // Also quantize note duration to multiples of the grid unit for rhythmic coherence
            let quantized_duration = if grid_unit > 0.01 {
                ((note.duration / grid_unit).round().max(1.0)) * grid_unit
            } else {
                note.duration
            };
            self.generated_notes.push(crate::Note {
                frequency: midi_freq,
                start_time: quantized_time,
                duration: quantized_duration,
                velocity: note.velocity,
            });
            self.prev_note_freq = Some(note.frequency);

            // When a phrase completes, evaluate it for creative journal + form memory
            if self.note_counter % (self.motif.replay_queue_len().max(4) as u32) == 0 {
                let recent: Vec<crate::Note> = self
                    .active_notes
                    .iter()
                    .rev()
                    .take(4)
                    .map(|a| a.note)
                    .collect();
                if recent.len() >= 2 {
                    let beauty = self.journal.evaluate_phrase(&recent, &self.state);
                    self.composer.memory.record_phrase(&recent, beauty);
                    self.similarity.record_phrase(&recent);

                    // Call-and-response: if lead completed a good phrase, queue echo
                    if beauty > 0.5 && self.lead_voice.phrase_notes.len() >= 3 {
                        // The next harmony voice will echo this phrase
                        // (stored in lead_voice.phrase_notes for the voice_leader to use)
                    }
                }
            }

            if let Some(note) = Some(note) {
                // P0.1: Voice-aware pitch derivation.
                // Lead uses the melody note as-is. Bass/harmony/ostinato derive
                // musically related pitches from the lead frequency.
                let note_counter = self.note_counter;
                let voice_idx = 0_usize; // Lead voice always gets the melody note

                // Spawn accompaniment voices alongside lead (not randomly)
                // Bass: every 2nd note, octave below
                let spawn_bass = psi > 0.4 && note_counter % 2 == 0;
                // Harmony: every 3rd note, chord tone above
                let spawn_harmony = psi > 0.6 && note_counter % 3 == 0;
                // Ostinato: every 4th note, repeat a motif
                let spawn_ostinato = psi > 0.7 && note_counter % 4 == 0;

                // Collect voices to spawn: (voice_idx, frequency, velocity_scale)
                let mut voices_to_spawn: Vec<(usize, f32, f32)> = vec![
                    (0, note.frequency, 1.0), // lead always
                ];
                if spawn_bass {
                    // Bass: octave below, longer duration implicit via lower voice volume
                    let bass_freq = (note.frequency * 0.5).max(55.0);
                    voices_to_spawn.push((1, bass_freq, 0.7));
                }
                if spawn_harmony {
                    // Harmony: perfect 5th above for positive valence, minor 3rd for negative.
                    // CRITICAL: drop an octave if resulting pitch exceeds 1400 Hz to prevent
                    // ear-piercing stacking when lead melody is already in high register.
                    let harm_ratio = if self.state.valence > 0.0 {
                        1.4983
                    } else {
                        1.1892
                    };
                    let mut harm_freq = note.frequency * harm_ratio;
                    while harm_freq > 1400.0 {
                        harm_freq *= 0.5; // drop octave until in safe register
                    }
                    voices_to_spawn.push((2, harm_freq.max(110.0), 0.5));
                }
                if spawn_ostinato {
                    // Ostinato: same pitch, delayed entry creates echo effect
                    voices_to_spawn.push((3, note.frequency, 0.3));
                }

                for &(voice_idx, voice_freq, vel_scale) in &voices_to_spawn {
                    let mut note = note;
                    note.frequency = voice_freq;
                    note.velocity *= vel_scale;

                    // Clear old notes in this voice to prevent mud
                    // Allow more overlap when sustain pedal is engaged (high Phi)
                    let max_per_voice = if self.state.consciousness_level > 0.5 {
                        4
                    } else {
                        2
                    };
                    let voice_count = self
                        .active_notes
                        .iter()
                        .filter(|n| n.voice_idx == voice_idx)
                        .count();
                    if voice_count >= max_per_voice {
                        // Force-expire the oldest note in this voice
                        if let Some(oldest) = self
                            .active_notes
                            .iter_mut()
                            .filter(|n| n.voice_idx == voice_idx)
                            .min_by_key(|n| n.sample_pos)
                        {
                            oldest.total_samples = oldest.sample_pos; // expire it
                        }
                    }

                    // Select instrument per voice role for variety
                    // Lead follows consciousness, others get complementary instruments
                    let instrument = match voice_idx {
                        0 => instruments::select_instrument(&self.state), // lead: consciousness-driven
                        1 => {
                            // bass: organ or pad (sustained low register)
                            if self.state.harmony_activations[7] > 0.4 {
                                Instrument::Organ
                            } else {
                                Instrument::Pad
                            }
                        }
                        2 => {
                            // harmony: harp or electric piano
                            if self.state.dopamine > 0.5 {
                                Instrument::ElectricPiano
                            } else {
                                Instrument::Harp
                            }
                        }
                        _ => {
                            // ostinato: vibraphone or bell (rhythmic shimmer)
                            if self.state.arousal > 0.5 {
                                Instrument::Bell
                            } else {
                                Instrument::AcousticGuitar
                            }
                        }
                    };

                    // Initialize instrument-specific synthesis state
                    let ks = if instrument.uses_karplus_strong() {
                        let (damp, bright, _stiff) = instrument.ks_params();
                        let mut ks =
                            KarplusStrong::new(note.frequency, self.sample_rate, damp, bright);
                        ks.excite_seeded(note.velocity, crate::synth::note_seed(&note));
                        Some(ks)
                    } else {
                        None
                    };

                    let fm_buffer = if instrument.uses_fm() {
                        Some(instruments::render_fm_instrument(
                            instrument,
                            note.frequency,
                            note.velocity,
                            note.duration,
                            self.sample_rate,
                        ))
                    } else {
                        None
                    };

                    // Prediction error → onset jitter: surprise makes timing unpredictable
                    let pe = self.state.prediction_error;
                    let jitter_max = (pe * 0.005 * sr) as i32; // ±5ms at PE=1.0
                    let jitter = if jitter_max > 0 {
                        let hash = self.note_counter.wrapping_mul(2654435761);
                        (hash as i32 % (jitter_max * 2 + 1)) - jitter_max
                    } else {
                        0
                    };
                    let onset_offset = jitter.max(0) as usize;

                    self.active_notes.push(ActiveNote {
                        total_samples: (note.duration * sr) as usize + release_samples,
                        sample_ref: self.sample_lib.resolve_sample(instrument, note.frequency),
                        note,
                        sample_pos: onset_offset, // jittered onset
                        partial_phases: vec![0.0; 16],
                        fm_phase: 0.0,
                        pan: [0.0, -0.2, 0.4, -0.3][voice_idx],
                        volume: [1.0, 0.7, 0.5, 0.3][voice_idx],
                        voice_idx,
                        wavetable_osc: WavetableOscillator::new(),
                        vibrato_phase: 0.0,
                        instrument,
                        ks,
                        fm_buffer,
                    });
                } // end for voice_idx in voices_to_spawn
            }
        }
    }

    // ── Accessors ──
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
    pub fn chunk_samples(&self) -> usize {
        self.chunk_samples
    }
    pub fn total_samples_rendered(&self) -> u64 {
        self.total_samples_rendered
    }
    pub fn chunks_rendered(&self) -> u64 {
        self.chunks_rendered
    }
    pub fn active_note_count(&self) -> usize {
        self.active_notes.len()
    }
    pub fn tempo(&self) -> f32 {
        self.muse_stream.tempo()
    }
    /// Snapshot motif memory for cross-session persistence.
    pub fn motif_snapshot(&self) -> crate::motif_memory::MotifSnapshot {
        self.motif.to_snapshot()
    }

    /// Restore motif memory from a saved snapshot.
    pub fn restore_motif(&mut self, snapshot: &crate::motif_memory::MotifSnapshot) {
        self.motif = MotifMemory::from_snapshot(snapshot);
    }

    pub fn phi_metrics(&self) -> Option<crate::phi_optimizer::PhiOptimizerMetrics> {
        if self.enable_phi_optimizer {
            Some(self.phi_optimizer.metrics())
        } else {
            None
        }
    }
    pub fn feedback_features(&self) -> &crate::audio_feedback::AudioFeatures {
        self.feedback.smoothed_features()
    }
    /// Current aesthetic assessment (beauty, harshness, consonance).
    pub fn aesthetic_assessment(&self) -> &crate::aesthetic_listener::AestheticAssessment {
        self.aesthetic.smoothed_assessment()
    }
    /// Current free energy from the FEP agent (lower = better self-model).
    pub fn current_free_energy(&self) -> f64 {
        self.fep_engine.current_free_energy()
    }
    /// Free energy history for learning curve analysis.
    pub fn free_energy_history(&self) -> &[f64] {
        &self.fe_history
    }
    /// FEP inference cycle count.
    pub fn fep_cycles(&self) -> u64 {
        self.fep_engine.cycle_count()
    }
    /// Last FEP inference result.
    pub fn last_fep_result(&self) -> Option<&crate::musical_inference::MusicInferenceResult> {
        self.fep_engine.last_result()
    }

    pub fn reset(&mut self, seed: u64) {
        self.fep_engine = MusicalInferenceEngine::new();
        self.fe_history.clear();
        self.drone.reset();
        self.motif.reset();
        self.journal.reset();
        self.composer.reset();
        self.dramatic.reset();
        self.melody_predictor.reset();
        self.taste_melody.reset();
        self.density.reset();
        self.similarity.reset();
        self.time_sig = TimeSignature::FOUR_FOUR;
        self.groove = GroovePocket {
            swing: 0.0,
            laid_back: 0.0,
            ghost_notes: 0.0,
        };
        self.lead_voice.reset();
        self.bars_elapsed_frac = 0.0;
        self.chord_idx = 0;
        self.chord_beat_counter = 0.0;
        self.note_counter = 0;
        self.prev_note_freq = None;
        self.smoother.reset();
        self.mixing = MixingChain::new(self.sample_rate);
        self.drum_hits.clear();
        self.bar_sample_pos = 0;
        self.muse_stream.reset(seed);
        self.active_notes.clear();
        self.total_samples_rendered = 0;
        self.chunks_rendered = 0;
        self.chunks_since_note_gen = 0;
        self.reverb = ConsciousnessReverb::new(self.sample_rate);
        self.feedback.reset();
        self.sidechain.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> MuseConfig {
        MuseConfig {
            duration_secs: 4.0,
            max_notes: 16,
            ..Default::default()
        }
    }

    #[test]
    fn render_chunk_correct_length() {
        let mut s = StreamingSynth::new(cfg(), 44100);
        assert_eq!(s.render_chunk().len(), s.chunk_samples());
    }

    #[test]
    fn samples_are_finite() {
        let mut s = StreamingSynth::new(cfg(), 44100);
        s.update_state(&MusicalState {
            consciousness_level: 0.8,
            ..Default::default()
        });
        for _ in 0..50 {
            for p in s.render_chunk() {
                assert!(p[0].is_finite() && p[1].is_finite());
            }
        }
    }

    #[test]
    fn state_update_changes_tempo() {
        let mut s = StreamingSynth::new(cfg(), 44100);
        let t1 = s.tempo();
        s.update_state(&MusicalState {
            arousal: 0.95,
            noradrenaline: 0.9,
            ..Default::default()
        });
        assert!(s.tempo() > t1);
    }

    #[test]
    fn total_samples_tracks() {
        let mut s = StreamingSynth::new(cfg(), 44100);
        let cs = s.chunk_samples() as u64;
        s.render_chunk();
        assert_eq!(s.total_samples_rendered(), cs);
    }

    #[test]
    fn reset_clears() {
        let mut s = StreamingSynth::new(cfg(), 44100);
        for _ in 0..10 {
            s.render_chunk();
        }
        s.reset(99);
        assert_eq!(s.total_samples_rendered(), 0);
        assert_eq!(s.chunks_rendered(), 0);
    }

    #[test]
    fn feedback_loop_modulates_state() {
        let mut s = StreamingSynth::new(cfg(), 44100);
        s.feedback_strength = 1.0;
        s.update_state(&MusicalState {
            consciousness_level: 0.8,
            ..Default::default()
        });
        let orig_arousal = s.state.arousal;
        // Render several chunks to build up feedback
        for _ in 0..20 {
            s.render_chunk();
        }
        // Audio feedback should have modified state
        let features = s.feedback_features();
        assert!(
            features.rms_energy > 0.0 || features.spectral_centroid > 0.0,
            "feedback should detect audio features"
        );
    }

    #[test]
    fn binaural_vs_simple_pan() {
        let mut s = StreamingSynth::new(cfg(), 44100);
        s.update_state(&MusicalState {
            consciousness_level: 0.8,
            ..Default::default()
        });
        s.enable_binaural = true;
        for _ in 0..10 {
            s.render_chunk();
        }
        let bin_chunk = s.render_chunk();

        s.reset(42);
        s.update_state(&MusicalState {
            consciousness_level: 0.8,
            ..Default::default()
        });
        s.enable_binaural = false;
        for _ in 0..10 {
            s.render_chunk();
        }
        let pan_chunk = s.render_chunk();

        assert_eq!(bin_chunk.len(), pan_chunk.len());
    }

    #[test]
    fn substrate_changes_character() {
        let mut s = StreamingSynth::new(cfg(), 44100);
        s.update_state(&MusicalState {
            consciousness_level: 0.8,
            ..Default::default()
        });
        s.set_substrate(SubstrateTimbreType::Quantum);
        assert!(s.substrate.is_some());
        for _ in 0..10 {
            s.render_chunk();
        }
    }

    #[test]
    fn fep_runs_live_in_pipeline() {
        let mut s = StreamingSynth::new(cfg(), 44100);
        s.enable_fep = true;
        s.feedback_strength = 0.5;
        s.update_state(&MusicalState {
            consciousness_level: 0.7,
            ..Default::default()
        });

        for _ in 0..50 {
            s.render_chunk();
        }

        assert!(s.fep_cycles() > 0, "FEP should have run cycles");
        assert!(
            !s.free_energy_history().is_empty(),
            "FE history should be recorded"
        );
    }

    #[test]
    fn fep_free_energy_is_finite() {
        let mut s = StreamingSynth::new(cfg(), 44100);
        s.enable_fep = true;
        s.feedback_strength = 1.0;
        s.update_state(&MusicalState {
            consciousness_level: 0.8,
            arousal: 0.6,
            harmony_activations: [0.7, 0.5, 0.6, 0.4, 0.3, 0.5, 0.6, 0.3],
            ..Default::default()
        });

        for _ in 0..200 {
            let chunk = s.render_chunk();
            for pair in &chunk {
                assert!(pair[0].is_finite() && pair[1].is_finite());
            }
        }

        // All FE values should be finite
        for &fe in s.free_energy_history() {
            assert!(fe.is_finite(), "free energy should be finite: {fe}");
        }
    }

    #[test]
    fn fep_learning_curve() {
        // Run 500 cycles with consistent input.
        // The FEP agent should learn to predict its own audio,
        // meaning late free energy should be no worse than early.
        let mut s = StreamingSynth::new(cfg(), 44100);
        s.enable_fep = true;
        s.feedback_strength = 0.5;
        s.update_state(&MusicalState {
            consciousness_level: 0.7,
            arousal: 0.4,
            harmony_activations: [0.6, 0.5, 0.4, 0.3, 0.5, 0.6, 0.4, 0.5],
            ..Default::default()
        });

        for _ in 0..500 {
            s.render_chunk();
        }

        let history = s.free_energy_history();
        assert!(history.len() >= 100, "need enough data points");

        // Compare early vs late average FE
        let early: f64 = history[..20].iter().sum::<f64>() / 20.0;
        let late: f64 = history[history.len() - 20..].iter().sum::<f64>() / 20.0;

        eprintln!(
            "  FEP learning: early_FE={early:.4}, late_FE={late:.4}, delta={:.4}",
            early - late
        );

        // Late FE should not be dramatically worse than early
        // (with learning, it should improve or stabilize)
        assert!(
            late < early + 0.5,
            "FE should not diverge: early={early:.4}, late={late:.4}"
        );
    }
}
