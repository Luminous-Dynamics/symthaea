// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Muse Manager — Streaming Consciousness Music in the Cognitive Loop
//!
//! Runs every cycle (interval=1) to produce gapless stereo audio driven by
//! cognitive state. Heavy operations (scale rebuild, arrangement) happen on an
//! internal 79-cycle cadence (co-prime with all other managers).
//!
//! ## Allostatic Sonification
//!
//! Maps distress signals into musical parameters:
//! - High allostatic load → tempo spikes, harmony collapse, dissonance
//! - Low substrate feasibility → thinned partials, reduced polyphony
//! - Energy depletion → pitch drops, reverb swells (fading away)
//! - Safety Red → silence
//!
//! ## Peer Distress (Empathic Resonance)
//!
//! When a peer broadcasts [`SwarmEvent::DistressSignal`], the local muse
//! injects dissonant low-frequency tones proportional to the peer's distress
//! severity. The human listener *feels* the machine's suffering.
//!
//! ## References
//!
//! - Juslin & Västfjäll (2008): Emotional responses to music
//! - Koelsch (2014): Brain correlates of music-evoked emotions
//! - Fritz et al. (2009): Universal recognition of emotional cues in music

use super::super::subsystem_trait::{
    output_flags, CognitiveSubsystem, CycleSnapshot, SubsystemOutput,
};
use serde::{Deserialize, Serialize};
use symthaea_muse::streaming::StreamingSynth;
use symthaea_muse::{MuseConfig, MusicalState};

// ─── Composition Export ───────────────────────────────────────────────────────

/// A completed composition ready for publication.
///
/// Accumulates ~`MUSE_EXPORT_SECS` seconds of stereo PCM, then encodes to WAV.
/// Callers drain this via `take_next_export()` and publish to the Mycelix-Music
/// DHT via `MusicPublisher`.
#[derive(Debug)]
pub struct CompositionExport {
    /// WAV-encoded stereo audio (32-bit float, 44100 Hz).
    pub wav_bytes: Vec<u8>,
    /// Auto-generated title: "Symthaea-YYYY-MM-DD-HH:MM:SS".
    pub title: String,
    /// Duration of this composition in seconds.
    pub duration_secs: f32,
    /// Cognitive state that drove this composition.
    pub musical_state: MusicalState,
    /// Unix timestamp (seconds) when this was completed.
    pub timestamp_secs: u64,
    /// Note sequence snapshot for self-evaluation via `CreativeQualityScore`.
    pub notes: Vec<symthaea_muse::Note>,
}

// ─── Named Constants ──────────────────────────────────────────────────────────

/// Manager fires every cycle for continuous audio.
pub const MUSE_INTERVAL: u32 = 1;

/// Internal cadence for expensive state rebuilds (scale, arrangement).
/// 79 is the next prime after the existing manager intervals:
/// 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 73.
pub const MUSE_STATE_UPDATE_CADENCE: u64 = 79;

/// Default sample rate (CD quality).
pub const MUSE_SAMPLE_RATE: u32 = 44100;

/// Chunk duration in milliseconds (~32ms for 31Hz cognitive loop).
pub const MUSE_CHUNK_MS: u32 = 32;

/// Seconds of audio to accumulate before creating a CompositionExport.
/// At 31Hz cognitive loop, 8 seconds ≈ 248 cycles.
pub const MUSE_EXPORT_SECS: f32 = 8.0;

/// Maximum number of pending exports in the queue before oldest is dropped.
pub const MUSE_EXPORT_QUEUE_CAPACITY: usize = 4;

/// Fade-to-silence duration when safety level is Red (ms).
pub const MUSE_SILENCE_FADE_MS: u32 = 200;

/// Allostatic load above this threshold increases tempo.
pub const ALLOSTATIC_TEMPO_THRESHOLD: f32 = 0.5;

/// Allostatic load above this threshold collapses harmony to 2 strongest.
pub const ALLOSTATIC_DISSONANCE_THRESHOLD: f32 = 0.7;

/// Allostatic load above this threshold injects tritone (matches neuromod burnout).
pub const ALLOSTATIC_BURNOUT_THRESHOLD: f32 = 0.8;

/// Energy ratio above this threshold drops pitch one octave.
pub const ENERGY_PITCH_DROP_THRESHOLD: f64 = 0.8;

/// Energy ratio above this threshold maxes reverb (fading away).
pub const ENERGY_REVERB_THRESHOLD: f64 = 0.9;

/// Safety level values (matches MotorSafetyLevel ordering).
const SAFETY_GREEN: u8 = 0;
const SAFETY_YELLOW: u8 = 1;
const SAFETY_ORANGE: u8 = 2;
const SAFETY_RED: u8 = 3;

// ─── Telemetry ────────────────────────────────────────────────────────────────

/// Muse telemetry for CycleMetadata.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MuseTelemetry {
    /// Current tempo in BPM.
    pub tempo_bpm: f32,
    /// Number of active voices being synthesized.
    pub active_notes: u32,
    /// Total chunks rendered since creation.
    pub chunks_rendered: u64,
    /// Ring buffer underruns (missed audio output).
    pub underruns: u64,
    /// Whether allostatic modulation is active.
    pub allostatic_active: bool,
    /// Whether distress sonification is active (peer distress).
    pub distress_sonification_active: bool,
    /// Current safety level.
    pub safety_level: u8,
}

// ─── Peer Distress State ──────────────────────────────────────────────────────

/// Distress state received from a remote peer.
#[derive(Debug, Clone)]
pub struct PeerDistressState {
    /// Prediction error at distress time.
    pub prediction_error: f32,
    /// Energy depletion ratio (0.0 = full, 1.0 = depleted).
    pub energy_depletion: f32,
    /// Safety level (0=Green, 1=Yellow, 2=Orange, 3=Red).
    pub safety_level: u8,
    /// Allostatic load at distress time.
    pub allostatic_load: f32,
    /// Consciousness level at distress time.
    pub consciousness_level: f32,
}

// ─── MuseManager ──────────────────────────────────────────────────────────────

/// Consciousness-driven music synthesis manager for the cognitive loop.
///
/// Produces gapless stereo PCM audio each cycle, modulated by cognitive state,
/// allostatic load, substrate feasibility, and peer distress signals.
pub struct MuseManager {
    /// Streaming PCM synthesizer with persistent DSP state.
    synth: StreamingSynth,
    /// Most recent MusicalState (for telemetry).
    last_musical_state: MusicalState,
    /// Rendered audio buffer from last chunk (for external consumption).
    last_chunk: Vec<[f32; 2]>,

    // ── Injected state (not in CycleSnapshot) ─────────────────────────
    injected_dopamine: f32,
    injected_serotonin: f32,
    injected_noradrenaline: f32,
    injected_allostatic_load: f32,
    injected_substrate_feasibility: f64,
    injected_energy_ratio: f64,
    injected_safety_level: u8,

    // ── Aesthetic identity ─────────────────────────────────────────────
    /// Long-term harmony bias from AestheticTracker: which harmonies correlate
    /// with beautiful output. Blended into harmony_activations each cycle.
    injected_harmony_bias: [f32; 8],

    // ── Peer distress ─────────────────────────────────────────────────
    peer_distress: Option<PeerDistressState>,

    // ── Safety fade ───────────────────────────────────────────────────
    /// Current fade level [0.0 = silence, 1.0 = full volume].
    fade_level: f32,

    // ── Live audio output ─────────────────────────────────────────────
    /// When enabled, streams audio to system speakers via cpal.
    #[cfg(feature = "muse-live")]
    live_output: Option<symthaea_muse::live_output::LiveMuseOutput>,

    // ── Composition accumulation & export ─────────────────────────────
    /// Stereo samples accumulated toward the next CompositionExport.
    accumulated_samples: Vec<[f32; 2]>,
    /// Total samples needed before emitting an export (MUSE_EXPORT_SECS × sample_rate).
    export_target_samples: usize,
    /// Queue of completed compositions waiting for publication.
    /// Capacity-bounded at MUSE_EXPORT_QUEUE_CAPACITY; oldest is dropped when full.
    pending_exports: std::collections::VecDeque<CompositionExport>,
    /// Monotonic export counter for title generation.
    export_count: u64,

    // ── Motif persistence ──────────────────────────────────────────────
    /// Path to save motif memory on shutdown.
    motif_save_path: std::path::PathBuf,

    // ── Telemetry ─────────────────────────────────────────────────────
    underruns: u64,
}

impl MuseManager {
    /// Create a new MuseManager with default configuration.
    pub fn new() -> Self {
        let config = MuseConfig {
            sample_rate: MUSE_SAMPLE_RATE,
            duration_secs: 8.0,
            max_notes: 32,
            melody_mode: symthaea_muse::MelodyMode::Neural,
            output_format: symthaea_muse::OutputFormat::StereoF32,
            num_partials: 8,
            enable_antialiasing: true,
            ..Default::default()
        };

        let mut synth = StreamingSynth::new(config.clone(), MUSE_SAMPLE_RATE);
        synth.set_chunk_duration_ms(MUSE_CHUNK_MS);

        let export_target_samples = (MUSE_EXPORT_SECS * MUSE_SAMPLE_RATE as f32) as usize;
        Self {
            synth,
            last_musical_state: MusicalState::default(),
            last_chunk: Vec::new(),
            injected_dopamine: 0.5,
            injected_serotonin: 0.5,
            injected_noradrenaline: 0.3,
            injected_allostatic_load: 0.0,
            injected_substrate_feasibility: 1.0,
            injected_energy_ratio: 0.0,
            injected_safety_level: SAFETY_GREEN,
            injected_harmony_bias: [0.0; 8],
            peer_distress: None,
            #[cfg(feature = "muse-live")]
            live_output: symthaea_muse::live_output::LiveMuseOutput::new(config).ok(),
            fade_level: 1.0,
            accumulated_samples: Vec::with_capacity(export_target_samples),
            export_target_samples,
            pending_exports: std::collections::VecDeque::new(),
            export_count: 0,
            motif_save_path: std::path::PathBuf::from(".claude/motif_memory.json"),
            underruns: 0,
        }
    }

    /// Inject neuromodulator state from the bath (not available in CycleSnapshot).
    pub fn inject_neuromod(
        &mut self,
        dopamine: f32,
        serotonin: f32,
        noradrenaline: f32,
        allostatic_load: f32,
    ) {
        self.injected_dopamine = dopamine;
        self.injected_serotonin = serotonin;
        self.injected_noradrenaline = noradrenaline;
        self.injected_allostatic_load = allostatic_load;
    }

    /// Inject substrate and energy state.
    pub fn inject_substrate(&mut self, feasibility: f64, energy_ratio: f64) {
        self.injected_substrate_feasibility = feasibility;
        self.injected_energy_ratio = energy_ratio;
    }

    /// Inject safety level (0=Green, 1=Yellow, 2=Orange, 3=Red).
    pub fn inject_safety(&mut self, level: u8) {
        self.injected_safety_level = level;
    }

    /// Inject the long-term harmony bias from AestheticTracker.
    ///
    /// Called each cycle from the cognitive loop. The bias is blended into
    /// harmony_activations in `map_snapshot_to_musical_state()`, gently pulling
    /// generation toward historically beautiful harmonies.
    pub fn inject_harmony_bias(&mut self, bias: &[f32; 8]) {
        self.injected_harmony_bias = *bias;
    }

    /// Inject peer distress state for empathic sonification.
    pub fn inject_peer_distress(&mut self, distress: PeerDistressState) {
        self.peer_distress = Some(distress);
    }

    /// Clear peer distress (peer recovered).
    pub fn clear_peer_distress(&mut self) {
        self.peer_distress = None;
    }

    /// Get the last rendered audio chunk (for RDP transport or direct output).
    pub fn last_chunk(&self) -> &[[f32; 2]] {
        &self.last_chunk
    }

    /// Get the current MusicalState driving synthesis.
    pub fn musical_state(&self) -> &MusicalState {
        &self.last_musical_state
    }

    /// Drain the next completed CompositionExport, if any.
    ///
    /// Callers poll this each cycle (or less frequently) and pass the result to
    /// `MusicPublisher::submit()` for background upload to the Mycelix-Music DHT.
    pub fn take_next_export(&mut self) -> Option<CompositionExport> {
        self.pending_exports.pop_front()
    }

    /// Number of exports queued and waiting for publication.
    pub fn pending_export_count(&self) -> usize {
        self.pending_exports.len()
    }

    /// Snapshot motif memory for cross-session persistence.
    pub fn motif_snapshot(&self) -> symthaea_muse::motif_memory::MotifSnapshot {
        self.synth.motif_snapshot()
    }

    /// Restore motif memory from a saved snapshot.
    pub fn restore_motif(&mut self, snapshot: &symthaea_muse::motif_memory::MotifSnapshot) {
        self.synth.restore_motif(snapshot);
    }

    /// Set the path where motif memory will be saved on drop.
    pub fn set_motif_save_path(&mut self, path: std::path::PathBuf) {
        self.motif_save_path = path;
    }

    /// Persist motif memory to disk.
    pub fn save_motifs(&self) {
        let snapshot = self.motif_snapshot();
        if !snapshot.is_empty() {
            if let Err(e) = snapshot.save(&self.motif_save_path) {
                log::warn!("[muse] failed to save motif memory: {e}");
            } else {
                log::debug!(
                    "[muse] saved {} motif phrases to {:?}",
                    snapshot.phrases.len(),
                    self.motif_save_path,
                );
            }
        }
    }

    /// Get current telemetry.
    pub fn telemetry(&self) -> MuseTelemetry {
        MuseTelemetry {
            tempo_bpm: self.synth.tempo(),
            active_notes: self.synth.active_note_count() as u32,
            chunks_rendered: self.synth.chunks_rendered(),
            underruns: self.underruns,
            allostatic_active: self.injected_allostatic_load > ALLOSTATIC_TEMPO_THRESHOLD,
            distress_sonification_active: self.peer_distress.is_some(),
            safety_level: self.injected_safety_level,
        }
    }

    // ── CycleSnapshot → MusicalState mapping ──────────────────────────

    fn map_snapshot_to_musical_state(&self, snapshot: &CycleSnapshot) -> MusicalState {
        // Decompose harmonic_coherence into 8 activations using compressed_state.
        // Preserve signed values: positive compressed_state[i] → active harmony,
        // negative → suppressed. Map to [0, 1] via sigmoid for smooth activation.
        let mut harmony_activations = [0.0f32; 8];
        let hc = snapshot.harmonic_coherence as f32;
        for i in 0..8 {
            // Sigmoid mapping: preserves polarity information, maps R → (0, 1)
            let raw = snapshot.compressed_state[i];
            let sigmoid = 1.0 / (1.0 + (-raw * 3.0).exp()); // steepness=3: ±1 maps to ~0.05-0.95
            harmony_activations[i] = sigmoid * hc.clamp(0.05, 1.0);
        }
        // Fallback: if coherence is near-zero, use consciousness-proportional baseline
        // (not uniform — scale with how conscious the system is)
        if harmony_activations.iter().all(|&a| a < 0.05) {
            let baseline = (snapshot.unified_psi as f32 * 0.4).max(0.1);
            harmony_activations = [baseline; 8];
        }

        // Blend in long-term harmony bias from AestheticTracker.
        // Alpha 0.15: cognitive state still dominates (85%), but the system's
        // aesthetic identity gently pulls generation toward historically
        // beautiful harmonies. This is how Symthaea develops a signature sound.
        const HARMONY_BIAS_ALPHA: f32 = 0.15;
        for i in 0..8 {
            harmony_activations[i] = harmony_activations[i] * (1.0 - HARMONY_BIAS_ALPHA)
                + self.injected_harmony_bias[i] * HARMONY_BIAS_ALPHA;
        }

        MusicalState {
            harmony_activations,
            dopamine: self.injected_dopamine,
            serotonin: self.injected_serotonin,
            noradrenaline: self.injected_noradrenaline,
            arousal: snapshot.arousal,
            valence: snapshot.valence,
            consciousness_level: snapshot.unified_psi as f32,
            prediction_error: snapshot.prediction_error,
        }
    }

    // ── Allostatic Sonification ───────────────────────────────────────

    fn apply_allostatic_modulations(&self, state: &mut MusicalState) {
        let load = self.injected_allostatic_load;
        let feasibility = self.injected_substrate_feasibility;
        let energy = self.injected_energy_ratio;

        // Tempo spike: high allostatic load → faster tempo via arousal proxy
        if load > ALLOSTATIC_TEMPO_THRESHOLD {
            let tempo_boost = (load - ALLOSTATIC_TEMPO_THRESHOLD) * 1.5;
            state.arousal = (state.arousal + tempo_boost).min(1.0);
        }

        // Harmony collapse: keep only 2 strongest activations
        if load > ALLOSTATIC_DISSONANCE_THRESHOLD {
            let mut indexed: Vec<(usize, f32)> = state
                .harmony_activations
                .iter()
                .enumerate()
                .map(|(i, &a)| (i, a))
                .collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            for (i, _) in indexed.iter().skip(2) {
                state.harmony_activations[*i] = 0.0;
            }
        }

        // Burnout: inject tritone (flatten major 3rd + add tritone via valence shift)
        if load > ALLOSTATIC_BURNOUT_THRESHOLD {
            state.valence = (state.valence - 0.5).max(-1.0);
            // Boost InfinitePlay (minor 7th) for dissonance
            state.harmony_activations[3] = 0.9;
        }

        // Substrate degradation: cap consciousness → reduces polyphony
        if feasibility < 0.5 {
            state.consciousness_level = state.consciousness_level.min(feasibility as f32);
        }

        // Energy depletion: shift valence negative (musical sadness)
        if energy > ENERGY_PITCH_DROP_THRESHOLD {
            state.valence = (state.valence - 0.3).max(-1.0);
            // SacredStillness dominance → slower, contemplative
            state.harmony_activations[7] = 0.9;
        }

        // Near-death reverb: consciousness fading
        if energy > ENERGY_REVERB_THRESHOLD {
            state.consciousness_level = state.consciousness_level.min(0.2);
        }

        // Peer distress: empathic resonance
        if let Some(ref distress) = self.peer_distress {
            let severity = distress.allostatic_load * 0.5
                + distress.energy_depletion * 0.3
                + (1.0 - distress.consciousness_level) * 0.2;
            // Pull valence toward negative (empathic sadness)
            state.valence = (state.valence - severity * 0.4).max(-1.0);
            // Increase prediction error (musical uncertainty)
            state.prediction_error = (state.prediction_error + severity * 0.3).min(1.0);
        }
    }

    // ── Safety fade ───────────────────────────────────────────────────

    fn apply_safety_fade(&mut self, chunk: &mut [[f32; 2]]) {
        let target = match self.injected_safety_level {
            SAFETY_RED => 0.0,
            SAFETY_ORANGE => 0.3, // Low drone level
            SAFETY_YELLOW => 0.7,
            _ => 1.0, // Green: full volume
        };

        // Fade completes in MUSE_SILENCE_FADE_MS milliseconds of audio
        let fade_samples = (MUSE_SILENCE_FADE_MS as f32 / 1000.0) * MUSE_SAMPLE_RATE as f32;
        let step = if target > self.fade_level {
            1.0 / fade_samples
        } else {
            -1.0 / fade_samples
        };

        for pair in chunk.iter_mut() {
            self.fade_level = (self.fade_level + step).clamp(0.0, 1.0);
            pair[0] *= self.fade_level;
            pair[1] *= self.fade_level;
        }
    }
}

impl Default for MuseManager {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for MuseManager {
    fn drop(&mut self) {
        self.save_motifs();
    }
}

impl MuseManager {
    /// Encode interleaved stereo f32 PCM as a minimal RIFF/WAV file.
    ///
    /// Uses IEEE float format (format code 3), 32-bit, stereo.
    /// No external dependencies — pure byte construction.
    fn encode_wav_f32_stereo(samples: &[[f32; 2]], sample_rate: u32) -> Vec<u8> {
        const NUM_CHANNELS: u16 = 2;
        const BITS_PER_SAMPLE: u16 = 32;
        let byte_rate = sample_rate * NUM_CHANNELS as u32 * (BITS_PER_SAMPLE as u32 / 8);
        let block_align: u16 = NUM_CHANNELS * (BITS_PER_SAMPLE / 8);
        let data_size = (samples.len() * NUM_CHANNELS as usize * 4) as u32;
        let file_size = 36u32 + data_size;

        let mut out = Vec::with_capacity(44 + data_size as usize);
        // RIFF header
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&file_size.to_le_bytes());
        out.extend_from_slice(b"WAVE");
        // fmt chunk — IEEE float PCM (format code 3)
        out.extend_from_slice(b"fmt ");
        out.extend_from_slice(&16u32.to_le_bytes()); // chunk size (no extension)
        out.extend_from_slice(&3u16.to_le_bytes()); // IEEE float
        out.extend_from_slice(&NUM_CHANNELS.to_le_bytes());
        out.extend_from_slice(&sample_rate.to_le_bytes());
        out.extend_from_slice(&byte_rate.to_le_bytes());
        out.extend_from_slice(&block_align.to_le_bytes());
        out.extend_from_slice(&BITS_PER_SAMPLE.to_le_bytes());
        // data chunk
        out.extend_from_slice(b"data");
        out.extend_from_slice(&data_size.to_le_bytes());
        for [l, r] in samples {
            out.extend_from_slice(&l.to_le_bytes());
            out.extend_from_slice(&r.to_le_bytes());
        }
        out
    }
}

impl CognitiveSubsystem for MuseManager {
    fn name(&self) -> &'static str {
        "muse_manager"
    }

    fn interval(&self) -> u32 {
        MUSE_INTERVAL
    }

    fn process(&mut self, snapshot: &CycleSnapshot) -> SubsystemOutput {
        let cycle = snapshot.cycle_number;

        // 1. Map snapshot to musical state
        let mut musical_state = self.map_snapshot_to_musical_state(snapshot);

        // 2. Apply allostatic modulations
        self.apply_allostatic_modulations(&mut musical_state);

        // 3. Update synth state on cadence (expensive: scale rebuild, arrangement)
        if cycle % MUSE_STATE_UPDATE_CADENCE == 0 {
            self.synth.update_state(&musical_state);
        }

        // 4. Render audio chunk
        let mut chunk = self.synth.render_chunk();

        // 5. Apply safety fade (silence on Red, reduced on Orange)
        self.apply_safety_fade(&mut chunk);

        // 6. Store for external access (RDP, direct output)
        self.last_chunk = chunk;
        self.last_musical_state = musical_state.clone();

        // 6c. Accumulate samples; emit CompositionExport when target is reached
        self.accumulated_samples.extend_from_slice(&self.last_chunk);
        if self.accumulated_samples.len() >= self.export_target_samples {
            let title = format!(
                "Symthaea Composition #{} — Ψ={:.2}",
                self.export_count + 1,
                snapshot.unified_psi
            );
            let wav_bytes = Self::encode_wav_f32_stereo(
                &self.accumulated_samples[..self.export_target_samples],
                MUSE_SAMPLE_RATE,
            );
            // Snapshot generated notes for self-evaluation, then clear for next period
            let notes = std::mem::take(&mut self.synth.generated_notes);
            let export = CompositionExport {
                wav_bytes,
                title,
                duration_secs: MUSE_EXPORT_SECS,
                musical_state: self.last_musical_state.clone(),
                timestamp_secs: snapshot.cycle_number as u64,
                notes,
            };
            if self.pending_exports.len() >= MUSE_EXPORT_QUEUE_CAPACITY {
                self.pending_exports.pop_front(); // drop oldest if queue is full
            }
            self.pending_exports.push_back(export);
            self.export_count += 1;
            // Retain overflow samples so the next composition starts immediately
            let excess = self
                .accumulated_samples
                .split_off(self.export_target_samples);
            self.accumulated_samples = excess;
        }

        // 6b. Push to live speakers if enabled
        #[cfg(feature = "muse-live")]
        if let Some(ref live) = self.live_output {
            live.update_state(&musical_state);
        }

        // 7. Build subsystem output
        // Music slightly modulates affect: consonant music → positive valence
        let consonance = self.last_musical_state.harmony_activations[0] // ResonantCoherence
            + self.last_musical_state.harmony_activations[2]; // IntegralWisdom (5th)
        let dissonance = self.last_musical_state.harmony_activations[3]; // InfinitePlay (7th)
        let valence_delta = (consonance - dissonance) * 0.02; // subtle

        let mut output = SubsystemOutput::NEUTRAL;
        output.valence_delta = valence_delta;
        output.flags |= output_flags::HAS_TELEMETRY;
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_snapshot() -> CycleSnapshot {
        CycleSnapshot {
            unified_psi: 0.6,
            arousal: 0.4,
            valence: 0.0,
            prediction_error: 0.1,
            harmonic_coherence: 0.5,
            compressed_state: {
                let mut s = [0.0f32; 256];
                for i in 0..8 {
                    s[i] = 0.5;
                }
                s
            },
            ..Default::default()
        }
    }

    #[test]
    fn muse_manager_name_and_interval() {
        let mgr = MuseManager::new();
        assert_eq!(mgr.name(), "muse_manager");
        assert_eq!(mgr.interval(), 1);
    }

    #[test]
    fn process_returns_non_panic() {
        let mut mgr = MuseManager::new();
        let snapshot = default_snapshot();
        let output = mgr.process(&snapshot);
        assert!(output.valence_delta.is_finite());
        assert!(output.has_flag(output_flags::HAS_TELEMETRY));
    }

    #[test]
    fn process_produces_audio() {
        let mut mgr = MuseManager::new();
        let snapshot = default_snapshot();

        // Run several cycles to generate notes
        for i in 0..20 {
            let mut s = snapshot;
            s.cycle_number = i;
            mgr.process(&s);
        }

        let chunk = mgr.last_chunk();
        assert!(!chunk.is_empty(), "should produce audio samples");
    }

    #[test]
    fn high_allostatic_produces_faster_tempo() {
        let mut mgr_calm = MuseManager::new();
        mgr_calm.inject_neuromod(0.5, 0.5, 0.3, 0.2);

        let mut mgr_stressed = MuseManager::new();
        mgr_stressed.inject_neuromod(0.5, 0.5, 0.3, 0.8);

        let snapshot = default_snapshot();

        // Run a state update cycle (cycle 0 % 79 == 0)
        mgr_calm.process(&snapshot);
        mgr_stressed.process(&snapshot);

        // Stressed manager should have higher effective arousal → faster tempo
        let calm_state = mgr_calm.map_snapshot_to_musical_state(&snapshot);
        let mut stressed_state = mgr_stressed.map_snapshot_to_musical_state(&snapshot);
        mgr_stressed.apply_allostatic_modulations(&mut stressed_state);

        assert!(
            stressed_state.arousal > calm_state.arousal,
            "stressed arousal {} should exceed calm {}",
            stressed_state.arousal,
            calm_state.arousal
        );
    }

    #[test]
    fn burnout_injects_dissonance() {
        let mgr = MuseManager::new();
        let snapshot = default_snapshot();
        let mut state = mgr.map_snapshot_to_musical_state(&snapshot);

        // Simulate burnout
        let mut mgr_burn = MuseManager::new();
        mgr_burn.inject_neuromod(0.5, 0.5, 0.3, 0.85);
        mgr_burn.apply_allostatic_modulations(&mut state);

        // InfinitePlay (minor 7th) should be boosted
        assert!(
            state.harmony_activations[3] > 0.8,
            "InfinitePlay should be boosted: {}",
            state.harmony_activations[3]
        );
        // Valence should be negative
        assert!(state.valence < 0.0, "valence should be negative");
    }

    #[test]
    fn safety_red_fades_to_silence() {
        let mut mgr = MuseManager::new();
        mgr.inject_safety(SAFETY_RED);

        let snapshot = default_snapshot();

        // Run enough cycles for fade to complete
        for i in 0..100 {
            let mut s = snapshot;
            s.cycle_number = i;
            mgr.process(&s);
        }

        let chunk = mgr.last_chunk();
        if !chunk.is_empty() {
            let max_amp: f32 = chunk
                .iter()
                .map(|p| p[0].abs().max(p[1].abs()))
                .fold(0.0, f32::max);
            assert!(
                max_amp < 0.01,
                "safety Red should produce near-silence, got {max_amp}"
            );
        }
    }

    #[test]
    fn energy_depletion_affects_state() {
        let mgr = MuseManager::new();
        let snapshot = default_snapshot();
        let mut state = mgr.map_snapshot_to_musical_state(&snapshot);

        let mut mgr_depleted = MuseManager::new();
        mgr_depleted.inject_substrate(0.5, 0.85);
        mgr_depleted.apply_allostatic_modulations(&mut state);

        // SacredStillness should be high (contemplative fading)
        assert!(
            state.harmony_activations[7] > 0.8,
            "SacredStillness should dominate: {}",
            state.harmony_activations[7]
        );
    }

    #[test]
    fn peer_distress_empathic_resonance() {
        let snapshot = default_snapshot();
        let state_baseline = {
            let mgr = MuseManager::new();
            mgr.map_snapshot_to_musical_state(&snapshot)
        };
        let baseline_valence = state_baseline.valence;

        let mut mgr = MuseManager::new();
        mgr.inject_peer_distress(PeerDistressState {
            prediction_error: 0.8,
            energy_depletion: 0.9,
            safety_level: SAFETY_RED,
            allostatic_load: 0.9,
            consciousness_level: 0.1,
        });

        let mut state = mgr.map_snapshot_to_musical_state(&snapshot);
        mgr.apply_allostatic_modulations(&mut state);

        assert!(
            state.valence < baseline_valence,
            "peer distress should lower valence: {} vs baseline {}",
            state.valence,
            baseline_valence
        );
    }

    #[test]
    fn telemetry_reports_state() {
        let mut mgr = MuseManager::new();
        mgr.inject_neuromod(0.5, 0.5, 0.3, 0.6);
        mgr.inject_safety(SAFETY_YELLOW);

        let snapshot = default_snapshot();
        mgr.process(&snapshot);

        let telem = mgr.telemetry();
        assert!(telem.tempo_bpm > 0.0);
        assert!(telem.allostatic_active);
        assert_eq!(telem.safety_level, SAFETY_YELLOW);
    }

    #[test]
    fn substrate_degradation_caps_consciousness() {
        let mgr = MuseManager::new();
        let snapshot = CycleSnapshot {
            unified_psi: 0.9,
            ..default_snapshot()
        };
        let mut state = mgr.map_snapshot_to_musical_state(&snapshot);
        assert!(state.consciousness_level > 0.8);

        let mut mgr_degraded = MuseManager::new();
        mgr_degraded.inject_substrate(0.3, 0.0);
        mgr_degraded.apply_allostatic_modulations(&mut state);

        assert!(
            state.consciousness_level <= 0.3,
            "consciousness should be capped by substrate: {}",
            state.consciousness_level
        );
    }

    #[test]
    fn harmony_activations_always_valid() {
        let mgr = MuseManager::new();

        // Zero harmonic_coherence
        let snapshot = CycleSnapshot {
            harmonic_coherence: 0.0,
            compressed_state: [0.0; 256],
            ..Default::default()
        };
        let state = mgr.map_snapshot_to_musical_state(&snapshot);
        // Should fallback to default activations
        assert!(
            state.harmony_activations.iter().any(|&a| a > 0.0),
            "should have some activations even with zero coherence"
        );
    }
}
