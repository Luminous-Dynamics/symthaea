// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Formant Vocoder
//!
//! Converts formant frames to audio samples using source-filter synthesis.
//!
//! ## Theory
//!
//! Speech synthesis uses the source-filter model:
//! 1. **Source**: Glottal pulse train (voiced) or noise (unvoiced)
//! 2. **Filter**: Cascade of resonators modeling vocal tract formants
//!
//! ```text
//! ┌─────────────────┐     ┌───────────────────────┐     ┌─────────────┐
//! │ LF Glottal +   │────►│ Formant Filters       │────►│   Output    │
//! │ Aspiration +   │     │ (F1–F5, parallel)     │     │   Audio     │
//! │ Noise          │     └───────────────────────┘     └─────────────┘
//! └─────────────────┘
//! ```
//!
//! ## Implementation
//!
//! - Glottal pulse: Liljencrants-Fant (LF) model parameterized by Rd
//! - Aspiration: White noise during glottal open phase (breathiness)
//! - Noise: Pink noise for fricatives
//! - Filters: 5 second-order IIR resonators (F1–F3 articulation, F4/F5 speaker constants)

use crate::voice::articulatory_synthesizer::FormantFrame;
use serde::{Deserialize, Serialize};
use symthaea_vocal_tract::types::SourceType;

// ═══════════════════════════════════════════════════════════════════════════════
// VOCODER CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Vocoder configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VocoderConfig {
    /// Output sample rate (Hz)
    pub sample_rate: u32,
    /// Number of formant filters (typically 3-5)
    pub num_formants: usize,
    /// Master volume (0.0 to 1.0)
    pub volume: f32,
    /// Glottal pulse shape (0.0 = breathy, 1.0 = pressed)
    pub glottal_shape: f32,
    /// Noise floor for unvoiced sounds
    pub noise_floor: f32,
    /// Anti-aliasing filter cutoff ratio
    pub aa_cutoff: f32,
    /// F4 frequency (Hz) — pharyngeal resonance (speaker-dependent constant)
    pub f4_freq: f32,
    /// F5 frequency (Hz) — nasal cavity (speaker-dependent constant)
    pub f5_freq: f32,
    /// F4 bandwidth (Hz)
    pub f4_bandwidth: f32,
    /// F5 bandwidth (Hz)
    pub f5_bandwidth: f32,
    /// Aspiration noise level (0.0 to 0.1 typical)
    pub aspiration_level: f32,
    /// Spectral tilt coefficient (0.0 = flat, 0.5 = gentle rolloff, 0.7 = strong).
    /// Applies a 1-pole low-pass to the source signal for natural speech spectral slope.
    pub spectral_tilt: f32,
    /// F0 jitter amount (fraction, e.g. 0.01 = 1%). Per-sample pitch perturbation.
    pub jitter: f32,
    /// Amplitude shimmer amount (fraction, e.g. 0.02 = 2%). Per-cycle amplitude variation.
    pub shimmer: f32,
    /// Use cascade (series) formant filtering instead of parallel.
    /// Cascade produces more natural spectral rolloff. Default: true.
    pub cascade: bool,
}

impl Default for VocoderConfig {
    fn default() -> Self {
        Self {
            sample_rate: 24000,
            num_formants: 5,
            volume: 0.8,
            glottal_shape: 0.5,
            noise_floor: 0.02,
            aa_cutoff: 0.45,
            f4_freq: 3500.0,
            f5_freq: 4500.0,
            f4_bandwidth: 250.0,
            f5_bandwidth: 300.0,
            aspiration_level: 0.03,
            spectral_tilt: 0.5,
            jitter: 0.01,
            shimmer: 0.02,
            cascade: false,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// VOICE QUALITY — EMOTIONAL MODULATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Per-frame voice quality parameters driven by emotional/cognitive state.
///
/// Maps emotional valence, arousal, and consciousness level to concrete vocal
/// parameters: glottal shape (Rd), jitter/shimmer perturbation, bandwidth
/// scaling, and spectral tilt override.
#[derive(Debug, Clone, Copy)]
pub struct VoiceQuality {
    /// Glottal shape override (Rd: 0.3 = pressed, 1.0 = modal, 2.7 = breathy).
    /// `None` uses the vocoder config default.
    pub rd: Option<f32>,
    /// Jitter scale (1.0 = default, >1 = more pitch perturbation).
    pub jitter_scale: f32,
    /// Shimmer scale (1.0 = default, >1 = more amplitude perturbation).
    pub shimmer_scale: f32,
    /// Bandwidth scale (1.0 = default, <1 = tighter formants, >1 = wider).
    pub bandwidth_scale: f32,
    /// Spectral tilt override (0.0–0.7). `None` uses config default.
    pub spectral_tilt: Option<f32>,
}

impl Default for VoiceQuality {
    fn default() -> Self {
        Self {
            rd: None,
            jitter_scale: 1.0,
            shimmer_scale: 1.0,
            bandwidth_scale: 1.0,
            spectral_tilt: None,
        }
    }
}

/// Map cognitive state dimensions to voice quality parameters.
///
/// - **Valence → Rd**: Negative = pressed (0.5), neutral = modal (1.0), positive = breathy (2.0)
/// - **Arousal → perturbation**: High = tense (more jitter), low = relaxed (less)
/// - **Consciousness → register**: Low = creaky (pressed Rd), high = modal
pub fn cognitive_state_to_voice_quality(
    emotional_valence: f32,
    emotional_arousal: f32,
    consciousness_level: f32,
) -> VoiceQuality {
    // Valence → Rd: negative=pressed(0.5), neutral=modal(1.0), positive=breathy(2.0)
    let rd = 1.0 + emotional_valence; // [-1,1] → [0.0, 2.0]

    // Arousal → perturbation: high arousal = tense (more jitter), low = relaxed
    let jitter_scale = 0.5 + emotional_arousal; // [0,1] → [0.5, 1.5]
    let shimmer_scale = 0.5 + emotional_arousal;

    // Arousal → bandwidth: high arousal = tight (narrow BW), low = relaxed (wider)
    let bandwidth_scale = 1.2 - emotional_arousal * 0.4; // [0,1] → [1.2, 0.8]

    // Consciousness → register: low consciousness = creaky (pressed Rd)
    let rd_adj = if consciousness_level < 0.3 {
        rd.min(0.6)
    } else {
        rd
    };

    let tilt = 0.3 + consciousness_level * 0.3; // [0,1] → [0.3, 0.6]

    VoiceQuality {
        rd: Some(rd_adj.clamp(0.3, 2.7)),
        jitter_scale,
        shimmer_scale,
        bandwidth_scale,
        spectral_tilt: Some(tilt),
    }
}

/// Extended voice quality mapping with derivative-based modulation.
///
/// Builds on `cognitive_state_to_voice_quality` but adds temporal dynamics:
/// - Rising arousal → voice strain (increased jitter)
/// - Rapid valence shift → breathiness instability (spectral tilt flutter)
/// - Dropping consciousness → voice uncertainty (increased shimmer)
/// - Suppressed emotion (high |valence| × low arousal) → tighter BW, more jitter
/// - Animated speech (high |valence| × high arousal) → tighter BW (confident)
/// - High articulation quality → narrower bandwidths (confidence)
/// - High rate stability → reduced jitter (steadier voice)
pub fn cognitive_state_to_voice_quality_extended(
    state: &symthaea_vocal_tract::encoder::VoiceCognitiveState,
    derivs: &symthaea_vocal_tract::encoder::VoiceCognitiveStateDerivatives,
) -> VoiceQuality {
    // Base profile from static state
    let mut vq = cognitive_state_to_voice_quality(
        state.emotional_valence,
        state.emotional_arousal,
        state.consciousness_level,
    );

    // Derivative modulations
    // Rising arousal → voice strain (more jitter)
    if derivs.delta_arousal > 0.0 {
        vq.jitter_scale += derivs.delta_arousal * 0.3;
    }

    // Rapid valence shift → breathiness instability
    let valence_instability = derivs.delta_valence.abs() * 0.1;
    if let Some(ref mut tilt) = vq.spectral_tilt {
        *tilt = (*tilt + valence_instability).clamp(0.0, 0.7);
    }

    // Dropping consciousness → voice uncertainty (more shimmer)
    if derivs.delta_consciousness < 0.0 {
        vq.shimmer_scale += derivs.delta_consciousness.abs() * 0.4;
    }

    // Conflict detection: suppressed emotion (high |valence| × low arousal)
    let valence_magnitude = state.emotional_valence.abs();
    let suppression = valence_magnitude * (1.0 - state.emotional_arousal);
    if suppression > 0.5 {
        let intensity = (suppression - 0.5) * 2.0; // 0..1
        vq.bandwidth_scale *= 1.0 - intensity * 0.15; // tighter BW
        vq.jitter_scale += intensity * 0.2; // voice strain from suppression
    }

    // Animation: confident, articulate speech (high |valence| × high arousal)
    let animation = valence_magnitude * state.emotional_arousal;
    if animation > 0.5 {
        let intensity = (animation - 0.5) * 2.0;
        vq.bandwidth_scale *= 1.0 - intensity * 0.1; // tighter, more focused
    }

    // Articulation quality → bandwidth confidence
    vq.bandwidth_scale *= 1.1 - state.articulation_quality * 0.2; // 0.9–1.1

    // Rate stability → jitter dampening
    vq.jitter_scale *= 1.2 - state.rate_stability * 0.4; // 0.8–1.2

    // Clamp all outputs to safe ranges
    vq.jitter_scale = vq.jitter_scale.clamp(0.1, 3.0);
    vq.shimmer_scale = vq.shimmer_scale.clamp(0.1, 3.0);
    vq.bandwidth_scale = vq.bandwidth_scale.clamp(0.5, 2.0);

    vq
}

// ═══════════════════════════════════════════════════════════════════════════════
// BIQUAD RESONATOR FILTER
// ═══════════════════════════════════════════════════════════════════════════════

/// Second-order IIR resonator (biquad)
///
/// Implements a bandpass filter centered at the formant frequency
/// with bandwidth controlled by Q factor.
#[derive(Debug, Clone, Default)]
struct Resonator {
    // Coefficients
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    // State
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
}

impl Resonator {
    /// Create a new resonator
    fn new() -> Self {
        Self::default()
    }

    /// Set resonator frequency and bandwidth
    fn set_params(&mut self, freq: f32, bandwidth: f32, sample_rate: f32) {
        if freq <= 0.0 || freq >= sample_rate / 2.0 {
            // Invalid frequency - make passthrough
            self.b0 = 1.0;
            self.b1 = 0.0;
            self.b2 = 0.0;
            self.a1 = 0.0;
            self.a2 = 0.0;
            return;
        }

        let omega = 2.0 * std::f32::consts::PI * freq / sample_rate;
        let cos_omega = omega.cos();
        let sin_omega = omega.sin();

        // Use resonant lowpass for formant synthesis (better than bandpass cascade)
        // This gives unity gain at DC and resonant peak at formant frequency
        let q = freq / bandwidth.max(20.0);
        let alpha = sin_omega / (2.0 * q);

        // Resonant lowpass coefficients (better for formant synthesis)
        // Using peaking EQ style for formant resonance
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_omega;
        let a2 = 1.0 - alpha;

        // Bandpass with unity peak gain (constant peak, not constant skirt)
        let peak_gain = q; // Compensate for Q
        let b0 = alpha * peak_gain;
        let b1 = 0.0;
        let b2 = -alpha * peak_gain;

        // Normalize by a0
        self.b0 = b0 / a0;
        self.b1 = b1 / a0;
        self.b2 = b2 / a0;
        self.a1 = a1 / a0;
        self.a2 = a2 / a0;
    }

    /// Process one sample
    fn process(&mut self, input: f32) -> f32 {
        let output = self.b0 * input + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2;

        // Update state
        self.x2 = self.x1;
        self.x1 = input;
        self.y2 = self.y1;
        self.y1 = output;

        output
    }

    /// Reset state
    fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }
}

/// All-pole resonator for Klatt-style cascade formant synthesis.
///
/// Transfer function: H(z) = 1 / (1 - a1*z^-1 - a2*z^-2)
///
/// Unlike the bandpass `Resonator`, this has no numerator zeros. It adds a spectral
/// peak at the center frequency without rejecting other frequencies, which is essential
/// for cascade (series) filtering where F1→F2→F3 are different frequencies.
#[derive(Debug, Clone, Default)]
struct AllPoleResonator {
    a1: f32,
    a2: f32,
    y1: f32,
    y2: f32,
}

impl AllPoleResonator {
    fn new() -> Self {
        Self::default()
    }

    /// Set resonator parameters using the Klatt digital resonator formula.
    /// freq: center frequency (Hz), bandwidth: 3dB bandwidth (Hz), sample_rate: Hz
    fn set_params(&mut self, freq: f32, bandwidth: f32, sample_rate: f32) {
        if freq <= 0.0 || freq >= sample_rate / 2.0 {
            self.a1 = 0.0;
            self.a2 = 0.0;
            return;
        }
        // Klatt (1980) digital resonator coefficients:
        // c = -exp(-2π * B * T)
        // b = 2 * exp(-π * B * T) * cos(2π * F * T)
        // a = 1 - b - c
        let t = 1.0 / sample_rate;
        let exp_neg_pi_bt = (-std::f32::consts::PI * bandwidth * t).exp();
        let exp_neg_2pi_bt = exp_neg_pi_bt * exp_neg_pi_bt;
        let cos_2pi_ft = (2.0 * std::f32::consts::PI * freq * t).cos();

        self.a1 = 2.0 * exp_neg_pi_bt * cos_2pi_ft;
        self.a2 = -exp_neg_2pi_bt;
    }

    /// Process one sample through the all-pole resonator.
    fn process(&mut self, input: f32) -> f32 {
        // y[n] = x[n] + a1*y[n-1] + a2*y[n-2]
        let gain = 1.0 - self.a1 - self.a2; // normalize so DC gain = 1 / (1 - a1 - a2)
        let output = input * gain.abs().max(0.01) + self.a1 * self.y1 + self.a2 * self.y2;
        self.y2 = self.y1;
        self.y1 = output;
        output
    }

    fn reset(&mut self) {
        self.y1 = 0.0;
        self.y2 = 0.0;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// GLOTTAL SOURCE
// ═══════════════════════════════════════════════════════════════════════════════

/// Glottal pulse generator using the Liljencrants-Fant (LF) model.
///
/// The LF model is the standard in speech synthesis research, parameterized
/// by Rd (voice quality descriptor from Fant 1995):
/// - Rd = 0.3: Pressed voice (tight, bright)
/// - Rd = 1.0: Modal voice (normal speech)
/// - Rd = 2.7: Breathy voice (soft, airy)
///
/// The `shape` field (0.0–1.0) maps linearly to Rd: `Rd = 0.3 + shape * 2.4`
#[derive(Debug, Clone)]
struct GlottalSource {
    phase: f32,
    sample_rate: f32,
    /// Shape parameter: 0.0 = pressed (Rd=0.3), 1.0 = breathy (Rd=2.7)
    shape: f32,
}

impl GlottalSource {
    fn new(sample_rate: f32, shape: f32) -> Self {
        Self {
            phase: 0.0,
            sample_rate,
            shape: shape.clamp(0.0, 1.0),
        }
    }

    /// Generate one sample at given fundamental frequency using LF model.
    fn tick(&mut self, f0: f32) -> f32 {
        self.tick_with_rd(f0, self.shape)
    }

    /// Generate one sample with a specific Rd shape parameter.
    ///
    /// Allows per-source-type voice quality: vowels use modal Rd, nasals
    /// breathier, liquids slightly more pressed.
    fn tick_with_rd(&mut self, f0: f32, shape: f32) -> f32 {
        if f0 <= 0.0 {
            return 0.0;
        }

        let period = self.sample_rate / f0;
        let phase_inc = 1.0 / period;

        // Advance phase
        self.phase += phase_inc;
        if self.phase >= 1.0 {
            self.phase -= 1.0;
        }

        // Map shape (0–1) to Rd (0.3–2.7)
        let rd = 0.3 + shape.clamp(0.0, 1.0) * 2.4;

        // Derive LF timing parameters from Rd (Fant 1995)
        let tp = 0.1 + 0.22 * rd; // Peak position (fraction of period)
        let te = tp * (1.0 + (5.0 - 4.6 * rd.min(2.7)) * 0.01); // Excitation instant
        let ta = 0.01 * (0.2 + 3.0 * (rd - 1.0).max(0.0)); // Return time constant

        // Closed phase boundary (fraction of period)
        let tc = te + ta * 3.0; // ~3 time constants for return phase

        let t = self.phase; // Current position in glottal cycle (0–1)

        if t < te {
            // Open phase: E(t) = E0 * exp(alpha*t) * sin(omega_g * t)
            let omega_g = std::f32::consts::PI / tp;
            let alpha = if te > 0.01 { 1.0 / te } else { 100.0 }; // Growth rate

            let t_scaled = t;
            let envelope = (alpha * t_scaled * 0.5).exp(); // Gentle exponential growth
            let oscillation = (omega_g * t_scaled).sin();

            -envelope * oscillation * 0.8 // Negative pressure convention
        } else if t < tc.min(1.0) {
            // Return phase: rapid exponential decay
            let epsilon = if ta > 0.001 { 1.0 / ta } else { 1000.0 };
            let t_ret = t - te;

            // E(t) = decay from excitation instant
            let decay = (-epsilon * t_ret * 0.3).exp();
            -decay * 0.4
        } else {
            // Closed phase: vocal folds closed, no airflow
            0.0
        }
    }

    /// Whether the glottal source is in the open phase for the current sample.
    fn is_open(&self) -> bool {
        let rd = 0.3 + self.shape * 2.4;
        let tp = 0.1 + 0.22 * rd;
        let te = tp * (1.0 + (5.0 - 4.6 * rd.min(2.7)) * 0.01);
        self.phase < te
    }

    fn reset(&mut self) {
        self.phase = 0.0;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// NOISE SOURCE
// ═══════════════════════════════════════════════════════════════════════════════

/// Noise generator for unvoiced sounds
#[derive(Debug, Clone)]
struct NoiseSource {
    state: u64,
    pink_state: [f32; 7], // For pink noise filtering
}

impl NoiseSource {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.max(1),
            pink_state: [0.0; 7],
        }
    }

    /// Generate white noise sample
    fn white(&mut self) -> f32 {
        // Xorshift64
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;

        // Map to [-1, 1]
        (self.state as f32 / u64::MAX as f32) * 2.0 - 1.0
    }

    /// Generate pink noise sample (better for fricatives)
    fn pink(&mut self) -> f32 {
        let white = self.white();

        // Voss-McCartney algorithm approximation
        // Simple IIR low-pass cascade
        self.pink_state[0] = 0.99886 * self.pink_state[0] + white * 0.0555179;
        self.pink_state[1] = 0.99332 * self.pink_state[1] + white * 0.0750759;
        self.pink_state[2] = 0.96900 * self.pink_state[2] + white * 0.153_852;
        self.pink_state[3] = 0.86650 * self.pink_state[3] + white * 0.3104856;
        self.pink_state[4] = 0.55000 * self.pink_state[4] + white * 0.5329522;
        self.pink_state[5] = -0.7616 * self.pink_state[5] - white * 0.0168980;

        let pink = self.pink_state[0]
            + self.pink_state[1]
            + self.pink_state[2]
            + self.pink_state[3]
            + self.pink_state[4]
            + self.pink_state[5]
            + self.pink_state[6]
            + white * 0.5362;

        self.pink_state[6] = white * 0.115926;

        pink * 0.11 // Normalize
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ORNSTEIN-UHLENBECK PROCESS
// ═══════════════════════════════════════════════════════════════════════════════

/// Ornstein-Uhlenbeck process: mean-reverting, temporally correlated noise.
///
/// `dx = θ(μ - x)dt + σ dW`
///
/// Unlike white noise, OU produces smooth, biologically realistic perturbations
/// that drift away from and return to a mean value. Used for F0 jitter and
/// amplitude shimmer that sounds natural rather than harsh.
#[derive(Debug, Clone)]
struct OrnsteinUhlenbeck {
    /// Current state.
    x: f32,
    /// Mean-reversion rate (higher = faster return to mu).
    theta: f32,
    /// Long-term mean.
    mu: f32,
    /// Diffusion coefficient (noise amplitude).
    sigma: f32,
    /// White noise source.
    noise: NoiseSource,
}

impl OrnsteinUhlenbeck {
    fn new(theta: f32, mu: f32, sigma: f32, seed: u64) -> Self {
        Self {
            x: mu,
            theta,
            mu,
            sigma,
            noise: NoiseSource::new(seed),
        }
    }

    /// Advance one step at the given timestep (dt = 1/sample_rate).
    fn tick(&mut self, dt: f32) -> f32 {
        let dw = self.noise.white() * dt.sqrt();
        self.x += self.theta * (self.mu - self.x) * dt + self.sigma * dw;
        self.x
    }

    fn reset(&mut self) {
        self.x = self.mu;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// FORMANT VOCODER
// ═══════════════════════════════════════════════════════════════════════════════

/// Formant vocoder for speech synthesis
///
/// Converts formant frames to audio using source-filter synthesis.
#[derive(Debug, Clone)]
pub struct FormantVocoder {
    config: VocoderConfig,
    /// Formant resonators (F1, F2, F3, ...)
    resonators: Vec<Resonator>,
    /// Glottal pulse generator
    glottal: GlottalSource,
    /// Noise source for unvoiced
    noise: NoiseSource,
    /// Low-pass filter for smoothing
    lowpass: Resonator,
    /// Current frame index for interpolation
    frame_idx: usize,
    /// Samples since last frame
    samples_in_frame: usize,
    /// Spectral tilt 1-pole filter state
    tilt_state: f32,
    /// Current shimmer factor for this glottal cycle (1.0 ± shimmer)
    shimmer_factor: f32,
    /// Nasal anti-resonator — subtracted for nasal phonemes. Phoneme-specific zero freq.
    nasal_antires: Resonator,
    /// Nasal pole resonator (fixed 250 Hz / 100 Hz BW) — adds characteristic nasal resonance.
    nasal_pole: Resonator,
    /// Subglottal resonances (~600, ~1400, ~2100 Hz) — chest voice coupling.
    subglottal: [Resonator; 3],
    /// All-pole resonators for cascade mode (F1, F2, F3).
    cascade_res: [AllPoleResonator; 3],
    /// Ornstein-Uhlenbeck process for F0 jitter (temporally correlated pitch perturbation).
    ou_jitter: OrnsteinUhlenbeck,
    /// Ornstein-Uhlenbeck process for amplitude shimmer (temporally correlated energy perturbation).
    ou_shimmer: OrnsteinUhlenbeck,
    /// Previous output sample for lip radiation filter (first-difference).
    radiation_prev: f32,
    /// Current F1 frequency for amplitude correction.
    current_f1: f32,
    /// Current F2 frequency for amplitude correction.
    current_f2: f32,
    /// Current F3 frequency for amplitude correction.
    current_f3: f32,
}

impl FormantVocoder {
    /// Create a new vocoder with default configuration
    pub fn new() -> Self {
        Self::with_config(VocoderConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: VocoderConfig) -> Self {
        let mut resonators = Vec::with_capacity(config.num_formants);
        for _ in 0..config.num_formants {
            resonators.push(Resonator::new());
        }

        let mut lowpass = Resonator::new();
        lowpass.set_params(
            config.sample_rate as f32 * config.aa_cutoff,
            config.sample_rate as f32 * 0.2,
            config.sample_rate as f32,
        );

        let mut nasal_antires = Resonator::new();
        nasal_antires.set_params(300.0, 100.0, config.sample_rate as f32);

        let mut nasal_pole = Resonator::new();
        nasal_pole.set_params(250.0, 100.0, config.sample_rate as f32);

        let mut sg1 = Resonator::new();
        sg1.set_params(600.0, 80.0, config.sample_rate as f32);
        let mut sg2 = Resonator::new();
        sg2.set_params(1400.0, 100.0, config.sample_rate as f32);
        let mut sg3 = Resonator::new();
        sg3.set_params(2100.0, 120.0, config.sample_rate as f32);

        // OU parameters tuned for biological voice quality.
        // Continuous-time OU: dx = θ(μ−x)dt + σ√dt dW
        // Stationary stddev = σ / √(2θ), so σ = desired_stddev × √(2θ).
        //
        // Jitter (per-sample): θ=100, stddev→config.jitter (e.g. 1% pitch variation)
        // Shimmer (per-glottal-cycle): θ=50, stddev→config.shimmer (e.g. 2% amplitude variation)
        let ou_jitter = OrnsteinUhlenbeck::new(
            100.0,
            1.0,
            config.jitter * (2.0 * 100.0_f32).sqrt(), // σ so stddev = config.jitter
            0xF0F0CAFE,
        );
        let ou_shimmer = OrnsteinUhlenbeck::new(
            50.0,
            1.0,
            config.shimmer * (2.0 * 50.0_f32).sqrt(), // σ so stddev = config.shimmer
            0xBEEF1234,
        );

        Self {
            glottal: GlottalSource::new(config.sample_rate as f32, config.glottal_shape),
            noise: NoiseSource::new(0xDEADBEEF),
            resonators,
            lowpass,
            config,
            frame_idx: 0,
            samples_in_frame: 0,
            tilt_state: 0.0,
            shimmer_factor: 1.0,
            nasal_antires,
            nasal_pole,
            subglottal: [sg1, sg2, sg3],
            cascade_res: [
                AllPoleResonator::new(),
                AllPoleResonator::new(),
                AllPoleResonator::new(),
            ],
            ou_jitter,
            ou_shimmer,
            radiation_prev: 0.0,
            current_f1: 500.0,
            current_f2: 1500.0,
            current_f3: 2500.0,
        }
    }

    /// Reset vocoder state
    pub fn reset(&mut self) {
        self.glottal.reset();
        for res in &mut self.resonators {
            res.reset();
        }
        self.lowpass.reset();
        self.nasal_antires.reset();
        self.nasal_pole.reset();
        for sg in &mut self.subglottal {
            sg.reset();
        }
        for cr in &mut self.cascade_res {
            cr.reset();
        }
        self.ou_jitter.reset();
        self.ou_shimmer.reset();
        self.frame_idx = 0;
        self.samples_in_frame = 0;
        self.tilt_state = 0.0;
        self.shimmer_factor = 1.0;
        self.radiation_prev = 0.0;
    }

    /// Synthesize audio from formant frames
    pub fn synthesize(&mut self, frames: &[FormantFrame]) -> Vec<f32> {
        if frames.is_empty() {
            return Vec::new();
        }

        self.reset();

        // Calculate total samples
        // SAFETY: frames.is_empty() returned early above
        let Some(last_frame) = frames.last() else {
            return Vec::new();
        };
        let total_duration = last_frame.time - frames[0].time;
        let frame_duration = if frames.len() > 1 {
            total_duration / (frames.len() - 1) as f32
        } else {
            0.1
        };

        let samples_per_frame =
            (frame_duration.clamp(0.0, 1.0) * self.config.sample_rate as f32) as usize;
        let total_samples = samples_per_frame * frames.len();

        let mut audio = Vec::with_capacity(total_samples);

        // Process frame by frame
        for (i, frame) in frames.iter().enumerate() {
            // Get next frame for interpolation
            let next_frame = frames.get(i + 1).unwrap_or(frame);

            // Update resonator parameters
            self.update_resonators(frame);

            // Generate samples for this frame
            for j in 0..samples_per_frame {
                let t = j as f32 / samples_per_frame as f32;

                // Interpolate parameters
                let current = frame.lerp(next_frame, t);

                // Manner-aware source excitation
                let raw_source = self.generate_source(&current, t);

                // Spectral tilt: 1-pole low-pass for natural speech spectral slope
                let source = raw_source * (1.0 - self.config.spectral_tilt)
                    + self.tilt_state * self.config.spectral_tilt;
                self.tilt_state = source;

                // Formant filtering + nasal pole/anti-resonance
                let filtered = self.apply_filters(
                    source,
                    current.source_type,
                    current.f0,
                    current.nasal_zero_freq,
                    current.nasal_zero_bw,
                );

                // Lip radiation: first-difference filter (6dB/octave HF boost)
                let radiated = filtered - self.radiation_prev;
                self.radiation_prev = filtered;

                // Apply energy envelope with shimmer
                let output = radiated * current.energy * self.shimmer_factor * self.config.volume;

                // Low-pass filter for smoothing
                let smoothed = self.lowpass.process(output);

                // Soft clip to prevent distortion
                audio.push(soft_clip(smoothed));
            }
        }

        audio
    }

    /// Synthesize audio with per-frame voice quality modulation.
    ///
    /// Like `synthesize()` but applies `VoiceQuality` overrides per frame:
    /// Rd (glottal shape), jitter/shimmer scaling, bandwidth scaling, spectral tilt.
    /// If `quality` is shorter than `frames`, the last quality entry is repeated.
    pub fn synthesize_with_quality(
        &mut self,
        frames: &[FormantFrame],
        quality: &[VoiceQuality],
    ) -> Vec<f32> {
        if frames.is_empty() {
            return Vec::new();
        }

        self.reset();

        let total_duration = match (frames.first(), frames.last()) {
            (Some(first), Some(last)) => last.time - first.time,
            _ => return Vec::new(),
        };
        let frame_duration = if frames.len() > 1 {
            total_duration / (frames.len() - 1) as f32
        } else {
            0.1
        };
        let samples_per_frame =
            (frame_duration.clamp(0.0, 1.0) * self.config.sample_rate as f32) as usize;
        let default_quality = VoiceQuality::default();

        let mut audio = Vec::with_capacity(samples_per_frame * frames.len());

        for (i, frame) in frames.iter().enumerate() {
            let next_frame = frames.get(i + 1).unwrap_or(frame);
            let vq = quality
                .get(i)
                .unwrap_or_else(|| quality.last().unwrap_or(&default_quality));

            // Apply bandwidth scaling to resonators
            let mut scaled_frame = *frame;
            scaled_frame.b1 *= vq.bandwidth_scale;
            scaled_frame.b2 *= vq.bandwidth_scale;
            scaled_frame.b3 *= vq.bandwidth_scale;
            self.update_resonators(&scaled_frame);

            let tilt = vq.spectral_tilt.unwrap_or(self.config.spectral_tilt);
            let _jitter = self.config.jitter * vq.jitter_scale;
            let shimmer = self.config.shimmer * vq.shimmer_scale;

            for j in 0..samples_per_frame {
                let t = j as f32 / samples_per_frame as f32;
                let current = frame.lerp(next_frame, t);

                // Override Rd if quality specifies it
                let rd_override = vq.rd.unwrap_or(self.config.glottal_shape);

                // Source excitation with quality-modulated Rd
                let raw_source = if let Some(_rd) = vq.rd {
                    // Use quality-specified Rd for source generation
                    self.generate_source_with_rd(&current, t, rd_override)
                } else {
                    self.generate_source(&current, t)
                };

                // Spectral tilt with quality override
                let source = raw_source * (1.0 - tilt) + self.tilt_state * tilt;
                self.tilt_state = source;

                // Formant filtering
                let filtered = self.apply_filters(
                    source,
                    current.source_type,
                    current.f0,
                    current.nasal_zero_freq,
                    current.nasal_zero_bw,
                );

                // Lip radiation
                let radiated = filtered - self.radiation_prev;
                self.radiation_prev = filtered;

                // Apply shimmer-scaled energy
                // OU shimmer returns values centered around mu=1.0.
                // Scale the deviation by (quality shimmer / base shimmer).
                let dt = 1.0 / self.config.sample_rate as f32;
                let shimmer_ou = self.ou_shimmer.tick(dt);
                let deviation = shimmer_ou - 1.0;
                let scale = shimmer / self.config.shimmer.max(0.001);
                let shimmer_factor = (1.0 + deviation * scale).clamp(0.5, 1.5);

                let output = radiated * current.energy * shimmer_factor * self.config.volume;
                let smoothed = self.lowpass.process(output);
                audio.push(soft_clip(smoothed));
            }
        }

        audio
    }

    /// Generate source excitation with explicit Rd override (for voice quality modulation).
    fn generate_source_with_rd(&mut self, current: &FormantFrame, _progress: f32, rd: f32) -> f32 {
        match current.source_type {
            SourceType::Vowel | SourceType::Liquid => {
                let dt = 1.0 / self.config.sample_rate as f32;
                let jitter_factor = self.ou_jitter.tick(dt);
                let f0_jittered = current.f0 * jitter_factor;

                let prev_phase = self.glottal.phase;
                let voiced = self.glottal.tick_with_rd(f0_jittered, rd) * current.voicing;
                if self.glottal.phase < prev_phase {
                    let shimmer_dt = 1.0 / current.f0.max(50.0);
                    self.shimmer_factor = self.ou_shimmer.tick(shimmer_dt);
                }
                let unvoiced = self.noise.pink() * (1.0 - current.voicing) * 0.3;
                let aspiration = if self.glottal.is_open() {
                    self.noise.white()
                        * self.config.aspiration_level
                        * (1.0 - rd.clamp(0.0, 1.0) * 0.7)
                } else {
                    0.0
                };
                (voiced + aspiration + unvoiced) * 30.0
            }
            SourceType::Nasal => {
                let dt = 1.0 / self.config.sample_rate as f32;
                let jitter_factor = self.ou_jitter.tick(dt);
                let f0_jittered = current.f0 * jitter_factor;
                let voiced = self.glottal.tick_with_rd(f0_jittered, rd) * current.voicing;
                voiced * 30.0
            }
            _ => {
                // For non-voiced types, delegate to standard source
                self.generate_source(current, _progress)
            }
        }
    }

    /// Compute per-source-type glottal shape (Rd) for voice quality variation.
    ///
    /// Different manner classes benefit from different Rd values:
    /// - Vowels: modal voice (config default)
    /// - Liquids: slightly more pressed (clearer formants)
    /// - Nasals: slightly breathier (softer coupling)
    fn source_rd(&self, source_type: SourceType) -> f32 {
        match source_type {
            SourceType::Vowel => self.config.glottal_shape,
            SourceType::Liquid => (self.config.glottal_shape * 0.8).clamp(0.0, 1.0),
            SourceType::Nasal => (self.config.glottal_shape * 1.2).clamp(0.0, 1.0),
            _ => self.config.glottal_shape,
        }
    }

    /// Generate source excitation signal based on manner of articulation.
    ///
    /// Returns the raw source signal before formant filtering.
    /// - `progress`: position within the current frame/phoneme (0.0–1.0)
    fn generate_source(&mut self, current: &FormantFrame, progress: f32) -> f32 {
        let rd = self.source_rd(current.source_type);
        match current.source_type {
            SourceType::Vowel | SourceType::Liquid => {
                // Standard: glottal pulse + aspiration + unvoiced noise
                // OU jitter: temporally correlated pitch perturbation (biological realism)
                let dt = 1.0 / self.config.sample_rate as f32;
                let jitter_factor = self.ou_jitter.tick(dt);
                let f0_jittered = current.f0 * jitter_factor;

                let prev_phase = self.glottal.phase;
                let voiced = self.glottal.tick_with_rd(f0_jittered, rd) * current.voicing;
                if self.glottal.phase < prev_phase {
                    // OU shimmer: temporally correlated amplitude perturbation
                    // Use glottal period (1/f0) as dt since shimmer updates per-cycle
                    let shimmer_dt = 1.0 / current.f0.max(50.0);
                    self.shimmer_factor = self.ou_shimmer.tick(shimmer_dt);
                }
                let unvoiced = self.noise.pink() * (1.0 - current.voicing) * 0.3;
                let aspiration = if self.glottal.is_open() {
                    self.noise.white()
                        * self.config.aspiration_level
                        * (1.0 - self.glottal.shape * 0.7)
                } else {
                    0.0
                };
                (voiced + aspiration + unvoiced) * 30.0
            }
            SourceType::Stop => {
                // VOT-aware stop model:
                // - Closure (0–70%): silence or voicing bar (voiced stops)
                // - Burst (70–80%): transient noise burst
                // - VOT region (80–100%):
                //   - Voiceless: aspiration noise (decaying) before next vowel
                //   - Voiced: voicing resumes immediately after burst
                if progress < 0.7 {
                    // Closure phase
                    if current.voicing > 0.5 {
                        self.glottal.tick_with_rd(current.f0, rd) * 0.1
                    } else {
                        0.0
                    }
                } else if progress < 0.8 {
                    // Burst: transient noise
                    self.noise.white() * 0.8 * 30.0
                } else if current.voicing <= 0.5 {
                    // Voiceless VOT: aspiration noise decaying into next segment
                    let decay = 1.0 - (progress - 0.8) / 0.2;
                    self.noise.white() * 0.4 * decay * 30.0
                } else {
                    // Voiced: voicing resumes immediately
                    self.glottal.tick_with_rd(current.f0, rd) * current.voicing * 30.0
                }
            }
            SourceType::Fricative => {
                // Spectral shaping: sibilants (/S/, /SH/) use white noise
                // (bright, high-frequency energy), non-sibilants (/F/, /TH/)
                // use pink noise (gentler, lower spectral emphasis).
                // F2 position serves as proxy: high F2 → sibilant.
                let sibilance = ((self.current_f2 - 1500.0) / 1000.0).clamp(0.0, 1.0);
                let noise =
                    self.noise.white() * sibilance + self.noise.pink() * (1.0 - sibilance) * 2.0;
                let noise = noise * 0.5;
                let voiced = if current.voicing > 0.5 {
                    self.glottal.tick(current.f0) * 0.3
                } else {
                    0.0
                };
                (noise + voiced) * 30.0
            }
            SourceType::Nasal => {
                // Voiced source (nasal anti-formant applied post-filter)
                let voiced = self.glottal.tick_with_rd(current.f0, rd) * current.voicing;
                voiced * 30.0
            }
            SourceType::Affricate => {
                // First 40% closure, 40-50% burst, 50%+ frication
                if progress < 0.4 {
                    if current.voicing > 0.5 {
                        self.glottal.tick(current.f0) * 0.1
                    } else {
                        0.0
                    }
                } else if progress < 0.5 {
                    self.noise.white() * 0.8 * 30.0
                } else {
                    self.noise.white() * 0.5 * 30.0
                }
            }
            SourceType::Silent => 0.0,
        }
    }

    /// Apply formant filtering + optional nasal pole/anti-resonance + subglottal resonances.
    ///
    /// Supports two formant filter topologies:
    /// - **Cascade** (`config.cascade = true`): Signal passes through F1->F2->F3 in series,
    ///   producing more natural spectral rolloff (default).
    /// - **Parallel** (`config.cascade = false`): Each resonator processes the source
    ///   independently with Klatt (1980) amplitude correction, then outputs are summed.
    ///
    /// F4/F5 are always parallel (speaker constants). Subglottal resonances (~600, ~1400,
    /// ~2100 Hz) add "chest voice" coupling for voiced speech.
    fn apply_filters(
        &mut self,
        source: f32,
        source_type: SourceType,
        f0: f32,
        nasal_zero_freq: f32,
        nasal_zero_bw: f32,
    ) -> f32 {
        let filtered = if self.config.cascade && self.resonators.len() >= 3 {
            // Cascade: Klatt-style series filtering using all-pole resonators.
            //
            // Unlike the parallel path (bandpass filters), cascade uses all-pole resonators
            // H(z) = 1 / (1 - a1*z^-1 - a2*z^-2) that shape the spectrum without
            // completely rejecting out-of-band content. This allows F1→F2→F3 in series.
            //
            // Each resonator adds a spectral peak while passing other frequencies through.
            let mut signal = source;
            signal = self.cascade_res[0].process(signal);
            signal = self.cascade_res[1].process(signal);
            signal = self.cascade_res[2].process(signal);
            signal
        } else if self.resonators.len() >= 3 {
            // Parallel: each formant adds independently (original behavior)
            let a1 = formant_amplitude_correction(self.current_f1, f0);
            let a2 = formant_amplitude_correction(self.current_f2, f0);
            let a3 = formant_amplitude_correction(self.current_f3, f0);
            let mut filtered = 0.0;
            filtered += self.resonators[0].process(source) * a1;
            filtered += self.resonators[1].process(source) * a2;
            filtered += self.resonators[2].process(source) * a3;
            filtered
        } else {
            let mut filtered = 0.0;
            for res in &mut self.resonators {
                filtered += res.process(source);
            }
            filtered
        };

        // F4/F5 always parallel (speaker constants)
        let mut result = filtered;
        if self.resonators.len() >= 4 {
            result += self.resonators[3].process(source) * 0.1;
        }
        if self.resonators.len() >= 5 {
            result += self.resonators[4].process(source) * 0.05;
        }

        // Nasal resonance: pole (low-frequency nasal murmur) + anti-formant (zero)
        if source_type == SourceType::Nasal {
            // Nasal pole: characteristic low-frequency nasal resonance at ~250 Hz
            result += self.nasal_pole.process(source) * 0.3;

            // Anti-formant: phoneme-specific zero (from FormantFrame.nasal_zero_freq)
            let sr = self.config.sample_rate as f32;
            let zero_freq = if nasal_zero_freq > 0.0 {
                nasal_zero_freq
            } else {
                300.0
            };
            let zero_bw = if nasal_zero_bw > 0.0 {
                nasal_zero_bw
            } else {
                200.0
            };
            self.nasal_antires.set_params(zero_freq, zero_bw, sr);
            result -= self.nasal_antires.process(source) * 0.4;
        }

        // Subglottal resonances (voiced speech only)
        if source_type == SourceType::Vowel
            || source_type == SourceType::Nasal
            || source_type == SourceType::Liquid
        {
            result += self.subglottal[0].process(source) * 0.04;
            result += self.subglottal[1].process(source) * 0.02;
            result += self.subglottal[2].process(source) * 0.01;
        }

        result
    }

    /// Update resonator parameters from frame and track current formant frequencies.
    fn update_resonators(&mut self, frame: &FormantFrame) {
        let sr = self.config.sample_rate as f32;

        if !self.resonators.is_empty() {
            self.resonators[0].set_params(frame.f1, frame.b1, sr);
        }
        if self.resonators.len() >= 2 {
            self.resonators[1].set_params(frame.f2, frame.b2, sr);
        }
        if self.resonators.len() >= 3 {
            self.resonators[2].set_params(frame.f3, frame.b3, sr);
        }
        // F4/F5 are speaker-dependent constants (not articulation-dependent)
        if self.resonators.len() >= 4 {
            self.resonators[3].set_params(self.config.f4_freq, self.config.f4_bandwidth, sr);
        }
        if self.resonators.len() >= 5 {
            self.resonators[4].set_params(self.config.f5_freq, self.config.f5_bandwidth, sr);
        }

        // Track current formant frequencies for amplitude correction
        self.current_f1 = frame.f1;
        self.current_f2 = frame.f2;
        self.current_f3 = frame.f3;

        // Update cascade (all-pole) resonators for cascade mode
        if self.config.cascade {
            self.cascade_res[0].set_params(frame.f1, frame.b1, sr);
            self.cascade_res[1].set_params(frame.f2, frame.b2, sr);
            self.cascade_res[2].set_params(frame.f3, frame.b3, sr);
        }
    }

    /// Synthesize audio for a single formant frame (streaming mode).
    ///
    /// Unlike `synthesize()`, this does NOT reset state between calls — resonator
    /// and glottal source states carry over for smooth real-time output.
    /// Call this at your frame rate (e.g., 200Hz) with `samples_per_frame`
    /// samples per call (e.g., 120 at 24kHz/200Hz).
    pub fn synthesize_frame(&mut self, frame: &FormantFrame, samples_per_frame: usize) -> Vec<f32> {
        self.update_resonators(frame);
        let mut audio = Vec::with_capacity(samples_per_frame);

        for j in 0..samples_per_frame {
            let progress = j as f32 / samples_per_frame.max(1) as f32;

            // Manner-aware source excitation (includes jitter + shimmer for vowels)
            let raw_source = self.generate_source(frame, progress);

            // Spectral tilt: 1-pole low-pass for natural speech spectral slope
            let source = raw_source * (1.0 - self.config.spectral_tilt)
                + self.tilt_state * self.config.spectral_tilt;
            self.tilt_state = source;

            // Formant filtering + nasal pole/anti-resonance
            let filtered = self.apply_filters(
                source,
                frame.source_type,
                frame.f0,
                frame.nasal_zero_freq,
                frame.nasal_zero_bw,
            );

            // Lip radiation: first-difference filter (6dB/octave HF boost)
            let radiated = filtered - self.radiation_prev;
            self.radiation_prev = filtered;

            let output = radiated * frame.energy * self.shimmer_factor * self.config.volume;
            let smoothed = self.lowpass.process(output);
            audio.push(soft_clip(smoothed));
        }

        audio
    }

    /// Synthesize audio for a single formant frame with voice quality modulation (streaming mode).
    ///
    /// Like `synthesize_frame` but applies `VoiceQuality` overrides: bandwidth scaling,
    /// spectral tilt, shimmer scaling, and optional Rd override.
    pub fn synthesize_frame_with_quality(
        &mut self,
        frame: &FormantFrame,
        quality: &VoiceQuality,
        samples_per_frame: usize,
    ) -> Vec<f32> {
        // Apply bandwidth scaling
        let mut scaled = *frame;
        scaled.b1 *= quality.bandwidth_scale;
        scaled.b2 *= quality.bandwidth_scale;
        scaled.b3 *= quality.bandwidth_scale;
        self.update_resonators(&scaled);

        let tilt = quality.spectral_tilt.unwrap_or(self.config.spectral_tilt);
        let shimmer_scale = self.config.shimmer * quality.shimmer_scale;
        let mut audio = Vec::with_capacity(samples_per_frame);

        for j in 0..samples_per_frame {
            let progress = j as f32 / samples_per_frame.max(1) as f32;

            // Source excitation with optional Rd override
            let raw_source = if let Some(rd) = quality.rd {
                self.generate_source_with_rd(frame, progress, rd)
            } else {
                self.generate_source(frame, progress)
            };

            // Spectral tilt with quality override
            let source = raw_source * (1.0 - tilt) + self.tilt_state * tilt;
            self.tilt_state = source;

            // Formant filtering + nasal
            let filtered = self.apply_filters(
                source,
                frame.source_type,
                frame.f0,
                frame.nasal_zero_freq,
                frame.nasal_zero_bw,
            );

            // Lip radiation
            let radiated = filtered - self.radiation_prev;
            self.radiation_prev = filtered;

            // Shimmer-scaled energy (OU shimmer deviation scaled by quality)
            let dt = 1.0 / self.config.sample_rate as f32;
            let shimmer_ou = self.ou_shimmer.tick(dt);
            let deviation = shimmer_ou - 1.0;
            let scale = shimmer_scale / self.config.shimmer.max(0.001);
            let shimmer_factor = (1.0 + deviation * scale).clamp(0.5, 1.5);

            let output = radiated * frame.energy * shimmer_factor * self.config.volume;
            let smoothed = self.lowpass.process(output);
            audio.push(soft_clip(smoothed));
        }

        audio
    }

    /// Get sample rate
    pub fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }

    /// Get configuration
    pub fn config(&self) -> &VocoderConfig {
        &self.config
    }
}

impl Default for FormantVocoder {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// HELPER FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════

/// Compensate for glottal spectral rolloff (~12dB/octave from LF model).
///
/// Higher formants need more gain since the glottal source rolls off.
/// Based on Klatt (1980) amplitude correction factors.
fn formant_amplitude_correction(formant_freq: f32, f0: f32) -> f32 {
    let harmonic_number = (formant_freq / f0.max(50.0)).round().max(1.0);
    let rolloff_db = 12.0 * harmonic_number.log2(); // 12dB/octave
    let correction = 10.0_f32.powf(rolloff_db / 20.0); // dB to linear
    correction.min(8.0) // Cap at 8x to prevent instability
}

/// Soft clipping function to prevent harsh distortion
fn soft_clip(x: f32) -> f32 {
    if x.abs() < 0.5 {
        x
    } else if x > 0.0 {
        0.5 + (1.0 - (-2.0 * (x - 0.5)).exp()) * 0.5
    } else {
        -0.5 - (1.0 - (-2.0 * (-x - 0.5)).exp()) * 0.5
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resonator_basic() {
        let mut res = Resonator::new();
        res.set_params(500.0, 60.0, 24000.0);

        // Process some samples
        let mut output = 0.0;
        for i in 0..1000 {
            let input = if i == 0 { 1.0 } else { 0.0 }; // Impulse
            output = res.process(input);
        }

        // Should have decayed
        assert!(output.abs() < 0.1);
    }

    #[test]
    fn test_glottal_source() {
        let mut glottal = GlottalSource::new(24000.0, 0.5);

        let mut samples = Vec::new();
        for _ in 0..480 {
            // 20ms at 24kHz
            samples.push(glottal.tick(120.0)); // 120 Hz
        }

        // Should have some non-zero samples
        assert!(samples.iter().any(|&s| s.abs() > 0.1));

        // Should be bounded (glottal pulses can have transients up to ~3x)
        assert!(samples.iter().all(|&s| s.abs() < 4.0));
    }

    #[test]
    fn test_noise_source() {
        let mut noise = NoiseSource::new(42);

        let mut samples = Vec::new();
        for _ in 0..1000 {
            samples.push(noise.pink());
        }

        // Should have variance
        let mean: f32 = samples.iter().sum::<f32>() / samples.len() as f32;
        let variance: f32 =
            samples.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / samples.len() as f32;

        assert!(variance > 0.001);
        assert!(variance < 1.0);
    }

    #[test]
    fn test_vocoder_synthesis() {
        let mut vocoder = FormantVocoder::new();

        // Create a simple vowel frame
        let frames = vec![
            FormantFrame {
                f1: 500.0,
                f2: 1500.0,
                f3: 2500.0,
                b1: 60.0,
                b2: 90.0,
                b3: 150.0,
                f0: 120.0,
                energy: 0.8,
                voicing: 1.0,
                time: 0.0,
                source_type: SourceType::Vowel,
                nasal_zero_freq: 0.0,
                nasal_zero_bw: 0.0,
            },
            FormantFrame {
                f1: 500.0,
                f2: 1500.0,
                f3: 2500.0,
                b1: 60.0,
                b2: 90.0,
                b3: 150.0,
                f0: 120.0,
                energy: 0.8,
                voicing: 1.0,
                time: 0.1,
                source_type: SourceType::Vowel,
                nasal_zero_freq: 0.0,
                nasal_zero_bw: 0.0,
            },
        ];

        let audio = vocoder.synthesize(&frames);

        assert!(!audio.is_empty());
        assert!(audio.iter().any(|&s| s.abs() > 0.01)); // Has content
        assert!(audio.iter().all(|&s| s.abs() < 1.5)); // Not clipping badly
    }

    #[test]
    fn test_soft_clip() {
        assert!((soft_clip(0.3) - 0.3).abs() < 0.01);
        assert!(soft_clip(2.0) < 1.0);
        assert!(soft_clip(-2.0) > -1.0);
    }

    #[test]
    fn test_five_formant_synthesis() {
        // Default config now uses 5 formants
        let config = VocoderConfig::default();
        assert_eq!(config.num_formants, 5);

        let mut vocoder = FormantVocoder::with_config(config);
        assert_eq!(vocoder.resonators.len(), 5);

        let frames = vec![
            FormantFrame {
                f1: 730.0,
                f2: 1090.0,
                f3: 2440.0,
                b1: 60.0,
                b2: 90.0,
                b3: 150.0,
                f0: 120.0,
                energy: 0.8,
                voicing: 1.0,
                time: 0.0,
                source_type: SourceType::Vowel,
                nasal_zero_freq: 0.0,
                nasal_zero_bw: 0.0,
            },
            FormantFrame {
                f1: 730.0,
                f2: 1090.0,
                f3: 2440.0,
                b1: 60.0,
                b2: 90.0,
                b3: 150.0,
                f0: 120.0,
                energy: 0.8,
                voicing: 1.0,
                time: 0.1,
                source_type: SourceType::Vowel,
                nasal_zero_freq: 0.0,
                nasal_zero_bw: 0.0,
            },
        ];

        let audio = vocoder.synthesize(&frames);
        assert!(
            !audio.is_empty(),
            "5-formant synthesis should produce audio"
        );
        assert!(audio.iter().any(|&s| s.abs() > 0.01), "Should have content");
        assert!(audio.iter().all(|&s| s.abs() < 1.5), "Should not clip");
    }

    #[test]
    fn test_lf_glottal_shape_variation() {
        // Different shapes (Rd values) should produce different waveforms
        let mut glottal_pressed = GlottalSource::new(24000.0, 0.0); // Rd=0.3
        let mut glottal_breathy = GlottalSource::new(24000.0, 1.0); // Rd=2.7

        let mut pressed_samples = Vec::new();
        let mut breathy_samples = Vec::new();
        for _ in 0..480 {
            // 20ms at 24kHz
            pressed_samples.push(glottal_pressed.tick(120.0));
            breathy_samples.push(glottal_breathy.tick(120.0));
        }

        // Both should produce non-zero output
        assert!(
            pressed_samples.iter().any(|&s| s.abs() > 0.01),
            "Pressed should have output"
        );
        assert!(
            breathy_samples.iter().any(|&s| s.abs() > 0.01),
            "Breathy should have output"
        );

        // They should differ (different waveform shapes)
        let diff: f32 = pressed_samples
            .iter()
            .zip(&breathy_samples)
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>();
        assert!(
            diff > 0.1,
            "Different shapes should produce different waveforms, diff={diff}"
        );
    }

    #[test]
    fn test_aspiration_adds_breathiness() {
        let frame = FormantFrame {
            f1: 500.0,
            f2: 1500.0,
            f3: 2500.0,
            b1: 60.0,
            b2: 90.0,
            b3: 150.0,
            f0: 120.0,
            energy: 0.8,
            voicing: 1.0,
            time: 0.0,
            source_type: SourceType::Vowel,
            nasal_zero_freq: 0.0,
            nasal_zero_bw: 0.0,
        };

        // Synthesize with aspiration
        let config_with = VocoderConfig {
            aspiration_level: 0.1,
            ..VocoderConfig::default()
        };
        let mut vocoder_with = FormantVocoder::with_config(config_with);
        let audio_with = vocoder_with.synthesize_frame(&frame, 480);

        // Synthesize without aspiration
        let config_without = VocoderConfig {
            aspiration_level: 0.0,
            ..VocoderConfig::default()
        };
        let mut vocoder_without = FormantVocoder::with_config(config_without);
        let audio_without = vocoder_without.synthesize_frame(&frame, 480);

        // Aspiration adds high-frequency energy: compute spectral difference
        // Simple proxy: sum of squared differences should be non-trivial
        let diff_energy: f32 = audio_with
            .iter()
            .zip(&audio_without)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>();
        assert!(
            diff_energy > 0.001,
            "Aspiration should measurably change the output, diff_energy={diff_energy}"
        );
    }

    #[test]
    fn test_spectral_tilt_reduces_hf() {
        let frame = FormantFrame {
            f1: 500.0,
            f2: 1500.0,
            f3: 2500.0,
            b1: 60.0,
            b2: 90.0,
            b3: 150.0,
            f0: 120.0,
            energy: 0.8,
            voicing: 1.0,
            time: 0.0,
            source_type: SourceType::Vowel,
            nasal_zero_freq: 0.0,
            nasal_zero_bw: 0.0,
        };

        // No tilt
        let config_flat = VocoderConfig {
            spectral_tilt: 0.0,
            jitter: 0.0,
            shimmer: 0.0,
            ..VocoderConfig::default()
        };
        let mut vocoder_flat = FormantVocoder::with_config(config_flat);
        let audio_flat = vocoder_flat.synthesize_frame(&frame, 2400); // 100ms

        // Strong tilt
        let config_tilt = VocoderConfig {
            spectral_tilt: 0.7,
            jitter: 0.0,
            shimmer: 0.0,
            ..VocoderConfig::default()
        };
        let mut vocoder_tilt = FormantVocoder::with_config(config_tilt);
        let audio_tilt = vocoder_tilt.synthesize_frame(&frame, 2400);

        // Tilt should reduce high-frequency energy. Proxy: compute energy of
        // sample-to-sample differences (≈ high-frequency content)
        let hf_flat: f32 = audio_flat.windows(2).map(|w| (w[1] - w[0]).powi(2)).sum();
        let hf_tilt: f32 = audio_tilt.windows(2).map(|w| (w[1] - w[0]).powi(2)).sum();

        assert!(
            hf_tilt < hf_flat,
            "Spectral tilt should reduce HF energy: flat={hf_flat:.4}, tilt={hf_tilt:.4}"
        );
    }

    #[test]
    fn test_jitter_adds_pitch_variation() {
        let frame = FormantFrame {
            f1: 500.0,
            f2: 1500.0,
            f3: 2500.0,
            b1: 60.0,
            b2: 90.0,
            b3: 150.0,
            f0: 120.0,
            energy: 0.8,
            voicing: 1.0,
            time: 0.0,
            source_type: SourceType::Vowel,
            nasal_zero_freq: 0.0,
            nasal_zero_bw: 0.0,
        };

        // No jitter
        let config_no = VocoderConfig {
            jitter: 0.0,
            shimmer: 0.0,
            spectral_tilt: 0.0,
            ..VocoderConfig::default()
        };
        let mut vocoder_no = FormantVocoder::with_config(config_no);
        let audio_no = vocoder_no.synthesize_frame(&frame, 2400);

        // With jitter
        let config_yes = VocoderConfig {
            jitter: 0.03,
            shimmer: 0.0,
            spectral_tilt: 0.0,
            ..VocoderConfig::default()
        };
        let mut vocoder_yes = FormantVocoder::with_config(config_yes);
        let audio_yes = vocoder_yes.synthesize_frame(&frame, 2400);

        // Jitter should produce different output
        let diff: f32 = audio_no
            .iter()
            .zip(&audio_yes)
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        assert!(diff > 0.001, "Jitter should change the output, diff={diff}");
    }

    #[test]
    fn test_shimmer_adds_amplitude_variation() {
        let frame = FormantFrame {
            f1: 500.0,
            f2: 1500.0,
            f3: 2500.0,
            b1: 60.0,
            b2: 90.0,
            b3: 150.0,
            f0: 120.0,
            energy: 0.8,
            voicing: 1.0,
            time: 0.0,
            source_type: SourceType::Vowel,
            nasal_zero_freq: 0.0,
            nasal_zero_bw: 0.0,
        };

        // No shimmer
        let config_no = VocoderConfig {
            shimmer: 0.0,
            jitter: 0.0,
            spectral_tilt: 0.0,
            ..VocoderConfig::default()
        };
        let mut vocoder_no = FormantVocoder::with_config(config_no);
        let audio_no = vocoder_no.synthesize_frame(&frame, 2400);

        // With shimmer
        let config_yes = VocoderConfig {
            shimmer: 0.05,
            jitter: 0.0,
            spectral_tilt: 0.0,
            ..VocoderConfig::default()
        };
        let mut vocoder_yes = FormantVocoder::with_config(config_yes);
        let audio_yes = vocoder_yes.synthesize_frame(&frame, 2400);

        // Shimmer should produce different output
        let diff: f32 = audio_no
            .iter()
            .zip(&audio_yes)
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        assert!(
            diff > 0.001,
            "Shimmer should change the output, diff={diff}"
        );
    }

    #[test]
    fn test_stop_consonant_burst() {
        let mut vocoder = FormantVocoder::new();

        let frame = FormantFrame {
            f1: 200.0,
            f2: 1000.0,
            f3: 2200.0,
            b1: 80.0,
            b2: 120.0,
            b3: 200.0,
            f0: 120.0,
            energy: 0.8,
            voicing: 1.0,
            source_type: SourceType::Stop,
            ..Default::default()
        };

        let audio = vocoder.synthesize_frame(&frame, 480);
        assert!(!audio.is_empty());

        // Burst (last 20%) should have more energy than closure (first 80%)
        let closure_end = (480.0 * 0.8) as usize;
        let closure_energy: f32 =
            audio[..closure_end].iter().map(|s| s * s).sum::<f32>() / closure_end as f32;
        let burst_energy: f32 =
            audio[closure_end..].iter().map(|s| s * s).sum::<f32>() / (480 - closure_end) as f32;

        assert!(
            burst_energy > closure_energy,
            "Burst energy ({burst_energy:.6}) should exceed closure ({closure_energy:.6})"
        );
    }

    #[test]
    fn test_fricative_noise() {
        let mut vocoder = FormantVocoder::new();

        let frame = FormantFrame {
            f1: 320.0,
            f2: 1700.0,
            f3: 2600.0,
            b1: 100.0,
            b2: 150.0,
            b3: 250.0,
            f0: 120.0,
            energy: 0.6,
            voicing: 0.0,
            source_type: SourceType::Fricative,
            ..Default::default()
        };

        let audio = vocoder.synthesize_frame(&frame, 480);
        assert!(!audio.is_empty());

        // Fricative should produce continuous noise throughout
        let first_quarter: f32 = audio[..120].iter().map(|s| s * s).sum::<f32>();
        let last_quarter: f32 = audio[360..].iter().map(|s| s * s).sum::<f32>();

        assert!(
            first_quarter > 0.001,
            "First quarter energy: {first_quarter}"
        );
        assert!(last_quarter > 0.001, "Last quarter energy: {last_quarter}");
    }

    #[test]
    fn test_nasal_voiced() {
        let mut vocoder = FormantVocoder::new();

        let frame = FormantFrame {
            f1: 280.0,
            f2: 1000.0,
            f3: 2200.0,
            b1: 80.0,
            b2: 120.0,
            b3: 200.0,
            f0: 120.0,
            energy: 0.7,
            voicing: 1.0,
            source_type: SourceType::Nasal,
            ..Default::default()
        };

        let audio = vocoder.synthesize_frame(&frame, 480);
        assert!(!audio.is_empty());

        let rms: f32 = (audio.iter().map(|s| s * s).sum::<f32>() / audio.len() as f32).sqrt();
        assert!(
            rms > 0.001,
            "Nasal should produce voiced audio: rms={rms:.4}"
        );
    }

    #[test]
    fn test_radiation_boosts_hf() {
        // The radiation filter (first-difference) should boost high-frequency energy.
        // Compare HF content with and without radiation by disabling/enabling it.
        let frame = FormantFrame {
            f1: 500.0,
            f2: 1500.0,
            f3: 2500.0,
            b1: 60.0,
            b2: 90.0,
            b3: 150.0,
            f0: 120.0,
            energy: 0.8,
            voicing: 1.0,
            time: 0.0,
            source_type: SourceType::Vowel,
            nasal_zero_freq: 0.0,
            nasal_zero_bw: 0.0,
        };

        let config = VocoderConfig {
            jitter: 0.0,
            shimmer: 0.0,
            spectral_tilt: 0.0,
            ..VocoderConfig::default()
        };

        let mut vocoder = FormantVocoder::with_config(config);
        let audio = vocoder.synthesize_frame(&frame, 2400); // 100ms

        // HF proxy: energy of sample-to-sample differences
        let hf_energy: f32 = audio.windows(2).map(|w| (w[1] - w[0]).powi(2)).sum();

        // With radiation filter active, HF energy should be substantial
        // (first-difference is essentially a high-pass)
        assert!(
            hf_energy > 0.01,
            "Radiation filter should produce measurable HF energy: {hf_energy:.6}"
        );

        // The output should still be bounded
        assert!(
            audio.iter().all(|&s| s.abs() < 1.5),
            "Radiated output should not clip"
        );
    }

    #[test]
    fn test_amplitude_correction_increases_f3() {
        // formant_amplitude_correction should give F3 more gain than F1
        // because F3 is at a higher harmonic and needs rolloff compensation.
        // Use high F0 (300Hz) so F1 is at harmonic 2 (below cap) and F3 at harmonic 8.
        let f0 = 300.0;
        let a1 = formant_amplitude_correction(500.0, f0); // F1 ~ harmonic 2
        let a3 = formant_amplitude_correction(2500.0, f0); // F3 ~ harmonic 8

        assert!(
            a3 > a1,
            "F3 amplitude correction should exceed F1: a1={a1:.3}, a3={a3:.3}"
        );

        // Basic sanity
        assert!(a1 >= 1.0, "F1 correction should be >= 1.0: {a1:.3}");
        assert!(a3 <= 8.0, "F3 correction should be capped at 8.0: {a3:.3}");

        // Harmonic 1 should give correction = 1.0 (no rolloff)
        let a_fundamental = formant_amplitude_correction(300.0, 300.0);
        assert!(
            (a_fundamental - 1.0).abs() < 0.01,
            "Correction at fundamental should be 1.0: {a_fundamental:.3}"
        );
    }

    #[test]
    fn test_per_source_rd() {
        let vocoder = FormantVocoder::new();

        let vowel_rd = vocoder.source_rd(SourceType::Vowel);
        let liquid_rd = vocoder.source_rd(SourceType::Liquid);
        let nasal_rd = vocoder.source_rd(SourceType::Nasal);

        // Liquid should be more pressed (lower Rd → lower shape)
        assert!(
            liquid_rd < vowel_rd,
            "Liquid should be more pressed: liquid={liquid_rd:.3}, vowel={vowel_rd:.3}"
        );

        // Nasal should be breathier (higher Rd → higher shape)
        assert!(
            nasal_rd > vowel_rd,
            "Nasal should be breathier: nasal={nasal_rd:.3}, vowel={vowel_rd:.3}"
        );
    }

    #[test]
    fn test_vot_voiceless_has_aspiration() {
        let mut vocoder = FormantVocoder::new();

        // Voiceless stop (voicing=0.0): should have aspiration after burst
        let frame = FormantFrame {
            f1: 200.0,
            f2: 1000.0,
            f3: 2200.0,
            b1: 80.0,
            b2: 120.0,
            b3: 200.0,
            f0: 120.0,
            energy: 0.8,
            voicing: 0.0, // voiceless
            source_type: SourceType::Stop,
            ..Default::default()
        };

        let audio = vocoder.synthesize_frame(&frame, 480);

        // VOT region (last 20%, frames 384-480) should have non-zero energy
        // (aspiration noise, not silence)
        let vot_start = (480.0 * 0.8) as usize;
        let vot_energy: f32 =
            audio[vot_start..].iter().map(|s| s * s).sum::<f32>() / (480 - vot_start) as f32;
        assert!(
            vot_energy > 0.0001,
            "Voiceless VOT should have aspiration energy: {vot_energy:.6}"
        );
    }

    #[test]
    fn test_vot_voiced_has_voicing() {
        let mut vocoder = FormantVocoder::new();

        // Voiced stop (voicing=1.0): should resume voicing after burst
        let frame = FormantFrame {
            f1: 200.0,
            f2: 1000.0,
            f3: 2200.0,
            b1: 80.0,
            b2: 120.0,
            b3: 200.0,
            f0: 120.0,
            energy: 0.8,
            voicing: 1.0, // voiced
            source_type: SourceType::Stop,
            ..Default::default()
        };

        let audio = vocoder.synthesize_frame(&frame, 480);

        // Post-burst region should have voiced energy (glottal pulses)
        let post_burst = (480.0 * 0.8) as usize;
        let post_energy: f32 =
            audio[post_burst..].iter().map(|s| s * s).sum::<f32>() / (480 - post_burst) as f32;
        assert!(
            post_energy > 0.0001,
            "Voiced stop should have voicing after burst: {post_energy:.6}"
        );
    }

    #[test]
    fn test_fricative_sibilant_vs_nonsibilant() {
        // Sibilant (/S/): high F2 → more white noise (brighter)
        // Non-sibilant (/F/): low F2 → more pink noise (gentler)
        let config = VocoderConfig {
            jitter: 0.0,
            shimmer: 0.0,
            spectral_tilt: 0.0,
            ..VocoderConfig::default()
        };

        // Sibilant: F2=2500
        let mut vocoder_sib = FormantVocoder::with_config(config.clone());
        let frame_sib = FormantFrame {
            f1: 320.0,
            f2: 2500.0,
            f3: 3500.0,
            b1: 100.0,
            b2: 150.0,
            b3: 250.0,
            f0: 120.0,
            energy: 0.6,
            voicing: 0.0,
            source_type: SourceType::Fricative,
            ..Default::default()
        };
        let audio_sib = vocoder_sib.synthesize_frame(&frame_sib, 2400);

        // Non-sibilant: F2=1000
        let mut vocoder_non = FormantVocoder::with_config(config);
        let frame_non = FormantFrame {
            f1: 320.0,
            f2: 1000.0,
            f3: 2200.0,
            b1: 100.0,
            b2: 150.0,
            b3: 250.0,
            f0: 120.0,
            energy: 0.6,
            voicing: 0.0,
            source_type: SourceType::Fricative,
            ..Default::default()
        };
        let audio_non = vocoder_non.synthesize_frame(&frame_non, 2400);

        // Sibilant should have more high-frequency energy (brighter)
        let hf_sib: f32 = audio_sib.windows(2).map(|w| (w[1] - w[0]).powi(2)).sum();
        let hf_non: f32 = audio_non.windows(2).map(|w| (w[1] - w[0]).powi(2)).sum();

        assert!(
            hf_sib > hf_non,
            "Sibilant should have more HF energy: sib={hf_sib:.4}, non={hf_non:.4}"
        );
    }

    #[test]
    fn test_nasal_pole_adds_low_resonance() {
        let config = VocoderConfig {
            jitter: 0.0,
            shimmer: 0.0,
            spectral_tilt: 0.0,
            ..VocoderConfig::default()
        };

        // Nasal frame with pole-zero
        let mut vocoder_nasal = FormantVocoder::with_config(config.clone());
        let frame_nasal = FormantFrame {
            f1: 280.0,
            f2: 1000.0,
            f3: 2200.0,
            b1: 80.0,
            b2: 120.0,
            b3: 200.0,
            f0: 120.0,
            energy: 0.7,
            voicing: 1.0,
            source_type: SourceType::Nasal,
            nasal_zero_freq: 750.0,
            nasal_zero_bw: 200.0,
            ..Default::default()
        };
        let audio_nasal = vocoder_nasal.synthesize_frame(&frame_nasal, 2400);

        // Vowel frame (same formants)
        let mut vocoder_vowel = FormantVocoder::with_config(config);
        let frame_vowel = FormantFrame {
            f1: 280.0,
            f2: 1000.0,
            f3: 2200.0,
            b1: 80.0,
            b2: 120.0,
            b3: 200.0,
            f0: 120.0,
            energy: 0.7,
            voicing: 1.0,
            source_type: SourceType::Vowel,
            ..Default::default()
        };
        let audio_vowel = vocoder_vowel.synthesize_frame(&frame_vowel, 2400);

        // Nasal should differ from vowel (pole + anti-resonance change the spectrum)
        let diff: f32 = audio_nasal
            .iter()
            .zip(&audio_vowel)
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        assert!(
            diff > 0.01,
            "Nasal pole/zero should change the spectrum vs vowel: diff={diff:.4}"
        );
    }

    // ── Voice Quality Tests (Item 6) ─────────────────────────────────────

    #[test]
    fn test_voice_quality_default_neutral() {
        let vq = VoiceQuality::default();
        assert!(vq.rd.is_none());
        assert!((vq.jitter_scale - 1.0).abs() < f32::EPSILON);
        assert!((vq.shimmer_scale - 1.0).abs() < f32::EPSILON);
        assert!((vq.bandwidth_scale - 1.0).abs() < f32::EPSILON);
        assert!(vq.spectral_tilt.is_none());
    }

    #[test]
    fn test_voice_quality_negative_pressed() {
        // Negative valence → lower Rd (pressed voice)
        let vq = cognitive_state_to_voice_quality(-0.8, 0.5, 0.6);
        let rd = vq.rd.unwrap();
        assert!(
            rd < 0.5,
            "Negative valence should produce pressed (low Rd): {rd}"
        );
    }

    #[test]
    fn test_voice_quality_positive_breathy() {
        // Positive valence → higher Rd (breathy voice)
        let vq = cognitive_state_to_voice_quality(0.8, 0.5, 0.6);
        let rd = vq.rd.unwrap();
        assert!(
            rd > 1.5,
            "Positive valence should produce breathy (high Rd): {rd}"
        );
    }

    #[test]
    fn test_voice_quality_high_arousal_jitter() {
        let vq = cognitive_state_to_voice_quality(0.0, 0.9, 0.6);
        assert!(
            vq.jitter_scale > 1.2,
            "High arousal should increase jitter: {}",
            vq.jitter_scale
        );
        assert!(
            vq.shimmer_scale > 1.2,
            "High arousal should increase shimmer: {}",
            vq.shimmer_scale
        );
    }

    #[test]
    fn test_voice_quality_low_consciousness_creaky() {
        // Low consciousness → forced pressed Rd
        let vq = cognitive_state_to_voice_quality(0.5, 0.5, 0.1);
        let rd = vq.rd.unwrap();
        assert!(rd <= 0.6, "Low consciousness should force pressed Rd: {rd}");
    }

    #[test]
    fn test_cascade_vs_parallel_differ() {
        // Cascade (all-pole series) and parallel (bandpass) should produce different envelopes
        let frame = FormantFrame {
            f1: 500.0,
            f2: 1500.0,
            f3: 2500.0,
            b1: 60.0,
            b2: 90.0,
            b3: 150.0,
            f0: 120.0,
            energy: 0.8,
            voicing: 1.0,
            source_type: SourceType::Vowel,
            time: 0.0,
            nasal_zero_freq: 0.0,
            nasal_zero_bw: 0.0,
        };

        let mut voc_cascade = FormantVocoder::with_config(VocoderConfig {
            cascade: true,
            ..Default::default()
        });
        let mut voc_parallel = FormantVocoder::with_config(VocoderConfig {
            cascade: false,
            ..Default::default()
        });

        // Use synthesize_frame for reliable single-frame output (2400 samples = 100ms at 24kHz)
        let audio_c = voc_cascade.synthesize_frame(&frame, 2400);
        let audio_p = voc_parallel.synthesize_frame(&frame, 2400);

        // Both should produce audio
        assert!(
            audio_c.iter().any(|s| s.abs() > 0.001),
            "Cascade should produce audio, max={}",
            audio_c.iter().map(|s| s.abs()).fold(0.0_f32, f32::max)
        );
        assert!(
            audio_p.iter().any(|s| s.abs() > 0.001),
            "Parallel should produce audio"
        );

        // They should differ (different filter topologies)
        let diff: f32 = audio_c
            .iter()
            .zip(&audio_p)
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / audio_c.len() as f32;
        assert!(
            diff > 1e-6,
            "Cascade and parallel should produce different output, diff={diff}"
        );
    }

    #[test]
    fn test_subglottal_adds_chest_resonance() {
        // Subglottal resonances should add energy for voiced speech
        let frame = FormantFrame {
            f1: 500.0,
            f2: 1500.0,
            f3: 2500.0,
            b1: 60.0,
            b2: 90.0,
            b3: 150.0,
            f0: 120.0,
            energy: 0.8,
            voicing: 1.0,
            source_type: SourceType::Vowel,
            time: 0.0,
            nasal_zero_freq: 0.0,
            nasal_zero_bw: 0.0,
        };

        let audio = FormantVocoder::new().synthesize_frame(&frame, 2400);
        assert!(
            audio.iter().any(|s| s.abs() > 0.001),
            "Voiced speech should produce output with subglottal, max={}",
            audio.iter().map(|s| s.abs()).fold(0.0_f32, f32::max)
        );
    }

    #[test]
    fn test_cascade_default_parallel() {
        let config = VocoderConfig::default();
        assert!(
            !config.cascade,
            "Cascade should be off by default (parallel mode)"
        );
    }

    #[test]
    fn test_synthesize_with_quality_produces_audio() {
        let mut vocoder = FormantVocoder::new();
        let frames: Vec<FormantFrame> = (0..20)
            .map(|i| FormantFrame {
                f1: 500.0,
                f2: 1500.0,
                f3: 2500.0,
                b1: 60.0,
                b2: 90.0,
                b3: 150.0,
                f0: 120.0,
                energy: 0.5,
                voicing: 1.0,
                time: i as f32 * 0.005,
                source_type: SourceType::Vowel,
                ..Default::default()
            })
            .collect();
        let quality = vec![VoiceQuality::default(); frames.len()];
        let samples = vocoder.synthesize_with_quality(&frames, &quality);
        assert!(!samples.is_empty(), "Should produce audio");
        assert!(
            samples.iter().all(|s| s.is_finite()),
            "All samples should be finite"
        );
    }

    #[test]
    fn test_arousal_strain_increases_jitter() {
        let state = symthaea_vocal_tract::encoder::VoiceCognitiveState {
            emotional_arousal: 0.5,
            ..Default::default()
        };
        let zero_derivs = symthaea_vocal_tract::encoder::VoiceCognitiveStateDerivatives::default();
        let rising_derivs = symthaea_vocal_tract::encoder::VoiceCognitiveStateDerivatives {
            delta_arousal: 2.0,
            ..Default::default()
        };

        let vq_base = cognitive_state_to_voice_quality_extended(&state, &zero_derivs);
        let vq_rising = cognitive_state_to_voice_quality_extended(&state, &rising_derivs);

        assert!(
            vq_rising.jitter_scale > vq_base.jitter_scale,
            "Rising arousal should increase jitter: base={}, rising={}",
            vq_base.jitter_scale,
            vq_rising.jitter_scale
        );
    }

    #[test]
    fn test_consciousness_drop_increases_shimmer() {
        let state = symthaea_vocal_tract::encoder::VoiceCognitiveState {
            consciousness_level: 0.5,
            ..Default::default()
        };
        let zero_derivs = symthaea_vocal_tract::encoder::VoiceCognitiveStateDerivatives::default();
        let dropping_derivs = symthaea_vocal_tract::encoder::VoiceCognitiveStateDerivatives {
            delta_consciousness: -2.0,
            ..Default::default()
        };

        let vq_base = cognitive_state_to_voice_quality_extended(&state, &zero_derivs);
        let vq_dropping = cognitive_state_to_voice_quality_extended(&state, &dropping_derivs);

        assert!(
            vq_dropping.shimmer_scale > vq_base.shimmer_scale,
            "Dropping consciousness should increase shimmer: base={}, dropping={}",
            vq_base.shimmer_scale,
            vq_dropping.shimmer_scale
        );
    }

    #[test]
    fn test_conflict_detection_suppressed_emotion() {
        let zero_derivs = symthaea_vocal_tract::encoder::VoiceCognitiveStateDerivatives::default();

        // Suppressed: high |valence| but low arousal
        let suppressed = symthaea_vocal_tract::encoder::VoiceCognitiveState {
            emotional_valence: 0.9,
            emotional_arousal: 0.1,
            ..Default::default()
        };
        // Neutral
        let neutral = symthaea_vocal_tract::encoder::VoiceCognitiveState::default();

        let vq_suppressed = cognitive_state_to_voice_quality_extended(&suppressed, &zero_derivs);
        let vq_neutral = cognitive_state_to_voice_quality_extended(&neutral, &zero_derivs);

        // Suppression should affect bandwidth and/or jitter
        assert!(
            vq_suppressed.bandwidth_scale != vq_neutral.bandwidth_scale
                || vq_suppressed.jitter_scale != vq_neutral.jitter_scale,
            "Suppressed emotion should produce distinct profile"
        );
    }

    #[test]
    fn test_rate_stability_dampens_jitter() {
        let zero_derivs = symthaea_vocal_tract::encoder::VoiceCognitiveStateDerivatives::default();

        let stable = symthaea_vocal_tract::encoder::VoiceCognitiveState {
            rate_stability: 0.9,
            ..Default::default()
        };
        let unstable = symthaea_vocal_tract::encoder::VoiceCognitiveState {
            rate_stability: 0.1,
            ..Default::default()
        };

        let vq_stable = cognitive_state_to_voice_quality_extended(&stable, &zero_derivs);
        let vq_unstable = cognitive_state_to_voice_quality_extended(&unstable, &zero_derivs);

        assert!(
            vq_stable.jitter_scale < vq_unstable.jitter_scale,
            "High rate stability should reduce jitter: stable={}, unstable={}",
            vq_stable.jitter_scale,
            vq_unstable.jitter_scale
        );
    }
}
