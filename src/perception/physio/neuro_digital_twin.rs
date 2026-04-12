// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Closed-loop neuro-digital twin: bidirectional bridge between
//! biological EEG signals and the artificial consciousness pipeline.
//!
//! The twin maintains a synchronized model of biological neural state
//! decoded from EEG, maps it to neuromodulator levels, and tracks
//! bio-digital divergence as a consciousness validation metric.
//!
//! Architecture:
//! ```text
//!   Live EEG → EegDecoder → EegNeuromodFeatures → NeuromodBath blend
//!              ↓                                         ↓
//!        Twin State ←── divergence tracking ←── Digital State
//!              ↓
//!   Calibration feedback (per-subject adaptation)
//! ```
//!
//! References:
//! - Nunez, P. L. & Srinivasan, R. (2006). EEG of the Cerebral Cortex.
//! - Klimesch, W. (1999). EEG alpha and theta oscillations.
//! - Davidson, R. J. (2004). Frontal asymmetry and emotion.
//! - Aston-Jones, G. & Cohen, J. D. (2005). LC-NE and arousal.
//! - Harmony, T. (2013). Theta activity and cognitive function.

use std::collections::VecDeque;

// ============================================================================
// EEG Channel Definitions
// ============================================================================

/// Standard 10-20 EEG electrode positions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EegChannel {
    /// Frontal pole midline to vertex (common sleep montage)
    FpzCz,
    /// Parietal midline to occipital (common sleep montage)
    PzOz,
    /// Left frontal pole
    Fp1,
    /// Right frontal pole
    Fp2,
    /// Left frontal
    F3,
    /// Right frontal
    F4,
    /// Left central
    C3,
    /// Right central
    C4,
    /// Left parietal
    P3,
    /// Right parietal
    P4,
    /// Left occipital
    O1,
    /// Right occipital
    O2,
    /// Frontal midline
    Fz,
    /// Central midline
    Cz,
    /// Parietal midline
    Pz,
}

impl EegChannel {
    /// Total number of channel variants.
    pub const COUNT: usize = 15;
}

// ============================================================================
// EEG Band Power
// ============================================================================

/// Power spectral density decomposed into canonical EEG frequency bands.
#[derive(Debug, Clone, Default)]
pub struct EegBandPower {
    /// Delta band (0.5–4 Hz): deep sleep, unconscious processes.
    pub delta: f32,
    /// Theta band (4–8 Hz): meditation, memory encoding.
    pub theta: f32,
    /// Alpha band (8–13 Hz): relaxed wakefulness, cortical inhibition.
    pub alpha: f32,
    /// Beta band (13–30 Hz): active cognition, anxiety.
    pub beta: f32,
    /// Gamma band (30–100 Hz): binding, conscious integration.
    pub gamma: f32,
}

impl EegBandPower {
    /// Total power across all bands.
    pub fn total(&self) -> f32 {
        self.delta + self.theta + self.alpha + self.beta + self.gamma
    }
}

// ============================================================================
// EEG-Derived Neuromodulator Features
// ============================================================================

/// Features extracted from multichannel EEG that map to neuromodulator axes.
///
/// Each feature has a well-established neuroscience correlate linking
/// scalp EEG patterns to subcortical neuromodulatory systems.
#[derive(Debug, Clone, Default)]
pub struct EegNeuromodFeatures {
    /// Frontal alpha asymmetry: (F4_alpha - F3_alpha).
    /// Positive → approach/positive affect → serotonin.
    /// Science: Davidson (2004).
    pub frontal_alpha_asymmetry: f32,

    /// Frontal midline theta power (Fz).
    /// High theta → sustained attention → acetylcholine.
    /// Science: Harmony (2013).
    pub theta_frontal_midline: f32,

    /// Beta-gamma power ratio across all channels.
    /// High ratio → cortical arousal → noradrenaline.
    /// Science: Aston-Jones & Cohen (2005).
    pub beta_gamma_ratio: f32,

    /// Delta band power (global average).
    /// High delta → sleep pressure → adenosine.
    pub delta_power: f32,

    /// Inter-hemispheric alpha coherence.
    /// High coherence → integration → oxytocin.
    pub alpha_coherence: f32,

    /// Gamma-band synchrony across channels.
    /// High synchrony → conscious binding.
    pub gamma_synchrony: f32,

    /// Theta/beta ratio (global average).
    /// Low ratio → good attentional regulation → dopamine.
    pub theta_beta_ratio: f32,
}

// ============================================================================
// Subject Calibration
// ============================================================================

/// Per-subject calibration parameters learned from resting-state EEG.
///
/// Individual differences in tonic EEG are substantial (Klimesch 1999);
/// calibration normalizes these before mapping to neuromodulators.
#[derive(Debug, Clone)]
pub struct SubjectCalibration {
    /// Individual alpha peak frequency baseline (typically 8–12 Hz).
    pub alpha_baseline: f32,
    /// Tonic frontal asymmetry offset at rest.
    pub asymmetry_offset: f32,
    /// Per-feature multiplicative scaling factors (learned).
    /// Order: [asymmetry, theta_fm, beta_gamma, delta, alpha_coh, gamma_sync, theta_beta]
    pub scaling_factors: [f32; 7],
    /// Number of calibration samples collected.
    pub samples_collected: usize,
    /// Whether calibration is complete.
    pub is_calibrated: bool,
}

impl Default for SubjectCalibration {
    fn default() -> Self {
        Self {
            alpha_baseline: 0.0,
            asymmetry_offset: 0.0,
            scaling_factors: [1.0; 7],
            samples_collected: 0,
            is_calibrated: false,
        }
    }
}

impl SubjectCalibration {
    /// Minimum samples required for calibration.
    const MIN_CALIBRATION_SAMPLES: usize = 30;

    /// Apply calibration: subtract baselines and scale features.
    pub fn apply(&self, raw: &EegNeuromodFeatures) -> EegNeuromodFeatures {
        if !self.is_calibrated {
            return raw.clone();
        }
        EegNeuromodFeatures {
            frontal_alpha_asymmetry: (raw.frontal_alpha_asymmetry - self.asymmetry_offset)
                * self.scaling_factors[0],
            theta_frontal_midline: raw.theta_frontal_midline * self.scaling_factors[1],
            beta_gamma_ratio: raw.beta_gamma_ratio * self.scaling_factors[2],
            delta_power: raw.delta_power * self.scaling_factors[3],
            alpha_coherence: raw.alpha_coherence * self.scaling_factors[4],
            gamma_synchrony: raw.gamma_synchrony * self.scaling_factors[5],
            theta_beta_ratio: raw.theta_beta_ratio * self.scaling_factors[6],
        }
    }
}

// ============================================================================
// Digital Neuromodulator Snapshot
// ============================================================================

/// Current artificial neuromodulator bath state, projected onto the
/// 7 axes that have EEG correlates.
#[derive(Debug, Clone, Default)]
pub struct DigitalNeuromodSnapshot {
    pub serotonin: f32,
    pub dopamine: f32,
    pub noradrenaline: f32,
    pub acetylcholine: f32,
    pub adenosine: f32,
    pub oxytocin: f32,
    pub gaba: f32,
}

impl DigitalNeuromodSnapshot {
    /// Return the snapshot as a 7-element array in canonical order.
    fn as_array(&self) -> [f32; 7] {
        [
            self.serotonin,
            self.dopamine,
            self.noradrenaline,
            self.acetylcholine,
            self.adenosine,
            self.oxytocin,
            self.gaba,
        ]
    }
}

// ============================================================================
// Twin State
// ============================================================================

/// Divergence tracking window capacity.
const DIVERGENCE_HISTORY_CAP: usize = 100;

/// Synchronized biological-digital twin state.
#[derive(Debug, Clone)]
pub struct TwinState {
    /// Latest EEG-derived neuromod features.
    pub biological: EegNeuromodFeatures,
    /// Current artificial bath state snapshot.
    pub digital: DigitalNeuromodSnapshot,
    /// Euclidean distance between biological and digital state vectors.
    pub divergence: f64,
    /// Rolling history of divergence values.
    pub divergence_history: VecDeque<f64>,
    /// Synchronization quality: 1.0 - normalized_divergence (clamped 0..1).
    pub sync_quality: f64,
    /// Cycle at which the twin was last updated.
    pub last_update_cycle: u64,
}

impl Default for TwinState {
    fn default() -> Self {
        Self {
            biological: EegNeuromodFeatures::default(),
            digital: DigitalNeuromodSnapshot::default(),
            divergence: 0.0,
            divergence_history: VecDeque::with_capacity(DIVERGENCE_HISTORY_CAP),
            sync_quality: 1.0,
            last_update_cycle: 0,
        }
    }
}

// ============================================================================
// Neuromod Delta (output)
// ============================================================================

/// Signed deltas to apply to the neuromodulator bath, derived from
/// the bio-digital mapping.
#[derive(Debug, Clone, Default)]
pub struct NeuromodDelta {
    pub delta_serotonin: f32,
    pub delta_dopamine: f32,
    pub delta_noradrenaline: f32,
    pub delta_acetylcholine: f32,
    pub delta_adenosine: f32,
    pub delta_oxytocin: f32,
}

// ============================================================================
// PID Controller
// ============================================================================

/// Simple PID controller for convergence tracking.
#[derive(Debug, Clone)]
pub struct PidController {
    pub kp: f64,
    pub ki: f64,
    pub kd: f64,
    integral: f64,
    prev_error: f64,
    output_min: f64,
    output_max: f64,
}

impl PidController {
    /// Create a new PID controller with given gains and output bounds.
    pub fn new(kp: f64, ki: f64, kd: f64) -> Self {
        Self {
            kp,
            ki,
            kd,
            integral: 0.0,
            prev_error: 0.0,
            output_min: -1.0,
            output_max: 1.0,
        }
    }

    /// Create a PID controller with custom output limits.
    pub fn with_limits(mut self, min: f64, max: f64) -> Self {
        self.output_min = min;
        self.output_max = max;
        self
    }

    /// Compute control output given target, actual measurement, and time step.
    pub fn update(&mut self, target: f64, actual: f64, dt: f64) -> f64 {
        let error = target - actual;
        self.integral += error * dt;
        // Anti-windup: clamp integral
        let integral_limit = (self.output_max - self.output_min) / self.ki.abs().max(1e-9);
        self.integral = self.integral.clamp(-integral_limit, integral_limit);

        let derivative = if dt > 0.0 {
            (error - self.prev_error) / dt
        } else {
            0.0
        };
        self.prev_error = error;

        let output = self.kp * error + self.ki * self.integral + self.kd * derivative;
        output.clamp(self.output_min, self.output_max)
    }

    /// Reset internal state.
    pub fn reset(&mut self) {
        self.integral = 0.0;
        self.prev_error = 0.0;
    }
}

// ============================================================================
// Neuro-Digital Twin (Main Engine)
// ============================================================================

/// Default EEG buffer capacity in samples.
const DEFAULT_BUFFER_CAPACITY: usize = 256;

/// Default sample rate in Hz.
const DEFAULT_SAMPLE_RATE: f32 = 256.0;

/// Default blend ratio (30% biological influence).
const DEFAULT_BLEND_RATIO: f32 = 0.3;

/// Normalization ceiling for divergence → sync_quality mapping.
/// A divergence of this value maps to sync_quality = 0.
const MAX_DIVERGENCE_FOR_QUALITY: f64 = 3.0;

/// Closed-loop neuro-digital twin engine.
///
/// Ingests raw EEG samples, extracts neuromodulator-relevant features,
/// computes divergence against the artificial bath, and produces blended
/// deltas for injection into the cognitive loop.
#[derive(Debug, Clone)]
pub struct NeuroDigitalTwin {
    /// Synchronized twin state.
    pub state: TwinState,
    /// Per-subject calibration parameters.
    pub calibration: SubjectCalibration,
    /// Blend ratio: 0.0 = fully digital, 1.0 = fully biological.
    pub blend_ratio: f32,
    /// Ring buffer of raw EEG samples (each sample is a channel-indexed vector).
    eeg_buffer: VecDeque<Vec<(EegChannel, f32)>>,
    /// Maximum samples in the ring buffer.
    buffer_capacity: usize,
    /// EEG sampling rate in Hz.
    sample_rate: f32,
    /// Whether the twin is actively processing.
    pub is_active: bool,
    /// Convergence PID controller (tracks divergence → 0).
    pub pid: PidController,
}

impl NeuroDigitalTwin {
    /// Create a new neuro-digital twin.
    ///
    /// # Arguments
    /// * `sample_rate` - EEG sample rate in Hz (typically 256).
    /// * `blend_ratio` - 0.0 = fully digital, 1.0 = fully biological.
    pub fn new(sample_rate: f32, blend_ratio: f32) -> Self {
        Self {
            state: TwinState::default(),
            calibration: SubjectCalibration::default(),
            blend_ratio: blend_ratio.clamp(0.0, 1.0),
            eeg_buffer: VecDeque::with_capacity(DEFAULT_BUFFER_CAPACITY),
            buffer_capacity: DEFAULT_BUFFER_CAPACITY,
            sample_rate: sample_rate.max(1.0),
            is_active: false,
            pid: PidController::new(0.5, 0.1, 0.05),
        }
    }

    /// Create with default parameters (256 Hz, 0.3 blend).
    pub fn default_twin() -> Self {
        Self::new(DEFAULT_SAMPLE_RATE, DEFAULT_BLEND_RATIO)
    }

    /// Feed a single EEG sample (one time-point, multiple channels).
    ///
    /// When the buffer reaches capacity, the oldest sample is evicted.
    pub fn feed_eeg_sample(&mut self, channels: &[(EegChannel, f32)]) {
        if self.eeg_buffer.len() >= self.buffer_capacity {
            self.eeg_buffer.pop_front();
        }
        self.eeg_buffer.push_back(channels.to_vec());
    }

    /// Whether the buffer has enough samples for feature extraction.
    pub fn buffer_ready(&self) -> bool {
        self.eeg_buffer.len() >= self.buffer_capacity / 2
    }

    /// Extract neuromod-relevant features from buffered EEG.
    ///
    /// Returns `None` if not calibrated or insufficient data.
    pub fn extract_features(&self) -> Option<EegNeuromodFeatures> {
        if !self.calibration.is_calibrated {
            return None;
        }
        if !self.buffer_ready() {
            return None;
        }

        let band_powers = self.compute_band_powers();
        let raw = self.features_from_band_powers(&band_powers);
        Some(self.calibration.apply(&raw))
    }

    /// Compute approximate band powers per channel using zero-crossing rate
    /// heuristic (lightweight Welch's approximation).
    ///
    /// For each channel, counts zero crossings to estimate dominant frequency,
    /// then distributes energy into the appropriate band.
    fn compute_band_powers(&self) -> Vec<(EegChannel, EegBandPower)> {
        // Collect per-channel time series from the buffer.
        let mut channel_series: std::collections::HashMap<EegChannel, Vec<f32>> =
            std::collections::HashMap::new();

        for sample in &self.eeg_buffer {
            for &(ch, val) in sample {
                channel_series.entry(ch).or_default().push(val);
            }
        }

        let mut results = Vec::new();
        for (ch, series) in &channel_series {
            if series.len() < 4 {
                continue;
            }
            let bp = Self::estimate_band_power(series, self.sample_rate);
            results.push((*ch, bp));
        }
        results
    }

    /// Estimate band power distribution from a single-channel time series.
    ///
    /// Uses a multi-pass bandpass energy estimation:
    /// 1. Compute total signal variance (total power).
    /// 2. Count zero-crossings → dominant frequency estimate.
    /// 3. Distribute power into bands based on frequency profile.
    ///
    /// This is a lightweight approximation; a real deployment would use FFT.
    fn estimate_band_power(series: &[f32], sample_rate: f32) -> EegBandPower {
        let n = series.len() as f32;
        if n < 4.0 {
            return EegBandPower::default();
        }

        // Total variance (proxy for total power).
        let mean = series.iter().sum::<f32>() / n;
        let variance = series.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;

        // Zero-crossing rate → dominant frequency estimate.
        let mut crossings = 0u32;
        let centered: Vec<f32> = series.iter().map(|x| x - mean).collect();
        for i in 1..centered.len() {
            if centered[i].signum() != centered[i - 1].signum() && centered[i - 1] != 0.0 {
                crossings += 1;
            }
        }
        let dominant_freq = (crossings as f32 * sample_rate) / (2.0 * (n - 1.0));

        // Distribute power into bands using Gaussian weighting around
        // the dominant frequency.
        let band_centers = [2.0f32, 6.0, 10.5, 21.5, 50.0]; // delta, theta, alpha, beta, gamma
        let band_widths = [3.5f32, 4.0, 5.0, 17.0, 70.0];

        let mut weights: [f32; 5] = [0.0; 5];
        for (i, (&center, &width)) in band_centers.iter().zip(band_widths.iter()).enumerate() {
            let d = (dominant_freq - center) / width;
            weights[i] = (-0.5 * d * d).exp();
        }
        let weight_sum: f32 = weights.iter().sum::<f32>().max(1e-9);
        for w in &mut weights {
            *w /= weight_sum;
        }

        EegBandPower {
            delta: variance * weights[0],
            theta: variance * weights[1],
            alpha: variance * weights[2],
            beta: variance * weights[3],
            gamma: variance * weights[4],
        }
    }

    /// Derive high-level neuromod features from per-channel band powers.
    fn features_from_band_powers(
        &self,
        band_powers: &[(EegChannel, EegBandPower)],
    ) -> EegNeuromodFeatures {
        let get = |ch: EegChannel| -> Option<&EegBandPower> {
            band_powers.iter().find(|(c, _)| *c == ch).map(|(_, bp)| bp)
        };

        // Frontal alpha asymmetry: F4 - F3
        let frontal_alpha_asymmetry = match (get(EegChannel::F4), get(EegChannel::F3)) {
            (Some(f4), Some(f3)) => f4.alpha - f3.alpha,
            _ => 0.0,
        };

        // Frontal midline theta
        let theta_frontal_midline = get(EegChannel::Fz).map_or(0.0, |bp| bp.theta);

        // Average beta and gamma across all channels
        let (mut avg_beta, mut avg_gamma, mut avg_delta, mut count) = (0.0f32, 0.0f32, 0.0f32, 0);
        for (_, bp) in band_powers {
            avg_beta += bp.beta;
            avg_gamma += bp.gamma;
            avg_delta += bp.delta;
            count += 1;
        }
        if count > 0 {
            avg_beta /= count as f32;
            avg_gamma /= count as f32;
            avg_delta /= count as f32;
        }

        let beta_gamma_ratio = if avg_gamma > 1e-9 {
            avg_beta / avg_gamma
        } else {
            0.0
        };

        // Alpha coherence: correlation between left and right hemisphere alpha.
        let alpha_coherence = match (get(EegChannel::F3), get(EegChannel::F4)) {
            (Some(f3), Some(f4)) => {
                let sum = f3.alpha + f4.alpha;
                if sum > 1e-9 {
                    1.0 - ((f3.alpha - f4.alpha).abs() / sum)
                } else {
                    0.0
                }
            }
            _ => 0.0,
        };

        // Gamma synchrony: variance of gamma across channels (low variance = high synchrony).
        let gamma_values: Vec<f32> = band_powers.iter().map(|(_, bp)| bp.gamma).collect();
        let gamma_synchrony = if gamma_values.len() > 1 {
            let gm = gamma_values.iter().sum::<f32>() / gamma_values.len() as f32;
            let gv = gamma_values.iter().map(|g| (g - gm).powi(2)).sum::<f32>()
                / gamma_values.len() as f32;
            // Invert: low variance = high synchrony. Scale to 0..1.
            1.0 / (1.0 + gv.sqrt() * 10.0)
        } else {
            0.0
        };

        // Theta/beta ratio
        let theta_beta_ratio = if avg_beta > 1e-9 {
            let avg_theta = band_powers
                .iter()
                .map(|(_, bp)| bp.theta)
                .sum::<f32>()
                .max(0.0)
                / (count as f32).max(1.0);
            avg_theta / avg_beta
        } else {
            0.0
        };

        EegNeuromodFeatures {
            frontal_alpha_asymmetry,
            theta_frontal_midline,
            beta_gamma_ratio,
            delta_power: avg_delta,
            alpha_coherence,
            gamma_synchrony,
            theta_beta_ratio,
        }
    }

    /// Decode EEG features into neuromodulator deltas.
    ///
    /// Each mapping is grounded in neuroscience literature:
    /// - frontal_alpha_asymmetry → serotonin (Davidson 2004)
    /// - theta_frontal_midline → acetylcholine (Harmony 2013)
    /// - beta_gamma_ratio → noradrenaline (Aston-Jones & Cohen 2005)
    /// - delta_power → adenosine (Borbely 1982, Porkka-Heiskanen 1997)
    /// - alpha_coherence → oxytocin (Kosfeld 2005)
    /// - theta_beta_ratio → dopamine (attention regulation)
    pub fn decode_to_neuromod(features: &EegNeuromodFeatures) -> NeuromodDelta {
        // Scaling constants: map EEG feature ranges to ±0.1 neuromod delta range.
        const SEROTONIN_SCALE: f32 = 0.5;
        const ACH_SCALE: f32 = 0.3;
        const NE_SCALE: f32 = 0.2;
        const ADENOSINE_SCALE: f32 = 0.4;
        const OXYTOCIN_SCALE: f32 = 0.3;
        const DA_SCALE: f32 = -0.3; // Inverted: low theta/beta = high DA

        NeuromodDelta {
            delta_serotonin: (features.frontal_alpha_asymmetry * SEROTONIN_SCALE).clamp(-0.1, 0.1),
            delta_acetylcholine: (features.theta_frontal_midline * ACH_SCALE).clamp(-0.1, 0.1),
            delta_noradrenaline: (features.beta_gamma_ratio * NE_SCALE).clamp(-0.1, 0.1),
            delta_adenosine: (features.delta_power * ADENOSINE_SCALE).clamp(-0.1, 0.1),
            delta_oxytocin: (features.alpha_coherence * OXYTOCIN_SCALE).clamp(-0.1, 0.1),
            delta_dopamine: (features.theta_beta_ratio * DA_SCALE).clamp(-0.1, 0.1),
        }
    }

    /// Update twin state with the current digital neuromodulator snapshot.
    ///
    /// Computes Euclidean divergence between biological EEG-derived features
    /// (projected onto the neuromod axes) and the digital bath state.
    pub fn update_twin_state(&mut self, digital_snapshot: &DigitalNeuromodSnapshot, cycle: u64) {
        self.state.digital = digital_snapshot.clone();
        self.state.last_update_cycle = cycle;

        // Project biological features onto the same 7D neuromod space.
        let bio_delta = Self::decode_to_neuromod(&self.state.biological);
        let bio_vec = [
            bio_delta.delta_serotonin,
            bio_delta.delta_dopamine,
            bio_delta.delta_noradrenaline,
            bio_delta.delta_acetylcholine,
            bio_delta.delta_adenosine,
            bio_delta.delta_oxytocin,
            0.0, // GABA has no direct EEG decode
        ];
        let dig_vec = digital_snapshot.as_array();

        // Euclidean distance.
        let divergence: f64 = bio_vec
            .iter()
            .zip(dig_vec.iter())
            .map(|(&b, &d)| ((b - d) as f64).powi(2))
            .sum::<f64>()
            .sqrt();

        self.state.divergence = divergence;
        if self.state.divergence_history.len() >= DIVERGENCE_HISTORY_CAP {
            self.state.divergence_history.pop_front();
        }
        self.state.divergence_history.push_back(divergence);

        // Sync quality: 1.0 when divergence=0, 0.0 when divergence >= MAX.
        self.state.sync_quality = (1.0 - divergence / MAX_DIVERGENCE_FOR_QUALITY).clamp(0.0, 1.0);
    }

    /// Produce blended neuromod deltas weighted by blend_ratio.
    ///
    /// At blend_ratio=0.0, returns zero deltas (fully digital, no bio influence).
    /// At blend_ratio=1.0, returns full biological deltas.
    pub fn blended_neuromod_delta(&self, features: &EegNeuromodFeatures) -> NeuromodDelta {
        let bio = Self::decode_to_neuromod(features);
        let r = self.blend_ratio;
        NeuromodDelta {
            delta_serotonin: bio.delta_serotonin * r,
            delta_dopamine: bio.delta_dopamine * r,
            delta_noradrenaline: bio.delta_noradrenaline * r,
            delta_acetylcholine: bio.delta_acetylcholine * r,
            delta_adenosine: bio.delta_adenosine * r,
            delta_oxytocin: bio.delta_oxytocin * r,
        }
    }

    /// Run calibration from resting-state EEG data.
    ///
    /// Computes baseline alpha power and frontal asymmetry offset,
    /// then normalizes scaling factors.
    ///
    /// # Arguments
    /// * `resting_eeg` - Vec of samples, each sample is channel-value pairs.
    /// * `_duration_secs` - Duration of recording (for metadata; not used in computation).
    pub fn calibrate(&mut self, resting_eeg: &[Vec<(EegChannel, f32)>], _duration_secs: f32) {
        if resting_eeg.len() < SubjectCalibration::MIN_CALIBRATION_SAMPLES {
            return;
        }

        // Feed all resting data through the buffer.
        let mut temp_twin = NeuroDigitalTwin::new(self.sample_rate, 0.0);
        temp_twin.calibration.is_calibrated = true; // Temporarily enable extraction.
        for sample in resting_eeg {
            temp_twin.feed_eeg_sample(sample);
        }

        let band_powers = temp_twin.compute_band_powers();
        let features = temp_twin.features_from_band_powers(&band_powers);

        // Set baselines.
        self.calibration.alpha_baseline = band_powers.iter().map(|(_, bp)| bp.alpha).sum::<f32>()
            / (band_powers.len() as f32).max(1.0);

        self.calibration.asymmetry_offset = features.frontal_alpha_asymmetry;

        // Compute scaling factors to normalize each feature to unit variance.
        // For resting state, we just store the raw magnitudes as denominators.
        let magnitudes = [
            features.frontal_alpha_asymmetry.abs().max(0.01),
            features.theta_frontal_midline.abs().max(0.01),
            features.beta_gamma_ratio.abs().max(0.01),
            features.delta_power.abs().max(0.01),
            features.alpha_coherence.abs().max(0.01),
            features.gamma_synchrony.abs().max(0.01),
            features.theta_beta_ratio.abs().max(0.01),
        ];
        for (i, &mag) in magnitudes.iter().enumerate() {
            self.calibration.scaling_factors[i] = 1.0 / mag;
        }

        self.calibration.samples_collected = resting_eeg.len();
        self.calibration.is_calibrated = true;
    }

    /// Activate the twin for closed-loop processing.
    pub fn activate(&mut self) {
        self.is_active = true;
    }

    /// Deactivate the twin.
    pub fn deactivate(&mut self) {
        self.is_active = false;
    }

    /// Current buffer fill level (0.0 to 1.0).
    pub fn buffer_fill(&self) -> f32 {
        self.eeg_buffer.len() as f32 / self.buffer_capacity as f32
    }

    /// Mean divergence over the history window.
    pub fn mean_divergence(&self) -> f64 {
        if self.state.divergence_history.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.state.divergence_history.iter().sum();
        sum / self.state.divergence_history.len() as f64
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sample(channels: &[(EegChannel, f32)]) -> Vec<(EegChannel, f32)> {
        channels.to_vec()
    }

    fn basic_channels(value: f32) -> Vec<(EegChannel, f32)> {
        vec![
            (EegChannel::F3, value),
            (EegChannel::F4, value * 1.1),
            (EegChannel::Fz, value * 0.9),
            (EegChannel::C3, value),
            (EegChannel::C4, value),
            (EegChannel::O1, value),
            (EegChannel::O2, value),
        ]
    }

    fn fill_buffer(twin: &mut NeuroDigitalTwin, n: usize) {
        for i in 0..n {
            let t = i as f32 / 256.0;
            // Simulate 10 Hz alpha oscillation with some noise.
            let val = (t * 10.0 * std::f32::consts::TAU).sin() * 50.0
                + (t * 6.0 * std::f32::consts::TAU).sin() * 20.0;
            twin.feed_eeg_sample(&basic_channels(val));
        }
    }

    fn make_resting_eeg(n: usize) -> Vec<Vec<(EegChannel, f32)>> {
        (0..n)
            .map(|i| {
                let t = i as f32 / 256.0;
                let val = (t * 10.0 * std::f32::consts::TAU).sin() * 50.0;
                basic_channels(val)
            })
            .collect()
    }

    #[test]
    fn test_new_twin_starts_inactive_and_uncalibrated() {
        let twin = NeuroDigitalTwin::new(256.0, 0.3);
        assert!(!twin.is_active);
        assert!(!twin.calibration.is_calibrated);
        assert_eq!(twin.state.divergence, 0.0);
        assert_eq!(twin.state.sync_quality, 1.0);
    }

    #[test]
    fn test_feed_eeg_fills_buffer() {
        let mut twin = NeuroDigitalTwin::new(256.0, 0.3);
        assert_eq!(twin.buffer_fill(), 0.0);
        fill_buffer(&mut twin, 128);
        assert!((twin.buffer_fill() - 0.5).abs() < 0.01);
        fill_buffer(&mut twin, 128);
        assert!((twin.buffer_fill() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_buffer_evicts_oldest_when_full() {
        let mut twin = NeuroDigitalTwin::new(256.0, 0.3);
        // Fill beyond capacity.
        fill_buffer(&mut twin, 300);
        assert_eq!(twin.eeg_buffer.len(), twin.buffer_capacity);
    }

    #[test]
    fn test_extract_features_returns_none_before_calibration() {
        let mut twin = NeuroDigitalTwin::new(256.0, 0.3);
        fill_buffer(&mut twin, 256);
        assert!(twin.extract_features().is_none());
    }

    #[test]
    fn test_calibration_sets_baseline() {
        let mut twin = NeuroDigitalTwin::new(256.0, 0.3);
        let resting = make_resting_eeg(100);
        twin.calibrate(&resting, 1.0);
        assert!(twin.calibration.is_calibrated);
        assert!(twin.calibration.samples_collected >= 100);
        assert!(twin.calibration.alpha_baseline > 0.0);
    }

    #[test]
    fn test_calibration_rejected_with_few_samples() {
        let mut twin = NeuroDigitalTwin::new(256.0, 0.3);
        let resting = make_resting_eeg(5); // Below MIN_CALIBRATION_SAMPLES.
        twin.calibrate(&resting, 0.1);
        assert!(!twin.calibration.is_calibrated);
    }

    #[test]
    fn test_decode_high_delta_positive_adenosine() {
        let features = EegNeuromodFeatures {
            delta_power: 0.8,
            ..Default::default()
        };
        let delta = NeuroDigitalTwin::decode_to_neuromod(&features);
        assert!(
            delta.delta_adenosine > 0.0,
            "High delta → positive adenosine"
        );
    }

    #[test]
    fn test_decode_positive_asymmetry_positive_serotonin() {
        let features = EegNeuromodFeatures {
            frontal_alpha_asymmetry: 0.5,
            ..Default::default()
        };
        let delta = NeuroDigitalTwin::decode_to_neuromod(&features);
        assert!(
            delta.delta_serotonin > 0.0,
            "Positive frontal asymmetry → positive serotonin (Davidson 2004)"
        );
    }

    #[test]
    fn test_divergence_zero_with_matching_states() {
        let mut twin = NeuroDigitalTwin::new(256.0, 0.3);
        // Both biological and digital at zero → divergence = 0.
        twin.update_twin_state(&DigitalNeuromodSnapshot::default(), 1);
        assert!(
            twin.state.divergence < 1e-9,
            "Matching zero states → zero divergence"
        );
        assert!(
            (twin.state.sync_quality - 1.0).abs() < 1e-9,
            "Zero divergence → sync quality 1.0"
        );
    }

    #[test]
    fn test_divergence_increases_when_states_differ() {
        let mut twin = NeuroDigitalTwin::new(256.0, 0.3);
        let snap = DigitalNeuromodSnapshot {
            serotonin: 0.8,
            dopamine: 0.7,
            noradrenaline: 0.6,
            acetylcholine: 0.5,
            adenosine: 0.4,
            oxytocin: 0.3,
            gaba: 0.2,
        };
        twin.update_twin_state(&snap, 1);
        assert!(
            twin.state.divergence > 0.1,
            "Different states → positive divergence"
        );
        assert!(
            twin.state.sync_quality < 1.0,
            "Positive divergence → sync quality < 1.0"
        );
    }

    #[test]
    fn test_blend_ratio_zero_no_biological_contribution() {
        let twin = NeuroDigitalTwin::new(256.0, 0.0);
        let features = EegNeuromodFeatures {
            frontal_alpha_asymmetry: 1.0,
            theta_frontal_midline: 1.0,
            beta_gamma_ratio: 1.0,
            delta_power: 1.0,
            alpha_coherence: 1.0,
            gamma_synchrony: 1.0,
            theta_beta_ratio: 1.0,
        };
        let delta = twin.blended_neuromod_delta(&features);
        assert_eq!(delta.delta_serotonin, 0.0);
        assert_eq!(delta.delta_dopamine, 0.0);
        assert_eq!(delta.delta_noradrenaline, 0.0);
        assert_eq!(delta.delta_acetylcholine, 0.0);
        assert_eq!(delta.delta_adenosine, 0.0);
        assert_eq!(delta.delta_oxytocin, 0.0);
    }

    #[test]
    fn test_blend_ratio_one_full_biological() {
        let twin = NeuroDigitalTwin::new(256.0, 1.0);
        let features = EegNeuromodFeatures {
            frontal_alpha_asymmetry: 0.5,
            ..Default::default()
        };
        let blended = twin.blended_neuromod_delta(&features);
        let full = NeuroDigitalTwin::decode_to_neuromod(&features);
        assert!(
            (blended.delta_serotonin - full.delta_serotonin).abs() < 1e-9,
            "Blend 1.0 produces full bio contribution"
        );
    }

    #[test]
    fn test_pid_converges_to_target() {
        let mut pid = PidController::new(1.0, 0.1, 0.01);
        let target = 0.5;
        let mut actual = 0.0;
        let dt = 0.01;
        for _ in 0..500 {
            let output = pid.update(target, actual, dt);
            actual += output * dt;
        }
        assert!(
            (actual - target).abs() < 0.05,
            "PID should converge near target: actual={actual}, target={target}"
        );
    }

    #[test]
    fn test_sync_quality_inversely_related_to_divergence() {
        let mut twin = NeuroDigitalTwin::new(256.0, 0.3);

        // Low divergence.
        twin.update_twin_state(&DigitalNeuromodSnapshot::default(), 1);
        let q1 = twin.state.sync_quality;

        // High divergence.
        let snap = DigitalNeuromodSnapshot {
            serotonin: 1.0,
            dopamine: 1.0,
            noradrenaline: 1.0,
            acetylcholine: 1.0,
            adenosine: 1.0,
            oxytocin: 1.0,
            gaba: 1.0,
        };
        twin.update_twin_state(&snap, 2);
        let q2 = twin.state.sync_quality;

        assert!(
            q1 > q2,
            "Higher divergence → lower sync quality: q1={q1}, q2={q2}"
        );
    }

    #[test]
    fn test_extract_features_after_calibration() {
        let mut twin = NeuroDigitalTwin::new(256.0, 0.3);
        let resting = make_resting_eeg(100);
        twin.calibrate(&resting, 1.0);
        fill_buffer(&mut twin, 200);
        let features = twin.extract_features();
        assert!(
            features.is_some(),
            "Should extract features after calibration"
        );
    }

    #[test]
    fn test_eeg_band_power_total() {
        let bp = EegBandPower {
            delta: 1.0,
            theta: 2.0,
            alpha: 3.0,
            beta: 4.0,
            gamma: 5.0,
        };
        assert!((bp.total() - 15.0).abs() < 1e-6);
    }
}
