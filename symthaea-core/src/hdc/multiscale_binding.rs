//! # Multi-Scale Temporal Binding
//!
//! Implements gamma binding at multiple timescales simultaneously for consciousness.
//!
//! ## The Binding Problem
//!
//! Consciousness binds disparate features into unified experience. This module
//! implements temporal binding through neural oscillations at multiple frequencies:
//!
//! - **Gamma (40 Hz)**: Standard binding frequency for local integration
//! - **High Gamma (80 Hz)**: Fine-grained feature binding
//! - **Beta (20 Hz)**: Motor/cognitive maintenance
//! - **Alpha (10 Hz)**: Attention and inhibition
//! - **Theta (4 Hz)**: Working memory and navigation
//!
//! ## Scientific Basis
//!
//! - Singer & Gray (1995): Gamma oscillations and binding
//! - Fries (2015): Communication through coherence
//! - Varela et al. (2001): Neural synchrony and consciousness
//!
//! ## Example Usage
//!
//! ```rust
//! use symthaea::hdc::multiscale_binding::{MultiScaleBinding, BindingConfig};
//! use symthaea::hdc::unified_hv::ContinuousHV;
//!
//! let mut binder = MultiScaleBinding::new(BindingConfig::default());
//!
//! let features: Vec<ContinuousHV> = (0..5)
//!     .map(|i| ContinuousHV::random(1024, i))
//!     .collect();
//!
//! // Step the binding system
//! let result = binder.step(&features, 0.001);  // 1ms time step
//! println!("Global coherence: {:.3}", result.global_coherence);
//! ```

use crate::hdc::unified_hv::ContinuousHV;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Configuration for multi-scale binding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BindingConfig {
    /// Frequencies to track (Hz)
    pub frequencies: Vec<f64>,

    /// Coupling strength between oscillators
    pub coupling_strength: f64,

    /// Phase noise level
    pub noise_level: f64,

    /// Minimum coherence to consider bound
    pub coherence_threshold: f64,

    /// Time constant for binding (seconds)
    pub tau_binding: f64,
}

impl Default for BindingConfig {
    fn default() -> Self {
        Self {
            frequencies: vec![
                40.0,   // Standard gamma
                80.0,   // High gamma
                20.0,   // Beta
                10.0,   // Alpha
                4.0,    // Theta
            ],
            coupling_strength: 0.5,
            noise_level: 0.05,
            coherence_threshold: 0.7,
            tau_binding: 0.025, // 25ms binding window
        }
    }
}

/// Neural oscillator for temporal binding
#[derive(Debug, Clone)]
pub struct GammaOscillator {
    /// Oscillation frequency (Hz)
    pub frequency: f64,

    /// Current phase (radians, 0 to 2π)
    pub phase: f64,

    /// Amplitude (0 to 1)
    pub amplitude: f64,

    /// Phase coupling strength
    pub coupling: f64,
}

impl GammaOscillator {
    /// Create a new oscillator at given frequency
    pub fn new(frequency: f64) -> Self {
        Self {
            frequency,
            phase: 0.0,
            amplitude: 1.0,
            coupling: 0.5,
        }
    }

    /// Create with initial phase
    pub fn with_phase(frequency: f64, phase: f64) -> Self {
        Self {
            frequency,
            phase: phase % (2.0 * PI),
            amplitude: 1.0,
            coupling: 0.5,
        }
    }

    /// Get current output value
    pub fn output(&self) -> f64 {
        self.amplitude * self.phase.sin()
    }

    /// Advance by time step
    pub fn step(&mut self, dt: f64) {
        self.phase += 2.0 * PI * self.frequency * dt;
        self.phase %= 2.0 * PI;
    }

    /// Step with coupling to another oscillator
    pub fn step_coupled(&mut self, dt: f64, other_phase: f64, coupling_strength: f64) {
        // Kuramoto coupling
        let phase_diff = other_phase - self.phase;
        let coupling_term = coupling_strength * phase_diff.sin();

        self.phase += 2.0 * PI * self.frequency * dt + coupling_term * dt;
        self.phase %= 2.0 * PI;
        if self.phase < 0.0 {
            self.phase += 2.0 * PI;
        }
    }

    /// Compute phase locking value with another oscillator
    pub fn plv(&self, other: &GammaOscillator) -> f64 {
        // Phase locking value = |E[exp(i * (phase_a - phase_b))]|
        // For instantaneous, just return coherence of phase difference
        let diff = (self.phase - other.phase).abs();
        let normalized_diff = diff.min(2.0 * PI - diff) / PI;
        1.0 - normalized_diff
    }
}

/// Binding state at a single scale
#[derive(Debug, Clone)]
pub struct ScaleState {
    /// Oscillator for this frequency
    pub oscillator: GammaOscillator,

    /// Currently bound features (indices)
    pub bound_features: Vec<usize>,

    /// Phase coherence at this scale
    pub coherence: f64,

    /// Binding strength at this scale
    pub strength: f64,
}

/// Result of a binding step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BindingResult {
    /// Output for each frequency (sin(phase))
    pub outputs: Vec<f64>,

    /// Coherence at each frequency
    pub scale_coherence: Vec<f64>,

    /// Global coherence across all scales
    pub global_coherence: f64,

    /// Bound representation (if coherent)
    pub bound: Option<ContinuousHV>,

    /// Which features are bound at each scale
    pub bound_features: Vec<Vec<usize>>,

    /// Total binding energy
    pub binding_energy: f64,

    /// Dominant frequency (highest coherence)
    pub dominant_frequency: f64,
}

/// Main multi-scale temporal binding system
#[derive(Debug)]
pub struct MultiScaleBinding {
    config: BindingConfig,

    /// Oscillators at each frequency
    oscillators: Vec<GammaOscillator>,

    /// Current time (seconds)
    time: f64,

    /// Phase coupling matrix between oscillators
    coupling_matrix: Vec<Vec<f64>>,

    /// Feature representations currently being bound
    current_features: Vec<ContinuousHV>,

    /// Phase history for coherence calculation
    phase_history: Vec<Vec<f64>>,

    /// History window size
    history_size: usize,

    /// Random state for noise
    rng_state: u64,
}

impl MultiScaleBinding {
    /// Create a new multi-scale binding system
    pub fn new(config: BindingConfig) -> Self {
        let n = config.frequencies.len();

        let oscillators: Vec<GammaOscillator> = config.frequencies.iter()
            .map(|&f| GammaOscillator::new(f))
            .collect();

        // Initialize coupling matrix (weaker coupling between distant frequencies)
        let mut coupling_matrix = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let freq_ratio = config.frequencies[i].min(config.frequencies[j])
                        / config.frequencies[i].max(config.frequencies[j]);
                    coupling_matrix[i][j] = config.coupling_strength * freq_ratio;
                }
            }
        }

        Self {
            config,
            oscillators,
            time: 0.0,
            coupling_matrix,
            current_features: Vec::new(),
            phase_history: vec![Vec::new(); n],
            history_size: 100,
            rng_state: 12345,
        }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(BindingConfig::default())
    }

    /// Perform one time step of the binding system
    pub fn step(&mut self, features: &[ContinuousHV], dt: f64) -> BindingResult {
        self.time += dt;
        self.current_features = features.to_vec();

        let n = self.oscillators.len();

        // Step 1: Update oscillators with mutual coupling
        let old_phases: Vec<f64> = self.oscillators.iter().map(|o| o.phase).collect();

        for i in 0..n {
            // Compute total coupling from other oscillators
            let mut total_coupling = 0.0;
            for j in 0..n {
                if i != j {
                    let phase_diff = old_phases[j] - old_phases[i];
                    total_coupling += self.coupling_matrix[i][j] * phase_diff.sin();
                }
            }

            // Add noise
            let noise = self.noise() * self.config.noise_level;

            // Update phase
            self.oscillators[i].phase += 2.0 * PI * self.oscillators[i].frequency * dt
                + total_coupling * dt
                + noise * dt.sqrt();
            self.oscillators[i].phase %= 2.0 * PI;
            if self.oscillators[i].phase < 0.0 {
                self.oscillators[i].phase += 2.0 * PI;
            }

            // Record phase history
            self.phase_history[i].push(self.oscillators[i].phase);
            if self.phase_history[i].len() > self.history_size {
                self.phase_history[i].remove(0);
            }
        }

        // Step 2: Compute coherence at each scale
        let scale_coherence: Vec<f64> = (0..n)
            .map(|i| self.compute_scale_coherence(i, features))
            .collect();

        // Step 3: Compute global coherence
        let global_coherence = if scale_coherence.is_empty() {
            0.0
        } else {
            scale_coherence.iter().sum::<f64>() / scale_coherence.len() as f64
        };

        // Step 4: Determine which features are bound at each scale
        let bound_features: Vec<Vec<usize>> = (0..n)
            .map(|i| self.compute_bound_features(i, features))
            .collect();

        // Step 5: Compute bound representation if coherent
        let bound = if global_coherence >= self.config.coherence_threshold && !features.is_empty() {
            Some(self.compute_bound_representation(features, &scale_coherence))
        } else {
            None
        };

        // Step 6: Compute binding energy
        let binding_energy = self.compute_binding_energy(&scale_coherence);

        // Step 7: Find dominant frequency
        let dominant_idx = scale_coherence.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let dominant_frequency = self.config.frequencies[dominant_idx];

        // Outputs
        let outputs: Vec<f64> = self.oscillators.iter()
            .map(|o| o.output())
            .collect();

        BindingResult {
            outputs,
            scale_coherence,
            global_coherence,
            bound,
            bound_features,
            binding_energy,
            dominant_frequency,
        }
    }

    /// Compute coherence at a single scale
    fn compute_scale_coherence(&self, scale_idx: usize, features: &[ContinuousHV]) -> f64 {
        if features.len() < 2 || self.phase_history[scale_idx].len() < 10 {
            return 0.0;
        }

        // Compute phase locking value from history
        let phases = &self.phase_history[scale_idx];
        let n = phases.len();

        // PLV = |mean(exp(i * phase))|
        let sum_cos: f64 = phases.iter().map(|p| p.cos()).sum();
        let sum_sin: f64 = phases.iter().map(|p| p.sin()).sum();

        let plv = ((sum_cos / n as f64).powi(2) + (sum_sin / n as f64).powi(2)).sqrt();

        // Modulate by feature coherence
        let feature_sim = self.compute_feature_coherence(features);

        plv * feature_sim
    }

    /// Compute average pairwise similarity of features
    fn compute_feature_coherence(&self, features: &[ContinuousHV]) -> f64 {
        if features.len() < 2 {
            return 1.0;
        }

        let mut sum: f32 = 0.0;
        let mut count: usize = 0;

        for i in 0..features.len() {
            for j in (i + 1)..features.len() {
                sum += features[i].similarity(&features[j]).abs();
                count += 1;
            }
        }

        if count > 0 {
            (sum / count as f32) as f64
        } else {
            0.0
        }
    }

    /// Determine which features are bound at a scale
    fn compute_bound_features(&self, scale_idx: usize, features: &[ContinuousHV]) -> Vec<usize> {
        if features.is_empty() {
            return Vec::new();
        }

        // Features are bound if oscillator output is above threshold
        let output = self.oscillators[scale_idx].output().abs();
        let threshold = 0.5;

        if output > threshold {
            (0..features.len()).collect()
        } else {
            Vec::new()
        }
    }

    /// Compute the bound representation from features
    fn compute_bound_representation(
        &self,
        features: &[ContinuousHV],
        coherence: &[f64],
    ) -> ContinuousHV {
        if features.is_empty() {
            return ContinuousHV::zero(1);
        }

        // Weight each scale's contribution by coherence
        let total_coherence: f64 = coherence.iter().sum();
        if total_coherence < 1e-10 {
            return ContinuousHV::bundle(&features.iter().collect::<Vec<_>>());
        }

        // Scale weights by oscillator output and coherence
        let weights: Vec<f32> = self.oscillators.iter()
            .zip(coherence.iter())
            .map(|(osc, &c)| (osc.output().abs() * c) as f32)
            .collect();

        let weight_sum: f32 = weights.iter().sum();
        if weight_sum < 1e-10 {
            return ContinuousHV::bundle(&features.iter().collect::<Vec<_>>());
        }

        // Create bound representation by bundling features
        let refs: Vec<&ContinuousHV> = features.iter().collect();
        let bundled = ContinuousHV::bundle(&refs);

        // Modulate by dominant oscillator phase
        let dominant_phase = self.oscillators[0].phase;
        let phase_factor = (dominant_phase.cos().abs() + 0.5).min(1.0);

        bundled.scale(phase_factor as f32)
    }

    /// Compute total binding energy
    fn compute_binding_energy(&self, coherence: &[f64]) -> f64 {
        // Energy = sum of (amplitude * coherence * frequency_weight)
        self.oscillators.iter()
            .zip(coherence.iter())
            .map(|(osc, &c)| {
                let freq_weight = osc.frequency / 40.0; // Normalize to gamma
                osc.amplitude * c * freq_weight
            })
            .sum()
    }

    /// Generate noise value
    fn noise(&mut self) -> f64 {
        // Simple xorshift for reproducibility
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;

        // Map to [-1, 1]
        (self.rng_state as f64 / u64::MAX as f64) * 2.0 - 1.0
    }

    /// Get current oscillator phases
    pub fn phases(&self) -> Vec<f64> {
        self.oscillators.iter().map(|o| o.phase).collect()
    }

    /// Get current oscillator outputs
    pub fn outputs(&self) -> Vec<f64> {
        self.oscillators.iter().map(|o| o.output()).collect()
    }

    /// Get current time
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Reset the binding system
    pub fn reset(&mut self) {
        self.time = 0.0;
        for osc in &mut self.oscillators {
            osc.phase = 0.0;
        }
        for history in &mut self.phase_history {
            history.clear();
        }
        self.current_features.clear();
    }

    /// Get statistics about current binding state
    pub fn stats(&self) -> BindingStats {
        let phases: Vec<f64> = self.oscillators.iter().map(|o| o.phase).collect();
        let amplitudes: Vec<f64> = self.oscillators.iter().map(|o| o.amplitude).collect();

        // Compute phase coherence between adjacent oscillators
        let mut phase_coherence = Vec::new();
        for i in 0..(self.oscillators.len() - 1) {
            let plv = self.oscillators[i].plv(&self.oscillators[i + 1]);
            phase_coherence.push(plv);
        }

        BindingStats {
            time: self.time,
            phases,
            amplitudes,
            phase_coherence,
            frequencies: self.config.frequencies.clone(),
            num_features: self.current_features.len(),
        }
    }
}

/// Statistics about the binding system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BindingStats {
    pub time: f64,
    pub phases: Vec<f64>,
    pub amplitudes: Vec<f64>,
    pub phase_coherence: Vec<f64>,
    pub frequencies: Vec<f64>,
    pub num_features: usize,
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_oscillator_basic() {
        let mut osc = GammaOscillator::new(40.0);

        // Should start at phase 0
        assert!((osc.phase - 0.0).abs() < 1e-10);

        // Step forward
        osc.step(0.001); // 1ms

        // Phase should have advanced
        let expected = 2.0 * PI * 40.0 * 0.001;
        assert!((osc.phase - expected).abs() < 1e-6);
    }

    #[test]
    fn test_oscillator_period() {
        let mut osc = GammaOscillator::new(40.0);

        // One full period at 40 Hz = 25ms
        let dt = 0.001; // 1ms steps
        for _ in 0..25 {
            osc.step(dt);
        }

        // Should be back near phase 0 (with floating point tolerance)
        assert!(osc.phase < 0.5 || osc.phase > 2.0 * PI - 0.5);
    }

    #[test]
    fn test_multiscale_binding() {
        let mut binder = MultiScaleBinding::default_config();
        let dim = 1024;

        let features: Vec<ContinuousHV> = (0..3)
            .map(|i| ContinuousHV::random(dim, i))
            .collect();

        // Run for 100ms
        let dt = 0.001;
        for _ in 0..100 {
            let result = binder.step(&features, dt);

            // Should have outputs for each frequency
            assert_eq!(result.outputs.len(), 5);

            // Coherence should be between 0 and 1
            for c in &result.scale_coherence {
                assert!(*c >= 0.0 && *c <= 1.0);
            }
        }
    }

    #[test]
    fn test_binding_with_similar_features() {
        let mut binder = MultiScaleBinding::default_config();
        let dim = 1024;

        // Create very similar features (should bind well)
        let base = ContinuousHV::random(dim, 42);
        let features: Vec<ContinuousHV> = (0..3)
            .map(|i| {
                let noise = ContinuousHV::random(dim, 100 + i).scale(0.1);
                base.add(&noise)
            })
            .collect();

        // Run until coherence builds
        let dt = 0.001;
        let mut max_coherence: f64 = 0.0;

        for _ in 0..200 {
            let result = binder.step(&features, dt);
            max_coherence = max_coherence.max(result.global_coherence);
        }

        println!("Max coherence achieved: {:.3}", max_coherence);
        // Similar features should achieve some coherence
        assert!(max_coherence > 0.1);
    }

    #[test]
    fn test_dominant_frequency() {
        let binder = MultiScaleBinding::default_config();

        let features: Vec<ContinuousHV> = (0..3)
            .map(|i| ContinuousHV::random(1024, i))
            .collect();

        let mut b = binder;
        let result = b.step(&features, 0.001);

        // Dominant frequency should be one of configured frequencies
        assert!(b.config.frequencies.contains(&result.dominant_frequency));
    }

    #[test]
    fn test_stats() {
        let mut binder = MultiScaleBinding::default_config();

        let features: Vec<ContinuousHV> = (0..2)
            .map(|i| ContinuousHV::random(1024, i))
            .collect();

        let _ = binder.step(&features, 0.001);
        let stats = binder.stats();

        assert_eq!(stats.phases.len(), 5);
        assert_eq!(stats.frequencies.len(), 5);
        assert_eq!(stats.num_features, 2);
    }

    #[test]
    fn test_reset() {
        let mut binder = MultiScaleBinding::default_config();

        let features: Vec<ContinuousHV> = (0..2)
            .map(|i| ContinuousHV::random(1024, i))
            .collect();

        // Run for a bit
        for _ in 0..50 {
            let _ = binder.step(&features, 0.001);
        }

        assert!(binder.time() > 0.0);

        // Reset
        binder.reset();

        assert!((binder.time() - 0.0).abs() < 1e-10);
        for phase in binder.phases() {
            assert!((phase - 0.0).abs() < 1e-10);
        }
    }
}
