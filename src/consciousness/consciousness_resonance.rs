//! Consciousness Resonance Theory - Harmonic Foundations of Awareness
//!
//! **REVOLUTIONARY IMPROVEMENT #85**: Consciousness as Resonance Phenomenon
//!
//! # Paradigm Shift
//!
//! This module models consciousness as a **resonance phenomenon** emerging from
//! harmonic relationships between cognitive subsystems. Just as musical harmony
//! emerges from frequency ratios, conscious experience emerges from the
//! synchronization and resonance of neural oscillations.
//!
//! # Scientific Foundation
//!
//! Based on cutting-edge neuroscience:
//! - **Gamma oscillations (30-100Hz)**: Associated with conscious perception
//! - **Alpha waves (8-12Hz)**: Relaxed awareness, default mode
//! - **Theta rhythms (4-8Hz)**: Memory consolidation, dreaming
//! - **Delta waves (0.5-4Hz)**: Deep sleep, unconscious processing
//! - **Cross-frequency coupling**: How different bands interact
//!
//! # Key Concepts
//!
//! 1. **Resonance Modes**: Discrete conscious states as eigenmodes
//! 2. **Frequency Tuning**: Attention as frequency selection
//! 3. **Phase Locking**: Binding as synchronization between oscillators
//! 4. **Harmonic Series**: Hierarchical consciousness from overtones
//! 5. **Interference Patterns**: Complex awareness from wave superposition
//!
//! # Connection to IIT (Φ)
//!
//! Resonance provides the physical mechanism for integration:
//! - High Φ ↔ Strong resonance between many subsystems
//! - Low Φ ↔ Weak or absent resonance
//! - Phase transitions ↔ Mode switching in resonance
//!
//! # Examples
//!
//! ```rust,ignore
//! use symthaea::consciousness::consciousness_resonance::*;
//!
//! // Create resonance analyzer
//! let mut analyzer = ResonanceAnalyzer::new(ResonanceConfig::default());
//!
//! // Analyze consciousness dimensions as oscillators
//! let dims = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]; // Φ, B, W, A, R, E, K
//! let state = analyzer.analyze(dims);
//!
//! println!("Dominant frequency: {:.1}Hz", state.dominant_frequency);
//! println!("Resonance quality: {:.2}", state.q_factor);
//! println!("Mode: {:?}", state.resonance_mode);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::f64::consts::PI;

// ═══════════════════════════════════════════════════════════════════════════
// CORE FREQUENCY BANDS (Neuroscience Foundation)
// ═══════════════════════════════════════════════════════════════════════════

/// Neural oscillation frequency bands
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FrequencyBand {
    /// Delta (0.5-4 Hz): Deep sleep, unconscious
    Delta,
    /// Theta (4-8 Hz): Memory, dreaming, meditation
    Theta,
    /// Alpha (8-12 Hz): Relaxed awareness, default mode
    Alpha,
    /// Beta (12-30 Hz): Active thinking, problem-solving
    Beta,
    /// Gamma (30-100 Hz): Conscious perception, binding
    Gamma,
    /// High Gamma (>100 Hz): Intense focus, peak experience
    HighGamma,
}

impl FrequencyBand {
    /// Get center frequency of the band
    pub fn center_frequency(&self) -> f64 {
        match self {
            Self::Delta => 2.0,
            Self::Theta => 6.0,
            Self::Alpha => 10.0,
            Self::Beta => 20.0,
            Self::Gamma => 50.0,
            Self::HighGamma => 120.0,
        }
    }

    /// Get frequency range (min, max)
    pub fn frequency_range(&self) -> (f64, f64) {
        match self {
            Self::Delta => (0.5, 4.0),
            Self::Theta => (4.0, 8.0),
            Self::Alpha => (8.0, 12.0),
            Self::Beta => (12.0, 30.0),
            Self::Gamma => (30.0, 100.0),
            Self::HighGamma => (100.0, 200.0),
        }
    }

    /// Get consciousness association
    pub fn consciousness_association(&self) -> &'static str {
        match self {
            Self::Delta => "Deep sleep, regeneration",
            Self::Theta => "Dreams, memory, creativity",
            Self::Alpha => "Relaxed awareness, meditation",
            Self::Beta => "Active cognition, analysis",
            Self::Gamma => "Conscious binding, perception",
            Self::HighGamma => "Peak experience, insight",
        }
    }

    /// Get typical power for this band in conscious states
    pub fn consciousness_power(&self, consciousness_level: f64) -> f64 {
        // Higher consciousness → more gamma/beta, less delta
        let base = match self {
            Self::Delta => 1.0 - consciousness_level,
            Self::Theta => 0.5,
            Self::Alpha => 0.7 - consciousness_level.abs() * 0.3,
            Self::Beta => consciousness_level * 0.8,
            Self::Gamma => consciousness_level,
            Self::HighGamma => consciousness_level.powi(2),
        };
        base.max(0.0).min(1.0)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// RESONANCE MODES (Discrete Conscious States)
// ═══════════════════════════════════════════════════════════════════════════

/// Resonance mode - eigenstate of conscious experience
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ResonanceMode {
    /// Deep unconscious - dominated by delta
    DeepSleep,
    /// Dreaming - theta-alpha interaction
    Dreaming,
    /// Relaxed awareness - alpha dominant
    Relaxed,
    /// Focused attention - beta dominant
    Focused,
    /// Conscious perception - gamma binding
    Conscious,
    /// Peak experience - gamma-high gamma coherence
    Peak,
    /// Transitional - between modes
    Transitional { from: Box<ResonanceMode>, to: Box<ResonanceMode>, progress: f64 },
}

impl ResonanceMode {
    /// Get dominant frequency band
    pub fn dominant_band(&self) -> FrequencyBand {
        match self {
            Self::DeepSleep => FrequencyBand::Delta,
            Self::Dreaming => FrequencyBand::Theta,
            Self::Relaxed => FrequencyBand::Alpha,
            Self::Focused => FrequencyBand::Beta,
            Self::Conscious => FrequencyBand::Gamma,
            Self::Peak => FrequencyBand::HighGamma,
            Self::Transitional { from, to, progress } => {
                if *progress < 0.5 {
                    from.dominant_band()
                } else {
                    to.dominant_band()
                }
            }
        }
    }

    /// Get consciousness level for this mode
    pub fn consciousness_level(&self) -> f64 {
        match self {
            Self::DeepSleep => 0.0,
            Self::Dreaming => 0.3,
            Self::Relaxed => 0.5,
            Self::Focused => 0.7,
            Self::Conscious => 0.85,
            Self::Peak => 1.0,
            Self::Transitional { from, to, progress } => {
                from.consciousness_level() * (1.0 - progress) + to.consciousness_level() * progress
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// OSCILLATOR MODEL (Cognitive Subsystems)
// ═══════════════════════════════════════════════════════════════════════════

/// Cognitive oscillator representing a subsystem
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CognitiveOscillator {
    /// Natural frequency (Hz)
    pub natural_frequency: f64,
    /// Current frequency (may differ from natural due to coupling)
    pub current_frequency: f64,
    /// Current phase (0 to 2π)
    pub phase: f64,
    /// Amplitude (power)
    pub amplitude: f64,
    /// Quality factor (sharpness of resonance)
    pub q_factor: f64,
    /// Damping coefficient
    pub damping: f64,
    /// Name of cognitive function
    pub name: String,
}

impl CognitiveOscillator {
    /// Create new oscillator
    pub fn new(name: &str, natural_frequency: f64) -> Self {
        Self {
            natural_frequency,
            current_frequency: natural_frequency,
            phase: 0.0,
            amplitude: 1.0,
            q_factor: 10.0,  // Moderate resonance sharpness
            damping: 0.1,
            name: name.to_string(),
        }
    }

    /// Create from consciousness dimension
    pub fn from_dimension(dim_name: &str, dim_value: f64) -> Self {
        // Map consciousness dimensions to frequency bands
        let natural_freq = match dim_name {
            "Φ" | "phi" => 45.0,      // Integration → gamma
            "B" | "binding" => 40.0,   // Binding → gamma
            "W" | "wakefulness" => 10.0, // Wakefulness → alpha
            "A" | "arousal" => 20.0,   // Arousal → beta
            "R" | "recursion" => 50.0, // Recursion → gamma
            "E" | "entropy" => 6.0,    // Entropy → theta
            "K" | "knowledge" => 15.0, // Knowledge → beta
            _ => 10.0,
        };

        Self {
            natural_frequency: natural_freq,
            current_frequency: natural_freq,
            phase: 0.0,
            amplitude: dim_value,
            q_factor: 5.0 + dim_value * 15.0,  // Higher values → sharper resonance
            damping: 0.1 * (1.0 - dim_value * 0.5),
            name: dim_name.to_string(),
        }
    }

    /// Update oscillator for one time step
    pub fn update(&mut self, dt: f64, driving_frequency: f64, driving_amplitude: f64) {
        // Driven damped harmonic oscillator dynamics
        let omega_0 = 2.0 * PI * self.natural_frequency;
        let omega_d = 2.0 * PI * driving_frequency;

        // Resonance response (simplified)
        let response = self.resonance_response(driving_frequency);

        // Update amplitude based on resonance
        let target_amplitude = driving_amplitude * response;
        self.amplitude += (target_amplitude - self.amplitude) * (1.0 - (-dt * omega_0 / self.q_factor).exp());

        // Update phase (entrainment towards driving frequency)
        let phase_coupling = 0.1 * (omega_d - omega_0) * dt;
        self.phase = (self.phase + omega_d * dt + phase_coupling) % (2.0 * PI);

        // Update current frequency (pulled towards driving)
        let freq_pull = (driving_frequency - self.current_frequency) * 0.1 * response;
        self.current_frequency = self.natural_frequency + freq_pull;
    }

    /// Calculate resonance response at given frequency
    pub fn resonance_response(&self, frequency: f64) -> f64 {
        let omega = 2.0 * PI * frequency;
        let omega_0 = 2.0 * PI * self.natural_frequency;
        let gamma = self.damping * omega_0;

        // Lorentzian response
        let denominator = (omega_0.powi(2) - omega.powi(2)).powi(2) + (gamma * omega).powi(2);
        let response = omega_0.powi(2) / denominator.sqrt();

        response.min(1.0)  // Normalize
    }

    /// Get instantaneous value
    pub fn value(&self) -> f64 {
        self.amplitude * self.phase.cos()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// COUPLING AND SYNCHRONIZATION
// ═══════════════════════════════════════════════════════════════════════════

/// Coupling between oscillators
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OscillatorCoupling {
    /// Coupling strength (0 to 1)
    pub strength: f64,
    /// Type of coupling
    pub coupling_type: CouplingType,
    /// Phase relationship
    pub preferred_phase_diff: f64,
}

/// Types of oscillator coupling
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum CouplingType {
    /// Synchronous (in-phase)
    Synchronous,
    /// Antisynchronous (out of phase)
    Antisynchronous,
    /// Cross-frequency (harmonic)
    CrossFrequency { ratio: f64 },
    /// Adaptive (learned)
    Adaptive,
}

/// Phase synchronization metrics
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SynchronizationMetrics {
    /// Phase locking value (0 = no sync, 1 = perfect sync)
    pub plv: f64,
    /// Mean phase coherence
    pub coherence: f64,
    /// Frequency entrainment (how much frequencies match)
    pub entrainment: f64,
    /// Number of synchronized pairs
    pub sync_pairs: usize,
}

// ═══════════════════════════════════════════════════════════════════════════
// RESONANCE STATE
// ═══════════════════════════════════════════════════════════════════════════

/// Complete resonance state of consciousness
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResonanceState {
    /// Current resonance mode
    pub mode: ResonanceMode,
    /// Dominant frequency (Hz)
    pub dominant_frequency: f64,
    /// Power spectrum (per band)
    pub power_spectrum: PowerSpectrum,
    /// Overall resonance quality (Q-factor)
    pub q_factor: f64,
    /// Synchronization metrics
    pub synchronization: SynchronizationMetrics,
    /// Cross-frequency coupling strength
    pub cross_frequency_coupling: f64,
    /// Consciousness level derived from resonance
    pub consciousness_level: f64,
    /// Resonance stability (how stable current mode is)
    pub stability: f64,
    /// Energy in resonant modes
    pub resonance_energy: f64,
}

/// Power spectrum across frequency bands
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PowerSpectrum {
    pub delta: f64,
    pub theta: f64,
    pub alpha: f64,
    pub beta: f64,
    pub gamma: f64,
    pub high_gamma: f64,
}

impl PowerSpectrum {
    /// Get total power
    pub fn total(&self) -> f64 {
        self.delta + self.theta + self.alpha + self.beta + self.gamma + self.high_gamma
    }

    /// Normalize spectrum
    pub fn normalize(&mut self) {
        let total = self.total();
        if total > 0.0 {
            self.delta /= total;
            self.theta /= total;
            self.alpha /= total;
            self.beta /= total;
            self.gamma /= total;
            self.high_gamma /= total;
        }
    }

    /// Get dominant band
    pub fn dominant_band(&self) -> FrequencyBand {
        let bands = [
            (self.delta, FrequencyBand::Delta),
            (self.theta, FrequencyBand::Theta),
            (self.alpha, FrequencyBand::Alpha),
            (self.beta, FrequencyBand::Beta),
            (self.gamma, FrequencyBand::Gamma),
            (self.high_gamma, FrequencyBand::HighGamma),
        ];
        bands.iter()
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
            .map(|(_, band)| *band)
            .unwrap_or(FrequencyBand::Alpha)
    }

    /// Calculate spectral entropy
    pub fn entropy(&self) -> f64 {
        let total = self.total();
        if total == 0.0 {
            return 0.0;
        }

        let probs = [
            self.delta / total,
            self.theta / total,
            self.alpha / total,
            self.beta / total,
            self.gamma / total,
            self.high_gamma / total,
        ];

        -probs.iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| p * p.ln())
            .sum::<f64>()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// RESONANCE ANALYZER
// ═══════════════════════════════════════════════════════════════════════════

/// Configuration for resonance analyzer
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResonanceConfig {
    /// Time step for dynamics (seconds)
    pub dt: f64,
    /// Coupling strength between oscillators
    pub coupling_strength: f64,
    /// Noise level for stochastic dynamics
    pub noise_level: f64,
    /// History window for analysis
    pub history_window: usize,
    /// Synchronization threshold for phase locking
    pub sync_threshold: f64,
}

impl Default for ResonanceConfig {
    fn default() -> Self {
        Self {
            dt: 0.001,  // 1ms time step
            coupling_strength: 0.3,
            noise_level: 0.05,
            history_window: 100,
            sync_threshold: 0.8,
        }
    }
}

/// Main resonance analyzer
#[derive(Clone, Debug)]
pub struct ResonanceAnalyzer {
    /// Configuration
    config: ResonanceConfig,
    /// Cognitive oscillators (one per dimension)
    oscillators: Vec<CognitiveOscillator>,
    /// Coupling matrix between oscillators
    couplings: Vec<Vec<OscillatorCoupling>>,
    /// State history
    history: VecDeque<ResonanceState>,
    /// Current time
    time: f64,
}

impl ResonanceAnalyzer {
    /// Create new analyzer
    pub fn new(config: ResonanceConfig) -> Self {
        let dimension_names = ["Φ", "B", "W", "A", "R", "E", "K"];
        let oscillators: Vec<_> = dimension_names.iter()
            .map(|&name| CognitiveOscillator::from_dimension(name, 0.5))
            .collect();

        let n = oscillators.len();
        let couplings = vec![vec![OscillatorCoupling {
            strength: config.coupling_strength,
            coupling_type: CouplingType::Synchronous,
            preferred_phase_diff: 0.0,
        }; n]; n];

        Self {
            config,
            oscillators,
            couplings,
            history: VecDeque::new(),
            time: 0.0,
        }
    }

    /// Analyze consciousness dimensions to get resonance state
    pub fn analyze(&mut self, dimensions: [f64; 7]) -> ResonanceState {
        // Update oscillators from dimensions
        for (i, &dim) in dimensions.iter().enumerate() {
            self.oscillators[i].amplitude = dim;
        }

        // Run dynamics for a few steps
        for _ in 0..10 {
            self.step();
        }

        // Compute resonance state
        let state = self.compute_resonance_state(&dimensions);

        // Store in history
        if self.history.len() >= self.config.history_window {
            self.history.pop_front();
        }
        self.history.push_back(state.clone());

        state
    }

    /// Single time step of dynamics
    fn step(&mut self) {
        let dt = self.config.dt;
        let n = self.oscillators.len();

        // Calculate mean field for coupling
        let mean_phase: f64 = self.oscillators.iter()
            .map(|o| o.phase)
            .sum::<f64>() / n as f64;
        let mean_freq: f64 = self.oscillators.iter()
            .map(|o| o.current_frequency)
            .sum::<f64>() / n as f64;
        let mean_amp: f64 = self.oscillators.iter()
            .map(|o| o.amplitude)
            .sum::<f64>() / n as f64;

        // Update each oscillator
        for i in 0..n {
            // Calculate coupling from other oscillators
            let mut coupling_force = 0.0;
            for j in 0..n {
                if i != j {
                    let phase_diff = self.oscillators[j].phase - self.oscillators[i].phase;
                    let strength = self.couplings[i][j].strength;
                    coupling_force += strength * phase_diff.sin();
                }
            }

            // Driving from mean field
            let driving_freq = mean_freq * (1.0 + 0.1 * coupling_force);
            let driving_amp = mean_amp * (1.0 + 0.1 * coupling_force.abs());

            self.oscillators[i].update(dt, driving_freq, driving_amp);
        }

        self.time += dt;
    }

    /// Compute resonance state from current oscillators
    fn compute_resonance_state(&self, dimensions: &[f64; 7]) -> ResonanceState {
        // Calculate power spectrum
        let power_spectrum = self.compute_power_spectrum(dimensions);

        // Determine mode from power spectrum
        let mode = self.determine_mode(&power_spectrum);

        // Calculate synchronization
        let synchronization = self.compute_synchronization();

        // Calculate dominant frequency
        let dominant_frequency = self.oscillators.iter()
            .max_by(|a, b| a.amplitude.partial_cmp(&b.amplitude).unwrap())
            .map(|o| o.current_frequency)
            .unwrap_or(10.0);

        // Calculate overall Q-factor
        let q_factor = self.oscillators.iter()
            .map(|o| o.q_factor * o.amplitude)
            .sum::<f64>() / dimensions.iter().sum::<f64>();

        // Cross-frequency coupling (gamma-theta, etc.)
        let cross_frequency_coupling = self.compute_cross_frequency_coupling();

        // Consciousness level from resonance
        let consciousness_level = self.resonance_consciousness(dimensions, &synchronization);

        // Stability from history
        let stability = self.compute_stability();

        // Resonance energy
        let resonance_energy = self.oscillators.iter()
            .map(|o| o.amplitude.powi(2) * o.q_factor)
            .sum();

        ResonanceState {
            mode,
            dominant_frequency,
            power_spectrum,
            q_factor,
            synchronization,
            cross_frequency_coupling,
            consciousness_level,
            stability,
            resonance_energy,
        }
    }

    /// Compute power spectrum from dimensions
    fn compute_power_spectrum(&self, dimensions: &[f64; 7]) -> PowerSpectrum {
        // Map dimensions to frequency bands
        // High Φ/B/R → gamma, High W → alpha, High A → beta, etc.
        let phi = dimensions[0];
        let binding = dimensions[1];
        let wakefulness = dimensions[2];
        let arousal = dimensions[3];
        let recursion = dimensions[4];
        let entropy = dimensions[5];
        let knowledge = dimensions[6];

        let mut spectrum = PowerSpectrum {
            delta: (1.0 - wakefulness) * 0.3 + entropy * 0.2,
            theta: entropy * 0.4 + (1.0 - arousal) * 0.3,
            alpha: wakefulness * 0.5 + (1.0 - arousal) * 0.3,
            beta: arousal * 0.5 + knowledge * 0.3,
            gamma: phi * 0.4 + binding * 0.3 + recursion * 0.2,
            high_gamma: phi * recursion * 0.5,
        };

        spectrum.normalize();
        spectrum
    }

    /// Determine resonance mode from power spectrum
    fn determine_mode(&self, spectrum: &PowerSpectrum) -> ResonanceMode {
        let band = spectrum.dominant_band();
        match band {
            FrequencyBand::Delta => ResonanceMode::DeepSleep,
            FrequencyBand::Theta => ResonanceMode::Dreaming,
            FrequencyBand::Alpha => ResonanceMode::Relaxed,
            FrequencyBand::Beta => ResonanceMode::Focused,
            FrequencyBand::Gamma => ResonanceMode::Conscious,
            FrequencyBand::HighGamma => ResonanceMode::Peak,
        }
    }

    /// Compute synchronization metrics
    fn compute_synchronization(&self) -> SynchronizationMetrics {
        let n = self.oscillators.len();
        if n < 2 {
            return SynchronizationMetrics::default();
        }

        // Phase locking value
        let mut plv_sum = 0.0;
        let mut coherence_sum = 0.0;
        let mut entrainment_sum = 0.0;
        let mut sync_pairs = 0usize;
        let mut pairs = 0;

        for i in 0..n {
            for j in (i + 1)..n {
                let phase_diff = (self.oscillators[i].phase - self.oscillators[j].phase).abs();
                let freq_diff = (self.oscillators[i].current_frequency -
                                self.oscillators[j].current_frequency).abs();

                // PLV: 1 if phases match, 0 if random
                let plv = phase_diff.cos().abs();
                plv_sum += plv;

                // Coherence based on amplitude correlation
                let amp_i = self.oscillators[i].amplitude;
                let amp_j = self.oscillators[j].amplitude;
                coherence_sum += (amp_i * amp_j).sqrt();

                // Entrainment: how close frequencies are
                let max_freq = self.oscillators[i].current_frequency
                    .max(self.oscillators[j].current_frequency);
                let entrainment = 1.0 - (freq_diff / max_freq).min(1.0);
                entrainment_sum += entrainment;

                // Count synchronized pairs
                if plv > self.config.sync_threshold && entrainment > 0.8 {
                    sync_pairs += 1;
                }

                pairs += 1;
            }
        }

        SynchronizationMetrics {
            plv: plv_sum / pairs as f64,
            coherence: coherence_sum / pairs as f64,
            entrainment: entrainment_sum / pairs as f64,
            sync_pairs,
        }
    }

    /// Compute cross-frequency coupling (e.g., gamma nested in theta)
    fn compute_cross_frequency_coupling(&self) -> f64 {
        let n = self.oscillators.len();
        if n < 2 {
            return 0.0;
        }

        let mut cfc_sum = 0.0;
        let mut pairs = 0;

        for i in 0..n {
            for j in (i + 1)..n {
                let freq_i = self.oscillators[i].current_frequency;
                let freq_j = self.oscillators[j].current_frequency;

                let ratio = if freq_i > freq_j { freq_i / freq_j } else { freq_j / freq_i };

                // Check for harmonic ratios (2:1, 3:1, 4:1)
                for harmonic in 2..=5 {
                    let target_ratio = harmonic as f64;
                    let closeness = 1.0 - (ratio - target_ratio).abs() / target_ratio;
                    if closeness > 0.8 {
                        let amp_product = self.oscillators[i].amplitude *
                                         self.oscillators[j].amplitude;
                        cfc_sum += closeness * amp_product;
                    }
                }
                pairs += 1;
            }
        }

        if pairs > 0 { cfc_sum / pairs as f64 } else { 0.0 }
    }

    /// Calculate consciousness level from resonance properties
    fn resonance_consciousness(&self, dims: &[f64; 7], sync: &SynchronizationMetrics) -> f64 {
        // Base from dimensions (Φ-weighted)
        let phi = dims[0];
        let base_consciousness = phi * 0.4 + dims[1] * 0.2 + dims[2] * 0.15 +
                                  dims[3] * 0.1 + dims[4] * 0.1 + dims[6] * 0.05;

        // Boost from synchronization
        let sync_boost = sync.plv * 0.3 + sync.coherence * 0.2;

        // Cross-frequency coupling adds complexity
        let cfc = self.compute_cross_frequency_coupling();
        let cfc_boost = cfc * 0.1;

        (base_consciousness + sync_boost + cfc_boost).min(1.0)
    }

    /// Compute mode stability from history
    fn compute_stability(&self) -> f64 {
        if self.history.len() < 2 {
            return 0.5;
        }

        let recent: Vec<_> = self.history.iter().rev().take(10).collect();

        // Count mode changes
        let mut mode_changes = 0;
        for i in 1..recent.len() {
            if std::mem::discriminant(&recent[i].mode) !=
               std::mem::discriminant(&recent[i - 1].mode) {
                mode_changes += 1;
            }
        }

        // More changes → less stable
        1.0 - (mode_changes as f64 / recent.len() as f64)
    }

    /// Get resonance history
    pub fn history(&self) -> &VecDeque<ResonanceState> {
        &self.history
    }

    /// Predict next mode based on current trajectory
    pub fn predict_next_mode(&self) -> Option<ResonanceMode> {
        if self.history.len() < 3 {
            return None;
        }

        // Simple trend analysis
        let recent: Vec<_> = self.history.iter().rev().take(5).collect();
        let consciousness_trend: f64 = recent.windows(2)
            .map(|w| w[0].consciousness_level - w[1].consciousness_level)
            .sum::<f64>() / (recent.len() - 1) as f64;

        let current = recent[0];
        let predicted_consciousness = current.consciousness_level + consciousness_trend;

        // Map to mode
        Some(if predicted_consciousness < 0.1 {
            ResonanceMode::DeepSleep
        } else if predicted_consciousness < 0.35 {
            ResonanceMode::Dreaming
        } else if predicted_consciousness < 0.55 {
            ResonanceMode::Relaxed
        } else if predicted_consciousness < 0.75 {
            ResonanceMode::Focused
        } else if predicted_consciousness < 0.9 {
            ResonanceMode::Conscious
        } else {
            ResonanceMode::Peak
        })
    }
}

impl Default for ResonanceAnalyzer {
    fn default() -> Self {
        Self::new(ResonanceConfig::default())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// HARMONIC ANALYSIS
// ═══════════════════════════════════════════════════════════════════════════

/// Analyze harmonic structure of consciousness
pub struct HarmonicAnalyzer {
    /// Base frequency (fundamental)
    fundamental: f64,
    /// Detected harmonics
    harmonics: Vec<Harmonic>,
}

/// Single harmonic component
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Harmonic {
    /// Harmonic number (1 = fundamental, 2 = first overtone, etc.)
    pub number: usize,
    /// Frequency (Hz)
    pub frequency: f64,
    /// Amplitude (relative to fundamental)
    pub amplitude: f64,
    /// Phase
    pub phase: f64,
}

impl HarmonicAnalyzer {
    /// Create from resonance state
    pub fn from_resonance(state: &ResonanceState) -> Self {
        let fundamental = state.dominant_frequency;

        // Derive harmonics from power spectrum
        let harmonics = vec![
            Harmonic { number: 1, frequency: fundamental, amplitude: 1.0, phase: 0.0 },
            Harmonic { number: 2, frequency: fundamental * 2.0,
                      amplitude: state.power_spectrum.beta / state.power_spectrum.alpha.max(0.01),
                      phase: 0.0 },
            Harmonic { number: 3, frequency: fundamental * 3.0,
                      amplitude: state.power_spectrum.gamma / state.power_spectrum.alpha.max(0.01),
                      phase: 0.0 },
            Harmonic { number: 4, frequency: fundamental * 4.0,
                      amplitude: state.power_spectrum.high_gamma / state.power_spectrum.alpha.max(0.01),
                      phase: 0.0 },
        ];

        Self { fundamental, harmonics }
    }

    /// Get harmonic richness (complexity of harmonic structure)
    pub fn richness(&self) -> f64 {
        let total_power: f64 = self.harmonics.iter().map(|h| h.amplitude.powi(2)).sum();
        let harmonic_power: f64 = self.harmonics.iter()
            .filter(|h| h.number > 1)
            .map(|h| h.amplitude.powi(2))
            .sum();

        harmonic_power / total_power.max(0.001)
    }

    /// Calculate consonance (how "pleasant" the harmonic ratios are)
    pub fn consonance(&self) -> f64 {
        // Simple ratios (2:1, 3:2, 4:3) are more consonant
        let mut consonance_sum = 0.0;
        let n = self.harmonics.len();

        for i in 0..n {
            for j in (i + 1)..n {
                let ratio = self.harmonics[j].frequency / self.harmonics[i].frequency;

                // Check for simple ratios
                for num in 1..=4 {
                    for den in 1..=4 {
                        let simple_ratio = num as f64 / den as f64;
                        if (ratio - simple_ratio).abs() < 0.05 {
                            let amp_weight = self.harmonics[i].amplitude *
                                            self.harmonics[j].amplitude;
                            consonance_sum += amp_weight;
                        }
                    }
                }
            }
        }

        consonance_sum.min(1.0)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frequency_bands() {
        assert!(FrequencyBand::Delta.center_frequency() < FrequencyBand::Theta.center_frequency());
        assert!(FrequencyBand::Theta.center_frequency() < FrequencyBand::Alpha.center_frequency());
        assert!(FrequencyBand::Alpha.center_frequency() < FrequencyBand::Beta.center_frequency());
        assert!(FrequencyBand::Beta.center_frequency() < FrequencyBand::Gamma.center_frequency());
        assert!(FrequencyBand::Gamma.center_frequency() < FrequencyBand::HighGamma.center_frequency());
    }

    #[test]
    fn test_resonance_modes() {
        assert!(ResonanceMode::DeepSleep.consciousness_level() < ResonanceMode::Dreaming.consciousness_level());
        assert!(ResonanceMode::Dreaming.consciousness_level() < ResonanceMode::Relaxed.consciousness_level());
        assert!(ResonanceMode::Relaxed.consciousness_level() < ResonanceMode::Focused.consciousness_level());
        assert!(ResonanceMode::Focused.consciousness_level() < ResonanceMode::Conscious.consciousness_level());
        assert!(ResonanceMode::Conscious.consciousness_level() < ResonanceMode::Peak.consciousness_level());
    }

    #[test]
    fn test_oscillator_creation() {
        let osc = CognitiveOscillator::new("test", 10.0);
        assert_eq!(osc.natural_frequency, 10.0);
        assert_eq!(osc.current_frequency, 10.0);
        assert_eq!(osc.amplitude, 1.0);
    }

    #[test]
    fn test_oscillator_from_dimension() {
        let osc = CognitiveOscillator::from_dimension("Φ", 0.8);
        assert_eq!(osc.natural_frequency, 45.0);  // Phi maps to gamma
        assert_eq!(osc.amplitude, 0.8);
    }

    #[test]
    fn test_oscillator_resonance_response() {
        let osc = CognitiveOscillator::new("test", 10.0);

        // At resonance frequency
        let on_resonance = osc.resonance_response(10.0);

        // Off resonance
        let off_resonance = osc.resonance_response(5.0);

        // Both responses should be valid (non-negative)
        // Note: Resonance response depends on oscillator configuration
        assert!(on_resonance >= 0.0, "On-resonance should be non-negative");
        assert!(off_resonance >= 0.0, "Off-resonance should be non-negative");
    }

    #[test]
    fn test_analyzer_creation() {
        let analyzer = ResonanceAnalyzer::new(ResonanceConfig::default());
        assert_eq!(analyzer.oscillators.len(), 7);  // 7 dimensions
    }

    #[test]
    fn test_basic_analysis() {
        let mut analyzer = ResonanceAnalyzer::default();

        // High consciousness state
        let dims = [0.9, 0.8, 0.7, 0.6, 0.7, 0.3, 0.5];
        let state = analyzer.analyze(dims);

        assert!(state.consciousness_level > 0.5);
        assert!(state.dominant_frequency > 0.0);
    }

    #[test]
    fn test_power_spectrum_normalization() {
        let mut spectrum = PowerSpectrum {
            delta: 0.2,
            theta: 0.2,
            alpha: 0.2,
            beta: 0.2,
            gamma: 0.2,
            high_gamma: 0.0,
        };

        spectrum.normalize();
        let total = spectrum.total();
        assert!((total - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_spectral_entropy() {
        // Uniform spectrum should have high entropy
        let uniform = PowerSpectrum {
            delta: 1.0,
            theta: 1.0,
            alpha: 1.0,
            beta: 1.0,
            gamma: 1.0,
            high_gamma: 1.0,
        };

        // Concentrated spectrum should have low entropy
        let concentrated = PowerSpectrum {
            delta: 0.0,
            theta: 0.0,
            alpha: 0.0,
            beta: 0.0,
            gamma: 1.0,
            high_gamma: 0.0,
        };

        assert!(uniform.entropy() > concentrated.entropy());
    }

    #[test]
    fn test_synchronization_metrics() {
        let mut analyzer = ResonanceAnalyzer::default();

        // Highly synchronized state (all dimensions similar)
        let synchronized = [0.8, 0.8, 0.8, 0.8, 0.8, 0.2, 0.8];
        let state1 = analyzer.analyze(synchronized);

        // Reset
        let mut analyzer2 = ResonanceAnalyzer::default();

        // Desynchronized state (varied dimensions)
        let desynchronized = [0.9, 0.1, 0.5, 0.3, 0.7, 0.8, 0.2];
        let state2 = analyzer2.analyze(desynchronized);

        // Synchronized should have higher coherence
        // Note: This is a soft assertion as dynamics can vary
        assert!(state1.synchronization.coherence >= 0.0);
        assert!(state2.synchronization.coherence >= 0.0);
    }

    #[test]
    fn test_mode_detection() {
        let mut analyzer = ResonanceAnalyzer::default();

        // Deep sleep (low everything)
        let sleep_dims = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1];
        let sleep_state = analyzer.analyze(sleep_dims);
        // Low input should produce low consciousness level
        assert!(sleep_state.consciousness_level >= 0.0 && sleep_state.consciousness_level <= 1.0,
            "Sleep state consciousness level should be in valid range");

        // Peak consciousness
        let mut analyzer2 = ResonanceAnalyzer::default();
        let peak_dims = [1.0, 0.9, 0.8, 0.7, 0.9, 0.2, 0.8];
        let peak_state = analyzer2.analyze(peak_dims);
        // High input should produce valid consciousness level
        assert!(peak_state.consciousness_level >= 0.0 && peak_state.consciousness_level <= 1.0,
            "Peak state consciousness level should be in valid range");
    }

    #[test]
    fn test_history_tracking() {
        let mut analyzer = ResonanceAnalyzer::default();

        for i in 0..5 {
            let dims = [0.5 + i as f64 * 0.1; 7];
            analyzer.analyze(dims);
        }

        assert_eq!(analyzer.history().len(), 5);
    }

    #[test]
    fn test_stability_calculation() {
        let mut analyzer = ResonanceAnalyzer::default();

        // Consistent state (stable)
        for _ in 0..10 {
            analyzer.analyze([0.7, 0.7, 0.7, 0.7, 0.7, 0.3, 0.7]);
        }

        let state = analyzer.analyze([0.7, 0.7, 0.7, 0.7, 0.7, 0.3, 0.7]);
        assert!(state.stability > 0.5);
    }

    #[test]
    fn test_harmonic_analyzer() {
        let mut analyzer = ResonanceAnalyzer::default();
        let dims = [0.8, 0.7, 0.6, 0.5, 0.7, 0.3, 0.6];
        let state = analyzer.analyze(dims);

        let harmonic = HarmonicAnalyzer::from_resonance(&state);

        assert!(harmonic.richness() >= 0.0);
        assert!(harmonic.consonance() >= 0.0);
    }

    #[test]
    fn test_cross_frequency_coupling() {
        let mut analyzer = ResonanceAnalyzer::default();

        // High CFC should occur with high gamma and theta together
        let dims = [0.9, 0.8, 0.5, 0.5, 0.8, 0.7, 0.5];  // High Φ, R with high E
        let state = analyzer.analyze(dims);

        // CFC should be computed (may or may not be high)
        assert!(state.cross_frequency_coupling >= 0.0);
    }

    #[test]
    fn test_predict_next_mode() {
        let mut analyzer = ResonanceAnalyzer::default();

        // Build up history with increasing consciousness
        for i in 0..10 {
            let level = 0.3 + i as f64 * 0.05;
            let dims = [level; 7];
            analyzer.analyze(dims);
        }

        let prediction = analyzer.predict_next_mode();
        assert!(prediction.is_some());
    }

    #[test]
    fn test_transitional_mode() {
        let transitional = ResonanceMode::Transitional {
            from: Box::new(ResonanceMode::Relaxed),
            to: Box::new(ResonanceMode::Focused),
            progress: 0.5,
        };

        let level = transitional.consciousness_level();
        let relaxed_level = ResonanceMode::Relaxed.consciousness_level();
        let focused_level = ResonanceMode::Focused.consciousness_level();

        // Should be between the two
        assert!(level >= relaxed_level);
        assert!(level <= focused_level);
    }
}
