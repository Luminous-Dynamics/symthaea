// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Central Pattern Generator (CPG) Manager
//!
//! Generates rhythmic motor timing signals using Kuramoto coupled oscillators.
//! This is a biologically grounded approach: real CPGs in the spinal cord produce
//! locomotion rhythms without cortical input (Brown 1911, Grillner 2006).
//!
//! The CfC neurons in the cognitive loop send frequency/amplitude commands to
//! the CPG, which handles the rhythm internally. This separation is critical
//! because the CfC closed-form equation is a contraction mapping — it decays
//! to equilibrium and cannot sustain oscillation natively.
//!
//! ## Kuramoto Model
//!
//! Each oscillator `i` evolves as:
//! ```text
//! dθ_i/dt = ω_i + Σ_j K_ij sin(θ_j - θ_i)
//! ```
//!
//! Where `ω_i` is the natural frequency and `K_ij` is the coupling strength.
//! The Kuramoto order parameter `r = |1/N Σ exp(iθ_j)|` measures global
//! synchronization (1.0 = perfect sync, 0.0 = incoherent).
//!
//! ## Gait Presets
//!
//! Coupling matrices encode phase relationships for locomotion:
//! - **Walk**: alternating limbs (180° phase offset between contralateral pairs)
//! - **Trot**: diagonal pairs in phase (LF+RH sync, RF+LH sync)
//! - **Gallop**: front pair leads, back pair follows (~90° offset)
//!
//! ## Integration
//!
//! - Reads `arousal` from `CycleSnapshot` to modulate oscillator frequency
//! - Outputs `SubsystemOutput` with exploration delta based on sync index
//! - Telemetry: `CpgTelemetry` with phases, sync index, mean frequency
//!
//! ## References
//!
//! - Brown, T.G. (1911). The intrinsic factors in the act of progression.
//! - Grillner, S. (2006). Biological pattern generation. Neuron.
//! - Kuramoto, Y. (1975). Self-entrainment of coupled oscillators.

use super::super::subsystem_trait::{CognitiveSubsystem, CycleSnapshot, SubsystemOutput};
use serde::{Deserialize, Serialize};
use std::f64::consts::TAU; // 2π

use crate::cognitive_loop::thresholds::{
    CPG_AROUSAL_FREQ_SCALE, CPG_CRITICAL_DESYNC, CPG_DEFAULT_COUPLING_K,
    CPG_DESYNC_EXPLORATION_BOOST, CPG_GALLOP_MIN_SYNC, CPG_INTERVAL, CPG_TROT_MIN_SYNC,
    CPG_WALK_MIN_SYNC,
};

// ═══════════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Gait pattern preset defining phase relationships between oscillators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GaitPreset {
    /// Alternating limbs: contralateral pairs 180° out of phase.
    /// Each limb has 2 oscillators (hip + knee), knee lags hip by ~60°.
    Walk,
    /// Diagonal pairs in phase: LF+RH sync, RF+LH sync.
    /// Faster than walk, pairs move together.
    Trot,
    /// Front pair leads, back pair follows with ~90° offset.
    /// Asymmetric — inherently lower coherence than walk/trot.
    Gallop,
}

impl GaitPreset {
    /// Minimum expected Kuramoto order parameter `r` for this gait.
    /// Below this during active motor output = safety-relevant desynchronization.
    pub fn min_sync(self) -> f64 {
        match self {
            GaitPreset::Walk => CPG_WALK_MIN_SYNC,
            GaitPreset::Trot => CPG_TROT_MIN_SYNC,
            GaitPreset::Gallop => CPG_GALLOP_MIN_SYNC,
        }
    }

    /// Base frequency (Hz) for this gait pattern.
    pub fn base_freq_hz(self) -> f64 {
        match self {
            GaitPreset::Walk => 2.0,
            GaitPreset::Trot => 4.0,
            GaitPreset::Gallop => 6.0,
        }
    }

    /// Build the 8×8 coupling matrix for this gait.
    ///
    /// Oscillator layout (quad locomotion):
    ///   0: LF-hip,  1: LF-knee
    ///   2: RF-hip,  3: RF-knee
    ///   4: LH-hip,  5: LH-knee
    ///   6: RH-hip,  7: RH-knee
    ///
    /// Hip-knee pairs within a limb are always positively coupled.
    /// Inter-limb coupling defines the gait pattern.
    pub fn coupling_matrix(self, k: f64) -> [[f64; 8]; 8] {
        let mut m = [[0.0f64; 8]; 8];

        // Intra-limb hip-knee coupling (always positive, half strength)
        let intra = k * 0.5;
        for limb in 0..4 {
            let hip = limb * 2;
            let knee = hip + 1;
            m[hip][knee] = intra;
            m[knee][hip] = intra;
        }

        // Inter-limb hip-hip coupling (defines gait)
        match self {
            GaitPreset::Walk => {
                // Alternating: LF-RH in phase, RF-LH in phase
                // LF(0) and RH(6) positively coupled
                // RF(2) and LH(4) positively coupled
                // LF(0) and RF(2) anti-phase (negative coupling)
                // LH(4) and RH(6) anti-phase
                m[0][6] = k;
                m[6][0] = k;
                m[2][4] = k;
                m[4][2] = k;
                m[0][2] = -k;
                m[2][0] = -k;
                m[4][6] = -k;
                m[6][4] = -k;
                // Front-back contralateral: anti-phase
                m[0][4] = -k;
                m[4][0] = -k;
                m[2][6] = -k;
                m[6][2] = -k;
            }
            GaitPreset::Trot => {
                // Diagonal pairs: LF(0)+RH(6) sync, RF(2)+LH(4) sync
                m[0][6] = k;
                m[6][0] = k;
                m[2][4] = k;
                m[4][2] = k;
                // Opposite diagonal: anti-phase
                m[0][4] = -k;
                m[4][0] = -k;
                m[2][6] = -k;
                m[6][2] = -k;
                m[0][2] = -k;
                m[2][0] = -k;
                m[4][6] = -k;
                m[6][4] = -k;
            }
            GaitPreset::Gallop => {
                // Front pair in phase: LF(0)+RF(2) sync
                m[0][2] = k;
                m[2][0] = k;
                // Back pair in phase: LH(4)+RH(6) sync
                m[4][6] = k;
                m[6][4] = k;
                // Front-back: mild positive coupling (asymmetric offset)
                let asym = k * 0.3;
                m[0][4] = asym;
                m[4][0] = asym;
                m[0][6] = asym;
                m[6][0] = asym;
                m[2][4] = asym;
                m[4][2] = asym;
                m[2][6] = asym;
                m[6][2] = asym;
            }
        }

        m
    }

    /// Target phase offsets (radians) for each oscillator relative to oscillator 0.
    /// Used for gait-specific initialization.
    pub fn target_phases(self) -> [f64; 8] {
        let pi = std::f64::consts::PI;
        let knee_lag = pi / 3.0; // 60° hip-knee lag

        match self {
            GaitPreset::Walk => {
                // LF=0, RF=π, LH=π, RH=0 (alternating)
                [
                    0.0,
                    knee_lag,
                    pi,
                    pi + knee_lag,
                    pi,
                    pi + knee_lag,
                    0.0,
                    knee_lag,
                ]
            }
            GaitPreset::Trot => {
                // LF=0, RF=π, LH=π, RH=0 (same as walk but different coupling)
                [
                    0.0,
                    knee_lag,
                    pi,
                    pi + knee_lag,
                    pi,
                    pi + knee_lag,
                    0.0,
                    knee_lag,
                ]
            }
            GaitPreset::Gallop => {
                // Front pair=0, back pair≈π/2 (90° offset)
                let back_offset = pi / 2.0;
                [
                    0.0,
                    knee_lag,
                    0.0,
                    knee_lag,
                    back_offset,
                    back_offset + knee_lag,
                    back_offset,
                    back_offset + knee_lag,
                ]
            }
        }
    }
}

/// Configuration for the CPG Manager.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpgConfig {
    /// Number of oscillators (default: 8 for quad locomotion).
    pub n_oscillators: usize,
    /// Global coupling strength K (default: CPG_DEFAULT_COUPLING_K).
    pub coupling_k: f64,
    /// Gait preset for coupling matrix initialization.
    pub gait: GaitPreset,
    /// Whether CPG is actively producing motor output (vs idle/monitoring).
    pub motor_active: bool,
    /// Substrate tau factor for frequency scaling.
    pub tau_factor: f64,
}

impl Default for CpgConfig {
    fn default() -> Self {
        Self {
            n_oscillators: 8,
            coupling_k: CPG_DEFAULT_COUPLING_K,
            gait: GaitPreset::Walk,
            motor_active: false,
            tau_factor: 1.0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CPG OSCILLATOR
// ═══════════════════════════════════════════════════════════════════════════════

/// Single Kuramoto oscillator with phase and frequency.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpgOscillator {
    /// Current phase [0, 2π)
    pub phase: f64,
    /// Natural frequency (Hz)
    pub natural_freq: f64,
    /// Output amplitude scaling [0, 1]
    pub amplitude: f64,
}

impl CpgOscillator {
    /// Current output signal: sin(phase) * amplitude
    #[inline]
    pub fn output(&self) -> f64 {
        self.phase.sin() * self.amplitude
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CPG TELEMETRY
// ═══════════════════════════════════════════════════════════════════════════════

/// Telemetry snapshot from the CPG manager.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CpgTelemetry {
    /// Current phase of each oscillator (radians)
    pub phases: Vec<f64>,
    /// Current output of each oscillator
    pub outputs: Vec<f64>,
    /// Kuramoto order parameter r (0.0 = incoherent, 1.0 = perfect sync)
    pub sync_index: f64,
    /// Mean oscillator frequency (Hz)
    pub mean_freq: f64,
    /// Current gait preset name
    pub gait: String,
    /// Whether motor output is active
    pub motor_active: bool,
    /// Whether desynchronization was detected this tick
    pub desync_alert: bool,
}

// ═══════════════════════════════════════════════════════════════════════════════
// CPG MANAGER
// ═══════════════════════════════════════════════════════════════════════════════

/// Central Pattern Generator Manager.
///
/// Implements `CognitiveSubsystem` at interval 59 (co-prime with 7, 11, 13, 19, 29, 37, 41, 53).
/// Generates rhythmic motor timing via Kuramoto coupled oscillators.
pub struct CpgManager {
    /// Oscillator bank
    oscillators: Vec<CpgOscillator>,
    /// Coupling matrix K_ij (n × n)
    coupling: Vec<Vec<f64>>,
    /// Configuration
    config: CpgConfig,
    /// Last computed sync index (Kuramoto order parameter)
    sync_index: f64,
    /// Accumulated simulation time (seconds)
    sim_time: f64,
    /// Last telemetry snapshot
    telemetry: CpgTelemetry,
    /// Cycles since last desync alert (cooldown to avoid spamming)
    desync_cooldown: u32,
}

impl CpgManager {
    /// Time step per cognitive cycle at 31 Hz (measured loop rate).
    const DT: f64 = 1.0 / 31.0;

    /// Desync alert cooldown (cycles) — don't spam safety with repeated alerts.
    const DESYNC_COOLDOWN_CYCLES: u32 = 10;

    /// Create a new CPG manager with the given configuration.
    pub fn new(config: CpgConfig) -> Self {
        let n = config.n_oscillators;
        let gait = config.gait;
        let base_freq = gait.base_freq_hz() * config.tau_factor;
        let target_phases = gait.target_phases();

        // Initialize oscillators with gait-specific phases
        let oscillators: Vec<CpgOscillator> = (0..n)
            .map(|i| {
                let phase = if i < 8 { target_phases[i] } else { 0.0 };
                CpgOscillator {
                    phase,
                    natural_freq: base_freq,
                    amplitude: 1.0,
                }
            })
            .collect();

        // Build coupling matrix from gait preset
        let coupling_8x8 = gait.coupling_matrix(config.coupling_k);
        let coupling: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        if i < 8 && j < 8 {
                            coupling_8x8[i][j]
                        } else {
                            0.0
                        }
                    })
                    .collect()
            })
            .collect();

        Self {
            oscillators,
            coupling,
            config,
            sync_index: 1.0,
            sim_time: 0.0,
            telemetry: CpgTelemetry::default(),
            desync_cooldown: 0,
        }
    }

    /// Step all oscillators forward by `dt` seconds using Kuramoto dynamics.
    ///
    /// dθ_i/dt = ω_i + Σ_j K_ij sin(θ_j - θ_i)
    fn step(&mut self, dt: f64) {
        let n = self.oscillators.len();
        // Compute phase derivatives (read-only pass)
        let mut d_phase = vec![0.0f64; n];
        for i in 0..n {
            let omega_i = self.oscillators[i].natural_freq * TAU; // Hz to rad/s
            let mut coupling_sum = 0.0;
            for j in 0..n {
                if i != j {
                    coupling_sum += self.coupling[i][j]
                        * (self.oscillators[j].phase - self.oscillators[i].phase).sin();
                }
            }
            d_phase[i] = omega_i + coupling_sum;
        }
        // Apply phase updates
        for i in 0..n {
            self.oscillators[i].phase += d_phase[i] * dt;
            // Wrap to [0, 2π)
            self.oscillators[i].phase = self.oscillators[i].phase.rem_euclid(TAU);
        }
        self.sim_time += dt;
    }

    /// Compute the Kuramoto order parameter r = |1/N Σ exp(iθ_j)|.
    ///
    /// r = 1.0 means perfect phase synchronization.
    /// r = 0.0 means phases are uniformly distributed (incoherent).
    fn compute_sync_index(&self) -> f64 {
        let n = self.oscillators.len();
        if n == 0 {
            return 0.0;
        }
        let mut re_sum = 0.0;
        let mut im_sum = 0.0;
        for osc in &self.oscillators {
            re_sum += osc.phase.cos();
            im_sum += osc.phase.sin();
        }
        let r = (re_sum * re_sum + im_sum * im_sum).sqrt() / n as f64;
        r.clamp(0.0, 1.0)
    }

    /// Modulate oscillator frequencies based on arousal level.
    ///
    /// Higher arousal = faster gait (Yerkes-Dodson effect on motor output).
    fn modulate_frequency(&mut self, arousal: f32) {
        let base = self.config.gait.base_freq_hz() * self.config.tau_factor;
        // Arousal 0.5 = neutral (base freq), 1.0 = max boost, 0.0 = slowest
        let arousal_factor = 1.0 + (arousal as f64 - 0.5) * CPG_AROUSAL_FREQ_SCALE;
        let freq = base * arousal_factor.max(0.3); // Floor at 30% base
        for osc in &mut self.oscillators {
            osc.natural_freq = freq;
        }
    }

    /// Update telemetry snapshot.
    fn update_telemetry(&mut self, desync_alert: bool) {
        self.telemetry = CpgTelemetry {
            phases: self.oscillators.iter().map(|o| o.phase).collect(),
            outputs: self.oscillators.iter().map(|o| o.output()).collect(),
            sync_index: self.sync_index,
            mean_freq: if self.oscillators.is_empty() {
                0.0
            } else {
                self.oscillators.iter().map(|o| o.natural_freq).sum::<f64>()
                    / self.oscillators.len() as f64
            },
            gait: format!("{:?}", self.config.gait),
            motor_active: self.config.motor_active,
            desync_alert,
        };
    }

    // ── Public Accessors ─────────────────────────────────────────────────

    /// Current sync index (Kuramoto order parameter).
    pub fn sync_index(&self) -> f64 {
        self.sync_index
    }

    /// Current oscillator outputs.
    pub fn outputs(&self) -> Vec<f64> {
        self.oscillators.iter().map(|o| o.output()).collect()
    }

    /// Current oscillator phases.
    pub fn phases(&self) -> Vec<f64> {
        self.oscillators.iter().map(|o| o.phase).collect()
    }

    /// Current telemetry.
    pub fn telemetry(&self) -> &CpgTelemetry {
        &self.telemetry
    }

    /// Set motor active state.
    pub fn set_motor_active(&mut self, active: bool) {
        self.config.motor_active = active;
    }

    /// Switch gait preset, rebuilding the coupling matrix.
    pub fn set_gait(&mut self, gait: GaitPreset) {
        self.config.gait = gait;
        let n = self.config.n_oscillators;
        let coupling_8x8 = gait.coupling_matrix(self.config.coupling_k);
        self.coupling = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        if i < 8 && j < 8 {
                            coupling_8x8[i][j]
                        } else {
                            0.0
                        }
                    })
                    .collect()
            })
            .collect();
        // Re-initialize phases for new gait
        let target_phases = gait.target_phases();
        for (i, osc) in self.oscillators.iter_mut().enumerate() {
            if i < 8 {
                osc.phase = target_phases[i];
            }
        }
    }

    /// Reset phase of a specific oscillator (perturbation recovery testing).
    pub fn reset_phase(&mut self, idx: usize, new_phase: f64) {
        if let Some(osc) = self.oscillators.get_mut(idx) {
            osc.phase = new_phase.rem_euclid(TAU);
        }
    }

    /// Update substrate tau factor and rescale frequencies.
    pub fn set_tau_factor(&mut self, tau_factor: f64) {
        self.config.tau_factor = tau_factor;
    }

    /// Number of oscillators.
    pub fn n_oscillators(&self) -> usize {
        self.oscillators.len()
    }

    /// Current configuration (read-only).
    pub fn config(&self) -> &CpgConfig {
        &self.config
    }
}

impl CognitiveSubsystem for CpgManager {
    fn name(&self) -> &'static str {
        "cpg"
    }

    fn interval(&self) -> u32 {
        CPG_INTERVAL
    }

    fn process(&mut self, snapshot: &CycleSnapshot) -> SubsystemOutput {
        // Modulate frequency based on arousal
        self.modulate_frequency(snapshot.arousal);

        // Step oscillators forward
        self.step(Self::DT);

        // Compute synchronization
        self.sync_index = self.compute_sync_index();

        // Check for desynchronization alert
        let mut desync_alert = false;
        if self.desync_cooldown > 0 {
            self.desync_cooldown -= 1;
        }

        if self.config.motor_active && self.desync_cooldown == 0 {
            if self.sync_index < CPG_CRITICAL_DESYNC {
                desync_alert = true;
                self.desync_cooldown = Self::DESYNC_COOLDOWN_CYCLES;
            } else if self.sync_index < self.config.gait.min_sync() {
                desync_alert = true;
                self.desync_cooldown = Self::DESYNC_COOLDOWN_CYCLES;
            }
        }

        self.update_telemetry(desync_alert);

        // Build output
        let mut output = SubsystemOutput::NEUTRAL;

        // Desynchronization during idle → exploration boost
        // (motor incoherence when not moving = curiosity signal)
        if !self.config.motor_active && self.sync_index < self.config.gait.min_sync() {
            output.exploration_delta = CPG_DESYNC_EXPLORATION_BOOST as f64;
        }

        // Critical desync during active motor → arousal boost + flag
        if desync_alert {
            output.arousal_delta = crate::cognitive_loop::thresholds::CPG_DESYNC_AROUSAL_DELTA;
            output.flags |= crate::cognitive_loop::subsystem_trait::output_flags::ANOMALY_DETECTED;
        }

        output
    }

    fn checkpoint(&self) -> Vec<u8> {
        // Serialize oscillator phases and config for restore
        let data = (
            &self.oscillators,
            self.sync_index,
            self.sim_time,
            &self.config,
        );
        serde_json::to_vec(&data).unwrap_or_else(|e| {
            tracing::warn!("CpgManager checkpoint serialization failed: {e}");
            Vec::new()
        })
    }

    fn restore(&mut self, data: &[u8]) -> Result<(), String> {
        if data.is_empty() {
            return Err("CpgManager restore: empty checkpoint data".to_string());
        }
        let (oscillators, sync_index, sim_time, config): (Vec<CpgOscillator>, f64, f64, CpgConfig) =
            serde_json::from_slice(data).map_err(|e| format!("CpgManager restore failed: {e}"))?;
        self.oscillators = oscillators;
        self.sync_index = sync_index;
        self.sim_time = sim_time;
        // Re-derive coupling from restored config
        let gait = config.gait;
        let n = config.n_oscillators;
        let arr = gait.coupling_matrix(n as f64);
        self.coupling = arr.iter().map(|row| row.to_vec()).collect();
        self.config = config;
        tracing::debug!(
            "CpgManager restored: {} oscillators, sim_time={sim_time:.2}s",
            n
        );
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn default_cpg() -> CpgManager {
        CpgManager::new(CpgConfig::default())
    }

    fn active_cpg(gait: GaitPreset) -> CpgManager {
        CpgManager::new(CpgConfig {
            gait,
            motor_active: true,
            ..CpgConfig::default()
        })
    }

    #[test]
    fn test_creation() {
        let cpg = default_cpg();
        assert_eq!(cpg.n_oscillators(), 8);
        assert!((cpg.sync_index() - 1.0).abs() < 0.01 || cpg.sync_index() > 0.0);
    }

    #[test]
    fn test_kuramoto_sync_from_random() {
        // Walk gait has anti-phase coupling (contralateral limbs at 180°),
        // so the global order parameter r converges near 0 in the stable state
        // (half at phase 0, half at phase π → they cancel in the sum).
        // Instead, verify that oscillators reach stable phase relationships
        // by checking that phases stop changing significantly.
        let mut cpg = CpgManager::new(CpgConfig {
            coupling_k: 4.0, // Strong coupling for fast convergence
            ..CpgConfig::default()
        });
        // Scramble phases
        for (i, osc) in cpg.oscillators.iter_mut().enumerate() {
            osc.phase = (i as f64 * 1.7) % TAU;
        }

        let snapshot = CycleSnapshot::default();
        // Run to steady state
        for _ in 0..500 {
            cpg.process(&snapshot);
        }
        let phases_early = cpg.phases();

        // Run 100 more steps
        for _ in 0..100 {
            cpg.process(&snapshot);
        }
        let phases_late = cpg.phases();

        // Phase differences between paired oscillators should be stable
        // (LF-hip vs RH-hip should maintain consistent offset)
        let diff_early = (phases_early[0] - phases_early[6]).rem_euclid(TAU);
        let diff_late = (phases_late[0] - phases_late[6]).rem_euclid(TAU);
        let drift = (diff_early - diff_late).abs();
        assert!(
            drift < 0.5 || (drift - TAU).abs() < 0.5,
            "Phase relationship should stabilize: early_diff={:.3}, late_diff={:.3}, drift={:.3}",
            diff_early,
            diff_late,
            drift
        );
    }

    #[test]
    fn test_walk_gait_phase_relationships() {
        let cpg = CpgManager::new(CpgConfig {
            gait: GaitPreset::Walk,
            ..CpgConfig::default()
        });
        let phases = cpg.phases();
        let pi = std::f64::consts::PI;

        // LF-hip (0) and RH-hip (6) should be in phase (both 0.0)
        let diff_lf_rh = (phases[0] - phases[6]).abs();
        assert!(
            diff_lf_rh < 0.1 || (diff_lf_rh - TAU).abs() < 0.1,
            "LF and RH should be in phase for walk, got diff={}",
            diff_lf_rh
        );

        // LF-hip (0) and RF-hip (2) should be anti-phase (~π apart)
        let diff_lf_rf = (phases[0] - phases[2]).abs();
        assert!(
            (diff_lf_rf - pi).abs() < 0.2,
            "LF and RF should be anti-phase for walk, got diff={}",
            diff_lf_rf
        );
    }

    #[test]
    fn test_trot_gait_diagonal_pairs() {
        let cpg = CpgManager::new(CpgConfig {
            gait: GaitPreset::Trot,
            ..CpgConfig::default()
        });
        let phases = cpg.phases();

        // LF-hip (0) and RH-hip (6) should be in phase
        let diff = (phases[0] - phases[6]).abs();
        assert!(
            diff < 0.1 || (diff - TAU).abs() < 0.1,
            "LF and RH should be in phase for trot, got diff={}",
            diff
        );

        // RF-hip (2) and LH-hip (4) should be in phase
        let diff2 = (phases[2] - phases[4]).abs();
        assert!(
            diff2 < 0.1 || (diff2 - TAU).abs() < 0.1,
            "RF and LH should be in phase for trot, got diff={}",
            diff2
        );
    }

    #[test]
    fn test_gallop_front_back_offset() {
        let cpg = CpgManager::new(CpgConfig {
            gait: GaitPreset::Gallop,
            ..CpgConfig::default()
        });
        let phases = cpg.phases();
        let pi = std::f64::consts::PI;

        // Front pair (0, 2) should be in phase
        let diff_front = (phases[0] - phases[2]).abs();
        assert!(
            diff_front < 0.1,
            "Front pair should be in phase for gallop, got diff={}",
            diff_front
        );

        // Back pair leads by ~π/2
        let diff_fb = (phases[4] - phases[0]).abs();
        assert!(
            (diff_fb - pi / 2.0).abs() < 0.3,
            "Front-back offset should be ~π/2 for gallop, got diff={}",
            diff_fb
        );
    }

    #[test]
    fn test_arousal_modulates_frequency() {
        let mut cpg = default_cpg();
        let base_freq = cpg.oscillators[0].natural_freq;

        // High arousal should increase frequency
        cpg.modulate_frequency(0.9);
        let high_freq = cpg.oscillators[0].natural_freq;
        assert!(
            high_freq > base_freq,
            "High arousal should increase freq: base={}, high={}",
            base_freq,
            high_freq
        );

        // Low arousal should decrease frequency
        cpg.modulate_frequency(0.1);
        let low_freq = cpg.oscillators[0].natural_freq;
        assert!(
            low_freq < base_freq,
            "Low arousal should decrease freq: base={}, low={}",
            base_freq,
            low_freq
        );
    }

    #[test]
    fn test_phase_reset_recovery() {
        let mut cpg = CpgManager::new(CpgConfig {
            coupling_k: 4.0,
            ..CpgConfig::default()
        });

        // Run to steady state
        let snapshot = CycleSnapshot::default();
        for _ in 0..100 {
            cpg.process(&snapshot);
        }
        let sync_before = cpg.sync_index();

        // Perturb one oscillator
        cpg.reset_phase(0, 3.14);

        // Run to recover
        for _ in 0..200 {
            cpg.process(&snapshot);
        }
        let sync_after = cpg.sync_index();

        assert!(
            (sync_after - sync_before).abs() < 0.3,
            "Should recover from perturbation: before={}, after={}",
            sync_before,
            sync_after
        );
    }

    #[test]
    fn test_gait_switch() {
        let mut cpg = default_cpg();
        assert_eq!(cpg.config().gait, GaitPreset::Walk);

        cpg.set_gait(GaitPreset::Trot);
        assert_eq!(cpg.config().gait, GaitPreset::Trot);

        // Phase should be re-initialized
        let phases = cpg.phases();
        let trot_targets = GaitPreset::Trot.target_phases();
        for (i, &phase) in phases.iter().enumerate() {
            if i < 8 {
                assert!(
                    (phase - trot_targets[i]).abs() < 0.01,
                    "Phase {} should match trot target: got={}, expected={}",
                    i,
                    phase,
                    trot_targets[i]
                );
            }
        }
    }

    #[test]
    fn test_desync_during_walk_triggers_alert() {
        let mut cpg = active_cpg(GaitPreset::Walk);

        // Scramble phases to force desync
        for (i, osc) in cpg.oscillators.iter_mut().enumerate() {
            osc.phase = (i as f64 * 0.9) % TAU;
        }
        // Use very weak coupling so they don't sync quickly
        cpg.coupling = vec![vec![0.0; 8]; 8];

        let snapshot = CycleSnapshot::default();
        let output = cpg.process(&snapshot);

        // Should detect desync (scrambled phases + zero coupling = low r)
        assert!(
            cpg.sync_index() < CPG_WALK_MIN_SYNC,
            "Scrambled phases should have low sync: r={}",
            cpg.sync_index()
        );
        assert!(
            cpg.telemetry().desync_alert,
            "Should trigger desync alert during active walk"
        );
        assert!(
            output.flags & crate::cognitive_loop::subsystem_trait::output_flags::ANOMALY_DETECTED
                != 0,
            "Should set ANOMALY_DETECTED flag"
        );
    }

    #[test]
    fn test_critical_desync_triggers_alert() {
        let mut cpg = active_cpg(GaitPreset::Walk);

        // Force completely incoherent phases with zero coupling
        cpg.coupling = vec![vec![0.0; 8]; 8];
        // Evenly spread phases so r ≈ 0
        for (i, osc) in cpg.oscillators.iter_mut().enumerate() {
            osc.phase = TAU * (i as f64) / 8.0;
        }

        let snapshot = CycleSnapshot::default();
        let output = cpg.process(&snapshot);

        assert!(
            cpg.sync_index() < CPG_CRITICAL_DESYNC,
            "Evenly spread phases should have near-zero sync: r={}",
            cpg.sync_index()
        );
        assert!(
            output.arousal_delta > 0.0,
            "Critical desync should boost arousal"
        );
    }

    #[test]
    fn test_desync_during_idle_no_alert() {
        let mut cpg = default_cpg(); // motor_active = false

        // Scramble phases
        cpg.coupling = vec![vec![0.0; 8]; 8];
        for (i, osc) in cpg.oscillators.iter_mut().enumerate() {
            osc.phase = TAU * (i as f64) / 8.0;
        }

        let snapshot = CycleSnapshot::default();
        let output = cpg.process(&snapshot);

        // Should NOT trigger safety alert when idle
        assert!(
            !cpg.telemetry().desync_alert,
            "Should not alert during idle even with low sync"
        );
        // But SHOULD boost exploration
        assert!(
            output.exploration_delta > 0.0,
            "Idle desync should boost exploration"
        );
    }

    #[test]
    fn test_gallop_tolerates_lower_sync() {
        // Gallop has lower min_sync than walk
        assert!(GaitPreset::Gallop.min_sync() < GaitPreset::Walk.min_sync());
        assert!(GaitPreset::Gallop.min_sync() < GaitPreset::Trot.min_sync());
    }

    #[test]
    fn test_output_is_oscillatory() {
        let mut cpg = default_cpg();
        let snapshot = CycleSnapshot::default();

        let mut outputs_over_time: Vec<f64> = Vec::new();
        for _ in 0..100 {
            cpg.process(&snapshot);
            outputs_over_time.push(cpg.oscillators[0].output());
        }

        // Check for sign changes (oscillation)
        let sign_changes: usize = outputs_over_time
            .windows(2)
            .filter(|w| w[0].signum() != w[1].signum())
            .count();

        assert!(
            sign_changes >= 2,
            "Oscillator output should change sign multiple times, got {} changes",
            sign_changes
        );
    }

    #[test]
    fn test_substrate_tau_scaling() {
        let cpg_1x = CpgManager::new(CpgConfig {
            tau_factor: 1.0,
            ..CpgConfig::default()
        });
        let cpg_10x = CpgManager::new(CpgConfig {
            tau_factor: 10.0,
            ..CpgConfig::default()
        });

        let freq_1x = cpg_1x.oscillators[0].natural_freq;
        let freq_10x = cpg_10x.oscillators[0].natural_freq;

        assert!(
            (freq_10x / freq_1x - 10.0).abs() < 0.01,
            "10x tau_factor should give 10x frequency: 1x={}, 10x={}",
            freq_1x,
            freq_10x
        );
    }

    #[test]
    fn test_telemetry_populated() {
        let mut cpg = default_cpg();
        let snapshot = CycleSnapshot::default();
        cpg.process(&snapshot);

        let telem = cpg.telemetry();
        assert_eq!(telem.phases.len(), 8);
        assert_eq!(telem.outputs.len(), 8);
        assert!(telem.sync_index >= 0.0 && telem.sync_index <= 1.0);
        assert!(telem.mean_freq > 0.0);
        assert_eq!(telem.gait, "Walk");
    }

    #[test]
    fn test_coupling_matrix_symmetric_walk() {
        let m = GaitPreset::Walk.coupling_matrix(2.0);
        // Coupling should be symmetric: K_ij = K_ji
        for i in 0..8 {
            for j in 0..8 {
                assert!(
                    (m[i][j] - m[j][i]).abs() < 1e-10,
                    "Coupling should be symmetric: K[{}][{}]={}, K[{}][{}]={}",
                    i,
                    j,
                    m[i][j],
                    j,
                    i,
                    m[j][i]
                );
            }
        }
    }

    #[test]
    fn test_coupling_matrix_zero_diagonal() {
        for gait in [GaitPreset::Walk, GaitPreset::Trot, GaitPreset::Gallop] {
            let m = gait.coupling_matrix(2.0);
            for i in 0..8 {
                assert!(
                    m[i][i].abs() < 1e-10,
                    "Self-coupling should be zero for {:?}: K[{}][{}]={}",
                    gait,
                    i,
                    i,
                    m[i][i]
                );
            }
        }
    }

    #[test]
    fn test_checkpoint_returns_bytes() {
        let cpg = default_cpg();
        let bytes = cpg.checkpoint();
        assert!(
            !bytes.is_empty(),
            "Checkpoint should produce non-empty bytes"
        );
    }

    #[test]
    fn test_phase_wrapping() {
        let mut cpg = default_cpg();
        // Set phase beyond 2π
        cpg.oscillators[0].phase = 10.0;
        cpg.step(0.01);
        // Phase should be wrapped to [0, 2π)
        assert!(
            cpg.oscillators[0].phase >= 0.0 && cpg.oscillators[0].phase < TAU,
            "Phase should wrap to [0, 2π), got {}",
            cpg.oscillators[0].phase
        );
    }

    #[test]
    fn test_sync_index_bounds() {
        let cpg = default_cpg();
        let r = cpg.compute_sync_index();
        assert!(
            r >= 0.0 && r <= 1.0,
            "Sync index should be in [0,1], got {}",
            r
        );
    }

    #[test]
    fn test_empty_oscillator_bank() {
        let cpg = CpgManager::new(CpgConfig {
            n_oscillators: 0,
            ..CpgConfig::default()
        });
        assert_eq!(cpg.compute_sync_index(), 0.0);
        assert_eq!(cpg.outputs().len(), 0);
    }

    #[test]
    fn test_desync_cooldown() {
        let mut cpg = active_cpg(GaitPreset::Walk);
        cpg.coupling = vec![vec![0.0; 8]; 8];
        for (i, osc) in cpg.oscillators.iter_mut().enumerate() {
            osc.phase = (i as f64 * 0.9) % TAU;
        }

        let snapshot = CycleSnapshot::default();

        // First tick: should alert
        cpg.process(&snapshot);
        assert!(cpg.telemetry().desync_alert);

        // Immediate next tick: cooldown should suppress
        cpg.process(&snapshot);
        assert!(
            !cpg.telemetry().desync_alert,
            "Cooldown should suppress repeated alerts"
        );
    }
}
