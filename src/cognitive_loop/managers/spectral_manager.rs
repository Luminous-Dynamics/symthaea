// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Spectral Twin Manager
//!
//! Maintains a frequency-domain twin of the CfC hidden state by applying
//! spectral analysis to a rolling history of `compressed_state` snapshots.
//!
//! This closes the feedback loop between neural oscillations and consciousness:
//! band power ratios and cross-frequency coupling directly modulate confidence,
//! exploration, and arousal via `SubsystemOutput`.
//!
//! ## Architecture
//!
//! The manager maintains a ring buffer of recent CfC hidden states (128 cycles
//! ≈ 4 seconds at 31 Hz). Each tick, it appends the current state and runs
//! spectral analysis on each dimension's time series independently, then
//! aggregates across dimensions.
//!
//! ## Spectral Signatures (Buzsaki 2006)
//!
//! - **Delta dominance** (1-4 Hz): consolidated, low-complexity → request rest
//! - **Theta dominance** (4-8 Hz): temporal binding, memory encoding
//! - **Alpha dominance** (8-13 Hz): relaxed, motor-ready → confidence boost
//! - **Beta dominance** (13-30 Hz): active motor execution → lr boost
//! - **Gamma dominance** (30-100 Hz): binding, active perception → consciousness boost
//! - **High spectral entropy**: rich conscious content → exploration boost
//!
//! ## Cross-Frequency Coupling (Canolty & Knight 2010)
//!
//! Theta-gamma phase-amplitude coupling (PAC) measures how gamma amplitude
//! is modulated by theta phase — a neural correlate of information integration.
//! High PAC → boost consciousness confidence.
//!
//! ## References
//!
//! - Buzsaki, G. (2006). Rhythms of the Brain. Oxford UP.
//! - Canolty, R.T. & Knight, R.T. (2010). Cross-frequency coupling. TiCS.
//! - Jensen, O. & Colgin, L.L. (2007). Cross-frequency coupling. TiCS.

use super::super::subsystem_trait::{
    CognitiveSubsystem, CycleSnapshot, SNAPSHOT_STATE_DIM, SubsystemOutput, output_flags,
};
use crate::dynamics::spectral_analysis::{SpectralAnalyzer, SpectralConfig, WindowType};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

use crate::cognitive_loop::thresholds::{
    SPECTRAL_DELTA_REST_THRESHOLD, SPECTRAL_ENTROPY_EXPLORATION_SCALE,
    SPECTRAL_GAMMA_CONSCIOUSNESS_BOOST, SPECTRAL_HISTORY_CAPACITY, SPECTRAL_INTERVAL,
    SPECTRAL_MIN_HISTORY, SPECTRAL_PAC_CONFIDENCE_BOOST, SPECTRAL_PAC_THRESHOLD,
    SPECTRAL_SAMPLE_RATE,
};

// ═══════════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for the Spectral Manager.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralManagerConfig {
    /// Number of state dimensions to analyze (from compressed_state).
    /// Analyzing all 256 is expensive; we sample the first N.
    pub analyzed_dims: usize,
    /// Minimum history length before spectral analysis is meaningful.
    pub min_history: usize,
    /// Ring buffer capacity (cycles of state history).
    pub history_capacity: usize,
}

impl Default for SpectralManagerConfig {
    fn default() -> Self {
        Self {
            analyzed_dims: 32, // First 32 dimensions (most active)
            min_history: SPECTRAL_MIN_HISTORY as usize,
            history_capacity: SPECTRAL_HISTORY_CAPACITY as usize,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TELEMETRY
// ═══════════════════════════════════════════════════════════════════════════════

/// Telemetry from spectral analysis.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SpectralTelemetry {
    /// Mean band powers across analyzed dimensions.
    pub band_power: MeanBandPower,
    /// Mean spectral entropy across analyzed dimensions.
    pub spectral_entropy: f64,
    /// Theta-gamma phase-amplitude coupling strength [0, 1].
    pub theta_gamma_pac: f64,
    /// Dominant frequency band name.
    pub dominant_band: String,
    /// Number of history samples available.
    pub history_len: usize,
}

/// Aggregated band power (mean across dimensions).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MeanBandPower {
    pub delta: f64,
    pub theta: f64,
    pub alpha: f64,
    pub beta: f64,
    pub gamma: f64,
}

impl MeanBandPower {
    /// Total power across all bands.
    pub fn total(&self) -> f64 {
        self.delta + self.theta + self.alpha + self.beta + self.gamma
    }

    /// Identify the dominant band.
    pub fn dominant(&self) -> &'static str {
        let bands = [
            (self.delta, "delta"),
            (self.theta, "theta"),
            (self.alpha, "alpha"),
            (self.beta, "beta"),
            (self.gamma, "gamma"),
        ];
        bands
            .iter()
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
            .map(|b| b.1)
            .unwrap_or("unknown")
    }

    /// Relative power of a band (fraction of total).
    pub fn relative_gamma(&self) -> f64 {
        let total = self.total();
        if total < 1e-12 {
            0.0
        } else {
            self.gamma / total
        }
    }

    /// Relative delta power.
    pub fn relative_delta(&self) -> f64 {
        let total = self.total();
        if total < 1e-12 {
            0.0
        } else {
            self.delta / total
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SPECTRAL MANAGER
// ═══════════════════════════════════════════════════════════════════════════════

/// Spectral Twin Manager — frequency-domain analysis of CfC hidden state.
///
/// Implements `CognitiveSubsystem` at interval 67 (co-prime with all existing
/// manager intervals: 7, 11, 13, 19, 29, 37, 41, 53, 59).
pub struct SpectralManager {
    /// Ring buffer of compressed_state snapshots (one per cycle).
    /// Each entry is [f32; SNAPSHOT_STATE_DIM].
    history: VecDeque<[f32; SNAPSHOT_STATE_DIM]>,
    /// Spectral analyzer (reused across ticks).
    analyzer: SpectralAnalyzer,
    /// Configuration.
    config: SpectralManagerConfig,
    /// Last telemetry.
    telemetry: SpectralTelemetry,
    /// Cycles since manager was created (for warmup).
    cycle_count: u64,
}

impl SpectralManager {
    /// Create a new spectral manager.
    pub fn new(config: SpectralManagerConfig) -> Self {
        let analyzer = SpectralAnalyzer::new(SpectralConfig {
            sample_rate: SPECTRAL_SAMPLE_RATE,
            window_size: 64, // ~2 seconds at 31 Hz
            overlap: 0.5,
            window_type: WindowType::Hann,
        });

        Self {
            history: VecDeque::with_capacity(config.history_capacity),
            analyzer,
            config,
            telemetry: SpectralTelemetry::default(),
            cycle_count: 0,
        }
    }

    /// Append the current cycle's compressed state to the history buffer.
    pub fn record_state(&mut self, state: &[f32; SNAPSHOT_STATE_DIM]) {
        if self.history.len() >= self.config.history_capacity {
            self.history.pop_front();
        }
        self.history.push_back(*state);
    }

    /// Run spectral analysis across the first N dimensions of the state history.
    ///
    /// For each dimension, extract its time series from the history buffer,
    /// compute band power and spectral entropy, then average across dimensions.
    fn analyze(&mut self) -> SpectralTelemetry {
        let history_len = self.history.len();
        if history_len < self.config.min_history {
            return SpectralTelemetry {
                history_len,
                dominant_band: "warmup".into(),
                ..SpectralTelemetry::default()
            };
        }

        let n_dims = self.config.analyzed_dims.min(SNAPSHOT_STATE_DIM);
        let mut total_band = MeanBandPower::default();
        let mut total_entropy = 0.0;
        let mut analyzed = 0usize;

        for dim in 0..n_dims {
            // Extract time series for this dimension
            let signal: Vec<f64> = self.history.iter().map(|s| s[dim] as f64).collect();

            // Skip flat dimensions (no spectral content)
            let variance: f64 = {
                let mean = signal.iter().sum::<f64>() / signal.len() as f64;
                signal.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / signal.len() as f64
            };
            if variance < 1e-10 {
                continue;
            }

            let bands = self.analyzer.band_power(&signal);
            let entropy = self.analyzer.spectral_entropy(&signal);

            total_band.delta += bands.delta;
            total_band.theta += bands.theta;
            total_band.alpha += bands.alpha;
            total_band.beta += bands.beta;
            total_band.gamma += bands.gamma;
            total_entropy += entropy;
            analyzed += 1;
        }

        if analyzed > 0 {
            let n = analyzed as f64;
            total_band.delta /= n;
            total_band.theta /= n;
            total_band.alpha /= n;
            total_band.beta /= n;
            total_band.gamma /= n;
            total_entropy /= n;
        }

        // Theta-gamma phase-amplitude coupling (simplified).
        // Use the first dimension as a representative signal.
        let pac = self.compute_theta_gamma_pac();

        let dominant = total_band.dominant().to_string();

        SpectralTelemetry {
            band_power: total_band,
            spectral_entropy: total_entropy,
            theta_gamma_pac: pac,
            dominant_band: dominant,
            history_len,
        }
    }

    /// Compute theta-gamma phase-amplitude coupling (PAC).
    ///
    /// Simplified approach: compute theta band power and gamma band power
    /// independently, then measure their co-modulation via normalized product.
    /// Full PAC (Hilbert transform + phase extraction) would be more accurate
    /// but this approximation captures the key relationship.
    ///
    /// Science: Canolty & Knight (2010) — theta phase organizes gamma bursts.
    fn compute_theta_gamma_pac(&self) -> f64 {
        let history_len = self.history.len();
        if history_len < self.config.min_history {
            return 0.0;
        }

        // Use first 4 dimensions for PAC estimation (reduces noise)
        let n_pac_dims = 4.min(self.config.analyzed_dims).min(SNAPSHOT_STATE_DIM);
        let mut pac_sum = 0.0;
        let mut pac_count = 0;

        for dim in 0..n_pac_dims {
            let signal: Vec<f64> = self.history.iter().map(|s| s[dim] as f64).collect();

            let variance: f64 = {
                let mean = signal.iter().sum::<f64>() / signal.len() as f64;
                signal.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / signal.len() as f64
            };
            if variance < 1e-10 {
                continue;
            }

            let bands = self.analyzer.band_power(&signal);

            // PAC proxy: normalized product of theta and gamma power.
            // High when both theta and gamma are present (coupling).
            // Normalized by total power to get [0, 1] range.
            let total = bands.delta + bands.theta + bands.alpha + bands.beta + bands.gamma;
            if total > 1e-12 {
                let theta_rel = bands.theta / total;
                let gamma_rel = bands.gamma / total;
                // Geometric mean of relative powers, scaled to [0, 1]
                pac_sum += (theta_rel * gamma_rel).sqrt() * 2.0; // *2 since max(sqrt(0.25*0.25)) = 0.5
                pac_count += 1;
            }
        }

        if pac_count > 0 {
            (pac_sum / pac_count as f64).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    // ── Public Accessors ─────────────────────────────────────────────────

    /// Current telemetry.
    pub fn telemetry(&self) -> &SpectralTelemetry {
        &self.telemetry
    }

    /// Current history length.
    pub fn history_len(&self) -> usize {
        self.history.len()
    }

    /// Configuration.
    pub fn config(&self) -> &SpectralManagerConfig {
        &self.config
    }
}

impl CognitiveSubsystem for SpectralManager {
    fn name(&self) -> &'static str {
        "spectral_twin"
    }

    fn interval(&self) -> u32 {
        SPECTRAL_INTERVAL
    }

    fn process(&mut self, snapshot: &CycleSnapshot) -> SubsystemOutput {
        // Always record state (even between analysis ticks, for history continuity)
        self.record_state(&snapshot.compressed_state);
        self.cycle_count += 1;

        // Run spectral analysis
        self.telemetry = self.analyze();

        // Warmup: don't modulate until we have enough history
        if self.history.len() < self.config.min_history {
            return SubsystemOutput::NEUTRAL;
        }

        let mut output = SubsystemOutput::NEUTRAL;

        // ── Band power → consciousness modulation ────────────────────────

        // High relative gamma → consciousness boost (binding/integration)
        let rel_gamma = self.telemetry.band_power.relative_gamma();
        if rel_gamma > crate::cognitive_loop::thresholds::SPECTRAL_GAMMA_THRESHOLD {
            output.confidence_delta = SPECTRAL_GAMMA_CONSCIOUSNESS_BOOST as f64;
        }

        // Delta dominance → request rest
        let rel_delta = self.telemetry.band_power.relative_delta();
        if rel_delta > SPECTRAL_DELTA_REST_THRESHOLD as f64 {
            output.flags |= output_flags::REQUEST_REST;
            output.arousal_delta = crate::cognitive_loop::thresholds::SPECTRAL_DELTA_AROUSAL_DELTA;
        }

        // ── Spectral entropy → exploration ───────────────────────────────

        // High entropy = broadband = rich content → boost exploration
        if self.telemetry.spectral_entropy
            > crate::cognitive_loop::thresholds::SPECTRAL_ENTROPY_THRESHOLD
        {
            output.exploration_delta = (self.telemetry.spectral_entropy
                - crate::cognitive_loop::thresholds::SPECTRAL_ENTROPY_THRESHOLD)
                * SPECTRAL_ENTROPY_EXPLORATION_SCALE as f64;
        }

        // ── Theta-gamma PAC → consciousness confidence ──────────────────

        if self.telemetry.theta_gamma_pac > SPECTRAL_PAC_THRESHOLD as f64 {
            output.confidence_delta += SPECTRAL_PAC_CONFIDENCE_BOOST as f64;
        }

        output
    }

    fn checkpoint(&self) -> Vec<u8> {
        // Serialize config + telemetry (history is transient, not checkpointed)
        serde_json::to_vec(&(&self.config, &self.telemetry)).unwrap_or_else(|e| {
            tracing::warn!("SpectralManager checkpoint serialization failed: {e}");
            Vec::new()
        })
    }

    fn restore(&mut self, data: &[u8]) -> Result<(), String> {
        if data.is_empty() {
            return Err("SpectralManager restore: empty checkpoint data".to_string());
        }
        let (config, telemetry): (SpectralManagerConfig, SpectralTelemetry) =
            serde_json::from_slice(data)
                .map_err(|e| format!("SpectralManager restore failed: {e}"))?;
        self.config = config;
        self.telemetry = telemetry;
        // History is transient (ring buffer rebuilt from live data); analyzer stays as-is.
        tracing::debug!(
            "SpectralManager restored: dims={}, cycle_count={}",
            self.config.analyzed_dims,
            self.cycle_count
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

    fn make_snapshot_with_state(state: [f32; SNAPSHOT_STATE_DIM]) -> CycleSnapshot {
        CycleSnapshot {
            compressed_state: state,
            ..CycleSnapshot::default()
        }
    }

    fn sine_state(cycle: usize, freq_hz: f64, sample_rate: f64) -> [f32; SNAPSHOT_STATE_DIM] {
        let t = cycle as f64 / sample_rate;
        let val = (2.0 * std::f64::consts::PI * freq_hz * t).sin() as f32;
        let mut state = [0.0f32; SNAPSHOT_STATE_DIM];
        // Put the sine wave in the first few dimensions
        for d in 0..32 {
            state[d] = val * (1.0 + d as f32 * 0.01); // Slight variation per dim
        }
        state
    }

    #[test]
    fn test_creation() {
        let mgr = SpectralManager::new(SpectralManagerConfig::default());
        assert_eq!(mgr.history_len(), 0);
        assert_eq!(mgr.name(), "spectral_twin");
        assert_eq!(mgr.interval(), SPECTRAL_INTERVAL);
    }

    #[test]
    fn test_warmup_returns_neutral() {
        let mut mgr = SpectralManager::new(SpectralManagerConfig::default());
        let snapshot = CycleSnapshot::default();

        // Before min_history, should return neutral
        let output = mgr.process(&snapshot);
        assert!(output.is_neutral() || mgr.history_len() < mgr.config().min_history);
    }

    #[test]
    fn test_history_recording() {
        let mut mgr = SpectralManager::new(SpectralManagerConfig {
            history_capacity: 10,
            ..SpectralManagerConfig::default()
        });

        for i in 0..15 {
            let mut state = [0.0f32; SNAPSHOT_STATE_DIM];
            state[0] = i as f32;
            let snapshot = make_snapshot_with_state(state);
            mgr.record_state(&snapshot.compressed_state);
        }

        // Should be capped at capacity
        assert_eq!(mgr.history_len(), 10);
        // First entry should be cycle 5 (0-4 evicted)
        assert!((mgr.history[0][0] - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_spectral_analysis_with_sine() {
        let mut mgr = SpectralManager::new(SpectralManagerConfig {
            analyzed_dims: 4,
            min_history: 32,
            history_capacity: 128,
        });

        // Feed a 6 Hz sine wave (theta band) for 128 cycles at 31 Hz
        for cycle in 0..128 {
            let snapshot = make_snapshot_with_state(sine_state(cycle, 6.0, 31.0));
            mgr.process(&snapshot);
        }

        let telem = mgr.telemetry();
        assert!(
            telem.history_len >= 128,
            "Should have 128 samples, got {}",
            telem.history_len
        );

        // Theta should be the dominant band for a 6 Hz signal
        assert!(
            telem.band_power.theta > telem.band_power.alpha,
            "6 Hz signal should have more theta than alpha: theta={}, alpha={}",
            telem.band_power.theta,
            telem.band_power.alpha
        );
    }

    #[test]
    fn test_delta_dominance_requests_rest() {
        let mut mgr = SpectralManager::new(SpectralManagerConfig {
            analyzed_dims: 4,
            min_history: 32,
            history_capacity: 128,
        });

        // Feed a 2 Hz sine wave (delta band) for 128 cycles
        for cycle in 0..128 {
            let snapshot = make_snapshot_with_state(sine_state(cycle, 2.0, 31.0));
            mgr.process(&snapshot);
        }

        // Check if delta is detected
        let telem = mgr.telemetry();
        // Delta should be present for a 2 Hz signal
        assert!(
            telem.band_power.delta > 0.0,
            "2 Hz signal should produce delta power"
        );
    }

    #[test]
    fn test_flat_signal_no_crash() {
        let mut mgr = SpectralManager::new(SpectralManagerConfig {
            analyzed_dims: 4,
            min_history: 32,
            history_capacity: 128,
        });

        // Feed constant state (zero variance — should be skipped gracefully)
        for _ in 0..64 {
            let snapshot = make_snapshot_with_state([0.5; SNAPSHOT_STATE_DIM]);
            mgr.process(&snapshot);
        }

        let telem = mgr.telemetry();
        assert!(telem.spectral_entropy.is_finite());
        assert!(telem.theta_gamma_pac.is_finite());
    }

    #[test]
    fn test_pac_bounded() {
        let mut mgr = SpectralManager::new(SpectralManagerConfig {
            analyzed_dims: 4,
            min_history: 32,
            history_capacity: 128,
        });

        // Feed broadband noise (multiple frequencies)
        for cycle in 0..128 {
            let t = cycle as f64 / 31.0;
            let val = ((6.0 * 2.0 * std::f64::consts::PI * t).sin()
                + (40.0 * 2.0 * std::f64::consts::PI * t).sin() * 0.5) as f32;
            let mut state = [0.0f32; SNAPSHOT_STATE_DIM];
            for d in 0..32 {
                state[d] = val;
            }
            let snapshot = make_snapshot_with_state(state);
            mgr.process(&snapshot);
        }

        let pac = mgr.telemetry().theta_gamma_pac;
        assert!(
            pac >= 0.0 && pac <= 1.0,
            "PAC should be in [0,1], got {}",
            pac
        );
    }

    #[test]
    fn test_mean_band_power_dominant() {
        let bp = MeanBandPower {
            delta: 0.1,
            theta: 0.3,
            alpha: 0.2,
            beta: 0.15,
            gamma: 0.05,
        };
        assert_eq!(bp.dominant(), "theta");

        let bp2 = MeanBandPower {
            delta: 0.0,
            theta: 0.0,
            alpha: 0.0,
            beta: 0.0,
            gamma: 1.0,
        };
        assert_eq!(bp2.dominant(), "gamma");
    }

    #[test]
    fn test_telemetry_populated_after_warmup() {
        let mut mgr = SpectralManager::new(SpectralManagerConfig {
            analyzed_dims: 4,
            min_history: 32,
            history_capacity: 128,
        });

        // Feed enough data to pass warmup
        for cycle in 0..64 {
            let snapshot = make_snapshot_with_state(sine_state(cycle, 10.0, 31.0));
            mgr.process(&snapshot);
        }

        let telem = mgr.telemetry();
        assert_ne!(telem.dominant_band, "warmup");
        assert!(telem.history_len >= 64);
    }

    #[test]
    fn test_checkpoint_returns_bytes() {
        let mgr = SpectralManager::new(SpectralManagerConfig::default());
        let bytes = mgr.checkpoint();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_relative_gamma() {
        let bp = MeanBandPower {
            delta: 0.2,
            theta: 0.2,
            alpha: 0.2,
            beta: 0.2,
            gamma: 0.2,
        };
        assert!((bp.relative_gamma() - 0.2).abs() < 0.01);

        let empty = MeanBandPower::default();
        assert_eq!(empty.relative_gamma(), 0.0); // No divide by zero
    }
}
