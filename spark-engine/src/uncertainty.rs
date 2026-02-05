//! # Uncertainty Quantification
//!
//! Monte Carlo uncertainty propagation for LCF rate calculations.
//! Provides confidence intervals and sensitivity analysis.

use crate::physics::GamowIntegration;
use crate::constants::*;
use serde::{Deserialize, Serialize};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

/// Parameter distributions for Monte Carlo.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterDistributions {
    /// Temperature: mean and std dev (K)
    pub temperature: (f64, f64),
    /// Screening energy: mean and std dev (eV)
    pub screening_ev: (f64, f64),
    /// Loading ratio: mean and std dev
    pub loading_ratio: (f64, f64),
    /// Volume: mean and std dev (cm³)
    pub volume_cm3: (f64, f64),
    /// Number of phonon modes: mean (discrete)
    pub phonon_modes: u32,
}

impl Default for ParameterDistributions {
    fn default() -> Self {
        Self {
            temperature: (300.0, 10.0),         // 300 ± 10 K
            screening_ev: (309.0, 12.0),        // 309 ± 12 eV (Raiola uncertainty)
            loading_ratio: (0.7, 0.05),         // 0.7 ± 0.05
            volume_cm3: (0.01, 0.002),          // 0.01 ± 0.002 cm³
            phonon_modes: 0,
        }
    }
}

/// Monte Carlo simulation results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonteCarloResults {
    /// Number of samples
    pub n_samples: usize,
    /// Reaction rate statistics
    pub sigma_v: RateStatistics,
    /// Neutron rate statistics (for given n_d, volume)
    pub neutron_rate: RateStatistics,
    /// Sensitivity indices (normalized)
    pub sensitivities: Sensitivities,
}

/// Statistics for a rate calculation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateStatistics {
    /// Mean value
    pub mean: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Coefficient of variation (std/mean)
    pub cv: f64,
    /// 5th percentile
    pub p5: f64,
    /// 50th percentile (median)
    pub p50: f64,
    /// 95th percentile
    pub p95: f64,
    /// Minimum sample
    pub min: f64,
    /// Maximum sample
    pub max: f64,
}

/// Sensitivity indices for input parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sensitivities {
    /// Temperature sensitivity (d(log rate)/d(log T))
    pub temperature: f64,
    /// Screening sensitivity (d(log rate)/d(log Ue))
    pub screening: f64,
    /// Loading sensitivity (d(log rate)/d(log loading))
    pub loading: f64,
    /// Volume sensitivity (should be ~2 since rate ∝ n²V)
    pub volume: f64,
    /// Dominant parameter
    pub dominant: String,
}

/// Monte Carlo engine for uncertainty propagation.
pub struct MonteCarloEngine {
    rng: ChaCha8Rng,
}

impl MonteCarloEngine {
    /// Create new engine with seed.
    pub fn new(seed: u64) -> Self {
        Self {
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    /// Run Monte Carlo simulation.
    pub fn run(&mut self, params: &ParameterDistributions, n_samples: usize) -> MonteCarloResults {
        let mut sigma_v_samples = Vec::with_capacity(n_samples);
        let mut neutron_samples = Vec::with_capacity(n_samples);

        for _ in 0..n_samples {
            // Sample parameters from distributions
            let temp = self.sample_normal(params.temperature.0, params.temperature.1).max(1.0);
            let ue = self.sample_normal(params.screening_ev.0, params.screening_ev.1).max(0.0);
            let loading = self.sample_normal(params.loading_ratio.0, params.loading_ratio.1).clamp(0.0, 1.0);
            let volume = self.sample_normal(params.volume_cm3.0, params.volume_cm3.1).max(1e-6);

            // Calculate rate
            let gamow = GamowIntegration::dd_rate(temp, ue, params.phonon_modes);
            sigma_v_samples.push(gamow.sigma_v_cm3_s);

            // Calculate neutron rate
            let n_d = loading * 12.02 * 6.022e23 / 106.42; // D atoms/cm³
            let neutron_rate = gamow.to_neutron_rate(n_d, volume);
            neutron_samples.push(neutron_rate);
        }

        // Compute statistics
        let sigma_v = Self::compute_stats(&sigma_v_samples);
        let neutron_rate = Self::compute_stats(&neutron_samples);

        // Compute sensitivities via finite difference
        let sensitivities = self.compute_sensitivities(params);

        MonteCarloResults {
            n_samples,
            sigma_v,
            neutron_rate,
            sensitivities,
        }
    }

    /// Sample from normal distribution.
    fn sample_normal(&mut self, mean: f64, std: f64) -> f64 {
        // Box-Muller transform
        let u1: f64 = self.rng.gen();
        let u2: f64 = self.rng.gen();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        mean + std * z
    }

    /// Compute statistics from samples.
    fn compute_stats(samples: &[f64]) -> RateStatistics {
        let n = samples.len() as f64;
        let mean = samples.iter().sum::<f64>() / n;
        let variance = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();

        let mut sorted = samples.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let p5_idx = (0.05 * n) as usize;
        let p50_idx = (0.50 * n) as usize;
        let p95_idx = (0.95 * n) as usize;

        RateStatistics {
            mean,
            std_dev,
            cv: if mean > 0.0 { std_dev / mean } else { 0.0 },
            p5: sorted.get(p5_idx).copied().unwrap_or(0.0),
            p50: sorted.get(p50_idx).copied().unwrap_or(mean),
            p95: sorted.get(p95_idx).copied().unwrap_or(0.0),
            min: sorted.first().copied().unwrap_or(0.0),
            max: sorted.last().copied().unwrap_or(0.0),
        }
    }

    /// Compute sensitivity indices via finite difference.
    fn compute_sensitivities(&self, params: &ParameterDistributions) -> Sensitivities {
        let delta = 0.01; // 1% perturbation

        // Base case
        let base = GamowIntegration::dd_rate(
            params.temperature.0,
            params.screening_ev.0,
            params.phonon_modes,
        );
        let base_n_d = params.loading_ratio.0 * 12.02 * 6.022e23 / 106.42;
        let base_rate = base.to_neutron_rate(base_n_d, params.volume_cm3.0);

        // Temperature sensitivity
        let temp_high = GamowIntegration::dd_rate(
            params.temperature.0 * (1.0 + delta),
            params.screening_ev.0,
            params.phonon_modes,
        );
        let temp_rate = temp_high.to_neutron_rate(base_n_d, params.volume_cm3.0);
        let temp_sens = if base_rate > 0.0 {
            ((temp_rate / base_rate).ln() / delta).abs()
        } else {
            0.0
        };

        // Screening sensitivity
        let ue_high = GamowIntegration::dd_rate(
            params.temperature.0,
            params.screening_ev.0 * (1.0 + delta),
            params.phonon_modes,
        );
        let ue_rate = ue_high.to_neutron_rate(base_n_d, params.volume_cm3.0);
        let ue_sens = if base_rate > 0.0 {
            ((ue_rate / base_rate).ln() / delta).abs()
        } else {
            0.0
        };

        // Loading sensitivity (rate ∝ n² so expect ~2)
        let loading_high = params.loading_ratio.0 * (1.0 + delta);
        let n_d_high = loading_high * 12.02 * 6.022e23 / 106.42;
        let loading_rate = base.to_neutron_rate(n_d_high, params.volume_cm3.0);
        let loading_sens = if base_rate > 0.0 {
            ((loading_rate / base_rate).ln() / delta).abs()
        } else {
            0.0
        };

        // Volume sensitivity (rate ∝ V so expect ~1)
        let vol_high = params.volume_cm3.0 * (1.0 + delta);
        let vol_rate = base.to_neutron_rate(base_n_d, vol_high);
        let vol_sens = if base_rate > 0.0 {
            ((vol_rate / base_rate).ln() / delta).abs()
        } else {
            0.0
        };

        // Find dominant
        let max_sens = temp_sens.max(ue_sens).max(loading_sens).max(vol_sens);
        let dominant = if (temp_sens - max_sens).abs() < 0.01 {
            "temperature".to_string()
        } else if (ue_sens - max_sens).abs() < 0.01 {
            "screening".to_string()
        } else if (loading_sens - max_sens).abs() < 0.01 {
            "loading".to_string()
        } else {
            "volume".to_string()
        };

        Sensitivities {
            temperature: temp_sens,
            screening: ue_sens,
            loading: loading_sens,
            volume: vol_sens,
            dominant,
        }
    }
}

/// Confidence interval for a measurement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceInterval {
    /// Point estimate
    pub estimate: f64,
    /// Lower bound (95% CI)
    pub lower: f64,
    /// Upper bound (95% CI)
    pub upper: f64,
    /// Confidence level (e.g., 0.95)
    pub confidence: f64,
}

impl ConfidenceInterval {
    /// Create from Monte Carlo results.
    pub fn from_mc(stats: &RateStatistics) -> Self {
        Self {
            estimate: stats.mean,
            lower: stats.p5,
            upper: stats.p95,
            confidence: 0.90, // p5 to p95 is 90% CI
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monte_carlo() {
        let mut engine = MonteCarloEngine::new(42);
        let params = ParameterDistributions::default();
        let results = engine.run(&params, 100);

        assert_eq!(results.n_samples, 100);
        assert!(results.sigma_v.mean > 0.0);
        assert!(results.sigma_v.p5 <= results.sigma_v.p50);
        assert!(results.sigma_v.p50 <= results.sigma_v.p95);
    }

    #[test]
    fn test_sensitivities() {
        let mut engine = MonteCarloEngine::new(42);
        let params = ParameterDistributions::default();
        let results = engine.run(&params, 100);

        // Temperature should have very high sensitivity at low T
        assert!(results.sensitivities.temperature > 0.0);
        // Loading should be ~2 (rate ∝ n²)
        assert!(results.sensitivities.loading > 1.5);
        assert!(results.sensitivities.loading < 2.5);
    }

    #[test]
    fn test_statistics() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = MonteCarloEngine::compute_stats(&samples);
        assert!((stats.mean - 3.0).abs() < 0.01);
        assert!(stats.min <= stats.p5);
        assert!(stats.p95 <= stats.max);
    }
}
