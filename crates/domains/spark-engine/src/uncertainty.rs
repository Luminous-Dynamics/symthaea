// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Uncertainty Quantification
//!
//! Monte Carlo uncertainty propagation for LCF rate calculations.
//! Provides confidence intervals and sensitivity analysis.

use crate::physics::GamowIntegration;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

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
            temperature: (300.0, 10.0),  // 300 ± 10 K
            screening_ev: (309.0, 12.0), // 309 ± 12 eV (Raiola uncertainty)
            loading_ratio: (0.7, 0.05),  // 0.7 ± 0.05
            volume_cm3: (0.01, 0.002),   // 0.01 ± 0.002 cm³
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
            let temp = self
                .sample_normal(params.temperature.0, params.temperature.1)
                .max(1.0);
            let ue = self
                .sample_normal(params.screening_ev.0, params.screening_ev.1)
                .max(0.0);
            let loading = self
                .sample_normal(params.loading_ratio.0, params.loading_ratio.1)
                .clamp(0.0, 1.0);
            let volume = self
                .sample_normal(params.volume_cm3.0, params.volume_cm3.1)
                .max(1e-6);

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
        let u1: f64 = self.rng.r#gen();
        let u2: f64 = self.rng.r#gen();
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

// ============================================================================
// Tornado Diagram (Direction C Enhancement)
// ============================================================================

/// Entry in a tornado diagram showing parameter impact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TornadoEntry {
    /// Parameter name
    pub parameter: String,
    /// Output value when parameter is at low bound (5th percentile)
    pub output_low: f64,
    /// Output value when parameter is at high bound (95th percentile)
    pub output_high: f64,
    /// Total swing (|high - low|)
    pub swing: f64,
    /// Rank by importance (1 = most important)
    pub rank: usize,
    /// Direction of effect (+1 if high parameter → high output)
    pub direction: i8,
}

/// Tornado diagram for one-at-a-time sensitivity analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TornadoDiagram {
    /// Baseline output value
    pub baseline: f64,
    /// Entries sorted by swing (largest first)
    pub entries: Vec<TornadoEntry>,
    /// Total output range from all parameters
    pub total_range: f64,
}

impl TornadoDiagram {
    /// Build tornado diagram from parameter distributions.
    pub fn build(params: &ParameterDistributions) -> Self {
        let base_gamow = GamowIntegration::dd_rate(
            params.temperature.0,
            params.screening_ev.0,
            params.phonon_modes,
        );
        let base_n_d = params.loading_ratio.0 * 12.02 * 6.022e23 / 106.42;
        let baseline = base_gamow.to_neutron_rate(base_n_d, params.volume_cm3.0);

        let mut entries = Vec::new();

        // Temperature sensitivity
        let t_low = params.temperature.0 - 2.0 * params.temperature.1;
        let t_high = params.temperature.0 + 2.0 * params.temperature.1;
        let gamow_t_low =
            GamowIntegration::dd_rate(t_low.max(1.0), params.screening_ev.0, params.phonon_modes);
        let gamow_t_high =
            GamowIntegration::dd_rate(t_high, params.screening_ev.0, params.phonon_modes);
        let out_t_low = gamow_t_low.to_neutron_rate(base_n_d, params.volume_cm3.0);
        let out_t_high = gamow_t_high.to_neutron_rate(base_n_d, params.volume_cm3.0);
        entries.push(TornadoEntry {
            parameter: "Temperature".to_string(),
            output_low: out_t_low,
            output_high: out_t_high,
            swing: (out_t_high - out_t_low).abs(),
            rank: 0,
            direction: if out_t_high > out_t_low { 1 } else { -1 },
        });

        // Screening sensitivity
        let ue_low = (params.screening_ev.0 - 2.0 * params.screening_ev.1).max(0.0);
        let ue_high = params.screening_ev.0 + 2.0 * params.screening_ev.1;
        let gamow_ue_low =
            GamowIntegration::dd_rate(params.temperature.0, ue_low, params.phonon_modes);
        let gamow_ue_high =
            GamowIntegration::dd_rate(params.temperature.0, ue_high, params.phonon_modes);
        let out_ue_low = gamow_ue_low.to_neutron_rate(base_n_d, params.volume_cm3.0);
        let out_ue_high = gamow_ue_high.to_neutron_rate(base_n_d, params.volume_cm3.0);
        entries.push(TornadoEntry {
            parameter: "Screening".to_string(),
            output_low: out_ue_low,
            output_high: out_ue_high,
            swing: (out_ue_high - out_ue_low).abs(),
            rank: 0,
            direction: if out_ue_high > out_ue_low { 1 } else { -1 },
        });

        // Loading sensitivity
        let load_low = (params.loading_ratio.0 - 2.0 * params.loading_ratio.1).clamp(0.1, 1.0);
        let load_high = (params.loading_ratio.0 + 2.0 * params.loading_ratio.1).clamp(0.1, 1.0);
        let n_d_low = load_low * 12.02 * 6.022e23 / 106.42;
        let n_d_high = load_high * 12.02 * 6.022e23 / 106.42;
        let out_load_low = base_gamow.to_neutron_rate(n_d_low, params.volume_cm3.0);
        let out_load_high = base_gamow.to_neutron_rate(n_d_high, params.volume_cm3.0);
        entries.push(TornadoEntry {
            parameter: "Loading".to_string(),
            output_low: out_load_low,
            output_high: out_load_high,
            swing: (out_load_high - out_load_low).abs(),
            rank: 0,
            direction: if out_load_high > out_load_low { 1 } else { -1 },
        });

        // Volume sensitivity
        let vol_low = (params.volume_cm3.0 - 2.0 * params.volume_cm3.1).max(0.001);
        let vol_high = params.volume_cm3.0 + 2.0 * params.volume_cm3.1;
        let out_vol_low = base_gamow.to_neutron_rate(base_n_d, vol_low);
        let out_vol_high = base_gamow.to_neutron_rate(base_n_d, vol_high);
        entries.push(TornadoEntry {
            parameter: "Volume".to_string(),
            output_low: out_vol_low,
            output_high: out_vol_high,
            swing: (out_vol_high - out_vol_low).abs(),
            rank: 0,
            direction: if out_vol_high > out_vol_low { 1 } else { -1 },
        });

        // Sort by swing and assign ranks
        entries.sort_by(|a, b| {
            b.swing
                .partial_cmp(&a.swing)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for (i, entry) in entries.iter_mut().enumerate() {
            entry.rank = i + 1;
        }

        let total_range = entries.iter().map(|e| e.swing).sum();

        Self {
            baseline,
            entries,
            total_range,
        }
    }
}

// ============================================================================
// Morris Method (Direction C Enhancement)
// ============================================================================

/// Morris elementary effect for one parameter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MorrisEffect {
    /// Parameter name
    pub parameter: String,
    /// μ* - mean of absolute elementary effects (importance)
    pub mu_star: f64,
    /// σ - standard deviation of elementary effects (nonlinearity/interaction)
    pub sigma: f64,
    /// Rank by μ* (1 = most important)
    pub rank: usize,
    /// Interpretation
    pub interpretation: String,
}

/// Morris sensitivity analysis results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MorrisAnalysis {
    /// Effects for each parameter
    pub effects: Vec<MorrisEffect>,
    /// Number of trajectories used
    pub n_trajectories: usize,
    /// Total number of model evaluations
    pub n_evaluations: usize,
}

impl MonteCarloEngine {
    /// Run Morris method sensitivity analysis.
    ///
    /// Morris 1991 screening method: efficient global sensitivity analysis
    /// using r trajectories through the parameter space.
    pub fn morris_analysis(
        &mut self,
        params: &ParameterDistributions,
        n_trajectories: usize,
    ) -> MorrisAnalysis {
        let n_params = 4; // T, Ue, loading, volume
        let delta = 0.5; // Step size in [0,1] space

        let mut temp_effects: Vec<f64> = Vec::new();
        let mut ue_effects: Vec<f64> = Vec::new();
        let mut loading_effects: Vec<f64> = Vec::new();
        let mut volume_effects: Vec<f64> = Vec::new();

        for _ in 0..n_trajectories {
            // Random starting point in [0,1]^4
            let x0: Vec<f64> = (0..n_params).map(|_| self.rng.r#gen()).collect();

            // Transform to physical space
            let transform = |x: &[f64]| -> (f64, f64, f64, f64) {
                let t = params.temperature.0 + (x[0] - 0.5) * 4.0 * params.temperature.1;
                let ue = params.screening_ev.0 + (x[1] - 0.5) * 4.0 * params.screening_ev.1;
                let load = params.loading_ratio.0 + (x[2] - 0.5) * 4.0 * params.loading_ratio.1;
                let vol = params.volume_cm3.0 + (x[3] - 0.5) * 4.0 * params.volume_cm3.1;
                (
                    t.max(1.0),
                    ue.max(0.0),
                    load.clamp(0.1, 1.0),
                    vol.max(0.001),
                )
            };

            let evaluate = |t: f64, ue: f64, load: f64, vol: f64| -> f64 {
                let gamow = GamowIntegration::dd_rate(t, ue, params.phonon_modes);
                let n_d = load * 12.02 * 6.022e23 / 106.42;
                gamow.to_neutron_rate(n_d, vol).ln() // Log-space for better scaling
            };

            let (t0, ue0, l0, v0) = transform(&x0);
            let y0 = evaluate(t0, ue0, l0, v0);

            // Perturb each parameter
            for i in 0..n_params {
                let mut x1 = x0.clone();
                x1[i] = (x1[i] + delta).min(1.0);

                let (t1, ue1, l1, v1) = transform(&x1);
                let y1 = evaluate(t1, ue1, l1, v1);

                let effect = (y1 - y0) / delta;

                match i {
                    0 => temp_effects.push(effect),
                    1 => ue_effects.push(effect),
                    2 => loading_effects.push(effect),
                    3 => volume_effects.push(effect),
                    _ => {}
                }
            }
        }

        // Compute μ* and σ for each parameter
        let compute_morris = |effects: &[f64], name: &str| -> MorrisEffect {
            let n = effects.len() as f64;
            let mu_star = effects.iter().map(|e| e.abs()).sum::<f64>() / n;
            let mean = effects.iter().sum::<f64>() / n;
            let sigma = (effects.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / n).sqrt();

            let interpretation = if sigma > mu_star {
                "Nonlinear or involved in interactions".to_string()
            } else if mu_star > 1.0 {
                "Important linear effect".to_string()
            } else {
                "Negligible effect".to_string()
            };

            MorrisEffect {
                parameter: name.to_string(),
                mu_star,
                sigma,
                rank: 0,
                interpretation,
            }
        };

        let mut effects = vec![
            compute_morris(&temp_effects, "Temperature"),
            compute_morris(&ue_effects, "Screening"),
            compute_morris(&loading_effects, "Loading"),
            compute_morris(&volume_effects, "Volume"),
        ];

        // Sort and rank by μ*
        effects.sort_by(|a, b| {
            b.mu_star
                .partial_cmp(&a.mu_star)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for (i, effect) in effects.iter_mut().enumerate() {
            effect.rank = i + 1;
        }

        MorrisAnalysis {
            effects,
            n_trajectories,
            n_evaluations: n_trajectories * (n_params + 1),
        }
    }
}

// ============================================================================
// Feasibility Probability (Direction C Enhancement)
// ============================================================================

/// Feasibility probability from Monte Carlo analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeasibilityProbability {
    /// Probability of meeting dose constraint (<2.5 μSv/hr at surface)
    pub p_dose_safe: f64,
    /// Probability of meeting temperature constraint (<500°C)
    pub p_temp_safe: f64,
    /// Probability of meeting lifetime requirement (>5 years)
    pub p_lifetime_met: f64,
    /// Overall feasibility probability
    pub p_feasible: f64,
    /// Primary failure mode
    pub primary_failure_mode: String,
    /// Number of samples that passed all constraints
    pub n_passed: usize,
    /// Total samples
    pub n_total: usize,
}

impl FeasibilityProbability {
    /// Compute feasibility probability from Monte Carlo samples.
    ///
    /// Constraints:
    /// - Dose rate < 2.5 μSv/hr at 1m (public limit)
    /// - Core temperature < 500°C (PdD stability)
    /// - Lifetime > 5 years (economic)
    pub fn compute(neutron_rates: &[f64], power_densities: &[f64], lifetime_years: &[f64]) -> Self {
        let n = neutron_rates.len();
        if n == 0 {
            return Self {
                p_dose_safe: 0.0,
                p_temp_safe: 0.0,
                p_lifetime_met: 0.0,
                p_feasible: 0.0,
                primary_failure_mode: "No samples".to_string(),
                n_passed: 0,
                n_total: 0,
            };
        }

        // Dose constraint: 2.5 μSv/hr at 1m corresponds to ~10^6 n/s unshielded
        let dose_threshold = 1e6;
        let n_dose_ok = neutron_rates
            .iter()
            .filter(|&&r| r < dose_threshold)
            .count();
        let p_dose_safe = n_dose_ok as f64 / n as f64;

        // Temperature constraint: assume 0.1 K/mW power density rise
        let temp_rise_limit = 200.0; // K above ambient
        let power_threshold = temp_rise_limit / 0.1 * 1e-3; // W/cm³
        let n_temp_ok = power_densities
            .iter()
            .filter(|&&p| p < power_threshold)
            .count();
        let p_temp_safe = n_temp_ok as f64 / n as f64;

        // Lifetime constraint
        let lifetime_threshold = 5.0; // years
        let n_life_ok = lifetime_years
            .iter()
            .filter(|&&l| l > lifetime_threshold)
            .count();
        let p_lifetime_met = n_life_ok as f64 / n as f64;

        // Overall: must pass all
        let mut n_passed = 0;
        for i in 0..n {
            if neutron_rates[i] < dose_threshold
                && power_densities[i] < power_threshold
                && lifetime_years[i] > lifetime_threshold
            {
                n_passed += 1;
            }
        }
        let p_feasible = n_passed as f64 / n as f64;

        // Primary failure mode
        let failure_counts = [
            (1.0 - p_dose_safe, "Dose rate exceeds limit"),
            (1.0 - p_temp_safe, "Temperature exceeds limit"),
            (1.0 - p_lifetime_met, "Lifetime too short"),
        ];
        let primary_failure_mode = failure_counts
            .iter()
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(_, mode)| mode.to_string())
            .unwrap_or("None".to_string());

        Self {
            p_dose_safe,
            p_temp_safe,
            p_lifetime_met,
            p_feasible,
            primary_failure_mode,
            n_passed,
            n_total: n,
        }
    }
}

// ============================================================================
// Break-Even Analysis (Direction C Enhancement)
// ============================================================================

/// Break-even analysis result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreakEvenResult {
    /// Target LCOE ($/MWh)
    pub target_lcoe_usd_mwh: f64,
    /// Current LCOE ($/MWh)
    pub current_lcoe_usd_mwh: f64,
    /// Capital cost break-even ($/W)
    pub capital_breakeven_usd_w: f64,
    /// Capacity factor break-even
    pub capacity_factor_breakeven: f64,
    /// Lifetime break-even (years)
    pub lifetime_breakeven_years: f64,
    /// O&M cost break-even ($/W/year)
    pub om_breakeven_usd_w_year: f64,
    /// Improvement factor needed for capital cost
    pub capital_improvement_needed: f64,
    /// Whether break-even is achievable (<10× improvement)
    pub achievable: bool,
    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Break-even analyzer for economic feasibility.
pub struct BreakEvenAnalyzer;

impl BreakEvenAnalyzer {
    /// Compute break-even requirements.
    ///
    /// Uses simplified LCOE model:
    /// LCOE = (capital × CRF + O&M) / (capacity_factor × 8760 hours) + fuel
    ///
    /// CRF = Capital Recovery Factor = r(1+r)^n / ((1+r)^n - 1)
    pub fn analyze(
        capital_usd_w: f64,
        om_usd_w_year: f64,
        fuel_usd_mwh: f64,
        capacity_factor: f64,
        lifetime_years: f64,
        target_lcoe_usd_mwh: f64,
    ) -> BreakEvenResult {
        let discount_rate = 0.08; // 8% discount rate

        // Capital recovery factor
        let crf = |n: f64| -> f64 {
            let r: f64 = discount_rate;
            r * (1.0_f64 + r).powf(n) / ((1.0_f64 + r).powf(n) - 1.0)
        };

        // LCOE calculation
        let lcoe = |cap: f64, om: f64, cf: f64, life: f64| -> f64 {
            let annual_generation_mwh = cf * 8760.0 / 1e6; // per W capacity
            if annual_generation_mwh <= 0.0 {
                return f64::INFINITY;
            }
            (cap * crf(life) + om) / annual_generation_mwh + fuel_usd_mwh
        };

        let current_lcoe = lcoe(
            capital_usd_w,
            om_usd_w_year,
            capacity_factor,
            lifetime_years,
        );

        // Binary search for break-even capital cost
        let capital_breakeven = Self::binary_search(0.0, capital_usd_w * 100.0, |cap| {
            lcoe(cap, om_usd_w_year, capacity_factor, lifetime_years) - target_lcoe_usd_mwh
        });

        // Binary search for break-even capacity factor
        let cf_breakeven = Self::binary_search(0.01, 1.0, |cf| {
            lcoe(capital_usd_w, om_usd_w_year, cf, lifetime_years) - target_lcoe_usd_mwh
        });

        // Binary search for break-even lifetime
        let life_breakeven = Self::binary_search(1.0, 100.0, |life| {
            lcoe(capital_usd_w, om_usd_w_year, capacity_factor, life) - target_lcoe_usd_mwh
        });

        // Binary search for break-even O&M
        let om_breakeven = Self::binary_search(0.0, om_usd_w_year * 100.0, |om| {
            lcoe(capital_usd_w, om, capacity_factor, lifetime_years) - target_lcoe_usd_mwh
        });

        let capital_improvement_needed = if capital_breakeven > 0.0 {
            capital_usd_w / capital_breakeven
        } else {
            f64::INFINITY
        };

        let achievable =
            capital_improvement_needed < 10.0 && cf_breakeven < 0.95 && life_breakeven < 50.0;

        let mut recommendations = Vec::new();
        if capital_improvement_needed > 1.0 {
            recommendations.push(format!(
                "Reduce capital cost by {:.0}× (from ${:.0}/W to ${:.2}/W)",
                capital_improvement_needed, capital_usd_w, capital_breakeven
            ));
        }
        if cf_breakeven > capacity_factor {
            recommendations.push(format!(
                "Increase capacity factor from {:.0}% to {:.0}%",
                capacity_factor * 100.0,
                cf_breakeven * 100.0
            ));
        }
        if life_breakeven > lifetime_years {
            recommendations.push(format!(
                "Extend lifetime from {:.0} to {:.0} years",
                lifetime_years, life_breakeven
            ));
        }

        BreakEvenResult {
            target_lcoe_usd_mwh,
            current_lcoe_usd_mwh: current_lcoe,
            capital_breakeven_usd_w: capital_breakeven,
            capacity_factor_breakeven: cf_breakeven,
            lifetime_breakeven_years: life_breakeven,
            om_breakeven_usd_w_year: om_breakeven,
            capital_improvement_needed,
            achievable,
            recommendations,
        }
    }

    /// Binary search for zero crossing.
    fn binary_search<F: Fn(f64) -> f64>(mut low: f64, mut high: f64, f: F) -> f64 {
        for _ in 0..50 {
            let mid = (low + high) / 2.0;
            let val = f(mid);
            if val.abs() < 1e-6 {
                return mid;
            }
            if val > 0.0 {
                high = mid;
            } else {
                low = mid;
            }
        }
        (low + high) / 2.0
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

    #[test]
    fn test_tornado_diagram() {
        let params = ParameterDistributions::default();
        let tornado = TornadoDiagram::build(&params);

        // Should have 4 parameters
        assert_eq!(tornado.entries.len(), 4);

        // Should be sorted by swing (rank 1 is largest swing)
        assert_eq!(tornado.entries[0].rank, 1);

        // Swings should be in descending order
        for i in 1..tornado.entries.len() {
            assert!(tornado.entries[i - 1].swing >= tornado.entries[i].swing);
        }
    }

    #[test]
    fn test_morris_analysis() {
        let mut engine = MonteCarloEngine::new(42);
        let params = ParameterDistributions::default();
        let morris = engine.morris_analysis(&params, 10);

        // Should have 4 parameters
        assert_eq!(morris.effects.len(), 4);

        // μ* should be non-negative
        for effect in &morris.effects {
            assert!(effect.mu_star >= 0.0);
            assert!(effect.sigma >= 0.0);
        }

        // Should have used n_trajectories * (n_params + 1) evaluations
        assert_eq!(morris.n_evaluations, 10 * 5);
    }

    #[test]
    fn test_feasibility_probability() {
        let neutron_rates = vec![1e3, 1e4, 1e5, 1e6, 1e7];
        let power_densities = vec![1e-10, 1e-10, 1e-10, 1e-10, 1e-10];
        let lifetimes = vec![10.0, 10.0, 10.0, 10.0, 10.0];

        let feasibility =
            FeasibilityProbability::compute(&neutron_rates, &power_densities, &lifetimes);

        assert!(feasibility.p_dose_safe > 0.0);
        assert!(feasibility.p_temp_safe > 0.0);
        assert!(feasibility.p_lifetime_met > 0.0);
        assert!(feasibility.n_total == 5);
    }

    #[test]
    fn test_break_even_analysis() {
        let result = BreakEvenAnalyzer::analyze(
            10.0, // $10/W capital (like solar)
            0.02, // $0.02/W/year O&M
            0.0,  // No fuel cost
            0.25, // 25% capacity factor
            25.0, // 25 year lifetime
            50.0, // Target $50/MWh LCOE
        );

        assert!(result.current_lcoe_usd_mwh > 0.0);
        assert!(result.capital_breakeven_usd_w > 0.0);
        assert!(result.capacity_factor_breakeven > 0.0);
        assert!(result.capacity_factor_breakeven <= 1.0);
    }
}
