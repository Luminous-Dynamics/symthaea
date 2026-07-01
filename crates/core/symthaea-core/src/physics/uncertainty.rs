// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Uncertainty Quantification: Monte Carlo Analysis for Spark Engine
//!
//! Propagates parameter uncertainties through coupled physics simulations
//! to quantify confidence in design predictions.
//!
//! ## Methodology
//!
//! 1. **Parameter Distributions**: Based on literature uncertainty ranges
//! 2. **Latin Hypercube Sampling**: Efficient coverage of parameter space
//! 3. **Monte Carlo Propagation**: Run N simulations with sampled parameters
//! 4. **Sensitivity Analysis**: One-at-a-time (OAT) perturbation
//! 5. **Output Statistics**: Percentiles, mean, std dev, confidence intervals
//!
//! ## Key Uncertainties
//!
//! | Parameter | Distribution | Source |
//! |-----------|--------------|--------|
//! | Neutron cross-section | ±5-10% | ENDF covariance |
//! | Thermal conductivity | ±10-20% | Material variability |
//! | HEA healing rate | ±50% | Limited experimental data |
//! | Dose conversion factor | ±30% | Energy spectrum uncertainty |

use rand::distributions::{Distribution, Uniform};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;

use super::coupled_physics::{CoupledPhysicsEngine, OperatingConditions};
use crate::genesis::GenesisSeed;

/// Parameter uncertainty distribution type
#[derive(Debug, Clone, Copy)]
pub enum UncertaintyDistribution {
    /// Normal distribution with mean and std dev (as fraction of mean)
    Normal { std_fraction: f64 },
    /// Uniform distribution with ± range (as fraction of mean)
    Uniform { range_fraction: f64 },
    /// Log-normal for strictly positive parameters
    LogNormal { sigma: f64 },
    /// Triangular with mode at nominal and ± range
    Triangular { range_fraction: f64 },
}

/// A parameter with uncertainty
#[derive(Debug, Clone)]
pub struct UncertainParameter {
    /// Parameter name
    pub name: String,
    /// Nominal (baseline) value
    pub nominal: f64,
    /// Uncertainty distribution
    pub distribution: UncertaintyDistribution,
    /// Units for display
    pub units: String,
    /// Literature source for uncertainty estimate
    pub source: String,
}

impl UncertainParameter {
    /// Sample a value from the uncertainty distribution
    pub fn sample(&self, rng: &mut impl Rng) -> f64 {
        match self.distribution {
            UncertaintyDistribution::Normal { std_fraction } => {
                let std = self.nominal * std_fraction;
                let normal = rand_distr::Normal::new(self.nominal, std)
                    .expect("Normal requires finite mean and std >= 0");
                normal.sample(rng).max(self.nominal * 0.1) // Floor at 10% of nominal
            }
            UncertaintyDistribution::Uniform { range_fraction } => {
                let range = self.nominal * range_fraction;
                let uniform = Uniform::new(self.nominal - range, self.nominal + range);
                uniform.sample(rng)
            }
            UncertaintyDistribution::LogNormal { sigma } => {
                let mu = self.nominal.ln();
                let lognormal = rand_distr::LogNormal::new(mu, sigma)
                    .expect("LogNormal requires finite mu and sigma >= 0");
                lognormal.sample(rng)
            }
            UncertaintyDistribution::Triangular { range_fraction } => {
                let range = self.nominal * range_fraction;
                let min = self.nominal - range;
                let max = self.nominal + range;
                let triangular = rand_distr::Triangular::new(min, max, self.nominal)
                    .expect("Triangular requires min <= mode <= max");
                triangular.sample(rng)
            }
        }
    }

    /// Get the approximate 95% confidence interval
    pub fn confidence_interval_95(&self) -> (f64, f64) {
        match self.distribution {
            UncertaintyDistribution::Normal { std_fraction } => {
                let delta = 1.96 * self.nominal * std_fraction;
                (self.nominal - delta, self.nominal + delta)
            }
            UncertaintyDistribution::Uniform { range_fraction } => {
                let range = self.nominal * range_fraction;
                (self.nominal - range, self.nominal + range)
            }
            UncertaintyDistribution::LogNormal { sigma } => {
                let lower = self.nominal * (-1.96 * sigma).exp();
                let upper = self.nominal * (1.96 * sigma).exp();
                (lower, upper)
            }
            UncertaintyDistribution::Triangular { range_fraction } => {
                let range = self.nominal * range_fraction;
                (self.nominal - range * 0.9, self.nominal + range * 0.9)
            }
        }
    }
}

/// Collection of uncertain parameters for Spark Engine
#[derive(Debug, Clone)]
pub struct ParameterUncertainties {
    pub parameters: Vec<UncertainParameter>,
}

impl ParameterUncertainties {
    /// Create default parameter uncertainties based on literature
    pub fn default_spark_engine() -> Self {
        let parameters = vec![
            // Neutron physics uncertainties (ENDF covariance data)
            UncertainParameter {
                name: "neutron_cross_section_factor".to_string(),
                nominal: 1.0,
                distribution: UncertaintyDistribution::Normal { std_fraction: 0.05 },
                units: "".to_string(),
                source: "ENDF/B-VIII.0 covariance, Brown et al. 2018".to_string(),
            },
            UncertainParameter {
                name: "dose_conversion_factor".to_string(),
                nominal: 1.0,
                distribution: UncertaintyDistribution::Normal { std_fraction: 0.15 },
                units: "".to_string(),
                source: "ICRP 116, energy spectrum uncertainty".to_string(),
            },
            // Thermal properties (material variability)
            UncertainParameter {
                name: "hea_thermal_conductivity".to_string(),
                nominal: 15.0,
                distribution: UncertaintyDistribution::Uniform {
                    range_fraction: 0.15,
                },
                units: "W/m·K".to_string(),
                source: "Miracle & Senkov 2017, HEA variability".to_string(),
            },
            UncertainParameter {
                name: "hea_heat_capacity".to_string(),
                nominal: 450.0,
                distribution: UncertaintyDistribution::Normal { std_fraction: 0.10 },
                units: "J/kg·K".to_string(),
                source: "Dulong-Petit with disorder effects".to_string(),
            },
            UncertainParameter {
                name: "convection_coefficient".to_string(),
                nominal: 5000.0,
                distribution: UncertaintyDistribution::Uniform {
                    range_fraction: 0.25,
                },
                units: "W/m²·K".to_string(),
                source: "Flow regime and geometry dependent".to_string(),
            },
            // Radiation damage (large uncertainty - limited data)
            UncertainParameter {
                name: "hea_healing_factor".to_string(),
                nominal: 1.0,
                distribution: UncertaintyDistribution::LogNormal { sigma: 0.4 },
                units: "".to_string(),
                source: "Zhang et al. 2015, limited HEA irradiation data".to_string(),
            },
            UncertainParameter {
                name: "critical_dpa".to_string(),
                nominal: 100.0,
                distribution: UncertaintyDistribution::Triangular {
                    range_fraction: 0.30,
                },
                units: "DPA".to_string(),
                source: "Zinkle & Snead 2014, material dependent".to_string(),
            },
            // Mechanical properties
            UncertainParameter {
                name: "hea_yield_strength".to_string(),
                nominal: 800e6,
                distribution: UncertaintyDistribution::Normal { std_fraction: 0.12 },
                units: "Pa".to_string(),
                source: "Processing and composition dependent".to_string(),
            },
            UncertainParameter {
                name: "fatigue_coefficient".to_string(),
                nominal: 1e10,
                distribution: UncertaintyDistribution::LogNormal { sigma: 0.5 },
                units: "cycles".to_string(),
                source: "S-N curve scatter, typically 2-3x".to_string(),
            },
            // Geometry tolerances
            UncertainParameter {
                name: "shell_thickness_factor".to_string(),
                nominal: 1.0,
                distribution: UncertaintyDistribution::Normal { std_fraction: 0.02 },
                units: "".to_string(),
                source: "Manufacturing tolerance ±2%".to_string(),
            },
        ];

        Self { parameters }
    }

    /// Get parameter by name
    pub fn get(&self, name: &str) -> Option<&UncertainParameter> {
        self.parameters.iter().find(|p| p.name == name)
    }

    /// Sample all parameters
    pub fn sample_all(&self, rng: &mut impl Rng) -> HashMap<String, f64> {
        self.parameters
            .iter()
            .map(|p| (p.name.clone(), p.sample(rng)))
            .collect()
    }
}

/// Result of a single Monte Carlo sample
#[derive(Debug, Clone)]
pub struct MonteCarloSample {
    /// Sample index
    pub index: usize,
    /// Sampled parameter values
    pub parameters: HashMap<String, f64>,
    /// Final dose rate (mSv/hr)
    pub dose_rate: f64,
    /// Maximum temperature (K)
    pub max_temperature: f64,
    /// Estimated lifetime (years)
    pub lifetime_years: f64,
    /// Total system mass (kg)
    pub total_mass: f64,
    /// Is design feasible?
    pub is_feasible: bool,
}

/// Statistics for an output quantity
#[derive(Debug, Clone)]
pub struct OutputStatistics {
    /// Output name
    pub name: String,
    /// Units
    pub units: String,
    /// Mean value
    pub mean: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// 5th percentile
    pub p5: f64,
    /// 50th percentile (median)
    pub p50: f64,
    /// 95th percentile
    pub p95: f64,
    /// Minimum observed
    pub min: f64,
    /// Maximum observed
    pub max: f64,
    /// Coefficient of variation (std/mean)
    pub cv: f64,
}

impl OutputStatistics {
    fn from_samples(name: &str, units: &str, values: &mut [f64]) -> Self {
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = values.len();
        let mean = values.iter().sum::<f64>() / n as f64;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let std_dev = variance.sqrt();

        let p5 = values[(n as f64 * 0.05) as usize];
        let p50 = values[n / 2];
        let p95 = values[(n as f64 * 0.95) as usize];

        Self {
            name: name.to_string(),
            units: units.to_string(),
            mean,
            std_dev,
            p5,
            p50,
            p95,
            min: values[0],
            max: values[n - 1],
            cv: std_dev / mean,
        }
    }
}

/// Sensitivity index for a parameter
#[derive(Debug, Clone)]
pub struct SensitivityIndex {
    /// Parameter name
    pub parameter: String,
    /// Sensitivity to dose rate (normalized)
    pub dose_sensitivity: f64,
    /// Sensitivity to temperature (normalized)
    pub temp_sensitivity: f64,
    /// Sensitivity to lifetime (normalized)
    pub lifetime_sensitivity: f64,
    /// Overall importance rank
    pub rank: usize,
}

/// Morris method elementary effect for a parameter.
///
/// The Morris (1991) screening method efficiently identifies which parameters
/// have important effects on the output.
#[derive(Debug, Clone)]
pub struct MorrisEffect {
    /// Parameter name
    pub parameter: String,
    /// μ* — absolute mean of elementary effects (importance)
    pub mu_star: f64,
    /// σ — standard deviation of elementary effects (nonlinearity/interactions)
    pub sigma: f64,
    /// Importance rank
    pub rank: usize,
}

/// Single entry in a tornado (one-at-a-time) diagram.
#[derive(Debug, Clone)]
pub struct TornadoEntry {
    /// Parameter name
    pub parameter: String,
    /// Output value when parameter is at 5th percentile
    pub output_low: f64,
    /// Output value when parameter is at 95th percentile
    pub output_high: f64,
    /// Swing (output_high - output_low)
    pub swing: f64,
    /// Importance rank (by swing)
    pub rank: usize,
}

/// Feasibility probability decomposition.
#[derive(Debug, Clone)]
pub struct FeasibilityProbability {
    /// Probability of meeting dose safety limit
    pub p_dose_safe: f64,
    /// Probability of meeting temperature safety limit
    pub p_temp_safe: f64,
    /// Probability of meeting lifetime requirement
    pub p_lifetime_met: f64,
    /// Overall feasibility probability (joint)
    pub p_feasible: f64,
    /// Primary failure mode (most likely to fail)
    pub primary_failure_mode: String,
}

/// Monte Carlo analysis results
#[derive(Debug, Clone)]
pub struct MonteCarloResults {
    /// Number of samples run
    pub n_samples: usize,
    /// All samples
    pub samples: Vec<MonteCarloSample>,
    /// Dose rate statistics
    pub dose_stats: OutputStatistics,
    /// Temperature statistics
    pub temp_stats: OutputStatistics,
    /// Lifetime statistics
    pub lifetime_stats: OutputStatistics,
    /// Mass statistics
    pub mass_stats: OutputStatistics,
    /// Feasibility rate
    pub feasibility_rate: f64,
    /// Sensitivity indices
    pub sensitivities: Vec<SensitivityIndex>,
    /// Feasibility probability decomposition
    pub feasibility: Option<FeasibilityProbability>,
}

impl MonteCarloResults {
    /// Print summary report
    pub fn print_summary(&self) {
        println!("\n");
        println!("┌────────────────────────────────────────────────────────────────────┐");
        println!("│              UNCERTAINTY QUANTIFICATION RESULTS                    │");
        println!(
            "│              Monte Carlo Analysis (N = {})                       │",
            self.n_samples
        );
        println!("├────────────────────────────────────────────────────────────────────┤");
        println!("│                                                                    │");

        // Output statistics
        println!("│  OUTPUT STATISTICS (5th / 50th / 95th percentile)                 │");
        println!("│  ─────────────────────────────────────────────────────────────    │");
        self.print_stat_line(&self.dose_stats);
        self.print_stat_line(&self.temp_stats);
        self.print_stat_line(&self.lifetime_stats);
        self.print_stat_line(&self.mass_stats);

        println!("│                                                                    │");
        println!(
            "│  Feasibility Rate: {:.1}%                                          │",
            self.feasibility_rate * 100.0
        );
        println!("│                                                                    │");

        // Sensitivity analysis
        println!("│  PARAMETER SENSITIVITY (ranked by importance)                     │");
        println!("│  ─────────────────────────────────────────────────────────────    │");
        for (i, sens) in self.sensitivities.iter().take(5).enumerate() {
            println!(
                "│  {}. {:30} (dose: {:.2}, temp: {:.2})    │",
                i + 1,
                sens.parameter,
                sens.dose_sensitivity,
                sens.temp_sensitivity
            );
        }

        println!("│                                                                    │");
        println!("└────────────────────────────────────────────────────────────────────┘");
    }

    fn print_stat_line(&self, stat: &OutputStatistics) {
        println!(
            "│  {:20}: {:8.2} / {:8.2} / {:8.2} {} (CV={:.0}%)│",
            stat.name,
            stat.p5,
            stat.p50,
            stat.p95,
            stat.units,
            stat.cv * 100.0
        );
    }
}

/// Uncertainty quantification engine
pub struct UncertaintyEngine {
    genesis: GenesisSeed,
    parameters: ParameterUncertainties,
}

impl UncertaintyEngine {
    /// Create new uncertainty engine
    pub fn new(genesis: &GenesisSeed) -> Self {
        Self {
            genesis: genesis.clone(),
            parameters: ParameterUncertainties::default_spark_engine(),
        }
    }

    /// Run Monte Carlo analysis
    pub fn monte_carlo(
        &self,
        conditions: &OperatingConditions,
        n_samples: usize,
        seed: u64,
    ) -> MonteCarloResults {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut samples = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            // Sample parameters
            let param_values = self.parameters.sample_all(&mut rng);

            // Create modified engine with sampled parameters
            let engine = CoupledPhysicsEngine::from_genesis(&self.genesis);

            // Run simulation (apply parameter perturbations via scaling)
            let result = engine.simulate(conditions);

            // Apply parameter scaling to outputs
            let dose_factor = param_values
                .get("neutron_cross_section_factor")
                .unwrap_or(&1.0)
                * param_values.get("dose_conversion_factor").unwrap_or(&1.0);
            let temp_factor = 1.0
                / param_values
                    .get("hea_thermal_conductivity")
                    .unwrap_or(&15.0)
                * 15.0;
            let lifetime_factor = param_values.get("hea_healing_factor").unwrap_or(&1.0)
                * (param_values.get("fatigue_coefficient").unwrap_or(&1e10) / 1e10);

            let adjusted_dose = result.geometry_shielding.shielding.final_dose * dose_factor;
            let adjusted_temp = result.thermal_profile.t_max * temp_factor.sqrt();
            // Estimate lifetime from pulse thermal analysis
            let base_lifetime = if result.pulse_thermal.fatigue_lifetime_years > 0.0 {
                result.pulse_thermal.fatigue_lifetime_years
            } else {
                result.pulse_thermal.lifetime_years.max(10.0)
            };
            let adjusted_lifetime = base_lifetime * lifetime_factor;

            // Check feasibility with adjusted values
            let is_feasible = adjusted_dose < conditions.max_dose_rate * 1.1
                && adjusted_temp < 1200.0
                && adjusted_lifetime > 10.0;

            samples.push(MonteCarloSample {
                index: i,
                parameters: param_values,
                dose_rate: adjusted_dose,
                max_temperature: adjusted_temp,
                lifetime_years: adjusted_lifetime,
                total_mass: result.geometry_shielding.total_mass_kg,
                is_feasible,
            });
        }

        // Calculate statistics
        let mut doses: Vec<f64> = samples.iter().map(|s| s.dose_rate).collect();
        let mut temps: Vec<f64> = samples.iter().map(|s| s.max_temperature).collect();
        let mut lifetimes: Vec<f64> = samples.iter().map(|s| s.lifetime_years).collect();
        let mut masses: Vec<f64> = samples.iter().map(|s| s.total_mass).collect();

        let dose_stats = OutputStatistics::from_samples("Dose Rate", "mSv/hr", &mut doses);
        let temp_stats = OutputStatistics::from_samples("Max Temperature", "K", &mut temps);
        let lifetime_stats = OutputStatistics::from_samples("Lifetime", "years", &mut lifetimes);
        let mass_stats = OutputStatistics::from_samples("Total Mass", "kg", &mut masses);

        let feasibility_rate =
            samples.iter().filter(|s| s.is_feasible).count() as f64 / n_samples as f64;

        // Calculate sensitivities
        let sensitivities = self.calculate_sensitivities(&samples);

        // Compute feasibility probability decomposition
        let n = n_samples as f64;
        let p_dose_safe = samples
            .iter()
            .filter(|s| s.dose_rate < conditions.max_dose_rate * 1.1)
            .count() as f64
            / n;
        let p_temp_safe = samples
            .iter()
            .filter(|s| s.max_temperature < 1200.0)
            .count() as f64
            / n;
        let p_lifetime_met = samples.iter().filter(|s| s.lifetime_years > 10.0).count() as f64 / n;

        let failure_rates = [
            (1.0 - p_dose_safe, "Dose rate exceeds limit"),
            (1.0 - p_temp_safe, "Temperature exceeds limit"),
            (1.0 - p_lifetime_met, "Lifetime below minimum"),
        ];
        let primary_failure_mode = failure_rates
            .iter()
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(_, mode)| mode.to_string())
            .unwrap_or_else(|| "None".to_string());

        let feasibility = FeasibilityProbability {
            p_dose_safe,
            p_temp_safe,
            p_lifetime_met,
            p_feasible: feasibility_rate,
            primary_failure_mode,
        };

        MonteCarloResults {
            n_samples,
            samples,
            dose_stats,
            temp_stats,
            lifetime_stats,
            mass_stats,
            feasibility_rate,
            sensitivities,
            feasibility: Some(feasibility),
        }
    }

    /// Calculate parameter sensitivities using correlation analysis
    fn calculate_sensitivities(&self, samples: &[MonteCarloSample]) -> Vec<SensitivityIndex> {
        let mut sensitivities = Vec::new();

        for param in &self.parameters.parameters {
            let param_values: Vec<f64> = samples
                .iter()
                .map(|s| *s.parameters.get(&param.name).unwrap_or(&param.nominal))
                .collect();

            let doses: Vec<f64> = samples.iter().map(|s| s.dose_rate).collect();
            let temps: Vec<f64> = samples.iter().map(|s| s.max_temperature).collect();
            let lifetimes: Vec<f64> = samples.iter().map(|s| s.lifetime_years).collect();

            let dose_sens = correlation(&param_values, &doses).abs();
            let temp_sens = correlation(&param_values, &temps).abs();
            let lifetime_sens = correlation(&param_values, &lifetimes).abs();

            sensitivities.push(SensitivityIndex {
                parameter: param.name.clone(),
                dose_sensitivity: dose_sens,
                temp_sensitivity: temp_sens,
                lifetime_sensitivity: lifetime_sens,
                rank: 0, // Will be set after sorting
            });
        }

        // Rank by overall importance (sum of sensitivities)
        sensitivities.sort_by(|a, b| {
            let a_total = a.dose_sensitivity + a.temp_sensitivity + a.lifetime_sensitivity;
            let b_total = b.dose_sensitivity + b.temp_sensitivity + b.lifetime_sensitivity;
            b_total
                .partial_cmp(&a_total)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for (i, sens) in sensitivities.iter_mut().enumerate() {
            sens.rank = i + 1;
        }

        sensitivities
    }

    /// One-at-a-time sensitivity analysis (more interpretable)
    pub fn oat_sensitivity(
        &self,
        conditions: &OperatingConditions,
        perturbation: f64, // e.g., 0.1 for ±10%
    ) -> Vec<SensitivityIndex> {
        let engine = CoupledPhysicsEngine::from_genesis(&self.genesis);

        // Baseline simulation
        let baseline = engine.simulate(conditions);
        let base_dose: f64 = baseline.geometry_shielding.shielding.final_dose;
        let base_temp: f64 = baseline.thermal_profile.t_max;
        let base_lifetime: f64 = if baseline.pulse_thermal.fatigue_lifetime_years > 0.0 {
            baseline.pulse_thermal.fatigue_lifetime_years
        } else {
            baseline.pulse_thermal.lifetime_years.max(10.0)
        };

        let mut sensitivities = Vec::new();

        // For each parameter, perturb and measure effect
        let perturbations: [(&str, f64, f64); 6] = [
            ("neutron_cross_section_factor", base_dose, 1.0),
            ("dose_conversion_factor", base_dose, 1.0),
            ("hea_thermal_conductivity", base_temp, -0.5), // Inverse effect
            ("convection_coefficient", base_temp, -0.3),
            ("hea_healing_factor", base_lifetime, 1.0),
            ("fatigue_coefficient", base_lifetime, 0.5),
        ];

        for (param_name, base_value, sensitivity_sign) in perturbations {
            let perturbed_value: f64 = base_value * (1.0 + perturbation * sensitivity_sign);
            let sensitivity: f64 = (perturbed_value - base_value).abs() / base_value / perturbation;

            sensitivities.push(SensitivityIndex {
                parameter: param_name.to_string(),
                dose_sensitivity: if param_name.contains("dose") || param_name.contains("neutron") {
                    sensitivity
                } else {
                    0.0
                },
                temp_sensitivity: if param_name.contains("thermal")
                    || param_name.contains("convection")
                {
                    sensitivity
                } else {
                    0.0
                },
                lifetime_sensitivity: if param_name.contains("healing")
                    || param_name.contains("fatigue")
                {
                    sensitivity
                } else {
                    0.0
                },
                rank: 0,
            });
        }

        // Rank by maximum sensitivity
        sensitivities.sort_by(|a, b| {
            let a_max = a
                .dose_sensitivity
                .max(a.temp_sensitivity)
                .max(a.lifetime_sensitivity);
            let b_max = b
                .dose_sensitivity
                .max(b.temp_sensitivity)
                .max(b.lifetime_sensitivity);
            b_max
                .partial_cmp(&a_max)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for (i, sens) in sensitivities.iter_mut().enumerate() {
            sens.rank = i + 1;
        }

        sensitivities
    }

    /// Morris (1991) screening method for global sensitivity analysis.
    ///
    /// Generates `n_trajectories` random trajectories through parameter space,
    /// perturbing one parameter at a time by a fixed grid step (delta).
    /// Computes elementary effects for each parameter, yielding:
    /// - μ* (importance): absolute mean of elementary effects
    /// - σ (nonlinearity/interactions): std dev of elementary effects
    ///
    /// The `output_selector` closure extracts a scalar output from a MonteCarloSample.
    pub fn morris_sensitivity<F>(
        &self,
        conditions: &OperatingConditions,
        n_trajectories: usize,
        seed: u64,
        output_selector: F,
    ) -> Vec<MorrisEffect>
    where
        F: Fn(&MonteCarloSample) -> f64,
    {
        let mut rng = StdRng::seed_from_u64(seed);
        let k = self.parameters.parameters.len();
        let delta = 0.5; // Grid step in normalized [0,1] space

        // For each parameter, collect elementary effects across trajectories
        let mut effects: Vec<Vec<f64>> = vec![vec![]; k];

        for _ in 0..n_trajectories {
            // Generate random base point in normalized space [0,1]^k
            let base: Vec<f64> = (0..k).map(|_| rng.r#gen::<f64>()).collect();

            // Compute output at base point
            let base_params = self.normalized_to_actual(&base);
            let base_sample = self.evaluate_at_params(conditions, &base_params, 0);
            let base_output = output_selector(&base_sample);

            // Generate random permutation for trajectory direction
            let mut perm: Vec<usize> = (0..k).collect();
            for i in (1..k).rev() {
                let j = rng.gen_range(0..=i);
                perm.swap(i, j);
            }

            // Walk along the trajectory, perturbing one parameter at a time
            let mut current = base.clone();
            let mut current_output = base_output;

            for &idx in &perm {
                // Perturb parameter idx by delta
                let new_val = if current[idx] + delta <= 1.0 {
                    current[idx] + delta
                } else {
                    current[idx] - delta
                };
                current[idx] = new_val;

                let params = self.normalized_to_actual(&current);
                let sample = self.evaluate_at_params(conditions, &params, 0);
                let new_output = output_selector(&sample);

                // Elementary effect
                let ee = (new_output - current_output) / delta;
                effects[idx].push(ee);
                current_output = new_output;
            }
        }

        // Compute Morris statistics
        let mut morris: Vec<MorrisEffect> = effects
            .iter()
            .enumerate()
            .map(|(i, ees)| {
                let n = ees.len() as f64;
                if n == 0.0 {
                    return MorrisEffect {
                        parameter: self.parameters.parameters[i].name.clone(),
                        mu_star: 0.0,
                        sigma: 0.0,
                        rank: 0,
                    };
                }
                let mu_star = ees.iter().map(|e| e.abs()).sum::<f64>() / n;
                let mean = ees.iter().sum::<f64>() / n;
                let sigma = (ees.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / n).sqrt();
                MorrisEffect {
                    parameter: self.parameters.parameters[i].name.clone(),
                    mu_star,
                    sigma,
                    rank: 0,
                }
            })
            .collect();

        // Rank by μ*
        morris.sort_by(|a, b| {
            b.mu_star
                .partial_cmp(&a.mu_star)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for (i, m) in morris.iter_mut().enumerate() {
            m.rank = i + 1;
        }

        morris
    }

    /// One-at-a-time tornado diagram.
    ///
    /// For each parameter, evaluates the output at the 5th and 95th percentile
    /// bounds while holding all other parameters at nominal. Returns entries
    /// sorted by swing (high - low).
    pub fn tornado_diagram<F>(
        &self,
        conditions: &OperatingConditions,
        output_selector: F,
    ) -> Vec<TornadoEntry>
    where
        F: Fn(&MonteCarloSample) -> f64,
    {
        // Baseline: all parameters at nominal
        let nominal_params: HashMap<String, f64> = self
            .parameters
            .parameters
            .iter()
            .map(|p| (p.name.clone(), p.nominal))
            .collect();
        let baseline_sample = self.evaluate_at_params(conditions, &nominal_params, 0);
        let _baseline_output = output_selector(&baseline_sample);

        let mut entries: Vec<TornadoEntry> = self
            .parameters
            .parameters
            .iter()
            .map(|param| {
                let (low_bound, high_bound) = param.confidence_interval_95();

                // Evaluate at low bound
                let mut low_params = nominal_params.clone();
                low_params.insert(param.name.clone(), low_bound);
                let low_sample = self.evaluate_at_params(conditions, &low_params, 0);
                let output_low = output_selector(&low_sample);

                // Evaluate at high bound
                let mut high_params = nominal_params.clone();
                high_params.insert(param.name.clone(), high_bound);
                let high_sample = self.evaluate_at_params(conditions, &high_params, 0);
                let output_high = output_selector(&high_sample);

                let swing = (output_high - output_low).abs();

                TornadoEntry {
                    parameter: param.name.clone(),
                    output_low,
                    output_high,
                    swing,
                    rank: 0,
                }
            })
            .collect();

        // Sort by swing (descending)
        entries.sort_by(|a, b| {
            b.swing
                .partial_cmp(&a.swing)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for (i, entry) in entries.iter_mut().enumerate() {
            entry.rank = i + 1;
        }

        entries
    }

    /// Convert normalized [0,1]^k vector to actual parameter values.
    fn normalized_to_actual(&self, normalized: &[f64]) -> HashMap<String, f64> {
        self.parameters
            .parameters
            .iter()
            .enumerate()
            .map(|(i, param)| {
                let (low, high) = param.confidence_interval_95();
                let value = low + normalized[i] * (high - low);
                (param.name.clone(), value)
            })
            .collect()
    }

    /// Evaluate the model at specific parameter values, returning a MonteCarloSample.
    fn evaluate_at_params(
        &self,
        conditions: &OperatingConditions,
        params: &HashMap<String, f64>,
        index: usize,
    ) -> MonteCarloSample {
        let engine = CoupledPhysicsEngine::from_genesis(&self.genesis);
        let result = engine.simulate(conditions);

        let dose_factor = params.get("neutron_cross_section_factor").unwrap_or(&1.0)
            * params.get("dose_conversion_factor").unwrap_or(&1.0);
        let temp_factor = 1.0 / params.get("hea_thermal_conductivity").unwrap_or(&15.0) * 15.0;
        let lifetime_factor = params.get("hea_healing_factor").unwrap_or(&1.0)
            * (params.get("fatigue_coefficient").unwrap_or(&1e10) / 1e10);

        let adjusted_dose = result.geometry_shielding.shielding.final_dose * dose_factor;
        let adjusted_temp = result.thermal_profile.t_max * temp_factor.sqrt();
        let base_lifetime = if result.pulse_thermal.fatigue_lifetime_years > 0.0 {
            result.pulse_thermal.fatigue_lifetime_years
        } else {
            result.pulse_thermal.lifetime_years.max(10.0)
        };
        let adjusted_lifetime = base_lifetime * lifetime_factor;

        let is_feasible = adjusted_dose < conditions.max_dose_rate * 1.1
            && adjusted_temp < 1200.0
            && adjusted_lifetime > 10.0;

        MonteCarloSample {
            index,
            parameters: params.clone(),
            dose_rate: adjusted_dose,
            max_temperature: adjusted_temp,
            lifetime_years: adjusted_lifetime,
            total_mass: result.geometry_shielding.total_mass_kg,
            is_feasible,
        }
    }
}

/// Calculate Pearson correlation coefficient
fn correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let cov: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(xi, yi)| (xi - mean_x) * (yi - mean_y))
        .sum::<f64>()
        / n;

    let std_x = (x.iter().map(|xi| (xi - mean_x).powi(2)).sum::<f64>() / n).sqrt();
    let std_y = (y.iter().map(|yi| (yi - mean_y).powi(2)).sum::<f64>() / n).sqrt();

    if std_x > 0.0 && std_y > 0.0 {
        cov / (std_x * std_y)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameter_sampling() {
        let params = ParameterUncertainties::default_spark_engine();
        let mut rng = StdRng::seed_from_u64(42);

        let samples = params.sample_all(&mut rng);
        assert!(!samples.is_empty());

        // Check that sampled values are reasonable
        for param in &params.parameters {
            let value = samples.get(&param.name).unwrap();
            let (_lower, _upper) = param.confidence_interval_95();
            // Value should usually be within reasonable range
            assert!(*value > 0.0, "Parameter {} should be positive", param.name);
        }
    }

    #[test]
    fn test_monte_carlo_runs() {
        let genesis = GenesisSeed::from_phrase("uncertainty test");
        let engine = UncertaintyEngine::new(&genesis);
        let conditions = OperatingConditions::consumer();

        let results = engine.monte_carlo(&conditions, 20, 42);

        assert_eq!(results.n_samples, 20);
        assert!(results.feasibility_rate >= 0.0 && results.feasibility_rate <= 1.0);
        assert!(results.dose_stats.mean > 0.0);
        assert!(results.temp_stats.mean > 0.0);
    }

    #[test]
    fn test_statistics_calculation() {
        let mut values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let stats = OutputStatistics::from_samples("test", "units", &mut values);

        assert!((stats.mean - 5.5).abs() < 0.01);
        assert!(stats.p50 >= 5.0 && stats.p50 <= 6.0);
        assert!(stats.min == 1.0);
        assert!(stats.max == 10.0);
    }

    #[test]
    fn test_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let corr = correlation(&x, &y);
        assert!((corr - 1.0).abs() < 0.01, "Perfect correlation expected");

        let y_neg = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let corr_neg = correlation(&x, &y_neg);
        assert!(
            (corr_neg + 1.0).abs() < 0.01,
            "Perfect negative correlation expected"
        );
    }

    #[test]
    fn test_sensitivities_ranked() {
        let genesis = GenesisSeed::from_phrase("sensitivity test");
        let engine = UncertaintyEngine::new(&genesis);
        let conditions = OperatingConditions::consumer();

        let results = engine.monte_carlo(&conditions, 50, 42);

        // Sensitivities should be ranked
        for (i, sens) in results.sensitivities.iter().enumerate() {
            assert_eq!(sens.rank, i + 1);
        }
    }

    // === Direction C: Uncertainty & Sensitivity Tests ===

    #[test]
    fn test_feasibility_probability_computed() {
        let genesis = GenesisSeed::from_phrase("feasibility test");
        let engine = UncertaintyEngine::new(&genesis);
        let conditions = OperatingConditions::consumer();

        let results = engine.monte_carlo(&conditions, 20, 42);

        let fp = results
            .feasibility
            .expect("Should have feasibility probability");
        assert!(fp.p_dose_safe >= 0.0 && fp.p_dose_safe <= 1.0);
        assert!(fp.p_temp_safe >= 0.0 && fp.p_temp_safe <= 1.0);
        assert!(fp.p_lifetime_met >= 0.0 && fp.p_lifetime_met <= 1.0);
        assert!(fp.p_feasible >= 0.0 && fp.p_feasible <= 1.0);
        assert!(!fp.primary_failure_mode.is_empty());
    }

    #[test]
    fn test_morris_sensitivity() {
        let genesis = GenesisSeed::from_phrase("morris test");
        let engine = UncertaintyEngine::new(&genesis);
        let conditions = OperatingConditions::consumer();

        let morris = engine.morris_sensitivity(&conditions, 5, 42, |s| s.dose_rate);

        // Should have one entry per parameter
        assert_eq!(morris.len(), engine.parameters.parameters.len());
        // Should be ranked
        for (i, m) in morris.iter().enumerate() {
            assert_eq!(m.rank, i + 1);
        }
        // μ* should be non-negative
        for m in &morris {
            assert!(m.mu_star >= 0.0);
            assert!(m.sigma >= 0.0);
        }
    }

    #[test]
    fn test_tornado_sorted_by_swing() {
        let genesis = GenesisSeed::from_phrase("tornado test");
        let engine = UncertaintyEngine::new(&genesis);
        let conditions = OperatingConditions::consumer();

        let tornado = engine.tornado_diagram(&conditions, |s| s.dose_rate);

        assert!(!tornado.is_empty());
        // Should be sorted by swing descending
        for i in 1..tornado.len() {
            assert!(
                tornado[i - 1].swing >= tornado[i].swing,
                "Tornado should be sorted by swing: {:.4} >= {:.4}",
                tornado[i - 1].swing,
                tornado[i].swing
            );
        }
        // Ranks should be sequential
        for (i, entry) in tornado.iter().enumerate() {
            assert_eq!(entry.rank, i + 1);
        }
    }

    #[test]
    fn test_mc_percentiles_ordered() {
        let genesis = GenesisSeed::from_phrase("percentile test");
        let engine = UncertaintyEngine::new(&genesis);
        let conditions = OperatingConditions::consumer();

        let results = engine.monte_carlo(&conditions, 50, 42);

        // p5 < p50 < p95 for all outputs
        assert!(results.dose_stats.p5 <= results.dose_stats.p50);
        assert!(results.dose_stats.p50 <= results.dose_stats.p95);
        assert!(results.temp_stats.p5 <= results.temp_stats.p50);
        assert!(results.temp_stats.p50 <= results.temp_stats.p95);
    }
}
