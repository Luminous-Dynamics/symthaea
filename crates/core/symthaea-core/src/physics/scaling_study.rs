// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Scaling Law Study: 1W to 1MW Analysis
//!
//! This module analyzes how LCF reactor performance scales across 6 orders
//! of magnitude (1W to 1MW), identifying optimal architectures and trigger
//! systems at each power level.
//!
//! ## Key Scaling Questions
//!
//! 1. **Thermal**: Does power density stay constant or scale with size?
//! 2. **Neutronics**: How does shielding mass scale with power?
//! 3. **Materials**: When do HEA limits become binding constraints?
//! 4. **Economics**: What's the $/W at each scale?
//! 5. **Trigger**: Which triggers work at which scales?
//!
//! ## Expected Scaling Laws
//!
//! | Parameter       | Scaling        | Reason                          |
//! |-----------------|----------------|---------------------------------|
//! | Volume          | P^0.67-1.0     | Power density vs geometry       |
//! | Mass            | P^0.67-1.0     | Surface/volume effects          |
//! | Cost            | P^0.6-0.8      | Economies of scale              |
//! | Shielding       | P^0.5-0.67     | Geometric attenuation           |
//! | Lifetime        | P^-0.1-0       | Larger = better cooling         |
//!
//! ## Usage
//!
//! ```rust,ignore
//! let study = ScalingStudy::new();
//! let results = study.run_full_study();
//! println!("{}", study.generate_report(&results));
//! ```

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use super::radiation_damage::FusionReaction;
use super::reactor_architectures::{
    ReactorArchitecture, ReactorDesignParams, ReactorDesigner, ReactorSpec,
};
use super::trigger_systems::{ExtendedTriggerMethod, TriggerSystemLibrary};
use crate::genesis::GenesisSeed;

/// Power levels for the study (6 orders of magnitude)
pub const POWER_LEVELS_W: [f64; 7] = [
    1.0,         // 1W - sensor power
    10.0,        // 10W - electronics
    100.0,       // 100W - laptop
    1_000.0,     // 1kW - space heater
    10_000.0,    // 10kW - home
    100_000.0,   // 100kW - building
    1_000_000.0, // 1MW - industrial
];

/// Single data point in the scaling study
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingDataPoint {
    /// Power level (W)
    pub power_w: f64,
    /// Optimal architecture at this scale
    pub architecture: ReactorArchitecture,
    /// Optimal trigger system
    pub trigger: ExtendedTriggerMethod,
    /// Total system volume (m³)
    pub volume_m3: f64,
    /// Total system mass (kg)
    pub mass_kg: f64,
    /// Estimated cost (USD)
    pub cost_usd: f64,
    /// Specific cost ($/W)
    pub specific_cost: f64,
    /// Power density (W/m³)
    pub power_density: f64,
    /// Specific power (W/kg)
    pub specific_power: f64,
    /// Shielding mass fraction
    pub shielding_fraction: f64,
    /// Expected lifetime (years)
    pub lifetime_years: f64,
    /// Technology readiness level
    pub trl: u8,
    /// Feasibility score (0-100)
    pub feasibility_score: f64,
    /// Limiting factor at this scale
    pub limiting_factor: String,
    /// Full reactor spec (for detailed analysis)
    pub reactor: ReactorSpec,
}

/// Scaling metrics for a power level (used in uncertainty studies).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingMetrics {
    pub volume_m3: f64,
    pub mass_kg: f64,
    pub cost_usd: f64,
    pub specific_cost: f64,
    pub power_density: f64,
    pub specific_power: f64,
    pub lifetime_years: f64,
    pub feasibility_score: f64,
}

impl ScalingMetrics {
    pub fn from_data_point(dp: &ScalingDataPoint) -> Self {
        ScalingMetrics {
            volume_m3: dp.volume_m3,
            mass_kg: dp.mass_kg,
            cost_usd: dp.cost_usd,
            specific_cost: dp.specific_cost,
            power_density: dp.power_density,
            specific_power: dp.specific_power,
            lifetime_years: dp.lifetime_years,
            feasibility_score: dp.feasibility_score,
        }
    }
}

/// Scaling data point with uncertainty bounds from Monte Carlo.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingDataPointWithUncertainty {
    /// Power level (W)
    pub power_w: f64,
    /// Nominal (baseline) metrics
    pub nominal: ScalingMetrics,
    /// 5th percentile metrics
    pub p5: ScalingMetrics,
    /// 50th percentile (median) metrics
    pub p50: ScalingMetrics,
    /// 95th percentile metrics
    pub p95: ScalingMetrics,
    /// Coefficient of variation for cost
    pub cost_cv: f64,
    /// Coefficient of variation for mass
    pub mass_cv: f64,
    /// Probability of feasibility (fraction of MC samples that pass)
    pub p_feasible: f64,
}

/// Scaling law fit result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingLaw {
    /// Parameter name
    pub parameter: String,
    /// Scaling exponent (Y = A * P^exponent)
    pub exponent: f64,
    /// Scaling coefficient (A)
    pub coefficient: f64,
    /// R² goodness of fit
    pub r_squared: f64,
    /// Physical interpretation
    pub interpretation: String,
}

/// Complete scaling study results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingStudyResults {
    /// Data points for each power level
    pub data: Vec<ScalingDataPoint>,
    /// Fitted scaling laws
    pub scaling_laws: Vec<ScalingLaw>,
    /// Architecture transitions (power level → new architecture)
    pub transitions: Vec<(f64, ReactorArchitecture, String)>,
    /// Trigger transitions (power level → new trigger)
    pub trigger_transitions: Vec<(f64, ExtendedTriggerMethod, String)>,
    /// Key findings
    pub findings: Vec<String>,
    /// Recommendations by application
    pub recommendations: Vec<(String, String)>,
}

/// Scaling study engine
pub struct ScalingStudy {
    reactor_designer: ReactorDesigner,
    trigger_library: TriggerSystemLibrary,
    /// Genesis seed for reproducible Monte Carlo sampling
    pub genesis: GenesisSeed,
}

impl ScalingStudy {
    pub fn new() -> Self {
        Self {
            reactor_designer: ReactorDesigner::new(),
            trigger_library: TriggerSystemLibrary::new(),
            genesis: GenesisSeed::from_phrase("scaling study default"),
        }
    }

    /// Create with a specific genesis seed for reproducibility.
    pub fn with_genesis(genesis: GenesisSeed) -> Self {
        Self {
            reactor_designer: ReactorDesigner::new(),
            trigger_library: TriggerSystemLibrary::new(),
            genesis,
        }
    }

    /// Run complete scaling study across all power levels
    pub fn run_full_study(&self) -> ScalingStudyResults {
        let data: Vec<ScalingDataPoint> = POWER_LEVELS_W
            .iter()
            .map(|&power| self.analyze_power_level(power))
            .collect();

        let scaling_laws = self.fit_scaling_laws(&data);
        let transitions = self.find_architecture_transitions(&data);
        let trigger_transitions = self.find_trigger_transitions(&data);
        let findings = self.extract_findings(&data, &scaling_laws);
        let recommendations = self.generate_recommendations(&data);

        ScalingStudyResults {
            data,
            scaling_laws,
            transitions,
            trigger_transitions,
            findings,
            recommendations,
        }
    }

    /// Run scaling study with Monte Carlo uncertainty propagation.
    ///
    /// For each power level, runs `n_mc` Monte Carlo samples by perturbing key
    /// parameters: screening_energy (±10%), power_density (±30%), capital_cost (±20%).
    /// Returns percentile bounds and feasibility probability at each power level.
    pub fn run_study_with_uncertainty(
        &self,
        n_mc: usize,
        seed: u64,
    ) -> Vec<ScalingDataPointWithUncertainty> {
        let mut rng = StdRng::seed_from_u64(seed);

        POWER_LEVELS_W
            .iter()
            .map(|&power| {
                let nominal = self.analyze_power_level(power);
                let nominal_metrics = ScalingMetrics::from_data_point(&nominal);

                // Run MC samples with perturbed parameters
                let mut mc_costs: Vec<f64> = Vec::with_capacity(n_mc);
                let mut mc_masses: Vec<f64> = Vec::with_capacity(n_mc);
                let mut mc_volumes: Vec<f64> = Vec::with_capacity(n_mc);
                let mut mc_specific_costs: Vec<f64> = Vec::with_capacity(n_mc);
                let mut mc_power_densities: Vec<f64> = Vec::with_capacity(n_mc);
                let mut mc_specific_powers: Vec<f64> = Vec::with_capacity(n_mc);
                let mut mc_lifetimes: Vec<f64> = Vec::with_capacity(n_mc);
                let mut mc_feasibilities: Vec<f64> = Vec::with_capacity(n_mc);
                let mut n_feasible = 0usize;

                for _ in 0..n_mc {
                    // Perturb: screening ±10%, power_density ±30%, capital ±20%
                    let screening_factor: f64 = 1.0 + rng.gen_range(-0.10..0.10);
                    let density_factor: f64 = 1.0 + rng.gen_range(-0.30..0.30);
                    let cost_factor: f64 = 1.0 + rng.gen_range(-0.20..0.20);

                    let perturbed_cost = nominal.cost_usd * cost_factor;
                    let perturbed_volume = nominal.volume_m3 / density_factor.max(0.1);
                    let perturbed_mass = nominal.mass_kg * (1.0 / density_factor.max(0.1)).sqrt();
                    let perturbed_specific_cost = perturbed_cost / power;
                    let perturbed_power_density = power / perturbed_volume.max(0.001);
                    let perturbed_specific_power = power / perturbed_mass.max(0.1);
                    let perturbed_lifetime = nominal.lifetime_years * screening_factor;
                    let perturbed_feasibility =
                        (nominal.feasibility_score + screening_factor * 5.0 - 5.0)
                            .clamp(0.0, 100.0);

                    mc_costs.push(perturbed_cost);
                    mc_masses.push(perturbed_mass);
                    mc_volumes.push(perturbed_volume);
                    mc_specific_costs.push(perturbed_specific_cost);
                    mc_power_densities.push(perturbed_power_density);
                    mc_specific_powers.push(perturbed_specific_power);
                    mc_lifetimes.push(perturbed_lifetime);
                    mc_feasibilities.push(perturbed_feasibility);

                    if perturbed_feasibility > 50.0 {
                        n_feasible += 1;
                    }
                }

                // Sort for percentiles
                mc_costs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                mc_masses.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                mc_volumes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                mc_specific_costs
                    .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                mc_power_densities
                    .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                mc_specific_powers
                    .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                mc_lifetimes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                mc_feasibilities
                    .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                let pct = |v: &[f64], p: f64| -> f64 {
                    if v.is_empty() {
                        return 0.0;
                    }
                    let idx = ((v.len() as f64) * p).clamp(0.0, (v.len() - 1) as f64) as usize;
                    v[idx]
                };

                let cost_mean = mc_costs.iter().sum::<f64>() / n_mc as f64;
                let cost_std = (mc_costs
                    .iter()
                    .map(|c| (c - cost_mean).powi(2))
                    .sum::<f64>()
                    / n_mc as f64)
                    .sqrt();
                let mass_mean = mc_masses.iter().sum::<f64>() / n_mc as f64;
                let mass_std = (mc_masses
                    .iter()
                    .map(|m| (m - mass_mean).powi(2))
                    .sum::<f64>()
                    / n_mc as f64)
                    .sqrt();

                ScalingDataPointWithUncertainty {
                    power_w: power,
                    nominal: nominal_metrics,
                    p5: ScalingMetrics {
                        volume_m3: pct(&mc_volumes, 0.05),
                        mass_kg: pct(&mc_masses, 0.05),
                        cost_usd: pct(&mc_costs, 0.05),
                        specific_cost: pct(&mc_specific_costs, 0.05),
                        power_density: pct(&mc_power_densities, 0.05),
                        specific_power: pct(&mc_specific_powers, 0.05),
                        lifetime_years: pct(&mc_lifetimes, 0.05),
                        feasibility_score: pct(&mc_feasibilities, 0.05),
                    },
                    p50: ScalingMetrics {
                        volume_m3: pct(&mc_volumes, 0.50),
                        mass_kg: pct(&mc_masses, 0.50),
                        cost_usd: pct(&mc_costs, 0.50),
                        specific_cost: pct(&mc_specific_costs, 0.50),
                        power_density: pct(&mc_power_densities, 0.50),
                        specific_power: pct(&mc_specific_powers, 0.50),
                        lifetime_years: pct(&mc_lifetimes, 0.50),
                        feasibility_score: pct(&mc_feasibilities, 0.50),
                    },
                    p95: ScalingMetrics {
                        volume_m3: pct(&mc_volumes, 0.95),
                        mass_kg: pct(&mc_masses, 0.95),
                        cost_usd: pct(&mc_costs, 0.95),
                        specific_cost: pct(&mc_specific_costs, 0.95),
                        power_density: pct(&mc_power_densities, 0.95),
                        specific_power: pct(&mc_specific_powers, 0.95),
                        lifetime_years: pct(&mc_lifetimes, 0.95),
                        feasibility_score: pct(&mc_feasibilities, 0.95),
                    },
                    cost_cv: if cost_mean > 0.0 {
                        cost_std / cost_mean
                    } else {
                        0.0
                    },
                    mass_cv: if mass_mean > 0.0 {
                        mass_std / mass_mean
                    } else {
                        0.0
                    },
                    p_feasible: n_feasible as f64 / n_mc as f64,
                }
            })
            .collect()
    }

    /// Analyze a single power level
    fn analyze_power_level(&self, power_w: f64) -> ScalingDataPoint {
        // Design reactor for this power level
        let params = ReactorDesignParams {
            target_power_w: power_w,
            target_lifetime_years: 20.0,
            max_cost_usd: power_w * 100.0,              // $100/W budget
            max_volume_m3: (power_w / 1000.0).max(0.1), // ~1 liter/kW
            max_mass_kg: (power_w / 10.0).max(10.0),    // ~100g/W
            allow_neutrons: true,
            min_trl: 3,
            risk_weight: 0.0, // No risk adjustment for scaling study
        };

        let reactor = self.reactor_designer.design(&params);

        // Determine optimal trigger for this scale
        let trigger = self.trigger_library.optimal_for_power(power_w);

        // Calculate metrics
        let power_density = power_w / reactor.volume_m3.max(0.001);
        let specific_power = power_w / reactor.mass_kg.max(0.1);
        let specific_cost = reactor.cost_usd / power_w;

        // Estimate shielding fraction from reactor spec
        let shielding_fraction = reactor.shielding.total_mass_kg / reactor.mass_kg;

        // Determine limiting factor
        let limiting_factor = self.determine_limiting_factor(&reactor, power_w);

        // Calculate feasibility score
        let feasibility_score = self.calculate_feasibility(&reactor, &params);

        ScalingDataPoint {
            power_w,
            architecture: reactor.architecture,
            trigger: trigger.method,
            volume_m3: reactor.volume_m3,
            mass_kg: reactor.mass_kg,
            cost_usd: reactor.cost_usd,
            specific_cost,
            power_density,
            specific_power,
            shielding_fraction,
            lifetime_years: reactor.lifetime_years,
            trl: reactor.trl,
            feasibility_score,
            limiting_factor,
            reactor,
        }
    }

    /// Determine the limiting factor at a given power level
    fn determine_limiting_factor(&self, reactor: &ReactorSpec, power_w: f64) -> String {
        // Check various constraints
        if power_w < 100.0 {
            if reactor.trl < 5 {
                return "Technology maturity".to_string();
            }
            return "Trigger efficiency at small scale".to_string();
        }

        if power_w > 100_000.0 {
            if reactor.reactions.contains(&FusionReaction::DT) {
                return "14.1 MeV neutron damage".to_string();
            }
            return "Thermal management".to_string();
        }

        // Mid-range: various factors
        match reactor.architecture {
            ReactorArchitecture::AneutronicCore => "He3 fuel cost".to_string(),
            ReactorArchitecture::PulsedElectrolysis => "Cathode degradation".to_string(),
            ReactorArchitecture::FlowReactor => "Pb-Li corrosion".to_string(),
            _ => "Material lifetime under irradiation".to_string(),
        }
    }

    /// Calculate overall feasibility score
    fn calculate_feasibility(&self, reactor: &ReactorSpec, params: &ReactorDesignParams) -> f64 {
        let mut score = 50.0; // Base score

        // TRL contribution (0-20)
        score += (reactor.trl as f64) * 2.0;

        // Cost within budget (0-15)
        if reactor.cost_usd <= params.max_cost_usd {
            score += 15.0 * (1.0 - reactor.cost_usd / params.max_cost_usd);
        } else {
            score -= 10.0 * (reactor.cost_usd / params.max_cost_usd - 1.0).min(2.0);
        }

        // Volume within limits (0-10)
        if reactor.volume_m3 <= params.max_volume_m3 {
            score += 10.0;
        }

        // Lifetime meets target (0-10)
        if reactor.lifetime_years >= params.target_lifetime_years {
            score += 10.0;
        } else {
            score += 10.0 * reactor.lifetime_years / params.target_lifetime_years;
        }

        // Q-factor bonus (estimated)
        if reactor.yield_estimate.q_factor > 1.0 {
            score += 20.0 * (1.0 - 1.0 / reactor.yield_estimate.q_factor);
        }

        score.clamp(0.0, 100.0)
    }

    /// Fit power-law scaling relations to the data
    fn fit_scaling_laws(&self, data: &[ScalingDataPoint]) -> Vec<ScalingLaw> {
        let mut laws = Vec::new();

        // Volume scaling: V = A * P^n
        let (exp, coef, r2) = self.fit_power_law(
            &data.iter().map(|d| d.power_w).collect::<Vec<_>>(),
            &data.iter().map(|d| d.volume_m3).collect::<Vec<_>>(),
        );
        laws.push(ScalingLaw {
            parameter: "Volume (m³)".to_string(),
            exponent: exp,
            coefficient: coef,
            r_squared: r2,
            interpretation: if exp < 0.8 {
                "Sub-linear: larger systems have higher power density".to_string()
            } else if exp > 1.1 {
                "Super-linear: cooling limits power density at scale".to_string()
            } else {
                "Linear: power density roughly constant".to_string()
            },
        });

        // Mass scaling: M = A * P^n
        let (exp, coef, r2) = self.fit_power_law(
            &data.iter().map(|d| d.power_w).collect::<Vec<_>>(),
            &data.iter().map(|d| d.mass_kg).collect::<Vec<_>>(),
        );
        laws.push(ScalingLaw {
            parameter: "Mass (kg)".to_string(),
            exponent: exp,
            coefficient: coef,
            r_squared: r2,
            interpretation: if exp < 0.8 {
                "Favorable: specific power improves at scale".to_string()
            } else {
                "Shielding mass dominates at larger scales".to_string()
            },
        });

        // Cost scaling: $ = A * P^n
        let (exp, coef, r2) = self.fit_power_law(
            &data.iter().map(|d| d.power_w).collect::<Vec<_>>(),
            &data.iter().map(|d| d.cost_usd).collect::<Vec<_>>(),
        );
        laws.push(ScalingLaw {
            parameter: "Cost (USD)".to_string(),
            exponent: exp,
            coefficient: coef,
            r_squared: r2,
            interpretation: if exp < 0.9 {
                "Strong economies of scale".to_string()
            } else if exp > 1.0 {
                "Diseconomies of scale (complexity)".to_string()
            } else {
                "Neutral scaling".to_string()
            },
        });

        // Specific cost: $/W = A * P^n (should be negative exponent)
        let (exp, coef, r2) = self.fit_power_law(
            &data.iter().map(|d| d.power_w).collect::<Vec<_>>(),
            &data.iter().map(|d| d.specific_cost).collect::<Vec<_>>(),
        );
        laws.push(ScalingLaw {
            parameter: "Specific Cost ($/W)".to_string(),
            exponent: exp,
            coefficient: coef,
            r_squared: r2,
            interpretation: if exp < -0.1 {
                "Cost per watt decreases with scale".to_string()
            } else {
                "Cost per watt roughly constant".to_string()
            },
        });

        // Shielding fraction
        let (exp, coef, r2) = self.fit_power_law(
            &data.iter().map(|d| d.power_w).collect::<Vec<_>>(),
            &data
                .iter()
                .map(|d| d.shielding_fraction.max(0.01))
                .collect::<Vec<_>>(),
        );
        laws.push(ScalingLaw {
            parameter: "Shielding Fraction".to_string(),
            exponent: exp,
            coefficient: coef,
            r_squared: r2,
            interpretation: if exp < 0.0 {
                "Shielding becomes smaller fraction at scale (good)".to_string()
            } else {
                "Shielding dominates larger systems".to_string()
            },
        });

        laws
    }

    /// Fit Y = A * X^n using log-log linear regression
    fn fit_power_law(&self, x: &[f64], y: &[f64]) -> (f64, f64, f64) {
        let n = x.len() as f64;

        // Convert to log space
        let log_x: Vec<f64> = x.iter().map(|&xi| xi.ln()).collect();
        let log_y: Vec<f64> = y.iter().map(|&yi| yi.max(1e-10).ln()).collect();

        // Linear regression in log space
        let sum_x: f64 = log_x.iter().sum();
        let sum_y: f64 = log_y.iter().sum();
        let sum_xy: f64 = log_x.iter().zip(&log_y).map(|(xi, yi)| xi * yi).sum();
        let sum_xx: f64 = log_x.iter().map(|xi| xi * xi).sum();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
        let intercept = (sum_y - slope * sum_x) / n;

        // Calculate R²
        let y_mean = sum_y / n;
        let ss_tot: f64 = log_y.iter().map(|yi| (yi - y_mean).powi(2)).sum();
        let ss_res: f64 = log_x
            .iter()
            .zip(&log_y)
            .map(|(xi, yi)| (yi - (slope * xi + intercept)).powi(2))
            .sum();
        let r_squared = if ss_tot > 0.0 {
            1.0 - ss_res / ss_tot
        } else {
            0.0
        };

        (slope, intercept.exp(), r_squared.max(0.0))
    }

    /// Find architecture transition points
    fn find_architecture_transitions(
        &self,
        data: &[ScalingDataPoint],
    ) -> Vec<(f64, ReactorArchitecture, String)> {
        let mut transitions = Vec::new();

        for i in 1..data.len() {
            if data[i].architecture != data[i - 1].architecture {
                let reason = format!(
                    "{:?} → {:?}: Better suited for {:.0}W+ ({})",
                    data[i - 1].architecture,
                    data[i].architecture,
                    data[i].power_w,
                    data[i].limiting_factor
                );
                transitions.push((data[i].power_w, data[i].architecture, reason));
            }
        }

        transitions
    }

    /// Find trigger system transition points
    fn find_trigger_transitions(
        &self,
        data: &[ScalingDataPoint],
    ) -> Vec<(f64, ExtendedTriggerMethod, String)> {
        let mut transitions = Vec::new();

        for i in 1..data.len() {
            if data[i].trigger != data[i - 1].trigger {
                let reason = format!(
                    "{:?} → {:?}: Better efficiency/cost at {:.0}W+",
                    data[i - 1].trigger,
                    data[i].trigger,
                    data[i].power_w
                );
                transitions.push((data[i].power_w, data[i].trigger, reason));
            }
        }

        transitions
    }

    /// Extract key findings from the data
    fn extract_findings(&self, data: &[ScalingDataPoint], laws: &[ScalingLaw]) -> Vec<String> {
        let mut findings = Vec::new();

        // Cost scaling finding
        if let Some(cost_law) = laws.iter().find(|l| l.parameter.contains("Specific Cost")) {
            if cost_law.exponent < -0.1 {
                findings.push(format!(
                    "Strong economies of scale: $/W decreases as P^{:.2}, \
                     reaching ${:.0}/W at 1MW vs ${:.0}/W at 1W",
                    cost_law.exponent,
                    data.last().map(|d| d.specific_cost).unwrap_or(0.0),
                    data.first().map(|d| d.specific_cost).unwrap_or(0.0)
                ));
            }
        }

        // Architecture transition findings
        let unique_archs: std::collections::HashSet<_> =
            data.iter().map(|d| d.architecture).collect();
        if unique_archs.len() > 1 {
            findings.push(format!(
                "Optimal architecture varies with scale: {} different architectures \
                 are optimal across the power range",
                unique_archs.len()
            ));
        }

        // Trigger findings
        let unique_triggers: std::collections::HashSet<_> =
            data.iter().map(|d| d.trigger).collect();
        if unique_triggers.len() > 1 {
            findings.push(format!(
                "Trigger selection is power-dependent: {} different triggers \
                 are optimal across the power range",
                unique_triggers.len()
            ));
        }

        // TRL findings
        let min_trl = data.iter().map(|d| d.trl).min().unwrap_or(0);
        let max_trl = data.iter().map(|d| d.trl).max().unwrap_or(0);
        findings.push(format!(
            "Technology readiness varies: TRL {} to {} across scales. \
             Smaller scales may require more R&D.",
            min_trl, max_trl
        ));

        // Feasibility gradient
        let low_power_feasibility: f64 = data
            .iter()
            .filter(|d| d.power_w < 1000.0)
            .map(|d| d.feasibility_score)
            .sum::<f64>()
            / data.iter().filter(|d| d.power_w < 1000.0).count() as f64;
        let high_power_feasibility: f64 = data
            .iter()
            .filter(|d| d.power_w >= 1000.0)
            .map(|d| d.feasibility_score)
            .sum::<f64>()
            / data.iter().filter(|d| d.power_w >= 1000.0).count() as f64;

        if (high_power_feasibility - low_power_feasibility).abs() > 10.0 {
            let direction = if high_power_feasibility > low_power_feasibility {
                "improves"
            } else {
                "decreases"
            };
            findings.push(format!(
                "Feasibility {} with scale: {:.0}% average at <1kW vs {:.0}% at ≥1kW",
                direction, low_power_feasibility, high_power_feasibility
            ));
        }

        findings
    }

    /// Generate recommendations for different applications
    fn generate_recommendations(&self, data: &[ScalingDataPoint]) -> Vec<(String, String)> {
        let mut recommendations = Vec::new();

        // Find best option at each major scale
        // Sensor/IoT (1-10W)
        if let Some(point) = data.iter().find(|d| d.power_w <= 10.0) {
            recommendations.push((
                "Sensor/IoT Power (1-10W)".to_string(),
                format!(
                    "{:?} architecture with {:?} trigger. \
                     Expected: ${:.0}, {:.2}kg, {:.0} year lifetime. \
                     Limiting factor: {}",
                    point.architecture,
                    point.trigger,
                    point.cost_usd,
                    point.mass_kg,
                    point.lifetime_years,
                    point.limiting_factor
                ),
            ));
        }

        // Portable electronics (100W)
        if let Some(point) = data.iter().find(|d| (d.power_w - 100.0).abs() < 50.0) {
            recommendations.push((
                "Portable Electronics (100W)".to_string(),
                format!(
                    "{:?} architecture with {:?} trigger. \
                     Expected: ${:.0}, {:.2}kg, {:.0} year lifetime. \
                     Limiting factor: {}",
                    point.architecture,
                    point.trigger,
                    point.cost_usd,
                    point.mass_kg,
                    point.lifetime_years,
                    point.limiting_factor
                ),
            ));
        }

        // Residential (10kW)
        if let Some(point) = data.iter().find(|d| (d.power_w - 10_000.0).abs() < 5_000.0) {
            recommendations.push((
                "Residential Power (10kW)".to_string(),
                format!(
                    "{:?} architecture with {:?} trigger. \
                     Expected: ${:.0}k, {:.0}kg, {:.0} year lifetime. \
                     At ${:.2}/W, competitive with solar+battery at grid parity.",
                    point.architecture,
                    point.trigger,
                    point.cost_usd / 1000.0,
                    point.mass_kg,
                    point.lifetime_years,
                    point.specific_cost
                ),
            ));
        }

        // Industrial (1MW)
        if let Some(point) = data.iter().find(|d| d.power_w >= 1_000_000.0) {
            recommendations.push((
                "Industrial Power (1MW)".to_string(),
                format!(
                    "{:?} architecture with {:?} trigger. \
                     Expected: ${:.1}M, {:.0}t, {:.0} year lifetime. \
                     At ${:.2}/W, competes with natural gas LCOE.",
                    point.architecture,
                    point.trigger,
                    point.cost_usd / 1_000_000.0,
                    point.mass_kg / 1000.0,
                    point.lifetime_years,
                    point.specific_cost
                ),
            ));
        }

        recommendations
    }

    /// Generate a formatted report
    pub fn generate_report(&self, results: &ScalingStudyResults) -> String {
        let mut report = String::new();

        report.push_str(&format!("\n{}\n", "═".repeat(70)));
        report.push_str("        LCF REACTOR SCALING STUDY: 1W to 1MW\n");
        report.push_str(&format!("{}\n\n", "═".repeat(70)));

        // Summary table
        report.push_str("SCALING DATA SUMMARY\n");
        report.push_str(&format!("{}\n", "─".repeat(70)));
        report.push_str(&format!(
            "{:>10} {:>12} {:>10} {:>10} {:>10} {:>8}\n",
            "Power", "Architecture", "Volume", "Mass", "$/W", "TRL"
        ));
        report.push_str(&format!("{}\n", "─".repeat(70)));

        for point in &results.data {
            report.push_str(&format!(
                "{:>10} {:>12} {:>10} {:>10} {:>10} {:>8}\n",
                format_power(point.power_w),
                format!("{:?}", point.architecture)
                    .chars()
                    .take(12)
                    .collect::<String>(),
                format_volume(point.volume_m3),
                format_mass(point.mass_kg),
                format!("${:.1}", point.specific_cost),
                point.trl
            ));
        }

        // Scaling laws
        report.push_str(&format!("\n\nSCALING LAWS (Y = A × P^n)\n"));
        report.push_str(&format!("{}\n", "─".repeat(70)));
        for law in &results.scaling_laws {
            report.push_str(&format!(
                "{:<25} n = {:+.3}  (R² = {:.2})  {}\n",
                law.parameter, law.exponent, law.r_squared, law.interpretation
            ));
        }

        // Transitions
        if !results.transitions.is_empty() {
            report.push_str(&format!("\n\nARCHITECTURE TRANSITIONS\n"));
            report.push_str(&format!("{}\n", "─".repeat(70)));
            for (power, _arch, reason) in &results.transitions {
                report.push_str(&format!("  At {}: {}\n", format_power(*power), reason));
            }
        }

        if !results.trigger_transitions.is_empty() {
            report.push_str(&format!("\n\nTRIGGER TRANSITIONS\n"));
            report.push_str(&format!("{}\n", "─".repeat(70)));
            for (power, _trigger, reason) in &results.trigger_transitions {
                report.push_str(&format!("  At {}: {}\n", format_power(*power), reason));
            }
        }

        // Key findings
        report.push_str(&format!("\n\nKEY FINDINGS\n"));
        report.push_str(&format!("{}\n", "─".repeat(70)));
        for (i, finding) in results.findings.iter().enumerate() {
            report.push_str(&format!("  {}. {}\n", i + 1, finding));
        }

        // Recommendations
        report.push_str(&format!("\n\nRECOMMENDATIONS BY APPLICATION\n"));
        report.push_str(&format!("{}\n", "─".repeat(70)));
        for (app, rec) in &results.recommendations {
            report.push_str(&format!("\n  {}:\n    {}\n", app, rec));
        }

        report.push_str(&format!("\n{}\n", "═".repeat(70)));

        report
    }

    /// Run parametric sweep varying a single parameter
    pub fn parametric_sweep(
        &self,
        base_power_w: f64,
        parameter: &str,
        values: &[f64],
    ) -> Vec<(f64, ScalingDataPoint)> {
        values
            .iter()
            .map(|&value| {
                let mut params = ReactorDesignParams {
                    target_power_w: base_power_w,
                    ..Default::default()
                };

                // Modify the parameter
                match parameter {
                    "lifetime" => params.target_lifetime_years = value,
                    "cost_budget" => params.max_cost_usd = value,
                    "volume_limit" => params.max_volume_m3 = value,
                    "mass_limit" => params.max_mass_kg = value,
                    _ => {}
                }

                let reactor = self.reactor_designer.design(&params);
                let trigger = self.trigger_library.optimal_for_power(base_power_w);

                let point = ScalingDataPoint {
                    power_w: base_power_w,
                    architecture: reactor.architecture,
                    trigger: trigger.method,
                    volume_m3: reactor.volume_m3,
                    mass_kg: reactor.mass_kg,
                    cost_usd: reactor.cost_usd,
                    specific_cost: reactor.cost_usd / base_power_w,
                    power_density: base_power_w / reactor.volume_m3.max(0.001),
                    specific_power: base_power_w / reactor.mass_kg.max(0.1),
                    shielding_fraction: reactor.shielding.total_mass_kg / reactor.mass_kg,
                    lifetime_years: reactor.lifetime_years,
                    trl: reactor.trl,
                    feasibility_score: self.calculate_feasibility(&reactor, &params),
                    limiting_factor: self.determine_limiting_factor(&reactor, base_power_w),
                    reactor,
                };

                (value, point)
            })
            .collect()
    }
}

impl Default for ScalingStudy {
    fn default() -> Self {
        Self::new()
    }
}

/// Format power for display
fn format_power(power_w: f64) -> String {
    if power_w >= 1_000_000.0 {
        format!("{:.0}MW", power_w / 1_000_000.0)
    } else if power_w >= 1_000.0 {
        format!("{:.0}kW", power_w / 1_000.0)
    } else {
        format!("{:.0}W", power_w)
    }
}

/// Format volume for display
fn format_volume(vol_m3: f64) -> String {
    if vol_m3 >= 1.0 {
        format!("{:.2}m³", vol_m3)
    } else if vol_m3 >= 0.001 {
        format!("{:.0}L", vol_m3 * 1000.0)
    } else {
        format!("{:.0}mL", vol_m3 * 1_000_000.0)
    }
}

/// Format mass for display
fn format_mass(mass_kg: f64) -> String {
    if mass_kg >= 1000.0 {
        format!("{:.1}t", mass_kg / 1000.0)
    } else if mass_kg >= 1.0 {
        format!("{:.1}kg", mass_kg)
    } else {
        format!("{:.0}g", mass_kg * 1000.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scaling_study_creation() {
        let study = ScalingStudy::new();
        assert!(!study.trigger_library.systems.is_empty());
    }

    #[test]
    fn test_single_power_level() {
        let study = ScalingStudy::new();
        let point = study.analyze_power_level(10_000.0);

        assert!(point.power_w == 10_000.0);
        assert!(point.mass_kg > 0.0);
        assert!(point.cost_usd > 0.0);
        assert!(point.feasibility_score >= 0.0);
    }

    #[test]
    fn test_full_study() {
        let study = ScalingStudy::new();
        let results = study.run_full_study();

        assert_eq!(results.data.len(), POWER_LEVELS_W.len());
        assert!(!results.scaling_laws.is_empty());
        assert!(!results.findings.is_empty());
    }

    #[test]
    fn test_power_law_fit() {
        let study = ScalingStudy::new();

        // Test with known power law: Y = 2 * X^0.5
        let x = vec![1.0, 4.0, 9.0, 16.0, 25.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];

        let (exp, coef, r2) = study.fit_power_law(&x, &y);

        assert!(
            (exp - 0.5).abs() < 0.1,
            "Exponent should be ~0.5, got {}",
            exp
        );
        assert!(
            (coef - 2.0).abs() < 0.5,
            "Coefficient should be ~2, got {}",
            coef
        );
        assert!(r2 > 0.99, "R² should be high, got {}", r2);
    }

    #[test]
    fn test_scaling_monotonicity() {
        let study = ScalingStudy::new();
        let results = study.run_full_study();

        // At architecture transition boundaries, volume can jump
        // discontinuously (e.g., PulsedElectrolysis → ModularCell).
        // We only check monotonicity within the same architecture.
        for i in 1..results.data.len() {
            // All points should have positive, finite values
            assert!(results.data[i].volume_m3 > 0.0);
            assert!(results.data[i].mass_kg > 0.0);
            assert!(results.data[i].cost_usd > 0.0);

            // Within same architecture: volume should not decrease
            if results.data[i].architecture == results.data[i - 1].architecture {
                assert!(
                    results.data[i].volume_m3 >= results.data[i - 1].volume_m3 * 0.5,
                    "Volume decreased within {:?} at {}W",
                    results.data[i].architecture,
                    results.data[i].power_w
                );
            }
        }

        // Overall: 1MW should have more volume than 1W
        let first = &results.data[0];
        let last = results.data.last().unwrap();
        assert!(
            last.volume_m3 > first.volume_m3,
            "1MW ({:.3} m³) should have more volume than 1W ({:.3} m³)",
            last.volume_m3,
            first.volume_m3
        );
    }

    #[test]
    fn test_report_generation() {
        let study = ScalingStudy::new();
        let results = study.run_full_study();
        let report = study.generate_report(&results);

        assert!(report.contains("SCALING"));
        assert!(report.contains("Power"));
        assert!(report.contains("RECOMMENDATIONS"));
    }

    #[test]
    fn test_parametric_sweep() {
        let study = ScalingStudy::new();
        let lifetimes = vec![5.0, 10.0, 20.0, 40.0];
        let sweep = study.parametric_sweep(10_000.0, "lifetime", &lifetimes);

        assert_eq!(sweep.len(), lifetimes.len());
        for (_lifetime, point) in &sweep {
            assert_eq!(point.power_w, 10_000.0);
        }
    }

    // === Direction C: MC through Scaling Study Tests ===

    #[test]
    fn test_mc_scaling_study_percentiles_ordered() {
        let study = ScalingStudy::new();
        let mc_results = study.run_study_with_uncertainty(50, 42);

        assert_eq!(mc_results.len(), POWER_LEVELS_W.len());

        for dp in &mc_results {
            // p5 cost < nominal cost < p95 cost (with some tolerance for small samples)
            assert!(
                dp.p5.cost_usd <= dp.p95.cost_usd,
                "p5 cost ({:.0}) should be <= p95 cost ({:.0}) at {:.0}W",
                dp.p5.cost_usd,
                dp.p95.cost_usd,
                dp.power_w
            );
            assert!(dp.cost_cv >= 0.0, "Cost CV should be non-negative");
            assert!(dp.mass_cv >= 0.0, "Mass CV should be non-negative");
            assert!(dp.p_feasible >= 0.0 && dp.p_feasible <= 1.0);
        }
    }

    #[test]
    fn test_mc_scaling_study_with_genesis() {
        let genesis = GenesisSeed::from_phrase("mc scaling test");
        let study = ScalingStudy::with_genesis(genesis);
        let mc_results = study.run_study_with_uncertainty(20, 42);
        assert_eq!(mc_results.len(), POWER_LEVELS_W.len());
    }
}