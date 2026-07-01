// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Design Space Mapping
//!
//! Parametric sweeps to map feasible design regions for Spark Engine:
//! - Power vs shielding thickness
//! - Duty cycle vs lifetime
//! - Material trade studies
//!
//! ## Design Envelope
//!
//! ```text
//! Shielding (cm)
//!     │
//! 150 ┤     ░░░░░░░░░░░░░░░░  FEASIBLE
//!     │   ░░░░░░░░░░░░░░░░░░
//! 100 ┤ ░░░░░░░░░░░░░░░░░░░░
//!     │ ░░░░░░░░░░░░░░░░░░
//!  50 ┤ ░░░░░░░░░░░░░░
//!     │ ░░░░░░░░░░            INFEASIBLE (dose too high)
//!   0 ┼─────────────────────► Power (kW)
//!     0    10   20   30   40   50
//! ```

use super::coupled_physics::{CoupledPhysicsEngine, OperatingConditions};
use super::radiation_damage::FusionReaction;
use crate::genesis::GenesisSeed;

/// Result of a parametric sweep point
#[derive(Debug, Clone)]
pub struct DesignPoint {
    /// Power level (kW)
    pub power_kw: f64,
    /// Other parameter value
    pub param_value: f64,
    /// Is this point feasible?
    pub feasible: bool,
    /// Key metric value (e.g., lifetime, dose, cost)
    pub metric_value: f64,
    /// Notes on limiting factors
    pub notes: String,
}

/// Type of parametric sweep
#[derive(Debug, Clone, Copy)]
pub enum SweepType {
    /// Power vs shielding thickness
    PowerVsShielding,
    /// Duty cycle vs lifetime
    DutyCycleVsLifetime,
    /// Power vs system mass
    PowerVsMass,
    /// Power vs cost per kW
    PowerVsCost,
}

/// Parametric sweep result
#[derive(Debug, Clone)]
pub struct ParametricSweep {
    /// Sweep type
    pub sweep_type: SweepType,
    /// Reaction type
    pub reaction: FusionReaction,
    /// X-axis values (e.g., power levels)
    pub x_values: Vec<f64>,
    /// X-axis label
    pub x_label: String,
    /// Y-axis label
    pub y_label: String,
    /// Design points
    pub points: Vec<DesignPoint>,
    /// Feasibility boundary (power, threshold)
    pub feasibility_boundary: Vec<(f64, f64)>,
}

/// Design space mapper
pub struct DesignSpaceMapper {
    _genesis: GenesisSeed,
    engine: CoupledPhysicsEngine,
}

impl DesignSpaceMapper {
    /// Create from genesis
    pub fn from_genesis(genesis: &GenesisSeed) -> Self {
        Self {
            _genesis: genesis.clone(),
            engine: CoupledPhysicsEngine::from_genesis(genesis),
        }
    }

    /// Sweep power vs shielding thickness
    pub fn sweep_power_vs_shielding(
        &self,
        reaction: FusionReaction,
        power_range: (f64, f64),
        n_points: usize,
    ) -> ParametricSweep {
        let mut points = Vec::new();
        let mut boundary = Vec::new();

        let power_step = (power_range.1 - power_range.0) / (n_points as f64 - 1.0);

        for i in 0..n_points {
            let power = power_range.0 + power_step * (i as f64);

            let conditions = OperatingConditions {
                power_kw: power,
                reaction,
                ambient_temp_k: 300.0,
                h_convection: if power > 1000.0 { 500.0 } else { 50.0 },
                target_lifetime_years: 25.0,
                max_dose_rate: 0.001,
            };

            let result = self.engine.simulate(&conditions);

            let shielding_cm = result.geometry_shielding.shielding.thickness_m * 100.0;

            points.push(DesignPoint {
                power_kw: power,
                param_value: shielding_cm,
                feasible: result.feasible,
                metric_value: result.geometry_shielding.shielding.final_dose,
                notes: if result.feasible {
                    "OK".to_string()
                } else {
                    result.limiting_factors.join("; ")
                },
            });

            // Track feasibility boundary
            if result.feasible {
                boundary.push((power, shielding_cm));
            }
        }

        ParametricSweep {
            sweep_type: SweepType::PowerVsShielding,
            reaction,
            x_values: points.iter().map(|p| p.power_kw).collect(),
            x_label: "Power (kW)".to_string(),
            y_label: "Shielding (cm)".to_string(),
            points,
            feasibility_boundary: boundary,
        }
    }

    /// Sweep power vs system mass
    pub fn sweep_power_vs_mass(
        &self,
        reaction: FusionReaction,
        power_range: (f64, f64),
        n_points: usize,
    ) -> ParametricSweep {
        let mut points = Vec::new();
        let mut boundary = Vec::new();

        let power_step = (power_range.1 - power_range.0) / (n_points as f64 - 1.0);

        for i in 0..n_points {
            let power = power_range.0 + power_step * (i as f64);

            let conditions = OperatingConditions {
                power_kw: power,
                reaction,
                ambient_temp_k: 300.0,
                h_convection: if power > 1000.0 { 500.0 } else { 50.0 },
                target_lifetime_years: 25.0,
                max_dose_rate: 0.001,
            };

            let result = self.engine.simulate(&conditions);

            let mass_kg = result.geometry_shielding.total_mass_kg;

            points.push(DesignPoint {
                power_kw: power,
                param_value: mass_kg,
                feasible: result.feasible,
                metric_value: mass_kg / power, // kg per kW
                notes: format!("{:.1} kg/kW", mass_kg / power),
            });

            if result.feasible {
                boundary.push((power, mass_kg));
            }
        }

        ParametricSweep {
            sweep_type: SweepType::PowerVsMass,
            reaction,
            x_values: points.iter().map(|p| p.power_kw).collect(),
            x_label: "Power (kW)".to_string(),
            y_label: "System Mass (kg)".to_string(),
            points,
            feasibility_boundary: boundary,
        }
    }

    /// Sweep power vs cost per kW
    pub fn sweep_power_vs_cost(
        &self,
        reaction: FusionReaction,
        power_range: (f64, f64),
        n_points: usize,
    ) -> ParametricSweep {
        let mut points = Vec::new();
        let mut boundary = Vec::new();

        let power_step = (power_range.1 - power_range.0) / (n_points as f64 - 1.0);

        for i in 0..n_points {
            let power = power_range.0 + power_step * (i as f64);

            let conditions = OperatingConditions {
                power_kw: power,
                reaction,
                ambient_temp_k: 300.0,
                h_convection: if power > 1000.0 { 500.0 } else { 50.0 },
                target_lifetime_years: 25.0,
                max_dose_rate: 0.001,
            };

            let result = self.engine.simulate(&conditions);

            // Estimate cost
            let spec =
                super::spark_prototype_spec::PrototypeSpecification::from_simulation(&result);
            let cost_per_kw = spec.estimated_total_cost_usd / power;

            points.push(DesignPoint {
                power_kw: power,
                param_value: cost_per_kw,
                feasible: result.feasible,
                metric_value: cost_per_kw,
                notes: format!("${cost_per_kw:.0}/kW"),
            });

            if result.feasible {
                boundary.push((power, cost_per_kw));
            }
        }

        ParametricSweep {
            sweep_type: SweepType::PowerVsCost,
            reaction,
            x_values: points.iter().map(|p| p.power_kw).collect(),
            x_label: "Power (kW)".to_string(),
            y_label: "Cost (USD/kW)".to_string(),
            points,
            feasibility_boundary: boundary,
        }
    }

    /// Find minimum feasible power for given reaction
    pub fn find_minimum_feasible_power(&self, reaction: FusionReaction) -> Option<f64> {
        // Binary search for minimum feasible power
        let mut low = 0.1;
        let mut high = 100.0;

        for _ in 0..20 {
            // 20 iterations for precision
            let mid = (low + high) / 2.0;

            let conditions = OperatingConditions {
                power_kw: mid,
                reaction,
                ambient_temp_k: 300.0,
                h_convection: 50.0,
                target_lifetime_years: 25.0,
                max_dose_rate: 0.001,
            };

            let result = self.engine.simulate(&conditions);

            if result.feasible {
                high = mid;
            } else {
                low = mid;
            }
        }

        let conditions = OperatingConditions {
            power_kw: high,
            reaction,
            ambient_temp_k: 300.0,
            h_convection: 50.0,
            target_lifetime_years: 25.0,
            max_dose_rate: 0.001,
        };

        let result = self.engine.simulate(&conditions);
        if result.feasible { Some(high) } else { None }
    }

    /// Generate ASCII chart of sweep results
    pub fn ascii_chart(&self, sweep: &ParametricSweep, width: usize, height: usize) -> String {
        let mut output = String::new();

        // Find data range
        let x_min = sweep
            .points
            .iter()
            .map(|p| p.power_kw)
            .fold(f64::INFINITY, f64::min);
        let x_max = sweep
            .points
            .iter()
            .map(|p| p.power_kw)
            .fold(f64::NEG_INFINITY, f64::max);
        let y_min = sweep
            .points
            .iter()
            .map(|p| p.param_value)
            .fold(f64::INFINITY, f64::min);
        let y_max = sweep
            .points
            .iter()
            .map(|p| p.param_value)
            .fold(f64::NEG_INFINITY, f64::max);

        // Add margin
        let y_range = y_max - y_min;
        let y_max = y_max + y_range * 0.1;
        let y_min = (y_min - y_range * 0.1).max(0.0);

        // Title
        output.push_str(&format!(
            "\n{:?} ({:?})\n",
            sweep.sweep_type, sweep.reaction
        ));
        output.push_str(&"─".repeat(width + 10));
        output.push('\n');

        // Y-axis label width
        let _y_label_width = 8;

        // Create chart grid
        for row in 0..height {
            let y_val = y_max - (y_max - y_min) * (row as f64 / (height - 1) as f64);

            // Y-axis label
            if row == 0 || row == height / 2 || row == height - 1 {
                output.push_str(&format!("{y_val:>7.0} │"));
            } else {
                output.push_str(&format!("{:>7} │", ""));
            }

            // Plot points
            for col in 0..width {
                let x_val = x_min + (x_max - x_min) * (col as f64 / (width - 1) as f64);

                // Find nearest point
                let nearest = sweep.points.iter().min_by(|a, b| {
                    let da = (a.power_kw - x_val).abs();
                    let db = (b.power_kw - x_val).abs();
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                });

                if let Some(point) = nearest {
                    let px = (point.power_kw - x_min) / (x_max - x_min) * (width - 1) as f64;
                    let py = (y_max - point.param_value) / (y_max - y_min) * (height - 1) as f64;

                    if (px - col as f64).abs() < 0.6 && (py - row as f64).abs() < 0.6 {
                        if point.feasible {
                            output.push('●');
                        } else {
                            output.push('○');
                        }
                    } else {
                        output.push(' ');
                    }
                } else {
                    output.push(' ');
                }
            }
            output.push('\n');
        }

        // X-axis
        output.push_str(&format!("{:>7} └", ""));
        output.push_str(&"─".repeat(width));
        output.push('\n');

        // X-axis labels
        output.push_str(&format!("{:>7}  ", ""));
        output.push_str(&format!(
            "{:<width$}",
            format!("{:.0}", x_min),
            width = width / 3
        ));
        output.push_str(&format!(
            "{:^width$}",
            format!("{:.0}", (x_min + x_max) / 2.0),
            width = width / 3
        ));
        output.push_str(&format!(
            "{:>width$}",
            format!("{:.0}", x_max),
            width = width / 3
        ));
        output.push('\n');

        // Axis labels
        output.push_str(&format!("\n{:>7}  {}\n", "", sweep.x_label));

        // Legend
        output.push_str("\nLegend: ● = Feasible, ○ = Needs Revision\n");

        // Summary stats
        let feasible_count = sweep.points.iter().filter(|p| p.feasible).count();
        output.push_str(&format!(
            "\nFeasible: {}/{} points\n",
            feasible_count,
            sweep.points.len()
        ));

        output
    }
}

/// Design envelope summary
#[derive(Debug, Clone)]
pub struct DesignEnvelope {
    /// Reaction type
    pub reaction: FusionReaction,
    /// Minimum feasible power (kW)
    pub min_power_kw: Option<f64>,
    /// Maximum tested power (kW)
    pub max_power_kw: f64,
    /// Shielding range for feasible designs (cm)
    pub shielding_range_cm: (f64, f64),
    /// Mass range for feasible designs (kg)
    pub mass_range_kg: (f64, f64),
    /// Cost range (USD/kW)
    pub cost_range_usd_kw: (f64, f64),
    /// Key constraints
    pub constraints: Vec<String>,
}

impl DesignEnvelope {
    /// Generate from sweeps
    pub fn from_sweeps(
        reaction: FusionReaction,
        shielding_sweep: &ParametricSweep,
        mass_sweep: &ParametricSweep,
        cost_sweep: &ParametricSweep,
    ) -> Self {
        let feasible_shielding: Vec<_> = shielding_sweep
            .points
            .iter()
            .filter(|p| p.feasible)
            .collect();

        let feasible_mass: Vec<_> = mass_sweep.points.iter().filter(|p| p.feasible).collect();

        let feasible_cost: Vec<_> = cost_sweep.points.iter().filter(|p| p.feasible).collect();

        let min_power = feasible_shielding
            .iter()
            .map(|p| p.power_kw)
            .fold(f64::INFINITY, f64::min);

        let shielding_min = feasible_shielding
            .iter()
            .map(|p| p.param_value)
            .fold(f64::INFINITY, f64::min);
        let shielding_max = feasible_shielding
            .iter()
            .map(|p| p.param_value)
            .fold(f64::NEG_INFINITY, f64::max);

        let mass_min = feasible_mass
            .iter()
            .map(|p| p.param_value)
            .fold(f64::INFINITY, f64::min);
        let mass_max = feasible_mass
            .iter()
            .map(|p| p.param_value)
            .fold(f64::NEG_INFINITY, f64::max);

        let cost_min = feasible_cost
            .iter()
            .map(|p| p.param_value)
            .fold(f64::INFINITY, f64::min);
        let cost_max = feasible_cost
            .iter()
            .map(|p| p.param_value)
            .fold(f64::NEG_INFINITY, f64::max);

        let mut constraints = Vec::new();
        constraints.push("Public dose rate < 0.001 mSv/hr".to_string());
        constraints.push("Shell temperature < 1200 K".to_string());
        constraints.push("25 year lifetime target".to_string());

        Self {
            reaction,
            min_power_kw: if min_power.is_finite() {
                Some(min_power)
            } else {
                None
            },
            max_power_kw: shielding_sweep
                .points
                .iter()
                .map(|p| p.power_kw)
                .fold(f64::NEG_INFINITY, f64::max),
            shielding_range_cm: (shielding_min, shielding_max),
            mass_range_kg: (mass_min, mass_max),
            cost_range_usd_kw: (cost_min, cost_max),
            constraints,
        }
    }
}

// ============================================================================
// PARETO MULTI-OBJECTIVE OPTIMIZATION
// ============================================================================

/// A design point with multiple objective values for Pareto analysis
#[derive(Debug, Clone)]
pub struct MultiObjectiveDesign {
    /// Power level (kW)
    pub power_kw: f64,
    /// System mass (kg) - minimize
    pub mass_kg: f64,
    /// Cost per kW (USD/kW) - minimize
    pub cost_per_kw: f64,
    /// Dose rate (mSv/hr) - minimize
    pub dose_rate: f64,
    /// Lifetime (years) - maximize
    pub lifetime_years: f64,
    /// Is feasible (meets all constraints)?
    pub feasible: bool,
    /// Pareto rank (0 = frontier, 1 = dominated by 1, etc.)
    pub pareto_rank: usize,
    /// Crowding distance for diversity preservation
    pub crowding_distance: f64,
}

impl MultiObjectiveDesign {
    /// Check if this design dominates another
    /// A dominates B if A is better or equal on all objectives and strictly better on at least one
    pub fn dominates(&self, other: &Self) -> bool {
        // Only feasible designs can dominate
        if !self.feasible {
            return false;
        }
        if !other.feasible {
            return true; // Feasible always dominates infeasible
        }

        // Check objectives (all minimize except lifetime which we maximize)
        let dominated_mass = self.mass_kg <= other.mass_kg;
        let dominated_cost = self.cost_per_kw <= other.cost_per_kw;
        let dominated_dose = self.dose_rate <= other.dose_rate;
        let dominated_lifetime = self.lifetime_years >= other.lifetime_years;

        let all_dominated =
            dominated_mass && dominated_cost && dominated_dose && dominated_lifetime;

        let strictly_better = self.mass_kg < other.mass_kg
            || self.cost_per_kw < other.cost_per_kw
            || self.dose_rate < other.dose_rate
            || self.lifetime_years > other.lifetime_years;

        all_dominated && strictly_better
    }

    /// Normalized objective vector for distance calculations
    pub fn objectives_normalized(&self, min_max: &ObjectiveRanges) -> [f64; 4] {
        [
            (self.mass_kg - min_max.mass_min) / (min_max.mass_max - min_max.mass_min + 1e-10),
            (self.cost_per_kw - min_max.cost_min) / (min_max.cost_max - min_max.cost_min + 1e-10),
            (self.dose_rate - min_max.dose_min) / (min_max.dose_max - min_max.dose_min + 1e-10),
            // Invert lifetime since we maximize it
            1.0 - (self.lifetime_years - min_max.lifetime_min)
                / (min_max.lifetime_max - min_max.lifetime_min + 1e-10),
        ]
    }
}

/// Ranges for objective normalization
#[derive(Debug, Clone)]
pub struct ObjectiveRanges {
    pub mass_min: f64,
    pub mass_max: f64,
    pub cost_min: f64,
    pub cost_max: f64,
    pub dose_min: f64,
    pub dose_max: f64,
    pub lifetime_min: f64,
    pub lifetime_max: f64,
}

impl ObjectiveRanges {
    fn from_designs(designs: &[MultiObjectiveDesign]) -> Self {
        Self {
            mass_min: designs
                .iter()
                .map(|d| d.mass_kg)
                .fold(f64::INFINITY, f64::min),
            mass_max: designs
                .iter()
                .map(|d| d.mass_kg)
                .fold(f64::NEG_INFINITY, f64::max),
            cost_min: designs
                .iter()
                .map(|d| d.cost_per_kw)
                .fold(f64::INFINITY, f64::min),
            cost_max: designs
                .iter()
                .map(|d| d.cost_per_kw)
                .fold(f64::NEG_INFINITY, f64::max),
            dose_min: designs
                .iter()
                .map(|d| d.dose_rate)
                .fold(f64::INFINITY, f64::min),
            dose_max: designs
                .iter()
                .map(|d| d.dose_rate)
                .fold(f64::NEG_INFINITY, f64::max),
            lifetime_min: designs
                .iter()
                .map(|d| d.lifetime_years)
                .fold(f64::INFINITY, f64::min),
            lifetime_max: designs
                .iter()
                .map(|d| d.lifetime_years)
                .fold(f64::NEG_INFINITY, f64::max),
        }
    }
}

/// Pareto frontier result
#[derive(Debug, Clone)]
pub struct ParetoFrontier {
    /// All evaluated designs
    pub all_designs: Vec<MultiObjectiveDesign>,
    /// Pareto-optimal designs (rank 0)
    pub frontier: Vec<MultiObjectiveDesign>,
    /// Objective ranges for context
    pub ranges: ObjectiveRanges,
    /// Best designs by each objective
    pub best_by_objective: BestByObjective,
}

/// Best designs for each individual objective
#[derive(Debug, Clone)]
pub struct BestByObjective {
    pub min_mass: Option<MultiObjectiveDesign>,
    pub min_cost: Option<MultiObjectiveDesign>,
    pub min_dose: Option<MultiObjectiveDesign>,
    pub max_lifetime: Option<MultiObjectiveDesign>,
}

/// Pareto optimizer for design space exploration
pub struct ParetoOptimizer {
    _genesis: GenesisSeed,
    engine: CoupledPhysicsEngine,
}

impl ParetoOptimizer {
    /// Create from genesis
    pub fn from_genesis(genesis: &GenesisSeed) -> Self {
        Self {
            _genesis: genesis.clone(),
            engine: CoupledPhysicsEngine::from_genesis(genesis),
        }
    }

    /// Evaluate a single design point
    fn evaluate_design(&self, power_kw: f64, reaction: FusionReaction) -> MultiObjectiveDesign {
        let conditions = OperatingConditions {
            power_kw,
            reaction,
            ambient_temp_k: 300.0,
            h_convection: if power_kw < 100.0 { 50.0 } else { 500.0 },
            target_lifetime_years: 25.0,
            max_dose_rate: 0.001,
        };

        let result = self.engine.simulate(&conditions);

        // Calculate cost per kW (simplified model)
        let base_cost = 50000.0 + power_kw * 1000.0; // Base + linear
        let mass_cost = result.geometry_shielding.total_mass_kg * 100.0; // $100/kg
        let total_cost = base_cost + mass_cost;
        let cost_per_kw = total_cost / power_kw;

        // Get lifetime
        let lifetime = result
            .pulse_thermal
            .lifetime_years
            .min(result.pulse_thermal.fatigue_lifetime_years)
            .min(100.0);

        MultiObjectiveDesign {
            power_kw,
            mass_kg: result.geometry_shielding.total_mass_kg,
            cost_per_kw,
            dose_rate: result.geometry_shielding.shielding.final_dose,
            lifetime_years: lifetime,
            feasible: result.feasible,
            pareto_rank: 0,
            crowding_distance: 0.0,
        }
    }

    /// Sweep design space and compute Pareto frontier
    pub fn compute_pareto_frontier(
        &self,
        reaction: FusionReaction,
        power_range: (f64, f64),
        n_points: usize,
    ) -> ParetoFrontier {
        // Evaluate all design points
        let mut designs: Vec<MultiObjectiveDesign> = Vec::with_capacity(n_points);

        let power_step = (power_range.1 - power_range.0) / (n_points as f64 - 1.0).max(1.0);

        for i in 0..n_points {
            let power = power_range.0 + i as f64 * power_step;
            let design = self.evaluate_design(power, reaction);
            designs.push(design);
        }

        // Compute Pareto ranks
        self.assign_pareto_ranks(&mut designs);

        // Compute crowding distance for diversity
        self.compute_crowding_distance(&mut designs);

        // Extract frontier
        let frontier: Vec<MultiObjectiveDesign> = designs
            .iter()
            .filter(|d| d.pareto_rank == 0 && d.feasible)
            .cloned()
            .collect();

        // Find best by each objective
        let feasible: Vec<_> = designs.iter().filter(|d| d.feasible).collect();

        let best_by_objective = BestByObjective {
            min_mass: feasible
                .iter()
                .min_by(|a, b| {
                    a.mass_kg
                        .partial_cmp(&b.mass_kg)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|d| (*d).clone()),
            min_cost: feasible
                .iter()
                .min_by(|a, b| {
                    a.cost_per_kw
                        .partial_cmp(&b.cost_per_kw)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|d| (*d).clone()),
            min_dose: feasible
                .iter()
                .min_by(|a, b| {
                    a.dose_rate
                        .partial_cmp(&b.dose_rate)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|d| (*d).clone()),
            max_lifetime: feasible
                .iter()
                .max_by(|a, b| {
                    a.lifetime_years
                        .partial_cmp(&b.lifetime_years)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|d| (*d).clone()),
        };

        let ranges = ObjectiveRanges::from_designs(&designs);

        ParetoFrontier {
            all_designs: designs,
            frontier,
            ranges,
            best_by_objective,
        }
    }

    /// Assign Pareto ranks using non-dominated sorting
    fn assign_pareto_ranks(&self, designs: &mut [MultiObjectiveDesign]) {
        let n = designs.len();
        let mut ranks = vec![0usize; n];
        let mut domination_count = vec![0usize; n];
        let mut dominated_by = vec![Vec::new(); n];

        // Count dominations
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                if designs[i].dominates(&designs[j]) {
                    dominated_by[i].push(j);
                } else if designs[j].dominates(&designs[i]) {
                    domination_count[i] += 1;
                }
            }
        }

        // Assign ranks iteratively
        let mut current_front: Vec<usize> = (0..n).filter(|&i| domination_count[i] == 0).collect();

        let mut rank = 0;
        while !current_front.is_empty() {
            let mut next_front = Vec::new();

            for &i in &current_front {
                ranks[i] = rank;

                for &j in &dominated_by[i] {
                    domination_count[j] = domination_count[j].saturating_sub(1);
                    if domination_count[j] == 0 {
                        next_front.push(j);
                    }
                }
            }

            rank += 1;
            current_front = next_front;
        }

        // Apply ranks
        for (i, design) in designs.iter_mut().enumerate() {
            design.pareto_rank = ranks[i];
        }
    }

    /// Compute crowding distance for diversity preservation
    fn compute_crowding_distance(&self, designs: &mut [MultiObjectiveDesign]) {
        let ranges = ObjectiveRanges::from_designs(designs);

        // Group by rank
        let max_rank = designs.iter().map(|d| d.pareto_rank).max().unwrap_or(0);

        for rank in 0..=max_rank {
            let mut indices: Vec<usize> = designs
                .iter()
                .enumerate()
                .filter(|(_, d)| d.pareto_rank == rank)
                .map(|(i, _)| i)
                .collect();

            if indices.len() < 3 {
                for &i in &indices {
                    designs[i].crowding_distance = f64::INFINITY;
                }
                continue;
            }

            // Initialize distances
            for &i in &indices {
                designs[i].crowding_distance = 0.0;
            }

            // For each objective, sort and add distance
            for obj in 0..4 {
                indices.sort_by(|&a, &b| {
                    let va = designs[a].objectives_normalized(&ranges)[obj];
                    let vb = designs[b].objectives_normalized(&ranges)[obj];
                    va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
                });

                // Boundary points get infinite distance
                designs[indices[0]].crowding_distance = f64::INFINITY;
                designs[indices[indices.len() - 1]].crowding_distance = f64::INFINITY;

                // Interior points
                for k in 1..(indices.len() - 1) {
                    let prev = designs[indices[k - 1]].objectives_normalized(&ranges)[obj];
                    let next = designs[indices[k + 1]].objectives_normalized(&ranges)[obj];
                    designs[indices[k]].crowding_distance += (next - prev).abs();
                }
            }
        }
    }

    /// Format Pareto frontier as ASCII report
    pub fn format_report(&self, pareto: &ParetoFrontier) -> String {
        let mut output = String::new();

        output.push_str("\n════════════════════════════════════════════════════════════════\n");
        output.push_str("                    PARETO FRONTIER ANALYSIS\n");
        output.push_str("════════════════════════════════════════════════════════════════\n\n");

        output.push_str(&format!(
            "Total designs evaluated: {}\n",
            pareto.all_designs.len()
        ));
        output.push_str(&format!(
            "Feasible designs: {}\n",
            pareto.all_designs.iter().filter(|d| d.feasible).count()
        ));
        output.push_str(&format!(
            "Pareto-optimal designs: {}\n\n",
            pareto.frontier.len()
        ));

        output.push_str("┌────────────────────────────────────────────────────────────────┐\n");
        output.push_str("│  PARETO FRONTIER (Non-Dominated Feasible Designs)             │\n");
        output.push_str("├────────┬────────┬──────────┬────────────┬──────────┬──────────┤\n");
        output.push_str("│ Power  │  Mass  │ Cost/kW  │ Dose Rate  │ Lifetime │ Crowding │\n");
        output.push_str("│  (kW)  │  (kg)  │ ($/kW)   │ (mSv/hr)   │ (years)  │ Distance │\n");
        output.push_str("├────────┼────────┼──────────┼────────────┼──────────┼──────────┤\n");

        for design in &pareto.frontier {
            output.push_str(&format!(
                "│ {:>6.1} │ {:>6.0} │ {:>8.0} │ {:>10.4} │ {:>8.0} │ {:>8.2} │\n",
                design.power_kw,
                design.mass_kg,
                design.cost_per_kw,
                design.dose_rate,
                design.lifetime_years.min(100.0),
                if design.crowding_distance.is_infinite() {
                    f64::NAN
                } else {
                    design.crowding_distance
                }
            ));
        }
        output.push_str("└────────┴────────┴──────────┴────────────┴──────────┴──────────┘\n");

        output.push_str("\n┌────────────────────────────────────────────────────────────────┐\n");
        output.push_str("│  BEST BY OBJECTIVE (Single-Objective Optima)                   │\n");
        output.push_str("├────────────────────────────────────────────────────────────────┤\n");

        if let Some(ref d) = pareto.best_by_objective.min_mass {
            output.push_str(&format!(
                "│  Min Mass:     {:>6.0} kg at {:>5.1} kW                         │\n",
                d.mass_kg, d.power_kw
            ));
        }
        if let Some(ref d) = pareto.best_by_objective.min_cost {
            output.push_str(&format!(
                "│  Min Cost:     ${:>5.0}/kW at {:>5.1} kW                        │\n",
                d.cost_per_kw, d.power_kw
            ));
        }
        if let Some(ref d) = pareto.best_by_objective.min_dose {
            output.push_str(&format!(
                "│  Min Dose:     {:>.4} mSv/hr at {:>5.1} kW                     │\n",
                d.dose_rate, d.power_kw
            ));
        }
        if let Some(ref d) = pareto.best_by_objective.max_lifetime {
            output.push_str(&format!(
                "│  Max Lifetime: {:>5.0} years at {:>5.1} kW                      │\n",
                d.lifetime_years.min(100.0),
                d.power_kw
            ));
        }
        output.push_str("└────────────────────────────────────────────────────────────────┘\n");

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> DesignSpaceMapper {
        let genesis = GenesisSeed::from_phrase("design space test");
        DesignSpaceMapper::from_genesis(&genesis)
    }

    #[test]
    fn test_power_vs_shielding() {
        let mapper = setup();
        let sweep = mapper.sweep_power_vs_shielding(FusionReaction::DD, (1.0, 10.0), 5);

        assert_eq!(sweep.points.len(), 5);
        println!("Sweep results:");
        for point in &sweep.points {
            println!(
                "  {:.1} kW: {:.1} cm shielding, feasible={}",
                point.power_kw, point.param_value, point.feasible
            );
        }
    }

    #[test]
    fn test_minimum_feasible_power() {
        let mapper = setup();
        let min_power = mapper.find_minimum_feasible_power(FusionReaction::DD);

        if let Some(power) = min_power {
            println!("Minimum feasible power (DD): {:.2} kW", power);
            assert!(power > 0.0);
        }
    }

    #[test]
    fn test_ascii_chart() {
        let mapper = setup();
        let sweep = mapper.sweep_power_vs_shielding(FusionReaction::DD, (1.0, 20.0), 10);
        let chart = mapper.ascii_chart(&sweep, 40, 15);

        println!("{}", chart);
        assert!(chart.contains("Feasible"));
    }

    #[test]
    fn test_pareto_optimization() {
        let genesis = GenesisSeed::from_phrase("pareto test");
        let optimizer = ParetoOptimizer::from_genesis(&genesis);

        let pareto = optimizer.compute_pareto_frontier(FusionReaction::DD, (1.0, 20.0), 10);

        println!("Pareto frontier has {} designs", pareto.frontier.len());
        println!("{}", optimizer.format_report(&pareto));

        // Should have some designs evaluated
        assert!(!pareto.all_designs.is_empty());

        // Pareto rank 0 designs should not dominate each other
        for i in 0..pareto.frontier.len() {
            for j in 0..pareto.frontier.len() {
                if i != j {
                    assert!(
                        !pareto.frontier[i].dominates(&pareto.frontier[j]),
                        "Frontier design {} dominates {}",
                        i,
                        j
                    );
                }
            }
        }
    }

    #[test]
    fn test_dominance() {
        let a = MultiObjectiveDesign {
            power_kw: 5.0,
            mass_kg: 1000.0,
            cost_per_kw: 10000.0,
            dose_rate: 0.001,
            lifetime_years: 50.0,
            feasible: true,
            pareto_rank: 0,
            crowding_distance: 0.0,
        };

        let b = MultiObjectiveDesign {
            power_kw: 5.0,
            mass_kg: 1500.0,      // Worse
            cost_per_kw: 15000.0, // Worse
            dose_rate: 0.002,     // Worse
            lifetime_years: 40.0, // Worse
            feasible: true,
            pareto_rank: 0,
            crowding_distance: 0.0,
        };

        assert!(
            a.dominates(&b),
            "A should dominate B (better on all objectives)"
        );
        assert!(!b.dominates(&a), "B should not dominate A");
    }
}
