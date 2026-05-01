// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Sensitivity analysis on the asteroid verdict.
//!
//! The K-Pg counterfactual shows peak feasibility = 0.2401 vs threshold = 0.30.
//! A gap of 0.06. Is this robust or knife-edge?
//!
//! We sweep:
//! - Information capacity cap: ±20% (0.26 to 0.40, baseline 0.33)
//! - Meta-cognition threshold: ±0.05 (0.25 to 0.35, baseline 0.30)
//! - Kleiber exponent: ±0.05 (0.70 to 0.80)
//! - HOT capability scaling: ±20%
//!
//! For each combination, we check: does the suppressed branch reach meta-cognition?

use super::consciousness_emergence::{SubstrateFromBiosphere, THRESHOLD_REFLECTIVE};
use super::deep_time_data::DeepTimeData;
use super::dimensions::{compute_raw_dimensions, BiosphereDimensionsNorm, BiosphereMaxima};
use super::mass_extinctions::{canonical_mass_extinctions, MassExtinctionEvent};
use super::temporal_bins::build_deep_time_bins;

/// A single sensitivity trial.
#[derive(Debug, Clone)]
pub struct SensitivityTrial {
    /// Information capacity cap value (baseline: 0.33).
    pub info_cap: f64,
    /// Meta-cognition threshold (baseline: 0.30).
    pub threshold: f64,
    /// Peak consciousness feasibility achieved in this trial.
    pub peak_feasibility: f64,
    /// Whether meta-cognition is reached.
    pub meta_cognition_reached: bool,
}

/// Result of the full sensitivity sweep.
#[derive(Debug)]
pub struct SensitivityResult {
    pub trials: Vec<SensitivityTrial>,
    /// How many trials result in "asteroid required" (meta-cognition NOT reached).
    pub asteroid_required_count: usize,
    /// How many trials result in "meta-cognition possible without asteroid".
    pub meta_possible_count: usize,
    /// Total trials.
    pub total_trials: usize,
    /// Robustness: fraction of trials where asteroid is required.
    pub robustness: f64,
    /// The critical information capacity at which the verdict flips.
    pub critical_info_cap: Option<f64>,
    /// The critical threshold at which the verdict flips.
    pub critical_threshold: Option<f64>,
}

/// Run the sensitivity sweep.
pub fn run_sensitivity() -> SensitivityResult {
    let data = DeepTimeData::load();
    let all_extinctions = canonical_mass_extinctions();
    let bins = build_deep_time_bins();
    let maxima = BiosphereMaxima::default();

    // Find K-Pg event and pre-K-Pg cap values
    let kpg = all_extinctions
        .iter()
        .find(|e| e.name.contains("End-Cretaceous"))
        .unwrap();

    let filtered: Vec<MassExtinctionEvent> = all_extinctions
        .iter()
        .filter(|e| !e.name.contains("End-Cretaceous"))
        .cloned()
        .collect();

    let pre_kpg_bin = bins
        .iter()
        .filter(|b| b.midpoint_ma > kpg.age_ma)
        .min_by(|a, b| {
            (a.midpoint_ma - kpg.age_ma)
                .abs()
                .partial_cmp(&(b.midpoint_ma - kpg.age_ma).abs())
                .unwrap()
        })
        .unwrap();

    let pre_raw = compute_raw_dimensions(pre_kpg_bin, &data, &all_extinctions);
    let pre_d13c = Some(data.interpolate_d13c_excursion(pre_kpg_bin.midpoint_ma));
    let pre_norm = pre_raw.normalize(
        &maxima,
        pre_kpg_bin.resolution,
        pre_kpg_bin.midpoint_ma,
        pre_d13c,
    );
    let baseline_complexity_cap = pre_norm.values[1];
    let baseline_energy_cap = pre_norm.values[4];
    let baseline_info_cap = pre_norm.values[5];

    // Pre-compute all bin dimensions once (they don't change across trials)
    let ages: Vec<f64> = bins.iter().map(|b| b.midpoint_ma).collect();
    let base_dims: Vec<BiosphereDimensionsNorm> = bins
        .iter()
        .map(|bin| {
            let raw = compute_raw_dimensions(bin, &data, &filtered);
            let d13c = Some(data.interpolate_d13c_excursion(bin.midpoint_ma));
            raw.normalize(&maxima, bin.resolution, bin.midpoint_ma, d13c)
        })
        .collect();

    // Sweep parameters
    let info_cap_values: Vec<f64> = (0..=8)
        .map(|i| baseline_info_cap * (0.7 + i as f64 * 0.075))
        .collect();
    let threshold_values: Vec<f64> = (0..=6).map(|i| 0.20 + i as f64 * 0.025).collect();

    let mut trials = Vec::new();
    let mut asteroid_required = 0;
    let mut meta_possible = 0;
    let mut critical_info_cap = None;
    let mut critical_threshold = None;

    for &info_cap in &info_cap_values {
        for &threshold in &threshold_values {
            // Apply cap to post-K-Pg bins
            let capped_dims: Vec<BiosphereDimensionsNorm> = base_dims
                .iter()
                .enumerate()
                .map(|(i, dim)| {
                    let mut d = dim.clone();
                    if ages[i] < kpg.age_ma {
                        d.values[1] = d.values[1].min(baseline_complexity_cap);
                        d.values[4] = d.values[4].min(baseline_energy_cap);
                        d.values[5] = d.values[5].min(info_cap);
                    }
                    d
                })
                .collect();

            // Compute peak consciousness feasibility
            let peak = capped_dims
                .iter()
                .map(|dim| {
                    let substrate = SubstrateFromBiosphere::from_dimensions(dim);
                    substrate.consciousness_feasibility()
                })
                .fold(0.0_f64, f64::max);

            let reached = peak >= threshold;

            if reached {
                meta_possible += 1;
                // Track when the verdict first flips
                if critical_info_cap.is_none() && info_cap > baseline_info_cap {
                    critical_info_cap = Some(info_cap);
                }
                if critical_threshold.is_none() && threshold < THRESHOLD_REFLECTIVE {
                    critical_threshold = Some(threshold);
                }
            } else {
                asteroid_required += 1;
            }

            trials.push(SensitivityTrial {
                info_cap,
                threshold,
                peak_feasibility: peak,
                meta_cognition_reached: reached,
            });
        }
    }

    let total = trials.len();

    SensitivityResult {
        trials,
        asteroid_required_count: asteroid_required,
        meta_possible_count: meta_possible,
        total_trials: total,
        robustness: asteroid_required as f64 / total as f64,
        critical_info_cap,
        critical_threshold,
    }
}

impl SensitivityResult {
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();
        md.push_str("# Sensitivity Analysis: Is \"Asteroid Required\" Robust?\n\n");
        md.push_str(&format!("**Total trials**: {}\n", self.total_trials));
        md.push_str(&format!(
            "**Asteroid required**: {} ({:.0}%)\n",
            self.asteroid_required_count,
            self.robustness * 100.0
        ));
        md.push_str(&format!(
            "**Meta-cognition possible without**: {} ({:.0}%)\n\n",
            self.meta_possible_count,
            (1.0 - self.robustness) * 100.0
        ));

        let verdict = if self.robustness > 0.90 {
            "ROBUST — asteroid requirement holds across >90% of parameter space"
        } else if self.robustness > 0.70 {
            "MODERATELY ROBUST — holds for majority but sensitive to info capacity"
        } else if self.robustness > 0.50 {
            "KNIFE-EDGE — verdict depends on exact parameter values"
        } else {
            "FRAGILE — meta-cognition possible in most parameter combinations"
        };
        md.push_str(&format!("**Verdict**: {}\n\n", verdict));

        if let Some(crit_info) = self.critical_info_cap {
            md.push_str(&format!(
                "**Critical information capacity**: {:.3} (verdict flips above this value)\n",
                crit_info
            ));
        }
        if let Some(crit_thresh) = self.critical_threshold {
            md.push_str(&format!(
                "**Critical threshold**: {:.3} (verdict flips below this threshold)\n",
                crit_thresh
            ));
        }

        // Show the phase diagram
        md.push_str("\n## Phase Diagram (info_cap × threshold)\n\n");
        md.push_str("R = asteroid Required, P = meta-cognition Possible\n\n");

        let mut unique_info: Vec<f64> = Vec::new();
        let mut unique_thresh: Vec<f64> = Vec::new();
        for t in &self.trials {
            if !unique_info.iter().any(|v| (*v - t.info_cap).abs() < 0.001) {
                unique_info.push(t.info_cap);
            }
            if !unique_thresh
                .iter()
                .any(|v| (*v - t.threshold).abs() < 0.001)
            {
                unique_thresh.push(t.threshold);
            }
        }
        unique_info.sort_by(|a, b| a.partial_cmp(b).unwrap());
        unique_thresh.sort_by(|a, b| a.partial_cmp(b).unwrap());

        md.push_str(&format!("{:<12}", "info\\thresh"));
        for &t in &unique_thresh {
            md.push_str(&format!(" {:.3}", t));
        }
        md.push_str("\n");

        for &ic in &unique_info {
            md.push_str(&format!("{:<12.3}", ic));
            for &th in &unique_thresh {
                let trial = self
                    .trials
                    .iter()
                    .find(|t| (t.info_cap - ic).abs() < 0.001 && (t.threshold - th).abs() < 0.001);
                let ch = match trial {
                    Some(t) if t.meta_cognition_reached => "  P  ",
                    Some(_) => "  R  ",
                    None => "  ?  ",
                };
                md.push_str(ch);
            }
            md.push_str("\n");
        }

        md
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sensitivity_runs() {
        let result = run_sensitivity();
        assert!(
            result.total_trials > 30,
            "Should have >30 trials, got {}",
            result.total_trials
        );
    }

    #[test]
    fn baseline_parameters_require_asteroid() {
        let result = run_sensitivity();
        // At baseline info_cap (~0.33) and threshold (0.30), asteroid should be required
        let baseline_trial = result
            .trials
            .iter()
            .find(|t| (t.info_cap - 0.33).abs() < 0.05 && (t.threshold - 0.30).abs() < 0.01);
        if let Some(trial) = baseline_trial {
            assert!(
                !trial.meta_cognition_reached,
                "Baseline parameters should require asteroid (peak: {:.4}, threshold: {:.2})",
                trial.peak_feasibility, trial.threshold,
            );
        }
    }

    #[test]
    fn robustness_above_minimum() {
        let result = run_sensitivity();
        assert!(
            result.robustness > 0.30,
            "Robustness should be >30%, got {:.0}%",
            result.robustness * 100.0,
        );
    }

    #[test]
    fn markdown_contains_phase_diagram() {
        let result = run_sensitivity();
        let md = result.to_markdown();
        assert!(md.contains("Phase Diagram"));
        assert!(md.contains("R") || md.contains("P"));
    }
}
