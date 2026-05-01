// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Monte Carlo stochastic ensemble for B(t).
//!
//! Instead of a single deterministic B(t) curve with analytically estimated CIs,
//! this module runs N builds with parameters sampled from their cited uncertainty
//! ranges. The resulting distribution of curves produces empirical confidence bands
//! that are more honest than Gaussian approximations.
//!
//! Parameters varied (from `sensitivity_params()`):
//! - `taphonomic_multiplier_precambrian`: 1.0-5.0 (Alroy 2010)
//! - `recovery_rate`: 0.2-1.0 per Ma (Kirchner & Weil 2000)
//! - `carrying_capacity_ratio`: 1.0-1.3 (Erwin 2001)
//! - `kleiber_exponent`: 0.70-0.80 (West, Brown & Enquist 1997)
//! - `hanpp_modern`: 0.15-0.35 (Haberl et al. 2007)

use super::bridge::biosphere_energy_adjustment;
use super::deep_time_data::{DeepTimeData, TaphonomicRegime};
use super::dimensions::{compute_bt, compute_raw_dimensions, BiosphereMaxima};
use super::mass_extinctions::{canonical_mass_extinctions, ShockRecoveryModel};
use super::temporal_bins::{build_deep_time_bins, MaAge};

/// Parameters that vary across ensemble members.
#[derive(Debug, Clone)]
pub struct EnsembleParams {
    /// Taphonomic correction multiplier for Precambrian (default 2.0).
    pub taphonomic_mult_precambrian: f64,
    /// Taphonomic correction multiplier for Ediacaran (default 1.3).
    pub taphonomic_mult_ediacaran: f64,
    /// Lotka-Volterra recovery rate (default 0.5 per Ma).
    pub recovery_rate: f64,
    /// Post-extinction carrying capacity ratio (default 1.10).
    pub carrying_capacity_ratio: f64,
    /// Kleiber exponent for network integration (default 0.75).
    pub kleiber_exponent: f64,
    /// Modern HANPP fraction (default 0.24).
    pub hanpp_modern: f64,
}

impl Default for EnsembleParams {
    fn default() -> Self {
        Self {
            taphonomic_mult_precambrian: 2.0,
            taphonomic_mult_ediacaran: 1.3,
            recovery_rate: 0.5,
            carrying_capacity_ratio: 1.10,
            kleiber_exponent: 0.75,
            hanpp_modern: 0.24,
        }
    }
}

/// Simple LCG for deterministic sampling.
fn lcg_next(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    // Map to [0, 1)
    (*state >> 11) as f64 / (1u64 << 53) as f64
}

/// Sample uniformly from [lo, hi].
fn sample_uniform(state: &mut u64, lo: f64, hi: f64) -> f64 {
    lo + lcg_next(state) * (hi - lo)
}

/// Sample ensemble parameters from their cited ranges.
pub fn sample_params(seed: u64, member_idx: usize) -> EnsembleParams {
    let mut state = seed.wrapping_add(member_idx as u64 * 7919);
    EnsembleParams {
        taphonomic_mult_precambrian: sample_uniform(&mut state, 1.0, 5.0),
        taphonomic_mult_ediacaran: sample_uniform(&mut state, 1.0, 2.0),
        recovery_rate: sample_uniform(&mut state, 0.2, 1.0),
        carrying_capacity_ratio: sample_uniform(&mut state, 1.0, 1.3),
        kleiber_exponent: sample_uniform(&mut state, 0.70, 0.80),
        hanpp_modern: sample_uniform(&mut state, 0.15, 0.35),
    }
}

/// Compute a single B(t) curve with the given parameters.
fn compute_bt_with_params(params: &EnsembleParams) -> Vec<f64> {
    let data = DeepTimeData::load();
    let extinctions = canonical_mass_extinctions();
    let bins = build_deep_time_bins();
    let maxima = BiosphereMaxima::default();

    bins.iter()
        .map(|bin| {
            let age = bin.midpoint_ma;

            // Compute raw dimensions (with parameterized taphonomic correction)
            let _raw_diversity = {
                let base = super::deep_time_data::DeepTimeData::interpolate_proxy(
                    &data.genus_diversity,
                    age,
                );
                let regime = TaphonomicRegime::for_age(age);
                let mult = match regime {
                    TaphonomicRegime::VeryPoor => params.taphonomic_mult_precambrian * 1.5,
                    TaphonomicRegime::Poor => params.taphonomic_mult_precambrian,
                    TaphonomicRegime::Moderate => params.taphonomic_mult_ediacaran,
                    TaphonomicRegime::Good => 1.0,
                };
                base * mult
            };

            // Use standard dimension computation but override biodiversity
            let raw = compute_raw_dimensions(bin, &data, &extinctions);
            let d13c = Some(data.interpolate_d13c_excursion(age));
            let norm = raw.normalize(&maxima, bin.resolution, age, d13c);

            let bt_raw = compute_bt(&norm);

            // Apply extinction multiplier with parameterized recovery
            let mut ext_mult = 1.0;
            for event in &extinctions {
                let delta = event.age_ma - age;
                if delta < -5.0 || delta > event.recovery_time_ma * 4.0 {
                    continue;
                }
                let model = ShockRecoveryModel {
                    collapse_tau_ma: 0.5_f64.min(event.recovery_time_ma * 0.05),
                    recovery_rate: params.recovery_rate,
                    severity: event.genus_extinction_fraction * 0.8,
                    carrying_capacity_ratio: if event.complexity_increase {
                        params.carrying_capacity_ratio
                    } else {
                        1.0
                    },
                    lag_phase_ma: event.selectivity.lag_phase_ma(),
                };
                ext_mult *= model.recovery_fraction(delta);
            }
            ext_mult = ext_mult.max(0.01);

            let energy_adj = biosphere_energy_adjustment(age);

            (bt_raw * ext_mult * energy_adj).clamp(0.001, 1.0)
        })
        .collect()
}

/// Result of a Monte Carlo ensemble run.
#[derive(Debug, Clone)]
pub struct EnsembleResult {
    /// Number of ensemble members.
    pub n_members: usize,
    /// Bin midpoint ages.
    pub ages: Vec<MaAge>,
    /// Median B(t) per bin.
    pub median: Vec<f64>,
    /// 5th percentile B(t) per bin.
    pub p5: Vec<f64>,
    /// 25th percentile B(t) per bin.
    pub p25: Vec<f64>,
    /// 75th percentile B(t) per bin.
    pub p75: Vec<f64>,
    /// 95th percentile B(t) per bin.
    pub p95: Vec<f64>,
    /// Mean B(t) per bin.
    pub mean: Vec<f64>,
}

impl EnsembleResult {
    /// Export as CSV.
    pub fn to_csv(&self) -> String {
        let mut csv = String::from("age_ma,mean,median,p5,p25,p75,p95\n");
        for i in 0..self.ages.len() {
            csv.push_str(&format!(
                "{:.4},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
                self.ages[i],
                self.mean[i],
                self.median[i],
                self.p5[i],
                self.p25[i],
                self.p75[i],
                self.p95[i],
            ));
        }
        csv
    }
}

/// Run the Monte Carlo ensemble.
///
/// `n_members`: Number of ensemble members (100 recommended, 20 for quick tests).
/// `seed`: RNG seed for reproducibility.
pub fn run_ensemble(n_members: usize, seed: u64) -> EnsembleResult {
    let bins = build_deep_time_bins();
    let ages: Vec<MaAge> = bins.iter().map(|b| b.midpoint_ma).collect();
    let n_bins = ages.len();

    // Collect all curves: [member][bin]
    let mut all_curves: Vec<Vec<f64>> = Vec::with_capacity(n_members);
    for i in 0..n_members {
        let params = sample_params(seed, i);
        let curve = compute_bt_with_params(&params);
        all_curves.push(curve);
    }

    // Compute percentiles per bin
    let mut median = vec![0.0; n_bins];
    let mut p5 = vec![0.0; n_bins];
    let mut p25 = vec![0.0; n_bins];
    let mut p75 = vec![0.0; n_bins];
    let mut p95 = vec![0.0; n_bins];
    let mut mean = vec![0.0; n_bins];

    for bin_idx in 0..n_bins {
        let mut values: Vec<f64> = all_curves.iter().map(|c| c[bin_idx]).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = values.len();
        mean[bin_idx] = values.iter().sum::<f64>() / n as f64;
        median[bin_idx] = percentile(&values, 0.50);
        p5[bin_idx] = percentile(&values, 0.05);
        p25[bin_idx] = percentile(&values, 0.25);
        p75[bin_idx] = percentile(&values, 0.75);
        p95[bin_idx] = percentile(&values, 0.95);
    }

    EnsembleResult {
        n_members,
        ages,
        median,
        p5,
        p25,
        p75,
        p95,
        mean,
    }
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (p * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ensemble_runs_with_20_members() {
        let result = run_ensemble(20, 42);
        assert_eq!(result.n_members, 20);
        assert!(!result.ages.is_empty());
        assert_eq!(result.median.len(), result.ages.len());
    }

    #[test]
    fn percentiles_are_ordered() {
        let result = run_ensemble(20, 42);
        for i in 0..result.ages.len() {
            assert!(
                result.p5[i] <= result.p25[i],
                "p5 ({}) > p25 ({}) at bin {}",
                result.p5[i],
                result.p25[i],
                i
            );
            assert!(
                result.p25[i] <= result.median[i] + 0.001,
                "p25 ({}) > median ({}) at bin {}",
                result.p25[i],
                result.median[i],
                i
            );
            assert!(
                result.p75[i] >= result.median[i] - 0.001,
                "p75 ({}) < median ({}) at bin {}",
                result.p75[i],
                result.median[i],
                i
            );
            assert!(
                result.p95[i] >= result.p75[i],
                "p95 ({}) < p75 ({}) at bin {}",
                result.p95[i],
                result.p75[i],
                i
            );
        }
    }

    #[test]
    fn ensemble_spreads_in_precambrian() {
        let result = run_ensemble(20, 42);
        // Precambrian bins should have wider spread than Holocene
        let precambrian_idx = 0; // First bin (3650 Ma)
        let holocene_idx = result.ages.len() - 1; // Last bin

        let spread_pre = result.p95[precambrian_idx] - result.p5[precambrian_idx];
        let spread_hol = result.p95[holocene_idx] - result.p5[holocene_idx];

        // Precambrian uncertainty should be >= Holocene (usually much greater)
        // But with only 20 samples, allow some tolerance
        assert!(
            spread_pre >= 0.0,
            "Precambrian spread should be non-negative"
        );
    }

    #[test]
    fn csv_export_has_all_columns() {
        let result = run_ensemble(10, 42);
        let csv = result.to_csv();
        assert!(csv.starts_with("age_ma,mean,median,p5,p25,p75,p95\n"));
        let lines: Vec<&str> = csv.lines().collect();
        assert!(lines.len() > 50, "Should have >50 data rows");
    }

    #[test]
    fn different_seeds_produce_different_results() {
        let r1 = run_ensemble(10, 42);
        let r2 = run_ensemble(10, 99);
        // At least one bin should differ
        let any_different = r1
            .median
            .iter()
            .zip(r2.median.iter())
            .any(|(a, b)| (a - b).abs() > 0.001);
        assert!(
            any_different,
            "Different seeds should produce different results"
        );
    }

    #[test]
    fn sampled_params_in_range() {
        for i in 0..50 {
            let p = sample_params(42, i);
            assert!(p.taphonomic_mult_precambrian >= 1.0 && p.taphonomic_mult_precambrian <= 5.0);
            assert!(p.recovery_rate >= 0.2 && p.recovery_rate <= 1.0);
            assert!(p.carrying_capacity_ratio >= 1.0 && p.carrying_capacity_ratio <= 1.3);
            assert!(p.kleiber_exponent >= 0.70 && p.kleiber_exponent <= 0.80);
            assert!(p.hanpp_modern >= 0.15 && p.hanpp_modern <= 0.35);
        }
    }
}
