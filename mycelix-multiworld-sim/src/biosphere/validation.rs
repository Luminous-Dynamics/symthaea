// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! B(t) validation against published paleontological data.
//!
//! Compares our biodiversity dimension against Sepkoski (1982) and
//! Rohde & Muller (2005) marine genus diversity curves. A Pearson r > 0.85
//! validates the data pipeline; divergences identify where taphonomic
//! correction or interpolation needs tuning.
//!
//! CITATION: Sepkoski (1982) "A compendium of fossil marine animal genera"
//!           Rohde & Muller (2005) "Cycles in fossil diversity" Nature 434.

use super::deep_time_data::DeepTimeData;
use super::dimensions::{compute_raw_dimensions, BiosphereMaxima};
use super::mass_extinctions::canonical_mass_extinctions;
use super::temporal_bins::{build_deep_time_bins, MaAge};

const SEPKOSKI_CSV: &str = include_str!("../../data/biosphere/sepkoski_reference.csv");

/// A reference data point from Sepkoski/Rohde-Muller.
#[derive(Debug, Clone)]
pub struct ReferencePoint {
    pub age_ma: MaAge,
    pub marine_genera: f64,
}

fn load_reference() -> Vec<ReferencePoint> {
    SEPKOSKI_CSV
        .lines()
        .skip(1)
        .filter_map(|line| {
            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() >= 2 {
                Some(ReferencePoint {
                    age_ma: parts[0].trim().parse().ok()?,
                    marine_genera: parts[1].trim().parse().ok()?,
                })
            } else {
                None
            }
        })
        .collect()
}

/// Validation result comparing B(t) biodiversity against Sepkoski.
#[derive(Debug)]
pub struct ValidationResult {
    /// Number of comparison points (Phanerozoic only).
    pub n_points: usize,
    /// Pearson correlation coefficient between our diversity and Sepkoski.
    pub pearson_r: f64,
    /// R² (coefficient of determination).
    pub r_squared: f64,
    /// Mean absolute error (normalized to [0,1] scale).
    pub mean_absolute_error: f64,
    /// Points where divergence exceeds 30% (needs investigation).
    pub divergent_points: Vec<DivergentPoint>,
    /// Per-point comparison data.
    pub comparisons: Vec<ComparisonPoint>,
}

#[derive(Debug, Clone)]
pub struct ComparisonPoint {
    pub age_ma: MaAge,
    /// Our B(t) biodiversity value (normalized).
    pub bt_biodiversity: f64,
    /// Sepkoski reference value (normalized to same scale).
    pub sepkoski_normalized: f64,
    /// Absolute difference.
    pub abs_error: f64,
}

#[derive(Debug, Clone)]
pub struct DivergentPoint {
    pub age_ma: MaAge,
    pub bt_value: f64,
    pub sepkoski_value: f64,
    pub pct_difference: f64,
    pub likely_cause: String,
}

/// Run validation against Sepkoski/Rohde-Muller reference data.
pub fn validate_against_sepkoski() -> ValidationResult {
    let reference = load_reference();
    let data = DeepTimeData::load();
    let extinctions = canonical_mass_extinctions();
    let bins = build_deep_time_bins();
    let maxima = BiosphereMaxima::default();

    // Normalize Sepkoski to [0,1] using same max as our biodiversity dimension
    let sepkoski_max = reference
        .iter()
        .map(|r| r.marine_genera)
        .fold(0.0_f64, f64::max);

    let mut comparisons = Vec::new();
    let mut our_values = Vec::new();
    let mut ref_values = Vec::new();

    for ref_pt in &reference {
        // Find the bin containing this age
        let bin = bins
            .iter()
            .find(|b| ref_pt.age_ma >= b.end_ma && ref_pt.age_ma <= b.start_ma);

        let bin = match bin {
            Some(b) => b,
            None => continue, // Skip if outside our bin range
        };

        let raw = compute_raw_dimensions(bin, &data, &extinctions);
        let bt_bio_norm = (raw.biodiversity / maxima.max_biodiversity).clamp(0.0, 1.0);
        let sepkoski_norm = (ref_pt.marine_genera / sepkoski_max).clamp(0.0, 1.0);
        let abs_error = (bt_bio_norm - sepkoski_norm).abs();

        our_values.push(bt_bio_norm);
        ref_values.push(sepkoski_norm);

        comparisons.push(ComparisonPoint {
            age_ma: ref_pt.age_ma,
            bt_biodiversity: bt_bio_norm,
            sepkoski_normalized: sepkoski_norm,
            abs_error,
        });
    }

    // Pearson correlation
    let n = our_values.len() as f64;
    let pearson_r = if our_values.len() >= 3 {
        let mean_x: f64 = our_values.iter().sum::<f64>() / n;
        let mean_y: f64 = ref_values.iter().sum::<f64>() / n;

        let mut cov = 0.0;
        let mut var_x = 0.0;
        let mut var_y = 0.0;
        for i in 0..our_values.len() {
            let dx = our_values[i] - mean_x;
            let dy = ref_values[i] - mean_y;
            cov += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }

        let denom = (var_x * var_y).sqrt();
        if denom > 1e-15 {
            cov / denom
        } else {
            0.0
        }
    } else {
        0.0
    };

    let r_squared = pearson_r * pearson_r;
    let mean_absolute_error =
        comparisons.iter().map(|c| c.abs_error).sum::<f64>() / comparisons.len().max(1) as f64;

    // Identify divergent points (>30% difference)
    let divergent_points: Vec<DivergentPoint> = comparisons
        .iter()
        .filter_map(|c| {
            let max_val = c.bt_biodiversity.max(c.sepkoski_normalized).max(0.01);
            let pct_diff = c.abs_error / max_val * 100.0;
            if pct_diff > 30.0 {
                let cause = if c.bt_biodiversity > c.sepkoski_normalized {
                    if c.age_ma > 600.0 {
                        "Taphonomic over-correction (Precambrian)"
                    } else {
                        "PBDB data higher than Sepkoski (possible sampling difference)"
                    }
                } else if c.age_ma > 440.0 && c.age_ma < 450.0 {
                    "End-Ordovician: PBDB interpolation misses exact trough"
                } else if c.age_ma > 250.0 && c.age_ma < 255.0 {
                    "End-Permian: interpolation resolution at extinction nadir"
                } else {
                    "Interpolation gap or normalization difference"
                };
                Some(DivergentPoint {
                    age_ma: c.age_ma,
                    bt_value: c.bt_biodiversity,
                    sepkoski_value: c.sepkoski_normalized,
                    pct_difference: pct_diff,
                    likely_cause: cause.to_string(),
                })
            } else {
                None
            }
        })
        .collect();

    ValidationResult {
        n_points: comparisons.len(),
        pearson_r,
        r_squared,
        mean_absolute_error,
        divergent_points,
        comparisons,
    }
}

/// Decoupled validation: correlate ONLY the non-genus-derived dimensions
/// against the Sepkoski curve to test whether the *structural* coherence
/// dimensions (complexity, resilience, network) track diversity independently.
///
/// This addresses the circularity criticism: genus count feeds Biodiversity,
/// Energy Throughput (via biomass), and Information Capacity (via functional groups).
/// The non-circular dimensions are: Complexity (ordinal scale from first appearances)
/// and Resilience (background extinction rate, independent of absolute diversity).
pub fn validate_non_circular() -> (f64, usize) {
    let reference = load_reference();
    let data = DeepTimeData::load();
    let extinctions = canonical_mass_extinctions();
    let bins = build_deep_time_bins();
    let maxima = BiosphereMaxima::default();

    let mut complexity_values: Vec<f64> = Vec::new();
    let mut ref_values: Vec<f64> = Vec::new();
    let sepkoski_max = reference
        .iter()
        .map(|r| r.marine_genera)
        .fold(0.0_f64, f64::max);

    for ref_pt in &reference {
        let bin = bins
            .iter()
            .find(|b| ref_pt.age_ma >= b.end_ma && ref_pt.age_ma <= b.start_ma);
        let bin = match bin {
            Some(b) => b,
            None => continue,
        };

        let raw = compute_raw_dimensions(bin, &data, &extinctions);
        let d13c = Some(data.interpolate_d13c_excursion(bin.midpoint_ma));
        let norm = raw.normalize(&maxima, bin.resolution, bin.midpoint_ma, d13c);
        let sepkoski_norm = (ref_pt.marine_genera / sepkoski_max).clamp(0.0, 1.0);

        // Non-circular: average of Complexity + Resilience
        let non_circular_avg = (norm.values[1] + norm.values[2]) / 2.0;
        complexity_values.push(non_circular_avg);
        ref_values.push(sepkoski_norm);
    }

    let n = complexity_values.len();
    let pearson = if n >= 3 {
        let mean_x: f64 = complexity_values.iter().sum::<f64>() / n as f64;
        let mean_y: f64 = ref_values.iter().sum::<f64>() / n as f64;
        let mut cov = 0.0;
        let mut var_x = 0.0;
        let mut var_y = 0.0;
        for i in 0..n {
            let dx = complexity_values[i] - mean_x;
            let dy = ref_values[i] - mean_y;
            cov += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }
        let denom = (var_x * var_y).sqrt();
        if denom > 1e-15 {
            cov / denom
        } else {
            0.0
        }
    } else {
        0.0
    };

    (pearson, n)
}

/// Validate B(t) composite against εp (isotopic fractionation) as an
/// independent thermodynamic proxy. εp is NOT derived from genus counts —
/// it comes from carbonate/organic carbon isotopic ratios.
/// This tests whether B(t) tracks the biosphere's thermodynamic state
/// independently of its taxonomic input.
pub fn validate_against_epsilon_p() -> (f64, usize) {
    let data = DeepTimeData::load();
    let extinctions = canonical_mass_extinctions();
    let bins = build_deep_time_bins();
    let maxima = BiosphereMaxima::default();

    let ep_max = 32.0; // Maximum εp in the record (Phanerozoic)

    let mut bt_values = Vec::new();
    let mut ep_values = Vec::new();

    for bin in &bins {
        let age = bin.midpoint_ma;
        if age > 541.0 {
            continue;
        } // Phanerozoic only for fair comparison

        let raw = compute_raw_dimensions(bin, &data, &extinctions);
        let d13c = Some(data.interpolate_d13c_excursion(age));
        let norm = raw.normalize(&maxima, bin.resolution, age, d13c);
        let bt = super::dimensions::compute_bt(&norm);

        let ep = data.interpolate_epsilon_p(age);
        let ep_norm = (ep / ep_max).clamp(0.0, 1.0);

        bt_values.push(bt);
        ep_values.push(ep_norm);
    }

    let n = bt_values.len();
    let pearson = if n >= 3 {
        let mean_x: f64 = bt_values.iter().sum::<f64>() / n as f64;
        let mean_y: f64 = ep_values.iter().sum::<f64>() / n as f64;
        let mut cov = 0.0;
        let mut var_x = 0.0;
        let mut var_y = 0.0;
        for i in 0..n {
            let dx = bt_values[i] - mean_x;
            let dy = ep_values[i] - mean_y;
            cov += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }
        let denom = (var_x * var_y).sqrt();
        if denom > 1e-15 {
            cov / denom
        } else {
            0.0
        }
    } else {
        0.0
    };

    (pearson, n)
}

impl ValidationResult {
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();
        md.push_str("# B(t) Validation: Sepkoski/Rohde-Muller Comparison\n\n");
        md.push_str(&format!(
            "**Comparison points**: {} (Phanerozoic, 541 Ma → present)\n",
            self.n_points
        ));
        md.push_str(&format!("**Pearson r**: {:.4}\n", self.pearson_r));
        md.push_str(&format!("**R²**: {:.4}\n", self.r_squared));
        md.push_str(&format!(
            "**Mean absolute error**: {:.4}\n\n",
            self.mean_absolute_error
        ));

        let verdict = if self.pearson_r > 0.90 {
            "EXCELLENT — B(t) biodiversity strongly tracks Sepkoski"
        } else if self.pearson_r > 0.85 {
            "GOOD — B(t) validated against published curves"
        } else if self.pearson_r > 0.70 {
            "MODERATE — significant correlation but divergences present"
        } else {
            "POOR — B(t) diverges from Sepkoski; calibration needed"
        };
        md.push_str(&format!("**Verdict**: {}\n\n", verdict));

        // Decoupled validation
        let (nc_r, nc_n) = validate_non_circular();
        md.push_str("## Decoupled Validation (Non-Circular)\n\n");
        md.push_str(&format!(
            "Non-genus-derived dimensions (Complexity+Resilience) vs Sepkoski:\n"
        ));
        md.push_str(&format!("Pearson $r$ = {:.4} ($n$ = {})\n\n", nc_r, nc_n));

        let (ep_r, ep_n) = validate_against_epsilon_p();
        md.push_str(&format!("Composite $B(t)$ vs $\\epsilon_p$ (isotopic fractionation, independent thermodynamic proxy):\n"));
        md.push_str(&format!("Pearson $r$ = {:.4} ($n$ = {})\n\n", ep_r, ep_n));

        if !self.divergent_points.is_empty() {
            md.push_str("## Divergent Points (>30% difference)\n\n");
            md.push_str("| Age (Ma) | B(t) | Sepkoski | Δ% | Likely Cause |\n");
            md.push_str("|---|---|---|---|---|\n");
            for dp in &self.divergent_points {
                md.push_str(&format!(
                    "| {:.0} | {:.3} | {:.3} | {:.0}% | {} |\n",
                    dp.age_ma, dp.bt_value, dp.sepkoski_value, dp.pct_difference, dp.likely_cause
                ));
            }
        }

        md
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_data_loads() {
        let reference = load_reference();
        assert!(
            reference.len() >= 30,
            "Should have >=30 reference points, got {}",
            reference.len()
        );
    }

    #[test]
    fn validation_produces_result() {
        let result = validate_against_sepkoski();
        assert!(result.n_points >= 25, "Should compare >=25 points");
        assert!(result.pearson_r.is_finite());
    }

    #[test]
    fn pearson_r_is_positive() {
        let result = validate_against_sepkoski();
        assert!(
            result.pearson_r > 0.0,
            "Correlation should be positive (both increase over time), got {}",
            result.pearson_r
        );
    }

    #[test]
    fn correlation_above_minimum() {
        let result = validate_against_sepkoski();
        assert!(
            result.pearson_r > 0.60,
            "Pearson r should be > 0.60, got {}. B(t) may need recalibration.",
            result.pearson_r
        );
    }

    #[test]
    fn mean_error_below_threshold() {
        let result = validate_against_sepkoski();
        assert!(
            result.mean_absolute_error < 0.30,
            "MAE should be < 0.30, got {}",
            result.mean_absolute_error
        );
    }

    #[test]
    fn markdown_contains_verdict() {
        let result = validate_against_sepkoski();
        let md = result.to_markdown();
        assert!(md.contains("Pearson r"));
        assert!(md.contains("Verdict"));
        assert!(md.contains("Decoupled Validation"));
        assert!(md.contains("epsilon_p"));
    }

    #[test]
    fn non_circular_validation_positive() {
        let (r, n) = validate_non_circular();
        assert!(n >= 20, "Should have >=20 comparison points");
        assert!(
            r > 0.0,
            "Non-circular correlation should be positive, got {}",
            r
        );
    }

    #[test]
    fn epsilon_p_validation_runs() {
        let (r, n) = validate_against_epsilon_p();
        assert!(n >= 10, "Should have >=10 comparison points");
        assert!(r.is_finite(), "εp correlation should be finite");
    }
}
