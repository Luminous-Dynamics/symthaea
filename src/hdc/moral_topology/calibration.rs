// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Calibration, ROC curves, and severity scoring for moral topology.

use super::*;

/// A labeled scenario set for threshold calibration.
#[derive(Debug, Clone)]
pub struct CalibrationScenario {
    pub scenarios: Vec<ContinuousHV>,
    pub is_adversarial: bool,
    pub label: String,
}

/// ROC point at a given threshold.
#[derive(Debug, Clone, Serialize)]
pub struct RocPoint {
    pub threshold: f64,
    pub true_positive_rate: f64,
    pub false_positive_rate: f64,
    pub precision: f64,
    pub f1: f64,
}

/// Result of a calibration sweep.
#[derive(Debug, Clone, Serialize)]
pub struct CalibrationResult {
    pub roc_curve: Vec<RocPoint>,
    pub auc: f64,
    pub best_f1_threshold: f64,
    pub best_f1: f64,
}

/// Sweep convergence thresholds to find optimal operating point.
pub fn calibrate_convergence_threshold(
    scenarios: &[CalibrationScenario],
    thresholds: &[f64],
    dim: usize,
    base_config: &MoralAnomalyConfig,
) -> CalibrationResult {
    let total_positive = scenarios.iter().filter(|s| s.is_adversarial).count();
    let total_negative = scenarios.len() - total_positive;
    let mut roc_curve = Vec::with_capacity(thresholds.len());

    for &threshold in thresholds {
        let mut config = base_config.clone();
        config.convergence_similarity_threshold = threshold;
        let mut true_pos = 0usize;
        let mut false_pos = 0usize;

        for scenario in scenarios {
            let basis = Arc::new(HarmonyBasis::new(dim));
            let mut topo = MoralTopology::with_anomaly_config(
                MoralTopologyConfig {
                    dim,
                    ..Default::default()
                },
                basis,
                config.clone(),
            );
            for hv in &scenario.scenarios {
                topo.add_scenario(hv.clone());
            }
            let report = topo.detect_trajectory_convergence();
            if report.convergence_detected {
                if scenario.is_adversarial {
                    true_pos += 1;
                } else {
                    false_pos += 1;
                }
            }
        }

        let tpr = if total_positive > 0 {
            true_pos as f64 / total_positive as f64
        } else {
            0.0
        };
        let fpr = if total_negative > 0 {
            false_pos as f64 / total_negative as f64
        } else {
            0.0
        };
        let precision = if true_pos + false_pos > 0 {
            true_pos as f64 / (true_pos + false_pos) as f64
        } else {
            0.0
        };
        let f1 = if precision + tpr > 0.0 {
            2.0 * precision * tpr / (precision + tpr)
        } else {
            0.0
        };

        roc_curve.push(RocPoint {
            threshold,
            true_positive_rate: tpr,
            false_positive_rate: fpr,
            precision,
            f1,
        });
    }

    roc_curve.sort_by(|a, b| {
        a.threshold
            .partial_cmp(&b.threshold)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut sorted_by_fpr = roc_curve.clone();
    sorted_by_fpr.sort_by(|a, b| {
        a.false_positive_rate
            .partial_cmp(&b.false_positive_rate)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let auc: f64 = if sorted_by_fpr.len() >= 2 {
        sorted_by_fpr
            .windows(2)
            .map(|w| {
                (w[1].false_positive_rate - w[0].false_positive_rate)
                    * (w[0].true_positive_rate + w[1].true_positive_rate)
                    / 2.0
            })
            .sum()
    } else {
        0.0
    };

    let best = roc_curve
        .iter()
        .max_by(|a, b| a.f1.partial_cmp(&b.f1).unwrap_or(std::cmp::Ordering::Equal))
        .cloned()
        .unwrap_or(RocPoint {
            threshold: 0.15,
            true_positive_rate: 0.0,
            false_positive_rate: 0.0,
            precision: 0.0,
            f1: 0.0,
        });

    CalibrationResult {
        roc_curve,
        auc,
        best_f1_threshold: best.threshold,
        best_f1: best.f1,
    }
}

impl MoralTopology {
    /// Learn optimal thresholds from calibration data.
    ///
    /// Takes a [`CalibrationResult`] and updates the anomaly config's
    /// `convergence_similarity_threshold` to the best-F1 operating point.
    /// Also builds the severity calibration CDF from the ROC curve.
    pub fn apply_calibration(&mut self, cal: &CalibrationResult) {
        if cal.best_f1 > 0.0 {
            self.anomaly_config.convergence_similarity_threshold = cal.best_f1_threshold;
        }
        // Build CDF from ROC: map severity → empirical percentile
        let mut cdf_points: Vec<(f64, f64)> = cal
            .roc_curve
            .iter()
            .filter(|p| p.f1 > 0.0)
            .map(|p| (p.threshold, p.true_positive_rate))
            .collect();
        cdf_points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        cdf_points.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-9);
        if !cdf_points.is_empty() {
            self.set_severity_calibration(cdf_points);
        }
    }

    /// Set the empirical CDF for severity calibration.
    ///
    /// The CDF is a sorted list of `(raw_severity, calibrated_percentile)` pairs.
    /// When `calibrate_severity()` is called, it linearly interpolates through
    /// these breakpoints. Build from [`CalibrationResult`] output.
    pub fn set_severity_calibration(&mut self, cdf: Vec<(f64, f64)>) {
        self.severity_calibration_cdf = cdf;
    }

    /// Map raw severity through the empirical CDF.
    ///
    /// If no calibration is set, returns raw severity unchanged.
    pub(super) fn calibrate_severity(&self, raw: f64) -> f64 {
        let cdf = &self.severity_calibration_cdf;
        if cdf.is_empty() {
            return raw;
        }
        // Binary search for interpolation bracket
        match cdf.binary_search_by(|probe| {
            probe
                .0
                .partial_cmp(&raw)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            Ok(idx) => cdf[idx].1,
            Err(0) => cdf[0].1,
            Err(idx) if idx >= cdf.len() => cdf.last().unwrap().1,
            Err(idx) => {
                let (x0, y0) = cdf[idx - 1];
                let (x1, y1) = cdf[idx];
                let t = (raw - x0) / (x1 - x0).max(1e-12);
                (y0 + t * (y1 - y0)).clamp(0.0, 1.0)
            }
        }
    }
}
