// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic diagnostics for auditing aesthetic score behavior.
//!
//! These are intentionally dependency-free so CI and evidence-bundle builders
//! can detect score collapse, saturation, and calibration regressions.

use crate::birkhoff::BirkhoffFeatures;

/// Summary of a deterministic Birkhoff feature-grid sweep.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BirkhoffSaturationReport {
    pub samples: usize,
    pub mean: f32,
    pub median: f32,
    pub minimum: f32,
    pub maximum: f32,
    /// Fraction of calibrated scores effectively pinned at one.
    pub fraction_at_ceiling: f32,
    /// Fraction of classical raw ratios above one. This quantifies how much a
    /// direct clamp would have collapsed.
    pub raw_ratio_above_one_fraction: f32,
}

/// Sweep the six scalar Birkhoff inputs over an evenly spaced deterministic
/// grid. `steps_per_axis` is clamped to 2..=11 to keep runtime bounded.
pub fn sweep_birkhoff_saturation(steps_per_axis: usize) -> BirkhoffSaturationReport {
    let steps = steps_per_axis.clamp(2, 11);
    let axis: Vec<f32> = (0..steps)
        .map(|index| index as f32 / (steps - 1) as f32)
        .collect();
    let capacity = steps.pow(6);
    let mut calibrated = Vec::with_capacity(capacity);
    let mut raw_above_one = 0usize;

    for &symmetry in &axis {
        for &harmony_balance in &axis {
            for &consciousness_coupling in &axis {
                for &structural_complexity in &axis {
                    for &topological_complexity in &axis {
                        for &diversity in &axis {
                            let features = BirkhoffFeatures {
                                symmetry,
                                harmony_balance,
                                consciousness_coupling,
                                structural_complexity,
                                topological_complexity,
                                diversity,
                                harmony_activations: [harmony_balance; 8],
                            };
                            if features.birkhoff_raw_ratio() > 1.0 {
                                raw_above_one += 1;
                            }
                            calibrated.push(features.birkhoff());
                        }
                    }
                }
            }
        }
    }

    calibrated.sort_by(|left, right| left.total_cmp(right));
    let samples = calibrated.len();
    let mean = calibrated.iter().sum::<f32>() / samples as f32;
    let median = if samples % 2 == 0 {
        (calibrated[samples / 2 - 1] + calibrated[samples / 2]) * 0.5
    } else {
        calibrated[samples / 2]
    };
    let ceiling = calibrated
        .iter()
        .filter(|&&score| score >= 1.0 - 1e-6)
        .count();

    BirkhoffSaturationReport {
        samples,
        mean,
        median,
        minimum: calibrated[0],
        maximum: calibrated[samples - 1],
        fraction_at_ceiling: ceiling as f32 / samples as f32,
        raw_ratio_above_one_fraction: raw_above_one as f32 / samples as f32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calibrated_measure_preserves_headroom() {
        let report = sweep_birkhoff_saturation(5);
        assert!(report.samples > 10_000);
        assert!(report.raw_ratio_above_one_fraction > 0.25);
        assert_eq!(report.fraction_at_ceiling, 0.0);
        assert!(report.maximum < 1.0);
        assert!(report.mean > 0.35 && report.mean < 0.65);
    }

    #[test]
    fn diagnostic_runtime_is_bounded() {
        assert_eq!(sweep_birkhoff_saturation(1).samples, 2usize.pow(6));
        assert_eq!(sweep_birkhoff_saturation(100).samples, 11usize.pow(6));
    }
}
