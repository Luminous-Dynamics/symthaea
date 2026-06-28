//! Robustness summaries derived from noise sweeps.

use crate::noise_sweep::NoiseSweepReport;
use crate::statistics::{
    first_threshold_crossing, linear_slope, non_increasing_violations, trapezoid_auc,
};

/// Compact robustness summary for one binding method over a noise sweep.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MethodRobustness {
    /// Trapezoidal area under similarity-vs-noise curve.
    pub auc: f32,
    /// Least-squares slope of similarity-vs-noise. More negative means faster degradation.
    pub degradation_slope: f32,
    /// First noise value where similarity falls below the configured floor.
    pub first_below_floor: Option<f32>,
    /// Count of local increases where a non-increasing curve was expected.
    pub monotonicity_violations: usize,
}

/// Robustness summary across all methods currently reported by [`NoiseSweepReport`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NoiseRobustnessSummary {
    /// Similarity floor used for threshold crossing.
    pub similarity_floor: f32,
    /// Classical XOR HDC robustness.
    pub classical: MethodRobustness,
    /// Phase-HDC simulation robustness.
    pub phase: MethodRobustness,
    /// Correlation-sketch robustness.
    pub correlation: MethodRobustness,
}

impl NoiseRobustnessSummary {
    /// Builds a summary from a noise sweep report.
    pub fn from_sweep(report: &NoiseSweepReport, similarity_floor: f32) -> Self {
        fn summarize(points: Vec<(f32, f32)>, floor: f32) -> MethodRobustness {
            MethodRobustness {
                auc: trapezoid_auc(&points).unwrap_or(0.0),
                degradation_slope: linear_slope(&points).unwrap_or(0.0),
                first_below_floor: first_threshold_crossing(&points, floor),
                monotonicity_violations: non_increasing_violations(&points, 1e-4),
            }
        }

        let classical = report
            .points
            .iter()
            .map(|p| (p.noise, p.report.result.classical_noisy_similarity))
            .collect();
        let phase = report
            .points
            .iter()
            .map(|p| (p.noise, p.report.result.phase_noisy_similarity))
            .collect();
        let correlation = report
            .points
            .iter()
            .map(|p| (p.noise, p.report.result.correlation_noisy_similarity))
            .collect();

        Self {
            similarity_floor,
            classical: summarize(classical, similarity_floor),
            phase: summarize(phase, similarity_floor),
            correlation: summarize(correlation, similarity_floor),
        }
    }

    /// Returns a compact text report.
    pub fn to_text(&self) -> String {
        format!(
            "similarity_floor={:.4}\nclassical_auc={:.6} classical_slope={:.6} classical_first_below={:?} classical_violations={}\nphase_auc={:.6} phase_slope={:.6} phase_first_below={:?} phase_violations={}\ncorrelation_auc={:.6} correlation_slope={:.6} correlation_first_below={:?} correlation_violations={}",
            self.similarity_floor,
            self.classical.auc,
            self.classical.degradation_slope,
            self.classical.first_below_floor,
            self.classical.monotonicity_violations,
            self.phase.auc,
            self.phase.degradation_slope,
            self.phase.first_below_floor,
            self.phase.monotonicity_violations,
            self.correlation.auc,
            self.correlation.degradation_slope,
            self.correlation.first_below_floor,
            self.correlation.monotonicity_violations,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::noise_sweep::{NoiseSweepConfig, NoiseSweepRunner};
    use crate::probe::BindingProbeConfig;

    #[test]
    fn robustness_summary_runs() {
        let base = BindingProbeConfig {
            dimension: 128,
            trials: 4,
            noise: 0.0,
            seed: 11,
            topology_threshold: 0.55,
        };
        let sweep = NoiseSweepRunner::new(NoiseSweepConfig {
            base,
            steps: 3,
            max_noise: 0.2,
        })
        .unwrap()
        .run()
        .unwrap();
        let summary = NoiseRobustnessSummary::from_sweep(&sweep, 0.75);
        assert!(summary.classical.auc >= 0.0);
    }
}
