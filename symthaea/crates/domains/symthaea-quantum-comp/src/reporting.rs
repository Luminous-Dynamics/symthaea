//! Dependency-free report export helpers.
//!
//! Alpha.5 adds CSV and Markdown strings so researchers can paste results into
//! lab notes without adopting a serialization dependency.

use crate::benchmark::BindingProbeReport;
use crate::comparative::ComparativeBindingReport;
use crate::noise_sweep::NoiseSweepReport;
use crate::robustness::NoiseRobustnessSummary;

/// Exports reports into simple tabular strings.
pub trait ReportTable {
    /// Returns a CSV representation with a header row.
    fn to_csv(&self) -> String;
    /// Returns a Markdown representation suitable for research notes.
    fn to_markdown(&self) -> String;
}

impl ReportTable for BindingProbeReport {
    fn to_csv(&self) -> String {
        let r = &self.result;
        format!(
            "name,dimension,trials,noise,seed,topology_threshold,fingerprint,classical_recovery,phase_recovery,correlation_recovery,classical_noisy,phase_noisy,correlation_noisy,beta1_proxy,edge_density,mean_degree\n{},{},{},{},{},{},{:016x},{},{},{},{},{},{},{},{},{}\n",
            self.manifest.name,
            self.manifest.dimension,
            self.manifest.trials,
            self.manifest.noise,
            self.manifest.seed,
            self.manifest.topology_threshold,
            self.manifest.reproducibility_fingerprint(),
            r.classical_recovery_similarity,
            r.phase_recovery_similarity,
            r.correlation_recovery_similarity,
            r.classical_noisy_similarity,
            r.phase_noisy_similarity,
            r.correlation_noisy_similarity,
            r.beta1_proxy,
            r.topology_edge_density,
            r.topology_mean_degree,
        )
    }

    fn to_markdown(&self) -> String {
        let r = &self.result;
        format!(
            "# Binding Probe Report\n\n- Name: {}\n- Dimension: {}\n- Trials: {}\n- Noise: {}\n- Seed: {}\n- Fingerprint: {:016x}\n\n| Metric | Value |\n|---|---:|\n| Classical recovery | {:.6} |\n| Phase recovery | {:.6} |\n| Correlation recovery | {:.6} |\n| Classical noisy | {:.6} |\n| Phase noisy | {:.6} |\n| Correlation noisy | {:.6} |\n| Beta-1 proxy | {} |\n| Edge density | {:.6} |\n| Mean degree | {:.6} |\n",
            self.manifest.name,
            self.manifest.dimension,
            self.manifest.trials,
            self.manifest.noise,
            self.manifest.seed,
            self.manifest.reproducibility_fingerprint(),
            r.classical_recovery_similarity,
            r.phase_recovery_similarity,
            r.correlation_recovery_similarity,
            r.classical_noisy_similarity,
            r.phase_noisy_similarity,
            r.correlation_noisy_similarity,
            r.beta1_proxy,
            r.topology_edge_density,
            r.topology_mean_degree,
        )
    }
}

impl ReportTable for NoiseSweepReport {
    fn to_csv(&self) -> String {
        let mut out = String::from(
            "noise,classical_noisy,phase_noisy,correlation_noisy,beta1_proxy,edge_density,mean_degree\n",
        );
        for point in &self.points {
            let r = &point.report.result;
            out.push_str(&format!(
                "{},{},{},{},{},{},{}\n",
                point.noise,
                r.classical_noisy_similarity,
                r.phase_noisy_similarity,
                r.correlation_noisy_similarity,
                r.beta1_proxy,
                r.topology_edge_density,
                r.topology_mean_degree,
            ));
        }
        out
    }

    fn to_markdown(&self) -> String {
        let mut out = String::from(
            "# Noise Sweep Report\n\n| Noise | Classical | Phase | Correlation | Beta-1 proxy | Edge density | Mean degree |\n|---:|---:|---:|---:|---:|---:|---:|\n",
        );
        for point in &self.points {
            let r = &point.report.result;
            out.push_str(&format!(
                "| {:.6} | {:.6} | {:.6} | {:.6} | {} | {:.6} | {:.6} |\n",
                point.noise,
                r.classical_noisy_similarity,
                r.phase_noisy_similarity,
                r.correlation_noisy_similarity,
                r.beta1_proxy,
                r.topology_edge_density,
                r.topology_mean_degree,
            ));
        }
        out
    }
}

impl ReportTable for ComparativeBindingReport {
    fn to_csv(&self) -> String {
        format!(
            "method,recovery_mean,recovery_stderr,noisy_mean,noisy_stderr\nclassical,{},{},{},{}\nphase,{},{},{},{}\ncorrelation,{},{},{},{}\n",
            self.classical.recovery.mean,
            self.classical.recovery.stderr,
            self.classical.noisy.mean,
            self.classical.noisy.stderr,
            self.phase.recovery.mean,
            self.phase.recovery.stderr,
            self.phase.noisy.mean,
            self.phase.noisy.stderr,
            self.correlation.recovery.mean,
            self.correlation.recovery.stderr,
            self.correlation.noisy.mean,
            self.correlation.noisy.stderr,
        )
    }

    fn to_markdown(&self) -> String {
        fn ci(summary: crate::statistics::SampleSummary) -> String {
            let (lo, hi) = summary.approximate_95_ci();
            format!("[{lo:.6}, {hi:.6}]")
        }
        format!(
            "# Comparative Binding Report\n\nReplicates: {}\n\n| Method | Recovery mean | Recovery CI95 | Noisy mean | Noisy CI95 |\n|---|---:|---:|---:|---:|\n| Classical | {:.6} | {} | {:.6} | {} |\n| Phase | {:.6} | {} | {:.6} | {} |\n| Correlation | {:.6} | {} | {:.6} | {} |\n\nClassical-minus-phase noisy dz: {:?}\n\nCorrelation-minus-classical noisy dz: {:?}\n",
            self.config.replicates,
            self.classical.recovery.mean,
            ci(self.classical.recovery),
            self.classical.noisy.mean,
            ci(self.classical.noisy),
            self.phase.recovery.mean,
            ci(self.phase.recovery),
            self.phase.noisy.mean,
            ci(self.phase.noisy),
            self.correlation.recovery.mean,
            ci(self.correlation.recovery),
            self.correlation.noisy.mean,
            ci(self.correlation.noisy),
            self.classical_minus_phase_noisy_dz,
            self.correlation_minus_classical_noisy_dz,
        )
    }
}

/// Converts a robustness summary into a Markdown table.
pub fn robustness_to_markdown(summary: &NoiseRobustnessSummary) -> String {
    format!(
        "# Noise Robustness Summary\n\nSimilarity floor: {:.4}\n\n| Method | AUC | Slope | First below floor | Monotonicity violations |\n|---|---:|---:|---:|---:|\n| Classical | {:.6} | {:.6} | {:?} | {} |\n| Phase | {:.6} | {:.6} | {:?} | {} |\n| Correlation | {:.6} | {:.6} | {:?} | {} |\n",
        summary.similarity_floor,
        summary.classical.auc,
        summary.classical.degradation_slope,
        summary.classical.first_below_floor,
        summary.classical.monotonicity_violations,
        summary.phase.auc,
        summary.phase.degradation_slope,
        summary.phase.first_below_floor,
        summary.phase.monotonicity_violations,
        summary.correlation.auc,
        summary.correlation.degradation_slope,
        summary.correlation.first_below_floor,
        summary.correlation.monotonicity_violations,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BindingProbeConfig, BindingProbeRunner};

    #[test]
    fn binding_report_exports_csv() {
        let cfg = BindingProbeConfig {
            dimension: 64,
            trials: 2,
            noise: 0.01,
            seed: 7,
            topology_threshold: 0.55,
        };
        let report = BindingProbeRunner::new(cfg).unwrap().run().unwrap();
        assert!(report.to_csv().contains("classical_recovery"));
        assert!(report.to_markdown().contains("Binding Probe Report"));
    }
}
