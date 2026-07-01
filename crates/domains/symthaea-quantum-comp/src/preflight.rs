//! Preflight validation helpers for local research runs.
//!
//! Alpha.7 adds a small report shape that can be used by CLIs, notebooks, and
//! CI scripts before running expensive replicated experiments. The preflight
//! layer does not prove scientific validity; it catches obvious local mistakes
//! and reports caveats in a stable format.

use crate::comparative::ComparativeBindingConfig;
use crate::matrix::ExperimentMatrixConfig;
use crate::noise_sweep::NoiseSweepConfig;
use crate::probe::BindingProbeConfig;

/// Local preflight severity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreflightSeverity {
    /// Informational note.
    Info,
    /// Caution that should be visible in reports.
    Warning,
    /// Configuration problem that should block execution.
    Error,
}

/// One preflight finding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreflightFinding {
    /// Finding severity.
    pub severity: PreflightSeverity,
    /// Stable finding code.
    pub code: &'static str,
    /// Human-readable message.
    pub message: String,
}

impl PreflightFinding {
    /// Creates a new finding.
    pub fn new(
        severity: PreflightSeverity,
        code: &'static str,
        message: impl Into<String>,
    ) -> Self {
        Self {
            severity,
            code,
            message: message.into(),
        }
    }
}

/// Aggregated preflight report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreflightReport {
    /// Findings emitted by local preflight checks.
    pub findings: Vec<PreflightFinding>,
}

impl PreflightReport {
    /// Returns true when no error was emitted.
    pub fn can_run(&self) -> bool {
        !self
            .findings
            .iter()
            .any(|f| f.severity == PreflightSeverity::Error)
    }

    /// Returns true when at least one warning was emitted.
    pub fn has_warnings(&self) -> bool {
        self.findings
            .iter()
            .any(|f| f.severity == PreflightSeverity::Warning)
    }

    /// Adds a finding to the report.
    pub fn push(
        &mut self,
        severity: PreflightSeverity,
        code: &'static str,
        message: impl Into<String>,
    ) {
        self.findings
            .push(PreflightFinding::new(severity, code, message));
    }

    /// Renders the report as a stable line-oriented string.
    pub fn to_text(&self) -> String {
        if self.findings.is_empty() {
            return "preflight: no findings\n".to_string();
        }
        let mut out = String::new();
        for finding in &self.findings {
            out.push_str(&format!(
                "{:?}: {} — {}\n",
                finding.severity, finding.code, finding.message
            ));
        }
        out
    }
}

/// Runs preflight checks for a single binding-probe configuration.
pub fn preflight_binding_config(config: &BindingProbeConfig) -> PreflightReport {
    let mut report = PreflightReport {
        findings: Vec::new(),
    };
    if config.dimension == 0 {
        report.push(
            PreflightSeverity::Error,
            "dimension-zero",
            "dimension must be greater than zero",
        );
    }
    if config.dimension < 64 {
        report.push(
            PreflightSeverity::Warning,
            "dimension-low",
            "dimension is very small; use only for smoke tests",
        );
    }
    if config.trials == 0 {
        report.push(
            PreflightSeverity::Error,
            "trials-zero",
            "trials must be greater than zero",
        );
    }
    if config.trials < 8 {
        report.push(
            PreflightSeverity::Warning,
            "trials-low",
            "trial count is low; do not report as a benchmark",
        );
    }
    if !(0.0..=1.0).contains(&config.noise) {
        report.push(
            PreflightSeverity::Error,
            "noise-out-of-range",
            "noise must be in [0, 1]",
        );
    }
    if !(0.0..=1.0).contains(&config.topology_threshold) {
        report.push(
            PreflightSeverity::Error,
            "topology-threshold-out-of-range",
            "topology threshold must be in [0, 1]",
        );
    }
    if config.noise > 0.5 {
        report.push(
            PreflightSeverity::Warning,
            "noise-high",
            "high noise may collapse all methods toward chance; useful for stress tests only",
        );
    }
    if report.findings.is_empty() {
        report.push(
            PreflightSeverity::Info,
            "binding-config-ok",
            "binding configuration passed local preflight checks",
        );
    }
    report
}

/// Runs preflight checks for a noise sweep.
pub fn preflight_noise_sweep_config(config: &NoiseSweepConfig) -> PreflightReport {
    let mut report = preflight_binding_config(&config.base);
    if config.steps == 0 {
        report.push(
            PreflightSeverity::Error,
            "sweep-steps-zero",
            "noise sweep steps must be greater than zero",
        );
    }
    if config.steps < 3 {
        report.push(
            PreflightSeverity::Warning,
            "sweep-steps-low",
            "fewer than three noise levels cannot show a useful curve",
        );
    }
    if !(0.0..=1.0).contains(&config.max_noise) {
        report.push(
            PreflightSeverity::Error,
            "sweep-max-noise-out-of-range",
            "max_noise must be in [0, 1]",
        );
    }
    report
}

/// Runs preflight checks for a replicated comparison.
pub fn preflight_comparative_config(config: &ComparativeBindingConfig) -> PreflightReport {
    let mut report = preflight_binding_config(&config.base);
    if config.replicates == 0 {
        report.push(
            PreflightSeverity::Error,
            "replicates-zero",
            "replicates must be greater than zero",
        );
    }
    if config.replicates < 4 {
        report.push(
            PreflightSeverity::Warning,
            "replicates-low",
            "replicate count is low; report as pilot data",
        );
    }
    report
}

/// Runs preflight checks for an experiment matrix.
pub fn preflight_matrix_config(config: &ExperimentMatrixConfig) -> PreflightReport {
    let mut report = PreflightReport {
        findings: Vec::new(),
    };
    if config.dimensions.is_empty() {
        report.push(
            PreflightSeverity::Error,
            "matrix-dimensions-empty",
            "matrix dimensions must be nonempty",
        );
    }
    for &dimension in &config.dimensions {
        if dimension == 0 {
            report.push(
                PreflightSeverity::Error,
                "matrix-dimension-zero",
                "all matrix dimensions must be greater than zero",
            );
        } else if dimension < 64 {
            report.push(
                PreflightSeverity::Warning,
                "matrix-dimension-low",
                format!("dimension {dimension} is a smoke-test dimension"),
            );
        }
    }
    if config.noise_levels.is_empty() {
        report.push(
            PreflightSeverity::Error,
            "matrix-noise-empty",
            "matrix noise levels must be nonempty",
        );
    }
    for &noise in &config.noise_levels {
        if !(0.0..=1.0).contains(&noise) {
            report.push(
                PreflightSeverity::Error,
                "matrix-noise-out-of-range",
                format!("noise level {noise} is outside [0, 1]"),
            );
        }
    }
    if config.trials == 0 {
        report.push(
            PreflightSeverity::Error,
            "matrix-trials-zero",
            "trials must be greater than zero",
        );
    } else if config.trials < 8 {
        report.push(
            PreflightSeverity::Warning,
            "matrix-trials-low",
            "matrix trial count is low; use as pilot data",
        );
    }
    if config.replicates == 0 {
        report.push(
            PreflightSeverity::Error,
            "matrix-replicates-zero",
            "replicates must be greater than zero",
        );
    } else if config.replicates < 4 {
        report.push(
            PreflightSeverity::Warning,
            "matrix-replicates-low",
            "matrix replicate count is low; use as pilot data",
        );
    }
    if !(0.0..=1.0).contains(&config.topology_threshold) {
        report.push(
            PreflightSeverity::Error,
            "matrix-topology-threshold-out-of-range",
            "topology threshold must be in [0, 1]",
        );
    }
    if report.findings.is_empty() {
        report.push(
            PreflightSeverity::Info,
            "matrix-config-ok",
            "matrix configuration passed local preflight checks",
        );
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binding_preflight_flags_bad_config() {
        let cfg = BindingProbeConfig {
            dimension: 0,
            trials: 0,
            noise: 1.2,
            seed: 1,
            topology_threshold: 0.55,
        };
        let report = preflight_binding_config(&cfg);
        assert!(!report.can_run());
        assert!(report.to_text().contains("dimension-zero"));
    }

    #[test]
    fn matrix_preflight_accepts_default() {
        let report = preflight_matrix_config(&ExperimentMatrixConfig::default());
        assert!(report.can_run());
    }
}
