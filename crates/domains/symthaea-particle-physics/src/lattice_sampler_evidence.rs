// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Reproducible evidence contracts for lattice sampler comparisons.
//!
//! This module deliberately does not run a sampler, estimate autocorrelation,
//! or select a winner. It binds already-computed chain statistics and timing
//! measurements to an exact transition-kernel identity, benchmark contract,
//! execution environment, and immutable artifacts so later ESS/compute claims
//! cannot silently compare different physics targets or different machines.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SamplerObservableClass {
    FastGauge,
    Topological,
    Spectroscopy,
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransitionKernelLineage {
    /// Version-stable transition identity, e.g. `cm_metropolis_v1:...`.
    pub sampler_id: String,
    /// Digest of the canonical sampler-only configuration.
    pub sampler_config_digest: String,
    /// Exact implementation revision or immutable external version label.
    pub code_revision: String,
    /// Evidence establishing the transition-law / local-kernel semantics.
    pub transition_law_evidence_id: String,
    /// Exact-head build/test qualification for this implementation revision.
    pub exact_head_ci_evidence_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimingEnvironmentLineage {
    /// Canonical digest of the complete timing environment declaration.
    pub environment_digest: String,
    pub architecture: String,
    pub cpu_model: String,
    pub toolchain: String,
    pub build_profile: String,
    pub logical_threads: usize,
    /// Timer/clock source used for wall timing.
    pub timing_source: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SamplerEfficiencyObservation {
    /// Digest of the common benchmark target: lattice/action/beta/start policy,
    /// burn-in, measurement schedule, observable definition, and campaign rules.
    pub benchmark_contract_digest: String,
    pub kernel: TransitionKernelLineage,
    pub environment: TimingEnvironmentLineage,
    pub observable_id: String,
    pub observable_class: SamplerObservableClass,
    /// Descriptive stationary estimate from the retained chain.
    pub mean: f64,
    pub standard_error: f64,
    /// ESS supplied by an independently identified chain-statistics analysis.
    pub effective_sample_size: f64,
    pub raw_measurements: usize,
    pub transition_cycles: u64,
    pub stochastic_subgroup_updates: u64,
    pub deterministic_subgroup_updates: u64,
    pub wall_seconds: f64,
    pub cpu_seconds: f64,
    pub peak_resident_bytes: u64,
    pub chain_statistics_evidence_id: String,
    pub output_artifact_digest: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct StationaryEstimateComparison {
    pub first_mean: f64,
    pub second_mean: f64,
    pub difference: f64,
    pub combined_standard_error: f64,
    /// `abs(first_mean-second_mean) / combined_standard_error`.
    pub normalized_difference: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SamplerEfficiencyComparison {
    pub first_ess_per_wall_second: f64,
    pub second_ess_per_wall_second: f64,
    /// `second / first`; descriptive only, not an automatic winner rule.
    pub ess_per_wall_second_ratio_second_over_first: f64,
    pub first_ess_per_cpu_second: f64,
    pub second_ess_per_cpu_second: f64,
    pub ess_per_cpu_second_ratio_second_over_first: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SamplerEvidenceError {
    EmptyField(&'static str),
    InvalidSha256Digest { field: &'static str, value: String },
    MissingExactHeadCi,
    InvalidThreadCount(usize),
    InvalidRawMeasurements(usize),
    InvalidTransitionCycles(u64),
    InvalidMean(f64),
    InvalidStandardError(f64),
    InvalidEffectiveSampleSize(f64),
    InvalidWallSeconds(f64),
    InvalidCpuSeconds(f64),
    BenchmarkContractMismatch,
    ObservableMismatch,
    ObservableClassMismatch,
    TimingEnvironmentMismatch,
}

fn require_nonempty(value: &str, field: &'static str) -> Result<(), SamplerEvidenceError> {
    if value.trim().is_empty() {
        Err(SamplerEvidenceError::EmptyField(field))
    } else {
        Ok(())
    }
}

fn require_sha256(value: &str, field: &'static str) -> Result<(), SamplerEvidenceError> {
    let Some(hex) = value.strip_prefix("sha256:") else {
        return Err(SamplerEvidenceError::InvalidSha256Digest {
            field,
            value: value.to_string(),
        });
    };
    if hex.len() != 64 || !hex.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(SamplerEvidenceError::InvalidSha256Digest {
            field,
            value: value.to_string(),
        });
    }
    Ok(())
}

impl TransitionKernelLineage {
    pub fn validate(&self, require_exact_head_ci: bool) -> Result<(), SamplerEvidenceError> {
        require_nonempty(&self.sampler_id, "kernel.sampler_id")?;
        require_nonempty(&self.code_revision, "kernel.code_revision")?;
        require_nonempty(
            &self.transition_law_evidence_id,
            "kernel.transition_law_evidence_id",
        )?;
        require_sha256(&self.sampler_config_digest, "kernel.sampler_config_digest")?;
        if require_exact_head_ci {
            require_nonempty(
                self.exact_head_ci_evidence_id
                    .as_deref()
                    .ok_or(SamplerEvidenceError::MissingExactHeadCi)?,
                "kernel.exact_head_ci_evidence_id",
            )?;
        } else if let Some(receipt) = &self.exact_head_ci_evidence_id {
            require_nonempty(receipt, "kernel.exact_head_ci_evidence_id")?;
        }
        Ok(())
    }
}

impl TimingEnvironmentLineage {
    pub fn validate(&self) -> Result<(), SamplerEvidenceError> {
        require_sha256(&self.environment_digest, "environment.environment_digest")?;
        require_nonempty(&self.architecture, "environment.architecture")?;
        require_nonempty(&self.cpu_model, "environment.cpu_model")?;
        require_nonempty(&self.toolchain, "environment.toolchain")?;
        require_nonempty(&self.build_profile, "environment.build_profile")?;
        require_nonempty(&self.timing_source, "environment.timing_source")?;
        if self.logical_threads == 0 {
            return Err(SamplerEvidenceError::InvalidThreadCount(0));
        }
        Ok(())
    }
}

impl SamplerEfficiencyObservation {
    /// Validate an observation as a reproducible timing/statistics record.
    ///
    /// `require_exact_head_ci=false` is useful for explicitly provisional pilot
    /// records. Scientific promotion should call this with `true`.
    pub fn validate(&self, require_exact_head_ci: bool) -> Result<(), SamplerEvidenceError> {
        require_sha256(
            &self.benchmark_contract_digest,
            "benchmark_contract_digest",
        )?;
        self.kernel.validate(require_exact_head_ci)?;
        self.environment.validate()?;
        require_nonempty(&self.observable_id, "observable_id")?;
        require_nonempty(
            &self.chain_statistics_evidence_id,
            "chain_statistics_evidence_id",
        )?;
        require_sha256(&self.output_artifact_digest, "output_artifact_digest")?;
        if !self.mean.is_finite() {
            return Err(SamplerEvidenceError::InvalidMean(self.mean));
        }
        if !self.standard_error.is_finite() || self.standard_error <= 0.0 {
            return Err(SamplerEvidenceError::InvalidStandardError(
                self.standard_error,
            ));
        }
        if !self.effective_sample_size.is_finite() || self.effective_sample_size <= 0.0 {
            return Err(SamplerEvidenceError::InvalidEffectiveSampleSize(
                self.effective_sample_size,
            ));
        }
        if self.raw_measurements == 0 {
            return Err(SamplerEvidenceError::InvalidRawMeasurements(0));
        }
        if self.transition_cycles == 0 {
            return Err(SamplerEvidenceError::InvalidTransitionCycles(0));
        }
        if !self.wall_seconds.is_finite() || self.wall_seconds <= 0.0 {
            return Err(SamplerEvidenceError::InvalidWallSeconds(self.wall_seconds));
        }
        if !self.cpu_seconds.is_finite() || self.cpu_seconds <= 0.0 {
            return Err(SamplerEvidenceError::InvalidCpuSeconds(self.cpu_seconds));
        }
        Ok(())
    }

    pub fn ess_per_wall_second(&self) -> f64 {
        self.effective_sample_size / self.wall_seconds
    }

    pub fn ess_per_cpu_second(&self) -> f64 {
        self.effective_sample_size / self.cpu_seconds
    }

    pub fn ess_per_transition_cycle(&self) -> f64 {
        self.effective_sample_size / self.transition_cycles as f64
    }

    pub fn ess_per_stochastic_subgroup_update(&self) -> Option<f64> {
        (self.stochastic_subgroup_updates > 0).then_some(
            self.effective_sample_size / self.stochastic_subgroup_updates as f64,
        )
    }
}

/// Compare stationary estimates only. Timing environments need not match.
///
/// This function deliberately encodes no pass/fail threshold.
pub fn compare_stationary_estimates(
    first: &SamplerEfficiencyObservation,
    second: &SamplerEfficiencyObservation,
) -> Result<StationaryEstimateComparison, SamplerEvidenceError> {
    first.validate(false)?;
    second.validate(false)?;
    if first.benchmark_contract_digest != second.benchmark_contract_digest {
        return Err(SamplerEvidenceError::BenchmarkContractMismatch);
    }
    if first.observable_id != second.observable_id {
        return Err(SamplerEvidenceError::ObservableMismatch);
    }
    if first.observable_class != second.observable_class {
        return Err(SamplerEvidenceError::ObservableClassMismatch);
    }
    let difference = first.mean - second.mean;
    let combined_standard_error =
        (first.standard_error.powi(2) + second.standard_error.powi(2)).sqrt();
    Ok(StationaryEstimateComparison {
        first_mean: first.mean,
        second_mean: second.mean,
        difference,
        combined_standard_error,
        normalized_difference: difference.abs() / combined_standard_error,
    })
}

/// Compare compute efficiency under an identical benchmark and timing environment.
///
/// Ratios are descriptive. In particular, no threshold or automatic winner is
/// encoded because a sampler may improve a fast observable while degrading
/// topology or another scientifically important slow mode.
pub fn compare_efficiency(
    first: &SamplerEfficiencyObservation,
    second: &SamplerEfficiencyObservation,
) -> Result<SamplerEfficiencyComparison, SamplerEvidenceError> {
    first.validate(true)?;
    second.validate(true)?;
    if first.benchmark_contract_digest != second.benchmark_contract_digest {
        return Err(SamplerEvidenceError::BenchmarkContractMismatch);
    }
    if first.observable_id != second.observable_id {
        return Err(SamplerEvidenceError::ObservableMismatch);
    }
    if first.observable_class != second.observable_class {
        return Err(SamplerEvidenceError::ObservableClassMismatch);
    }
    if first.environment.environment_digest != second.environment.environment_digest {
        return Err(SamplerEvidenceError::TimingEnvironmentMismatch);
    }
    let first_wall = first.ess_per_wall_second();
    let second_wall = second.ess_per_wall_second();
    let first_cpu = first.ess_per_cpu_second();
    let second_cpu = second.ess_per_cpu_second();
    Ok(SamplerEfficiencyComparison {
        first_ess_per_wall_second: first_wall,
        second_ess_per_wall_second: second_wall,
        ess_per_wall_second_ratio_second_over_first: second_wall / first_wall,
        first_ess_per_cpu_second: first_cpu,
        second_ess_per_cpu_second: second_cpu,
        ess_per_cpu_second_ratio_second_over_first: second_cpu / first_cpu,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(ch: char) -> String {
        format!("sha256:{}", ch.to_string().repeat(64))
    }

    fn observation(sampler_id: &str, mean: f64, ess: f64, wall: f64) -> SamplerEfficiencyObservation {
        SamplerEfficiencyObservation {
            benchmark_contract_digest: digest('a'),
            kernel: TransitionKernelLineage {
                sampler_id: sampler_id.into(),
                sampler_config_digest: digest('b'),
                code_revision: "0123456789abcdef".into(),
                transition_law_evidence_id: "LQCD-transition-law".into(),
                exact_head_ci_evidence_id: Some("ci:exact-head".into()),
            },
            environment: TimingEnvironmentLineage {
                environment_digest: digest('c'),
                architecture: "x86_64".into(),
                cpu_model: "qualification-cpu".into(),
                toolchain: "rustc-qualified".into(),
                build_profile: "release".into(),
                logical_threads: 1,
                timing_source: "monotonic".into(),
            },
            observable_id: "average_plaquette_v1".into(),
            observable_class: SamplerObservableClass::FastGauge,
            mean,
            standard_error: 0.01,
            effective_sample_size: ess,
            raw_measurements: 100,
            transition_cycles: 1_000,
            stochastic_subgroup_updates: 192_000,
            deterministic_subgroup_updates: 0,
            wall_seconds: wall,
            cpu_seconds: wall,
            peak_resident_bytes: 16 * 1024 * 1024,
            chain_statistics_evidence_id: "LQCD-003:chain".into(),
            output_artifact_digest: digest('d'),
        }
    }

    #[test]
    fn efficiency_ratio_is_descriptive_only() {
        let first = observation("cm_metropolis_v1:max_angle=5e-1", 0.60, 25.0, 10.0);
        let mut second = observation("cm_heatbath_or_v1:or=1", 0.601, 60.0, 12.0);
        second.deterministic_subgroup_updates = 192_000;
        let comparison = compare_efficiency(&first, &second).unwrap();
        assert!((comparison.first_ess_per_wall_second - 2.5).abs() < 1e-12);
        assert!((comparison.second_ess_per_wall_second - 5.0).abs() < 1e-12);
        assert!((comparison.ess_per_wall_second_ratio_second_over_first - 2.0).abs() < 1e-12);
    }

    #[test]
    fn efficiency_comparison_requires_same_timing_environment() {
        let first = observation("a", 0.60, 25.0, 10.0);
        let mut second = observation("b", 0.60, 30.0, 10.0);
        second.environment.environment_digest = digest('e');
        assert!(matches!(
            compare_efficiency(&first, &second),
            Err(SamplerEvidenceError::TimingEnvironmentMismatch)
        ));
    }

    #[test]
    fn stationary_comparison_can_cross_timing_environments() {
        let first = observation("a", 0.60, 25.0, 10.0);
        let mut second = observation("b", 0.62, 30.0, 10.0);
        second.environment.environment_digest = digest('e');
        let comparison = compare_stationary_estimates(&first, &second).unwrap();
        assert!((comparison.difference + 0.02).abs() < 1e-12);
        assert!((comparison.combined_standard_error - (2.0f64).sqrt() * 0.01).abs() < 1e-12);
    }

    #[test]
    fn benchmark_mismatch_fails_closed() {
        let first = observation("a", 0.60, 25.0, 10.0);
        let mut second = observation("b", 0.60, 30.0, 10.0);
        second.benchmark_contract_digest = digest('f');
        assert!(matches!(
            compare_stationary_estimates(&first, &second),
            Err(SamplerEvidenceError::BenchmarkContractMismatch)
        ));
    }

    #[test]
    fn provisional_record_can_be_valid_without_ci_but_efficiency_cannot() {
        let mut first = observation("a", 0.60, 25.0, 10.0);
        first.kernel.exact_head_ci_evidence_id = None;
        assert!(first.validate(false).is_ok());
        let second = observation("b", 0.60, 30.0, 10.0);
        assert!(matches!(
            compare_efficiency(&first, &second),
            Err(SamplerEvidenceError::MissingExactHeadCi)
        ));
    }

    #[test]
    fn ess_is_not_artificially_capped_at_raw_count() {
        let mut record = observation("negative-autocorrelation-case", 0.60, 120.0, 10.0);
        record.raw_measurements = 100;
        assert!(record.validate(true).is_ok());
    }
}
