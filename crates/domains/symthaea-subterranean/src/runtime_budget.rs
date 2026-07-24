// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Real-time control-loop timing contracts.
//!
//! Wall-clock performance is environment-dependent and therefore must not be
//! hidden inside pass/fail unit tests. This module provides an explicit timing
//! campaign with percentile, deadline-miss, and headroom evidence. CI or field
//! hardware can run it under controlled scheduling and archive the JSON report.

use crate::embodiment::SubterraneanEmbodiment;
use serde::{Deserialize, Serialize};
use std::time::Instant;
use symthaea_core::hdc::ContinuousHV;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ControlLoopBudget {
    pub target_hz: f64,
    pub warmup_steps: usize,
    pub measured_steps: usize,
    /// Fraction of the hard frame deadline allocated to p99 execution.
    pub p99_headroom_ratio: f64,
}

impl ControlLoopBudget {
    pub const fn for_200_hz() -> Self {
        Self {
            target_hz: 200.0,
            warmup_steps: 200,
            measured_steps: 2_000,
            p99_headroom_ratio: 0.8,
        }
    }

    pub fn validate(self) -> Result<(), RuntimeBudgetError> {
        if !self.target_hz.is_finite() || self.target_hz <= 0.0 {
            return Err(RuntimeBudgetError::InvalidTargetRate);
        }
        if self.measured_steps == 0 {
            return Err(RuntimeBudgetError::ZeroMeasuredSteps);
        }
        if !self.p99_headroom_ratio.is_finite()
            || self.p99_headroom_ratio <= 0.0
            || self.p99_headroom_ratio > 1.0
        {
            return Err(RuntimeBudgetError::InvalidHeadroomRatio);
        }
        Ok(())
    }

    pub fn deadline_micros(self) -> f64 {
        1_000_000.0 / self.target_hz
    }

    pub fn p99_budget_micros(self) -> f64 {
        self.deadline_micros() * self.p99_headroom_ratio
    }
}

impl Default for ControlLoopBudget {
    fn default() -> Self {
        Self::for_200_hz()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeBudgetError {
    InvalidTargetRate,
    ZeroMeasuredSteps,
    InvalidHeadroomRatio,
}

impl std::fmt::Display for RuntimeBudgetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::InvalidTargetRate => "target_hz must be finite and greater than zero",
            Self::ZeroMeasuredSteps => "measured_steps must be greater than zero",
            Self::InvalidHeadroomRatio => "p99_headroom_ratio must be finite and in (0, 1]",
        };
        f.write_str(message)
    }
}

impl std::error::Error for RuntimeBudgetError {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ControlLoopTimingReport {
    pub target_hz: f64,
    pub warmup_steps: usize,
    pub measured_steps: usize,
    pub deadline_micros: f64,
    pub p99_budget_micros: f64,
    pub min_micros: f64,
    pub mean_micros: f64,
    pub p50_micros: f64,
    pub p95_micros: f64,
    pub p99_micros: f64,
    pub max_micros: f64,
    pub deadline_misses: usize,
    pub p99_within_budget: bool,
    pub no_deadline_misses: bool,
    pub passed: bool,
}

impl ControlLoopTimingReport {
    pub fn to_pretty_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    fn from_nanos(
        budget: ControlLoopBudget,
        mut samples: Vec<u128>,
    ) -> Result<Self, RuntimeBudgetError> {
        budget.validate()?;
        if samples.is_empty() {
            return Err(RuntimeBudgetError::ZeroMeasuredSteps);
        }
        samples.sort_unstable();
        let micros = |nanos: u128| nanos as f64 / 1_000.0;
        let percentile = |ratio: f64| {
            let index = ((samples.len() - 1) as f64 * ratio).round() as usize;
            micros(samples[index.min(samples.len() - 1)])
        };
        let sum_nanos: u128 = samples.iter().copied().sum();
        let mean_micros = sum_nanos as f64 / samples.len() as f64 / 1_000.0;
        let deadline_micros = budget.deadline_micros();
        let p99_budget_micros = budget.p99_budget_micros();
        let deadline_misses = samples
            .iter()
            .filter(|sample| micros(**sample) > deadline_micros)
            .count();
        let p99_micros = percentile(0.99);
        let p99_within_budget = p99_micros <= p99_budget_micros;
        let no_deadline_misses = deadline_misses == 0;
        Ok(Self {
            target_hz: budget.target_hz,
            warmup_steps: budget.warmup_steps,
            measured_steps: samples.len(),
            deadline_micros,
            p99_budget_micros,
            min_micros: micros(samples[0]),
            mean_micros,
            p50_micros: percentile(0.50),
            p95_micros: percentile(0.95),
            p99_micros,
            max_micros: micros(samples[samples.len() - 1]),
            deadline_misses,
            p99_within_budget,
            no_deadline_misses,
            passed: p99_within_budget && no_deadline_misses,
        })
    }
}

pub fn benchmark_control_loop(
    embodiment: &mut SubterraneanEmbodiment,
    thought: &ContinuousHV,
    phi: f64,
    budget: ControlLoopBudget,
) -> Result<ControlLoopTimingReport, RuntimeBudgetError> {
    budget.validate()?;
    let dt = (1.0 / budget.target_hz) as f32;
    for _ in 0..budget.warmup_steps {
        let _ = embodiment.step(thought, dt, phi);
    }

    let mut samples = Vec::with_capacity(budget.measured_steps);
    for _ in 0..budget.measured_steps {
        let started = Instant::now();
        let _ = embodiment.step(thought, dt, phi);
        samples.push(started.elapsed().as_nanos());
    }
    ControlLoopTimingReport::from_nanos(budget, samples)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SubterraneanConfig;
    use symthaea_core::genesis::GenesisSeed;

    #[test]
    fn percentile_report_is_deterministic_for_synthetic_samples() {
        let budget = ControlLoopBudget {
            target_hz: 1_000.0,
            warmup_steps: 0,
            measured_steps: 5,
            p99_headroom_ratio: 0.8,
        };
        let report = ControlLoopTimingReport::from_nanos(
            budget,
            vec![100_000, 200_000, 300_000, 400_000, 500_000],
        )
        .expect("valid timing report");
        assert_eq!(report.min_micros, 100.0);
        assert_eq!(report.p50_micros, 300.0);
        assert_eq!(report.max_micros, 500.0);
        assert_eq!(report.deadline_misses, 0);
        assert!(report.passed);
        assert!(report.to_pretty_json().is_ok());
    }

    #[test]
    fn invalid_budget_is_rejected_before_benchmarking() {
        let budget = ControlLoopBudget {
            target_hz: 0.0,
            ..ControlLoopBudget::default()
        };
        assert_eq!(
            budget.validate(),
            Err(RuntimeBudgetError::InvalidTargetRate)
        );
    }

    /// Run this only on controlled hardware with fixed CPU governor and no
    /// competing workloads. The result is evidence, not a portable unit test.
    #[test]
    #[ignore = "wall-clock benchmark requires controlled execution environment"]
    fn reference_200_hz_control_loop_budget() {
        let genesis = GenesisSeed::from_phrase("subterranean-runtime-budget");
        let mut config = SubterraneanConfig::default();
        config.evidence_capacity = 4_096;
        let mut embodiment = SubterraneanEmbodiment::with_config(&genesis, config);
        let thought = ContinuousHV::random(symthaea_core::hdc::HDC_DIMENSION, 404);
        let report = benchmark_control_loop(
            &mut embodiment,
            &thought,
            0.9,
            ControlLoopBudget::for_200_hz(),
        )
        .expect("valid benchmark configuration");
        eprintln!("{}", report.to_pretty_json().expect("serializable report"));
        assert!(report.passed, "200 Hz timing contract failed: {report:?}");
    }
}
