// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Stage-structured population projection with Leslie matrices.
//!
//! This module exposes the dominant finite-rate growth factor, net
//! reproductive rate, stable stage distribution, and deterministic projection
//! trajectories without a general linear-algebra dependency.

use crate::error::{ModelError, require_finite, require_non_negative};

pub const MAX_STAGES: usize = 128;
const EIGEN_TOLERANCE: f64 = 1.0e-12;
const MAX_BISECTION_ITERATIONS: usize = 256;

#[derive(Debug, Clone, PartialEq)]
pub struct LeslieMatrix {
    fecundities: Vec<f64>,
    survivals: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AsymptoticGrowth {
    Declining,
    Stationary,
    Growing,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LeslieAnalysis {
    /// Dominant finite-rate growth factor per projection interval.
    pub dominant_growth_factor: f64,
    /// Expected lifetime production of stage-zero recruits.
    pub net_reproductive_rate: f64,
    pub asymptotic_growth: AsymptoticGrowth,
    /// Right Perron eigenvector normalized to sum to one.
    pub stable_stage_distribution: Vec<f64>,
    /// Maximum absolute residual in `L v = lambda v`.
    pub eigen_residual: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StagePopulationSample {
    pub step: usize,
    pub stages: Vec<f64>,
    pub total_population: f64,
}

impl LeslieMatrix {
    pub fn try_new(fecundities: Vec<f64>, survivals: Vec<f64>) -> Result<Self, ModelError> {
        let matrix = Self {
            fecundities,
            survivals,
        };
        matrix.validate()?;
        Ok(matrix)
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        if self.fecundities.len() < 2 {
            return Err(ModelError::InsufficientSamples {
                required: 2,
                found: self.fecundities.len(),
            });
        }
        if self.fecundities.len() > MAX_STAGES {
            return Err(ModelError::OutOfRange {
                parameter: "stage_count",
                value: self.fecundities.len() as f64,
                min: 2.0,
                max: MAX_STAGES as f64,
            });
        }
        if self.survivals.len() + 1 != self.fecundities.len() {
            return Err(ModelError::DimensionMismatch {
                context: "Leslie survival vector",
                expected: self.fecundities.len() - 1,
                found: self.survivals.len(),
            });
        }
        for &fecundity in &self.fecundities {
            require_non_negative("fecundity", fecundity)?;
        }
        for &survival in &self.survivals {
            require_finite("survival", survival)?;
            if !(0.0..=1.0).contains(&survival) {
                return Err(ModelError::OutOfRange {
                    parameter: "survival",
                    value: survival,
                    min: 0.0,
                    max: 1.0,
                });
            }
        }
        if !self.fecundities.iter().any(|value| *value > 0.0) {
            return Err(ModelError::SingularCalibration {
                reason: "at least one fecundity must be positive",
            });
        }
        let net_reproductive_rate = self.net_reproductive_rate_unchecked();
        require_finite("net_reproductive_rate", net_reproductive_rate)?;
        if net_reproductive_rate == 0.0 {
            return Err(ModelError::SingularCalibration {
                reason: "no fecund stage is reachable through survival",
            });
        }
        Ok(())
    }

    pub fn stage_count(&self) -> usize {
        self.fecundities.len()
    }

    pub fn fecundities(&self) -> &[f64] {
        &self.fecundities
    }

    pub fn survivals(&self) -> &[f64] {
        &self.survivals
    }

    pub fn net_reproductive_rate(&self) -> Result<f64, ModelError> {
        self.validate()?;
        Ok(self.net_reproductive_rate_unchecked())
    }

    fn net_reproductive_rate_unchecked(&self) -> f64 {
        let mut survivorship = 1.0;
        let mut total = self.fecundities[0];
        for stage in 1..self.stage_count() {
            survivorship *= self.survivals[stage - 1];
            total += self.fecundities[stage] * survivorship;
        }
        total
    }

    pub fn project(&self, stages: &[f64]) -> Result<Vec<f64>, ModelError> {
        self.validate()?;
        self.validate_stage_vector(stages)?;
        let mut next = vec![0.0; self.stage_count()];
        next[0] = self
            .fecundities
            .iter()
            .zip(stages)
            .map(|(fecundity, population)| fecundity * population)
            .sum();
        for stage in 1..self.stage_count() {
            next[stage] = self.survivals[stage - 1] * stages[stage - 1];
        }
        if next.iter().all(|value| value.is_finite()) {
            Ok(next)
        } else {
            Err(ModelError::IntegrationDomainViolation {
                step: 1,
                component: "stage_population",
                value: f64::NAN,
            })
        }
    }

    pub fn analyze(&self) -> Result<LeslieAnalysis, ModelError> {
        self.validate()?;
        let dominant_growth_factor = self.dominant_growth_factor()?;
        let mut log_weights = Vec::with_capacity(self.stage_count());
        let mut log_survivorship: f64 = 0.0;
        let log_growth = dominant_growth_factor.ln();
        log_weights.push(0.0);
        for stage in 1..self.stage_count() {
            let survival = self.survivals[stage - 1];
            if survival == 0.0 || !log_survivorship.is_finite() {
                log_survivorship = f64::NEG_INFINITY;
            } else {
                log_survivorship += survival.ln();
            }
            log_weights.push(log_survivorship - stage as f64 * log_growth);
        }
        let maximum_log_weight = log_weights
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let mut stable_stage_distribution: Vec<f64> = log_weights
            .iter()
            .map(|value| (*value - maximum_log_weight).exp())
            .collect();
        let sum: f64 = stable_stage_distribution.iter().sum();
        require_finite("stable_stage_distribution_sum", sum)?;
        for value in &mut stable_stage_distribution {
            *value /= sum;
        }
        let projected = self.project(&stable_stage_distribution)?;
        let eigen_residual = projected
            .iter()
            .zip(&stable_stage_distribution)
            .map(|(projected, stable)| (projected - dominant_growth_factor * stable).abs())
            .fold(0.0, f64::max);
        let asymptotic_growth = if (dominant_growth_factor - 1.0).abs() <= 1.0e-10 {
            AsymptoticGrowth::Stationary
        } else if dominant_growth_factor > 1.0 {
            AsymptoticGrowth::Growing
        } else {
            AsymptoticGrowth::Declining
        };
        Ok(LeslieAnalysis {
            dominant_growth_factor,
            net_reproductive_rate: self.net_reproductive_rate_unchecked(),
            asymptotic_growth,
            stable_stage_distribution,
            eigen_residual,
        })
    }

    pub fn project_trajectory(
        &self,
        initial_stages: &[f64],
        steps: usize,
    ) -> Result<Vec<StagePopulationSample>, ModelError> {
        self.validate_stage_vector(initial_stages)?;
        crate::integration::validate_step_count(steps)?;
        let mut stages = initial_stages.to_vec();
        let mut samples = Vec::with_capacity(steps + 1);
        samples.push(sample(0, &stages));
        for step in 1..=steps {
            stages = self.project(&stages)?;
            samples.push(sample(step, &stages));
        }
        Ok(samples)
    }

    fn dominant_growth_factor(&self) -> Result<f64, ModelError> {
        let mut high = self
            .fecundities
            .iter()
            .sum::<f64>()
            .max(self.survivals.iter().copied().fold(0.0, f64::max))
            .max(1.0);
        while self.euler_lotka_residual(high) > 0.0 {
            high *= 2.0;
            if !high.is_finite() {
                return Err(ModelError::NoConvergence {
                    context: "Leslie dominant growth factor upper bracket",
                    iterations: MAX_BISECTION_ITERATIONS,
                });
            }
        }

        let low = f64::from_bits(1);
        if self.euler_lotka_residual(low) <= 0.0 {
            return Err(ModelError::NoConvergence {
                context: "Leslie dominant growth factor lower bracket",
                iterations: MAX_BISECTION_ITERATIONS,
            });
        }
        let mut log_low = low.ln();
        let mut log_high = high.ln();
        for _ in 0..MAX_BISECTION_ITERATIONS {
            let log_middle = 0.5 * (log_low + log_high);
            let middle = log_middle.exp();
            if self.euler_lotka_residual(middle) > 0.0 {
                log_low = log_middle;
            } else {
                log_high = log_middle;
            }
            if (log_high - log_low).abs() <= EIGEN_TOLERANCE {
                return Ok((0.5 * (log_low + log_high)).exp());
            }
        }
        Err(ModelError::NoConvergence {
            context: "Leslie dominant growth factor log-bisection",
            iterations: MAX_BISECTION_ITERATIONS,
        })
    }

    fn euler_lotka_residual(&self, growth_factor: f64) -> f64 {
        let log_growth = growth_factor.ln();
        let mut log_survivorship: f64 = 0.0;
        let mut reproduction = 0.0;
        for stage in 0..self.stage_count() {
            if stage > 0 {
                let survival = self.survivals[stage - 1];
                if survival == 0.0 || !log_survivorship.is_finite() {
                    log_survivorship = f64::NEG_INFINITY;
                } else {
                    log_survivorship += survival.ln();
                }
            }
            let fecundity = self.fecundities[stage];
            if fecundity == 0.0 || !log_survivorship.is_finite() {
                continue;
            }
            let log_term = fecundity.ln() + log_survivorship - (stage as f64 + 1.0) * log_growth;
            if log_term > f64::MAX.ln() {
                return f64::INFINITY;
            }
            reproduction += log_term.exp();
            if !reproduction.is_finite() {
                return f64::INFINITY;
            }
        }
        reproduction - 1.0
    }

    fn validate_stage_vector(&self, stages: &[f64]) -> Result<(), ModelError> {
        if stages.len() != self.stage_count() {
            return Err(ModelError::DimensionMismatch {
                context: "stage population vector",
                expected: self.stage_count(),
                found: stages.len(),
            });
        }
        for &population in stages {
            require_non_negative("stage_population", population)?;
        }
        Ok(())
    }
}

fn sample(step: usize, stages: &[f64]) -> StagePopulationSample {
    StagePopulationSample {
        step,
        stages: stages.to_vec(),
        total_population: stages.iter().sum(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_stage_replacement_case_has_unit_growth() {
        let matrix = LeslieMatrix::try_new(vec![0.0, 2.0], vec![0.5]).unwrap();
        let analysis = matrix.analyze().unwrap();
        assert!((analysis.dominant_growth_factor - 1.0).abs() < 1e-10);
        assert!((analysis.net_reproductive_rate - 1.0).abs() < 1e-12);
        assert_eq!(analysis.asymptotic_growth, AsymptoticGrowth::Stationary);
        assert!((analysis.stable_stage_distribution[0] - 2.0 / 3.0).abs() < 1e-10);
        assert!((analysis.stable_stage_distribution[1] - 1.0 / 3.0).abs() < 1e-10);
        assert!(analysis.eigen_residual < 1e-10);
    }

    #[test]
    fn growth_factor_matches_closed_form_two_stage_case() {
        let matrix = LeslieMatrix::try_new(vec![0.0, 4.0], vec![0.5]).unwrap();
        let analysis = matrix.analyze().unwrap();
        assert!((analysis.dominant_growth_factor - 2.0_f64.sqrt()).abs() < 1e-10);
        assert_eq!(analysis.asymptotic_growth, AsymptoticGrowth::Growing);
    }

    #[test]
    fn projection_preserves_stage_contract() {
        let matrix = LeslieMatrix::try_new(vec![0.0, 2.0], vec![0.5]).unwrap();
        assert_eq!(matrix.project(&[10.0, 5.0]).unwrap(), vec![10.0, 5.0]);
        let samples = matrix.project_trajectory(&[10.0, 5.0], 4).unwrap();
        assert_eq!(samples.len(), 5);
        assert_eq!(samples[0].step, 0);
        assert!(samples.iter().all(|sample| sample.total_population == 15.0));
    }

    #[test]
    fn log_scaled_distribution_handles_extreme_growth_factors() {
        let matrix =
            LeslieMatrix::try_new(vec![1.0e-100, 0.0, 0.0, 0.0, 0.0, 0.0], vec![1.0; 5]).unwrap();
        let analysis = matrix.analyze().unwrap();
        assert!(
            analysis
                .stable_stage_distribution
                .iter()
                .all(|value| value.is_finite())
        );
        assert!((analysis.stable_stage_distribution.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(analysis.eigen_residual < 1e-12);
    }

    #[test]
    fn invalid_dimensions_and_survival_are_rejected() {
        assert!(LeslieMatrix::try_new(vec![1.0, 1.0], vec![]).is_err());
        assert!(LeslieMatrix::try_new(vec![1.0, 1.0], vec![1.2]).is_err());
        let matrix = LeslieMatrix::try_new(vec![0.0, 2.0], vec![0.5]).unwrap();
        assert!(matrix.project(&[1.0]).is_err());
    }
}
