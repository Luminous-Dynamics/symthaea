// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Exact two-patch source-sink demography.
//!
//! Local per-capita growth may be positive or negative, while non-negative
//! directional dispersal moves individuals between patches. The resulting
//! linear Metzler system has an exact matrix-exponential solution and a
//! dominant-growth persistence criterion. It is a low-density rescue oracle,
//! not a density-regulated metapopulation model.

use crate::error::{ModelError, require_finite, require_non_negative};
use crate::integration::MAX_TRAJECTORY_STEPS;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SourceSinkState {
    pub first_population: f64,
    pub second_population: f64,
}

impl SourceSinkState {
    pub fn total_population(&self) -> f64 {
        self.first_population + self.second_population
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceSinkRegime {
    PersistentNetwork,
    CriticalNetwork,
    DecliningNetwork,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SourceSinkDiagnostic {
    pub dominant_growth_rate: f64,
    pub subordinate_growth_rate: f64,
    pub regime: SourceSinkRegime,
    pub first_is_local_source: bool,
    pub second_is_local_source: bool,
    pub rescue_effect: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SourceSinkSample {
    pub time: f64,
    pub state: SourceSinkState,
    pub total_population: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TwoPatchSourceSink {
    /// Local per-capita growth rate in patch one.
    pub first_local_growth: f64,
    /// Local per-capita growth rate in patch two.
    pub second_local_growth: f64,
    /// Per-capita movement rate from patch one to patch two.
    pub first_to_second: f64,
    /// Per-capita movement rate from patch two to patch one.
    pub second_to_first: f64,
}

impl TwoPatchSourceSink {
    pub fn try_new(
        first_local_growth: f64,
        second_local_growth: f64,
        first_to_second: f64,
        second_to_first: f64,
    ) -> Result<Self, ModelError> {
        let model = Self {
            first_local_growth,
            second_local_growth,
            first_to_second,
            second_to_first,
        };
        model.validate()?;
        Ok(model)
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        require_finite("first_local_growth", self.first_local_growth)?;
        require_finite("second_local_growth", self.second_local_growth)?;
        require_non_negative("first_to_second", self.first_to_second)?;
        require_non_negative("second_to_first", self.second_to_first)
    }

    pub fn validate_state(&self, state: SourceSinkState) -> Result<(), ModelError> {
        require_non_negative("first_population", state.first_population)?;
        require_non_negative("second_population", state.second_population)
    }

    /// System matrix entries `(a, b, c, d)` for `dN/dt = A N`.
    pub fn matrix(&self) -> Result<(f64, f64, f64, f64), ModelError> {
        self.validate()?;
        Ok((
            self.first_local_growth - self.first_to_second,
            self.second_to_first,
            self.first_to_second,
            self.second_local_growth - self.second_to_first,
        ))
    }

    pub fn growth_rates(&self) -> Result<(f64, f64), ModelError> {
        let (a, b, c, d) = self.matrix()?;
        let midpoint = 0.5 * (a + d);
        let discriminant = (0.5 * (a - d)).powi(2) + b * c;
        let delta = discriminant.sqrt();
        Ok((midpoint + delta, midpoint - delta))
    }

    pub fn diagnostic(&self) -> Result<SourceSinkDiagnostic, ModelError> {
        let (dominant_growth_rate, subordinate_growth_rate) = self.growth_rates()?;
        let scale = dominant_growth_rate
            .abs()
            .max(subordinate_growth_rate.abs())
            .max(1.0);
        let tolerance = 1.0e-12 * scale;
        let regime = if dominant_growth_rate > tolerance {
            SourceSinkRegime::PersistentNetwork
        } else if dominant_growth_rate < -tolerance {
            SourceSinkRegime::DecliningNetwork
        } else {
            SourceSinkRegime::CriticalNetwork
        };
        let first_is_local_source = self.first_local_growth > 0.0;
        let second_is_local_source = self.second_local_growth > 0.0;
        let rescue_effect = regime == SourceSinkRegime::PersistentNetwork
            && first_is_local_source != second_is_local_source;
        Ok(SourceSinkDiagnostic {
            dominant_growth_rate,
            subordinate_growth_rate,
            regime,
            first_is_local_source,
            second_is_local_source,
            rescue_effect,
        })
    }

    pub fn derivative(&self, state: SourceSinkState) -> Result<SourceSinkState, ModelError> {
        self.validate_state(state)?;
        let (a, b, c, d) = self.matrix()?;
        Ok(SourceSinkState {
            first_population: a * state.first_population + b * state.second_population,
            second_population: c * state.first_population + d * state.second_population,
        })
    }

    /// Exact matrix-exponential state after `elapsed_time`.
    pub fn exact_state(
        &self,
        initial_state: SourceSinkState,
        elapsed_time: f64,
    ) -> Result<SourceSinkState, ModelError> {
        self.validate_state(initial_state)?;
        require_non_negative("elapsed_time", elapsed_time)?;
        if elapsed_time == 0.0 {
            return Ok(initial_state);
        }
        let (a, b, c, d) = self.matrix()?;
        let midpoint = 0.5 * (a + d);
        let half_difference = 0.5 * (a - d);
        let delta = (half_difference.powi(2) + b * c).sqrt();
        let scaled_delta = delta * elapsed_time;
        let common = (midpoint * elapsed_time).exp();
        let cosh = scaled_delta.cosh();
        let sinh_over_delta = if delta <= 1.0e-14 {
            elapsed_time
        } else {
            scaled_delta.sinh() / delta
        };
        let b_first =
            half_difference * initial_state.first_population + b * initial_state.second_population;
        let b_second =
            c * initial_state.first_population - half_difference * initial_state.second_population;
        let mut state = SourceSinkState {
            first_population: common
                * (cosh * initial_state.first_population + sinh_over_delta * b_first),
            second_population: common
                * (cosh * initial_state.second_population + sinh_over_delta * b_second),
        };
        require_finite("first_population", state.first_population)?;
        require_finite("second_population", state.second_population)?;
        let scale = initial_state.total_population().max(1.0) * common.abs().max(1.0);
        let tolerance = 1.0e-12 * scale;
        if state.first_population < 0.0 && state.first_population >= -tolerance {
            state.first_population = 0.0;
        }
        if state.second_population < 0.0 && state.second_population >= -tolerance {
            state.second_population = 0.0;
        }
        self.validate_state(state)?;
        Ok(state)
    }

    pub fn exact_trajectory(
        &self,
        initial_state: SourceSinkState,
        dt: f64,
        steps: usize,
    ) -> Result<Vec<SourceSinkSample>, ModelError> {
        self.validate_state(initial_state)?;
        crate::error::require_positive("dt", dt)?;
        if steps == 0 {
            return Err(ModelError::ZeroSteps);
        }
        if steps > MAX_TRAJECTORY_STEPS {
            return Err(ModelError::TrajectoryTooLarge {
                requested: steps,
                maximum: MAX_TRAJECTORY_STEPS,
            });
        }
        let capacity = steps.checked_add(1).ok_or(ModelError::TrajectoryTooLarge {
            requested: usize::MAX,
            maximum: MAX_TRAJECTORY_STEPS,
        })?;
        require_finite("duration", dt * steps as f64)?;
        let mut samples = Vec::with_capacity(capacity);
        for step in 0..=steps {
            let time = step as f64 * dt;
            let state = self.exact_state(initial_state, time)?;
            samples.push(SourceSinkSample {
                time,
                state,
                total_population: state.total_population(),
            });
        }
        Ok(samples)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_patch_can_rescue_a_sink() {
        let model = TwoPatchSourceSink::try_new(0.2, -0.5, 0.1, 0.2).unwrap();
        let diagnostic = model.diagnostic().unwrap();
        assert_eq!(diagnostic.regime, SourceSinkRegime::PersistentNetwork);
        assert!(diagnostic.first_is_local_source);
        assert!(!diagnostic.second_is_local_source);
        assert!(diagnostic.rescue_effect);
        assert!(diagnostic.dominant_growth_rate > 0.0);
    }

    #[test]
    fn migration_cancels_from_total_derivative() {
        let model = TwoPatchSourceSink::try_new(0.2, -0.5, 0.1, 0.2).unwrap();
        let state = SourceSinkState {
            first_population: 30.0,
            second_population: 20.0,
        };
        let derivative = model.derivative(state).unwrap();
        let expected = model.first_local_growth * state.first_population
            + model.second_local_growth * state.second_population;
        assert!((derivative.total_population() - expected).abs() < 1.0e-12);
    }

    #[test]
    fn exact_flow_has_semigroup_property() {
        let model = TwoPatchSourceSink::try_new(0.2, -0.5, 0.1, 0.2).unwrap();
        let initial = SourceSinkState {
            first_population: 10.0,
            second_population: 5.0,
        };
        let direct = model.exact_state(initial, 7.0).unwrap();
        let staged = model
            .exact_state(model.exact_state(initial, 3.0).unwrap(), 4.0)
            .unwrap();
        assert!((direct.first_population - staged.first_population).abs() < 1.0e-11);
        assert!((direct.second_population - staged.second_population).abs() < 1.0e-11);
    }

    #[test]
    fn two_sinks_without_rescue_decline() {
        let model = TwoPatchSourceSink::try_new(-0.1, -0.2, 0.05, 0.05).unwrap();
        assert_eq!(
            model.diagnostic().unwrap().regime,
            SourceSinkRegime::DecliningNetwork
        );
        let samples = model
            .exact_trajectory(
                SourceSinkState {
                    first_population: 10.0,
                    second_population: 10.0,
                },
                1.0,
                20,
            )
            .unwrap();
        assert!(samples.last().unwrap().total_population < samples[0].total_population);
    }
}
