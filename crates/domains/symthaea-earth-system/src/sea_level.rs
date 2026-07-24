// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Transparent multi-timescale sea-level response baseline.
//!
//! Each component relaxes linearly toward a temperature-dependent equilibrium.
//! The components may be interpreted structurally as faster and slower
//! contributions, but they are not identified with calibrated thermal-
//! expansion, glacier, ice-sheet, or land-water projections unless the caller
//! supplies and documents such a calibration.

use crate::error::{ModelError, require_finite, require_non_negative, require_positive};
use crate::transient::MAX_TRAJECTORY_STEPS;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SeaLevelComponent {
    /// Equilibrium sea-level anomaly per kelvin, m/K.
    pub equilibrium_sensitivity_m_per_k: f64,
    /// E-folding response time, years.
    pub response_time_years: f64,
}

impl SeaLevelComponent {
    pub fn try_new(
        equilibrium_sensitivity_m_per_k: f64,
        response_time_years: f64,
    ) -> Result<Self, ModelError> {
        require_non_negative(
            "equilibrium_sensitivity_m_per_k",
            equilibrium_sensitivity_m_per_k,
        )?;
        require_positive("response_time_years", response_time_years)?;
        Ok(Self {
            equilibrium_sensitivity_m_per_k,
            response_time_years,
        })
    }

    pub fn equilibrium_anomaly_m(&self, warming_k: f64) -> Result<f64, ModelError> {
        require_finite("warming_k", warming_k)?;
        Ok(self.equilibrium_sensitivity_m_per_k * warming_k)
    }

    pub fn tendency_m_per_year(&self, anomaly_m: f64, warming_k: f64) -> Result<f64, ModelError> {
        require_finite("sea_level_anomaly_m", anomaly_m)?;
        let equilibrium = self.equilibrium_anomaly_m(warming_k)?;
        Ok((equilibrium - anomaly_m) / self.response_time_years)
    }

    pub fn exact_constant_warming(
        &self,
        initial_anomaly_m: f64,
        warming_k: f64,
        elapsed_years: f64,
    ) -> Result<f64, ModelError> {
        require_finite("initial_sea_level_anomaly_m", initial_anomaly_m)?;
        require_non_negative("elapsed_years", elapsed_years)?;
        let equilibrium = self.equilibrium_anomaly_m(warming_k)?;
        Ok(equilibrium
            + (initial_anomaly_m - equilibrium) * (-elapsed_years / self.response_time_years).exp())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SeaLevelState {
    pub fast_anomaly_m: f64,
    pub slow_anomaly_m: f64,
}

impl SeaLevelState {
    pub fn total_anomaly_m(&self) -> f64 {
        self.fast_anomaly_m + self.slow_anomaly_m
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SeaLevelResponseModel {
    pub fast: SeaLevelComponent,
    pub slow: SeaLevelComponent,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SeaLevelSample {
    pub time_years: f64,
    pub warming_k: f64,
    pub state: SeaLevelState,
    pub total_anomaly_m: f64,
}

impl SeaLevelResponseModel {
    pub fn try_new(fast: SeaLevelComponent, slow: SeaLevelComponent) -> Result<Self, ModelError> {
        let model = Self { fast, slow };
        model.validate()?;
        Ok(model)
    }

    /// Illustrative fast/slow response parameters for numerical experiments.
    /// They are not an assessed sea-level projection.
    pub fn illustrative() -> Self {
        Self {
            fast: SeaLevelComponent {
                equilibrium_sensitivity_m_per_k: 0.30,
                response_time_years: 30.0,
            },
            slow: SeaLevelComponent {
                equilibrium_sensitivity_m_per_k: 5.0,
                response_time_years: 1_000.0,
            },
        }
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        SeaLevelComponent::try_new(
            self.fast.equilibrium_sensitivity_m_per_k,
            self.fast.response_time_years,
        )?;
        SeaLevelComponent::try_new(
            self.slow.equilibrium_sensitivity_m_per_k,
            self.slow.response_time_years,
        )?;
        Ok(())
    }

    pub fn equilibrium_state(&self, warming_k: f64) -> Result<SeaLevelState, ModelError> {
        self.validate()?;
        Ok(SeaLevelState {
            fast_anomaly_m: self.fast.equilibrium_anomaly_m(warming_k)?,
            slow_anomaly_m: self.slow.equilibrium_anomaly_m(warming_k)?,
        })
    }

    pub fn exact_constant_warming(
        &self,
        initial_state: SeaLevelState,
        warming_k: f64,
        elapsed_years: f64,
    ) -> Result<SeaLevelState, ModelError> {
        self.validate()?;
        self.validate_state(initial_state, "initial")?;
        Ok(SeaLevelState {
            fast_anomaly_m: self.fast.exact_constant_warming(
                initial_state.fast_anomaly_m,
                warming_k,
                elapsed_years,
            )?,
            slow_anomaly_m: self.slow.exact_constant_warming(
                initial_state.slow_anomaly_m,
                warming_k,
                elapsed_years,
            )?,
        })
    }

    pub fn tendencies(
        &self,
        state: SeaLevelState,
        warming_k: f64,
    ) -> Result<SeaLevelState, ModelError> {
        self.validate()?;
        self.validate_state(state, "state")?;
        Ok(SeaLevelState {
            fast_anomaly_m: self
                .fast
                .tendency_m_per_year(state.fast_anomaly_m, warming_k)?,
            slow_anomaly_m: self
                .slow
                .tendency_m_per_year(state.slow_anomaly_m, warming_k)?,
        })
    }

    pub fn step_rk4(
        &self,
        state: SeaLevelState,
        warming_k: f64,
        dt_years: f64,
    ) -> Result<SeaLevelState, ModelError> {
        require_positive("dt_years", dt_years)?;
        let k1 = self.tendencies(state, warming_k)?;
        let stage2 = add_scaled(state, k1, 0.5 * dt_years);
        self.validate_state(stage2, "stage_2")?;
        let k2 = self.tendencies(stage2, warming_k)?;
        let stage3 = add_scaled(state, k2, 0.5 * dt_years);
        self.validate_state(stage3, "stage_3")?;
        let k3 = self.tendencies(stage3, warming_k)?;
        let stage4 = add_scaled(state, k3, dt_years);
        self.validate_state(stage4, "stage_4")?;
        let k4 = self.tendencies(stage4, warming_k)?;
        let next = SeaLevelState {
            fast_anomaly_m: state.fast_anomaly_m
                + dt_years
                    * (k1.fast_anomaly_m
                        + 2.0 * k2.fast_anomaly_m
                        + 2.0 * k3.fast_anomaly_m
                        + k4.fast_anomaly_m)
                    / 6.0,
            slow_anomaly_m: state.slow_anomaly_m
                + dt_years
                    * (k1.slow_anomaly_m
                        + 2.0 * k2.slow_anomaly_m
                        + 2.0 * k3.slow_anomaly_m
                        + k4.slow_anomaly_m)
                    / 6.0,
        };
        self.validate_state(next, "final")?;
        Ok(next)
    }

    pub fn simulate_constant_warming(
        &self,
        initial_state: SeaLevelState,
        warming_k: f64,
        dt_years: f64,
        steps: usize,
    ) -> Result<Vec<SeaLevelSample>, ModelError> {
        self.validate()?;
        self.validate_state(initial_state, "initial")?;
        require_finite("warming_k", warming_k)?;
        require_positive("dt_years", dt_years)?;
        if steps == 0 {
            return Err(ModelError::ZeroSteps);
        }
        if steps > MAX_TRAJECTORY_STEPS {
            return Err(ModelError::TrajectoryTooLarge {
                requested: steps,
                maximum: MAX_TRAJECTORY_STEPS,
            });
        }
        require_finite("duration_years", dt_years * steps as f64)?;
        let capacity = steps.checked_add(1).ok_or(ModelError::TrajectoryTooLarge {
            requested: usize::MAX,
            maximum: MAX_TRAJECTORY_STEPS,
        })?;
        let mut samples = Vec::with_capacity(capacity);
        let mut state = initial_state;
        samples.push(SeaLevelSample {
            time_years: 0.0,
            warming_k,
            state,
            total_anomaly_m: state.total_anomaly_m(),
        });
        for step in 1..=steps {
            state = self.step_rk4(state, warming_k, dt_years)?;
            samples.push(SeaLevelSample {
                time_years: step as f64 * dt_years,
                warming_k,
                state,
                total_anomaly_m: state.total_anomaly_m(),
            });
        }
        Ok(samples)
    }

    fn validate_state(&self, state: SeaLevelState, stage: &'static str) -> Result<(), ModelError> {
        let fast_name = match stage {
            "initial" => "fast_sea_level_initial_m",
            "stage_2" => "fast_sea_level_stage_2_m",
            "stage_3" => "fast_sea_level_stage_3_m",
            "stage_4" => "fast_sea_level_stage_4_m",
            _ => "fast_sea_level_m",
        };
        let slow_name = match stage {
            "initial" => "slow_sea_level_initial_m",
            "stage_2" => "slow_sea_level_stage_2_m",
            "stage_3" => "slow_sea_level_stage_3_m",
            "stage_4" => "slow_sea_level_stage_4_m",
            _ => "slow_sea_level_m",
        };
        require_finite(fast_name, state.fast_anomaly_m)?;
        require_finite(slow_name, state.slow_anomaly_m)
    }
}

fn add_scaled(state: SeaLevelState, tendency: SeaLevelState, scale: f64) -> SeaLevelState {
    SeaLevelState {
        fast_anomaly_m: state.fast_anomaly_m + scale * tendency.fast_anomaly_m,
        slow_anomaly_m: state.slow_anomaly_m + scale * tendency.slow_anomaly_m,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zero_state() -> SeaLevelState {
        SeaLevelState {
            fast_anomaly_m: 0.0,
            slow_anomaly_m: 0.0,
        }
    }

    #[test]
    fn exact_solution_starts_at_initial_and_relaxes_to_equilibrium() {
        let model = SeaLevelResponseModel::illustrative();
        let initial = SeaLevelState {
            fast_anomaly_m: 0.1,
            slow_anomaly_m: -0.2,
        };
        let at_zero = model.exact_constant_warming(initial, 2.0, 0.0).unwrap();
        assert!((at_zero.fast_anomaly_m - initial.fast_anomaly_m).abs() < 1e-9);
        assert!((at_zero.slow_anomaly_m - initial.slow_anomaly_m).abs() < 1e-9);
        let late = model
            .exact_constant_warming(initial, 2.0, 20_000.0)
            .unwrap();
        let equilibrium = model.equilibrium_state(2.0).unwrap();
        assert!((late.fast_anomaly_m - equilibrium.fast_anomaly_m).abs() < 1e-12);
        assert!((late.slow_anomaly_m - equilibrium.slow_anomaly_m).abs() < 1e-7);
    }

    #[test]
    fn rk4_matches_exact_constant_warming() {
        let model = SeaLevelResponseModel::illustrative();
        let samples = model
            .simulate_constant_warming(zero_state(), 1.5, 0.1, 1_000)
            .unwrap();
        let exact = model
            .exact_constant_warming(zero_state(), 1.5, 100.0)
            .unwrap();
        let numerical = samples.last().unwrap().state;
        assert!((numerical.fast_anomaly_m - exact.fast_anomaly_m).abs() < 1e-10);
        assert!((numerical.slow_anomaly_m - exact.slow_anomaly_m).abs() < 1e-12);
    }

    #[test]
    fn warming_and_cooling_are_signed_responses() {
        let model = SeaLevelResponseModel::illustrative();
        assert!(model.equilibrium_state(2.0).unwrap().total_anomaly_m() > 0.0);
        assert!(model.equilibrium_state(-1.0).unwrap().total_anomaly_m() < 0.0);
    }

    #[test]
    fn mutated_invalid_components_fail_closed() {
        let mut model = SeaLevelResponseModel::illustrative();
        model.fast.response_time_years = 0.0;
        assert!(
            model
                .exact_constant_warming(zero_state(), 1.0, 10.0)
                .is_err()
        );
    }

    #[test]
    fn trajectory_contract_is_bounded_and_includes_initial_state() {
        let model = SeaLevelResponseModel::illustrative();
        let samples = model
            .simulate_constant_warming(zero_state(), 1.0, 1.0, 10)
            .unwrap();
        assert_eq!(samples.len(), 11);
        assert_eq!(samples[0].time_years, 0.0);
        assert_eq!(samples[10].time_years, 10.0);
        assert!(matches!(
            model.simulate_constant_warming(zero_state(), 1.0, 1.0, MAX_TRAJECTORY_STEPS + 1,),
            Err(ModelError::TrajectoryTooLarge { .. })
        ));
    }
}
