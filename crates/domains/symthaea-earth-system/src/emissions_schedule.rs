// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Exact propagation through piecewise-constant emissions pathways.
//!
//! Each stage declares a duration and a constant emissions rate. Because the
//! reduced two-box carbon cycle is linear, stages can be propagated exactly
//! without numerical time stepping. This makes mitigation, overshoot, and
//! removal pathways independently auditable while preserving the model's
//! carbon-budget invariant.

use crate::carbon_cycle::{CarbonState, TwoBoxCarbonCycle};
use crate::error::{ModelError, require_finite, require_positive};

pub const MAX_EMISSIONS_STAGES: usize = 4096;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EmissionsStage {
    /// Stage duration in years.
    pub duration_years: f64,
    /// Constant emissions rate in GtC/year. Negative values represent removal.
    pub rate_gtc_per_year: f64,
}

impl EmissionsStage {
    pub fn try_new(duration_years: f64, rate_gtc_per_year: f64) -> Result<Self, ModelError> {
        require_positive("duration_years", duration_years)?;
        require_finite("rate_gtc_per_year", rate_gtc_per_year)?;
        Ok(Self {
            duration_years,
            rate_gtc_per_year,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PiecewiseConstantEmissions {
    stages: Vec<EmissionsStage>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScheduledCarbonSample {
    /// Time at the stage boundary, years since the pathway began.
    pub time_years: f64,
    pub state: CarbonState,
    /// Cumulative pathway emissions through this boundary, GtC.
    pub cumulative_emissions_gtc: f64,
}

impl PiecewiseConstantEmissions {
    pub fn try_new(stages: Vec<EmissionsStage>) -> Result<Self, ModelError> {
        if stages.is_empty() {
            return Err(ModelError::EmptySeries {
                series: "emissions_stages",
            });
        }
        if stages.len() > MAX_EMISSIONS_STAGES {
            return Err(ModelError::ScheduleTooLarge {
                requested: stages.len(),
                maximum: MAX_EMISSIONS_STAGES,
            });
        }
        let pathway = Self { stages };
        pathway.validate()?;
        Ok(pathway)
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        if self.stages.is_empty() {
            return Err(ModelError::EmptySeries {
                series: "emissions_stages",
            });
        }
        if self.stages.len() > MAX_EMISSIONS_STAGES {
            return Err(ModelError::ScheduleTooLarge {
                requested: self.stages.len(),
                maximum: MAX_EMISSIONS_STAGES,
            });
        }
        let mut duration = 0.0;
        let mut cumulative = 0.0;
        for stage in &self.stages {
            require_positive("duration_years", stage.duration_years)?;
            require_finite("rate_gtc_per_year", stage.rate_gtc_per_year)?;
            duration += stage.duration_years;
            require_finite("pathway_duration_years", duration)?;
            cumulative += stage.duration_years * stage.rate_gtc_per_year;
            require_finite("pathway_cumulative_emissions_gtc", cumulative)?;
        }
        Ok(())
    }

    pub fn stages(&self) -> &[EmissionsStage] {
        &self.stages
    }

    pub fn duration_years(&self) -> f64 {
        self.stages.iter().map(|stage| stage.duration_years).sum()
    }

    pub fn cumulative_emissions_gtc(&self) -> f64 {
        self.stages
            .iter()
            .map(|stage| stage.duration_years * stage.rate_gtc_per_year)
            .sum()
    }

    /// Emissions rate at a time inside the pathway. The final boundary uses the
    /// final stage's rate; times beyond the pathway are rejected.
    pub fn rate_at(&self, time_years: f64) -> Result<f64, ModelError> {
        self.validate()?;
        require_finite("time_years", time_years)?;
        let duration = self.duration_years();
        if !(0.0..=duration).contains(&time_years) {
            return Err(ModelError::OutOfRange {
                parameter: "time_years",
                value: time_years,
                min: 0.0,
                max: duration,
            });
        }
        let mut boundary = 0.0;
        for stage in &self.stages {
            boundary += stage.duration_years;
            if time_years < boundary {
                return Ok(stage.rate_gtc_per_year);
            }
        }
        Ok(self.stages[self.stages.len() - 1].rate_gtc_per_year)
    }

    /// Exact carbon states at the initial time and every stage boundary.
    pub fn exact_boundary_trajectory(
        &self,
        model: &TwoBoxCarbonCycle,
        initial_state: CarbonState,
    ) -> Result<Vec<ScheduledCarbonSample>, ModelError> {
        self.validate()?;
        model.validate()?;
        let capacity = self
            .stages
            .len()
            .checked_add(1)
            .ok_or(ModelError::ScheduleTooLarge {
                requested: usize::MAX,
                maximum: MAX_EMISSIONS_STAGES,
            })?;
        let mut samples = Vec::with_capacity(capacity);
        let mut state = initial_state;
        let mut time_years = 0.0;
        let mut cumulative_emissions_gtc = 0.0;
        // Validate the initial state through a zero-duration exact propagation.
        state = model.exact_constant_emissions(state, 0.0, 0.0)?;
        samples.push(ScheduledCarbonSample {
            time_years,
            state,
            cumulative_emissions_gtc,
        });
        for stage in &self.stages {
            state = model.exact_constant_emissions(
                state,
                stage.rate_gtc_per_year,
                stage.duration_years,
            )?;
            time_years += stage.duration_years;
            cumulative_emissions_gtc += stage.rate_gtc_per_year * stage.duration_years;
            require_finite("pathway_time_years", time_years)?;
            require_finite("cumulative_emissions_gtc", cumulative_emissions_gtc)?;
            samples.push(ScheduledCarbonSample {
                time_years,
                state,
                cumulative_emissions_gtc,
            });
        }
        Ok(samples)
    }

    pub fn exact_final_state(
        &self,
        model: &TwoBoxCarbonCycle,
        initial_state: CarbonState,
    ) -> Result<CarbonState, ModelError> {
        self.exact_boundary_trajectory(model, initial_state)?
            .last()
            .map(|sample| sample.state)
            .ok_or(ModelError::EmptySeries {
                series: "scheduled_carbon_trajectory",
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stage(duration: f64, rate: f64) -> EmissionsStage {
        EmissionsStage::try_new(duration, rate).unwrap()
    }

    #[test]
    fn pathway_budget_and_rate_contract_are_exact() {
        let pathway = PiecewiseConstantEmissions::try_new(vec![
            stage(10.0, 5.0),
            stage(20.0, 2.0),
            stage(5.0, -3.0),
        ])
        .unwrap();
        assert_eq!(pathway.duration_years(), 35.0);
        assert_eq!(pathway.cumulative_emissions_gtc(), 75.0);
        assert_eq!(pathway.rate_at(0.0).unwrap(), 5.0);
        assert_eq!(pathway.rate_at(10.0).unwrap(), 2.0);
        assert_eq!(pathway.rate_at(35.0).unwrap(), -3.0);
        assert!(pathway.rate_at(35.1).is_err());
    }

    #[test]
    fn one_stage_matches_constant_emissions_oracle() {
        let model = TwoBoxCarbonCycle::illustrative();
        let initial = CarbonState {
            atmosphere_gtc: 12.0,
            reservoir_gtc: 4.0,
        };
        let pathway = PiecewiseConstantEmissions::try_new(vec![stage(40.0, 3.0)]).unwrap();
        let scheduled = pathway.exact_final_state(&model, initial).unwrap();
        let direct = model.exact_constant_emissions(initial, 3.0, 40.0).unwrap();
        assert!((scheduled.atmosphere_gtc - direct.atmosphere_gtc).abs() < 1e-12);
        assert!((scheduled.reservoir_gtc - direct.reservoir_gtc).abs() < 1e-12);
    }

    #[test]
    fn sequential_pathway_preserves_integrated_carbon_budget() {
        let model = TwoBoxCarbonCycle::illustrative();
        let initial = CarbonState {
            atmosphere_gtc: 0.0,
            reservoir_gtc: 0.0,
        };
        let pathway = PiecewiseConstantEmissions::try_new(vec![
            stage(20.0, 8.0),
            stage(20.0, 4.0),
            stage(20.0, 0.0),
            stage(10.0, -2.0),
        ])
        .unwrap();
        let samples = pathway.exact_boundary_trajectory(&model, initial).unwrap();
        assert_eq!(samples.len(), 5);
        let final_sample = samples.last().unwrap();
        let residual = model
            .integrated_mass_balance_residual(
                initial,
                final_sample.state,
                final_sample.cumulative_emissions_gtc,
            )
            .unwrap();
        assert!(residual.abs() < 1e-10);
    }

    #[test]
    fn overflowed_pathway_budgets_are_rejected() {
        assert!(PiecewiseConstantEmissions::try_new(vec![stage(f64::MAX, 2.0),]).is_err());
    }

    #[test]
    fn empty_or_oversized_pathways_fail_before_allocation() {
        assert!(PiecewiseConstantEmissions::try_new(Vec::new()).is_err());
        let stages = vec![stage(1.0, 0.0); MAX_EMISSIONS_STAGES + 1];
        assert!(matches!(
            PiecewiseConstantEmissions::try_new(stages),
            Err(ModelError::ScheduleTooLarge { .. })
        ));
    }
}
