// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Monod chemostat resource-consumer baseline.
//!
//! The model makes resource limitation explicit rather than hiding it inside a
//! carrying capacity. It is a perfectly mixed, constant-volume deterministic
//! reactor with one substrate and one consumer population.

use crate::error::{ModelError, require_finite, require_non_negative, require_positive};
use crate::integration::{MAX_TRAJECTORY_STEPS, validate_trajectory_request};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChemostatRegime {
    Washout,
    BreakEven,
    Persistence,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChemostatState {
    pub substrate: f64,
    pub biomass: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChemostatSample {
    pub time: f64,
    pub state: ChemostatState,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChemostatModel {
    /// Dilution rate, inverse model-time.
    pub dilution_rate: f64,
    /// Substrate concentration in the inflow.
    pub inflow_substrate: f64,
    /// Maximum consumer growth rate, inverse model-time.
    pub maximum_growth_rate: f64,
    /// Monod half-saturation concentration.
    pub half_saturation: f64,
    /// Biomass produced per unit substrate consumed.
    pub yield_coefficient: f64,
}

impl ChemostatModel {
    pub fn try_new(
        dilution_rate: f64,
        inflow_substrate: f64,
        maximum_growth_rate: f64,
        half_saturation: f64,
        yield_coefficient: f64,
    ) -> Result<Self, ModelError> {
        let model = Self {
            dilution_rate,
            inflow_substrate,
            maximum_growth_rate,
            half_saturation,
            yield_coefficient,
        };
        model.validate()?;
        Ok(model)
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        require_positive("dilution_rate", self.dilution_rate)?;
        require_positive("inflow_substrate", self.inflow_substrate)?;
        require_positive("maximum_growth_rate", self.maximum_growth_rate)?;
        require_positive("half_saturation", self.half_saturation)?;
        require_positive("yield_coefficient", self.yield_coefficient)
    }

    pub fn specific_growth_rate(&self, substrate: f64) -> Result<f64, ModelError> {
        self.validate()?;
        require_non_negative("substrate", substrate)?;
        Ok(self.maximum_growth_rate * substrate / (self.half_saturation + substrate))
    }

    pub fn regime(&self) -> Result<ChemostatRegime, ModelError> {
        let inflow_growth = self.specific_growth_rate(self.inflow_substrate)?;
        let scale = self.maximum_growth_rate.max(self.dilution_rate).max(1.0);
        if (inflow_growth - self.dilution_rate).abs() <= 16.0 * f64::EPSILON * scale {
            Ok(ChemostatRegime::BreakEven)
        } else if inflow_growth < self.dilution_rate {
            Ok(ChemostatRegime::Washout)
        } else {
            Ok(ChemostatRegime::Persistence)
        }
    }

    pub fn washout_equilibrium(&self) -> Result<ChemostatState, ModelError> {
        self.validate()?;
        Ok(ChemostatState {
            substrate: self.inflow_substrate,
            biomass: 0.0,
        })
    }

    pub fn coexistence_equilibrium(&self) -> Result<Option<ChemostatState>, ModelError> {
        self.validate()?;
        if self.regime()? != ChemostatRegime::Persistence {
            return Ok(None);
        }
        let substrate = self.dilution_rate * self.half_saturation
            / (self.maximum_growth_rate - self.dilution_rate);
        let biomass = self.yield_coefficient * (self.inflow_substrate - substrate);
        Ok(Some(ChemostatState { substrate, biomass }))
    }

    pub fn derivatives(&self, state: ChemostatState) -> Result<ChemostatState, ModelError> {
        self.validate_state(state, 0, "state")?;
        let growth = self.specific_growth_rate(state.substrate)?;
        Ok(ChemostatState {
            substrate: self.dilution_rate * (self.inflow_substrate - state.substrate)
                - growth * state.biomass / self.yield_coefficient,
            biomass: (growth - self.dilution_rate) * state.biomass,
        })
    }

    pub fn jacobian(&self, state: ChemostatState) -> Result<[[f64; 2]; 2], ModelError> {
        self.validate()?;
        self.validate_state(state, 0, "state")?;
        let denominator = self.half_saturation + state.substrate;
        let growth = self.maximum_growth_rate * state.substrate / denominator;
        let growth_derivative =
            self.maximum_growth_rate * self.half_saturation / denominator.powi(2);
        Ok([
            [
                -self.dilution_rate - growth_derivative * state.biomass / self.yield_coefficient,
                -growth / self.yield_coefficient,
            ],
            [
                growth_derivative * state.biomass,
                growth - self.dilution_rate,
            ],
        ])
    }

    pub fn step_rk4(
        &self,
        state: ChemostatState,
        dt: f64,
        step: usize,
    ) -> Result<ChemostatState, ModelError> {
        require_positive("dt", dt)?;
        self.validate_state(state, step, "state")?;
        let k1 = self.derivatives(state)?;
        let stage2 = add_scaled(state, k1, 0.5 * dt);
        self.validate_state(stage2, step, "stage_2")?;
        let k2 = self.derivatives(stage2)?;
        let stage3 = add_scaled(state, k2, 0.5 * dt);
        self.validate_state(stage3, step, "stage_3")?;
        let k3 = self.derivatives(stage3)?;
        let stage4 = add_scaled(state, k3, dt);
        self.validate_state(stage4, step, "stage_4")?;
        let k4 = self.derivatives(stage4)?;
        let next = ChemostatState {
            substrate: state.substrate
                + dt * (k1.substrate + 2.0 * k2.substrate + 2.0 * k3.substrate + k4.substrate)
                    / 6.0,
            biomass: state.biomass
                + dt * (k1.biomass + 2.0 * k2.biomass + 2.0 * k3.biomass + k4.biomass) / 6.0,
        };
        self.validate_state(next, step, "final")?;
        Ok(next)
    }

    pub fn try_simulate(
        &self,
        initial_state: ChemostatState,
        dt: f64,
        steps: usize,
    ) -> Result<Vec<ChemostatSample>, ModelError> {
        self.validate()?;
        validate_trajectory_request(dt, steps)?;
        self.validate_state(initial_state, 0, "initial")?;
        let capacity = steps.checked_add(1).ok_or(ModelError::TrajectoryTooLarge {
            requested: usize::MAX,
            maximum: MAX_TRAJECTORY_STEPS,
        })?;
        let mut samples = Vec::with_capacity(capacity);
        let mut state = initial_state;
        samples.push(ChemostatSample { time: 0.0, state });
        for step in 1..=steps {
            state = self.step_rk4(state, dt, step)?;
            samples.push(ChemostatSample {
                time: step as f64 * dt,
                state,
            });
        }
        Ok(samples)
    }

    fn validate_state(
        &self,
        state: ChemostatState,
        step: usize,
        stage: &'static str,
    ) -> Result<(), ModelError> {
        let substrate_name = match stage {
            "initial" => "initial_substrate",
            "stage_2" => "substrate_stage_2",
            "stage_3" => "substrate_stage_3",
            "stage_4" => "substrate_stage_4",
            _ => "substrate",
        };
        let biomass_name = match stage {
            "initial" => "initial_biomass",
            "stage_2" => "biomass_stage_2",
            "stage_3" => "biomass_stage_3",
            "stage_4" => "biomass_stage_4",
            _ => "biomass",
        };
        if !state.substrate.is_finite() || state.substrate < 0.0 {
            return Err(ModelError::IntegrationDomainViolation {
                step,
                component: substrate_name,
                value: state.substrate,
            });
        }
        if !state.biomass.is_finite() || state.biomass < 0.0 {
            return Err(ModelError::IntegrationDomainViolation {
                step,
                component: biomass_name,
                value: state.biomass,
            });
        }
        Ok(())
    }
}

fn add_scaled(state: ChemostatState, derivative: ChemostatState, scale: f64) -> ChemostatState {
    ChemostatState {
        substrate: state.substrate + scale * derivative.substrate,
        biomass: state.biomass + scale * derivative.biomass,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn persistent_model() -> ChemostatModel {
        ChemostatModel::try_new(0.5, 10.0, 2.0, 1.0, 0.5).unwrap()
    }

    #[test]
    fn washout_and_persistence_follow_the_inflow_growth_threshold() {
        let washout = ChemostatModel::try_new(1.5, 1.0, 2.0, 1.0, 0.5).unwrap();
        assert_eq!(washout.regime().unwrap(), ChemostatRegime::Washout);
        assert!(washout.coexistence_equilibrium().unwrap().is_none());

        let persistent = persistent_model();
        assert_eq!(persistent.regime().unwrap(), ChemostatRegime::Persistence);
        assert!(persistent.coexistence_equilibrium().unwrap().is_some());
    }

    #[test]
    fn coexistence_equilibrium_zeroes_both_tendencies() {
        let model = persistent_model();
        let equilibrium = model.coexistence_equilibrium().unwrap().unwrap();
        let derivative = model.derivatives(equilibrium).unwrap();
        assert!(derivative.substrate.abs() < 1e-12);
        assert!(derivative.biomass.abs() < 1e-12);
    }

    #[test]
    fn resolved_trajectory_converges_to_coexistence() {
        let model = persistent_model();
        let equilibrium = model.coexistence_equilibrium().unwrap().unwrap();
        let samples = model
            .try_simulate(
                ChemostatState {
                    substrate: 8.0,
                    biomass: 0.5,
                },
                0.01,
                5_000,
            )
            .unwrap();
        let final_state = samples.last().unwrap().state;
        assert!((final_state.substrate - equilibrium.substrate).abs() < 1e-6);
        assert!((final_state.biomass - equilibrium.biomass).abs() < 1e-6);
    }

    #[test]
    fn washout_equilibrium_is_invariant() {
        let model = ChemostatModel::try_new(1.5, 1.0, 2.0, 1.0, 0.5).unwrap();
        let washout = model.washout_equilibrium().unwrap();
        let derivative = model.derivatives(washout).unwrap();
        assert_eq!(
            derivative,
            ChemostatState {
                substrate: 0.0,
                biomass: 0.0
            }
        );
    }

    #[test]
    fn mutated_invalid_model_is_rejected_by_analysis_methods() {
        let mut model = persistent_model();
        model.yield_coefficient = 0.0;
        assert!(
            model
                .jacobian(ChemostatState {
                    substrate: 1.0,
                    biomass: 1.0
                })
                .is_err()
        );
    }

    #[test]
    fn oversized_step_fails_before_negative_substrate_propagates() {
        let model = persistent_model();
        let result = model.try_simulate(
            ChemostatState {
                substrate: 0.1,
                biomass: 100.0,
            },
            10.0,
            1,
        );
        assert!(matches!(
            result,
            Err(ModelError::IntegrationDomainViolation { .. })
        ));
    }
}
