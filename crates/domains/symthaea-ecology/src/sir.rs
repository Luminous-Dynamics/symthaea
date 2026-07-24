// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Closed-population susceptible-infectious-removed epidemic oracle.
//!
//! This is the frequency-dependent Kermack-McKendrick SIR baseline with fixed
//! population, homogeneous mixing, immediate infectiousness, exponential
//! removal, and permanent removal from susceptibility. It exposes the epidemic
//! threshold, conserved phase-plane invariant, final-size root, peak diagnostic,
//! and guarded RK4 trajectories. It is not an individual-based transmission
//! model and does not represent births, waning immunity, latency, or networks.

use crate::error::{ModelError, require_finite, require_non_negative, require_positive};
use crate::integration::MAX_TRAJECTORY_STEPS;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SirState {
    pub susceptible: f64,
    pub infectious: f64,
    pub removed: f64,
}

impl SirState {
    pub fn total(&self) -> f64 {
        self.susceptible + self.infectious + self.removed
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpidemicRegime {
    Growing,
    Critical,
    Declining,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EpidemicDiagnostic {
    pub basic_reproduction_number: f64,
    pub effective_reproduction_number: f64,
    pub regime: EpidemicRegime,
    pub peak_susceptible: f64,
    pub predicted_peak_infectious: f64,
    pub final_susceptible: f64,
    pub final_removed: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SirSample {
    pub time: f64,
    pub state: SirState,
    pub effective_reproduction_number: f64,
    pub invariant: f64,
    pub population_residual: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SirModel {
    /// Transmission rate per model-time unit.
    pub transmission_rate: f64,
    /// Removal or recovery rate per model-time unit.
    pub removal_rate: f64,
}

impl SirModel {
    pub fn try_new(transmission_rate: f64, removal_rate: f64) -> Result<Self, ModelError> {
        let model = Self {
            transmission_rate,
            removal_rate,
        };
        model.validate()?;
        Ok(model)
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        require_positive("transmission_rate", self.transmission_rate)?;
        require_positive("removal_rate", self.removal_rate)
    }

    pub fn validate_state(&self, state: SirState) -> Result<(), ModelError> {
        require_non_negative("susceptible", state.susceptible)?;
        require_non_negative("infectious", state.infectious)?;
        require_non_negative("removed", state.removed)?;
        require_positive("total_population", state.total())
    }

    pub fn basic_reproduction_number(&self) -> Result<f64, ModelError> {
        self.validate()?;
        Ok(self.transmission_rate / self.removal_rate)
    }

    pub fn effective_reproduction_number(&self, state: SirState) -> Result<f64, ModelError> {
        self.validate_state(state)?;
        Ok(self.basic_reproduction_number()? * state.susceptible / state.total())
    }

    pub fn regime(&self, state: SirState) -> Result<EpidemicRegime, ModelError> {
        let effective = self.effective_reproduction_number(state)?;
        let tolerance = 1.0e-12 * effective.abs().max(1.0);
        Ok(if effective > 1.0 + tolerance {
            EpidemicRegime::Growing
        } else if effective < 1.0 - tolerance {
            EpidemicRegime::Declining
        } else {
            EpidemicRegime::Critical
        })
    }

    pub fn derivative(&self, state: SirState) -> Result<SirState, ModelError> {
        self.validate_state(state)?;
        let population = state.total();
        let incidence = self.transmission_rate * state.susceptible * state.infectious / population;
        let removal = self.removal_rate * state.infectious;
        Ok(SirState {
            susceptible: -incidence,
            infectious: incidence - removal,
            removed: removal,
        })
    }

    /// Conserved phase-plane quantity for fixed total population.
    pub fn invariant(&self, state: SirState) -> Result<f64, ModelError> {
        self.validate_state(state)?;
        if state.susceptible <= 0.0 {
            return Err(ModelError::NonPositive {
                parameter: "susceptible",
                value: state.susceptible,
            });
        }
        let population = state.total();
        Ok(state.infectious + state.susceptible
            - (self.removal_rate * population / self.transmission_rate) * state.susceptible.ln())
    }

    pub fn final_susceptible(&self, initial: SirState) -> Result<f64, ModelError> {
        self.validate_state(initial)?;
        if initial.susceptible <= 0.0 {
            return Err(ModelError::NonPositive {
                parameter: "susceptible",
                value: initial.susceptible,
            });
        }
        if initial.infectious == 0.0 {
            return Ok(initial.susceptible);
        }
        let population = initial.total();
        let coefficient = self.removal_rate * population / self.transmission_rate;
        let constant =
            initial.infectious + initial.susceptible - coefficient * initial.susceptible.ln();
        let residual = |susceptible: f64| susceptible - coefficient * susceptible.ln() - constant;
        let mut lower = (initial.susceptible * 1.0e-15).max(f64::MIN_POSITIVE);
        let mut upper = initial.susceptible;
        if residual(lower) <= 0.0 || residual(upper) >= 0.0 {
            return Err(ModelError::NoConvergence {
                context: "SIR final-size bracket",
                iterations: 0,
            });
        }
        for _ in 0..200 {
            let midpoint = 0.5 * (lower + upper);
            let value = residual(midpoint);
            if value > 0.0 {
                lower = midpoint;
            } else {
                upper = midpoint;
            }
            if (upper - lower).abs() <= 1.0e-13 * initial.susceptible.max(1.0) {
                return Ok(0.5 * (lower + upper));
            }
        }
        Err(ModelError::NoConvergence {
            context: "SIR final-size root",
            iterations: 200,
        })
    }

    pub fn diagnostic(&self, initial: SirState) -> Result<EpidemicDiagnostic, ModelError> {
        self.validate_state(initial)?;
        let population = initial.total();
        let basic = self.basic_reproduction_number()?;
        let effective = self.effective_reproduction_number(initial)?;
        let peak_susceptible = population / basic;
        let predicted_peak_infectious =
            if initial.infectious == 0.0 || initial.susceptible <= peak_susceptible {
                initial.infectious
            } else {
                initial.infectious + initial.susceptible
                    - peak_susceptible
                    - peak_susceptible * (initial.susceptible / peak_susceptible).ln()
            };
        let final_susceptible = self.final_susceptible(initial)?;
        require_finite("predicted_peak_infectious", predicted_peak_infectious)?;
        require_non_negative("predicted_peak_infectious", predicted_peak_infectious)?;
        require_finite("final_susceptible", final_susceptible)?;
        Ok(EpidemicDiagnostic {
            basic_reproduction_number: basic,
            effective_reproduction_number: effective,
            regime: self.regime(initial)?,
            peak_susceptible,
            predicted_peak_infectious,
            final_susceptible,
            final_removed: population - final_susceptible,
        })
    }

    fn checked_stage(&self, step: usize, state: SirState) -> Result<SirState, ModelError> {
        for (component, value) in [
            ("susceptible", state.susceptible),
            ("infectious", state.infectious),
            ("removed", state.removed),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(ModelError::IntegrationDomainViolation {
                    step,
                    component,
                    value,
                });
            }
        }
        Ok(state)
    }

    fn add_scaled(state: SirState, tendency: SirState, scale: f64) -> SirState {
        SirState {
            susceptible: state.susceptible + scale * tendency.susceptible,
            infectious: state.infectious + scale * tendency.infectious,
            removed: state.removed + scale * tendency.removed,
        }
    }

    pub fn simulate(
        &self,
        initial: SirState,
        dt: f64,
        steps: usize,
    ) -> Result<Vec<SirSample>, ModelError> {
        self.validate_state(initial)?;
        require_positive("dt", dt)?;
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
        let population = initial.total();
        let initial_invariant = self.invariant(initial)?;
        let mut state = initial;
        let mut samples = Vec::with_capacity(capacity);
        for step in 0..=steps {
            samples.push(SirSample {
                time: step as f64 * dt,
                state,
                effective_reproduction_number: self.effective_reproduction_number(state)?,
                invariant: self.invariant(state)?,
                population_residual: state.total() - population,
            });
            if step == steps {
                break;
            }
            let k1 = self.derivative(state)?;
            let s2 = self.checked_stage(step, Self::add_scaled(state, k1, 0.5 * dt))?;
            let k2 = self.derivative(s2)?;
            let s3 = self.checked_stage(step, Self::add_scaled(state, k2, 0.5 * dt))?;
            let k3 = self.derivative(s3)?;
            let s4 = self.checked_stage(step, Self::add_scaled(state, k3, dt))?;
            let k4 = self.derivative(s4)?;
            state = self.checked_stage(
                step,
                SirState {
                    susceptible: state.susceptible
                        + dt * (k1.susceptible
                            + 2.0 * k2.susceptible
                            + 2.0 * k3.susceptible
                            + k4.susceptible)
                            / 6.0,
                    infectious: state.infectious
                        + dt * (k1.infectious
                            + 2.0 * k2.infectious
                            + 2.0 * k3.infectious
                            + k4.infectious)
                            / 6.0,
                    removed: state.removed
                        + dt * (k1.removed + 2.0 * k2.removed + 2.0 * k3.removed + k4.removed)
                            / 6.0,
                },
            )?;
            let residual = state.total() - population;
            if residual.abs() > 1.0e-9 * population.max(1.0) {
                return Err(ModelError::IntegrationDomainViolation {
                    step,
                    component: "population_residual",
                    value: residual,
                });
            }
            let invariant_drift = self.invariant(state)? - initial_invariant;
            require_finite("sir_invariant_drift", invariant_drift)?;
        }
        Ok(samples)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn initial() -> SirState {
        SirState {
            susceptible: 999.0,
            infectious: 1.0,
            removed: 0.0,
        }
    }

    #[test]
    fn threshold_uses_effective_susceptible_fraction() {
        let model = SirModel::try_new(0.3, 0.1).unwrap();
        assert_eq!(model.regime(initial()).unwrap(), EpidemicRegime::Growing);
        let late = SirState {
            susceptible: 200.0,
            infectious: 10.0,
            removed: 790.0,
        };
        assert_eq!(model.regime(late).unwrap(), EpidemicRegime::Declining);
    }

    #[test]
    fn derivative_conserves_population() {
        let model = SirModel::try_new(0.3, 0.1).unwrap();
        let tendency = model.derivative(initial()).unwrap();
        assert!(tendency.total().abs() < 1.0e-12);
    }

    #[test]
    fn final_size_root_matches_long_trajectory() {
        let model = SirModel::try_new(0.3, 0.1).unwrap();
        let predicted = model.final_susceptible(initial()).unwrap();
        let samples = model.simulate(initial(), 0.05, 6000).unwrap();
        let final_state = samples.last().unwrap().state;
        assert!(final_state.infectious < 1.0e-6);
        assert!((final_state.susceptible - predicted).abs() < 1.0e-5);
        assert!(
            samples
                .iter()
                .all(|sample| sample.population_residual.abs() < 1.0e-9)
        );
    }

    #[test]
    fn predicted_peak_is_reached_numerically() {
        let model = SirModel::try_new(0.3, 0.1).unwrap();
        let diagnostic = model.diagnostic(initial()).unwrap();
        let samples = model.simulate(initial(), 0.02, 5000).unwrap();
        let peak = samples
            .iter()
            .map(|sample| sample.state.infectious)
            .fold(0.0, f64::max);
        assert!((peak - diagnostic.predicted_peak_infectious).abs() < 0.01);
    }

    #[test]
    fn subcritical_outbreak_declines_but_still_has_finite_final_size() {
        let model = SirModel::try_new(0.05, 0.1).unwrap();
        let diagnostic = model.diagnostic(initial()).unwrap();
        assert_eq!(diagnostic.regime, EpidemicRegime::Declining);
        assert!(diagnostic.final_susceptible < initial().susceptible);
        assert!(diagnostic.final_susceptible > 990.0);
    }

    #[test]
    fn oversized_step_fails_before_negative_compartments_propagate() {
        let model = SirModel::try_new(10.0, 0.1).unwrap();
        let error = model.simulate(initial(), 10.0, 2).unwrap_err();
        assert!(matches!(
            error,
            ModelError::IntegrationDomainViolation { .. }
        ));
    }
}
