// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Generic Dynamical System Framework with ODE Integrators
//!
//! This module provides a trait-based dynamical systems framework for numerical
//! simulation and HDC encoding of continuous-time dynamics. It includes:
//!
//! - **`DynamicalSystem` trait**: Define any ODE system dx/dt = f(x, t)
//! - **`Integrator` trait** with four implementations:
//!   - Euler (1st order)
//!   - RK4 (classic 4th order Runge-Kutta)
//!   - RK45 (adaptive Runge-Kutta-Fehlberg)
//!   - Verlet (symplectic, for Hamiltonian systems)
//! - **Built-in systems**: Harmonic oscillator, N-body gravity, Lorenz attractor,
//!   double pendulum
//! - **HDC bridge**: Encode dynamical states and trajectories as `ContinuousHV`
//!
//! ## Design Principles
//!
//! - Pure numerical: no external crates beyond `std`
//! - Self-contained: all integration and system definitions in one module
//! - HDC encoding is a thin bridge layer on top of the numerical core
//!
//! ## References
//!
//! - Butcher (2008) - "Numerical Methods for Ordinary Differential Equations"
//! - Hairer, Lubich, Wanner (2006) - "Geometric Numerical Integration"
//! - Strogatz (2015) - "Nonlinear Dynamics and Chaos"

use crate::hdc::primitive_system::seed_from_name;
use crate::hdc::unified_hv::ContinuousHV;

use std::f64::consts::PI;

// =============================================================================
// CORE TRAITS
// =============================================================================

/// A continuous-time dynamical system defined by dx/dt = f(x, t).
///
/// Implementors define the dimensionality, the derivative function, and a
/// human-readable name. All implementations must be `Send + Sync` to allow
/// parallel simulation.
pub trait DynamicalSystem: Send + Sync {
    /// Number of state variables (dimension of the phase space).
    fn state_dim(&self) -> usize;

    /// Compute the time derivatives: dx/dt = f(x, t).
    ///
    /// # Arguments
    /// * `state` - Current state vector of length `state_dim()`
    /// * `t` - Current time
    ///
    /// # Returns
    /// Vector of derivatives, same length as `state`
    fn derivatives(&self, state: &[f64], t: f64) -> Vec<f64>;

    /// Human-readable name of the system.
    fn name(&self) -> &str;
}

/// An ODE integrator that advances a dynamical system by one time step.
pub trait Integrator {
    /// Validate integrator configuration before a checked simulation starts.
    fn validate(&self) -> Result<(), SimulationError> {
        Ok(())
    }

    /// Advance the system by one step of size `dt`.
    ///
    /// # Arguments
    /// * `system` - The dynamical system to integrate
    /// * `state` - Current state vector
    /// * `t` - Current time
    /// * `dt` - Requested step size
    ///
    /// # Returns
    /// New state vector at time t + dt. Adaptive integrators should implement
    /// step_with_dt so callers can observe the accepted step size.
    fn step(&self, system: &dyn DynamicalSystem, state: &[f64], t: f64, dt: f64) -> Vec<f64>;

    /// Advance a state and return both the new state and the accepted time step.
    ///
    /// Fixed-step integrators can use this default. Adaptive integrators must
    /// override it when the accepted step may be smaller than the requested step.
    fn step_with_dt(
        &self,
        system: &dyn DynamicalSystem,
        state: &[f64],
        t: f64,
        dt: f64,
    ) -> (Vec<f64>, f64) {
        (self.step(system, state, t, dt), dt)
    }

    /// Fallible step contract for checked simulations.
    ///
    /// Fixed-step integrators inherit this adapter. Adaptive integrators should
    /// override it when satisfying the requested tolerance is not guaranteed.
    fn try_step_with_dt(
        &self,
        system: &dyn DynamicalSystem,
        state: &[f64],
        t: f64,
        dt: f64,
    ) -> Result<(Vec<f64>, f64), SimulationError> {
        Ok(self.step_with_dt(system, state, t, dt))
    }

    /// Human-readable name of the integrator.
    fn name(&self) -> &str;
}

// =============================================================================
// SIMULATION RESULT
// =============================================================================

/// Result of a numerical simulation, storing the full trajectory.
#[derive(Debug, Clone, PartialEq)]
pub struct SimulationResult {
    /// Time values at each recorded step.
    pub times: Vec<f64>,
    /// State vectors at each recorded step.
    pub states: Vec<Vec<f64>>,
    /// Total elapsed simulation time (t_end - t_start).
    pub total_time: f64,
    /// Number of integration steps taken.
    pub steps: usize,
    /// Name of the system that was simulated.
    pub system_name: String,
}

impl SimulationResult {
    /// Extract the time series of a single state variable.
    pub fn variable(&self, index: usize) -> Vec<f64> {
        self.states.iter().map(|s| s[index]).collect()
    }

    /// Get the final state.
    pub fn final_state(&self) -> &[f64] {
        self.states.last().map(|s| s.as_slice()).unwrap_or(&[])
    }
}

// =============================================================================
// SIMULATE FUNCTION
// =============================================================================

/// Maximum number of integration steps retained by the checked simulator.
pub const MAX_SIMULATION_STEPS: usize = 1_000_000;

/// Maximum number of state scalars retained across a complete trajectory.
pub const MAX_TRAJECTORY_STATE_VALUES: usize = 10_000_000;

/// Errors returned when a simulation request or integrator output is invalid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SimulationError {
    NonFiniteTimeBounds,
    ReversedTimeRange,
    InvalidTimeStep,
    NonFiniteTimeSpan,
    NonFiniteStepCount,
    StepBudgetExceeded {
        maximum_steps: usize,
    },
    TrajectoryBudgetExceeded {
        maximum_state_values: usize,
    },
    StateDimensionMismatch {
        expected: usize,
        actual: usize,
    },
    NonFiniteInitialState {
        index: usize,
    },
    InvalidIntegratorStateDimension {
        step: usize,
        expected: usize,
        actual: usize,
    },
    NonFiniteIntegratorState {
        step: usize,
        index: usize,
    },
    InvalidIntegratorConfiguration,
    /// RK45 derivative evaluation returned a vector with the wrong dimension.
    DerivativeDimensionMismatch {
        stage: u8,
        expected: usize,
        actual: usize,
    },
    /// RK45 derivative evaluation returned a non-finite component.
    NonFiniteDerivative {
        stage: u8,
        index: usize,
    },
    /// The per-component absolute/relative tolerance profile is invalid.
    InvalidRk45Tolerances,
    /// RK45 generated a non-finite embedded error estimate.
    NonFiniteErrorEstimate {
        index: usize,
    },
    /// RK45 reached its minimum step while the embedded error still exceeded tolerance.
    AdaptiveToleranceNotMet,
    InvalidIntegratorStepSize {
        step: usize,
    },
    TimeDidNotAdvance {
        step: usize,
    },
}

/// Checked simulation entry point.
///
/// Rejects invalid/non-finite inputs and integrator outputs, and bounds trajectory
/// memory before allocation. A zero-duration run is valid and returns the initial
/// state. The step budget applies to requested fixed-step samples; adaptive
/// integrators may take internal substeps according to their own contract.
///
/// This validates the numerical integration boundary; it does not establish that
/// the selected equations or parameters accurately represent a physical system.
pub fn try_simulate(
    system: &dyn DynamicalSystem,
    integrator: &dyn Integrator,
    initial_state: &[f64],
    t_start: f64,
    t_end: f64,
    dt: f64,
) -> Result<SimulationResult, SimulationError> {
    integrator.validate()?;
    if !t_start.is_finite() || !t_end.is_finite() {
        return Err(SimulationError::NonFiniteTimeBounds);
    }
    if t_end < t_start {
        return Err(SimulationError::ReversedTimeRange);
    }
    if !dt.is_finite() || dt <= 0.0 {
        return Err(SimulationError::InvalidTimeStep);
    }

    let elapsed = t_end - t_start;
    if !elapsed.is_finite() {
        return Err(SimulationError::NonFiniteTimeSpan);
    }

    let expected_dim = system.state_dim();
    if initial_state.len() != expected_dim {
        return Err(SimulationError::StateDimensionMismatch {
            expected: expected_dim,
            actual: initial_state.len(),
        });
    }
    if let Some(index) = initial_state.iter().position(|value| !value.is_finite()) {
        return Err(SimulationError::NonFiniteInitialState { index });
    }

    let raw_steps = if elapsed == 0.0 {
        0.0
    } else {
        (elapsed / dt).ceil().max(1.0)
    };
    if !raw_steps.is_finite() || raw_steps < 0.0 {
        return Err(SimulationError::NonFiniteStepCount);
    }
    if raw_steps > MAX_SIMULATION_STEPS as f64 {
        return Err(SimulationError::StepBudgetExceeded {
            maximum_steps: MAX_SIMULATION_STEPS,
        });
    }
    let n_steps = raw_steps as usize;
    let sample_count = n_steps
        .checked_add(1)
        .ok_or(SimulationError::NonFiniteStepCount)?;
    let state_value_count = sample_count.checked_mul(expected_dim).ok_or(
        SimulationError::TrajectoryBudgetExceeded {
            maximum_state_values: MAX_TRAJECTORY_STATE_VALUES,
        },
    )?;
    if state_value_count > MAX_TRAJECTORY_STATE_VALUES {
        return Err(SimulationError::TrajectoryBudgetExceeded {
            maximum_state_values: MAX_TRAJECTORY_STATE_VALUES,
        });
    }

    let mut times = Vec::with_capacity(sample_count);
    let mut states = Vec::with_capacity(sample_count);
    let mut state = initial_state.to_vec();
    let mut t = t_start;

    times.push(t);
    states.push(state.clone());

    while t < t_end {
        let completed_steps = times.len() - 1;
        if completed_steps >= MAX_SIMULATION_STEPS {
            return Err(SimulationError::StepBudgetExceeded {
                maximum_steps: MAX_SIMULATION_STEPS,
            });
        }

        let next_sample_count =
            states
                .len()
                .checked_add(1)
                .ok_or(SimulationError::TrajectoryBudgetExceeded {
                    maximum_state_values: MAX_TRAJECTORY_STATE_VALUES,
                })?;
        let next_state_value_count = next_sample_count.checked_mul(expected_dim).ok_or(
            SimulationError::TrajectoryBudgetExceeded {
                maximum_state_values: MAX_TRAJECTORY_STATE_VALUES,
            },
        )?;
        if next_state_value_count > MAX_TRAJECTORY_STATE_VALUES {
            return Err(SimulationError::TrajectoryBudgetExceeded {
                maximum_state_values: MAX_TRAJECTORY_STATE_VALUES,
            });
        }

        let remaining = t_end - t;
        let requested_dt = dt.min(remaining);
        if !requested_dt.is_finite() || requested_dt <= 0.0 {
            return Err(SimulationError::TimeDidNotAdvance {
                step: completed_steps + 1,
            });
        }

        let (next_state, accepted_dt) =
            integrator.try_step_with_dt(system, &state, t, requested_dt)?;
        if !accepted_dt.is_finite() || accepted_dt <= 0.0 || accepted_dt > requested_dt {
            return Err(SimulationError::InvalidIntegratorStepSize {
                step: completed_steps + 1,
            });
        }
        if next_state.len() != expected_dim {
            return Err(SimulationError::InvalidIntegratorStateDimension {
                step: completed_steps + 1,
                expected: expected_dim,
                actual: next_state.len(),
            });
        }
        if let Some(index) = next_state.iter().position(|value| !value.is_finite()) {
            return Err(SimulationError::NonFiniteIntegratorState {
                step: completed_steps + 1,
                index,
            });
        }

        let next_t = if accepted_dt >= remaining {
            t_end
        } else {
            t + accepted_dt
        };
        if !next_t.is_finite() || next_t <= t {
            return Err(SimulationError::TimeDidNotAdvance {
                step: completed_steps + 1,
            });
        }

        state = next_state;
        t = next_t;
        times.push(t);
        states.push(state.clone());
    }

    let steps = times.len() - 1;
    Ok(SimulationResult {
        times,
        states,
        total_time: elapsed,
        steps,
        system_name: system.name().to_string(),
    })
}

/// Run a full simulation of a dynamical system.
///
/// Integrates from `t_start` to `t_end` using the given integrator and step
/// size. Records the state at every step.
///
/// # Arguments
/// * `system` - The dynamical system to simulate
/// * `integrator` - The ODE integrator to use
/// * `initial_state` - Initial conditions (length must equal `system.state_dim()`)
/// * `t_start` - Start time
/// * `t_end` - End time
/// * `dt` - Time step size
///
/// # Panics
/// Panics if the request is invalid or the trajectory exceeds the checked simulator's resource limits.
pub fn simulate(
    system: &dyn DynamicalSystem,
    integrator: &dyn Integrator,
    initial_state: &[f64],
    t_start: f64,
    t_end: f64,
    dt: f64,
) -> SimulationResult {
    try_simulate(system, integrator, initial_state, t_start, t_end, dt)
        .unwrap_or_else(|error| panic!("Simulation failed: {error:?}"))
}

// =============================================================================
// INTEGRATORS
// =============================================================================

/// Forward Euler integrator (1st order).
///
/// The simplest integrator: x_{n+1} = x_n + dt * f(x_n, t_n).
/// First-order accurate. Not symplectic. Use only for rough estimates or
/// as a baseline.
pub struct EulerIntegrator;

impl Integrator for EulerIntegrator {
    fn step(&self, system: &dyn DynamicalSystem, state: &[f64], t: f64, dt: f64) -> Vec<f64> {
        let deriv = system.derivatives(state, t);
        state
            .iter()
            .zip(deriv.iter())
            .map(|(&x, &dx)| x + dt * dx)
            .collect()
    }

    fn name(&self) -> &str {
        "Euler"
    }
}

/// Classic 4th-order Runge-Kutta integrator.
///
/// The workhorse of numerical ODE integration:
///   k1 = f(x, t)
///   k2 = f(x + dt/2 * k1, t + dt/2)
///   k3 = f(x + dt/2 * k2, t + dt/2)
///   k4 = f(x + dt * k3, t + dt)
///   x_{n+1} = x_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
///
/// 4th-order accurate. Not symplectic.
pub struct Rk4Integrator;

impl Integrator for Rk4Integrator {
    fn step(&self, system: &dyn DynamicalSystem, state: &[f64], t: f64, dt: f64) -> Vec<f64> {
        let dim = state.len();

        let k1 = system.derivatives(state, t);

        let x2: Vec<f64> = (0..dim).map(|i| state[i] + 0.5 * dt * k1[i]).collect();
        let k2 = system.derivatives(&x2, t + 0.5 * dt);

        let x3: Vec<f64> = (0..dim).map(|i| state[i] + 0.5 * dt * k2[i]).collect();
        let k3 = system.derivatives(&x3, t + 0.5 * dt);

        let x4: Vec<f64> = (0..dim).map(|i| state[i] + dt * k3[i]).collect();
        let k4 = system.derivatives(&x4, t + dt);

        (0..dim)
            .map(|i| state[i] + dt * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0)
            .collect()
    }

    fn name(&self) -> &str {
        "RK4"
    }
}

/// Adaptive Runge-Kutta-Fehlberg (RK45) integrator.
///
/// Uses embedded 4th and 5th order methods to estimate the local truncation
/// error. The step is accepted if the error is below `tol`; otherwise the
/// step size is reduced and retried. When the error is much smaller than
/// `tol`, the step size grows for efficiency.
///
/// The Butcher tableau used is Cash-Karp (a popular RK45 variant).
pub struct Rk45Integrator {
    /// Error tolerance for step-size control.
    pub tol: f64,
    /// Minimum allowed step size.
    pub dt_min: f64,
    /// Maximum allowed step size.
    pub dt_max: f64,
}

impl Default for Rk45Integrator {
    fn default() -> Self {
        Self {
            tol: 1e-8,
            dt_min: 1e-12,
            dt_max: 1.0,
        }
    }
}

/// Component-aware error tolerances for adaptive RK45 integration.
///
/// Each absolute tolerance must be finite and strictly positive. The relative
/// tolerance may be zero, but must be finite and non-negative. A component's
/// error scale is `absolute[i] + relative * max(abs(y_old[i]), abs(y_new[i]))`.
#[derive(Debug, Clone, PartialEq)]
pub struct Rk45Tolerances {
    pub relative: f64,
    pub absolute: Vec<f64>,
}

impl Rk45Tolerances {
    pub fn new(relative: f64, absolute: Vec<f64>) -> Self {
        Self { relative, absolute }
    }

    fn validate(&self, dimension: usize) -> Result<(), SimulationError> {
        if dimension == 0
            || !self.relative.is_finite()
            || self.relative < 0.0
            || self.absolute.len() != dimension
            || self.absolute.iter().any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(SimulationError::InvalidRk45Tolerances);
        }
        Ok(())
    }
}

/// One accepted RK45 step and the next controller proposal.
#[derive(Debug, Clone, PartialEq)]
pub struct Rk45StepReport {
    pub state: Vec<f64>,
    pub accepted_dt: f64,
    /// Maximum absolute difference between the embedded formulas.
    pub max_absolute_error: f64,
    /// Dimensionless error score; accepted steps have `error_norm <= 1`.
    pub error_norm: f64,
    pub proposed_next_dt: f64,
    pub attempted_steps: usize,
    pub rejected_steps: usize,
    pub derivative_evaluations: usize,
}

/// Stateful step-size controller. It persists the next-step proposal across
/// accepted steps without making the integrator itself mutable or shared stateful.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Rk45Controller {
    next_step_size: Option<f64>,
}

impl Rk45Controller {
    pub fn new(initial_step_size: f64) -> Result<Self, SimulationError> {
        if !initial_step_size.is_finite() || initial_step_size <= 0.0 {
            return Err(SimulationError::InvalidTimeStep);
        }
        Ok(Self {
            next_step_size: Some(initial_step_size),
        })
    }

    pub fn next_step_size(&self) -> Option<f64> {
        self.next_step_size
    }

    pub fn reset(&mut self, initial_step_size: f64) -> Result<(), SimulationError> {
        if !initial_step_size.is_finite() || initial_step_size <= 0.0 {
            return Err(SimulationError::InvalidTimeStep);
        }
        self.next_step_size = Some(initial_step_size);
        Ok(())
    }

    pub fn try_step(
        &mut self,
        integrator: &Rk45Integrator,
        system: &dyn DynamicalSystem,
        state: &[f64],
        t: f64,
        maximum_dt: f64,
        tolerances: &Rk45Tolerances,
    ) -> Result<Rk45StepReport, SimulationError> {
        if !maximum_dt.is_finite() || maximum_dt <= 0.0 || !t.is_finite() {
            return Err(SimulationError::InvalidTimeStep);
        }
        let requested_dt = self
            .next_step_size
            .unwrap_or(maximum_dt)
            .min(maximum_dt);
        let report = integrator.try_adaptive_step_with_tolerances(
            system,
            state,
            t,
            requested_dt,
            tolerances,
        )?;
        self.next_step_size = Some(report.proposed_next_dt);
        Ok(report)
    }
}

/// Aggregate diagnostics for a scale-aware adaptive simulation.
#[derive(Debug, Clone, PartialEq)]
pub struct Rk45Diagnostics {
    pub accepted_steps: usize,
    pub rejected_steps: usize,
    pub derivative_evaluations: usize,
    pub min_accepted_dt: Option<f64>,
    pub max_accepted_dt: Option<f64>,
    pub max_error_norm: f64,
}

/// Numerical trajectory plus adaptive-controller diagnostics.
#[derive(Debug, Clone, PartialEq)]
pub struct Rk45SimulationReport {
    pub simulation: SimulationResult,
    pub diagnostics: Rk45Diagnostics,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Rk45ErrorNorm {
    WeightedInfinity,
    WeightedRms,
}

impl Rk45Integrator {
    /// Create a new adaptive integrator with given tolerance.
    pub fn new(tol: f64) -> Self {
        Self {
            tol,
            ..Default::default()
        }
    }

    fn checked_derivatives(
        system: &dyn DynamicalSystem,
        state: &[f64],
        t: f64,
        expected_dim: usize,
        stage: u8,
    ) -> Result<Vec<f64>, SimulationError> {
        if let Some(index) = state.iter().position(|value| !value.is_finite()) {
            return Err(SimulationError::NonFiniteIntegratorState {
                step: usize::from(stage),
                index,
            });
        }
        let derivatives = system.derivatives(state, t);
        if derivatives.len() != expected_dim {
            return Err(SimulationError::DerivativeDimensionMismatch {
                stage,
                expected: expected_dim,
                actual: derivatives.len(),
            });
        }
        if let Some(index) = derivatives.iter().position(|value| !value.is_finite()) {
            return Err(SimulationError::NonFiniteDerivative { stage, index });
        }
        Ok(derivatives)
    }

    /// Compute one Cash–Karp step with the selected error norm.
    fn try_adaptive_step_impl(
        &self,
        system: &dyn DynamicalSystem,
        state: &[f64],
        t: f64,
        dt: f64,
        tolerances: &Rk45Tolerances,
        norm: Rk45ErrorNorm,
    ) -> Result<Rk45StepReport, SimulationError> {
        self.validate()?;
        if !t.is_finite() {
            return Err(SimulationError::NonFiniteTimeBounds);
        }
        if !dt.is_finite() || dt <= 0.0 {
            return Err(SimulationError::InvalidTimeStep);
        }
        let expected_dim = system.state_dim();
        if state.len() != expected_dim {
            return Err(SimulationError::StateDimensionMismatch {
                expected: expected_dim,
                actual: state.len(),
            });
        }
        if let Some(index) = state.iter().position(|value| !value.is_finite()) {
            return Err(SimulationError::NonFiniteInitialState { index });
        }
        tolerances.validate(expected_dim)?;
        let dim = state.len();
        let mut h = dt.min(self.dt_max);
        let mut attempted_steps = 0usize;

        loop {
            attempted_steps += 1;
            // Cash–Karp 4(5) embedded pair. The six checked derivative calls
            // are deliberately kept in this one implementation for both legacy
            // and scale-aware callers.
            let k1 = Self::checked_derivatives(system, state, t, dim, 1)?;

            let x2: Vec<f64> = (0..dim)
                .map(|i| state[i] + h * (1.0 / 5.0) * k1[i])
                .collect();
            let k2 = Self::checked_derivatives(system, &x2, t + h / 5.0, dim, 2)?;

            let x3: Vec<f64> = (0..dim)
                .map(|i| state[i] + h * (3.0 / 40.0 * k1[i] + 9.0 / 40.0 * k2[i]))
                .collect();
            let k3 =
                Self::checked_derivatives(system, &x3, t + 3.0 * h / 10.0, dim, 3)?;

            let x4: Vec<f64> = (0..dim)
                .map(|i| {
                    state[i] + h * (3.0 / 10.0 * k1[i] - 9.0 / 10.0 * k2[i]
                        + 6.0 / 5.0 * k3[i])
                })
                .collect();
            let k4 = Self::checked_derivatives(system, &x4, t + 3.0 * h / 5.0, dim, 4)?;

            let x5: Vec<f64> = (0..dim)
                .map(|i| {
                    state[i] + h * (-11.0 / 54.0 * k1[i] + 5.0 / 2.0 * k2[i]
                        - 70.0 / 27.0 * k3[i] + 35.0 / 27.0 * k4[i])
                })
                .collect();
            let k5 = Self::checked_derivatives(system, &x5, t + h, dim, 5)?;

            let x6: Vec<f64> = (0..dim)
                .map(|i| {
                    state[i] + h * (1631.0 / 55296.0 * k1[i]
                        + 175.0 / 512.0 * k2[i]
                        + 575.0 / 13824.0 * k3[i]
                        + 44275.0 / 110592.0 * k4[i]
                        + 253.0 / 4096.0 * k5[i])
                })
                .collect();
            let k6 = Self::checked_derivatives(system, &x6, t + 7.0 * h / 8.0, dim, 6)?;

            // Fifth-order solution used to advance the state.
            let y5: Vec<f64> = (0..dim)
                .map(|i| {
                    state[i] + h * (37.0 / 378.0 * k1[i]
                        + 250.0 / 621.0 * k3[i]
                        + 125.0 / 594.0 * k4[i]
                        + 512.0 / 1771.0 * k6[i])
                })
                .collect();

            // Fourth-order embedded solution used only for error estimation.
            let y4: Vec<f64> = (0..dim)
                .map(|i| {
                    state[i] + h * (2825.0 / 27648.0 * k1[i]
                        + 18575.0 / 48384.0 * k3[i]
                        + 13525.0 / 55296.0 * k4[i]
                        + 277.0 / 14336.0 * k5[i]
                        + 1.0 / 4.0 * k6[i])
                })
                .collect();

            let mut max_absolute_error = 0.0f64;
            let mut max_scaled_error = 0.0f64;
            let mut scaled_error_squares = 0.0f64;
            for index in 0..dim {
                if !y5[index].is_finite() || !y4[index].is_finite() {
                    return Err(SimulationError::NonFiniteErrorEstimate { index });
                }
                let component_error = (y5[index] - y4[index]).abs();
                if !component_error.is_finite() {
                    return Err(SimulationError::NonFiniteErrorEstimate { index });
                }
                max_absolute_error = max_absolute_error.max(component_error);

                let component_scale = tolerances.absolute[index]
                    + tolerances.relative * state[index].abs().max(y5[index].abs());
                if !component_scale.is_finite() || component_scale <= 0.0 {
                    return Err(SimulationError::InvalidRk45Tolerances);
                }
                let scaled = component_error / component_scale;
                if !scaled.is_finite() {
                    return Err(SimulationError::NonFiniteErrorEstimate { index });
                }
                max_scaled_error = max_scaled_error.max(scaled);
                scaled_error_squares += scaled * scaled;
                if !scaled_error_squares.is_finite() {
                    return Err(SimulationError::NonFiniteErrorEstimate { index });
                }
            }

            let error_norm = match norm {
                Rk45ErrorNorm::WeightedInfinity => max_scaled_error,
                Rk45ErrorNorm::WeightedRms => (scaled_error_squares / dim as f64).sqrt(),
            };
            if !error_norm.is_finite() {
                return Err(SimulationError::NonFiniteErrorEstimate { index: 0 });
            }

            if error_norm <= 1.0 {
                // Propose a next step using the embedded-pair order (5). The
                // caller-owned controller persists this proposal across steps.
                let factor = if error_norm <= 1.0e-14 {
                    5.0
                } else {
                    (0.9 * error_norm.powf(-0.2)).clamp(0.2, 5.0)
                };
                let proposed_next_dt = (h * factor).clamp(self.dt_min, self.dt_max);
                return Ok(Rk45StepReport {
                    state: y5,
                    accepted_dt: h,
                    max_absolute_error,
                    error_norm,
                    proposed_next_dt,
                    attempted_steps,
                    rejected_steps: attempted_steps - 1,
                    derivative_evaluations: attempted_steps * 6,
                });
            }

            if h <= self.dt_min {
                return Err(SimulationError::AdaptiveToleranceNotMet);
            }

            // Rejected steps always shrink. Bound the shrink factor to avoid
            // both near-zero progress and repeated unchanged attempts.
            let factor = (0.9 * error_norm.powf(-0.2)).clamp(0.1, 0.5);
            h = (h * factor).max(self.dt_min);
        }
    }

    /// Advance using the legacy scalar absolute tolerance and weighted-infinity norm.
    ///
    /// The tuple shape and absolute error return value remain compatible with
    /// existing callers. New code should prefer `try_adaptive_step_with_tolerances`.
    pub fn try_adaptive_step(
        &self,
        system: &dyn DynamicalSystem,
        state: &[f64],
        t: f64,
        dt: f64,
    ) -> Result<(Vec<f64>, f64, f64), SimulationError> {
        self.validate()?;
        let tolerances = Rk45Tolerances::new(0.0, vec![self.tol; state.len()]);
        let report = self.try_adaptive_step_impl(
            system,
            state,
            t,
            dt,
            &tolerances,
            Rk45ErrorNorm::WeightedInfinity,
        )?;
        Ok((report.state, report.accepted_dt, report.max_absolute_error))
    }

    /// Advance one step using weighted-RMS scale-aware error control.
    ///
    /// A step is accepted when the weighted RMS norm of its embedded error is
    /// at most one. Every state component uses its own absolute tolerance plus
    /// a shared relative tolerance, making the criterion meaningful when state
    /// components have different magnitudes or units.
    pub fn try_adaptive_step_with_tolerances(
        &self,
        system: &dyn DynamicalSystem,
        state: &[f64],
        t: f64,
        dt: f64,
        tolerances: &Rk45Tolerances,
    ) -> Result<Rk45StepReport, SimulationError> {
        self.try_adaptive_step_impl(
            system,
            state,
            t,
            dt,
            tolerances,
            Rk45ErrorNorm::WeightedRms,
        )
    }

    /// Integrate a complete trajectory with scale-aware error control and a
    /// persistent step-size controller.
    ///
    /// Unlike the legacy `try_simulate` entry point, this API explicitly accepts
    /// component-wise absolute tolerances and a relative tolerance, and returns
    /// accepted/rejected-step and derivative-evaluation diagnostics.
    pub fn try_simulate_with_tolerances(
        &self,
        system: &dyn DynamicalSystem,
        initial_state: &[f64],
        t_start: f64,
        t_end: f64,
        initial_dt: f64,
        tolerances: &Rk45Tolerances,
    ) -> Result<Rk45SimulationReport, SimulationError> {
        self.validate()?;
        if !t_start.is_finite() || !t_end.is_finite() {
            return Err(SimulationError::NonFiniteTimeBounds);
        }
        if t_end < t_start {
            return Err(SimulationError::ReversedTimeRange);
        }
        if !initial_dt.is_finite() || initial_dt <= 0.0 {
            return Err(SimulationError::InvalidTimeStep);
        }
        let elapsed = t_end - t_start;
        if !elapsed.is_finite() {
            return Err(SimulationError::NonFiniteTimeSpan);
        }

        let expected_dim = system.state_dim();
        if initial_state.len() != expected_dim {
            return Err(SimulationError::StateDimensionMismatch {
                expected: expected_dim,
                actual: initial_state.len(),
            });
        }
        if let Some(index) = initial_state.iter().position(|value| !value.is_finite()) {
            return Err(SimulationError::NonFiniteInitialState { index });
        }
        tolerances.validate(expected_dim)?;
        if expected_dim > MAX_TRAJECTORY_STATE_VALUES {
            return Err(SimulationError::TrajectoryBudgetExceeded {
                maximum_state_values: MAX_TRAJECTORY_STATE_VALUES,
            });
        }

        let mut times = Vec::new();
        let mut states = Vec::new();
        times.push(t_start);
        states.push(initial_state.to_vec());
        let mut time = t_start;
        let mut state = initial_state.to_vec();
        let mut controller = Rk45Controller::new(initial_dt)?;
        let mut diagnostics = Rk45Diagnostics {
            accepted_steps: 0,
            rejected_steps: 0,
            derivative_evaluations: 0,
            min_accepted_dt: None,
            max_accepted_dt: None,
            max_error_norm: 0.0,
        };

        while time < t_end {
            if diagnostics.accepted_steps >= MAX_SIMULATION_STEPS {
                return Err(SimulationError::StepBudgetExceeded {
                    maximum_steps: MAX_SIMULATION_STEPS,
                });
            }

            let next_sample_count =
                states
                    .len()
                    .checked_add(1)
                    .ok_or(SimulationError::TrajectoryBudgetExceeded {
                        maximum_state_values: MAX_TRAJECTORY_STATE_VALUES,
                    })?;
            let state_value_count = next_sample_count.checked_mul(expected_dim).ok_or(
                SimulationError::TrajectoryBudgetExceeded {
                    maximum_state_values: MAX_TRAJECTORY_STATE_VALUES,
                },
            )?;
            if state_value_count > MAX_TRAJECTORY_STATE_VALUES {
                return Err(SimulationError::TrajectoryBudgetExceeded {
                    maximum_state_values: MAX_TRAJECTORY_STATE_VALUES,
                });
            }

            let remaining = t_end - time;
            let maximum_dt = self.dt_max.min(remaining);
            let step = controller.try_step(
                self,
                system,
                &state,
                time,
                maximum_dt,
                tolerances,
            )?;
            if !step.accepted_dt.is_finite()
                || step.accepted_dt <= 0.0
                || step.accepted_dt > remaining
                || step.state.len() != expected_dim
                || step.state.iter().any(|value| !value.is_finite())
            {
                return Err(SimulationError::InvalidIntegratorStepSize {
                    step: diagnostics.accepted_steps + 1,
                });
            }

            let next_time = if step.accepted_dt >= remaining {
                t_end
            } else {
                time + step.accepted_dt
            };
            if !next_time.is_finite() || next_time <= time {
                return Err(SimulationError::TimeDidNotAdvance {
                    step: diagnostics.accepted_steps + 1,
                });
            }

            diagnostics.accepted_steps += 1;
            diagnostics.rejected_steps += step.rejected_steps;
            diagnostics.derivative_evaluations += step.derivative_evaluations;
            diagnostics.min_accepted_dt = Some(
                diagnostics
                    .min_accepted_dt
                    .map_or(step.accepted_dt, |value| value.min(step.accepted_dt)),
            );
            diagnostics.max_accepted_dt = Some(
                diagnostics
                    .max_accepted_dt
                    .map_or(step.accepted_dt, |value| value.max(step.accepted_dt)),
            );
            diagnostics.max_error_norm = diagnostics.max_error_norm.max(step.error_norm);

            time = next_time;
            state = step.state;
            times.push(time);
            states.push(state.clone());
        }

        let simulation = SimulationResult {
            times,
            steps: diagnostics.accepted_steps,
            states,
            total_time: elapsed,
            system_name: system.name().to_string(),
        };
        Ok(Rk45SimulationReport {
            simulation,
            diagnostics,
        })
    }

    /// Compatibility wrapper for callers that require a tuple-returning API.
    ///
    /// Panics explicitly when the configured tolerance cannot be met. Checked
    /// simulations should use `try_simulate` so this condition is returned as
    /// `SimulationError::AdaptiveToleranceNotMet`.
    pub fn adaptive_step(
        &self,
        system: &dyn DynamicalSystem,
        state: &[f64],
        t: f64,
        dt: f64,
    ) -> (Vec<f64>, f64, f64) {
        self.try_adaptive_step(system, state, t, dt)
            .unwrap_or_else(|error| panic!("RK45 could not satisfy its tolerance: {error:?}"))
    }
}

impl Integrator for Rk45Integrator {
    fn validate(&self) -> Result<(), SimulationError> {
        if !self.tol.is_finite()
            || self.tol <= 0.0
            || !self.dt_min.is_finite()
            || self.dt_min <= 0.0
            || !self.dt_max.is_finite()
            || self.dt_max < self.dt_min
        {
            return Err(SimulationError::InvalidIntegratorConfiguration);
        }
        Ok(())
    }

    fn step(&self, system: &dyn DynamicalSystem, state: &[f64], t: f64, dt: f64) -> Vec<f64> {
        if self.validate().is_err() || !t.is_finite() || !dt.is_finite() || dt < 0.0 {
            panic!("RK45 received an invalid integration request or configuration");
        }
        if dt == 0.0 {
            return state.to_vec();
        }

        let target_t = t + dt;
        if !target_t.is_finite() || target_t <= t {
            panic!("RK45 step cannot advance the requested time interval");
        }

        let mut current_state = state.to_vec();
        let mut current_t = t;
        for _ in 0..MAX_SIMULATION_STEPS {
            let remaining = target_t - current_t;
            if remaining <= 0.0 {
                return current_state;
            }

            let (next_state, accepted_dt, _error) =
                self.adaptive_step(system, &current_state, current_t, remaining);
            if !accepted_dt.is_finite() || accepted_dt <= 0.0 || accepted_dt > remaining {
                panic!("RK45 returned an invalid accepted timestep");
            }
            if next_state.len() != state.len() || next_state.iter().any(|value| !value.is_finite())
            {
                panic!("RK45 returned an invalid state vector");
            }

            let next_t = if accepted_dt >= remaining {
                target_t
            } else {
                current_t + accepted_dt
            };
            if !next_t.is_finite() || next_t <= current_t {
                panic!("RK45 timestep failed to advance time");
            }
            current_state = next_state;
            current_t = next_t;
        }
        panic!("RK45 exceeded the maximum accepted-step budget");
    }

    fn step_with_dt(
        &self,
        system: &dyn DynamicalSystem,
        state: &[f64],
        t: f64,
        dt: f64,
    ) -> (Vec<f64>, f64) {
        let (new_state, actual_dt, _error) = self.adaptive_step(system, state, t, dt);
        (new_state, actual_dt)
    }

    fn try_step_with_dt(
        &self,
        system: &dyn DynamicalSystem,
        state: &[f64],
        t: f64,
        dt: f64,
    ) -> Result<(Vec<f64>, f64), SimulationError> {
        let (new_state, accepted_dt, _error) =
            self.try_adaptive_step(system, state, t, dt)?;
        Ok((new_state, accepted_dt))
    }

    fn name(&self) -> &str {
        "RK45 (adaptive)"
    }
}

/// Stormer-Verlet (leapfrog) integrator for Hamiltonian systems.
///
/// Assumes the state vector is structured as [positions..., velocities...],
/// with equal numbers of position and velocity variables. The system's
/// derivatives must return [velocities..., accelerations...].
///
/// Symplectic: preserves the phase-space volume, leading to excellent
/// long-term energy conservation for conservative (Hamiltonian) systems.
pub struct VerletIntegrator;

impl Integrator for VerletIntegrator {
    fn step(&self, system: &dyn DynamicalSystem, state: &[f64], t: f64, dt: f64) -> Vec<f64> {
        let dim = state.len();
        assert!(
            dim.is_multiple_of(2),
            "Verlet requires even-dimensional state [q, p]"
        );
        let half = dim / 2;

        // state = [q_0 .. q_{n-1}, v_0 .. v_{n-1}]
        let q = &state[..half];
        let v = &state[half..];

        // Half-step velocity: v_{1/2} = v + (dt/2) * a(q, t)
        let deriv0 = system.derivatives(state, t);
        let a0 = &deriv0[half..]; // accelerations
        let v_half: Vec<f64> = v
            .iter()
            .zip(a0.iter())
            .map(|(&vi, &ai)| vi + 0.5 * dt * ai)
            .collect();

        // Full-step position: q_{1} = q + dt * v_{1/2}
        let q_new: Vec<f64> = q
            .iter()
            .zip(v_half.iter())
            .map(|(&qi, &vh)| qi + dt * vh)
            .collect();

        // Build intermediate state for acceleration evaluation
        let mut state_mid: Vec<f64> = Vec::with_capacity(dim);
        state_mid.extend_from_slice(&q_new);
        state_mid.extend_from_slice(&v_half);

        // Full-step velocity: v_{1} = v_{1/2} + (dt/2) * a(q_{1}, t+dt)
        let deriv1 = system.derivatives(&state_mid, t + dt);
        let a1 = &deriv1[half..];
        let v_new: Vec<f64> = v_half
            .iter()
            .zip(a1.iter())
            .map(|(&vh, &ai)| vh + 0.5 * dt * ai)
            .collect();

        let mut result = Vec::with_capacity(dim);
        result.extend_from_slice(&q_new);
        result.extend_from_slice(&v_new);
        result
    }

    fn name(&self) -> &str {
        "Verlet (symplectic)"
    }
}

// =============================================================================
// BUILT-IN DYNAMICAL SYSTEMS
// =============================================================================

/// Damped harmonic oscillator.
///
/// Equations of motion:
///   dx/dt = v
///   dv/dt = -omega^2 * x - damping * v
///
/// State: [x, v]
///
/// When damping = 0 the system is conservative (Hamiltonian) and energy
/// E = 0.5 * v^2 + 0.5 * omega^2 * x^2 is exactly conserved.
pub struct HarmonicOscillator {
    /// Angular frequency
    pub omega: f64,
    /// Damping coefficient (gamma >= 0)
    pub damping: f64,
}

impl HarmonicOscillator {
    /// Create a simple harmonic oscillator (no damping).
    pub fn simple(omega: f64) -> Self {
        Self {
            omega,
            damping: 0.0,
        }
    }

    /// Compute the total energy E = KE + PE = 0.5*v^2 + 0.5*omega^2*x^2.
    pub fn energy(&self, state: &[f64]) -> f64 {
        let x = state[0];
        let v = state[1];
        0.5 * v * v + 0.5 * self.omega * self.omega * x * x
    }

    /// Exact analytical period T = 2*pi/omega.
    pub fn period(&self) -> f64 {
        2.0 * PI / self.omega
    }
}

impl DynamicalSystem for HarmonicOscillator {
    fn state_dim(&self) -> usize {
        2
    }

    fn derivatives(&self, state: &[f64], _t: f64) -> Vec<f64> {
        let x = state[0];
        let v = state[1];
        vec![v, -self.omega * self.omega * x - self.damping * v]
    }

    fn name(&self) -> &str {
        "HarmonicOscillator"
    }
}

/// Gravitational N-body system.
///
/// State layout for `n` bodies in `dim` spatial dimensions:
///   [x1_1, x1_2, ..., x1_dim, x2_1, ..., xn_dim, v1_1, ..., vn_dim]
///
/// First `n * dim` entries are positions, next `n * dim` are velocities.
///
/// Gravitational acceleration on body i from body j:
///   a_i += G * m_j * (r_j - r_i) / |r_j - r_i|^3
///
/// Softening length `epsilon` prevents singularities at close approach.
pub struct NBodyGravity {
    /// Mass of each body.
    pub masses: Vec<f64>,
    /// Gravitational constant.
    pub g: f64,
    /// Spatial dimensionality (2 or 3).
    pub dim: usize,
    /// Softening length to prevent singularities.
    pub epsilon: f64,
}

impl NBodyGravity {
    /// Create a new N-body system.
    pub fn new(masses: Vec<f64>, g: f64, dim: usize) -> Self {
        assert!(dim == 2 || dim == 3, "Only 2D and 3D supported");
        Self {
            masses,
            g,
            dim,
            epsilon: 1e-6,
        }
    }

    /// Create a two-body system with given masses.
    pub fn two_body(m1: f64, m2: f64, g: f64, dim: usize) -> Self {
        Self::new(vec![m1, m2], g, dim)
    }

    /// Number of bodies.
    pub fn n_bodies(&self) -> usize {
        self.masses.len()
    }

    /// Compute total energy (kinetic + gravitational potential).
    pub fn energy(&self, state: &[f64]) -> f64 {
        let n = self.n_bodies();
        let d = self.dim;
        let half = n * d;

        // Kinetic energy
        let mut ke = 0.0;
        for i in 0..n {
            let mut v2 = 0.0;
            for k in 0..d {
                let v = state[half + i * d + k];
                v2 += v * v;
            }
            ke += 0.5 * self.masses[i] * v2;
        }

        // Gravitational potential energy
        let mut pe = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                let mut r2 = 0.0;
                for k in 0..d {
                    let dx = state[j * d + k] - state[i * d + k];
                    r2 += dx * dx;
                }
                let r = (r2 + self.epsilon * self.epsilon).sqrt();
                pe -= self.g * self.masses[i] * self.masses[j] / r;
            }
        }

        ke + pe
    }

    /// Compute total momentum.
    pub fn momentum(&self, state: &[f64]) -> Vec<f64> {
        let n = self.n_bodies();
        let d = self.dim;
        let half = n * d;
        let mut p = vec![0.0; d];
        for i in 0..n {
            for k in 0..d {
                p[k] += self.masses[i] * state[half + i * d + k];
            }
        }
        p
    }
}

impl DynamicalSystem for NBodyGravity {
    fn state_dim(&self) -> usize {
        2 * self.n_bodies() * self.dim
    }

    fn derivatives(&self, state: &[f64], _t: f64) -> Vec<f64> {
        let n = self.n_bodies();
        let d = self.dim;
        let half = n * d;
        let total = 2 * half;
        let mut deriv = vec![0.0; total];

        // Positions' derivatives are velocities
        deriv[..half].copy_from_slice(&state[half..(half + half)]);

        // Velocities' derivatives are gravitational accelerations
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let mut r2 = 0.0;
                let mut dr = vec![0.0; d];
                for k in 0..d {
                    dr[k] = state[j * d + k] - state[i * d + k];
                    r2 += dr[k] * dr[k];
                }
                let r = (r2 + self.epsilon * self.epsilon).sqrt();
                let inv_r3 = 1.0 / (r * r * r);
                for k in 0..d {
                    deriv[half + i * d + k] += self.g * self.masses[j] * dr[k] * inv_r3;
                }
            }
        }

        deriv
    }

    fn name(&self) -> &str {
        "NBodyGravity"
    }
}

/// Lorenz system (atmospheric convection model).
///
/// Equations:
///   dx/dt = sigma * (y - x)
///   dy/dt = x * (rho - z) - y
///   dz/dt = x * y - beta * z
///
/// State: [x, y, z]
///
/// Standard chaotic parameters: sigma=10, rho=28, beta=8/3.
pub struct LorenzSystem {
    /// Prandtl number
    pub sigma: f64,
    /// Rayleigh number
    pub rho: f64,
    /// Geometric factor
    pub beta: f64,
}

impl LorenzSystem {
    /// Create a Lorenz system with the standard chaotic parameters.
    pub fn standard() -> Self {
        Self {
            sigma: 10.0,
            rho: 28.0,
            beta: 8.0 / 3.0,
        }
    }
}

impl DynamicalSystem for LorenzSystem {
    fn state_dim(&self) -> usize {
        3
    }

    fn derivatives(&self, state: &[f64], _t: f64) -> Vec<f64> {
        let x = state[0];
        let y = state[1];
        let z = state[2];
        vec![
            self.sigma * (y - x),
            x * (self.rho - z) - y,
            x * y - self.beta * z,
        ]
    }

    fn name(&self) -> &str {
        "LorenzSystem"
    }
}

/// Double pendulum system (chaotic mechanical system).
///
/// A double pendulum consists of two masses m1, m2 on massless rods of
/// lengths l1, l2 under gravity g. The dynamics are described by the
/// Lagrangian equations of motion.
///
/// State: [theta1, theta2, omega1, omega2]
///   where theta = angle from vertical, omega = angular velocity
///
/// The equations of motion are derived from the Lagrangian and are
/// highly nonlinear, producing chaotic behavior for most initial conditions.
pub struct DoublePendulum {
    /// Mass of first bob
    pub m1: f64,
    /// Mass of second bob
    pub m2: f64,
    /// Length of first rod
    pub l1: f64,
    /// Length of second rod
    pub l2: f64,
    /// Gravitational acceleration
    pub g: f64,
}

impl DoublePendulum {
    /// Create a double pendulum with equal masses and lengths.
    pub fn symmetric(m: f64, l: f64, g: f64) -> Self {
        Self {
            m1: m,
            m2: m,
            l1: l,
            l2: l,
            g,
        }
    }

    /// Compute total energy (kinetic + potential).
    ///
    /// E = KE + PE where PE is measured from the pivot point.
    pub fn energy(&self, state: &[f64]) -> f64 {
        let th1 = state[0];
        let th2 = state[1];
        let w1 = state[2];
        let w2 = state[3];

        let cos_delta = (th1 - th2).cos();

        // Kinetic energy
        let ke = 0.5 * (self.m1 + self.m2) * self.l1 * self.l1 * w1 * w1
            + 0.5 * self.m2 * self.l2 * self.l2 * w2 * w2
            + self.m2 * self.l1 * self.l2 * w1 * w2 * cos_delta;

        // Potential energy (measured from pivot, so both negative)
        let pe = -(self.m1 + self.m2) * self.g * self.l1 * th1.cos()
            - self.m2 * self.g * self.l2 * th2.cos();

        ke + pe
    }
}

impl DynamicalSystem for DoublePendulum {
    fn state_dim(&self) -> usize {
        4
    }

    fn derivatives(&self, state: &[f64], _t: f64) -> Vec<f64> {
        let th1 = state[0];
        let th2 = state[1];
        let w1 = state[2];
        let w2 = state[3];

        let m1 = self.m1;
        let m2 = self.m2;
        let l1 = self.l1;
        let l2 = self.l2;
        let g = self.g;

        let delta = th1 - th2;
        let cos_d = delta.cos();
        let sin_d = delta.sin();

        // Denominator for the coupled equations
        let denom = (2.0 * m1 + m2 - m2 * (2.0 * delta).cos()) * l1;

        // Angular accelerations from the Lagrangian equations of motion
        let alpha1 = (-g * (2.0 * m1 + m2) * th1.sin()
            - m2 * g * (th1 - 2.0 * th2).sin()
            - 2.0 * sin_d * m2 * (w2 * w2 * l2 + w1 * w1 * l1 * cos_d))
            / denom;

        let denom2 = denom * l2 / l1;
        let alpha2 = (2.0
            * sin_d
            * (w1 * w1 * l1 * (m1 + m2) + g * (m1 + m2) * th1.cos() + w2 * w2 * l2 * m2 * cos_d))
            / denom2;

        vec![w1, w2, alpha1, alpha2]
    }

    fn name(&self) -> &str {
        "DoublePendulum"
    }
}

// =============================================================================
// ELECTROMAGNETIC: CHARGED PARTICLE IN EM FIELD (LORENTZ FORCE)
// =============================================================================

/// Charged particle moving in static electric and magnetic fields.
///
/// Equations of motion (Lorentz force law):
///   dx/dt = v
///   dv/dt = (q/m)(E + v × B)
///
/// State vector: [x, y, z, vx, vy, vz]
///
/// Notable cases:
/// - Pure B field → cyclotron motion (circular orbit, frequency ω = qB/m)
/// - E ⊥ B → E×B drift
/// - E ∥ B → helical acceleration
pub struct ChargedParticle {
    /// Charge-to-mass ratio (q/m) in SI units
    pub charge_mass_ratio: f64,
    /// Electric field vector [Ex, Ey, Ez] (V/m)
    pub e_field: [f64; 3],
    /// Magnetic field vector [Bx, By, Bz] (Tesla)
    pub b_field: [f64; 3],
}

impl ChargedParticle {
    pub fn new(charge_mass_ratio: f64, e_field: [f64; 3], b_field: [f64; 3]) -> Self {
        Self {
            charge_mass_ratio,
            e_field,
            b_field,
        }
    }

    /// Cyclotron motion: particle in uniform B field along z-axis.
    pub fn cyclotron(charge_mass_ratio: f64, b_z: f64) -> Self {
        Self::new(charge_mass_ratio, [0.0, 0.0, 0.0], [0.0, 0.0, b_z])
    }

    /// E×B drift: crossed electric and magnetic fields.
    pub fn e_cross_b_drift(e_y: f64, b_z: f64) -> Self {
        Self::new(1.0, [0.0, e_y, 0.0], [0.0, 0.0, b_z])
    }

    /// Expected cyclotron frequency: ω_c = |q/m| * |B|
    pub fn cyclotron_frequency(&self) -> f64 {
        let b_mag =
            (self.b_field[0].powi(2) + self.b_field[1].powi(2) + self.b_field[2].powi(2)).sqrt();
        self.charge_mass_ratio.abs() * b_mag
    }

    /// Expected cyclotron period: T = 2π / ω_c
    pub fn cyclotron_period(&self) -> f64 {
        let omega = self.cyclotron_frequency();
        if omega > 0.0 {
            2.0 * PI / omega
        } else {
            f64::INFINITY
        }
    }

    /// Kinetic energy: 0.5 * |v|²  (per unit mass, since we use q/m)
    pub fn kinetic_energy(&self, state: &[f64]) -> f64 {
        let vx = state[3];
        let vy = state[4];
        let vz = state[5];
        0.5 * (vx * vx + vy * vy + vz * vz)
    }
}

impl DynamicalSystem for ChargedParticle {
    fn state_dim(&self) -> usize {
        6
    }

    fn derivatives(&self, state: &[f64], _t: f64) -> Vec<f64> {
        let (vx, vy, vz) = (state[3], state[4], state[5]);
        let (ex, ey, ez) = (self.e_field[0], self.e_field[1], self.e_field[2]);
        let (bx, by, bz) = (self.b_field[0], self.b_field[1], self.b_field[2]);
        let qm = self.charge_mass_ratio;

        // Lorentz force: F/m = (q/m)(E + v × B)
        // v × B = (vy*bz - vz*by, vz*bx - vx*bz, vx*by - vy*bx)
        let ax = qm * (ex + vy * bz - vz * by);
        let ay = qm * (ey + vz * bx - vx * bz);
        let az = qm * (ez + vx * by - vy * bx);

        vec![vx, vy, vz, ax, ay, az]
    }

    fn name(&self) -> &str {
        "ChargedParticle"
    }
}

// =============================================================================
// THERMODYNAMIC: 1D HEAT EQUATION (DISCRETIZED)
// =============================================================================

/// 1D heat diffusion equation discretized to a system of ODEs.
///
/// The continuous PDE: ∂T/∂t = α ∂²T/∂x²
///
/// is discretized on N grid points with spacing Δx:
///   dT_i/dt = α (T_{i-1} - 2T_i + T_{i+1}) / Δx²
///
/// Boundary conditions: fixed temperature at both ends (Dirichlet).
///
/// State vector: [T_1, T_2, ..., T_N] (interior grid point temperatures)
pub struct HeatEquation1D {
    /// Thermal diffusivity (m²/s)
    pub alpha: f64,
    /// Grid spacing
    pub dx: f64,
    /// Number of interior grid points (state dimension)
    pub n_points: usize,
    /// Left boundary temperature
    pub t_left: f64,
    /// Right boundary temperature
    pub t_right: f64,
}

impl HeatEquation1D {
    pub fn new(alpha: f64, length: f64, n_points: usize, t_left: f64, t_right: f64) -> Self {
        let dx = length / (n_points + 1) as f64;
        Self {
            alpha,
            dx,
            n_points,
            t_left,
            t_right,
        }
    }

    /// Uniform rod with hot left end and cold right end.
    pub fn hot_cold_rod(alpha: f64, n_points: usize) -> Self {
        Self::new(alpha, 1.0, n_points, 100.0, 0.0)
    }

    /// Total thermal energy (proportional to sum of temperatures).
    pub fn total_energy(&self, state: &[f64]) -> f64 {
        self.t_left + state.iter().sum::<f64>() + self.t_right
    }

    /// Steady-state solution: linear interpolation between boundaries.
    pub fn steady_state(&self) -> Vec<f64> {
        (0..self.n_points)
            .map(|i| {
                let x = (i + 1) as f64 / (self.n_points + 1) as f64;
                self.t_left + (self.t_right - self.t_left) * x
            })
            .collect()
    }

    /// Maximum temperature deviation from steady state.
    pub fn max_deviation_from_steady(&self, state: &[f64]) -> f64 {
        let ss = self.steady_state();
        state
            .iter()
            .zip(ss.iter())
            .map(|(t, s)| (t - s).abs())
            .fold(0.0f64, f64::max)
    }
}

impl DynamicalSystem for HeatEquation1D {
    fn state_dim(&self) -> usize {
        self.n_points
    }

    fn derivatives(&self, state: &[f64], _t: f64) -> Vec<f64> {
        let c = self.alpha / (self.dx * self.dx);
        let n = self.n_points;
        let mut dt = vec![0.0; n];

        for i in 0..n {
            let t_prev = if i == 0 { self.t_left } else { state[i - 1] };
            let t_next = if i == n - 1 {
                self.t_right
            } else {
                state[i + 1]
            };
            dt[i] = c * (t_prev - 2.0 * state[i] + t_next);
        }

        dt
    }

    fn name(&self) -> &str {
        "HeatEquation1D"
    }
}

// =============================================================================
// HDC BRIDGE
// =============================================================================

/// Encode a dynamical system state as a ContinuousHV.
///
/// The state is expanded to the standard HDC dimension by repeating and
/// scaling the components. Each component modulates a deterministic random
/// basis vector, and the results are bundled.
///
/// # Arguments
/// * `state` - The dynamical state vector (arbitrary length)
///
/// # Returns
/// A ContinuousHV encoding of the state
pub fn state_to_hv(state: &[f64]) -> ContinuousHV {
    let dim = ContinuousHV::DEFAULT_DIM;
    let n = state.len();

    if n == 0 {
        return ContinuousHV::zero(dim);
    }

    // Generate a deterministic basis vector for each state variable
    let mut values = vec![0.0f32; dim];

    for (idx, &val) in state.iter().enumerate() {
        let seed = seed_from_name(&format!("dynstate::{idx}"));
        let basis = ContinuousHV::random(dim, seed);
        let weight = val as f32;
        for (v, b) in values.iter_mut().zip(basis.values.iter()) {
            *v += weight * b;
        }
    }

    // Normalize
    let norm: f32 = values.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-15 {
        for v in values.iter_mut() {
            *v /= norm;
        }
    }

    ContinuousHV::from_vec(values)
}

/// Encode an entire trajectory as a sequence of ContinuousHVs.
///
/// Each state in the simulation result is individually encoded.
pub fn trajectory_to_hvs(result: &SimulationResult) -> Vec<ContinuousHV> {
    result.states.iter().map(|s| state_to_hv(s)).collect()
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    struct FixedOutputIntegrator {
        output: Vec<f64>,
    }

    impl Integrator for FixedOutputIntegrator {
        fn step(
            &self,
            _system: &dyn DynamicalSystem,
            _state: &[f64],
            _t: f64,
            _dt: f64,
        ) -> Vec<f64> {
            self.output.clone()
        }

        fn name(&self) -> &str {
            "FixedOutput"
        }
    }

    #[test]
    fn checked_simulation_rejects_invalid_time_inputs() {
        let osc = HarmonicOscillator::simple(1.0);
        let integrator = Rk4Integrator;
        let initial = [1.0, 0.0];

        for invalid_dt in [0.0, -0.1, f64::NAN, f64::INFINITY] {
            assert_eq!(
                try_simulate(&osc, &integrator, &initial, 0.0, 1.0, invalid_dt),
                Err(SimulationError::InvalidTimeStep)
            );
        }
        assert_eq!(
            try_simulate(&osc, &integrator, &initial, f64::NAN, 1.0, 0.1),
            Err(SimulationError::NonFiniteTimeBounds)
        );
        assert_eq!(
            try_simulate(&osc, &integrator, &initial, 1.0, 0.0, 0.1),
            Err(SimulationError::ReversedTimeRange)
        );
        assert_eq!(
            try_simulate(&osc, &integrator, &initial, -f64::MAX, f64::MAX, 0.1),
            Err(SimulationError::NonFiniteTimeSpan)
        );
        assert_eq!(
            try_simulate(&osc, &integrator, &initial, 0.0, 1.0, 1.0e-12),
            Err(SimulationError::StepBudgetExceeded {
                maximum_steps: MAX_SIMULATION_STEPS
            })
        );

        let invalid_rk45 = Rk45Integrator {
            tol: f64::NAN,
            ..Rk45Integrator::default()
        };
        assert_eq!(
            try_simulate(&osc, &invalid_rk45, &initial, 0.0, 1.0, 0.1),
            Err(SimulationError::InvalidIntegratorConfiguration)
        );
    }

    #[test]
    fn checked_simulation_rejects_invalid_states_and_integrator_outputs() {
        let osc = HarmonicOscillator::simple(1.0);
        let integrator = Rk4Integrator;

        assert_eq!(
            try_simulate(&osc, &integrator, &[f64::NAN, 0.0], 0.0, 1.0, 0.1),
            Err(SimulationError::NonFiniteInitialState { index: 0 })
        );
        assert_eq!(
            try_simulate(&osc, &integrator, &[1.0], 0.0, 1.0, 0.1),
            Err(SimulationError::StateDimensionMismatch {
                expected: 2,
                actual: 1
            })
        );

        let wrong_dimension = FixedOutputIntegrator { output: vec![1.0] };
        assert_eq!(
            try_simulate(&osc, &wrong_dimension, &[1.0, 0.0], 0.0, 0.1, 0.1),
            Err(SimulationError::InvalidIntegratorStateDimension {
                step: 1,
                expected: 2,
                actual: 1
            })
        );
        let non_finite = FixedOutputIntegrator {
            output: vec![f64::INFINITY, 0.0],
        };
        assert_eq!(
            try_simulate(&osc, &non_finite, &[1.0, 0.0], 0.0, 0.1, 0.1),
            Err(SimulationError::NonFiniteIntegratorState { step: 1, index: 0 })
        );
    }

    #[test]
    fn rk45_uses_accepted_step_size_to_advance_simulation_time() {
        let osc = HarmonicOscillator::simple(1.0);
        let initial = [1.0, 0.0];
        let t_end = 10.0;
        let integrator = Rk45Integrator::new(1.0e-12);
        let result = try_simulate(&osc, &integrator, &initial, 0.0, t_end, 0.5)
            .expect("valid adaptive simulation should complete");

        // The requested interval is intentionally coarse so the tolerance forces
        // smaller accepted substeps; the clock must advance by those accepted steps.
        assert!(result.steps > 20);
        assert_eq!(result.times.last().copied(), Some(t_end));
        let expected_x = t_end.cos();
        let expected_v = -t_end.sin();
        assert!(
            (result.final_state()[0] - expected_x).abs() < 1.0e-8,
            "adaptive RK45 x error was {:.3e}",
            (result.final_state()[0] - expected_x).abs()
        );
        assert!(
            (result.final_state()[1] - expected_v).abs() < 1.0e-8,
            "adaptive RK45 v error was {:.3e}",
            (result.final_state()[1] - expected_v).abs()
        );

        // The legacy one-step API must also advance across the full requested
        // interval, not just one accepted adaptive substep.
        let short_end = 0.1;
        let one_step = integrator.step(&osc, &initial, 0.0, short_end);
        assert!(
            (one_step[0] - short_end.cos()).abs() < 1.0e-8,
            "direct RK45 x error was {:.3e}",
            (one_step[0] - short_end.cos()).abs()
        );
        assert!(
            (one_step[1] + short_end.sin()).abs() < 1.0e-8,
            "direct RK45 v error was {:.3e}",
            (one_step[1] + short_end.sin()).abs()
        );
    }

    #[test]
    fn checked_simulation_accepts_zero_duration() {
        let osc = HarmonicOscillator::simple(1.0);
        let result = try_simulate(&osc, &Rk4Integrator, &[1.0, 0.0], 2.0, 2.0, 0.1)
            .expect("zero-duration request should be valid");
        assert_eq!(result.steps, 0);
        assert_eq!(result.times, vec![2.0]);
        assert_eq!(result.states, vec![vec![1.0, 0.0]]);
    }

    #[test]
    fn rk4_exhibits_fourth_order_convergence_on_harmonic_oscillator() {
        let osc = HarmonicOscillator::simple(1.0);
        let initial = [1.0, 0.0];
        let exact = [1.0f64.cos(), -1.0f64.sin()];

        let error_at = |dt: f64| {
            let result = try_simulate(&osc, &Rk4Integrator, &initial, 0.0, 1.0, dt)
                .expect("valid convergence-study request");
            result
                .final_state()
                .iter()
                .zip(exact.iter())
                .map(|(actual, target)| (actual - target).powi(2))
                .sum::<f64>()
                .sqrt()
        };

        let coarse = error_at(0.2);
        let medium = error_at(0.1);
        let fine = error_at(0.05);
        let first_ratio = coarse / medium;
        let second_ratio = medium / fine;

        assert!(
            (12.0..20.0).contains(&first_ratio),
            "RK4 coarse/medium error ratio {first_ratio:.3} does not support fourth-order convergence"
        );
        assert!(
            (12.0..20.0).contains(&second_ratio),
            "RK4 medium/fine error ratio {second_ratio:.3} does not support fourth-order convergence"
        );
    }

    // -------------------------------------------------------------------------
    // Harmonic oscillator: energy conservation
    // -------------------------------------------------------------------------

    #[test]
    fn test_harmonic_energy_conservation_rk4() {
        let osc = HarmonicOscillator::simple(2.0);
        let integrator = Rk4Integrator;

        // x(0) = 1, v(0) = 0 => E = 0.5 * omega^2
        let initial = [1.0, 0.0];
        let result = simulate(&osc, &integrator, &initial, 0.0, 50.0, 0.001);

        let e0 = osc.energy(&initial);
        let mut max_drift = 0.0f64;

        for state in &result.states {
            let e = osc.energy(state);
            let drift = (e - e0).abs() / e0;
            if drift > max_drift {
                max_drift = drift;
            }
        }

        assert!(
            max_drift < 1e-10,
            "RK4 energy drift {:.2e} should be < 1e-10 for harmonic oscillator",
            max_drift
        );
    }

    #[test]
    fn test_harmonic_energy_conservation_verlet() {
        let osc = HarmonicOscillator::simple(2.0);
        let integrator = VerletIntegrator;

        let initial = [1.0, 0.0];
        let result = simulate(&osc, &integrator, &initial, 0.0, 50.0, 0.001);

        let e0 = osc.energy(&initial);
        let mut max_drift = 0.0f64;

        for state in &result.states {
            let e = osc.energy(state);
            let drift = (e - e0).abs() / e0;
            if drift > max_drift {
                max_drift = drift;
            }
        }

        assert!(
            max_drift < 1e-5,
            "Verlet energy drift {:.2e} should be < 1e-5 for harmonic oscillator",
            max_drift
        );
    }

    #[test]
    fn test_harmonic_period() {
        let omega = 3.0;
        let osc = HarmonicOscillator::simple(omega);
        let integrator = Rk4Integrator;

        // x(0) = 1, v(0) = 0
        let initial = [1.0, 0.0];
        let expected_period = osc.period();
        let result = simulate(
            &osc,
            &integrator,
            &initial,
            0.0,
            expected_period * 3.0,
            0.0001,
        );

        // After one full period, x should return close to 1.0
        // Find the time closest to T
        let mut best_idx = 0;
        let mut best_diff = f64::MAX;
        for (i, &t) in result.times.iter().enumerate() {
            let diff = (t - expected_period).abs();
            if diff < best_diff {
                best_diff = diff;
                best_idx = i;
            }
        }

        let x_at_period = result.states[best_idx][0];
        assert!(
            (x_at_period - 1.0).abs() < 0.001,
            "x at T={:.4} should be ~1.0, got {:.6}",
            expected_period,
            x_at_period
        );
    }

    // -------------------------------------------------------------------------
    // N-body: two-body orbit (Kepler period)
    // -------------------------------------------------------------------------

    #[test]
    fn test_two_body_kepler_period() {
        // Two equal masses in circular orbit
        // For circular orbit: G*M/r = v^2, period T = 2*pi*r / v
        // With m1 = m2 = 1, G = 1, separation = 2 (each at r=1 from COM):
        //   Force on each = G*m1*m2/(2r)^2 = 1/4
        //   Centripetal: m*v^2/r = 1/4 => v^2 = r/4 = 0.25 => v = 0.5
        //   Period: T = 2*pi*r/v = 2*pi*1/0.5 = 4*pi
        let g = 1.0;
        let nbody = NBodyGravity::new(vec![1.0, 1.0], g, 2);

        // Body 1 at (-1, 0) with velocity (0, -0.5)
        // Body 2 at (1, 0) with velocity (0, 0.5)
        let initial = vec![
            -1.0, 0.0, // pos body 1
            1.0, 0.0, // pos body 2
            0.0, -0.5, // vel body 1
            0.0, 0.5, // vel body 2
        ];

        let expected_period = 4.0 * PI; // ~12.566
        let integrator = Rk4Integrator;
        let result = simulate(
            &nbody,
            &integrator,
            &initial,
            0.0,
            expected_period * 1.1,
            0.001,
        );

        // After one period, body 1 should return near its initial position
        let mut best_idx = 0;
        let mut best_diff = f64::MAX;
        for (i, &t) in result.times.iter().enumerate() {
            if i == 0 {
                continue;
            }
            let diff = (t - expected_period).abs();
            if diff < best_diff {
                best_diff = diff;
                best_idx = i;
            }
        }

        let x1_at_period = result.states[best_idx][0]; // x of body 1
        let y1_at_period = result.states[best_idx][1]; // y of body 1

        assert!(
            (x1_at_period - (-1.0)).abs() < 0.05,
            "Body 1 x at period should be ~-1.0, got {:.4}",
            x1_at_period
        );
        assert!(
            y1_at_period.abs() < 0.05,
            "Body 1 y at period should be ~0.0, got {:.4}",
            y1_at_period
        );
    }

    #[test]
    fn test_two_body_energy_conservation() {
        let nbody = NBodyGravity::new(vec![1.0, 2.0], 1.0, 2);

        let initial = vec![-1.0, 0.0, 0.5, 0.0, 0.0, -0.3, 0.0, 0.15];

        let integrator = Rk4Integrator;
        let result = simulate(&nbody, &integrator, &initial, 0.0, 20.0, 0.001);

        let e0 = nbody.energy(&initial);
        let mut max_drift = 0.0f64;

        for state in &result.states {
            let e = nbody.energy(state);
            let drift = (e - e0).abs() / e0.abs().max(1e-15);
            if drift > max_drift {
                max_drift = drift;
            }
        }

        assert!(
            max_drift < 1e-4,
            "Two-body energy drift {:.2e} should be < 1e-4",
            max_drift
        );
    }

    #[test]
    fn test_two_body_momentum_conservation() {
        let nbody = NBodyGravity::new(vec![1.0, 2.0], 1.0, 2);

        // Set up with zero total momentum
        let initial = vec![-1.0, 0.0, 0.5, 0.0, 0.0, -0.4, 0.0, 0.2];

        let integrator = Rk4Integrator;
        let result = simulate(&nbody, &integrator, &initial, 0.0, 10.0, 0.001);

        let p0 = nbody.momentum(&initial);

        for state in &result.states {
            let p = nbody.momentum(state);
            for k in 0..2 {
                assert!(
                    (p[k] - p0[k]).abs() < 1e-10,
                    "Momentum component {} drifted: {:.2e}",
                    k,
                    (p[k] - p0[k]).abs()
                );
            }
        }
    }

    // -------------------------------------------------------------------------
    // Lorenz: verify chaotic behavior (positive Lyapunov exponent)
    // -------------------------------------------------------------------------

    #[test]
    fn test_lorenz_chaotic_divergence() {
        let lorenz = LorenzSystem::standard();
        let integrator = Rk4Integrator;

        let state1 = [1.0, 1.0, 1.0];
        let state2 = [1.0 + 1e-6, 1.0, 1.0]; // Small but measurable perturbation

        let res1 = simulate(&lorenz, &integrator, &state1, 0.0, 40.0, 0.005);
        let res2 = simulate(&lorenz, &integrator, &state2, 0.0, 40.0, 0.005);

        // After sufficient time, trajectories should diverge exponentially.
        // With lambda ~ 0.9, a perturbation of 1e-6 reaches O(1) after
        // t ~ ln(1e6) / 0.9 ~ 15 seconds. Check at the end.
        let n = res1.states.len();
        let late_idx = n - 1;

        let dist: f64 = res1.states[late_idx]
            .iter()
            .zip(res2.states[late_idx].iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();

        assert!(
            dist > 1.0,
            "Lorenz trajectories should diverge significantly, got dist={:.4}",
            dist
        );
    }

    #[test]
    fn test_lorenz_positive_lyapunov() {
        // Estimate the maximal Lyapunov exponent by tracking divergence
        let lorenz = LorenzSystem::standard();
        let integrator = Rk4Integrator;
        let dt = 0.01;

        let mut state = vec![1.0, 1.0, 1.0];
        let perturbation = 1e-8;
        let mut perturbed = vec![1.0 + perturbation, 1.0, 1.0];

        // Skip transient
        for _ in 0..2000 {
            state = integrator.step(&lorenz, &state, 0.0, dt);
            perturbed = integrator.step(&lorenz, &perturbed, 0.0, dt);
        }

        // Estimate Lyapunov exponent with renormalization
        let mut lambda_sum = 0.0;
        let mut count = 0;
        let renorm_interval = 10;

        // Reinitialize perturbation after transient
        perturbed = state.clone();
        perturbed[0] += perturbation;

        for step in 0..5000 {
            state = integrator.step(&lorenz, &state, 0.0, dt);
            perturbed = integrator.step(&lorenz, &perturbed, 0.0, dt);

            if (step + 1) % renorm_interval == 0 {
                let dist: f64 = state
                    .iter()
                    .zip(perturbed.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f64>()
                    .sqrt();

                if dist > 1e-15 {
                    lambda_sum += (dist / perturbation).ln();
                    count += 1;

                    // Renormalize perturbation
                    for i in 0..3 {
                        perturbed[i] = state[i] + (perturbed[i] - state[i]) * perturbation / dist;
                    }
                }
            }
        }

        let lambda = lambda_sum / (count as f64 * renorm_interval as f64 * dt);

        // Lorenz maximal Lyapunov exponent is approximately 0.9
        assert!(
            lambda > 0.5 && lambda < 1.5,
            "Lorenz lambda = {:.3}, expected ~0.9",
            lambda
        );
    }

    #[test]
    fn test_lorenz_bounded_attractor() {
        let lorenz = LorenzSystem::standard();
        let integrator = Rk4Integrator;

        let initial = [1.0, 1.0, 1.0];
        let result = simulate(&lorenz, &integrator, &initial, 0.0, 100.0, 0.01);

        // All states should remain bounded (Lorenz attractor is compact)
        for state in &result.states {
            for &v in state {
                assert!(
                    v.abs() < 100.0,
                    "Lorenz state should be bounded, got {:.1}",
                    v
                );
            }
        }
    }

    // -------------------------------------------------------------------------
    // Verlet vs RK4: energy drift comparison for Hamiltonian systems
    // -------------------------------------------------------------------------

    #[test]
    fn test_verlet_vs_rk4_energy_drift() {
        let osc = HarmonicOscillator::simple(1.0);
        let initial = [1.0, 0.0];
        let t_end = 100.0;
        let dt = 0.01;

        // Run with RK4
        let rk4 = Rk4Integrator;
        let res_rk4 = simulate(&osc, &rk4, &initial, 0.0, t_end, dt);

        // Run with Verlet
        let verlet = VerletIntegrator;
        let res_verlet = simulate(&osc, &verlet, &initial, 0.0, t_end, dt);

        let e0 = osc.energy(&initial);

        // Compute max energy drift for each
        let rk4_max_drift: f64 = res_rk4
            .states
            .iter()
            .map(|s| ((osc.energy(s) - e0) / e0).abs())
            .fold(0.0f64, f64::max);

        let verlet_max_drift: f64 = res_verlet
            .states
            .iter()
            .map(|s| ((osc.energy(s) - e0) / e0).abs())
            .fold(0.0f64, f64::max);

        // Both should be small for this simple system
        assert!(
            rk4_max_drift < 1e-8,
            "RK4 energy drift {:.2e} too large",
            rk4_max_drift
        );
        assert!(
            verlet_max_drift < 1e-4,
            "Verlet energy drift {:.2e} too large",
            verlet_max_drift
        );

        // For long-term integration, Verlet's drift should not grow secularly
        // (it oscillates, while RK4's can grow). Check last 10% of trajectory.
        let n = res_verlet.states.len();
        let late_start = n * 9 / 10;
        let verlet_late_drifts: Vec<f64> = res_verlet.states[late_start..]
            .iter()
            .map(|s| ((osc.energy(s) - e0) / e0).abs())
            .collect();
        let verlet_late_max = verlet_late_drifts.iter().cloned().fold(0.0f64, f64::max);

        // Late drift should not be much worse than overall drift
        assert!(
            verlet_late_max < verlet_max_drift * 2.0,
            "Verlet late drift {:.2e} should not grow much beyond overall {:.2e}",
            verlet_late_max,
            verlet_max_drift
        );
    }

    // -------------------------------------------------------------------------
    // Adaptive RK45: verify step-size adaptation
    // -------------------------------------------------------------------------

    #[test]
    fn test_rk45_adaptive_accuracy() {
        let osc = HarmonicOscillator::simple(1.0);
        let rk45 = Rk45Integrator::new(1e-10);

        let initial = [1.0, 0.0];
        let result = simulate(&osc, &rk45, &initial, 0.0, 10.0, 0.1);

        let e0 = osc.energy(&initial);
        let max_drift: f64 = result
            .states
            .iter()
            .map(|s| ((osc.energy(s) - e0) / e0).abs())
            .fold(0.0f64, f64::max);

        assert!(
            max_drift < 1e-8,
            "RK45 energy drift {:.2e} should be < 1e-8",
            max_drift
        );
    }

    #[test]
    fn scale_aware_rk45_reuses_controller_proposal_and_reports_work() {
        let oscillator = HarmonicOscillator::simple(1.0);
        let integrator = Rk45Integrator {
            tol: 1.0e-8,
            dt_min: 1.0e-12,
            dt_max: 0.1,
        };
        let tolerances = Rk45Tolerances::new(1.0e-6, vec![1.0e-8; 2]);
        let mut controller = Rk45Controller::new(1.0e-3).unwrap();

        let first = controller
            .try_step(&integrator, &oscillator, &[1.0, 0.0], 0.0, 0.1, &tolerances)
            .unwrap();
        assert!(first.error_norm <= 1.0);
        assert_eq!(first.derivative_evaluations, first.attempted_steps * 6);
        assert_eq!(first.rejected_steps + 1, first.attempted_steps);
        assert_eq!(controller.next_step_size(), Some(first.proposed_next_dt));

        let second = controller
            .try_step(
                &integrator,
                &oscillator,
                &first.state,
                first.accepted_dt,
                0.1,
                &tolerances,
            )
            .unwrap();
        assert_eq!(
            second.attempted_steps, 1,
            "the persisted proposal should avoid repeating the rejected startup attempts"
        );
        assert!(
            (second.accepted_dt - first.proposed_next_dt).abs() < 1.0e-14,
            "second step {} should reuse proposal {}",
            second.accepted_dt,
            first.proposed_next_dt
        );
    }

    #[test]
    fn scale_aware_simulation_is_invariant_to_uniform_state_rescaling() {
        let oscillator = HarmonicOscillator::simple(1.0);
        let integrator = Rk45Integrator::new(1.0e-8);
        let scale = 1.0e6;
        let small_tolerances = Rk45Tolerances::new(1.0e-7, vec![1.0e-10; 2]);
        let large_tolerances = Rk45Tolerances::new(1.0e-7, vec![1.0e-4; 2]);

        let small = integrator
            .try_simulate_with_tolerances(
                &oscillator,
                &[1.0, 0.0],
                0.0,
                10.0,
                0.2,
                &small_tolerances,
            )
            .unwrap();
        let large = integrator
            .try_simulate_with_tolerances(
                &oscillator,
                &[scale, 0.0],
                0.0,
                10.0,
                0.2,
                &large_tolerances,
            )
            .unwrap();

        assert_eq!(small.simulation.steps, large.simulation.steps);
        assert_eq!(
            small.diagnostics.rejected_steps,
            large.diagnostics.rejected_steps
        );
        assert!(
            (small.diagnostics.max_error_norm - large.diagnostics.max_error_norm).abs() < 1.0e-5,
            "normalized error should be invariant under scaled units: small={}, large={}",
            small.diagnostics.max_error_norm,
            large.diagnostics.max_error_norm
        );
        for (small_state, large_state) in small
            .simulation
            .final_state()
            .iter()
            .zip(large.simulation.final_state())
        {
            assert!(
                (scale * small_state - large_state).abs() / scale < 1.0e-8,
                "scaled final states should agree: small={}, large={}",
                small_state,
                large_state
            );
        }
        assert_eq!(
            large.diagnostics.derivative_evaluations,
            6 * (large.diagnostics.accepted_steps + large.diagnostics.rejected_steps)
        );
        assert!(large.diagnostics.max_error_norm <= 1.0);
        assert_eq!(large.simulation.times.last().copied(), Some(10.0));
    }

    #[test]
    fn scale_aware_rk45_rejects_invalid_tolerance_profiles() {
        let oscillator = HarmonicOscillator::simple(1.0);
        let integrator = Rk45Integrator::new(1.0e-8);
        let invalid_profiles = [
            Rk45Tolerances::new(f64::NAN, vec![1.0e-8; 2]),
            Rk45Tolerances::new(-1.0, vec![1.0e-8; 2]),
            Rk45Tolerances::new(1.0e-6, vec![1.0e-8]),
            Rk45Tolerances::new(1.0e-6, vec![0.0, 1.0e-8]),
            Rk45Tolerances::new(1.0e-6, vec![f64::INFINITY, 1.0e-8]),
        ];
        for profile in invalid_profiles {
            assert_eq!(
                integrator.try_adaptive_step_with_tolerances(
                    &oscillator,
                    &[1.0, 0.0],
                    0.0,
                    0.1,
                    &profile,
                ),
                Err(SimulationError::InvalidRk45Tolerances)
            );
        }
    }

    #[test]
    fn checked_simulation_rejects_rk45_minimum_step_tolerance_shortfall() {
        let oscillator = HarmonicOscillator::simple(1.0);
        let integrator = Rk45Integrator {
            tol: 1e-14,
            dt_min: 0.1,
            dt_max: 0.1,
        };
        let result = try_simulate(&oscillator, &integrator, &[1.0, 0.0], 0.0, 0.1, 0.1);
        assert_eq!(result, Err(SimulationError::AdaptiveToleranceNotMet));
    }

    struct MalformedDerivativeSystem {
        output: Vec<f64>,
    }

    impl DynamicalSystem for MalformedDerivativeSystem {
        fn state_dim(&self) -> usize {
            2
        }

        fn derivatives(&self, _state: &[f64], _t: f64) -> Vec<f64> {
            self.output.clone()
        }

        fn name(&self) -> &str {
            "MalformedDerivativeSystem"
        }
    }

    #[test]
    fn checked_rk45_rejects_wrong_dimension_derivatives() {
        let system = MalformedDerivativeSystem { output: vec![0.0] };
        let result = Rk45Integrator::new(1e-8).try_adaptive_step(
            &system,
            &[1.0, 0.0],
            0.0,
            0.1,
        );
        assert_eq!(
            result,
            Err(SimulationError::DerivativeDimensionMismatch {
                stage: 1,
                expected: 2,
                actual: 1,
            })
        );
    }

    #[test]
    fn checked_rk45_rejects_non_finite_derivatives() {
        let system = MalformedDerivativeSystem {
            output: vec![f64::NAN, 0.0],
        };
        let result = Rk45Integrator::new(1e-8).try_adaptive_step(
            &system,
            &[1.0, 0.0],
            0.0,
            0.1,
        );
        assert_eq!(
            result,
            Err(SimulationError::NonFiniteDerivative { stage: 1, index: 0 })
        );
    }

    #[test]
    fn test_rk45_error_estimate() {
        let lorenz = LorenzSystem::standard();
        let rk45 = Rk45Integrator::new(1e-6);

        let state = [1.0, 1.0, 1.0];
        let (_new_state, actual_dt, error) = rk45.adaptive_step(&lorenz, &state, 0.0, 0.1);

        // The step should have been taken (actual_dt > 0)
        assert!(actual_dt > 0.0, "Step should have been taken");

        // Error should be within tolerance
        assert!(
            error <= 1e-6 || actual_dt <= rk45.dt_min,
            "Error {:.2e} should be within tolerance or at minimum step",
            error
        );
    }

    #[test]
    fn test_rk45_vs_rk4_accuracy() {
        // RK45 with tight tolerance should be at least as accurate as RK4
        let osc = HarmonicOscillator::simple(1.0);
        let initial = [1.0, 0.0];
        let t_end = 10.0;

        let rk4 = Rk4Integrator;
        let res_rk4 = simulate(&osc, &rk4, &initial, 0.0, t_end, 0.01);

        let rk45 = Rk45Integrator::new(1e-12);
        let res_rk45 = simulate(&osc, &rk45, &initial, 0.0, t_end, 0.01);

        // Exact solution at t_end: x = cos(t_end), v = -sin(t_end)
        let exact_x = t_end.cos();
        let exact_v = -t_end.sin();

        let rk4_error = (res_rk4.final_state()[0] - exact_x).abs();
        let rk45_error = (res_rk45.final_state()[0] - exact_x).abs();
        let rk4_velocity_error = (res_rk4.final_state()[1] - exact_v).abs();
        let rk45_velocity_error = (res_rk45.final_state()[1] - exact_v).abs();

        assert!(
            rk45_error < 1e-8,
            "RK45 position error {:.2e} should be very small",
            rk45_error
        );
        assert!(
            rk45_velocity_error < 1e-8,
            "RK45 velocity error {:.2e} should be very small",
            rk45_velocity_error
        );
        // RK4 with dt=0.01 should also be quite good for both state variables.
        assert!(
            rk4_error < 1e-6,
            "RK4 position error {:.2e} should be small",
            rk4_error
        );
        assert!(
            rk4_velocity_error < 1e-6,
            "RK4 velocity error {:.2e} should be small",
            rk4_velocity_error
        );
    }

    // -------------------------------------------------------------------------
    // Euler integrator basic test
    // -------------------------------------------------------------------------

    #[test]
    fn test_euler_basic() {
        // Euler should work but with more error
        let osc = HarmonicOscillator::simple(1.0);
        let euler = EulerIntegrator;

        let initial = [1.0, 0.0];
        let result = simulate(&osc, &euler, &initial, 0.0, 1.0, 0.0001);

        // Euler is not symplectic, so energy will drift, but with small dt
        // it should still give a reasonable answer
        let exact_x = 1.0f64.cos();
        let final_x = result.final_state()[0];

        assert!(
            (final_x - exact_x).abs() < 0.01,
            "Euler final x = {:.6}, expected ~{:.6}",
            final_x,
            exact_x
        );
    }

    // -------------------------------------------------------------------------
    // Double pendulum tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_double_pendulum_small_angle() {
        // For very small angles, the double pendulum should behave
        // approximately like two coupled harmonic oscillators
        let dp = DoublePendulum::symmetric(1.0, 1.0, 9.81);
        let integrator = Rk4Integrator;

        // Very small initial angle
        let initial = [0.01, 0.005, 0.0, 0.0];
        let result = simulate(&dp, &integrator, &initial, 0.0, 10.0, 0.001);

        // Energy should be approximately conserved
        let e0 = dp.energy(&initial);
        let mut max_drift = 0.0f64;

        for state in &result.states {
            let e = dp.energy(state);
            let drift = (e - e0).abs() / e0.abs().max(1e-15);
            if drift > max_drift {
                max_drift = drift;
            }
        }

        assert!(
            max_drift < 1e-6,
            "Double pendulum energy drift {:.2e} should be < 1e-6 for small angles",
            max_drift
        );
    }

    #[test]
    fn test_double_pendulum_chaotic_sensitivity() {
        let dp = DoublePendulum::symmetric(1.0, 1.0, 9.81);
        let integrator = Rk4Integrator;

        // Large initial angle => chaotic regime
        let state1 = [PI / 2.0, PI / 2.0, 0.0, 0.0];
        let state2 = [PI / 2.0 + 1e-8, PI / 2.0, 0.0, 0.0];

        let res1 = simulate(&dp, &integrator, &state1, 0.0, 20.0, 0.001);
        let res2 = simulate(&dp, &integrator, &state2, 0.0, 20.0, 0.001);

        // After sufficient time, small perturbation should grow significantly
        let n = res1.states.len();
        let late_idx = n - 1;

        let dist: f64 = res1.states[late_idx]
            .iter()
            .zip(res2.states[late_idx].iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();

        assert!(
            dist > 0.01,
            "Double pendulum should show sensitivity to initial conditions, got dist={:.6}",
            dist
        );
    }

    // -------------------------------------------------------------------------
    // Electromagnetic: cyclotron motion
    // -------------------------------------------------------------------------

    #[test]
    fn test_cyclotron_circular_orbit() {
        // Particle in uniform B field along z: should trace a circle in xy plane
        let qm = 1.0;
        let b_z = 1.0;
        let particle = ChargedParticle::cyclotron(qm, b_z);
        let integrator = Rk4Integrator;

        // Initial: at (1,0,0), velocity (0,1,0) => circular orbit of radius 1
        let initial = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let period = particle.cyclotron_period();
        let result = simulate(&particle, &integrator, &initial, 0.0, period * 2.0, 0.001);

        // After one period, should return to starting position
        let mut best_idx = 0;
        let mut best_diff = f64::MAX;
        for (i, &t) in result.times.iter().enumerate() {
            if i == 0 {
                continue;
            }
            let diff = (t - period).abs();
            if diff < best_diff {
                best_diff = diff;
                best_idx = i;
            }
        }

        let x = result.states[best_idx][0];
        let y = result.states[best_idx][1];
        assert!(
            (x - 1.0).abs() < 0.01,
            "x at period should be ~1.0, got {:.4}",
            x
        );
        assert!(y.abs() < 0.01, "y at period should be ~0.0, got {:.4}", y);
    }

    #[test]
    fn test_cyclotron_energy_conservation() {
        // In pure B field, kinetic energy is conserved (magnetic force does no work)
        let particle = ChargedParticle::cyclotron(2.0, 3.0);
        let integrator = Rk4Integrator;

        let initial = [0.0, 0.0, 0.0, 1.0, 0.5, 0.3];
        let result = simulate(&particle, &integrator, &initial, 0.0, 20.0, 0.001);

        let ke0 = particle.kinetic_energy(&initial);
        let mut max_drift = 0.0f64;
        for state in &result.states {
            let ke = particle.kinetic_energy(state);
            let drift = (ke - ke0).abs() / ke0;
            if drift > max_drift {
                max_drift = drift;
            }
        }

        assert!(
            max_drift < 1e-8,
            "Kinetic energy drift {:.2e} should be < 1e-8 in pure B field",
            max_drift
        );
    }

    #[test]
    fn test_e_cross_b_drift() {
        // In crossed E and B fields, the particle drifts with velocity v_d = E×B/|B|²
        let e_y = 1.0;
        let b_z = 2.0;
        let particle = ChargedParticle::e_cross_b_drift(e_y, b_z);
        let integrator = Rk4Integrator;

        // Start at rest
        let initial = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let result = simulate(&particle, &integrator, &initial, 0.0, 50.0, 0.001);

        // Expected drift velocity: v_x = E_y / B_z = 0.5
        let n = result.states.len();
        let late_start = n * 3 / 4;

        // Average vx in late portion should approach drift velocity
        let avg_vx: f64 = result.states[late_start..]
            .iter()
            .map(|s| s[3])
            .sum::<f64>()
            / (n - late_start) as f64;

        let expected_drift = e_y / b_z;
        assert!(
            (avg_vx - expected_drift).abs() < 0.2,
            "Average vx drift should be ~{:.2}, got {:.4}",
            expected_drift,
            avg_vx
        );
    }

    // -------------------------------------------------------------------------
    // Thermodynamic: heat equation
    // -------------------------------------------------------------------------

    #[test]
    fn test_heat_equation_convergence_to_steady_state() {
        // A rod with hot left and cold right should converge to linear profile
        let heat = HeatEquation1D::hot_cold_rod(0.01, 20);
        let integrator = Rk4Integrator;

        // Initial: uniform 50°C
        let initial: Vec<f64> = vec![50.0; 20];
        let result = simulate(&heat, &integrator, &initial, 0.0, 100.0, 0.01);

        let final_state = result.final_state();
        let dev = heat.max_deviation_from_steady(final_state);

        assert!(
            dev < 1.0,
            "Should converge near steady state, max deviation = {:.4}",
            dev
        );
    }

    #[test]
    fn test_heat_equation_temperature_bounded() {
        // Temperature should stay between boundary values (maximum principle)
        let heat = HeatEquation1D::hot_cold_rod(0.01, 10);
        let integrator = Rk4Integrator;

        // Initial: temperature spike in the middle
        let mut initial = vec![0.0; 10];
        initial[5] = 200.0; // Hot spot

        let result = simulate(&heat, &integrator, &initial, 0.0, 50.0, 0.01);

        // After diffusion, max temp should not exceed max(boundaries, initial max)
        let max_initial = 200.0f64;
        for state in &result.states {
            for &t in state {
                assert!(
                    t <= max_initial + 1.0 && t >= -1.0,
                    "Temperature {:.2} out of expected bounds",
                    t
                );
            }
        }
    }

    #[test]
    fn test_heat_equation_energy_conservation() {
        // With Dirichlet boundaries, total energy changes as heat flows
        // through boundaries, but for insulated (Neumann) we'd get conservation.
        // Here just verify it's monotonically approaching steady-state energy.
        let heat = HeatEquation1D::new(0.01, 1.0, 10, 100.0, 100.0);
        let integrator = Rk4Integrator;

        // Initial: all at 50 (below boundary temps of 100)
        let initial = vec![50.0; 10];
        let result = simulate(&heat, &integrator, &initial, 0.0, 50.0, 0.01);

        // Energy should increase monotonically toward steady state (100 everywhere)
        let mut prev_energy = heat.total_energy(&initial);
        let check_interval = result.states.len() / 20;

        for (i, state) in result.states.iter().enumerate() {
            if i % check_interval.max(1) != 0 {
                continue;
            }
            let energy = heat.total_energy(state);
            assert!(
                energy >= prev_energy - 0.01,
                "Energy should not decrease: {:.4} < {:.4} at step {}",
                energy,
                prev_energy,
                i
            );
            prev_energy = energy;
        }
    }

    // -------------------------------------------------------------------------
    // HDC bridge tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_state_to_hv_basic() {
        let state = [1.0, 0.0, -1.0];
        let hv = state_to_hv(&state);

        assert_eq!(hv.dim(), ContinuousHV::DEFAULT_DIM);

        // Should be normalized
        let norm: f32 = hv.values.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 0.01,
            "HV should be approximately normalized, got norm={:.4}",
            norm
        );
    }

    #[test]
    fn test_state_to_hv_different_states() {
        let s1 = [1.0, 0.0];
        let s2 = [0.0, 1.0];
        let s3 = [1.0, 0.0]; // Same as s1

        let hv1 = state_to_hv(&s1);
        let hv2 = state_to_hv(&s2);
        let hv3 = state_to_hv(&s3);

        // Same state should produce same HV
        let sim_same = hv1.similarity(&hv3);
        assert!(
            (sim_same - 1.0).abs() < 0.001,
            "Same state should produce identical HV, got sim={:.4}",
            sim_same
        );

        // Different states should produce different HVs (low similarity)
        let sim_diff = hv1.similarity(&hv2);
        assert!(
            sim_diff.abs() < 0.5,
            "Different states should produce dissimilar HVs, got sim={:.4}",
            sim_diff
        );
    }

    #[test]
    fn test_trajectory_to_hvs() {
        let osc = HarmonicOscillator::simple(1.0);
        let integrator = Rk4Integrator;
        let initial = [1.0, 0.0];
        let result = simulate(&osc, &integrator, &initial, 0.0, 1.0, 0.1);

        let hvs = trajectory_to_hvs(&result);

        assert_eq!(hvs.len(), result.states.len());

        // Consecutive states should be similar (small dt => small change)
        if hvs.len() >= 2 {
            let sim = hvs[0].similarity(&hvs[1]);
            assert!(
                sim > 0.8,
                "Consecutive states should be similar, got sim={:.4}",
                sim
            );
        }
    }

    #[test]
    fn test_state_to_hv_empty() {
        let hv = state_to_hv(&[]);
        assert_eq!(hv.dim(), ContinuousHV::DEFAULT_DIM);

        // Should be all zeros
        let norm: f32 = hv.values.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            norm < 1e-10,
            "Empty state should produce zero HV, got norm={:.6}",
            norm
        );
    }

    // -------------------------------------------------------------------------
    // SimulationResult helpers
    // -------------------------------------------------------------------------

    #[test]
    fn test_simulation_result_variable() {
        let osc = HarmonicOscillator::simple(1.0);
        let integrator = Rk4Integrator;
        let initial = [1.0, 0.0];
        let result = simulate(&osc, &integrator, &initial, 0.0, 1.0, 0.1);

        let x_series = result.variable(0);
        let v_series = result.variable(1);

        assert_eq!(x_series.len(), result.states.len());
        assert_eq!(v_series.len(), result.states.len());
        assert!((x_series[0] - 1.0).abs() < 1e-15);
        assert!((v_series[0] - 0.0).abs() < 1e-15);
    }

    // -------------------------------------------------------------------------
    // Integration: full system pipeline
    // -------------------------------------------------------------------------

    #[test]
    fn test_full_pipeline_lorenz_to_hdc() {
        let lorenz = LorenzSystem::standard();
        let integrator = Rk4Integrator;

        let initial = [1.0, 1.0, 1.0];
        let result = simulate(&lorenz, &integrator, &initial, 0.0, 5.0, 0.01);

        assert!(result.steps > 0);
        assert_eq!(result.system_name, "LorenzSystem");

        let hvs = trajectory_to_hvs(&result);
        assert_eq!(hvs.len(), result.states.len());

        // Early and late HVs should be different (chaotic trajectory explores space)
        let early = &hvs[10];
        let late = &hvs[hvs.len() - 10];
        let sim = early.similarity(late);
        // They can be somewhat similar or not, but the test is that the pipeline works
        assert!(sim.is_finite(), "Similarity should be finite");
    }
}
