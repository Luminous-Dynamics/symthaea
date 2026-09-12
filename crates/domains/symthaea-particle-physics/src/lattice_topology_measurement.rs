// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Provenance-bearing flowed-topology measurements.
//!
//! This module deliberately owns no gauge algebra. It wraps separately qualified
//! flow implementations with version-stable measurement definitions so topology
//! histories cannot silently mix different operators, integrators, step sizes,
//! gradient probes, step counts, or smoothing times behind one `f64`.

use crate::lattice_gauge::WilsonGaugeField;
use crate::lattice_topology_flow::{
    LatticeTopologyFlowError, clover_topological_charge,
    finite_difference_wilson_flow_step_reference,
};
use crate::lattice_topology_flow_rk3::{Rk3FlowError, rk3_wilson_flow_step};

pub const CLOVER_TOPOLOGY_OPERATOR_ID: &str = "symthaea_clover_q_v1";
pub const REFERENCE_WILSON_FLOW_ID: &str = "wilson_action_fd_lie_euler_v1";
pub const RK3_WILSON_FLOW_ID: &str = "wilson_action_staple_rk3_v1";

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReferenceFlowSchedule {
    pub steps: usize,
    pub dt: f64,
    pub gradient_epsilon: f64,
}

impl ReferenceFlowSchedule {
    pub fn validate(self) -> Result<(), FlowedTopologyMeasurementError> {
        validate_steps_dt(self.steps, self.dt)?;
        if !self.gradient_epsilon.is_finite() || self.gradient_epsilon <= 0.0 {
            return Err(FlowedTopologyMeasurementError::InvalidGradientEpsilon(
                self.gradient_epsilon,
            ));
        }
        Ok(())
    }

    pub fn flow_time(self) -> Result<f64, FlowedTopologyMeasurementError> {
        self.validate()?;
        finite_flow_time(self.steps, self.dt)
    }

    pub fn stable_id(self) -> Result<String, FlowedTopologyMeasurementError> {
        let flow_time = self.flow_time()?;
        Ok(format!(
            "{REFERENCE_WILSON_FLOW_ID}:steps={}:dt={:.17e}:grad_eps={:.17e}:t={:.17e}",
            self.steps, self.dt, self.gradient_epsilon, flow_time,
        ))
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rk3FlowSchedule {
    pub steps: usize,
    pub dt: f64,
}

impl Rk3FlowSchedule {
    pub fn validate(self) -> Result<(), FlowedTopologyMeasurementError> {
        validate_steps_dt(self.steps, self.dt)
    }

    pub fn flow_time(self) -> Result<f64, FlowedTopologyMeasurementError> {
        self.validate()?;
        finite_flow_time(self.steps, self.dt)
    }

    pub fn stable_id(self) -> Result<String, FlowedTopologyMeasurementError> {
        let flow_time = self.flow_time()?;
        Ok(format!(
            "{RK3_WILSON_FLOW_ID}:steps={}:dt={:.17e}:t={:.17e}",
            self.steps, self.dt, flow_time,
        ))
    }
}

fn validate_steps_dt(steps: usize, dt: f64) -> Result<(), FlowedTopologyMeasurementError> {
    if steps == 0 {
        return Err(FlowedTopologyMeasurementError::InvalidStepCount(0));
    }
    if !dt.is_finite() || dt <= 0.0 {
        return Err(FlowedTopologyMeasurementError::InvalidFlowStep(dt));
    }
    Ok(())
}

fn finite_flow_time(steps: usize, dt: f64) -> Result<f64, FlowedTopologyMeasurementError> {
    let flow_time = steps as f64 * dt;
    if !flow_time.is_finite() {
        return Err(FlowedTopologyMeasurementError::NonFiniteFlowTime);
    }
    Ok(flow_time)
}

#[derive(Debug, Clone, PartialEq)]
pub struct FlowedTopologyDefinition {
    pub operator_id: &'static str,
    pub smoothing_id: String,
    pub smoothing_time: f64,
    pub steps: usize,
    pub dt: f64,
    pub gradient_epsilon: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Rk3FlowedTopologyDefinition {
    pub operator_id: &'static str,
    pub smoothing_id: String,
    pub smoothing_time: f64,
    pub steps: usize,
    pub dt: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FlowedTopologyMeasurement {
    pub definition: FlowedTopologyDefinition,
    pub charge: f64,
    /// Wilson action at t=0 followed by the action after every declared step.
    pub action_history: Vec<f64>,
    pub max_unitarity_error: f64,
    pub max_determinant_error: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Rk3FlowedTopologyMeasurement {
    pub definition: Rk3FlowedTopologyDefinition,
    pub charge: f64,
    pub action_history: Vec<f64>,
    pub max_unitarity_error: f64,
    pub max_determinant_error: f64,
    /// Exactly three full Wilson-force evaluations per retained RK3 step.
    pub force_evaluations: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FlowedTopologyMeasurementError {
    Flow(LatticeTopologyFlowError),
    Rk3(Rk3FlowError),
    InvalidStepCount(usize),
    InvalidFlowStep(f64),
    InvalidGradientEpsilon(f64),
    NonFiniteFlowTime,
    ForceEvaluationOverflow,
    ActionIncreased {
        step: usize,
        before: f64,
        after: f64,
    },
}

impl From<LatticeTopologyFlowError> for FlowedTopologyMeasurementError {
    fn from(value: LatticeTopologyFlowError) -> Self {
        Self::Flow(value)
    }
}

impl From<Rk3FlowError> for FlowedTopologyMeasurementError {
    fn from(value: Rk3FlowError) -> Self {
        Self::Rk3(value)
    }
}

fn enforce_action_descent(
    step: usize,
    before: f64,
    after: f64,
) -> Result<(), FlowedTopologyMeasurementError> {
    let tolerance = 1.0e-12 * (1.0 + before.abs());
    if after > before + tolerance {
        return Err(FlowedTopologyMeasurementError::ActionIncreased {
            step,
            before,
            after,
        });
    }
    Ok(())
}

/// Run the declared semantic-reference Lie-Euler flow on a clone and measure Q.
pub fn measure_flowed_clover_topology_reference(
    field: &WilsonGaugeField,
    schedule: ReferenceFlowSchedule,
) -> Result<FlowedTopologyMeasurement, FlowedTopologyMeasurementError> {
    schedule.validate()?;
    let smoothing_time = schedule.flow_time()?;
    let smoothing_id = schedule.stable_id()?;

    let mut flowed = field.clone();
    let initial_action = flowed
        .wilson_action(1.0)
        .map_err(LatticeTopologyFlowError::from)?;
    let mut action_history = Vec::with_capacity(schedule.steps + 1);
    action_history.push(initial_action);
    let mut max_unitarity_error: f64 = 0.0;
    let mut max_determinant_error: f64 = 0.0;

    for step in 1..=schedule.steps {
        let stats = finite_difference_wilson_flow_step_reference(
            &mut flowed,
            schedule.dt,
            schedule.gradient_epsilon,
        )?;
        enforce_action_descent(step, stats.action_before, stats.action_after)?;
        action_history.push(stats.action_after);
        max_unitarity_error = max_unitarity_error.max(stats.max_unitarity_error);
        max_determinant_error = max_determinant_error.max(stats.max_determinant_error);
    }

    Ok(FlowedTopologyMeasurement {
        definition: FlowedTopologyDefinition {
            operator_id: CLOVER_TOPOLOGY_OPERATOR_ID,
            smoothing_id,
            smoothing_time,
            steps: schedule.steps,
            dt: schedule.dt,
            gradient_epsilon: schedule.gradient_epsilon,
        },
        charge: clover_topological_charge(&flowed)?,
        action_history,
        max_unitarity_error,
        max_determinant_error,
    })
}

/// Run the independently qualified third-order flow on a clone and measure Q.
///
/// The RK3 identity is deliberately separate from the finite-difference
/// Lie-Euler identity. A history consumer must therefore choose one explicit
/// smoothing lineage rather than mixing integrators at the same nominal `t`.
pub fn measure_flowed_clover_topology_rk3(
    field: &WilsonGaugeField,
    schedule: Rk3FlowSchedule,
) -> Result<Rk3FlowedTopologyMeasurement, FlowedTopologyMeasurementError> {
    schedule.validate()?;
    let smoothing_time = schedule.flow_time()?;
    let smoothing_id = schedule.stable_id()?;

    let mut flowed = field.clone();
    let initial_action = flowed
        .wilson_action(1.0)
        .map_err(LatticeTopologyFlowError::from)?;
    let mut action_history = Vec::with_capacity(schedule.steps + 1);
    action_history.push(initial_action);
    let mut max_unitarity_error: f64 = 0.0;
    let mut max_determinant_error: f64 = 0.0;
    let mut force_evaluations = 0usize;

    for step in 1..=schedule.steps {
        let stats = rk3_wilson_flow_step(&mut flowed, schedule.dt)?;
        enforce_action_descent(step, stats.flow.action_before, stats.flow.action_after)?;
        action_history.push(stats.flow.action_after);
        max_unitarity_error = max_unitarity_error.max(stats.flow.max_unitarity_error);
        max_determinant_error = max_determinant_error.max(stats.flow.max_determinant_error);
        force_evaluations = force_evaluations
            .checked_add(stats.force_evaluations)
            .ok_or(FlowedTopologyMeasurementError::ForceEvaluationOverflow)?;
    }

    Ok(Rk3FlowedTopologyMeasurement {
        definition: Rk3FlowedTopologyDefinition {
            operator_id: CLOVER_TOPOLOGY_OPERATOR_ID,
            smoothing_id,
            smoothing_time,
            steps: schedule.steps,
            dt: schedule.dt,
        },
        charge: clover_topological_charge(&flowed)?,
        action_history,
        max_unitarity_error,
        max_determinant_error,
        force_evaluations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_schedule() -> ReferenceFlowSchedule {
        ReferenceFlowSchedule {
            steps: 2,
            dt: 1.0e-3,
            gradient_epsilon: 2.0e-6,
        }
    }

    #[test]
    fn reference_schedule_identity_binds_every_numerical_parameter() {
        let base = base_schedule();
        assert_eq!(base.flow_time().unwrap(), 2.0e-3);
        assert_ne!(
            base.stable_id().unwrap(),
            ReferenceFlowSchedule { steps: 3, ..base }.stable_id().unwrap()
        );
        assert_ne!(
            base.stable_id().unwrap(),
            ReferenceFlowSchedule { dt: 5.0e-4, ..base }.stable_id().unwrap()
        );
        assert_ne!(
            base.stable_id().unwrap(),
            ReferenceFlowSchedule {
                gradient_epsilon: 1.0e-6,
                ..base
            }
            .stable_id()
            .unwrap()
        );
    }

    #[test]
    fn rk3_identity_is_distinct_and_binds_schedule() {
        let schedule = Rk3FlowSchedule {
            steps: 4,
            dt: 1.0e-3,
        };
        assert_eq!(schedule.flow_time().unwrap(), 4.0e-3);
        let id = schedule.stable_id().unwrap();
        assert!(id.starts_with(RK3_WILSON_FLOW_ID));
        assert!(!id.contains(REFERENCE_WILSON_FLOW_ID));
        assert_ne!(
            id,
            Rk3FlowSchedule {
                steps: 8,
                dt: 5.0e-4,
            }
            .stable_id()
            .unwrap(),
            "same total flow time but different numerical schedules need distinct identities"
        );
    }

    #[test]
    fn identity_measurements_are_zero_and_do_not_mutate_input() {
        let field = WilsonGaugeField::identity([2, 2, 2, 2]).unwrap();
        let before = field.wilson_action(1.0).unwrap();

        let reference = measure_flowed_clover_topology_reference(
            &field,
            ReferenceFlowSchedule {
                steps: 1,
                dt: 1.0e-3,
                gradient_epsilon: 2.0e-6,
            },
        )
        .unwrap();
        assert_eq!(reference.action_history, vec![0.0, 0.0]);
        assert!(reference.charge.abs() < 1.0e-15);

        let rk3 = measure_flowed_clover_topology_rk3(
            &field,
            Rk3FlowSchedule {
                steps: 1,
                dt: 1.0e-3,
            },
        )
        .unwrap();
        assert_eq!(rk3.definition.operator_id, CLOVER_TOPOLOGY_OPERATOR_ID);
        assert_eq!(rk3.definition.smoothing_time, 1.0e-3);
        assert_eq!(rk3.action_history, vec![0.0, 0.0]);
        assert_eq!(rk3.force_evaluations, 3);
        assert!(rk3.charge.abs() < 1.0e-15);
        assert_eq!(field.wilson_action(1.0).unwrap(), before);
    }

    #[test]
    fn invalid_schedules_fail_closed() {
        let zero_steps = ReferenceFlowSchedule { steps: 0, ..base_schedule() };
        assert!(matches!(
            zero_steps.validate(),
            Err(FlowedTopologyMeasurementError::InvalidStepCount(0))
        ));

        let zero_dt = Rk3FlowSchedule { steps: 1, dt: 0.0 };
        assert!(matches!(
            zero_dt.validate(),
            Err(FlowedTopologyMeasurementError::InvalidFlowStep(value)) if value == 0.0
        ));

        let zero_epsilon = ReferenceFlowSchedule {
            gradient_epsilon: 0.0,
            ..base_schedule()
        };
        assert!(matches!(
            zero_epsilon.validate(),
            Err(FlowedTopologyMeasurementError::InvalidGradientEpsilon(value)) if value == 0.0
        ));
    }
}
