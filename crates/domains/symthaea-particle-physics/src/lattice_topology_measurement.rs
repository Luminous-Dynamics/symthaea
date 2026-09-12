// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Provenance-bearing flowed-topology measurements.
//!
//! This module deliberately owns no gauge algebra. It wraps the LQCD-017E
//! semantic reference with a version-stable measurement definition so topology
//! histories cannot silently mix different operators, flow algorithms, step
//! sizes, gradient probes, step counts, or smoothing times behind one `f64`.

use crate::lattice_gauge::WilsonGaugeField;
use crate::lattice_topology_flow::{
    LatticeTopologyFlowError, clover_topological_charge,
    finite_difference_wilson_flow_step_reference,
};

pub const CLOVER_TOPOLOGY_OPERATOR_ID: &str = "symthaea_clover_q_v1";
pub const REFERENCE_WILSON_FLOW_ID: &str = "wilson_action_fd_lie_euler_v1";

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReferenceFlowSchedule {
    pub steps: usize,
    pub dt: f64,
    pub gradient_epsilon: f64,
}

impl ReferenceFlowSchedule {
    pub fn validate(self) -> Result<(), FlowedTopologyMeasurementError> {
        if self.steps == 0 {
            return Err(FlowedTopologyMeasurementError::InvalidStepCount(0));
        }
        if !self.dt.is_finite() || self.dt <= 0.0 {
            return Err(FlowedTopologyMeasurementError::InvalidFlowStep(self.dt));
        }
        if !self.gradient_epsilon.is_finite() || self.gradient_epsilon <= 0.0 {
            return Err(FlowedTopologyMeasurementError::InvalidGradientEpsilon(
                self.gradient_epsilon,
            ));
        }
        Ok(())
    }

    pub fn flow_time(self) -> Result<f64, FlowedTopologyMeasurementError> {
        self.validate()?;
        let flow_time = self.steps as f64 * self.dt;
        if !flow_time.is_finite() {
            return Err(FlowedTopologyMeasurementError::NonFiniteFlowTime);
        }
        Ok(flow_time)
    }

    pub fn stable_id(self) -> Result<String, FlowedTopologyMeasurementError> {
        let flow_time = self.flow_time()?;
        Ok(format!(
            "{REFERENCE_WILSON_FLOW_ID}:steps={}:dt={:.17e}:grad_eps={:.17e}:t={:.17e}",
            self.steps, self.dt, self.gradient_epsilon, flow_time,
        ))
    }
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
pub struct FlowedTopologyMeasurement {
    pub definition: FlowedTopologyDefinition,
    pub charge: f64,
    /// Wilson action at t=0 followed by the action after every declared step.
    pub action_history: Vec<f64>,
    pub max_unitarity_error: f64,
    pub max_determinant_error: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FlowedTopologyMeasurementError {
    Flow(LatticeTopologyFlowError),
    InvalidStepCount(usize),
    InvalidFlowStep(f64),
    InvalidGradientEpsilon(f64),
    NonFiniteFlowTime,
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

/// Run the declared semantic-reference flow on a clone and measure clover Q.
///
/// The caller's field is never mutated. Each Lie-Euler step must lower or
/// preserve the Wilson action within a tiny floating tolerance; otherwise the
/// requested finite step is rejected rather than being recorded as a valid
/// flowed-topology measurement.
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
        let tolerance = 1.0e-12 * (1.0 + stats.action_before.abs());
        if stats.action_after > stats.action_before + tolerance {
            return Err(FlowedTopologyMeasurementError::ActionIncreased {
                step,
                before: stats.action_before,
                after: stats.action_after,
            });
        }
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
    fn schedule_identity_binds_every_numerical_parameter() {
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
    fn identity_measurement_is_zero_and_does_not_mutate_input() {
        let field = WilsonGaugeField::identity([2, 2, 2, 2]).unwrap();
        let before = field.wilson_action(1.0).unwrap();
        let measurement = measure_flowed_clover_topology_reference(
            &field,
            ReferenceFlowSchedule {
                steps: 1,
                dt: 1.0e-3,
                gradient_epsilon: 2.0e-6,
            },
        )
        .unwrap();
        assert_eq!(measurement.definition.operator_id, CLOVER_TOPOLOGY_OPERATOR_ID);
        assert_eq!(measurement.definition.smoothing_time, 1.0e-3);
        assert_eq!(measurement.action_history, vec![0.0, 0.0]);
        assert!(measurement.charge.abs() < 1.0e-15);
        assert_eq!(field.wilson_action(1.0).unwrap(), before);
    }

    #[test]
    fn invalid_schedules_fail_closed() {
        let zero_steps = ReferenceFlowSchedule { steps: 0, ..base_schedule() };
        assert!(matches!(
            zero_steps.validate(),
            Err(FlowedTopologyMeasurementError::InvalidStepCount(0))
        ));

        let zero_dt = ReferenceFlowSchedule { dt: 0.0, ..base_schedule() };
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
