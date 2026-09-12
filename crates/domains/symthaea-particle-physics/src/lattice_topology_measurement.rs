// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Provenance-bearing flowed-topology measurements.
//!
//! This module wraps the slow LQCD-017E semantic reference with an explicit,
//! version-stable measurement definition. It exists so downstream topology
//! histories cannot silently mix different clover conventions, flow methods,
//! step sizes, finite-difference probes, or flow times behind the same `f64`.

use crate::lattice_gauge::{Site4, WilsonGaugeField, su3_mul};
use crate::lattice_topology_flow::{
    LatticeTopologyFlowError, clover_topological_charge,
    finite_difference_wilson_flow_step_reference,
};
use crate::symmetry_groups::gell_mann_matrix;

pub const CLOVER_TOPOLOGY_OPERATOR_ID: &str = "symthaea_clover_q_v1";
pub const REFERENCE_WILSON_FLOW_ID: &str = "wilson_action_fd_lie_euler_v1";

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReferenceFlowSchedule {
    pub steps: usize,
    pub dt: f64,
    pub gradient_epsilon: f64,
}

impl ReferenceFlowSchedule {
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
/// The input field is never mutated. Each Lie-Euler step must lower or preserve
/// the Wilson action within a tiny floating tolerance; otherwise the requested
/// finite step is rejected rather than being recorded as a valid Wilson-flow
/// measurement.
pub fn measure_flowed_clover_topology_reference(
    field: &WilsonGaugeField,
    schedule: ReferenceFlowSchedule,
) -> Result<FlowedTopologyMeasurement, FlowedTopologyMeasurementError> {
    schedule.validate()?;
    let smoothing_time = schedule.flow_time()?;
    let smoothing_id = schedule.stable_id()?;
    let mut flowed = field.clone();
    let mut action_history = Vec::with_capacity(schedule.steps + 1);
    action_history.push(flowed.wilson_action(1.0).map_err(LatticeTopologyFlowError::from)?);
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

    let charge = clover_topological_charge(&flowed)?;
    Ok(FlowedTopologyMeasurement {
        definition: FlowedTopologyDefinition {
            operator_id: CLOVER_TOPOLOGY_OPERATOR_ID,
            smoothing_id,
            smoothing_time,
            steps: schedule.steps,
            dt: schedule.dt,
            gradient_epsilon: schedule.gradient_epsilon,
        },
        charge,
        action_history,
        max_unitarity_error,
        max_determinant_error,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice_gauge::WilsonGaugeField;
    use crate::lattice_topology_flow::ReferenceFlowStepStats;
    use crate::symmetry_groups::Complex;

    const FIXTURE_OPS: [(Site4, usize, usize, f64); 40] = [
        ([0, 0, 1, 0], 0, 3, -0.5819354339098058),
        ([0, 1, 1, 1], 0, 2, -0.32393958624710995),
        ([0, 1, 1, 1], 0, 6, -0.1250464331865352),
        ([1, 0, 1, 0], 1, 2, 0.19161237542499698),
        ([0, 1, 0, 1], 1, 2, 0.22983089548025992),
        ([1, 0, 1, 1], 1, 3, 0.4032003002590304),
        ([1, 1, 1, 1], 3, 5, 0.5114974521920254),
        ([1, 1, 1, 1], 2, 4, -0.09168927490735845),
        ([1, 1, 1, 0], 1, 6, -0.4726273806149346),
        ([0, 1, 0, 0], 2, 2, 0.3411282616087933),
        ([1, 1, 0, 0], 3, 3, -0.341627594608156),
        ([1, 0, 1, 1], 2, 5, -0.11361497865301007),
        ([1, 0, 1, 0], 1, 4, 0.003219526879812973),
        ([0, 1, 1, 1], 3, 1, 0.2092875716255429),
        ([0, 0, 1, 1], 1, 0, -0.2872940926296875),
        ([1, 0, 1, 1], 1, 3, -0.33032520366686874),
        ([1, 1, 1, 0], 3, 1, 0.5401678740668779),
        ([0, 0, 1, 0], 2, 2, 0.31304313667691264),
        ([1, 1, 0, 0], 0, 1, 0.2375215627437346),
        ([0, 1, 0, 0], 0, 0, 0.3958604638855575),
        ([1, 1, 0, 0], 0, 5, -0.42942950746005015),
        ([1, 1, 1, 1], 2, 6, 0.5455840065263183),
        ([1, 0, 0, 0], 2, 3, 0.017115673418346744),
        ([1, 0, 0, 1], 1, 2, 0.13107989509472173),
        ([1, 0, 1, 1], 1, 7, -0.4551702244963054),
        ([1, 0, 0, 0], 0, 0, -0.5395094927947152),
        ([1, 1, 1, 1], 2, 2, 0.3670444298654235),
        ([0, 0, 0, 0], 2, 6, -0.24861863368491577),
        ([1, 1, 1, 1], 2, 3, 0.335177509798799),
        ([0, 1, 1, 1], 2, 5, 0.5656866202369394),
        ([1, 1, 1, 1], 2, 2, 0.47986932117951764),
        ([1, 0, 0, 0], 3, 4, 0.013402168869181441),
        ([0, 0, 0, 1], 2, 4, 0.10603947222843657),
        ([0, 1, 0, 0], 2, 0, 0.23018975518126505),
        ([0, 0, 0, 0], 1, 0, 0.19239685997822842),
        ([1, 0, 1, 0], 0, 6, -0.08507387388380139),
        ([1, 1, 0, 0], 0, 2, -0.22404873035871248),
        ([1, 0, 1, 0], 3, 0, -0.35908049548949106),
        ([1, 1, 0, 0], 0, 6, 0.07694552778636243),
        ([1, 0, 0, 0], 0, 5, -0.12989873801836926),
    ];

    fn c_mul(a: Complex, b: Complex) -> Complex {
        Complex::new(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re)
    }

    fn matrix_scale(a: &[[Complex; 3]; 3], scalar: Complex) -> [[Complex; 3]; 3] {
        let mut out = [[Complex::ZERO; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                out[i][j] = c_mul(scalar, a[i][j]);
            }
        }
        out
    }

    fn matrix_add(a: &[[Complex; 3]; 3], b: &[[Complex; 3]; 3]) -> [[Complex; 3]; 3] {
        let mut out = [[Complex::ZERO; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                out[i][j] = Complex::new(a[i][j].re + b[i][j].re, a[i][j].im + b[i][j].im);
            }
        }
        out
    }

    fn matrix_exp(a: &[[Complex; 3]; 3]) -> [[Complex; 3]; 3] {
        let norm = a.iter().flatten().map(|z| z.norm_sq()).sum::<f64>().sqrt();
        let squarings = if norm > 0.5 { (norm / 0.5).log2().ceil() as u32 } else { 0 };
        let x = matrix_scale(a, Complex::new(1.0 / 2.0_f64.powi(squarings as i32), 0.0));
        let mut out = crate::lattice_gauge::su3_identity();
        let mut term = crate::lattice_gauge::su3_identity();
        for k in 1..=50 {
            term = matrix_scale(&su3_mul(&term, &x), Complex::new(1.0 / k as f64, 0.0));
            out = matrix_add(&out, &term);
        }
        for _ in 0..squarings {
            out = su3_mul(&out, &out);
        }
        out
    }

    fn generator_rotation(generator: usize, theta: f64) -> [[Complex; 3]; 3] {
        matrix_exp(&matrix_scale(&gell_mann_matrix(generator), Complex::new(0.0, theta)))
    }

    fn fixture() -> WilsonGaugeField {
        let mut field = WilsonGaugeField::identity([2, 2, 2, 2]).unwrap();
        for (site, mu, generator, theta) in FIXTURE_OPS {
            let rotation = generator_rotation(generator, theta);
            let original = *field.link(site, mu).unwrap();
            field.set_link(site, mu, su3_mul(&rotation, &original)).unwrap();
        }
        field
    }

    #[test]
    fn schedule_identity_binds_all_numerical_flow_parameters() {
        let base = ReferenceFlowSchedule { steps: 4, dt: 1.0e-3, gradient_epsilon: 2.0e-6 };
        assert_eq!(base.flow_time().unwrap(), 4.0e-3);
        assert_ne!(base.stable_id().unwrap(), ReferenceFlowSchedule { dt: 2.0e-3, ..base }.stable_id().unwrap());
        assert_ne!(base.stable_id().unwrap(), ReferenceFlowSchedule { gradient_epsilon: 1.0e-6, ..base }.stable_id().unwrap());
        assert_ne!(base.stable_id().unwrap(), ReferenceFlowSchedule { steps: 5, ..base }.stable_id().unwrap());
    }

    #[test]
    fn one_step_measurement_matches_lqcd_017d_oracle() {
        let field = fixture();
        let measurement = measure_flowed_clover_topology_reference(
            &field,
            ReferenceFlowSchedule { steps: 1, dt: 1.0e-3, gradient_epsilon: 2.0e-6 },
        ).unwrap();
        assert_eq!(measurement.definition.operator_id, CLOVER_TOPOLOGY_OPERATOR_ID);
        assert_eq!(measurement.definition.smoothing_time, 1.0e-3);
        assert_eq!(measurement.action_history.len(), 2);
        assert!((measurement.action_history[0] - 8.222_361_191_068_600_3).abs() < 2.0e-12);
        assert!((measurement.action_history[1] - 8.131_857_099_782_706_7).abs() < 3.0e-10);
        assert!((measurement.charge - (-0.000_411_907_836_303_515_9)).abs() < 3.0e-14);
    }

    #[test]
    fn zero_step_schedule_fails_closed() {
        let schedule = ReferenceFlowSchedule { steps: 0, dt: 1.0e-3, gradient_epsilon: 2.0e-6 };
        assert!(matches!(schedule.validate(), Err(FlowedTopologyMeasurementError::InvalidStepCount(0))));
    }
}
