// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Auditable one-dimensional tether mechanics reference models.
//!
//! The v0.1 solver evaluates prescribed geometry under signed effective
//! acceleration and optional lumped axial loads. It is deliberately limited to
//! quasi-static axial equilibrium. It does not model transverse dynamics,
//! libration, creep, fatigue, thermal strain, radiation, MMOD, splices, or
//! climber transients.

#![deny(unsafe_code)]

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TetherError {
    FewerThanTwoNodes,
    NonFiniteInput,
    NonIncreasingPosition,
    NonPositiveArea,
    NonPositiveDensity,
    NegativeTipTension,
    NonPositiveAllowableStress,
}

/// One sample along the tether coordinate `s`, which increases toward the
/// positive/end direction.
///
/// `effective_acceleration_m_s2` is signed along +s. `lumped_force_n` is also
/// signed along +s and is applied at this node. The reported node tension is
/// the axial tension immediately on the -s side of the node after including
/// that lumped force.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TetherNodeInput {
    pub s_m: f64,
    pub effective_acceleration_m_s2: f64,
    pub area_m2: f64,
    pub lumped_force_n: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StaticTetherModel {
    pub density_kg_m3: f64,
    /// Tensile boundary load immediately beyond the final +s node.
    pub tip_tension_n: f64,
    /// Optional design/allowable stress used only for utilization reporting.
    pub allowable_stress_pa: Option<f64>,
    pub nodes: Vec<TetherNodeInput>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TetherNodeState {
    pub s_m: f64,
    pub tension_n: f64,
    pub stress_pa: f64,
    pub utilization: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StaticTetherResult {
    pub nodes: Vec<TetherNodeState>,
    pub distributed_mass_kg: f64,
    pub integrated_signed_distributed_force_n: f64,
    pub integrated_signed_lumped_force_n: f64,
    pub minimum_tension_n: f64,
    pub maximum_tension_n: f64,
    pub maximum_tensile_stress_pa: f64,
    pub maximum_utilization: Option<f64>,
}

impl StaticTetherResult {
    pub fn remains_in_tension(&self) -> bool {
        self.minimum_tension_n >= 0.0
    }
}

impl StaticTetherModel {
    pub fn validate(&self) -> Result<(), TetherError> {
        if self.nodes.len() < 2 {
            return Err(TetherError::FewerThanTwoNodes);
        }
        if !self.density_kg_m3.is_finite() || !self.tip_tension_n.is_finite() {
            return Err(TetherError::NonFiniteInput);
        }
        if self.density_kg_m3 <= 0.0 {
            return Err(TetherError::NonPositiveDensity);
        }
        if self.tip_tension_n < 0.0 {
            return Err(TetherError::NegativeTipTension);
        }
        if let Some(allowable) = self.allowable_stress_pa {
            if !allowable.is_finite() {
                return Err(TetherError::NonFiniteInput);
            }
            if allowable <= 0.0 {
                return Err(TetherError::NonPositiveAllowableStress);
            }
        }

        for (index, node) in self.nodes.iter().enumerate() {
            if !node.s_m.is_finite()
                || !node.effective_acceleration_m_s2.is_finite()
                || !node.area_m2.is_finite()
                || !node.lumped_force_n.is_finite()
            {
                return Err(TetherError::NonFiniteInput);
            }
            if node.area_m2 <= 0.0 {
                return Err(TetherError::NonPositiveArea);
            }
            if index > 0 && node.s_m <= self.nodes[index - 1].s_m {
                return Err(TetherError::NonIncreasingPosition);
            }
        }

        Ok(())
    }

    /// Solve static axial equilibrium using trapezoidal integration of the
    /// distributed signed body force `rho * A(s) * a_eff(s)`.
    ///
    /// With `s` increasing toward the final node, equilibrium is
    /// `dT/ds = -lambda(s) a_eff(s)`. Integrating backward gives the tension
    /// required toward the -s side of the tether.
    pub fn solve(&self) -> Result<StaticTetherResult, TetherError> {
        self.validate()?;

        let count = self.nodes.len();
        let mut tensions = vec![0.0_f64; count];
        let mut distributed_mass_kg = 0.0;
        let mut integrated_signed_distributed_force_n = 0.0;
        let integrated_signed_lumped_force_n = self
            .nodes
            .iter()
            .map(|node| node.lumped_force_n)
            .sum::<f64>();

        // Boundary traction plus any point load carried by the final node.
        tensions[count - 1] = self.tip_tension_n + self.nodes[count - 1].lumped_force_n;

        for i in (0..count - 1).rev() {
            let left = self.nodes[i];
            let right = self.nodes[i + 1];
            let ds = right.s_m - left.s_m;

            let lambda_left = self.density_kg_m3 * left.area_m2;
            let lambda_right = self.density_kg_m3 * right.area_m2;
            let segment_mass = 0.5 * (lambda_left + lambda_right) * ds;
            distributed_mass_kg += segment_mass;

            let force_density_left = lambda_left * left.effective_acceleration_m_s2;
            let force_density_right = lambda_right * right.effective_acceleration_m_s2;
            let segment_force = 0.5 * (force_density_left + force_density_right) * ds;
            integrated_signed_distributed_force_n += segment_force;

            tensions[i] = tensions[i + 1] + segment_force + left.lumped_force_n;
        }

        let mut states = Vec::with_capacity(count);
        let mut minimum_tension_n = f64::INFINITY;
        let mut maximum_tension_n = f64::NEG_INFINITY;
        let mut maximum_tensile_stress_pa = 0.0_f64;
        let mut maximum_utilization = self.allowable_stress_pa.map(|_| 0.0_f64);

        for (node, tension_n) in self.nodes.iter().zip(tensions.into_iter()) {
            let stress_pa = tension_n / node.area_m2;
            minimum_tension_n = minimum_tension_n.min(tension_n);
            maximum_tension_n = maximum_tension_n.max(tension_n);
            maximum_tensile_stress_pa = maximum_tensile_stress_pa.max(stress_pa.max(0.0));
            let utilization = self.allowable_stress_pa.map(|allowable| stress_pa.max(0.0) / allowable);
            if let (Some(current), Some(value)) = (&mut maximum_utilization, utilization) {
                *current = current.max(value);
            }
            states.push(TetherNodeState {
                s_m: node.s_m,
                tension_n,
                stress_pa,
                utilization,
            });
        }

        Ok(StaticTetherResult {
            nodes: states,
            distributed_mass_kg,
            integrated_signed_distributed_force_n,
            integrated_signed_lumped_force_n,
            minimum_tension_n,
            maximum_tension_n,
            maximum_tensile_stress_pa,
            maximum_utilization,
        })
    }

    /// Minimum non-negative boundary tension required so every sampled node is
    /// at zero or positive axial tension for the prescribed profile.
    ///
    /// This is a tension-only boundary helper, not a counterweight optimizer.
    pub fn minimum_tip_tension_for_tension_only(&self) -> Result<f64, TetherError> {
        let mut zero_tip = self.clone();
        zero_tip.tip_tension_n = 0.0;
        let result = zero_tip.solve()?;
        Ok((-result.minimum_tension_n).max(0.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tolerance: f64) {
        assert!(
            (a - b).abs() <= tolerance,
            "{a} !~= {b} (tol {tolerance})"
        );
    }

    fn uniform_model(acceleration: f64, tip_tension_n: f64) -> StaticTetherModel {
        StaticTetherModel {
            density_kg_m3: 1_000.0,
            tip_tension_n,
            allowable_stress_pa: Some(1.0e6),
            nodes: vec![
                TetherNodeInput {
                    s_m: 0.0,
                    effective_acceleration_m_s2: acceleration,
                    area_m2: 0.01,
                    lumped_force_n: 0.0,
                },
                TetherNodeInput {
                    s_m: 5.0,
                    effective_acceleration_m_s2: acceleration,
                    area_m2: 0.01,
                    lumped_force_n: 0.0,
                },
                TetherNodeInput {
                    s_m: 10.0,
                    effective_acceleration_m_s2: acceleration,
                    area_m2: 0.01,
                    lumped_force_n: 0.0,
                },
            ],
        }
    }

    #[test]
    fn uniform_area_and_acceleration_match_closed_form_linear_tension() {
        // lambda = rho*A = 10 kg/m; distributed force = 20 N/m.
        // T(s) = T_tip + 20*(10-s).
        let result = uniform_model(2.0, 100.0).solve().unwrap();
        close(result.nodes[0].tension_n, 300.0, 1.0e-12);
        close(result.nodes[1].tension_n, 200.0, 1.0e-12);
        close(result.nodes[2].tension_n, 100.0, 1.0e-12);
        close(result.distributed_mass_kg, 100.0, 1.0e-12);
        close(result.integrated_signed_distributed_force_n, 200.0, 1.0e-12);
        close(result.maximum_tensile_stress_pa, 30_000.0, 1.0e-9);
        assert!(result.remains_in_tension());
    }

    #[test]
    fn zero_effective_acceleration_produces_constant_tension() {
        let result = uniform_model(0.0, 123.0).solve().unwrap();
        for state in result.nodes {
            close(state.tension_n, 123.0, 1.0e-12);
        }
    }

    #[test]
    fn minimum_tip_tension_prevents_compression_for_negative_body_force() {
        let model = uniform_model(-2.0, 0.0);
        let required = model.minimum_tip_tension_for_tension_only().unwrap();
        close(required, 200.0, 1.0e-12);

        let mut tensioned = model;
        tensioned.tip_tension_n = required;
        let result = tensioned.solve().unwrap();
        close(result.minimum_tension_n, 0.0, 1.0e-12);
        assert!(result.remains_in_tension());
    }

    #[test]
    fn internal_lumped_force_creates_exact_tension_jump() {
        let mut model = uniform_model(0.0, 100.0);
        model.nodes[1].lumped_force_n = 50.0;
        let result = model.solve().unwrap();
        close(result.nodes[2].tension_n, 100.0, 1.0e-12);
        close(result.nodes[1].tension_n, 150.0, 1.0e-12);
        close(result.nodes[0].tension_n, 150.0, 1.0e-12);
        close(result.integrated_signed_lumped_force_n, 50.0, 1.0e-12);
    }

    #[test]
    fn sign_change_can_create_internal_tension_extremum() {
        let model = StaticTetherModel {
            density_kg_m3: 1_000.0,
            tip_tension_n: 100.0,
            allowable_stress_pa: None,
            nodes: vec![
                TetherNodeInput {
                    s_m: 0.0,
                    effective_acceleration_m_s2: 2.0,
                    area_m2: 0.01,
                    lumped_force_n: 0.0,
                },
                TetherNodeInput {
                    s_m: 5.0,
                    effective_acceleration_m_s2: 0.0,
                    area_m2: 0.01,
                    lumped_force_n: 0.0,
                },
                TetherNodeInput {
                    s_m: 10.0,
                    effective_acceleration_m_s2: -2.0,
                    area_m2: 0.01,
                    lumped_force_n: 0.0,
                },
            ],
        };
        let result = model.solve().unwrap();
        // Each half contributes +/-50 N, so the center is the minimum here.
        close(result.nodes[0].tension_n, 100.0, 1.0e-12);
        close(result.nodes[1].tension_n, 50.0, 1.0e-12);
        close(result.nodes[2].tension_n, 100.0, 1.0e-12);
        close(result.minimum_tension_n, 50.0, 1.0e-12);
    }

    #[test]
    fn mesh_refinement_preserves_uniform_closed_form_solution() {
        let coarse = uniform_model(2.0, 100.0).solve().unwrap();
        let fine = StaticTetherModel {
            density_kg_m3: 1_000.0,
            tip_tension_n: 100.0,
            allowable_stress_pa: Some(1.0e6),
            nodes: (0..=10)
                .map(|i| TetherNodeInput {
                    s_m: i as f64,
                    effective_acceleration_m_s2: 2.0,
                    area_m2: 0.01,
                    lumped_force_n: 0.0,
                })
                .collect(),
        }
        .solve()
        .unwrap();
        close(coarse.nodes[0].tension_n, fine.nodes[0].tension_n, 1.0e-12);
        close(coarse.distributed_mass_kg, fine.distributed_mass_kg, 1.0e-12);
    }

    #[test]
    fn malformed_inputs_fail_closed() {
        let mut model = uniform_model(1.0, 0.0);
        model.nodes[1].s_m = model.nodes[0].s_m;
        assert_eq!(model.solve(), Err(TetherError::NonIncreasingPosition));

        let mut model = uniform_model(1.0, 0.0);
        model.nodes[1].area_m2 = 0.0;
        assert_eq!(model.solve(), Err(TetherError::NonPositiveArea));

        let mut model = uniform_model(1.0, 0.0);
        model.nodes[1].effective_acceleration_m_s2 = f64::NAN;
        assert_eq!(model.solve(), Err(TetherError::NonFiniteInput));
    }
}
