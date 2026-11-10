// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Implicit lumped-node thermal network for electric-aircraft subsystem trades.
//!
//! The model solves a backward-Euler thermal RC network:
//!
//! C_i dT_i/dt = Q_i + G_i,amb (T_amb - T_i)
//!                 + sum_j G_ij (T_j - T_i)
//!
//! It supports coupled battery, inverter, motor, enclosure and coolant nodes.
//! A coolant node may be represented with its own heat capacity and ambient
//! rejection conductance. Flow, fluid advection, phase change, radiation,
//! nonlinear material properties, local cell hot spots and thermal runaway
//! are outside this module's scope.
//!
//! Backward Euler is unconditionally stable for the passive linear network,
//! but stability does not imply physical validity or thermal-safety evidence.

use std::collections::BTreeMap;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub struct ThermalNode {
    pub id: String,
    /// Lumped thermal capacitance for this node (J/K).
    pub heat_capacity_j_per_k: f64,
    /// Linear conductance from this node to the declared ambient sink (W/K).
    pub ambient_conductance_w_per_k: f64,
    /// Screening ceiling for this lumped node (K).
    pub maximum_temperature_k: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ThermalLink {
    pub first_node_id: String,
    pub second_node_id: String,
    /// Symmetric thermal conductance between two nodes (W/K).
    pub conductance_w_per_k: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ThermalNetwork {
    pub nodes: Vec<ThermalNode>,
    pub links: Vec<ThermalLink>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ThermalState {
    /// Temperatures are aligned with ThermalNetwork.nodes order.
    pub temperatures_k: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ThermalStepReport {
    pub state: ThermalState,
    pub maximum_temperature_k: f64,
    pub maximum_temperature_node_id: String,
    pub heat_input_j: f64,
    pub ambient_rejected_energy_j: f64,
    pub stored_energy_change_j: f64,
    /// Residual of the network-level energy balance (J).
    pub energy_balance_residual_j: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ThermalNetworkError {
    EmptyNetwork,
    InvalidNode(String),
    DuplicateNodeId(String),
    InvalidLink(String),
    UnknownNodeInLink(String),
    InvalidState(&'static str),
    InvalidStep(&'static str),
    InvalidHeatInput(String),
    SingularSystem,
    TemperatureLimitExceeded {
        node_id: String,
        temperature_k: f64,
        maximum_temperature_k: f64,
    },
}

impl fmt::Display for ThermalNetworkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for ThermalNetworkError {}

impl ThermalNetwork {
    pub fn validate(&self) -> Result<(), ThermalNetworkError> {
        if self.nodes.is_empty() {
            return Err(ThermalNetworkError::EmptyNetwork);
        }
        let mut indices = BTreeMap::new();
        for (index, node) in self.nodes.iter().enumerate() {
            if node.id.trim().is_empty()
                || !node.heat_capacity_j_per_k.is_finite()
                || node.heat_capacity_j_per_k <= 0.0
                || !node.ambient_conductance_w_per_k.is_finite()
                || node.ambient_conductance_w_per_k < 0.0
                || !node.maximum_temperature_k.is_finite()
                || node.maximum_temperature_k <= 0.0
            {
                return Err(ThermalNetworkError::InvalidNode(node.id.clone()));
            }
            if indices.insert(node.id.as_str(), index).is_some() {
                return Err(ThermalNetworkError::DuplicateNodeId(node.id.clone()));
            }
        }
        for link in &self.links {
            if link.first_node_id == link.second_node_id
                || !link.conductance_w_per_k.is_finite()
                || link.conductance_w_per_k <= 0.0
            {
                return Err(ThermalNetworkError::InvalidLink(format!(
                    "{} <-> {}",
                    link.first_node_id, link.second_node_id
                )));
            }
            if !indices.contains_key(link.first_node_id.as_str()) {
                return Err(ThermalNetworkError::UnknownNodeInLink(
                    link.first_node_id.clone(),
                ));
            }
            if !indices.contains_key(link.second_node_id.as_str()) {
                return Err(ThermalNetworkError::UnknownNodeInLink(
                    link.second_node_id.clone(),
                ));
            }
        }
        Ok(())
    }

    /// Advance the network by one implicit time step.
    ///
    /// heat_generation_w must have one finite, nonnegative value per node.
    /// Ambient is a fixed thermal boundary for this step. No node or link is
    /// mutated; a failure returns no partially advanced state.
    pub fn step(
        &self,
        state: &ThermalState,
        heat_generation_w: &[f64],
        dt_s: f64,
        ambient_temperature_k: f64,
    ) -> Result<ThermalStepReport, ThermalNetworkError> {
        self.validate()?;
        let n = self.nodes.len();
        if state.temperatures_k.len() != n {
            return Err(ThermalNetworkError::InvalidState(
                "temperature vector must match network node count",
            ));
        }
        if !dt_s.is_finite() || dt_s <= 0.0
            || !ambient_temperature_k.is_finite() || ambient_temperature_k <= 0.0
        {
            return Err(ThermalNetworkError::InvalidStep(
                "time step and ambient temperature must be finite and positive",
            ));
        }
        if heat_generation_w.len() != n {
            return Err(ThermalNetworkError::InvalidHeatInput(
                "heat vector must match network node count".into(),
            ));
        }
        for (node, temperature) in self.nodes.iter().zip(&state.temperatures_k) {
            if !temperature.is_finite() || *temperature <= 0.0 {
                return Err(ThermalNetworkError::InvalidState(
                    "initial temperatures must be finite and above absolute zero",
                ));
            }
            if *temperature > node.maximum_temperature_k {
                return Err(ThermalNetworkError::TemperatureLimitExceeded {
                    node_id: node.id.clone(),
                    temperature_k: *temperature,
                    maximum_temperature_k: node.maximum_temperature_k,
                });
            }
        }
        for (node, heat) in self.nodes.iter().zip(heat_generation_w) {
            if !heat.is_finite() || *heat < 0.0 {
                return Err(ThermalNetworkError::InvalidHeatInput(format!(
                    "node {} heat must be finite and nonnegative",
                    node.id
                )));
            }
        }

        let indices = self.nodes.iter()
            .enumerate()
            .map(|(index, node)| (node.id.as_str(), index))
            .collect::<BTreeMap<_, _>>();
        let mut matrix = vec![vec![0.0_f64; n]; n];
        let mut rhs = vec![0.0_f64; n];

        for i in 0..n {
            let node = &self.nodes[i];
            let storage_rate = node.heat_capacity_j_per_k / dt_s;
            let ambient_g = node.ambient_conductance_w_per_k;
            matrix[i][i] = storage_rate + ambient_g;
            rhs[i] = storage_rate * state.temperatures_k[i]
                + heat_generation_w[i]
                + ambient_g * ambient_temperature_k;
        }

        for link in &self.links {
            let i = *indices.get(link.first_node_id.as_str())
                .ok_or_else(|| ThermalNetworkError::UnknownNodeInLink(link.first_node_id.clone()))?;
            let j = *indices.get(link.second_node_id.as_str())
                .ok_or_else(|| ThermalNetworkError::UnknownNodeInLink(link.second_node_id.clone()))?;
            let conductance = link.conductance_w_per_k;
            matrix[i][i] += conductance;
            matrix[j][j] += conductance;
            matrix[i][j] -= conductance;
            matrix[j][i] -= conductance;
        }

        let next_temperatures = solve_dense_system(matrix, rhs)
            .ok_or(ThermalNetworkError::SingularSystem)?;

        let mut maximum_temperature_k = f64::NEG_INFINITY;
        let mut maximum_temperature_node_id = String::new();
        for (node, temperature) in self.nodes.iter().zip(&next_temperatures) {
            if !temperature.is_finite() || *temperature <= 0.0 {
                return Err(ThermalNetworkError::InvalidState(
                    "implicit solve produced a non-finite or non-positive temperature",
                ));
            }
            if *temperature > node.maximum_temperature_k {
                return Err(ThermalNetworkError::TemperatureLimitExceeded {
                    node_id: node.id.clone(),
                    temperature_k: *temperature,
                    maximum_temperature_k: node.maximum_temperature_k,
                });
            }
            if *temperature > maximum_temperature_k {
                maximum_temperature_k = *temperature;
                maximum_temperature_node_id = node.id.clone();
            }
        }

        let heat_input_j = heat_generation_w.iter().sum::<f64>() * dt_s;
        let ambient_rejected_energy_j = self.nodes.iter()
            .zip(&next_temperatures)
            .map(|(node, temperature)| {
                node.ambient_conductance_w_per_k
                    * (temperature - ambient_temperature_k)
                    * dt_s
            })
            .sum::<f64>();
        let stored_energy_change_j = self.nodes.iter()
            .zip(state.temperatures_k.iter().zip(&next_temperatures))
            .map(|(node, (old, new))| node.heat_capacity_j_per_k * (new - old))
            .sum::<f64>();
        let energy_balance_residual_j =
            heat_input_j - ambient_rejected_energy_j - stored_energy_change_j;

        Ok(ThermalStepReport {
            state: ThermalState { temperatures_k: next_temperatures },
            maximum_temperature_k,
            maximum_temperature_node_id,
            heat_input_j,
            ambient_rejected_energy_j,
            stored_energy_change_j,
            energy_balance_residual_j,
        })
    }
}

/// Deterministic Gaussian elimination with partial pivoting.
fn solve_dense_system(mut matrix: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Option<Vec<f64>> {
    let n = rhs.len();
    if matrix.len() != n || matrix.iter().any(|row| row.len() != n) {
        return None;
    }
    for pivot in 0..n {
        let mut best_row = pivot;
        let mut best_value = matrix[pivot][pivot].abs();
        for row in (pivot + 1)..n {
            let candidate = matrix[row][pivot].abs();
            if candidate > best_value {
                best_value = candidate;
                best_row = row;
            }
        }
        if !best_value.is_finite() || best_value <= f64::MIN_POSITIVE {
            return None;
        }
        if best_row != pivot {
            matrix.swap(best_row, pivot);
            rhs.swap(best_row, pivot);
        }

        let pivot_value = matrix[pivot][pivot];
        for row in (pivot + 1)..n {
            let factor = matrix[row][pivot] / pivot_value;
            matrix[row][pivot] = 0.0;
            for column in (pivot + 1)..n {
                matrix[row][column] -= factor * matrix[pivot][column];
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }

    let mut solution = vec![0.0; n];
    for row in (0..n).rev() {
        let mut residual = rhs[row];
        for column in (row + 1)..n {
            residual -= matrix[row][column] * solution[column];
        }
        let diagonal = matrix[row][row];
        if !diagonal.is_finite() || diagonal.abs() <= f64::MIN_POSITIVE {
            return None;
        }
        solution[row] = residual / diagonal;
        if !solution[row].is_finite() {
            return None;
        }
    }
    Some(solution)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one_node_network() -> ThermalNetwork {
        ThermalNetwork {
            nodes: vec![ThermalNode {
                id: "battery".into(),
                heat_capacity_j_per_k: 100.0,
                ambient_conductance_w_per_k: 10.0,
                maximum_temperature_k: 400.0,
            }],
            links: Vec::new(),
        }
    }

    #[test]
    fn backward_euler_matches_single_node_analytic_solution() {
        let network = one_node_network();
        let state = ThermalState { temperatures_k: vec![293.0] };
        let report = network.step(&state, &[20.0], 10.0, 293.0).unwrap();
        assert!((report.state.temperatures_k[0] - 294.0).abs() < 1e-12);
    }

    #[test]
    fn coupled_nodes_transfer_heat_and_conserve_energy() {
        let network = ThermalNetwork {
            nodes: vec![
                ThermalNode {
                    id: "battery".into(),
                    heat_capacity_j_per_k: 100.0,
                    ambient_conductance_w_per_k: 0.0,
                    maximum_temperature_k: 400.0,
                },
                ThermalNode {
                    id: "coolant".into(),
                    heat_capacity_j_per_k: 200.0,
                    ambient_conductance_w_per_k: 0.0,
                    maximum_temperature_k: 400.0,
                },
            ],
            links: vec![ThermalLink {
                first_node_id: "battery".into(),
                second_node_id: "coolant".into(),
                conductance_w_per_k: 5.0,
            }],
        };
        let state = ThermalState { temperatures_k: vec![320.0, 290.0] };
        let report = network.step(&state, &[0.0, 0.0], 10.0, 293.0).unwrap();
        assert!(report.state.temperatures_k[0] < 320.0);
        assert!(report.state.temperatures_k[1] > 290.0);
        assert!(report.energy_balance_residual_j.abs() < 1e-9);
    }

    #[test]
    fn ambient_rejection_balances_heat_input_and_stored_energy() {
        let network = one_node_network();
        let state = ThermalState { temperatures_k: vec![293.0] };
        let report = network.step(&state, &[20.0], 10.0, 293.0).unwrap();
        assert!((report.heat_input_j
            - report.ambient_rejected_energy_j
            - report.stored_energy_change_j).abs() < 1e-9);
    }

    #[test]
    fn thermal_limit_failure_is_explicit_and_does_not_return_state() {
        let mut network = one_node_network();
        network.nodes[0].maximum_temperature_k = 294.5;
        let state = ThermalState { temperatures_k: vec![293.0] };
        assert!(matches!(
            network.step(&state, &[100.0], 10.0, 293.0),
            Err(ThermalNetworkError::TemperatureLimitExceeded { node_id, .. })
                if node_id == "battery"
        ));
    }

    #[test]
    fn invalid_node_links_and_heat_inputs_fail_closed() {
        let mut network = one_node_network();
        network.links.push(ThermalLink {
            first_node_id: "battery".into(),
            second_node_id: "missing".into(),
            conductance_w_per_k: 1.0,
        });
        assert!(matches!(
            network.validate(),
            Err(ThermalNetworkError::UnknownNodeInLink(_))
        ));

        let network = one_node_network();
        let state = ThermalState { temperatures_k: vec![293.0] };
        assert!(matches!(
            network.step(&state, &[-1.0], 1.0, 293.0),
            Err(ThermalNetworkError::InvalidHeatInput(_))
        ));
    }

    #[test]
    fn isolated_adiabatic_node_conserves_temperature_without_heat() {
        let network = ThermalNetwork {
            nodes: vec![ThermalNode {
                id: "motor".into(),
                heat_capacity_j_per_k: 1_000.0,
                ambient_conductance_w_per_k: 0.0,
                maximum_temperature_k: 500.0,
            }],
            links: Vec::new(),
        };
        let state = ThermalState { temperatures_k: vec![350.0] };
        let report = network.step(&state, &[0.0], 60.0, 293.0).unwrap();
        assert!((report.state.temperatures_k[0] - 350.0).abs() < 1e-12);
    }
}
