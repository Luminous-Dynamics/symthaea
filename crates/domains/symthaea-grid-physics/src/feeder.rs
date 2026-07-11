// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Radial distribution feeder power flow via the linearized DistFlow
//! equations (Baran & Wu, "Optimal sizing of capacitors placed on a radial
//! distribution system", IEEE Trans. Power Delivery, 1989).
//!
//! Scope, stated honestly (matching the project's narrow-but-real pattern
//! for physics models — see e.g. symthaea-auv's drag/buoyancy scope note):
//! single-phase equivalent, balanced, and the quadratic branch-loss term is
//! dropped (the "linearized" part of LinDistFlow) — this is the standard
//! simplification used for fast voltage-profile estimation in distribution
//! planning studies, not a full nonlinear AC power-flow solver. Real branch
//! losses are estimated separately, post-hoc, from the linear solution as a
//! diagnostic (`FeederSolution::estimate_branch_loss_kw`), not fed back into
//! the voltage solve.

use serde::{Deserialize, Serialize};

/// Line (branch) electrical parameters, in ohms.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Line {
    pub resistance_ohm: f64,
    pub reactance_ohm: f64,
}

/// A single bus in the radial feeder tree.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Node {
    /// Index of the parent bus. `None` only for the root (substation/slack bus).
    pub parent: Option<usize>,
    /// Line from `parent` to this node. Ignored (may be zero) for the root.
    pub line_to_parent: Line,
    /// Net real power consumed at this bus (kW). Negative = net injection
    /// (e.g. distributed generation/DER exceeding local load).
    pub load_kw: f64,
    /// Net reactive power consumed at this bus (kVAR). Negative = net
    /// reactive injection (e.g. capacitor bank, inverter VAR support).
    pub load_kvar: f64,
}

impl Node {
    pub fn root() -> Self {
        Self {
            parent: None,
            line_to_parent: Line {
                resistance_ohm: 0.0,
                reactance_ohm: 0.0,
            },
            load_kw: 0.0,
            load_kvar: 0.0,
        }
    }

    pub fn load(parent: usize, line_to_parent: Line, load_kw: f64, load_kvar: f64) -> Self {
        Self {
            parent: Some(parent),
            line_to_parent,
            load_kw,
            load_kvar,
        }
    }
}

/// A radial distribution feeder: a tree of buses rooted at the substation.
///
/// Precondition (validated by `new`): `nodes[0]` is the root (`parent ==
/// None`); every other node's `parent` index is strictly less than its own
/// index (i.e. nodes are given in topological, parent-before-child order).
/// This lets both sweep passes run in a single linear scan with no
/// iteration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Feeder {
    /// Substation (slack bus) voltage magnitude, volts (line-to-neutral).
    pub base_voltage_v: f64,
    pub nodes: Vec<Node>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FeederError {
    EmptyTopology,
    RootMustHaveNoParent,
    NonRootMustHaveParent { index: usize },
    ParentMustPrecedeChild { index: usize, parent: usize },
    NonFiniteBaseVoltage,
}

/// Solved bus voltages and branch power flows for a `Feeder`.
#[derive(Debug, Clone)]
pub struct FeederSolution {
    base_voltage_v: f64,
    /// Voltage magnitude squared at each bus (volts^2).
    voltage_sq_v2: Vec<f64>,
    /// Cumulative real power flowing on the branch feeding this bus (this
    /// bus's own load plus all downstream descendants), kW.
    pub branch_power_kw: Vec<f64>,
    /// Cumulative reactive power flowing on the branch feeding this bus, kVAR.
    pub branch_power_kvar: Vec<f64>,
}

impl Feeder {
    pub fn new(base_voltage_v: f64, nodes: Vec<Node>) -> Result<Self, FeederError> {
        if nodes.is_empty() {
            return Err(FeederError::EmptyTopology);
        }
        if !base_voltage_v.is_finite() || base_voltage_v <= 0.0 {
            return Err(FeederError::NonFiniteBaseVoltage);
        }
        if nodes[0].parent.is_some() {
            return Err(FeederError::RootMustHaveNoParent);
        }
        for (i, node) in nodes.iter().enumerate().skip(1) {
            match node.parent {
                None => return Err(FeederError::NonRootMustHaveParent { index: i }),
                Some(p) if p >= i => {
                    return Err(FeederError::ParentMustPrecedeChild {
                        index: i,
                        parent: p,
                    });
                }
                Some(_) => {}
            }
        }
        Ok(Self {
            base_voltage_v,
            nodes,
        })
    }

    /// Solve for bus voltages via one backward (power aggregation) sweep and
    /// one forward (voltage) sweep — exact for the linearized DistFlow model
    /// (no iteration needed since the quadratic loss term is dropped).
    pub fn solve(&self) -> FeederSolution {
        let n = self.nodes.len();
        let mut branch_power_kw: Vec<f64> = self.nodes.iter().map(|node| node.load_kw).collect();
        let mut branch_power_kvar: Vec<f64> =
            self.nodes.iter().map(|node| node.load_kvar).collect();

        // Backward sweep: fold each node's cumulative branch power into its
        // parent. Safe in reverse order because every child has a strictly
        // higher index than its parent (validated in `new`).
        for i in (1..n).rev() {
            let parent = self.nodes[i].parent.expect("validated non-root");
            branch_power_kw[parent] += branch_power_kw[i];
            branch_power_kvar[parent] += branch_power_kvar[i];
        }

        // Forward sweep: v_child^2 = v_parent^2 - 2*(R*P_watts + X*Q_var).
        let mut voltage_sq_v2 = vec![0.0; n];
        voltage_sq_v2[0] = self.base_voltage_v * self.base_voltage_v;
        for i in 1..n {
            let parent = self.nodes[i].parent.expect("validated non-root");
            let line = self.nodes[i].line_to_parent;
            let p_watts = branch_power_kw[i] * 1000.0;
            let q_var = branch_power_kvar[i] * 1000.0;
            voltage_sq_v2[i] = voltage_sq_v2[parent]
                - 2.0 * (line.resistance_ohm * p_watts + line.reactance_ohm * q_var);
        }

        FeederSolution {
            base_voltage_v: self.base_voltage_v,
            voltage_sq_v2,
            branch_power_kw,
            branch_power_kvar,
        }
    }
}

impl FeederSolution {
    /// Voltage magnitude at bus `i`, volts. Panics (via indexing) if `i` is
    /// out of range — same contract as `Vec` indexing.
    pub fn voltage_v(&self, i: usize) -> f64 {
        self.voltage_sq_v2[i].max(0.0).sqrt()
    }

    /// Voltage in per-unit of the substation base voltage. IEEE 1547 trip
    /// thresholds are conventionally expressed in this unit.
    pub fn voltage_pu(&self, i: usize) -> f64 {
        self.voltage_v(i) / self.base_voltage_v
    }

    /// Approximate real-power loss on the branch feeding bus `i` (kW),
    /// estimated post-hoc from the linear solution as `I^2 * R` with
    /// `I ~= S_branch / V_child` (S = apparent power magnitude). This is a
    /// diagnostic — it is NOT fed back into the (deliberately linearized)
    /// voltage solve.
    pub fn estimate_branch_loss_kw(&self, feeder: &Feeder, i: usize) -> f64 {
        if i == 0 {
            return 0.0;
        }
        let v = self.voltage_v(i);
        if v <= 0.0 {
            return 0.0;
        }
        let p_kw = self.branch_power_kw[i];
        let q_kvar = self.branch_power_kvar[i];
        let s_va = ((p_kw * 1000.0).powi(2) + (q_kvar * 1000.0).powi(2)).sqrt();
        let current_a = s_va / v;
        let r_ohm = feeder.nodes[i].line_to_parent.resistance_ohm;
        current_a.powi(2) * r_ohm / 1000.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_two_node_real_power_only_voltage_drop() {
        // root --(1 ohm, 0 ohm)--> child, child load = 100 kW, unity PF.
        let feeder = Feeder::new(
            7200.0,
            vec![
                Node::root(),
                Node::load(
                    0,
                    Line {
                        resistance_ohm: 1.0,
                        reactance_ohm: 0.0,
                    },
                    100.0,
                    0.0,
                ),
            ],
        )
        .unwrap();
        let solution = feeder.solve();
        // v_child^2 = 7200^2 - 2*(1 * 100_000) = 51_840_000 - 200_000 = 51_640_000
        let expected_v_sq = 7200.0_f64.powi(2) - 200_000.0;
        assert!((solution.voltage_v(1).powi(2) - expected_v_sq).abs() < 1e-6);
        assert_eq!(solution.branch_power_kw[1], 100.0);
    }

    #[test]
    fn test_reactive_power_only_voltage_drop() {
        // root --(0 ohm, 1 ohm)--> child, child load = 100 kVAR, zero real power.
        let feeder = Feeder::new(
            7200.0,
            vec![
                Node::root(),
                Node::load(
                    0,
                    Line {
                        resistance_ohm: 0.0,
                        reactance_ohm: 1.0,
                    },
                    0.0,
                    100.0,
                ),
            ],
        )
        .unwrap();
        let solution = feeder.solve();
        let expected_v_sq = 7200.0_f64.powi(2) - 200_000.0;
        assert!((solution.voltage_v(1).powi(2) - expected_v_sq).abs() < 1e-6);
    }

    #[test]
    fn test_pass_through_node_aggregates_downstream_power() {
        // root -> A (no load, 0.5 ohm) -> B (50 kW load, 0.5 ohm)
        let feeder = Feeder::new(
            12_000.0,
            vec![
                Node::root(),
                Node::load(
                    0,
                    Line {
                        resistance_ohm: 0.5,
                        reactance_ohm: 0.0,
                    },
                    0.0,
                    0.0,
                ),
                Node::load(
                    1,
                    Line {
                        resistance_ohm: 0.5,
                        reactance_ohm: 0.0,
                    },
                    50.0,
                    0.0,
                ),
            ],
        )
        .unwrap();
        let solution = feeder.solve();
        // A carries B's 50kW even though A itself has no load.
        assert_eq!(solution.branch_power_kw[1], 50.0);
        assert_eq!(solution.branch_power_kw[2], 50.0);

        let v_a_sq = 12_000.0_f64.powi(2) - 2.0 * 0.5 * 50_000.0;
        assert!((solution.voltage_v(1).powi(2) - v_a_sq).abs() < 1e-6);
        let v_b_sq = v_a_sq - 2.0 * 0.5 * 50_000.0;
        assert!((solution.voltage_v(2).powi(2) - v_b_sq).abs() < 1e-6);
    }

    #[test]
    fn test_der_injection_causes_voltage_rise_not_drop() {
        // Same topology/impedance as the real-power-drop test, but load_kw
        // is negative (net generation exceeds local load) — a real DER
        // overvoltage scenario, not just a sign flip for its own sake.
        let feeder = Feeder::new(
            7200.0,
            vec![
                Node::root(),
                Node::load(
                    0,
                    Line {
                        resistance_ohm: 1.0,
                        reactance_ohm: 0.0,
                    },
                    -100.0,
                    0.0,
                ),
            ],
        )
        .unwrap();
        let solution = feeder.solve();
        assert!(
            solution.voltage_v(1) > 7200.0,
            "DER injection should raise downstream voltage above substation voltage, got {}",
            solution.voltage_v(1)
        );
    }

    #[test]
    fn test_root_voltage_is_exactly_base_and_one_pu() {
        let feeder = Feeder::new(7200.0, vec![Node::root()]).unwrap();
        let solution = feeder.solve();
        assert_eq!(solution.voltage_v(0), 7200.0);
        assert_eq!(solution.voltage_pu(0), 1.0);
    }

    #[test]
    fn test_child_index_must_exceed_parent_index() {
        let result = Feeder::new(
            7200.0,
            vec![
                Node::root(),
                Node::load(
                    5, // parent index >= own index (1) -> invalid
                    Line {
                        resistance_ohm: 1.0,
                        reactance_ohm: 0.0,
                    },
                    10.0,
                    0.0,
                ),
            ],
        );
        assert_eq!(
            result,
            Err(FeederError::ParentMustPrecedeChild {
                index: 1,
                parent: 5
            })
        );
    }

    #[test]
    fn test_root_with_parent_rejected() {
        let mut root = Node::root();
        root.parent = Some(0);
        assert_eq!(
            Feeder::new(7200.0, vec![root]),
            Err(FeederError::RootMustHaveNoParent)
        );
    }

    #[test]
    fn test_empty_topology_rejected() {
        assert_eq!(Feeder::new(7200.0, vec![]), Err(FeederError::EmptyTopology));
    }

    #[test]
    fn test_estimated_branch_loss_is_positive_for_loaded_line() {
        let feeder = Feeder::new(
            7200.0,
            vec![
                Node::root(),
                Node::load(
                    0,
                    Line {
                        resistance_ohm: 1.0,
                        reactance_ohm: 0.0,
                    },
                    100.0,
                    0.0,
                ),
            ],
        )
        .unwrap();
        let solution = feeder.solve();
        let loss = solution.estimate_branch_loss_kw(&feeder, 1);
        assert!(loss > 0.0, "loaded line must show nonzero estimated loss");
        // Sanity bound: losses should be a small fraction of the 100kW load
        // for this modest impedance, not comparable to or exceeding it.
        assert!(loss < 5.0, "loss implausibly large: {loss} kW");
    }

    #[test]
    fn test_root_branch_loss_is_zero() {
        let feeder = Feeder::new(7200.0, vec![Node::root()]).unwrap();
        let solution = feeder.solve();
        assert_eq!(solution.estimate_branch_loss_kw(&feeder, 0), 0.0);
    }
}
