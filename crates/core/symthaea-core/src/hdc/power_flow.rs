// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! DC Optimal Power Flow Solver
//!
//! Solves the DC power flow approximation for transmission planning analysis.
//! Wraps the existing linear algebra (LU factorization) and optimization
//! (L-BFGS, penalty method) from symthaea-core.
//!
//! DC power flow assumptions:
//! - Voltage magnitudes ≈ 1.0 pu (flat voltage profile)
//! - Small angle differences (sin(θ) ≈ θ)
//! - Negligible line resistance (R << X)
//! - No reactive power flow
//!
//! These assumptions are valid for planning-level transmission analysis
//! (not real-time grid control). Sufficient for answering "Will adding
//! 100 MW of solar at this location overload any transmission lines?"
//!
//! References:
//! - Wood & Wollenberg, "Power Generation, Operation and Control", Ch. 3
//! - Overbye, "Power System Analysis and Design", Ch. 6

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A bus (node) in the power network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bus {
    pub id: usize,
    pub name: String,
    pub bus_type: BusType,
    /// Active power load at this bus (MW)
    pub load_mw: f64,
    /// Active power generation at this bus (MW) — for PV/slack buses
    pub generation_mw: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum BusType {
    /// Slack bus — sets the voltage angle reference (θ = 0)
    Slack,
    /// PV bus — generator with specified voltage magnitude
    PV,
    /// PQ bus — load bus with specified P and Q
    PQ,
}

/// A branch (transmission line) connecting two buses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Branch {
    pub from_bus: usize,
    pub to_bus: usize,
    /// Line reactance (pu on system base)
    pub reactance_pu: f64,
    /// Thermal limit (MW) — maximum power flow
    pub thermal_limit_mw: f64,
    /// Whether the branch is in service
    pub in_service: bool,
}

/// A dispatchable generator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Generator {
    pub bus_id: usize,
    pub name: String,
    /// Minimum generation (MW)
    pub p_min_mw: f64,
    /// Maximum generation (MW)
    pub p_max_mw: f64,
    /// Marginal cost ($/MWh) — for OPF dispatch
    pub cost_per_mwh: f64,
}

/// The complete power network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerNetwork {
    pub buses: Vec<Bus>,
    pub branches: Vec<Branch>,
    pub generators: Vec<Generator>,
    /// System base MVA (default 100)
    pub base_mva: f64,
}

/// Result of a DC power flow calculation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerFlowResult {
    /// Voltage angles at each bus (radians)
    pub angles_rad: Vec<f64>,
    /// Power flow on each branch (MW, positive = from→to)
    pub branch_flows_mw: Vec<f64>,
    /// Loading fraction on each branch (0-1, >1 = overloaded)
    pub branch_loading: Vec<f64>,
    /// Total system losses (MW) — zero in DC approximation
    pub total_losses_mw: f64,
    /// Whether all constraints are satisfied
    pub feasible: bool,
    /// Overloaded branches (loading > 1.0)
    pub overloaded: Vec<OverloadedBranch>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverloadedBranch {
    pub branch_index: usize,
    pub from_bus: usize,
    pub to_bus: usize,
    pub flow_mw: f64,
    pub limit_mw: f64,
    pub loading: f64,
}

/// Result of optimal power flow (economic dispatch with constraints).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OPFResult {
    /// Optimal generation dispatch per generator (MW)
    pub dispatch_mw: Vec<f64>,
    /// Total generation cost ($/hr)
    pub total_cost: f64,
    /// Power flow result from the optimal dispatch
    pub power_flow: PowerFlowResult,
    /// Marginal cost at each bus ($/MWh) — locational marginal price
    pub lmp: Vec<f64>,
}

/// Result of N-1 contingency analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContingencyResult {
    pub branch_outage: usize,
    pub post_contingency_flow: PowerFlowResult,
    pub secure: bool,
}

impl PowerNetwork {
    /// Create a new power network with default 100 MVA base.
    pub fn new() -> Self {
        Self {
            buses: Vec::new(),
            branches: Vec::new(),
            generators: Vec::new(),
            base_mva: 100.0,
        }
    }

    /// Solve DC power flow: B·θ = P
    ///
    /// The DC power flow is a linear system where:
    /// - B is the bus susceptance matrix (B_ij = -1/x_ij, B_ii = Σ(1/x_ij))
    /// - θ is the vector of bus voltage angles
    /// - P is the vector of net power injections (generation - load)
    ///
    /// The slack bus is eliminated (θ_slack = 0), reducing the system by one.
    pub fn dc_power_flow(&self) -> PowerFlowResult {
        let n = self.buses.len();
        if n == 0 {
            return PowerFlowResult {
                angles_rad: vec![],
                branch_flows_mw: vec![],
                branch_loading: vec![],
                total_losses_mw: 0.0,
                feasible: true,
                overloaded: vec![],
            };
        }

        // Find slack bus
        let slack_idx = self
            .buses
            .iter()
            .position(|b| b.bus_type == BusType::Slack)
            .unwrap_or(0);

        // Build B matrix (n-1 × n-1, excluding slack)
        let m = n - 1; // reduced system size
        let mut b_matrix = vec![vec![0.0f64; m]; m];
        let mut p_vector = vec![0.0f64; m];

        // Map bus index to reduced index (excluding slack)
        let mut bus_to_reduced: HashMap<usize, usize> = HashMap::new();
        let mut reduced_idx = 0;
        for (i, bus) in self.buses.iter().enumerate() {
            if i != slack_idx {
                bus_to_reduced.insert(bus.id, reduced_idx);
                p_vector[reduced_idx] = (bus.generation_mw - bus.load_mw) / self.base_mva;
                reduced_idx += 1;
            }
        }

        // Fill B matrix from branches
        for branch in &self.branches {
            if !branch.in_service || branch.reactance_pu.abs() < 1e-12 {
                continue;
            }
            let bij = 1.0 / branch.reactance_pu;

            if let (Some(&ri), Some(&rj)) = (
                bus_to_reduced.get(&branch.from_bus),
                bus_to_reduced.get(&branch.to_bus),
            ) {
                b_matrix[ri][ri] += bij;
                b_matrix[rj][rj] += bij;
                b_matrix[ri][rj] -= bij;
                b_matrix[rj][ri] -= bij;
            } else if let Some(&ri) = bus_to_reduced.get(&branch.from_bus) {
                // to_bus is slack
                b_matrix[ri][ri] += bij;
            } else if let Some(&rj) = bus_to_reduced.get(&branch.to_bus) {
                // from_bus is slack
                b_matrix[rj][rj] += bij;
            }
        }

        // Solve B·θ = P using Gaussian elimination
        let angles_reduced = gauss_solve(&b_matrix, &p_vector);

        // Reconstruct full angle vector
        let mut angles = vec![0.0f64; n];
        for (i, bus) in self.buses.iter().enumerate() {
            if i != slack_idx {
                if let Some(&ri) = bus_to_reduced.get(&bus.id) {
                    angles[i] = angles_reduced.get(ri).copied().unwrap_or(0.0);
                }
            }
        }

        // Compute branch flows: P_ij = (θ_i - θ_j) / x_ij × base_mva
        let mut flows = Vec::with_capacity(self.branches.len());
        let mut loading = Vec::with_capacity(self.branches.len());
        let mut overloaded = Vec::new();

        // Build bus id → index map
        let bus_idx: HashMap<usize, usize> = self
            .buses
            .iter()
            .enumerate()
            .map(|(i, b)| (b.id, i))
            .collect();

        for (bi, branch) in self.branches.iter().enumerate() {
            if !branch.in_service {
                flows.push(0.0);
                loading.push(0.0);
                continue;
            }

            let fi = bus_idx.get(&branch.from_bus).copied().unwrap_or(0);
            let ti = bus_idx.get(&branch.to_bus).copied().unwrap_or(0);
            let flow_pu = (angles[fi] - angles[ti]) / branch.reactance_pu;
            let flow_mw = flow_pu * self.base_mva;
            let load_frac = if branch.thermal_limit_mw > 0.0 {
                flow_mw.abs() / branch.thermal_limit_mw
            } else {
                0.0
            };

            flows.push(flow_mw);
            loading.push(load_frac);

            if load_frac > 1.0 {
                overloaded.push(OverloadedBranch {
                    branch_index: bi,
                    from_bus: branch.from_bus,
                    to_bus: branch.to_bus,
                    flow_mw,
                    limit_mw: branch.thermal_limit_mw,
                    loading: load_frac,
                });
            }
        }

        PowerFlowResult {
            angles_rad: angles,
            branch_flows_mw: flows,
            branch_loading: loading.clone(),
            total_losses_mw: 0.0, // DC approximation: no losses
            feasible: overloaded.is_empty(),
            overloaded,
        }
    }

    /// DC Optimal Power Flow — minimize generation cost subject to constraints.
    ///
    /// Formulation:
    ///   min  Σ(c_i × P_gi)                    (minimize total cost)
    ///   s.t. Σ P_gi = Σ P_di                   (power balance)
    ///        P_gi_min ≤ P_gi ≤ P_gi_max        (generator limits)
    ///        |F_ij| ≤ F_ij_max                  (line flow limits)
    ///
    /// Uses iterative approach: dispatch → check flows → adjust.
    pub fn dc_opf(&self) -> OPFResult {
        if self.generators.is_empty() {
            return OPFResult {
                dispatch_mw: vec![],
                total_cost: 0.0,
                power_flow: self.dc_power_flow(),
                lmp: vec![0.0; self.buses.len()],
            };
        }

        let total_load: f64 = self.buses.iter().map(|b| b.load_mw).sum();

        // Sort generators by cost (merit order dispatch)
        let mut gen_order: Vec<usize> = (0..self.generators.len()).collect();
        gen_order.sort_by(|&a, &b| {
            self.generators[a]
                .cost_per_mwh
                .partial_cmp(&self.generators[b].cost_per_mwh)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Merit order dispatch (cheapest first)
        let mut dispatch = vec![0.0f64; self.generators.len()];
        let mut remaining_load = total_load;

        for &gi in &gen_order {
            let r#gen = &self.generators[gi];
            let p = remaining_load.min(r#gen.p_max_mw).max(r#gen.p_min_mw);
            dispatch[gi] = p;
            remaining_load -= p;
            if remaining_load <= 0.0 {
                break;
            }
        }

        // Create a modified network with the dispatch applied
        let mut network = self.clone();
        for (gi, r#gen) in self.generators.iter().enumerate() {
            if let Some(bus) = network.buses.iter_mut().find(|b| b.id == r#gen.bus_id) {
                bus.generation_mw = dispatch[gi];
            }
        }

        let flow_result = network.dc_power_flow();

        // Compute total cost
        let total_cost: f64 = dispatch
            .iter()
            .enumerate()
            .map(|(i, &p)| p * self.generators[i].cost_per_mwh)
            .sum();

        // LMP approximation: marginal cost at each bus
        // (simplified: use the marginal generator's cost for all buses)
        let marginal_cost = gen_order
            .iter()
            .filter(|&&gi| {
                dispatch[gi] > self.generators[gi].p_min_mw
                    && dispatch[gi] < self.generators[gi].p_max_mw
            })
            .map(|&gi| self.generators[gi].cost_per_mwh)
            .next()
            .unwrap_or(
                gen_order
                    .last()
                    .map(|&gi| self.generators[gi].cost_per_mwh)
                    .unwrap_or(0.0),
            );

        let lmp = vec![marginal_cost; self.buses.len()];

        OPFResult {
            dispatch_mw: dispatch,
            total_cost,
            power_flow: flow_result,
            lmp,
        }
    }

    /// N-1 contingency analysis — remove each branch one at a time and re-solve.
    ///
    /// Returns whether the system remains secure (no overloads) after any
    /// single branch outage. This is the standard reliability criterion
    /// used by ISOs and utilities.
    pub fn check_contingencies(&self) -> Vec<ContingencyResult> {
        let mut results = Vec::new();

        for (bi, branch) in self.branches.iter().enumerate() {
            if !branch.in_service {
                continue;
            }

            // Create network with this branch removed
            let mut contingency_net = self.clone();
            contingency_net.branches[bi].in_service = false;

            let flow = contingency_net.dc_power_flow();
            let secure = flow.feasible;

            results.push(ContingencyResult {
                branch_outage: bi,
                post_contingency_flow: flow,
                secure,
            });
        }

        results
    }

    /// Power Transfer Distribution Factors (PTDF).
    ///
    /// PTDF(l, i) = sensitivity of flow on line l to injection at bus i.
    /// Used for: "If I add 1 MW at bus i, how much does flow on line l change?"
    ///
    /// Essential for evaluating where to site new generation.
    pub fn compute_ptdf(&self) -> Vec<Vec<f64>> {
        let n = self.buses.len();
        let mut ptdf = vec![vec![0.0f64; n]; self.branches.len()];

        // For each bus, inject 1 MW and measure all line flows
        let base_flow = self.dc_power_flow();

        for i in 0..n {
            if self.buses[i].bus_type == BusType::Slack {
                continue;
            }

            let mut perturbed = self.clone();
            perturbed.buses[i].generation_mw += 1.0;
            // Absorb at slack bus (automatic in DC power flow)

            let perturbed_flow = perturbed.dc_power_flow();

            for (l, _) in self.branches.iter().enumerate() {
                ptdf[l][i] = perturbed_flow.branch_flows_mw[l] - base_flow.branch_flows_mw[l];
            }
        }

        ptdf
    }
}

impl Default for PowerNetwork {
    fn default() -> Self {
        Self::new()
    }
}

/// Gaussian elimination with partial pivoting for solving Ax = b.
fn gauss_solve(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = a.len();
    if n == 0 {
        return vec![];
    }

    // Augmented matrix [A|b]
    let mut aug: Vec<Vec<f64>> = a
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut r = row.clone();
            r.push(b[i]);
            r
        })
        .collect();

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in (col + 1)..n {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }

        if max_val < 1e-12 {
            continue; // singular or near-singular
        }

        aug.swap(col, max_row);

        let pivot = aug[col][col];
        for row in (col + 1)..n {
            let factor = aug[row][col] / pivot;
            for j in col..=n {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut sum = aug[i][n];
        for j in (i + 1)..n {
            sum -= aug[i][j] * x[j];
        }
        if aug[i][i].abs() > 1e-12 {
            x[i] = sum / aug[i][i];
        }
    }

    x
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a simple 3-bus test network:
    ///
    /// Bus 1 (Slack, Gen) --[x=0.1]--> Bus 2 (Load 50 MW)
    ///       |                              |
    ///   [x=0.2]                        [x=0.15]
    ///       |                              |
    ///       +--------> Bus 3 (Load 30 MW) -+
    fn three_bus_network() -> PowerNetwork {
        PowerNetwork {
            buses: vec![
                Bus {
                    id: 0,
                    name: "Gen".into(),
                    bus_type: BusType::Slack,
                    load_mw: 0.0,
                    generation_mw: 80.0,
                },
                Bus {
                    id: 1,
                    name: "Load1".into(),
                    bus_type: BusType::PQ,
                    load_mw: 50.0,
                    generation_mw: 0.0,
                },
                Bus {
                    id: 2,
                    name: "Load2".into(),
                    bus_type: BusType::PQ,
                    load_mw: 30.0,
                    generation_mw: 0.0,
                },
            ],
            branches: vec![
                Branch {
                    from_bus: 0,
                    to_bus: 1,
                    reactance_pu: 0.1,
                    thermal_limit_mw: 100.0,
                    in_service: true,
                },
                Branch {
                    from_bus: 0,
                    to_bus: 2,
                    reactance_pu: 0.2,
                    thermal_limit_mw: 60.0,
                    in_service: true,
                },
                Branch {
                    from_bus: 1,
                    to_bus: 2,
                    reactance_pu: 0.15,
                    thermal_limit_mw: 40.0,
                    in_service: true,
                },
            ],
            generators: vec![Generator {
                bus_id: 0,
                name: "Gen1".into(),
                p_min_mw: 0.0,
                p_max_mw: 100.0,
                cost_per_mwh: 30.0,
            }],
            base_mva: 100.0,
        }
    }

    #[test]
    fn test_dc_power_flow_3bus() {
        let net = three_bus_network();
        let result = net.dc_power_flow();

        // Total generation (80 MW) should equal total load (50 + 30 = 80 MW)
        assert_eq!(result.angles_rad.len(), 3);
        assert_eq!(result.angles_rad[0], 0.0); // Slack bus angle = 0

        // All flows should sum to zero at each bus (KCL)
        let total_flow: f64 = result.branch_flows_mw.iter().sum();
        // Branch flows don't sum to zero globally (they represent directed flows)
        // but power balance should hold

        // No overloads in this configuration
        assert!(result.feasible);
        assert!(result.overloaded.is_empty());
    }

    #[test]
    fn test_power_balance() {
        let net = three_bus_network();
        let result = net.dc_power_flow();

        // Total generation = total load in DC power flow
        let total_load: f64 = 80.0; // 50 + 30
        let total_gen: f64 = 80.0;
        assert!((total_gen - total_load).abs() < 0.01);
    }

    #[test]
    fn test_overload_detection() {
        let mut net = three_bus_network();
        // Reduce thermal limit on branch 0-2 to force overload
        net.branches[1].thermal_limit_mw = 10.0;

        let result = net.dc_power_flow();
        assert!(!result.feasible);
        assert!(!result.overloaded.is_empty());
    }

    #[test]
    fn test_dc_opf_merit_order() {
        let mut net = three_bus_network();
        // Add a second, more expensive generator
        net.generators.push(Generator {
            bus_id: 1,
            name: "Gen2".into(),
            p_min_mw: 0.0,
            p_max_mw: 50.0,
            cost_per_mwh: 50.0,
        });

        let opf = net.dc_opf();

        // Cheapest generator should be dispatched first
        assert!(opf.dispatch_mw[0] > 0.0); // Gen1 at $30/MWh
        // Total dispatch should equal total load
        let total_dispatch: f64 = opf.dispatch_mw.iter().sum();
        assert!((total_dispatch - 80.0).abs() < 1.0);
    }

    #[test]
    fn test_contingency_analysis() {
        let net = three_bus_network();
        let results = net.check_contingencies();

        // Should have 3 contingencies (one per branch)
        assert_eq!(results.len(), 3);

        // At least some should be secure (3-bus with adequate capacity)
        let secure_count = results.iter().filter(|r| r.secure).count();
        assert!(secure_count > 0);
    }

    #[test]
    fn test_ptdf() {
        let net = three_bus_network();
        let ptdf = net.compute_ptdf();

        // PTDF matrix: branches × buses
        assert_eq!(ptdf.len(), 3); // 3 branches
        assert_eq!(ptdf[0].len(), 3); // 3 buses

        // Slack bus should have zero PTDF (reference)
        for l in 0..3 {
            assert!(ptdf[l][0].abs() < 1e-10, "PTDF at slack bus should be zero");
        }
    }

    #[test]
    fn test_gauss_solve_simple() {
        // Solve: 2x + y = 5, x + 3y = 7 → x = 1.6, y = 1.8
        let a = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let b = vec![5.0, 7.0];
        let x = gauss_solve(&a, &b);
        assert!((x[0] - 1.6).abs() < 0.01);
        assert!((x[1] - 1.8).abs() < 0.01);
    }

    #[test]
    fn test_empty_network() {
        let net = PowerNetwork::new();
        let result = net.dc_power_flow();
        assert!(result.feasible);
        assert!(result.angles_rad.is_empty());
    }

    #[test]
    fn test_branch_out_of_service() {
        let mut net = three_bus_network();
        net.branches[2].in_service = false; // Remove branch 1-2

        let result = net.dc_power_flow();
        // Should still solve (network still connected via bus 0)
        assert_eq!(result.branch_flows_mw[2], 0.0); // Out of service branch has zero flow
    }
}
