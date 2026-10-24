//! Correctness-first MIS-to-QUBO/Ising mapping for photonic-optimization research.
//!
//! This module is a mapping and oracle primitive, not an optimizer or a hardware
//! adapter. It does not claim photonic execution, speedup, quantum advantage, or
//! state-of-the-art performance. Returned assignments must still be checked by an
//! independent graph verifier before they count as solutions.

use core::fmt;

/// A simple, upper-triangular QUBO convention for maximum independent set.
///
/// The energy is minimized:
///
/// `E(x) = -sum_i weights[i] * x[i] + penalty * sum_{(u,v) in edges} x[u] * x[v]`.
///
/// Here `x[i]` is a binary selected/not-selected variable, and each canonical
/// undirected edge appears exactly once. The offset is zero. The strict penalty
/// condition `penalty > max(weights)` makes every global minimizer feasible for
/// this ideal objective when weights are finite and non-negative.
#[derive(Debug, Clone, PartialEq)]
pub struct MisQuboModel {
    vertex_count: usize,
    edges: Vec<(usize, usize)>,
    weights: Vec<f64>,
    penalty: f64,
}

/// A sparse Ising model using spins in `{-1, +1}`.
///
/// Spin `+1` corresponds to a selected MIS vertex. Energy is minimized:
///
/// `E(s) = constant + sum_i fields[i] * s[i] + sum_edges couplers[u,v] * s[u] * s[v]`.
#[derive(Debug, Clone, PartialEq)]
pub struct MisIsingModel {
    constant: f64,
    fields: Vec<f64>,
    couplers: Vec<(usize, usize, f64)>,
}

/// Errors in graph validation, mapping, decoding, or energy evaluation.
#[derive(Debug, Clone, PartialEq)]
pub enum MisQuboError {
    /// An edge refers to a vertex outside `0..vertex_count`.
    VertexOutOfRange {
        /// First endpoint supplied by the caller.
        u: usize,
        /// Second endpoint supplied by the caller.
        v: usize,
        /// Number of vertices in the problem.
        vertex_count: usize,
    },
    /// A simple graph cannot contain a self-loop.
    SelfLoop {
        /// Vertex with the self-loop.
        vertex: usize,
    },
    /// The same undirected edge was provided more than once, including reversed copies.
    DuplicateEdge {
        /// Canonical lower endpoint.
        u: usize,
        /// Canonical higher endpoint.
        v: usize,
    },
    /// The weight vector does not have one entry per vertex.
    WeightCountMismatch {
        /// Required number of weights.
        expected: usize,
        /// Number of weights supplied.
        actual: usize,
    },
    /// A weight is negative or non-finite.
    InvalidWeight {
        /// Vertex with invalid weight.
        vertex: usize,
        /// Invalid value.
        value: f64,
    },
    /// The penalty is non-finite or does not strictly exceed every vertex weight.
    InvalidPenalty {
        /// Supplied penalty.
        penalty: f64,
        /// Largest vertex weight (zero for an empty graph).
        maximum_weight: f64,
    },
    /// An assignment has the wrong length.
    AssignmentLengthMismatch {
        /// Required assignment/spin length.
        expected: usize,
        /// Actual assignment/spin length.
        actual: usize,
    },
    /// A spin is not -1 or +1.
    InvalidSpin {
        /// Spin index.
        index: usize,
        /// Supplied spin value.
        value: i8,
    },
    /// An intermediate or result is not finite in f64 arithmetic.
    NumericOverflow,
}

impl fmt::Display for MisQuboError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::VertexOutOfRange { u, v, vertex_count } => write!(
                f,
                "edge ({u}, {v}) references a vertex outside 0..{vertex_count}"
            ),
            Self::SelfLoop { vertex } => write!(f, "self-loop at vertex {vertex} is not allowed"),
            Self::DuplicateEdge { u, v } => write!(f, "duplicate undirected edge ({u}, {v})"),
            Self::WeightCountMismatch { expected, actual } => {
                write!(f, "expected {expected} vertex weights, got {actual}")
            }
            Self::InvalidWeight { vertex, value } => {
                write!(f, "vertex {vertex} has invalid non-negative weight {value}")
            }
            Self::InvalidPenalty { penalty, maximum_weight } => write!(
                f,
                "penalty {penalty} must be finite and strictly greater than max weight {maximum_weight}"
            ),
            Self::AssignmentLengthMismatch { expected, actual } => {
                write!(f, "expected assignment length {expected}, got {actual}")
            }
            Self::InvalidSpin { index, value } => {
                write!(f, "spin {index} must be -1 or +1, got {value}")
            }
            Self::NumericOverflow => write!(f, "non-finite result in QUBO/Ising arithmetic"),
        }
    }
}

impl std::error::Error for MisQuboError {}

impl MisQuboModel {
    /// Construct the canonical weighted MIS QUBO from a graph edge list.
    ///
    /// Edges may be supplied in either endpoint order but must be unique after
    /// canonicalization. Duplicate and reversed-duplicate edges are rejected
    /// instead of silently changing the penalty. Weights must be finite and
    /// non-negative, and the penalty must be finite and strictly greater than
    /// the maximum weight. The coefficient convention is independent of any
    /// target device's matrix or spin convention.
    pub fn new(
        vertex_count: usize,
        edges: &[(usize, usize)],
        weights: &[f64],
        penalty: f64,
    ) -> Result<Self, MisQuboError> {
        if weights.len() != vertex_count {
            return Err(MisQuboError::WeightCountMismatch {
                expected: vertex_count,
                actual: weights.len(),
            });
        }

        for (vertex, &weight) in weights.iter().enumerate() {
            if !weight.is_finite() || weight < 0.0 {
                return Err(MisQuboError::InvalidWeight {
                    vertex,
                    value: weight,
                });
            }
        }

        let maximum_weight = weights.iter().copied().fold(0.0_f64, f64::max);
        if !penalty.is_finite() || penalty <= maximum_weight {
            return Err(MisQuboError::InvalidPenalty {
                penalty,
                maximum_weight,
            });
        }

        let mut canonical_edges = Vec::with_capacity(edges.len());
        for &(u, v) in edges {
            if u >= vertex_count || v >= vertex_count {
                return Err(MisQuboError::VertexOutOfRange { u, v, vertex_count });
            }
            if u == v {
                return Err(MisQuboError::SelfLoop { vertex: u });
            }
            canonical_edges.push(if u < v { (u, v) } else { (v, u) });
        }
        canonical_edges.sort_unstable();
        for pair in canonical_edges.windows(2) {
            if pair[0] == pair[1] {
                return Err(MisQuboError::DuplicateEdge {
                    u: pair[0].0,
                    v: pair[0].1,
                });
            }
        }

        Ok(Self {
            vertex_count,
            edges: canonical_edges,
            weights: weights.to_vec(),
            penalty,
        })
    }

    /// Construct an unweighted MIS QUBO with unit vertex weights.
    pub fn unweighted(
        vertex_count: usize,
        edges: &[(usize, usize)],
        penalty: f64,
    ) -> Result<Self, MisQuboError> {
        let weights = vec![1.0; vertex_count];
        Self::new(vertex_count, edges, &weights, penalty)
    }

    /// Number of binary decision variables.
    pub fn vertex_count(&self) -> usize {
        self.vertex_count
    }

    /// Canonical sorted unique edges.
    pub fn edges(&self) -> &[(usize, usize)] {
        &self.edges
    }

    /// Vertex weights, in canonical vertex-index order.
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Quadratic penalty coefficient.
    pub fn penalty(&self) -> f64 {
        self.penalty
    }

    /// Linear coefficients for the convention `sum_i linear[i]*x[i]`.
    pub fn linear_coefficients(&self) -> Vec<f64> {
        self.weights.iter().map(|weight| -*weight).collect()
    }

    /// Sparse quadratic terms `(u, v, coefficient)`, with each edge exactly once.
    pub fn quadratic_terms(&self) -> Vec<(usize, usize, f64)> {
        self.edges
            .iter()
            .map(|&(u, v)| (u, v, self.penalty))
            .collect()
    }

    /// Constant offset in the canonical QUBO convention (always zero).
    pub fn offset(&self) -> f64 {
        0.0
    }

    /// Evaluate the canonical QUBO energy for one binary assignment.
    pub fn energy(&self, assignment: &[bool]) -> Result<f64, MisQuboError> {
        self.check_assignment_length(assignment.len())?;

        let mut energy = 0.0;
        for (selected, weight) in assignment.iter().zip(&self.weights) {
            if *selected {
                energy -= weight;
            }
        }
        for &(u, v) in &self.edges {
            if assignment[u] && assignment[v] {
                energy += self.penalty;
            }
        }
        if energy.is_finite() {
            Ok(energy)
        } else {
            Err(MisQuboError::NumericOverflow)
        }
    }

    /// Return the weighted objective of a binary assignment.
    pub fn objective_weight(&self, assignment: &[bool]) -> Result<f64, MisQuboError> {
        self.check_assignment_length(assignment.len())?;
        let mut objective = 0.0;
        for (selected, weight) in assignment.iter().zip(&self.weights) {
            if *selected {
                objective += weight;
            }
        }
        if objective.is_finite() {
            Ok(objective)
        } else {
            Err(MisQuboError::NumericOverflow)
        }
    }

    /// Check independent-set feasibility without trusting the QUBO energy.
    pub fn is_feasible(&self, assignment: &[bool]) -> Result<bool, MisQuboError> {
        self.check_assignment_length(assignment.len())?;
        Ok(self
            .edges
            .iter()
            .all(|&(u, v)| !(assignment[u] && assignment[v])))
    }

    /// Convert this canonical QUBO to the matching sparse Ising convention.
    ///
    /// The transform uses `x_i=(1+s_i)/2), so spin +1 means selected:
    /// `constant = -sum(weights)/2 + penalty*|E|/4`,
    /// `field[i] = -weight[i]/2 + penalty*degree[i]/4`, and
    /// `coupler(u,v)=penalty/4`.
    pub fn to_ising(&self) -> Result<MisIsingModel, MisQuboError> {
        let mut degree = vec![0_usize; self.vertex_count];
        for &(u, v) in &self.edges {
            degree[u] += 1;
            degree[v] += 1;
        }

        let mut weight_sum = 0.0;
        for &weight in &self.weights {
            weight_sum += weight;
        }
        let constant = -weight_sum / 2.0
            + self.penalty * (self.edges.len() as f64) / 4.0;
        let fields: Vec<f64> = self
            .weights
            .iter()
            .zip(&degree)
            .map(|(&weight, &d)| -weight / 2.0 + self.penalty * (d as f64) / 4.0)
            .collect();
        let couplers: Vec<(usize, usize, f64)> = self
            .edges
            .iter()
            .map(|&(u, v)| (u, v, self.penalty / 4.0))
            .collect();

        if !weight_sum.is_finite()
            || !constant.is_finite()
            || fields.iter().any(|value| !value.is_finite())
            || couplers.iter().any(|(_, _, value)| !value.is_finite())
        {
            return Err(MisQuboError::NumericOverflow);
        }

        Ok(MisIsingModel {
            constant,
            fields,
            couplers,
        })
    }

    fn check_assignment_length(&self, actual: usize) -> Result<(), MisQuboError> {
        if actual != self.vertex_count {
            return Err(MisQuboError::AssignmentLengthMismatch {
                expected: self.vertex_count,
                actual,
            });
        }
        Ok(())
    }
}

impl MisIsingModel {
    /// Constant energy offset.
    pub fn constant(&self) -> f64 {
        self.constant
    }

    /// Local-field coefficients in canonical vertex order.
    pub fn fields(&self) -> &[f64] {
        &self.fields
    }

    /// Sparse pairwise couplers `(u, v, J_uv)`.
    pub fn couplers(&self) -> &[(usize, usize, f64)] {
        &self.couplers
    }

    /// Evaluate Ising energy for spins in `{-1, +1}`.
    pub fn energy(&self, spins: &[i8]) -> Result<f64, MisQuboError> {
        if spins.len() != self.fields.len() {
            return Err(MisQuboError::AssignmentLengthMismatch {
                expected: self.fields.len(),
                actual: spins.len(),
            });
        }
        for (index, &value) in spins.iter().enumerate() {
            if value != -1 && value != 1 {
                return Err(MisQuboError::InvalidSpin { index, value });
            }
        }

        let mut energy = self.constant;
        for (&spin, &field) in spins.iter().zip(&self.fields) {
            energy += field * f64::from(spin);
        }
        for &(u, v, coupling) in &self.couplers {
            energy += coupling * f64::from(spins[u]) * f64::from(spins[v]);
        }
        if energy.is_finite() {
            Ok(energy)
        } else {
            Err(MisQuboError::NumericOverflow)
        }
    }

    /// Decode spins to binary decisions; spin +1 means selected.
    pub fn decode_spins(spins: &[i8]) -> Result<Vec<bool>, MisQuboError> {
        let mut assignment = Vec::with_capacity(spins.len());
        for (index, &spin) in spins.iter().enumerate() {
            match spin {
                -1 => assignment.push(false),
                1 => assignment.push(true),
                value => return Err(MisQuboError::InvalidSpin { index, value }),
            }
        }
        Ok(assignment)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assignment_from_mask(vertex_count: usize, mask: usize) -> Vec<bool> {
        (0..vertex_count)
            .map(|vertex| mask & (1_usize << vertex) != 0)
            .collect()
    }

    fn spin_encoding(assignment: &[bool]) -> Vec<i8> {
        assignment
            .iter()
            .map(|&selected| if selected { 1 } else { -1 })
            .collect()
    }

    #[test]
    fn hand_checkable_unweighted_fixtures_have_expected_minimum_energy() {
        let fixtures: Vec<(usize, Vec<(usize, usize)>, usize, f64)> = vec![
            (3, vec![], 3, -3.0),
            (2, vec![(0, 1)], 1, -1.0),
            (3, vec![(0, 1), (1, 2)], 2, -2.0),
            (3, vec![(0, 1), (0, 2), (1, 2)], 1, -1.0),
            (3, vec![(0, 1)], 2, -2.0),
        ];
        for (n, edges, expected_cardinality, expected_energy) in fixtures {
            let model = MisQuboModel::unweighted(n, &edges, 2.0).expect("valid mapping");
            let mut best_energy = f64::INFINITY;
            let mut best_cardinality = 0;
            for mask in 0..(1_usize << n) {
                let assignment = assignment_from_mask(n, mask);
                let energy = model.energy(&assignment).expect("finite fixture energy");
                if energy < best_energy {
                    best_energy = energy;
                    best_cardinality = assignment.iter().filter(|&&v| v).count();
                }
            }
            assert_eq!(best_energy, expected_energy, "edges={edges:?}");
            assert_eq!(best_cardinality, expected_cardinality, "edges={edges:?}");
        }
    }

    #[test]
    fn weighted_single_edge_selects_the_higher_weight_vertex() {
        let model = MisQuboModel::new(2, &[(0, 1)], &[3.0, 1.0], 4.0)
            .expect("penalty strictly exceeds maximum weight");
        let left = [true, false];
        let right = [false, true];
        assert_eq!(model.energy(&left).expect("energy"), -3.0);
        assert_eq!(model.energy(&right).expect("energy"), -1.0);
        assert!(model.is_feasible(&left).expect("feasibility"));
        assert!(model.is_feasible(&right).expect("feasibility"));
    }

    #[test]
    fn rejects_penalty_not_strictly_above_maximum_weight() {
        assert!(matches!(
            MisQuboModel::unweighted(2, &[(0, 1)], 1.0),
            Err(MisQuboError::InvalidPenalty { .. })
        ));
        assert!(matches!(
            MisQuboModel::unweighted(2, &[(0, 1)], 0.5),
            Err(MisQuboError::InvalidPenalty { .. })
        ));
    }

    #[test]
    fn negative_control_exposes_under_penalized_conflicting_assignment() {
        // For K2, A=0.5 makes both-selected energy -2 + 0.5 = -1.5,
        // below either feasible singleton's energy of -1. The public constructor
        // correctly refuses this configuration instead of treating it as valid.
        let invalid_both_selected_energy = -2.0 + 0.5;
        let feasible_singleton_energy = -1.0;
        assert!(invalid_both_selected_energy < feasible_singleton_energy);
        assert!(matches!(
            MisQuboModel::unweighted(2, &[(0, 1)], 0.5),
            Err(MisQuboError::InvalidPenalty { .. })
        ));
    }

    #[test]
    fn rejects_invalid_graph_coefficients_and_assignment_shapes() {
        assert!(matches!(
            MisQuboModel::new(2, &[(0, 1)], &[1.0], 2.0),
            Err(MisQuboError::WeightCountMismatch { .. })
        ));
        assert!(matches!(
            MisQuboModel::new(1, &[], &[-1.0], 2.0),
            Err(MisQuboError::InvalidWeight { .. })
        ));
        assert!(matches!(
            MisQuboModel::new(1, &[], &[f64::NAN], 2.0),
            Err(MisQuboError::InvalidWeight { .. })
        ));
        assert!(matches!(
            MisQuboModel::new(2, &[(0, 2)], &[1.0, 1.0], 2.0),
            Err(MisQuboError::VertexOutOfRange { .. })
        ));
        assert!(matches!(
            MisQuboModel::new(1, &[(0, 0)], &[1.0], 2.0),
            Err(MisQuboError::SelfLoop { .. })
        ));
        assert!(matches!(
            MisQuboModel::new(2, &[(0, 1), (1, 0)], &[1.0, 1.0], 2.0),
            Err(MisQuboError::DuplicateEdge { .. })
        ));
        let model = MisQuboModel::unweighted(2, &[(0, 1)], 2.0).expect("valid model");
        assert!(matches!(
            model.energy(&[true]),
            Err(MisQuboError::AssignmentLengthMismatch { .. })
        ));
    }

    #[test]
    fn ising_mapping_matches_qubo_energy_for_every_assignment_on_all_graphs_up_to_five_vertices() {
        for vertex_count in 0..=5 {
            let possible_edges: Vec<(usize, usize)> = (0..vertex_count)
                .flat_map(|u| ((u + 1)..vertex_count).map(move |v| (u, v)))
                .collect();
            let graph_count = 1_usize << possible_edges.len();
            for graph_mask in 0..graph_count {
                let edges: Vec<(usize, usize)> = possible_edges
                    .iter()
                    .enumerate()
                    .filter_map(|(index, &edge)| {
                        (graph_mask & (1_usize << index) != 0).then_some(edge)
                    })
                    .collect();
                let model =
                    MisQuboModel::unweighted(vertex_count, &edges, 2.0).expect("valid graph");
                let ising = model.to_ising().expect("finite Ising coefficients");

                for assignment_mask in 0..(1_usize << vertex_count) {
                    let assignment = assignment_from_mask(vertex_count, assignment_mask);
                    let spins = spin_encoding(&assignment);
                    let qubo_energy = model.energy(&assignment).expect("finite QUBO energy");
                    let ising_energy = ising.energy(&spins).expect("finite Ising energy");
                    assert!(
                        (qubo_energy - ising_energy).abs() < 1e-10,
                        "energy mismatch n={vertex_count} graph={graph_mask} assignment={assignment_mask}: QUBO={qubo_energy} Ising={ising_energy}"
                    );
                    assert_eq!(
                        MisIsingModel::decode_spins(&spins).expect("valid spin decode"),
                        assignment
                    );
                }
            }
        }
    }

    #[test]
    fn energy_minima_are_feasible_and_optimal_for_every_graph_up_to_five_vertices() {
        for vertex_count in 0..=5 {
            let possible_edges: Vec<(usize, usize)> = (0..vertex_count)
                .flat_map(|u| ((u + 1)..vertex_count).map(move |v| (u, v)))
                .collect();
            let graph_count = 1_usize << possible_edges.len();
            for graph_mask in 0..graph_count {
                let edges: Vec<(usize, usize)> = possible_edges
                    .iter()
                    .enumerate()
                    .filter_map(|(index, &edge)| {
                        (graph_mask & (1_usize << index) != 0).then_some(edge)
                    })
                    .collect();
                let model =
                    MisQuboModel::unweighted(vertex_count, &edges, 2.0).expect("valid graph");

                let mut optimum_cardinality = 0_usize;
                let mut minimum_energy = f64::INFINITY;
                let mut minimizers = Vec::new();
                for mask in 0..(1_usize << vertex_count) {
                    let assignment = assignment_from_mask(vertex_count, mask);
                    if model.is_feasible(&assignment).expect("assignment shape") {
                        optimum_cardinality = optimum_cardinality
                            .max(assignment.iter().filter(|&&selected| selected).count());
                    }
                    let energy = model.energy(&assignment).expect("finite energy");
                    if energy < minimum_energy {
                        minimum_energy = energy;
                        minimizers.clear();
                        minimizers.push(assignment);
                    } else if energy == minimum_energy {
                        minimizers.push(assignment);
                    }
                }

                for candidate in minimizers {
                    assert!(
                        model.is_feasible(&candidate).expect("assignment shape"),
                        "infeasible QUBO minimum n={vertex_count}, graph mask={graph_mask}"
                    );
                    assert_eq!(
                        candidate.iter().filter(|&&selected| selected).count(),
                        optimum_cardinality,
                        "non-optimal QUBO minimum n={vertex_count}, graph mask={graph_mask}"
                    );
                }
            }
        }
    }


    #[test]
    fn weighted_qubo_and_ising_match_and_minima_are_optimal_for_all_graphs_up_to_three_vertices() {
        for vertex_count in 0..=3 {
            let possible_edges: Vec<(usize, usize)> = (0..vertex_count)
                .flat_map(|u| ((u + 1)..vertex_count).map(move |v| (u, v)))
                .collect();
            let graph_count = 1_usize << possible_edges.len();
            let weight_vector_count = 3_usize.pow(vertex_count as u32);

            for graph_mask in 0..graph_count {
                let edges: Vec<(usize, usize)> = possible_edges
                    .iter()
                    .enumerate()
                    .filter_map(|(index, &edge)| {
                        (graph_mask & (1_usize << index) != 0).then_some(edge)
                    })
                    .collect();

                for weight_code in 0..weight_vector_count {
                    let mut code = weight_code;
                    let mut weights = Vec::with_capacity(vertex_count);
                    for _ in 0..vertex_count {
                        weights.push((code % 3) as f64);
                        code /= 3;
                    }
                    let maximum_weight = weights.iter().copied().fold(0.0_f64, f64::max);
                    let penalty = maximum_weight + 1.0;
                    let model = MisQuboModel::new(vertex_count, &edges, &weights, penalty)
                        .expect("valid weighted mapping");
                    let ising = model.to_ising().expect("finite weighted Ising mapping");

                    let mut optimum_weight = 0.0_f64;
                    let mut minimum_energy = f64::INFINITY;
                    let mut minimizers = Vec::new();
                    for assignment_mask in 0..(1_usize << vertex_count) {
                        let assignment = assignment_from_mask(vertex_count, assignment_mask);
                        let spins = spin_encoding(&assignment);
                        let qubo_energy = model.energy(&assignment).expect("finite QUBO energy");
                        let ising_energy = ising.energy(&spins).expect("finite Ising energy");
                        assert!(
                            (qubo_energy - ising_energy).abs() < 1e-10,
                            "weighted energy mismatch n={vertex_count} graph={graph_mask} weights={weights:?} assignment={assignment_mask}"
                        );

                        if model.is_feasible(&assignment).expect("assignment shape") {
                            optimum_weight =
                                optimum_weight.max(model.objective_weight(&assignment).expect("finite objective"));
                        }
                        if qubo_energy < minimum_energy {
                            minimum_energy = qubo_energy;
                            minimizers.clear();
                            minimizers.push(assignment);
                        } else if qubo_energy == minimum_energy {
                            minimizers.push(assignment);
                        }
                    }

                    for candidate in minimizers {
                        assert!(
                            model.is_feasible(&candidate).expect("assignment shape"),
                            "infeasible weighted minimum n={vertex_count} graph={graph_mask} weights={weights:?}"
                        );
                        assert_eq!(
                            model.objective_weight(&candidate).expect("finite objective"),
                            optimum_weight,
                            "non-optimal weighted minimum n={vertex_count} graph={graph_mask} weights={weights:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn ising_energy_rejects_non_spin_values_and_decode_is_fail_closed() {
        let model = MisQuboModel::unweighted(1, &[], 2.0).expect("valid model");
        let ising = model.to_ising().expect("finite coefficients");
        assert!(matches!(
            ising.energy(&[0]),
            Err(MisQuboError::InvalidSpin { .. })
        ));
        assert!(matches!(
            MisIsingModel::decode_spins(&[2]),
            Err(MisQuboError::InvalidSpin { .. })
        ));
    }
}
