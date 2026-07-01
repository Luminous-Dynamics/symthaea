// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Approximate MIP Algorithms for Large Systems
//!
//! Efficient heuristics for N > 10 where exhaustive search is infeasible.

use crate::hdc::unified_hv::ContinuousHV;

use super::{TruePartition, TruePhiCalculator};

/// Approximate MIP finder using multiple heuristics
#[derive(Debug, Clone)]
pub struct ApproximateMIPFinder {
    /// Maximum iterations for optimization algorithms
    max_iterations: usize,
    /// Temperature schedule for simulated annealing
    initial_temperature: f64,
    /// Cooling rate for simulated annealing
    cooling_rate: f64,
    /// Number of random restarts
    num_restarts: usize,
}

impl Default for ApproximateMIPFinder {
    fn default() -> Self {
        Self::new()
    }
}

impl ApproximateMIPFinder {
    /// Create with default parameters
    pub fn new() -> Self {
        Self {
            max_iterations: 1000,
            initial_temperature: 1.0,
            cooling_rate: 0.995,
            num_restarts: 5,
        }
    }

    /// Create with custom parameters
    pub fn with_params(max_iterations: usize, initial_temperature: f64, cooling_rate: f64) -> Self {
        Self {
            max_iterations,
            initial_temperature,
            cooling_rate,
            num_restarts: 5,
        }
    }

    /// Find approximate MIP using simulated annealing
    pub fn find_mip_simulated_annealing(
        &self,
        _mi_matrix: &[Vec<f64>],
        calc: &TruePhiCalculator,
        components: &[ContinuousHV],
    ) -> (TruePartition, f64) {
        let n = components.len();
        if n < 2 {
            return (
                TruePartition {
                    part_a: vec![0],
                    part_b: vec![],
                },
                0.0,
            );
        }

        let mut best_partition = self.random_partition(n);
        let mut best_ei = calc.partition_effective_information(components, &best_partition);

        for _ in 0..self.num_restarts {
            let (partition, ei) = self.single_annealing_run(n, calc, components);
            if ei < best_ei {
                best_ei = ei;
                best_partition = partition;
            }
        }

        (best_partition, best_ei)
    }

    /// Single simulated annealing run
    fn single_annealing_run(
        &self,
        n: usize,
        calc: &TruePhiCalculator,
        components: &[ContinuousHV],
    ) -> (TruePartition, f64) {
        let mut current = self.random_partition(n);
        let mut current_ei = calc.partition_effective_information(components, &current);
        let mut best = current.clone();
        let mut best_ei = current_ei;
        let mut temp = self.initial_temperature;

        for iter in 0..self.max_iterations {
            // Generate neighbor by moving one element
            let neighbor = self.neighbor_partition(&current, n);
            let neighbor_ei = calc.partition_effective_information(components, &neighbor);

            // Accept if better, or with probability exp(-delta/T)
            let delta = neighbor_ei - current_ei;
            let accept = delta < 0.0 || {
                let r: f64 = (iter as f64 * 7.0 + 3.0).sin().abs(); // Pseudo-random
                r < (-delta / temp).exp()
            };

            if accept {
                current = neighbor;
                current_ei = neighbor_ei;

                if current_ei < best_ei {
                    best = current.clone();
                    best_ei = current_ei;
                }
            }

            temp *= self.cooling_rate;
        }

        (best, best_ei)
    }

    /// Generate a random partition
    fn random_partition(&self, n: usize) -> TruePartition {
        let mid = n / 2;
        TruePartition {
            part_a: (0..mid).collect(),
            part_b: (mid..n).collect(),
        }
    }

    /// Generate a neighbor partition by moving one element
    fn neighbor_partition(&self, partition: &TruePartition, _n: usize) -> TruePartition {
        let mut new_a = partition.part_a.clone();
        let mut new_b = partition.part_b.clone();

        // Simple deterministic move based on partition state
        let hash = new_a.len() * 17 + new_b.len() * 31;
        let move_from_a = hash.is_multiple_of(2) && new_a.len() > 1;

        if move_from_a {
            let idx = hash % new_a.len();
            let elem = new_a.remove(idx);
            new_b.push(elem);
        } else if new_b.len() > 1 {
            let idx = hash % new_b.len();
            let elem = new_b.remove(idx);
            new_a.push(elem);
        }

        // Ensure non-empty partitions
        if new_a.is_empty() && !new_b.is_empty() {
            new_a.push(new_b.pop().expect("new_b verified non-empty"));
        }
        if new_b.is_empty() && !new_a.is_empty() {
            new_b.push(new_a.pop().expect("new_a verified non-empty"));
        }

        TruePartition {
            part_a: new_a,
            part_b: new_b,
        }
    }

    /// Find MIP using greedy graph cut
    pub fn find_mip_graph_cut(&self, mi_matrix: &[Vec<f64>]) -> TruePartition {
        let n = mi_matrix.len();
        if n < 2 {
            return TruePartition {
                part_a: (0..n).collect(),
                part_b: vec![],
            };
        }

        // Build weighted adjacency from MI matrix
        // Use Kernighan-Lin style heuristic
        let mut part_a: Vec<usize> = (0..n / 2).collect();
        let mut part_b: Vec<usize> = (n / 2..n).collect();

        let mut improved = true;
        while improved {
            improved = false;

            // Try swapping each pair
            for i in 0..part_a.len() {
                for j in 0..part_b.len() {
                    let gain = self.swap_gain(mi_matrix, &part_a, &part_b, i, j);
                    if gain > 1e-10 {
                        // Swap elements
                        let a_elem = part_a[i];
                        let b_elem = part_b[j];
                        part_a[i] = b_elem;
                        part_b[j] = a_elem;
                        improved = true;
                    }
                }
            }
        }

        TruePartition { part_a, part_b }
    }

    /// Compute gain from swapping elements between partitions
    fn swap_gain(
        &self,
        mi_matrix: &[Vec<f64>],
        part_a: &[usize],
        part_b: &[usize],
        i: usize,
        j: usize,
    ) -> f64 {
        let a_elem = part_a[i];
        let b_elem = part_b[j];

        // Current cost: MI from a_elem to part_a + MI from b_elem to part_b
        let mut current_internal = 0.0;
        for &a in part_a {
            if a != a_elem {
                current_internal += mi_matrix[a_elem][a];
            }
        }
        for &b in part_b {
            if b != b_elem {
                current_internal += mi_matrix[b_elem][b];
            }
        }

        // New cost after swap
        let mut new_internal = 0.0;
        for &a in part_a {
            if a != a_elem {
                new_internal += mi_matrix[b_elem][a]; // b_elem now in A
            }
        }
        for &b in part_b {
            if b != b_elem {
                new_internal += mi_matrix[a_elem][b]; // a_elem now in B
            }
        }

        // Positive gain means swap reduces internal MI (good for MIP)
        current_internal - new_internal
    }
}
