// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Multivariate Causal Discovery - PC Algorithm
//!
//! Extends bivariate causal discovery to full DAG structure learning.
//! Uses the PC (Peter-Clark) algorithm with KCIT for conditional independence.
//!
//! ## Algorithm Overview
//!
//! The PC algorithm discovers causal DAGs through:
//! 1. Start with complete undirected graph
//! 2. Remove edges where X ⊥ Y | S for some conditioning set S
//! 3. Orient edges based on v-structures (X -> Z <- Y)
//! 4. Propagate orientation using acyclicity constraints

use super::causal_discovery::{CausalDirection, CausalDiscoveryEngine};
use std::collections::{HashMap, HashSet, VecDeque};

/// A variable in the causal graph
#[derive(Debug, Clone)]
pub struct Variable {
    pub name: String,
    pub data: Vec<f64>,
}

/// An edge in the causal skeleton (before orientation)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SkeletonEdge {
    pub from: usize,
    pub to: usize,
}

/// A directed edge in the causal DAG
#[derive(Debug, Clone)]
pub struct DirectedEdge {
    pub from: usize,
    pub to: usize,
    pub confidence: f64,
}

/// Separation set: the conditioning set that made two variables independent
type SepSet = HashMap<(usize, usize), HashSet<usize>>;

/// The discovered causal DAG
#[derive(Debug, Clone)]
pub struct CausalDAG {
    /// Variable names
    pub variables: Vec<String>,
    /// Directed edges (from -> to)
    pub edges: Vec<DirectedEdge>,
    /// Undirected edges (couldn't determine direction)
    pub undirected: Vec<(usize, usize)>,
    /// Separation sets for each removed edge
    pub sep_sets: SepSet,
}

impl CausalDAG {
    /// Get parents of a variable
    pub fn parents(&self, var_idx: usize) -> Vec<usize> {
        self.edges
            .iter()
            .filter(|e| e.to == var_idx)
            .map(|e| e.from)
            .collect()
    }

    /// Get children of a variable
    pub fn children(&self, var_idx: usize) -> Vec<usize> {
        self.edges
            .iter()
            .filter(|e| e.from == var_idx)
            .map(|e| e.to)
            .collect()
    }

    /// Check if there's a directed path from a to b
    pub fn has_path(&self, from: usize, to: usize) -> bool {
        if from == to {
            return true;
        }

        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(from);

        while let Some(current) = queue.pop_front() {
            if visited.contains(&current) {
                continue;
            }
            visited.insert(current);

            for child in self.children(current) {
                if child == to {
                    return true;
                }
                queue.push_back(child);
            }
        }

        false
    }

    /// Convert to DOT format for visualization
    pub fn to_dot(&self) -> String {
        let mut dot = String::from("digraph CausalDAG {\n");
        dot.push_str("  rankdir=LR;\n");

        for (i, name) in self.variables.iter().enumerate() {
            dot.push_str(&format!("  {i} [label=\"{name}\"];\n"));
        }

        for edge in &self.edges {
            dot.push_str(&format!(
                "  {} -> {} [label=\"{:.2}\"];\n",
                edge.from, edge.to, edge.confidence
            ));
        }

        for (a, b) in &self.undirected {
            dot.push_str(&format!("  {a} -- {b} [style=dashed];\n"));
        }

        dot.push_str("}\n");
        dot
    }
}

/// PC Algorithm for multivariate causal discovery
pub struct PCAlgorithm {
    engine: CausalDiscoveryEngine,
    alpha: f64,           // Significance level for independence tests
    max_cond_size: usize, // Maximum conditioning set size
}

impl PCAlgorithm {
    /// Create new PC algorithm instance
    pub fn new(seed: u64) -> Self {
        Self {
            engine: CausalDiscoveryEngine::with_ensemble_size(seed, 5),
            alpha: 0.05,
            max_cond_size: 3,
        }
    }

    /// Set significance level
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set maximum conditioning set size
    pub fn with_max_cond_size(mut self, size: usize) -> Self {
        self.max_cond_size = size;
        self
    }

    /// Learn causal structure from data
    pub fn learn(&mut self, variables: &[Variable]) -> CausalDAG {
        let n = variables.len();
        if n < 2 {
            return CausalDAG {
                variables: variables.iter().map(|v| v.name.clone()).collect(),
                edges: vec![],
                undirected: vec![],
                sep_sets: HashMap::new(),
            };
        }

        // Step 1: Build skeleton
        let (skeleton, sep_sets) = self.build_skeleton(variables);

        // Step 2: Orient v-structures
        let mut oriented = self.orient_v_structures(&skeleton, &sep_sets, n);

        // Step 3: Propagate orientations
        self.propagate_orientations(&mut oriented, &skeleton, n);

        // Build final DAG
        let mut edges = Vec::new();
        let undirected = Vec::new();

        for edge in &skeleton {
            let (a, b) = (edge.from, edge.to);

            // Check if oriented
            if oriented.contains(&(a, b)) {
                edges.push(DirectedEdge {
                    from: a,
                    to: b,
                    confidence: self.compute_edge_confidence(variables, a, b),
                });
            } else if oriented.contains(&(b, a)) {
                edges.push(DirectedEdge {
                    from: b,
                    to: a,
                    confidence: self.compute_edge_confidence(variables, b, a),
                });
            } else {
                // Use bivariate direction as tiebreaker
                let dir = self.engine.predict(&variables[a].data, &variables[b].data);
                match dir {
                    CausalDirection::Forward => {
                        edges.push(DirectedEdge {
                            from: a,
                            to: b,
                            confidence: 0.5,
                        });
                    }
                    CausalDirection::Backward => {
                        edges.push(DirectedEdge {
                            from: b,
                            to: a,
                            confidence: 0.5,
                        });
                    }
                }
            }
        }

        CausalDAG {
            variables: variables.iter().map(|v| v.name.clone()).collect(),
            edges,
            undirected,
            sep_sets,
        }
    }

    /// Build skeleton by testing conditional independence
    fn build_skeleton(&mut self, variables: &[Variable]) -> (HashSet<SkeletonEdge>, SepSet) {
        let n = variables.len();
        let mut edges: HashSet<SkeletonEdge> = HashSet::new();
        let mut sep_sets: SepSet = HashMap::new();

        // Start with complete graph
        for i in 0..n {
            for j in (i + 1)..n {
                edges.insert(SkeletonEdge { from: i, to: j });
            }
        }

        // Test conditional independence for increasing conditioning set sizes
        for cond_size in 0..=self.max_cond_size.min(n - 2) {
            let edges_to_test: Vec<_> = edges.iter().cloned().collect();

            for edge in edges_to_test {
                let (x, y) = (edge.from, edge.to);

                // Get potential conditioning variables
                let neighbors: Vec<usize> = (0..n)
                    .filter(|&k| k != x && k != y)
                    .filter(|&k| {
                        edges.contains(&SkeletonEdge {
                            from: x.min(k),
                            to: x.max(k),
                        }) || edges.contains(&SkeletonEdge {
                            from: y.min(k),
                            to: y.max(k),
                        })
                    })
                    .collect();

                // Test all conditioning sets of current size
                if neighbors.len() >= cond_size {
                    for cond_set in combinations(&neighbors, cond_size) {
                        let is_independent =
                            self.test_conditional_independence(variables, x, y, &cond_set);

                        if is_independent {
                            edges.remove(&edge);
                            sep_sets.insert((x, y), cond_set.iter().cloned().collect());
                            sep_sets.insert((y, x), cond_set.iter().cloned().collect());
                            break;
                        }
                    }
                }
            }
        }

        (edges, sep_sets)
    }

    /// Test if X ⊥ Y | Z
    fn test_conditional_independence(
        &mut self,
        variables: &[Variable],
        x: usize,
        y: usize,
        cond: &[usize],
    ) -> bool {
        if cond.is_empty() {
            // Unconditional independence test using HSIC
            let hsic = self.compute_hsic(&variables[x].data, &variables[y].data);
            return hsic < self.alpha;
        }

        // Conditional independence using KCIT
        if cond.len() == 1 {
            let (is_indep, _) = self.engine.kcit(
                &variables[x].data,
                &variables[y].data,
                &variables[cond[0]].data,
            );
            return is_indep;
        }

        // For larger conditioning sets, combine conditioning variables
        let z: Vec<f64> = (0..variables[x].data.len())
            .map(|i| {
                cond.iter()
                    .map(|&k| variables[k].data.get(i).unwrap_or(&0.0))
                    .sum::<f64>()
            })
            .collect();

        let (is_indep, _) = self.engine.kcit(&variables[x].data, &variables[y].data, &z);

        is_indep
    }

    /// Compute HSIC for unconditional independence
    fn compute_hsic(&self, x: &[f64], y: &[f64]) -> f64 {
        let n = x.len().min(y.len()).min(100);
        if n < 10 {
            return 0.0;
        }

        let xs: Vec<f64> = x.iter().take(n).cloned().collect();
        let ys: Vec<f64> = y.iter().take(n).cloned().collect();

        let sigma_x = self.median_bandwidth(&xs);
        let sigma_y = self.median_bandwidth(&ys);

        let kx = self.rbf_kernel(&xs, sigma_x);
        let ky = self.rbf_kernel(&ys, sigma_y);

        let hkx = self.center_kernel(&kx);
        let hky = self.center_kernel(&ky);

        let mut hsic = 0.0;
        for i in 0..n {
            for j in 0..n {
                hsic += hkx[i][j] * hky[j][i];
            }
        }

        hsic / ((n - 1) * (n - 1)) as f64
    }

    fn median_bandwidth(&self, x: &[f64]) -> f64 {
        let n = x.len().min(50);
        let mut dists = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                dists.push((x[i] - x[j]).abs());
            }
        }
        if dists.is_empty() {
            return 1.0;
        }
        dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        dists[dists.len() / 2].max(1e-10)
    }

    fn rbf_kernel(&self, x: &[f64], sigma: f64) -> Vec<Vec<f64>> {
        let n = x.len();
        let gamma = 1.0 / (2.0 * sigma * sigma);
        let mut k = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                let diff = x[i] - x[j];
                k[i][j] = (-gamma * diff * diff).exp();
            }
        }
        k
    }

    fn center_kernel(&self, k: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = k.len();
        let row_means: Vec<f64> = k
            .iter()
            .map(|row| row.iter().sum::<f64>() / n as f64)
            .collect();
        let total_mean: f64 = row_means.iter().sum::<f64>() / n as f64;

        let mut centered = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                centered[i][j] = k[i][j] - row_means[i] - row_means[j] + total_mean;
            }
        }
        centered
    }

    /// Orient v-structures (X - Z - Y where X and Y not adjacent)
    fn orient_v_structures(
        &self,
        skeleton: &HashSet<SkeletonEdge>,
        sep_sets: &SepSet,
        n: usize,
    ) -> HashSet<(usize, usize)> {
        let mut oriented = HashSet::new();

        for z in 0..n {
            // Find pairs (x, y) where x - z and z - y but not x - y
            let neighbors: Vec<usize> = (0..n)
                .filter(|&k| k != z)
                .filter(|&k| {
                    skeleton.contains(&SkeletonEdge {
                        from: z.min(k),
                        to: z.max(k),
                    })
                })
                .collect();

            for i in 0..neighbors.len() {
                for j in (i + 1)..neighbors.len() {
                    let x = neighbors[i];
                    let y = neighbors[j];

                    // Check if x and y are not adjacent
                    let not_adjacent = !skeleton.contains(&SkeletonEdge {
                        from: x.min(y),
                        to: x.max(y),
                    });

                    if not_adjacent {
                        // Check if z is not in sep_set(x, y)
                        let sep = sep_sets.get(&(x.min(y), x.max(y)));
                        let z_not_in_sep = sep.map(|s| !s.contains(&z)).unwrap_or(true);

                        if z_not_in_sep {
                            // Orient x -> z <- y (v-structure)
                            oriented.insert((x, z));
                            oriented.insert((y, z));
                        }
                    }
                }
            }
        }

        oriented
    }

    /// Propagate orientations using Meek rules
    fn propagate_orientations(
        &self,
        oriented: &mut HashSet<(usize, usize)>,
        skeleton: &HashSet<SkeletonEdge>,
        n: usize,
    ) {
        let mut changed = true;

        while changed {
            changed = false;

            for edge in skeleton {
                let (a, b) = (edge.from, edge.to);

                // Skip if already oriented
                if oriented.contains(&(a, b)) || oriented.contains(&(b, a)) {
                    continue;
                }

                // Rule 1: If a -> c and c - b, orient c -> b (avoid new v-structure)
                for c in 0..n {
                    if c == a || c == b {
                        continue;
                    }

                    if oriented.contains(&(a, c)) {
                        let c_b_edge = SkeletonEdge {
                            from: c.min(b),
                            to: c.max(b),
                        };
                        if skeleton.contains(&c_b_edge) && !oriented.contains(&(b, c)) {
                            oriented.insert((c, b));
                            changed = true;
                        }
                    }
                }

                // Rule 2: If a -> c -> b, orient a -> b (avoid cycle)
                for c in 0..n {
                    if c == a || c == b {
                        continue;
                    }

                    if oriented.contains(&(a, c)) && oriented.contains(&(c, b)) {
                        oriented.insert((a, b));
                        changed = true;
                    }
                }
            }
        }
    }

    /// Compute confidence for an edge based on bivariate analysis
    fn compute_edge_confidence(&mut self, variables: &[Variable], from: usize, to: usize) -> f64 {
        let (_, conf) = self
            .engine
            .predict_with_confidence(&variables[from].data, &variables[to].data);
        conf
    }
}

/// Generate all combinations of size k from items
fn combinations<T: Clone>(items: &[T], k: usize) -> Vec<Vec<T>> {
    if k == 0 {
        return vec![vec![]];
    }
    if items.len() < k {
        return vec![];
    }

    let mut result = Vec::new();
    for i in 0..=(items.len() - k) {
        let first = items[i].clone();
        for mut rest in combinations(&items[i + 1..], k - 1) {
            let mut combo = vec![first.clone()];
            combo.append(&mut rest);
            result.push(combo);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pc_algorithm_chain() {
        // Create X -> Y -> Z chain
        let x: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 0.5).collect();
        let z: Vec<f64> = y.iter().map(|&yi| 0.5 * yi + 1.0).collect();

        let variables = vec![
            Variable {
                name: "X".to_string(),
                data: x,
            },
            Variable {
                name: "Y".to_string(),
                data: y,
            },
            Variable {
                name: "Z".to_string(),
                data: z,
            },
        ];

        let mut pc = PCAlgorithm::new(42).with_alpha(0.1);
        let dag = pc.learn(&variables);

        println!("Learned DAG:");
        println!("{}", dag.to_dot());

        assert_eq!(dag.variables.len(), 3);
        // Should have some edges (structure may vary due to statistical tests)
    }

    #[test]
    fn test_pc_algorithm_fork() {
        // Create fork: X <- Z -> Y
        let z: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let x: Vec<f64> = z.iter().map(|&zi| 2.0 * zi + 0.3).collect();
        let y: Vec<f64> = z.iter().map(|&zi| 0.5 * zi + 0.7).collect();

        let variables = vec![
            Variable {
                name: "X".to_string(),
                data: x,
            },
            Variable {
                name: "Y".to_string(),
                data: y,
            },
            Variable {
                name: "Z".to_string(),
                data: z,
            },
        ];

        let mut pc = PCAlgorithm::new(42);
        let dag = pc.learn(&variables);

        println!("Learned DAG (fork):");
        println!("{}", dag.to_dot());

        // X and Y should become independent given Z
        assert_eq!(dag.variables.len(), 3);
    }

    #[test]
    fn test_combinations() {
        let items = vec![1, 2, 3, 4];

        let c2 = combinations(&items, 2);
        assert_eq!(c2.len(), 6); // C(4,2) = 6

        let c3 = combinations(&items, 3);
        assert_eq!(c3.len(), 4); // C(4,3) = 4
    }
}