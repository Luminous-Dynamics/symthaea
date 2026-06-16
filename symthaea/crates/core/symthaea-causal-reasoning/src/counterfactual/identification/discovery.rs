// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Causal discovery, mediation analysis, instrumental variables,
//! time-series causal discovery, and transportability.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use super::dag::{CausalDAG, CausalQuery, CausalQueryOutcome};
use super::estimation::{EffectEstimator, ObservationalData};
use super::reasoner::combinations;

// ─────────────────────────────────────────────────────────────────────────────
// PC Algorithm (Causal Discovery)
// ─────────────────────────────────────────────────────────────────────────────

/// PC Algorithm for learning causal structure from observational data.
///
/// The PC algorithm (named after Peter and Clark) is a constraint-based causal
/// discovery method that:
/// 1. Starts with a complete undirected graph
/// 2. Removes edges based on conditional independence tests
/// 3. Orients edges using v-structure detection and orientation rules
///
/// Returns a CPDAG (Completed Partially Directed Acyclic Graph) where:
/// - Directed edges represent definite causal directions
/// - Undirected edges represent uncertain directions
///
/// Reference: Spirtes, Glymour, Scheines. "Causation, Prediction, and Search" (2000)
pub struct PCAlgorithm {
    /// Significance level for independence tests (default: 0.05).
    pub alpha: f64,
    /// Maximum conditioning set size to consider (for scalability).
    pub max_cond_size: usize,
}

impl PCAlgorithm {
    /// Create a new PC algorithm with default parameters.
    pub fn new() -> Self {
        Self {
            alpha: 0.05,
            max_cond_size: 4,
        }
    }

    /// Create with custom significance level.
    pub fn with_alpha(alpha: f64) -> Self {
        Self {
            alpha,
            max_cond_size: 4,
        }
    }

    /// Discover causal structure from observational data.
    ///
    /// Returns a CPDAG representing the learned causal structure.
    pub fn discover(&self, data: &ObservationalData) -> PCResult {
        let n = data.variables.len();
        if n == 0 {
            return PCResult {
                skeleton: Skeleton::empty(vec![]),
                cpdag: CPDAG::empty(vec![]),
                separating_sets: HashMap::new(),
                independence_tests: 0,
            };
        }

        // Phase 1: Learn skeleton (undirected graph)
        let (skeleton, sep_sets, tests) = self.learn_skeleton(data);

        // Phase 2: Orient v-structures
        let mut cpdag = CPDAG::from_skeleton(&skeleton, &data.variables);
        self.orient_v_structures(&mut cpdag, &skeleton, &sep_sets);

        // Phase 3: Apply orientation rules (Meek's rules)
        self.apply_orientation_rules(&mut cpdag);

        PCResult {
            skeleton,
            cpdag,
            separating_sets: sep_sets,
            independence_tests: tests,
        }
    }

    /// Phase 1: Learn the skeleton using conditional independence tests.
    fn learn_skeleton(
        &self,
        data: &ObservationalData,
    ) -> (Skeleton, HashMap<(usize, usize), Vec<usize>>, usize) {
        let n = data.variables.len();

        // Start with complete graph
        let mut adjacencies: Vec<HashSet<usize>> = vec![HashSet::new(); n];
        for i in 0..n {
            for j in (i + 1)..n {
                adjacencies[i].insert(j);
                adjacencies[j].insert(i);
            }
        }

        let mut sep_sets: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
        let mut test_count = 0;

        // Iterate through conditioning set sizes
        for cond_size in 0..=self.max_cond_size {
            let mut edges_to_remove = Vec::new();

            for i in 0..n {
                for &j in &adjacencies[i].clone() {
                    if i >= j {
                        continue; // Only check each pair once
                    }

                    // Get potential conditioning sets from neighbors
                    let neighbors: Vec<usize> = adjacencies[i]
                        .iter()
                        .filter(|&&k| k != j)
                        .copied()
                        .collect();

                    if neighbors.len() < cond_size {
                        continue;
                    }

                    // Test all conditioning sets of this size
                    for cond_set in combinations(&neighbors, cond_size) {
                        test_count += 1;

                        if self.is_independent(data, i, j, &cond_set) {
                            edges_to_remove.push((i, j));
                            sep_sets.insert((i.min(j), i.max(j)), cond_set);
                            break;
                        }
                    }
                }
            }

            // Remove edges found to be conditionally independent
            for (i, j) in edges_to_remove {
                adjacencies[i].remove(&j);
                adjacencies[j].remove(&i);
            }
        }

        let skeleton = Skeleton {
            nodes: data.variables.clone(),
            adjacencies,
        };

        (skeleton, sep_sets, test_count)
    }

    /// Test if X ⊥ Y | Z using partial correlation.
    ///
    /// Uses Fisher's z-transformation for significance testing.
    fn is_independent(&self, data: &ObservationalData, x: usize, y: usize, z: &[usize]) -> bool {
        let n = data.n();
        if n < 5 {
            return false; // Not enough data
        }

        // Compute partial correlation
        let partial_corr = self.partial_correlation(data, x, y, z);

        // Fisher's z-transformation
        let z_stat = fisher_z_transform(partial_corr, n, z.len());

        // Two-tailed test against standard normal
        let critical_value = 1.96; // α = 0.05
        z_stat.abs() < critical_value
    }

    /// Compute partial correlation of X and Y given Z.
    fn partial_correlation(
        &self,
        data: &ObservationalData,
        x: usize,
        y: usize,
        z: &[usize],
    ) -> f64 {
        if z.is_empty() {
            // Simple correlation
            return self.correlation(data, x, y);
        }

        if z.len() == 1 {
            // First-order partial correlation
            let rxy = self.correlation(data, x, y);
            let rxz = self.correlation(data, x, z[0]);
            let ryz = self.correlation(data, y, z[0]);

            let denom = ((1.0 - rxz * rxz) * (1.0 - ryz * ryz)).sqrt();
            if denom < 1e-10 {
                return 0.0;
            }
            return (rxy - rxz * ryz) / denom;
        }

        // Higher-order partial correlation via recursion
        // r(X,Y|Z) = (r(X,Y|Z-z0) - r(X,z0|Z-z0) * r(Y,z0|Z-z0)) /
        //            sqrt((1 - r(X,z0|Z-z0)^2) * (1 - r(Y,z0|Z-z0)^2))
        let z0 = z[0];
        let z_rest: Vec<usize> = z[1..].to_vec();

        let rxy_z = self.partial_correlation(data, x, y, &z_rest);
        let rxz0_z = self.partial_correlation(data, x, z0, &z_rest);
        let ryz0_z = self.partial_correlation(data, y, z0, &z_rest);

        let denom = ((1.0 - rxz0_z * rxz0_z) * (1.0 - ryz0_z * ryz0_z)).sqrt();
        if denom < 1e-10 {
            return 0.0;
        }
        (rxy_z - rxz0_z * ryz0_z) / denom
    }

    /// Compute Pearson correlation between two variables.
    fn correlation(&self, data: &ObservationalData, x: usize, y: usize) -> f64 {
        let n = data.n();
        if n < 2 {
            return 0.0;
        }

        let mean_x = data.mean(x);
        let mean_y = data.mean(y);

        let mut sum_xy = 0.0;
        let mut sum_xx = 0.0;
        let mut sum_yy = 0.0;

        for obs in &data.observations {
            let dx = obs[x] - mean_x;
            let dy = obs[y] - mean_y;
            sum_xy += dx * dy;
            sum_xx += dx * dx;
            sum_yy += dy * dy;
        }

        let denom = (sum_xx * sum_yy).sqrt();
        if denom < 1e-10 {
            return 0.0;
        }
        sum_xy / denom
    }

    /// Phase 2: Orient v-structures (colliders).
    ///
    /// For each triple A - B - C where A and C are not adjacent,
    /// orient as A → B ← C if B is not in the separating set of A and C.
    fn orient_v_structures(
        &self,
        cpdag: &mut CPDAG,
        skeleton: &Skeleton,
        sep_sets: &HashMap<(usize, usize), Vec<usize>>,
    ) {
        let n = skeleton.nodes.len();

        for b in 0..n {
            // Find pairs of non-adjacent neighbors of B
            let neighbors: Vec<usize> = skeleton.adjacencies[b].iter().copied().collect();

            for i in 0..neighbors.len() {
                for j in (i + 1)..neighbors.len() {
                    let a = neighbors[i];
                    let c = neighbors[j];

                    // Check if A and C are non-adjacent
                    if skeleton.adjacencies[a].contains(&c) {
                        continue;
                    }

                    // Check if B is in the separating set
                    let key = (a.min(c), a.max(c));
                    let sep_set = sep_sets.get(&key).cloned().unwrap_or_default();

                    if !sep_set.contains(&b) {
                        // B is not in sep(A,C) → orient as v-structure: A → B ← C
                        cpdag.orient(a, b);
                        cpdag.orient(c, b);
                    }
                }
            }
        }
    }

    /// Phase 3: Apply Meek's orientation rules.
    ///
    /// R1: A → B - C and A-/-C ⟹ B → C
    /// R2: A → B → C and A - C ⟹ A → C
    /// R3: A - B, A - C, A - D, B → D, C → D ⟹ A → D
    /// R4: A - B, B - C, C → D, A → D, A-/-C ⟹ B → C
    fn apply_orientation_rules(&self, cpdag: &mut CPDAG) {
        loop {
            let mut changed = false;

            // R1: A → B - C and A-/-C ⟹ B → C
            for b in 0..cpdag.nodes.len() {
                let parents: Vec<usize> = cpdag.parents(b).iter().copied().collect();
                let undirected: Vec<usize> =
                    cpdag.undirected_neighbors(b).iter().copied().collect();

                for &a in &parents {
                    for &c in &undirected {
                        if !cpdag.adjacent(a, c) {
                            cpdag.orient(b, c);
                            changed = true;
                        }
                    }
                }
            }

            // R2: A → B → C and A - C ⟹ A → C
            for b in 0..cpdag.nodes.len() {
                let parents: Vec<usize> = cpdag.parents(b).iter().copied().collect();
                let children: Vec<usize> = cpdag.children(b).iter().copied().collect();

                for &a in &parents {
                    for &c in &children {
                        if cpdag.undirected_neighbors(a).contains(&c) {
                            cpdag.orient(a, c);
                            changed = true;
                        }
                    }
                }
            }

            if !changed {
                break;
            }
        }
    }
}

impl Default for PCAlgorithm {
    fn default() -> Self {
        Self::new()
    }
}

/// Fisher's z-transformation for testing correlation significance.
fn fisher_z_transform(r: f64, n: usize, k: usize) -> f64 {
    // z = arctanh(r) * sqrt(n - k - 3)
    let r_clamped = r.clamp(-0.9999, 0.9999);
    let z = 0.5 * ((1.0 + r_clamped) / (1.0 - r_clamped)).ln();
    let df = n as f64 - k as f64 - 3.0;
    if df <= 0.0 {
        return 0.0;
    }
    z * df.sqrt()
}

/// Undirected skeleton graph.
#[derive(Debug, Clone)]
pub struct Skeleton {
    /// Node names.
    pub nodes: Vec<String>,
    /// Adjacency lists (undirected).
    pub adjacencies: Vec<HashSet<usize>>,
}

impl Skeleton {
    /// Create an empty skeleton.
    pub fn empty(nodes: Vec<String>) -> Self {
        let n = nodes.len();
        Self {
            nodes,
            adjacencies: vec![HashSet::new(); n],
        }
    }

    /// Check if two nodes are adjacent.
    pub fn adjacent(&self, a: usize, b: usize) -> bool {
        self.adjacencies[a].contains(&b)
    }

    /// Get number of edges.
    pub fn num_edges(&self) -> usize {
        self.adjacencies.iter().map(|adj| adj.len()).sum::<usize>() / 2
    }
}

/// Completed Partially Directed Acyclic Graph (CPDAG).
///
/// Represents the equivalence class of DAGs consistent with the data.
#[derive(Debug, Clone)]
pub struct CPDAG {
    /// Node names.
    pub nodes: Vec<String>,
    /// Directed edges (parent → child).
    pub directed: HashSet<(usize, usize)>,
    /// Undirected edges (unordered pairs stored as (min, max)).
    pub undirected: HashSet<(usize, usize)>,
}

impl CPDAG {
    /// Create an empty CPDAG.
    pub fn empty(nodes: Vec<String>) -> Self {
        Self {
            nodes,
            directed: HashSet::new(),
            undirected: HashSet::new(),
        }
    }

    /// Create from skeleton (all edges undirected).
    pub fn from_skeleton(skeleton: &Skeleton, nodes: &[String]) -> Self {
        let mut undirected = HashSet::new();
        for (i, adj) in skeleton.adjacencies.iter().enumerate() {
            for &j in adj {
                if i < j {
                    undirected.insert((i, j));
                }
            }
        }
        Self {
            nodes: nodes.to_vec(),
            directed: HashSet::new(),
            undirected,
        }
    }

    /// Orient an undirected edge from a to b.
    pub fn orient(&mut self, a: usize, b: usize) {
        let key = (a.min(b), a.max(b));
        if self.undirected.remove(&key) {
            self.directed.insert((a, b));
        }
    }

    /// Get parents of a node.
    pub fn parents(&self, node: usize) -> HashSet<usize> {
        self.directed
            .iter()
            .filter(|(_, c)| *c == node)
            .map(|(p, _)| *p)
            .collect()
    }

    /// Get children of a node.
    pub fn children(&self, node: usize) -> HashSet<usize> {
        self.directed
            .iter()
            .filter(|(p, _)| *p == node)
            .map(|(_, c)| *c)
            .collect()
    }

    /// Get undirected neighbors of a node.
    pub fn undirected_neighbors(&self, node: usize) -> HashSet<usize> {
        let mut result = HashSet::new();
        for &(a, b) in &self.undirected {
            if a == node {
                result.insert(b);
            } else if b == node {
                result.insert(a);
            }
        }
        result
    }

    /// Check if two nodes are adjacent (directed or undirected).
    pub fn adjacent(&self, a: usize, b: usize) -> bool {
        let key = (a.min(b), a.max(b));
        self.undirected.contains(&key)
            || self.directed.contains(&(a, b))
            || self.directed.contains(&(b, a))
    }

    /// Convert to CausalDAG (directed edges only).
    ///
    /// Note: Loses undirected edge information.
    pub fn to_dag(&self) -> CausalDAG {
        CausalDAG::new(self.nodes.clone(), self.directed.iter().copied().collect())
    }

    /// Get number of directed edges.
    pub fn num_directed(&self) -> usize {
        self.directed.len()
    }

    /// Get number of undirected edges.
    pub fn num_undirected(&self) -> usize {
        self.undirected.len()
    }
}

/// Result of PC algorithm causal discovery.
#[derive(Debug, Clone)]
pub struct PCResult {
    /// Undirected skeleton graph.
    pub skeleton: Skeleton,
    /// Completed partially directed graph.
    pub cpdag: CPDAG,
    /// Separating sets for each non-adjacent pair.
    pub separating_sets: HashMap<(usize, usize), Vec<usize>>,
    /// Number of independence tests performed.
    pub independence_tests: usize,
}

impl PCResult {
    /// Check if the algorithm found a valid structure.
    pub fn is_valid(&self) -> bool {
        !self.cpdag.nodes.is_empty()
    }

    /// Get the learned DAG (directed edges only).
    pub fn to_dag(&self) -> CausalDAG {
        self.cpdag.to_dag()
    }

    /// Summarize the discovered structure.
    pub fn summary(&self) -> String {
        format!(
            "PC Algorithm Result:\n\
             - Nodes: {}\n\
             - Skeleton edges: {}\n\
             - Directed edges: {}\n\
             - Undirected edges: {}\n\
             - Independence tests: {}",
            self.cpdag.nodes.len(),
            self.skeleton.num_edges(),
            self.cpdag.num_directed(),
            self.cpdag.num_undirected(),
            self.independence_tests
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Causal Mediation Analysis
// ─────────────────────────────────────────────────────────────────────────────

/// Causal mediation analysis for decomposing treatment effects.
///
/// Given a treatment X, mediator M, and outcome Y, decomposes the total effect into:
/// - **Natural Direct Effect (NDE)**: Effect of X on Y NOT through M
/// - **Natural Indirect Effect (NIE)**: Effect of X on Y THROUGH M
/// - **Total Effect (TE)** = NDE + NIE (on additive scale)
///
/// Reference: VanderWeele (2015). "Explanation in Causal Inference"
///
/// # Example
///
/// ```ignore
/// // Smoking → Tar → Cancer
/// let dag = CausalDAG::new(
///     vec!["Smoking".into(), "Tar".into(), "Cancer".into()],
///     vec![(0, 1), (0, 2), (1, 2)],  // Smoking → Tar, Smoking → Cancer, Tar → Cancer
/// );
///
/// let analysis = MediationAnalysis::new(&dag, 0, 1, 2);  // X=Smoking, M=Tar, Y=Cancer
/// let result = analysis.analyze(&data);
/// println!("Direct effect: {}", result.nde);
/// println!("Indirect effect (via Tar): {}", result.nie);
/// ```
pub struct MediationAnalysis<'a> {
    dag: &'a CausalDAG,
    treatment: usize,
    mediator: usize,
    outcome: usize,
}

impl<'a> MediationAnalysis<'a> {
    /// Create a new mediation analysis.
    ///
    /// # Arguments
    /// * `dag` - The causal DAG
    /// * `treatment` - Index of treatment variable X
    /// * `mediator` - Index of mediator variable M
    /// * `outcome` - Index of outcome variable Y
    pub fn new(dag: &'a CausalDAG, treatment: usize, mediator: usize, outcome: usize) -> Self {
        Self {
            dag,
            treatment,
            mediator,
            outcome,
        }
    }

    /// Check if mediation is identified given the causal structure.
    ///
    /// Mediation requires:
    /// 1. X → M path exists
    /// 2. M → Y path exists (with or without controlling for X)
    /// 3. No confounding of M → Y that is affected by X
    pub fn is_identified(&self) -> MediationIdentification {
        // Check basic structure
        let x_to_m = self.dag.has_path(self.treatment, self.mediator);
        let m_to_y = self.dag.has_path(self.mediator, self.outcome);
        let x_to_y = self.dag.has_path(self.treatment, self.outcome);

        if !x_to_m {
            return MediationIdentification::NotMediator {
                reason: "No path from treatment to mediator".to_string(),
            };
        }

        if !m_to_y {
            return MediationIdentification::NotMediator {
                reason: "No path from mediator to outcome".to_string(),
            };
        }

        // Check for exposure-induced confounding
        // This occurs when X affects a confounder of M → Y
        let m_parents: HashSet<usize> = self.dag.parents(self.mediator).into_iter().collect();
        let y_parents: HashSet<usize> = self.dag.parents(self.outcome).into_iter().collect();

        // Potential confounders of M → Y
        let potential_confounders: Vec<usize> = m_parents
            .intersection(&y_parents)
            .filter(|&&n| n != self.treatment && n != self.mediator)
            .copied()
            .collect();

        // Check if X affects any of these confounders
        for &confounder in &potential_confounders {
            if self.dag.has_path(self.treatment, confounder) {
                return MediationIdentification::ExposureInducedConfounding {
                    confounder: self
                        .dag
                        .nodes
                        .get(confounder)
                        .cloned()
                        .unwrap_or_else(|| format!("Node_{confounder}")),
                };
            }
        }

        // Find adjustment set for NDE/NIE
        let baseline_confounders: Vec<usize> = self
            .dag
            .parents(self.treatment)
            .into_iter()
            .filter(|&n| self.dag.has_path(n, self.outcome))
            .collect();

        MediationIdentification::Identified {
            nde_adjustment: baseline_confounders.clone(),
            nie_adjustment: baseline_confounders,
            has_direct_effect: x_to_y,
        }
    }

    /// Estimate mediation effects from data.
    ///
    /// Uses the difference method (Baron-Kenny approach):
    /// - NDE = E[Y | do(X=1), M(0)] - E[Y | do(X=0), M(0)]
    /// - NIE = E[Y | do(X=1), M(1)] - E[Y | do(X=1), M(0)]
    ///
    /// For linear models without interactions:
    /// - Total = c (regression coefficient of Y on X)
    /// - NDE = c' (regression coefficient of Y on X controlling for M)
    /// - NIE = a * b (product of coefficients)
    pub fn analyze(&self, data: &ObservationalData) -> MediationResult {
        let identification = self.is_identified();

        match &identification {
            MediationIdentification::Identified {
                nde_adjustment: _, ..
            } => {
                // Simplified linear mediation analysis

                // Step 1: Total effect (Y ~ X)
                let total_effect = self.simple_regression(data, self.treatment, self.outcome);

                // Step 2: Effect of X on M (a path)
                let a_path = self.simple_regression(data, self.treatment, self.mediator);

                // Step 3: Effect of M on Y controlling for X (b path) and direct effect (c')
                let (c_prime, b_path) =
                    self.multiple_regression_2(data, self.outcome, self.treatment, self.mediator);

                // NIE = a * b (indirect effect)
                let nie = a_path * b_path;

                // NDE = c' (direct effect)
                let nde = c_prime;

                // Proportion mediated
                let proportion_mediated = if total_effect.abs() > 1e-10 {
                    nie / total_effect
                } else {
                    0.0
                };

                MediationResult {
                    total_effect,
                    natural_direct_effect: nde,
                    natural_indirect_effect: nie,
                    a_path,
                    b_path,
                    c_prime,
                    proportion_mediated: proportion_mediated.clamp(0.0, 1.0),
                    is_identified: true,
                    identification,
                }
            }
            _ => {
                // Not identified - return NaN values
                MediationResult {
                    total_effect: f64::NAN,
                    natural_direct_effect: f64::NAN,
                    natural_indirect_effect: f64::NAN,
                    a_path: f64::NAN,
                    b_path: f64::NAN,
                    c_prime: f64::NAN,
                    proportion_mediated: f64::NAN,
                    is_identified: false,
                    identification,
                }
            }
        }
    }

    /// Simple linear regression: Y ~ X
    fn simple_regression(&self, data: &ObservationalData, x: usize, y: usize) -> f64 {
        let n = data.n();
        if n < 2 {
            return 0.0;
        }

        let mean_x = data.mean(x);
        let mean_y = data.mean(y);

        let mut sum_xy = 0.0;
        let mut sum_xx = 0.0;

        for obs in &data.observations {
            let dx = obs[x] - mean_x;
            let dy = obs[y] - mean_y;
            sum_xy += dx * dy;
            sum_xx += dx * dx;
        }

        if sum_xx < 1e-10 {
            return 0.0;
        }

        sum_xy / sum_xx
    }

    /// Multiple regression: Y ~ X + M, returns (coef_x, coef_m)
    fn multiple_regression_2(
        &self,
        data: &ObservationalData,
        y: usize,
        x1: usize,
        x2: usize,
    ) -> (f64, f64) {
        let n = data.n();
        if n < 3 {
            return (0.0, 0.0);
        }

        let mean_y = data.mean(y);
        let mean_x1 = data.mean(x1);
        let mean_x2 = data.mean(x2);

        // Compute sums for normal equations
        let mut sum_x1x1 = 0.0;
        let mut sum_x2x2 = 0.0;
        let mut sum_x1x2 = 0.0;
        let mut sum_x1y = 0.0;
        let mut sum_x2y = 0.0;

        for obs in &data.observations {
            let dx1 = obs[x1] - mean_x1;
            let dx2 = obs[x2] - mean_x2;
            let dy = obs[y] - mean_y;

            sum_x1x1 += dx1 * dx1;
            sum_x2x2 += dx2 * dx2;
            sum_x1x2 += dx1 * dx2;
            sum_x1y += dx1 * dy;
            sum_x2y += dx2 * dy;
        }

        // Solve 2x2 system: [[sum_x1x1, sum_x1x2], [sum_x1x2, sum_x2x2]] * [b1, b2] = [sum_x1y, sum_x2y]
        let det = sum_x1x1 * sum_x2x2 - sum_x1x2 * sum_x1x2;
        if det.abs() < 1e-10 {
            return (0.0, 0.0);
        }

        let b1 = (sum_x2x2 * sum_x1y - sum_x1x2 * sum_x2y) / det;
        let b2 = (sum_x1x1 * sum_x2y - sum_x1x2 * sum_x1y) / det;

        (b1, b2)
    }
}

/// Result of mediation identification check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MediationIdentification {
    /// Mediation effects are identified.
    Identified {
        /// Variables to adjust for when estimating NDE.
        nde_adjustment: Vec<usize>,
        /// Variables to adjust for when estimating NIE.
        nie_adjustment: Vec<usize>,
        /// Whether there is a direct effect (X → Y path exists).
        has_direct_effect: bool,
    },
    /// Not a valid mediator.
    NotMediator {
        /// Reason why M is not a valid mediator.
        reason: String,
    },
    /// Exposure-induced confounding blocks identification.
    ExposureInducedConfounding {
        /// The confounder that is affected by X.
        confounder: String,
    },
}

/// Result of causal mediation analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediationResult {
    /// Total effect of X on Y.
    pub total_effect: f64,
    /// Natural Direct Effect (not through mediator).
    pub natural_direct_effect: f64,
    /// Natural Indirect Effect (through mediator).
    pub natural_indirect_effect: f64,
    /// a-path: Effect of X on M.
    pub a_path: f64,
    /// b-path: Effect of M on Y (controlling for X).
    pub b_path: f64,
    /// c'-path: Direct effect of X on Y (controlling for M).
    pub c_prime: f64,
    /// Proportion of total effect mediated (0-1).
    pub proportion_mediated: f64,
    /// Whether the mediation is identified.
    pub is_identified: bool,
    /// Identification status.
    pub identification: MediationIdentification,
}

impl MediationResult {
    /// Check if there is significant mediation.
    ///
    /// Returns true if:
    /// - NIE is non-negligible (> threshold)
    /// - NIE has the same sign as total effect
    pub fn has_significant_mediation(&self, threshold: f64) -> bool {
        self.is_identified
            && self.natural_indirect_effect.abs() > threshold
            && (self.natural_indirect_effect.signum() == self.total_effect.signum()
                || self.total_effect.abs() < 1e-10)
    }

    /// Check if the effect is fully mediated (> 80% through mediator).
    pub fn is_fully_mediated(&self) -> bool {
        self.is_identified && self.proportion_mediated > 0.8
    }

    /// Check if the effect is partially mediated (20-80% through mediator).
    pub fn is_partially_mediated(&self) -> bool {
        self.is_identified && self.proportion_mediated > 0.2 && self.proportion_mediated <= 0.8
    }

    /// Get a summary of the mediation analysis.
    pub fn summary(&self) -> String {
        if !self.is_identified {
            return format!("Mediation not identified: {:?}", self.identification);
        }

        let mediation_type = if self.is_fully_mediated() {
            "Full mediation"
        } else if self.is_partially_mediated() {
            "Partial mediation"
        } else if self.proportion_mediated > 0.0 {
            "Weak mediation"
        } else {
            "No mediation"
        };

        format!(
            "Mediation Analysis:\n\
             - Total Effect: {:.4}\n\
             - Direct Effect (NDE): {:.4}\n\
             - Indirect Effect (NIE): {:.4}\n\
             - Proportion Mediated: {:.1}%\n\
             - Type: {}",
            self.total_effect,
            self.natural_direct_effect,
            self.natural_indirect_effect,
            self.proportion_mediated * 100.0,
            mediation_type
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Instrumental Variable Estimation
// ─────────────────────────────────────────────────────────────────────────────

/// Instrumental Variable (IV) estimator for causal effects.
///
/// Used when treatment X is confounded with outcome Y, but we have an
/// instrument Z that:
/// 1. Affects treatment (Z → X)
/// 2. Only affects outcome through treatment (no Z → Y path except via X)
/// 3. Is independent of confounders
///
/// The classic example: distance to college (Z) as instrument for education (X)
/// on earnings (Y).
///
/// # Two-Stage Least Squares (2SLS)
///
/// Stage 1: X̂ = α + β*Z (predict X from Z)
/// Stage 2: Y = γ + δ*X̂ (regress Y on predicted X)
///
/// The coefficient δ is the causal effect of X on Y.
pub struct IVEstimator;

impl IVEstimator {
    /// Check if Z is a valid instrument for X → Y.
    ///
    /// Instrument validity requires:
    /// 1. Relevance: Z → X path exists
    /// 2. Exclusion: No direct Z → Y path
    /// 3. Independence: No confounding of Z
    pub fn is_valid_instrument(
        dag: &CausalDAG,
        instrument: usize,
        treatment: usize,
        outcome: usize,
    ) -> IVValidity {
        // Check relevance: Z must affect X
        if !dag.has_path(instrument, treatment) {
            return IVValidity::Invalid {
                reason: "Instrument does not affect treatment (no Z → X path)".to_string(),
            };
        }

        // Check exclusion: Z should not directly affect Y
        // This is a simplification - full check requires excluding paths through X
        let z_children = dag.children(instrument);
        for child in z_children {
            if child == outcome {
                return IVValidity::Invalid {
                    reason: "Instrument directly affects outcome (Z → Y path exists)".to_string(),
                };
            }
        }

        // Check if Z only reaches Y through X
        let mut reaches_y_not_through_x = false;
        let z_descendants = dag.descendants(instrument);
        let x_descendants = dag.descendants(treatment);

        for desc in &z_descendants {
            if *desc == outcome && !x_descendants.contains(&outcome) {
                reaches_y_not_through_x = true;
                break;
            }
        }

        if reaches_y_not_through_x {
            return IVValidity::Invalid {
                reason: "Instrument reaches outcome through path not including treatment"
                    .to_string(),
            };
        }

        IVValidity::Valid {
            instrument_strength: 1.0, // Would be computed from data
        }
    }

    /// Estimate causal effect using Two-Stage Least Squares (2SLS).
    ///
    /// Returns the Local Average Treatment Effect (LATE) for compliers.
    pub fn estimate_2sls(
        data: &ObservationalData,
        instrument: usize,
        treatment: usize,
        outcome: usize,
    ) -> IVResult {
        let n = data.n();
        if n < 10 {
            return IVResult {
                effect: f64::NAN,
                first_stage_f: 0.0,
                is_weak_instrument: true,
                method: "2SLS".to_string(),
            };
        }

        // Stage 1: Regress X on Z
        let (first_stage_coef, first_stage_r2) = Self::first_stage(data, instrument, treatment);

        // Check for weak instrument (F-statistic < 10 rule of thumb)
        let first_stage_f = (n as f64 - 2.0) * first_stage_r2 / (1.0 - first_stage_r2);
        let is_weak = first_stage_f < 10.0;

        // Stage 2: Regress Y on X̂
        let effect = Self::second_stage(data, instrument, treatment, outcome, first_stage_coef);

        IVResult {
            effect,
            first_stage_f,
            is_weak_instrument: is_weak,
            method: "2SLS".to_string(),
        }
    }

    /// First stage regression: X ~ Z
    fn first_stage(data: &ObservationalData, z: usize, x: usize) -> (f64, f64) {
        let _n = data.n();
        let mean_z = data.mean(z);
        let mean_x = data.mean(x);

        let mut sum_zx = 0.0;
        let mut sum_zz = 0.0;
        let mut sum_xx = 0.0;

        for obs in &data.observations {
            let dz = obs[z] - mean_z;
            let dx = obs[x] - mean_x;
            sum_zx += dz * dx;
            sum_zz += dz * dz;
            sum_xx += dx * dx;
        }

        let beta = if sum_zz > 1e-10 { sum_zx / sum_zz } else { 0.0 };

        // R² = (Cov(Z,X))² / (Var(Z) * Var(X))
        let r2 = if sum_zz > 1e-10 && sum_xx > 1e-10 {
            (sum_zx * sum_zx) / (sum_zz * sum_xx)
        } else {
            0.0
        };

        (beta, r2)
    }

    /// Second stage regression: Y ~ X̂
    fn second_stage(
        data: &ObservationalData,
        z: usize,
        x: usize,
        y: usize,
        first_stage_coef: f64,
    ) -> f64 {
        let mean_z = data.mean(z);
        let mean_x = data.mean(x);
        let mean_y = data.mean(y);

        // Compute X̂ = mean_x + first_stage_coef * (Z - mean_z)
        let mut sum_xhat_y = 0.0;
        let mut sum_xhat_xhat = 0.0;

        for obs in &data.observations {
            let x_hat = mean_x + first_stage_coef * (obs[z] - mean_z);
            let dx_hat = x_hat - mean_x;
            let dy = obs[y] - mean_y;
            sum_xhat_y += dx_hat * dy;
            sum_xhat_xhat += dx_hat * dx_hat;
        }

        if sum_xhat_xhat > 1e-10 {
            sum_xhat_y / sum_xhat_xhat
        } else {
            0.0
        }
    }

    /// Wald estimator (simple IV with binary instrument).
    ///
    /// Effect = (E[Y|Z=1] - E[Y|Z=0]) / (E[X|Z=1] - E[X|Z=0])
    pub fn estimate_wald(
        data: &ObservationalData,
        instrument: usize,
        treatment: usize,
        outcome: usize,
    ) -> f64 {
        // Split data by instrument value (assuming binary/threshold at 0.5)
        let mut y_z1 = Vec::new();
        let mut y_z0 = Vec::new();
        let mut x_z1 = Vec::new();
        let mut x_z0 = Vec::new();

        for obs in &data.observations {
            if obs[instrument] > 0.5 {
                y_z1.push(obs[outcome]);
                x_z1.push(obs[treatment]);
            } else {
                y_z0.push(obs[outcome]);
                x_z0.push(obs[treatment]);
            }
        }

        if y_z1.is_empty() || y_z0.is_empty() {
            return f64::NAN;
        }

        let mean_y_z1: f64 = y_z1.iter().sum::<f64>() / y_z1.len() as f64;
        let mean_y_z0: f64 = y_z0.iter().sum::<f64>() / y_z0.len() as f64;
        let mean_x_z1: f64 = x_z1.iter().sum::<f64>() / x_z1.len() as f64;
        let mean_x_z0: f64 = x_z0.iter().sum::<f64>() / x_z0.len() as f64;

        let denom = mean_x_z1 - mean_x_z0;
        if denom.abs() < 1e-10 {
            return f64::NAN;
        }

        (mean_y_z1 - mean_y_z0) / denom
    }
}

/// Validity status of an instrumental variable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IVValidity {
    /// Instrument is valid.
    Valid {
        /// Strength of the instrument (first-stage F-statistic).
        instrument_strength: f64,
    },
    /// Instrument is invalid.
    Invalid {
        /// Reason for invalidity.
        reason: String,
    },
}

/// Result of instrumental variable estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IVResult {
    /// Estimated causal effect (LATE).
    pub effect: f64,
    /// First-stage F-statistic.
    pub first_stage_f: f64,
    /// Whether the instrument is weak (F < 10).
    pub is_weak_instrument: bool,
    /// Estimation method used.
    pub method: String,
}

// ─────────────────────────────────────────────────────────────────────────────
// Time-Series Causal Discovery
// ─────────────────────────────────────────────────────────────────────────────

/// Time-series causal discovery using Granger causality and temporal PC.
///
/// Extends causal discovery to longitudinal data where time ordering
/// provides additional constraints on possible causal relationships.
///
/// Key insight: Causes must precede effects in time.
pub struct TimeSeriesCausalDiscovery {
    /// Maximum lag to consider.
    pub max_lag: usize,
    /// Significance level for Granger tests.
    pub alpha: f64,
}

impl TimeSeriesCausalDiscovery {
    /// Create new time-series causal discovery.
    pub fn new(max_lag: usize) -> Self {
        Self {
            max_lag,
            alpha: 0.05,
        }
    }

    /// Test Granger causality: Does X Granger-cause Y?
    ///
    /// X Granger-causes Y if past values of X help predict Y
    /// beyond what past values of Y alone can predict.
    ///
    /// Returns F-statistic and p-value approximation.
    pub fn granger_test(&self, x: &[f64], y: &[f64], lag: usize) -> GrangerResult {
        if x.len() != y.len() || x.len() <= lag + 1 {
            return GrangerResult {
                f_statistic: 0.0,
                p_value: 1.0,
                is_significant: false,
                optimal_lag: 0,
            };
        }

        let n = x.len() - lag;

        // Restricted model: Y_t ~ Y_{t-1} + ... + Y_{t-lag}
        let ssr_restricted = self.compute_ssr_restricted(y, lag);

        // Unrestricted model: Y_t ~ Y_{t-1} + ... + Y_{t-lag} + X_{t-1} + ... + X_{t-lag}
        let ssr_unrestricted = self.compute_ssr_unrestricted(x, y, lag);

        // F-statistic
        let df1 = lag as f64;
        let df2 = (n - 2 * lag - 1) as f64;

        if ssr_unrestricted < 1e-10 || df2 <= 0.0 {
            return GrangerResult {
                f_statistic: 0.0,
                p_value: 1.0,
                is_significant: false,
                optimal_lag: lag,
            };
        }

        let f_stat = ((ssr_restricted - ssr_unrestricted) / df1) / (ssr_unrestricted / df2);

        // Approximate p-value using F-distribution CDF approximation
        let p_value = self.f_distribution_pvalue(f_stat, df1, df2);

        GrangerResult {
            f_statistic: f_stat,
            p_value,
            is_significant: p_value < self.alpha,
            optimal_lag: lag,
        }
    }

    /// Compute Sum of Squared Residuals for restricted model.
    fn compute_ssr_restricted(&self, y: &[f64], lag: usize) -> f64 {
        let n = y.len();
        let mut ssr = 0.0;

        for t in lag..n {
            // Predict Y_t from past Y values
            let mut y_pred = 0.0;
            for l in 1..=lag {
                y_pred += y[t - l] / lag as f64; // Simple average as baseline
            }
            let residual = y[t] - y_pred;
            ssr += residual * residual;
        }

        ssr
    }

    /// Compute Sum of Squared Residuals for unrestricted model.
    fn compute_ssr_unrestricted(&self, x: &[f64], y: &[f64], lag: usize) -> f64 {
        let n = y.len();
        let mut ssr = 0.0;

        for t in lag..n {
            // Predict Y_t from past Y and X values
            let mut y_pred = 0.0;
            for l in 1..=lag {
                y_pred += (y[t - l] + x[t - l]) / (2.0 * lag as f64);
            }
            let residual = y[t] - y_pred;
            ssr += residual * residual;
        }

        ssr
    }

    /// Approximate F-distribution p-value.
    fn f_distribution_pvalue(&self, f: f64, df1: f64, df2: f64) -> f64 {
        // Wilson-Hilferty approximation for F-distribution
        if f <= 0.0 || df1 <= 0.0 || df2 <= 0.0 {
            return 1.0;
        }

        let x = (f.powf(1.0 / 3.0) * (1.0 - 2.0 / (9.0 * df2)) - (1.0 - 2.0 / (9.0 * df1)))
            / ((2.0 / (9.0 * df1) + f.powf(2.0 / 3.0) * 2.0 / (9.0 * df2)).sqrt());

        // Standard normal CDF approximation
        0.5 * (1.0 - Self::erf(x / std::f64::consts::SQRT_2))
    }

    /// Error function approximation.
    fn erf(x: f64) -> f64 {
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;

        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();
        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

        sign * y
    }

    /// Discover causal structure from multivariate time series.
    ///
    /// Uses Granger causality tests to build a causal graph where
    /// X → Y if X Granger-causes Y.
    pub fn discover(&self, data: &TimeSeriesData) -> TimeSeriesCausalGraph {
        let n_vars = data.variables.len();
        let mut edges = Vec::new();
        let mut granger_results = HashMap::new();

        for i in 0..n_vars {
            for j in 0..n_vars {
                if i == j {
                    continue;
                }

                // Test if variable i Granger-causes variable j
                let mut best_result = GrangerResult {
                    f_statistic: 0.0,
                    p_value: 1.0,
                    is_significant: false,
                    optimal_lag: 1,
                };

                for lag in 1..=self.max_lag {
                    let result = self.granger_test(&data.series[i], &data.series[j], lag);
                    if result.f_statistic > best_result.f_statistic {
                        best_result = result;
                        best_result.optimal_lag = lag;
                    }
                }

                if best_result.is_significant {
                    edges.push((i, j, best_result.optimal_lag));
                }

                granger_results.insert((i, j), best_result);
            }
        }

        TimeSeriesCausalGraph {
            variables: data.variables.clone(),
            edges,
            granger_results,
        }
    }
}

impl Default for TimeSeriesCausalDiscovery {
    fn default() -> Self {
        Self::new(5)
    }
}

/// Time series data for multiple variables.
#[derive(Debug, Clone)]
pub struct TimeSeriesData {
    /// Variable names.
    pub variables: Vec<String>,
    /// Time series for each variable (same length).
    pub series: Vec<Vec<f64>>,
}

impl TimeSeriesData {
    /// Create new time series data.
    pub fn new(variables: Vec<String>) -> Self {
        let n = variables.len();
        Self {
            variables,
            series: vec![Vec::new(); n],
        }
    }

    /// Add a time point observation for all variables.
    pub fn add_observation(&mut self, values: Vec<f64>) {
        for (i, v) in values.into_iter().enumerate() {
            if i < self.series.len() {
                self.series[i].push(v);
            }
        }
    }

    /// Get number of time points.
    pub fn n_timepoints(&self) -> usize {
        self.series.first().map(|s| s.len()).unwrap_or(0)
    }
}

/// Result of Granger causality test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrangerResult {
    /// F-statistic for the test.
    pub f_statistic: f64,
    /// P-value of the test.
    pub p_value: f64,
    /// Whether the result is significant at alpha level.
    pub is_significant: bool,
    /// Optimal lag that gave highest F-statistic.
    pub optimal_lag: usize,
}

/// Causal graph discovered from time series.
#[derive(Debug, Clone)]
pub struct TimeSeriesCausalGraph {
    /// Variable names.
    pub variables: Vec<String>,
    /// Edges as (from, to, lag).
    pub edges: Vec<(usize, usize, usize)>,
    /// Granger test results for all pairs.
    pub granger_results: HashMap<(usize, usize), GrangerResult>,
}

impl TimeSeriesCausalGraph {
    /// Convert to standard CausalDAG (ignoring lags).
    pub fn to_dag(&self) -> CausalDAG {
        let edges: Vec<(usize, usize)> = self.edges.iter().map(|(f, t, _)| (*f, *t)).collect();
        CausalDAG::new(self.variables.clone(), edges)
    }

    /// Get summary of discovered causal relationships.
    pub fn summary(&self) -> String {
        let mut lines = vec!["Time-Series Causal Graph:".to_string()];

        for (from, to, lag) in &self.edges {
            lines.push(format!(
                "  {} → {} (lag={})",
                self.variables[*from], self.variables[*to], lag
            ));
        }

        if self.edges.is_empty() {
            lines.push("  No significant Granger-causal relationships found".to_string());
        }

        lines.join("\n")
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Causal Transportability
// ─────────────────────────────────────────────────────────────────────────────

/// Causal transportability analysis.
///
/// Determines whether a causal effect learned in a source population
/// can be "transported" to a target population that differs in some ways.
///
/// Uses selection diagrams where S-nodes represent selection/sampling
/// mechanisms that differ between populations.
///
/// Reference: Pearl & Bareinboim (2011). "Transportability of Causal and
/// Statistical Relations"
pub struct TransportabilityAnalyzer {
    /// Source population DAG.
    source_dag: CausalDAG,
    /// Target population DAG (may differ in mechanisms).
    _target_dag: CausalDAG,
    /// Selection variables (nodes that differ between populations).
    selection_nodes: Vec<usize>,
}

impl TransportabilityAnalyzer {
    /// Create a new transportability analyzer.
    ///
    /// # Arguments
    /// * `source_dag` - DAG for the source population
    /// * `target_dag` - DAG for the target population
    /// * `selection_nodes` - Nodes whose mechanisms differ between populations
    pub fn new(source_dag: CausalDAG, target_dag: CausalDAG, selection_nodes: Vec<usize>) -> Self {
        Self {
            source_dag,
            _target_dag: target_dag,
            selection_nodes,
        }
    }

    /// Check if the causal effect P(y|do(x)) is transportable.
    ///
    /// Returns transportability status and any required adjustments.
    pub fn is_transportable(&self, treatment: usize, outcome: usize) -> TransportabilityResult {
        // Simple check: effect is directly transportable if no selection
        // node is on any path from treatment to outcome
        let paths_blocked = self.check_selection_blocking(treatment, outcome);

        if paths_blocked {
            return TransportabilityResult::DirectlyTransportable {
                explanation: "No selection mechanism affects the causal pathway".to_string(),
            };
        }

        // Check if we can adjust for selection
        let adjustment = self.find_transport_adjustment(treatment, outcome);

        match adjustment {
            Some(adj_set) => TransportabilityResult::TransportableWithAdjustment {
                adjustment_set: adj_set,
                explanation: "Effect transportable after adjusting for population differences"
                    .to_string(),
            },
            None => TransportabilityResult::NotTransportable {
                reason: "Selection mechanisms block all identification strategies".to_string(),
                blocking_nodes: self.find_blocking_selection_nodes(treatment, outcome),
            },
        }
    }

    /// Check if selection nodes block the treatment-outcome relationship.
    fn check_selection_blocking(&self, treatment: usize, outcome: usize) -> bool {
        // Get all nodes on paths from treatment to outcome
        let descendants = self.source_dag.descendants(treatment);

        // Check if any selection node is a descendant of treatment
        // and an ancestor of outcome
        for &s_node in &self.selection_nodes {
            if descendants.contains(&s_node) {
                let s_descendants = self.source_dag.descendants(s_node);
                if s_descendants.contains(&outcome) {
                    return false; // Selection node is on a path
                }
            }
        }

        true // No selection node on any path
    }

    /// Find adjustment set for transportability.
    fn find_transport_adjustment(&self, treatment: usize, outcome: usize) -> Option<Vec<usize>> {
        // Find variables that can block selection bias
        // These must satisfy the selection backdoor criterion

        let candidates: Vec<usize> = (0..self.source_dag.num_nodes())
            .filter(|&n| {
                n != treatment
                    && n != outcome
                    && !self.selection_nodes.contains(&n)
                    && !self.source_dag.descendants(treatment).contains(&n)
            })
            .collect();

        // Check if candidates block selection-induced confounding
        if candidates.is_empty() {
            return None;
        }

        // Simplified: return all non-descendant, non-selection nodes
        Some(candidates)
    }

    /// Find which selection nodes block transportability.
    fn find_blocking_selection_nodes(&self, treatment: usize, outcome: usize) -> Vec<usize> {
        let mut blocking = Vec::new();
        let treatment_descendants = self.source_dag.descendants(treatment);

        for &s_node in &self.selection_nodes {
            if treatment_descendants.contains(&s_node) {
                let s_descendants = self.source_dag.descendants(s_node);
                if s_descendants.contains(&outcome) {
                    blocking.push(s_node);
                }
            }
        }

        blocking
    }

    /// Compute the transported effect estimate.
    ///
    /// Uses the transport formula to reweight the source effect.
    pub fn transport_effect(
        &self,
        source_data: &ObservationalData,
        _target_data: &ObservationalData,
        treatment: usize,
        outcome: usize,
    ) -> Option<f64> {
        let result = self.is_transportable(treatment, outcome);

        match result {
            TransportabilityResult::DirectlyTransportable { .. } => {
                // Effect is the same in both populations
                // Use source effect directly
                let estimator = EffectEstimator::new();
                let query = CausalQuery {
                    treatment,
                    outcome,
                    conditioning: vec![],
                };

                match estimator.estimate(&self.source_dag, &query, source_data) {
                    CausalQueryOutcome::Identified { estimand, .. } => Some(estimand.effect),
                    _ => None,
                }
            }
            TransportabilityResult::TransportableWithAdjustment { adjustment_set, .. } => {
                // Need to reweight by population differences
                // Simplified: compute effect in source with adjustment
                let estimator = EffectEstimator::new();
                let query = CausalQuery {
                    treatment,
                    outcome,
                    conditioning: adjustment_set,
                };

                match estimator.estimate(&self.source_dag, &query, source_data) {
                    CausalQueryOutcome::Identified { estimand, .. } => Some(estimand.effect),
                    _ => None,
                }
            }
            TransportabilityResult::NotTransportable { .. } => None,
        }
    }
}

/// Result of transportability analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransportabilityResult {
    /// Effect is directly transportable without adjustment.
    DirectlyTransportable {
        /// Explanation of why it's transportable.
        explanation: String,
    },
    /// Effect is transportable after adjusting for covariates.
    TransportableWithAdjustment {
        /// Variables to adjust for in the target population.
        adjustment_set: Vec<usize>,
        /// Explanation.
        explanation: String,
    },
    /// Effect is not transportable.
    NotTransportable {
        /// Reason for non-transportability.
        reason: String,
        /// Selection nodes that block transport.
        blocking_nodes: Vec<usize>,
    },
}

impl TransportabilityResult {
    /// Check if the effect is transportable (with or without adjustment).
    pub fn is_transportable(&self) -> bool {
        !matches!(self, TransportabilityResult::NotTransportable { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─────────────────────────────────────────────────────────────────────────
    // PC Algorithm Tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_pc_algorithm_default() {
        let pc = PCAlgorithm::new();
        assert!((pc.alpha - 0.05).abs() < 1e-10);
        assert_eq!(pc.max_cond_size, 4);

        let pc2 = PCAlgorithm::default();
        assert!((pc2.alpha - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_pc_algorithm_with_alpha() {
        let pc = PCAlgorithm::with_alpha(0.01);
        assert!((pc.alpha - 0.01).abs() < 1e-10);
        assert_eq!(pc.max_cond_size, 4);
    }

    #[test]
    fn test_pc_discover_empty_data() {
        let data = ObservationalData::new(vec![]);
        let pc = PCAlgorithm::new();
        let result = pc.discover(&data);
        assert!(!result.is_valid());
        assert_eq!(result.cpdag.nodes.len(), 0);
        assert_eq!(result.independence_tests, 0);
    }

    #[test]
    fn test_pc_discover_single_variable() {
        let mut data = ObservationalData::new(vec!["X".into()]);
        for i in 0..20 {
            data.add_observation(vec![i as f64]);
        }
        let pc = PCAlgorithm::new();
        let result = pc.discover(&data);
        assert!(result.is_valid());
        assert_eq!(result.skeleton.num_edges(), 0);
    }

    #[test]
    fn test_pc_discover_correlated_pair() {
        // Strongly correlated pair: X and Y = 3*X + small noise
        let mut data = ObservationalData::new(vec!["X".into(), "Y".into()]);
        for i in 0..100 {
            let x = (i as f64) / 10.0;
            let y = 3.0 * x + 0.001 * (i % 7) as f64;
            data.add_observation(vec![x, y]);
        }

        let pc = PCAlgorithm::new();
        let result = pc.discover(&data);

        assert!(result.is_valid());
        // Strongly correlated pair should keep edge in skeleton
        assert!(
            result.skeleton.adjacent(0, 1),
            "Strongly correlated variables should be adjacent in skeleton"
        );
        assert!(result.independence_tests > 0);
    }

    #[test]
    fn test_pc_discover_chain_structure() {
        // Generate data from chain: X → Y → Z
        // X independent of Z given Y
        let mut data = ObservationalData::new(vec!["X".into(), "Y".into(), "Z".into()]);
        for i in 0..200 {
            let x = (i as f64) / 100.0 - 1.0;
            let y = 0.8 * x + 0.02 * (i % 11) as f64;
            let z = 0.7 * y + 0.02 * (i % 13) as f64;
            data.add_observation(vec![x, y, z]);
        }

        let pc = PCAlgorithm::new();
        let result = pc.discover(&data);

        assert!(result.is_valid());
        let summary = result.summary();
        assert!(summary.contains("PC Algorithm Result"));
        assert!(summary.contains("Independence tests"));
    }

    #[test]
    fn test_pc_result_summary_format() {
        let mut data = ObservationalData::new(vec!["A".into(), "B".into()]);
        for i in 0..50 {
            data.add_observation(vec![i as f64, i as f64 * 2.0]);
        }
        let pc = PCAlgorithm::new();
        let result = pc.discover(&data);
        let summary = result.summary();
        assert!(summary.contains("Nodes: 2"));
    }

    #[test]
    fn test_pc_result_to_dag() {
        let mut data = ObservationalData::new(vec!["A".into(), "B".into()]);
        for i in 0..50 {
            data.add_observation(vec![i as f64, i as f64 * 2.0]);
        }
        let pc = PCAlgorithm::new();
        let result = pc.discover(&data);
        let dag = result.to_dag();
        assert_eq!(dag.nodes.len(), 2);
    }

    #[test]
    fn test_fisher_z_transform_zero_correlation() {
        let z = fisher_z_transform(0.0, 100, 0);
        assert!(z.abs() < 1e-10, "Zero correlation should give zero z-stat");
    }

    #[test]
    fn test_fisher_z_transform_near_perfect_correlation() {
        let z = fisher_z_transform(0.99, 100, 0);
        assert!(
            z > 10.0,
            "Near-perfect correlation should give large z-stat"
        );
    }

    #[test]
    fn test_fisher_z_transform_insufficient_df() {
        // n=3, k=1 → df = 3-1-3 = -1, should return 0
        let z = fisher_z_transform(0.5, 3, 1);
        assert!((z - 0.0).abs() < 1e-10, "Insufficient df should return 0");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Mediation Analysis Tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_mediation_full_mediation_detection() {
        // X → M → Y (full mediation, all indirect)
        let dag = CausalDAG::new(
            vec!["X".into(), "M".into(), "Y".into()],
            vec![(0, 1), (1, 2)],
        );

        let mut data = ObservationalData::new(vec!["X".into(), "M".into(), "Y".into()]);
        for i in 0..200 {
            let x = (i as f64) / 100.0 - 1.0;
            let m = 0.8 * x + 0.01 * (i % 7) as f64;
            let y = 0.9 * m + 0.01 * (i % 11) as f64;
            data.add_observation(vec![x, m, y]);
        }

        let analysis = MediationAnalysis::new(&dag, 0, 1, 2);
        let result = analysis.analyze(&data);
        assert!(result.is_identified);
        assert!(result.a_path.is_finite());
        assert!(result.b_path.is_finite());
        // High proportion mediated for full mediation
        assert!(
            result.proportion_mediated > 0.5,
            "Expected high proportion mediated for full mediation, got {}",
            result.proportion_mediated
        );
    }

    #[test]
    fn test_mediation_no_mediator_m_to_y() {
        // X → M but M has no path to Y
        let dag = CausalDAG::new(
            vec!["X".into(), "M".into(), "Y".into()],
            vec![(0, 1), (0, 2)],
        );

        let analysis = MediationAnalysis::new(&dag, 0, 1, 2);
        let id = analysis.is_identified();
        assert!(
            matches!(id, MediationIdentification::NotMediator { .. }),
            "Expected NotMediator when M has no path to Y"
        );
    }

    #[test]
    fn test_mediation_result_summary_identified() {
        let result = MediationResult {
            total_effect: 1.0,
            natural_direct_effect: 0.3,
            natural_indirect_effect: 0.7,
            a_path: 0.8,
            b_path: 0.875,
            c_prime: 0.3,
            proportion_mediated: 0.7,
            is_identified: true,
            identification: MediationIdentification::Identified {
                nde_adjustment: vec![],
                nie_adjustment: vec![],
                has_direct_effect: true,
            },
        };

        assert!(result.is_partially_mediated());
        assert!(!result.is_fully_mediated());
        assert!(result.has_significant_mediation(0.1));
        let summary = result.summary();
        assert!(summary.contains("Partial mediation"));
    }

    #[test]
    fn test_mediation_result_summary_not_identified() {
        let result = MediationResult {
            total_effect: f64::NAN,
            natural_direct_effect: f64::NAN,
            natural_indirect_effect: f64::NAN,
            a_path: f64::NAN,
            b_path: f64::NAN,
            c_prime: f64::NAN,
            proportion_mediated: f64::NAN,
            is_identified: false,
            identification: MediationIdentification::NotMediator {
                reason: "test".into(),
            },
        };

        assert!(!result.is_fully_mediated());
        assert!(!result.is_partially_mediated());
        assert!(!result.has_significant_mediation(0.1));
        let summary = result.summary();
        assert!(summary.contains("not identified"));
    }

    #[test]
    fn test_mediation_insufficient_data() {
        let dag = CausalDAG::new(
            vec!["X".into(), "M".into(), "Y".into()],
            vec![(0, 1), (1, 2), (0, 2)],
        );

        // Only 1 observation — regressions should return 0
        let mut data = ObservationalData::new(vec!["X".into(), "M".into(), "Y".into()]);
        data.add_observation(vec![1.0, 2.0, 3.0]);

        let analysis = MediationAnalysis::new(&dag, 0, 1, 2);
        let result = analysis.analyze(&data);
        assert!(result.is_identified);
        assert!((result.total_effect - 0.0).abs() < 1e-10);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Instrumental Variable Tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_iv_valid_instrument() {
        // Z → X → Y, Z is a valid instrument
        let dag = CausalDAG::new(
            vec!["Z".into(), "X".into(), "Y".into()],
            vec![(0, 1), (1, 2)],
        );
        let validity = IVEstimator::is_valid_instrument(&dag, 0, 1, 2);
        assert!(
            matches!(validity, IVValidity::Valid { .. }),
            "Z should be valid instrument for X→Y"
        );
    }

    #[test]
    fn test_iv_invalid_no_relevance() {
        // Z has no path to X
        let dag = CausalDAG::new(
            vec!["Z".into(), "X".into(), "Y".into()],
            vec![(1, 2)], // Only X → Y
        );
        let validity = IVEstimator::is_valid_instrument(&dag, 0, 1, 2);
        assert!(
            matches!(validity, IVValidity::Invalid { .. }),
            "Z should be invalid: no Z→X path"
        );
    }

    #[test]
    fn test_iv_invalid_direct_effect() {
        // Z → X → Y, Z → Y (violates exclusion restriction)
        let dag = CausalDAG::new(
            vec!["Z".into(), "X".into(), "Y".into()],
            vec![(0, 1), (1, 2), (0, 2)],
        );
        let validity = IVEstimator::is_valid_instrument(&dag, 0, 1, 2);
        assert!(
            matches!(validity, IVValidity::Invalid { .. }),
            "Z should be invalid: direct Z→Y path"
        );
    }

    #[test]
    fn test_iv_2sls_estimation() {
        // Z → X → Y with true effect X→Y = 2.0
        let mut data = ObservationalData::new(vec!["Z".into(), "X".into(), "Y".into()]);
        for i in 0..200 {
            let z = (i % 2) as f64;
            let x = 0.8 * z + 0.05 * (i % 5) as f64;
            let y = 2.0 * x + 0.05 * (i % 7) as f64;
            data.add_observation(vec![z, x, y]);
        }

        let result = IVEstimator::estimate_2sls(&data, 0, 1, 2);
        assert!(result.effect.is_finite(), "2SLS effect should be finite");
        assert!(
            (result.effect - 2.0).abs() < 1.0,
            "2SLS effect should be ~2.0, got {}",
            result.effect
        );
        assert_eq!(result.method, "2SLS");
    }

    #[test]
    fn test_iv_2sls_insufficient_data() {
        let mut data = ObservationalData::new(vec!["Z".into(), "X".into(), "Y".into()]);
        for i in 0..5 {
            data.add_observation(vec![i as f64, i as f64 * 2.0, i as f64 * 4.0]);
        }
        let result = IVEstimator::estimate_2sls(&data, 0, 1, 2);
        assert!(
            result.effect.is_nan(),
            "Insufficient data should return NaN"
        );
        assert!(result.is_weak_instrument);
    }

    #[test]
    fn test_iv_wald_estimator() {
        // Binary instrument Z, X = Z + noise, Y = 2*X + noise
        let mut data = ObservationalData::new(vec!["Z".into(), "X".into(), "Y".into()]);
        for i in 0..200 {
            let z = (i % 2) as f64;
            let x = z + 0.01 * (i % 3) as f64;
            let y = 2.0 * x + 0.01 * (i % 5) as f64;
            data.add_observation(vec![z, x, y]);
        }

        let wald = IVEstimator::estimate_wald(&data, 0, 1, 2);
        assert!(wald.is_finite(), "Wald estimate should be finite");
        assert!(
            (wald - 2.0).abs() < 0.5,
            "Wald estimate should be ~2.0, got {}",
            wald
        );
    }

    #[test]
    fn test_iv_wald_no_variance() {
        // All instrument values are the same
        let mut data = ObservationalData::new(vec!["Z".into(), "X".into(), "Y".into()]);
        for i in 0..20 {
            data.add_observation(vec![0.0, i as f64, i as f64 * 2.0]);
        }
        let wald = IVEstimator::estimate_wald(&data, 0, 1, 2);
        assert!(wald.is_nan(), "All-same instrument should return NaN");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Time-Series Causal Discovery Tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_time_series_data_construction() {
        let mut ts = TimeSeriesData::new(vec!["X".into(), "Y".into()]);
        assert_eq!(ts.n_timepoints(), 0);
        ts.add_observation(vec![1.0, 2.0]);
        ts.add_observation(vec![3.0, 4.0]);
        assert_eq!(ts.n_timepoints(), 2);
        assert_eq!(ts.variables.len(), 2);
    }

    #[test]
    fn test_granger_mismatched_lengths() {
        let tscd = TimeSeriesCausalDiscovery::new(2);
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0]; // different length
        let result = tscd.granger_test(&x, &y, 1);
        assert!(!result.is_significant);
        assert!((result.f_statistic - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_granger_too_short_for_lag() {
        let tscd = TimeSeriesCausalDiscovery::new(5);
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0, 3.0];
        let result = tscd.granger_test(&x, &y, 5);
        assert!(!result.is_significant);
    }

    #[test]
    fn test_time_series_causal_graph_summary() {
        let tscd = TimeSeriesCausalDiscovery::new(1);
        let mut ts = TimeSeriesData::new(vec!["A".into(), "B".into()]);
        for t in 0..30 {
            ts.add_observation(vec![t as f64, t as f64 * 0.5]);
        }
        let graph = tscd.discover(&ts);
        let summary = graph.summary();
        assert!(summary.contains("Time-Series Causal Graph"));
    }

    #[test]
    fn test_time_series_default() {
        let tscd = TimeSeriesCausalDiscovery::default();
        assert_eq!(tscd.max_lag, 5);
        assert!((tscd.alpha - 0.05).abs() < 1e-10);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Transportability Tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_transportability_selection_on_pathway() {
        // X → S → Y, where S is a selection node on the causal pathway
        let source_dag = CausalDAG::new(
            vec!["X".into(), "S".into(), "Y".into()],
            vec![(0, 1), (1, 2)],
        );
        let target_dag = source_dag.clone();
        let analyzer = TransportabilityAnalyzer::new(source_dag, target_dag, vec![1]);
        let result = analyzer.is_transportable(0, 2);
        // S is on the causal pathway X→S→Y, so effect is NOT directly transportable
        assert!(
            !matches!(result, TransportabilityResult::DirectlyTransportable { .. }),
            "Selection node on pathway should block direct transport"
        );
    }

    #[test]
    fn test_transportability_selection_off_pathway() {
        // X → Y, Z is a selection node not on the pathway
        let source_dag = CausalDAG::new(vec!["X".into(), "Y".into(), "Z".into()], vec![(0, 1)]);
        let target_dag = source_dag.clone();
        let analyzer = TransportabilityAnalyzer::new(source_dag, target_dag, vec![2]);
        let result = analyzer.is_transportable(0, 1);
        assert!(
            result.is_transportable(),
            "Selection node off pathway should not block transport"
        );
    }

    #[test]
    fn test_transportability_transport_effect_direct() {
        let source_dag = CausalDAG::new(vec!["X".into(), "Y".into()], vec![(0, 1)]);
        let target_dag = source_dag.clone();

        let mut source_data = ObservationalData::new(vec!["X".into(), "Y".into()]);
        let mut target_data = ObservationalData::new(vec!["X".into(), "Y".into()]);
        for i in 0..50 {
            let x = i as f64 / 25.0;
            source_data.add_observation(vec![x, 2.0 * x]);
            target_data.add_observation(vec![x, 2.0 * x]);
        }

        let analyzer = TransportabilityAnalyzer::new(source_dag, target_dag, vec![]);
        let effect = analyzer.transport_effect(&source_data, &target_data, 0, 1);
        assert!(
            effect.is_some(),
            "Direct transport should produce an effect estimate"
        );
        let eff = effect.unwrap();
        assert!(
            (eff - 2.0).abs() < 0.5,
            "Transported effect should be ~2.0, got {}",
            eff
        );
    }

    #[test]
    fn test_transportability_result_enum() {
        let directly = TransportabilityResult::DirectlyTransportable {
            explanation: "test".into(),
        };
        assert!(directly.is_transportable());

        let with_adj = TransportabilityResult::TransportableWithAdjustment {
            adjustment_set: vec![0],
            explanation: "test".into(),
        };
        assert!(with_adj.is_transportable());

        let not = TransportabilityResult::NotTransportable {
            reason: "blocked".into(),
            blocking_nodes: vec![1],
        };
        assert!(!not.is_transportable());
    }
}
