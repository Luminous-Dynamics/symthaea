// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! TruePhiCalculator - Core IIT computation using Shannon entropy.

use crate::hdc::unified_hv::ContinuousHV;
use nalgebra::DMatrix;

use super::{EntropyConfig, JointDistribution, TruePartition, TruePhiResult, VectorDistribution};

/// Maximum graph size for the dense spectral heuristic. Larger inputs abstain
/// rather than paying unbounded cubic eigensolver cost.
const MAX_DENSE_SPECTRAL_NODES: usize = 256;
const SPECTRAL_SYMMETRY_TOLERANCE: f64 = 1e-10;
const SPECTRAL_RESIDUAL_TOLERANCE: f64 = 1e-8;
const SPECTRAL_EIGENGAP_TOLERANCE: f64 = 1e-8;
const SPECTRAL_NULLSPACE_TOLERANCE: f64 = 1e-10;

/// Project an MI reporting matrix into a weighted graph Laplacian.
///
/// The MI diagonal is marginal entropy, not adjacency. It is validated as a
/// finite non-negative reported value but intentionally excluded from degrees.
/// Off-diagonal weights are normalized by the largest edge to make the spectral
/// calculation invariant to a common positive scale and avoid overflow in degree
/// accumulation.
fn checked_mi_graph_laplacian(mi_matrix: &[Vec<f64>], n: usize) -> Option<DMatrix<f64>> {
    if n < 3 || n > MAX_DENSE_SPECTRAL_NODES || mi_matrix.len() != n {
        return None;
    }
    if mi_matrix.iter().any(|row| row.len() != n) {
        return None;
    }
    if mi_matrix
        .iter()
        .flatten()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return None;
    }

    let mut maximum_edge = 0.0_f64;
    for i in 0..n {
        for j in (i + 1)..n {
            let a = mi_matrix[i][j];
            let b = mi_matrix[j][i];
            let symmetry_scale = a.abs().max(b.abs()).max(1.0);
            if (a - b).abs() > SPECTRAL_SYMMETRY_TOLERANCE * symmetry_scale {
                return None;
            }
            maximum_edge = maximum_edge.max(a.max(b));
        }
    }

    // There is no meaningful spectral cut without at least one nonzero edge.
    if maximum_edge == 0.0 {
        return None;
    }

    let mut laplacian = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        let mut degree = 0.0;
        for j in 0..n {
            if i == j {
                continue;
            }
            // Scale before averaging; (a + b) could overflow for finite inputs.
            let weight =
                0.5 * (mi_matrix[i][j] / maximum_edge + mi_matrix[j][i] / maximum_edge);
            if !weight.is_finite() || weight < 0.0 {
                return None;
            }
            degree += weight;
            laplacian[(i, j)] = -weight;
        }
        if !degree.is_finite() {
            return None;
        }
        laplacian[(i, i)] = degree;
    }

    Some(laplacian)
}

/// Compute and qualify the Fiedler eigenpair with a dense symmetric solver.
///
/// Ordinary power iteration targets the largest-magnitude eigenmode, not the
/// second-smallest Laplacian eigenmode. The returned pair must be positive beyond
/// the numerical nullspace floor, separated from the next eigenvalue, orthogonal
/// to the constant vector, and independently satisfy the eigenpair residual gate.
/// Disconnected or spectrally ambiguous graphs abstain so another heuristic can
/// supply a candidate instead of fabricating a Fiedler cut.
fn checked_fiedler_eigenpair(laplacian: &DMatrix<f64>) -> Option<(f64, Vec<f64>)> {
    let n = laplacian.nrows();
    let max_iterations = n.saturating_mul(n).saturating_mul(20).max(1);
    checked_fiedler_eigenpair_with_budget(laplacian, max_iterations)
}

/// Testable bounded variant: exhausting the symmetric solver budget must abstain.
fn checked_fiedler_eigenpair_with_budget(
    laplacian: &DMatrix<f64>,
    max_iterations: usize,
) -> Option<(f64, Vec<f64>)> {
    let n = laplacian.nrows();
    if n < 3 || n != laplacian.ncols() || max_iterations == 0 {
        return None;
    }

    // Use the fallible API: a decomposition that exhausts its iteration budget
    // is a missing candidate, not an apparently verified Fiedler vector.
    let eigen = nalgebra::linalg::SymmetricEigen::try_new(
        laplacian.clone(),
        f64::EPSILON,
        max_iterations,
    )?;
    if eigen.eigenvalues.iter().any(|value| !value.is_finite())
        || eigen.eigenvectors.iter().any(|value| !value.is_finite())
    {
        return None;
    }

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| eigen.eigenvalues[a].total_cmp(&eigen.eigenvalues[b]));
    let fiedler_index = order[1];
    let next_index = order[2];
    let lambda = eigen.eigenvalues[fiedler_index];
    let next_lambda = eigen.eigenvalues[next_index];
    let spectral_scale = eigen
        .eigenvalues
        .iter()
        .map(|value| value.abs())
        .fold(1.0_f64, f64::max);

    // λ₂≈0 means disconnected or numerically disconnected; λ₂≈λ₃ means the
    // Fiedler direction is non-unique/ill-conditioned (e.g. a complete graph).
    if lambda <= SPECTRAL_NULLSPACE_TOLERANCE * spectral_scale
        || next_lambda - lambda <= SPECTRAL_EIGENGAP_TOLERANCE * spectral_scale
    {
        return None;
    }

    let vector: Vec<f64> = eigen
        .eigenvectors
        .column(fiedler_index)
        .iter()
        .copied()
        .collect();
    let vector_norm = vector.iter().map(|value| value * value).sum::<f64>().sqrt();
    if !vector_norm.is_finite() || vector_norm <= f64::MIN_POSITIVE {
        return None;
    }

    // Independently compute ||L v - λv|| / ((||L||_F + |λ|) ||v||).
    let mut residual_squared = 0.0;
    let mut matrix_norm_squared = 0.0;
    for i in 0..n {
        let mut row_product = 0.0;
        for j in 0..n {
            let value = laplacian[(i, j)];
            row_product += value * vector[j];
            matrix_norm_squared += value * value;
        }
        let residual = row_product - lambda * vector[i];
        residual_squared += residual * residual;
    }
    let matrix_norm = matrix_norm_squared.sqrt();
    let residual_scale = ((matrix_norm + lambda.abs()) * vector_norm).max(f64::MIN_POSITIVE);
    let relative_residual = residual_squared.sqrt() / residual_scale;
    let constant_overlap = vector.iter().sum::<f64>().abs()
        / ((n as f64).sqrt() * vector_norm).max(f64::MIN_POSITIVE);

    if !relative_residual.is_finite()
        || relative_residual > SPECTRAL_RESIDUAL_TOLERANCE
        || !constant_overlap.is_finite()
        || constant_overlap > SPECTRAL_RESIDUAL_TOLERANCE
    {
        return None;
    }

    Some((lambda, vector))
}

/// Calculator for true integrated information using Shannon entropy
#[derive(Debug, Clone)]
pub struct TruePhiCalculator {
    pub(crate) config: EntropyConfig,
}

impl TruePhiCalculator {
    /// Create a new calculator with default config
    pub fn new() -> Self {
        Self {
            config: EntropyConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: EntropyConfig) -> Self {
        Self { config }
    }

    /// Get the log function based on config
    pub(crate) fn log(&self, x: f64) -> f64 {
        if self.config.use_bits {
            x.log2()
        } else {
            x.ln()
        }
    }

    /// Convert HDC vector to probability distribution via binning
    pub fn to_distribution(&self, hv: &ContinuousHV) -> VectorDistribution {
        VectorDistribution::from_hv(hv, self.config.num_bins)
    }

    /// Compute Shannon entropy H(X) from a distribution
    ///
    /// H(X) = -Σ p(x) log p(x)
    pub fn entropy_from_distribution(&self, dist: &VectorDistribution) -> f64 {
        let mut h = 0.0;
        for &p in &dist.probabilities {
            if p > 0.0 {
                h -= p * self.log(p);
            }
        }
        h
    }

    /// Compute Shannon entropy H(X) directly from a hypervector
    pub fn entropy(&self, hv: &ContinuousHV) -> f64 {
        let dist = self.to_distribution(hv);
        self.entropy_from_distribution(&dist)
    }

    /// Compute joint entropy H(X,Y) via 2D histogram
    pub fn joint_entropy(&self, hv1: &ContinuousHV, hv2: &ContinuousHV) -> f64 {
        let joint = JointDistribution::from_hvs(hv1, hv2, self.config.num_bins);

        let mut h = 0.0;
        for row in &joint.probabilities {
            for &p in row {
                if p > 0.0 {
                    h -= p * self.log(p);
                }
            }
        }
        h
    }

    /// Compute mutual information I(X;Y) = H(X) + H(Y) - H(X,Y)
    ///
    /// Mutual information measures how much knowing X reduces uncertainty about Y.
    /// I(X;Y) = 0 for independent variables
    /// I(X;Y) = H(X) = H(Y) for perfectly correlated variables
    pub fn mutual_information(&self, hv1: &ContinuousHV, hv2: &ContinuousHV) -> f64 {
        let h_x = self.entropy(hv1);
        let h_y = self.entropy(hv2);
        let h_xy = self.joint_entropy(hv1, hv2);

        // I(X;Y) = H(X) + H(Y) - H(X,Y)
        // Due to numerical precision, ensure non-negative
        (h_x + h_y - h_xy).max(0.0)
    }

    /// Compute effective information as sum of MI between all component pairs
    ///
    /// EI = Σ_{i<j} I(X_i; X_j)
    ///
    /// This measures the total pairwise information integration.
    pub fn effective_information(&self, components: &[ContinuousHV]) -> f64 {
        if components.len() < 2 {
            return 0.0;
        }

        let mut ei = 0.0;
        for i in 0..components.len() {
            for j in (i + 1)..components.len() {
                ei += self.mutual_information(&components[i], &components[j]);
            }
        }
        ei
    }

    /// Compute effective information for a partition
    pub(crate) fn partition_effective_information(
        &self,
        components: &[ContinuousHV],
        partition: &TruePartition,
    ) -> f64 {
        // EI(partition) = EI(part_a) + EI(part_b)
        // This is the sum of information within each part, ignoring cross-part info

        let mut ei = 0.0;

        // EI within part A
        for i in 0..partition.part_a.len() {
            for j in (i + 1)..partition.part_a.len() {
                let idx_i = partition.part_a[i];
                let idx_j = partition.part_a[j];
                ei += self.mutual_information(&components[idx_i], &components[idx_j]);
            }
        }

        // EI within part B
        for i in 0..partition.part_b.len() {
            for j in (i + 1)..partition.part_b.len() {
                let idx_i = partition.part_b[i];
                let idx_j = partition.part_b[j];
                ei += self.mutual_information(&components[idx_i], &components[idx_j]);
            }
        }

        ei
    }

    /// Build mutual information matrix for all component pairs
    pub(crate) fn build_mi_matrix(&self, components: &[ContinuousHV]) -> Vec<Vec<f64>> {
        let n = components.len();
        let mut matrix = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in (i + 1)..n {
                let mi = self.mutual_information(&components[i], &components[j]);
                matrix[i][j] = mi;
                matrix[j][i] = mi; // Symmetric
            }
            // Diagonal: self-information = entropy
            matrix[i][i] = self.entropy(&components[i]);
        }

        matrix
    }

    /// Find the Minimum Information Partition (MIP) using true entropy measures
    ///
    /// The MIP is the partition that minimizes information loss.
    /// Φ = EI(system) - EI(MIP)
    pub fn find_true_mip(&self, components: &[ContinuousHV]) -> (TruePartition, f64) {
        let n = components.len();

        if n < 2 {
            return (
                TruePartition {
                    part_a: (0..n).collect(),
                    part_b: vec![],
                },
                0.0,
            );
        }

        if n == 2 {
            // Only one partition possible: {0} | {1}
            let partition = TruePartition {
                part_a: vec![0],
                part_b: vec![1],
            };
            let ei = self.partition_effective_information(components, &partition);
            return (partition, ei);
        }

        // For small N (≤8), exhaustive search
        if n <= 8 {
            self.exhaustive_mip_search(components)
        } else {
            // For large N, use heuristic search
            self.heuristic_mip_search(components)
        }
    }

    /// Exhaustive MIP search for small systems (N ≤ 8)
    fn exhaustive_mip_search(&self, components: &[ContinuousHV]) -> (TruePartition, f64) {
        let n = components.len();
        let mut min_ei = f64::MAX;
        let mut mip = TruePartition {
            part_a: vec![0],
            part_b: (1..n).collect(),
        };

        // Iterate through all bipartitions
        // Use bit masks: for each subset of {0, 1, ..., n-1}
        for mask in 1..(1 << n) - 1 {
            let partition = TruePartition::from_mask(mask, n);

            // Skip trivial partitions (one part empty)
            if partition.part_a.is_empty() || partition.part_b.is_empty() {
                continue;
            }

            let ei = self.partition_effective_information(components, &partition);

            if ei < min_ei {
                min_ei = ei;
                mip = partition;
            }
        }

        (mip, min_ei)
    }

    /// Heuristic MIP search for large systems (N > 8)
    ///
    /// Uses multiple strategies to find the minimum information partition:
    /// 1. Spectral clustering based on MI matrix (Fiedler vector)
    /// 2. Simulated annealing for global optimization
    /// 3. Greedy bisection with local search refinement
    /// 4. MI-based heuristics (total MI split, index split, highest MI pair)
    fn heuristic_mip_search(&self, components: &[ContinuousHV]) -> (TruePartition, f64) {
        let n = components.len();
        let mi_matrix = self.build_mi_matrix(components);

        let mut candidates = Vec::new();

        // Strategy 1: Spectral clustering via Fiedler vector
        if let Some(partition) = self.spectral_partition(&mi_matrix, n) {
            candidates.push(partition);
        }

        // Strategy 2: Simulated annealing
        if let Some(partition) = self.simulated_annealing_partition(components, n) {
            candidates.push(partition);
        }

        // Strategy 3: Greedy bisection with local search
        let greedy = self.greedy_bisection_partition(&mi_matrix, n);
        let refined = self.local_search_refinement(components, greedy);
        candidates.push(refined);

        // Strategy 4: Split by total MI (separate high-MI from low-MI components)
        let total_mi: Vec<f64> = (0..n).map(|i| mi_matrix[i].iter().sum::<f64>()).collect();
        let mean_mi = total_mi.iter().sum::<f64>() / n as f64;
        let part_a: Vec<usize> = (0..n).filter(|&i| total_mi[i] >= mean_mi).collect();
        let part_b: Vec<usize> = (0..n).filter(|&i| total_mi[i] < mean_mi).collect();
        if !part_a.is_empty() && !part_b.is_empty() {
            candidates.push(TruePartition { part_a, part_b });
        }

        // Strategy 5: Split in half by index
        let mid = n / 2;
        candidates.push(TruePartition {
            part_a: (0..mid).collect(),
            part_b: (mid..n).collect(),
        });

        // Strategy 6: Greedy clustering based on highest MI pair
        let mut used = vec![false; n];
        let mut part_a = Vec::new();
        let mut part_b = Vec::new();

        let mut max_mi = 0.0;
        let mut best_i = 0;
        let mut best_j = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                if mi_matrix[i][j] > max_mi {
                    max_mi = mi_matrix[i][j];
                    best_i = i;
                    best_j = j;
                }
            }
        }

        part_a.push(best_i);
        part_b.push(best_j);
        used[best_i] = true;
        used[best_j] = true;

        for k in 0..n {
            if used[k] {
                continue;
            }

            let mi_to_a: f64 = part_a.iter().map(|&i| mi_matrix[k][i]).sum();
            let mi_to_b: f64 = part_b.iter().map(|&i| mi_matrix[k][i]).sum();

            if mi_to_a >= mi_to_b {
                part_a.push(k);
            } else {
                part_b.push(k);
            }
            used[k] = true;
        }

        candidates.push(TruePartition { part_a, part_b });

        // Find partition with minimum EI
        let mut min_ei = f64::MAX;
        let mut mip = candidates[0].clone();

        for partition in &candidates {
            let ei = self.partition_effective_information(components, partition);
            if ei < min_ei {
                min_ei = ei;
                mip = partition.clone();
            }
        }

        (mip, min_ei)
    }

    /// Spectral partition from a checked second-smallest Laplacian eigenpair.
    ///
    /// The graph uses only off-diagonal pairwise mutual information. Diagonal
    /// entries store marginal entropy and are not graph self-edges. This dense
    /// solver is deliberately bounded; for larger graphs the heuristic abstains
    /// here and leaves candidate generation to the other MIP strategies.
    pub(crate) fn spectral_partition(
        &self,
        mi_matrix: &[Vec<f64>],
        n: usize,
    ) -> Option<TruePartition> {
        let laplacian = checked_mi_graph_laplacian(mi_matrix, n)?;
        let (_eigenvalue, fiedler) = checked_fiedler_eigenpair(&laplacian)?;

        // The Fiedler vector's sign is arbitrary, so either orientation is valid.
        let mut part_a = Vec::new();
        let mut part_b = Vec::new();
        for (index, value) in fiedler.iter().enumerate() {
            if *value >= 0.0 {
                part_a.push(index);
            } else {
                part_b.push(index);
            }
        }

        // Some valid Fiedler vectors contain near-zero entries. If sign thresholding
        // degenerates, use a deterministic median cut rather than an empty side.
        if part_a.is_empty() || part_b.is_empty() {
            let mut indices: Vec<usize> = (0..n).collect();
            indices.sort_by(|&a, &b| fiedler[a].total_cmp(&fiedler[b]));
            let mid = n / 2;
            return Some(TruePartition {
                part_a: indices[..mid].to_vec(),
                part_b: indices[mid..].to_vec(),
            });
        }

        Some(TruePartition { part_a, part_b })
    }

    /// Simulated annealing for MIP search
    pub(crate) fn simulated_annealing_partition(
        &self,
        components: &[ContinuousHV],
        n: usize,
    ) -> Option<TruePartition> {
        if n < 3 {
            return None;
        }

        let mut assignment = vec![false; n];
        for i in 0..(n / 2) {
            assignment[i] = true;
        }

        let mut current = self.assignment_to_partition(&assignment);
        let mut current_ei = self.partition_effective_information(components, &current);
        let mut best_partition = current.clone();
        let mut best_ei = current_ei;

        let initial_temp = 1.0;
        let final_temp = 0.001;
        let cooling_rate = 0.95;
        let iterations_per_temp = n * 2;

        let mut temp = initial_temp;
        let mut rng_state = 42u64;

        while temp > final_temp {
            for _ in 0..iterations_per_temp {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let idx = (rng_state as usize) % n;

                let current_side = assignment[idx];
                let count_same = assignment.iter().filter(|&&x| x == current_side).count();
                if count_same <= 1 {
                    continue;
                }

                assignment[idx] = !assignment[idx];
                let new_partition = self.assignment_to_partition(&assignment);
                let new_ei = self.partition_effective_information(components, &new_partition);

                let delta = new_ei - current_ei;
                let accept = if delta < 0.0 {
                    true
                } else {
                    rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let r = (rng_state as f64) / (u64::MAX as f64);
                    r < (-delta / temp).exp()
                };

                if accept {
                    current = new_partition;
                    current_ei = new_ei;
                    if current_ei < best_ei {
                        best_partition = current.clone();
                        best_ei = current_ei;
                    }
                } else {
                    assignment[idx] = !assignment[idx];
                }
            }
            temp *= cooling_rate;
        }

        Some(best_partition)
    }

    /// Convert boolean assignment to TruePartition
    pub(crate) fn assignment_to_partition(&self, assignment: &[bool]) -> TruePartition {
        let mut part_a = Vec::new();
        let mut part_b = Vec::new();
        for (i, &in_a) in assignment.iter().enumerate() {
            if in_a {
                part_a.push(i);
            } else {
                part_b.push(i);
            }
        }
        TruePartition { part_a, part_b }
    }

    /// Greedy bisection partition
    pub(crate) fn greedy_bisection_partition(
        &self,
        mi_matrix: &[Vec<f64>],
        n: usize,
    ) -> TruePartition {
        let mut min_mi = f64::MAX;
        let mut seed_a = 0;
        let mut seed_b = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                if mi_matrix[i][j] < min_mi {
                    min_mi = mi_matrix[i][j];
                    seed_a = i;
                    seed_b = j;
                }
            }
        }

        let mut part_a = vec![seed_a];
        let mut part_b = vec![seed_b];
        let mut assigned = vec![false; n];
        assigned[seed_a] = true;
        assigned[seed_b] = true;

        for _ in 2..n {
            let mut best_idx = 0;
            let mut best_to_a = false;
            let mut best_cost = f64::MAX;

            for i in 0..n {
                if assigned[i] {
                    continue;
                }

                let mi_to_a: f64 = part_a.iter().map(|&j| mi_matrix[i][j]).sum();
                let mi_to_b: f64 = part_b.iter().map(|&j| mi_matrix[i][j]).sum();

                if mi_to_a < best_cost {
                    best_cost = mi_to_a;
                    best_idx = i;
                    best_to_a = true;
                }
                if mi_to_b < best_cost {
                    best_cost = mi_to_b;
                    best_idx = i;
                    best_to_a = false;
                }
            }

            assigned[best_idx] = true;
            if best_to_a {
                part_a.push(best_idx);
            } else {
                part_b.push(best_idx);
            }
        }

        if part_a.is_empty() && !part_b.is_empty() {
            let moved = part_b.pop().expect("part_b verified non-empty");
            part_a.push(moved);
        } else if part_b.is_empty() && !part_a.is_empty() {
            let moved = part_a.pop().expect("part_a verified non-empty");
            part_b.push(moved);
        }

        TruePartition { part_a, part_b }
    }

    /// Local search refinement
    pub(crate) fn local_search_refinement(
        &self,
        components: &[ContinuousHV],
        initial: TruePartition,
    ) -> TruePartition {
        let mut current = initial;
        let mut current_ei = self.partition_effective_information(components, &current);
        let mut improved = true;

        while improved {
            improved = false;

            for i in 0..current.part_a.len() {
                if current.part_a.len() <= 1 {
                    break;
                }

                let elem = current.part_a[i];
                let mut new_a = current.part_a.clone();
                new_a.remove(i);
                let mut new_b = current.part_b.clone();
                new_b.push(elem);

                let new_partition = TruePartition {
                    part_a: new_a,
                    part_b: new_b,
                };
                let new_ei = self.partition_effective_information(components, &new_partition);

                if new_ei < current_ei {
                    current = new_partition;
                    current_ei = new_ei;
                    improved = true;
                    break;
                }
            }

            if improved {
                continue;
            }

            for i in 0..current.part_b.len() {
                if current.part_b.len() <= 1 {
                    break;
                }

                let elem = current.part_b[i];
                let mut new_b = current.part_b.clone();
                new_b.remove(i);
                let mut new_a = current.part_a.clone();
                new_a.push(elem);

                let new_partition = TruePartition {
                    part_a: new_a,
                    part_b: new_b,
                };
                let new_ei = self.partition_effective_information(components, &new_partition);

                if new_ei < current_ei {
                    current = new_partition;
                    current_ei = new_ei;
                    improved = true;
                    break;
                }
            }

            if improved {
                continue;
            }

            'swap: for i in 0..current.part_a.len() {
                for j in 0..current.part_b.len() {
                    let elem_a = current.part_a[i];
                    let elem_b = current.part_b[j];

                    let mut new_a = current.part_a.clone();
                    let mut new_b = current.part_b.clone();
                    new_a[i] = elem_b;
                    new_b[j] = elem_a;

                    let new_partition = TruePartition {
                        part_a: new_a,
                        part_b: new_b,
                    };
                    let new_ei = self.partition_effective_information(components, &new_partition);

                    if new_ei < current_ei {
                        current = new_partition;
                        current_ei = new_ei;
                        improved = true;
                        break 'swap;
                    }
                }
            }
        }

        current
    }

    /// Compute true Φ = EI(system) - EI(MIP)
    ///
    /// This is the core IIT calculation using genuine Shannon entropy.
    ///
    /// # Arguments
    /// * `components` - System components as hypervectors
    ///
    /// # Returns
    /// Detailed Φ result including:
    /// - phi: The integrated information value
    /// - system_ei: Whole system effective information
    /// - mip_ei: MIP effective information
    /// - mip: The minimum information partition
    /// - component_entropies: Individual H(X_i) values
    pub fn compute_true_phi(&self, components: &[ContinuousHV]) -> TruePhiResult {
        if components.len() < 2 {
            return TruePhiResult {
                phi: 0.0,
                system_ei: 0.0,
                mip_ei: 0.0,
                mip: TruePartition {
                    part_a: (0..components.len()).collect(),
                    part_b: vec![],
                },
                component_entropies: components.iter().map(|c| self.entropy(c)).collect(),
                mutual_information_matrix: vec![],
            };
        }

        // 1. Compute component entropies
        let component_entropies: Vec<f64> = components.iter().map(|c| self.entropy(c)).collect();

        // 2. Build MI matrix
        let mi_matrix = self.build_mi_matrix(components);

        // 3. Compute system effective information
        let system_ei = self.effective_information(components);

        // 4. Find MIP and its effective information
        let (mip, mip_ei) = self.find_true_mip(components);

        // 5. Φ = EI(system) - EI(MIP)
        let phi = (system_ei - mip_ei).max(0.0);

        TruePhiResult {
            phi,
            system_ei,
            mip_ei,
            mip,
            component_entropies,
            mutual_information_matrix: mi_matrix,
        }
    }

    /// Fast effective information (EI) estimation for real-time use.
    ///
    /// **WARNING**: This computes total mutual information, NOT IIT Φ.
    /// True Φ = system_EI - min_partition_EI (requires MIP search).
    /// This method skips the partition step and returns raw EI only.
    ///
    /// For actual Φ, use `TieredPhi` with `ExhaustivePartition` or
    /// `SampledPartition` tiers.
    #[deprecated(note = "returns raw EI, not IIT Φ — use TieredPhi for actual Φ")]
    pub fn compute_phi_fast(&self, components: &[ContinuousHV]) -> f64 {
        self.compute_effective_information_normalized(components)
    }

    /// Normalized effective information (total MI / theoretical max).
    ///
    /// Returns a [0, 1] scalar measuring raw information content.
    /// This is NOT integrated information (Φ) — it does not account
    /// for the Minimum Information Partition.
    pub fn compute_effective_information_normalized(&self, components: &[ContinuousHV]) -> f64 {
        if components.len() < 2 {
            return 0.0;
        }

        // Total effective information (sum of pairwise MI)
        let ei = self.effective_information(components);

        // Normalize by theoretical maximum
        // Max EI would be if all pairs had max MI
        let n = components.len();
        let num_pairs = (n * (n - 1)) / 2;
        let max_entropy = self.log(self.config.num_bins as f64);
        let theoretical_max = num_pairs as f64 * max_entropy;

        if theoretical_max > 0.0 {
            (ei / theoretical_max).min(1.0)
        } else {
            0.0
        }
    }
}

impl Default for TruePhiCalculator {
    fn default() -> Self {
        Self::new()
    }
}
