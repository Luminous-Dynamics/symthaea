// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! TruePhiCalculator - Core IIT computation using Shannon entropy.

use crate::hdc::unified_hv::ContinuousHV;

use super::{EntropyConfig, JointDistribution, TruePartition, TruePhiResult, VectorDistribution};

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

    /// Spectral partition using the Fiedler vector (second smallest eigenvector of Laplacian)
    ///
    /// The Fiedler vector reveals natural clusters in the MI graph.
    /// Components with the same sign tend to be more connected.
    pub(crate) fn spectral_partition(
        &self,
        mi_matrix: &[Vec<f64>],
        n: usize,
    ) -> Option<TruePartition> {
        if n < 3 {
            return None;
        }

        // Build Laplacian: L = D - A where A is the MI adjacency matrix
        let mut laplacian = vec![vec![0.0; n]; n];
        for i in 0..n {
            let degree: f64 = mi_matrix[i].iter().sum();
            laplacian[i][i] = degree;
            for j in 0..n {
                if i != j {
                    laplacian[i][j] = -mi_matrix[i][j];
                }
            }
        }

        // Power iteration to find Fiedler vector
        let fiedler = self.power_iteration_fiedler(&laplacian, n, 100);

        // Partition based on sign of Fiedler vector components
        let mut part_a = Vec::new();
        let mut part_b = Vec::new();
        for i in 0..n {
            if fiedler[i] >= 0.0 {
                part_a.push(i);
            } else {
                part_b.push(i);
            }
        }

        if part_a.is_empty() || part_b.is_empty() {
            let mut indices: Vec<usize> = (0..n).collect();
            indices.sort_by(|&a, &b| {
                fiedler[a]
                    .partial_cmp(&fiedler[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let mid = n / 2;
            return Some(TruePartition {
                part_a: indices[..mid].to_vec(),
                part_b: indices[mid..].to_vec(),
            });
        }

        Some(TruePartition { part_a, part_b })
    }

    /// Power iteration to find the Fiedler vector
    pub(crate) fn power_iteration_fiedler(
        &self,
        laplacian: &[Vec<f64>],
        n: usize,
        max_iter: usize,
    ) -> Vec<f64> {
        let mut v: Vec<f64> = (0..n).map(|i| i as f64 - n as f64 / 2.0).collect();

        // Normalize
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-10 {
            for x in &mut v {
                *x /= norm;
            }
        }

        // Add regularization
        let epsilon = 0.01;
        let mut reg_laplacian = laplacian.to_vec();
        for i in 0..n {
            reg_laplacian[i][i] += epsilon;
        }

        for _ in 0..max_iter {
            let mut new_v = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    new_v[i] += reg_laplacian[i][j] * v[j];
                }
            }

            // Deflate by constant vector
            let mean: f64 = new_v.iter().sum::<f64>() / n as f64;
            for x in &mut new_v {
                *x -= mean;
            }

            let norm: f64 = new_v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < 1e-10 {
                break;
            }
            for x in &mut new_v {
                *x /= norm;
            }

            v = new_v;
        }

        v
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
