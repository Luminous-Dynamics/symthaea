// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Persistent homology and Betti number computation for moral topology.

use super::*;

impl MoralTopology {
    /// Compute n×n pairwise cosine similarity matrix (flat, row-major).
    pub(super) fn pairwise_similarities(&self) -> Vec<f64> {
        let n = self.window.len();
        let mut sim = vec![0.0f64; n * n];
        for i in 0..n {
            sim[i * n + i] = 1.0;
            for j in (i + 1)..n {
                let s = self.window[i].similarity(&self.window[j]) as f64;
                sim[i * n + j] = s;
                sim[j * n + i] = s;
            }
        }
        sim
    }

    /// Median of upper-triangle pairwise similarities.
    pub(super) fn characteristic_scale(sim: &[f64], n: usize) -> f64 {
        let mut upper: Vec<f64> = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                upper.push(sim[i * n + j]);
            }
        }
        if upper.is_empty() {
            return 0.5;
        }
        upper.sort_by(|a, b| a.total_cmp(b));
        upper[upper.len() / 2]
    }

    /// Compute Betti numbers at a given scale threshold.
    pub(super) fn compute_betti(sim: &[f64], n: usize, scale: f64) -> BettiNumbers {
        // Build adjacency
        let mut adj = vec![vec![false; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                if sim[i * n + j] >= scale {
                    adj[i][j] = true;
                    adj[j][i] = true;
                }
            }
        }

        let beta_0 = Self::count_components(&adj, n);
        let beta_1 = Self::count_triangles(&adj, n) / 3;
        let beta_2 = Self::count_tetrahedra(&adj, n) / 4;

        BettiNumbers::new(beta_0, beta_1, beta_2)
    }

    /// Exact Betti computation via Hodge Laplacian on the Rips complex.
    ///
    /// More accurate than triangle/tetrahedra counting but O(n³) for
    /// boundary matrix operations. Use for small windows (n ≤ 32).
    pub(super) fn compute_betti_exact(sim: &[f64], n: usize, scale: f64) -> BettiNumbers {
        let mut complex = SimplicialComplex::new();
        // Add vertices
        for i in 0..n {
            complex.add_simplex(vec![i]);
        }
        // Add edges (1-simplices) where similarity ≥ scale
        for i in 0..n {
            for j in (i + 1)..n {
                if sim[i * n + j] >= scale {
                    complex.add_simplex(vec![i, j]);
                    // Add triangles (2-simplices)
                    for k in (j + 1)..n {
                        if sim[i * n + k] >= scale && sim[j * n + k] >= scale {
                            complex.add_simplex(vec![i, j, k]);
                            // Add tetrahedra (3-simplices)
                            for l in (k + 1)..n {
                                if sim[i * n + l] >= scale
                                    && sim[j * n + l] >= scale
                                    && sim[k * n + l] >= scale
                                {
                                    complex.add_simplex(vec![i, j, k, l]);
                                }
                            }
                        }
                    }
                }
            }
        }
        let laplacian = HodgeLaplacian::new(complex);
        let hodge_betti = laplacian.betti_numbers();
        BettiNumbers::new(hodge_betti.get(0), hodge_betti.get(1), hodge_betti.get(2))
    }

    /// DFS-based connected component counting (β₀).
    fn count_components(adj: &[Vec<bool>], n: usize) -> usize {
        let mut visited = vec![false; n];
        let mut count = 0;
        for i in 0..n {
            if !visited[i] {
                Self::dfs(i, adj, &mut visited);
                count += 1;
            }
        }
        count
    }

    fn dfs(node: usize, adj: &[Vec<bool>], visited: &mut [bool]) {
        visited[node] = true;
        for (neighbor, connected) in adj[node].iter().enumerate() {
            if *connected && !visited[neighbor] {
                Self::dfs(neighbor, adj, visited);
            }
        }
    }

    /// Triangle counting (for β₁ estimation; divide by 3 externally).
    fn count_triangles(adj: &[Vec<bool>], n: usize) -> usize {
        let mut count = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                if adj[i][j] {
                    for k in (j + 1)..n {
                        if adj[i][k] && adj[j][k] {
                            count += 1;
                        }
                    }
                }
            }
        }
        count
    }

    /// Tetrahedra counting (for β₂ estimation; divide by 4 externally).
    fn count_tetrahedra(adj: &[Vec<bool>], n: usize) -> usize {
        let mut count = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                if adj[i][j] {
                    for k in (j + 1)..n {
                        if adj[i][k] && adj[j][k] {
                            for l in (k + 1)..n {
                                if adj[i][l] && adj[j][l] && adj[k][l] {
                                    count += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        count
    }

    /// Multi-scale sweep to find persistent topological features.
    pub(super) fn persistent_features(&self, sim: &[f64], n: usize) -> Vec<PersistentFeature> {
        let num_scales = self.config.num_scales;
        let min_persistence = self.config.min_persistence;

        // Generate scale thresholds from 0.0 to 1.0
        let scales: Vec<f64> = (0..num_scales)
            .map(|i| i as f64 / (num_scales - 1).max(1) as f64)
            .collect();

        // Track Betti numbers at each scale
        let betti_at_scale: Vec<BettiNumbers> = scales
            .iter()
            .map(|&s| Self::compute_betti(sim, n, s))
            .collect();

        let mut features = Vec::new();

        // Track β₀ feature births/deaths
        Self::track_dimension_features(
            &scales,
            &betti_at_scale,
            TopologicalFeature::Component,
            |b| b.beta_0,
            min_persistence,
            &mut features,
        );

        // Track β₁ feature births/deaths
        Self::track_dimension_features(
            &scales,
            &betti_at_scale,
            TopologicalFeature::Cycle,
            |b| b.beta_1,
            min_persistence,
            &mut features,
        );

        // Track β₂ feature births/deaths
        Self::track_dimension_features(
            &scales,
            &betti_at_scale,
            TopologicalFeature::Void,
            |b| b.beta_2,
            min_persistence,
            &mut features,
        );

        features
    }

    /// Track birth/death of features for one Betti dimension.
    fn track_dimension_features(
        scales: &[f64],
        betti_at_scale: &[BettiNumbers],
        feature_type: TopologicalFeature,
        extract: impl Fn(&BettiNumbers) -> usize,
        min_persistence: f64,
        features: &mut Vec<PersistentFeature>,
    ) {
        if scales.len() < 2 {
            return;
        }
        let mut prev = extract(&betti_at_scale[0]);
        let mut births: Vec<f64> = (0..prev).map(|_| scales[0]).collect();

        for i in 1..scales.len() {
            let curr = extract(&betti_at_scale[i]);
            if curr > prev {
                // New features born
                for _ in 0..(curr - prev) {
                    births.push(scales[i]);
                }
            } else if curr < prev {
                // Features died — oldest first
                for _ in 0..(prev - curr) {
                    if let Some(birth) = births.pop() {
                        let pf = PersistentFeature::new(feature_type, birth, scales[i]);
                        if pf.persistence >= min_persistence {
                            features.push(pf);
                        }
                    }
                }
            }
            prev = curr;
        }

        // Features still alive at the last scale get death = last scale
        // SAFETY: scales.len() >= 2 checked at entry
        let last_scale = *scales.last().expect("scales.len() >= 2 verified at entry");
        for birth in births.drain(..) {
            let pf = PersistentFeature::new(feature_type, birth, last_scale);
            if pf.persistence >= min_persistence {
                features.push(pf);
            }
        }
    }

    /// Compute persistence-weighted Hodge decomposition fractions.
    ///
    /// Sweeps the Rips threshold across `num_scales` values from 0 to 1,
    /// computes the Hodge decomposition at each scale that has enough edges,
    /// and returns the integral weighted by the scale interval width.
    ///
    /// This captures the *persistent* harmonic structure — harmonics that
    /// survive across many thresholds are topologically robust, while
    /// transient ones at a single scale are noise.
    ///
    /// Science: Hodge (1941); persistence-weighted integration extends
    /// standard persistent homology (Edelsbrunner & Harer 2010) into
    /// the Hodge signal processing domain.
    pub(super) fn compute_persistent_hodge_fractions(
        sim: &[f64],
        n: usize,
        num_scales: usize,
    ) -> Option<HodgeFractions> {
        if n < 4 || num_scales < 3 {
            return None;
        }

        let scales: Vec<f64> = (0..num_scales)
            .map(|i| i as f64 / (num_scales - 1).max(1) as f64)
            .collect();

        let mut weighted_gradient = 0.0_f64;
        let mut weighted_curl = 0.0_f64;
        let mut weighted_harmonic = 0.0_f64;
        let mut total_weight = 0.0_f64;
        let mut scales_sampled = 0_usize;

        for w in 0..(num_scales - 1) {
            let scale = scales[w];
            let interval_width = scales[w + 1] - scales[w];

            // Build Rips complex at this scale
            let mut complex = SimplicialComplex::new();
            let mut edge_signal: Vec<f64> = Vec::new();

            for i in 0..n {
                complex.add_simplex(vec![i]);
            }
            for i in 0..n {
                for j in (i + 1)..n {
                    if sim[i * n + j] >= scale {
                        complex.add_simplex(vec![i, j]);
                        edge_signal.push(sim[i * n + j]);
                        for k in (j + 1)..n {
                            if sim[i * n + k] >= scale && sim[j * n + k] >= scale {
                                complex.add_simplex(vec![i, j, k]);
                            }
                        }
                    }
                }
            }

            let edge_count = complex.count(1);
            if edge_count < 3 {
                continue;
            }

            // Center the edge signal
            let mean = edge_signal.iter().sum::<f64>() / edge_signal.len() as f64;
            let centered: Vec<f64> = edge_signal.iter().map(|s| s - mean).collect();

            let laplacian = HodgeLaplacian::new(complex);
            if let Some(decomp) = laplacian.hodge_decompose(1, &centered) {
                let (g, c, h) = decomp.fractions();
                weighted_gradient += g * interval_width;
                weighted_curl += c * interval_width;
                weighted_harmonic += h * interval_width;
                total_weight += interval_width;
                scales_sampled += 1;
            }
        }

        if total_weight < 1e-15 || scales_sampled == 0 {
            return None;
        }

        // Normalize by total weight to get persistence-weighted averages
        Some(HodgeFractions {
            gradient: weighted_gradient / total_weight,
            curl: weighted_curl / total_weight,
            harmonic: weighted_harmonic / total_weight,
            scales_sampled,
            total_weight,
        })
    }

    /// Compute per-harmony variance across all 8D coordinates.
    pub(super) fn harmony_variance(coords: &[[f64; N_HARMONIES]]) -> [f64; N_HARMONIES] {
        let n = coords.len();
        if n == 0 {
            return [0.0; N_HARMONIES];
        }
        let mut mean = [0.0f64; N_HARMONIES];
        for c in coords {
            for (i, v) in c.iter().enumerate() {
                mean[i] += v;
            }
        }
        for m in &mut mean {
            *m /= n as f64;
        }
        let mut var = [0.0f64; N_HARMONIES];
        for c in coords {
            for (i, v) in c.iter().enumerate() {
                let d = v - mean[i];
                var[i] += d * d;
            }
        }
        for v in &mut var {
            *v /= n as f64;
        }
        var
    }
}
