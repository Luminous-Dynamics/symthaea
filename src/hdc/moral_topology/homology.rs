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
    /// More accurate than triangle/tetrahedra counting but the Laplacian's
    /// boundary-matrix products are dense in the SIMPLEX COUNT, which for a
    /// dense similarity window grows like C(n,4). Two guards keep this
    /// tractable (added 2026-07-16 after this function was caught as the
    /// ">24h at 97.7% CPU" empty-input loop hang — a long run of
    /// near-identical inputs makes every pair clear `scale`, and n=64 fully
    /// connected is ~677k simplices, a ~10^15-op multiply; see
    /// docs/PHI_SIGNAL_TRACE_2026-07-15.md follow-up 6):
    ///
    /// 1. A complete 1-skeleton means the Rips complex is the full simplex,
    ///    which is contractible — Betti = (1, 0, 0) in CLOSED FORM. This is
    ///    exact mathematics, not an approximation, and it is precisely the
    ///    degenerate-window case that used to hang.
    /// 2. Past `MAX_EXACT_SIMPLICES`, fall back to the counting-based
    ///    `compute_betti` — an approximation, but a bounded one.
    pub(super) fn compute_betti_exact(sim: &[f64], n: usize, scale: f64) -> BettiNumbers {
        // Guard 1: complete 1-skeleton → contractible → closed form.
        let mut edge_count = 0usize;
        for i in 0..n {
            for j in (i + 1)..n {
                if sim[i * n + j] >= scale {
                    edge_count += 1;
                }
            }
        }
        if n >= 2 && edge_count == n * (n - 1) / 2 {
            return BettiNumbers::new(1, 0, 0);
        }

        // Guard 2: simplex budget — exact homology only while tractable.
        const MAX_EXACT_SIMPLICES: usize = 20_000;
        let mut remaining = MAX_EXACT_SIMPLICES.saturating_sub(n);

        let mut complex = SimplicialComplex::new();
        // Add vertices
        for i in 0..n {
            complex.add_simplex(vec![i]);
        }
        // Add edges (1-simplices) where similarity ≥ scale
        'build: for i in 0..n {
            for j in (i + 1)..n {
                if sim[i * n + j] >= scale {
                    if remaining == 0 {
                        break 'build;
                    }
                    remaining -= 1;
                    complex.add_simplex(vec![i, j]);
                    // Add triangles (2-simplices)
                    for k in (j + 1)..n {
                        if sim[i * n + k] >= scale && sim[j * n + k] >= scale {
                            if remaining == 0 {
                                break 'build;
                            }
                            remaining -= 1;
                            complex.add_simplex(vec![i, j, k]);
                            // Add tetrahedra (3-simplices)
                            for l in (k + 1)..n {
                                if sim[i * n + l] >= scale
                                    && sim[j * n + l] >= scale
                                    && sim[k * n + l] >= scale
                                {
                                    if remaining == 0 {
                                        break 'build;
                                    }
                                    remaining -= 1;
                                    complex.add_simplex(vec![i, j, k, l]);
                                }
                            }
                        }
                    }
                }
            }
        }
        if remaining == 0 {
            // Budget exhausted: the complex is too dense for in-loop exact
            // homology. The bounded counting approximation is the honest
            // alternative to a multi-hour Laplacian.
            return Self::compute_betti(sim, n, scale);
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

    /// Compute persistence-weighted vertex Hodge decomposition (L₀).
    ///
    /// Decomposes the *harmony coordinates* on vertices (0-simplices) across
    /// a multi-scale Rips filtration. This measures how moral meaning fragments
    /// into disconnected clusters as the connectivity threshold tightens.
    ///
    /// For the vertex Laplacian L₀:
    /// - **Gradient**: moral meaning flowing hierarchically between connected scenarios
    /// - **Harmonic**: moral meaning trapped in disconnected clusters (β₀ > 1)
    /// - The transition from curl-dominated to harmonic-dominated reveals the
    ///   **moral coherence phase transition** — the critical scale at which
    ///   unified reasoning fragments into isolated echo chambers.
    ///
    /// The function averages across all 8 harmony dimensions to produce a single
    /// composite fraction, weighted by each scale's interval width (persistence).
    ///
    /// Science: Hodge (1941); Tononi (2004) — integrated information collapses
    /// when β₀ > 1 (disconnected components cannot integrate). The harmonic
    /// fraction directly measures this topological disintegration.
    ///
    /// `vertex_signals` is an N×D array of harmony coordinates (one D-vector per scenario).
    /// `adaptive_center`: when Some, focuses the sweep around this center ± HODGE_ADAPTIVE_RIPS_RADIUS.
    pub(super) fn compute_persistent_hodge_fractions(
        sim: &[f64],
        n: usize,
        num_scales: usize,
        vertex_signals: &[[f64; N_HARMONIES]],
        adaptive_center: Option<f64>,
    ) -> Option<HodgeFractions> {
        if n < 4 || num_scales < 3 || vertex_signals.len() != n {
            return None;
        }

        let scales: Vec<f64> = if let Some(center) = adaptive_center {
            // Adaptive: focus sweep around the critical zone
            const RADIUS: f64 = 0.15; // HODGE_ADAPTIVE_RIPS_RADIUS
            let radius = RADIUS;
            let lo = (center - radius).max(0.0);
            let hi = (center + radius).min(1.0);
            (0..num_scales)
                .map(|i| lo + (hi - lo) * i as f64 / (num_scales - 1).max(1) as f64)
                .collect()
        } else {
            // Uniform: sweep full range 0.0 to 1.0
            (0..num_scales)
                .map(|i| i as f64 / (num_scales - 1).max(1) as f64)
                .collect()
        };

        let mut weighted_gradient = 0.0_f64;
        let mut weighted_curl = 0.0_f64;
        let mut weighted_harmonic = 0.0_f64;
        let mut total_weight = 0.0_f64;
        let mut scales_sampled = 0_usize;
        let mut critical_scale = f64::NAN;
        let mut found_critical = false;

        for w in 0..(num_scales - 1) {
            let scale = scales[w];
            let interval_width = scales[w + 1] - scales[w];

            // Build Rips complex at this scale
            let mut complex = SimplicialComplex::new();
            for i in 0..n {
                complex.add_simplex(vec![i]);
            }
            for i in 0..n {
                for j in (i + 1)..n {
                    if sim[i * n + j] >= scale {
                        complex.add_simplex(vec![i, j]);
                    }
                }
            }

            // Need at least 2 vertices with edges for meaningful L₀ decomposition
            if complex.count(0) < 2 || complex.count(1) == 0 {
                continue;
            }

            let laplacian = HodgeLaplacian::new(complex);

            // Decompose each harmony dimension on vertices, then average
            let mut dim_gradient = 0.0_f64;
            let mut dim_harmonic = 0.0_f64;
            let mut dim_curl = 0.0_f64;
            let mut dim_count = 0_usize;

            for d in 0..N_HARMONIES {
                let signal: Vec<f64> = vertex_signals.iter().map(|c| c[d]).collect();
                // Center the signal
                let mean = signal.iter().sum::<f64>() / signal.len() as f64;
                let centered: Vec<f64> = signal.iter().map(|v| v - mean).collect();

                if let Some(decomp) = laplacian.hodge_decompose(0, &centered) {
                    let (g, c, h) = decomp.fractions();
                    dim_gradient += g;
                    dim_curl += c;
                    dim_harmonic += h;
                    dim_count += 1;
                }
            }

            if dim_count > 0 {
                let inv = 1.0 / dim_count as f64;
                let scale_harmonic = dim_harmonic * inv;
                weighted_gradient += dim_gradient * inv * interval_width;
                weighted_curl += dim_curl * inv * interval_width;
                weighted_harmonic += scale_harmonic * interval_width;
                total_weight += interval_width;
                scales_sampled += 1;

                // Detect the critical scale: first scale where harmonic > 0.5
                if !found_critical && scale_harmonic > 0.5 {
                    critical_scale = scale;
                    found_critical = true;
                }
            }
        }

        if total_weight < 1e-15 || scales_sampled == 0 {
            return None;
        }

        // Normalize by total weight to get persistence-weighted averages
        let harmonic_avg = weighted_harmonic / total_weight;
        Some(HodgeFractions {
            gradient: weighted_gradient / total_weight,
            curl: weighted_curl / total_weight,
            harmonic: harmonic_avg,
            scales_sampled,
            total_weight,
            critical_scale,
            // At criticality: harmonic fraction in the "Goldilocks zone" [0.2, 0.8]
            // where the system is neither fully synchronized nor fully fragmented.
            at_criticality: harmonic_avg >= 0.2 && harmonic_avg <= 0.8,
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
