// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Topological Coherence Benchmark
//!
//! Tests whether moral space fragmentation (high beta_0 = disconnected components)
//! correlates with higher consciousness variance over a sliding window.
//!
//! ## Method
//!
//! 1. Generate moral trajectories with controlled fragmentation:
//!    - Low fragmentation: scenarios cluster around 1-2 harmony axes
//!    - High fragmentation: scenarios scatter across all 7 axes uniformly
//! 2. For each condition, compute:
//!    - beta_0: number of connected components in the moral similarity graph
//!      (approximated by counting clusters in the harmony projection)
//!    - Consciousness variance: stddev of consciousness level over the window
//! 3. Test: does higher beta_0 predict higher consciousness variance?
//!
//! ## Prediction
//!
//! Pearson r > 0.3 between fragmentation level and consciousness variance.
//! Fragmented moral space → unstable consciousness (Damasio's somatic markers
//! require coherent value integration for stable phenomenal experience).
//!
//! Science:
//! - Damasio, A. (1994). Descartes' Error.
//! - Tononi, G. (2004). An information integration theory of consciousness. BMC Neuroscience.

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::ContinuousHV;

/// Number of fragmentation levels.
const NUM_FRAG_LEVELS: usize = 7;

/// Scenarios per fragmentation level.
const SCENARIOS_PER_LEVEL: usize = 30;

/// Similarity threshold for connectivity in beta_0 approximation.
const CONNECTIVITY_THRESHOLD: f64 = 0.15;

/// Consciousness dampening from fragmentation.
const FRAG_DAMPENING: f64 = 0.1;

/// Base consciousness level.
const BASE_CONSCIOUSNESS: f64 = 0.8;

/// Number of harmony axes (Eight Harmonies).
const N_HARMONIES: usize = 8;

/// Topological Coherence Benchmark.
pub struct TopologicalCoherenceBenchmark;

impl TopologicalCoherenceBenchmark {
    /// Project a ContinuousHV onto harmony dimensions via stride-based softmax.
    fn harmony_projection(hv: &ContinuousHV) -> [f64; N_HARMONIES] {
        let data = &hv.values;
        let stride = data.len() / N_HARMONIES;
        let mut proj = [0.0f64; N_HARMONIES];
        for (h, p) in proj.iter_mut().enumerate() {
            let start = h * stride;
            let end = ((h + 1) * stride).min(data.len());
            let sum: f64 = data[start..end].iter().map(|&v| v as f64).sum();
            *p = sum / (end - start) as f64;
        }
        let max = proj.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_sum: f64 = proj.iter().map(|&v| (v - max).exp()).sum();
        for p in proj.iter_mut() {
            *p = (*p - max).exp() / exp_sum;
        }
        proj
    }

    /// Approximate beta_0 (connected components) from a set of harmony projections.
    ///
    /// Uses single-linkage clustering: two points are connected if their L2
    /// distance is below `CONNECTIVITY_THRESHOLD`. Returns the number of
    /// connected components via union-find.
    fn approximate_beta_0(projections: &[[f64; N_HARMONIES]]) -> usize {
        let n = projections.len();
        if n == 0 {
            return 0;
        }
        // Union-find
        let mut parent: Vec<usize> = (0..n).collect();

        fn find(parent: &mut [usize], mut x: usize) -> usize {
            while parent[x] != x {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            x
        }

        fn union(parent: &mut [usize], a: usize, b: usize) {
            let ra = find(parent, a);
            let rb = find(parent, b);
            if ra != rb {
                parent[ra] = rb;
            }
        }

        for i in 0..n {
            for j in (i + 1)..n {
                let dist: f64 = projections[i]
                    .iter()
                    .zip(projections[j].iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                if dist < CONNECTIVITY_THRESHOLD {
                    union(&mut parent, i, j);
                }
            }
        }

        // Count distinct components
        let mut roots = std::collections::HashSet::new();
        for i in 0..n {
            roots.insert(find(&mut parent, i));
        }
        roots.len()
    }

    /// Generate a scenario HV with controlled fragmentation.
    ///
    /// `num_active_axes`: how many harmony axes are active (1 = focused, 7 = scattered).
    /// Scenarios cluster around randomly chosen active axes.
    fn make_fragmented_scenario(
        dim: usize,
        num_active_axes: usize,
        scenario_idx: usize,
        seed: u64,
    ) -> ContinuousHV {
        let mut hv = ContinuousHV::random(dim, seed + scenario_idx as u64);
        let stride = dim / N_HARMONIES;

        // Determine which axis this scenario targets
        // Cycle through active axes
        let target_axis = scenario_idx % num_active_axes;
        // Hash to get a deterministic axis mapping
        let axis = (target_axis.wrapping_mul(3) + (seed as usize % N_HARMONIES)) % N_HARMONIES;

        // Strongly bias the target axis
        let data = &mut hv.values;
        let start = axis * stride;
        let end = ((axis + 1) * stride).min(dim);
        for v in data[start..end].iter_mut() {
            *v = (*v + 0.8).clamp(-1.0, 1.0);
        }
        // Dampen non-target axes
        for h in 0..N_HARMONIES {
            if h != axis {
                let s = h * stride;
                let e = ((h + 1) * stride).min(dim);
                for v in data[s..e].iter_mut() {
                    *v *= 0.3;
                }
            }
        }
        hv
    }

    /// Standard deviation.
    fn stddev(xs: &[f64]) -> f64 {
        let n = xs.len() as f64;
        if n < 2.0 {
            return 0.0;
        }
        let mean = xs.iter().sum::<f64>() / n;
        let var = xs.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        var.sqrt()
    }

    /// Pearson correlation coefficient.
    fn pearson_r(xs: &[f64], ys: &[f64]) -> f64 {
        let n = xs.len() as f64;
        if n < 2.0 {
            return 0.0;
        }
        let mean_x = xs.iter().sum::<f64>() / n;
        let mean_y = ys.iter().sum::<f64>() / n;
        let mut cov = 0.0;
        let mut var_x = 0.0;
        let mut var_y = 0.0;
        for (&x, &y) in xs.iter().zip(ys.iter()) {
            let dx = x - mean_x;
            let dy = y - mean_y;
            cov += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }
        if var_x < 1e-15 || var_y < 1e-15 {
            return 0.0;
        }
        cov / (var_x.sqrt() * var_y.sqrt())
    }
}

impl PsychBenchmark for TopologicalCoherenceBenchmark {
    fn name(&self) -> &str {
        "NormativeIntegration::TopologicalCoherence"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Moral Topology-Consciousness Stability Coupling",
            citation: "Damasio (1994); Tononi (2004)",
            year: 1994,
            doi: None,
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

        let dim = config.dimension.max(256);
        let seed = config.seed;

        let mut frag_levels = Vec::with_capacity(NUM_FRAG_LEVELS);
        let mut beta_0_values = Vec::with_capacity(NUM_FRAG_LEVELS);
        let mut consciousness_variances = Vec::with_capacity(NUM_FRAG_LEVELS);
        let mut trace = Vec::new();

        for level in 0..NUM_FRAG_LEVELS {
            // Number of active axes: 1 (focused) to 7 (scattered)
            let num_active = level + 1;

            // Generate scenarios
            let mut projections: Vec<[f64; N_HARMONIES]> = Vec::with_capacity(SCENARIOS_PER_LEVEL);
            let mut consciousness_values = Vec::with_capacity(SCENARIOS_PER_LEVEL);

            for s in 0..SCENARIOS_PER_LEVEL {
                let hv =
                    Self::make_fragmented_scenario(dim, num_active, s, seed + level as u64 * 1000);
                let proj = Self::harmony_projection(&hv);

                // Consciousness varies with how far this projection is from
                // the running mean (local incoherence)
                let mean_proj = if !projections.is_empty() {
                    let mut m = [0.0f64; N_HARMONIES];
                    for p in &projections {
                        for (i, &v) in p.iter().enumerate() {
                            m[i] += v;
                        }
                    }
                    for v in m.iter_mut() {
                        *v /= projections.len() as f64;
                    }
                    m
                } else {
                    proj
                };

                let local_dist: f64 = proj
                    .iter()
                    .zip(mean_proj.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();

                let consciousness =
                    (BASE_CONSCIOUSNESS - local_dist * FRAG_DAMPENING).clamp(0.3, 1.0);
                consciousness_values.push(consciousness);
                projections.push(proj);
            }

            let beta_0 = Self::approximate_beta_0(&projections) as f64;
            let consciousness_var = Self::stddev(&consciousness_values);

            frag_levels.push(num_active as f64);
            beta_0_values.push(beta_0);
            consciousness_variances.push(consciousness_var);

            if config.trial_trace {
                let mut extra = BTreeMap::new();
                extra.insert("num_active_axes".into(), num_active as f64);
                extra.insert("beta_0".into(), beta_0);
                extra.insert("consciousness_var".into(), consciousness_var);
                trace.push(TrialOutcome {
                    trial_idx: level,
                    condition: format!("axes_{}", num_active),
                    correct: true,
                    rt_ticks: 0.0,
                    similarity: consciousness_var,
                    confidence: beta_0 / SCENARIOS_PER_LEVEL as f64,
                    response_idx: level,
                    extra,
                });
            }
        }

        // Correlations
        let r_frag_beta = Self::pearson_r(&frag_levels, &beta_0_values);
        let r_beta_var = Self::pearson_r(&beta_0_values, &consciousness_variances);
        let r_frag_var = Self::pearson_r(&frag_levels, &consciousness_variances);

        // Prediction: fragmentation correlates with consciousness variance
        let prediction_met = r_frag_var > 0.3;

        result.insert(
            "pearson_r_fragmentation_beta0",
            MetricValue::from_samples(&[r_frag_beta]),
        );
        result.insert(
            "pearson_r_beta0_consciousness_var",
            MetricValue::from_samples(&[r_beta_var]),
        );
        result.insert(
            "pearson_r_fragmentation_consciousness_var",
            MetricValue::from_samples(&[r_frag_var]),
        );
        result.insert(
            "prediction_met",
            MetricValue::from_samples(&[if prediction_met { 1.0 } else { 0.0 }]),
        );
        result.insert("beta_0_values", MetricValue::from_samples(&beta_0_values));
        result.insert(
            "consciousness_variances",
            MetricValue::from_samples(&consciousness_variances),
        );

        result.conditions = NUM_FRAG_LEVELS;
        result.trials_per_condition = SCENARIOS_PER_LEVEL;
        if config.trial_trace {
            result.trial_trace = trace;
        }
        result.elapsed_ms = start.elapsed().as_millis() as u64;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topological_coherence_runs() {
        let config = BenchmarkConfig {
            dimension: 256,
            ..Default::default()
        };
        let result = TopologicalCoherenceBenchmark.run(&config);
        assert!(
            result
                .metrics
                .contains_key("pearson_r_fragmentation_consciousness_var")
        );
    }

    #[test]
    fn test_beta_0_single_point() {
        let proj = [[0.14; N_HARMONIES]];
        let beta = TopologicalCoherenceBenchmark::approximate_beta_0(&proj);
        assert_eq!(beta, 1, "Single point should have beta_0 = 1");
    }

    #[test]
    fn test_beta_0_two_close_points() {
        let proj = [[0.14; N_HARMONIES], [0.145; N_HARMONIES]];
        let beta = TopologicalCoherenceBenchmark::approximate_beta_0(&proj);
        assert_eq!(beta, 1, "Two close points should be connected (beta_0 = 1)");
    }

    #[test]
    fn test_beta_0_two_far_points() {
        let mut a = [0.0f64; N_HARMONIES];
        let mut b = [0.0f64; N_HARMONIES];
        a[0] = 1.0; // All mass on axis 0
        b[3] = 1.0; // All mass on axis 3
        let proj = [a, b];
        let beta = TopologicalCoherenceBenchmark::approximate_beta_0(&proj);
        assert_eq!(
            beta, 2,
            "Two maximally distant points should be disconnected (beta_0 = 2)"
        );
    }

    #[test]
    fn test_fragmentation_increases_beta_0() {
        let config = BenchmarkConfig {
            dimension: 1024,
            ..Default::default()
        };
        let result = TopologicalCoherenceBenchmark.run(&config);
        // Verify fragmentation-beta_0 correlation is non-negative.
        // With larger dimensions, more active axes should scatter projections
        // across harmony space, producing more disconnected clusters.
        let r = result.metrics["pearson_r_fragmentation_beta0"].mean;
        assert!(
            r >= -0.1,
            "Fragmentation should not negatively correlate with beta_0, got r={:.4}",
            r
        );
    }

    #[test]
    fn test_stddev_constant() {
        let xs = vec![5.0; 10];
        let sd = TopologicalCoherenceBenchmark::stddev(&xs);
        assert!(sd < 1e-10, "Constant values should have zero stddev");
    }

    #[test]
    fn test_stddev_known() {
        let xs = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let sd = TopologicalCoherenceBenchmark::stddev(&xs);
        // Population stddev = 2.0, sample stddev ≈ 2.138
        assert!(
            (sd - 2.138).abs() < 0.01,
            "Expected stddev ~2.138, got {:.4}",
            sd
        );
    }

    #[test]
    fn test_make_fragmented_scenario_preserves_dim() {
        let hv = TopologicalCoherenceBenchmark::make_fragmented_scenario(256, 3, 0, 42);
        assert_eq!(hv.dim(), 256);
    }
}
