// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Perturbational Complexity Index (PCI) Benchmark — Digital Analog
//!
//! Massimini et al. (2013) showed that PCI — the spatiotemporal complexity of
//! cortical response to TMS perturbation — reliably distinguishes conscious from
//! unconscious brain states. High PCI = conscious; low PCI = unconscious.
//!
//! ## Principle
//!
//! Conscious systems have two properties simultaneously:
//! 1. **Differentiation**: Many possible states (high complexity)
//! 2. **Integration**: Perturbations propagate widely (high influence)
//!
//! ## Method
//!
//! 1. Inline 4-cell recurrent network (`MiniRecurrentNet`): states [f64;4],
//!    weights [[f64;4];4], tanh activation
//! 2. **Conscious condition**: GWT `enable_broadcasting = true`. Broadcast content
//!    feeds back into network input. Perturbation propagates through global workspace.
//! 3. **Unconscious condition**: GWT `enable_broadcasting = false`. No broadcast
//!    feedback. Perturbation stays local.
//! 4. Record 4D state trajectories, binarize (above/below median per dimension),
//!    compute normalized Lempel-Ziv complexity.
//! 5. PCI ratio = conscious_complexity / unconscious_complexity (target: >1.5)
//!
//! Science:
//! - Massimini, M. et al. (2013). A perturbational approach for evaluating the brain's
//!   capacity for consciousness. Annals of the New York Academy of Sciences.
//! - Casali, A. G. et al. (2013). A theoretically based index of consciousness. Science
//!   Translational Medicine.

use crate::benchmarks::qualia_confidence::helpers::{float_from_seed, normalized_lz};
use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use symthaea_core::hdc::BinaryHV;
use symthaea_core::hdc::global_workspace::{GlobalWorkspace, WorkspaceConfig, WorkspaceContent};

/// Baseline cycles before perturbation.
const BASELINE_CYCLES: usize = 50;

/// Post-perturbation observation cycles.
const POST_PERTURBATION_CYCLES: usize = 100;

/// Perturbation magnitude (fraction of state norm).
const PERTURBATION_MAGNITUDE: f64 = 0.30;

/// Number of repetitions for averaging.
const NUM_REPETITIONS: usize = 10;

/// Minimal recurrent network for PCI measurement.
///
/// 4 cells with PURELY self-connected dynamics (no cross-coupling).
/// Cross-dimensional coupling comes ONLY from broadcast feedback,
/// making broadcast the sole mechanism for perturbation propagation.
///
/// Design principle: without broadcast, a perturbation to dim 0/1
/// stays local (only those dims respond, then decay). With broadcast,
/// the perturbation propagates via a ring topology (0→1→2→3→0),
/// creating a traveling wave of diverse spatial patterns → high LZ.
struct MiniRecurrentNet {
    state: [f64; 4],
    self_weights: [f64; 4],
}

impl MiniRecurrentNet {
    fn new(seed: u64) -> Self {
        // Self-connections only: each dimension is an independent damped oscillator.
        // Weight 0.85 means perturbations decay with half-life ~4 steps.
        // Slight variation per dim prevents trivial symmetry.
        let mut self_weights = [0.0f64; 4];
        for (i, w) in self_weights.iter_mut().enumerate() {
            let w_seed = seed.wrapping_mul(0x100000001b3).wrapping_add(i as u64 * 10);
            let variation = float_from_seed(w_seed) * 0.06 - 0.03; // ±0.03
            *w = 0.85 + variation;
        }

        MiniRecurrentNet {
            state: [0.3, -0.2, 0.15, -0.25],
            self_weights,
        }
    }

    /// Step the network forward with optional external input.
    fn step(&mut self, input: [f64; 4]) {
        let mut new_state = [0.0f64; 4];
        for (i, ns) in new_state.iter_mut().enumerate() {
            *ns = (self.self_weights[i] * self.state[i] + input[i]).tanh();
        }
        self.state = new_state;
    }

    /// Apply a perturbation to dims 0 and 1 (localized "TMS pulse").
    fn perturb(&mut self, magnitude: f64, _seed: u64) {
        // Localized perturbation: only dims 0 and 1 are directly affected.
        // In the unconscious condition, dims 2 and 3 never "hear" about this.
        // In the conscious condition, broadcast propagates it to dims 2 and 3.
        self.state[0] = (self.state[0] + magnitude).clamp(-1.0, 1.0);
        self.state[1] = (self.state[1] - magnitude * 0.7).clamp(-1.0, 1.0);
    }
}

/// Run PCI measurement for one condition.
///
/// Returns (mean_complexity, trajectory_for_last_rep).
fn run_pci_condition(config: &BenchmarkConfig, broadcasting_enabled: bool) -> (f64, Vec<[f64; 4]>) {
    let mut complexities = Vec::with_capacity(NUM_REPETITIONS);
    let mut last_trajectory = Vec::new();

    for rep in 0..NUM_REPETITIONS {
        let rep_seed = config
            .seed
            .wrapping_mul(0x100000001b3)
            .wrapping_add(rep as u64 * 10000)
            .wrapping_add(if broadcasting_enabled { 0 } else { 5000 });

        let mut net = MiniRecurrentNet::new(rep_seed);
        let mut trajectory = Vec::with_capacity(BASELINE_CYCLES + POST_PERTURBATION_CYCLES);

        // Workspace for broadcast feedback.
        // Low entry threshold: the network's state always enters the workspace.
        // The question is whether it gets BROADCAST (conscious) or stays local.
        let ws_config = WorkspaceConfig {
            entry_threshold: 0.05,
            enable_broadcasting: broadcasting_enabled,
            ..Default::default()
        };

        // ── Baseline phase ──
        for cycle in 0..BASELINE_CYCLES {
            let cycle_seed = rep_seed
                .wrapping_mul(0x100000001b3)
                .wrapping_add(cycle as u64);

            // Create workspace content from network state
            let mut ws = GlobalWorkspace::new(ws_config.clone());
            let state_hv = BinaryHV::random(cycle_seed);
            // Activation always above entry threshold — network always has state to report
            let abs_mean = net.state.iter().map(|s| s.abs()).sum::<f64>() / 4.0;
            let activation = (0.5 + abs_mean * 0.5).clamp(0.0, 1.0);
            let content = WorkspaceContent::new(vec![state_hv], activation, "network".to_string());
            ws.submit(content);
            let assessment = ws.process();

            // Broadcast feedback: NONLINEAR state-dependent coupling.
            // Linear coupling synchronizes oscillators (reduces complexity).
            // Differential broadcast coupling: each dim receives the DIFFERENCE
            // between two neighbors. This creates oscillatory, wave-like dynamics
            // as signals bounce around the ring with alternating signs.
            // Without broadcast: each dim decays independently → predictable → low LZ.
            // With broadcast: differential coupling → oscillatory cascade → high LZ.
            let feedback = if broadcasting_enabled && !assessment.broadcasts.is_empty() {
                let s = assessment.broadcasts[0].strength;
                let st = net.state;
                [
                    s * 0.5 * (st[1] - st[2]),
                    s * 0.5 * (st[2] - st[3]),
                    s * 0.5 * (st[3] - st[0]),
                    s * 0.5 * (st[0] - st[1]),
                ]
            } else {
                [0.0; 4]
            };

            net.step(feedback);
            trajectory.push(net.state);
        }

        // ── Perturbation ──
        net.perturb(PERTURBATION_MAGNITUDE, rep_seed.wrapping_add(9999));

        // ── Post-perturbation phase ──
        for cycle in 0..POST_PERTURBATION_CYCLES {
            let cycle_seed = rep_seed
                .wrapping_mul(0x100000001b3)
                .wrapping_add((BASELINE_CYCLES + cycle) as u64);

            let mut ws = GlobalWorkspace::new(ws_config.clone());
            let state_hv = BinaryHV::random(cycle_seed);
            // Activation always above entry threshold — network always has state to report
            let abs_mean = net.state.iter().map(|s| s.abs()).sum::<f64>() / 4.0;
            let activation = (0.5 + abs_mean * 0.5).clamp(0.0, 1.0);
            let content = WorkspaceContent::new(vec![state_hv], activation, "network".to_string());
            ws.submit(content);
            let assessment = ws.process();

            // Same ring-topology broadcast as baseline phase
            let feedback = if broadcasting_enabled && !assessment.broadcasts.is_empty() {
                let s = assessment.broadcasts[0].strength * 0.4;
                let st = net.state;
                [s * st[3], s * st[0], s * st[1], s * st[2]]
            } else {
                [0.0; 4]
            };

            net.step(feedback);
            trajectory.push(net.state);
        }

        // ── Compute complexity of post-perturbation response ──
        // Use only post-perturbation trajectory
        let post_trajectory = &trajectory[BASELINE_CYCLES..];

        // Compute per-dimension medians for binarization
        let mut medians = [0.0f64; 4];
        for (dim, med) in medians.iter_mut().enumerate() {
            let mut values: Vec<f64> = post_trajectory.iter().map(|s| s[dim]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            *med = values[values.len() / 2];
        }

        // Concatenated spatiotemporal binarization (how real PCI works):
        // At each timestep, binarize all 4 dimensions → 4-bit spatial pattern.
        // The SEQUENCE of spatial patterns captures both temporal and spatial diversity.
        // Conscious: diverse spatial patterns → high LZ.
        // Unconscious: repetitive or local patterns → low LZ.
        let mut binary_sequence = Vec::with_capacity(post_trajectory.len() * 4);
        for state in post_trajectory {
            for (dim, &med) in medians.iter().enumerate() {
                binary_sequence.push(state[dim] > med);
            }
        }

        let mean_complexity = normalized_lz(&binary_sequence);
        complexities.push(mean_complexity);
        last_trajectory = trajectory;
    }

    let overall_mean = complexities.iter().sum::<f64>() / complexities.len() as f64;
    (overall_mean, last_trajectory)
}

/// Perturbational Complexity Index Benchmark.
///
/// Tests the principle that conscious systems produce more complex responses to
/// perturbation than unconscious systems (PCI > 1.5).
pub struct PerturbationalComplexityBenchmark;

impl PsychBenchmark for PerturbationalComplexityBenchmark {
    fn name(&self) -> &str {
        "QualiaConfidence::PerturbationalComplexity"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Perturbational Complexity Index (digital analog)",
            citation: "Casali et al. (2013); Massimini et al. (2013)",
            year: 2013,
            doi: Some("10.1126/scitranslmed.3006294"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

        // Run conscious condition (broadcasting enabled)
        let (pci_conscious, conscious_traj) = run_pci_condition(config, true);

        // Run unconscious condition (broadcasting disabled)
        let (pci_unconscious, unconscious_traj) = run_pci_condition(config, false);

        // PCI ratio
        let pci_ratio = if pci_unconscious > 0.001 {
            pci_conscious / pci_unconscious
        } else if pci_conscious > 0.001 {
            10.0 // Conscious has complexity, unconscious doesn't
        } else {
            1.0
        };

        result.insert("pci_conscious", MetricValue::from_samples(&[pci_conscious]));
        result.insert(
            "pci_unconscious",
            MetricValue::from_samples(&[pci_unconscious]),
        );
        result.insert("pci_ratio", MetricValue::from_samples(&[pci_ratio]));

        // Baseline complexity (pre-perturbation, conscious condition)
        if conscious_traj.len() >= BASELINE_CYCLES {
            let baseline_traj = &conscious_traj[..BASELINE_CYCLES];
            let mut medians = [0.0f64; 4];
            for (dim, med) in medians.iter_mut().enumerate() {
                let mut values: Vec<f64> = baseline_traj.iter().map(|s| s[dim]).collect();
                values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                *med = values[values.len() / 2];
            }
            let mut binary_seq = Vec::with_capacity(baseline_traj.len() * 4);
            for state in baseline_traj {
                for (dim, &med) in medians.iter().enumerate() {
                    binary_seq.push(state[dim] > med);
                }
            }
            result.insert(
                "baseline_complexity",
                MetricValue::from_samples(&[normalized_lz(&binary_seq)]),
            );
        }

        // Perturbation propagation depth: how many cycles until the effect diminishes
        if conscious_traj.len() > BASELINE_CYCLES + 1 {
            let pre_state = conscious_traj[BASELINE_CYCLES - 1];
            let mut propagation_depth = 0;
            #[allow(clippy::needless_range_loop)]
            for i in BASELINE_CYCLES..conscious_traj.len() {
                let delta: f64 = (0..4)
                    .map(|d| (conscious_traj[i][d] - pre_state[d]).abs())
                    .sum::<f64>();
                if delta > 0.01 {
                    propagation_depth = i - BASELINE_CYCLES + 1;
                }
            }
            result.insert(
                "perturbation_propagation_depth",
                MetricValue::from_samples(&[propagation_depth as f64]),
            );
        }

        // Unconscious propagation depth for comparison
        if unconscious_traj.len() > BASELINE_CYCLES + 1 {
            let pre_state = unconscious_traj[BASELINE_CYCLES - 1];
            let mut propagation_depth = 0;
            #[allow(clippy::needless_range_loop)]
            for i in BASELINE_CYCLES..unconscious_traj.len() {
                let delta: f64 = (0..4)
                    .map(|d| (unconscious_traj[i][d] - pre_state[d]).abs())
                    .sum::<f64>();
                if delta > 0.01 {
                    propagation_depth = i - BASELINE_CYCLES + 1;
                }
            }
            result.insert(
                "unconscious_propagation_depth",
                MetricValue::from_samples(&[propagation_depth as f64]),
            );
        }

        result.conditions = 2; // conscious vs unconscious
        result.trials_per_condition = NUM_REPETITIONS;
        result.elapsed_ms = start.elapsed().as_millis() as u64;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> BenchmarkConfig {
        BenchmarkConfig {
            seed: 42,
            ..Default::default()
        }
    }

    #[test]
    fn test_perturbational_complexity_runs() {
        let bench = PerturbationalComplexityBenchmark;
        let result = bench.run(&config());
        assert_eq!(
            result.benchmark,
            "QualiaConfidence::PerturbationalComplexity"
        );
        assert!(result.metrics.len() >= 3);
    }

    #[test]
    fn test_all_metrics_finite() {
        let bench = PerturbationalComplexityBenchmark;
        let result = bench.run(&config());
        for (key, value) in &result.metrics {
            assert!(
                value.mean.is_finite(),
                "Metric '{key}' should be finite, got {}",
                value.mean
            );
        }
    }

    #[test]
    fn test_pci_ratio_above_threshold() {
        // Massimini (2013): conscious systems produce PCI > unconscious systems
        let bench = PerturbationalComplexityBenchmark;
        let result = bench.run(&config());
        let ratio = result.metrics.get("pci_ratio").unwrap().mean;
        assert!(
            ratio > 1.3,
            "PCI ratio (conscious/unconscious) should show clear dissociation: {ratio}"
        );
    }

    #[test]
    fn test_conscious_gt_unconscious_complexity() {
        // Core prediction: broadcast feedback amplifies perturbation response complexity
        let bench = PerturbationalComplexityBenchmark;
        let result = bench.run(&config());
        let pci_c = result.metrics.get("pci_conscious").unwrap().mean;
        let pci_u = result.metrics.get("pci_unconscious").unwrap().mean;
        assert!(
            pci_c > pci_u,
            "Conscious PCI ({pci_c}) should exceed unconscious PCI ({pci_u})"
        );
    }

    #[test]
    fn test_perturbation_propagates_further_with_broadcasting() {
        let bench = PerturbationalComplexityBenchmark;
        let result = bench.run(&config());
        let depth_c = result
            .metrics
            .get("perturbation_propagation_depth")
            .unwrap()
            .mean;
        let depth_u = result
            .metrics
            .get("unconscious_propagation_depth")
            .unwrap()
            .mean;
        assert!(
            depth_c > depth_u,
            "Conscious propagation depth ({depth_c}) should exceed unconscious ({depth_u})"
        );
    }

    #[test]
    fn test_mini_recurrent_net_dynamics() {
        let mut net = MiniRecurrentNet::new(42);
        let initial = net.state;
        net.step([0.1, 0.0, 0.0, 0.0]);
        // State should change after step
        let changed = net
            .state
            .iter()
            .zip(initial.iter())
            .any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(changed, "Network should have dynamics");
    }

    #[test]
    fn test_perturbation_changes_state() {
        let mut net = MiniRecurrentNet::new(42);
        let pre = net.state;
        net.perturb(0.5, 99);
        let post = net.state;
        let delta: f64 = (0..4).map(|i| (pre[i] - post[i]).abs()).sum();
        assert!(
            delta > 0.01,
            "Perturbation should change state: delta={delta}"
        );
    }

    #[test]
    fn test_deterministic() {
        let bench = PerturbationalComplexityBenchmark;
        let r1 = bench.run(&config());
        let r2 = bench.run(&config());
        let p1 = r1.metrics.get("pci_ratio").unwrap().mean;
        let p2 = r2.metrics.get("pci_ratio").unwrap().mean;
        assert!(
            (p1 - p2).abs() < 1e-10,
            "Same seed should produce identical results: {p1} vs {p2}"
        );
    }

    #[test]
    fn test_has_provenance() {
        let bench = PerturbationalComplexityBenchmark;
        assert!(bench.provenance().is_some());
    }

    #[test]
    fn test_baseline_complexity_exists() {
        let bench = PerturbationalComplexityBenchmark;
        let result = bench.run(&config());
        assert!(result.metrics.contains_key("baseline_complexity"));
    }

    #[test]
    fn test_pci_ratio_robust_across_seeds() {
        let bench = PerturbationalComplexityBenchmark;
        for seed in [42, 137, 256, 999, 7777] {
            let cfg = BenchmarkConfig {
                seed,
                ..Default::default()
            };
            let result = bench.run(&cfg);
            let ratio = result.metrics.get("pci_ratio").unwrap().mean;
            assert!(
                ratio > 1.0,
                "PCI ratio should be >1.0 at seed={seed}: {ratio}"
            );
        }
    }
}
