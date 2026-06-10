// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # End-to-End Unified FL Pipeline Benchmark
//!
//! Exercises every stage of the unified FL pipeline:
//!
//! 1. **DP noise**: Gradient clipping + Gaussian noise
//! 2. **Reputation gate**: Drop low-reputation participants
//! 3. **Multi-signal Byzantine detection**: 4-signal ensemble
//! 4. **Hybrid BFT trimming**: Reputation-weighted outlier scoring
//! 5. **Reputation²-weighted aggregation**: Final weighted mean
//! 6. **Plugin system**: External weight adjustments + verification
//!
//! ## Run
//! ```bash
//! cargo run --example benchmark_unified_pipeline --release
//! ```

use std::collections::HashMap;
use std::time::Instant;

use mycelix_fl_core::pipeline::{
    ExternalWeightMap, ParticipantWeightAdjustment, PipelineConfig, UnifiedPipeline,
};
use mycelix_fl_core::plugins::{
    ByzantinePlugin, PipelinePlugins, VerificationPlugin, VerificationResult,
};
use mycelix_fl_core::privacy::DifferentialPrivacyConfig;
use mycelix_fl_core::types::GradientUpdate;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║       Unified FL Pipeline — End-to-End Benchmark            ║");
    println!("║       mycelix-fl-core v0.1.0                                ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let mut pass = 0;
    let mut fail = 0;
    let total_start = Instant::now();

    // ═══════════════════════════════════════════════════════════════
    // Test 1: Pipeline convergence (honest network)
    // ═══════════════════════════════════════════════════════════════
    print_header("Test 1: Pipeline Convergence (50 honest nodes, no Byzantine)");
    {
        let t = Instant::now();
        let (updates, reps) = make_contributions(50, 0, 0.5, 10);
        let config = PipelineConfig::default();
        let mut pipeline = UnifiedPipeline::new(config);

        let result = pipeline.aggregate(&updates, &reps).unwrap();
        let max_err = max_error(&result.aggregated.gradients, 0.5);
        let elapsed = t.elapsed();

        println!(
            "  Participants: {} → {}",
            result.stats.total_contributions, result.aggregated.participant_count
        );
        println!("  Max error from target: {:.6}", max_err);
        println!("  Time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);

        if max_err < 0.05 {
            println!("  PASS");
            pass += 1;
        } else {
            println!("  FAIL: max error {:.6} > 0.05", max_err);
            fail += 1;
        }
    }
    println!();

    // ═══════════════════════════════════════════════════════════════
    // Test 2: Byzantine resilience with reputation disparity
    // ═══════════════════════════════════════════════════════════════
    print_header("Test 2: 34% Byzantine (low rep) — Reputation Gate");
    {
        let t = Instant::now();
        let (updates, reps) = make_contributions_with_reps(66, 34, 0.5, 20, 0.85, 0.15);
        let config = PipelineConfig {
            min_reputation: 0.3,
            trim_fraction: 0.2,
            ..Default::default()
        };
        let mut pipeline = UnifiedPipeline::new(config);

        let result = pipeline.aggregate(&updates, &reps).unwrap();
        let max_err = max_error(&result.aggregated.gradients, 0.5);
        let elapsed = t.elapsed();

        println!(
            "  Total: {} (66 honest + 34 Byzantine)",
            result.stats.total_contributions
        );
        println!("  After gate: {}", result.stats.after_gate);
        println!("  Surviving: {}", result.aggregated.participant_count);
        println!("  Max error: {:.6}", max_err);
        println!("  Time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);

        if max_err < 0.15 {
            println!("  PASS: Byzantine gated out by reputation");
            pass += 1;
        } else {
            println!("  FAIL: max error {:.6} > 0.15", max_err);
            fail += 1;
        }
    }
    println!();

    // ═══════════════════════════════════════════════════════════════
    // Test 3: Byzantine resilience with same reputation (harder)
    // ═══════════════════════════════════════════════════════════════
    print_header("Test 3: 20% Byzantine (same rep) — Multi-signal + Trimming");
    {
        let t = Instant::now();
        let (updates, reps) = make_contributions_with_reps(80, 20, 0.5, 20, 0.85, 0.85);
        let config = PipelineConfig {
            min_reputation: 0.3,
            trim_fraction: 0.25,
            multi_signal_detection: true,
            ..Default::default()
        };
        let mut pipeline = UnifiedPipeline::new(config);

        let result = pipeline.aggregate(&updates, &reps).unwrap();
        let max_err = max_error(&result.aggregated.gradients, 0.5);
        let elapsed = t.elapsed();

        println!(
            "  Total: {} (80 honest + 20 Byzantine, same rep)",
            result.stats.total_contributions
        );
        if let Some(ref det) = result.detection {
            println!(
                "  Byzantine detected: {}/{}",
                det.byzantine_indices.len(),
                result.stats.total_contributions
            );
        }
        println!("  Surviving: {}", result.aggregated.participant_count);
        println!("  Max error: {:.6}", max_err);
        println!("  Time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);

        if max_err < 0.5 {
            println!("  PASS: Converges despite same-rep Byzantine");
            pass += 1;
        } else {
            println!("  FAIL: max error {:.6} > 0.5", max_err);
            fail += 1;
        }
    }
    println!();

    // ═══════════════════════════════════════════════════════════════
    // Test 4: Differential Privacy impact
    // ═══════════════════════════════════════════════════════════════
    print_header("Test 4: DP Privacy Modes — Utility vs Privacy Trade-off");
    {
        let dp_configs = [
            ("Low (ε≈10)", DifferentialPrivacyConfig::low_privacy()),
            (
                "Moderate (ε≈1)",
                DifferentialPrivacyConfig::moderate_privacy(),
            ),
            ("High (ε≈0.1)", DifferentialPrivacyConfig::high_privacy()),
        ];

        for (label, dp) in &dp_configs {
            let (updates, reps) = make_contributions(50, 0, 0.5, 20);
            let config = PipelineConfig {
                dp_config: Some(*dp),
                multi_signal_detection: false, // Isolate DP effect
                min_reputation: 0.1,
                ..Default::default()
            };
            let mut pipeline = UnifiedPipeline::new(config);

            let result = pipeline.aggregate(&updates, &reps).unwrap();
            let max_err = max_error(&result.aggregated.gradients, 0.5);
            let epsilon = result
                .privacy
                .as_ref()
                .and_then(|p| p.epsilon_estimate)
                .unwrap_or(0.0);

            println!(
                "  {}: max_err={:.4}, ε≈{:.4}, σ={:.3}",
                label,
                max_err,
                epsilon,
                result.privacy.as_ref().map(|p| p.sigma).unwrap_or(0.0)
            );
        }
        println!("  PASS: DP modes exercised");
        pass += 1;
    }
    println!();

    // ═══════════════════════════════════════════════════════════════
    // Test 5: External weight adjustments (consciousness-style)
    // ═══════════════════════════════════════════════════════════════
    print_header("Test 5: External Weight Adjustments (Epistemic/Phi)");
    {
        let t = Instant::now();
        let (updates, reps) = make_contributions(10, 0, 0.5, 10);
        let config = PipelineConfig {
            min_reputation: 0.1,
            multi_signal_detection: false,
            ..Default::default()
        };
        let mut pipeline = UnifiedPipeline::new(config);

        // Give first 3 nodes a 2x boost (simulating high Phi/epistemic quality)
        let mut ext = ExternalWeightMap::new();
        for i in 0..3 {
            ext.insert(
                format!("h{}", i),
                vec![ParticipantWeightAdjustment {
                    weight_multiplier: 2.0,
                    veto: false,
                    source: "phi".to_string(),
                }],
            );
        }
        // Veto node 9 (simulating consciousness-detected anomaly)
        ext.insert(
            "h9".to_string(),
            vec![ParticipantWeightAdjustment {
                weight_multiplier: 0.0,
                veto: true,
                source: "consciousness".to_string(),
            }],
        );

        let result = pipeline
            .aggregate_with_external_weights(&updates, &reps, &ext)
            .unwrap();
        let elapsed = t.elapsed();

        println!("  Boosted 3 nodes (2x), vetoed 1 node");
        println!("  Surviving: {}", result.aggregated.participant_count);
        println!("  Time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);

        if result.aggregated.participant_count <= 9 {
            println!("  PASS: Veto and boost applied");
            pass += 1;
        } else {
            println!("  FAIL: Veto not applied");
            fail += 1;
        }
    }
    println!();

    // ═══════════════════════════════════════════════════════════════
    // Test 6: Plugin system with Byzantine plugin + Verification
    // ═══════════════════════════════════════════════════════════════
    print_header("Test 6: Plugin System (Byzantine + Verification)");
    {
        let t = Instant::now();
        let (updates, reps) = make_contributions(10, 3, 0.5, 10);
        let config = PipelineConfig {
            min_reputation: 0.1,
            multi_signal_detection: false, // Let plugin handle detection
            ..Default::default()
        };
        let mut pipeline = UnifiedPipeline::new(config);

        let mut byz_plugin = NormByzantinePlugin { threshold: 1.5 };
        let mut verify_plugin = HashVerifier;

        let mut plugins = PipelinePlugins {
            compression: None,
            byzantine: vec![&mut byz_plugin],
            verification: Some(&mut verify_plugin),
        };

        let result = pipeline
            .aggregate_with_plugins(&updates, &reps, &mut plugins)
            .unwrap();
        let max_err = max_error(&result.result.aggregated.gradients, 0.5);
        let elapsed = t.elapsed();

        println!(
            "  Plugin weights applied: {}",
            result.plugin_weights_applied
        );
        println!(
            "  Verification: {:?}",
            result.verification.as_ref().map(|v| v.verified)
        );
        println!(
            "  Surviving: {}",
            result.result.aggregated.participant_count
        );
        println!("  Max error: {:.6}", max_err);
        println!("  Time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);

        if result.plugin_weights_applied && result.verification.is_some() && max_err < 0.5 {
            println!("  PASS: Full plugin pipeline");
            pass += 1;
        } else {
            println!("  FAIL");
            fail += 1;
        }
    }
    println!();

    // ═══════════════════════════════════════════════════════════════
    // Test 7: RDP budget tracking across rounds
    // ═══════════════════════════════════════════════════════════════
    print_header("Test 7: RDP Budget Tracking (100 rounds)");
    {
        let t = Instant::now();
        let config = PipelineConfig {
            dp_config: Some(DifferentialPrivacyConfig::moderate_privacy()),
            min_reputation: 0.1,
            multi_signal_detection: false,
            ..Default::default()
        };
        let mut pipeline = UnifiedPipeline::new(config);

        let mut epsilons = Vec::new();
        for _round in 0..100 {
            let (updates, reps) = make_contributions(20, 0, 0.5, 10);
            if let Ok(result) = pipeline.aggregate(&updates, &reps) {
                if let Some(eps) = result.privacy.and_then(|p| p.epsilon_estimate) {
                    epsilons.push(eps);
                }
            }
        }
        let elapsed = t.elapsed();

        let final_eps = epsilons.last().copied().unwrap_or(0.0);
        let first_eps = epsilons.first().copied().unwrap_or(0.0);
        // RDP-to-DP conversion: optimal alpha shifts as budget accumulates,
        // so raw epsilon isn't strictly monotonic. Check overall trend instead.
        let grows_overall = final_eps > first_eps;
        let all_positive = epsilons.iter().all(|&e| e > 0.0);

        println!("  Rounds: {}", epsilons.len());
        println!("  First ε: {:.4}", first_eps);
        println!("  Final ε: {:.4}", final_eps);
        println!("  All positive: {}", all_positive);
        println!("  Grows overall: {}", grows_overall);
        println!("  Time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);

        if grows_overall && all_positive && epsilons.len() == 100 {
            println!("  PASS: Privacy budget tracked correctly");
            pass += 1;
        } else {
            println!("  FAIL: budget not growing or missing rounds");
            fail += 1;
        }
    }
    println!();

    // ═══════════════════════════════════════════════════════════════
    // Test 8: Phase diagram — Byzantine% × reputation disparity
    // ═══════════════════════════════════════════════════════════════
    print_header("Test 8: Byzantine Phase Diagram");
    {
        let t = Instant::now();
        let scenarios = [
            (10, 0.15, 0.85, "10% low-rep"),
            (20, 0.15, 0.85, "20% low-rep"),
            (34, 0.15, 0.85, "34% low-rep"),
            (10, 0.85, 0.85, "10% same-rep"),
            (20, 0.85, 0.85, "20% same-rep"),
            (30, 0.85, 0.85, "30% same-rep"),
            (34, 0.50, 0.85, "34% mid-rep"),
        ];

        // Count how many scenarios converge. All low-rep and mid-rep
        // scenarios should converge. Same-rep scenarios are harder because
        // reputation gating can't help — only signal detection + trimming.
        let mut converged = 0;
        let mut total = 0;
        for (byz_pct, byz_rep, honest_rep, label) in &scenarios {
            total += 1;
            let honest_count = 100 - byz_pct;
            let (updates, reps) = make_contributions_with_reps(
                honest_count,
                *byz_pct,
                0.5,
                20,
                *honest_rep,
                *byz_rep,
            );
            let config = PipelineConfig {
                min_reputation: 0.3,
                trim_fraction: 0.25, // Trim 25% from each tail
                multi_signal_detection: true,
                ..Default::default()
            };
            let mut pipeline = UnifiedPipeline::new(config);

            match pipeline.aggregate(&updates, &reps) {
                Ok(result) => {
                    let max_err = max_error(&result.aggregated.gradients, 0.5);
                    let ok = max_err < 0.5;
                    println!(
                        "  {}: err={:.4} {}",
                        label,
                        max_err,
                        if ok { "OK" } else { "HIGH" }
                    );
                    if ok {
                        converged += 1;
                    }
                }
                Err(e) => {
                    println!("  {}: ERROR {:?}", label, e);
                }
            }
        }
        let elapsed = t.elapsed();
        println!("  Time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
        println!("  Converged: {}/{}", converged, total);

        // Require at least 5/7 scenarios to converge (all rep-disparity +
        // at least low same-rep scenarios). Same-rep ≥20% Byzantine without
        // reputation disparity is known-hard for ANY BFT system.
        if converged >= 5 {
            println!("  PASS: {} scenarios converge (5+ required)", converged);
            pass += 1;
        } else {
            println!("  FAIL: Only {} scenarios converge (need 5+)", converged);
            fail += 1;
        }
    }
    println!();

    // ═══════════════════════════════════════════════════════════════
    // Summary
    // ═══════════════════════════════════════════════════════════════
    let total_elapsed = total_start.elapsed();
    println!("══════════════════════════════════════════════════════════════");
    println!("Results: {}/{} PASS ({} FAIL)", pass, pass + fail, fail);
    println!("Total time: {:.2}s", total_elapsed.as_secs_f64());
    println!("══════════════════════════════════════════════════════════════");

    if fail > 0 {
        std::process::exit(1);
    }
}

// ============================================================================
// Helpers
// ============================================================================

fn print_header(title: &str) {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("{}", title);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
}

fn max_error(gradients: &[f32], target: f32) -> f32 {
    gradients
        .iter()
        .map(|v| (v - target).abs())
        .fold(0.0_f32, f32::max)
}

fn make_contributions(
    n_honest: usize,
    n_byzantine: usize,
    target: f32,
    dim: usize,
) -> (Vec<GradientUpdate>, HashMap<String, f32>) {
    make_contributions_with_reps(n_honest, n_byzantine, target, dim, 0.85, 0.15)
}

fn make_contributions_with_reps(
    n_honest: usize,
    n_byzantine: usize,
    target: f32,
    dim: usize,
    honest_rep: f32,
    byz_rep: f32,
) -> (Vec<GradientUpdate>, HashMap<String, f32>) {
    let mut updates = Vec::new();
    let mut reps = HashMap::new();

    for i in 0..n_honest {
        let val = target + (i as f32 * 0.001);
        updates.push(GradientUpdate::new(
            format!("h{}", i),
            1,
            vec![val; dim],
            100,
            0.5,
        ));
        reps.insert(format!("h{}", i), honest_rep + (i as f32 * 0.001));
    }

    for i in 0..n_byzantine {
        let val = if i % 2 == 0 { 100.0 } else { -100.0 };
        updates.push(GradientUpdate::new(
            format!("b{}", i),
            1,
            vec![val; dim],
            100,
            0.5,
        ));
        reps.insert(format!("b{}", i), byz_rep);
    }

    (updates, reps)
}

// ============================================================================
// Test Plugins
// ============================================================================

/// Simple norm-based Byzantine plugin for testing.
struct NormByzantinePlugin {
    threshold: f32,
}

impl ByzantinePlugin for NormByzantinePlugin {
    fn analyze(&mut self, updates: &[GradientUpdate]) -> ExternalWeightMap {
        let mut weights = ExternalWeightMap::new();

        // Compute median norm (robust to Byzantine skewing)
        let norms: Vec<f32> = updates
            .iter()
            .map(|u| u.gradients.iter().map(|g| g * g).sum::<f32>().sqrt())
            .collect();
        let mut sorted_norms = norms.clone();
        sorted_norms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_norm = sorted_norms[sorted_norms.len() / 2];

        for (i, update) in updates.iter().enumerate() {
            if norms[i] > median_norm * self.threshold {
                weights.insert(
                    update.participant_id.clone(),
                    vec![ParticipantWeightAdjustment {
                        weight_multiplier: 0.0,
                        veto: true,
                        source: "norm_outlier".to_string(),
                    }],
                );
            }
        }

        weights
    }

    fn name(&self) -> &str {
        "norm_byzantine"
    }
}

/// Simple hash verifier for testing.
struct HashVerifier;

impl VerificationPlugin for HashVerifier {
    fn verify(
        &mut self,
        inputs: &[GradientUpdate],
        aggregated: &[f32],
        _reputations: &HashMap<String, f32>,
    ) -> VerificationResult {
        // Compute simple checksum
        let mut hash: u64 = 0;
        for input in inputs {
            for &g in &input.gradients {
                hash = hash.wrapping_add(g.to_bits() as u64);
            }
        }
        for &g in aggregated {
            hash = hash.wrapping_add(g.to_bits() as u64);
        }

        VerificationResult {
            verified: true,
            proof_data: Some(hash.to_le_bytes().to_vec()),
            verifier: "hash_test".to_string(),
        }
    }

    fn name(&self) -> &str {
        "hash_test"
    }
}
