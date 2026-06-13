// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Reasoning Engine Benchmarks
//!
//! Criterion benchmarks for the ConsciousReasoningEngine.
//!
//! ## Benchmarks
//!
//! - `conflict_detection`: <500μs
//! - `effective_phi`: <50μs
//! - `tool_classification`: <200μs
//! - `full_tier0_cycle`: <2ms
//! - `full_tier1_cycle`: <8ms
//! - `full_tier2_cycle`: <20ms
//!
//! ```bash
//! cargo bench --bench reasoning_engine --features reasoning_engine
//! ```

use criterion::{Criterion, black_box, criterion_group, criterion_main};

#[cfg(feature = "reasoning_engine")]
mod benches {
    use super::*;
    use symthaea::consciousness::epistemic_conflict::{
        ConflictDetector, MultiTheoryMetrics, TheoryCalibrator, phi_integration::effective_phi,
    };
    use symthaea::consciousness::reasoning_engine::{ConsciousReasoningEngine, ReasoningContext};
    use symthaea::consciousness::temporal_planning::types::PlannedAction;
    use symthaea::consciousness::tool_gate::classifier;
    use symthaea::consciousness::tool_gate::types::ToolDescriptor;

    fn make_metrics(consensus: f64) -> MultiTheoryMetrics {
        MultiTheoryMetrics {
            phi: 0.8,
            gwt: consensus,
            ast: consensus,
            pp: consensus,
            rpt: consensus,
            embodiment: consensus,
            unified: 0.8 * 0.2 + consensus * 0.8,
        }
    }

    fn make_actions() -> Vec<PlannedAction> {
        (0..5)
            .map(|i| PlannedAction {
                id: format!("action_{}", i),
                description: format!("Action {}", i),
                embedding: vec![i as f32 * 0.2; 4],
                prior: 1.0 / 5.0,
                is_epistemic: i % 2 == 0,
            })
            .collect()
    }

    pub fn bench_conflict_detection(c: &mut Criterion) {
        let mut detector = ConflictDetector::new();
        let metrics = make_metrics(0.7);

        c.bench_function("conflict_detection", |b| {
            b.iter(|| {
                detector.detect(black_box(&metrics));
            })
        });
    }

    pub fn bench_effective_phi(c: &mut Criterion) {
        c.bench_function("effective_phi", |b| {
            b.iter(|| effective_phi(black_box(0.8), black_box(0.7), black_box(2.0)))
        });
    }

    pub fn bench_tool_classification(c: &mut Criterion) {
        let tool = ToolDescriptor::from_command("nixos-rebuild switch")
            .with_domain("nixos")
            .with_rollback("nixos-rebuild switch --rollback")
            .with_calibration_count(100);

        c.bench_function("tool_classification", |b| {
            b.iter(|| classifier::gate(black_box(&tool), black_box(0.7), black_box(0.8)))
        });
    }

    pub fn bench_tier0_cycle(c: &mut Criterion) {
        let mut engine = ConsciousReasoningEngine::new();
        let ctx = ReasoningContext {
            negative_prototypes: Default::default(),
            substrate_cost_model: Default::default(),
            theory_metrics: make_metrics(0.8),
            phi: 0.8,
            available_budget_us: 1_000, // force Tier 0
            available_actions: make_actions(),
            code_context: None,
            tool: None,
            recent_utility: 0.5,
            cycle_id: 0,
            neuromod_exploration_mod: 1.0,
            epistemic_quality: 0.5,
        };

        c.bench_function("full_tier0_cycle", |b| {
            b.iter(|| {
                engine.reason(black_box(&ctx));
            })
        });
    }

    pub fn bench_tier1_cycle(c: &mut Criterion) {
        let mut engine = ConsciousReasoningEngine::new();
        let ctx = ReasoningContext {
            negative_prototypes: Default::default(),
            substrate_cost_model: Default::default(),
            theory_metrics: make_metrics(0.7),
            phi: 0.8,
            available_budget_us: 8_000, // Tier 1
            available_actions: make_actions(),
            code_context: None,
            tool: None,
            recent_utility: 0.5,
            cycle_id: 0,
            neuromod_exploration_mod: 1.0,
            epistemic_quality: 0.5,
        };

        c.bench_function("full_tier1_cycle", |b| {
            b.iter(|| {
                engine.reason(black_box(&ctx));
            })
        });
    }

    pub fn bench_tier2_cycle(c: &mut Criterion) {
        let mut engine = ConsciousReasoningEngine::new();
        let ctx = ReasoningContext {
            negative_prototypes: Default::default(),
            substrate_cost_model: Default::default(),
            theory_metrics: make_metrics(0.8),
            phi: 0.8,
            available_budget_us: 25_000, // Tier 2
            available_actions: make_actions(),
            code_context: None,
            tool: Some(
                ToolDescriptor::from_command("nix build .#pkg")
                    .with_domain("nixos")
                    .with_rollback("nix store delete")
                    .with_calibration_count(100),
            ),
            recent_utility: 0.5,
            cycle_id: 0,
            neuromod_exploration_mod: 1.0,
            epistemic_quality: 0.5,
        };

        c.bench_function("full_tier2_cycle", |b| {
            b.iter(|| {
                engine.reason(black_box(&ctx));
            })
        });
    }

    pub fn bench_evs(c: &mut Criterion) {
        use symthaea::consciousness::temporal_planning::mcts::evs;

        c.bench_function("evs_calculation", |b| {
            b.iter(|| evs(black_box(0.5), black_box(0.7), black_box(5), black_box(0.5)))
        });
    }

    pub fn bench_multi_cycle_stability(c: &mut Criterion) {
        c.bench_function("50_cycle_stability", |b| {
            b.iter(|| {
                let mut engine = ConsciousReasoningEngine::new();
                for i in 0..50 {
                    let consensus = 0.5 + 0.3 * ((i as f64 * 0.1).sin());
                    let ctx = ReasoningContext {
                        negative_prototypes: Default::default(),
                        substrate_cost_model: Default::default(),
                        theory_metrics: MultiTheoryMetrics {
                            phi: 0.8,
                            gwt: consensus,
                            ast: consensus,
                            pp: consensus,
                            rpt: consensus,
                            embodiment: consensus,
                            unified: 0.8 * 0.2 + consensus * 0.8,
                        },
                        phi: 0.8,
                        available_budget_us: 20_000,
                        available_actions: make_actions(),
                        code_context: None,
                        tool: None,
                        recent_utility: 0.5,
                        cycle_id: i,
                        neuromod_exploration_mod: 1.0,
                        epistemic_quality: 0.5,
                    };
                    black_box(engine.reason(&ctx));
                }
                engine.stats()
            })
        });
    }
}

#[cfg(feature = "reasoning_engine")]
criterion_group!(
    reasoning_engine,
    benches::bench_conflict_detection,
    benches::bench_effective_phi,
    benches::bench_tool_classification,
    benches::bench_tier0_cycle,
    benches::bench_tier1_cycle,
    benches::bench_tier2_cycle,
    benches::bench_evs,
    benches::bench_multi_cycle_stability,
);

#[cfg(feature = "reasoning_engine")]
criterion_main!(reasoning_engine);

// Stub main for when feature is not enabled
#[cfg(not(feature = "reasoning_engine"))]
fn main() {
    eprintln!("Enable the 'reasoning_engine' feature to run these benchmarks:");
    eprintln!("  cargo bench --bench reasoning_engine --features reasoning_engine");
}
