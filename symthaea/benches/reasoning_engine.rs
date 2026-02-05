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

use criterion::{black_box, criterion_group, criterion_main, Criterion};

#[cfg(feature = "reasoning_engine")]
mod benches {
    use super::*;
    use symthaea::consciousness::epistemic_conflict::{
        ConflictDetector, TheoryCalibrator, MultiTheoryMetrics,
        phi_integration::effective_phi,
    };
    use symthaea::consciousness::tool_gate::types::ToolDescriptor;
    use symthaea::consciousness::tool_gate::classifier;
    use symthaea::consciousness::reasoning_engine::{
        ConsciousReasoningEngine, ReasoningContext,
    };
    use symthaea::consciousness::temporal_planning::types::PlannedAction;

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
            b.iter(|| {
                effective_phi(black_box(0.8), black_box(0.7), black_box(2.0))
            })
        });
    }

    pub fn bench_tool_classification(c: &mut Criterion) {
        let tool = ToolDescriptor::from_command("nixos-rebuild switch")
            .with_domain("nixos")
            .with_rollback("nixos-rebuild switch --rollback")
            .with_calibration_count(100);

        c.bench_function("tool_classification", |b| {
            b.iter(|| {
                classifier::gate(black_box(&tool), black_box(0.7), black_box(0.8))
            })
        });
    }

    pub fn bench_tier0_cycle(c: &mut Criterion) {
        let mut engine = ConsciousReasoningEngine::new();
        let ctx = ReasoningContext {
            theory_metrics: make_metrics(0.8),
            phi: 0.8,
            available_budget_us: 1_000, // force Tier 0
            available_actions: make_actions(),
            tool: None,
            recent_utility: 0.5,
            cycle_id: 0,
        };

        c.bench_function("full_tier0_cycle", |b| {
            b.iter(|| {
                engine.reason(black_box(&ctx));
            })
        });
    }

    pub fn bench_tier2_cycle(c: &mut Criterion) {
        let mut engine = ConsciousReasoningEngine::new();
        let ctx = ReasoningContext {
            theory_metrics: make_metrics(0.8),
            phi: 0.8,
            available_budget_us: 25_000, // Tier 2
            available_actions: make_actions(),
            tool: Some(
                ToolDescriptor::from_command("nix build .#pkg")
                    .with_domain("nixos")
                    .with_rollback("nix store delete")
                    .with_calibration_count(100),
            ),
            recent_utility: 0.5,
            cycle_id: 0,
        };

        c.bench_function("full_tier2_cycle", |b| {
            b.iter(|| {
                engine.reason(black_box(&ctx));
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
    benches::bench_tier2_cycle,
);

#[cfg(feature = "reasoning_engine")]
criterion_main!(reasoning_engine);

// Stub main for when feature is not enabled
#[cfg(not(feature = "reasoning_engine"))]
fn main() {
    eprintln!("Enable the 'reasoning_engine' feature to run these benchmarks:");
    eprintln!("  cargo bench --bench reasoning_engine --features reasoning_engine");
}
