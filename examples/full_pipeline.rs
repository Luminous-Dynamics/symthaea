// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Full Pipeline Demo
//!
//! End-to-end demonstration of Symthaea's cognitive architecture:
//!
//! 1. **Learning**: CognitiveLoopService processes patterned input, reducing prediction error
//! 2. **Composition**: CompositionalityEngine composes primitives via sequential + parallel ops
//! 3. **Adaptation**: Harder input drives adaptive HDC dimension scaling up
//! 4. **Consolidation**: Easy input lets dimensions scale back down
//!
//! ```bash
//! cargo run --example full_pipeline
//! ```

use std::sync::Arc;

use symthaea::cognitive_loop::{
    CognitiveLoopConfig, CognitiveLoopService, TemporalBackend, TrainingMethod,
};
use symthaea::consciousness::{CompositionalityConfig, CompositionalityEngine};
use symthaea::hdc::binary_hv::BinaryHV;
use symthaea::hdc::primitive_system::PrimitiveSystem;
use symthaea::hdc_ltc_bridge::{AdaptiveDimConfig, HdcLtcBridgeConfig};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║           Symthaea Full Pipeline Demonstration              ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // ── Configure ──────────────────────────────────────────────────────────
    let adaptive = AdaptiveDimConfig {
        min_dim: 1024,
        max_dim: 8192,
        upscale_error_threshold: 0.7,
        downscale_error_threshold: 0.25,
        scale_step: 1024,
        cooldown_cycles: 10,
    };

    let config = CognitiveLoopConfig {
        temporal_backend: TemporalBackend::HdcLtcUnified,
        genesis_phrase: Some("symthaea-demo-2026".to_string()),
        training_method: TrainingMethod::BpttWithSpsaFallback,
        hdc_ltc_config: HdcLtcBridgeConfig {
            hdc_dim: 2048,
            adaptive_dim: Some(adaptive),
            ..HdcLtcBridgeConfig::default()
        },
        ..Default::default()
    };

    let training_method = format!("{:?}", config.training_method);
    let mut service =
        CognitiveLoopService::new(config).expect("Failed to create CognitiveLoopService");

    println!("Backend:          {:?}", service.temporal_backend());
    println!("Training method:  {}", training_method);
    println!("Genesis phrase:   symthaea-demo-2026");
    println!();

    // ══════════════════════════════════════════════════════════════════════
    // Phase 1: Learning  (50 cycles of patterned input)
    // ══════════════════════════════════════════════════════════════════════
    println!("── Phase 1: Learning (50 cycles, patterned input) ─────────────");
    println!(
        "{:<8} {:>10} {:>12} {:>10}",
        "Cycle", "Phi", "Coherence", "Loss"
    );

    let patterns = [
        "cause leads to effect",
        "the sun warms the earth",
        "water flows downhill",
        "action produces reaction",
        "seeds grow into trees",
    ];

    for i in 0..50 {
        let input = patterns[i % patterns.len()];
        let result = service.cycle(input);

        if (i + 1) % 10 == 0 {
            let s = service.stats();
            println!(
                "{:<8} {:>10.4} {:>12.4} {:>10.4}",
                i + 1,
                s.unified_psi,
                s.temporal_coherence,
                result.training_loss.unwrap_or(0.0),
            );
        }
    }
    println!();

    // ══════════════════════════════════════════════════════════════════════
    // Phase 2: Composition  (primitive algebra)
    // ══════════════════════════════════════════════════════════════════════
    println!("── Phase 2: Composition (primitive algebra) ───────────────────");

    let ps = Arc::new(PrimitiveSystem::new());
    let mut comp = CompositionalityEngine::new(ps, CompositionalityConfig::default());

    // Sequential: (cause ∘ similarity)
    let seq = comp
        .compose_sequential("cause", "similarity")
        .expect("sequential composition failed");
    println!(
        "  Sequential:  {} | depth={} | est. Phi={:.3}",
        seq.name, seq.metadata.depth, seq.metadata.expected_phi_contribution,
    );

    // Parallel: (time_arrow || negation)
    let par = comp
        .compose_parallel("time_arrow", "negation")
        .expect("parallel composition failed");
    println!(
        "  Parallel:    {} | depth={} | est. Phi={:.3}",
        par.name, par.metadata.depth, par.metadata.expected_phi_contribution,
    );

    // Nested: sequential of (seq result) and parallel result
    let nested = comp
        .compose_sequential(&seq.id, &par.id)
        .expect("nested composition failed");
    println!(
        "  Nested:      {} | depth={} | est. Phi={:.3}",
        nested.name, nested.metadata.depth, nested.metadata.expected_phi_contribution,
    );

    // Execute the nested composition
    let test_input = BinaryHV::random(12345);
    let exec_result = comp
        .execute(&nested.id, &test_input)
        .expect("execution failed");
    println!(
        "  Execution:   confidence={:.3} | steps={} | path={:?}",
        exec_result.confidence, exec_result.iterations, exec_result.execution_path,
    );
    println!();

    // ══════════════════════════════════════════════════════════════════════
    // Phase 3: Adaptation  (30 cycles of harder / novel input)
    // ══════════════════════════════════════════════════════════════════════
    println!("── Phase 3: Adaptation (30 cycles, harder input) ──────────────");
    println!(
        "{:<8} {:>10} {:>12} {:>10}",
        "Cycle", "Phi", "Coherence", "Loss"
    );

    let hard_patterns = [
        "quantum entanglement violates local realism through nonlocal correlations",
        "goedel incompleteness means no consistent system proves its own consistency",
        "epigenetic methylation patterns modulate gene expression across generations",
        "turbulent navier-stokes solutions remain an open millennium problem",
        "consciousness may arise from integrated information across recurrent networks",
        "category theory unifies algebraic structures through functorial mappings",
    ];

    for i in 0..30 {
        let input = hard_patterns[i % hard_patterns.len()];
        let result = service.cycle(input);

        if (i + 1) % 10 == 0 {
            let s = service.stats();
            println!(
                "{:<8} {:>10.4} {:>12.4} {:>10.4}",
                50 + i + 1,
                s.unified_psi,
                s.temporal_coherence,
                result.training_loss.unwrap_or(0.0),
            );
        }
    }
    println!();

    // ══════════════════════════════════════════════════════════════════════
    // Phase 4: Consolidation  (30 cycles of easy input)
    // ══════════════════════════════════════════════════════════════════════
    println!("── Phase 4: Consolidation (30 cycles, easy input) ─────────────");
    println!(
        "{:<8} {:>10} {:>12} {:>10}",
        "Cycle", "Phi", "Coherence", "Loss"
    );

    let easy_patterns = ["hello world", "yes or no", "one two three"];

    for i in 0..30 {
        let input = easy_patterns[i % easy_patterns.len()];
        let result = service.cycle(input);

        if (i + 1) % 10 == 0 {
            let s = service.stats();
            println!(
                "{:<8} {:>10.4} {:>12.4} {:>10.4}",
                80 + i + 1,
                s.unified_psi,
                s.temporal_coherence,
                result.training_loss.unwrap_or(0.0),
            );
        }
    }
    println!();

    // ══════════════════════════════════════════════════════════════════════
    // Final Summary
    // ══════════════════════════════════════════════════════════════════════
    let s = service.stats();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                     Final Summary                           ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Total cycles:        {:<37}║", s.total_cycles);
    println!("║  Final Phi:           {:<37.4}║", s.unified_psi);
    println!("║  Final coherence:     {:<37.4}║", s.temporal_coherence);
    println!("║  Avg prediction err:  {:<37.4}║", s.avg_prediction_error);
    println!("║  Compositions made:   {:<37}║", 3);
    println!("║  Training method:     {:<37}║", training_method);
    println!("║  Consciousness:       {:<37}║", s.consciousness_pattern);
    println!("║  Strategy:            {:<37}║", s.current_strategy);
    println!("╚══════════════════════════════════════════════════════════════╝");
}
