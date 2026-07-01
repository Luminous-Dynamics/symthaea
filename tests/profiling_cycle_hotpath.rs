// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Profiling test: identifies cognitive cycle hot-path modules.
//!
//! Runs 100 warm cycles and reports the top-20 slowest modules
//! using the existing ModuleTimings telemetry (zero overhead instrumentation).
//!
//! ## Usage
//!
//! ```bash
//! cargo test --test profiling_cycle_hotpath -- --nocapture
//! ```

use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

/// Collect module timings from a single CycleResult into (name, microseconds) pairs.
fn extract_timings(t: &symthaea::cognitive_loop::ModuleTimings) -> Vec<(&'static str, u64)> {
    vec![
        ("affective_bridge", t.affective_bridge),
        ("predictive_processing", t.predictive_processing),
        ("cross_modal_binding", t.cross_modal_binding),
        ("surprise_exploration", t.surprise_exploration),
        ("prefrontal", t.prefrontal),
        ("meta_cognition", t.meta_cognition),
        ("narrative_self", t.narrative_self),
        ("gwt", t.gwt),
        ("virtual_body", t.virtual_body),
        ("embodied_cognition", t.embodied_cognition),
        ("dream_replay", t.dream_replay),
        ("moral_algebra", t.moral_algebra),
        ("consciousness_resonance", t.consciousness_resonance),
        ("temporal_consciousness", t.temporal_consciousness),
        ("attention_schema", t.attention_schema),
        ("narrative_gwt", t.narrative_gwt),
        (
            "consciousness_thermodynamics",
            t.consciousness_thermodynamics,
        ),
        ("phenomenal_binding", t.phenomenal_binding),
        ("hierarchical_free_energy", t.hierarchical_free_energy),
        ("resonator_recall", t.resonator_recall),
        ("support_intelligence", t.support_intelligence),
        ("temporal_analyzer", t.temporal_analyzer),
        ("primitive_lattice", t.primitive_lattice),
        ("compositionality", t.compositionality),
        ("value_evaluator", t.value_evaluator),
        ("consciousness_profile", t.consciousness_profile),
        ("harmonics", t.harmonics),
        ("primitive_reasoning", t.primitive_reasoning),
        ("causal_explanation", t.causal_explanation),
        ("adaptive_reasoning", t.adaptive_reasoning),
        ("epistemic_tiers", t.epistemic_tiers),
        ("phi_validation", t.phi_validation),
        ("dissipative_consciousness", t.dissipative_consciousness),
        ("epistemic_conflict", t.epistemic_conflict),
        ("consciousness_equation_v2", t.consciousness_equation_v2),
        ("hierarchical_ltc", t.hierarchical_ltc),
        ("primitive_evolution", t.primitive_evolution),
        ("consciousness_holography", t.consciousness_holography),
        (
            "differentiable_consciousness",
            t.differentiable_consciousness,
        ),
        ("affective_consciousness", t.affective_consciousness),
        (
            "unified_consciousness_pipeline",
            t.unified_consciousness_pipeline,
        ),
        ("multi_modal_integration", t.multi_modal_integration),
        ("synthetic_grounding", t.synthetic_grounding),
        ("epistemic_gate", t.epistemic_gate),
        ("semantic_value_embedder", t.semantic_value_embedder),
        ("composition_rules", t.composition_rules),
        ("harmonies_integration", t.harmonies_integration),
        ("meta_cognitive_reasoning", t.meta_cognitive_reasoning),
        ("code_primitive_routing", t.code_primitive_routing),
        ("empathic_unification", t.empathic_unification),
        ("multi_objective_evolution", t.multi_objective_evolution),
        ("grid_encoder", t.grid_encoder),
        ("stability_regime", t.stability_regime),
        // Core pipeline phases
        ("core_hdc_encode", t.core_hdc_encode),
        ("core_compress", t.core_compress),
        ("core_semantic_lookup", t.core_semantic_lookup),
        ("core_cfc_step", t.core_cfc_step),
        ("core_predict", t.core_predict),
        ("core_training", t.core_training),
        ("core_parallel_postprocess", t.core_parallel_postprocess),
        // End-of-cycle
        ("living_mind", t.living_mind),
        (
            "master_consciousness_equation",
            t.master_consciousness_equation,
        ),
        ("homeostasis", t.homeostasis),
        ("spectral_mip", t.spectral_mip),
        ("soul_experience", t.soul_experience),
        ("metadata_assembly", t.metadata_assembly),
        // Consciousness engine
        ("consciousness_engine", t.consciousness_engine),
        (
            "consciousness_engine_equation_v2",
            t.consciousness_engine_equation_v2,
        ),
        (
            "consciousness_engine_pipeline",
            t.consciousness_engine_pipeline,
        ),
        (
            "consciousness_engine_multimodal",
            t.consciousness_engine_multimodal,
        ),
        // Ethics engine
        ("ethics_engine", t.ethics_engine),
        ("ethics_engine_moral", t.ethics_engine_moral),
        ("ethics_engine_value", t.ethics_engine_value),
        ("ethics_engine_harmonies", t.ethics_engine_harmonies),
        // Mid-cycle
        ("world_model", t.world_model),
        ("resonator_codebook", t.resonator_codebook),
        ("high_phi_promotion", t.high_phi_promotion),
        ("demand_consolidation", t.demand_consolidation),
        ("episodic_replay", t.episodic_replay),
        ("parameter_optimization", t.parameter_optimization),
        ("dream_cycle", t.dream_cycle),
        ("moral_topology", t.moral_topology),
        ("math_service", t.math_service),
    ]
}

#[test]
fn profile_cycle_hotpath_top20() {
    let config = CognitiveLoopConfig {
        async_training: false,
        enable_surprise_exploration: true,
        enable_prefrontal: true,
        enable_meta_cognition: true,
        enable_predictive_processing: true,
        enable_cross_modal_binding: true,
        enable_affective_bridge: true,
        enable_embodied_cognition: true,
        enable_narrative_gwt: true,
        causal_enhancement: true,
        ..Default::default()
    };
    let mut service = CognitiveLoopService::new(config).unwrap();

    // Warm up: 10 cycles
    for _ in 0..10 {
        service.cycle("warm up input for profiling");
    }

    // Profile: accumulate timings over 100 cycles
    let n_cycles = 100;
    let mut accum: Vec<(&str, u64)> = Vec::new();

    let mut total_cycle_us: u64 = 0;

    let inputs = [
        "the sun rises in the east",
        "cause and effect are linked",
        "patterns emerge from chaos",
        "consciousness arises from integration",
        "temporal prediction drives learning",
        "moral reasoning requires empathy",
        "exploration balances exploitation",
        "free energy minimization is universal",
        "hyperdimensional binding creates meaning",
        "precision weighting modulates attention",
    ];

    for i in 0..n_cycles {
        let result = service.cycle(inputs[i % inputs.len()]);
        total_cycle_us += result.cycle_time_us;

        let timings = extract_timings(&result.metadata.module_timings_us);
        if accum.is_empty() {
            accum = timings;
        } else {
            for (j, (_, us)) in timings.iter().enumerate() {
                accum[j].1 += us;
            }
        }
    }

    // Sort by total time descending
    accum.sort_by(|a, b| b.1.cmp(&a.1));

    // Report
    let avg_cycle_us = total_cycle_us / n_cycles as u64;
    println!("\n{}", "=".repeat(72));
    println!("COGNITIVE CYCLE PROFILING ({n_cycles} cycles, avg {avg_cycle_us} us/cycle)");
    println!("{}", "=".repeat(72));
    println!(
        "{:<40} {:>10} {:>10} {:>6}",
        "MODULE", "TOTAL (us)", "AVG (us)", "% CYCLE"
    );
    println!("{}", "-".repeat(72));

    let instrumented_total: u64 = accum.iter().map(|(_, us)| us).sum();

    for (name, total_us) in accum.iter().take(25) {
        if *total_us == 0 {
            continue;
        }
        let avg_us = total_us / n_cycles as u64;
        let pct = if total_cycle_us > 0 {
            (*total_us as f64 / total_cycle_us as f64) * 100.0
        } else {
            0.0
        };
        println!("{:<40} {:>10} {:>10} {:>5.1}%", name, total_us, avg_us, pct);
    }

    println!("{}", "-".repeat(72));
    println!(
        "{:<40} {:>10} {:>10} {:>5.1}%",
        "INSTRUMENTED TOTAL",
        instrumented_total,
        instrumented_total / n_cycles as u64,
        (instrumented_total as f64 / total_cycle_us as f64) * 100.0
    );
    println!(
        "{:<40} {:>10} {:>10} {:>5.1}%",
        "UNINSTRUMENTED (overhead)",
        total_cycle_us.saturating_sub(instrumented_total),
        total_cycle_us.saturating_sub(instrumented_total) / n_cycles as u64,
        ((total_cycle_us.saturating_sub(instrumented_total)) as f64 / total_cycle_us as f64)
            * 100.0
    );
    println!(
        "{:<40} {:>10} {:>10}",
        "TOTAL", total_cycle_us, avg_cycle_us
    );

    // Sanity check: cycle should complete (debug builds are 5-10x slower than release).
    // In release with async_training=true, target is 50ms (20Hz).
    // Debug with sync training: ~150-200ms is expected.
    assert!(
        avg_cycle_us < 500_000,
        "avg cycle time {avg_cycle_us} us exceeds 500ms — something is very wrong"
    );
}

#[test]
fn profile_core_pipeline_breakdown() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        async_training: false,
        ..Default::default()
    })
    .unwrap();

    // Warm up
    for _ in 0..10 {
        service.cycle("warmup");
    }

    // Collect 50 cycles of core pipeline timings
    let n = 50;
    let mut encode_total = 0u64;
    let mut compress_total = 0u64;
    let mut semantic_total = 0u64;
    let mut cfc_total = 0u64;
    let mut predict_total = 0u64;
    let mut training_total = 0u64;
    let mut parallel_total = 0u64;
    let mut cycle_total = 0u64;

    for i in 0..n {
        let r = service.cycle(&format!("profiling core pipeline iteration {i}"));
        let t = &r.metadata.module_timings_us;
        encode_total += t.core_hdc_encode;
        compress_total += t.core_compress;
        semantic_total += t.core_semantic_lookup;
        cfc_total += t.core_cfc_step;
        predict_total += t.core_predict;
        training_total += t.core_training;
        parallel_total += t.core_parallel_postprocess;
        cycle_total += r.cycle_time_us;
    }

    println!(
        "\nCORE PIPELINE BREAKDOWN ({n} cycles, avg {} us/cycle)",
        cycle_total / n as u64
    );
    println!("{:<30} {:>8} {:>8} {:>6}", "PHASE", "AVG(us)", "TOTAL", "%");
    println!("{}", "-".repeat(56));
    let phases = [
        ("hdc_encode", encode_total),
        ("compress", compress_total),
        ("semantic_lookup", semantic_total),
        ("cfc_step", cfc_total),
        ("predict", predict_total),
        ("training", training_total),
        ("parallel_postprocess", parallel_total),
    ];
    let core_total: u64 = phases.iter().map(|(_, t)| t).sum();
    for (name, total) in &phases {
        println!(
            "{:<30} {:>8} {:>8} {:>5.1}%",
            name,
            total / n as u64,
            total,
            (*total as f64 / cycle_total as f64) * 100.0
        );
    }
    println!("{}", "-".repeat(56));
    println!(
        "{:<30} {:>8} {:>8} {:>5.1}%",
        "core_pipeline_total",
        core_total / n as u64,
        core_total,
        (core_total as f64 / cycle_total as f64) * 100.0
    );
}