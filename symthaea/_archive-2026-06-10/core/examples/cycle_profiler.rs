// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cycle Profiler — reveals where time is spent in the full cognitive cycle.
//!
//! Runs N text cycles, aggregates per-module timing, and prints a sorted
//! breakdown showing the heaviest subsystems. This is the tool for finding
//! what to optimize when text cycles exceed the 50Hz target (20ms budget).
//!
//! Run: cargo run --example cycle_profiler --release

use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

fn main() {
    let config = CognitiveLoopConfig::default();
    let mut loop_service = CognitiveLoopService::new(config).expect("init cognitive loop");

    let inputs = [
        "the agent perceives a bright red object moving quickly",
        "consciousness emerges from integrated information processing",
        "moral reasoning requires empathy and fairness",
        "the quick brown fox jumps over the lazy dog",
        "causal inference connects observed patterns to explanations",
    ];

    let warmup_cycles = 10;
    let measure_cycles = 50;
    let total_cycles = warmup_cycles + measure_cycles;

    println!("Cycle Profiler: {warmup_cycles} warmup + {measure_cycles} measured cycles");
    println!("=========================================================\n");

    // Collect per-module timing across measured cycles
    let mut timing_sums: Vec<(&str, u64)> = Vec::new();
    let mut total_cycle_us: u64 = 0;

    for i in 0..total_cycles {
        let input = inputs[i % inputs.len()];
        let result = loop_service.cycle(input);

        if i < warmup_cycles {
            continue; // Skip warmup
        }

        total_cycle_us += result.cycle_time_us;

        let t = &result.metadata.module_timings_us;

        // Build timing entries (only on first measured cycle to init, then accumulate)
        let entries: Vec<(&str, u64)> = vec![
            ("core_hdc_encode", t.core_hdc_encode),
            ("core_compress", t.core_compress),
            ("core_semantic_lookup", t.core_semantic_lookup),
            ("core_cfc_step", t.core_cfc_step),
            ("core_predict", t.core_predict),
            ("core_training", t.core_training),
            ("core_parallel_postprocess", t.core_parallel_postprocess),
            ("moral_algebra", t.moral_algebra),
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
            ("stability_regime", t.stability_regime),
            ("world_model", t.world_model),
            ("resonator_codebook", t.resonator_codebook),
            ("high_phi_promotion", t.high_phi_promotion),
            ("demand_consolidation", t.demand_consolidation),
            ("episodic_replay", t.episodic_replay),
            ("living_mind", t.living_mind),
            (
                "master_consciousness_equation",
                t.master_consciousness_equation,
            ),
            ("homeostasis", t.homeostasis),
            ("spectral_mip", t.spectral_mip),
            ("soul_experience", t.soul_experience),
            ("metadata_assembly", t.metadata_assembly),
        ];

        if timing_sums.is_empty() {
            timing_sums = entries.iter().map(|(name, val)| (*name, *val)).collect();
        } else {
            for (idx, (_, val)) in entries.iter().enumerate() {
                timing_sums[idx].1 += val;
            }
        }
    }

    // Compute averages and sort by time descending
    let mut averages: Vec<(&str, f64)> = timing_sums
        .iter()
        .map(|(name, sum)| (*name, *sum as f64 / measure_cycles as f64))
        .collect();
    averages.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let avg_total = total_cycle_us as f64 / measure_cycles as f64;
    let instrumented_sum: f64 = averages.iter().map(|(_, v)| v).sum();
    let uninstrumented = avg_total - instrumented_sum;

    println!(
        "Average cycle time: {:.0} µs ({:.1} ms)",
        avg_total,
        avg_total / 1000.0
    );
    println!(
        "Target: 20,000 µs (50Hz) | Headroom: {:.0} µs\n",
        20_000.0 - avg_total
    );

    println!("{:<40} {:>10} {:>8}", "Module", "Avg (µs)", "% Total");
    println!("{}", "-".repeat(60));

    for (name, avg_us) in &averages {
        if *avg_us < 1.0 {
            continue; // Skip zero-time modules
        }
        let pct = avg_us / avg_total * 100.0;
        println!("{:<40} {:>10.0} {:>7.1}%", name, avg_us, pct);
    }

    if uninstrumented > 0.0 {
        let pct = uninstrumented / avg_total * 100.0;
        println!(
            "{:<40} {:>10.0} {:>7.1}%",
            "(uninstrumented / overhead)", uninstrumented, pct
        );
    }

    println!("{}", "-".repeat(60));
    println!("{:<40} {:>10.0} {:>7.1}%", "TOTAL", avg_total, 100.0);
}