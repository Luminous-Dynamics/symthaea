// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # Abstract Thought Demo — Empirical Exploration
//!
//! Runs Symthaea's `ConjectureEngine` with `abstract_thought` enabled on real
//! mathematical sequences from multiple domains, then reports what the system
//! actually discovers: concept vector clusters, recurring subtrees, promoted
//! macro-operators, and cross-domain formula matches.
//!
//! This is an honest empirical exploration — not a test. Its purpose is to
//! observe the system's behavior on real input and surface what works and
//! what doesn't.
//!
//! ## Run
//!
//! ```sh
//! cargo run -p symthaea-core --example abstract_thought_demo --features abstract_thought --release
//! ```

use symthaea_core::hdc::abstract_thought::macro_quality::{
    MacroQualityReport, MacroQualityThresholds, evaluate_common_metrics, maybe_enforce,
    print_report,
};
use symthaea_core::hdc::conjecture_engine::{
    ConjectureEngine, ConjectureStatus, MathDomain, ObservedSequence, observe_fibonacci_ratios,
    observe_partitions,
};
use symthaea_core::hdc::primitive_system::PrimitiveSystem;

fn section(title: &str) {
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  {}", title);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
}

fn print_macro_pool_snapshot(engine: &ConjectureEngine, label: &str) {
    let Some(metrics) = engine.macro_pool_metrics() else {
        return;
    };

    println!("\n  ── Macro pool quality ({label}) ──");
    println!(
        "    cycle={}  operators={}  candidates={}  promoted={}  pruned={}  survival={:.0}%",
        metrics.cycle,
        metrics.total_operators,
        metrics.total_candidates,
        metrics.total_promoted,
        metrics.total_pruned,
        metrics.survival_rate * 100.0
    );
    println!(
        "    tiers: formal={}  recurrent={}  quarantined={}",
        metrics.formal_operators, metrics.recurrent_operators, metrics.quarantined_operators
    );
    println!(
        "    usage: active_precision={:.0}%  mature_precision={:.0}%  avg_usage={:.2}  avg_sources={:.2}",
        metrics.active_precision * 100.0,
        metrics.mature_precision * 100.0,
        metrics.avg_usage_count,
        metrics.avg_source_count
    );

    if metrics.signature_stats.is_empty() {
        println!("    top signatures: (none)");
    } else {
        let mut signature_stats = metrics.signature_stats;
        signature_stats.sort_by(|a, b| {
            b.used_operator_count
                .cmp(&a.used_operator_count)
                .then_with(|| b.total_usage_count.cmp(&a.total_usage_count))
                .then_with(|| b.operator_count.cmp(&a.operator_count))
                .then_with(|| a.signature.cmp(&b.signature))
        });

        println!("    top signatures:");
        for stat in signature_stats.iter().take(5) {
            println!(
                "      · {:<18} ops={} used={} hits={}",
                stat.signature,
                stat.operator_count,
                stat.used_operator_count,
                stat.total_usage_count
            );
        }
    }

    println!("    macro metadata:");
    for op in engine.macro_operators().iter().take(8) {
        let parent = op
            .parent_formulas
            .first()
            .map(|s| s.as_str())
            .unwrap_or("(none)");
        println!(
            "      · {} | sig={} tier={:?} sources={} usage={} arity={} cycle={} parent={}",
            op.template,
            op.signature,
            op.promotion_tier,
            op.source_count,
            op.usage_count,
            op.arity,
            op.created_at,
            parent
        );
    }
}

/// Triangular numbers: T(n) = n(n+1)/2
fn triangular_sequence(max_n: usize) -> ObservedSequence {
    let data: Vec<(f64, f64)> = (1..=max_n)
        .map(|n| (n as f64, (n * (n + 1)) as f64 / 2.0))
        .collect();
    ObservedSequence::new("triangular(n)", MathDomain::Combinatorics, data)
}

/// Square numbers: n²
fn squares_sequence(max_n: usize) -> ObservedSequence {
    let data: Vec<(f64, f64)> = (1..=max_n).map(|n| (n as f64, (n * n) as f64)).collect();
    ObservedSequence::new("squares(n)", MathDomain::NumberTheory, data)
}

/// Cubes: n³
fn cubes_sequence(max_n: usize) -> ObservedSequence {
    let data: Vec<(f64, f64)> = (1..=max_n)
        .map(|n| (n as f64, (n * n * n) as f64))
        .collect();
    ObservedSequence::new("cubes(n)", MathDomain::NumberTheory, data)
}

/// Harmonic numbers: H(n) = 1 + 1/2 + ... + 1/n
fn harmonic_sequence(max_n: usize) -> ObservedSequence {
    let mut sum = 0.0;
    let data: Vec<(f64, f64)> = (1..=max_n)
        .map(|n| {
            sum += 1.0 / n as f64;
            (n as f64, sum)
        })
        .collect();
    ObservedSequence::new("harmonic(n)", MathDomain::NumberTheory, data)
}

/// "Physics-like" sequence: kinetic energy E = 0.5 * n² (analogue of E ~ v²)
fn kinetic_energy_sequence(max_n: usize) -> ObservedSequence {
    let data: Vec<(f64, f64)> = (1..=max_n)
        .map(|n| (n as f64, 0.5 * (n * n) as f64))
        .collect();
    ObservedSequence::new("kinetic_energy(n)", MathDomain::Physics, data)
}

/// "Biology-like": logistic population growth 1 / (1 + exp(-(n-10)))
fn logistic_sequence(max_n: usize) -> ObservedSequence {
    let data: Vec<(f64, f64)> = (1..=max_n)
        .map(|n| {
            let x = n as f64 - 10.0;
            (n as f64, 100.0 / (1.0 + (-x * 0.5).exp()))
        })
        .collect();
    ObservedSequence::new("logistic_growth(n)", MathDomain::Biology, data)
}

fn main() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  Symthaea Abstract Thought — Empirical Exploration             ║");
    println!("║  Meta-HDC · Dynamic Grammar · Category Discovery               ║");
    println!("╚════════════════════════════════════════════════════════════════╝");

    let prims = PrimitiveSystem::new();
    let mut engine = ConjectureEngine::new();
    engine.enable_abstract_thought();

    // ── 1. Feed real sequences from 4 domains ──
    section("Phase 1: Observing sequences from 4 domains");
    engine.observe(triangular_sequence(25));
    engine.observe(squares_sequence(25));
    engine.observe(cubes_sequence(25));
    engine.observe(harmonic_sequence(25));
    engine.observe(observe_fibonacci_ratios(25));
    engine.observe(observe_partitions(20));
    engine.observe(kinetic_energy_sequence(25));
    engine.observe(logistic_sequence(25));

    println!("  Observed {} sequences:", engine.observations.len());
    for seq in &engine.observations {
        println!(
            "    · {:<25} ({:?}, {} data points)",
            seq.name,
            seq.domain,
            seq.data.len()
        );
    }

    // ── 2. Generate conjectures ──
    section("Phase 2: First GP run — no macros yet");
    let round1_start = std::time::Instant::now();
    engine.generate_conjectures(3);
    let round1_time = round1_start.elapsed();
    let round1_count = engine.conjectures.len();
    println!(
        "  Generated {} conjectures in {:.2}s",
        round1_count,
        round1_time.as_secs_f64()
    );

    // Show the top conjecture per sequence
    println!("\n  Top conjecture per source sequence:");
    let mut shown_sources = std::collections::HashSet::new();
    for c in engine.conjectures.iter() {
        if shown_sources.insert(c.source.clone()) {
            let status_str = match &c.status {
                ConjectureStatus::Proposed => "proposed",
                ConjectureStatus::NumericallyTested { .. } => "tested",
                ConjectureStatus::SymbolicallyChecked => "checked",
                ConjectureStatus::FormallyVerified { .. } => "VERIFIED",
                ConjectureStatus::Refuted { .. } => "refuted",
            };
            println!(
                "    · {:<25} → {} [{}] (mse={:.3e})",
                c.source, c.formula_str, status_str, c.training_mse
            );
        }
    }

    // ── 3. Verify to upgrade status ──
    section("Phase 3: Verification");
    engine.verify_numerical();
    engine.verify_formal(50);
    let verified_count = engine
        .conjectures
        .iter()
        .filter(|c| matches!(c.status, ConjectureStatus::FormallyVerified { .. }))
        .count();
    let tested_count = engine
        .conjectures
        .iter()
        .filter(|c| matches!(c.status, ConjectureStatus::NumericallyTested { .. }))
        .count();
    println!("  Formally verified: {}", verified_count);
    println!("  Numerically tested: {}", tested_count);

    // ── 4. Reflect — the moment abstract thought kicks in ──
    section("Phase 4: Reflect — Meta-HDC + Grammar + Category");
    engine.reflect(&prims);

    let at = engine.abstract_thought.as_ref().unwrap();

    println!("\n  ── Meta-HDC state ──");
    println!(
        "    Concept vectors encoded: {}",
        at.meta_hdc.concepts.len()
    );
    println!(
        "    Clusters formed:         {}",
        at.meta_hdc.clusters.len()
    );

    for (i, cluster) in at.meta_hdc.clusters.iter().enumerate() {
        println!(
            "    Cluster #{}: {} members, {} distinct domains",
            i,
            cluster.members.len(),
            cluster.domain_diversity
        );
        // Show the source sequences in this cluster
        let sources: Vec<String> = cluster
            .members
            .iter()
            .filter_map(|&m| {
                let conj_idx = at.meta_hdc.concepts[m].conjecture_idx;
                engine
                    .conjectures
                    .get(conj_idx)
                    .map(|c| format!("{:?}:{}", c.domain, c.source))
            })
            .collect();
        println!("      → {}", sources.join(", "));
    }

    println!("\n  ── Dynamic Grammar state ──");
    println!(
        "    Candidate subtrees tracked: {}",
        at.dynamic_grammar.candidates.len()
    );
    println!(
        "    Promoted macro-operators:   {}",
        at.dynamic_grammar.operators.len()
    );
    print_macro_pool_snapshot(&engine, "after first reflection");

    if !at.dynamic_grammar.candidates.is_empty() {
        println!("\n    Top 5 candidates (by occurrence count):");
        let mut cands: Vec<_> = at.dynamic_grammar.candidates.iter().collect();
        cands.sort_by(|a, b| b.occurrences.len().cmp(&a.occurrences.len()));
        for cand in cands.iter().take(5) {
            println!(
                "      · {} (appears in {} conjectures, {} sources, {} verified)",
                cand.canonical,
                cand.occurrences.len(),
                cand.sources.len(),
                cand.verified_count
            );
        }
    }

    if !at.dynamic_grammar.operators.is_empty() {
        println!("\n    Promoted macros:");
        for op in &at.dynamic_grammar.operators {
            println!(
                "      · {} (arity {}, from {} conjectures)",
                op.canonical,
                op.arity,
                op.source_conjectures.len()
            );
        }
    }

    println!("\n  ── Category Discovery state ──");
    println!(
        "    Domain morphisms found: {}",
        at.category_discovery.morphisms.len()
    );
    println!(
        "    Functorial structures:  {}",
        at.category_discovery.functors.len()
    );
    if let Some(cat) = &at.category_discovery.math_category {
        println!(
            "    Category objects:       {} (math domains)",
            cat.num_objects()
        );
    }

    if !at.category_discovery.functors.is_empty() {
        println!("\n    Discovered functors:");
        for f in &at.category_discovery.functors {
            println!(
                "      · {} (confidence={:.2}, paths={})",
                f.description, f.confidence, f.verified_paths
            );
        }
    }

    // ── 5. Cross-domain matches (the raw input to category discovery) ──
    section("Phase 5: Cross-domain formula matches");
    let matches = engine.discover_cross_domain_formulas(3.0);
    println!(
        "  Found {} cross-domain matches (mse_ratio < 3.0)",
        matches.len()
    );
    for (i, m) in matches.iter().take(10).enumerate() {
        println!(
            "    {}. {} ({:?}) → {} ({:?}): ratio={:.2}",
            i + 1,
            m.source_seq,
            m.source_domain,
            m.target_seq,
            m.target_domain,
            m.mse_ratio
        );
    }

    // Snapshot the abstract thought state before running round 2 (which needs &mut engine)
    let concept_count = at.meta_hdc.concepts.len();
    let cross_domain_clusters = at
        .meta_hdc
        .clusters
        .iter()
        .filter(|c| c.domain_diversity >= 2)
        .count();
    let candidate_count = at.dynamic_grammar.candidates.len();
    let operator_count = at.dynamic_grammar.operators.len();
    let functor_count = at.category_discovery.functors.len();
    let _ = at; // release immutable borrow

    // ── 6. Second run WITH macro injection ──
    section("Phase 6: Second GP run — macros now injected into population");
    let macros_before = engine.macro_operators().len();
    println!("  Macros available for injection: {}", macros_before);

    // Clear the conjectures but keep abstract_thought state (which holds macros)
    engine.conjectures.clear();

    let round2_start = std::time::Instant::now();
    engine.generate_conjectures(3);
    let round2_time = round2_start.elapsed();
    let round2_count = engine.conjectures.len();
    println!(
        "  Generated {} conjectures in {:.2}s",
        round2_count,
        round2_time.as_secs_f64()
    );
    print_macro_pool_snapshot(&engine, "after macro-injected round 2");

    // ── 7. Summary ──
    section("Summary");
    println!(
        "  Round 1: {} conjectures in {:.2}s",
        round1_count,
        round1_time.as_secs_f64()
    );
    println!(
        "  Round 2: {} conjectures in {:.2}s",
        round2_count,
        round2_time.as_secs_f64()
    );
    println!();
    println!("  Architecture performance on real sequences:");
    println!(
        "    · Conjectures encoded as concept vectors: {}",
        concept_count
    );
    println!(
        "    · Cross-domain clusters detected:         {}",
        cross_domain_clusters
    );
    println!(
        "    · Recurring subtree candidates:           {}",
        candidate_count
    );
    println!(
        "    · Promoted macro-operators:               {}",
        operator_count
    );
    println!(
        "    · Functorial structures detected:         {}",
        functor_count
    );
    println!();
    println!("  Honest assessment:");
    if operator_count == 0 {
        println!("    ⚠ No macros promoted — promotion thresholds may be too strict");
        println!("      for this input size, OR the sequences don't share enough structure.");
    } else {
        println!(
            "    ✓ Grammar promotion worked — {} macros available for future runs",
            operator_count
        );
    }
    if cross_domain_clusters > 0 {
        println!("    ✓ Cross-domain patterns detected — Meta-HDC found structural echoes");
    } else {
        println!("    ⚠ No cross-domain clusters — either sequences are too different,");
        println!("      or k-means split them before diversity could emerge.");
    }
    if functor_count == 0 {
        println!("    ⚠ No functorial structure — category_discovery needs composable paths");
        println!("      (A→B, B→C, A→C) and we may not have enough matches for that.");
    } else {
        println!("    ✓ Category discovery found functorial structure");
    }

    println!("\n  Macro pool quality gates:");
    let metrics = engine.macro_pool_metrics();
    let mut report = MacroQualityReport::new();
    report.push(
        "macros_promoted",
        operator_count > 0,
        format!("operators={}", operator_count),
    );
    for gate in
        evaluate_common_metrics(metrics.as_ref(), &MacroQualityThresholds::one_dimensional()).gates
    {
        report.gates.push(gate);
    }
    print_report(&report, "overall_abstract_thought_quality");
    maybe_enforce(&report);

    println!("\n  Next steps (based on actual observations above):");
    println!("    1. If no macros promoted: lower min_occurrences or min_sources");
    println!("    2. If no cross-domain clusters: tune k in k-means");
    println!("    3. If no functors: need more cross-domain sequences to generate triangles");
}
