// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Guided Design Sweep: Physics-bridge-enhanced fusion reactor design exploration.
//!
//! Demonstrates `GuidedDesignExplorer` which runs parametric sweeps over
//! fusion reactor design parameters and annotates plateau regions with
//! physics analogies from the HDC equation catalog.
//!
//! ```bash
//! cargo run --features physics-bridge --example guided_design_sweep
//! ```

use symthaea::spark_physics_bridge::{GuidedDesignExplorer, SweepKind};
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::physics::FusionReaction;

fn main() {
    let genesis = GenesisSeed::from_phrase("spark_guided_design_demo");
    let mut explorer = GuidedDesignExplorer::from_genesis(&genesis);

    println!("═══════════════════════════════════════════════════════════");
    println!("  Guided Design Sweep: D-D Fusion (1–50 kW)");
    println!("═══════════════════════════════════════════════════════════\n");

    // ── Sweep: Power vs Shielding (default) ──
    let result = explorer.guided_sweep(FusionReaction::DD, (1.0, 50.0), 30);

    println!("{}", explorer.ascii_chart(&result.sweep, 60, 15));
    println!();

    println!(
        "Sweep (shielding): {} points, {} feasible, {} infeasible",
        result.sweep.points.len(),
        result.sweep.points.iter().filter(|p| p.feasible).count(),
        result.sweep.points.iter().filter(|p| !p.feasible).count(),
    );
    println!();

    if result.plateau_regions.is_empty() {
        println!("No plateau regions detected.");
    } else {
        println!("Plateau regions ({}):", result.plateau_regions.len());
        for (i, plateau) in result.plateau_regions.iter().enumerate() {
            println!(
                "  [{i}] idx {}-{}: {} (avg metric = {:.3})",
                plateau.start_idx,
                plateau.end_idx,
                if plateau.feasible {
                    "FEASIBLE"
                } else {
                    "infeasible"
                },
                plateau.avg_metric,
            );
        }
        println!();
    }

    if result.analogies.is_empty() {
        println!("No physics analogies found.");
    } else {
        println!("Physics analogies ({}):", result.analogies.len());
        for analogy in &result.analogies {
            println!(
                "  Point [{}]: {} ({:?}) — similarity {:.1}%",
                analogy.design_point_idx,
                analogy.equation_name,
                analogy.domain,
                analogy.similarity * 100.0,
            );
            println!("    {}", analogy.insight);
        }
        println!();
    }

    // ── Sweep: Power vs Mass ──
    println!("───────────────────────────────────────────────────────────");
    println!("  Sweep: Power vs Mass\n");
    let mass_result = explorer.guided_sweep_with_kind(
        SweepKind::PowerVsMass,
        FusionReaction::DD,
        (1.0, 50.0),
        20,
    );
    println!(
        "  {} points, {} feasible",
        mass_result.sweep.points.len(),
        mass_result
            .sweep
            .points
            .iter()
            .filter(|p| p.feasible)
            .count(),
    );
    println!();

    // ── Sweep: Power vs Cost ──
    println!("───────────────────────────────────────────────────────────");
    println!("  Sweep: Power vs Cost\n");
    let cost_result = explorer.guided_sweep_with_kind(
        SweepKind::PowerVsCost,
        FusionReaction::DD,
        (1.0, 50.0),
        20,
    );
    println!(
        "  {} points, {} feasible",
        cost_result.sweep.points.len(),
        cost_result
            .sweep
            .points
            .iter()
            .filter(|p| p.feasible)
            .count(),
    );
    println!();

    // ── Pareto Frontier ──
    println!("═══════════════════════════════════════════════════════════");
    println!("  Pareto Frontier: D-D Fusion (1–50 kW)");
    println!("═══════════════════════════════════════════════════════════\n");

    let pareto_result = explorer.guided_pareto(FusionReaction::DD, (1.0, 50.0), 30);
    println!(
        "Evaluated {} designs, {} on Pareto frontier",
        pareto_result.frontier.all_designs.len(),
        pareto_result.frontier.frontier.len(),
    );

    let report = explorer.format_pareto_report(&pareto_result);
    println!("{report}");

    // ── Refinement Suggestions ──
    println!("───────────────────────────────────────────────────────────");
    let suggestions = explorer.suggest_refinement(&result.sweep);
    if suggestions.is_empty() {
        println!("No refinement suggestions (no feasibility boundaries found).");
    } else {
        println!("Refinement suggestions ({}):", suggestions.len());
        for s in &suggestions {
            println!(
                "  Point [{}]: {} (sim {:.1}%)",
                s.point_idx,
                s.equation_name,
                s.physics_similarity * 100.0,
            );
            println!("    {}", s.reason);
        }
        println!();
    }

    // ── Multi-Sweep with Cross-Correlation ──
    println!("═══════════════════════════════════════════════════════════");
    println!("  Multi-Sweep: D-D Fusion (1–50 kW)");
    println!("═══════════════════════════════════════════════════════════\n");

    let multi = explorer.multi_sweep(FusionReaction::DD, (1.0, 50.0), 20);
    println!(
        "Shielding: {} pts, {} plateaus",
        multi.shielding.sweep.points.len(),
        multi.shielding.plateau_regions.len(),
    );
    println!(
        "Mass:      {} pts, {} plateaus",
        multi.mass.sweep.points.len(),
        multi.mass.plateau_regions.len(),
    );
    println!(
        "Cost:      {} pts, {} plateaus",
        multi.cost.sweep.points.len(),
        multi.cost.plateau_regions.len(),
    );
    println!("Cross-correlated plateaus: {}", multi.cross_plateaus.len());
    for (i, cp) in multi.cross_plateaus.iter().enumerate() {
        println!(
            "  [{i}] {:?} @ {:.1}–{:.1} kW (all_feasible={})",
            cp.sweep_kinds, cp.power_range.0, cp.power_range.1, cp.all_feasible,
        );
    }
    println!();

    // Minimum feasible power
    if let Some(min_kw) = explorer.find_minimum_feasible_power(FusionReaction::DD) {
        println!("Minimum feasible power (D-D): {min_kw:.2} kW");
    } else {
        println!("No feasible power found for D-D reaction.");
    }

    // ── Pareto Context Injection Demo ──
    println!("\n═══════════════════════════════════════════════════════════");
    println!("  Pareto Context → Cognitive Loop");
    println!("═══════════════════════════════════════════════════════════\n");

    // Build Pareto context from explorer results
    let pareto_result = explorer.guided_pareto(FusionReaction::DD, (1.0, 50.0), 20);
    let pareto_ctx = symthaea::cognitive_loop::ParetoContext {
        frontier_size: pareto_result.frontier.frontier.len(),
        power_range: (1.0, 50.0),
        dominant_domain: pareto_result
            .frontier_analogies
            .first()
            .map(|a| a.equation_name.clone())
            .unwrap_or_default(),
        best_analogy_score: pareto_result
            .frontier_analogies
            .first()
            .map(|a| a.similarity)
            .unwrap_or(0.0),
    };
    println!(
        "Pareto frontier: {} points, best analogy: {} ({:.1}%)",
        pareto_ctx.frontier_size,
        pareto_ctx.dominant_domain,
        pareto_ctx.best_analogy_score * 100.0,
    );

    // Inject into cognitive loop
    let mut config = symthaea::cognitive_loop::CognitiveLoopConfig::default();
    config.enable_physics_bridge = true;
    config.physics_bridge_query_interval = 5;
    let mut service = symthaea::cognitive_loop::CognitiveLoopService::new(config).unwrap();

    service.set_physics_pareto_context(pareto_ctx);

    let result = service.cycle("fusion reactor design optimization");
    if let Some(ref pb) = result.metadata.physics_bridge {
        println!(
            "Physics bridge: catalog={}, queried={}, top_match={}",
            pb.catalog_size, pb.queried_this_cycle, pb.top_match,
        );
    }
    println!("\nDone.");
}
