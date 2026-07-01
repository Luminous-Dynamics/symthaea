// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Complete Causal Tower Benchmark
//!
//! Tests all phases of the Causal Understanding Tower on the Tübingen benchmark
//! Including the improved SmartTower with ANM and oracle selection

use symthaea::benchmarks::{
    AnmDiscovery,
    CamDiscovery,
    // Complete System
    CausalTower,
    // Advanced methods (Jan 2026)
    EnhancedReci,
    FinalBoss,
    // Phase 2: Classic Algorithms
    IgciDiscovery,
    // Phase 1: Improved Primitives
    ImprovedHdcCompression,
    ImprovedLtcDynamics,
    ImprovedPhiFlow,
    LingamDiscovery,
    NeuralCausalDiscovery,
    ReciDiscovery,
    SlopeDiscovery,
    TuebingenAdapter,
    UltimateEnsemble,
    discover_by_enhanced_reci,
    discover_by_final_boss,
    discover_by_neural,
    discover_by_oracle,
    // SmartTower (improved ensemble)
    discover_by_smart_tower,
    discover_by_tower,
    discover_by_ultimate_ensemble,
    // Previous methods for comparison
    discover_information_theoretic,
    discover_majority_voting,
};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║              CAUSAL UNDERSTANDING TOWER - COMPLETE BENCHMARK              ║");
    println!("║                        Approaching 100% Accuracy                          ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝\n");

    // Load Tübingen benchmark
    let tuebingen_path = "benchmarks/external/tuebingen";
    let adapter = match TuebingenAdapter::load(tuebingen_path) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("Failed to load dataset: {}", e);
            return;
        }
    };
    println!("Loaded {} cause-effect pairs\n", adapter.len());

    // =========================================================================
    // PHASE 1: IMPROVED PRIMITIVES
    // =========================================================================
    println!("┌──────────────────────────────────────────────────────────────────────────┐");
    println!("│ PHASE 1: IMPROVED PRIMITIVES                                            │");
    println!("└──────────────────────────────────────────────────────────────────────────┘\n");

    let hdc = ImprovedHdcCompression::new();
    let results_hdc = adapter.run(|x, y| hdc.discover(x, y).direction);
    println!("  [1/3] Improved HDC Compression (multi-scale)...");
    println!(
        "         Accuracy: {:.1}% ({}/{})",
        results_hdc.accuracy() * 100.0,
        results_hdc.correct,
        results_hdc.total
    );

    let ltc = ImprovedLtcDynamics::new();
    let results_ltc = adapter.run(|x, y| ltc.discover(x, y).direction);
    println!("  [2/3] Improved LTC Dynamics (ODE-style)...");
    println!(
        "         Accuracy: {:.1}% ({}/{})",
        results_ltc.accuracy() * 100.0,
        results_ltc.correct,
        results_ltc.total
    );

    let phi = ImprovedPhiFlow::new();
    let results_phi = adapter.run(|x, y| phi.discover(x, y).direction);
    println!("  [3/3] Improved Phi Flow (HSIC)...");
    println!(
        "         Accuracy: {:.1}% ({}/{})",
        results_phi.accuracy() * 100.0,
        results_phi.correct,
        results_phi.total
    );

    // =========================================================================
    // PHASE 2: CLASSIC ALGORITHMS
    // =========================================================================
    println!("\n┌──────────────────────────────────────────────────────────────────────────┐");
    println!("│ PHASE 2: CLASSIC ALGORITHMS                                             │");
    println!("└──────────────────────────────────────────────────────────────────────────┘\n");

    let igci = IgciDiscovery::new();
    let results_igci = adapter.run(|x, y| igci.discover(x, y).direction);
    println!("  [1/4] IGCI (Information Geometric)...");
    println!(
        "         Accuracy: {:.1}% ({}/{})",
        results_igci.accuracy() * 100.0,
        results_igci.correct,
        results_igci.total
    );

    let lingam = LingamDiscovery::new();
    let results_lingam = adapter.run(|x, y| lingam.discover(x, y).direction);
    println!("  [2/4] LiNGaM (Linear Non-Gaussian)...");
    println!(
        "         Accuracy: {:.1}% ({}/{})",
        results_lingam.accuracy() * 100.0,
        results_lingam.correct,
        results_lingam.total
    );

    let reci = ReciDiscovery::new();
    let results_reci = adapter.run(|x, y| reci.discover(x, y).direction);
    println!("  [3/4] RECI (Regression Error)...");
    println!(
        "         Accuracy: {:.1}% ({}/{})",
        results_reci.accuracy() * 100.0,
        results_reci.correct,
        results_reci.total
    );

    let cam = CamDiscovery::new();
    let results_cam = adapter.run(|x, y| cam.discover(x, y).direction);
    println!("  [4/4] CAM (Causal Additive Model)...");
    println!(
        "         Accuracy: {:.1}% ({}/{})",
        results_cam.accuracy() * 100.0,
        results_cam.correct,
        results_cam.total
    );

    // =========================================================================
    // NEW METHODS: ANM and Slope-based
    // =========================================================================
    println!("\n┌──────────────────────────────────────────────────────────────────────────┐");
    println!("│ NEW ALGORITHMS                                                           │");
    println!("└──────────────────────────────────────────────────────────────────────────┘\n");

    let anm = AnmDiscovery::new();
    let results_anm = adapter.run(|x, y| anm.discover(x, y).direction);
    println!("  [1/2] ANM (Additive Noise Model)...");
    println!(
        "         Accuracy: {:.1}% ({}/{})",
        results_anm.accuracy() * 100.0,
        results_anm.correct,
        results_anm.total
    );

    let slope = SlopeDiscovery::new();
    let results_slope = adapter.run(|x, y| slope.discover(x, y).direction);
    println!("  [2/2] Slope-based Discovery...");
    println!(
        "         Accuracy: {:.1}% ({}/{})",
        results_slope.accuracy() * 100.0,
        results_slope.correct,
        results_slope.total
    );

    // =========================================================================
    // PHASE 3-5: COMPLETE TOWER (Meta-Learning + Semantic + Uncertainty)
    // =========================================================================
    println!("\n┌──────────────────────────────────────────────────────────────────────────┐");
    println!("│ ENSEMBLE METHODS                                                         │");
    println!("└──────────────────────────────────────────────────────────────────────────┘\n");

    let results_tower = adapter.run(|x, y| discover_by_tower(x, y).direction);
    println!("  Causal Tower (weighted ensemble)...");
    println!(
        "         Accuracy: {:.1}% ({}/{})",
        results_tower.accuracy() * 100.0,
        results_tower.correct,
        results_tower.total
    );

    let results_smart = adapter.run(|x, y| discover_by_smart_tower(x, y).direction);
    println!("  SmartTower (confidence-weighted)...");
    println!(
        "         Accuracy: {:.1}% ({}/{})",
        results_smart.accuracy() * 100.0,
        results_smart.correct,
        results_smart.total
    );

    let results_oracle = adapter.run(|x, y| discover_by_oracle(x, y).direction);
    println!("  Oracle Selection (best per data type)...");
    println!(
        "         Accuracy: {:.1}% ({}/{})",
        results_oracle.accuracy() * 100.0,
        results_oracle.correct,
        results_oracle.total
    );

    // =========================================================================
    // ADVANCED METHODS (Jan 2026)
    // =========================================================================
    println!("\n┌──────────────────────────────────────────────────────────────────────────┐");
    println!("│ ADVANCED METHODS (JAN 2026)                                              │");
    println!("└──────────────────────────────────────────────────────────────────────────┘\n");

    let results_enhanced_reci = adapter.run(|x, y| discover_by_enhanced_reci(x, y).direction);
    println!("  Enhanced RECI (adaptive bandwidth)...");
    println!(
        "         Accuracy: {:.1}% ({}/{})",
        results_enhanced_reci.accuracy() * 100.0,
        results_enhanced_reci.correct,
        results_enhanced_reci.total
    );

    let results_ultimate = adapter.run(|x, y| discover_by_ultimate_ensemble(x, y).direction);
    println!("  Ultimate Ensemble (CV-optimized)...");
    println!(
        "         Accuracy: {:.1}% ({}/{})",
        results_ultimate.accuracy() * 100.0,
        results_ultimate.correct,
        results_ultimate.total
    );

    let results_neural = adapter.run(|x, y| discover_by_neural(x, y).direction);
    println!("  Neural Discovery (feature-based)...");
    println!(
        "         Accuracy: {:.1}% ({}/{})",
        results_neural.accuracy() * 100.0,
        results_neural.correct,
        results_neural.total
    );

    let results_final_boss = adapter.run(|x, y| discover_by_final_boss(x, y).direction);
    println!("  FinalBoss (all combined)...");
    println!(
        "         Accuracy: {:.1}% ({}/{})",
        results_final_boss.accuracy() * 100.0,
        results_final_boss.correct,
        results_final_boss.total
    );

    // =========================================================================
    // COMPARISON WITH PREVIOUS METHODS
    // =========================================================================
    println!("\n┌──────────────────────────────────────────────────────────────────────────┐");
    println!("│ COMPARISON WITH PREVIOUS METHODS                                        │");
    println!("└──────────────────────────────────────────────────────────────────────────┘\n");

    let results_info = adapter.run(discover_information_theoretic);
    println!("  Info-Theoretic (previous best primitive)...");
    println!(
        "         Accuracy: {:.1}% ({}/{})",
        results_info.accuracy() * 100.0,
        results_info.correct,
        results_info.total
    );

    let results_majority = adapter.run(discover_majority_voting);
    println!("  Majority Voting (previous best ensemble)...");
    println!(
        "         Accuracy: {:.1}% ({}/{})",
        results_majority.accuracy() * 100.0,
        results_majority.correct,
        results_majority.total
    );

    // =========================================================================
    // SUMMARY
    // =========================================================================
    println!("\n╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║                            FINAL SUMMARY                                 ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝\n");

    let methods = vec![
        ("Improved HDC", results_hdc.accuracy()),
        ("Improved LTC", results_ltc.accuracy()),
        ("Improved Phi (HSIC)", results_phi.accuracy()),
        ("IGCI", results_igci.accuracy()),
        ("LiNGaM", results_lingam.accuracy()),
        ("RECI", results_reci.accuracy()),
        ("CAM", results_cam.accuracy()),
        ("ANM", results_anm.accuracy()),
        ("Slope", results_slope.accuracy()),
        ("─────────────", 0.0), // Separator
        ("Causal Tower", results_tower.accuracy()),
        ("SmartTower", results_smart.accuracy()),
        ("Oracle Selection", results_oracle.accuracy()),
        ("Info-Theoretic", results_info.accuracy()),
        ("Majority Voting", results_majority.accuracy()),
        ("─────────────", 0.0), // Separator
        ("Enhanced RECI", results_enhanced_reci.accuracy()),
        ("Ultimate Ensemble", results_ultimate.accuracy()),
        ("Neural Discovery", results_neural.accuracy()),
        ("FinalBoss", results_final_boss.accuracy()),
    ];

    println!("  Method                    Accuracy    vs Random    vs 100%    Status");
    println!("  ────────────────────────────────────────────────────────────────────");

    for (name, acc) in &methods {
        if name.starts_with("───") {
            println!("  ────────────────────────────────────────────────────────────────────");
            continue;
        }

        let delta_random = (acc - 0.5) * 100.0;
        let gap_to_100 = (1.0 - acc) * 100.0;
        let marker = if *acc >= 0.80 {
            "★★★"
        } else if *acc >= 0.75 {
            "★★"
        } else if *acc >= 0.70 {
            "★"
        } else if *acc >= 0.60 {
            "✓"
        } else if *acc > 0.50 {
            "~"
        } else {
            "✗"
        };

        println!(
            "  {:22} {:5.1}%      {:+5.1}%      -{:4.1}%      {}",
            name,
            acc * 100.0,
            delta_random,
            gap_to_100,
            marker
        );
    }

    // Best method
    let best = methods
        .iter()
        .filter(|(n, _)| !n.starts_with("───"))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();

    println!("\n  ════════════════════════════════════════════════════════════════════");
    println!("  BEST METHOD: {} ({:.1}%)", best.0, best.1 * 100.0);
    println!("  GAP TO 100%: {:.1}%", (1.0 - best.1) * 100.0);

    // Tower vs previous best
    let tower_acc = results_tower.accuracy();
    let majority_acc = results_majority.accuracy();

    if tower_acc > majority_acc {
        println!(
            "\n  ✓ Causal Tower EXCEEDS Majority Voting by {:.1}%!",
            (tower_acc - majority_acc) * 100.0
        );
    } else if tower_acc == majority_acc {
        println!("\n  ~ Causal Tower matches Majority Voting");
    } else {
        println!(
            "\n  Gap: {:.1}% (Tower: {:.1}% vs Majority: {:.1}%)",
            (majority_acc - tower_acc) * 100.0,
            tower_acc * 100.0,
            majority_acc * 100.0
        );
    }

    // =========================================================================
    // ANALYZE HARD CASES
    // =========================================================================
    println!("\n╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║                       HARD CASE ANALYSIS                                 ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝\n");

    // Find pairs where Tower got it wrong
    let tower = CausalTower::new();
    let pairs = adapter.get_pairs();
    let mut wrong_cases = Vec::new();
    let mut right_cases = Vec::new();

    for (i, pair) in pairs.iter().enumerate() {
        let result = tower.discover(&pair.x, &pair.y);
        let predicted = result.direction;
        let actual = &pair.ground_truth;

        let is_correct = match (&predicted, actual) {
            (
                symthaea::benchmarks::CausalDirection::Forward,
                symthaea::benchmarks::CausalDirection::Forward,
            ) => true,
            (
                symthaea::benchmarks::CausalDirection::Backward,
                symthaea::benchmarks::CausalDirection::Backward,
            ) => true,
            _ => false,
        };

        if is_correct {
            right_cases.push((i, result.confidence));
        } else {
            wrong_cases.push((i, result.confidence, result.p_forward));
        }
    }

    println!("  Total pairs: {}", pairs.len());
    println!(
        "  Correct: {} ({:.1}%)",
        right_cases.len(),
        right_cases.len() as f64 / pairs.len() as f64 * 100.0
    );
    println!(
        "  Wrong: {} ({:.1}%)",
        wrong_cases.len(),
        wrong_cases.len() as f64 / pairs.len() as f64 * 100.0
    );

    if !wrong_cases.is_empty() {
        println!("\n  Most confident wrong predictions:");
        let mut sorted_wrong = wrong_cases.clone();
        sorted_wrong.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        for (idx, conf, p_fwd) in sorted_wrong.iter().take(5) {
            println!(
                "    Pair {}: confidence={:.2}, p_forward={:.2}",
                idx + 1,
                conf,
                p_fwd
            );
        }

        println!("\n  Low-confidence wrong predictions (potentially undetermined):");
        let mut low_conf: Vec<_> = wrong_cases.iter().filter(|(_, c, _)| *c < 0.3).collect();
        low_conf.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        for (idx, conf, p_fwd) in low_conf.iter().take(5) {
            println!(
                "    Pair {}: confidence={:.2}, p_forward={:.2}",
                idx + 1,
                conf,
                p_fwd
            );
        }
    }

    println!("\n  Done!");
}
