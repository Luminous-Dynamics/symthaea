// ==================================================================================
// Tübingen Cause-Effect Pairs Evaluation
// ==================================================================================
//
// Tests Symthaea's causal discovery on real-world data from diverse domains.
//
// **Dataset**: 108 cause-effect pairs from meteorology, biology, economics, etc.
// **Task**: Determine causal direction (X→Y or Y→X) from observational data
//
// **Methods**:
//   Linear:
//   - ANM (Additive Noise Model): Fits residuals, checks independence
//   - HSIC: Kernel-based independence test
//   - Combined: Voting ensemble
//
//   Nonlinear (improved):
//   - Nonlinear ANM: Kernel regression instead of linear
//   - IGCI: Information-Geometric Causal Inference
//   - RECI: Regression Error Causal Inference
//   - Combined Nonlinear: Ensemble of nonlinear methods
//
// Usage: cargo run --example tuebingen_evaluation
//
// ==================================================================================

use symthaea::benchmarks::{
    TuebingenAdapter,
    CausalDirection,
    discover_by_anm, discover_by_hsic, discover_combined,
    discover_by_nonlinear_anm, discover_by_igci, discover_by_reci,
    discover_combined_nonlinear,
    // Learned methods (self-contained)
    discover_by_learned,
    discover_enhanced_learned,
    create_learned_discoverer,
    // HDC methods (Symthaea's unique approach)
    discover_by_hdc,
    discover_hdc_ensemble,
    discover_ultimate_ensemble,
    // Advanced HDC (4-part improvement system)
    AdvancedHdcCausalDiscovery,
    discover_advanced_hdc,
    discover_sota_ensemble,
    // Information-theoretic methods (Phi-inspired)
    discover_by_conditional_entropy,
    discover_information_theoretic,
    // Smart ensemble (based on diagnostic analysis)
    discover_smart_ensemble,
    discover_majority_voting,
    // Principled Symthaea primitives (Dec 2025)
    discover_by_principled_hdc,
    discover_by_phi,
    discover_by_unified_primitives,
};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║       Tübingen Cause-Effect Pairs Evaluation                 ║");
    println!("║                                                              ║");
    println!("║   Real-world causal discovery benchmark (108 pairs)          ║");
    println!("║   Domains: meteorology, biology, economics, physics          ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Load dataset
    let tuebingen_path = "benchmarks/external/tuebingen";

    println!("Loading Tübingen dataset from {}...", tuebingen_path);

    let adapter = match TuebingenAdapter::load(tuebingen_path) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("Failed to load Tübingen dataset: {}", e);
            eprintln!("\nMake sure to download the dataset first:");
            eprintln!("  mkdir -p benchmarks/external/tuebingen");
            eprintln!("  cd benchmarks/external/tuebingen");
            eprintln!("  curl -LO https://webdav.tuebingen.mpg.de/cause-effect/pairs.zip");
            eprintln!("  curl -LO https://webdav.tuebingen.mpg.de/cause-effect/pairmeta.txt");
            eprintln!("  unzip pairs.zip");
            return;
        }
    };

    println!("Loaded {} cause-effect pairs\n", adapter.len());

    // ==========================================
    // LINEAR METHODS (Baseline)
    // ==========================================
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║              LINEAR METHODS (Baseline)                       ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Method 1: ANM (Additive Noise Model)
    println!("Testing Linear ANM...");
    let results_anm = adapter.run(discover_by_anm);
    println!("  Accuracy: {:.1}% ({}/{})",
             results_anm.accuracy() * 100.0,
             results_anm.correct,
             results_anm.total);

    // Method 2: HSIC
    println!("Testing HSIC...");
    let results_hsic = adapter.run(discover_by_hsic);
    println!("  Accuracy: {:.1}% ({}/{})",
             results_hsic.accuracy() * 100.0,
             results_hsic.correct,
             results_hsic.total);

    // Method 3: Combined Linear
    println!("Testing Combined Linear...");
    let results_combined = adapter.run(discover_combined);
    println!("  Accuracy: {:.1}% ({}/{})\n",
             results_combined.accuracy() * 100.0,
             results_combined.correct,
             results_combined.total);

    // ==========================================
    // NONLINEAR METHODS (Improved)
    // ==========================================
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║              NONLINEAR METHODS (Improved)                    ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Method 4: Nonlinear ANM
    println!("Testing Nonlinear ANM (Kernel Regression)...");
    let results_nonlinear_anm = adapter.run(discover_by_nonlinear_anm);
    println!("  Accuracy: {:.1}% ({}/{})",
             results_nonlinear_anm.accuracy() * 100.0,
             results_nonlinear_anm.correct,
             results_nonlinear_anm.total);

    // Method 5: IGCI
    println!("Testing IGCI (Information-Geometric)...");
    let results_igci = adapter.run(discover_by_igci);
    println!("  Accuracy: {:.1}% ({}/{})",
             results_igci.accuracy() * 100.0,
             results_igci.correct,
             results_igci.total);

    // Method 6: RECI
    println!("Testing RECI (Regression Error)...");
    let results_reci = adapter.run(discover_by_reci);
    println!("  Accuracy: {:.1}% ({}/{})",
             results_reci.accuracy() * 100.0,
             results_reci.correct,
             results_reci.total);

    // Method 7: Combined Nonlinear
    println!("Testing Combined Nonlinear...");
    let results_combined_nl = adapter.run(discover_combined_nonlinear);
    println!("  Accuracy: {:.1}% ({}/{})\n",
             results_combined_nl.accuracy() * 100.0,
             results_combined_nl.correct,
             results_combined_nl.total);

    // ==========================================
    // LEARNED METHODS
    // ==========================================
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║              LEARNED METHODS                                 ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Method 8: Learned model with default weights
    println!("Testing Learned Model (default weights)...");
    let results_learned = adapter.run(discover_by_learned);
    println!("  Accuracy: {:.1}% ({}/{})",
             results_learned.accuracy() * 100.0,
             results_learned.correct,
             results_learned.total);

    // Method 9: Ensemble with trained model
    println!("Testing Learned + Ensemble...");
    let trained_model = create_learned_discoverer();
    let results_ensemble = adapter.run(|x, y| {
        discover_enhanced_learned(x, y, &trained_model)
    });
    println!("  Accuracy: {:.1}% ({}/{})\n",
             results_ensemble.accuracy() * 100.0,
             results_ensemble.correct,
             results_ensemble.total);

    // ==========================================
    // INFORMATION-THEORETIC METHODS (Phi-Inspired)
    // ==========================================
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║    INFORMATION-THEORETIC METHODS (Phi-Inspired)            ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Method: Conditional Entropy Asymmetry
    println!("Testing Conditional Entropy Asymmetry...");
    let results_cea = adapter.run(discover_by_conditional_entropy);
    println!("  Accuracy: {:.1}% ({}/{})",
             results_cea.accuracy() * 100.0,
             results_cea.correct,
             results_cea.total);

    // Method: Combined Information-Theoretic (IGCI + CEA + RECI)
    println!("Testing Combined Info-Theoretic (IGCI + CEA + Residuals)...");
    let results_info_theory = adapter.run(discover_information_theoretic);
    println!("  Accuracy: {:.1}% ({}/{})\n",
             results_info_theory.accuracy() * 100.0,
             results_info_theory.correct,
             results_info_theory.total);

    // ==========================================
    // HDC METHODS (Symthaea's Unique Approach)
    // ==========================================
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║      HDC METHODS (Symthaea's Unique Approach)               ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Method 10: Pure HDC causal discovery
    println!("Testing HDC Causal Discovery (4D Cantor + CausalRoleMarkers)...");
    let results_hdc = adapter.run(discover_by_hdc);
    println!("  Accuracy: {:.1}% ({}/{})",
             results_hdc.accuracy() * 100.0,
             results_hdc.correct,
             results_hdc.total);

    // Method 11: HDC + Learned ensemble
    println!("Testing HDC + Learned Ensemble...");
    let results_hdc_learned = adapter.run(discover_hdc_ensemble);
    println!("  Accuracy: {:.1}% ({}/{})",
             results_hdc_learned.accuracy() * 100.0,
             results_hdc_learned.correct,
             results_hdc_learned.total);

    // Method 12: Ultimate ensemble (all methods)
    println!("Testing Ultimate Ensemble (HDC + Learned + RECI + ANM + IGCI)...");
    let results_ultimate = adapter.run(discover_ultimate_ensemble);
    println!("  Accuracy: {:.1}% ({}/{})\n",
             results_ultimate.accuracy() * 100.0,
             results_ultimate.correct,
             results_ultimate.total);

    // ==========================================
    // ADVANCED HDC (Four-Part Improvement System)
    // ==========================================
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   ADVANCED HDC (Trainable + LTC + CGNN + Domain Priors)     ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Method 13: Advanced HDC (untrained)
    println!("Testing Advanced HDC (4-part system, untrained)...");
    let results_advanced_untrained = adapter.run(discover_advanced_hdc);
    println!("  Accuracy: {:.1}% ({}/{})",
             results_advanced_untrained.accuracy() * 100.0,
             results_advanced_untrained.correct,
             results_advanced_untrained.total);

    // Method 14: Advanced HDC with 3-fold cross-validation training
    println!("Testing Advanced HDC with 3-fold CV training...");
    let pairs = adapter.get_pairs();
    let n_pairs = pairs.len();
    let fold_size = n_pairs / 3;

    let mut total_correct = 0;
    let mut total_tested = 0;

    for fold in 0..3 {
        // Split into train/test
        let test_start = fold * fold_size;
        let test_end = if fold == 2 { n_pairs } else { (fold + 1) * fold_size };

        let mut train_data: Vec<(&[f64], &[f64], CausalDirection)> = Vec::new();
        let mut test_data: Vec<(&[f64], &[f64], CausalDirection)> = Vec::new();

        for (i, pair) in pairs.iter().enumerate() {
            let dir = pair.ground_truth.clone();  // Use actual ground truth from dataset
            if i >= test_start && i < test_end {
                test_data.push((&pair.x, &pair.y, dir));
            } else {
                train_data.push((&pair.x, &pair.y, dir));
            }
        }

        // Train model
        let mut model = AdvancedHdcCausalDiscovery::new();
        for _epoch in 0..3 {
            for &(x, y, dir) in &train_data {
                model.train(x, y, dir);
            }
        }

        // Test
        for &(x, y, true_dir) in &test_data {
            let result = model.discover(x, y);
            if result.direction == true_dir {
                total_correct += 1;
            }
            total_tested += 1;
        }
    }

    let trained_accuracy = total_correct as f64 / total_tested.max(1) as f64;
    println!("  Accuracy: {:.1}% ({}/{})",
             trained_accuracy * 100.0,
             total_correct,
             total_tested);

    // Method 15: SOTA Ensemble (Advanced HDC + All methods)
    println!("Testing SOTA Ensemble (Advanced HDC + Learned + RECI + ANM)...");
    let results_sota = adapter.run(discover_sota_ensemble);
    println!("  Accuracy: {:.1}% ({}/{})\n",
             results_sota.accuracy() * 100.0,
             results_sota.correct,
             results_sota.total);

    // ==========================================
    // SMART ENSEMBLE (Based on Deep Analysis)
    // ==========================================
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   SMART ENSEMBLE (Based on Diagnostic Analysis)             ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Majority Voting baseline
    println!("Testing Majority Voting (7 methods)...");
    let results_majority = adapter.run(discover_majority_voting);
    println!("  Accuracy: {:.1}% ({}/{})",
             results_majority.accuracy() * 100.0,
             results_majority.correct,
             results_majority.total);

    // Smart Ensemble with complementary weighting
    println!("Testing Smart Ensemble (complementary weighting)...");
    let results_smart = adapter.run(discover_smart_ensemble);
    println!("  Accuracy: {:.1}% ({}/{})\n",
             results_smart.accuracy() * 100.0,
             results_smart.correct,
             results_smart.total);

    // ==========================================
    // PRINCIPLED SYMTHAEA PRIMITIVES
    // ==========================================
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   PRINCIPLED SYMTHAEA PRIMITIVES (True Domain-Agnostic)     ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Principled HDC: Functional complexity via bundle entropy
    println!("Testing Principled HDC (Functional Complexity + Independence)...");
    let results_principled_hdc = adapter.run(discover_by_principled_hdc);
    println!("  Accuracy: {:.1}% ({}/{})",
             results_principled_hdc.accuracy() * 100.0,
             results_principled_hdc.correct,
             results_principled_hdc.total);

    // Phi-based: Effective Information (true IIT-inspired)
    println!("Testing Phi-based (Effective Information + Determinism)...");
    let results_phi = adapter.run(discover_by_phi);
    println!("  Accuracy: {:.1}% ({}/{})",
             results_phi.accuracy() * 100.0,
             results_phi.correct,
             results_phi.total);

    // Unified primitives ensemble
    println!("Testing Unified Symthaea Primitives (HDC + Phi + Info-Theoretic)...");
    let results_unified = adapter.run(discover_by_unified_primitives);
    println!("  Accuracy: {:.1}% ({}/{})\n",
             results_unified.accuracy() * 100.0,
             results_unified.correct,
             results_unified.total);

    // ==========================================
    // SUMMARY
    // ==========================================
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                       SUMMARY                                ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Find best methods
    let methods = vec![
        ("Linear ANM", results_anm.accuracy()),
        ("HSIC", results_hsic.accuracy()),
        ("Combined Linear", results_combined.accuracy()),
        ("Nonlinear ANM", results_nonlinear_anm.accuracy()),
        ("IGCI (fixed)", results_igci.accuracy()),
        ("RECI", results_reci.accuracy()),
        ("Combined Nonlinear", results_combined_nl.accuracy()),
        ("Learned (default)", results_learned.accuracy()),
        ("Learned+Ensemble", results_ensemble.accuracy()),
        ("CEA (Phi-inspired)", results_cea.accuracy()),
        ("Info-Theoretic", results_info_theory.accuracy()),
        ("HDC (4D Cantor)", results_hdc.accuracy()),
        ("HDC+Learned", results_hdc_learned.accuracy()),
        ("Ultimate Ensemble", results_ultimate.accuracy()),
        ("Majority Voting", results_majority.accuracy()),
        ("Smart Ensemble", results_smart.accuracy()),
        ("Advanced HDC", results_advanced_untrained.accuracy()),
        ("Adv HDC Trained", trained_accuracy),
        ("SOTA Ensemble", results_sota.accuracy()),
        // New principled primitives
        ("Principled HDC", results_principled_hdc.accuracy()),
        ("Phi-based (EI)", results_phi.accuracy()),
        ("Unified Primitives", results_unified.accuracy()),
    ];

    println!("  Method                   Accuracy    vs Random");
    println!("  ─────────────────────────────────────────────────");

    for (name, acc) in &methods {
        let delta = (acc - 0.5) * 100.0;
        let marker = if *acc > 0.6 { "✓" } else if *acc > 0.5 { "~" } else { "✗" };
        println!("  {:22} {:5.1}%      {:+5.1}%  {}", name, acc * 100.0, delta, marker);
    }

    let best = methods.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
    let best_linear = methods.iter().take(3).max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
    let best_nonlinear = methods.iter().skip(3).max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();

    println!();
    println!("  Best Linear:     {} ({:.1}%)", best_linear.0, best_linear.1 * 100.0);
    println!("  Best Nonlinear:  {} ({:.1}%)", best_nonlinear.0, best_nonlinear.1 * 100.0);
    println!("  Overall Best:    {} ({:.1}%)", best.0, best.1 * 100.0);

    let improvement = (best_nonlinear.1 - best_linear.1) * 100.0;
    println!("\n  Improvement from Nonlinear: {:+.1}%", improvement);

    // State of the art comparison
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║              STATE-OF-THE-ART COMPARISON                     ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    println!("  Published SOTA methods on Tübingen:");
    println!("    Random Baseline:      50.0%");
    println!("    ANM (original):       ~65-70%");
    println!("    IGCI (published):     ~70%");
    println!("    CGNN (neural):        ~80%");
    println!("    Deep learning:        ~85%");
    println!();
    println!("  Symthaea Best ({:}): {:.1}%", best.0, best.1 * 100.0);

    let sota_gap = 70.0 - best.1 * 100.0;
    if sota_gap <= 0.0 {
        println!("\n  ✅ Matches or exceeds published IGCI!");
    } else if sota_gap <= 10.0 {
        println!("\n  ⚠️  Close to published methods ({:.1}% gap)", sota_gap);
    } else {
        println!("\n  📈 Room for improvement ({:.1}% below SOTA)", sota_gap);
    }

    println!("\n  Done!");
}
