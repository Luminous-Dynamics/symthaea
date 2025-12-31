// ==================================================================================
// Tübingen Diagnostic Analysis
// ==================================================================================
//
// Deep analysis of causal discovery performance to identify:
// 1. Which pairs are consistently misclassified
// 2. What data characteristics correlate with errors
// 3. Where methods agree/disagree
// 4. Algorithmic gaps vs tuning issues
//
// ==================================================================================

use symthaea::benchmarks::{
    TuebingenAdapter,
    CausalDirection,
    CauseEffectPair,
    discover_by_anm, discover_by_hsic,
    discover_by_nonlinear_anm, discover_by_igci, discover_by_reci,
    discover_by_learned,
    discover_by_hdc,
    discover_by_conditional_entropy,
    discover_information_theoretic,
    discover_by_information_theoretic,
};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║         Tübingen Diagnostic Analysis                        ║");
    println!("║                                                              ║");
    println!("║   Deep dive into misclassifications and method behavior     ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let tuebingen_path = "benchmarks/external/tuebingen";
    let adapter = match TuebingenAdapter::load(tuebingen_path) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("Failed to load: {}", e);
            return;
        }
    };

    let pairs = adapter.get_pairs();
    println!("Analyzing {} pairs...\n", pairs.len());

    // ==========================================
    // PHASE 1: Per-pair method agreement analysis
    // ==========================================
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║              PHASE 1: Method Agreement Analysis             ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    #[derive(Debug, Clone)]
    struct PairAnalysis {
        pair_id: usize,
        n_samples: usize,
        ground_truth: CausalDirection,
        // Method predictions
        anm: CausalDirection,
        igci: CausalDirection,
        reci: CausalDirection,
        learned: CausalDirection,
        cea: CausalDirection,
        info_theory: CausalDirection,
        hdc: CausalDirection,
        // Data characteristics
        correlation: f64,
        x_std: f64,
        y_std: f64,
        nonlinearity: f64,
    }

    let mut analyses: Vec<PairAnalysis> = Vec::new();

    for (i, pair) in pairs.iter().enumerate() {
        let analysis = PairAnalysis {
            pair_id: i + 1,
            n_samples: pair.x.len(),
            ground_truth: pair.ground_truth.clone(),
            anm: discover_by_anm(&pair.x, &pair.y),
            igci: discover_by_igci(&pair.x, &pair.y),
            reci: discover_by_reci(&pair.x, &pair.y),
            learned: discover_by_learned(&pair.x, &pair.y),
            cea: discover_by_conditional_entropy(&pair.x, &pair.y),
            info_theory: discover_information_theoretic(&pair.x, &pair.y),
            hdc: discover_by_hdc(&pair.x, &pair.y),
            correlation: compute_correlation(&pair.x, &pair.y),
            x_std: std_dev(&pair.x),
            y_std: std_dev(&pair.y),
            nonlinearity: estimate_nonlinearity(&pair.x, &pair.y),
        };
        analyses.push(analysis);
    }

    // Count method correctness
    let mut method_correct = vec![0usize; 7];
    let method_names = ["ANM", "IGCI", "RECI", "Learned", "CEA", "InfoTheory", "HDC"];

    for a in &analyses {
        if a.anm == a.ground_truth { method_correct[0] += 1; }
        if a.igci == a.ground_truth { method_correct[1] += 1; }
        if a.reci == a.ground_truth { method_correct[2] += 1; }
        if a.learned == a.ground_truth { method_correct[3] += 1; }
        if a.cea == a.ground_truth { method_correct[4] += 1; }
        if a.info_theory == a.ground_truth { method_correct[5] += 1; }
        if a.hdc == a.ground_truth { method_correct[6] += 1; }
    }

    println!("Method Accuracies:");
    for (i, name) in method_names.iter().enumerate() {
        println!("  {:12} {:3}/108 ({:.1}%)", name, method_correct[i],
                 method_correct[i] as f64 / 108.0 * 100.0);
    }

    // ==========================================
    // PHASE 2: Find hard pairs (all methods wrong)
    // ==========================================
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║              PHASE 2: Hard Pairs (Most Methods Wrong)       ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let mut hard_pairs: Vec<&PairAnalysis> = Vec::new();
    let mut easy_pairs: Vec<&PairAnalysis> = Vec::new();

    for a in &analyses {
        let correct_count = [
            a.anm == a.ground_truth,
            a.igci == a.ground_truth,
            a.reci == a.ground_truth,
            a.learned == a.ground_truth,
            a.cea == a.ground_truth,
            a.info_theory == a.ground_truth,
            a.hdc == a.ground_truth,
        ].iter().filter(|&&x| x).count();

        if correct_count <= 2 {
            hard_pairs.push(a);
        } else if correct_count >= 5 {
            easy_pairs.push(a);
        }
    }

    println!("Hard pairs (≤2 methods correct): {} pairs", hard_pairs.len());
    println!("Easy pairs (≥5 methods correct): {} pairs\n", easy_pairs.len());

    // Analyze characteristics of hard vs easy pairs
    let hard_avg_samples: f64 = hard_pairs.iter().map(|a| a.n_samples as f64).sum::<f64>() / hard_pairs.len().max(1) as f64;
    let easy_avg_samples: f64 = easy_pairs.iter().map(|a| a.n_samples as f64).sum::<f64>() / easy_pairs.len().max(1) as f64;

    let hard_avg_corr: f64 = hard_pairs.iter().map(|a| a.correlation.abs()).sum::<f64>() / hard_pairs.len().max(1) as f64;
    let easy_avg_corr: f64 = easy_pairs.iter().map(|a| a.correlation.abs()).sum::<f64>() / easy_pairs.len().max(1) as f64;

    let hard_avg_nonlin: f64 = hard_pairs.iter().map(|a| a.nonlinearity).sum::<f64>() / hard_pairs.len().max(1) as f64;
    let easy_avg_nonlin: f64 = easy_pairs.iter().map(|a| a.nonlinearity).sum::<f64>() / easy_pairs.len().max(1) as f64;

    println!("Characteristic comparison (Hard vs Easy):");
    println!("  Avg samples:     {:.0} vs {:.0}", hard_avg_samples, easy_avg_samples);
    println!("  Avg |correlation|: {:.3} vs {:.3}", hard_avg_corr, easy_avg_corr);
    println!("  Avg nonlinearity:  {:.3} vs {:.3}", hard_avg_nonlin, easy_avg_nonlin);

    // ==========================================
    // PHASE 3: Method disagreement analysis
    // ==========================================
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║              PHASE 3: Method Disagreement Patterns          ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // When InfoTheory is wrong, what do other methods say?
    let mut info_wrong_count = 0;
    let mut when_info_wrong_others_right = vec![0usize; 6];

    for a in &analyses {
        if a.info_theory != a.ground_truth {
            info_wrong_count += 1;
            if a.anm == a.ground_truth { when_info_wrong_others_right[0] += 1; }
            if a.igci == a.ground_truth { when_info_wrong_others_right[1] += 1; }
            if a.reci == a.ground_truth { when_info_wrong_others_right[2] += 1; }
            if a.learned == a.ground_truth { when_info_wrong_others_right[3] += 1; }
            if a.cea == a.ground_truth { when_info_wrong_others_right[4] += 1; }
            if a.hdc == a.ground_truth { when_info_wrong_others_right[5] += 1; }
        }
    }

    println!("When Info-Theoretic is WRONG ({} pairs):", info_wrong_count);
    let other_names = ["ANM", "IGCI", "RECI", "Learned", "CEA", "HDC"];
    for (i, name) in other_names.iter().enumerate() {
        println!("  {:12} correct: {:2} ({:.1}%)", name, when_info_wrong_others_right[i],
                 when_info_wrong_others_right[i] as f64 / info_wrong_count.max(1) as f64 * 100.0);
    }

    // When CEA and InfoTheory disagree, who is right?
    let mut disagree_count = 0;
    let mut cea_right_when_disagree = 0;
    let mut info_right_when_disagree = 0;

    for a in &analyses {
        if a.cea != a.info_theory {
            disagree_count += 1;
            if a.cea == a.ground_truth { cea_right_when_disagree += 1; }
            if a.info_theory == a.ground_truth { info_right_when_disagree += 1; }
        }
    }

    println!("\nWhen CEA and InfoTheory disagree ({} pairs):", disagree_count);
    println!("  CEA correct:        {} ({:.1}%)", cea_right_when_disagree,
             cea_right_when_disagree as f64 / disagree_count.max(1) as f64 * 100.0);
    println!("  InfoTheory correct: {} ({:.1}%)", info_right_when_disagree,
             info_right_when_disagree as f64 / disagree_count.max(1) as f64 * 100.0);

    // ==========================================
    // PHASE 4: Optimal ensemble analysis
    // ==========================================
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║              PHASE 4: Oracle Ensemble Analysis              ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // What's the best possible accuracy with majority voting?
    let mut majority_correct = 0;
    for a in &analyses {
        let votes_forward = [
            a.anm == CausalDirection::Forward,
            a.igci == CausalDirection::Forward,
            a.reci == CausalDirection::Forward,
            a.learned == CausalDirection::Forward,
            a.cea == CausalDirection::Forward,
            a.info_theory == CausalDirection::Forward,
            a.hdc == CausalDirection::Forward,
        ].iter().filter(|&&x| x).count();

        let majority = if votes_forward > 3 {
            CausalDirection::Forward
        } else {
            CausalDirection::Backward
        };

        if majority == a.ground_truth {
            majority_correct += 1;
        }
    }

    println!("Simple Majority Voting (7 methods): {}/108 ({:.1}%)",
             majority_correct, majority_correct as f64 / 108.0 * 100.0);

    // Oracle: pick the best method for each pair
    let mut oracle_correct = 0;
    for a in &analyses {
        let any_correct = a.anm == a.ground_truth
            || a.igci == a.ground_truth
            || a.reci == a.ground_truth
            || a.learned == a.ground_truth
            || a.cea == a.ground_truth
            || a.info_theory == a.ground_truth
            || a.hdc == a.ground_truth;
        if any_correct {
            oracle_correct += 1;
        }
    }

    println!("Oracle (any method correct):       {}/108 ({:.1}%)",
             oracle_correct, oracle_correct as f64 / 108.0 * 100.0);

    // Weighted ensemble: give more weight to better methods
    let mut weighted_correct = 0;
    for a in &analyses {
        // Weights based on individual accuracy
        let score =
            (if a.anm == CausalDirection::Forward { 1.0 } else { -1.0 }) * 0.47 +
            (if a.igci == CausalDirection::Forward { 1.0 } else { -1.0 }) * 0.54 +
            (if a.reci == CausalDirection::Forward { 1.0 } else { -1.0 }) * 0.56 +
            (if a.learned == CausalDirection::Forward { 1.0 } else { -1.0 }) * 0.59 +
            (if a.cea == CausalDirection::Forward { 1.0 } else { -1.0 }) * 0.62 +
            (if a.info_theory == CausalDirection::Forward { 1.0 } else { -1.0 }) * 0.68 +
            (if a.hdc == CausalDirection::Forward { 1.0 } else { -1.0 }) * 0.53;

        let weighted_pred = if score > 0.0 {
            CausalDirection::Forward
        } else {
            CausalDirection::Backward
        };

        if weighted_pred == a.ground_truth {
            weighted_correct += 1;
        }
    }

    println!("Weighted Ensemble (by accuracy):   {}/108 ({:.1}%)",
             weighted_correct, weighted_correct as f64 / 108.0 * 100.0);

    // ==========================================
    // PHASE 5: Specific failure analysis
    // ==========================================
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║              PHASE 5: Specific Failure Patterns             ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // List the hardest pairs
    println!("Hardest pairs (0-1 methods correct):");
    for a in &analyses {
        let correct_count = [
            a.anm == a.ground_truth,
            a.igci == a.ground_truth,
            a.reci == a.ground_truth,
            a.learned == a.ground_truth,
            a.cea == a.ground_truth,
            a.info_theory == a.ground_truth,
            a.hdc == a.ground_truth,
        ].iter().filter(|&&x| x).count();

        if correct_count <= 1 {
            let truth_str = match a.ground_truth {
                CausalDirection::Forward => "X→Y",
                CausalDirection::Backward => "Y→X",
                _ => "?",
            };
            println!("  Pair {:3}: n={:4}, corr={:+.2}, nonlin={:.2}, truth={}",
                     a.pair_id, a.n_samples, a.correlation, a.nonlinearity, truth_str);
        }
    }

    // ==========================================
    // RECOMMENDATIONS
    // ==========================================
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                    RECOMMENDATIONS                          ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let theoretical_max = oracle_correct as f64 / 108.0 * 100.0;
    let current_best = 67.6;

    println!("Current Best (Info-Theoretic): {:.1}%", current_best);
    println!("Theoretical Max (Oracle):      {:.1}%", theoretical_max);
    println!("Improvement Potential:         {:.1}%\n", theoretical_max - current_best);

    if theoretical_max - current_best > 10.0 {
        println!("→ SIGNIFICANT room for ensemble improvement");
        println!("  - Methods capture complementary information");
        println!("  - Smart meta-learning could help select best method per-pair");
    } else if theoretical_max - current_best > 5.0 {
        println!("→ MODERATE room for ensemble improvement");
        println!("  - Better weighting could help");
        println!("  - Confidence-based selection recommended");
    } else {
        println!("→ LIMITED room for ensemble improvement");
        println!("  - Need fundamentally new algorithms");
        println!("  - Consider: neural approaches, more features");
    }

    println!("\nDone!");
}

// Helper functions
fn compute_correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.is_empty() { return 0.0; }
    let n = x.len() as f64;
    let mean_x: f64 = x.iter().sum::<f64>() / n;
    let mean_y: f64 = y.iter().sum::<f64>() / n;
    let std_x = std_dev(x);
    let std_y = std_dev(y);
    if std_x == 0.0 || std_y == 0.0 { return 0.0; }
    let cov: f64 = x.iter().zip(y.iter())
        .map(|(xi, yi)| (xi - mean_x) * (yi - mean_y))
        .sum::<f64>() / n;
    cov / (std_x * std_y)
}

fn std_dev(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    let n = data.len() as f64;
    let mean: f64 = data.iter().sum::<f64>() / n;
    let var: f64 = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    var.sqrt()
}

fn estimate_nonlinearity(x: &[f64], y: &[f64]) -> f64 {
    // Estimate nonlinearity as 1 - R² of linear fit
    if x.len() != y.len() || x.len() < 3 { return 0.0; }
    let n = x.len() as f64;
    let mean_x: f64 = x.iter().sum::<f64>() / n;
    let mean_y: f64 = y.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
    }

    let slope = if var_x > 1e-10 { cov / var_x } else { 0.0 };
    let intercept = mean_y - slope * mean_x;

    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;
    for i in 0..x.len() {
        let pred = slope * x[i] + intercept;
        ss_res += (y[i] - pred).powi(2);
        ss_tot += (y[i] - mean_y).powi(2);
    }

    let r2 = if ss_tot > 0.0 { 1.0 - ss_res / ss_tot } else { 0.0 };
    1.0 - r2.max(0.0).min(1.0)
}
