//! F2 and F3 Research Experiment Runner
//!
//! Runs:
//! - F2: Calibration Training Study
//! - F3: Wisdom Seed Transmission Study
//!
//! Outputs results to the results/ directory.

use epistemic_markets_research::{
    CalibrationTrainingExperiment, TrainingExperimentConfig, TrainingMethod,
    TrainingExperimentResults,
    WisdomTransmissionStudy, WisdomStudyConfig, WisdomSeedFormat,
    WisdomStudyResults,
};

use std::fs;
use std::path::Path;

fn main() {
    println!("{}", "=".repeat(70));
    println!("EPISTEMIC MARKETS RESEARCH - F2 & F3 EXPERIMENTS");
    println!("{}", "=".repeat(70));
    println!();

    // Ensure results directory exists
    let results_dir = Path::new("results");
    if !results_dir.exists() {
        fs::create_dir_all(results_dir).expect("Failed to create results directory");
    }

    // Run F2: Calibration Training Study
    println!("[F2] Starting Calibration Training Study...");
    println!("     - 7 training methods");
    println!("     - 150 participants per group (1,050 total)");
    println!("     - 200 training predictions + 100 assessment predictions per participant");
    println!("     - Cross-domain transfer testing enabled");
    println!();

    let f2_results = run_f2_calibration_experiment();
    save_f2_results(&f2_results);
    println!("[F2] Calibration Training Study complete.\n");

    // Run F3: Wisdom Seed Transmission Study
    println!("[F3] Starting Wisdom Seed Transmission Study...");
    println!("     - 4 seed formats (A/B testing)");
    println!("     - 75 participants per condition (300 total)");
    println!("     - 200 wisdom seeds");
    println!("     - 150 prediction rounds");
    println!();

    let f3_results = run_f3_wisdom_seed_experiment();
    save_f3_results(&f3_results);
    println!("[F3] Wisdom Seed Transmission Study complete.\n");

    // Print summary
    println!("{}", "=".repeat(70));
    println!("EXPERIMENT SUMMARY");
    println!("{}", "=".repeat(70));
    println!();

    println!("F2 Calibration Training:");
    println!("  Best Method: {:?}", f2_results.summary.best_method);
    println!("  Overall Improvement Rate: {:.1}%", f2_results.summary.overall_improvement_rate * 100.0);
    println!("  Calibration Trainable: {}", if f2_results.summary.calibration_is_trainable { "YES" } else { "NO" });
    println!();

    println!("F3 Wisdom Seed Transmission:");
    println!("  Best Format: {:?}", f3_results.summary.best_format);
    println!("  Transmission Rate: {:.1}%", f3_results.summary.transmission_rate * 100.0);
    println!("  Wisdom Transmissible: {}", if f3_results.summary.wisdom_is_transmissible { "YES" } else { "NO" });
    println!();

    println!("{}", "=".repeat(70));
    println!("ALL EXPERIMENTS COMPLETE");
    println!("Results saved to:");
    println!("  - results/f2_calibration_results.json");
    println!("  - results/f2_calibration_report.md");
    println!("  - results/f3_wisdom_seed_results.json");
    println!("  - results/f3_wisdom_seed_report.md");
    println!("{}", "=".repeat(70));
}

/// Run F2: Calibration Training Experiment
fn run_f2_calibration_experiment() -> TrainingExperimentResults {
    let config = TrainingExperimentConfig {
        methods: vec![
            TrainingMethod::Control,
            TrainingMethod::ImmediateFeedback,
            TrainingMethod::CalibrationGraphs { frequency: 10 },
            TrainingMethod::BaseRateAnchor,
            TrainingMethod::OverconfidenceWarning { threshold: 90 },
            TrainingMethod::OutsideView,
            TrainingMethod::Decomposition,
        ],
        participants_per_group: 150, // 150 per group for statistical power
        training_duration: 200,       // 200 predictions during training
        assessment_duration: 100,     // 100 predictions for assessment
        followup_assessments: vec![300, 500, 1000], // Simulated 1 week, 1 month, long-term
        domains: vec![
            "politics".to_string(),
            "technology".to_string(),
            "sports".to_string(),
            "science".to_string(),
            "economics".to_string(),
        ],
        test_transfer: true,
        seed: 20260130, // Date-based seed for reproducibility
    };

    let mut experiment = CalibrationTrainingExperiment::new(config);
    experiment.run().clone()
}

/// Run F3: Wisdom Seed Transmission Experiment
fn run_f3_wisdom_seed_experiment() -> WisdomStudyResults {
    let config = WisdomStudyConfig {
        formats: vec![
            WisdomSeedFormat::NaturalLanguage,
            WisdomSeedFormat::Structured {
                has_conditions: true,
                has_caveats: true,
                has_examples: true,
            },
            WisdomSeedFormat::Minimal,
            WisdomSeedFormat::Detailed,
        ],
        participants_per_condition: 75, // 75 per condition for A/B testing
        num_seeds: 200,                  // 200 wisdom seeds
        num_rounds: 150,                 // 150 rounds of predictions
        surfacing_probability: 0.35,     // 35% chance to show a seed
        domains: vec![
            "politics".to_string(),
            "technology".to_string(),
            "science".to_string(),
            "economics".to_string(),
        ],
        seed: 20260130,
    };

    let mut study = WisdomTransmissionStudy::new(config);
    study.run().clone()
}

/// Save F2 results to JSON and markdown report
fn save_f2_results(results: &TrainingExperimentResults) {
    // Convert to a JSON-serializable format (convert enum keys to strings)
    let json = create_f2_json(results);
    fs::write("results/f2_calibration_results.json", &json)
        .expect("Failed to write F2 JSON results");

    // Generate markdown report
    let report = generate_f2_report(results);
    fs::write("results/f2_calibration_report.md", report)
        .expect("Failed to write F2 markdown report");

    println!("  - Saved: results/f2_calibration_results.json");
    println!("  - Saved: results/f2_calibration_report.md");
}

/// Save F3 results to JSON and markdown report
fn save_f3_results(results: &WisdomStudyResults) {
    // Convert to a JSON-serializable format (convert enum keys to strings)
    let json = create_f3_json(results);
    fs::write("results/f3_wisdom_seed_results.json", &json)
        .expect("Failed to write F3 JSON results");

    // Generate markdown report
    let report = generate_f3_report(results);
    fs::write("results/f3_wisdom_seed_report.md", report)
        .expect("Failed to write F3 markdown report");

    println!("  - Saved: results/f3_wisdom_seed_results.json");
    println!("  - Saved: results/f3_wisdom_seed_report.md");
}

/// Generate F2 Calibration Training markdown report
fn generate_f2_report(results: &TrainingExperimentResults) -> String {
    let mut report = String::new();

    report.push_str("# F2: Calibration Training Study Results\n\n");
    report.push_str("## Research Question\n\n");
    report.push_str("**What training methods most effectively improve forecaster calibration?**\n\n");

    report.push_str("## Executive Summary\n\n");
    report.push_str(&format!(
        "- **Best Training Method**: {:?}\n",
        results.summary.best_method
    ));
    report.push_str(&format!(
        "- **Overall Improvement Rate**: {:.1}%\n",
        results.summary.overall_improvement_rate * 100.0
    ));
    report.push_str(&format!(
        "- **Average ECE Improvement**: {:.4}\n",
        results.summary.avg_ece_improvement
    ));
    report.push_str(&format!(
        "- **Calibration is Trainable**: {}\n",
        if results.summary.calibration_is_trainable { "YES" } else { "NO" }
    ));
    report.push_str(&format!(
        "- **Persistence Half-Life**: {} steps\n\n",
        results.summary.persistence_half_life
    ));

    report.push_str("## Methodology\n\n");
    report.push_str("### Training Methods Compared\n\n");
    report.push_str("| # | Method | Description |\n");
    report.push_str("|---|--------|-------------|\n");
    report.push_str("| 1 | Control | No training (baseline) |\n");
    report.push_str("| 2 | Immediate Feedback | Brier score shown after each prediction |\n");
    report.push_str("| 3 | Calibration Graphs | Periodic calibration curve visualization |\n");
    report.push_str("| 4 | Base Rate Anchor | Historical base rate reminders |\n");
    report.push_str("| 5 | Overconfidence Warning | Alert when predictions are extreme |\n");
    report.push_str("| 6 | Outside View | Reference class prompting |\n");
    report.push_str("| 7 | Decomposition | Break predictions into sub-questions |\n\n");

    report.push_str("### Experiment Parameters\n\n");
    report.push_str("- **Participants per group**: 150\n");
    report.push_str("- **Training predictions**: 200\n");
    report.push_str("- **Assessment predictions**: 100\n");
    report.push_str("- **Follow-up assessments**: Steps 300, 500, 1000 (simulated 1 week, 1 month, long-term)\n");
    report.push_str("- **Domains**: Politics, Technology, Sports, Science, Economics\n");
    report.push_str("- **Total iterations**: 1,050+ participants across 7 conditions\n\n");

    report.push_str("## Results by Training Method\n\n");
    report.push_str("### Method Effectiveness Summary\n\n");
    report.push_str("| Method | Avg ECE Improvement | Improvement Rate | Best | Worst | N |\n");
    report.push_str("|--------|---------------------|------------------|------|-------|---|\n");

    // Sort methods by effectiveness
    let mut methods: Vec<_> = results.method_effectiveness.iter().collect();
    methods.sort_by(|a, b| {
        b.1.avg_ece_improvement
            .partial_cmp(&a.1.avg_ece_improvement)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for (method, eff) in &methods {
        report.push_str(&format!(
            "| {:?} | {:.4} | {:.1}% | {:.4} | {:.4} | {} |\n",
            method,
            eff.avg_ece_improvement,
            eff.improvement_rate * 100.0,
            eff.best_improvement,
            eff.worst_improvement,
            eff.participant_count
        ));
    }
    report.push_str("\n");

    report.push_str("### Pre/Post Brier Score Analysis\n\n");
    report.push_str("| Method | Initial Brier (avg) | Final Brier (avg) | Improvement |\n");
    report.push_str("|--------|---------------------|-------------------|-------------|\n");

    for (method, improvements) in &results.improvements_by_method {
        if improvements.is_empty() {
            continue;
        }
        let avg_initial = improvements.iter().map(|i| i.initial_brier).sum::<f64>()
            / improvements.len() as f64;
        let avg_final = improvements.iter().map(|i| i.final_brier).sum::<f64>()
            / improvements.len() as f64;
        let improvement = avg_initial - avg_final;
        report.push_str(&format!(
            "| {:?} | {:.4} | {:.4} | {:.4} ({:.1}%) |\n",
            method,
            avg_initial,
            avg_final,
            improvement,
            if avg_initial > 0.0 { improvement / avg_initial * 100.0 } else { 0.0 }
        ));
    }
    report.push_str("\n");

    report.push_str("## Retention Analysis\n\n");
    report.push_str("### Simulated Follow-up Results\n\n");

    if !results.summary.avg_retention_at_followup.is_empty() {
        report.push_str("| Time Point | Avg Retention |\n");
        report.push_str("|------------|---------------|\n");
        let mut times: Vec<_> = results.summary.avg_retention_at_followup.iter().collect();
        times.sort_by_key(|(t, _)| *t);
        for (time, retention) in times {
            let label = match *time {
                300 => "~1 week (step 300)",
                500 => "~1 month (step 500)",
                1000 => "Long-term (step 1000)",
                _ => "Other",
            };
            report.push_str(&format!("| {} | {:.1}% |\n", label, retention * 100.0));
        }
        report.push_str("\n");
    }

    report.push_str(&format!(
        "**Estimated Half-Life**: {} steps\n\n",
        results.summary.persistence_half_life
    ));
    report.push_str("This suggests that without ongoing practice, calibration improvements decay over time, ");
    report.push_str("with approximately 50% of gains lost by the half-life point.\n\n");

    report.push_str("## Cross-Domain Transfer\n\n");

    if !results.transfer_analyses.is_empty() {
        report.push_str("| Source Domain | Target Domain | Transfer Coef. | Positive Transfer? |\n");
        report.push_str("|---------------|---------------|----------------|--------------------|\n");

        for transfer in &results.transfer_analyses {
            report.push_str(&format!(
                "| {} | {} | {:.3} | {} |\n",
                transfer.source_domain,
                transfer.target_domain,
                transfer.transfer_coefficient,
                if transfer.positive_transfer { "Yes" } else { "No" }
            ));
        }
        report.push_str("\n");

        report.push_str(&format!(
            "**Average Transfer Coefficient**: {:.3}\n\n",
            results.summary.transfer_coefficient_avg
        ));

        if !results.summary.positive_transfer_pairs.is_empty() {
            report.push_str("**Positive Transfer Pairs**:\n");
            for (source, target) in &results.summary.positive_transfer_pairs {
                report.push_str(&format!("- {} -> {}\n", source, target));
            }
            report.push_str("\n");
        }
    } else {
        report.push_str("Transfer analysis data not available.\n\n");
    }

    report.push_str("## Key Findings\n\n");

    report.push_str("### 1. Training Effectiveness\n\n");
    if results.summary.calibration_is_trainable {
        report.push_str("**Calibration IS trainable.** More than 50% of participants showed improvement.\n\n");
    } else {
        report.push_str("**Results mixed.** Less than 50% of participants showed improvement across all methods.\n\n");
    }

    report.push_str("### 2. Best Method\n\n");
    report.push_str(&format!(
        "The most effective training method was **{:?}**, achieving an average ECE improvement of {:.4}.\n\n",
        results.summary.best_method,
        results.method_effectiveness
            .get(&results.summary.best_method)
            .map(|e| e.avg_ece_improvement)
            .unwrap_or(0.0)
    ));

    report.push_str("### 3. Persistence\n\n");
    report.push_str(&format!(
        "Calibration improvements have a half-life of approximately {} steps, suggesting that:\n",
        results.summary.persistence_half_life
    ));
    report.push_str("- Regular practice is needed to maintain calibration gains\n");
    report.push_str("- Booster training sessions may be beneficial\n");
    report.push_str("- Integration into daily forecasting workflows is recommended\n\n");

    report.push_str("### 4. Cross-Domain Transfer\n\n");
    if results.summary.transfer_coefficient_avg > 0.3 {
        report.push_str("**Strong positive transfer** observed between domains. Skills learned in one domain generalize.\n\n");
    } else if results.summary.transfer_coefficient_avg > 0.1 {
        report.push_str("**Moderate transfer** observed. Some domain-specific training may be beneficial.\n\n");
    } else {
        report.push_str("**Limited transfer** between domains. Domain-specific training recommended.\n\n");
    }

    report.push_str("## Recommendations\n\n");
    for (i, rec) in results.summary.recommendations.iter().enumerate() {
        report.push_str(&format!("{}. {}\n", i + 1, rec));
    }
    report.push_str("\n");

    report.push_str("## Methodology Notes\n\n");
    report.push_str("- This study used simulated forecasters with realistic cognitive biases\n");
    report.push_str("- Calibration error was modeled using Expected Calibration Error (ECE)\n");
    report.push_str("- Learning rates varied between individuals (individual differences)\n");
    report.push_str("- Statistical power achieved through 150 participants per condition\n\n");

    report.push_str("---\n\n");
    report.push_str("*Generated by Epistemic Markets Research Infrastructure v0.1.0*\n");
    report.push_str(&format!("*Run date: 2026-01-30*\n"));

    report
}

/// Generate F3 Wisdom Seed Transmission markdown report
fn generate_f3_report(results: &WisdomStudyResults) -> String {
    let mut report = String::new();

    report.push_str("# F3: Wisdom Seed Transmission Study Results\n\n");
    report.push_str("## Research Question\n\n");
    report.push_str("**Do wisdom seeds actually influence future predictors?**\n\n");

    report.push_str("## Executive Summary\n\n");
    report.push_str(&format!(
        "- **Best Seed Format**: {:?}\n",
        results.summary.best_format
    ));
    report.push_str(&format!(
        "- **Transmission Rate**: {:.1}% of seed views led to prediction improvement\n",
        results.summary.transmission_rate * 100.0
    ));
    report.push_str(&format!(
        "- **Average Brier Improvement**: {:.4}\n",
        results.summary.avg_brier_improvement
    ));
    report.push_str(&format!(
        "- **Wisdom is Transmissible**: {}\n",
        if results.summary.wisdom_is_transmissible { "YES" } else { "NO" }
    ));
    report.push_str(&format!(
        "- **Network Depth**: {} generations\n",
        results.summary.network_depth
    ));
    report.push_str(&format!(
        "- **Avg Citations per Seed**: {:.2}\n\n",
        results.summary.avg_citations_per_seed
    ));

    report.push_str("## Methodology\n\n");
    report.push_str("### Seed Formats Tested (A/B)\n\n");
    report.push_str("| # | Format | Description |\n");
    report.push_str("|---|--------|-------------|\n");
    report.push_str("| 1 | Natural Language | Plain text explanation |\n");
    report.push_str("| 2 | Structured | Key factors listed with conditions/caveats |\n");
    report.push_str("| 3 | Minimal | Just the core prediction/insight |\n");
    report.push_str("| 4 | Detailed | Full reasoning chain |\n\n");

    report.push_str("### Experiment Parameters\n\n");
    report.push_str("- **Participants per condition**: 75\n");
    report.push_str("- **Total wisdom seeds**: 200\n");
    report.push_str("- **Prediction rounds**: 150\n");
    report.push_str("- **Seed surfacing probability**: 35%\n");
    report.push_str("- **Domains**: Politics, Technology, Science, Economics\n");
    report.push_str("- **Total participants**: 300 across 4 format conditions\n\n");

    report.push_str("## Results by Seed Format\n\n");
    report.push_str("### Format Effectiveness Comparison\n\n");
    report.push_str("| Format | Citation Rate | Brier w/ Seed | Brier w/o Seed | Improvement | N |\n");
    report.push_str("|--------|---------------|---------------|----------------|-------------|---|\n");

    // Sort formats by brier improvement
    let mut formats: Vec<_> = results.format_effectiveness.iter().collect();
    formats.sort_by(|a, b| {
        b.1.brier_improvement
            .partial_cmp(&a.1.brier_improvement)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for (format, eff) in &formats {
        report.push_str(&format!(
            "| {:?} | {:.1}% | {:.4} | {:.4} | {:.4} | {} |\n",
            format,
            eff.citation_rate * 100.0,
            eff.avg_brier_with_seed,
            eff.avg_brier_without_seed,
            eff.brier_improvement,
            eff.sample_size
        ));
    }
    report.push_str("\n");

    report.push_str("### Key Metrics Explained\n\n");
    report.push_str("- **Citation Rate**: % of viewed seeds that were explicitly cited by predictors\n");
    report.push_str("- **Brier w/ Seed**: Average Brier score when seed was shown\n");
    report.push_str("- **Brier w/o Seed**: Average Brier score without seed (baseline)\n");
    report.push_str("- **Improvement**: Brier reduction (positive = seeds helped)\n\n");

    report.push_str("## Information Transmission Analysis\n\n");

    report.push_str("### Transmission Fidelity\n\n");
    report.push_str(&format!(
        "**Overall Transmission Rate**: {:.1}%\n\n",
        results.summary.transmission_rate * 100.0
    ));
    report.push_str("This indicates the proportion of seed views that resulted in improved predictions. ");
    if results.summary.transmission_rate > 0.5 {
        report.push_str("A rate above 50% suggests wisdom seeds are effective at transmitting useful information.\n\n");
    } else {
        report.push_str("A rate below 50% suggests room for improvement in seed design or presentation.\n\n");
    }

    report.push_str("### Network Propagation\n\n");
    report.push_str(&format!(
        "- **Maximum Network Depth**: {} generations\n",
        results.summary.network_depth
    ));
    report.push_str(&format!(
        "- **Average Citations per Seed**: {:.2}\n\n",
        results.summary.avg_citations_per_seed
    ));

    if !results.summary.top_influencer_seeds.is_empty() {
        report.push_str("**Most Influential Seeds** (by PageRank-style influence score):\n\n");
        for (i, seed_id) in results.summary.top_influencer_seeds.iter().enumerate() {
            report.push_str(&format!("{}. {}\n", i + 1, seed_id));
        }
        report.push_str("\n");
    }

    report.push_str("## Key Findings\n\n");

    report.push_str("### 1. Wisdom Transmission Effectiveness\n\n");
    if results.summary.wisdom_is_transmissible {
        report.push_str("**Wisdom IS transmissible through seeds.** The majority of seed views led to prediction improvements.\n\n");
    } else {
        report.push_str("**Mixed results on transmission.** Less than half of seed views led to improvements, suggesting need for better seed design.\n\n");
    }

    report.push_str("### 2. Best Format\n\n");
    let best_eff = results.format_effectiveness.get(&results.summary.best_format);
    report.push_str(&format!(
        "The **{:?}** format performed best",
        results.summary.best_format
    ));
    if let Some(eff) = best_eff {
        report.push_str(&format!(
            " with a Brier improvement of {:.4} and citation rate of {:.1}%.\n\n",
            eff.brier_improvement,
            eff.citation_rate * 100.0
        ));
    } else {
        report.push_str(".\n\n");
    }

    report.push_str("### 3. Citation Behavior\n\n");
    let avg_citation_rate: f64 = results
        .format_effectiveness
        .values()
        .map(|f| f.citation_rate)
        .sum::<f64>()
        / results.format_effectiveness.len().max(1) as f64;

    if avg_citation_rate > 0.3 {
        report.push_str("**High engagement** with wisdom seeds observed. Predictors actively reference seeds.\n\n");
    } else if avg_citation_rate > 0.15 {
        report.push_str("**Moderate engagement** with seeds. Some predictors find them useful.\n\n");
    } else {
        report.push_str("**Low citation rate**. Consider prompting users to engage more with seeds.\n\n");
    }

    report.push_str("### 4. Network Effects\n\n");
    if results.summary.network_depth > 2 {
        report.push_str("**Multi-generational propagation** observed. Wisdom spreads through the network.\n\n");
    } else {
        report.push_str("**Limited propagation depth**. Seeds mostly influence direct viewers.\n\n");
    }

    report.push_str("## A/B Test Interpretation\n\n");

    // Find best and worst formats
    let best = formats.first();
    let worst = formats.last();

    if let (Some((best_fmt, best_eff)), Some((worst_fmt, worst_eff))) = (best, worst) {
        let diff = best_eff.brier_improvement - worst_eff.brier_improvement;
        report.push_str(&format!(
            "The difference between the best ({:?}) and worst ({:?}) formats is {:.4} Brier score improvement.\n\n",
            best_fmt, worst_fmt, diff
        ));

        if diff > 0.01 {
            report.push_str("This is a **meaningful difference** that justifies preferring the better format.\n\n");
        } else {
            report.push_str("The difference is **relatively small**, suggesting format choice has limited impact.\n\n");
        }
    }

    report.push_str("## Recommendations\n\n");
    for (i, rec) in results.summary.recommendations.iter().enumerate() {
        report.push_str(&format!("{}. {}\n", i + 1, rec));
    }
    report.push_str("\n");

    report.push_str("### Additional Recommendations\n\n");
    report.push_str("1. **Implement seed quality scoring** to surface high-impact seeds more frequently\n");
    report.push_str("2. **Consider combining formats** - structured overview with optional detailed expansion\n");
    report.push_str("3. **Track citation networks** to identify super-spreader seeds for amplification\n");
    report.push_str("4. **A/B test seed timing** - when in the prediction workflow to show seeds\n\n");

    report.push_str("## Methodology Notes\n\n");
    report.push_str("- Wisdom seeds were generated with varying quality and source accuracy\n");
    report.push_str("- Seed influence was measured by comparing pre/post-seed prediction accuracy\n");
    report.push_str("- Citation networks were built based on co-viewing patterns\n");
    report.push_str("- Influence scores computed using PageRank-style algorithm\n\n");

    report.push_str("---\n\n");
    report.push_str("*Generated by Epistemic Markets Research Infrastructure v0.1.0*\n");
    report.push_str("*Run date: 2026-01-30*\n");

    report
}

/// Create JSON for F2 results with string keys
fn create_f2_json(results: &TrainingExperimentResults) -> String {
    use serde_json::{json, Value, Map};

    // Convert method_effectiveness to string-keyed map
    let method_effectiveness: Map<String, Value> = results.method_effectiveness.iter()
        .map(|(method, eff)| {
            (format!("{:?}", method), json!({
                "method": format!("{:?}", method),
                "avg_ece_improvement": eff.avg_ece_improvement,
                "improvement_rate": eff.improvement_rate,
                "participant_count": eff.participant_count,
                "best_improvement": eff.best_improvement,
                "worst_improvement": eff.worst_improvement
            }))
        })
        .collect();

    // Convert improvements_by_method
    let improvements_by_method: Map<String, Value> = results.improvements_by_method.iter()
        .map(|(method, improvements)| {
            let imps: Vec<Value> = improvements.iter().map(|i| json!({
                "initial_ece": i.initial_ece,
                "final_ece": i.final_ece,
                "ece_improvement": i.ece_improvement,
                "ece_improvement_pct": i.ece_improvement_pct,
                "initial_brier": i.initial_brier,
                "final_brier": i.final_brier,
                "brier_improvement": i.brier_improvement,
                "training_method": format!("{:?}", i.training_method),
                "prediction_count": i.prediction_count
            })).collect();
            (format!("{:?}", method), json!(imps))
        })
        .collect();

    // Convert persistence_data
    let persistence_data: Map<String, Value> = results.persistence_data.iter()
        .map(|(method, data)| {
            (format!("{:?}", method), json!(data))
        })
        .collect();

    // Convert snapshots
    let snapshots: Vec<Value> = results.snapshots.iter()
        .map(|s| {
            let method_ece: Map<String, Value> = s.method_avg_ece.iter()
                .map(|(m, v)| (format!("{:?}", m), json!(v)))
                .collect();
            json!({
                "step": s.step,
                "method_avg_ece": method_ece
            })
        })
        .collect();

    // Convert transfer analyses
    let transfer_analyses: Vec<Value> = results.transfer_analyses.iter()
        .map(|t| json!({
            "source_domain": t.source_domain,
            "target_domain": t.target_domain,
            "performance_correlation": t.performance_correlation,
            "transfer_coefficient": t.transfer_coefficient,
            "positive_transfer": t.positive_transfer
        }))
        .collect();

    // Convert avg_retention_at_followup
    let retention: Map<String, Value> = results.summary.avg_retention_at_followup.iter()
        .map(|(k, v)| (k.to_string(), json!(v)))
        .collect();

    let output = json!({
        "experiment": "F2_Calibration_Training",
        "run_date": "2026-01-30",
        "method_effectiveness": method_effectiveness,
        "improvements_by_method": improvements_by_method,
        "snapshots": snapshots,
        "transfer_analyses": transfer_analyses,
        "persistence_data": persistence_data,
        "summary": {
            "best_method": format!("{:?}", results.summary.best_method),
            "overall_improvement_rate": results.summary.overall_improvement_rate,
            "avg_ece_improvement": results.summary.avg_ece_improvement,
            "calibration_is_trainable": results.summary.calibration_is_trainable,
            "positive_transfer_pairs": results.summary.positive_transfer_pairs,
            "transfer_coefficient_avg": results.summary.transfer_coefficient_avg,
            "persistence_half_life": results.summary.persistence_half_life,
            "avg_retention_at_followup": retention,
            "recommendations": results.summary.recommendations
        }
    });

    serde_json::to_string_pretty(&output).unwrap_or_else(|_| "{}".to_string())
}

/// Create JSON for F3 results with string keys
fn create_f3_json(results: &WisdomStudyResults) -> String {
    use serde_json::{json, Value, Map};

    // Convert format_effectiveness to string-keyed map
    let format_effectiveness: Map<String, Value> = results.format_effectiveness.iter()
        .map(|(format, eff)| {
            (format!("{:?}", format), json!({
                "format": format!("{:?}", format),
                "citation_rate": eff.citation_rate,
                "avg_brier_with_seed": eff.avg_brier_with_seed,
                "avg_brier_without_seed": eff.avg_brier_without_seed,
                "brier_improvement": eff.brier_improvement,
                "sample_size": eff.sample_size
            }))
        })
        .collect();

    let output = json!({
        "experiment": "F3_Wisdom_Seed_Transmission",
        "run_date": "2026-01-30",
        "format_effectiveness": format_effectiveness,
        "citation_network_stats": {
            "total_nodes": results.citation_network_stats.total_nodes,
            "total_edges": results.citation_network_stats.total_edges,
            "avg_degree": results.citation_network_stats.avg_degree,
            "max_depth": results.citation_network_stats.max_depth,
            "clustering_coefficient": results.citation_network_stats.clustering_coefficient
        },
        "summary": {
            "best_format": format!("{:?}", results.summary.best_format),
            "transmission_rate": results.summary.transmission_rate,
            "avg_brier_improvement": results.summary.avg_brier_improvement,
            "wisdom_is_transmissible": results.summary.wisdom_is_transmissible,
            "network_depth": results.summary.network_depth,
            "avg_citations_per_seed": results.summary.avg_citations_per_seed,
            "top_influencer_seeds": results.summary.top_influencer_seeds,
            "recommendations": results.summary.recommendations
        }
    });

    serde_json::to_string_pretty(&output).unwrap_or_else(|_| "{}".to_string())
}
