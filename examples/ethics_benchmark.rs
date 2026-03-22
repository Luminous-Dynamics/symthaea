//! Hendrycks ETHICS Benchmark Runner — External moral reasoning validation.
//!
//! Evaluates Symthaea's keyword-based moral algebra against the Hendrycks ETHICS
//! benchmark across 5 domains: commonsense, deontology, justice, virtue, utilitarianism.
//!
//! ## Expected Results
//!
//! Commonsense ~55-65%, Deontology ~50-60%, Justice ~50-60%, Virtue ~45-55%,
//! Utilitarianism ~50-55%. Near chance on nuanced ethics — keyword parsing captures
//! structure but not semantics.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example ethics_benchmark --release
//! ```

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Hendrycks ETHICS Benchmark — Moral Reasoning Validation   ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    #[cfg(not(feature = "external-benchmarks"))]
    {
        // Run directly without feature gate for the built-in corpus
        run_ethics_benchmark();
    }

    #[cfg(feature = "external-benchmarks")]
    {
        run_ethics_benchmark();
    }
}

fn run_ethics_benchmark() {
    use symthaea_psych_bench::benchmarks::external::hendrycks_ethics::HendrycksEthicsBenchmark;
    use symthaea_psych_bench::harness::{config::BenchmarkConfig, PsychBenchmark};

    let bench = HendrycksEthicsBenchmark;
    let config = BenchmarkConfig::default();
    let result = bench.run(&config);

    // Print results
    println!("Benchmark: {}", result.benchmark);
    println!("Conditions: {} domains", result.conditions);
    println!("Scenarios per domain: {}", result.trials_per_condition);
    println!("Time: {}ms\n", result.elapsed_ms);

    // Per-domain results
    println!("┌──────────────────┬──────────┬──────────┐");
    println!("│ Domain           │ Accuracy │ Baseline │");
    println!("├──────────────────┼──────────┼──────────┤");

    let domains = [
        ("commonsense", "~85%"),
        ("deontology", "~85%"),
        ("justice", "~85%"),
        ("virtue", "~85%"),
        ("utilitarianism", "~85%"),
    ];
    for (domain, gpt4_baseline) in &domains {
        let key = format!("{}_accuracy", domain);
        if let Some(value) = result.metrics.get(&key) {
            println!(
                "│ {:<16} │ {:>6.1}%  │ GPT4:{}│",
                domain,
                value.mean * 100.0,
                gpt4_baseline
            );
        }
    }
    if let Some(composite) = result.metrics.get("composite_accuracy") {
        println!("├──────────────────┼──────────┼──────────┤");
        println!(
            "│ {:<16} │ {:>6.1}%  │ GPT4:~85%│",
            "COMPOSITE",
            composite.mean * 100.0
        );
    }
    println!("└──────────────────┴──────────┴──────────┘");

    // Comparison table
    println!("\n┌────────────────────┬──────────┐");
    println!("│ System             │ Accuracy │");
    println!("├────────────────────┼──────────┤");
    let composite_acc = result
        .metrics
        .get("composite_accuracy")
        .map(|m| m.mean)
        .unwrap_or(0.0);
    println!("│ HDC-LTC (ours)     │ {:>6.1}%  │", composite_acc * 100.0);
    println!("│ GPT-4              │  ~85.0%  │");
    println!("│ BERT-large         │  ~70.0%  │");
    println!("│ Random baseline    │   50.0%  │");
    println!("└────────────────────┴──────────┘");

    // Notes
    println!("\nNotes:");
    for note in &result.notes {
        println!("  {}", note);
    }

    // Save JSON
    let output_dir = std::path::Path::new("data/benchmarks/external");
    let _ = std::fs::create_dir_all(output_dir);

    let mut json = String::from("{\n");
    json.push_str(&format!(
        "  \"benchmark\": \"{}\",\n",
        result.benchmark
    ));
    json.push_str("  \"reference\": \"Hendrycks et al., 2021. Aligning AI With Shared Human Values. ICLR 2021.\",\n");
    json.push_str("  \"metrics\": {\n");
    let mut first = true;
    for (name, value) in &result.metrics {
        if !first {
            json.push_str(",\n");
        }
        first = false;
        json.push_str(&format!(
            "    \"{}\": {{ \"mean\": {:.6}, \"sd\": {:.6}, \"ci_lower\": {:.6}, \"ci_upper\": {:.6} }}",
            name, value.mean, value.std_dev, value.ci_lower, value.ci_upper
        ));
    }
    json.push_str("\n  },\n");
    json.push_str("  \"notes\": [\n");
    for (i, note) in result.notes.iter().enumerate() {
        json.push_str(&format!(
            "    \"{}\"",
            note.replace('"', "\\\"")
        ));
        if i + 1 < result.notes.len() {
            json.push(',');
        }
        json.push('\n');
    }
    json.push_str("  ]\n}\n");

    let path = output_dir.join("hendrycks_ethics.json");
    match std::fs::write(&path, &json) {
        Ok(()) => println!("\nResults saved to {}", path.display()),
        Err(e) => eprintln!("Warning: Failed to save JSON: {}", e),
    }
}
