// ==================================================================================
// CLadder Benchmark Evaluation
// ==================================================================================
//
// Run the standard CLadder causal reasoning benchmark against Symthaea's solver.
// Compares original adapter with new NLP-enhanced adapter.
//
// CLadder tests causal reasoning across Pearl's 3 rungs:
//   Rung 1: Association (P(Y|X))
//   Rung 2: Intervention (P(Y|do(X)))
//   Rung 3: Counterfactual (Y_{X=x})
//
// GPT-4 Baseline: 64.28%
// Symthaea Target: >90%
//
// Usage: cargo run --example cladder_evaluation
//
// ==================================================================================

use symthaea::benchmarks::{CLadderAdapter, CLadderNLPAdapter, SymthaeaSolver};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║           CLadder Benchmark Evaluation                       ║");
    println!("║                                                              ║");
    println!("║   Standard causal reasoning benchmark (10,112 questions)     ║");
    println!("║   GPT-4 baseline: 64.28%                                     ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Load CLadder benchmark
    let cladder_path = "benchmarks/external/cladder-balanced.csv";

    println!("Loading CLadder benchmark from {}...", cladder_path);

    let adapter = match CLadderAdapter::load(cladder_path) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("Failed to load CLadder: {}", e);
            eprintln!("\nMake sure to download CLadder first:");
            eprintln!("  mkdir -p benchmarks/external");
            eprintln!("  curl -L 'https://huggingface.co/datasets/causal-nlp/CLadder/resolve/main/data/test-balanced-v1.5.csv' -o benchmarks/external/cladder-balanced.csv");
            return;
        }
    };

    println!("Loaded {} questions\n", adapter.len());

    // ==========================================
    // ORIGINAL ADAPTER (Baseline)
    // ==========================================
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║               Original Adapter (Baseline)                    ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let mut solver = SymthaeaSolver::new();
    let original_results = adapter.run_sample(&mut solver, 100);

    println!("  Original Adapter Accuracy: {:.1}%\n", original_results.accuracy() * 100.0);
    println!("  By Rung:");
    for rung in 1..=3 {
        let acc = original_results.accuracy_by_rung(rung);
        let (correct, total) = original_results.by_rung.get(&rung).unwrap_or(&(0, 0));
        let label = match rung {
            1 => "Association    ",
            2 => "Intervention   ",
            3 => "Counterfactual ",
            _ => "Unknown        ",
        };
        println!("    Rung {}: {} {:.1}% ({}/{})", rung, label, acc * 100.0, correct, total);
    }

    // ==========================================
    // NLP ADAPTER (Enhanced)
    // ==========================================
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                NLP Adapter (Enhanced)                        ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Load questions directly for NLP adapter
    let questions = load_questions(cladder_path, 100);
    if let Some(questions) = questions {
        let mut nlp_adapter = CLadderNLPAdapter::new();
        let mut nlp_solver = SymthaeaSolver::new();
        let nlp_results = nlp_adapter.run(&questions, &mut nlp_solver);

        println!("  NLP Adapter Accuracy: {:.1}%\n", nlp_results.accuracy() * 100.0);
        println!("  By Rung:");
        for rung in 1..=3 {
            let acc = nlp_results.accuracy_by_rung(rung);
            let (correct, total) = nlp_results.by_rung.get(&rung).unwrap_or(&(0, 0));
            let label = match rung {
                1 => "Association    ",
                2 => "Intervention   ",
                3 => "Counterfactual ",
                _ => "Unknown        ",
            };
            println!("    Rung {}: {} {:.1}% ({}/{})", rung, label, acc * 100.0, correct, total);
        }

        // ==========================================
        // COMPARISON
        // ==========================================
        println!("\n╔══════════════════════════════════════════════════════════════╗");
        println!("║                      Comparison                              ║");
        println!("╚══════════════════════════════════════════════════════════════╝\n");

        let original_acc = original_results.accuracy() * 100.0;
        let nlp_acc = nlp_results.accuracy() * 100.0;
        let improvement = nlp_acc - original_acc;
        let gpt4_baseline = 64.28;

        println!("  Original Adapter:    {:.1}%", original_acc);
        println!("  NLP Adapter:         {:.1}%", nlp_acc);
        println!("  Improvement:         {:+.1}%", improvement);
        println!("  ────────────────────────────────");
        println!("  GPT-4 Baseline:      {:.1}%", gpt4_baseline);
        println!("  Delta from GPT-4:    {:+.1}%", nlp_acc - gpt4_baseline);

        if nlp_acc > gpt4_baseline {
            println!("\n  SYMTHAEA BEATS GPT-4!");
        }

        // Detailed query type comparison
        println!("\n  By Query Type (NLP Adapter):");
        let mut qt_vec: Vec<_> = nlp_results.by_query_type.iter().collect();
        qt_vec.sort_by(|a, b| b.1.1.cmp(&a.1.1));
        for (qt, (correct, total)) in qt_vec.iter().take(10) {
            let acc = if *total > 0 { *correct as f64 / *total as f64 } else { 0.0 };
            println!("    {}: {:.1}% ({}/{})", qt, acc * 100.0, correct, total);
        }
    } else {
        println!("  Could not load questions for NLP adapter");
    }

    // Full evaluation option
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                   Full Evaluation                            ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!("\n  To run the full 10,112 question evaluation:");
    println!("  Set CLADDER_FULL=1 environment variable");

    if std::env::var("CLADDER_FULL").is_ok() {
        println!("\n  Running full evaluation (this may take a few minutes)...");

        let questions = load_questions(cladder_path, usize::MAX);
        if let Some(questions) = questions {
            let mut nlp_adapter = CLadderNLPAdapter::new();
            let mut nlp_solver = SymthaeaSolver::new();
            let full_results = nlp_adapter.run(&questions, &mut nlp_solver);

            println!("\n  Full Results (NLP Adapter):");
            println!("    Total:    {}", full_results.total);
            println!("    Correct:  {}", full_results.correct);
            println!("    Accuracy: {:.2}%", full_results.accuracy() * 100.0);

            for rung in 1..=3 {
                let acc = full_results.accuracy_by_rung(rung);
                println!("    Rung {}: {:.2}%", rung, acc * 100.0);
            }
        }
    }

    println!("\n  Done!");
}

/// Load questions directly from CSV
fn load_questions(path: &str, limit: usize) -> Option<Vec<symthaea::benchmarks::CLadderQuestion>> {
    use std::fs::File;
    use std::io::BufReader;

    let file = File::open(path).ok()?;
    let reader = BufReader::new(file);
    let mut csv_reader = csv::Reader::from_reader(reader);

    let mut questions = Vec::new();
    for result in csv_reader.deserialize() {
        if let Ok(question) = result {
            questions.push(question);
            if questions.len() >= limit {
                break;
            }
        }
    }

    Some(questions)
}
