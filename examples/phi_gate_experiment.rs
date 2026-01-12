//! # Φ-Gate Experiment - Phase 2 Validation
//!
//! Tests the hypothesis: "Does Φ correlate with reasoning accuracy?"
//!
//! ## Target: r > 0.3 (Pearson correlation coefficient)
//!
//! ## Usage:
//! ```bash
//! cargo run --example phi_gate_experiment --release
//! ```

use symthaea::benchmarks::mmlu::{MMLUQuestion, MMLUBenchmark, BenchmarkResults, sample_questions};

/// Generate extended test questions across multiple subjects
fn generate_extended_questions() -> Vec<MMLUQuestion> {
    let mut questions = Vec::new();

    // === MATHEMATICS ===
    questions.push(MMLUQuestion::new(
        "What is 7 × 8?".to_string(),
        "mathematics".to_string(),
        ["54".to_string(), "56".to_string(), "58".to_string(), "64".to_string()],
        1,
    ));
    questions.push(MMLUQuestion::new(
        "If x + 5 = 12, what is x?".to_string(),
        "mathematics".to_string(),
        ["5".to_string(), "7".to_string(), "12".to_string(), "17".to_string()],
        1,
    ));
    questions.push(MMLUQuestion::new(
        "What is the square root of 144?".to_string(),
        "mathematics".to_string(),
        ["10".to_string(), "11".to_string(), "12".to_string(), "14".to_string()],
        2,
    ));
    questions.push(MMLUQuestion::new(
        "What is 15% of 200?".to_string(),
        "mathematics".to_string(),
        ["20".to_string(), "25".to_string(), "30".to_string(), "35".to_string()],
        2,
    ));

    // === LOGIC ===
    questions.push(MMLUQuestion::new(
        "If all dogs are mammals, and all mammals are animals, then:".to_string(),
        "philosophy".to_string(),
        [
            "Some dogs are not animals".to_string(),
            "All dogs are animals".to_string(),
            "No dogs are animals".to_string(),
            "Some animals are dogs".to_string(),
        ],
        1,
    ));
    questions.push(MMLUQuestion::new(
        "If P implies Q, and P is true, then:".to_string(),
        "philosophy".to_string(),
        [
            "Q is false".to_string(),
            "Q is true".to_string(),
            "Q is unknown".to_string(),
            "P is false".to_string(),
        ],
        1,
    ));
    questions.push(MMLUQuestion::new(
        "The statement 'Not (A and B)' is equivalent to:".to_string(),
        "philosophy".to_string(),
        [
            "A and B".to_string(),
            "(Not A) or (Not B)".to_string(),
            "(Not A) and (Not B)".to_string(),
            "A or B".to_string(),
        ],
        1,
    ));

    // === PHYSICS ===
    questions.push(MMLUQuestion::new(
        "What is the SI unit of force?".to_string(),
        "physics".to_string(),
        ["Watt".to_string(), "Newton".to_string(), "Joule".to_string(), "Pascal".to_string()],
        1,
    ));
    questions.push(MMLUQuestion::new(
        "If you double the speed of an object, its kinetic energy:".to_string(),
        "physics".to_string(),
        [
            "Doubles".to_string(),
            "Quadruples".to_string(),
            "Halves".to_string(),
            "Stays the same".to_string(),
        ],
        1,
    ));
    questions.push(MMLUQuestion::new(
        "The law of conservation of energy states that:".to_string(),
        "physics".to_string(),
        [
            "Energy can be created".to_string(),
            "Energy can be destroyed".to_string(),
            "Energy cannot be created or destroyed".to_string(),
            "Energy always increases".to_string(),
        ],
        2,
    ));

    // === PSYCHOLOGY ===
    questions.push(MMLUQuestion::new(
        "Classical conditioning was discovered by:".to_string(),
        "psychology".to_string(),
        [
            "Freud".to_string(),
            "Pavlov".to_string(),
            "Skinner".to_string(),
            "Jung".to_string(),
        ],
        1,
    ));
    questions.push(MMLUQuestion::new(
        "The 'fight or flight' response is controlled by:".to_string(),
        "psychology".to_string(),
        [
            "Parasympathetic nervous system".to_string(),
            "Sympathetic nervous system".to_string(),
            "Central nervous system".to_string(),
            "Peripheral nervous system".to_string(),
        ],
        1,
    ));

    // === ECONOMICS ===
    questions.push(MMLUQuestion::new(
        "When demand increases and supply stays constant, price will:".to_string(),
        "economics".to_string(),
        [
            "Decrease".to_string(),
            "Stay the same".to_string(),
            "Increase".to_string(),
            "Become zero".to_string(),
        ],
        2,
    ));
    questions.push(MMLUQuestion::new(
        "GDP measures:".to_string(),
        "economics".to_string(),
        [
            "Only imports".to_string(),
            "Total economic output".to_string(),
            "Only exports".to_string(),
            "Government debt".to_string(),
        ],
        1,
    ));

    // === BIOLOGY ===
    questions.push(MMLUQuestion::new(
        "The powerhouse of the cell is:".to_string(),
        "biology".to_string(),
        [
            "Nucleus".to_string(),
            "Mitochondria".to_string(),
            "Ribosome".to_string(),
            "Golgi apparatus".to_string(),
        ],
        1,
    ));
    questions.push(MMLUQuestion::new(
        "DNA is composed of:".to_string(),
        "biology".to_string(),
        [
            "Amino acids".to_string(),
            "Nucleotides".to_string(),
            "Lipids".to_string(),
            "Carbohydrates".to_string(),
        ],
        1,
    ));

    // === COMPUTER SCIENCE ===
    questions.push(MMLUQuestion::new(
        "The time complexity of binary search is:".to_string(),
        "computer_science".to_string(),
        [
            "O(n)".to_string(),
            "O(n^2)".to_string(),
            "O(log n)".to_string(),
            "O(1)".to_string(),
        ],
        2,
    ));
    questions.push(MMLUQuestion::new(
        "A stack data structure follows:".to_string(),
        "computer_science".to_string(),
        [
            "First In First Out".to_string(),
            "Last In First Out".to_string(),
            "Random access".to_string(),
            "Priority ordering".to_string(),
        ],
        1,
    ));

    // === ETHICS ===
    questions.push(MMLUQuestion::new(
        "Utilitarianism judges actions by their:".to_string(),
        "ethics".to_string(),
        [
            "Intent".to_string(),
            "Consequences".to_string(),
            "Divine command".to_string(),
            "Tradition".to_string(),
        ],
        1,
    ));
    questions.push(MMLUQuestion::new(
        "Kant's categorical imperative emphasizes:".to_string(),
        "ethics".to_string(),
        [
            "Maximizing happiness".to_string(),
            "Following universal moral laws".to_string(),
            "Self-interest".to_string(),
            "Situational ethics".to_string(),
        ],
        1,
    ));

    // Add the original sample questions
    questions.extend(sample_questions());

    questions
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║           Φ-GATE EXPERIMENT - Phase 2 Validation             ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║ Hypothesis: Φ correlates with reasoning accuracy             ║");
    println!("║ Target: Pearson r > 0.3                                      ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Generate questions
    let questions = generate_extended_questions();
    println!("Generated {} questions across multiple subjects", questions.len());

    // Subject breakdown
    let mut subject_counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for q in &questions {
        *subject_counts.entry(q.subject.as_str()).or_insert(0) += 1;
    }
    println!("\nSubject distribution:");
    for (subject, count) in &subject_counts {
        println!("  {}: {} questions", subject, count);
    }
    println!();

    // Run the benchmark
    println!("Running benchmark...");
    let benchmark = MMLUBenchmark::new().with_max_steps(10);
    let start = std::time::Instant::now();
    let results = benchmark.run_benchmark(&questions);
    let elapsed = start.elapsed();

    // Print detailed results
    println!("\n{}", results.summary());

    // Print timing
    println!("Total time: {:.2}s ({:.1}ms per question)",
        elapsed.as_secs_f64(),
        elapsed.as_millis() as f64 / questions.len() as f64
    );

    // Per-subject breakdown
    println!("\n=== Per-Subject Breakdown ===");
    for subject in subject_counts.keys() {
        let subject_results: Vec<_> = results.results.iter()
            .filter(|r| r.question.subject == *subject)
            .collect();

        if subject_results.is_empty() {
            continue;
        }

        let correct = subject_results.iter().filter(|r| r.is_correct).count();
        let total = subject_results.len();
        let acc = correct as f64 / total as f64;
        let mean_phi: f64 = subject_results.iter().map(|r| r.phi).sum::<f64>() / total as f64;

        println!("{:20} | Acc: {:5.1}% | Mean Φ: {:.4} | n={}",
            subject, acc * 100.0, mean_phi, total);
    }

    // Φ-Gate decision
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    if results.phi_gate_passed {
        println!("║  Φ-GATE: PASSED ✓                                            ║");
        println!("║  The correlation r = {:.4} exceeds threshold 0.3            ║", results.phi_accuracy_correlation);
        println!("║  Evidence supports Φ-accuracy hypothesis!                    ║");
    } else {
        println!("║  Φ-GATE: NOT YET PASSED                                      ║");
        println!("║  The correlation r = {:.4} is below threshold 0.3           ║", results.phi_accuracy_correlation);
        println!("║  Further refinement needed before NixOS integration          ║");
    }
    println!("╚══════════════════════════════════════════════════════════════╝");

    // Detailed analysis
    println!("\n=== Detailed Φ-Accuracy Analysis ===");

    // Sort by Φ and show quartiles
    let mut sorted_results: Vec<_> = results.results.iter().collect();
    sorted_results.sort_by(|a, b| a.phi.partial_cmp(&b.phi).unwrap());

    let n = sorted_results.len();
    let q1_idx = n / 4;
    let q2_idx = n / 2;
    let q3_idx = 3 * n / 4;

    let q1_acc = sorted_results[..q1_idx].iter().filter(|r| r.is_correct).count() as f64 / q1_idx as f64;
    let q2_acc = sorted_results[q1_idx..q2_idx].iter().filter(|r| r.is_correct).count() as f64 / (q2_idx - q1_idx) as f64;
    let q3_acc = sorted_results[q2_idx..q3_idx].iter().filter(|r| r.is_correct).count() as f64 / (q3_idx - q2_idx) as f64;
    let q4_acc = sorted_results[q3_idx..].iter().filter(|r| r.is_correct).count() as f64 / (n - q3_idx) as f64;

    println!("Accuracy by Φ quartile (should increase if correlated):");
    println!("  Q1 (lowest Φ):  {:.1}%", q1_acc * 100.0);
    println!("  Q2:             {:.1}%", q2_acc * 100.0);
    println!("  Q3:             {:.1}%", q3_acc * 100.0);
    println!("  Q4 (highest Φ): {:.1}%", q4_acc * 100.0);

    // Check if accuracy increases with Φ (monotonic trend)
    let monotonic_increasing = q1_acc <= q2_acc && q2_acc <= q3_acc && q3_acc <= q4_acc;
    println!("\nMonotonic increasing trend: {}", if monotonic_increasing { "Yes ✓" } else { "No" });

    // Interpretation
    println!("\n=== Interpretation ===");
    if results.phi_accuracy_correlation > 0.5 {
        println!("Strong positive correlation - Φ is a good predictor of reasoning ability");
    } else if results.phi_accuracy_correlation > 0.3 {
        println!("Moderate positive correlation - Φ has predictive value");
    } else if results.phi_accuracy_correlation > 0.1 {
        println!("Weak positive correlation - some relationship exists");
    } else if results.phi_accuracy_correlation > -0.1 {
        println!("No significant correlation - Φ may need refinement");
    } else {
        println!("Negative correlation - unexpected, investigate measurement");
    }
}
