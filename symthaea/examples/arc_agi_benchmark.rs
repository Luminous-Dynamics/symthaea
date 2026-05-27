// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! ARC-AGI Benchmark Runner — Strict + 2-AFC scoring on Chollet's dataset.
//!
//! Runs both the strict (pixel-perfect) and lenient (2-AFC) ARC evaluations,
//! comparing against published baselines.
//!
//! ## Two Modes
//!
//! 1. **Synthetic** (default): Procedurally generated tasks — tests encoding algebra.
//! 2. **Real dataset** (`--arc-data-dir`): Chollet's ARC JSON files — tests on real tasks.
//!
//! ## Expected Results
//!
//! - 2-AFC accuracy: ~85-95% (encoding fidelity)
//! - Strict solve rate: ~0-3% (grid reconstruction is noisy)
//! - The gap is the key finding: HDC encodes structure but can't reconstruct exactly.
//!
//! ## Usage
//!
//! ```bash
//! # Synthetic tasks (no data needed)
//! cargo run --example arc_agi_benchmark --release
//!
//! # Real ARC dataset
//! cargo run --example arc_agi_benchmark --release -- --arc-data-dir /path/to/ARC-AGI/data/training/
//! ```

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  ARC-AGI Benchmark — Strict + 2-AFC Scoring                ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let args: Vec<String> = std::env::args().collect();
    let arc_data_dir = args
        .windows(2)
        .find(|w| w[0] == "--arc-data-dir")
        .map(|w| w[1].clone());

    // Run synthetic strict benchmark
    run_synthetic_benchmark();

    // If real ARC data is provided, run on that too
    if let Some(dir) = arc_data_dir {
        println!("\n");
        run_real_arc_benchmark(&dir);
    } else {
        println!("\nTip: Pass --arc-data-dir /path/to/ARC-AGI/data/training/ for real ARC tasks.");
    }
}

fn run_synthetic_benchmark() {
    use symthaea_psych_bench::benchmarks::reasoning::arc_strict::ArcStrictBenchmark;
    use symthaea_psych_bench::harness::{PsychBenchmark, config::BenchmarkConfig};

    println!("━━━ Synthetic ARC Tasks (Strict Scoring) ━━━\n");

    let config = BenchmarkConfig {
        trials_per_condition: 200,
        ..BenchmarkConfig::default()
    };
    let bench = ArcStrictBenchmark;
    let result = bench.run(&config);

    // Print results
    println!("Trials: {}", result.trials_per_condition);
    println!("Time: {}ms\n", result.elapsed_ms);

    for note in &result.notes {
        println!("  {}", note);
    }

    println!("\nMetrics:");
    for (name, value) in &result.metrics {
        println!("  {}: {:.4} (±{:.4})", name, value.mean, value.std_dev);
    }

    // Comparison table
    println!("\n┌────────────────────┬──────────┐");
    println!("│ System             │ Solve %  │");
    println!("├────────────────────┼──────────┤");
    let strict = result
        .metrics
        .get("strict_solve_rate")
        .map(|m| m.mean)
        .unwrap_or(0.0);
    let twoafc = result
        .metrics
        .get("twoafc_accuracy")
        .map(|m| m.mean)
        .unwrap_or(0.0);
    println!("│ HDC-LTC (strict)   │ {:>6.1}%  │", strict * 100.0);
    println!("│ HDC-LTC (2-AFC)    │ {:>6.1}%  │", twoafc * 100.0);
    println!("│ GPT-4              │   ~5.0%  │");
    println!("│ o3-high            │  ~87.5%  │");
    println!("│ Human average      │  ~84.0%  │");
    println!("└────────────────────┴──────────┘");

    // Save JSON
    save_json("arc_agi_synthetic.json", &result);
}

fn run_real_arc_benchmark(dir: &str) {
    use symthaea_psych_bench::benchmarks::reasoning::arc_dataset::{
        evaluate_arc_tasks, load_arc_tasks,
    };

    println!("━━━ Real ARC Dataset (from {}) ━━━\n", dir);

    let path = std::path::Path::new(dir);
    match load_arc_tasks(path) {
        Ok(tasks) => {
            println!("Loaded {} ARC tasks", tasks.len());

            // Run 2-AFC evaluation (existing)
            let result = evaluate_arc_tasks(&tasks, 16384, 42);
            println!("\n2-AFC Results:");
            println!("  Tasks: {}", result.tasks_loaded);
            println!(
                "  Correct: {} ({:.1}%)",
                result.tasks_correct,
                result.tasks_correct as f64 / result.tasks_loaded.max(1) as f64 * 100.0
            );
            println!("  Mean similarity: {:.4}", result.mean_similarity);
            println!(
                "  Mean rule consistency: {:.4}",
                result.mean_rule_consistency
            );

            // Run strict evaluation
            println!("\nStrict (pixel-perfect) Results:");
            let strict_result = evaluate_arc_strict(&tasks, 42);
            println!("  Tasks evaluated: {}", strict_result.total);
            println!(
                "  Pixel-perfect solves: {} ({:.1}%)",
                strict_result.exact_matches,
                strict_result.exact_matches as f64 / strict_result.total.max(1) as f64 * 100.0
            );
            println!(
                "  Mean pixel accuracy: {:.1}%",
                strict_result.mean_pixel_accuracy * 100.0
            );

            // Breakdown by grid size
            println!("\n  By grid size:");
            for (size, (total, correct, mean_acc)) in &strict_result.by_grid_size {
                println!(
                    "    {}×{}: {}/{} exact, {:.1}% pixel accuracy",
                    size,
                    size,
                    correct,
                    total,
                    mean_acc * 100.0
                );
            }
        }
        Err(e) => {
            eprintln!("Failed to load ARC tasks: {}", e);
        }
    }
}

struct StrictArcResult {
    total: usize,
    exact_matches: usize,
    mean_pixel_accuracy: f64,
    by_grid_size: std::collections::BTreeMap<usize, (usize, usize, f64)>,
}

fn evaluate_arc_strict(
    tasks: &std::collections::BTreeMap<
        String,
        symthaea_psych_bench::benchmarks::reasoning::arc_dataset::ArcTask,
    >,
    seed: u64,
) -> StrictArcResult {
    use symthaea_core::hdc::binary_grid_encoder::BinaryGridEncoder;

    let mut total = 0usize;
    let mut exact_matches = 0usize;
    let mut pixel_acc_sum = 0.0f64;
    let mut by_grid_size: std::collections::BTreeMap<usize, (usize, usize, f64)> =
        std::collections::BTreeMap::new();

    for (idx, (_task_id, task)) in tasks.iter().enumerate() {
        if task.train.is_empty() || task.test.is_empty() {
            continue;
        }

        let max_rows = task
            .train
            .iter()
            .chain(task.test.iter())
            .flat_map(|p| [p.input.len(), p.output.len()])
            .max()
            .unwrap_or(30);
        let max_cols = task
            .train
            .iter()
            .chain(task.test.iter())
            .flat_map(|p| {
                [
                    p.input.iter().map(|r| r.len()).max().unwrap_or(0),
                    p.output.iter().map(|r| r.len()).max().unwrap_or(0),
                ]
            })
            .max()
            .unwrap_or(30);

        let encoder =
            BinaryGridEncoder::new(max_rows.max(1), max_cols.max(1), 10, seed ^ (idx as u64));

        // Build consensus rule
        let rules: Vec<_> = task
            .train
            .iter()
            .map(|pair| {
                let hv_in = encoder.encode_grid(&pair.input);
                let hv_out = encoder.encode_grid(&pair.output);
                encoder.encode_rule(&hv_in, &hv_out)
            })
            .collect();
        let consensus = encoder.bundle_rules(&rules);

        // Apply to test
        let test = &task.test[0];
        let test_in_hv = encoder.encode_grid(&test.input);
        let predicted_hv = encoder.apply_rule(&test_in_hv, &consensus);

        let out_rows = test.output.len();
        let out_cols = test.output.first().map(|r| r.len()).unwrap_or(0);
        let predicted_grid = encoder.decode_grid(&predicted_hv, out_rows, out_cols);

        let pixel_acc = BinaryGridEncoder::grid_accuracy(&predicted_grid, &test.output) as f64;
        let exact = BinaryGridEncoder::grid_exact_match(&predicted_grid, &test.output);

        total += 1;
        pixel_acc_sum += pixel_acc;
        if exact {
            exact_matches += 1;
        }

        let grid_size = max_rows.max(max_cols);
        let entry = by_grid_size.entry(grid_size).or_insert((0, 0, 0.0));
        entry.0 += 1;
        if exact {
            entry.1 += 1;
        }
        entry.2 += pixel_acc;
    }

    // Normalize accumulated accuracies
    for (_size, (count, _correct, acc)) in by_grid_size.iter_mut() {
        if *count > 0 {
            *acc /= *count as f64;
        }
    }

    StrictArcResult {
        total,
        exact_matches,
        mean_pixel_accuracy: if total > 0 {
            pixel_acc_sum / total as f64
        } else {
            0.0
        },
        by_grid_size,
    }
}

fn save_json(filename: &str, result: &symthaea_psych_bench::harness::report::BenchmarkResult) {
    let output_dir = std::path::Path::new("data/benchmarks/external");
    let _ = std::fs::create_dir_all(output_dir);
    let path = output_dir.join(filename);

    let mut json = String::from("{\n");
    json.push_str(&format!("  \"benchmark\": \"{}\",\n", result.benchmark));
    json.push_str(&format!("  \"trials\": {},\n", result.trials_per_condition));
    json.push_str(&format!("  \"elapsed_ms\": {},\n", result.elapsed_ms));
    json.push_str("  \"metrics\": {\n");
    let mut first = true;
    for (name, value) in &result.metrics {
        if !first {
            json.push_str(",\n");
        }
        first = false;
        json.push_str(&format!(
            "    \"{}\": {{ \"mean\": {:.6}, \"sd\": {:.6} }}",
            name, value.mean, value.std_dev
        ));
    }
    json.push_str("\n  },\n");
    json.push_str("  \"notes\": [\n");
    for (i, note) in result.notes.iter().enumerate() {
        json.push_str(&format!("    \"{}\"", note.replace('"', "\\\"")));
        if i + 1 < result.notes.len() {
            json.push(',');
        }
        json.push('\n');
    }
    json.push_str("  ]\n}\n");

    match std::fs::write(&path, &json) {
        Ok(()) => println!("\nResults saved to {}", path.display()),
        Err(e) => eprintln!("Warning: Failed to save JSON: {}", e),
    }
}
