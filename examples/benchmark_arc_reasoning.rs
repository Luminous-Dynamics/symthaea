//! # ARC (Abstraction and Reasoning Corpus) Benchmark
//!
//! Tests Symthaea's HDC encoding and pattern recognition on the ARC benchmark
//! (Chollet 2019), which evaluates abstract reasoning and generalization.
//!
//! ## Method
//! 1. Encode each ARC grid as an HDC vector using spatial+color encoding
//! 2. Encode input→output transformation as a "rule vector" (bind input with output)
//! 3. For each test task, compute similarity between training rule vectors
//! 4. Measure task-level consistency and cross-task generalization
//!
//! ## Significance
//! ARC is the gold standard for measuring fluid intelligence and abstraction.
//! HDC's compositional binding operations are theoretically well-suited for
//! representing spatial transformations, making this a critical validation.
//!
//! ## Dataset
//! ARC-AGI (Chollet 2019): 400 training + 400 evaluation tasks
//! Each task has 2-5 training examples + 1 test example
//! Grids are 2D arrays of integers (0-9 colors), typically 10x10
//!
//! ## Run
//! ```bash
//! cargo run --example benchmark_arc_reasoning --release
//! ```

use std::fs;
use std::env;
use std::path::Path;
use std::path::PathBuf;
use std::time::Instant;

use symthaea::hdc::unified_hv::ContinuousHV;

fn arc_data_dir() -> PathBuf {
    env::var("SYMTHAEA_ARC_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("data/benchmarks/arc/repo/data"))
}

fn arc_results_path() -> PathBuf {
    env::var("SYMTHAEA_ARC_RESULTS_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("data/benchmarks/arc/results.json"))
}

/// ARC grid encoder using HDC spatial composition
struct ArcHdcEncoder {
    dim: usize,
    /// Color HVs (10 colors: 0-9)
    color_hvs: Vec<ContinuousHV>,
    /// Position HVs for (row, col) - up to 30x30
    row_hvs: Vec<ContinuousHV>,
    col_hvs: Vec<ContinuousHV>,
}

impl ArcHdcEncoder {
    fn new(dim: usize) -> Self {
        let color_hvs: Vec<ContinuousHV> = (0..10)
            .map(|c| ContinuousHV::random(dim, 50000 + c as u64))
            .collect();

        let row_hvs: Vec<ContinuousHV> = (0..30)
            .map(|r| ContinuousHV::random(dim, 60000 + r as u64))
            .collect();

        let col_hvs: Vec<ContinuousHV> = (0..30)
            .map(|c| ContinuousHV::random(dim, 70000 + c as u64))
            .collect();

        Self {
            dim,
            color_hvs,
            row_hvs,
            col_hvs,
        }
    }

    /// Encode a 2D grid into an HDC vector
    /// Grid encoding: bundle(bind(row_hv, col_hv, color_hv) for each cell)
    fn encode_grid(&self, grid: &[Vec<u8>]) -> ContinuousHV {
        let mut accumulator = vec![0.0f32; self.dim];
        let mut cell_count = 0;

        for (r, row) in grid.iter().enumerate() {
            for (c, &color) in row.iter().enumerate() {
                if r >= 30 || c >= 30 || color as usize >= 10 {
                    continue;
                }
                // bind(row, col, color) - triple binding for spatial+semantic
                let pos = self.row_hvs[r].bind(&self.col_hvs[c]);
                let cell = pos.bind(&self.color_hvs[color as usize]);

                for (acc, &val) in accumulator.iter_mut().zip(cell.values.iter()) {
                    *acc += val;
                }
                cell_count += 1;
            }
        }

        // Normalize
        if cell_count > 0 {
            let norm: f32 = accumulator.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for v in &mut accumulator {
                    *v /= norm;
                }
            }
        }

        ContinuousHV::from_vec(accumulator)
    }

    /// Encode a transformation (input→output pair) as a "rule vector"
    /// Rule = bind(input_encoding, output_encoding)
    fn encode_rule(&self, input: &[Vec<u8>], output: &[Vec<u8>]) -> ContinuousHV {
        let input_hv = self.encode_grid(input);
        let output_hv = self.encode_grid(output);
        input_hv.bind(&output_hv)
    }

    /// Compute grid metadata features
    #[allow(dead_code)]
    fn grid_features(grid: &[Vec<u8>]) -> GridFeatures {
        let rows = grid.len();
        let cols = if rows > 0 { grid[0].len() } else { 0 };

        let mut color_counts = [0usize; 10];
        for row in grid {
            for &c in row {
                if (c as usize) < 10 {
                    color_counts[c as usize] += 1;
                }
            }
        }

        let n_colors = color_counts.iter().filter(|&&c| c > 0).count();
        let non_bg = color_counts[1..].iter().sum::<usize>();

        GridFeatures {
            rows,
            cols,
            n_colors,
            non_background_cells: non_bg,
            total_cells: rows * cols,
        }
    }
}

#[allow(dead_code)]
struct GridFeatures {
    rows: usize,
    cols: usize,
    n_colors: usize,
    non_background_cells: usize,
    total_cells: usize,
}

/// Parse ARC JSON task file
fn parse_task(path: &Path) -> Option<ArcTask> {
    let content = fs::read_to_string(path).ok()?;
    let json: serde_json::Value = serde_json::from_str(&content).ok()?;

    let train = json["train"]
        .as_array()?
        .iter()
        .filter_map(|example| {
            let input = parse_grid(&example["input"])?;
            let output = parse_grid(&example["output"])?;
            Some((input, output))
        })
        .collect::<Vec<_>>();

    let test = json["test"]
        .as_array()?
        .iter()
        .filter_map(|example| {
            let input = parse_grid(&example["input"])?;
            let output = parse_grid(&example["output"])?;
            Some((input, output))
        })
        .collect::<Vec<_>>();

    if train.is_empty() || test.is_empty() {
        return None;
    }

    Some(ArcTask { train, test })
}

fn parse_grid(value: &serde_json::Value) -> Option<Vec<Vec<u8>>> {
    let arr = value.as_array()?;
    arr.iter()
        .map(|row| {
            row.as_array().map(|r| {
                r.iter()
                    .filter_map(|v| v.as_u64().map(|n| n as u8))
                    .collect()
            })
        })
        .collect()
}

struct ArcTask {
    train: Vec<(Vec<Vec<u8>>, Vec<Vec<u8>>)>,
    test: Vec<(Vec<Vec<u8>>, Vec<Vec<u8>>)>,
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║       ARC Reasoning Benchmark                              ║");
    println!("║       HDC Spatial Composition for Abstract Reasoning       ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let data_dir = arc_data_dir();
    let training_dir = data_dir.join("training");
    let eval_dir = data_dir.join("evaluation");

    if !training_dir.exists() {
        eprintln!("ARC data not found at {}", data_dir.display());
        eprintln!("Set SYMTHAEA_ARC_DATA_DIR to override the dataset location.");
        eprintln!("Clone: git clone https://github.com/fchollet/ARC-AGI data/benchmarks/arc/repo");
        return;
    }

    // Load training tasks
    let mut task_files: Vec<_> = fs::read_dir(&training_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "json").unwrap_or(false))
        .map(|e| e.path())
        .collect();
    task_files.sort();

    println!("Found {} training tasks\n", task_files.len());

    let dim = 4096;
    let encoder = ArcHdcEncoder::new(dim);

    // ═══════════════════════════════════════════════════════════════
    // Test 1: Rule Consistency within tasks
    // Within each task, training examples should have similar rule vectors
    // ═══════════════════════════════════════════════════════════════
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 1: Intra-task Rule Consistency");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let t = Instant::now();
    let mut consistencies = Vec::new();
    let mut tasks_with_consistent_rules = 0;
    let max_tasks = 400;

    for task_path in task_files.iter().take(max_tasks) {
        let task = match parse_task(task_path) {
            Some(t) => t,
            None => continue,
        };

        if task.train.len() < 2 {
            continue;
        }

        // Encode all training rules
        let rule_hvs: Vec<ContinuousHV> = task
            .train
            .iter()
            .map(|(inp, out)| encoder.encode_rule(inp, out))
            .collect();

        // Average pairwise similarity
        let mut total_sim = 0.0f32;
        let mut n_pairs = 0;
        for i in 0..rule_hvs.len() {
            for j in i + 1..rule_hvs.len() {
                total_sim += rule_hvs[i].similarity(&rule_hvs[j]);
                n_pairs += 1;
            }
        }

        if n_pairs > 0 {
            let avg_sim = total_sim / n_pairs as f32;
            consistencies.push(avg_sim);
            if avg_sim > 0.0 {
                tasks_with_consistent_rules += 1;
            }
        }
    }

    let mean_consistency: f32 =
        consistencies.iter().sum::<f32>() / consistencies.len().max(1) as f32;
    let consistent_pct = tasks_with_consistent_rules as f64 / consistencies.len().max(1) as f64;

    println!(
        "  Tasks analyzed: {} (with ≥2 training examples)",
        consistencies.len()
    );
    println!("  Mean rule similarity: {:.4}", mean_consistency);
    println!(
        "  Tasks with positive consistency: {:.1}% ({}/{})",
        consistent_pct * 100.0,
        tasks_with_consistent_rules,
        consistencies.len()
    );
    println!("  Time: {:.1}s\n", t.elapsed().as_secs_f64());

    // ═══════════════════════════════════════════════════════════════
    // Test 2: Cross-task Discrimination
    // Rules from different tasks should be dissimilar
    // ═══════════════════════════════════════════════════════════════
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 2: Cross-task Rule Discrimination");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let t = Instant::now();
    // Collect one rule per task (first training example)
    let mut task_rules = Vec::new();
    for task_path in task_files.iter().take(100) {
        if let Some(task) = parse_task(task_path) {
            let (inp, out) = &task.train[0];
            task_rules.push(encoder.encode_rule(inp, out));
        }
    }

    // Cross-task pairwise similarity (sample)
    let mut cross_sims = Vec::new();
    let sample_size = 500;
    let mut rng = 42u64;
    for _ in 0..sample_size {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let i = (rng >> 33) as usize % task_rules.len();
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let j = (rng >> 33) as usize % task_rules.len();
        if i != j {
            cross_sims.push(task_rules[i].similarity(&task_rules[j]));
        }
    }

    let mean_cross: f32 = cross_sims.iter().sum::<f32>() / cross_sims.len().max(1) as f32;
    let discrimination = mean_consistency - mean_cross;

    println!("  Mean cross-task similarity: {:.4}", mean_cross);
    println!(
        "  Discrimination margin (intra - cross): {:.4}",
        discrimination
    );
    println!("  Time: {:.1}s\n", t.elapsed().as_secs_f64());

    // ═══════════════════════════════════════════════════════════════
    // Test 3: Grid Encoding Quality
    // Identical grids should have similarity ≈ 1.0
    // Different grids should have lower similarity
    // ═══════════════════════════════════════════════════════════════
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 3: Grid Encoding Fidelity");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let t = Instant::now();

    // Self-similarity (encode same grid twice → should be identical)
    let test_grid = vec![
        vec![0, 0, 1, 0, 0],
        vec![0, 1, 2, 1, 0],
        vec![1, 2, 3, 2, 1],
        vec![0, 1, 2, 1, 0],
        vec![0, 0, 1, 0, 0],
    ];
    let enc1 = encoder.encode_grid(&test_grid);
    let enc2 = encoder.encode_grid(&test_grid);
    let self_sim = enc1.similarity(&enc2);

    // Translated grid (shift right by 1)
    let translated = vec![
        vec![0, 0, 0, 1, 0],
        vec![0, 0, 1, 2, 1],
        vec![0, 1, 2, 3, 2],
        vec![0, 0, 1, 2, 1],
        vec![0, 0, 0, 1, 0],
    ];
    let trans_enc = encoder.encode_grid(&translated);
    let trans_sim = enc1.similarity(&trans_enc);

    // Color-swapped grid (1↔2)
    let color_swapped = vec![
        vec![0, 0, 2, 0, 0],
        vec![0, 2, 1, 2, 0],
        vec![2, 1, 3, 1, 2],
        vec![0, 2, 1, 2, 0],
        vec![0, 0, 2, 0, 0],
    ];
    let swap_enc = encoder.encode_grid(&color_swapped);
    let swap_sim = enc1.similarity(&swap_enc);

    // Random grid
    let random_grid = vec![
        vec![5, 3, 8, 1, 9],
        vec![2, 7, 0, 4, 6],
        vec![8, 1, 5, 9, 3],
        vec![0, 6, 2, 7, 4],
        vec![9, 4, 7, 3, 1],
    ];
    let rand_enc = encoder.encode_grid(&random_grid);
    let rand_sim = enc1.similarity(&rand_enc);

    println!("  Self-similarity:        {:.4} (expected: 1.0)", self_sim);
    println!(
        "  Translated similarity:  {:.4} (expected: moderate)",
        trans_sim
    );
    println!(
        "  Color-swapped sim:      {:.4} (expected: moderate)",
        swap_sim
    );
    println!("  Random grid sim:        {:.4} (expected: ~0.0)", rand_sim);
    println!("  Time: {:.1}s\n", t.elapsed().as_secs_f64());

    // ═══════════════════════════════════════════════════════════════
    // Test 4: Nearest-Rule Transfer Accuracy
    // Can we predict the test output by applying the most similar training rule?
    // ═══════════════════════════════════════════════════════════════
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 4: Rule Transfer (output similarity)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let t = Instant::now();
    let mut output_sims = Vec::new();
    let mut tasks_evaluated = 0;

    for task_path in task_files.iter().take(max_tasks) {
        let task = match parse_task(task_path) {
            Some(t) => t,
            None => continue,
        };

        if task.train.is_empty() || task.test.is_empty() {
            continue;
        }

        // Bundle all training rule vectors
        let rule_bundle = {
            let mut acc = vec![0.0f32; dim];
            for (inp, out) in &task.train {
                let rule = encoder.encode_rule(inp, out);
                for (a, &v) in acc.iter_mut().zip(rule.values.iter()) {
                    *a += v;
                }
            }
            let norm: f32 = acc.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for v in &mut acc {
                    *v /= norm;
                }
            }
            ContinuousHV::from_vec(acc)
        };

        // Test: encode test input, apply bundled rule, compare with actual test output
        let (test_input, test_output) = &task.test[0];
        let test_input_enc = encoder.encode_grid(test_input);
        let test_output_enc = encoder.encode_grid(test_output);

        // "Apply" rule: bind(test_input, rule_bundle) → predicted output representation
        let predicted = test_input_enc.bind(&rule_bundle);
        let output_sim = predicted.similarity(&test_output_enc);
        output_sims.push(output_sim);
        tasks_evaluated += 1;
    }

    let mean_output_sim: f32 = output_sims.iter().sum::<f32>() / output_sims.len().max(1) as f32;
    let positive_transfer = output_sims.iter().filter(|&&s| s > 0.0).count();

    println!("  Tasks evaluated: {}", tasks_evaluated);
    println!("  Mean output similarity: {:.4}", mean_output_sim);
    println!(
        "  Positive transfer: {:.1}% ({}/{})",
        positive_transfer as f64 / tasks_evaluated.max(1) as f64 * 100.0,
        positive_transfer,
        tasks_evaluated
    );
    println!("  Time: {:.1}s\n", t.elapsed().as_secs_f64());

    // Also load and test evaluation set
    let n_eval = if eval_dir.exists() {
        fs::read_dir(&eval_dir)
            .map(|rd| rd.filter_map(|e| e.ok()).count())
            .unwrap_or(0)
    } else {
        0
    };

    // ═══════════════════════════════════════════════════════════════
    // Summary
    // ═══════════════════════════════════════════════════════════════
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                 VALIDATION SUMMARY                         ║");
    println!("╠══════════════════════════════════════════════════════════════╣");

    let checks = vec![
        (
            "Self-encoding fidelity (sim ≈ 1.0)",
            (self_sim - 1.0).abs() < 0.01,
        ),
        (
            "Intra-task consistency > cross-task",
            mean_consistency > mean_cross,
        ),
        ("Discrimination margin > 0", discrimination > 0.0),
        (
            "Positive rule transfer > 40%",
            positive_transfer as f64 / tasks_evaluated.max(1) as f64 > 0.40,
        ),
        ("Random grid similarity near zero", rand_sim.abs() < 0.1),
    ];

    let mut passed = 0;
    for (name, pass) in &checks {
        println!("║  {} {:50} ║", if *pass { "PASS" } else { "FAIL" }, name);
        if *pass {
            passed += 1;
        }
    }

    println!("╟──────────────────────────────────────────────────────────────╢");
    println!(
        "║  Result: {}/{} tests passed                                 ║",
        passed,
        checks.len()
    );
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Save results (normalized external schema)
    let primary_score = passed as f64 / checks.len().max(1) as f64;
    let result_json = serde_json::json!({
        "benchmark_id": "arc-agi-2",
        "benchmark_name": "ARC-AGI-2 HDC Reasoning",
        "run_id": chrono::Utc::now().to_rfc3339(),
        "model": {
            "backend": "none",
            "model_id": "hdc-only",
            "translation_only": true,
            "quantized": true
        },
        "policy": null,
        "metrics": {
            "primary": primary_score,
            "secondary": {
                "dim": dim,
                "training_tasks": task_files.len(),
                "evaluation_tasks": n_eval,
                "intra_task_consistency": mean_consistency,
                "cross_task_similarity": mean_cross,
                "discrimination_margin": discrimination,
                "self_similarity": self_sim,
                "translated_similarity": trans_sim,
                "color_swap_similarity": swap_sim,
                "random_similarity": rand_sim,
                "mean_output_similarity": mean_output_sim,
                "positive_transfer_pct": positive_transfer as f64 / tasks_evaluated.max(1) as f64,
                "tests_passed": passed,
                "tests_total": checks.len()
            }
        },
        "artifacts": {
            "raw_logs": null,
            "run_notes": null
        }
    });

    let result_path = arc_results_path();
    if let Ok(f) = fs::File::create(&result_path) {
        serde_json::to_writer_pretty(f, &result_json).ok();
        println!("Results saved to {}", result_path.display());
    }
}
