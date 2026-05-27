// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # GSM8K Benchmark (Grade School Math 8K)
//!
//! Tests Symthaea's HDC numerical reasoning on the GSM8K dataset (Cobbe et al. 2021),
//! which contains 1,319 grade-school math word problems with step-by-step solutions.
//!
//! ## Method
//! 1. Extract all numbers from each question via regex
//! 2. Detect arithmetic operation keywords (total, each, per, times, etc.)
//! 3. Encode numbers as BinaryHV vectors using thermometer coding
//! 4. Attempt a simple arithmetic chain based on detected operations
//! 5. Compare computed answer to gold answer (within tolerance)
//!
//! ## Honest Assessment
//! GSM8K is designed for chain-of-thought LLM reasoning. HDC pattern-matching
//! over extracted numbers and operation keywords is a deliberately naive baseline.
//! The value here is measuring what fraction of grade-school math can be captured
//! by pure pattern-matching without language understanding, and validating that
//! HDC number encoding preserves arithmetic relationships.
//!
//! ## Dataset
//! GSM8K (Cobbe et al. 2021): 1,319 test problems
//! Format: JSONL with `question` and `answer` fields
//! Gold answer extracted from `#### <number>` suffix in the answer field
//!
//! ## Run
//! ```bash
//! cargo run --example benchmark_gsm8k --release
//! ```

use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use symthaea::hdc::binary_hv::BinaryHV;

// ─────────────────────────────────────────────────────────────────────
// Paths
// ─────────────────────────────────────────────────────────────────────

fn gsm8k_data_path() -> PathBuf {
    env::var("SYMTHAEA_GSM8K_DATA_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("data/benchmarks/gsm8k/test.jsonl"))
}

fn gsm8k_results_path() -> PathBuf {
    env::var("SYMTHAEA_GSM8K_RESULTS_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("data/benchmarks/gsm8k/results.json"))
}

// ─────────────────────────────────────────────────────────────────────
// Data structures
// ─────────────────────────────────────────────────────────────────────

struct Gsm8kProblem {
    question: String,
    gold_answer: f64,
    /// Raw answer text (for debugging)
    _answer_text: String,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ArithOp {
    Add,
    Subtract,
    Multiply,
    Divide,
}

struct SolveAttempt {
    /// Numbers extracted from the question
    numbers: Vec<f64>,
    /// Operations detected from keywords
    operations: Vec<ArithOp>,
    /// Computed answer (if any)
    answer: Option<f64>,
    /// Whether we consider this "attempted" (had numbers + operations)
    attempted: bool,
    /// Whether our answer matched gold (within tolerance)
    correct: bool,
}

// ─────────────────────────────────────────────────────────────────────
// Parsing
// ─────────────────────────────────────────────────────────────────────

/// Extract gold numeric answer from the answer field.
/// GSM8K format: "Step 1...\nStep 2...\n#### 72"
fn extract_gold_answer(answer_text: &str) -> Option<f64> {
    let marker = "####";
    let idx = answer_text.rfind(marker)?;
    let after = &answer_text[idx + marker.len()..];
    // Strip whitespace and commas from the number
    let cleaned: String = after.trim().chars().filter(|c| *c != ',').collect();
    cleaned.parse::<f64>().ok()
}

/// Extract all numbers from question text.
/// Handles integers, decimals, and numbers with commas (e.g., "1,200").
fn extract_numbers(text: &str) -> Vec<f64> {
    let mut numbers = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        // Check if we're at the start of a number (digit, or negative sign before digit)
        let is_negative = chars[i] == '-' && i + 1 < len && chars[i + 1].is_ascii_digit();
        let starts_digit = chars[i].is_ascii_digit();

        if starts_digit || is_negative {
            // Don't capture numbers that are part of a word (e.g., "GSM8K")
            if i > 0 && chars[i - 1].is_ascii_alphabetic() {
                i += 1;
                continue;
            }

            let start = i;
            if is_negative {
                i += 1;
            }
            // Consume digits, commas between digits, and decimal points
            while i < len
                && (chars[i].is_ascii_digit()
                    || chars[i] == '.'
                    || (chars[i] == ',' && i + 1 < len && chars[i + 1].is_ascii_digit()))
            {
                i += 1;
            }

            // Don't include trailing periods (end of sentence)
            let mut end = i;
            if end > start && chars[end - 1] == '.' {
                end -= 1;
            }

            let num_str: String = chars[start..end].iter().filter(|c| **c != ',').collect();

            if let Ok(n) = num_str.parse::<f64>() {
                // Skip numbers that are likely not operands (years, etc.)
                // Keep all for now; filtering is done at solve time
                numbers.push(n);
            }
        } else {
            i += 1;
        }
    }

    numbers
}

/// Detect arithmetic operations from keyword patterns in the question.
fn detect_operations(text: &str) -> Vec<ArithOp> {
    let lower = text.to_lowercase();
    let mut ops = Vec::new();

    // Multiplication keywords
    let mul_keywords = [
        "times",
        "multiplied by",
        "each",
        "per",
        "every",
        "for each",
        "at a rate",
        "costs",
        "price of",
        "rate of",
    ];
    for kw in &mul_keywords {
        if lower.contains(kw) {
            ops.push(ArithOp::Multiply);
            break;
        }
    }

    // Addition keywords
    let add_keywords = [
        "total",
        "altogether",
        "combined",
        "sum",
        "in all",
        "plus",
        "more than",
        "added",
        "increased by",
        "additional",
        "extra",
        "and",
    ];
    for kw in &add_keywords {
        if lower.contains(kw) {
            ops.push(ArithOp::Add);
            break;
        }
    }

    // Subtraction keywords
    let sub_keywords = [
        "left",
        "remaining",
        "minus",
        "fewer",
        "less than",
        "difference",
        "reduced by",
        "gave away",
        "lost",
        "spent",
        "sold",
        "removed",
        "after giving",
    ];
    for kw in &sub_keywords {
        if lower.contains(kw) {
            ops.push(ArithOp::Subtract);
            break;
        }
    }

    // Division keywords
    let div_keywords = [
        "divided",
        "split",
        "shared equally",
        "per person",
        "average",
        "half",
        "third",
        "quarter",
        "ratio",
        "out of",
    ];
    for kw in &div_keywords {
        if lower.contains(kw) {
            ops.push(ArithOp::Divide);
            break;
        }
    }

    ops
}

/// Attempt to solve a problem by applying detected operations to extracted numbers.
///
/// Strategy: try a simple left-fold of all operations over the numbers.
/// For single-operation questions (e.g., "5 times 3"), this often works.
/// For multi-step problems, this is a naive heuristic.
fn attempt_solve(numbers: &[f64], operations: &[ArithOp]) -> Option<f64> {
    if numbers.is_empty() {
        return None;
    }
    if numbers.len() == 1 {
        return Some(numbers[0]);
    }
    if operations.is_empty() {
        return None;
    }

    // Strategy 1: If exactly one operation and two numbers, apply directly
    if numbers.len() == 2 && operations.len() == 1 {
        return apply_op(numbers[0], numbers[1], operations[0]);
    }

    // Strategy 2: For multiply+add patterns (very common in GSM8K):
    // e.g., "3 boxes of 12 apples each, plus 5 loose" → 3*12 + 5
    if operations.contains(&ArithOp::Multiply) && operations.contains(&ArithOp::Add) {
        if numbers.len() >= 3 {
            // Try: first two multiply, then add rest
            if let Some(prod) = apply_op(numbers[0], numbers[1], ArithOp::Multiply) {
                let sum: f64 = numbers[2..].iter().sum();
                return Some(prod + sum);
            }
        }
    }

    // Strategy 3: For multiply+subtract patterns:
    // e.g., "had 50, gave away 3 groups of 5" → 50 - 3*5
    if operations.contains(&ArithOp::Multiply) && operations.contains(&ArithOp::Subtract) {
        if numbers.len() >= 3 {
            if let Some(prod) = apply_op(numbers[1], numbers[2], ArithOp::Multiply) {
                return apply_op(numbers[0], prod, ArithOp::Subtract);
            }
        }
    }

    // Strategy 4: Single operation, fold over all numbers
    if operations.len() == 1 {
        let op = operations[0];
        let mut result = numbers[0];
        for &n in &numbers[1..] {
            result = apply_op(result, n, op)?;
        }
        return Some(result);
    }

    // Strategy 5: Two numbers, pick the first detected operation
    if numbers.len() == 2 {
        return apply_op(numbers[0], numbers[1], operations[0]);
    }

    // Fallback: fold with first operation
    let op = operations[0];
    let mut result = numbers[0];
    for &n in &numbers[1..] {
        result = apply_op(result, n, op)?;
    }
    Some(result)
}

fn apply_op(a: f64, b: f64, op: ArithOp) -> Option<f64> {
    match op {
        ArithOp::Add => Some(a + b),
        ArithOp::Subtract => Some(a - b),
        ArithOp::Multiply => Some(a * b),
        ArithOp::Divide => {
            if b.abs() < 1e-10 {
                None
            } else {
                Some(a / b)
            }
        }
    }
}

/// Check if two answers are "close enough".
/// Tolerance: exact integer match, or within 1% for large numbers.
fn answers_match(computed: f64, gold: f64) -> bool {
    if (computed - gold).abs() < 0.5 {
        return true;
    }
    if gold.abs() > 1.0 && ((computed - gold) / gold).abs() < 0.01 {
        return true;
    }
    false
}

// ─────────────────────────────────────────────────────────────────────
// HDC Number Encoding Validation
// ─────────────────────────────────────────────────────────────────────

/// Encode a number as a BinaryHV using thermometer coding.
///
/// Thermometer coding preserves ordinality: nearby numbers produce similar vectors.
/// We use a hash-based encoding where the number of set basis vectors is proportional
/// to the magnitude, creating a monotonic similarity gradient.
fn encode_number_hv(value: f64, scale: f64) -> BinaryHV {
    // Normalize to [0, 1] range using the scale
    let normalized = (value / scale).clamp(0.0, 1.0);

    // Thermometer: bundle the first N basis vectors where N = normalized * 128
    let n_active = (normalized * 128.0).round() as usize;
    if n_active == 0 {
        return BinaryHV::zero();
    }

    let vectors: Vec<BinaryHV> = (0..n_active)
        .map(|i| BinaryHV::basis(i + 500_000)) // offset to avoid collision with other uses
        .collect();

    BinaryHV::bundle(&vectors)
}

/// Validate that HDC number encoding preserves arithmetic relationships.
fn validate_hdc_number_encoding() -> (f64, f64, f64) {
    let scale = 100.0;

    // Test 1: Ordinality — nearby numbers should be more similar
    let mut ordinality_passes = 0;
    let n_tests = 20;
    for i in 0..n_tests {
        let base = (i as f64 + 1.0) * 4.0;
        let near = base + 2.0;
        let far = base + 40.0;

        let hv_base = encode_number_hv(base, scale);
        let hv_near = encode_number_hv(near, scale);
        let hv_far = encode_number_hv(far.min(scale), scale);

        let sim_near = hv_base.similarity(&hv_near);
        let sim_far = hv_base.similarity(&hv_far);

        if sim_near > sim_far {
            ordinality_passes += 1;
        }
    }
    let ordinality_score = ordinality_passes as f64 / n_tests as f64;

    // Test 2: Monotonicity — encoding should increase with value
    let mut monotonic_passes = 0;
    for i in 0..n_tests {
        let v1 = (i as f64 + 1.0) * 3.0;
        let v2 = v1 + 5.0;
        let hv1 = encode_number_hv(v1, scale);
        let hv2 = encode_number_hv(v2, scale);
        // Higher value should have more bits set (higher density)
        if hv2.density() >= hv1.density() {
            monotonic_passes += 1;
        }
    }
    let monotonicity_score = monotonic_passes as f64 / n_tests as f64;

    // Test 3: Self-similarity — same number always produces same vector
    let mut self_sim_total = 0.0f64;
    for i in 0..10 {
        let v = (i as f64 + 1.0) * 7.0;
        let hv_a = encode_number_hv(v, scale);
        let hv_b = encode_number_hv(v, scale);
        self_sim_total += hv_a.similarity(&hv_b) as f64;
    }
    let self_sim_score = self_sim_total / 10.0;

    (ordinality_score, monotonicity_score, self_sim_score)
}

// ─────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────

fn main() {
    println!("======================================================================");
    println!("       GSM8K Benchmark (Grade School Math)");
    println!("       HDC Pattern-Matching Arithmetic Baseline");
    println!("======================================================================\n");

    let data_path = gsm8k_data_path();

    if !data_path.exists() {
        eprintln!("GSM8K data not found at {}", data_path.display());
        eprintln!("Set SYMTHAEA_GSM8K_DATA_PATH to override the dataset location.");
        eprintln!(
            "Download: https://github.com/openai/grade-school-math \
             (test.jsonl → data/benchmarks/gsm8k/test.jsonl)"
        );
        return;
    }

    // ═══════════════════════════════════════════════════════════════
    // Load dataset
    // ═══════════════════════════════════════════════════════════════
    println!("Loading dataset from {}\n", data_path.display());

    let content = match fs::read_to_string(&data_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to read {}: {}", data_path.display(), e);
            return;
        }
    };

    let mut problems = Vec::new();
    let mut parse_failures = 0;

    for (line_no, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let parsed: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => {
                parse_failures += 1;
                continue;
            }
        };

        let question = match parsed["question"].as_str() {
            Some(q) => q.to_string(),
            None => {
                parse_failures += 1;
                continue;
            }
        };

        let answer_text = match parsed["answer"].as_str() {
            Some(a) => a.to_string(),
            None => {
                parse_failures += 1;
                continue;
            }
        };

        match extract_gold_answer(&answer_text) {
            Some(gold) => problems.push(Gsm8kProblem {
                question,
                gold_answer: gold,
                _answer_text: answer_text,
            }),
            None => {
                if line_no < 5 {
                    eprintln!(
                        "  Warning: could not extract gold answer from line {}",
                        line_no + 1
                    );
                }
                parse_failures += 1;
            }
        }
    }

    println!("  Loaded:         {} problems", problems.len());
    println!("  Parse failures: {}", parse_failures);
    if problems.is_empty() {
        eprintln!("No valid problems loaded. Check data format.");
        return;
    }
    println!();

    // ═══════════════════════════════════════════════════════════════
    // Test 1: HDC Number Encoding Validation
    // ═══════════════════════════════════════════════════════════════
    println!("------------------------------------------------------------------");
    println!("Test 1: HDC Number Encoding Properties");
    println!("------------------------------------------------------------------");

    let t = Instant::now();
    let (ordinality, monotonicity, self_sim) = validate_hdc_number_encoding();

    println!(
        "  Ordinality (nearby > far similarity): {:.1}%",
        ordinality * 100.0
    );
    println!(
        "  Monotonicity (density increases):     {:.1}%",
        monotonicity * 100.0
    );
    println!("  Self-similarity (deterministic):      {:.4}", self_sim);
    println!("  Time: {:.1}ms\n", t.elapsed().as_secs_f64() * 1000.0);

    // ═══════════════════════════════════════════════════════════════
    // Test 2: Number Extraction Coverage
    // ═══════════════════════════════════════════════════════════════
    println!("------------------------------------------------------------------");
    println!("Test 2: Number Extraction Coverage");
    println!("------------------------------------------------------------------");

    let t = Instant::now();
    let mut questions_with_numbers = 0;
    let mut total_numbers_extracted = 0;
    let mut questions_with_ops = 0;

    for prob in &problems {
        let nums = extract_numbers(&prob.question);
        if !nums.is_empty() {
            questions_with_numbers += 1;
            total_numbers_extracted += nums.len();
        }
        let ops = detect_operations(&prob.question);
        if !ops.is_empty() {
            questions_with_ops += 1;
        }
    }

    let extraction_rate = questions_with_numbers as f64 / problems.len() as f64;
    let op_detection_rate = questions_with_ops as f64 / problems.len() as f64;
    let avg_numbers = total_numbers_extracted as f64 / problems.len() as f64;

    println!(
        "  Questions with numbers: {:.1}% ({}/{})",
        extraction_rate * 100.0,
        questions_with_numbers,
        problems.len()
    );
    println!("  Avg numbers per question: {:.1}", avg_numbers);
    println!(
        "  Questions with ops detected: {:.1}% ({}/{})",
        op_detection_rate * 100.0,
        questions_with_ops,
        problems.len()
    );
    println!("  Time: {:.1}ms\n", t.elapsed().as_secs_f64() * 1000.0);

    // ═══════════════════════════════════════════════════════════════
    // Test 3: Arithmetic Solving
    // ═══════════════════════════════════════════════════════════════
    println!("------------------------------------------------------------------");
    println!("Test 3: Arithmetic Pattern-Matching Solve");
    println!("------------------------------------------------------------------");

    let t = Instant::now();
    let mut attempts: Vec<SolveAttempt> = Vec::with_capacity(problems.len());

    for prob in &problems {
        let numbers = extract_numbers(&prob.question);
        let operations = detect_operations(&prob.question);
        let answer = attempt_solve(&numbers, &operations);
        let attempted = answer.is_some();
        let correct = answer
            .map(|a| answers_match(a, prob.gold_answer))
            .unwrap_or(false);

        attempts.push(SolveAttempt {
            numbers,
            operations,
            answer,
            attempted,
            correct,
        });
    }

    let n_attempted = attempts.iter().filter(|a| a.attempted).count();
    let n_correct = attempts.iter().filter(|a| a.correct).count();
    let attempt_rate = n_attempted as f64 / problems.len() as f64;
    let accuracy_overall = n_correct as f64 / problems.len() as f64;
    let accuracy_attempted = if n_attempted > 0 {
        n_correct as f64 / n_attempted as f64
    } else {
        0.0
    };

    println!("  Total problems:     {}", problems.len());
    println!(
        "  Attempted (had numbers + ops): {:.1}% ({}/{})",
        attempt_rate * 100.0,
        n_attempted,
        problems.len()
    );
    println!(
        "  Correct (overall):  {:.1}% ({}/{})",
        accuracy_overall * 100.0,
        n_correct,
        problems.len()
    );
    println!(
        "  Correct (attempted only): {:.1}% ({}/{})",
        accuracy_attempted * 100.0,
        n_correct,
        n_attempted
    );
    println!("  Time: {:.1}ms\n", t.elapsed().as_secs_f64() * 1000.0);

    // ═══════════════════════════════════════════════════════════════
    // Test 4: Operation-type breakdown
    // ═══════════════════════════════════════════════════════════════
    println!("------------------------------------------------------------------");
    println!("Test 4: Accuracy by Operation Type");
    println!("------------------------------------------------------------------");

    for op in &[
        ArithOp::Add,
        ArithOp::Subtract,
        ArithOp::Multiply,
        ArithOp::Divide,
    ] {
        let relevant: Vec<&SolveAttempt> = attempts
            .iter()
            .filter(|a| a.operations.contains(op))
            .collect();
        let op_attempted = relevant.iter().filter(|a| a.attempted).count();
        let op_correct = relevant.iter().filter(|a| a.correct).count();
        let op_acc = if op_attempted > 0 {
            op_correct as f64 / op_attempted as f64
        } else {
            0.0
        };
        println!(
            "  {:12} — {}/{} correct of {} attempted ({:.1}%)",
            format!("{:?}", op),
            op_correct,
            op_attempted,
            relevant.len(),
            op_acc * 100.0
        );
    }
    println!();

    // ═══════════════════════════════════════════════════════════════
    // Test 5: HDC encoding of gold answers — distribution analysis
    // ═══════════════════════════════════════════════════════════════
    println!("------------------------------------------------------------------");
    println!("Test 5: HDC Gold Answer Encoding Distribution");
    println!("------------------------------------------------------------------");

    let t = Instant::now();

    // Find max gold answer for scaling
    let max_gold = problems
        .iter()
        .map(|p| p.gold_answer.abs())
        .fold(0.0f64, f64::max);
    let scale = if max_gold > 0.0 { max_gold } else { 1.0 };

    // Encode all gold answers and check pairwise similarity structure
    let gold_hvs: Vec<BinaryHV> = problems
        .iter()
        .take(200) // sample for speed
        .map(|p| encode_number_hv(p.gold_answer.abs(), scale))
        .collect();

    // Check that similar gold answers produce similar HVs
    let mut sim_close = 0.0f64;
    let mut sim_far = 0.0f64;
    let mut n_close = 0;
    let mut n_far = 0;

    for i in 0..gold_hvs.len().min(100) {
        for j in (i + 1)..gold_hvs.len().min(100) {
            let val_diff = (problems[i].gold_answer - problems[j].gold_answer).abs() / scale;
            let sim = gold_hvs[i].similarity(&gold_hvs[j]) as f64;
            if val_diff < 0.05 {
                sim_close += sim;
                n_close += 1;
            } else if val_diff > 0.3 {
                sim_far += sim;
                n_far += 1;
            }
        }
    }

    let avg_sim_close = if n_close > 0 {
        sim_close / n_close as f64
    } else {
        0.0
    };
    let avg_sim_far = if n_far > 0 {
        sim_far / n_far as f64
    } else {
        0.0
    };

    println!("  Gold answer range: 0 to {:.0}", max_gold);
    println!(
        "  Close-value pairs (diff < 5%): avg sim = {:.4} (n={})",
        avg_sim_close, n_close
    );
    println!(
        "  Far-value pairs (diff > 30%):  avg sim = {:.4} (n={})",
        avg_sim_far, n_far
    );
    println!(
        "  Encoding preserves ordinality: {}",
        if avg_sim_close > avg_sim_far {
            "YES"
        } else {
            "NO"
        }
    );
    println!("  Time: {:.1}ms\n", t.elapsed().as_secs_f64() * 1000.0);

    // ═══════════════════════════════════════════════════════════════
    // Show sample results (first 5 correct, first 5 incorrect attempted)
    // ═══════════════════════════════════════════════════════════════
    println!("------------------------------------------------------------------");
    println!("Sample Results");
    println!("------------------------------------------------------------------");

    println!("\n  Correct examples:");
    let mut shown = 0;
    for (i, (prob, att)) in problems.iter().zip(attempts.iter()).enumerate() {
        if att.correct && shown < 5 {
            println!(
                "    [{}] gold={} computed={:.0} nums={:?}",
                i,
                prob.gold_answer,
                att.answer.unwrap_or(f64::NAN),
                att.numbers
            );
            shown += 1;
        }
    }
    if shown == 0 {
        println!("    (none)");
    }

    println!("\n  Incorrect attempted examples:");
    shown = 0;
    for (i, (prob, att)) in problems.iter().zip(attempts.iter()).enumerate() {
        if att.attempted && !att.correct && shown < 5 {
            println!(
                "    [{}] gold={} computed={:.0} nums={:?} ops={:?}",
                i,
                prob.gold_answer,
                att.answer.unwrap_or(f64::NAN),
                att.numbers,
                att.operations
            );
            shown += 1;
        }
    }
    if shown == 0 {
        println!("    (none)");
    }
    println!();

    // ═══════════════════════════════════════════════════════════════
    // Validation Summary
    // ═══════════════════════════════════════════════════════════════
    println!("======================================================================");
    println!("                     VALIDATION SUMMARY");
    println!("======================================================================");

    let checks = vec![
        ("Number extraction rate > 90%", extraction_rate > 0.90),
        ("Operation detection rate > 70%", op_detection_rate > 0.70),
        ("HDC ordinality > 80%", ordinality > 0.80),
        ("HDC monotonicity > 90%", monotonicity > 0.90),
        ("HDC self-similarity > 0.99", self_sim > 0.99),
        (
            "HDC encoding preserves ordinality",
            avg_sim_close > avg_sim_far,
        ),
    ];

    let mut passed = 0;
    for (name, pass) in &checks {
        let mark = if *pass { "PASS" } else { "FAIL" };
        println!("  [{}] {}", mark, name);
        if *pass {
            passed += 1;
        }
    }

    println!();
    println!(
        "  Result: {}/{} validation checks passed",
        passed,
        checks.len()
    );
    println!();
    println!("  NOTE: GSM8K requires multi-step chain-of-thought reasoning.");
    println!("  A pattern-matching arithmetic baseline is expected to score low.");
    println!("  The value is in validating HDC number encoding properties and");
    println!("  measuring the ceiling of keyword-based operation detection.");
    println!("  State-of-the-art LLMs score 50-95% on GSM8K; pure pattern-matching");
    println!("  without language understanding is a fundamentally different approach.");
    println!();

    // ═══════════════════════════════════════════════════════════════
    // Save results
    // ═══════════════════════════════════════════════════════════════
    let primary_score = accuracy_overall;
    let result_json = serde_json::json!({
        "benchmark_id": "gsm8k-hdc-arithmetic",
        "benchmark_name": "GSM8K HDC Pattern-Matching Arithmetic",
        "run_id": chrono::Utc::now().to_rfc3339(),
        "model": {
            "backend": "none",
            "model_id": "hdc-pattern-matching",
            "translation_only": true,
            "quantized": true
        },
        "policy": null,
        "dataset": {
            "name": "GSM8K",
            "split": "test",
            "n_problems": problems.len(),
            "parse_failures": parse_failures
        },
        "metrics": {
            "primary": primary_score,
            "secondary": {
                "n_problems": problems.len(),
                "n_attempted": n_attempted,
                "n_correct": n_correct,
                "accuracy_overall": accuracy_overall,
                "accuracy_attempted": accuracy_attempted,
                "attempt_rate": attempt_rate,
                "extraction_rate": extraction_rate,
                "op_detection_rate": op_detection_rate,
                "avg_numbers_per_question": avg_numbers,
                "hdc_ordinality": ordinality,
                "hdc_monotonicity": monotonicity,
                "hdc_self_similarity": self_sim,
                "hdc_close_pair_sim": avg_sim_close,
                "hdc_far_pair_sim": avg_sim_far,
                "validation_checks_passed": passed,
                "validation_checks_total": checks.len()
            }
        },
        "notes": [
            "GSM8K is designed for chain-of-thought LLM reasoning.",
            "This benchmark tests HDC number encoding and keyword-based operation detection.",
            "Low overall accuracy is expected and scientifically honest.",
            "Primary value: validates HDC thermometer encoding preserves numeric ordinality."
        ],
        "artifacts": {
            "raw_logs": null,
            "run_notes": null
        }
    });

    let result_path = gsm8k_results_path();
    if let Some(parent) = result_path.parent() {
        fs::create_dir_all(parent).ok();
    }
    match fs::File::create(&result_path) {
        Ok(f) => {
            serde_json::to_writer_pretty(f, &result_json).ok();
            println!("Results saved to {}", result_path.display());
        }
        Err(e) => {
            eprintln!("Could not save results to {}: {}", result_path.display(), e);
        }
    }
}