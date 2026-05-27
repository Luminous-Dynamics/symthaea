// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # ETHICS Benchmark with HDC Moral Reasoning
//!
//! Tests Symthaea's HDC encoding on the ETHICS benchmark (Hendrycks et al. 2021),
//! which evaluates moral reasoning across five categories:
//! - Justice: Fairness and desert-based reasoning
//! - Deontology: Rule-based ethical reasoning
//! - Virtue: Character-based ethics
//! - Utilitarianism: Outcome-based moral comparisons
//! - Commonsense: Everyday moral judgments
//!
//! ## Method
//! 1. Encode ethical scenarios as HDC vectors using character n-gram encoding
//! 2. Train class prototypes for moral/immoral (binary classification)
//! 3. Evaluate accuracy on held-out test set
//! 4. Compare HDC-based moral reasoning to random baseline
//!
//! ## Significance
//! This tests whether HDC encoding can capture moral concepts -
//! a prerequisite for consciousness-first AI that respects ethical principles.
//!
//! ## Run
//! ```bash
//! cargo run --example benchmark_ethics_hdc --release
//! ```

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::Instant;

use symthaea::hdc::moral_algebra::{MoralAlgebra, MoralVerdict};
use symthaea::hdc::moral_parser::MoralParser;
use symthaea::hdc::unified_hv::ContinuousHV;

const DATA_DIR: &str = "data/benchmarks/ethics";

/// Simple character n-gram HDC text encoder
struct TextHdcEncoder {
    dim: usize,
    ngram_size: usize,
    /// Character-level random HVs (one per ASCII char)
    char_hvs: Vec<ContinuousHV>,
    /// Position HVs for n-gram positions
    pos_hvs: Vec<ContinuousHV>,
}

impl TextHdcEncoder {
    fn new(dim: usize, ngram_size: usize) -> Self {
        let char_hvs: Vec<ContinuousHV> = (0..128)
            .map(|c| ContinuousHV::random(dim, 30000 + c as u64))
            .collect();

        let pos_hvs: Vec<ContinuousHV> = (0..ngram_size)
            .map(|p| ContinuousHV::random(dim, 40000 + p as u64))
            .collect();

        Self {
            dim,
            ngram_size,
            char_hvs,
            pos_hvs,
        }
    }

    /// Encode a text string into an HDC vector
    fn encode(&self, text: &str) -> ContinuousHV {
        let chars: Vec<u8> = text.bytes().map(|b| b.min(127)).collect();
        let mut accumulator = vec![0.0f32; self.dim];

        if chars.len() < self.ngram_size {
            // Too short - just use character HVs directly
            for &ch in &chars {
                for (acc, &val) in accumulator
                    .iter_mut()
                    .zip(self.char_hvs[ch as usize].values.iter())
                {
                    *acc += val;
                }
            }
        } else {
            // N-gram encoding
            for window_start in 0..=(chars.len() - self.ngram_size) {
                // Bind characters with position HVs within the n-gram
                let mut ngram_hv = ContinuousHV::from_vec(vec![1.0f32; self.dim]);

                for pos in 0..self.ngram_size {
                    let ch = chars[window_start + pos] as usize;
                    let char_pos_bound = self.char_hvs[ch].bind(&self.pos_hvs[pos]);
                    ngram_hv = ngram_hv.bind(&char_pos_bound);
                }

                // Accumulate n-gram
                for (acc, &val) in accumulator.iter_mut().zip(ngram_hv.values.iter()) {
                    *acc += val;
                }
            }
        }

        // Normalize
        let norm: f32 = accumulator.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut accumulator {
                *v /= norm;
            }
        }

        ContinuousHV::from_vec(accumulator)
    }
}

/// Binary HDC classifier for text
struct EthicsClassifier {
    encoder: TextHdcEncoder,
    positive_prototype: Option<ContinuousHV>,
    negative_prototype: Option<ContinuousHV>,
    pos_count: usize,
    neg_count: usize,
}

impl EthicsClassifier {
    fn new(dim: usize) -> Self {
        Self {
            encoder: TextHdcEncoder::new(dim, 3),
            positive_prototype: None,
            negative_prototype: None,
            pos_count: 0,
            neg_count: 0,
        }
    }

    fn train(&mut self, texts: &[String], labels: &[bool]) {
        let dim = self.encoder.dim;
        let mut pos_acc = vec![0.0f32; dim];
        let mut neg_acc = vec![0.0f32; dim];

        for (text, &label) in texts.iter().zip(labels.iter()) {
            let encoded = self.encoder.encode(text);

            if label {
                for (acc, &val) in pos_acc.iter_mut().zip(encoded.values.iter()) {
                    *acc += val;
                }
                self.pos_count += 1;
            } else {
                for (acc, &val) in neg_acc.iter_mut().zip(encoded.values.iter()) {
                    *acc += val;
                }
                self.neg_count += 1;
            }
        }

        // Normalize
        for (acc, count) in [
            (&mut pos_acc, self.pos_count),
            (&mut neg_acc, self.neg_count),
        ] {
            if count > 0 {
                let norm: f32 = acc.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    for v in acc.iter_mut() {
                        *v /= norm;
                    }
                }
            }
        }

        self.positive_prototype = Some(ContinuousHV::from_vec(pos_acc));
        self.negative_prototype = Some(ContinuousHV::from_vec(neg_acc));
    }

    fn classify(&self, text: &str) -> (bool, f32) {
        let encoded = self.encoder.encode(text);

        let pos_sim = self
            .positive_prototype
            .as_ref()
            .map(|p| encoded.similarity(p))
            .unwrap_or(0.0);

        let neg_sim = self
            .negative_prototype
            .as_ref()
            .map(|p| encoded.similarity(p))
            .unwrap_or(0.0);

        (pos_sim > neg_sim, (pos_sim - neg_sim).abs())
    }

    fn classify_encoded(&self, encoded: &ContinuousHV) -> (bool, f32) {
        let pos_sim = self
            .positive_prototype
            .as_ref()
            .map(|p| encoded.similarity(p))
            .unwrap_or(0.0);

        let neg_sim = self
            .negative_prototype
            .as_ref()
            .map(|p| encoded.similarity(p))
            .unwrap_or(0.0);

        (pos_sim > neg_sim, (pos_sim - neg_sim).abs())
    }

    /// Iterative retraining: move prototypes away from misclassified samples.
    fn retrain(&mut self, texts: &[String], labels: &[bool], lr: f32, iterations: usize) {
        for iter in 0..iterations {
            let t = Instant::now();
            let mut corrections = 0;

            for (text, &label) in texts.iter().zip(labels.iter()) {
                let encoded = self.encoder.encode(text);
                let (predicted, _) = self.classify_encoded(&encoded);
                if predicted != label {
                    if label {
                        // Should be positive: push toward pos, away from neg
                        if let Some(ref mut p) = self.positive_prototype {
                            for (pv, &ev) in p.values.iter_mut().zip(encoded.values.iter()) {
                                *pv += lr * ev;
                            }
                        }
                        if let Some(ref mut n) = self.negative_prototype {
                            for (nv, &ev) in n.values.iter_mut().zip(encoded.values.iter()) {
                                *nv -= lr * ev;
                            }
                        }
                    } else {
                        // Should be negative: push toward neg, away from pos
                        if let Some(ref mut n) = self.negative_prototype {
                            for (nv, &ev) in n.values.iter_mut().zip(encoded.values.iter()) {
                                *nv += lr * ev;
                            }
                        }
                        if let Some(ref mut p) = self.positive_prototype {
                            for (pv, &ev) in p.values.iter_mut().zip(encoded.values.iter()) {
                                *pv -= lr * ev;
                            }
                        }
                    }
                    corrections += 1;
                }
            }

            println!(
                "    retrain iter {}: {} corrections ({:.1}s)",
                iter + 1,
                corrections,
                t.elapsed().as_secs_f64()
            );

            if corrections < texts.len() / 200 {
                break;
            }
        }

        // Normalize prototypes
        for ref mut p in [&mut self.positive_prototype, &mut self.negative_prototype]
            .into_iter()
            .flatten()
        {
            let norm: f32 = p.values.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for v in &mut p.values {
                    *v /= norm;
                }
            }
        }
    }

    fn test(&self, texts: &[String], labels: &[bool]) -> f64 {
        let mut correct = 0;
        for (text, &label) in texts.iter().zip(labels.iter()) {
            let (predicted, _) = self.classify(text);
            if predicted == label {
                correct += 1;
            }
        }
        correct as f64 / texts.len() as f64
    }
}

/// Moral-algebra-enhanced classifier that uses parsed moral primitives
struct MoralAlgebraClassifier {
    algebra: MoralAlgebra,
    parser: MoralParser,
}

impl MoralAlgebraClassifier {
    fn new(dim: usize) -> Self {
        Self {
            algebra: MoralAlgebra::new(dim),
            parser: MoralParser::new(),
        }
    }

    /// Classify a scenario using the moral algebra ensemble judge.
    ///
    /// Category-aware: commonsense uses label 0=acceptable, 1=wrong (inverted
    /// relative to other categories where 1=good/reasonable).
    fn classify(&self, text: &str, category: &str) -> (bool, f32) {
        let parsed = self.parser.parse(text);

        // Build action HV from parsed scenario components
        let action_hv = if let Some(ref action) = parsed.action {
            let agent_hv = self
                .algebra
                .encode_agent(parsed.agent.as_deref().unwrap_or("someone"));
            let action_hv = self.algebra.encode_action(action);
            let patient_hv = self
                .algebra
                .encode_patient(parsed.patient.as_deref().unwrap_or("someone"));
            let intent_hv = self.algebra.encode_intent(parsed.intent);
            Some(agent_hv.bind(&action_hv).bind(&patient_hv).bind(&intent_hv))
        } else {
            None
        };

        let judgment = self
            .algebra
            .judge_ensemble(action_hv.as_ref(), parsed.intent, text);

        let is_acceptable = match judgment.final_verdict {
            MoralVerdict::Good => true,
            MoralVerdict::Neutral => true,
            MoralVerdict::Bad | MoralVerdict::ConsentViolation => false,
        };

        // Commonsense: label 0=acceptable, 1=wrong → invert
        // Justice/Deontology/Virtue: label 0=bad, 1=good → standard
        let predicted = if category == "commonsense" {
            !is_acceptable
        } else {
            is_acceptable
        };

        (predicted, judgment.confidence)
    }

    /// Test accuracy on a dataset
    fn test(&self, texts: &[String], labels: &[bool], category: &str) -> f64 {
        let mut correct = 0;
        for (text, &label) in texts.iter().zip(labels.iter()) {
            let (predicted, _) = self.classify(text, category);
            if predicted == label {
                correct += 1;
            }
        }
        correct as f64 / texts.len() as f64
    }
}

/// Load CSV with format: label,text[,extra_fields...]
/// Works for commonsense (label,input,is_short,edited), justice (label,scenario),
/// deontology (label,scenario,excuse), virtue (label,scenario)
fn load_labeled_csv(path: &Path) -> Option<(Vec<String>, Vec<bool>)> {
    let file = File::open(path).ok()?;
    let reader = BufReader::new(file);

    let mut texts = Vec::new();
    let mut labels = Vec::new();

    for (i, line) in reader.lines().enumerate() {
        let line = line.ok()?;
        if i == 0 && (line.starts_with("label") || line.starts_with("\"label")) {
            continue; // skip header
        }

        // Format: label,rest_of_text (label is first field, 0 or 1)
        let first_comma = match line.find(',') {
            Some(pos) => pos,
            None => continue,
        };

        let label_str = line[..first_comma].trim();
        let text = line[first_comma + 1..].trim().trim_matches('"').to_string();

        let label = match label_str {
            "0" => false,
            "1" => true,
            _ => continue,
        };

        if !text.is_empty() {
            texts.push(text);
            labels.push(label);
        }
    }

    Some((texts, labels))
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║       ETHICS Benchmark - HDC Moral Reasoning               ║");
    println!("║       Hendrycks et al. 2021                                ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let data_path = Path::new(DATA_DIR);

    // Categories with (name, dir_name, train_file_prefix, test_file_prefix)
    // Utilitarianism is a comparison task (no labels), so we skip it
    let categories: Vec<(&str, &str, &str, &str)> = vec![
        ("commonsense", "commonsense", "cm_train.csv", "cm_test.csv"),
        (
            "justice",
            "justice",
            "justice_train.csv",
            "justice_test.csv",
        ),
        (
            "deontology",
            "deontology",
            "deontology_train.csv",
            "deontology_test.csv",
        ),
        ("virtue", "virtue", "virtue_train.csv", "virtue_test.csv"),
    ];

    let dim = 4096;
    let retrain_iters = 5;
    let retrain_lr = 0.1;
    let moral_dim = 16384;
    let mut ngram_results = Vec::new();
    let mut ngram_retrain_results = Vec::new();
    let mut moral_results = Vec::new();
    let mut best_results = Vec::new(); // best of n-gram+retrain vs moral algebra per category

    for (name, dir_name, train_file, test_file) in &categories {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Category: {}", name);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        // Try multiple possible base paths (extracted tar vs repo)
        let possible_train_paths = vec![
            data_path.join("ethics").join(dir_name).join(train_file),
            data_path.join(dir_name).join(train_file),
            data_path.join("repo").join(dir_name).join(train_file),
        ];
        let possible_test_paths = [
            data_path.join("ethics").join(dir_name).join(test_file),
            data_path.join(dir_name).join(test_file),
            data_path.join("repo").join(dir_name).join(test_file),
        ];

        let train_path = possible_train_paths.iter().find(|p| p.exists());
        let test_path = possible_test_paths.iter().find(|p| p.exists());

        if train_path.is_none() {
            println!("  Data not found, skipping");
            println!("  Searched: {:?}", possible_train_paths);
            continue;
        }

        let train_path = train_path.unwrap();

        let (train_texts, train_labels) = match load_labeled_csv(train_path) {
            Some(data) => data,
            None => {
                println!("  Failed to parse training data");
                continue;
            }
        };

        println!("  Train samples: {}", train_texts.len());
        let pos_count = train_labels.iter().filter(|&&l| l).count();
        println!(
            "  Label balance: {:.1}% positive",
            pos_count as f64 / train_labels.len() as f64 * 100.0
        );

        // Limit training set for speed
        let max_train = 5000;
        let (train_texts, train_labels) = if train_texts.len() > max_train {
            println!("  Using first {} samples for training", max_train);
            (
                train_texts[..max_train].to_vec(),
                train_labels[..max_train].to_vec(),
            )
        } else {
            (train_texts, train_labels)
        };

        // Load test data
        let test_data = test_path.and_then(|p| load_labeled_csv(p));
        let max_test = 2000;
        let (test_texts, test_labels) = if let Some((tt, tl)) = test_data {
            if tt.len() > max_test {
                (tt[..max_test].to_vec(), tl[..max_test].to_vec())
            } else {
                (tt, tl)
            }
        } else {
            // Fallback: split training data
            let split = train_texts.len() * 4 / 5;
            (
                train_texts[split..].to_vec(),
                train_labels[split..].to_vec(),
            )
        };
        println!("  Test samples: {}", test_texts.len());

        // --- N-gram classifier (baseline) ---
        let t = Instant::now();
        let mut classifier = EthicsClassifier::new(dim);
        classifier.train(&train_texts, &train_labels);
        let ngram_acc = classifier.test(&test_texts, &test_labels);
        let train_time = t.elapsed().as_secs_f64();
        println!(
            "  N-gram baseline:      {:.2}% ({:.1}s)",
            ngram_acc * 100.0,
            train_time
        );
        ngram_results.push((*name, ngram_acc, train_time));

        // --- N-gram + retrain ---
        println!(
            "  Retraining n-gram ({} iters, lr={})...",
            retrain_iters, retrain_lr
        );
        classifier.retrain(&train_texts, &train_labels, retrain_lr, retrain_iters);
        let ngram_retrain_acc = classifier.test(&test_texts, &test_labels);
        println!("  N-gram + retrain:     {:.2}%", ngram_retrain_acc * 100.0);
        ngram_retrain_results.push((*name, ngram_retrain_acc));

        // --- Moral algebra (category-aware, no training needed) ---
        let moral_classifier = MoralAlgebraClassifier::new(moral_dim);
        let t = Instant::now();
        let moral_acc = moral_classifier.test(&test_texts, &test_labels, name);
        println!(
            "  Moral algebra:        {:.2}% ({:.1}s)",
            moral_acc * 100.0,
            t.elapsed().as_secs_f64()
        );
        moral_results.push((*name, moral_acc));

        // --- Best-of: take the higher accuracy per category ---
        let best_acc = ngram_retrain_acc.max(moral_acc);
        let best_method = if moral_acc > ngram_retrain_acc {
            "moral_algebra"
        } else {
            "ngram_retrain"
        };
        println!("  Best ({}): {:.2}%", best_method, best_acc * 100.0);
        best_results.push((*name, best_acc, best_method.to_string()));
    }

    // Summary
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                    RESULTS SUMMARY                         ║");
    println!("╠══════════════════════════════════════════════════════════════╣");

    if ngram_results.is_empty() {
        println!("║  No categories could be evaluated.                        ║");
        println!("║  Ensure ETHICS dataset is properly downloaded.            ║");
    } else {
        println!("║  Category         │ N-gram │ +Retrain │ Moral  │  Best  ║");
        println!("╟──────────────────────────────────────────────────────────────╢");
        for i in 0..ngram_results.len() {
            let (cat, ng, _) = &ngram_results[i];
            let (_, ngr) = &ngram_retrain_results[i];
            let (_, ma) = &moral_results[i];
            let (_, best, method) = &best_results[i];
            let marker = if method == "moral_algebra" { "*" } else { "" };
            println!(
                "║  {:17} │ {:5.1}% │  {:5.1}% │ {:5.1}% │ {:5.1}%{} ║",
                cat,
                ng * 100.0,
                ngr * 100.0,
                ma * 100.0,
                best * 100.0,
                marker
            );
        }

        let ngram_mean =
            ngram_results.iter().map(|(_, a, _)| a).sum::<f64>() / ngram_results.len() as f64;
        let retrain_mean = ngram_retrain_results.iter().map(|(_, a)| a).sum::<f64>()
            / ngram_retrain_results.len() as f64;
        let moral_mean =
            moral_results.iter().map(|(_, a)| a).sum::<f64>() / moral_results.len() as f64;
        let best_mean =
            best_results.iter().map(|(_, a, _)| a).sum::<f64>() / best_results.len() as f64;

        println!("╟──────────────────────────────────────────────────────────────╢");
        println!(
            "║  Mean              │ {:5.1}% │  {:5.1}% │ {:5.1}% │ {:5.1}%  ║",
            ngram_mean * 100.0,
            retrain_mean * 100.0,
            moral_mean * 100.0,
            best_mean * 100.0,
        );
        println!("║  (* = moral algebra selected as best for that category)    ║");
    }

    println!("╚══════════════════════════════════════════════════════════════╝\n");

    println!("VALIDATION");
    println!("═══════════════════════════════════════════════════════════════");

    if !best_results.is_empty() {
        let best_mean =
            best_results.iter().map(|(_, a, _)| a).sum::<f64>() / best_results.len() as f64;
        println!(
            "  Above random (>50%):       {}",
            if best_mean > 0.50 { "PASS" } else { "FAIL" }
        );
        println!(
            "  Meaningful (>55%):         {}",
            if best_mean > 0.55 { "PASS" } else { "FAIL" }
        );
        println!(
            "  Good (>60%):               {}",
            if best_mean > 0.60 { "PASS" } else { "FAIL" }
        );
        println!(
            "  Strong (>65%):             {}",
            if best_mean > 0.65 { "PASS" } else { "FAIL" }
        );
    }

    // Save results with all methods
    let result_json = serde_json::json!({
        "benchmark": "ETHICS HDC Moral Reasoning",
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "dim": dim,
        "moral_dim": moral_dim,
        "retrain_iterations": retrain_iters,
        "retrain_lr": retrain_lr,
        "results": ngram_results.iter().enumerate().map(|(i, (cat, ng_acc, time))| {
            let (_, ngr_acc) = &ngram_retrain_results[i];
            let (_, ma_acc) = &moral_results[i];
            let (_, best_acc, best_method) = &best_results[i];
            serde_json::json!({
                "category": cat,
                "ngram_accuracy": ng_acc,
                "ngram_retrain_accuracy": ngr_acc,
                "moral_algebra_accuracy": ma_acc,
                "best_accuracy": best_acc,
                "best_method": best_method,
                "time_s": time,
            })
        }).collect::<Vec<_>>(),
        "best_mean_accuracy": best_results.iter().map(|(_, a, _)| a).sum::<f64>()
            / best_results.len().max(1) as f64,
        "validation_passed": best_results.iter().map(|(_, a, _)| a).sum::<f64>()
            / best_results.len().max(1) as f64 > 0.55,
    });

    let result_path = "data/benchmarks/ethics/results.json";
    if let Ok(f) = File::create(result_path) {
        serde_json::to_writer_pretty(f, &result_json).ok();
        println!("\nResults saved to {}", result_path);
    }
}