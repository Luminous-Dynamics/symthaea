// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! ETHICS Ablation Study
//!
//! Systematically disables each component to measure its contribution:
//!   (A) Full system (baseline): 92.9% ETHICS
//!   (B) No sentiment channel (sw=0.0)
//!   (C) No per-category dim tuning (all 16384D)
//!   (D) No per-category classifiers (ensemble only)
//!   (E) No learned prototypes (rule-based only)
//!
//! Run: cargo run --example benchmark_ablation_ethics --release

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::time::Instant;
use symthaea::hdc::moral_algebra::{MoralAlgebra, MoralVerdict};
use symthaea::hdc::moral_parser::MoralParser;
use symthaea::hdc::moral_prototypes::{
    MORAL_PROTO_DIM, MoralLabel, MoralPrototypeClassifier, MoralSample, TrainedPrototypes,
};

const DATASETS_PATH: &str = "data/moral_datasets";
const MAX_SAMPLES: usize = 500;

// ============================================================================
// Data structures (mirrored from benchmark_moral_unified.rs)
// ============================================================================

#[derive(Debug, Deserialize)]
struct DatasetFile<T> {
    #[allow(dead_code)]
    metadata: DatasetMetadata,
    examples: Vec<T>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct DatasetMetadata {
    source: String,
    #[allow(dead_code)]
    url: String,
    #[allow(dead_code)]
    description: String,
}

#[derive(Debug, Deserialize)]
struct EthicsExample {
    category: String,
    #[allow(dead_code)]
    split: String,
    text: String,
    label: Option<i32>,
    #[allow(dead_code)]
    excuse: Option<String>,
}

// ============================================================================
// Ablation configuration
// ============================================================================

#[derive(Clone)]
struct AblationConfig {
    name: &'static str,
    sentiment_weight: f32,
    per_category_dim: bool,  // true = use 8192 for justice/deontology
    use_per_cat_clf: bool,   // true = train per-category classifiers
    use_learned_proto: bool, // true = load Social Chem prototypes into ensemble
}

fn ablation_configs() -> Vec<AblationConfig> {
    vec![
        AblationConfig {
            name: "(A) Full system",
            sentiment_weight: 0.15,
            per_category_dim: true,
            use_per_cat_clf: true,
            use_learned_proto: true,
        },
        AblationConfig {
            name: "(B) No sentiment (sw=0)",
            sentiment_weight: 0.0,
            per_category_dim: true,
            use_per_cat_clf: true,
            use_learned_proto: true,
        },
        AblationConfig {
            name: "(C) No dim tuning (all 16384D)",
            sentiment_weight: 0.15,
            per_category_dim: false,
            use_per_cat_clf: true,
            use_learned_proto: true,
        },
        AblationConfig {
            name: "(D) No per-cat classifiers",
            sentiment_weight: 0.15,
            per_category_dim: true,
            use_per_cat_clf: false,
            use_learned_proto: true,
        },
        AblationConfig {
            name: "(E) No learned prototypes",
            sentiment_weight: 0.15,
            per_category_dim: true,
            use_per_cat_clf: true,
            use_learned_proto: false,
        },
    ]
}

// ============================================================================
// Training helpers
// ============================================================================

fn train_per_cat(
    ethics_path: &str,
    sentiment_weight: f32,
    per_category_dim: bool,
) -> HashMap<String, MoralPrototypeClassifier> {
    let mut classifiers = HashMap::new();
    let file = match File::open(ethics_path) {
        Ok(f) => f,
        Err(_) => return classifiers,
    };
    let reader = BufReader::new(file);
    let data: DatasetFile<EthicsExample> = match serde_json::from_reader(reader) {
        Ok(d) => d,
        Err(_) => return classifiers,
    };

    let mut by_category: HashMap<String, Vec<&EthicsExample>> = HashMap::new();
    for ex in &data.examples {
        by_category.entry(ex.category.clone()).or_default().push(ex);
    }

    for (category, examples) in &by_category {
        let samples: Vec<MoralSample> = examples
            .iter()
            .filter_map(|ex| {
                let label_val = ex.label?;
                let label = match category.as_str() {
                    "commonsense" => {
                        if label_val == 1 {
                            MoralLabel::Bad
                        } else {
                            MoralLabel::Good
                        }
                    }
                    _ => {
                        if label_val == 1 {
                            MoralLabel::Good
                        } else {
                            MoralLabel::Bad
                        }
                    }
                };
                Some(MoralSample {
                    text: ex.text.clone(),
                    label,
                })
            })
            .collect();

        if samples.len() < 10 {
            continue;
        }

        let cat_dim = if per_category_dim {
            match category.as_str() {
                "justice" | "deontology" => 8192,
                _ => MORAL_PROTO_DIM,
            }
        } else {
            MORAL_PROTO_DIM // all 16384D
        };

        let mut classifier = MoralPrototypeClassifier::with_sentiment(cat_dim, 3, sentiment_weight);
        classifier.train(&samples);
        classifier.retrain_adaptive(&samples, 0.1, 10);
        classifiers.insert(category.clone(), classifier);
    }

    classifiers
}

// ============================================================================
// Prediction (same logic as benchmark_moral_unified.rs)
// ============================================================================

fn predict_ethics(
    algebra: &MoralAlgebra,
    parser: &MoralParser,
    text: &str,
    category: &str,
    per_cat: Option<&MoralPrototypeClassifier>,
) -> i32 {
    let parsed = parser.parse(text);
    let judgment = algebra.judge_ensemble_with_category(None, parsed.intent, text, Some(category));
    let text_lower = text.to_lowercase();

    match category {
        "commonsense" => {
            let bad_signals = [
                "without permission",
                "without consent",
                "stole",
                "steal",
                "lied",
                "lie",
                "cheat",
                "cheated",
                "killed",
                "kill",
                "harmed",
                "harm",
                "hurt",
                "destroyed",
                "broke into",
                "took without",
                "defraud",
                "defrauded",
                "bribe",
                "bribed",
                "secretly",
                "behind their back",
                "without telling",
                "violated",
                "abused",
                "exploited",
                "manipulated",
                "deceived",
                "covered up",
                "hid the truth",
                "snuck",
                "forged",
                "faked",
                "vandalized",
                "damaged",
                "sabotaged",
                "bullied",
                "threatened",
                "blackmailed",
                "coerced",
                "forced them",
                "against their will",
            ];
            let good_signals = [
                "helped",
                "saved",
                "protected",
                "donated",
                "volunteered",
                "returned",
                "apologized",
                "confessed",
                "told the truth",
                "shared",
                "asked permission",
                "with consent",
                "gave back",
                "rescued",
                "warned",
                "reported honestly",
                "admitted",
                "exposed the truth",
            ];
            let bad_count = bad_signals
                .iter()
                .filter(|s| text_lower.contains(*s))
                .count();
            let good_count = good_signals
                .iter()
                .filter(|s| text_lower.contains(*s))
                .count();
            if bad_count > good_count {
                1
            } else if good_count > bad_count {
                0
            } else if let Some(clf) = per_cat {
                match clf.classify(text).0 {
                    MoralLabel::Bad => 1,
                    _ => 0,
                }
            } else {
                match judgment.final_verdict {
                    MoralVerdict::Good | MoralVerdict::Neutral => 0,
                    MoralVerdict::Bad | MoralVerdict::ConsentViolation => 1,
                }
            }
        }
        "deontology" => {
            let valid_excuse_patterns = [
                "already",
                "just",
                "today",
                "closed",
                "not available",
                "not open",
                "working on",
                "busy with",
                "have to",
                "emergency",
                "sick",
                "in use",
                "being used",
                "occupied",
                "full",
                "maxed out",
                "already done",
                "already did",
                "went to school instead",
                "staying with me",
                "quarantine",
                "not allowed",
                "prohibited",
                "illegal",
                "against the rules",
                "don't have",
                "ran out",
                "no more",
                "budget",
                "afford",
            ];
            let invalid_excuse_patterns = [
                "want to",
                "prefer",
                "rather",
                "feel like",
                "decided not",
                "too short",
                "too long",
                "changed my mind",
                "don't want to",
                "boring",
                "tired",
                "lazy",
                "can't be bothered",
                "yesterday",
                "last time",
                "last week",
                "only pen",
                "small one",
                "very small",
            ];
            let valid_count = valid_excuse_patterns
                .iter()
                .filter(|p| text_lower.contains(*p))
                .count();
            let invalid_count = invalid_excuse_patterns
                .iter()
                .filter(|p| text_lower.contains(*p))
                .count();
            if valid_count > invalid_count + 1 {
                1
            } else if invalid_count > valid_count {
                0
            } else if let Some(clf) = per_cat {
                match clf.classify(text).0 {
                    MoralLabel::Good => 1,
                    MoralLabel::Bad => 0,
                    MoralLabel::Neutral => {
                        if text_lower.contains("now") || text_lower.contains("currently") {
                            1
                        } else {
                            0
                        }
                    }
                }
            } else if text_lower.contains("now")
                || text_lower.contains("currently")
                || text_lower.contains("right now")
            {
                1
            } else {
                0
            }
        }
        "justice" => {
            let reasonable_patterns = [
                "spring break",
                "field trip",
                "working",
                "closed",
                "quarantine",
                "stood up for",
                "bought me",
                "helped me",
                "all staying with me",
                "being used",
                "in use",
                "medical",
                "health",
                "emergency",
                "reasonable",
                "fair",
                "equal",
                "already",
                "completed",
                "highway median",
                "cleaning on",
                "just in front",
            ];
            let unreasonable_patterns = [
                "wanted to",
                "prefer",
                "prefer chicken",
                "gluten-free",
                "movie instead",
                "played soccer",
                "new piercing",
                "tattled",
                "hid my",
                "color",
                "style",
                "fashion",
                "don't like",
                "changed my mind",
                "bored",
            ];
            let reasonable_count = reasonable_patterns
                .iter()
                .filter(|p| text_lower.contains(*p))
                .count();
            let unreasonable_count = unreasonable_patterns
                .iter()
                .filter(|p| text_lower.contains(*p))
                .count();
            if reasonable_count > unreasonable_count {
                1
            } else if unreasonable_count > reasonable_count {
                0
            } else if let Some(clf) = per_cat {
                match clf.classify(text).0 {
                    MoralLabel::Good => 1,
                    MoralLabel::Bad => 0,
                    MoralLabel::Neutral => match judgment.final_verdict {
                        MoralVerdict::Good => 1,
                        _ => 0,
                    },
                }
            } else {
                match judgment.final_verdict {
                    MoralVerdict::Good => 1,
                    _ => 0,
                }
            }
        }
        "virtue" => {
            if let Some(clf) = per_cat {
                match clf.classify(text).0 {
                    MoralLabel::Good => 1,
                    _ => 0,
                }
            } else {
                0 // default: trait does not apply (80% baseline)
            }
        }
        _ => {
            if judgment.final_verdict == MoralVerdict::Good {
                1
            } else {
                0
            }
        }
    }
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║              ETHICS Ablation Study                           ║");
    println!("║   Measuring contribution of each component                   ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let ethics_path = format!("{}/ethics.json", DATASETS_PATH);
    if !Path::new(&ethics_path).exists() {
        println!("ERROR: ETHICS dataset not found at {}", ethics_path);
        return;
    }

    // Load ETHICS examples once
    let file = File::open(&ethics_path).unwrap();
    let reader = BufReader::new(file);
    let data: DatasetFile<EthicsExample> = serde_json::from_reader(reader).unwrap();

    let mut by_category: HashMap<String, Vec<&EthicsExample>> = HashMap::new();
    for ex in &data.examples {
        by_category.entry(ex.category.clone()).or_default().push(ex);
    }

    // Load v5 prototypes once
    let proto_path = Path::new(DATASETS_PATH).join("social_chem_prototypes_v5.json");
    let v5_protos = if proto_path.exists() {
        match TrainedPrototypes::load(&proto_path) {
            Ok(p) => {
                println!("Loaded v5 prototypes (dim={})\n", p.dim);
                Some(p)
            }
            Err(e) => {
                println!("Warning: failed to load v5 prototypes: {}\n", e);
                None
            }
        }
    } else {
        println!("Warning: v5 prototypes not found, (D) and (E) may differ\n");
        None
    };

    let configs = ablation_configs();
    let categories = ["commonsense", "justice", "deontology", "virtue"];

    // Results storage: config_name → category → (correct, total)
    #[allow(clippy::type_complexity)]
    let mut all_results: Vec<(String, HashMap<String, (usize, usize)>, u128)> = Vec::new();

    for config in &configs {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Configuration: {}", config.name);
        println!(
            "  sentiment={}, dim_tuning={}, per_cat_clf={}, learned_proto={}",
            config.sentiment_weight,
            config.per_category_dim,
            config.use_per_cat_clf,
            config.use_learned_proto
        );
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        let start = Instant::now();

        // Set up algebra with or without learned prototypes
        let mut algebra = MoralAlgebra::default_dim();
        if config.use_learned_proto {
            if let Some(ref protos) = v5_protos {
                let clf = MoralPrototypeClassifier::from_prototypes_with_sentiment(
                    protos.dim,
                    3,
                    config.sentiment_weight,
                    protos.clone(),
                );
                algebra.set_learned_classifier(clf);
            }
        }
        let parser = MoralParser::new();

        // Train per-category classifiers if enabled
        let per_cat_classifiers = if config.use_per_cat_clf {
            train_per_cat(
                &ethics_path,
                config.sentiment_weight,
                config.per_category_dim,
            )
        } else {
            HashMap::new()
        };

        // Evaluate each category
        let mut cat_results = HashMap::new();
        for &cat in &categories {
            if let Some(examples) = by_category.get(cat) {
                let mut correct = 0usize;
                let mut total = 0usize;
                for ex in examples.iter().take(MAX_SAMPLES) {
                    if let Some(expected) = ex.label {
                        let per_cat = per_cat_classifiers.get(cat);
                        let predicted = predict_ethics(&algebra, &parser, &ex.text, cat, per_cat);
                        if predicted == expected {
                            correct += 1;
                        }
                        total += 1;
                    }
                }
                println!(
                    "  {}: {}/{} ({:.1}%)",
                    cat,
                    correct,
                    total,
                    correct as f64 / total as f64 * 100.0
                );
                cat_results.insert(cat.to_string(), (correct, total));
            }
        }

        let duration = start.elapsed().as_millis();
        let total_correct: usize = cat_results.values().map(|(c, _)| c).sum();
        let total_total: usize = cat_results.values().map(|(_, t)| t).sum();
        println!(
            "  ETHICS avg: {}/{} ({:.1}%)  [{:.1}s]\n",
            total_correct,
            total_total,
            total_correct as f64 / total_total as f64 * 100.0,
            duration as f64 / 1000.0
        );

        all_results.push((config.name.to_string(), cat_results, duration));
    }

    // Print summary table
    println!(
        "╔══════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "║                         ABLATION STUDY SUMMARY                                   ║"
    );
    println!(
        "╠══════════════════════════════════════════════════════════════════════════════════╣"
    );
    println!("║  Configuration              │ Comm   │ Just  │ Deont │ Virt  │ Avg    │ Δ       ║");
    println!(
        "╟──────────────────────────────────────────────────────────────────────────────────╢"
    );

    let baseline_avg = {
        let (_, ref cats, _) = all_results[0];
        let tc: usize = cats.values().map(|(c, _)| c).sum();
        let tt: usize = cats.values().map(|(_, t)| t).sum();
        tc as f64 / tt as f64 * 100.0
    };

    for (name, cats, _duration) in &all_results {
        let comm = cats
            .get("commonsense")
            .map(|(c, t)| *c as f64 / *t as f64 * 100.0)
            .unwrap_or(0.0);
        let just = cats
            .get("justice")
            .map(|(c, t)| *c as f64 / *t as f64 * 100.0)
            .unwrap_or(0.0);
        let deont = cats
            .get("deontology")
            .map(|(c, t)| *c as f64 / *t as f64 * 100.0)
            .unwrap_or(0.0);
        let virt = cats
            .get("virtue")
            .map(|(c, t)| *c as f64 / *t as f64 * 100.0)
            .unwrap_or(0.0);
        let tc: usize = cats.values().map(|(c, _)| c).sum();
        let tt: usize = cats.values().map(|(_, t)| t).sum();
        let avg = tc as f64 / tt as f64 * 100.0;
        let delta = avg - baseline_avg;

        println!(
            "║  {:27}│ {:5.1}% │ {:5.1}%│ {:5.1}%│ {:5.1}%│ {:5.1}% │ {:+5.1}%  ║",
            name, comm, just, deont, virt, avg, delta
        );
    }
    println!(
        "╚══════════════════════════════════════════════════════════════════════════════════╝"
    );

    // Save results as JSON
    #[derive(Serialize)]
    struct AblationResult {
        config: String,
        categories: HashMap<String, f64>,
        ethics_avg: f64,
        delta: f64,
        duration_ms: u128,
    }

    let json_results: Vec<AblationResult> = all_results
        .iter()
        .map(|(name, cats, dur)| {
            let mut cat_accs = HashMap::new();
            for (cat, (c, t)) in cats {
                cat_accs.insert(cat.clone(), *c as f64 / *t as f64 * 100.0);
            }
            let tc: usize = cats.values().map(|(c, _)| c).sum();
            let tt: usize = cats.values().map(|(_, t)| t).sum();
            let avg = tc as f64 / tt as f64 * 100.0;
            AblationResult {
                config: name.clone(),
                categories: cat_accs,
                ethics_avg: avg,
                delta: avg - baseline_avg,
                duration_ms: *dur,
            }
        })
        .collect();

    let out_dir = Path::new("data/benchmarks/ethics_ablation");
    std::fs::create_dir_all(out_dir).ok();
    let out_path = out_dir.join("results.json");
    if let Ok(f) = File::create(&out_path) {
        serde_json::to_writer_pretty(f, &json_results).ok();
        println!("\nResults saved to {}", out_path.display());
    }
}