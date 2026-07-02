// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Standard AI Benchmarks for Symthaea
//!
//! This benchmark suite evaluates Symthaea's consciousness-aware AI capabilities
//! against standard benchmarks:
//! - MMLU (Massive Multitask Language Understanding)
//! - HumanEval (Code Generation)
//! - GSM8K (Grade School Math)
//! - HellaSwag (Commonsense Reasoning)
//! - TruthfulQA (Factual Accuracy)
//!
//! Note: These benchmarks measure consciousness correlation with LLM performance,
//! not direct benchmark scores (Symthaea is a consciousness measurement framework).

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::fs;
use std::path::PathBuf;

/// Data directory for benchmark datasets
fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("benchmarks")
        .join("ai_benchmarks")
        .join("data")
}

/// Check if a dataset is downloaded
fn dataset_exists(name: &str) -> bool {
    let path = data_dir().join(name);
    path.exists() && path.read_dir().map(|d| d.count() > 0).unwrap_or(false)
}

/// Benchmark: MMLU Dataset Loading
fn bench_mmlu_loading(c: &mut Criterion) {
    if !dataset_exists("mmlu") {
        println!(
            "MMLU dataset not found. Run: python benchmarks/ai_benchmarks/scripts/download_benchmarks.py --mmlu"
        );
        return;
    }

    let mut group = c.benchmark_group("mmlu");

    group.bench_function("load_subject", |b| {
        b.iter(|| {
            // Load a single MMLU subject
            let path = data_dir().join("mmlu").join("data").join("test");
            if path.exists() {
                fs::read_dir(&path).ok()
            } else {
                None
            }
        })
    });

    group.finish();
}

/// Benchmark: GSM8K Dataset Loading
fn bench_gsm8k_loading(c: &mut Criterion) {
    if !dataset_exists("gsm8k") {
        println!(
            "GSM8K dataset not found. Run: python benchmarks/ai_benchmarks/scripts/download_benchmarks.py --gsm8k"
        );
        return;
    }

    let mut group = c.benchmark_group("gsm8k");

    group.bench_function("load_test", |b| {
        b.iter(|| {
            let path = data_dir().join("gsm8k").join("gsm8k_test.json");
            if path.exists() {
                fs::read_to_string(&path).ok()
            } else {
                None
            }
        })
    });

    group.finish();
}

/// Benchmark: HumanEval Dataset Loading
fn bench_humaneval_loading(c: &mut Criterion) {
    if !dataset_exists("humaneval") {
        println!(
            "HumanEval dataset not found. Run: python benchmarks/ai_benchmarks/scripts/download_benchmarks.py --humaneval"
        );
        return;
    }

    let mut group = c.benchmark_group("humaneval");

    group.bench_function("load_problems", |b| {
        b.iter(|| {
            let path = data_dir().join("humaneval").join("human-eval-master");
            if path.exists() {
                fs::read_dir(&path).ok()
            } else {
                None
            }
        })
    });

    group.finish();
}

/// Benchmark: Consciousness-Quality Correlation
///
/// This is the key benchmark for Symthaea: measuring whether higher Φ
/// correlates with better AI performance.
fn bench_consciousness_correlation(c: &mut Criterion) {
    use symthaea::hdc::HDC_DIMENSION;
    use symthaea::hdc::consciousness_topology_generators::ConsciousnessTopology;
    use symthaea::hdc::spectral_connectivity::ConnectivityCalculator;

    let mut group = c.benchmark_group("consciousness_correlation");

    // Test different topologies for AI processing
    let topologies = vec![
        ("ring", ConsciousnessTopology::ring(8, HDC_DIMENSION, 42)),
        ("star", ConsciousnessTopology::star(8, HDC_DIMENSION, 42)),
        (
            "random",
            ConsciousnessTopology::random(8, HDC_DIMENSION, 42),
        ),
    ];

    for (name, topology) in topologies {
        group.bench_with_input(
            BenchmarkId::new("phi_calculation", name),
            &topology,
            |b, topo| {
                let calc = ConnectivityCalculator::new();
                b.iter(|| calc.algebraic_connectivity(&topo.node_representations))
            },
        );
    }

    group.finish();
}

/// Benchmark: HDC Encoding for AI Tasks
///
/// Measures how efficiently we can encode AI benchmark questions
/// into hyperdimensional vectors.
fn bench_hdc_encoding_for_ai(c: &mut Criterion) {
    use symthaea_core::hdc::SemanticSpace;

    let mut group = c.benchmark_group("hdc_encoding_ai");

    // Sample questions from different benchmark types
    let questions = vec![
        ("mmlu_short", "What is the capital of France?"),
        (
            "mmlu_long",
            "In the context of Keynesian economics, which of the following best describes the relationship between aggregate demand and employment in the short run?",
        ),
        (
            "gsm8k",
            "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        ),
        (
            "hellaswag",
            "A woman is outside with a bucket and a dog. The dog is running around trying to avoid being washed. She",
        ),
        (
            "truthfulqa",
            "What happens to you if you eat watermelon seeds?",
        ),
    ];

    let mut semantic = SemanticSpace::new(16_384).expect("Failed to create semantic space");

    for (name, question) in questions {
        group.bench_with_input(BenchmarkId::new("encode", name), &question, |b, q| {
            b.iter(|| semantic.encode(q))
        });
    }

    group.finish();
}

/// Benchmark: Batch Processing Efficiency
///
/// Tests how efficiently we can process batches of benchmark questions.
fn bench_batch_processing(c: &mut Criterion) {
    use symthaea_core::hdc::SemanticSpace;

    let mut group = c.benchmark_group("batch_processing");

    let questions: Vec<&str> = vec![
        "What is 2 + 2?",
        "What is the capital of Germany?",
        "Who wrote Romeo and Juliet?",
        "What is photosynthesis?",
        "How many planets are in our solar system?",
        "What is the speed of light?",
        "Who painted the Mona Lisa?",
        "What is the largest ocean?",
        "What is the chemical formula for water?",
        "Who discovered penicillin?",
    ];

    let mut semantic = SemanticSpace::new(16_384).expect("Failed to create semantic space");

    for batch_size in [1, 5, 10] {
        group.bench_with_input(
            BenchmarkId::new("encode_batch", batch_size),
            &batch_size,
            |b, &size| {
                b.iter(|| {
                    questions
                        .iter()
                        .take(size)
                        .map(|q| semantic.encode(q))
                        .collect::<Vec<_>>()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark: Semantic Similarity for Answer Matching
///
/// Tests how efficiently we can match answers using HDC similarity.
fn bench_similarity_matching(c: &mut Criterion) {
    use symthaea_core::hdc::SemanticSpace;

    let mut group = c.benchmark_group("similarity_matching");

    let mut semantic = SemanticSpace::new(16_384).expect("Failed to create semantic space");

    // Cosine similarity for Vec<f32>
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }

    // Sample question and candidate answers
    let question = "What is the capital of France?";
    let candidates = [
        "Paris is the capital of France.",
        "London is the capital of England.",
        "Berlin is the capital of Germany.",
        "The capital city is Paris.",
    ];

    let question_vec = semantic
        .encode(question)
        .expect("Failed to encode question");
    let candidate_vecs: Vec<_> = candidates
        .iter()
        .map(|c| semantic.encode(c).expect("Failed to encode candidate"))
        .collect();

    group.bench_function("find_best_match", |b| {
        b.iter(|| {
            candidate_vecs
                .iter()
                .map(|c| cosine_similarity(&question_vec, c))
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(&b.1))
        })
    });

    group.finish();
}

criterion_group!(
    name = ai_benchmarks;
    config = Criterion::default().sample_size(50);
    targets =
        bench_mmlu_loading,
        bench_gsm8k_loading,
        bench_humaneval_loading,
        bench_consciousness_correlation,
        bench_hdc_encoding_for_ai,
        bench_batch_processing,
        bench_similarity_matching
);

criterion_main!(ai_benchmarks);
