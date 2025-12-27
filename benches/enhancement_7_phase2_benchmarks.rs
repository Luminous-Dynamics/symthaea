//! Performance Benchmarks for Enhancement #7 Phase 2
//!
//! Measures performance improvements from integrating Enhancement #4 components:
//! 1. Synthesis time (with vs without components)
//! 2. Verification accuracy (counterfactual testing)
//! 3. Overall workflow performance
//!
//! Run with: cargo bench --bench enhancement_7_phase2_benchmarks

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use symthaea::observability::{
    CausalInterventionEngine, CounterfactualEngine, ActionPlanner,
    ProbabilisticCausalGraph,
};
use symthaea::synthesis::{
    CausalProgramSynthesizer, CounterfactualVerifier, SynthesisConfig,
    VerificationConfig, CausalSpec,
};

// ============================================================================
// Benchmark 1: Synthesis Performance (Phase 1 vs Phase 2)
// ============================================================================

fn bench_synthesis_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("synthesis_baseline");

    let spec = CausalSpec::MakeCause {
        cause: "exercise".to_string(),
        effect: "health".to_string(),
        strength: 0.75,
    };

    group.bench_function("phase1_no_components", |b| {
        b.iter(|| {
            let mut synthesizer = CausalProgramSynthesizer::new(SynthesisConfig::default());
            synthesizer.synthesize(black_box(&spec)).unwrap()
        });
    });

    group.finish();
}

fn bench_synthesis_with_intervention_engine(c: &mut Criterion) {
    let mut group = c.benchmark_group("synthesis_with_intervention");

    let spec = CausalSpec::MakeCause {
        cause: "exercise".to_string(),
        effect: "health".to_string(),
        strength: 0.75,
    };

    group.bench_function("phase2_intervention_engine", |b| {
        b.iter(|| {
            let graph = ProbabilisticCausalGraph::new();
            let intervention_engine = CausalInterventionEngine::new(graph);
            let mut synthesizer = CausalProgramSynthesizer::new(SynthesisConfig::default())
                .with_intervention_engine(intervention_engine);
            synthesizer.synthesize(black_box(&spec)).unwrap()
        });
    });

    group.finish();
}

fn bench_synthesis_with_all_components(c: &mut Criterion) {
    let mut group = c.benchmark_group("synthesis_complete");

    let spec = CausalSpec::Strengthen {
        cause: "exercise".to_string(),
        effect: "health".to_string(),
        target_strength: 0.8,
    };

    group.bench_function("phase2_all_components", |b| {
        b.iter(|| {
            let graph = ProbabilisticCausalGraph::new();
            let intervention_engine = CausalInterventionEngine::new(graph.clone());
            let action_planner = ActionPlanner::new(graph);
            let mut synthesizer = CausalProgramSynthesizer::new(SynthesisConfig::default())
                .with_intervention_engine(intervention_engine)
                .with_action_planner(action_planner);
            synthesizer.synthesize(black_box(&spec)).unwrap()
        });
    });

    group.finish();
}

// ============================================================================
// Benchmark 2: Verification Performance (Phase 1 vs Phase 2)
// ============================================================================

fn bench_verification_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("verification_baseline");

    // Pre-synthesize a program
    let mut synthesizer = CausalProgramSynthesizer::new(SynthesisConfig::default());
    let spec = CausalSpec::MakeCause {
        cause: "a".to_string(),
        effect: "b".to_string(),
        strength: 0.5,
    };
    let program = synthesizer.synthesize(&spec).unwrap();

    group.bench_function("phase1_no_counterfactual", |b| {
        b.iter(|| {
            let config = VerificationConfig {
                num_counterfactuals: 100,
                min_accuracy: 0.95,
                test_edge_cases: true,
                max_complexity: 10,
            };
            let mut verifier = CounterfactualVerifier::new(config);
            verifier.verify(black_box(&program))
        });
    });

    group.finish();
}

fn bench_verification_with_counterfactual_engine(c: &mut Criterion) {
    let mut group = c.benchmark_group("verification_with_counterfactual");

    // Pre-synthesize a program
    let mut synthesizer = CausalProgramSynthesizer::new(SynthesisConfig::default());
    let spec = CausalSpec::MakeCause {
        cause: "a".to_string(),
        effect: "b".to_string(),
        strength: 0.5,
    };
    let program = synthesizer.synthesize(&spec).unwrap();

    group.bench_function("phase2_counterfactual_engine", |b| {
        b.iter(|| {
            let graph = ProbabilisticCausalGraph::new();
            let counterfactual_engine = CounterfactualEngine::new(graph);
            let config = VerificationConfig {
                num_counterfactuals: 100,
                min_accuracy: 0.95,
                test_edge_cases: true,
                max_complexity: 10,
            };
            let mut verifier = CounterfactualVerifier::new(config)
                .with_counterfactual_engine(counterfactual_engine);
            verifier.verify(black_box(&program))
        });
    });

    group.finish();
}

// ============================================================================
// Benchmark 3: Complete Workflow Performance
// ============================================================================

fn bench_complete_workflow_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("complete_workflow");

    let spec = CausalSpec::Strengthen {
        cause: "exercise".to_string(),
        effect: "health".to_string(),
        target_strength: 0.8,
    };

    group.bench_function("phase1_baseline_workflow", |b| {
        b.iter(|| {
            // Synthesis (Phase 1)
            let mut synthesizer = CausalProgramSynthesizer::new(SynthesisConfig::default());
            let program = synthesizer.synthesize(black_box(&spec)).unwrap();

            // Verification (Phase 1)
            let config = VerificationConfig {
                num_counterfactuals: 50,
                min_accuracy: 0.90,
                test_edge_cases: true,
                max_complexity: 10,
            };
            let mut verifier = CounterfactualVerifier::new(config);
            let _result = verifier.verify(&program);
        });
    });

    group.bench_function("phase2_enhanced_workflow", |b| {
        b.iter(|| {
            // Synthesis (Phase 2 - all components)
            let graph = ProbabilisticCausalGraph::new();
            let intervention_engine = CausalInterventionEngine::new(graph.clone());
            let counterfactual_engine = CounterfactualEngine::new(graph.clone());
            let action_planner = ActionPlanner::new(graph);

            let mut synthesizer = CausalProgramSynthesizer::new(SynthesisConfig::default())
                .with_intervention_engine(intervention_engine)
                .with_action_planner(action_planner);

            let program = synthesizer.synthesize(black_box(&spec)).unwrap();

            // Verification (Phase 2 - with counterfactual engine)
            let config = VerificationConfig {
                num_counterfactuals: 50,
                min_accuracy: 0.90,
                test_edge_cases: true,
                max_complexity: 10,
            };
            let mut verifier = CounterfactualVerifier::new(config)
                .with_counterfactual_engine(counterfactual_engine);
            let _result = verifier.verify(&program);
        });
    });

    group.finish();
}

// ============================================================================
// Benchmark 4: Scalability (Different Complexity Levels)
// ============================================================================

fn bench_scalability_by_complexity(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalability");

    for path_length in [1, 2, 3, 5].iter() {
        let through: Vec<String> = (0..*path_length)
            .map(|i| format!("step{}", i))
            .collect();

        let spec = CausalSpec::CreatePath {
            from: "start".to_string(),
            through,
            to: "end".to_string(),
        };

        group.bench_with_input(
            BenchmarkId::new("synthesis_complexity", path_length),
            path_length,
            |b, _| {
                b.iter(|| {
                    let mut synthesizer = CausalProgramSynthesizer::new(SynthesisConfig::default());
                    synthesizer.synthesize(black_box(&spec)).unwrap()
                });
            },
        );
    }

    group.finish();
}

fn bench_scalability_by_verification_tests(c: &mut Criterion) {
    let mut group = c.benchmark_group("verification_scalability");

    // Pre-synthesize a program
    let mut synthesizer = CausalProgramSynthesizer::new(SynthesisConfig::default());
    let spec = CausalSpec::MakeCause {
        cause: "a".to_string(),
        effect: "b".to_string(),
        strength: 0.5,
    };
    let program = synthesizer.synthesize(&spec).unwrap();

    for num_tests in [10, 50, 100, 500].iter() {
        group.bench_with_input(
            BenchmarkId::new("verification_tests", num_tests),
            num_tests,
            |b, &num| {
                b.iter(|| {
                    let config = VerificationConfig {
                        num_counterfactuals: num,
                        min_accuracy: 0.95,
                        test_edge_cases: true,
                        max_complexity: 10,
                    };
                    let mut verifier = CounterfactualVerifier::new(config);
                    verifier.verify(black_box(&program))
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Benchmark Configuration
// ============================================================================

criterion_group!(
    benches,
    bench_synthesis_baseline,
    bench_synthesis_with_intervention_engine,
    bench_synthesis_with_all_components,
    bench_verification_baseline,
    bench_verification_with_counterfactual_engine,
    bench_complete_workflow_comparison,
    bench_scalability_by_complexity,
    bench_scalability_by_verification_tests,
);

criterion_main!(benches);
