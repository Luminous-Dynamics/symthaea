//! Performance Benchmarks for Enhancement #4: Causal Reasoning
//!
//! This benchmark suite measures the performance of all four phases:
//! - Phase 1: Causal Intervention
//! - Phase 2: Counterfactual Reasoning
//! - Phase 3: Action Planning
//! - Phase 4: Causal Explanation
//!
//! Run with: cargo bench --bench causal_reasoning_benchmark

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use symthaea::observability::{
    // Phase 1: Causal Intervention
    CausalInterventionEngine, InterventionSpec, InterventionType,
    // Phase 2: Counterfactual Reasoning
    CounterfactualEngine, CounterfactualQuery,
    // Phase 3: Action Planning
    ActionPlanner, Goal, GoalDirection, PlannerConfig,
    // Phase 4: Causal Explanation
    ExplanationGenerator, ExplanationLevel,
    // Core types
    CausalGraph, EdgeType,
    ProbabilisticCausalGraph, ProbabilisticConfig,
};

/// Create a simple test graph for benchmarking
fn create_simple_graph() -> ProbabilisticCausalGraph {
    let mut graph = ProbabilisticCausalGraph::new();

    // Simple chain: A → B → C
    for _ in 0..9 {
        graph.observe_edge("A", "B", EdgeType::Direct, true);
        graph.observe_edge("B", "C", EdgeType::Direct, true);
    }

    graph
}

/// Create a medium complexity graph
fn create_medium_graph() -> ProbabilisticCausalGraph {
    let mut graph = ProbabilisticCausalGraph::new();

    // Multi-path graph:
    //     A
    //    / \
    //   B   C
    //    \ /
    //     D
    for _ in 0..8 {
        graph.observe_edge("A", "B", EdgeType::Direct, true);
        graph.observe_edge("A", "C", EdgeType::Direct, true);
        graph.observe_edge("B", "D", EdgeType::Direct, true);
        graph.observe_edge("C", "D", EdgeType::Direct, true);
    }

    graph
}

/// Create a complex graph with many nodes
fn create_complex_graph() -> ProbabilisticCausalGraph {
    let mut graph = ProbabilisticCausalGraph::new();

    // Network with 10 nodes and 20 edges
    let nodes = vec!["N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8", "N9", "N10"];

    for _ in 0..7 {
        // Create various connections
        graph.observe_edge("N1", "N2", EdgeType::Direct, true);
        graph.observe_edge("N1", "N3", EdgeType::Direct, true);
        graph.observe_edge("N2", "N4", EdgeType::Direct, true);
        graph.observe_edge("N3", "N4", EdgeType::Direct, true);
        graph.observe_edge("N4", "N5", EdgeType::Direct, true);
        graph.observe_edge("N5", "N6", EdgeType::Direct, true);
        graph.observe_edge("N5", "N7", EdgeType::Direct, true);
        graph.observe_edge("N6", "N8", EdgeType::Direct, true);
        graph.observe_edge("N7", "N8", EdgeType::Direct, true);
        graph.observe_edge("N8", "N9", EdgeType::Direct, true);
        graph.observe_edge("N9", "N10", EdgeType::Direct, true);
    }

    graph
}

// ============================================================================
// PHASE 1: CAUSAL INTERVENTION BENCHMARKS
// ============================================================================

fn bench_intervention_simple(c: &mut Criterion) {
    let graph = create_simple_graph();

    c.bench_function("intervention/simple_graph/single_intervention", |b| {
        b.iter(|| {
            let mut engine = CausalInterventionEngine::new(graph.clone());
            let spec = InterventionSpec::new()
                .set_value("A", 1.0);

            black_box(engine.predict_intervention_spec(&spec, "C"))
        });
    });
}

fn bench_intervention_medium(c: &mut Criterion) {
    let graph = create_medium_graph();

    c.bench_function("intervention/medium_graph/multi_path", |b| {
        b.iter(|| {
            let mut engine = CausalInterventionEngine::new(graph.clone());
            let spec = InterventionSpec::new()
                .set_value("A", 1.0);

            black_box(engine.predict_intervention_spec(&spec, "D"))
        });
    });
}

fn bench_intervention_complex(c: &mut Criterion) {
    let graph = create_complex_graph();

    c.bench_function("intervention/complex_graph/long_chain", |b| {
        b.iter(|| {
            let mut engine = CausalInterventionEngine::new(graph.clone());
            let spec = InterventionSpec::new()
                .set_value("N1", 1.0)
                .set_value("N2", 0.8);

            black_box(engine.predict_intervention_spec(&spec, "N10"))
        });
    });
}

// ============================================================================
// PHASE 2: COUNTERFACTUAL REASONING BENCHMARKS
// ============================================================================

fn bench_counterfactual_simple(c: &mut Criterion) {
    let graph = create_simple_graph();

    c.bench_function("counterfactual/simple_graph/single_query", |b| {
        b.iter(|| {
            let mut engine = CounterfactualEngine::new(graph.clone());
            let query = CounterfactualQuery::new("C")
                .with_evidence("A", 0.0)
                .with_evidence("C", 0.0)
                .with_counterfactual("A", 1.0);

            black_box(engine.compute_counterfactual(&query))
        });
    });
}

fn bench_counterfactual_medium(c: &mut Criterion) {
    let graph = create_medium_graph();

    c.bench_function("counterfactual/medium_graph/multiple_evidence", |b| {
        b.iter(|| {
            let mut engine = CounterfactualEngine::new(graph.clone());
            let query = CounterfactualQuery::new("D")
                .with_evidence("A", 0.5)
                .with_evidence("B", 0.3)
                .with_evidence("C", 0.4)
                .with_counterfactual("A", 1.0);

            black_box(engine.compute_counterfactual(&query))
        });
    });
}

fn bench_counterfactual_complex(c: &mut Criterion) {
    let graph = create_complex_graph();

    c.bench_function("counterfactual/complex_graph/deep_inference", |b| {
        b.iter(|| {
            let mut engine = CounterfactualEngine::new(graph.clone());
            let query = CounterfactualQuery::new("N10")
                .with_evidence("N1", 0.2)
                .with_evidence("N5", 0.5)
                .with_evidence("N10", 0.3)
                .with_counterfactual("N1", 1.0)
                .with_counterfactual("N2", 0.8);

            black_box(engine.compute_counterfactual(&query))
        });
    });
}

// ============================================================================
// PHASE 3: ACTION PLANNING BENCHMARKS
// ============================================================================

fn bench_planning_simple(c: &mut Criterion) {
    let graph = create_simple_graph();

    c.bench_function("planning/simple_graph/single_goal", |b| {
        b.iter(|| {
            let mut planner = ActionPlanner::new(graph.clone(), PlannerConfig::default());
            let goal = Goal::new("C", 0.9, GoalDirection::Maximize);

            black_box(planner.plan(&[goal]))
        });
    });
}

fn bench_planning_medium(c: &mut Criterion) {
    let graph = create_medium_graph();

    c.bench_function("planning/medium_graph/multiple_goals", |b| {
        b.iter(|| {
            let mut planner = ActionPlanner::new(graph.clone(), PlannerConfig::default());
            let goals = vec![
                Goal::new("B", 0.8, GoalDirection::Maximize),
                Goal::new("C", 0.8, GoalDirection::Maximize),
                Goal::new("D", 0.9, GoalDirection::Maximize),
            ];

            black_box(planner.plan(&goals))
        });
    });
}

fn bench_planning_complex(c: &mut Criterion) {
    let graph = create_complex_graph();

    c.bench_function("planning/complex_graph/optimized_path", |b| {
        b.iter(|| {
            let config = PlannerConfig {
                max_interventions: 5,
                min_confidence: 0.6,
                max_steps: 10,
                optimization_metric: "utility".to_string(),
            };
            let mut planner = ActionPlanner::new(graph.clone(), config);
            let goal = Goal::new("N10", 0.95, GoalDirection::Maximize);

            black_box(planner.plan(&[goal]))
        });
    });
}

// ============================================================================
// PHASE 4: CAUSAL EXPLANATION BENCHMARKS
// ============================================================================

fn bench_explanation_simple(c: &mut Criterion) {
    let graph = create_simple_graph();

    c.bench_function("explanation/simple_graph/basic_causation", |b| {
        b.iter(|| {
            let generator = ExplanationGenerator::new(graph.graph().clone());

            black_box(generator.explain_causation("A", "C", ExplanationLevel::Simple))
        });
    });
}

fn bench_explanation_medium(c: &mut Criterion) {
    let graph = create_medium_graph();

    c.bench_function("explanation/medium_graph/detailed_paths", |b| {
        b.iter(|| {
            let generator = ExplanationGenerator::new(graph.graph().clone());

            black_box(generator.explain_causation("A", "D", ExplanationLevel::Detailed))
        });
    });
}

fn bench_explanation_complex(c: &mut Criterion) {
    let graph = create_complex_graph();

    c.bench_function("explanation/complex_graph/technical_analysis", |b| {
        b.iter(|| {
            let generator = ExplanationGenerator::new(graph.graph().clone());

            black_box(generator.explain_causation("N1", "N10", ExplanationLevel::Technical))
        });
    });
}

// ============================================================================
// INTEGRATED WORKFLOW BENCHMARKS
// ============================================================================

fn bench_integrated_workflow(c: &mut Criterion) {
    c.bench_function("integrated/full_workflow/intervention_to_explanation", |b| {
        b.iter(|| {
            let graph = create_medium_graph();

            // Step 1: Plan an action
            let mut planner = ActionPlanner::new(graph.clone(), PlannerConfig::default());
            let goal = Goal::new("D", 0.9, GoalDirection::Maximize);
            let plan = planner.plan(&[goal]);

            // Step 2: Simulate intervention
            let mut intervention_engine = CausalInterventionEngine::new(graph.clone());
            let spec = InterventionSpec::new().set_value("A", 1.0);
            let intervention_result = intervention_engine.predict_intervention_spec(&spec, "D");

            // Step 3: Counterfactual reasoning
            let mut counterfactual_engine = CounterfactualEngine::new(graph.clone());
            let query = CounterfactualQuery::new("D")
                .with_evidence("A", 0.5)
                .with_counterfactual("A", 1.0);
            let counterfactual_result = counterfactual_engine.compute_counterfactual(&query);

            // Step 4: Generate explanation
            let generator = ExplanationGenerator::new(graph.graph().clone());
            let explanation = generator.explain_causation("A", "D", ExplanationLevel::Detailed);

            black_box((plan, intervention_result, counterfactual_result, explanation))
        });
    });
}

// ============================================================================
// SCALING BENCHMARKS
// ============================================================================

fn bench_scaling_graph_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling/graph_size");

    for num_nodes in [5, 10, 20, 50].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_nodes),
            num_nodes,
            |b, &num_nodes| {
                // Create graph with specified number of nodes
                let mut graph = ProbabilisticCausalGraph::new();

                for i in 0..num_nodes-1 {
                    let from = format!("N{}", i);
                    let to = format!("N{}", i+1);

                    for _ in 0..8 {
                        graph.observe_edge(&from, &to, EdgeType::Direct, true);
                    }
                }

                b.iter(|| {
                    let mut engine = CausalInterventionEngine::new(graph.clone());
                    let spec = InterventionSpec::new().set_value("N0", 1.0);
                    let target = format!("N{}", num_nodes-1);

                    black_box(engine.predict_intervention_spec(&spec, &target))
                });
            },
        );
    }

    group.finish();
}

fn bench_scaling_evidence_count(c: &mut Criterion) {
    let graph = create_complex_graph();
    let mut group = c.benchmark_group("scaling/evidence_count");

    for num_evidence in [1, 3, 5, 10].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_evidence),
            num_evidence,
            |b, &num_evidence| {
                b.iter(|| {
                    let mut engine = CounterfactualEngine::new(graph.clone());
                    let mut query = CounterfactualQuery::new("N10");

                    for i in 0..*num_evidence {
                        let node = format!("N{}", i % 10 + 1);
                        query = query.with_evidence(&node, 0.5);
                    }

                    query = query.with_counterfactual("N1", 1.0);

                    black_box(engine.compute_counterfactual(&query))
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// BENCHMARK GROUPS
// ============================================================================

criterion_group!(
    intervention_benches,
    bench_intervention_simple,
    bench_intervention_medium,
    bench_intervention_complex
);

criterion_group!(
    counterfactual_benches,
    bench_counterfactual_simple,
    bench_counterfactual_medium,
    bench_counterfactual_complex
);

criterion_group!(
    planning_benches,
    bench_planning_simple,
    bench_planning_medium,
    bench_planning_complex
);

criterion_group!(
    explanation_benches,
    bench_explanation_simple,
    bench_explanation_medium,
    bench_explanation_complex
);

criterion_group!(
    integrated_benches,
    bench_integrated_workflow
);

criterion_group!(
    scaling_benches,
    bench_scaling_graph_size,
    bench_scaling_evidence_count
);

criterion_main!(
    intervention_benches,
    counterfactual_benches,
    planning_benches,
    explanation_benches,
    integrated_benches,
    scaling_benches
);
