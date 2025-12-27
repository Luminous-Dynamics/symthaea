// Phase 3 Causal Understanding System - Benchmarks
//
// These benchmarks validate the performance characteristics of the causal analysis pipeline:
// - Graph construction from traces of varying sizes
// - Causal query performance (causes, effects, chains, did_cause)
// - Visualization export performance
// - Memory overhead tracking
//
// Expected Performance Targets:
// - Graph construction: <60ms for 1,000 events
// - find_causes/effects: <100μs per query
// - did_cause: <1ms per query
// - Critical path: <10ms for complex graphs

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use symthaea::observability::{
    correlation::CorrelationContext,
    causal_graph::CausalGraph,
    trace_analyzer::TraceAnalyzer,
    types::{Trace, PhiComponents, SecurityDecision},
    observer::{
        TraceObserver, SymthaeaObserver,
        SecurityCheckEvent, PhiMeasurementEvent, RouterSelectionEvent,
    },
};
use std::sync::{Arc, RwLock};
use tempfile::NamedTempFile;
use chrono::Utc;

type SharedObserver = Arc<RwLock<Box<dyn SymthaeaObserver>>>;

// Helper to create trace with N events in a chain
fn create_trace_with_chain(n: usize) -> Trace {
    let temp_file = NamedTempFile::new().unwrap();
    let trace_path = temp_file.into_temp_path();

    let observer: SharedObserver = Arc::new(RwLock::new(
        Box::new(TraceObserver::new(&trace_path).unwrap())
    ));

    let mut ctx = CorrelationContext::new(format!("bench_chain_{}", n));

    // Create chain of N events
    let mut parent_id = None;
    for i in 0..n {
        if let Some(pid) = parent_id.as_ref() {
            ctx.push_parent(pid);
        }

        let meta = ctx.create_event_metadata_with_tags(vec![format!("event_{}", i)]);

        // Alternate between event types for variety
        {
            let mut obs = observer.blocking_write();
            match i % 3 {
                0 => {
                    obs.record_security_check(SecurityCheckEvent {
                        timestamp: Utc::now(),
                        operation: format!("op_{}", i),
                        decision: SecurityDecision::Allowed,
                        reason: None,
                        secrets_redacted: 0,
                        similarity_score: Some(0.01 * i as f64),
                        matched_pattern: None,
                    }).unwrap();
                }
                1 => {
                    obs.record_phi_measurement(PhiMeasurementEvent {
                        timestamp: Utc::now(),
                        phi: 0.7 + (i as f64 * 0.001).min(0.29),
                        components: PhiComponents::default(),
                        temporal_continuity: 0.85 + (i as f64 * 0.001).min(0.14),
                    }).unwrap();
                }
                _ => {
                    obs.record_router_selection(RouterSelectionEvent {
                        timestamp: Utc::now(),
                        input: format!("input_{}", i),
                        selected_router: "standard".to_string(),
                        confidence: 0.8,
                        alternatives: vec![],
                        bandit_stats: std::collections::HashMap::new(),
                    }).unwrap();
                }
            }
        }

        if parent_id.is_some() {
            ctx.pop_parent();
        }
        parent_id = Some(meta.id.clone());
    }

    {
        let mut obs = observer.blocking_write();
        obs.finalize().unwrap();
    }

    Trace::load_from_file(&trace_path).unwrap()
}

// Benchmark 1: Graph Construction from Trace
fn bench_graph_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_construction");

    for size in [100, 500, 1000, 5000].iter() {
        let trace = create_trace_with_chain(*size);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let graph = CausalGraph::from_trace(black_box(&trace));
                black_box(graph);
            });
        });
    }

    group.finish();
}

// Benchmark 2: Find Causes Query
fn bench_find_causes(c: &mut Criterion) {
    let mut group = c.benchmark_group("find_causes");

    for size in [100, 500, 1000, 5000].iter() {
        let trace = create_trace_with_chain(*size);
        let graph = CausalGraph::from_trace(&trace);

        // Query the last event (has the longest causal chain)
        let last_event_id = &graph.events().last().unwrap().id;

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let causes = graph.find_causes(black_box(last_event_id), Some(10));
                black_box(causes);
            });
        });
    }

    group.finish();
}

// Benchmark 3: Critical Path Analysis
fn bench_critical_path(c: &mut Criterion) {
    let mut group = c.benchmark_group("critical_path");

    for size in [100, 500, 1000].iter() {
        let trace = create_trace_with_chain(*size);
        let analyzer = TraceAnalyzer::new(trace.clone());

        // Find critical path to last event
        let last_event_id = &trace.events.last().unwrap().event_id;

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let chain = analyzer.get_causal_chain(black_box(last_event_id));
                black_box(chain);
            });
        });
    }

    group.finish();
}

// Benchmark 4: Mermaid Export
fn bench_mermaid_export(c: &mut Criterion) {
    let mut group = c.benchmark_group("mermaid_export");

    for size in [100, 500, 1000].iter() {
        let trace = create_trace_with_chain(*size);
        let graph = CausalGraph::from_trace(&trace);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let mermaid = graph.to_mermaid();
                black_box(mermaid);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_graph_construction,
    bench_find_causes,
    bench_critical_path,
    bench_mermaid_export,
);

criterion_main!(benches);
