//! Benchmarks for HDC + LTC + Resonator modules
//!
//! Validates O(log N) complexity claims and measures real performance.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::collections::HashMap;
use chrono::Utc;

// Import from symthaea
use symthaea::observability::{
    ResonantPatternMatcher, ResonantMatcherConfig,
    ResonantCausalAnalyzer, ResonantCausalConfig,
    ResonantByzantineDefender, ResonantDefenseConfig,
    CausalGraph, CausalNode, CausalEdge, EdgeType,
    CausalMotif, MotifSeverity,
    byzantine_defense::{AttackPattern, AttackType, SystemState},
};
use symthaea::consciousness::recursive_improvement::routers::{
    ResonantConsciousnessRouter, ResonantRouterConfig, LatentConsciousnessState,
};

// ═══════════════════════════════════════════════════════════════════════════
// HELPER FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

fn create_causal_graph(n_events: usize) -> CausalGraph {
    let now = Utc::now();
    let mut nodes = HashMap::new();
    let mut edges = Vec::new();

    for i in 0..n_events {
        nodes.insert(
            format!("evt_{}", i),
            CausalNode {
                id: format!("evt_{}", i),
                event_type: format!("type_{}", i % 10),
                timestamp: now,
                correlation_id: Some(format!("corr_{}", i / 10)),
                parent_id: if i > 0 { Some(format!("evt_{}", i - 1)) } else { None },
                duration_ms: Some(10),
                metadata: HashMap::new(),
            },
        );

        if i > 0 {
            edges.push(CausalEdge {
                from: format!("evt_{}", i - 1),
                to: format!("evt_{}", i),
                strength: 0.8 + (i as f64 % 3.0) * 0.05,
                edge_type: EdgeType::Direct,
            });
        }
    }

    CausalGraph {
        nodes,
        edges,
        root_events: vec!["evt_0".to_string()],
        leaf_events: vec![format!("evt_{}", n_events - 1)],
    }
}

fn create_system_state() -> SystemState {
    SystemState {
        honest_nodes: 10,
        suspicious_nodes: 2,
        network_connectivity: 0.9,
        resource_utilization: 0.5,
        consensus_round: Some(100),
        recent_patterns: vec![],
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CONSCIOUSNESS ROUTER BENCHMARKS
// ═══════════════════════════════════════════════════════════════════════════

fn bench_resonant_router(c: &mut Criterion) {
    let mut group = c.benchmark_group("ResonantConsciousnessRouter");

    let mut router = ResonantConsciousnessRouter::new(ResonantRouterConfig::default());

    // Warm up
    for _ in 0..100 {
        let state = LatentConsciousnessState::from_observables(0.5, 0.5, 0.5, 0.5);
        let _ = router.route(&state);
    }

    group.bench_function("single_route", |b| {
        b.iter(|| {
            let state = LatentConsciousnessState::from_observables(
                black_box(0.7),
                black_box(0.6),
                black_box(0.8),
                black_box(0.5),
            );
            router.route(&state)
        })
    });

    // Varying phi values
    for phi in [0.3, 0.5, 0.7, 0.9].iter() {
        group.bench_with_input(BenchmarkId::new("route_phi", phi), phi, |b, &phi| {
            b.iter(|| {
                let state = LatentConsciousnessState::from_observables(phi, 0.5, 0.5, 0.5);
                router.route(&state)
            })
        });
    }

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════
// CAUSAL ANALYZER BENCHMARKS
// ═══════════════════════════════════════════════════════════════════════════

fn bench_resonant_causal(c: &mut Criterion) {
    let mut group = c.benchmark_group("ResonantCausalAnalyzer");

    // Test O(log N) scaling
    for n_events in [10, 50, 100, 500, 1000].iter() {
        let graph = create_causal_graph(*n_events);
        let mut analyzer = ResonantCausalAnalyzer::new(ResonantCausalConfig::default());
        analyzer.index_graph(&graph);

        group.bench_with_input(
            BenchmarkId::new("query_causes", n_events),
            n_events,
            |b, &n| {
                let target = format!("evt_{}", n - 1);
                b.iter(|| analyzer.query_causes(black_box(&target)))
            },
        );
    }

    // Indexing benchmark
    for n_events in [10, 50, 100].iter() {
        let graph = create_causal_graph(*n_events);

        group.bench_with_input(
            BenchmarkId::new("index_graph", n_events),
            n_events,
            |b, _| {
                b.iter(|| {
                    let mut analyzer = ResonantCausalAnalyzer::new(ResonantCausalConfig::default());
                    analyzer.index_graph(black_box(&graph))
                })
            },
        );
    }

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════
// BYZANTINE DEFENSE BENCHMARKS
// ═══════════════════════════════════════════════════════════════════════════

fn bench_resonant_byzantine(c: &mut Criterion) {
    let mut group = c.benchmark_group("ResonantByzantineDefender");

    let mut defender = ResonantByzantineDefender::new(ResonantDefenseConfig::default());
    let state = create_system_state();

    // Register some patterns
    for attack_type in [
        AttackType::SybilAttack,
        AttackType::EclipseAttack,
        AttackType::DoubleSpendAttack,
    ].iter() {
        let pattern = AttackPattern {
            event_sequence: vec!["login".to_string(), "transfer".to_string()],
            timing_constraints: vec![(0, 1, 5.0)],
            anomalies: vec!["rapid".to_string()],
        };
        defender.register_pattern(*attack_type, &pattern);
    }

    // Warm up
    for i in 0..100 {
        let _ = defender.process_event(&format!("event_{}", i % 5), &format!("id_{}", i), &state);
    }

    group.bench_function("process_event", |b| {
        let mut counter = 0u64;
        b.iter(|| {
            counter += 1;
            defender.process_event(
                black_box(&format!("event_{}", counter % 5)),
                black_box(&format!("id_{}", counter)),
                &state,
            )
        })
    });

    // Pattern registration
    group.bench_function("register_pattern", |b| {
        b.iter(|| {
            let pattern = AttackPattern {
                event_sequence: vec![black_box("login".to_string())],
                timing_constraints: vec![],
                anomalies: vec![],
            };
            let mut new_defender = ResonantByzantineDefender::new(ResonantDefenseConfig::default());
            new_defender.register_pattern(black_box(AttackType::SybilAttack), &pattern)
        })
    });

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════
// PATTERN MATCHER BENCHMARKS
// ═══════════════════════════════════════════════════════════════════════════

fn bench_resonant_pattern_matcher(c: &mut Criterion) {
    let mut group = c.benchmark_group("ResonantPatternMatcher");

    let mut matcher = ResonantPatternMatcher::new(ResonantMatcherConfig::default());

    // Register some patterns
    for i in 0..10 {
        let motif = CausalMotif {
            id: format!("motif_{}", i),
            name: format!("Pattern {}", i),
            description: format!("Test pattern {}", i),
            event_sequence: vec![format!("event_{}", i), format!("event_{}", i + 1)],
            timing_constraints: vec![(0, 1, 5.0)],
            min_occurrences: 1,
            severity: MotifSeverity::Warning,
            tags: vec!["test".to_string()],
        };
        matcher.register_motif(&motif);
    }

    // Warm up
    for i in 0..50 {
        let _ = matcher.process_event(&format!("event_{}", i % 5));
    }

    group.bench_function("process_event", |b| {
        let mut counter = 0u64;
        b.iter(|| {
            counter += 1;
            matcher.process_event(black_box(&format!("event_{}", counter % 5)))
        })
    });

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════
// HDC VECTOR OPERATIONS BENCHMARKS
// ═══════════════════════════════════════════════════════════════════════════

fn bench_hdv_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("HDV_Operations");

    // Compare different dimensions
    for dim in [512, 1024, 4096, 16384].iter() {
        let a: Vec<f32> = (0..*dim).map(|i| ((i * 17 + 31) % 1000) as f32 / 500.0 - 1.0).collect();
        let b: Vec<f32> = (0..*dim).map(|i| ((i * 23 + 47) % 1000) as f32 / 500.0 - 1.0).collect();

        group.bench_with_input(BenchmarkId::new("similarity", dim), dim, |bench, _| {
            bench.iter(|| {
                let dot: f32 = black_box(&a).iter().zip(black_box(&b).iter()).map(|(x, y)| x * y).sum();
                let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
                let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
                dot / (norm_a * norm_b)
            })
        });

        group.bench_with_input(BenchmarkId::new("bind", dim), dim, |bench, _| {
            bench.iter(|| {
                black_box(&a).iter().zip(black_box(&b).iter()).map(|(x, y)| x * y).collect::<Vec<f32>>()
            })
        });
    }

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════
// CRITERION CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════

criterion_group!(
    benches,
    bench_resonant_router,
    bench_resonant_causal,
    bench_resonant_byzantine,
    bench_resonant_pattern_matcher,
    bench_hdv_operations,
);

criterion_main!(benches);
