// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Performance benchmarks for the Epistemic-Aware AI Agency framework
//!
//! Run with: `cargo bench`
//!
//! Benchmarks:
//! - K-Vector updates and trust computation
//! - Phi coherence measurement
//! - Storage routing decisions
//! - Byzantine/cartel detection

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use mycelix_sdk::agentic::{
    analyze_behavior, calculate_kredit_from_trust, compute_kvector_update, compute_trust_score,
    measure_coherence, ActionOutcome, AgentClass, AgentConstraints, AgentId, AgentOutput,
    AgentOutputBuilder, AgentStatus, CoherenceMeasurementConfig, InstrumentalActor,
    KVectorBridgeConfig, OutputContent,
};
use mycelix_sdk::epistemic::{
    EmpiricalLevel, EpistemicClassification, HarmonicLevel, MaterialityLevel, NormativeLevel,
};
use mycelix_sdk::matl::CartelDetector;
use mycelix_sdk::matl::KVector;
use mycelix_sdk::storage::StorageRouter;

/// Benchmark K-Vector trust score computation
fn bench_kvector_trust_score(c: &mut Criterion) {
    let kvector = KVector {
        k_r: 0.8,
        k_a: 0.6,
        k_i: 0.9,
        k_p: 0.7,
        k_m: 0.5,
        k_s: 0.4,
        k_h: 0.6,
        k_topo: 0.3,
        k_v: 0.7, // Verification dimension
    };

    c.bench_function("kvector_trust_score", |b| {
        b.iter(|| compute_trust_score(black_box(&kvector)))
    });
}

/// Benchmark K-Vector update from behavior analysis
fn bench_kvector_update(c: &mut Criterion) {
    let mut group = c.benchmark_group("kvector_update");

    for action_count in [10, 50, 100, 500].iter() {
        let mut agent = create_test_agent();

        // Record actions
        for i in 0..*action_count {
            let outcome = if i % 10 == 0 {
                ActionOutcome::Error
            } else {
                ActionOutcome::Success
            };
            agent.record_action("test_action", 10, outcome);
        }

        let config = KVectorBridgeConfig::default();
        let analysis = analyze_behavior(&agent.behavior_log);

        group.bench_with_input(
            BenchmarkId::new("actions", action_count),
            action_count,
            |b, _| {
                b.iter(|| {
                    compute_kvector_update(
                        black_box(&agent.k_vector),
                        black_box(&analysis),
                        black_box(&config),
                        black_box(7.0),
                    )
                })
            },
        );
    }

    group.finish();
}

/// Benchmark KREDIT derivation from trust
fn bench_kredit_derivation(c: &mut Criterion) {
    c.bench_function("kredit_from_trust", |b| {
        b.iter(|| calculate_kredit_from_trust(black_box(0.75)))
    });
}

/// Benchmark Phi coherence measurement with varying output counts
fn bench_phi_measurement(c: &mut Criterion) {
    let mut group = c.benchmark_group("phi_measurement");
    let config = CoherenceMeasurementConfig::default();

    for output_count in [5, 10, 20, 50].iter() {
        let outputs = create_test_outputs(*output_count);

        group.bench_with_input(
            BenchmarkId::new("outputs", output_count),
            output_count,
            |b, _| b.iter(|| measure_coherence(black_box(&outputs), black_box(&config))),
        );
    }

    group.finish();
}

/// Benchmark storage routing decisions
fn bench_storage_routing(c: &mut Criterion) {
    let router = StorageRouter::default_router();

    let classifications = vec![
        EpistemicClassification {
            empirical: EmpiricalLevel::E0Null,
            normative: NormativeLevel::N0Personal,
            materiality: MaterialityLevel::M0Ephemeral,
        },
        EpistemicClassification {
            empirical: EmpiricalLevel::E2PrivateVerify,
            normative: NormativeLevel::N1Communal,
            materiality: MaterialityLevel::M2Persistent,
        },
        EpistemicClassification {
            empirical: EmpiricalLevel::E4PublicRepro,
            normative: NormativeLevel::N3Axiomatic,
            materiality: MaterialityLevel::M3Foundational,
        },
    ];

    c.bench_function("storage_routing", |b| {
        b.iter(|| {
            for class in &classifications {
                black_box(router.route(black_box(class)));
            }
        })
    });
}

/// Benchmark cartel detection with varying node counts
fn bench_cartel_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("cartel_detection");

    for node_count in [10, 50, 100, 200].iter() {
        let mut detector = CartelDetector::new(0.8, 3);

        // Create similarity matrix
        for i in 0..*node_count {
            for j in (i + 1)..*node_count {
                // Create a cartel pattern: nodes 0-4 are highly similar
                let similarity = if i < 5 && j < 5 {
                    0.9 + (rand_float(i * j) * 0.1)
                } else {
                    rand_float(i * j) * 0.5
                };
                detector.record_similarity(
                    &format!("node_{}", i),
                    &format!("node_{}", j),
                    similarity,
                );
            }
        }

        group.bench_with_input(BenchmarkId::new("nodes", node_count), node_count, |b, _| {
            b.iter(|| {
                let mut d = detector.clone();
                d.detect()
            })
        });
    }

    group.finish();
}

/// Benchmark behavior analysis
fn bench_behavior_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("behavior_analysis");

    for action_count in [10, 50, 100, 500, 1000].iter() {
        let mut agent = create_test_agent();

        for i in 0..*action_count {
            let outcome = match i % 20 {
                0 => ActionOutcome::Error,
                1 => ActionOutcome::ConstraintViolation,
                _ => ActionOutcome::Success,
            };
            agent.record_action("action", 10, outcome);
            if let Some(entry) = agent.behavior_log.last_mut() {
                entry.counterparties = vec![format!("user_{}", i % 50)];
            }
        }

        group.bench_with_input(
            BenchmarkId::new("actions", action_count),
            action_count,
            |b, _| b.iter(|| analyze_behavior(black_box(&agent.behavior_log))),
        );
    }

    group.finish();
}

// Helper functions

fn create_test_agent() -> InstrumentalActor {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    InstrumentalActor {
        agent_id: AgentId::from_string("bench-agent".to_string()),
        sponsor_did: "did:mycelix:sponsor".to_string(),
        agent_class: AgentClass::Supervised,
        kredit_balance: 5000,
        kredit_cap: 10000,
        constraints: AgentConstraints::default(),
        behavior_log: vec![],
        status: AgentStatus::Active,
        created_at: now,
        last_activity: now,
        actions_this_hour: 0,
        k_vector: KVector::new_participant(),
    }
}

fn create_test_outputs(count: usize) -> Vec<AgentOutput> {
    let levels_e = [
        EmpiricalLevel::E0Null,
        EmpiricalLevel::E1Testimonial,
        EmpiricalLevel::E2PrivateVerify,
        EmpiricalLevel::E3Cryptographic,
    ];
    let levels_n = [
        NormativeLevel::N0Personal,
        NormativeLevel::N1Communal,
        NormativeLevel::N2Network,
    ];
    let levels_m = [
        MaterialityLevel::M0Ephemeral,
        MaterialityLevel::M1Temporal,
        MaterialityLevel::M2Persistent,
    ];

    (0..count)
        .map(|i| {
            AgentOutputBuilder::new("bench-agent")
                .content(OutputContent::Text(format!("output_{}", i)))
                .classification(
                    levels_e[i % levels_e.len()],
                    levels_n[i % levels_n.len()],
                    levels_m[i % levels_m.len()],
                    HarmonicLevel::H0None,
                )
                .build()
                .expect("Failed to build output")
        })
        .collect()
}

/// Simple deterministic pseudo-random for benchmarks
fn rand_float(seed: usize) -> f64 {
    let x = seed.wrapping_mul(1103515245).wrapping_add(12345);
    (x % 1000) as f64 / 1000.0
}

criterion_group!(
    benches,
    bench_kvector_trust_score,
    bench_kvector_update,
    bench_kredit_derivation,
    bench_phi_measurement,
    bench_storage_routing,
    bench_cartel_detection,
    bench_behavior_analysis,
);

criterion_main!(benches);
