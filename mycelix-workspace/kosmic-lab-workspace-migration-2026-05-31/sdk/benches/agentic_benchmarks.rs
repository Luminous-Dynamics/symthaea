// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Agentic Framework Performance Benchmarks
//!
//! Benchmarks for AI agent management operations:
//! - K-Vector trust profile computation for agents
//! - Gaming detection and adversarial analysis
//! - Uncertainty gating and escalation
//! - Coherence (Phi) measurement
//!
//! Run with: `cargo bench --bench agentic_benchmarks --features simulation`

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};

use mycelix_sdk::agentic::adversarial::{SybilDetectionConfig, SybilDetector, SybilEvidence};
use mycelix_sdk::agentic::lifecycle::{
    UncertaintyCheckResult, gate_action_on_uncertainty, process_resolved_escalations,
};
use mycelix_sdk::agentic::{
    ActionOutcome, AgentClass, AgentConstraints, AgentId, AgentOutput, AgentStatus,
    BehaviorLogEntry, CoherenceState, EpistemicStats, EscalationRequest, GamingDetectionConfig,
    GamingDetector, InstrumentalActor, KVectorBridgeConfig, MoralActionGuidance, MoralUncertainty,
    MoralUncertaintyType, PhiMeasurementConfig, UncertaintyCalibration,
    calculate_kredit_from_trust, check_coherence_for_action, compute_trust_score,
    update_kvector_from_behavior,
};
use mycelix_sdk::matl::KVector;

// =============================================================================
// K-VECTOR TRUST PROFILE BENCHMARKS
// =============================================================================

/// Benchmark K-Vector trust score computation for agents
fn benchmark_kvector_trust_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("agentic_kvector_trust");

    // Single trust score computation
    group.bench_function("compute_trust_score", |b| {
        let kvector = create_test_kvector();
        let config = KVectorBridgeConfig::default();

        b.iter(|| compute_trust_score(black_box(&kvector), black_box(&config)))
    });

    // KREDIT calculation from trust
    group.bench_function("calculate_kredit_from_trust", |b| {
        let kvector = create_test_kvector();
        let config = KVectorBridgeConfig::default();
        let trust = compute_trust_score(&kvector, &config);

        b.iter(|| calculate_kredit_from_trust(black_box(trust)))
    });

    // Batch trust score computation
    for count in [10, 50, 100, 500] {
        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(
            BenchmarkId::new("batch_trust_scores", count),
            &count,
            |b, &count| {
                let kvectors: Vec<KVector> =
                    (0..count).map(|i| create_kvector_with_seed(i)).collect();
                let config = KVectorBridgeConfig::default();

                b.iter(|| {
                    kvectors
                        .iter()
                        .map(|kv| compute_trust_score(black_box(kv), black_box(&config)))
                        .collect::<Vec<_>>()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark K-Vector update from agent behavior
fn benchmark_kvector_behavior_update(c: &mut Criterion) {
    let mut group = c.benchmark_group("agentic_kvector_update");

    // Single behavior entry update
    group.bench_function("single_behavior_update", |b| {
        let mut agent = create_test_agent();
        let behavior = create_test_behavior(ActionOutcome::Success);
        let config = KVectorBridgeConfig::default();

        b.iter(|| {
            let mut a = agent.clone();
            update_kvector_from_behavior(
                black_box(&mut a),
                black_box(&behavior),
                black_box(&config),
            )
        })
    });

    // Batch behavior updates (simulating agent activity)
    for count in [10, 50, 100] {
        group.bench_with_input(
            BenchmarkId::new("batch_behavior_update", count),
            &count,
            |b, &count| {
                let mut agent = create_test_agent();
                let behaviors: Vec<BehaviorLogEntry> = (0..count)
                    .map(|i| {
                        if i % 10 == 0 {
                            create_test_behavior(ActionOutcome::Failure)
                        } else {
                            create_test_behavior(ActionOutcome::Success)
                        }
                    })
                    .collect();
                let config = KVectorBridgeConfig::default();

                b.iter(|| {
                    let mut a = agent.clone();
                    for behavior in &behaviors {
                        update_kvector_from_behavior(
                            black_box(&mut a),
                            black_box(behavior),
                            black_box(&config),
                        );
                    }
                    a.k_vector.trust_score()
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// GAMING DETECTION BENCHMARKS
// =============================================================================

/// Benchmark gaming detection analysis
fn benchmark_gaming_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("agentic_gaming_detection");

    // Single agent analysis
    group.bench_function("single_agent_analysis", |b| {
        let agent = create_test_agent_with_history(50);
        let mut detector = GamingDetector::new(GamingDetectionConfig::default());
        let now = current_timestamp();

        b.iter(|| detector.analyze(black_box(&agent), black_box(now)))
    });

    // Analysis with suspicious patterns
    group.bench_function("suspicious_agent_analysis", |b| {
        let agent = create_suspicious_agent();
        let mut detector = GamingDetector::new(GamingDetectionConfig::default());
        let now = current_timestamp();

        b.iter(|| detector.analyze(black_box(&agent), black_box(now)))
    });

    // Batch gaming detection
    for count in [10, 50, 100] {
        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(
            BenchmarkId::new("batch_detection", count),
            &count,
            |b, &count| {
                let agents: Vec<InstrumentalActor> = (0..count)
                    .map(|i| {
                        if i % 10 == 0 {
                            create_suspicious_agent()
                        } else {
                            create_test_agent_with_history(20)
                        }
                    })
                    .collect();
                let mut detector = GamingDetector::new(GamingDetectionConfig::default());
                let now = current_timestamp();

                b.iter(|| {
                    agents
                        .iter()
                        .map(|a| detector.analyze(black_box(a), black_box(now)))
                        .collect::<Vec<_>>()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark Sybil detection
fn benchmark_sybil_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("agentic_sybil_detection");

    // Small network
    group.bench_function("small_network_10_agents", |b| {
        let agents: Vec<InstrumentalActor> =
            (0..10).map(|i| create_kvector_agent_with_seed(i)).collect();
        let detector = SybilDetector::new(SybilDetectionConfig::default());

        b.iter(|| detector.detect(black_box(&agents)))
    });

    // Medium network
    group.bench_function("medium_network_50_agents", |b| {
        let agents: Vec<InstrumentalActor> =
            (0..50).map(|i| create_kvector_agent_with_seed(i)).collect();
        let detector = SybilDetector::new(SybilDetectionConfig::default());

        b.iter(|| detector.detect(black_box(&agents)))
    });

    // Network with Sybil cluster
    group.bench_function("sybil_cluster_detection", |b| {
        // Create 40 honest agents + 10 Sybil agents with similar K-Vectors
        let mut agents: Vec<InstrumentalActor> =
            (0..40).map(|i| create_kvector_agent_with_seed(i)).collect();

        // Add Sybil cluster (nearly identical K-Vectors)
        let sybil_base = create_kvector_agent_with_seed(1000);
        for i in 0..10 {
            let mut sybil = sybil_base.clone();
            sybil.agent_id = AgentId::from_string(format!("sybil-{}", i));
            // Slightly perturb to avoid exact duplicates
            sybil.k_vector = perturb_kvector(&sybil.k_vector, 0.01);
            agents.push(sybil);
        }

        let detector = SybilDetector::new(SybilDetectionConfig::default());

        b.iter(|| detector.detect(black_box(&agents)))
    });

    group.finish();
}

// =============================================================================
// UNCERTAINTY GATING BENCHMARKS
// =============================================================================

/// Benchmark moral uncertainty gating
fn benchmark_uncertainty_gating(c: &mut Criterion) {
    let mut group = c.benchmark_group("agentic_uncertainty_gating");

    // Low uncertainty (proceed)
    group.bench_function("low_uncertainty", |b| {
        let agent = create_test_agent();
        let uncertainty = MoralUncertainty::new(0.1, 0.15, 0.1);

        b.iter(|| {
            gate_action_on_uncertainty(
                black_box(&agent),
                black_box(&uncertainty),
                black_box("test_action"),
                None,
            )
        })
    });

    // High uncertainty (escalation)
    group.bench_function("high_uncertainty", |b| {
        let agent = create_test_agent();
        let uncertainty = MoralUncertainty::new(0.7, 0.65, 0.75);

        b.iter(|| {
            gate_action_on_uncertainty(
                black_box(&agent),
                black_box(&uncertainty),
                black_box("test_action"),
                None,
            )
        })
    });

    // Batch uncertainty checks
    for count in [10, 50, 100] {
        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(
            BenchmarkId::new("batch_uncertainty_check", count),
            &count,
            |b, &count| {
                let agent = create_test_agent();
                let uncertainties: Vec<MoralUncertainty> = (0..count)
                    .map(|i| {
                        let base = (i as f32 % 10) / 10.0;
                        MoralUncertainty::new(base, base + 0.1, base + 0.05)
                    })
                    .collect();

                b.iter(|| {
                    uncertainties
                        .iter()
                        .map(|u| {
                            gate_action_on_uncertainty(
                                black_box(&agent),
                                black_box(u),
                                black_box("test_action"),
                                None,
                            )
                        })
                        .collect::<Vec<_>>()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark escalation resolution
fn benchmark_escalation_resolution(c: &mut Criterion) {
    let mut group = c.benchmark_group("agentic_escalation_resolution");

    // Single escalation resolution
    group.bench_function("single_resolution", |b| {
        let mut agent = create_agent_with_escalations(1);
        let resolutions = vec![("action_0".to_string(), true)];

        b.iter(|| {
            let mut a = agent.clone();
            process_resolved_escalations(black_box(&mut a), black_box(&resolutions))
        })
    });

    // Batch escalation resolution
    for count in [5, 10, 20] {
        group.bench_with_input(
            BenchmarkId::new("batch_resolution", count),
            &count,
            |b, &count| {
                let mut agent = create_agent_with_escalations(*count);
                let resolutions: Vec<(String, bool)> = (0..*count)
                    .map(|i| (format!("action_{}", i), i % 2 == 0))
                    .collect();

                b.iter(|| {
                    let mut a = agent.clone();
                    process_resolved_escalations(black_box(&mut a), black_box(&resolutions))
                })
            },
        );
    }

    group.finish();
}

/// Benchmark calibration tracking
fn benchmark_calibration_tracking(c: &mut Criterion) {
    let mut group = c.benchmark_group("agentic_calibration");

    // Single calibration record
    group.bench_function("record_event", |b| {
        let mut calibration = UncertaintyCalibration::default();

        b.iter(|| {
            let mut c = calibration.clone();
            c.record(black_box(true), black_box(true))
        })
    });

    // Calibration score computation
    group.bench_function("compute_score", |b| {
        let mut calibration = UncertaintyCalibration::default();
        // Simulate some history
        for i in 0..100 {
            calibration.record(i % 3 == 0, i % 4 != 0);
        }

        b.iter(|| black_box(&calibration).calibration_score())
    });

    // Bias detection
    group.bench_function("detect_bias", |b| {
        let mut calibration = UncertaintyCalibration::default();
        for i in 0..100 {
            calibration.record(i % 3 == 0, i % 4 != 0);
        }

        b.iter(|| {
            let c = black_box(&calibration);
            (c.is_overconfident(), c.is_overcautious())
        })
    });

    group.finish();
}

// =============================================================================
// COHERENCE (PHI) BENCHMARKS
// =============================================================================

/// Benchmark coherence checks
fn benchmark_coherence_checks(c: &mut Criterion) {
    let mut group = c.benchmark_group("agentic_coherence");

    // Coherence state check
    group.bench_function("check_coherence", |b| {
        let state = CoherenceState {
            phi: 0.75,
            is_coherent: true,
            trend: 0.02,
            last_measured: current_timestamp(),
        };

        b.iter(|| check_coherence_for_action(black_box(&state), black_box(true)))
    });

    // Low coherence check
    group.bench_function("check_low_coherence", |b| {
        let state = CoherenceState {
            phi: 0.3,
            is_coherent: false,
            trend: -0.05,
            last_measured: current_timestamp(),
        };

        b.iter(|| check_coherence_for_action(black_box(&state), black_box(true)))
    });

    group.finish();
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Create a test K-Vector
fn create_test_kvector() -> KVector {
    KVector::new(0.8, 0.6, 0.9, 0.7, 0.5, 0.4, 0.6, 0.5, 0.75, 0.7)
}

/// Create a K-Vector with seed for variation
fn create_kvector_with_seed(seed: usize) -> KVector {
    let base = rand_float(seed);
    KVector::new(
        (base + 0.1).min(1.0),
        (base + 0.05).min(1.0),
        (base + 0.15).min(1.0),
        (base + 0.02).min(1.0),
        (base - 0.1).max(0.0),
        (base - 0.05).max(0.0),
        (base + 0.08).min(1.0),
        (base - 0.2).max(0.0),
        (base + 0.12).min(1.0),
        (base + 0.07).min(1.0),
    )
}

/// Perturb a K-Vector slightly
fn perturb_kvector(kv: &KVector, amount: f32) -> KVector {
    let arr = kv.to_array();
    let perturbed: [f32; 10] = [
        (arr[0] + amount).clamp(0.0, 1.0),
        (arr[1] - amount).clamp(0.0, 1.0),
        (arr[2] + amount * 0.5).clamp(0.0, 1.0),
        (arr[3] - amount * 0.5).clamp(0.0, 1.0),
        arr[4],
        arr[5],
        (arr[6] + amount).clamp(0.0, 1.0),
        arr[7],
        arr[8],
        arr[9],
    ];
    KVector::from_array(perturbed)
}

/// Create a test agent
fn create_test_agent() -> InstrumentalActor {
    InstrumentalActor {
        agent_id: AgentId::from_string("test-agent".to_string()),
        sponsor_did: "did:test:sponsor".to_string(),
        agent_class: AgentClass::Supervised,
        kredit_balance: 5000,
        kredit_cap: 10000,
        constraints: AgentConstraints::default(),
        behavior_log: vec![],
        status: AgentStatus::Active,
        created_at: current_timestamp() - 86400,
        last_activity: current_timestamp(),
        actions_this_hour: 10,
        k_vector: create_test_kvector(),
        epistemic_stats: EpistemicStats::default(),
        output_history: vec![],
        uncertainty_calibration: UncertaintyCalibration::default(),
        pending_escalations: vec![],
    }
}

/// Create a test agent with behavior history
fn create_test_agent_with_history(history_size: usize) -> InstrumentalActor {
    let mut agent = create_test_agent();
    agent.behavior_log = (0..history_size)
        .map(|i| {
            if i % 10 == 0 {
                create_test_behavior(ActionOutcome::Failure)
            } else {
                create_test_behavior(ActionOutcome::Success)
            }
        })
        .collect();
    agent
}

/// Create an agent that looks suspicious (gaming patterns)
fn create_suspicious_agent() -> InstrumentalActor {
    let mut agent = create_test_agent();
    // Perfect success rate (suspicious)
    agent.behavior_log = (0..50)
        .map(|_| create_test_behavior(ActionOutcome::Success))
        .collect();
    // Constant timing (bot-like)
    let base_time = current_timestamp() - 5000;
    for (i, entry) in agent.behavior_log.iter_mut().enumerate() {
        entry.timestamp = base_time + (i as u64 * 100); // Exact 100s intervals
    }
    agent
}

/// Create an agent with K-Vector from seed
fn create_kvector_agent_with_seed(seed: usize) -> InstrumentalActor {
    let mut agent = create_test_agent();
    agent.agent_id = AgentId::from_string(format!("agent-{}", seed));
    agent.k_vector = create_kvector_with_seed(seed);
    agent
}

/// Create agent with pending escalations
fn create_agent_with_escalations(count: usize) -> InstrumentalActor {
    let mut agent = create_test_agent();
    agent.pending_escalations = (0..count)
        .map(|i| EscalationRequest {
            agent_id: agent.agent_id.as_str().to_string(),
            blocked_action: format!("action_{}", i),
            uncertainty: MoralUncertainty::new(0.6, 0.5, 0.7),
            guidance: MoralActionGuidance::SeekConsultation,
            recommendations: vec!["Review action".to_string()],
            context: None,
            timestamp: current_timestamp(),
        })
        .collect();
    agent
}

/// Create a test behavior log entry
fn create_test_behavior(outcome: ActionOutcome) -> BehaviorLogEntry {
    BehaviorLogEntry {
        action_type: "test_action".to_string(),
        timestamp: current_timestamp(),
        outcome,
        kredit_delta: match outcome {
            ActionOutcome::Success => -10,
            ActionOutcome::Failure => -5,
            ActionOutcome::Reverted => 0,
        },
        context: None,
    }
}

/// Get current timestamp
fn current_timestamp() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Simple deterministic pseudo-random
fn rand_float(seed: usize) -> f32 {
    let x = seed.wrapping_mul(1103515245).wrapping_add(12345);
    (x % 1000) as f32 / 1000.0
}

// =============================================================================
// CRITERION GROUPS
// =============================================================================

criterion_group!(
    name = kvector_benchmarks;
    config = Criterion::default().sample_size(100);
    targets = benchmark_kvector_trust_computation, benchmark_kvector_behavior_update
);

criterion_group!(
    name = gaming_benchmarks;
    config = Criterion::default().sample_size(50);
    targets = benchmark_gaming_detection, benchmark_sybil_detection
);

criterion_group!(
    name = uncertainty_benchmarks;
    config = Criterion::default().sample_size(100);
    targets = benchmark_uncertainty_gating, benchmark_escalation_resolution, benchmark_calibration_tracking
);

criterion_group!(
    name = coherence_benchmarks;
    config = Criterion::default().sample_size(100);
    targets = benchmark_coherence_checks
);

criterion_main!(
    kvector_benchmarks,
    gaming_benchmarks,
    uncertainty_benchmarks,
    coherence_benchmarks
);
