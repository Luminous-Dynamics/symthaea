// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Agentic Framework Performance Benchmarks
//!
//! Benchmarks for AI agent management operations:
//! - K-Vector trust profile computation for agents
//! - Gaming detection and adversarial analysis
//! - Uncertainty gating and escalation
//! - Coherence (Phi) measurement
//!
//! Run with: `cargo bench --bench agentic_benchmarks --features simulation`

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use mycelix_sdk::agentic::adversarial::{SybilDetectionConfig, SybilDetector, SybilEvidence};
use mycelix_sdk::agentic::lifecycle::{
    gate_action_on_uncertainty, process_resolved_escalations, UncertaintyCheckResult,
};
use mycelix_sdk::agentic::{
    calculate_kredit_from_trust, check_coherence_for_action, compute_trust_score,
    update_kvector_from_behavior, ActionOutcome, AgentClass, AgentConstraints, AgentId,
    AgentOutput, AgentStatus, BehaviorLogEntry, CoherenceMeasurementConfig, CoherenceState,
    EpistemicStats, EscalationRequest, GamingDetectionConfig, GamingDetector, InstrumentalActor,
    KVectorBridgeConfig, MoralActionGuidance, MoralUncertainty, MoralUncertaintyType,
    UncertaintyCalibration,
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

// =============================================================================
// DASHBOARD BENCHMARKS
// =============================================================================

use mycelix_sdk::agentic::dashboard::{
    AlertPanel, AlertSeverity, ChartDataBuilder, ChartType, Dashboard, DashboardAlert,
    DashboardConfig, DashboardEventType, EventPriority, EventStream, MetricsAggregator,
    MetricsInput, TimeSeries,
};

/// Benchmark dashboard metrics aggregation
fn benchmark_dashboard_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("agentic_dashboard_metrics");

    // Single metrics update
    group.bench_function("single_update", |b| {
        let config = DashboardConfig::default();
        let mut agg = MetricsAggregator::new(config);
        let input = create_metrics_input(10);

        b.iter(|| {
            let mut a = agg.clone();
            a.update(black_box(input.clone()), black_box(current_timestamp()))
        })
    });

    // Batch metrics updates
    for count in [10, 50, 100] {
        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(
            BenchmarkId::new("batch_update", count),
            &count,
            |b, &count| {
                let config = DashboardConfig::default();
                let mut agg = MetricsAggregator::new(config);
                let inputs: Vec<MetricsInput> =
                    (0..count).map(|i| create_metrics_input(10 + i)).collect();

                b.iter(|| {
                    let mut a = agg.clone();
                    for (i, input) in inputs.iter().enumerate() {
                        a.update(black_box(input.clone()), black_box(1000 + i as u64 * 60));
                    }
                    a.latest().cloned()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark alert panel operations
fn benchmark_alert_panel(c: &mut Criterion) {
    let mut group = c.benchmark_group("agentic_alert_panel");

    // Create alert
    group.bench_function("create_alert", |b| {
        let config = DashboardConfig::default();
        let mut panel = AlertPanel::new(config);

        b.iter(|| {
            let mut p = panel.clone();
            p.create_alert(
                black_box(AlertSeverity::High),
                black_box("Test Alert"),
                black_box("Description"),
                black_box("test"),
            )
        })
    });

    // Get active alerts with many alerts
    group.bench_function("get_active_100_alerts", |b| {
        let config = DashboardConfig::default();
        let mut panel = AlertPanel::new(config);
        for i in 0..100 {
            panel.create_alert(
                AlertSeverity::Medium,
                &format!("Alert {}", i),
                "Description",
                "test",
            );
        }

        b.iter(|| black_box(&panel).active_alerts())
    });

    // Critical alerts filter
    group.bench_function("get_critical_alerts", |b| {
        let config = DashboardConfig::default();
        let mut panel = AlertPanel::new(config);
        for i in 0..100 {
            let severity = if i % 10 == 0 {
                AlertSeverity::Critical
            } else {
                AlertSeverity::Low
            };
            panel.create_alert(severity, &format!("Alert {}", i), "Description", "test");
        }

        b.iter(|| black_box(&panel).critical_alerts())
    });

    group.finish();
}

/// Benchmark event stream operations
fn benchmark_event_stream(c: &mut Criterion) {
    let mut group = c.benchmark_group("agentic_event_stream");

    // Emit event
    group.bench_function("emit_event", |b| {
        let config = DashboardConfig::default();
        let mut stream = EventStream::new(config);

        b.iter(|| {
            let mut s = stream.clone();
            s.emit(
                black_box(DashboardEventType::TrustUpdate),
                black_box(EventPriority::Medium),
                black_box("agent-1"),
                black_box("Trust updated"),
            )
        })
    });

    // Filter by priority
    group.bench_function("filter_by_priority", |b| {
        let config = DashboardConfig::default();
        let mut stream = EventStream::new(config);
        for i in 0..100 {
            let priority = match i % 4 {
                0 => EventPriority::Critical,
                1 => EventPriority::High,
                2 => EventPriority::Medium,
                _ => EventPriority::Low,
            };
            stream.emit(
                DashboardEventType::Custom,
                priority,
                &format!("src-{}", i),
                "Event",
            );
        }

        b.iter(|| black_box(&stream).by_priority(black_box(EventPriority::High)))
    });

    group.finish();
}

/// Benchmark chart data building
fn benchmark_chart_building(c: &mut Criterion) {
    let mut group = c.benchmark_group("agentic_chart_building");

    // Build trust over time chart
    group.bench_function("trust_over_time_100_points", |b| {
        let config = DashboardConfig::default();
        let mut agg = MetricsAggregator::new(config);

        for i in 0..100 {
            let input = create_metrics_input(10);
            agg.update(input, 1000 + i * 60);
        }

        b.iter(|| ChartDataBuilder::trust_over_time(black_box(agg.history())))
    });

    // Build multiple charts
    group.bench_function("build_all_charts", |b| {
        let config = DashboardConfig::default();
        let mut agg = MetricsAggregator::new(config);

        for i in 0..100 {
            let input = create_metrics_input(10);
            agg.update(input, 1000 + i * 60);
        }

        b.iter(|| {
            let h = agg.history();
            let _ = ChartDataBuilder::trust_over_time(black_box(h));
            let _ = ChartDataBuilder::health_over_time(black_box(h));
            let _ = ChartDataBuilder::tps_over_time(black_box(h));
            ChartDataBuilder::alert_trend(black_box(h))
        })
    });

    group.finish();
}

// =============================================================================
// VERIFICATION BENCHMARKS
// =============================================================================

use mycelix_sdk::agentic::verification::{
    CompareOp, Invariant, InvariantType, SystemState, VerificationEngine,
    ViolationSeverity as VerifSeverity,
};

/// Benchmark verification engine operations
fn benchmark_verification_engine(c: &mut Criterion) {
    let mut group = c.benchmark_group("agentic_verification");

    // Check invariants with small state
    group.bench_function("check_invariants_10_agents", |b| {
        let mut engine = VerificationEngine::with_defaults();
        let state = create_system_state(10, 0);

        b.iter(|| {
            let mut e = engine.clone();
            e.check_invariants(black_box(&state))
        })
    });

    // Check invariants with larger state
    group.bench_function("check_invariants_100_agents", |b| {
        let mut engine = VerificationEngine::with_defaults();
        let state = create_system_state(100, 0);

        b.iter(|| {
            let mut e = engine.clone();
            e.check_invariants(black_box(&state))
        })
    });

    // Check with violations
    group.bench_function("check_with_violations", |b| {
        let mut engine = VerificationEngine::with_defaults();
        let state = create_system_state(50, 30); // 60% Byzantine

        b.iter(|| {
            let mut e = engine.clone();
            e.check_invariants(black_box(&state))
        })
    });

    // Batch state checks
    for count in [10, 50] {
        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(
            BenchmarkId::new("batch_checks", count),
            &count,
            |b, &count| {
                let mut engine = VerificationEngine::with_defaults();
                let states: Vec<SystemState> =
                    (0..count).map(|i| create_system_state(20, i % 3)).collect();

                b.iter(|| {
                    let mut e = engine.clone();
                    for state in &states {
                        e.check_invariants(black_box(state));
                    }
                    e.summary()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark comparison operator evaluations
fn benchmark_compare_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("agentic_compare_ops");

    // Float comparisons
    group.bench_function("float_comparison_batch", |b| {
        let pairs: Vec<(f64, f64)> = (0..1000)
            .map(|i| (rand_float(i) as f64, rand_float(i + 1000) as f64))
            .collect();

        b.iter(|| {
            pairs
                .iter()
                .map(|(a, b)| CompareOp::Lt.eval(black_box(*a), black_box(*b)))
                .filter(|&x| x)
                .count()
        })
    });

    // Integer comparisons
    group.bench_function("usize_comparison_batch", |b| {
        let pairs: Vec<(usize, usize)> = (0..1000).map(|i| (i * 17 % 100, i * 31 % 100)).collect();

        b.iter(|| {
            pairs
                .iter()
                .map(|(a, b)| CompareOp::Lt.eval_usize(black_box(*a), black_box(*b)))
                .filter(|&x| x)
                .count()
        })
    });

    group.finish();
}

// =============================================================================
// INTEGRATION BENCHMARKS
// =============================================================================

use mycelix_sdk::agentic::epistemic_classifier::OutputContent;
use mycelix_sdk::agentic::integration::{
    AttackResponseConfig, EpistemicLifecycleConfig, IntegratedAttackResponse,
    IntegratedEpistemicLifecycle, IntegratedTrustPipeline, TrustPipelineConfig,
};

/// Benchmark trust pipeline operations
fn benchmark_trust_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("agentic_trust_pipeline");

    // Register agent
    group.bench_function("register_agent", |b| {
        let config = TrustPipelineConfig::default();
        let mut pipeline = IntegratedTrustPipeline::new(config.clone());
        let agent = create_test_agent();

        b.iter(|| {
            let mut p = IntegratedTrustPipeline::new(config.clone());
            p.register_agent(black_box(agent.clone()))
        })
    });

    // Process attestation
    group.bench_function("process_attestation", |b| {
        let config = TrustPipelineConfig::default();
        let mut pipeline = IntegratedTrustPipeline::new(config);

        for i in 0..10 {
            let agent = create_kvector_agent_with_seed(i);
            pipeline.register_agent(agent);
        }

        b.iter(|| {
            let mut p = pipeline.clone();
            p.process_attestation(black_box("agent-0"), black_box("agent-1"), black_box(0.8))
        })
    });

    // Verify invariants
    group.bench_function("verify_invariants", |b| {
        let config = TrustPipelineConfig::default();
        let mut pipeline = IntegratedTrustPipeline::new(config);

        for i in 0..20 {
            let agent = create_kvector_agent_with_seed(i);
            pipeline.register_agent(agent);
        }

        b.iter(|| {
            let mut p = pipeline.clone();
            p.verify_invariants()
        })
    });

    // Cascade simulation
    group.bench_function("simulate_cascade", |b| {
        let config = TrustPipelineConfig::default();
        let mut pipeline = IntegratedTrustPipeline::new(config);

        for i in 0..20 {
            let agent = create_kvector_agent_with_seed(i);
            pipeline.register_agent(agent);
        }

        // Create attestation chain
        for i in 0..19 {
            let _ = pipeline.process_attestation(
                &format!("agent-{}", i),
                &format!("agent-{}", i + 1),
                0.8,
            );
        }

        b.iter(|| {
            let mut p = pipeline.clone();
            p.simulate_cascade(black_box("agent-0"), black_box(0.5))
        })
    });

    group.finish();
}

/// Benchmark epistemic lifecycle operations
fn benchmark_epistemic_lifecycle(c: &mut Criterion) {
    let mut group = c.benchmark_group("agentic_epistemic_lifecycle");

    // Create agent
    group.bench_function("create_agent", |b| {
        let config = EpistemicLifecycleConfig::default();
        let lifecycle = IntegratedEpistemicLifecycle::new(config);

        b.iter(|| {
            lifecycle.create_agent(
                black_box("did:test:sponsor"),
                black_box(AgentClass::Supervised),
            )
        })
    });

    // Process output
    group.bench_function("process_output", |b| {
        let config = EpistemicLifecycleConfig::default();
        let lifecycle = IntegratedEpistemicLifecycle::new(config);
        let mut agent = lifecycle.create_agent("did:test:sponsor", AgentClass::Supervised);

        b.iter(|| {
            let mut a = agent.clone();
            lifecycle.process_output(
                black_box(&mut a),
                black_box(OutputContent::Text("Test output".to_string())),
            )
        })
    });

    // Batch output processing
    for count in [5, 10, 20] {
        group.bench_with_input(
            BenchmarkId::new("batch_outputs", count),
            &count,
            |b, &count| {
                let config = EpistemicLifecycleConfig::default();
                let lifecycle = IntegratedEpistemicLifecycle::new(config);
                let mut agent = lifecycle.create_agent("did:test:sponsor", AgentClass::Supervised);

                b.iter(|| {
                    let mut a = agent.clone();
                    for i in 0..count {
                        let _ = lifecycle.process_output(
                            black_box(&mut a),
                            black_box(OutputContent::Text(format!("Output {}", i))),
                        );
                    }
                    a.k_vector.trust_score()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark attack response operations
fn benchmark_attack_response(c: &mut Criterion) {
    let mut group = c.benchmark_group("agentic_attack_response");

    // Process normal behavior
    group.bench_function("process_normal_behavior", |b| {
        let config = AttackResponseConfig::default();
        let mut response = IntegratedAttackResponse::new(config);
        let agent = create_test_agent_with_history(20);

        b.iter(|| {
            let mut r = response.clone();
            let mut a = agent.clone();
            r.process_behavior(black_box(&mut a))
        })
    });

    // Process suspicious behavior
    group.bench_function("process_suspicious_behavior", |b| {
        let config = AttackResponseConfig::default();
        let mut response = IntegratedAttackResponse::new(config);
        let agent = create_suspicious_agent();

        b.iter(|| {
            let mut r = response.clone();
            let mut a = agent.clone();
            r.process_behavior(black_box(&mut a))
        })
    });

    group.finish();
}

// =============================================================================
// ADDITIONAL HELPER FUNCTIONS
// =============================================================================

/// Create metrics input for dashboard testing
fn create_metrics_input(seed: usize) -> MetricsInput {
    let trust_scores: Vec<f64> = (0..10).map(|i| rand_float(seed + i) as f64).collect();
    let phi_values: Vec<f64> = (0..3).map(|i| rand_float(seed + i + 100) as f64).collect();

    MetricsInput {
        trust_scores,
        transaction_count: seed % 100,
        alerts: vec![],
        phi_values,
        threats: vec![rand_float(seed + 200) as f64 * 0.3],
    }
}

/// Create a system state for verification testing
fn create_system_state(num_agents: usize, byzantine_count: usize) -> SystemState {
    let mut trust_scores = std::collections::HashMap::new();
    for i in 0..num_agents {
        trust_scores.insert(format!("agent-{}", i), rand_float(i) as f64);
    }

    SystemState {
        index: 0,
        timestamp: current_timestamp(),
        trust_scores,
        byzantine_count,
        network_health: 0.9,
        variables: std::collections::HashMap::new(),
    }
}

impl Clone for IntegratedTrustPipeline {
    fn clone(&self) -> Self {
        // Create a fresh pipeline for benchmarking
        // (Full clone not needed, just re-create with same config)
        let config = TrustPipelineConfig::default();
        let mut new_pipeline = IntegratedTrustPipeline::new(config);
        for (id, agent) in self.agents() {
            new_pipeline.register_agent(agent.clone());
        }
        new_pipeline
    }
}

impl Clone for IntegratedAttackResponse {
    fn clone(&self) -> Self {
        IntegratedAttackResponse::new(AttackResponseConfig::default())
    }
}

impl Clone for MetricsAggregator {
    fn clone(&self) -> Self {
        MetricsAggregator::new(DashboardConfig::default())
    }
}

impl Clone for AlertPanel {
    fn clone(&self) -> Self {
        AlertPanel::new(DashboardConfig::default())
    }
}

impl Clone for EventStream {
    fn clone(&self) -> Self {
        EventStream::new(DashboardConfig::default())
    }
}

impl Clone for VerificationEngine {
    fn clone(&self) -> Self {
        VerificationEngine::with_defaults()
    }
}

// =============================================================================
// ZK TRUST BENCHMARKS
// =============================================================================

use mycelix_sdk::agentic::integration::{ZKIntegratedPipeline, ZKTrustConfig};
use mycelix_sdk::agentic::zk_trust::ProofStatement;
use mycelix_sdk::matl::KVectorDimension;

/// Benchmark ZK proof generation
fn benchmark_zk_proof_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("agentic_zk_proof_generation");

    // Simple threshold proof
    group.bench_function("threshold_proof", |b| {
        let config = ZKTrustConfig::default();
        let mut pipeline = ZKIntegratedPipeline::new(config);
        let agent = create_test_agent();
        pipeline.register_agent_with_commitment(agent);

        b.iter(|| {
            let mut p = pipeline.clone();
            p.generate_trust_proof(
                black_box("test-agent"),
                black_box(ProofStatement::TrustExceedsThreshold { threshold: 0.5 }),
            )
        })
    });

    // Dimension proof
    group.bench_function("dimension_proof", |b| {
        let config = ZKTrustConfig::default();
        let mut pipeline = ZKIntegratedPipeline::new(config);
        let agent = create_test_agent();
        pipeline.register_agent_with_commitment(agent);

        b.iter(|| {
            let mut p = pipeline.clone();
            p.generate_trust_proof(
                black_box("test-agent"),
                black_box(ProofStatement::DimensionExceedsThreshold {
                    dimension: KVectorDimension::Integrity,
                    threshold: 0.5,
                }),
            )
        })
    });

    // Compound AND proof
    group.bench_function("compound_and_proof", |b| {
        let config = ZKTrustConfig::default();
        let mut pipeline = ZKIntegratedPipeline::new(config);
        let agent = create_test_agent();
        pipeline.register_agent_with_commitment(agent);

        let statement = ProofStatement::And(vec![
            ProofStatement::TrustExceedsThreshold { threshold: 0.5 },
            ProofStatement::DimensionExceedsThreshold {
                dimension: KVectorDimension::Integrity,
                threshold: 0.5,
            },
            ProofStatement::IsVerified,
        ]);

        b.iter(|| {
            let mut p = pipeline.clone();
            p.generate_trust_proof(black_box("test-agent"), black_box(statement.clone()))
        })
    });

    // WellFormed proof
    group.bench_function("wellformed_proof", |b| {
        let config = ZKTrustConfig::default();
        let mut pipeline = ZKIntegratedPipeline::new(config);
        let agent = create_test_agent();
        pipeline.register_agent_with_commitment(agent);

        b.iter(|| {
            let mut p = pipeline.clone();
            p.generate_trust_proof(
                black_box("test-agent"),
                black_box(ProofStatement::WellFormed),
            )
        })
    });

    group.finish();
}

/// Benchmark ZK attestation processing
fn benchmark_zk_attestation(c: &mut Criterion) {
    let mut group = c.benchmark_group("agentic_zk_attestation");

    // Single attestation
    group.bench_function("single_attestation", |b| {
        let config = ZKTrustConfig::default();
        let mut pipeline = ZKIntegratedPipeline::new(config);

        let agent1 = create_kvector_agent_with_seed(0);
        let agent2 = create_kvector_agent_with_seed(1);
        pipeline.register_agent_with_commitment(agent1);
        pipeline.register_agent_with_commitment(agent2);

        b.iter(|| {
            let mut p = pipeline.clone();
            p.process_zk_attestation(black_box("agent-0"), black_box("agent-1"), black_box(0.8))
        })
    });

    group.finish();
}

/// Benchmark ZK improvement proofs
fn benchmark_zk_improvement(c: &mut Criterion) {
    let mut group = c.benchmark_group("agentic_zk_improvement");

    // Improvement proof
    group.bench_function("improvement_proof", |b| {
        let config = ZKTrustConfig::default();
        let mut pipeline = ZKIntegratedPipeline::new(config);
        let agent = create_test_agent();
        pipeline.register_agent_with_commitment(agent);

        let prev_kv = KVector::new(0.4, 0.3, 0.5, 0.4, 0.3, 0.2, 0.4, 0.3, 0.5, 0.4);

        b.iter(|| {
            let mut p = pipeline.clone();
            p.generate_improvement_proof(
                black_box("test-agent"),
                black_box(&prev_kv),
                black_box(1000),
            )
        })
    });

    group.finish();
}

/// Benchmark ZK aggregate proofs
fn benchmark_zk_aggregate(c: &mut Criterion) {
    let mut group = c.benchmark_group("agentic_zk_aggregate");

    // Aggregate 10 proofs
    for count in [5, 10, 20] {
        group.bench_with_input(
            BenchmarkId::new("aggregate_proofs", count),
            &count,
            |b, &count| {
                let config = ZKTrustConfig::default();
                let mut pipeline = ZKIntegratedPipeline::new(config);

                // Register agents and generate proofs
                let statement = ProofStatement::TrustExceedsThreshold { threshold: 0.5 };
                let mut proofs = Vec::new();

                for i in 0..count {
                    let agent = create_kvector_agent_with_seed(i);
                    pipeline.register_agent_with_commitment(agent);
                    let result =
                        pipeline.generate_trust_proof(&format!("agent-{}", i), statement.clone());
                    if let Ok(r) = result {
                        proofs.push(r.proof);
                    }
                }

                b.iter(|| {
                    pipeline.aggregate_trust_proofs(
                        black_box(proofs.clone()),
                        black_box(statement.clone()),
                    )
                })
            },
        );
    }

    group.finish();
}

/// Benchmark ZK network health computation
fn benchmark_zk_network_health(c: &mut Criterion) {
    let mut group = c.benchmark_group("agentic_zk_network_health");

    // Health with many agents
    for count in [10, 50, 100] {
        group.bench_with_input(
            BenchmarkId::new("network_health", count),
            &count,
            |b, &count| {
                let config = ZKTrustConfig::default();
                let mut pipeline = ZKIntegratedPipeline::new(config);

                for i in 0..*count {
                    let agent = create_kvector_agent_with_seed(i);
                    pipeline.register_agent_with_commitment(agent);
                    let _ = pipeline
                        .generate_trust_proof(&format!("agent-{}", i), ProofStatement::WellFormed);
                }

                b.iter(|| pipeline.zk_network_health())
            },
        );
    }

    group.finish();
}

impl Clone for ZKIntegratedPipeline {
    fn clone(&self) -> Self {
        let config = ZKTrustConfig::default();
        let mut new_pipeline = ZKIntegratedPipeline::new(config);
        for (id, agent) in self.matl_pipeline().trust_pipeline().agents() {
            new_pipeline.register_agent_with_commitment(agent.clone());
        }
        new_pipeline
    }
}

// =============================================================================
// CRITERION GROUPS
// =============================================================================

criterion_group!(
    name = dashboard_benchmarks;
    config = Criterion::default().sample_size(100);
    targets = benchmark_dashboard_metrics, benchmark_alert_panel, benchmark_event_stream, benchmark_chart_building
);

criterion_group!(
    name = verification_benchmarks;
    config = Criterion::default().sample_size(100);
    targets = benchmark_verification_engine, benchmark_compare_ops
);

criterion_group!(
    name = integration_benchmarks;
    config = Criterion::default().sample_size(50);
    targets = benchmark_trust_pipeline, benchmark_epistemic_lifecycle, benchmark_attack_response
);

criterion_group!(
    name = zk_trust_benchmarks;
    config = Criterion::default().sample_size(50);
    targets = benchmark_zk_proof_generation, benchmark_zk_attestation, benchmark_zk_improvement, benchmark_zk_aggregate, benchmark_zk_network_health
);

criterion_main!(
    kvector_benchmarks,
    gaming_benchmarks,
    uncertainty_benchmarks,
    coherence_benchmarks,
    dashboard_benchmarks,
    verification_benchmarks,
    integration_benchmarks,
    zk_trust_benchmarks
);
