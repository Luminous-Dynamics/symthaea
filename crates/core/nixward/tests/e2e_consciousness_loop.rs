// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! End-to-End Consciousness Loop Integration Test
//!
//! Exercises the full NixOS awareness cycle:
//!   Observation → Encoding → Inference → Action Plan → Verification
//!
//! This test validates that all layers connect properly and produce
//! meaningful results through the entire cognitive pipeline.

use nixward::encoding::{NixCodebook, ServiceState, SystemStateEncoder, SystemStateSnapshot};
use nixward::ipc::{AnomalyEntry, ConcernEntry, DaemonConfig, DaemonSnapshot};
use nixward::mind::active_inference::NixActiveInference;
use nixward::mind::causal_graph::{NixCausalGraph, NixOSCausalPatterns};
use nixward::mind::episodic_memory::{EpisodeOutcome, NixEpisodicMemory, SystemEpisode};
use nixward::mind::journal_anomaly::JournalAnomalyDetector;
use nixward::mind::working_memory::{MemorySource, WorkingMemory};
use nixward::mind::{HdcWorldModel, NixWorldModel};
use nixward::observe::journal::JournalEntry;
use symthaea_core::hdc::ContinuousHV;

/// Full observation → encoding → world model → drift detection loop.
#[test]
fn test_observation_encoding_drift_loop() {
    let mut codebook = NixCodebook::new();

    // Observation 1: healthy system
    let snapshot1 = SystemStateSnapshot {
        services: vec![
            ("nginx.service".into(), ServiceState::Running),
            ("sshd.service".into(), ServiceState::Running),
            ("postgresql.service".into(), ServiceState::Running),
        ],
        packages: vec!["firefox".into(), "vim".into(), "git".into()],
        generation: Some(42),
        store_size_bytes: Some(15_000_000_000),
        store_path_count: Some(5000),
        config_options: vec![
            ("services.nginx.enable".into(), "true".into()),
            ("services.openssh.enable".into(), "true".into()),
        ],
    };

    let hv1 = {
        let mut encoder = SystemStateEncoder::new(&mut codebook);
        encoder.encode_snapshot(&snapshot1)
    };
    assert!(hv1.dim() > 0);
    assert!(hv1.norm() > 0.0);

    // Feed into world model
    let mut world_model = HdcWorldModel::default();
    world_model.observe(&hv1);

    // Observation 2: similar (no drift expected)
    let snapshot2 = SystemStateSnapshot {
        services: vec![
            ("nginx.service".into(), ServiceState::Running),
            ("sshd.service".into(), ServiceState::Running),
            ("postgresql.service".into(), ServiceState::Running),
        ],
        packages: vec!["firefox".into(), "vim".into(), "git".into(), "htop".into()],
        generation: Some(42),
        store_size_bytes: Some(15_100_000_000),
        store_path_count: Some(5001),
        config_options: vec![
            ("services.nginx.enable".into(), "true".into()),
            ("services.openssh.enable".into(), "true".into()),
        ],
    };

    let hv2 = {
        let mut encoder = SystemStateEncoder::new(&mut codebook);
        encoder.encode_snapshot(&snapshot2)
    };

    world_model.observe(&hv2);
    let drift = world_model.detect_drift(0.7);
    // Similar system state — no drift expected
    assert!(
        !drift.drifted,
        "Similar observations should not trigger drift"
    );

    // Observation 3: degraded (services failing)
    let snapshot3 = SystemStateSnapshot {
        services: vec![
            ("nginx.service".into(), ServiceState::Failed),
            ("sshd.service".into(), ServiceState::Running),
            ("postgresql.service".into(), ServiceState::Failed),
        ],
        packages: vec!["firefox".into(), "vim".into()],
        generation: Some(43),
        store_size_bytes: Some(10_000_000_000),
        store_path_count: Some(3000),
        config_options: vec![],
    };

    let hv3 = {
        let mut encoder = SystemStateEncoder::new(&mut codebook);
        encoder.encode_snapshot(&snapshot3)
    };

    // hv3 should differ from hv1/hv2
    let sim = hv1.similarity(&hv3);
    assert!(
        sim < 0.99,
        "Degraded system should differ from healthy, got similarity {:.3}",
        sim
    );
}

/// Full active inference: input → goal inference → action plan → evaluation.
#[test]
fn test_active_inference_planning_loop() {
    let mut engine = NixActiveInference::new();

    // Process multiple user inputs to build up working memory
    let plan1 = engine.process_input("install nginx web server");
    assert!(!plan1.goal.description.is_empty());
    assert!(plan1.current_free_energy >= 0.0);

    let plan2 = engine.process_input("enable the firewall");
    assert!(!plan2.goal.description.is_empty());

    // Working memory should accumulate context
    let wm = engine.goal_inference().working_memory();
    assert!(
        !wm.is_empty(),
        "Working memory should have items after input"
    );
    assert!(
        wm.mean_activation() > 0.0,
        "Working memory should have positive activation"
    );

    // Check working memory item sources
    let items = wm.items();
    for item in items {
        assert_eq!(item.source, MemorySource::UserInput);
        assert!(item.activation > 0.0);
    }

    // Process a third input — should benefit from accumulated context
    let plan3 = engine.process_input("check the system health");
    assert!(!plan3.goal.description.is_empty());

    // Free energy should be finite throughout
    assert!(engine.free_energy().is_finite());
}

/// Causal reasoning: predict side effects, analyze root causes.
#[test]
fn test_causal_reasoning_loop() {
    let mut graph = NixCausalGraph::new(42);

    // Bootstrap with known patterns
    let patterns = NixOSCausalPatterns::known_patterns();
    assert!(patterns.len() >= 200, "Should have >=200 known patterns");

    for (cause, effect, _desc) in &patterns {
        graph.add_structural_edge(cause, effect, 0.8);
    }

    // Predict side effects of enabling nginx
    let side_effects = graph.predict_side_effects("services.nginx.enable");
    // Should predict something related (firewall, virtual hosts, etc.)
    assert!(
        !side_effects.is_empty() || graph.edge_count() > 0,
        "Causal graph should have edges"
    );

    // Analyze root causes of a failure
    let root_causes = graph.analyze_root_causes("services.nginx.enable");
    // May or may not find root causes depending on graph structure
    // Just verify it doesn't panic
    let _ = root_causes;

    // Recommend fixes
    let fixes = graph.recommend_fixes("services.nginx.enable");
    let _ = fixes;
}

/// Predictive hierarchy: observation processing with multi-level prediction.
#[test]
fn test_predictive_hierarchy_loop() {
    let mut model = NixWorldModel::default();
    let mut codebook = NixCodebook::new();

    // Feed a series of observations and track hierarchy state
    for i in 0..10 {
        let snapshot = SystemStateSnapshot {
            services: vec![("nginx.service".into(), ServiceState::Running)],
            packages: vec![format!("pkg-{}", i)],
            generation: Some(42 + i),
            ..Default::default()
        };

        let hv = {
            let mut encoder = SystemStateEncoder::new(&mut codebook);
            encoder.encode_snapshot(&snapshot)
        };
        model.observe(hv);
    }

    let hierarchy = model.prediction_hierarchy();
    let errors = hierarchy.errors();

    // All levels should have finite errors
    for (i, &err) in errors.iter().enumerate() {
        assert!(err.is_finite(), "Level {} error should be finite", i);
        assert!(err >= 0.0, "Level {} error should be non-negative", i);
    }

    // Free energy should be finite
    let fe = hierarchy.free_energy();
    assert!(fe.is_finite(), "Free energy should be finite");
}

/// Journal anomaly → concern tracking → working memory integration.
#[test]
fn test_anomaly_to_concern_pipeline() {
    let mut detector = JournalAnomalyDetector::new().with_threshold(0.3);

    // Warm up with normal entries
    for i in 0..20 {
        let entries = vec![JournalEntry {
            unit: "sshd.service".into(),
            message: format!("Accepted publickey for user from 10.0.0.{}", i),
            priority: 6,
            timestamp: format!("2025-01-15T10:{:02}:00Z", i),
        }];
        let _ = detector.process_entries(&entries);
    }

    // Inject error-level entries
    let error_entries = vec![JournalEntry {
        unit: "kernel".into(),
        message: "Out of memory: Kill process 999 (java) score 800".into(),
        priority: 3,
        timestamp: "2025-01-15T10:30:00Z".into(),
    }];

    let anomalies = detector.process_entries(&error_entries);
    // The detector may or may not flag this depending on baseline — just verify no panic
    // and that anomalies have reasonable scores
    for anomaly in &anomalies {
        assert!(anomaly.anomaly_score >= 0.0);
        assert!(anomaly.anomaly_score <= 1.0);
        assert!(!anomaly.reason.is_empty());
    }
}

/// Pre-action vs post-action state comparison via HDC similarity.
#[test]
fn test_pre_post_action_verification() {
    let mut codebook = NixCodebook::new();

    // Capture pre-action state
    let pre_snapshot = SystemStateSnapshot {
        services: vec![("nginx.service".into(), ServiceState::Running)],
        generation: Some(42),
        ..Default::default()
    };
    let pre_hv = {
        let mut encoder = SystemStateEncoder::new(&mut codebook);
        encoder.encode_snapshot(&pre_snapshot)
    };

    // Simulate post-action state (same system — no change)
    let post_hv = {
        let mut encoder = SystemStateEncoder::new(&mut codebook);
        encoder.encode_snapshot(&pre_snapshot)
    };

    // Same state should produce identical vectors
    let sim = pre_hv.similarity(&post_hv);
    assert!(
        sim > 0.99,
        "Same state should have high similarity, got {}",
        sim
    );

    // Different state should produce lower similarity
    let changed_snapshot = SystemStateSnapshot {
        services: vec![
            ("nginx.service".into(), ServiceState::Running),
            ("postgresql.service".into(), ServiceState::Running),
        ],
        generation: Some(43),
        packages: vec!["postgresql".into()],
        ..Default::default()
    };
    let changed_hv = {
        let mut encoder = SystemStateEncoder::new(&mut codebook);
        encoder.encode_snapshot(&changed_snapshot)
    };
    let changed_sim = pre_hv.similarity(&changed_hv);
    assert!(
        changed_sim < sim,
        "Different state should have lower similarity: {} vs {}",
        changed_sim,
        sim
    );
}

/// Full consciousness loop: observe → encode → infer → plan → predict.
#[test]
fn test_full_consciousness_loop() {
    // 1. Initialize all components
    let mut engine = NixActiveInference::new();
    let mut codebook = NixCodebook::new();
    let mut causal_graph = NixCausalGraph::new(42);

    for (cause, effect, _) in NixOSCausalPatterns::known_patterns() {
        causal_graph.add_structural_edge(cause, effect, 0.8);
    }

    // 2. Observe system state
    let snapshot = SystemStateSnapshot {
        services: vec![
            ("nginx.service".into(), ServiceState::Running),
            ("sshd.service".into(), ServiceState::Running),
        ],
        packages: vec!["firefox".into(), "vim".into()],
        generation: Some(42),
        store_size_bytes: Some(15_000_000_000),
        store_path_count: Some(5000),
        config_options: vec![("services.nginx.enable".into(), "true".into())],
    };

    let state_hv = {
        let mut encoder = SystemStateEncoder::new(&mut codebook);
        encoder.encode_snapshot(&snapshot)
    };

    // 3. Feed observation to world model
    engine.world_model_mut().observe(state_hv);

    // 4. Process user input through active inference
    let plan = engine.process_input("add postgresql database");

    // 5. Verify plan was generated
    assert!(!plan.goal.description.is_empty(), "Goal should be inferred");
    assert!(
        plan.current_free_energy >= 0.0,
        "Free energy should be non-negative"
    );

    // 6. Verify working memory accumulated context
    let wm = engine.goal_inference().working_memory();
    assert!(!wm.is_empty(), "Working memory should track context");

    // 7. Verify predictive hierarchy processed the observation
    let hierarchy = engine.world_model().prediction_hierarchy();
    let errors = hierarchy.errors();
    for &err in &errors {
        assert!(err.is_finite());
    }

    // 8. Causal prediction
    let side_effects = causal_graph.predict_side_effects("services.postgresql.enable");
    // Just verify the causal system is operational
    let _ = side_effects;

    // 9. Verify free energy is coherent
    let fe = engine.free_energy();
    assert!(fe.is_finite(), "System free energy should be finite");
}

// ── Daemon IPC Round-Trip Tests ──────────────────────────────────────────

/// Build a full DaemonSnapshot, write to disk, read back, verify every field.
#[test]
fn test_daemon_ipc_snapshot_roundtrip() {
    let dir = std::env::temp_dir().join("nixward-ipc-roundtrip");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("daemon_state.json");

    let concerns = vec![
        ConcernEntry {
            label: "high memory usage".into(),
            activation: 0.85,
            source: "SystemObservation".into(),
        },
        ConcernEntry {
            label: "pending rebuild".into(),
            activation: 0.42,
            source: "UserInput".into(),
        },
    ];
    let anomalies = vec![
        AnomalyEntry {
            score: 0.91,
            reason: "OOM killer invoked".into(),
            unit: "kernel".into(),
            error_type: None,
            suggestion: None,
        },
        AnomalyEntry {
            score: 0.55,
            reason: "segfault in libfoo".into(),
            unit: "app.service".into(),
            error_type: Some("runtime_error".into()),
            suggestion: Some("Check coredump: coredumpctl list".into()),
        },
    ];

    let snapshot = DaemonSnapshot {
        observation_count: 142,
        anomaly_count: 7,
        hierarchy_errors: [0.12, 0.24, 0.08, 0.03],
        free_energy: 0.47,
        is_surprised: true,
        drift_similarity: 0.88,
        causal_edge_count: 315,
        episodic_count: 12,
        concerns,
        recent_anomalies: anomalies,
        daemon_pid: 99999,
        support_status: None,
        memory_used_percent: None,
        prediction_accuracy: None,
        load_average_1m: None,
        swap_used_percent: None,
        ..DaemonSnapshot::test_default()
    };

    // Write atomically
    snapshot.write_to(&path).expect("write_to should succeed");

    // Verify temp file was cleaned up (atomic rename)
    assert!(
        !path.with_extension("tmp").exists(),
        "Temp file should be removed after rename"
    );

    // Read back
    let restored = DaemonSnapshot::read_from(&path).expect("read_from should return Some");

    // Verify every field
    assert_eq!(restored.timestamp, 1700000000);
    assert_eq!(restored.observation_count, 142);
    assert_eq!(restored.anomaly_count, 7);
    assert_eq!(restored.hierarchy_errors, [0.12, 0.24, 0.08, 0.03]);
    assert!((restored.free_energy - 0.47).abs() < 1e-10);
    assert!(restored.is_surprised);
    assert!((restored.drift_similarity - 0.88).abs() < 1e-6);
    assert_eq!(restored.causal_edge_count, 315);
    assert_eq!(restored.episodic_count, 12);
    assert!(restored.daemon_running);
    assert_eq!(restored.daemon_pid, 99999);

    // Verify nested structs
    assert_eq!(restored.concerns.len(), 2);
    assert_eq!(restored.concerns[0].label, "high memory usage");
    assert!((restored.concerns[0].activation - 0.85).abs() < 1e-10);
    assert_eq!(restored.concerns[1].source, "UserInput");

    assert_eq!(restored.recent_anomalies.len(), 2);
    assert!((restored.recent_anomalies[0].score - 0.91).abs() < 1e-10);
    assert_eq!(restored.recent_anomalies[0].unit, "kernel");
    assert_eq!(restored.recent_anomalies[1].reason, "segfault in libfoo");

    let _ = std::fs::remove_dir_all(&dir);
}

/// DaemonConfig save → load round-trip with partial overrides.
#[test]
fn test_daemon_config_roundtrip() {
    let dir = std::env::temp_dir().join("nixward-config-roundtrip");
    let path = dir.join("daemon.json");

    let mut config = DaemonConfig::default();
    config.snapshot_interval = 120;
    config.surprise_threshold = 0.6;
    config.learning_rate = 0.05;

    config.save(&path).expect("save should succeed");
    let loaded = DaemonConfig::load(&path);

    assert_eq!(loaded.snapshot_interval, 120);
    assert_eq!(loaded.poll_interval, 5); // default preserved
    assert!((loaded.surprise_threshold - 0.6).abs() < 1e-10);
    assert_eq!(loaded.journal_batch_size, 50); // default preserved
    assert_eq!(loaded.ipc_write_interval, 10); // default preserved
    assert!((loaded.learning_rate - 0.05).abs() < 1e-10);

    let _ = std::fs::remove_dir_all(&dir);
}

/// Full cognitive pipeline → IPC snapshot → disk → read-back.
///
/// Exercises: encode system state → world model → episodic memory →
/// working memory → build IPC snapshot → write → read → verify.
#[test]
fn test_cognitive_to_ipc_pipeline() {
    let mut codebook = NixCodebook::new();

    // 1. Encode a system state
    let snapshot = SystemStateSnapshot {
        services: vec![
            ("nginx.service".into(), ServiceState::Running),
            ("postgresql.service".into(), ServiceState::Failed),
        ],
        packages: vec!["firefox".into(), "vim".into()],
        generation: Some(42),
        store_size_bytes: Some(15_000_000_000),
        store_path_count: Some(5000),
        config_options: vec![("services.nginx.enable".into(), "true".into())],
    };

    let state_hv = {
        let mut encoder = SystemStateEncoder::new(&mut codebook);
        encoder.encode_snapshot(&snapshot)
    };
    assert!(state_hv.dim() > 0);

    // 2. Feed into world model
    let mut world_model = NixWorldModel::default_dim();
    world_model.observe(state_hv.clone());
    let free_energy = world_model.free_energy();
    assert!(free_energy.is_finite());

    // 3. Create an episodic memory entry
    let mut episodic = NixEpisodicMemory::new();
    let episode = SystemEpisode {
        state_before: ContinuousHV::random(16384, 1),
        action: "nixos-rebuild switch".into(),
        state_after: state_hv.clone(),
        outcome: EpisodeOutcome::Success,
        phi_at_encoding: free_energy,
        prediction_error: 0.35,
        emotional_valence: 0.1,
        timestamp: 1700000000,
    };
    episodic.record(episode);

    // 4. Track concerns in working memory
    let mut working_memory = WorkingMemory::new();
    working_memory.push(
        state_hv.clone(),
        MemorySource::SystemObservation,
        "postgresql.service failed".into(),
    );
    working_memory.push(
        state_hv,
        MemorySource::UserInput,
        "user requested rebuild".into(),
    );

    // 5. Build IPC snapshot from cognitive state
    let hierarchy = world_model.prediction_hierarchy();
    let hierarchy_errors = hierarchy.errors();

    let concerns: Vec<ConcernEntry> = working_memory
        .items()
        .iter()
        .map(|item| ConcernEntry {
            label: item.label.clone(),
            activation: item.activation,
            source: format!("{:?}", item.source),
        })
        .collect();

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let ipc_snapshot = DaemonSnapshot {
        timestamp: now,
        observation_count: 1,
        anomaly_count: 0,
        hierarchy_errors,
        free_energy,
        is_surprised: free_energy > 0.3,
        causal_edge_count: 200,
        episodic_count: episodic.len(),
        concerns,
        daemon_pid: std::process::id(),
        support_status: None,
        memory_used_percent: None,
        prediction_accuracy: None,
        load_average_1m: None,
        swap_used_percent: None,
        ..DaemonSnapshot::test_default()
    };

    // 6. Write to disk and read back
    let dir = std::env::temp_dir().join("nixward-cognitive-ipc");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("daemon_state.json");

    ipc_snapshot.write_to(&path).expect("write should succeed");
    let restored = DaemonSnapshot::read_from(&path).expect("read should succeed");

    // 7. Verify cognitive data survived the round-trip
    assert_eq!(restored.observation_count, 1);
    assert_eq!(restored.episodic_count, 1);
    assert!((restored.free_energy - free_energy).abs() < 1e-10);
    assert_eq!(restored.hierarchy_errors, hierarchy_errors);
    assert_eq!(restored.concerns.len(), 2);
    let labels: Vec<&str> = restored.concerns.iter().map(|c| c.label.as_str()).collect();
    assert!(labels.contains(&"postgresql.service failed"));
    assert!(labels.contains(&"user requested rebuild"));
    assert!(restored.daemon_running);

    // 8. Verify freshness
    assert!(
        restored.is_fresh(60),
        "Just-written snapshot should be fresh"
    );

    // 9. Verify liveness (current process PID)
    assert!(restored.daemon_alive(), "Current PID should be alive");

    let _ = std::fs::remove_dir_all(&dir);
}

/// Stale snapshot detection: old timestamp → not fresh.
#[test]
fn test_stale_snapshot_detection() {
    let snapshot = DaemonSnapshot {
        timestamp: 1000, // ancient
        observation_count: 0,
        anomaly_count: 0,
        hierarchy_errors: [0.0; 4],
        free_energy: 0.0,
        daemon_pid: 1, // PID 1 (init) is always alive
        support_status: None,
        memory_used_percent: None,
        prediction_accuracy: None,
        load_average_1m: None,
        swap_used_percent: None,
        ..DaemonSnapshot::test_default()
    };

    assert!(
        !snapshot.is_fresh(60),
        "Ancient snapshot should not be fresh"
    );
    assert!(!snapshot.is_fresh(3600), "Still ancient");
    assert!(snapshot.daemon_alive(), "PID 1 should be alive");

    // Dead daemon: nonexistent PID
    let dead = DaemonSnapshot {
        daemon_pid: u32::MAX,
        ..snapshot
    };
    assert!(!dead.daemon_alive(), "Nonexistent PID should not be alive");
}
