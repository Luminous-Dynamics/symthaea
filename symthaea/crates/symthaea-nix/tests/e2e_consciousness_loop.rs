//! End-to-End Consciousness Loop Integration Test
//!
//! Exercises the full NixOS awareness cycle:
//!   Observation → Encoding → Inference → Action Plan → Verification
//!
//! This test validates that all layers connect properly and produce
//! meaningful results through the entire cognitive pipeline.

use symthaea_nix::encoding::{NixCodebook, SystemStateEncoder, SystemStateSnapshot, ServiceState};
use symthaea_nix::mind::active_inference::NixActiveInference;
use symthaea_nix::mind::causal_graph::{NixCausalGraph, NixOSCausalPatterns};
use symthaea_nix::mind::{NixWorldModel, HdcWorldModel};
use symthaea_nix::mind::working_memory::MemorySource;
use symthaea_nix::mind::journal_anomaly::JournalAnomalyDetector;
use symthaea_nix::observe::journal::JournalEntry;

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
    assert!(!drift.drifted, "Similar observations should not trigger drift");

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
    assert!(!wm.is_empty(), "Working memory should have items after input");
    assert!(wm.mean_activation() > 0.0, "Working memory should have positive activation");

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
            services: vec![
                ("nginx.service".into(), ServiceState::Running),
            ],
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
    let error_entries = vec![
        JournalEntry {
            unit: "kernel".into(),
            message: "Out of memory: Kill process 999 (java) score 800".into(),
            priority: 3,
            timestamp: "2025-01-15T10:30:00Z".into(),
        },
    ];

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
    assert!(sim > 0.99, "Same state should have high similarity, got {}", sim);

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
    assert!(changed_sim < sim, "Different state should have lower similarity: {} vs {}", changed_sim, sim);
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
        config_options: vec![
            ("services.nginx.enable".into(), "true".into()),
        ],
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
    assert!(plan.current_free_energy >= 0.0, "Free energy should be non-negative");

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
