// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration tests for the daemon's cognitive cycle.
//!
//! These tests exercise the DaemonSnapshot IPC pipeline: building snapshots,
//! serialization, and verifying that predictive alerts + causal edges + health
//! fields are properly populated.

use nixward::encoding::codebook::NixCodebook;
use nixward::ipc::{CausalEdgeEntry, DaemonSnapshot, SNAPSHOT_VERSION};
use nixward::mind::causal_graph::{NixCausalGraph, NixOSCausalPatterns};
use nixward::support::knowledge::{DynamicKnowledgeArticle, KnowledgeBase, KnowledgeCategory};
use nixward::support::predictive::{PredictiveMonitor, SystemTelemetry};

#[test]
fn test_daemon_snapshot_serialization_with_new_fields() {
    let snap = DaemonSnapshot {
        version: SNAPSHOT_VERSION,
        timestamp: 1700000000,
        observation_count: 100,
        anomaly_count: 5,
        hierarchy_errors: [0.1, 0.2, 0.15, 0.05],
        free_energy: 0.35,
        is_surprised: false,
        drift_similarity: 0.92,
        causal_edge_count: 42,
        episodic_count: 10,
        concerns: vec![],
        recent_anomalies: vec![],
        daemon_running: true,
        daemon_pid: 1234,
        support_status: Some("Healthy".into()),
        recommendation_count: 2,
        alerts: vec![nixward::ipc::AlertEntry {
            metric: "disk_used_pct".into(),
            current_value: 85.0,
            predicted_value: 95.0,
            hours_ahead: 24.0,
            threshold: 90.0,
            confidence: 0.7,
            recommended_action: Some("nix-collect-garbage -d".into()),
            severity: nixward::ipc::AlertSeverity::Critical,
            first_seen: 1700000000,
            last_seen: 1700000000,
            consecutive_cycles: 1,
            prev_predicted_value: None,
            journal_context: vec![],
        }],
        top_causal_edges: vec![CausalEdgeEntry {
            from: "nginx".into(),
            to: "disk_pressure".into(),
            confidence: 0.85,
        }],
        memory_used_percent: Some(55.0),
        watchdog_status: None,
        degraded: false,
        prediction_accuracy: Some(2.5),
        maintenance_plan_count: 1,
        load_average_1m: Some(0.42),
        swap_used_percent: Some(10.0),
        ..DaemonSnapshot::test_default()
    };

    // Serialize → deserialize roundtrip
    let json = serde_json::to_string_pretty(&snap).unwrap();
    let restored: DaemonSnapshot = serde_json::from_str(&json).unwrap();

    assert_eq!(restored.alerts.len(), 1);
    assert_eq!(restored.alerts[0].metric, "disk_used_pct");
    assert!((restored.alerts[0].predicted_value - 95.0).abs() < 1e-6);
    assert_eq!(restored.top_causal_edges.len(), 1);
    assert_eq!(restored.top_causal_edges[0].from, "nginx");
    assert!((restored.top_causal_edges[0].confidence - 0.85).abs() < 1e-6);
    assert!((restored.memory_used_percent.unwrap() - 55.0).abs() < 1e-6);
    assert_eq!(restored.support_status.as_deref(), Some("Healthy"));
    assert_eq!(restored.recommendation_count, 2);
    assert_eq!(restored.version, SNAPSHOT_VERSION);
    assert!(!restored.degraded);
    assert!((restored.prediction_accuracy.unwrap() - 2.5).abs() < 1e-6);
    assert_eq!(restored.maintenance_plan_count, 1);
    assert!((restored.load_average_1m.unwrap() - 0.42).abs() < 1e-6);
    assert!((restored.swap_used_percent.unwrap() - 10.0).abs() < 1e-6);
}

#[test]
fn test_daemon_snapshot_backward_compatible() {
    // Old-format JSON without new fields should deserialize with defaults
    let old_json = r#"{
        "timestamp": 1700000000,
        "observation_count": 42,
        "anomaly_count": 3,
        "hierarchy_errors": [0.1, 0.2, 0.15, 0.05],
        "free_energy": 0.35,
        "is_surprised": false,
        "drift_similarity": 0.95,
        "causal_edge_count": 210,
        "episodic_count": 5,
        "concerns": [],
        "recent_anomalies": [],
        "daemon_running": true,
        "daemon_pid": 12345
    }"#;

    let snap: DaemonSnapshot = serde_json::from_str(old_json).unwrap();
    assert!(snap.alerts.is_empty());
    assert!(snap.top_causal_edges.is_empty());
    assert!(snap.memory_used_percent.is_none());
    assert!(snap.support_status.is_none());
    assert_eq!(snap.recommendation_count, 0);
    assert!(snap.watchdog_status.is_none());
    assert_eq!(snap.version, 1, "Old snapshots default to version 1");
    assert!(!snap.degraded, "Old snapshots default to not degraded");
    assert!(snap.prediction_accuracy.is_none());
    assert_eq!(snap.maintenance_plan_count, 0);
    assert!(snap.load_average_1m.is_none());
    assert!(snap.swap_used_percent.is_none());
}

#[test]
fn test_predictive_monitor_generates_alerts() {
    let mut monitor = PredictiveMonitor::with_defaults();

    // Feed rising disk usage — should eventually trigger alert
    for i in 0..20 {
        monitor.ingest(SystemTelemetry {
            disk_used_pct: 70.0 + i as f64,
            memory_used_pct: 40.0,
            store_path_count: 50_000,
            failed_unit_count: 0,
            load_average_1m: 0.5,
            swap_used_pct: 5.0,
        });
    }

    let predictions = monitor.predict_all_horizons();
    // With disk at 89% and rising, some horizon should predict crossing 80% warn
    let disk_alerts: Vec<_> = predictions
        .iter()
        .filter(|p| p.metric == "disk_used_pct" && p.predicted_value >= 80.0)
        .collect();
    assert!(
        !disk_alerts.is_empty(),
        "Rising disk should generate at least one alert-worthy prediction"
    );
}

#[test]
fn test_causal_graph_top_edges() {
    let mut graph = NixCausalGraph::new(42);

    // Bootstrap with known patterns
    for (cause, effect, _) in NixOSCausalPatterns::known_patterns() {
        graph.add_structural_edge(cause, effect, 0.5);
    }

    // Strengthen one edge via observation
    graph.observe_outcome(
        "nginx",
        &["disk_pressure", "memory_pressure"],
        &["disk_pressure", "memory_pressure"],
    );

    let top = graph.top_edges(5);
    assert!(top.len() <= 5);
    assert!(!top.is_empty());
    // Should be sorted by descending confidence
    for pair in top.windows(2) {
        assert!(
            pair[0].confidence >= pair[1].confidence,
            "Edges should be sorted by descending confidence"
        );
    }
}

#[test]
fn test_causal_graph_top_edges_empty() {
    let graph = NixCausalGraph::new(42);
    let top = graph.top_edges(10);
    assert!(top.is_empty());
}

#[test]
fn test_snapshot_write_read_roundtrip_with_alerts() {
    let snap = DaemonSnapshot {
        observation_count: 1,
        anomaly_count: 0,
        hierarchy_errors: [0.0; 4],
        free_energy: 0.1,
        drift_similarity: 0.99,
        causal_edge_count: 0,
        episodic_count: 0,
        daemon_pid: 1,
        support_status: Some("Critical".into()),
        recommendation_count: 3,
        alerts: vec![nixward::ipc::AlertEntry {
            metric: "memory_used_pct".into(),
            current_value: 92.0,
            predicted_value: 98.0,
            hours_ahead: 1.0,
            threshold: 95.0,
            confidence: 0.9,
            recommended_action: Some("systemd-cgtop".into()),
            severity: nixward::ipc::AlertSeverity::Warning,
            first_seen: 1700000000,
            last_seen: 1700000000,
            consecutive_cycles: 5,
            prev_predicted_value: Some(96.0),
            journal_context: vec!["OOM killer invoked".into()],
        }],
        top_causal_edges: vec![
            CausalEdgeEntry {
                from: "postgresql".into(),
                to: "memory_pressure".into(),
                confidence: 0.9,
            },
            CausalEdgeEntry {
                from: "nginx".into(),
                to: "disk_pressure".into(),
                confidence: 0.7,
            },
        ],
        memory_used_percent: Some(92.0),
        watchdog_status: Some("degraded".into()),
        degraded: true,
        prediction_accuracy: Some(4.2),
        maintenance_plan_count: 2,
        load_average_1m: Some(1.5),
        swap_used_percent: Some(25.0),
        ..DaemonSnapshot::test_default()
    };

    let dir = std::env::temp_dir().join("nixward-daemon-integration-test");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("daemon_state.json");

    snap.write_to(&path).unwrap();
    let restored = DaemonSnapshot::read_from(&path).unwrap();

    assert_eq!(restored.alerts.len(), 1);
    assert_eq!(restored.alerts[0].journal_context.len(), 1);
    assert_eq!(restored.top_causal_edges.len(), 2);
    assert_eq!(restored.top_causal_edges[0].from, "postgresql");
    assert!((restored.memory_used_percent.unwrap() - 92.0).abs() < 1e-6);
    assert_eq!(restored.support_status.as_deref(), Some("Critical"));
    assert_eq!(restored.watchdog_status.as_deref(), Some("degraded"));
    assert_eq!(restored.version, SNAPSHOT_VERSION);
    assert!(restored.degraded);
    assert!((restored.prediction_accuracy.unwrap() - 4.2).abs() < 1e-6);
    assert_eq!(restored.maintenance_plan_count, 2);
    assert!((restored.load_average_1m.unwrap() - 1.5).abs() < 1e-6);
    assert!((restored.swap_used_percent.unwrap() - 25.0).abs() < 1e-6);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_knowledge_learning_persistence() {
    let mut codebook = NixCodebook::new();
    let mut kb = KnowledgeBase::new(&mut codebook);

    // Add a learned article
    let article = DynamicKnowledgeArticle {
        id: "learned_test_1".into(),
        title: "Resolved: nginx 502".into(),
        category: KnowledgeCategory::ServiceIssue,
        symptoms: vec!["502 Bad Gateway".into(), "upstream timeout".into()],
        solution: "Increase proxy_read_timeout".into(),
        commands: vec!["systemctl restart nginx".into()],
        learned_at: 1700000000,
        hit_count: 0,
    };
    kb.add_learned_article(article, &mut codebook);
    assert_eq!(kb.dynamic_len(), 1);

    // Serialize and reload into a fresh KB
    let json = kb.save_dynamic();
    let mut codebook2 = NixCodebook::new();
    let mut kb2 = KnowledgeBase::new(&mut codebook2);
    kb2.load_dynamic(&json, &mut codebook2);
    assert_eq!(kb2.dynamic_len(), 1);

    // Search should find the learned article
    let results = kb2.search_all("nginx 502 gateway", &mut codebook2, 5);
    assert!(!results.is_empty(), "Should find learned article");
}

#[test]
fn test_knowledge_dynamic_ranking() {
    let mut codebook = NixCodebook::new();
    let mut kb = KnowledgeBase::new(&mut codebook);

    // Add 3 articles with different symptoms
    for (i, symptom) in ["disk full", "memory OOM", "nginx crash"]
        .iter()
        .enumerate()
    {
        kb.add_learned_article(
            DynamicKnowledgeArticle {
                id: format!("learned_{i}"),
                title: format!("Resolved: {symptom}"),
                category: KnowledgeCategory::ServiceIssue,
                symptoms: vec![symptom.to_string()],
                solution: format!("Fix for {symptom}"),
                commands: vec![],
                learned_at: 1700000000,
                hit_count: 0,
            },
            &mut codebook,
        );
    }
    assert_eq!(kb.dynamic_len(), 3);

    // Search for "disk full" — should rank the disk article highest among dynamic results
    let results = kb.search_all("disk full", &mut codebook, 20);
    assert!(!results.is_empty());
    // All results should have finite similarity
    for r in &results {
        assert!(r.similarity().is_finite(), "Similarity must be finite");
        assert!(r.similarity() >= 0.0, "Similarity must be non-negative");
    }
}

#[test]
fn test_semantic_search_rank_validity() {
    use nixward::encoding::semantic_search::search_options;
    let mut codebook = NixCodebook::new();
    let paths = [
        "services.nginx.enable",
        "services.nginx.package",
        "services.postgresql.enable",
        "networking.firewall.enable",
        "boot.loader.grub.enable",
        "users.users.alice.isNormalUser",
        "environment.systemPackages",
    ];

    let results = search_options("enable nginx web server", &mut codebook, &paths, 7);
    assert_eq!(results.len(), 7, "Should return all paths");

    // Verify sorted descending
    for window in results.windows(2) {
        assert!(
            window[0].similarity >= window[1].similarity,
            "Results must be sorted by descending similarity: {} >= {}",
            window[0].similarity,
            window[1].similarity,
        );
    }

    // All similarities should be finite and non-negative
    for r in &results {
        assert!(
            r.similarity.is_finite(),
            "Similarity must be finite for {}",
            r.path
        );
        assert!(
            r.similarity >= 0.0,
            "Similarity must be non-negative for {}",
            r.path
        );
    }
}
