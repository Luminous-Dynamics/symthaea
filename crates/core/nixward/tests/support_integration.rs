// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration tests for the support modules.
//!
//! These tests verify that the support modules work together correctly,
//! exercising the full pipeline: scrub → health → knowledge → predict.
//! Note: watchdog tests are unit-only (the full monitor requires a live system).

use nixward::encoding::{NixCodebook, ServiceState, SystemStateSnapshot};
use nixward::observe::hardware::{DiskInfo, HardwareInfo};
use nixward::support::health_check::{HealthAssessor, HealthStatus};
use nixward::support::knowledge::{KnowledgeBase, KnowledgeCategory};
use nixward::support::predictive::{PredictiveMonitor, SystemTelemetry};
use nixward::support::scrubber::Scrubber;

// ============================================================================
// Scrubber Integration
// ============================================================================

#[test]
fn test_scrubber_on_realistic_nixos_log() {
    let scrubber = Scrubber::new();
    let log = "\
Jan 15 10:23:45 my-nixos-host systemd[1]: Started nginx.service.
Jan 15 10:23:46 my-nixos-host nginx[1234]: 192.168.1.50 - alice@company.com \"GET /api HTTP/1.1\" 200
Jan 15 10:23:47 my-nixos-host sshd[5678]: Accepted publickey for user from 10.0.0.42 port 22 ssh2: ED25519 SHA256:abcdef123456
Jan 15 10:24:00 my-nixos-host kernel: wlan0: associated with AA:BB:CC:DD:EE:FF
error: builder for '/nix/store/abc123-glibc-2.38.drv' failed with exit code 1
reading file /home/tristan/.config/nixpkgs/config.nix";

    let result = scrubber.scrub(log);

    // PII should be redacted
    assert!(!result.scrubbed_text.contains("192.168.1.50"));
    assert!(!result.scrubbed_text.contains("10.0.0.42"));
    assert!(!result.scrubbed_text.contains("alice@company.com"));
    assert!(!result.scrubbed_text.contains("AA:BB:CC:DD:EE:FF"));
    assert!(!result.scrubbed_text.contains("/home/tristan"));

    // Non-PII should be preserved
    assert!(result.scrubbed_text.contains("nginx.service"));
    assert!(
        result
            .scrubbed_text
            .contains("/nix/store/abc123-glibc-2.38.drv")
    );
    assert!(result.scrubbed_text.contains("exit code 1"));

    assert!(result.redaction_count >= 4);
}

// ============================================================================
// Health + Knowledge Integration
// ============================================================================

#[test]
fn test_health_check_drives_knowledge_search() {
    // Simulate: health check detects disk critical → search knowledge base for solutions
    let assessor = HealthAssessor::default();

    let hardware = HardwareInfo {
        cpu_model: "Test CPU".to_string(),
        cpu_cores: 8,
        memory_total_mb: 32000,
        memory_available_mb: 28000,
        gpus: vec![],
        disks: vec![DiskInfo {
            device: "/dev/sda1".to_string(),
            mount_point: "/".to_string(),
            total_bytes: 500_000_000_000,
            used_bytes: 475_000_000_000, // 95% full
        }],
        load_average: [0.5, 0.4, 0.3],
        swap_total_mb: 8192,
        swap_used_mb: 0,
    };

    let snapshot = SystemStateSnapshot {
        services: vec![("nginx.service".to_string(), ServiceState::Running)],
        store_path_count: Some(150_000),
        ..Default::default()
    };

    let (overall, checks) = assessor.assess_all(&snapshot, Some(&hardware));
    assert_eq!(overall, HealthStatus::Critical);

    // Find the critical disk check
    let disk_check = checks.iter().find(|c| c.name == "disk_usage").unwrap();
    assert_eq!(disk_check.status, HealthStatus::Critical);

    // Use the health check message to search the knowledge base
    let mut codebook = NixCodebook::new();
    let kb = KnowledgeBase::new(&mut codebook);
    let results = kb.search(&disk_check.message, &mut codebook, 5);

    assert!(!results.is_empty());
    // Should find store management articles since disk is full
    let has_store_article = results
        .iter()
        .any(|r| r.article.category == KnowledgeCategory::StoreManagement);
    // Also acceptable: build error articles about disk space
    let has_disk_article = results.iter().any(|r| {
        r.article.id.contains("disk")
            || r.article.id.contains("gc")
            || r.article.id.contains("store")
    });
    assert!(
        has_store_article || has_disk_article,
        "Disk-full health check should find relevant articles, got: {:?}",
        results.iter().map(|r| r.article.id).collect::<Vec<_>>()
    );
}

// ============================================================================
// Predictive Monitor Integration
// ============================================================================

#[test]
fn test_predictive_monitor_with_rising_disk() {
    let mut monitor = PredictiveMonitor::with_defaults();

    // Simulate gradually rising disk usage over several samples
    for i in 0..20 {
        monitor.ingest(SystemTelemetry {
            disk_used_pct: 70.0 + i as f64 * 0.5, // 70% → 79.5%
            memory_used_pct: 50.0,
            store_path_count: 50_000 + i * 1000,
            failed_unit_count: 0,
            load_average_1m: 0.5,
            swap_used_pct: 5.0,
        });
    }

    let predictions = monitor.predict_all_horizons();
    assert_eq!(predictions.len(), 24); // 6 metrics × 4 horizons

    // 7-day disk prediction should be at or above the latest value (79.5)
    // Note: since samples are ingested nearly simultaneously, the time delta
    // is very small, making the per-hour trend very large. The predicted value
    // is clamped to 100.0 but should be >= the last ingested value.
    let disk_7d: Vec<_> = predictions
        .iter()
        .filter(|p| p.metric == "disk_used_pct" && p.hours_ahead == 168.0)
        .collect();
    assert_eq!(disk_7d.len(), 1);
    assert!(
        disk_7d[0].predicted_value >= 79.5,
        "Rising disk should predict at or above current, got {}",
        disk_7d[0].predicted_value
    );
}

#[test]
fn test_predictive_monitor_stable_system_no_alerts() {
    let mut monitor = PredictiveMonitor::with_defaults();

    // Stable system — no trends
    for _ in 0..10 {
        monitor.ingest(SystemTelemetry {
            disk_used_pct: 50.0,
            memory_used_pct: 40.0,
            store_path_count: 50_000,
            failed_unit_count: 0,
            load_average_1m: 0.5,
            swap_used_pct: 5.0,
        });
    }

    let predictions = monitor.predict(24.0);
    // All predictions should be near current values, no threshold crossings
    for pred in &predictions {
        assert!(
            !pred.crosses_threshold,
            "Stable system should have no threshold crossings for {}: current={}, pred={}",
            pred.metric, pred.current_value, pred.predicted_value,
        );
    }
}

// ============================================================================
// Knowledge Base Corpus Integrity
// ============================================================================

#[test]
fn test_knowledge_base_covers_all_categories() {
    let mut codebook = NixCodebook::new();
    let kb = KnowledgeBase::new(&mut codebook);

    let categories = [
        KnowledgeCategory::BuildError,
        KnowledgeCategory::ServiceIssue,
        KnowledgeCategory::StoreManagement,
        KnowledgeCategory::FlakePattern,
        KnowledgeCategory::HardwareDriver,
        KnowledgeCategory::EvaluationError,
    ];

    for cat in &categories {
        let articles = kb.articles_in_category(*cat);
        assert!(
            !articles.is_empty(),
            "Category {:?} should have at least one article",
            cat
        );
    }
}

#[test]
fn test_knowledge_base_error_search_relevance() {
    let mut codebook = NixCodebook::new();
    let kb = KnowledgeBase::new(&mut codebook);

    // Common NixOS error messages → should find relevant articles
    let test_cases = [
        (
            "hash mismatch in fixed-output derivation",
            "build-hash-mismatch",
        ),
        ("infinite recursion encountered", "eval-infinite-recursion"),
        ("No space left on device", "build-out-of-disk"),
    ];

    for (error_msg, expected_id) in &test_cases {
        let results = kb.search_by_error(error_msg, &mut codebook, 5);
        assert!(
            !results.is_empty(),
            "Search for '{}' returned no results",
            error_msg
        );
        let found = results.iter().any(|r| r.article.id == *expected_id);
        assert!(
            found,
            "Expected to find article '{}' for error '{}', got: {:?}",
            expected_id,
            error_msg,
            results.iter().map(|r| r.article.id).collect::<Vec<_>>()
        );
    }
}

// ============================================================================
// Full Pipeline: Observe → Health → Predict → Recommend
// ============================================================================

#[test]
fn test_full_support_pipeline() {
    // 1. Start with a system snapshot
    let snapshot = SystemStateSnapshot {
        services: vec![
            ("nginx.service".to_string(), ServiceState::Running),
            ("sshd.service".to_string(), ServiceState::Running),
            ("broken.service".to_string(), ServiceState::Failed),
        ],
        store_path_count: Some(120_000),
        store_size_bytes: Some(50_000_000_000),
        ..Default::default()
    };

    let hardware = HardwareInfo {
        cpu_model: "Test CPU".to_string(),
        cpu_cores: 8,
        memory_total_mb: 32000,
        memory_available_mb: 6000, // Only ~19% available = ~81% used
        gpus: vec![],
        disks: vec![DiskInfo {
            device: "/dev/sda1".to_string(),
            mount_point: "/".to_string(),
            total_bytes: 500_000_000_000,
            used_bytes: 420_000_000_000, // 84% used
        }],
        load_average: [1.5, 1.0, 0.8],
        swap_total_mb: 8192,
        swap_used_mb: 1024,
    };

    // 2. Health assessment
    let assessor = HealthAssessor::default();
    let (overall, checks) = assessor.assess_all(&snapshot, Some(&hardware));
    assert_eq!(overall, HealthStatus::Warning); // disk 84% + memory 81% + failed svc

    // 3. Collect recommendations from all checks
    let all_recs: Vec<String> = checks
        .iter()
        .flat_map(|c| c.recommendations.clone())
        .collect();
    assert!(
        !all_recs.is_empty(),
        "A system with warnings should produce recommendations"
    );

    // 4. Knowledge search for the failed service
    let mut codebook = NixCodebook::new();
    let kb = KnowledgeBase::new(&mut codebook);
    let svc_results = kb.search("service failed to start", &mut codebook, 3);
    assert!(!svc_results.is_empty());

    // 5. Feed predictive monitor
    let mut monitor = PredictiveMonitor::with_defaults();
    monitor.ingest(SystemTelemetry {
        disk_used_pct: 84.0,
        memory_used_pct: 81.0,
        store_path_count: 120_000,
        failed_unit_count: 1,
        load_average_1m: 1.5,
        swap_used_pct: 20.0,
    });

    let predictions = monitor.predict(24.0);
    assert!(!predictions.is_empty());

    // 6. Scrub a recommendation for safe sharing
    let scrubber = Scrubber::new();
    let sample_rec =
        "Check: journalctl -u broken.service on host 192.168.1.5 as user admin@corp.com";
    let scrubbed = scrubber.scrub(sample_rec);
    assert!(!scrubbed.scrubbed_text.contains("192.168.1.5"));
    assert!(!scrubbed.scrubbed_text.contains("admin@corp.com"));
}
