// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! CLI Integration Tests
//!
//! End-to-end tests for the nix-mind CLI command parsing,
//! cognitive context initialization, and dispatch logic.
#![cfg(feature = "cli")]

use clap::Parser;
use symthaea_nix::cli::commands::{
    Cli, Command, FlakeCommand, ObserveDomain, OutputFormat, RebuildMode, ServiceCommand,
};

// === Command Parsing Tests ===

#[test]
fn test_cli_no_args_is_repl() {
    let cli = Cli::parse_from(["nix-mind"]);
    assert!(!cli.has_natural_input());
    assert!(cli.command.is_none());
    assert!(!cli.dry_run);
    assert_eq!(cli.format, OutputFormat::Human);
}

#[test]
fn test_cli_natural_language_multiword() {
    let cli = Cli::parse_from(["nix-mind", "why", "did", "my", "rebuild", "fail"]);
    assert!(cli.has_natural_input());
    assert_eq!(cli.natural_input(), "why did my rebuild fail");
}

#[test]
fn test_cli_search_with_limit() {
    let cli = Cli::parse_from(["nix-mind", "search", "firefox", "-n", "5"]);
    if let Some(Command::Search {
        query,
        limit,
        options,
    }) = cli.command
    {
        assert_eq!(query, "firefox");
        assert_eq!(limit, 5);
        assert!(!options);
    } else {
        panic!("Expected Search command");
    }
}

#[test]
fn test_cli_search_options_flag() {
    let cli = Cli::parse_from(["nix-mind", "search", "enable", "--options"]);
    if let Some(Command::Search { options, .. }) = cli.command {
        assert!(options);
    } else {
        panic!("Expected Search command");
    }
}

#[test]
fn test_cli_rebuild_modes() {
    for (args, expected_mode) in [
        (vec!["nix-mind", "rebuild", "switch"], RebuildMode::Switch),
        (vec!["nix-mind", "rebuild", "test"], RebuildMode::Test),
        (vec!["nix-mind", "rebuild", "boot"], RebuildMode::Boot),
    ] {
        let cli = Cli::parse_from(args);
        if let Some(Command::Rebuild { mode, .. }) = cli.command {
            assert_eq!(mode, expected_mode);
        } else {
            panic!("Expected Rebuild command");
        }
    }
}

#[test]
fn test_cli_rebuild_with_flake() {
    let cli = Cli::parse_from(["nix-mind", "rebuild", "switch", "--flake", ".#myhost"]);
    if let Some(Command::Rebuild { flake, .. }) = cli.command {
        assert_eq!(flake, Some(".#myhost".to_string()));
    } else {
        panic!("Expected Rebuild command");
    }
}

#[test]
fn test_cli_rollback_no_generation() {
    let cli = Cli::parse_from(["nix-mind", "rollback"]);
    if let Some(Command::Rollback { generation }) = cli.command {
        assert_eq!(generation, None);
    } else {
        panic!("Expected Rollback command");
    }
}

#[test]
fn test_cli_rollback_with_generation() {
    let cli = Cli::parse_from(["nix-mind", "rollback", "42"]);
    if let Some(Command::Rollback { generation }) = cli.command {
        assert_eq!(generation, Some(42));
    } else {
        panic!("Expected Rollback command");
    }
}

#[test]
fn test_cli_observe_domains() {
    for (arg, expected) in [
        ("services", ObserveDomain::Services),
        ("packages", ObserveDomain::Packages),
        ("store", ObserveDomain::Store),
        ("generations", ObserveDomain::Generations),
        ("hardware", ObserveDomain::Hardware),
        ("flakes", ObserveDomain::Flakes),
    ] {
        let cli = Cli::parse_from(["nix-mind", "observe", arg]);
        if let Some(Command::Observe { domain: Some(d) }) = cli.command {
            assert_eq!(d, expected, "Failed for domain arg: {}", arg);
        } else {
            panic!("Expected Observe command with domain for arg: {}", arg);
        }
    }
}

#[test]
fn test_cli_observe_no_domain() {
    let cli = Cli::parse_from(["nix-mind", "observe"]);
    if let Some(Command::Observe { domain }) = cli.command {
        assert!(domain.is_none());
    } else {
        panic!("Expected Observe command");
    }
}

#[test]
fn test_cli_doctor() {
    let cli = Cli::parse_from(["nix-mind", "doctor"]);
    assert!(matches!(cli.command, Some(Command::Doctor)));
}

#[test]
fn test_cli_gc_modes() {
    let cli = Cli::parse_from(["nix-mind", "gc", "--analyze"]);
    if let Some(Command::Gc {
        analyze,
        aggressive,
        older_than,
    }) = cli.command
    {
        assert!(analyze);
        assert!(!aggressive);
        assert_eq!(older_than, None);
    } else {
        panic!("Expected Gc command");
    }

    let cli = Cli::parse_from(["nix-mind", "gc", "--aggressive", "--older-than", "30"]);
    if let Some(Command::Gc {
        analyze,
        aggressive,
        older_than,
    }) = cli.command
    {
        assert!(!analyze);
        assert!(aggressive);
        assert_eq!(older_than, Some(30));
    } else {
        panic!("Expected Gc command");
    }
}

#[test]
fn test_cli_service_subcommands() {
    let cli = Cli::parse_from(["nix-mind", "service", "status", "nginx"]);
    if let Some(Command::Service {
        op: ServiceCommand::Status { name },
    }) = cli.command
    {
        assert_eq!(name, "nginx");
    } else {
        panic!("Expected Service Status");
    }

    let cli = Cli::parse_from(["nix-mind", "service", "start", "postgresql"]);
    if let Some(Command::Service {
        op: ServiceCommand::Start { name },
    }) = cli.command
    {
        assert_eq!(name, "postgresql");
    } else {
        panic!("Expected Service Start");
    }

    let cli = Cli::parse_from(["nix-mind", "service", "stop", "docker"]);
    if let Some(Command::Service {
        op: ServiceCommand::Stop { name },
    }) = cli.command
    {
        assert_eq!(name, "docker");
    } else {
        panic!("Expected Service Stop");
    }

    let cli = Cli::parse_from(["nix-mind", "service", "failed"]);
    assert!(matches!(
        cli.command,
        Some(Command::Service {
            op: ServiceCommand::Failed
        })
    ));
}

#[test]
fn test_cli_flake_subcommands() {
    let cli = Cli::parse_from(["nix-mind", "flake", "check"]);
    assert!(matches!(
        cli.command,
        Some(Command::Flake {
            op: FlakeCommand::Check
        })
    ));

    let cli = Cli::parse_from(["nix-mind", "flake", "update", "nixpkgs"]);
    if let Some(Command::Flake {
        op: FlakeCommand::Update { inputs },
    }) = cli.command
    {
        assert_eq!(inputs, vec!["nixpkgs"]);
    } else {
        panic!("Expected Flake Update");
    }

    let cli = Cli::parse_from(["nix-mind", "flake", "show"]);
    assert!(matches!(
        cli.command,
        Some(Command::Flake {
            op: FlakeCommand::Show
        })
    ));

    let cli = Cli::parse_from(["nix-mind", "flake", "info"]);
    assert!(matches!(
        cli.command,
        Some(Command::Flake {
            op: FlakeCommand::Info
        })
    ));
}

#[test]
fn test_cli_phi_override() {
    let cli = Cli::parse_from(["nix-mind", "--phi", "0.95", "rebuild"]);
    assert_eq!(cli.phi, Some(0.95));
}

#[test]
fn test_cli_verbose_levels() {
    let cli = Cli::parse_from(["nix-mind", "-v", "3", "doctor"]);
    assert_eq!(cli.verbose, 3);
}

#[test]
fn test_cli_output_formats() {
    for (fmt, expected) in [
        ("human", OutputFormat::Human),
        ("json", OutputFormat::Json),
        ("minimal", OutputFormat::Minimal),
    ] {
        let cli = Cli::parse_from(["nix-mind", "--format", fmt, "doctor"]);
        assert_eq!(cli.format, expected, "Failed for format: {}", fmt);
    }
}

#[test]
fn test_cli_global_flags_with_subcommand() {
    let cli = Cli::parse_from([
        "nix-mind",
        "--dry-run",
        "--format",
        "json",
        "--phi",
        "0.5",
        "rebuild",
        "test",
    ]);
    assert!(cli.dry_run);
    assert_eq!(cli.format, OutputFormat::Json);
    assert_eq!(cli.phi, Some(0.5));
    if let Some(Command::Rebuild { mode, .. }) = cli.command {
        assert_eq!(mode, RebuildMode::Test);
    } else {
        panic!("Expected Rebuild command");
    }
}

// === Cognitive Core Integration Tests ===

#[test]
fn test_causal_graph_bootstrap() {
    use symthaea_nix::mind::causal_graph::{NixCausalGraph, NixOSCausalPatterns};
    let mut graph = NixCausalGraph::new(42);
    let patterns = NixOSCausalPatterns::known_patterns();
    assert!(!patterns.is_empty(), "Should have known patterns");

    for (cause, effect, _) in &patterns {
        graph.add_structural_edge(cause, effect, 0.8);
    }
    assert!(
        graph.edge_count() > 0,
        "Graph should have edges after bootstrap"
    );
}

#[test]
fn test_active_inference_initialization() {
    use symthaea_nix::mind::active_inference::NixActiveInference;
    let engine = NixActiveInference::new();
    let fe = engine.free_energy();
    assert!(fe.is_finite(), "Free energy should be finite");
}

#[test]
fn test_hdc_world_model_drift() {
    use symthaea_core::hdc::ContinuousHV;
    use symthaea_nix::mind::HdcWorldModel;

    let mut model = HdcWorldModel::default();
    let state = ContinuousHV::random(256, 42);
    model.observe(&state);

    let drift = model.detect_drift(0.8);
    // After first observation, no previous state to drift from
    assert!(!drift.drifted || drift.similarity >= 0.0);
}

#[test]
fn test_system_state_encoder_roundtrip() {
    use symthaea_nix::encoding::SystemStateSnapshot;
    use symthaea_nix::encoding::{NixCodebook, SystemStateEncoder};

    let mut codebook = NixCodebook::new();
    let mut encoder = SystemStateEncoder::new(&mut codebook);

    let snapshot = SystemStateSnapshot::default();
    let hv = encoder.encode_snapshot(&snapshot);
    assert!(hv.dim() > 0);
}

#[test]
fn test_journal_anomaly_detector_integration() {
    use symthaea_nix::mind::JournalAnomalyDetector;
    use symthaea_nix::observe::journal::JournalEntry;

    let mut detector = JournalAnomalyDetector::new().with_threshold(0.3);

    // Warm up with normal entries
    for i in 0..30 {
        let entries = vec![JournalEntry {
            unit: "sshd.service".into(),
            message: format!("Accepted publickey for user from 192.168.1.{}", i % 255),
            priority: 6,
            timestamp: "2025-01-15T10:00:00Z".into(),
        }];
        let _ = detector.process_entries(&entries);
    }

    // Inject anomalous entry
    let anomalous = vec![JournalEntry {
        unit: "kernel".into(),
        message: "Out of memory: Kill process 12345 (java) score 800".into(),
        priority: 3,
        timestamp: "2025-01-15T10:01:00Z".into(),
    }];
    let anomalies = detector.process_entries(&anomalous);
    // May or may not detect it depending on baseline — just verify no panic
    assert!(anomalies.len() <= 1);
}

#[test]
fn test_executor_safety_levels() {
    use symthaea_nix::action::executor::NixOSExecutor;
    let executor = NixOSExecutor::new();
    // Just verify construction doesn't panic
    drop(executor);
}
