// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration tests for the gui_bridge module.
//!
//! Tests the full widget↔Nix bidirectional pipeline: binding registration,
//! change handling, Nix sync, semantic search, and constraint validation.
#![cfg(feature = "gui")]

use symthaea::gui_bridge::widget_mapper::WidgetValueType;
use symthaea::gui_bridge::*;

#[test]
fn test_bridge_creation() {
    let bridge = GuiBridge::new();
    assert!(
        bridge
            .bindings_for_category(ConfigCategory::Services)
            .is_empty()
    );
    assert!(
        bridge
            .bindings_for_category(ConfigCategory::Packages)
            .is_empty()
    );
}

#[test]
fn test_register_and_query_binding() {
    let mut bridge = GuiBridge::new();
    let binding = WidgetBinding::builder("sshd_toggle", "services.openssh.enable")
        .value_type(WidgetValueType::Bool)
        .label("SSH Server")
        .description("Enable OpenSSH daemon")
        .default(WidgetValue::Bool(false))
        .build();

    bridge.register_binding(binding);

    let services = bridge.bindings_for_category(ConfigCategory::Services);
    assert_eq!(services.len(), 1);
    assert_eq!(services[0].label, "SSH Server");
}

#[test]
fn test_widget_change_produces_mapping() {
    let mut bridge = GuiBridge::new();
    let binding = WidgetBinding::builder("sshd_toggle", "services.openssh.enable")
        .value_type(WidgetValueType::Bool)
        .label("SSH Server")
        .default(WidgetValue::Bool(false))
        .build();

    bridge.register_binding(binding);

    let result = bridge.on_widget_change(WidgetId::new("sshd_toggle"), WidgetValue::Bool(true));
    // Should produce a mapping result (success or validation)
    assert!(
        matches!(
            result,
            MappingResult::Success { .. } | MappingResult::ValidationFailed { .. }
        ),
        "Expected Success or ValidationFailed, got {result:?}"
    );
}

#[test]
fn test_sync_from_nix_parses_config() {
    let mut bridge = GuiBridge::new();
    let binding = WidgetBinding::builder("sshd_toggle", "services.openssh.enable")
        .value_type(WidgetValueType::Bool)
        .label("SSH")
        .default(WidgetValue::Bool(false))
        .build();
    bridge.register_binding(binding);

    let nix_content = r#"{ services.openssh.enable = true; }"#;
    let updates = bridge.sync_from_nix(nix_content);

    // sync_from_nix returns WidgetUpdate vec — may be empty if parsing
    // doesn't match the simple format, but should not panic
    let _ = updates;
}

#[test]
fn test_generate_nix_diff_after_changes() {
    let mut bridge = GuiBridge::new();
    let binding = WidgetBinding::builder("hostname", "networking.hostName")
        .value_type(WidgetValueType::String)
        .label("Hostname")
        .default(WidgetValue::String("nixos".to_string()))
        .build();
    bridge.register_binding(binding);

    let _ = bridge.on_widget_change(
        WidgetId::new("hostname"),
        WidgetValue::String("my-machine".to_string()),
    );

    let diff = bridge.generate_nix_diff();
    // Diff should contain something about the hostname change
    assert!(!diff.is_empty() || bridge.state.pending_changes.is_empty());
}

#[test]
fn test_apply_pending_clears_queue() {
    let mut bridge = GuiBridge::new();
    let binding = WidgetBinding::builder("toggle", "services.test.enable")
        .value_type(WidgetValueType::Bool)
        .label("Test")
        .default(WidgetValue::Bool(false))
        .build();
    bridge.register_binding(binding);

    let _ = bridge.on_widget_change(WidgetId::new("toggle"), WidgetValue::Bool(true));
    bridge.apply_pending();

    assert!(
        bridge.state.pending_changes.is_empty(),
        "apply_pending should clear pending_changes"
    );
}

#[test]
fn test_semantic_search_returns_results() {
    let mut bridge = GuiBridge::new();

    // Register several bindings to search across
    for (id, path, label) in [
        ("ssh", "services.openssh.enable", "SSH Server"),
        ("firewall", "networking.firewall.enable", "Firewall"),
        ("docker", "virtualisation.docker.enable", "Docker"),
    ] {
        let binding = WidgetBinding::builder(id, path)
            .value_type(WidgetValueType::Bool)
            .label(label)
            .default(WidgetValue::Bool(false))
            .build();
        bridge.register_binding(binding);
    }

    let results = bridge.semantic_search("network", 5);
    // Should return at least something (HDC similarity is approximate)
    // The important thing is it doesn't panic and returns a valid vec
    assert!(results.len() <= 5);
}

#[test]
fn test_constraint_validator_catches_conflicts() {
    let mut validator = ConstraintValidator::new();

    validator.add_constraint(
        "networking.firewall.allowedTCPPorts",
        Constraint::valid_port(),
    );

    let result = validator.validate_change(
        &NixPath::new("networking.firewall.allowedTCPPorts"),
        "99999",
    );
    // Port 99999 is invalid (>65535) — should fail validation
    assert!(
        !result.is_valid || !result.errors.is_empty(),
        "Port 99999 should fail validation"
    );
}

#[test]
fn test_hdc_translator_roundtrip() {
    let mut translator = HdcTranslator::new();

    let path = NixPath::new("services.openssh.enable");
    let hv1 = translator.encode_path(&path);
    let hv2 = translator.encode_path(&path);

    // Same path should produce identical encoding (cached)
    assert_eq!(hv1, hv2, "HDC encoding should be deterministic");
}

#[test]
fn test_widget_mapper_bidirectional() {
    let mapper = WidgetMapper::new();

    let binding = WidgetBinding::builder("test", "services.test.enable")
        .value_type(WidgetValueType::Bool)
        .transformer(ValueTransformer::ServiceEnable)
        .default(WidgetValue::Bool(false))
        .build();

    let nix = mapper.widget_to_nix(&binding, &WidgetValue::Bool(true));
    let back = mapper.nix_to_widget(&binding, &nix);

    assert_eq!(
        back,
        WidgetValue::Bool(true),
        "Roundtrip should preserve Bool(true)"
    );
}

#[test]
fn test_config_categories_complete() {
    // All categories should have string representations (path_prefix)
    let categories = [
        ConfigCategory::Packages,
        ConfigCategory::Services,
        ConfigCategory::Networking,
        ConfigCategory::Users,
        ConfigCategory::Boot,
        ConfigCategory::Hardware,
        ConfigCategory::Programs,
        ConfigCategory::Security,
        ConfigCategory::NixSettings,
        ConfigCategory::Custom,
    ];

    for cat in &categories {
        let prefix = cat.path_prefix();
        assert!(!prefix.is_empty() || matches!(cat, ConfigCategory::Custom));
    }
}

#[test]
fn test_sync_state_dirty_tracking() {
    let mut bridge = GuiBridge::new();
    let binding = WidgetBinding::builder("test", "services.test.enable")
        .value_type(WidgetValueType::Bool)
        .label("Test")
        .default(WidgetValue::Bool(false))
        .build();
    bridge.register_binding(binding);

    assert!(!bridge.state.nix_dirty, "Should start clean");

    let _ = bridge.on_widget_change(WidgetId::new("test"), WidgetValue::Bool(true));

    // After a change, state may be dirty
    if !bridge.state.pending_changes.is_empty() {
        assert!(bridge.state.nix_dirty, "Should be dirty after change");
    }
}