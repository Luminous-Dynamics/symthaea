// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Defense cascade and immune system integration tests.
//!
//! Tests the graduated defense response: SafetyLevel → proposed actions →
//! moral algebra filter → approved actions. Verifies proportionality,
//! moral filtering, and escalation/de-escalation dynamics.
//!
//! Run: `cargo test --test defense_cascade_tests`

use symthaea::cognitive_loop::defense::{
    DefenseAction, DefenseActionKind, moral_filter, propose_defense_actions,
};
use symthaea::safety::SafetyLevel;

// ═══════════════════════════════════════════════════════════════════════════════
// Graduated response — proportionality
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_green_proposes_no_actions() {
    let actions = propose_defense_actions(SafetyLevel::Green, 100);
    assert!(
        actions.is_empty(),
        "Green level should propose no defensive actions"
    );
}

#[test]
fn test_yellow_proposes_monitoring_only() {
    let actions = propose_defense_actions(SafetyLevel::Yellow, 100);
    assert!(!actions.is_empty(), "Yellow should propose some actions");

    // Yellow actions should be low-severity monitoring
    for action in &actions {
        assert!(
            action.moral_severity < 0.3,
            "Yellow actions should have low severity: {} has {:.2}",
            action.kind,
            action.moral_severity
        );
    }
}

#[test]
fn test_orange_proposes_intervention() {
    let actions = propose_defense_actions(SafetyLevel::Orange, 100);
    assert!(
        !actions.is_empty(),
        "Orange should propose intervention actions"
    );

    // Should include some actions that affect others
    let has_quarantine_or_suspend = actions.iter().any(|a| {
        matches!(
            a.kind,
            DefenseActionKind::QuarantinePeer { .. }
                | DefenseActionKind::SuspendNonGuardianProposals
                | DefenseActionKind::RestrictMotorToReadOnly
        )
    });
    assert!(
        has_quarantine_or_suspend,
        "Orange should include intervention actions"
    );
}

#[test]
fn test_red_proposes_emergency_halt() {
    let actions = propose_defense_actions(SafetyLevel::Red, 100);
    assert!(!actions.is_empty(), "Red should propose emergency actions");

    let has_halt = actions
        .iter()
        .any(|a| matches!(a.kind, DefenseActionKind::HaltMotor));
    assert!(has_halt, "Red should include motor halt");
}

#[test]
fn test_escalation_increases_action_count() {
    let yellow = propose_defense_actions(SafetyLevel::Yellow, 100);
    let orange = propose_defense_actions(SafetyLevel::Orange, 100);
    let red = propose_defense_actions(SafetyLevel::Red, 100);

    assert!(
        orange.len() >= yellow.len(),
        "Orange should have >= Yellow actions: {} vs {}",
        orange.len(),
        yellow.len()
    );
    assert!(
        red.len() >= orange.len(),
        "Red should have >= Orange actions: {} vs {}",
        red.len(),
        orange.len()
    );
}

#[test]
fn test_escalation_increases_max_severity() {
    let yellow = propose_defense_actions(SafetyLevel::Yellow, 100);
    let orange = propose_defense_actions(SafetyLevel::Orange, 100);
    let red = propose_defense_actions(SafetyLevel::Red, 100);

    let max_sev = |actions: &[DefenseAction]| {
        actions
            .iter()
            .map(|a| a.moral_severity)
            .fold(0.0f32, f32::max)
    };

    assert!(
        max_sev(&orange) >= max_sev(&yellow),
        "Orange max severity should >= Yellow"
    );
    assert!(
        max_sev(&red) >= max_sev(&orange),
        "Red max severity should >= Orange"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Moral algebra filter
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_moral_filter_approves_low_severity() {
    let mut actions = propose_defense_actions(SafetyLevel::Yellow, 100);
    moral_filter(&mut actions);

    // Low-severity actions should be approved
    for action in &actions {
        if action.moral_severity < 0.2 {
            assert!(
                action.morally_approved,
                "Low-severity action should be approved: {} (sev={:.2})",
                action.kind, action.moral_severity
            );
        }
    }
}

#[test]
fn test_all_actions_have_expiry() {
    for level in [SafetyLevel::Yellow, SafetyLevel::Orange, SafetyLevel::Red] {
        let actions = propose_defense_actions(level, 100);
        for action in &actions {
            assert!(
                action.expires_at > action.proposed_at,
                "{:?} action {:?} should have expiry > proposed",
                level,
                action.kind
            );
        }
    }
}

#[test]
fn test_actions_all_start_unapproved() {
    for level in [SafetyLevel::Yellow, SafetyLevel::Orange, SafetyLevel::Red] {
        let actions = propose_defense_actions(level, 100);
        for action in &actions {
            assert!(
                !action.morally_approved,
                "Actions should start unapproved before moral filter"
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Action properties
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_base_severity_bounded() {
    let all_kinds = [
        DefenseActionKind::IncreasePeerScoring {
            interval_reduction: 0.5,
        },
        DefenseActionKind::RateLimitProposals {
            agent_id: "test".into(),
            max_per_minute: 5,
        },
        DefenseActionKind::BoostVigilance { ne_delta: 0.1 },
        DefenseActionKind::QuarantinePeer {
            peer_id: "test".into(),
            duration_cycles: 100,
        },
        DefenseActionKind::SuspendNonGuardianProposals,
        DefenseActionKind::RestrictMotorToReadOnly,
        DefenseActionKind::StressResponse {
            cortisol_delta: 0.2,
        },
        DefenseActionKind::EmergencyGovernanceFreeze,
        DefenseActionKind::DisconnectUntrusted,
        DefenseActionKind::PhysicalSafePosition,
        DefenseActionKind::EmergencyBeacon,
        DefenseActionKind::HaltMotor,
    ];

    for kind in &all_kinds {
        let sev = kind.base_severity();
        assert!(
            sev >= 0.0 && sev <= 1.0,
            "{} has severity {sev} outside [0,1]",
            kind
        );
    }
}

#[test]
fn test_affects_others_classification() {
    // Actions affecting others should have higher base severity
    let self_only = [
        DefenseActionKind::BoostVigilance { ne_delta: 0.1 },
        DefenseActionKind::RestrictMotorToReadOnly,
        DefenseActionKind::PhysicalSafePosition,
        DefenseActionKind::HaltMotor,
    ];

    let affects_others = [
        DefenseActionKind::QuarantinePeer {
            peer_id: "test".into(),
            duration_cycles: 100,
        },
        DefenseActionKind::DisconnectUntrusted,
        DefenseActionKind::EmergencyGovernanceFreeze,
    ];

    let max_self_sev = self_only
        .iter()
        .map(|k| k.base_severity())
        .fold(0.0f32, f32::max);
    let min_other_sev = affects_others
        .iter()
        .map(|k| k.base_severity())
        .fold(f32::MAX, f32::min);

    // Actions affecting others should generally have higher severity
    assert!(
        min_other_sev > max_self_sev,
        "Other-affecting actions should have higher severity: min_other={min_other_sev} vs max_self={max_self_sev}"
    );
}

#[test]
fn test_defense_action_display() {
    let kind = DefenseActionKind::HaltMotor;
    let display = format!("{}", kind);
    assert!(!display.is_empty());
    assert_eq!(display, "HaltMotor");
}