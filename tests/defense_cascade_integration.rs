// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration tests for the decentralized defensive force cascade.
//!
//! Tests the full immune system pipeline:
//! SafetyAgent → safety_enforcement → defense actions → moral filter
//! Sentinel → threat memory → collective immunity → guardian posture

use symthaea::cognitive_loop::defense::{DefenseActionKind, moral_filter, propose_defense_actions};
use symthaea::cognitive_loop::guardian::{GuardianPosture, GuardianState};
use symthaea::cognitive_loop::safety_enforcement::compute_enforcement;
use symthaea::safety::{SafetyAgent, SafetyLevel};

// ═══════════════════════════════════════════════════════════════════════════════
// Phase 0: Safety enforcement cascade
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_safety_enforcement_green_no_gates() {
    let mut agent = SafetyAgent::new();
    let result = compute_enforcement(&mut agent, 0.8, 0.2, 0.9, false, 100, None);
    assert_eq!(result.level, SafetyLevel::Green);
    assert!((result.lr_multiplier - 1.0).abs() < 1e-6);
    assert!(!result.motor_halt);
    assert!(!result.motor_readonly);
}

#[test]
fn test_safety_enforcement_red_halts_everything() {
    let mut agent = SafetyAgent::new();
    let result = compute_enforcement(&mut agent, 0.05, 0.9, 0.1, true, 100, None);
    assert_eq!(result.level, SafetyLevel::Red);
    assert!(result.motor_halt);
    assert!(result.lr_multiplier < 0.1);
    assert!(result.exploration_multiplier < 0.01);
    assert!(result.ne_nudge > 0.0);
    assert!(result.allostatic_load > 0.0);
}

#[test]
fn test_safety_enforcement_sustained_yellow_escalates() {
    let mut agent = SafetyAgent::new();
    // Feed 3 cycles of Yellow-level consciousness (below 0.6 but above 0.35)
    for i in 0..4 {
        let result = compute_enforcement(&mut agent, 0.5, 0.2, 0.9, false, i, None);
        if i < 3 {
            assert_eq!(result.level, SafetyLevel::Yellow);
        }
    }
    // After 3 sustained Yellow cycles, trend escalation kicks in
    let result = compute_enforcement(&mut agent, 0.65, 0.2, 0.9, false, 5, None);
    // Even with 0.65 (normally Green), sustained Yellow history escalates
    assert!(result.level >= SafetyLevel::Yellow);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Phase 3: Graduated defense response
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_defense_cascade_green_to_red() {
    let green = propose_defense_actions(SafetyLevel::Green, 0);
    assert!(green.is_empty(), "Green should produce no defense actions");

    let yellow = propose_defense_actions(SafetyLevel::Yellow, 100);
    assert!(!yellow.is_empty());
    assert!(yellow.iter().all(|a| !a.kind.affects_others()
        || a.kind == DefenseActionKind::BoostVigilance { ne_delta: 0.03 }
        || matches!(a.kind, DefenseActionKind::IncreasePeerScoring { .. })));

    let orange = propose_defense_actions(SafetyLevel::Orange, 200);
    assert!(
        orange
            .iter()
            .any(|a| matches!(a.kind, DefenseActionKind::SuspendNonGuardianProposals))
    );

    let red = propose_defense_actions(SafetyLevel::Red, 300);
    assert!(
        red.iter()
            .any(|a| matches!(a.kind, DefenseActionKind::EmergencyGovernanceFreeze))
    );
    assert!(
        red.iter()
            .any(|a| matches!(a.kind, DefenseActionKind::HaltMotor))
    );
}

#[test]
fn test_moral_filter_blocks_excessive_severity() {
    let mut actions = propose_defense_actions(SafetyLevel::Red, 0);
    moral_filter(&mut actions);

    // Self-protective actions should always be approved
    for action in &actions {
        if !action.kind.affects_others() {
            assert!(
                action.morally_approved,
                "Self-protective action {:?} should be approved",
                action.kind
            );
        }
    }

    // Actions affecting others need severity < 0.7
    for action in &actions {
        if action.kind.affects_others() {
            assert!(
                action.moral_severity < 0.7,
                "Action {:?} severity {} should be < 0.7 for approval",
                action.kind,
                action.moral_severity
            );
        }
    }
}

#[test]
fn test_all_defense_actions_have_expiry() {
    for level in [SafetyLevel::Yellow, SafetyLevel::Orange, SafetyLevel::Red] {
        let actions = propose_defense_actions(level, 100);
        for action in &actions {
            assert!(
                action.expires_at > action.proposed_at,
                "Action {:?} at {:?} should have expiry > proposal time",
                action.kind,
                level
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Phase 4: Guardian posture transitions
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_guardian_full_lifecycle() {
    let mut state = GuardianState::default();

    // Start at Hold (default)
    assert_eq!(state.posture, GuardianPosture::Hold);
    assert!(!state.posture.locomotion_permitted());

    // High Phi + Green → Normal
    state.update(SafetyLevel::Green, 0.8, 10);
    assert_eq!(state.posture, GuardianPosture::Normal);
    assert!(state.posture.locomotion_permitted());

    // Start patrol
    state.start_patrol(5);
    assert!(state.patrol_active);

    // Yellow → Cautious (patrol continues, walk gait only)
    state.update(SafetyLevel::Yellow, 0.8, 20);
    assert_eq!(state.posture, GuardianPosture::Cautious);
    assert!(state.posture.locomotion_permitted());

    // Orange → Defensive (patrol stops)
    state.update(SafetyLevel::Orange, 0.8, 30);
    assert_eq!(state.posture, GuardianPosture::Defensive);
    assert!(!state.posture.locomotion_permitted());
    assert!(!state.patrol_active);

    // Red → Emergency (e-stop)
    state.update(SafetyLevel::Red, 0.8, 40);
    assert_eq!(state.posture, GuardianPosture::Emergency);
    assert!(state.posture.e_stop_engaged());

    // Recovery: back to Green
    state.update(SafetyLevel::Green, 0.8, 50);
    assert_eq!(state.posture, GuardianPosture::Normal);
    assert!(!state.posture.e_stop_engaged());

    // Consciousness drop → Hold (overrides safety level)
    state.update(SafetyLevel::Green, 0.1, 60);
    assert_eq!(state.posture, GuardianPosture::Hold);
}

#[test]
fn test_guardian_emergency_cycle_tracking() {
    let mut state = GuardianState::default();
    state.update(SafetyLevel::Red, 0.8, 0);

    for i in 1..=100 {
        state.update(SafetyLevel::Red, 0.8, i);
    }
    assert_eq!(state.emergency_cycles, 100);
}

// ═══════════════════════════════════════════════════════════════════════════════
// End-to-end: Safety → Defense → Guardian cascade
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_full_cascade_safety_to_guardian() {
    let mut agent = SafetyAgent::new();
    let mut guardian = GuardianState::default();
    guardian.update(SafetyLevel::Green, 0.8, 0);
    assert_eq!(guardian.posture, GuardianPosture::Normal);

    // Simulate consciousness degradation over cycles
    let degradation_steps = vec![
        (0.7, SafetyLevel::Green),
        (0.5, SafetyLevel::Yellow),
        (0.3, SafetyLevel::Orange),
        (0.1, SafetyLevel::Red),
    ];

    for (i, (consciousness, expected_level)) in degradation_steps.iter().enumerate() {
        let enforcement =
            compute_enforcement(&mut agent, *consciousness, 0.3, 0.5, false, i + 1, None);
        assert_eq!(
            enforcement.level, *expected_level,
            "Cycle {}: expected {:?}",
            i, expected_level
        );

        // Update guardian posture
        guardian.update(enforcement.level, *consciousness as f64, i + 1);

        // Propose and filter defense actions
        let mut actions = propose_defense_actions(enforcement.level, i + 1);
        moral_filter(&mut actions);

        // Verify proportionality
        match expected_level {
            SafetyLevel::Green => {
                assert!(actions.is_empty());
                assert!(guardian.posture.locomotion_permitted());
            }
            SafetyLevel::Yellow => {
                assert!(!actions.is_empty());
                assert!(guardian.posture.locomotion_permitted());
            }
            SafetyLevel::Orange => {
                assert!(
                    actions
                        .iter()
                        .any(|a| matches!(a.kind, DefenseActionKind::SuspendNonGuardianProposals))
                );
                assert!(!guardian.posture.locomotion_permitted());
            }
            SafetyLevel::Red => {
                assert!(enforcement.motor_halt);
                assert!(guardian.posture.e_stop_engaged());
            }
        }
    }

    // Recovery: consciousness returns — use fresh agent to avoid trend escalation
    // from the degradation history above (SafetyAgent tracks recent levels and
    // escalates when it detects sustained Yellow+ trends).
    let mut fresh_agent = SafetyAgent::new();
    let recovery = compute_enforcement(&mut fresh_agent, 0.8, 0.2, 0.9, false, 10, None);
    assert_eq!(recovery.level, SafetyLevel::Green);
    guardian.update(recovery.level, 0.8, 10);
    assert_eq!(guardian.posture, GuardianPosture::Normal);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Full security pipeline: threat → collective immunity → safety escalation →
//                         defense cascade → guardian posture
// ═══════════════════════════════════════════════════════════════════════════════

/// Integration test: threat detected → collective immunity activates →
/// safety escalates → defense actions fire → guardian posture changes.
#[test]
fn security_pipeline_threat_to_guardian_posture() {
    use symthaea::cognitive_loop::collective_immunity::CollectiveImmuneState;

    // Step 1: Simulate a collective immune threat
    let mut immune = CollectiveImmuneState::default();
    immune.update(
        0.8,                                                     // local_threat_level: high
        &[("peer1".into(), 0.7), ("peer2".into(), 0.6)],         // peer reports
        0.4,                       // collective_phi: low (confused swarm)
        5,                         // total_peers
        &["ProposalFlood".into()], // local threats
        &["ProposalFlood".into(), "GradientFingerprint".into()], // swarm threats
    );

    // Verify immune response activated
    assert!(
        immune.immune_response_active,
        "Immune response should activate with high threats"
    );
    assert!(
        immune.coherence_adjusted_severity > 0.5,
        "Severity should be amplified by low phi, got {}",
        immune.coherence_adjusted_severity
    );
    assert_eq!(
        immune.blind_spots.len(),
        1,
        "Should detect blind spot: GradientFingerprint"
    );
    assert_eq!(immune.blind_spots[0], "GradientFingerprint");

    // Step 2: Safety enforcement with collective immunity
    let mut agent = SafetyAgent::new();
    let result = compute_enforcement(
        &mut agent,
        0.8,   // consciousness_level: normal
        0.2,   // prediction_error: low
        0.9,   // temporal_coherence: high
        false, // integrity_critical: no
        100,   // cycle
        Some(&immune),
    );

    // Should escalate from Green to at least Yellow (collective immunity triggers escalation)
    assert!(
        result.level >= SafetyLevel::Yellow,
        "Collective threat should escalate safety to at least Yellow, got {:?}",
        result.level
    );

    // Step 3: Propose defense actions based on escalated level
    let mut actions = propose_defense_actions(result.level, 100);
    assert!(
        !actions.is_empty(),
        "Should propose defense actions at {:?}",
        result.level
    );

    // Step 4: Moral algebra filter
    moral_filter(&mut actions);
    let approved_count = actions.iter().filter(|a| a.morally_approved).count();
    assert!(
        approved_count > 0,
        "At least some defense actions should be morally approved"
    );

    // Step 5: Guardian posture should reflect the safety level
    let mut guardian = GuardianState::default();
    let phi = 0.8;
    guardian.update(result.level, phi, 100);

    match result.level {
        SafetyLevel::Green => assert_eq!(guardian.posture, GuardianPosture::Normal),
        SafetyLevel::Yellow => assert_eq!(guardian.posture, GuardianPosture::Cautious),
        SafetyLevel::Orange => assert_eq!(guardian.posture, GuardianPosture::Defensive),
        SafetyLevel::Red => assert_eq!(guardian.posture, GuardianPosture::Emergency),
    }

    // Verify defense actions match the safety level
    for action in &actions {
        assert_eq!(action.trigger_level, result.level);
    }

    // Verify neuromodulator coupling is active under immune response
    let (ne, cortisol, oxy) = immune.neuromod_deltas();
    assert!(ne > 0.0, "NE should increase under active immune response");
    assert!(
        cortisol > 0.0,
        "Cortisol should increase under active immune response"
    );
    assert!(
        oxy > 0.0,
        "Oxytocin should increase for in-group solidarity"
    );
}

/// Test the full escalation ladder: Green → Yellow → Orange with collective immunity.
#[test]
fn escalation_ladder_green_to_orange() {
    use symthaea::cognitive_loop::collective_immunity::CollectiveImmuneState;

    let mut agent = SafetyAgent::new();
    let mut guardian = GuardianState::default();

    // Phase 1: No threat → Green
    let result = compute_enforcement(&mut agent, 0.8, 0.2, 0.9, false, 1, None);
    assert_eq!(result.level, SafetyLevel::Green);
    guardian.update(result.level, 0.8, 1);
    assert_eq!(guardian.posture, GuardianPosture::Normal);

    // Phase 2: Moderate threat → Yellow
    // sentinel_threat_level > 0.5 triggers Yellow escalation
    let moderate_immune = CollectiveImmuneState {
        swarm_threat_level: 0.6,
        coherence_adjusted_severity: 0.55,
        immune_response_active: false,
        ..Default::default()
    };
    let result = compute_enforcement(&mut agent, 0.8, 0.2, 0.9, false, 2, Some(&moderate_immune));
    assert!(
        result.level >= SafetyLevel::Yellow,
        "Moderate threat should reach at least Yellow, got {:?}",
        result.level
    );
    guardian.update(result.level, 0.8, 2);
    assert!(
        guardian.posture == GuardianPosture::Cautious
            || guardian.posture == GuardianPosture::Defensive,
        "Guardian should be at least Cautious, got {:?}",
        guardian.posture
    );

    // Phase 3: Severe threat → Orange (high threat + active immune response)
    let severe_immune = CollectiveImmuneState {
        swarm_threat_level: 0.8,
        coherence_adjusted_severity: 0.8,
        immune_response_active: true,
        ..Default::default()
    };
    let result = compute_enforcement(&mut agent, 0.8, 0.2, 0.9, false, 3, Some(&severe_immune));
    assert!(
        result.level >= SafetyLevel::Orange,
        "Severe threat with active immune response should reach at least Orange, got {:?}",
        result.level
    );
    guardian.update(result.level, 0.8, 3);
    assert!(
        !guardian.posture.locomotion_permitted(),
        "Guardian should not permit locomotion at {:?}",
        guardian.posture
    );

    // Verify guardian tracked transitions
    assert!(
        guardian.transition_count >= 2,
        "Should have recorded at least 2 transitions, got {}",
        guardian.transition_count
    );
}

/// Verify moral algebra blocks actions that affect others at extreme severity.
#[test]
fn moral_algebra_blocks_excessive_defense() {
    // Red level proposes maximum actions
    let mut actions = propose_defense_actions(SafetyLevel::Red, 0);
    moral_filter(&mut actions);

    // Self-protective actions should always be approved
    let halt = actions
        .iter()
        .find(|a| matches!(a.kind, DefenseActionKind::HaltMotor))
        .expect("Red should propose HaltMotor");
    assert!(
        halt.morally_approved,
        "HaltMotor (self-protective) should always be approved"
    );

    let safe_pos = actions
        .iter()
        .find(|a| matches!(a.kind, DefenseActionKind::PhysicalSafePosition))
        .expect("Red should propose PhysicalSafePosition");
    assert!(
        safe_pos.morally_approved,
        "PhysicalSafePosition should always be approved"
    );

    let beacon = actions
        .iter()
        .find(|a| matches!(a.kind, DefenseActionKind::EmergencyBeacon))
        .expect("Red should propose EmergencyBeacon");
    assert!(
        beacon.morally_approved,
        "EmergencyBeacon (self-protective) should always be approved"
    );

    // All approved actions affecting others should have bounded severity
    for action in &actions {
        if action.morally_approved && action.kind.affects_others() {
            assert!(
                action.moral_severity < 0.6,
                "Approved other-affecting action {:?} should have severity < 0.6, got {}",
                action.kind,
                action.moral_severity
            );
        }
    }

    // Verify at least one other-affecting action exists and is approved
    let other_affecting_approved = actions
        .iter()
        .filter(|a| a.kind.affects_others() && a.morally_approved)
        .count();
    assert!(
        other_affecting_approved > 0,
        "At least some other-affecting actions should pass moral filter at Red"
    );
}

/// Verify collective immunity produces neuromodulator deltas when active.
#[test]
fn collective_immunity_neuromod_coupling() {
    use symthaea::cognitive_loop::collective_immunity::CollectiveImmuneState;

    // Inactive state → no deltas
    let inactive = CollectiveImmuneState::default();
    let (ne, cortisol, oxy) = inactive.neuromod_deltas();
    assert_eq!(
        ne, 0.0,
        "Inactive immune state should produce zero NE delta"
    );
    assert_eq!(
        cortisol, 0.0,
        "Inactive immune state should produce zero cortisol delta"
    );
    assert_eq!(
        oxy, 0.0,
        "Inactive immune state should produce zero oxytocin delta"
    );

    // Active state with threat → positive deltas
    let mut active = CollectiveImmuneState::default();
    active.update(
        0.8,                             // high local threat
        &[("p1".into(), 0.7)],           // peer under threat
        0.3,                             // low collective phi (confused swarm)
        3,                               // total peers
        &[],                             // no local threat kinds
        &["GradientFingerprint".into()], // swarm-wide threats
    );
    assert!(
        active.immune_response_active,
        "Should activate with high threat severity"
    );

    let (ne, cortisol, oxy) = active.neuromod_deltas();
    assert!(ne > 0.0, "NE should increase under threat, got {}", ne);
    assert!(
        cortisol > 0.0,
        "Cortisol should increase under threat, got {}",
        cortisol
    );
    assert!(
        oxy > 0.0,
        "Oxytocin should increase for in-group solidarity, got {}",
        oxy
    );

    // Verify magnitudes scale with threat level
    // NE = swarm_threat_level * 0.04
    let expected_ne = active.swarm_threat_level as f64 * 0.04;
    assert!(
        (ne - expected_ne).abs() < 1e-6,
        "NE should be proportional to swarm_threat_level: expected {}, got {}",
        expected_ne,
        ne
    );

    // Cortisol = coherence_adjusted_severity * 0.03
    let expected_cortisol = active.coherence_adjusted_severity as f64 * 0.03;
    assert!(
        (cortisol - expected_cortisol).abs() < 1e-6,
        "Cortisol should be proportional to coherence_adjusted_severity: expected {}, got {}",
        expected_cortisol,
        cortisol
    );
}
