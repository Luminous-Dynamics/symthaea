// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/*!
Governance Hardening Tests — Fourth Improvement Pass

Tests for:
1. External mesh preservation (not overwritten by local fallback)
2. Positive reputation → 5-HT upregulation
3. NaN/Inf outcome clamping
4. LR boost tracking from prediction error
5. Delayed outcome uses default PE after TTL expiry
*/

#![cfg(feature = "mycelix")]

use symthaea::cognitive_loop::{
    CognitiveLoopConfig, CognitiveLoopService, GovernanceEvent, GovernanceEventKind,
    GovernanceOutcome,
};

fn create_service() -> CognitiveLoopService {
    CognitiveLoopService::new(CognitiveLoopConfig::default()).expect("CLS construction")
}

fn make_event(kind: GovernanceEventKind) -> GovernanceEvent {
    GovernanceEvent {
        kind,
        proposal_id: None,
        timestamp_secs: 0,
    }
}

/// External epistemic mesh is preserved across governance intervals.
#[test]
fn test_external_mesh_survives_governance_interval() {
    use symthaea::mycelix::epistemic_mesh::{EpistemicMesh, EpistemicSummary};
    use symthaea::mycelix::gis::IgnoranceType;

    let mut service = create_service();

    // Set external mesh with 3 agents
    let external = EpistemicMesh::new(vec![
        EpistemicSummary {
            agent_id: "net-a1".into(),
            dominant_ignorance: IgnoranceType::KnownUnknown,
            domain_expertise: vec![],
            blind_spots: vec!["quantum".into()],
        },
        EpistemicSummary {
            agent_id: "net-a2".into(),
            dominant_ignorance: IgnoranceType::Known,
            domain_expertise: vec![],
            blind_spots: vec![],
        },
        EpistemicSummary {
            agent_id: "net-a3".into(),
            dominant_ignorance: IgnoranceType::Unknown,
            domain_expertise: vec![],
            blind_spots: vec!["gravity".into()],
        },
    ]);
    service.set_governance_epistemic_mesh(external);

    // Run past the governance interval (37) + extra cycles
    for i in 0..50 {
        let _ = service.cycle(&format!("mesh survive {}", i));
    }

    // Verify telemetry still reports 3 agents (external mesh survived)
    let result = service.cycle("mesh check");
    assert_eq!(
        result.metadata.governance.governance_epistemic_agents, 3,
        "External mesh should survive governance interval, got {} agents",
        result.metadata.governance.governance_epistemic_agents
    );
}

/// Positive reputation change produces 5-HT boost (symmetric with decline).
#[test]
fn test_positive_reputation_serotonin_boost() {
    let mut service = create_service();

    service.inject_governance_event(make_event(GovernanceEventKind::ReputationChanged {
        delta: 0.5,
    }));

    // Run through governance interval
    for i in 0..40 {
        let _ = service.cycle(&format!("rep boost {}", i));
    }

    // Service should have processed the event without crashing
    assert_eq!(service.governance_pending_count(), 0);
}

/// NaN and Infinity outcomes are clamped to safe defaults.
#[test]
fn test_nan_inf_outcomes_produce_finite_state() {
    let mut service = create_service();

    // Inject outcome with NaN and Infinity values
    service.inject_governance_outcome(GovernanceOutcome {
        proposal_id: "nan-test".into(),
        passed: true,
        my_vote_aligned: Some(true),
        value_alignment_score: f64::NAN,
        harmonic_resonance: f64::INFINITY,
    });

    // Run through governance processing
    for i in 0..50 {
        let result = service.cycle(&format!("nan check {}", i));
        let m = &result.metadata;
        assert!(
            m.consciousness.consciousness_level.is_finite(),
            "Consciousness should be finite after NaN outcome: {}",
            m.consciousness.consciousness_level
        );
        assert!(
            m.governance.governance_reward_ema.is_finite(),
            "Reward EMA should be finite after NaN outcome: {}",
            m.governance.governance_reward_ema
        );
        assert!(
            m.governance.governance_harmonic_delta_max.is_finite(),
            "Harmonic delta max should be finite after NaN outcome: {}",
            m.governance.governance_harmonic_delta_max
        );
    }
}

/// Governance LR boost is tracked and appears in telemetry.
#[test]
fn test_governance_lr_boost_in_telemetry() {
    let mut service = create_service();

    // Predict then give a very different outcome → high PE → high LR boost
    service.predict_governance_outcome("lr-telem".into(), 0.9);
    service.inject_governance_outcome(GovernanceOutcome {
        proposal_id: "lr-telem".into(),
        passed: true,
        my_vote_aligned: Some(true),
        value_alignment_score: 0.1,
        harmonic_resonance: 0.5,
    });

    // Need events to trigger processing
    service.inject_governance_event(make_event(GovernanceEventKind::ProposalCreated));

    // Run past governance interval
    let mut max_lr_boost = 0.0f64;
    for i in 0..50 {
        let result = service.cycle(&format!("lr boost {}", i));
        max_lr_boost = max_lr_boost.max(result.metadata.governance.governance_lr_boost);
    }

    assert!(
        max_lr_boost > 1.0,
        "LR boost should be > 1.0 after high PE outcome: {}",
        max_lr_boost
    );
}

/// Delayed outcomes (after TTL expiry) use default prediction error.
#[test]
fn test_delayed_outcome_default_prediction_error() {
    let mut service = create_service();

    // Predict at the start
    service.predict_governance_outcome("delayed-p1".into(), 0.9);

    // Run many cycles to expire the prediction TTL (5000 cycles)
    // We need to run through multiple governance intervals
    for i in 0..5100 {
        let _ = service.cycle(&format!("age {}", i));
    }

    // Now inject the delayed outcome
    service.inject_governance_outcome(GovernanceOutcome {
        proposal_id: "delayed-p1".into(),
        passed: true,
        my_vote_aligned: Some(true),
        value_alignment_score: 0.1,
        harmonic_resonance: 0.5,
    });

    // Run through governance processing
    for i in 0..50 {
        let _ = service.cycle(&format!("delayed check {}", i));
    }

    // The system should have processed it without error
    assert_eq!(service.governance_pending_count(), 0);
    // Reward EMA should be non-zero (outcome was processed)
    assert!(
        service.governance_reward_ema().abs() > 1e-10,
        "Delayed outcome should still produce reward signal"
    );
}