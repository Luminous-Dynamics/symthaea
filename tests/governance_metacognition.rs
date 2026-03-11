// ==================================================================================
// Integration Test: Governance ↔ Consciousness Metacognitive Loop
// ==================================================================================
//
// Verifies the complete Symthaea-Mycelix metacognitive integration:
//   1. Governance events flow into CLS and produce stable cycles
//   2. Governance outcomes are processed without crashes
//   3. Epistemic mesh detects blind spots and escalates proposals
//   4. KosmicSong credentials reflect governance identity
//   5. Collective identity derives community mode
//
// Requires: --features mycelix
// ==================================================================================

#![cfg(feature = "mycelix")]

use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};
use symthaea::cognitive_loop::{GovernanceEvent, GovernanceEventKind, GovernanceOutcome};

// ── Helper ──────────────────────────────────────────────────────────

fn create_service() -> CognitiveLoopService {
    CognitiveLoopService::new(CognitiveLoopConfig {
        learning_threshold: 0.0,
        ..Default::default()
    })
    .unwrap()
}

fn make_event(kind: GovernanceEventKind) -> GovernanceEvent {
    GovernanceEvent {
        kind,
        proposal_id: Some("test-proposal".into()),
        timestamp_secs: 0,
    }
}

// ── Test 1: Emergency event processed without crash ─────────────────

#[test]
fn test_governance_emergency_processed_stably() {
    let mut service = create_service();

    // Inject emergency event
    service.inject_governance_event(make_event(GovernanceEventKind::EmergencyDeclared));

    // Run enough cycles for interval 37 to fire and neuromod to apply
    for i in 0..50 {
        let result = service.cycle(&format!("governance cycle {}", i));
        assert!(result.prediction_error >= 0.0);
        assert!(result.prediction_error <= 1.0);
    }

    assert_eq!(service.stats().total_cycles, 50);
}

// ── Test 2: Reciprocity pledges processed ───────────────────────────

#[test]
fn test_reciprocity_pledges_processed() {
    let mut service = create_service();

    // Inject 5 reciprocity pledges
    for _ in 0..5 {
        service.inject_governance_event(make_event(GovernanceEventKind::ReciprocityPledge {
            amount: 1.0,
        }));
    }

    // Run cycles to process
    for i in 0..50 {
        service.cycle(&format!("reciprocity {}", i));
    }

    assert_eq!(service.stats().total_cycles, 50);
}

// ── Test 3: Aligned outcome processed stably ────────────────────────

#[test]
fn test_aligned_outcome_processed() {
    let mut service = create_service();

    // Run a few baseline cycles
    for i in 0..10 {
        service.cycle(&format!("baseline {}", i));
    }

    // Inject a positive aligned outcome
    service.inject_governance_outcome(GovernanceOutcome {
        proposal_id: "p1".into(),
        passed: true,
        my_vote_aligned: Some(true),
        value_alignment_score: 0.9,
        harmonic_resonance: 0.8,
    });

    // Run enough cycles for GovernanceManager (interval 37) to process
    for i in 0..50 {
        let result = service.cycle(&format!("post-outcome {}", i));
        assert!(result.prediction_error >= 0.0);
    }

    assert_eq!(service.stats().total_cycles, 60);
}

// ── Test 4: Misaligned outcome processed stably ─────────────────────

#[test]
fn test_misaligned_outcome_processed() {
    let mut service = create_service();

    for i in 0..10 {
        service.cycle(&format!("baseline {}", i));
    }

    // Inject a negative misaligned outcome
    service.inject_governance_outcome(GovernanceOutcome {
        proposal_id: "p2".into(),
        passed: true,
        my_vote_aligned: Some(false),
        value_alignment_score: 0.1,
        harmonic_resonance: 0.1,
    });

    for i in 0..50 {
        service.cycle(&format!("post-misalign {}", i));
    }

    // System should remain stable even with negative governance signals
    assert_eq!(service.stats().total_cycles, 60);
}

// ── Test 5: Epistemic mesh blind spot detection ─────────────────────

#[test]
fn test_epistemic_mesh_blind_spot_escalation() {
    use symthaea::mycelix::epistemic_mesh::{EpistemicMesh, EpistemicSummary, EscalationTier};
    use symthaea::mycelix::gis::IgnoranceType;

    // 4 out of 5 agents uncertain about "quantum" → severity 0.8 → Guardian
    let summaries = vec![
        EpistemicSummary {
            agent_id: "a1".into(),
            dominant_ignorance: IgnoranceType::KnownUnknown,
            domain_expertise: vec![],
            blind_spots: vec!["quantum".into()],
        },
        EpistemicSummary {
            agent_id: "a2".into(),
            dominant_ignorance: IgnoranceType::KnownUnknown,
            domain_expertise: vec![],
            blind_spots: vec!["quantum".into()],
        },
        EpistemicSummary {
            agent_id: "a3".into(),
            dominant_ignorance: IgnoranceType::KnownUnknown,
            domain_expertise: vec![],
            blind_spots: vec!["quantum".into()],
        },
        EpistemicSummary {
            agent_id: "a4".into(),
            dominant_ignorance: IgnoranceType::KnownUnknown,
            domain_expertise: vec![],
            blind_spots: vec!["quantum".into()],
        },
        EpistemicSummary {
            agent_id: "a5".into(),
            dominant_ignorance: IgnoranceType::Known,
            domain_expertise: vec![("quantum".into(), 0.9)],
            blind_spots: vec![],
        },
    ];

    let mesh = EpistemicMesh::new(summaries);

    let tier = mesh.proposal_escalation_required(&["quantum".into()]);
    assert_eq!(tier, Some(EscalationTier::Guardian));

    let expert = EpistemicSummary {
        agent_id: "a5".into(),
        dominant_ignorance: IgnoranceType::Known,
        domain_expertise: vec![("quantum".into(), 0.9)],
        blind_spots: vec![],
    };
    let boost = mesh.expertise_boost(&expert, &["quantum".into()]);
    assert!(boost > 1.0, "Expert should get boost: {}", boost);
}

// ── Test 6: KosmicSong governance credential ────────────────────────

#[test]
fn test_kosmic_song_governance_credential() {
    let ks = symthaea::mycelix::KosmicSong::default();
    let cred = ks.governance_credential();

    assert!(!cred.agent_id.is_empty());
    assert!(cred.coherence_score >= 0.0);
    assert!(cred.maturation_signal >= 0.0);
}

// ── Test 7: Collective identity from credentials ────────────────────

#[test]
fn test_collective_identity_from_credentials() {
    use symthaea::mycelix::collective_identity::{CollectiveKosmicSong, CommunityMode};
    use symthaea::mycelix::gis::{IgnoranceType, MoralUncertainty};
    use symthaea::mycelix::kosmic_song::KosmicCredential;
    use symthaea_types::Harmony;

    let credentials = vec![
        KosmicCredential {
            agent_id: "a1".into(),
            dominant_harmony: Harmony::PanSentientFlourishing,
            coherence_score: 0.8,
            moral_uncertainty: MoralUncertainty::default(),
            gis_type: IgnoranceType::None,
            maturation_signal: 0.9,
        },
        KosmicCredential {
            agent_id: "a2".into(),
            dominant_harmony: Harmony::PanSentientFlourishing,
            coherence_score: 0.7,
            moral_uncertainty: MoralUncertainty::default(),
            gis_type: IgnoranceType::None,
            maturation_signal: 0.8,
        },
        KosmicCredential {
            agent_id: "a3".into(),
            dominant_harmony: Harmony::SacredReciprocity,
            coherence_score: 0.6,
            moral_uncertainty: MoralUncertainty::default(),
            gis_type: IgnoranceType::None,
            maturation_signal: 0.7,
        },
    ];

    let song = CollectiveKosmicSong::from_credentials(&credentials);
    assert_eq!(song.n_agents, 3);
    assert_eq!(song.community_mode, CommunityMode::Protective);
    assert!(song.coherence > 0.0);
}

// ── Test 8: Full loop — diverse governance events through 50 cycles ──

#[test]
fn test_full_governance_loop_50_cycles() {
    let mut service = create_service();

    // Phase 1: Inject diverse governance events
    service.inject_governance_event(make_event(GovernanceEventKind::EmergencyDeclared));
    service.inject_governance_event(make_event(GovernanceEventKind::ReciprocityPledge {
        amount: 5.0,
    }));
    service.inject_governance_event(make_event(GovernanceEventKind::JusticeDispute {
        involves_self: true,
    }));
    service.inject_governance_event(make_event(GovernanceEventKind::ReputationChanged {
        delta: -0.3,
    }));
    service.inject_governance_event(make_event(GovernanceEventKind::TallyCompleted {
        passed: true,
        collective_phi: 0.7,
    }));

    // Phase 2: Inject governance outcomes
    service.inject_governance_outcome(GovernanceOutcome {
        proposal_id: "p1".into(),
        passed: true,
        my_vote_aligned: Some(true),
        value_alignment_score: 0.85,
        harmonic_resonance: 0.7,
    });
    service.inject_governance_outcome(GovernanceOutcome {
        proposal_id: "p2".into(),
        passed: false,
        my_vote_aligned: Some(false),
        value_alignment_score: 0.2,
        harmonic_resonance: 0.3,
    });

    // Run 50 cycles
    for i in 0..50 {
        let result = service.cycle(&format!("governance integration cycle {}", i));

        // Every cycle should produce valid output
        assert!(
            result.prediction_error >= 0.0 && result.prediction_error <= 1.0,
            "Cycle {} PE out of bounds: {}",
            i,
            result.prediction_error,
        );
    }

    // Verify system is stable after governance processing
    assert_eq!(service.stats().total_cycles, 50);
    assert!(
        service.stats().avg_prediction_error >= 0.0,
        "Average PE should remain non-negative"
    );
}
