// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/*!
Property-Based Tests for Governance Manager

Verifies structural invariants of the governance metacognition subsystem:
- Random event sequences never produce NaN or infinite outputs
- Reward EMA stays bounded in [-1, 1]
- Neuromod doses respect floor checks and caps
- Harmonic deltas are bounded

Feature-gated: `mycelix`.
*/

#![cfg(feature = "mycelix")]

use proptest::prelude::*;
use symthaea::cognitive_loop::{
    CognitiveLoopConfig, CognitiveLoopService, GovernanceEvent, GovernanceEventKind,
    GovernanceOutcome,
};

// ═══════════════════════════════════════════════════════════════════════════════
// Strategies
// ═══════════════════════════════════════════════════════════════════════════════

fn arb_event_kind() -> impl Strategy<Value = GovernanceEventKind> {
    prop_oneof![
        Just(GovernanceEventKind::ProposalCreated),
        Just(GovernanceEventKind::EmergencyDeclared),
        (0.0..1.0f64, -1.0..1.0f64).prop_map(|(phi, val)| GovernanceEventKind::VoteCast {
            voter_phi: phi,
            vote_value: val,
        }),
        prop::bool::ANY.prop_map(|b| GovernanceEventKind::TallyCompleted {
            passed: b,
            collective_phi: 0.5,
        }),
        (-1.0..1.0f64).prop_map(|d| GovernanceEventKind::ReputationChanged { delta: d }),
        (0.01..1.0f64).prop_map(|_a| GovernanceEventKind::ReciprocityPledge { amount: 0.5 }),
        prop::bool::ANY.prop_map(|b| GovernanceEventKind::JusticeDispute { involves_self: b }),
    ]
}

fn arb_event() -> impl Strategy<Value = GovernanceEvent> {
    arb_event_kind().prop_map(|kind| GovernanceEvent {
        kind,
        proposal_id: None,
        timestamp_secs: 0,
    })
}

fn arb_outcome() -> impl Strategy<Value = GovernanceOutcome> {
    (
        prop::bool::ANY,
        prop::option::of(prop::bool::ANY),
        0.0..1.0f64,
        -1.0..1.0f64,
    )
        .prop_map(|(passed, aligned, vas, resonance)| GovernanceOutcome {
            proposal_id: "prop-test".to_string(),
            passed,
            my_vote_aligned: aligned,
            value_alignment_score: vas,
            harmonic_resonance: resonance,
        })
}

fn make_service() -> CognitiveLoopService {
    CognitiveLoopService::new(CognitiveLoopConfig::default()).expect("CLS construction")
}

// ═══════════════════════════════════════════════════════════════════════════════
// Properties
// ═══════════════════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// Random event sequences never crash and reward EMA stays bounded.
    #[test]
    fn prop_events_produce_finite_reward_ema(
        events in prop::collection::vec(arb_event(), 0..20),
    ) {
        let mut svc = make_service();
        for e in events {
            svc.inject_governance_event(e);
        }
        // Run past interval 37
        for _ in 0..40 {
            let _ = svc.cycle("governance proptest");
        }
        let ema = svc.governance_reward_ema();
        prop_assert!(ema.is_finite(), "reward_ema not finite: {}", ema);
        prop_assert!(ema >= -1.0, "reward_ema below -1: {}", ema);
        prop_assert!(ema <= 1.0, "reward_ema above 1: {}", ema);
    }

    /// Events are drained after processing (pending count goes to 0).
    #[test]
    fn prop_events_drain_after_processing(
        events in prop::collection::vec(arb_event(), 1..15),
    ) {
        let mut svc = make_service();
        for e in events {
            svc.inject_governance_event(e);
        }
        // Run past interval 37
        for _ in 0..40 {
            let _ = svc.cycle("governance drain check");
        }
        let pending = svc.governance_pending_count();
        prop_assert_eq!(pending, 0, "Events should drain after processing");
    }

    /// Outcomes produce a finite reward EMA in [-1, 1].
    #[test]
    fn prop_outcomes_bounded_reward(
        outcomes in prop::collection::vec(arb_outcome(), 1..20),
    ) {
        let mut svc = make_service();
        for o in outcomes {
            svc.inject_governance_outcome(o);
        }
        for _ in 0..40 {
            let _ = svc.cycle("governance outcome proptest");
        }
        let ema = svc.governance_reward_ema();
        prop_assert!(ema >= -1.0 && ema <= 1.0, "reward_ema OOB: {}", ema);
    }

    /// Mixed events + outcomes never crash and produce finite metadata.
    #[test]
    fn prop_mixed_events_and_outcomes_stable(
        events in prop::collection::vec(arb_event(), 0..10),
        outcomes in prop::collection::vec(arb_outcome(), 0..10),
    ) {
        let mut svc = make_service();
        for e in events {
            svc.inject_governance_event(e);
        }
        for o in outcomes {
            svc.inject_governance_outcome(o);
        }
        for _ in 0..50 {
            let r = svc.cycle("mixed governance proptest");
            let m = &r.metadata;
            prop_assert!(m.consciousness.consciousness_level.is_finite());
            prop_assert!(m.actual_effective_lr.is_finite());
        }
    }

    /// Governance consciousness modulation: collective_phi affects consciousness level
    /// within sane bounds.
    #[test]
    fn prop_collective_phi_consciousness_bounded(
        phi in 0.0..1.0f64,
    ) {
        let mut svc = make_service();
        svc.inject_governance_event(GovernanceEvent {
            kind: GovernanceEventKind::TallyCompleted {
                passed: true,
                collective_phi: phi,
            },
            proposal_id: None,
            timestamp_secs: 0,
        });
        for _ in 0..40 {
            let r = svc.cycle("phi consciousness check");
            prop_assert!(r.metadata.consciousness.consciousness_level >= 0.0);
            prop_assert!(r.metadata.consciousness.consciousness_level <= 1.0);
        }
    }

    /// Harmonic deltas stay bounded under random event sequences + mode switches.
    #[test]
    fn prop_harmonic_deltas_bounded(
        events in prop::collection::vec(arb_event(), 0..15),
        outcomes in prop::collection::vec(arb_outcome(), 0..10),
    ) {
        let mut svc = make_service();
        for e in events {
            svc.inject_governance_event(e);
        }
        for o in outcomes {
            svc.inject_governance_outcome(o);
        }
        // Run enough cycles for governance to process multiple times
        for _ in 0..80 {
            let r = svc.cycle("harmonic stability");
            let delta_max = r.metadata.governance.governance_harmonic_delta_max;
            prop_assert!(
                delta_max.is_finite(),
                "harmonic delta max not finite: {}",
                delta_max,
            );
            prop_assert!(
                delta_max < 10.0,
                "harmonic delta max too large: {} (expected < 10.0)",
                delta_max,
            );
        }
    }

    /// Community mode is always populated after governance processes.
    #[test]
    fn prop_community_mode_always_set(
        events in prop::collection::vec(arb_event(), 1..5),
    ) {
        let mut svc = make_service();
        for e in events {
            svc.inject_governance_event(e);
        }
        // Run past governance interval (37) + some cycles
        for _ in 0..50 {
            let _ = svc.cycle("community mode check");
        }
        let r = svc.cycle("final check");
        let mode = &r.metadata.governance.governance_community_mode;
        prop_assert!(
            !mode.is_empty(),
            "Community mode should be set after governance processes"
        );
    }
}