// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

// Compile and exercise the source module directly until a later public-surface PR
// adds the module export to symthaea-communication::lib.rs.
#[path = "../src/adult_fantasy_session.rs"]
mod adult_fantasy_session;

use adult_fantasy_session::*;

struct ToggleVerifier(bool);

impl AdultEligibilityVerifier for ToggleVerifier {
    fn verify_adult_eligibility(
        &self,
        _evidence: &AdultEligibilityEvidenceHandleV1,
        _participant_id: &str,
        _now_ns: u64,
    ) -> bool {
        self.0
    }
}

fn activation(epoch: u64) -> AdultFantasyActivationV1 {
    AdultFantasyActivationV1::new(
        "participant-a",
        "session-a",
        epoch,
        FantasyRealityFrameV1::ExplicitRoleplay,
        FantasyRetentionPolicyV1::Ephemeral,
        FantasyIdentityPolicyV1::OriginalOrFictionalOnly,
    )
    .unwrap()
}

fn evidence() -> AdultEligibilityEvidenceHandleV1 {
    AdultEligibilityEvidenceHandleV1::new("eligibility-1", "participant-a", 100, 300).unwrap()
}

#[test]
fn end_to_end_stop_then_fresh_epoch_reentry() {
    let mut session = AdultFantasySessionV1::new("participant-a", "session-a").unwrap();
    session
        .activate(activation(4), evidence(), &ToggleVerifier(true), 150)
        .unwrap();
    assert!(session.is_active());

    session.stop(AdultFantasyStopReasonV1::ExplicitExit).unwrap();
    assert!(!session.is_active());
    session
        .acknowledge_exit_to_ordinary_conversation()
        .unwrap();

    assert_eq!(
        session.activate(activation(4), evidence(), &ToggleVerifier(true), 160),
        Err(AdultFantasySessionError::StaleSessionEpoch)
    );
    session
        .activate(activation(5), evidence(), &ToggleVerifier(true), 160)
        .unwrap();
    assert!(session.is_active());
}

#[test]
fn live_verifier_loss_terminates_fantasy_eligibility() {
    let mut session = AdultFantasySessionV1::new("participant-a", "session-a").unwrap();
    session
        .activate(activation(1), evidence(), &ToggleVerifier(true), 150)
        .unwrap();
    assert!(!session.revalidate(&ToggleVerifier(false), 160).unwrap());
    assert_eq!(
        session.state(),
        AdultFantasySessionStateV1::Stopped {
            terminal_session_epoch: 1,
            reason: AdultFantasyStopReasonV1::AdultEligibilityNoLongerValid,
        }
    );
}

#[test]
fn activation_cannot_swap_participant_identity() {
    let wrong = AdultEligibilityEvidenceHandleV1::new(
        "eligibility-other",
        "participant-b",
        100,
        300,
    )
    .unwrap();
    let mut session = AdultFantasySessionV1::new("participant-a", "session-a").unwrap();
    assert_eq!(
        session.activate(activation(1), wrong, &ToggleVerifier(true), 150),
        Err(AdultFantasySessionError::ParticipantMismatch)
    );
}
