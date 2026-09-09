// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_maritime_core::{
    AuthenticatedMachineSession, MachineSessionContext, MachineSessionTrust,
    evaluate_machine_session,
};

const FIXTURE: &str = include_str!("../fixtures/xenia-verified-machine-session-v1.json");

#[test]
fn xenia_v1_provider_fixture_is_consumable_without_xenia_dependency() {
    // Xenia carries provider metadata (`schema`, negotiated-context binding) in
    // addition to the minimal maritime consumer fields. Serde's normal
    // forward-compatible handling lets maritime core consume only the trust
    // facts it owns without reimplementing Xenia's schema.
    let session: AuthenticatedMachineSession = serde_json::from_str(FIXTURE).unwrap();

    assert_eq!(session.session_id, "session-fixture-01");
    assert_eq!(session.authenticated_at_ms, 1_700_000_000_000);
    assert_eq!(session.expires_at_ms, 1_700_000_300_000);
    assert_eq!(session.authority_epoch, 9);
    assert!(session.validate_shape().is_ok());

    let live = MachineSessionContext {
        now_ms: 1_700_000_100_000,
        authority_epoch: 9,
        trusted_time_available: true,
        revoked: false,
    };
    assert_eq!(
        evaluate_machine_session(&session, live),
        MachineSessionTrust::Trusted
    );
}

#[test]
fn same_immutable_xenia_fixture_fails_closed_when_live_authority_changes() {
    let session: AuthenticatedMachineSession = serde_json::from_str(FIXTURE).unwrap();

    let revoked = MachineSessionContext {
        now_ms: 1_700_000_100_000,
        authority_epoch: 9,
        trusted_time_available: true,
        revoked: true,
    };
    assert_eq!(
        evaluate_machine_session(&session, revoked),
        MachineSessionTrust::Revoked
    );

    let replaced_authority = MachineSessionContext {
        now_ms: 1_700_000_100_000,
        authority_epoch: 10,
        trusted_time_available: true,
        revoked: false,
    };
    assert_eq!(
        evaluate_machine_session(&session, replaced_authority),
        MachineSessionTrust::EpochMismatch
    );

    let no_trusted_time = MachineSessionContext {
        now_ms: 1_700_000_100_000,
        authority_epoch: 9,
        trusted_time_available: false,
        revoked: false,
    };
    assert_eq!(
        evaluate_machine_session(&session, no_trusted_time),
        MachineSessionTrust::UntrustedTime
    );
}
