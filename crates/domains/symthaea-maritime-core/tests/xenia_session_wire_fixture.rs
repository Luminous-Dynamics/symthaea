// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_maritime_core::{
    AuthenticatedMachineSession, MachineSessionContext, MachineSessionTrust, evaluate_machine_session,
};

const EVIDENCE_FIXTURE: &str =
    include_str!("../fixtures/xenia-verified-machine-session-evidence-v1.json");
const CONTEXT_FIXTURE: &str =
    include_str!("../fixtures/xenia-machine-session-authority-context-v1.json");

#[test]
fn xenia_v1_session_evidence_projects_into_provider_neutral_contract() {
    // Xenia carries provider-specific `schema` and negotiated-context fields.
    // Serde intentionally ignores those here: maritime core owns only the
    // provider-neutral subset needed for point-of-use authority evaluation.
    let session: AuthenticatedMachineSession = serde_json::from_str(EVIDENCE_FIXTURE).unwrap();

    assert_eq!(session.session_id, "session-fixture-001");
    assert_eq!(session.authenticated_at_ms, 1_700_000_000_000);
    assert_eq!(session.expires_at_ms, 1_700_000_060_000);
    assert_eq!(session.authority_epoch, 9);
    assert!(session
        .peer_identity_binding
        .starts_with("xenia-signing-identity-v1:blake3-256:"));
    assert!(session
        .evidence_binding
        .starts_with("xenia-handshake-transcript-v1:blake3-256:"));
    assert_eq!(session.validate_shape(), Ok(()));
}

#[test]
fn xenia_live_authority_context_drives_fail_closed_session_evaluation() {
    let session: AuthenticatedMachineSession = serde_json::from_str(EVIDENCE_FIXTURE).unwrap();
    let context: MachineSessionContext = serde_json::from_str(CONTEXT_FIXTURE).unwrap();

    assert_eq!(
        evaluate_machine_session(&session, context),
        MachineSessionTrust::Trusted
    );

    let mut revoked = context;
    revoked.revoked = true;
    assert_eq!(
        evaluate_machine_session(&session, revoked),
        MachineSessionTrust::Revoked
    );

    let mut rotated_authority = context;
    rotated_authority.authority_epoch = 10;
    assert_eq!(
        evaluate_machine_session(&session, rotated_authority),
        MachineSessionTrust::EpochMismatch
    );

    let mut time_untrusted = context;
    time_untrusted.trusted_time_available = false;
    assert_eq!(
        evaluate_machine_session(&session, time_untrusted),
        MachineSessionTrust::UntrustedTime
    );
}
