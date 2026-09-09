// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_maritime_core::{
    AuthenticatedMachineSession, MachineSessionContext, MachineSessionPolicy, MachineSessionTrust,
    evaluate_machine_session,
};

const XENIA_SCHEMA_V1: &str = "xenia-verified-machine-session-evidence-v1";
const EVIDENCE_FIXTURE: &str =
    include_str!("../fixtures/xenia-verified-machine-session-evidence-v1.json");

fn policy() -> MachineSessionPolicy<'static> {
    MachineSessionPolicy {
        accepted_schemas: &[XENIA_SCHEMA_V1],
        max_validity_ms: 60_000,
    }
}

#[test]
fn xenia_v1_session_evidence_projects_into_provider_neutral_contract() {
    // Xenia carries provider-specific negotiated-context metadata in addition
    // to the maritime-owned immutable trust subset. The provider schema itself
    // is retained and explicitly allowlisted at point of use.
    let session: AuthenticatedMachineSession = serde_json::from_str(EVIDENCE_FIXTURE).unwrap();

    assert_eq!(session.schema, XENIA_SCHEMA_V1);
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
fn fresh_local_policy_and_authority_context_drive_fail_closed_evaluation() {
    let session: AuthenticatedMachineSession = serde_json::from_str(EVIDENCE_FIXTURE).unwrap();

    // Intentionally constructed in-process. Current time, authority generation,
    // revocation state and provider-schema acceptance are local policy facts,
    // not replayable cross-repo wire artifacts.
    let context = MachineSessionContext {
        now_ms: 1_700_000_030_000,
        authority_epoch: 9,
        trusted_time_available: true,
        revoked: false,
    };

    assert_eq!(
        evaluate_machine_session(&session, context, policy()),
        MachineSessionTrust::Trusted
    );

    let mut revoked = context;
    revoked.revoked = true;
    assert_eq!(
        evaluate_machine_session(&session, revoked, policy()),
        MachineSessionTrust::Revoked
    );

    let mut rotated_authority = context;
    rotated_authority.authority_epoch = 10;
    assert_eq!(
        evaluate_machine_session(&session, rotated_authority, policy()),
        MachineSessionTrust::EpochMismatch
    );

    let mut time_untrusted = context;
    time_untrusted.trusted_time_available = false;
    assert_eq!(
        evaluate_machine_session(&session, time_untrusted, policy()),
        MachineSessionTrust::UntrustedTime
    );
}

#[test]
fn future_xenia_schema_and_excessive_validity_require_explicit_review() {
    let context = MachineSessionContext {
        now_ms: 1_700_000_030_000,
        authority_epoch: 9,
        trusted_time_available: true,
        revoked: false,
    };
    let mut session: AuthenticatedMachineSession = serde_json::from_str(EVIDENCE_FIXTURE).unwrap();

    session.schema = "xenia-verified-machine-session-evidence-v2".into();
    assert_eq!(
        evaluate_machine_session(&session, context, policy()),
        MachineSessionTrust::UnsupportedSchema
    );

    session.schema = XENIA_SCHEMA_V1.into();
    session.expires_at_ms += 1;
    assert_eq!(
        evaluate_machine_session(&session, context, policy()),
        MachineSessionTrust::ValidityTooLong
    );
}
