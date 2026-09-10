// SPDX-License-Identifier: AGPL-3.0-or-later

use serde::Deserialize;
use symthaea_maritime_core::{
    AuthenticatedMachineSession, MachineSessionContext, MachineSessionPolicy, MachineSessionTrust,
    evaluate_machine_session,
};

const XENIA_SCHEMA_V1: &str = "xenia-verified-machine-session-evidence-v1";
const XENIA_IDENTITY_BINDING: &str =
    "xenia-signing-identity-v1:blake3-256:12fb7634b7b55b14b996d5b984e4ce93f5db26e0aac9d5d177f2622dbeda412e";
const XENIA_TRANSCRIPT_BINDING: &str =
    "xenia-handshake-transcript-v1:blake3-256:2222222222222222222222222222222222222222222222222222222222222222";
const EVIDENCE_FIXTURE: &str =
    include_str!("../fixtures/xenia-verified-machine-session-evidence-v1.json");

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct XeniaSessionFixtureV1 {
    schema: String,
    session_id: String,
    peer_identity_binding: String,
    authenticated_at_ms: u64,
    expires_at_ms: u64,
    authority_epoch: u64,
    evidence_binding: String,
    negotiated_context_binding: Option<String>,
}

fn policy() -> MachineSessionPolicy<'static> {
    MachineSessionPolicy {
        accepted_schemas: &[XENIA_SCHEMA_V1],
        max_validity_ms: 60_000,
    }
}

/// Test-only stand-in for the future Xenia adapter.
///
/// The real adapter must receive provider-verified Xenia evidence; parsing this
/// JSON fixture is only an executable schema-compatibility check. The explicit
/// constructor call mirrors the trust-boundary crossing without claiming that
/// JSON parsing itself verifies Xenia cryptography.
fn project_fixture_after_provider_verification() -> AuthenticatedMachineSession {
    let wire: XeniaSessionFixtureV1 = serde_json::from_str(EVIDENCE_FIXTURE).unwrap();
    assert_eq!(wire.schema, XENIA_SCHEMA_V1);
    assert_eq!(wire.peer_identity_binding, XENIA_IDENTITY_BINDING);
    assert!(wire
        .negotiated_context_binding
        .as_deref()
        .is_some_and(|binding| binding.starts_with("xenia-negotiated-session-context:blake3-256:")));

    AuthenticatedMachineSession::from_verified_provider(
        wire.schema,
        wire.session_id,
        wire.peer_identity_binding,
        wire.authenticated_at_ms,
        wire.expires_at_ms,
        wire.authority_epoch,
        wire.evidence_binding,
    )
    .unwrap()
}

fn context_for(
    session: &AuthenticatedMachineSession,
    now_ms: u64,
    authority_epoch: u64,
    trusted_time_available: bool,
    revoked: bool,
) -> MachineSessionContext {
    MachineSessionContext::from_authority_provider(
        session.peer_identity_binding(),
        now_ms,
        authority_epoch,
        trusted_time_available,
        revoked,
    )
    .unwrap()
}

#[test]
fn xenia_v1_session_evidence_projects_into_provider_neutral_contract() {
    let session = project_fixture_after_provider_verification();

    assert_eq!(session.schema(), XENIA_SCHEMA_V1);
    assert_eq!(session.session_id(), "session-fixture-001");
    assert_eq!(session.authenticated_at_ms(), 1_700_000_000_000);
    assert_eq!(session.expires_at_ms(), 1_700_000_060_000);
    assert_eq!(session.authority_epoch(), 9);
    assert_eq!(session.peer_identity_binding(), XENIA_IDENTITY_BINDING);
    assert!(session
        .evidence_binding()
        .starts_with("xenia-handshake-transcript-v1:blake3-256:"));
    assert_eq!(session.validate_shape(), Ok(()));
}

#[test]
fn fresh_local_policy_and_authority_context_drive_fail_closed_evaluation() {
    let session = project_fixture_after_provider_verification();

    let context = context_for(&session, 1_700_000_030_000, 9, true, false);
    assert_eq!(
        evaluate_machine_session(&session, &context, policy()),
        MachineSessionTrust::Trusted
    );

    let revoked = context_for(&session, 1_700_000_030_000, 9, true, true);
    assert_eq!(
        evaluate_machine_session(&session, &revoked, policy()),
        MachineSessionTrust::Revoked
    );

    let rotated_authority = context_for(&session, 1_700_000_030_000, 10, true, false);
    assert_eq!(
        evaluate_machine_session(&session, &rotated_authority, policy()),
        MachineSessionTrust::EpochMismatch
    );

    let time_untrusted = context_for(&session, 1_700_000_030_000, 9, false, false);
    assert_eq!(
        evaluate_machine_session(&session, &time_untrusted, policy()),
        MachineSessionTrust::UntrustedTime
    );

    let wrong_identity = MachineSessionContext::from_authority_provider(
        "xenia-signing-identity-v1:blake3-256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        1_700_000_030_000,
        9,
        true,
        false,
    )
    .unwrap();
    assert_eq!(
        evaluate_machine_session(&session, &wrong_identity, policy()),
        MachineSessionTrust::ContextIdentityMismatch
    );
}

#[test]
fn future_xenia_schema_and_excessive_validity_require_explicit_review() {
    let future = AuthenticatedMachineSession::from_verified_provider(
        "xenia-verified-machine-session-evidence-v2",
        "session-fixture-001",
        XENIA_IDENTITY_BINDING,
        1_700_000_000_000,
        1_700_000_060_000,
        9,
        XENIA_TRANSCRIPT_BINDING,
    )
    .unwrap();
    let future_context = context_for(&future, 1_700_000_030_000, 9, true, false);
    assert_eq!(
        evaluate_machine_session(&future, &future_context, policy()),
        MachineSessionTrust::UnsupportedSchema
    );

    let too_long = AuthenticatedMachineSession::from_verified_provider(
        XENIA_SCHEMA_V1,
        "session-fixture-001",
        XENIA_IDENTITY_BINDING,
        1_700_000_000_000,
        1_700_000_060_001,
        9,
        XENIA_TRANSCRIPT_BINDING,
    )
    .unwrap();
    let too_long_context = context_for(&too_long, 1_700_000_030_000, 9, true, false);
    assert_eq!(
        evaluate_machine_session(&too_long, &too_long_context, policy()),
        MachineSessionTrust::ValidityTooLong
    );
}
