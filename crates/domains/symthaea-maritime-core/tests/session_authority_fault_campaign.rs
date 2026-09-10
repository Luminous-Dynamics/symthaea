// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_maritime_core::{
    AuthenticatedMachineSession, MachineSessionContext, MachineSessionPolicy, MachineSessionTrust,
    MaritimeObservationKind, ObservationBindingError, ObservationSourceGrantV1,
    ObservationSourcePolicyV1, bind_observation_to_trusted_session, evaluate_machine_session,
};

const SCHEMA: &str = "xenia-verified-machine-session-evidence-v1";
const IDENTITY: &str =
    "xenia-signing-identity-v1:blake3-256:12fb7634b7b55b14b996d5b984e4ce93f5db26e0aac9d5d177f2622dbeda412e";
const PLATFORM: &str = "auv-01";
const AUTHENTICATED_AT_MS: u64 = 1_700_000_000_000;
const EXPIRES_AT_MS: u64 = 1_700_000_060_000;
const EPOCH: u64 = 9;

fn session() -> AuthenticatedMachineSession {
    AuthenticatedMachineSession::from_verified_provider(
        SCHEMA,
        "session-fixture-001",
        IDENTITY,
        AUTHENTICATED_AT_MS,
        EXPIRES_AT_MS,
        EPOCH,
        "xenia-handshake-transcript-v1:blake3-256:2222222222222222222222222222222222222222222222222222222222222222",
    )
    .unwrap()
}

fn context(
    provider: &str,
    identity: &str,
    now_ms: u64,
    epoch: u64,
    trusted_time: bool,
    revoked: bool,
) -> MachineSessionContext {
    MachineSessionContext::from_authority_provider(
        provider,
        identity,
        now_ms,
        epoch,
        trusted_time,
        revoked,
    )
    .unwrap()
}

fn session_policy() -> MachineSessionPolicy<'static> {
    MachineSessionPolicy {
        accepted_schemas: &[SCHEMA],
        max_validity_ms: 60_000,
    }
}

fn source_policy() -> ObservationSourcePolicyV1 {
    ObservationSourcePolicyV1::new([ObservationSourceGrantV1::new(
        SCHEMA, IDENTITY, PLATFORM,
    )
    .unwrap()])
    .unwrap()
}

fn try_bind(ctx: &MachineSessionContext) -> Result<String, ObservationBindingError> {
    bind_observation_to_trusted_session(
        &session(),
        ctx,
        session_policy(),
        &source_policy(),
        PLATFORM,
        1_700_000_030_000_000,
        MaritimeObservationKind::HealthObservation,
        r#"{"severity":"healthy"}"#,
        &["mycelix-position:measurement:001".into()],
    )
    .map(|bound| bound.evidence_binding().to_owned())
}

#[test]
fn progressive_authority_failures_never_mint_a_new_observation_binding() {
    let trusted = context(
        SCHEMA,
        IDENTITY,
        1_700_000_030_000,
        EPOCH,
        true,
        false,
    );
    let baseline = try_bind(&trusted).expect("baseline fixture must bind");
    assert!(baseline.starts_with("symthaea-maritime-session-bound-observation-v1:blake3-256:"));

    let cases = [
        (
            context(
                SCHEMA,
                IDENTITY,
                1_700_000_030_001,
                EPOCH,
                true,
                true,
            ),
            MachineSessionTrust::Revoked,
        ),
        (
            context(
                SCHEMA,
                IDENTITY,
                1_700_000_030_002,
                EPOCH + 1,
                true,
                false,
            ),
            MachineSessionTrust::EpochMismatch,
        ),
        (
            context(
                SCHEMA,
                IDENTITY,
                1_700_000_030_003,
                EPOCH,
                false,
                false,
            ),
            MachineSessionTrust::UntrustedTime,
        ),
        (
            context(
                "other-provider-v1",
                IDENTITY,
                1_700_000_030_004,
                EPOCH,
                true,
                false,
            ),
            MachineSessionTrust::ContextProviderMismatch,
        ),
        (
            context(
                SCHEMA,
                "xenia-signing-identity-v1:blake3-256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                1_700_000_030_005,
                EPOCH,
                true,
                false,
            ),
            MachineSessionTrust::ContextIdentityMismatch,
        ),
        (
            context(SCHEMA, IDENTITY, EXPIRES_AT_MS, EPOCH, true, false),
            MachineSessionTrust::Expired,
        ),
        (
            context(
                SCHEMA,
                IDENTITY,
                AUTHENTICATED_AT_MS - 1,
                EPOCH,
                true,
                false,
            ),
            MachineSessionTrust::NotYetValid,
        ),
    ];

    for (ctx, expected_trust) in cases {
        assert_eq!(
            evaluate_machine_session(&session(), &ctx, session_policy()),
            expected_trust
        );
        assert_eq!(
            try_bind(&ctx),
            Err(ObservationBindingError::SessionNotTrusted(expected_trust))
        );
    }
}

#[test]
fn fresh_session_authority_does_not_override_platform_source_policy() {
    let trusted = context(
        SCHEMA,
        IDENTITY,
        1_700_000_030_000,
        EPOCH,
        true,
        false,
    );
    let deny_all = ObservationSourcePolicyV1::new([]).unwrap();

    assert_eq!(
        bind_observation_to_trusted_session(
            &session(),
            &trusted,
            session_policy(),
            &deny_all,
            PLATFORM,
            1_700_000_030_000_000,
            MaritimeObservationKind::HealthObservation,
            r#"{"severity":"healthy"}"#,
            &[],
        ),
        Err(ObservationBindingError::SourceNotAuthorized)
    );
}
