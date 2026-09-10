// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_maritime_core::{
    AuthenticatedMachineSession, ComponentHealth, DependencyState, HealthSeverity,
    MachineSessionContext, MachineSessionPolicy, MachineSessionTrust, MaritimeObservationKind,
    NavigationQuality, ObservationBindingError, ObservationSourceGrantV1,
    ObservationSourcePolicyV1, OperatingEnvelope, PlatformHealth, baseline_envelope,
    bind_observation_to_trusted_session,
};

const SCHEMA: &str = "xenia-verified-machine-session-evidence-v1";
const PRINCIPAL: &str = "xenia-signing-identity-v1:blake3-256:fixture-principal";
const PLATFORM: &str = "auv-01";

fn source_policy() -> ObservationSourcePolicyV1 {
    ObservationSourcePolicyV1::new([ObservationSourceGrantV1::new(
        SCHEMA, PRINCIPAL, PLATFORM,
    )
    .unwrap()])
    .unwrap()
}

fn session(
    session_id: &str,
    authenticated_at_ms: u64,
    expires_at_ms: u64,
    authority_epoch: u64,
    evidence_binding: &str,
) -> AuthenticatedMachineSession {
    AuthenticatedMachineSession::from_verified_provider(
        SCHEMA,
        session_id,
        PRINCIPAL,
        authenticated_at_ms,
        expires_at_ms,
        authority_epoch,
        evidence_binding,
    )
    .unwrap()
}

fn context(
    now_ms: u64,
    authority_epoch: u64,
    trusted_time: bool,
    revoked: bool,
) -> MachineSessionContext {
    MachineSessionContext::from_authority_provider(
        SCHEMA,
        PRINCIPAL,
        now_ms,
        authority_epoch,
        trusted_time,
        revoked,
    )
    .unwrap()
}

fn session_policy() -> MachineSessionPolicy<'static> {
    MachineSessionPolicy {
        accepted_schemas: &[SCHEMA],
        max_validity_ms: 1_000,
    }
}

fn healthy_platform() -> PlatformHealth {
    PlatformHealth {
        platform_id: PLATFORM.into(),
        observed_at_ms: 1_500,
        components: vec![ComponentHealth {
            component_id: "navigation".into(),
            severity: HealthSeverity::Healthy,
            code: "nominal".into(),
            detail: String::new(),
        }],
    }
}

#[test]
fn partition_revocation_and_readmission_never_inflate_authority() {
    let old_session = session(
        "session-before-partition",
        1_000,
        2_000,
        9,
        "xenia-handshake-transcript-v1:blake3-256:old-session",
    );
    let source_policy = source_policy();

    let historical = bind_observation_to_trusted_session(
        &old_session,
        &context(1_500, 9, true, false),
        session_policy(),
        &source_policy,
        PLATFORM,
        1_400_000,
        MaritimeObservationKind::HealthObservation,
        r#"{"severity":"healthy"}"#,
        &[],
    )
    .unwrap();
    let historical_binding = historical.evidence_binding().to_owned();

    // Losing fleet/operator reachability reduces the local operating envelope but does not
    // manufacture new authority or force a healthy platform into a connectivity-dependent stop.
    let partitioned = DependencyState {
        fleet_link_available: false,
        remote_operator_available: false,
        trusted_time_available: true,
        external_positioning_available: true,
    };
    assert_eq!(
        baseline_envelope(
            partitioned,
            NavigationQuality::Nominal,
            &healthy_platform(),
        ),
        OperatingEnvelope::ReducedCapability
    );

    // Once the authority source reports revocation, no new observation may be associated with
    // the old session. Previously minted evidence remains an immutable historical association;
    // it must not be reinterpreted as current authorization.
    assert_eq!(
        bind_observation_to_trusted_session(
            &old_session,
            &context(1_600, 9, true, true),
            session_policy(),
            &source_policy,
            PLATFORM,
            1_550_000,
            MaritimeObservationKind::HealthObservation,
            r#"{"severity":"degraded"}"#,
            &[],
        ),
        Err(ObservationBindingError::SessionNotTrusted(
            MachineSessionTrust::Revoked
        ))
    );
    assert_eq!(historical.evidence_binding(), historical_binding.as_str());

    // Rotating the authority generation also invalidates the old session even if it has not
    // reached its timestamp expiry.
    assert_eq!(
        bind_observation_to_trusted_session(
            &old_session,
            &context(1_600, 10, true, false),
            session_policy(),
            &source_policy,
            PLATFORM,
            1_550_000,
            MaritimeObservationKind::HealthObservation,
            r#"{"severity":"degraded"}"#,
            &[],
        ),
        Err(ObservationBindingError::SessionNotTrusted(
            MachineSessionTrust::EpochMismatch
        ))
    );

    // Recovery begins with a newly admitted session in the new epoch. Its evidence is a new
    // association, never a revival or mutation of the pre-revocation binding.
    let new_session = session(
        "session-after-readmission",
        1_600,
        2_600,
        10,
        "xenia-handshake-transcript-v1:blake3-256:new-session",
    );
    let recovered = bind_observation_to_trusted_session(
        &new_session,
        &context(1_700, 10, true, false),
        session_policy(),
        &source_policy,
        PLATFORM,
        1_650_000,
        MaritimeObservationKind::HealthObservation,
        r#"{"severity":"healthy"}"#,
        &[],
    )
    .unwrap();

    assert_ne!(recovered.evidence_binding(), historical_binding.as_str());
    assert_eq!(historical.evidence_binding(), historical_binding.as_str());
}

#[test]
fn loss_of_trusted_time_blocks_new_binding_during_partition() {
    let session = session(
        "session-time-loss",
        1_000,
        2_000,
        9,
        "xenia-handshake-transcript-v1:blake3-256:time-loss",
    );
    let source_policy = source_policy();

    assert_eq!(
        bind_observation_to_trusted_session(
            &session,
            &context(1_500, 9, false, false),
            session_policy(),
            &source_policy,
            PLATFORM,
            1_400_000,
            MaritimeObservationKind::StateObservation,
            "{}",
            &[],
        ),
        Err(ObservationBindingError::SessionNotTrusted(
            MachineSessionTrust::UntrustedTime
        ))
    );

    let dependencies = DependencyState {
        fleet_link_available: false,
        remote_operator_available: false,
        trusted_time_available: false,
        external_positioning_available: true,
    };
    assert_eq!(
        baseline_envelope(
            dependencies,
            NavigationQuality::Nominal,
            &healthy_platform(),
        ),
        OperatingEnvelope::SafeTransit
    );
}
