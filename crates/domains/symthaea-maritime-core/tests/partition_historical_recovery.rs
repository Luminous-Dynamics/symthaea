// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_maritime_core::{
    AuthenticatedMachineSession, ComponentHealth, DependencyState, HealthSeverity,
    HistoricallyQualifiedMachineSessionV1, MachineSessionContext, MachineSessionPolicy,
    MachineSessionTrust, MaritimeObservationKind, NavigationQuality, ObservationBindingError,
    ObservationSourceGrantV1, ObservationSourcePolicyV1, OperatingEnvelope, PlatformHealth,
    baseline_envelope, bind_observation_to_historically_qualified_session,
    bind_observation_to_trusted_session,
};

const SCHEMA: &str = "xenia-verified-machine-session-evidence-v1";
const PRINCIPAL: &str = "xenia-signing-identity-v1:blake3-256:fixture-principal";
const PLATFORM: &str = "auv-01";
const OLD_EVIDENCE: &str = "xenia-handshake-transcript-v1:blake3-256:old-session";
const NEW_EVIDENCE: &str = "xenia-handshake-transcript-v1:blake3-256:new-session";
const OLD_ADMISSION: &str = "xenia-machine-session-admission-v1:blake3-256:old-session";

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

fn session_policy() -> MachineSessionPolicy<'static> {
    MachineSessionPolicy {
        accepted_schemas: &[SCHEMA],
        max_validity_ms: 1_000,
    }
}

fn source_policy() -> ObservationSourcePolicyV1 {
    ObservationSourcePolicyV1::new([ObservationSourceGrantV1::new(
        SCHEMA, PRINCIPAL, PLATFORM,
    )
    .unwrap()])
    .unwrap()
}

fn context(now_ms: u64, authority_epoch: u64, revoked: bool) -> MachineSessionContext {
    MachineSessionContext::from_authority_provider(
        SCHEMA,
        PRINCIPAL,
        now_ms,
        authority_epoch,
        true,
        revoked,
    )
    .unwrap()
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
fn partition_reconnect_uses_history_for_the_past_and_new_admission_for_the_future() {
    let source_policy = source_policy();
    let old_session = session("session-before-partition", 1_000, 2_000, 9, OLD_EVIDENCE);

    let live_before_partition = bind_observation_to_trusted_session(
        &old_session,
        &context(1_500, 9, false),
        session_policy(),
        &source_policy,
        PLATFORM,
        1_400_000,
        MaritimeObservationKind::HealthObservation,
        r#"{"severity":"healthy"}"#,
        &["mycelix-position:measurement:001".into()],
    )
    .unwrap();
    let stable_pre_partition_binding = live_before_partition.evidence_binding().to_owned();

    assert_eq!(
        baseline_envelope(
            DependencyState {
                fleet_link_available: false,
                remote_operator_available: false,
                trusted_time_available: true,
                external_positioning_available: true,
            },
            NavigationQuality::Nominal,
            &healthy_platform(),
        ),
        OperatingEnvelope::ReducedCapability
    );

    assert_eq!(
        bind_observation_to_trusted_session(
            &old_session,
            &context(1_700, 9, true),
            session_policy(),
            &source_policy,
            PLATFORM,
            1_650_000,
            MaritimeObservationKind::HealthObservation,
            r#"{"severity":"degraded"}"#,
            &[],
        ),
        Err(ObservationBindingError::SessionNotTrusted(
            MachineSessionTrust::Revoked
        ))
    );

    let historical = HistoricallyQualifiedMachineSessionV1::from_verified_provider_result(
        &old_session,
        session_policy(),
        SCHEMA,
        "session-before-partition",
        PRINCIPAL,
        OLD_EVIDENCE,
        9,
        1_000,
        2_000,
        1_400,
        OLD_ADMISSION,
        "xenia-machine-authority-history-head-v1:blake3-256:revoked-at-1600",
        1,
        1_700,
    )
    .unwrap();

    let recovered_after_reconnect = bind_observation_to_historically_qualified_session(
        &old_session,
        &historical,
        session_policy(),
        &source_policy,
        PLATFORM,
        1_400_000,
        MaritimeObservationKind::HealthObservation,
        r#"{"severity":"healthy"}"#,
        &["mycelix-position:measurement:001".into()],
    )
    .unwrap();

    assert_eq!(recovered_after_reconnect, live_before_partition);
    assert_eq!(
        recovered_after_reconnect.evidence_binding(),
        stable_pre_partition_binding
    );
    assert_eq!(historical.session_evidence_binding(), OLD_EVIDENCE);
    assert_eq!(
        historical.provider_session_admission_binding(),
        OLD_ADMISSION
    );
    assert_eq!(historical.history_head_sequence(), 1);

    assert_eq!(
        bind_observation_to_trusted_session(
            &old_session,
            &context(1_700, 10, false),
            session_policy(),
            &source_policy,
            PLATFORM,
            1_650_000,
            MaritimeObservationKind::HealthObservation,
            r#"{"severity":"healthy"}"#,
            &[],
        ),
        Err(ObservationBindingError::SessionNotTrusted(
            MachineSessionTrust::EpochMismatch
        ))
    );

    let new_session = session("session-after-reconnect", 1_700, 2_700, 10, NEW_EVIDENCE);
    let live_after_readmission = bind_observation_to_trusted_session(
        &new_session,
        &context(1_800, 10, false),
        session_policy(),
        &source_policy,
        PLATFORM,
        1_750_000,
        MaritimeObservationKind::RecoveryEvent,
        r#"{"state":"readmitted"}"#,
        &[],
    )
    .unwrap();

    assert_ne!(
        live_after_readmission.evidence_binding(),
        stable_pre_partition_binding
    );
}

#[test]
fn historical_recovery_cannot_rebind_different_bytes_or_time() {
    let source_policy = source_policy();
    let old_session = session("session-before-partition", 1_000, 2_000, 9, OLD_EVIDENCE);
    let historical = HistoricallyQualifiedMachineSessionV1::from_verified_provider_result(
        &old_session,
        session_policy(),
        SCHEMA,
        "session-before-partition",
        PRINCIPAL,
        OLD_EVIDENCE,
        9,
        1_000,
        2_000,
        1_400,
        OLD_ADMISSION,
        "xenia-machine-authority-history-head-v1:blake3-256:revoked-at-1600",
        1,
        1_700,
    )
    .unwrap();

    assert_eq!(
        bind_observation_to_historically_qualified_session(
            &old_session,
            &historical,
            session_policy(),
            &source_policy,
            PLATFORM,
            1_401_000,
            MaritimeObservationKind::HealthObservation,
            r#"{"severity":"healthy"}"#,
            &[],
        ),
        Err(ObservationBindingError::HistoricalQualificationMismatch)
    );

    let changed_bytes = bind_observation_to_historically_qualified_session(
        &old_session,
        &historical,
        session_policy(),
        &source_policy,
        PLATFORM,
        1_400_000,
        MaritimeObservationKind::HealthObservation,
        r#"{"severity":"degraded"}"#,
        &[],
    )
    .unwrap();
    let original_bytes = bind_observation_to_historically_qualified_session(
        &old_session,
        &historical,
        session_policy(),
        &source_policy,
        PLATFORM,
        1_400_000,
        MaritimeObservationKind::HealthObservation,
        r#"{"severity":"healthy"}"#,
        &[],
    )
    .unwrap();
    assert_ne!(changed_bytes.evidence_binding(), original_bytes.evidence_binding());
}
