// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_maritime_core::{
    AuthenticatedMachineSession, HistoricallyQualifiedMachineSessionV1, MachineSessionContext,
    MachineSessionPolicy, MaritimeObservationKind, ObservationSourceGrantV1,
    ObservationSourcePolicyV1, bind_observation_to_historically_qualified_session,
    bind_observation_to_trusted_session,
};

const SCHEMA: &str = "xenia-verified-machine-session-evidence-v1";
const PRINCIPAL: &str = "xenia-signing-identity-v1:blake3-256:abc";
const SESSION_EVIDENCE: &str = "xenia-handshake-transcript-v1:blake3-256:def";

fn session() -> AuthenticatedMachineSession {
    AuthenticatedMachineSession::from_verified_provider(
        SCHEMA,
        "session-1",
        PRINCIPAL,
        1_000,
        2_000,
        9,
        SESSION_EVIDENCE,
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
        SCHEMA, PRINCIPAL, "auv-01",
    )
    .unwrap()])
    .unwrap()
}

#[test]
fn delayed_historical_reconciliation_preserves_the_live_observation_id() {
    let session = session();
    let live_context = MachineSessionContext::from_authority_provider(
        SCHEMA, PRINCIPAL, 1_500, 9, true, false,
    )
    .unwrap();
    let live = bind_observation_to_trusted_session(
        &session,
        &live_context,
        session_policy(),
        &source_policy(),
        "auv-01",
        1_400_000,
        MaritimeObservationKind::HealthObservation,
        r#"{"severity":"healthy"}"#,
        &["mycelix-position:measurement:001".into()],
    )
    .unwrap();

    let historical = HistoricallyQualifiedMachineSessionV1::from_verified_provider_result(
        &session,
        session_policy(),
        SCHEMA,
        "session-1",
        PRINCIPAL,
        SESSION_EVIDENCE,
        9,
        1_000,
        2_000,
        1_400,
        "xenia-machine-authority-history-head-v1:blake3-256:abc",
        7,
        1_600,
    )
    .unwrap();
    let delayed = bind_observation_to_historically_qualified_session(
        &session,
        &historical,
        session_policy(),
        &source_policy(),
        "auv-01",
        1_400_000,
        MaritimeObservationKind::HealthObservation,
        r#"{"severity":"healthy"}"#,
        &["mycelix-position:measurement:001".into()],
    )
    .unwrap();

    assert_eq!(live, delayed);
    assert_eq!(historical.session_evidence_binding(), SESSION_EVIDENCE);
    assert_eq!(historical.history_head_sequence(), 7);
}
