// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_maritime_core::{
    AuthenticatedMachineSession, HistoricallyQualifiedMachineSessionV1, MachineSessionPolicy,
};

#[test]
fn historical_handoff_records_provider_proof_positions_without_live_authority() {
    const SCHEMA: &str = "xenia-verified-machine-session-evidence-v1";
    const PRINCIPAL: &str = "xenia-signing-identity-v1:blake3-256:abc";
    const SESSION_EVIDENCE: &str = "xenia-handshake-transcript-v1:blake3-256:def";
    const ADMISSION: &str = "xenia-machine-session-admission-v1:blake3-256:abc";
    let session = AuthenticatedMachineSession::from_verified_provider(
        SCHEMA, "session-1", PRINCIPAL, 1_000, 2_000, 9, SESSION_EVIDENCE,
    )
    .unwrap();
    let policy = MachineSessionPolicy {
        accepted_schemas: &[SCHEMA],
        max_validity_ms: 1_000,
    };
    let historical = HistoricallyQualifiedMachineSessionV1::from_verified_provider_result(
        &session,
        policy,
        SCHEMA,
        "session-1",
        PRINCIPAL,
        SESSION_EVIDENCE,
        9,
        1_000,
        2_000,
        1_400,
        ADMISSION,
        "xenia-machine-authority-history-head-v1:blake3-256:abc",
        11,
        1_600,
    )
    .unwrap();

    assert_eq!(historical.provider_session_admission_binding(), ADMISSION);
    assert_eq!(historical.history_head_sequence(), 11);
    assert_eq!(historical.observation_at_ms(), 1_400);
}
