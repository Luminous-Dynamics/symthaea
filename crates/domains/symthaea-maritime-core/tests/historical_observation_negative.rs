// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_maritime_core::{
    AuthenticatedMachineSession, HistoricalSessionHandoffError,
    HistoricallyQualifiedMachineSessionV1, MachineSessionPolicy,
};

#[test]
fn provider_historical_result_cannot_substitute_another_session_proof() {
    const SCHEMA: &str = "xenia-verified-machine-session-evidence-v1";
    const PRINCIPAL: &str = "xenia-signing-identity-v1:blake3-256:abc";
    let session = AuthenticatedMachineSession::from_verified_provider(
        SCHEMA,
        "session-1",
        PRINCIPAL,
        1_000,
        2_000,
        9,
        "xenia-handshake-transcript-v1:blake3-256:expected",
    )
    .unwrap();
    let policy = MachineSessionPolicy {
        accepted_schemas: &[SCHEMA],
        max_validity_ms: 1_000,
    };

    assert_eq!(
        HistoricallyQualifiedMachineSessionV1::from_verified_provider_result(
            &session,
            policy,
            SCHEMA,
            "session-1",
            PRINCIPAL,
            "xenia-handshake-transcript-v1:blake3-256:other",
            9,
            1_000,
            2_000,
            1_400,
            "xenia-machine-authority-history-head-v1:blake3-256:abc",
            7,
            1_600,
        ),
        Err(HistoricalSessionHandoffError::SessionEvidenceMismatch)
    );
}
