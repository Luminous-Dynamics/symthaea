// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::convert::Infallible;

use symthaea_wisdom::{
    ActionExecutionCoordinator, ArchiveAuthorityOperationalStartupEvidence,
    ArchiveOperationalRestorePolicy, EvidenceLedger, EvidenceSigner, EvidenceVerifier,
    LedgerRevision, OperationalReplayPolicy, OperationalStartupRequirements, RuntimeIntegration,
    RuntimeSourceIdentity, SingleArchiveVerifier, TrustRegistry, TrustRole, TrustedKey,
    TrustedKeyStatus, WisdomConfig, WisdomObservation, WisdomState, attest_authority_checkpoint,
    create_archive_segment, fingerprint_bytes, recover_authority_from_checkpoint,
    restore_operational_state_from_archive, validate_archive_operational_startup_with_authority,
};

#[derive(Debug)]
struct ToyKey;

impl EvidenceSigner for ToyKey {
    type Error = Infallible;

    fn algorithm(&self) -> &str {
        "toy"
    }

    fn key_id(&self) -> &str {
        "archive-startup"
    }

    fn sign(&self, message: &[u8]) -> Result<Vec<u8>, Self::Error> {
        Ok(fingerprint_bytes(message).to_le_bytes().to_vec())
    }
}

impl EvidenceVerifier for ToyKey {
    type Error = Infallible;

    fn algorithm(&self) -> &str {
        "toy"
    }

    fn key_id(&self) -> &str {
        "archive-startup"
    }

    fn verify(&self, message: &[u8], signature: &[u8]) -> Result<bool, Self::Error> {
        Ok(signature == fingerprint_bytes(message).to_le_bytes())
    }
}

fn trust_registry() -> TrustRegistry {
    let mut registry = TrustRegistry::new();
    registry
        .register_initial(TrustedKey {
            role: TrustRole::EvidenceAttestation,
            algorithm: "toy".into(),
            key_id: "archive-startup".into(),
            valid_from_millis: 0,
            valid_until_millis: None,
            status: TrustedKeyStatus::Active,
        })
        .unwrap();
    registry
}

fn observe(state: &mut WisdomState, timestamp: u64) {
    state.update_from_observation_with_evidence(
        WisdomObservation::legacy(0.3, 0.2, 0.8),
        timestamp,
        "archive-startup",
    );
}

#[test]
fn archive_replay_authority_recovery_and_startup_admission_form_one_closed_path() {
    let trust = trust_registry();
    let sources = Vec::<RuntimeSourceIdentity>::new();
    let integrations = Vec::<RuntimeIntegration>::new();
    let execution = ActionExecutionCoordinator::new();

    let mut state = WisdomState::new();
    *state.evidence_ledger_mut() = EvidenceLedger::new(4);
    for timestamp in 0..4 {
        observe(&mut state, timestamp);
    }
    let segment = create_archive_segment(0, None, &state, 4, 3, &ToyKey).unwrap();
    let checkpoint = attest_authority_checkpoint(
        &state,
        &execution,
        &integrations,
        &sources,
        &trust,
        &segment.checkpoint,
        LedgerRevision::from_ledger(state.evidence_ledger()),
        4,
        &ToyKey,
    )
    .unwrap();

    observe(&mut state, 4);
    observe(&mut state, 5);
    assert_eq!(state.evidence_ledger().evicted_records(), 2);

    let restored = restore_operational_state_from_archive(
        &[segment.clone()],
        state.evidence_ledger(),
        &SingleArchiveVerifier::new(&ToyKey),
        &trust,
        WisdomConfig::default(),
        OperationalReplayPolicy::default(),
        ArchiveOperationalRestorePolicy::default(),
    )
    .unwrap();
    let mut restored_state = restored.state;
    let authority = recover_authority_from_checkpoint(
        &mut restored_state,
        state.evidence_ledger(),
        &checkpoint,
        &segment.checkpoint,
        &ToyKey,
        &ToyKey,
        &trust,
        &sources,
        6,
    )
    .unwrap();

    let report = validate_archive_operational_startup_with_authority(
        &restored_state,
        ArchiveAuthorityOperationalStartupEvidence {
            durable_revision: LedgerRevision::from_ledger(state.evidence_ledger()),
            archive_restore: &restored.report,
            authority_recovery: &authority.report,
            trust_registry: &trust,
            expected_trust_registry_fingerprint: trust.fingerprint(),
            runtime_sources: &sources,
            retention_continuity: &authority.report.continuity,
            now_millis: 6,
        },
        OperationalStartupRequirements {
            minimum_runtime_sources: 0,
            ..OperationalStartupRequirements::default()
        },
    );

    assert!(report.is_ready(), "startup findings: {:?}", report.findings);
    let admission = report.admission_permit().unwrap();
    admission
        .validate(
            &restored_state,
            LedgerRevision::from_ledger(state.evidence_ledger()),
            &trust,
            &sources,
            6,
            0,
        )
        .unwrap();
}

#[test]
fn archive_signature_tampering_blocks_state_before_authority_recovery() {
    let trust = trust_registry();
    let mut state = WisdomState::new();
    *state.evidence_ledger_mut() = EvidenceLedger::new(4);
    for timestamp in 0..4 {
        observe(&mut state, timestamp);
    }
    let mut segment = create_archive_segment(0, None, &state, 4, 2, &ToyKey).unwrap();
    segment.signature[0] ^= 0x80;

    assert!(
        restore_operational_state_from_archive(
            &[segment],
            state.evidence_ledger(),
            &SingleArchiveVerifier::new(&ToyKey),
            &trust,
            WisdomConfig::default(),
            OperationalReplayPolicy::default(),
            ArchiveOperationalRestorePolicy::default(),
        )
        .is_err()
    );
}
