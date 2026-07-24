// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::collections::BTreeSet;

use symthaea_fabrication_kernel::{
    FabricationGatewayState, GatewayConsensusEvidence, GatewayConsensusPolicy,
    GatewayConsensusTracker, GatewayEndorsementSigner, GatewayEndorsementVerifier,
    GatewayRecoveryBundle, GatewayRecoveryCheckpoint, GatewayStateEnvelope, IncidentBundle,
    IncidentBundleSigner, IncidentBundleVerifier, IncidentKind, IncidentLedger, KeyLifecycleStatus,
    KeyTrustRecord, KeyUsage, MachineSessionTracker, MachineTelemetryTracker, OperatorCommand,
    OperatorCommandExpectation, OperatorCommandKind, OperatorCommandPolicy, OperatorCommandSigner,
    OperatorCommandTracker, OperatorCommandVerifier, ReleaseCandidateEvidence,
    ReleaseCandidateSigner, ReleaseCandidateVerifier, ReleaseCertificationPolicy, Sha256Digest,
    SignatureAlgorithm, SubmissionLedger, TrustSnapshot, attest_fabrication_manifest,
    digest_release_candidate, endorse_gateway_state, sha256, sign_incident_bundle,
    sign_operator_command, sign_release_candidate, verify_gateway_consensus,
    verify_incident_bundle, verify_operator_command, verify_release_candidate,
    verify_release_candidate_evidence,
};

#[derive(Clone)]
struct HashSigner {
    algorithm: SignatureAlgorithm,
    key_id: &'static str,
}

impl HashSigner {
    fn signature(message: &[u8]) -> Vec<u8> {
        sha256(message).0.to_vec()
    }
}

struct HashVerifier;

impl GatewayEndorsementSigner for HashSigner {
    fn algorithm(&self) -> SignatureAlgorithm {
        self.algorithm.clone()
    }
    fn key_id(&self) -> &str {
        self.key_id
    }
    fn sign_gateway_endorsement(&self, message: &[u8]) -> Result<Vec<u8>, String> {
        Ok(Self::signature(message))
    }
}

impl GatewayEndorsementVerifier for HashVerifier {
    fn verify_gateway_endorsement(
        &self,
        _algorithm: &SignatureAlgorithm,
        _key_id: &str,
        message: &[u8],
        signature: &[u8],
    ) -> Result<bool, String> {
        Ok(signature == sha256(message).0.as_slice())
    }
}

impl OperatorCommandSigner for HashSigner {
    fn algorithm(&self) -> SignatureAlgorithm {
        self.algorithm.clone()
    }
    fn key_id(&self) -> &str {
        self.key_id
    }
    fn sign_operator_command(&self, message: &[u8]) -> Result<Vec<u8>, String> {
        Ok(Self::signature(message))
    }
}

impl OperatorCommandVerifier for HashVerifier {
    fn verify_operator_command(
        &self,
        _algorithm: &SignatureAlgorithm,
        _key_id: &str,
        message: &[u8],
        signature: &[u8],
    ) -> Result<bool, String> {
        Ok(signature == sha256(message).0.as_slice())
    }
}

impl IncidentBundleSigner for HashSigner {
    fn algorithm(&self) -> SignatureAlgorithm {
        self.algorithm.clone()
    }
    fn key_id(&self) -> &str {
        self.key_id
    }
    fn sign_incident_bundle(&self, message: &[u8]) -> Result<Vec<u8>, String> {
        Ok(Self::signature(message))
    }
}

impl IncidentBundleVerifier for HashVerifier {
    fn verify_incident_bundle(
        &self,
        _algorithm: &SignatureAlgorithm,
        _key_id: &str,
        message: &[u8],
        signature: &[u8],
    ) -> Result<bool, String> {
        Ok(signature == sha256(message).0.as_slice())
    }
}

impl ReleaseCandidateSigner for HashSigner {
    fn algorithm(&self) -> SignatureAlgorithm {
        self.algorithm.clone()
    }
    fn key_id(&self) -> &str {
        self.key_id
    }
    fn sign_release_candidate(&self, message: &[u8]) -> Result<Vec<u8>, String> {
        Ok(Self::signature(message))
    }
}

impl ReleaseCandidateVerifier for HashVerifier {
    fn verify_release_candidate(
        &self,
        _algorithm: &SignatureAlgorithm,
        _key_id: &str,
        message: &[u8],
        signature: &[u8],
    ) -> Result<bool, String> {
        Ok(signature == sha256(message).0.as_slice())
    }
}

fn key(
    algorithm: SignatureAlgorithm,
    key_id: &str,
    usages: impl IntoIterator<Item = KeyUsage>,
) -> KeyTrustRecord {
    KeyTrustRecord {
        algorithm,
        key_id: key_id.into(),
        not_before_unix_s: 100,
        not_after_unix_s: None,
        status: KeyLifecycleStatus::Active,
        usages: usages.into_iter().collect(),
    }
}

fn trust_snapshot() -> TrustSnapshot {
    TrustSnapshot::new(
        1,
        100,
        1_000,
        vec![
            key(
                SignatureAlgorithm::Ed25519,
                "gateway-a-key",
                [KeyUsage::GatewayConsensus],
            ),
            key(
                SignatureAlgorithm::MlDsa65,
                "gateway-b-key",
                [KeyUsage::GatewayConsensus],
            ),
            key(
                SignatureAlgorithm::Ed25519,
                "operator-key",
                [KeyUsage::OperatorCommand],
            ),
            key(
                SignatureAlgorithm::Ed25519,
                "incident-key",
                [KeyUsage::IncidentEvidence],
            ),
            key(
                SignatureAlgorithm::Ed25519,
                "release-classical",
                [KeyUsage::ReleaseCertification],
            ),
            key(
                SignatureAlgorithm::MlDsa65,
                "release-pq",
                [KeyUsage::ReleaseCertification],
            ),
        ],
    )
    .unwrap()
}

fn gateway_state() -> FabricationGatewayState {
    FabricationGatewayState::genesis(
        500_000,
        trust_snapshot(),
        Default::default(),
        MachineSessionTracker::default(),
        MachineTelemetryTracker::default(),
        SubmissionLedger::default(),
        OperatorCommandTracker::default(),
        GatewayConsensusTracker::default(),
        IncidentLedger::default(),
    )
    .unwrap()
}

#[test]
fn federated_release_requires_quorum_recovery_and_closed_incidents() {
    let state = gateway_state();
    let state_envelope = GatewayStateEnvelope::seal(state.clone()).unwrap();
    let gateway_a = HashSigner {
        algorithm: SignatureAlgorithm::Ed25519,
        key_id: "gateway-a-key",
    };
    let gateway_b = HashSigner {
        algorithm: SignatureAlgorithm::MlDsa65,
        key_id: "gateway-b-key",
    };
    let endorsements = vec![
        endorse_gateway_state(&state, "gateway-a", 500_000, 520_000, &gateway_a).unwrap(),
        endorse_gateway_state(&state, "gateway-b", 500_000, 520_000, &gateway_b).unwrap(),
    ];
    let consensus = verify_gateway_consensus(
        &state,
        &endorsements,
        &GatewayConsensusPolicy {
            minimum_distinct_gateways: 2,
            maximum_endorsements: 8,
            require_algorithm_diversity: true,
            required_gateway_ids: BTreeSet::from([
                "gateway-a".to_string(),
                "gateway-b".to_string(),
            ]),
            allowed_gateway_ids: None,
        },
        &state.trust_snapshot,
        501_000,
        &HashVerifier,
    )
    .unwrap();
    let mut consensus_tracker = GatewayConsensusTracker::default();
    consensus_tracker.accept(&state, &consensus).unwrap();

    let operator = HashSigner {
        algorithm: SignatureAlgorithm::Ed25519,
        key_id: "operator-key",
    };
    let operator_command = OperatorCommand::new(
        sha256(b"manifest"),
        "machine-1",
        sha256(b"session"),
        "job-1",
        1,
        501_000,
        510_000,
        OperatorCommandKind::Pause,
        "inspect layer adhesion",
    )
    .unwrap();
    let signed_command = sign_operator_command(operator_command, &[&operator]).unwrap();
    let verified_command = verify_operator_command(
        signed_command,
        &OperatorCommandPolicy::default(),
        OperatorCommandExpectation {
            manifest_digest: sha256(b"manifest"),
            machine_id: "machine-1",
            session_digest: sha256(b"session"),
            printer_job_id: "job-1",
            now_unix_ms: 502_000,
            trust_snapshot: &state.trust_snapshot,
        },
        &HashVerifier,
    )
    .unwrap();
    let mut command_tracker = OperatorCommandTracker::default();
    command_tracker.apply(&verified_command).unwrap();

    let recovery = GatewayRecoveryBundle::build(
        "primary-site",
        530_000,
        vec![GatewayRecoveryCheckpoint {
            backup_id: "generation-1".into(),
            captured_at_unix_ms: 510_000,
            envelope: state_envelope.clone(),
            consensus: GatewayConsensusEvidence::from_verified(&consensus),
        }],
    )
    .unwrap();

    let incident_signer = HashSigner {
        algorithm: SignatureAlgorithm::Ed25519,
        key_id: "incident-key",
    };
    let incident = IncidentBundle::capture(
        "incident-layer-adhesion",
        503_000,
        IncidentKind::OperatorPause,
        "operator paused the job for layer inspection",
        Some(sha256(b"manifest")),
        Some("machine-1".into()),
        Some(sha256(b"session")),
        Some("job-1".into()),
        verified_command.command_digest(),
        state_envelope,
    )
    .unwrap();
    let signed_incident = sign_incident_bundle(incident, &[&incident_signer]).unwrap();
    let verified_incident = verify_incident_bundle(
        signed_incident,
        &state.trust_snapshot,
        503,
        1,
        &HashVerifier,
    )
    .unwrap();
    let incident_digest = verified_incident.bundle_digest();
    let mut incident_ledger = IncidentLedger::default();
    incident_ledger.register(503, &verified_incident).unwrap();

    let source_tree_digest = sha256(b"source-tree");
    let manifest_digest = sha256(b"manifest");
    let governed_replay_digest = sha256(b"governed-replay");
    let gateway_replay_digest = sha256(b"gateway-replay");
    let blocked = ReleaseCandidateEvidence::build_from_incident_ledger(
        "v0.14.0-rc.1",
        "0.14.0",
        504,
        900,
        source_tree_digest,
        manifest_digest,
        governed_replay_digest,
        gateway_replay_digest,
        &state,
        &consensus,
        &recovery,
        &incident_ledger,
    )
    .unwrap();
    assert_eq!(blocked.unresolved_incident_digests, vec![incident_digest]);

    incident_ledger
        .resolve(505, incident_digest, sha256(b"inspection-closed"))
        .unwrap();
    let candidate = ReleaseCandidateEvidence::build_from_incident_ledger(
        "v0.14.0-rc.2",
        "0.14.0",
        506,
        900,
        source_tree_digest,
        manifest_digest,
        governed_replay_digest,
        gateway_replay_digest,
        &state,
        &consensus,
        &recovery,
        &incident_ledger,
    )
    .unwrap();
    assert!(candidate.unresolved_incident_digests.is_empty());

    let release_classical = HashSigner {
        algorithm: SignatureAlgorithm::Ed25519,
        key_id: "release-classical",
    };
    let release_pq = HashSigner {
        algorithm: SignatureAlgorithm::MlDsa65,
        key_id: "release-pq",
    };
    let signed_candidate =
        sign_release_candidate(candidate, &[&release_classical, &release_pq]).unwrap();
    let certified = verify_release_candidate(
        signed_candidate,
        &ReleaseCertificationPolicy::default(),
        &state.trust_snapshot,
        507,
        &HashVerifier,
    )
    .unwrap();
    assert_eq!(certified.valid_signers().len(), 2);
    assert!(
        verify_release_candidate_evidence(
            certified.candidate(),
            source_tree_digest,
            manifest_digest,
            governed_replay_digest,
            gateway_replay_digest,
            &state,
            &consensus,
            &recovery,
        )
        .unwrap()
        .exact()
    );
    assert_eq!(
        digest_release_candidate(certified.candidate()).unwrap(),
        certified.candidate_digest()
    );
}
