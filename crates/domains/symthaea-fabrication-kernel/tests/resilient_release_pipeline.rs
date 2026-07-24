// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::collections::{BTreeMap, BTreeSet};
use symthaea_fabrication_kernel::*;

#[derive(Clone)]
struct HashProvider {
    algorithm: SignatureAlgorithm,
    key_id: &'static str,
}

fn signature(message: &[u8]) -> Vec<u8> {
    sha256(message).0.to_vec()
}

impl GatewayEndorsementSigner for HashProvider {
    fn algorithm(&self) -> SignatureAlgorithm {
        self.algorithm.clone()
    }
    fn key_id(&self) -> &str {
        self.key_id
    }
    fn sign_gateway_endorsement(&self, message: &[u8]) -> Result<Vec<u8>, String> {
        Ok(signature(message))
    }
}
impl GatewayEndorsementVerifier for HashProvider {
    fn verify_gateway_endorsement(
        &self,
        _algorithm: &SignatureAlgorithm,
        _key_id: &str,
        message: &[u8],
        signature_bytes: &[u8],
    ) -> Result<bool, String> {
        Ok(signature_bytes == signature(message).as_slice())
    }
}
impl TransparencyCheckpointSigner for HashProvider {
    fn algorithm(&self) -> SignatureAlgorithm {
        self.algorithm.clone()
    }
    fn key_id(&self) -> &str {
        self.key_id
    }
    fn sign_transparency_checkpoint(&self, message: &[u8]) -> Result<Vec<u8>, String> {
        Ok(signature(message))
    }
}
impl TransparencyCheckpointVerifier for HashProvider {
    fn verify_transparency_checkpoint(
        &self,
        _algorithm: &SignatureAlgorithm,
        _key_id: &str,
        message: &[u8],
        signature_bytes: &[u8],
    ) -> Result<bool, String> {
        Ok(signature_bytes == signature(message).as_slice())
    }
}
impl TransparencyWitnessSigner for HashProvider {
    fn algorithm(&self) -> SignatureAlgorithm {
        self.algorithm.clone()
    }
    fn key_id(&self) -> &str {
        self.key_id
    }
    fn sign_transparency_witness(&self, message: &[u8]) -> Result<Vec<u8>, String> {
        Ok(signature(message))
    }
}
impl TransparencyWitnessVerifier for HashProvider {
    fn verify_transparency_witness(
        &self,
        _algorithm: &SignatureAlgorithm,
        _key_id: &str,
        message: &[u8],
        signature_bytes: &[u8],
    ) -> Result<bool, String> {
        Ok(signature_bytes == signature(message).as_slice())
    }
}
impl ArtifactProvenanceSigner for HashProvider {
    fn algorithm(&self) -> SignatureAlgorithm {
        self.algorithm.clone()
    }
    fn key_id(&self) -> &str {
        self.key_id
    }
    fn sign_artifact_provenance(&self, message: &[u8]) -> Result<Vec<u8>, String> {
        Ok(signature(message))
    }
}
impl ArtifactProvenanceVerifier for HashProvider {
    fn verify_artifact_provenance(
        &self,
        _algorithm: &SignatureAlgorithm,
        _key_id: &str,
        message: &[u8],
        signature_bytes: &[u8],
    ) -> Result<bool, String> {
        Ok(signature_bytes == signature(message).as_slice())
    }
}

fn key(algorithm: SignatureAlgorithm, key_id: &str, usage: KeyUsage) -> KeyTrustRecord {
    KeyTrustRecord {
        algorithm,
        key_id: key_id.into(),
        not_before_unix_s: 100,
        not_after_unix_s: None,
        status: KeyLifecycleStatus::Active,
        usages: BTreeSet::from([usage]),
    }
}

fn trust() -> TrustSnapshot {
    TrustSnapshot::new(
        1,
        100,
        1_000,
        vec![
            key(
                SignatureAlgorithm::Ed25519,
                "gateway-a-key",
                KeyUsage::GatewayConsensus,
            ),
            key(
                SignatureAlgorithm::MlDsa65,
                "gateway-b-key",
                KeyUsage::GatewayConsensus,
            ),
            key(
                SignatureAlgorithm::Ed25519,
                "log-key",
                KeyUsage::TransparencyLog,
            ),
            key(
                SignatureAlgorithm::Ed25519,
                "witness-a",
                KeyUsage::TransparencyWitness,
            ),
            key(
                SignatureAlgorithm::MlDsa65,
                "witness-b",
                KeyUsage::TransparencyWitness,
            ),
            key(
                SignatureAlgorithm::Ed25519,
                "builder-a",
                KeyUsage::ArtifactProvenance,
            ),
            key(
                SignatureAlgorithm::MlDsa65,
                "builder-b",
                KeyUsage::ArtifactProvenance,
            ),
        ],
    )
    .unwrap()
}

#[test]
fn regional_consensus_witnesses_and_provenance_form_independent_evidence() {
    let trust = trust();
    let state = FabricationGatewayState::genesis(
        500_000,
        trust.clone(),
        Default::default(),
        MachineSessionTracker::default(),
        MachineTelemetryTracker::default(),
        SubmissionLedger::default(),
        OperatorCommandTracker::default(),
        GatewayConsensusTracker::default(),
        IncidentLedger::default(),
    )
    .unwrap();
    let gateway_a = HashProvider {
        algorithm: SignatureAlgorithm::Ed25519,
        key_id: "gateway-a-key",
    };
    let gateway_b = HashProvider {
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
            required_gateway_ids: BTreeSet::new(),
            allowed_gateway_ids: None,
        },
        &trust,
        501_000,
        &gateway_a,
    )
    .unwrap();
    let membership = GatewayMembership::new(
        1,
        100,
        1_000,
        vec![
            GatewayMember {
                gateway_id: "gateway-a".into(),
                voting_weight: 1,
                failure_domain: "africa-south".into(),
            },
            GatewayMember {
                gateway_id: "gateway-b".into(),
                voting_weight: 1,
                failure_domain: "europe-west".into(),
            },
        ],
    )
    .unwrap();
    let regional = build_regional_quorum_evidence(
        &consensus,
        &membership,
        501,
        &RegionalQuorumPolicy {
            minimum_distinct_regions: 2,
            minimum_represented_weight_basis_points: 10_000,
            maximum_single_region_weight_basis_points: 5_000,
            required_regions: BTreeSet::new(),
        },
    )
    .unwrap();
    assert_eq!(regional.represented_regions.len(), 2);

    let log_provider = HashProvider {
        algorithm: SignatureAlgorithm::Ed25519,
        key_id: "log-key",
    };
    let mut log = TransparencyLog::default();
    log.append(510, "release-candidate", sha256(b"candidate"))
        .unwrap();
    let signed_checkpoint =
        sign_transparency_checkpoint(&log, None, 510, 700, &log_provider).unwrap();
    let checkpoint =
        verify_transparency_checkpoint(&signed_checkpoint, &log, &trust, 520, &log_provider)
            .unwrap();
    let witness_a = HashProvider {
        algorithm: SignatureAlgorithm::Ed25519,
        key_id: "witness-a",
    };
    let witness_b = HashProvider {
        algorithm: SignatureAlgorithm::MlDsa65,
        key_id: "witness-b",
    };
    let signed_witnesses = vec![
        sign_transparency_witness(&checkpoint, "org-a", "africa-south", 515, &witness_a).unwrap(),
        sign_transparency_witness(&checkpoint, "org-b", "europe-west", 516, &witness_b).unwrap(),
    ];
    let witness_quorum = verify_transparency_witness_quorum(
        &checkpoint,
        &signed_witnesses,
        &TransparencyWitnessPolicy::default(),
        &trust,
        520,
        &witness_a,
    )
    .unwrap();
    assert_eq!(witness_quorum.witnesses().len(), 2);

    let mut files = BTreeMap::new();
    files.insert(
        "bin/gateway".into(),
        ("application/octet-stream".into(), b"same-build".to_vec()),
    );
    let artifacts = build_release_artifact_set(sha256(b"tree"), &files).unwrap();
    let inputs = vec![ProvenanceInput {
        name: "Cargo.lock".into(),
        digest: sha256(b"lock"),
    }];
    let statement_a = build_artifact_provenance_statement(
        &artifacts,
        "builder-a",
        "africa-south",
        sha256(b"env-a"),
        sha256(b"lock"),
        inputs.clone(),
        2,
        515,
    )
    .unwrap();
    let statement_b = build_artifact_provenance_statement(
        &artifacts,
        "builder-b",
        "europe-west",
        sha256(b"env-b"),
        sha256(b"lock"),
        inputs,
        2,
        516,
    )
    .unwrap();
    let builder_a = HashProvider {
        algorithm: SignatureAlgorithm::Ed25519,
        key_id: "builder-a",
    };
    let builder_b = HashProvider {
        algorithm: SignatureAlgorithm::MlDsa65,
        key_id: "builder-b",
    };
    let signed_provenance = vec![
        sign_artifact_provenance(statement_a, &builder_a).unwrap(),
        sign_artifact_provenance(statement_b, &builder_b).unwrap(),
    ];
    let provenance = verify_artifact_provenance(
        &artifacts,
        &signed_provenance,
        &ArtifactProvenancePolicy::default(),
        &trust,
        520,
        &builder_a,
    )
    .unwrap();
    assert_eq!(provenance.builders().len(), 2);
}

#[test]
fn resilience_state_successor_cannot_drop_lineage() {
    let genesis = ReleaseResilienceState::genesis();
    let successor = genesis.successor().unwrap();
    verify_release_resilience_successor(&genesis, &successor).unwrap();

    let mut invalid = successor.clone();
    invalid.previous_state_digest = Some(sha256(b"wrong"));
    assert!(verify_release_resilience_successor(&genesis, &invalid).is_err());
}
