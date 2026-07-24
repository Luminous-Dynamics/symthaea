// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::collections::{BTreeMap, BTreeSet};
use symthaea_fabrication_kernel::*;

struct Provider {
    algorithm: SignatureAlgorithm,
    key_id: &'static str,
}

impl ThresholdApprovalSigner for Provider {
    fn algorithm(&self) -> SignatureAlgorithm {
        self.algorithm.clone()
    }
    fn key_id(&self) -> &str {
        self.key_id
    }
    fn sign_threshold_approval(&self, message: &[u8]) -> Result<Vec<u8>, String> {
        Ok(sha256(message).0.to_vec())
    }
}

impl ThresholdApprovalVerifier for Provider {
    fn verify_threshold_approval(
        &self,
        _algorithm: &SignatureAlgorithm,
        _key_id: &str,
        message: &[u8],
        signature: &[u8],
    ) -> Result<bool, String> {
        Ok(signature == sha256(message).0.as_slice())
    }
}

impl TransparencyCheckpointSigner for Provider {
    fn algorithm(&self) -> SignatureAlgorithm {
        self.algorithm.clone()
    }
    fn key_id(&self) -> &str {
        self.key_id
    }
    fn sign_transparency_checkpoint(&self, message: &[u8]) -> Result<Vec<u8>, String> {
        Ok(sha256(message).0.to_vec())
    }
}

impl TransparencyCheckpointVerifier for Provider {
    fn verify_transparency_checkpoint(
        &self,
        _algorithm: &SignatureAlgorithm,
        _key_id: &str,
        message: &[u8],
        signature: &[u8],
    ) -> Result<bool, String> {
        Ok(signature == sha256(message).0.as_slice())
    }
}

fn member(gateway_id: &str, failure_domain: &str) -> GatewayMember {
    GatewayMember {
        gateway_id: gateway_id.into(),
        voting_weight: 1,
        failure_domain: failure_domain.into(),
    }
}

fn trust() -> TrustSnapshot {
    TrustSnapshot::new(
        1,
        100,
        1_000,
        vec![
            KeyTrustRecord {
                algorithm: SignatureAlgorithm::Ed25519,
                key_id: "operator-a".into(),
                not_before_unix_s: 100,
                not_after_unix_s: None,
                status: KeyLifecycleStatus::Active,
                usages: BTreeSet::from([KeyUsage::GatewayMembership, KeyUsage::TransparencyLog]),
            },
            KeyTrustRecord {
                algorithm: SignatureAlgorithm::MlDsa65,
                key_id: "operator-b".into(),
                not_before_unix_s: 100,
                not_after_unix_s: None,
                status: KeyLifecycleStatus::Active,
                usages: BTreeSet::from([KeyUsage::GatewayMembership]),
            },
        ],
    )
    .unwrap()
}

#[test]
fn membership_rotation_and_transparency_checkpoint_are_verifiable() {
    let current = GatewayMembership::new(
        1,
        100,
        1_000,
        vec![
            member("gateway-a", "rack-a"),
            member("gateway-b", "rack-b"),
            member("gateway-c", "rack-c"),
        ],
    )
    .unwrap();
    let proposed = GatewayMembership::new(
        2,
        200,
        1_000,
        vec![
            member("gateway-a", "rack-a"),
            member("gateway-b", "rack-b"),
            member("gateway-d", "rack-d"),
        ],
    )
    .unwrap();
    let transition =
        build_membership_transition(&current, proposed, 200, "replace failed gateway").unwrap();
    let transition_digest = digest_membership_transition(&transition, &current).unwrap();
    let a = Provider {
        algorithm: SignatureAlgorithm::Ed25519,
        key_id: "operator-a",
    };
    let b = Provider {
        algorithm: SignatureAlgorithm::MlDsa65,
        key_id: "operator-b",
    };
    let approvals = vec![
        sign_threshold_approval(
            "gateway-membership-rotation",
            transition_digest,
            150,
            300,
            &a,
        )
        .unwrap(),
        sign_threshold_approval(
            "gateway-membership-rotation",
            transition_digest,
            150,
            300,
            &b,
        )
        .unwrap(),
    ];
    let ceremony = verify_threshold_ceremony(
        "gateway-membership-rotation",
        transition_digest,
        &approvals,
        &ThresholdCeremonyPolicy {
            key_usage: KeyUsage::GatewayMembership,
            ..ThresholdCeremonyPolicy::default()
        },
        &trust(),
        200,
        &a,
    )
    .unwrap();
    let authorized = authorize_membership_transition(
        &current,
        transition,
        &GatewayMembershipPolicy::default(),
        &ceremony,
    )
    .unwrap();
    assert_eq!(authorized.proposed_membership().epoch, 2);

    let mut log = TransparencyLog::default();
    let candidate = sha256(b"candidate");
    log.append(200, "release-candidate", candidate).unwrap();
    let proof = log.inclusion_proof(0).unwrap();
    verify_transparency_inclusion(&proof).unwrap();
    let signed_checkpoint = sign_transparency_checkpoint(&log, None, 200, 400, &a).unwrap();
    let verified_checkpoint =
        verify_transparency_checkpoint(&signed_checkpoint, &log, &trust(), 250, &a).unwrap();
    assert_eq!(
        verified_checkpoint.checkpoint().root_digest,
        proof.root_digest
    );
}

#[test]
fn artifact_inventory_rejects_unlisted_output() {
    let mut build = BTreeMap::new();
    build.insert(
        "bin/fabrication-gateway".into(),
        ("application/octet-stream".into(), b"binary".to_vec()),
    );
    let set = build_release_artifact_set(sha256(b"tree"), &build).unwrap();
    let mut supplied = BTreeMap::new();
    supplied.insert("bin/fabrication-gateway".into(), b"binary".to_vec());
    verify_release_artifact_set(&set, &supplied).unwrap();
    supplied.insert("debug/private-key".into(), b"forbidden".to_vec());
    assert!(verify_release_artifact_set(&set, &supplied).is_err());
}
