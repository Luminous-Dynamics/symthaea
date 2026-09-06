// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use ed25519_dalek::{Signer, SigningKey};
use symthaea_authority::Digest32;
use symthaea_authority_time::{
    AUTHORITY_TIME_SCHEMA_VERSION, AuthorityTimeStatementV1, PendingAuthorityTimeChallenge,
    TimeAuthorityId, TrustedTimeAuthorityV1, TrustedTimePolicyV1, VerifiedAuthorityTime,
    verify_authority_time_v1,
};
use symthaea_qualification_witness_frontier::{
    WitnessFrontierPublicationDispositionV1, WitnessFrontierRecoveryRelationV1,
};
use symthaea_qualification_witness_frontier_sqlite::SqliteWitnessFrontierPublicationGuard;
use symthaea_qualification_witness_sequence::{
    SqliteWitnessSequenceStore, WitnessSequenceAttemptBindingV1, WitnessSequenceFrontierStatementV1,
};
use symthaea_xenia_authority::{
    ED25519_SIGNATURE_ALGORITHM, PendingXeniaWitnessCurrentnessChallengeV1,
    XENIA_WITNESS_FRONTIER_ANCHOR_SCHEMA_VERSION, XeniaSignatureEnvelopeV1,
    XeniaSignedWitnessFrontierAnchorV1, XeniaSignedWitnessFrontierObservationV1,
    XeniaWitnessCurrentnessScopeV1, XeniaWitnessFrontierAnchorSummaryV1,
    XeniaWitnessFrontierAnchorTargetV1, XeniaWitnessFrontierFreshnessPolicyV1,
    derive_xenia_witness_frontier_source_id,
};
use symthaea_xenia_guarded_witness_frontier::classify_guarded_xenia_witness_frontier_v1;
use symthaea_xenia_witness_frontier_adapter::adapt_verified_xenia_witness_frontier_v1;

const WITNESS: [u8; 16] = [0x51; 16];
const ANCHOR_POLICY: [u8; 32] = [0x33; 32];
static DB_COUNTER: AtomicU64 = AtomicU64::new(1);

fn xenia_key() -> SigningKey {
    SigningKey::from_bytes(&[7; 32])
}

fn envelope(signature: [u8; 64]) -> XeniaSignatureEnvelopeV1 {
    XeniaSignatureEnvelopeV1 {
        algorithm: ED25519_SIGNATURE_ALGORITHM.to_string(),
        signature: signature.to_vec(),
    }
}

fn binding(attempt: u8) -> WitnessSequenceAttemptBindingV1 {
    WitnessSequenceAttemptBindingV1 {
        attempt_id: [attempt; 16],
        witness_id: WITNESS,
        witness_epoch: 7,
        archive_sha256: Digest32([0x11; 32]),
        git_head: [0x22; 20],
        git_tree: [0x33; 20],
        verifier_digest: Digest32([0x44; 32]),
        witness_policy_digest: Digest32([0x55; 32]),
    }
}

fn db_path(label: &str) -> PathBuf {
    let serial = DB_COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "symthaea-xenia-guarded-{label}-{}-{serial}.sqlite3",
        std::process::id()
    ));
    cleanup(&path);
    path
}

fn signed_anchor(
    statement: WitnessSequenceFrontierStatementV1,
) -> XeniaSignedWitnessFrontierAnchorV1 {
    let key = xenia_key();
    let public_key = key.verifying_key().to_bytes();
    let source_id = derive_xenia_witness_frontier_source_id(public_key, ANCHOR_POLICY).unwrap();
    let mut target = XeniaWitnessFrontierAnchorTargetV1 {
        schema_version: XENIA_WITNESS_FRONTIER_ANCHOR_SCHEMA_VERSION,
        operation_id: [0; 32],
        source_id,
        source_epoch: 3,
        anchor_policy_digest: ANCHOR_POLICY,
        witness_id: statement.witness_id(),
        high_watermark: statement.high_watermark(),
        reservation_head: statement.reservation_head().0,
        frontier_statement_digest: statement.digest().0,
    };
    target.operation_id = target.recompute_operation_id();

    let mut anchor = XeniaSignedWitnessFrontierAnchorV1 {
        schema_version: XENIA_WITNESS_FRONTIER_ANCHOR_SCHEMA_VERSION,
        target,
        anchor_sequence: 1,
        previous_anchor_fingerprint: [0; 32],
        ledger_entry_count: 12,
        ledger_head_hash: [0x77; 32],
        ledger_public_key: public_key,
        issued_at_unix_s: 1_000,
        signature: envelope([0; 64]),
    };
    anchor.signature = envelope(key.sign(&anchor.canonical_message().unwrap()).to_bytes());
    anchor
}

fn signed_observation(
    anchor: &XeniaSignedWitnessFrontierAnchorV1,
    challenge: [u8; 32],
) -> XeniaSignedWitnessFrontierObservationV1 {
    let key = xenia_key();
    let mut observation = XeniaSignedWitnessFrontierObservationV1 {
        schema_version: XENIA_WITNESS_FRONTIER_ANCHOR_SCHEMA_VERSION,
        source_id: anchor.target.source_id,
        source_epoch: anchor.target.source_epoch,
        anchor_policy_digest: anchor.target.anchor_policy_digest,
        witness_id: anchor.target.witness_id,
        challenge,
        observed_at_unix_s: 1_010,
        current: Some(XeniaWitnessFrontierAnchorSummaryV1 {
            anchor_sequence: anchor.anchor_sequence,
            anchor_fingerprint: anchor.fingerprint().unwrap(),
            operation_id: anchor.target.operation_id,
            high_watermark: anchor.target.high_watermark,
            reservation_head: anchor.target.reservation_head,
            frontier_statement_digest: anchor.target.frontier_statement_digest,
        }),
        ledger_entry_count: 13,
        ledger_head_hash: [0x88; 32],
        ledger_public_key: key.verifying_key().to_bytes(),
        signature: envelope([0; 64]),
    };
    observation.signature =
        envelope(key.sign(&observation.canonical_message().unwrap()).to_bytes());
    observation
}

fn time_policy() -> (TrustedTimePolicyV1, SigningKey, SigningKey) {
    let key_a = SigningKey::from_bytes(&[31; 32]);
    let key_b = SigningKey::from_bytes(&[32; 32]);
    (
        TrustedTimePolicyV1 {
            schema_version: AUTHORITY_TIME_SCHEMA_VERSION,
            policy_id: [33; 16],
            authorities: vec![
                TrustedTimeAuthorityV1 {
                    authority_id: TimeAuthorityId([1; 16]),
                    verifying_key: key_a.verifying_key().to_bytes(),
                    organization_binding: [41; 32],
                    service_binding: [51; 32],
                },
                TrustedTimeAuthorityV1 {
                    authority_id: TimeAuthorityId([2; 16]),
                    verifying_key: key_b.verifying_key().to_bytes(),
                    organization_binding: [42; 32],
                    service_binding: [52; 32],
                },
            ],
            threshold: 2,
            minimum_organizations: 2,
            maximum_uncertainty_s: 1,
            maximum_challenge_age_ns: 5_000_000_000,
            maximum_post_verification_age_ns: 5_000_000_000,
        },
        key_a,
        key_b,
    )
}

fn verified_time(
    policy: &TrustedTimePolicyV1,
    key_a: &SigningKey,
    key_b: &SigningKey,
    subject: [u8; 32],
) -> VerifiedAuthorityTime {
    let pending = PendingAuthorityTimeChallenge::new(policy, subject).unwrap();
    let challenge = pending.wire();
    let sign = |authority_id: TimeAuthorityId, key: &SigningKey| {
        let mut statement = AuthorityTimeStatementV1 {
            schema_version: AUTHORITY_TIME_SCHEMA_VERSION,
            authority_id,
            policy_digest: challenge.policy_digest,
            subject_digest: challenge.subject_digest,
            challenge_nonce: challenge.nonce,
            witnessed_unix_s: 1_010,
            uncertainty_s: 1,
            signature: Vec::new(),
        };
        statement.signature = key
            .sign(&statement.canonical_message().unwrap())
            .to_bytes()
            .to_vec();
        statement
    };
    verify_authority_time_v1(
        policy,
        pending,
        &[
            sign(TimeAuthorityId([1; 16]), key_a),
            sign(TimeAuthorityId([2; 16]), key_b),
        ],
    )
    .unwrap()
}

fn verified_xenia_for_statement(
    statement: WitnessSequenceFrontierStatementV1,
) -> symthaea_xenia_witness_frontier_adapter::XeniaExternalWitnessFrontierV1 {
    let anchor = signed_anchor(statement);
    let pending = PendingXeniaWitnessCurrentnessChallengeV1::generate(
        XeniaWitnessCurrentnessScopeV1 {
            trusted_ledger_public_key: xenia_key().verifying_key().to_bytes(),
            source_epoch: 3,
            anchor_policy_digest: ANCHOR_POLICY,
            witness_id: WITNESS,
        },
    )
    .unwrap();
    let observation = signed_observation(&anchor, pending.challenge());

    let (time_policy, key_a, key_b) = time_policy();
    let freshness = XeniaWitnessFrontierFreshnessPolicyV1::strict(
        time_policy.digest().unwrap(),
        30,
        2,
    );
    let subject = pending
        .authority_time_subject_digest(&anchor, freshness)
        .unwrap();
    let time = verified_time(&time_policy, &key_a, &key_b, subject);
    let verified = pending
        .verify(&anchor, &observation, &time, freshness)
        .unwrap();
    adapt_verified_xenia_witness_frontier_v1(verified).unwrap()
}

#[test]
fn exact_current_xenia_anchor_yields_only_provenance_retaining_publication_permit() {
    let path = db_path("publish");
    let store = SqliteWitnessSequenceStore::open(&path).unwrap();
    store.reserve_attempt(binding(1)).unwrap();
    let statement = store.frontier_statement(WITNESS).unwrap().unwrap();
    let xenia = verified_xenia_for_statement(statement);

    let guard = SqliteWitnessFrontierPublicationGuard::acquire(&store, WITNESS).unwrap();
    let decision = classify_guarded_xenia_witness_frontier_v1(&guard, &xenia).unwrap();
    assert_eq!(
        decision.publication_disposition(),
        WitnessFrontierPublicationDispositionV1::PublishAllowed
    );
    assert!(matches!(
        decision.relation(),
        WitnessFrontierRecoveryRelationV1::AnchoredCurrent {
            high_watermark: 1,
            ..
        }
    ));
    assert!(decision.anchor_permit().is_none());
    {
        let permit = decision.publication_permit().unwrap();
        assert_eq!(permit.witness_id(), WITNESS);
        assert_eq!(permit.frontier().high_watermark, 1);
        assert_eq!(permit.xenia_evidence().xenia().high_watermark(), 1);
        assert_eq!(
            permit.xenia_evidence().xenia().operation_id(),
            xenia.xenia().operation_id()
        );
    }
    drop(decision);
    guard.release().unwrap();
    cleanup(&path);
}

#[test]
fn newer_local_descendant_yields_only_provenance_retaining_anchor_permit() {
    let path = db_path("reanchor");
    let store = SqliteWitnessSequenceStore::open(&path).unwrap();
    store.reserve_attempt(binding(1)).unwrap();
    let anchored_statement = store.frontier_statement(WITNESS).unwrap().unwrap();
    let xenia = verified_xenia_for_statement(anchored_statement);
    store.reserve_attempt(binding(2)).unwrap();

    let guard = SqliteWitnessFrontierPublicationGuard::acquire(&store, WITNESS).unwrap();
    let decision = classify_guarded_xenia_witness_frontier_v1(&guard, &xenia).unwrap();
    assert_eq!(
        decision.publication_disposition(),
        WitnessFrontierPublicationDispositionV1::AnchorRequired
    );
    assert!(matches!(
        decision.relation(),
        WitnessFrontierRecoveryRelationV1::LocalAheadVerifiedDescendant {
            trusted_high_watermark: 1,
            local_high_watermark: 2,
            ..
        }
    ));
    assert!(decision.publication_permit().is_none());
    {
        let permit = decision.anchor_permit().unwrap();
        assert_eq!(permit.witness_id(), WITNESS);
        assert_eq!(permit.frontier().high_watermark, 2);
        assert_eq!(permit.xenia_evidence().xenia().high_watermark(), 1);
        assert!(matches!(
            permit.relation(),
            WitnessFrontierRecoveryRelationV1::LocalAheadVerifiedDescendant { .. }
        ));
    }
    drop(decision);
    guard.release().unwrap();
    cleanup(&path);
}

fn cleanup(path: &Path) {
    let _ = fs::remove_file(path);
    let _ = fs::remove_file(format!("{}-wal", path.display()));
    let _ = fs::remove_file(format!("{}-shm", path.display()));
}
