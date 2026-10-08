use symthaea_swarm::holochain_projection::{
    EvidenceAnchorKind, HolochainActionHash, HolochainEvidenceAnchor,
    HolochainProjectionError, ReceiptSelectionContext, HOLOCHAIN_ACTION_HASH_BYTES,
    HOLOCHAIN_ACTION_HASH_PREFIX, MAX_RECEIPT_SELECTION_CANDIDATES,
};
use uuid::Uuid;

fn valid_action_hash(fill: u8) -> HolochainActionHash {
    let mut bytes = [fill; HOLOCHAIN_ACTION_HASH_BYTES];
    bytes[..3].copy_from_slice(&HOLOCHAIN_ACTION_HASH_PREFIX);
    HolochainActionHash::from_raw(bytes).unwrap()
}

fn anchor() -> HolochainEvidenceAnchor {
    HolochainEvidenceAnchor {
        anchor_id: Uuid::from_u128(1),
        origin_node_id: Uuid::from_u128(2),
        kind: EvidenceAnchorKind::Attestation,
        evidence_digest: [1; 32],
        parent_evidence_digest: Some([2; 32]),
        parent_action_hash: Some(valid_action_hash(7)),
        vds_root: Some([3; 32]),
        receipt_selection: Some(ReceiptSelectionContext {
            collection_sha256: [4; 32],
            collection_len: 2,
            selected_index: 1,
            selected_receipt_sha256: [5; 32],
            verified_capability_sha256: [6; 32],
            verified_composition_capability_sha256: [7; 32],
            selection_decision_sha256: [8; 32],
            selection_policy: "rfc9942/priority-first-valid-v1".into(),
            selection_policy_version: 1,
        }),
        selection_decision_action_hash: Some(valid_action_hash(8)),
        context: b"qualification".to_vec(),
    }
}

#[test]
fn projection_is_deterministic() {
    assert_eq!(anchor().canonical_bytes(), anchor().canonical_bytes());
}

#[test]
fn native_action_hashes_are_opaque_but_exactly_address_sized() {
    let hash = HolochainActionHash::from_raw([9; HOLOCHAIN_ACTION_HASH_BYTES]);
    assert_eq!(hash.as_ref().map(HolochainActionHash::as_bytes).unwrap().len(), HOLOCHAIN_ACTION_HASH_BYTES);
    assert!(HolochainActionHash::from_raw([0; HOLOCHAIN_ACTION_HASH_BYTES]).is_err());
    assert_eq!(
        HolochainActionHash::from_raw([9; HOLOCHAIN_ACTION_HASH_BYTES]),
        Err(HolochainProjectionError::InvalidActionHashType)
    );
}

#[test]
fn durable_dependency_claims_fail_closed_without_native_addresses() {
    let mut missing_parent = anchor();
    missing_parent.parent_action_hash = None;
    assert_eq!(
        missing_parent.validate(),
        Err(HolochainProjectionError::UnaddressableDependency("parent_evidence"))
    );

    let mut missing_selection = anchor();
    missing_selection.selection_decision_action_hash = None;
    assert_eq!(
        missing_selection.validate(),
        Err(HolochainProjectionError::UnaddressableDependency("selection_decision"))
    );
}

#[test]
fn selection_context_changes_projection_identity() {
    let mut first = anchor();
    let first_bytes = first.canonical_bytes().unwrap();
    first.receipt_selection.as_mut().unwrap().selected_index = 0;
    assert_ne!(first_bytes, first.canonical_bytes().unwrap());
}

#[test]
fn verified_capability_identity_changes_projection() {
    let mut first = anchor();
    let before = first.canonical_bytes().unwrap();
    first
        .receipt_selection
        .as_mut()
        .unwrap()
        .verified_capability_sha256[0] ^= 1;
    assert_ne!(before, first.canonical_bytes().unwrap());
}

#[test]
fn verified_composition_capability_identity_changes_projection() {
    let mut first = anchor();
    let before = first.canonical_bytes().unwrap();
    first
        .receipt_selection
        .as_mut()
        .unwrap()
        .verified_composition_capability_sha256[0] ^= 1;
    assert_ne!(before, first.canonical_bytes().unwrap());
}

#[test]
fn selection_decision_digest_changes_projection() {
    let mut first = anchor();
    let before = first.canonical_bytes().unwrap();
    first
        .receipt_selection
        .as_mut()
        .unwrap()
        .selection_decision_sha256[0] ^= 1;
    assert_ne!(before, first.canonical_bytes().unwrap());
}

#[test]
fn selection_dependency_identity_is_canonical() {
    let mut first = anchor();
    let before = first.canonical_bytes().unwrap();
    first.selection_decision_action_hash = Some(valid_action_hash(9));
    assert_ne!(before, first.canonical_bytes().unwrap());
}

#[cfg(feature = "semantic-receipts")]
use sha2::Digest;

#[cfg(feature = "semantic-receipts")]
use ring::{
    rand::SystemRandom,
    signature::{EcdsaKeyPair, KeyPair},
};

#[cfg(feature = "semantic-receipts")]
use symthaea_swarm::rfc9942_selection::evaluate_priority_first_valid;

#[cfg(feature = "semantic-receipts")]
use symthaea_swarm::semantic_evidence_vds::{
    Rfc9942ProofKind, Rfc9942ReceiptCollection, Rfc9942ReceiptEnvelope,
    Rfc9942ReceiptPayload, Rfc9942SignatureWithReceipts, Rfc9942Vdp,
    Rfc9162InclusionProof, Rfc9162Sha256Vds, COSE_ES256_ALGORITHM_ID,
};

#[cfg(feature = "semantic-receipts")]
fn signing_key(rng: &SystemRandom) -> EcdsaKeyPair {
    let pkcs8 = EcdsaKeyPair::generate_pkcs8(
        &ring::signature::ECDSA_P256_SHA256_FIXED_SIGNING,
        rng,
    )
    .unwrap();
    EcdsaKeyPair::from_pkcs8(
        &ring::signature::ECDSA_P256_SHA256_FIXED_SIGNING,
        pkcs8.as_ref(),
        rng,
    )
    .unwrap()
}

#[cfg(feature = "semantic-receipts")]
fn signed_inclusion_receipt(
    leaves: &[Vec<u8>],
    signer: &EcdsaKeyPair,
    rng: &SystemRandom,
) -> Rfc9942ReceiptEnvelope {
    let vds = Rfc9162Sha256Vds;
    let leaves = leaves.to_vec();
    let head = vds.tree_head(&leaves);
    let proof = vds.prove(&leaves, 0).unwrap().to_cbor();
    let vdp = Rfc9942Vdp::new(Rfc9942ProofKind::Inclusion, vec![proof]).unwrap();

    let unsigned = Rfc9942ReceiptEnvelope::new(
        COSE_ES256_ALGORITHM_ID,
        vdp.clone(),
        Rfc9942ReceiptPayload::Attached(head.root()),
        vec![0; 64],
    )
    .unwrap();
    let signature = signer
        .sign(rng, &unsigned.signature1_tbs(&[], None).unwrap())
        .unwrap()
        .as_ref()
        .to_vec();

    Rfc9942ReceiptEnvelope::new(
        COSE_ES256_ALGORITHM_ID,
        vdp,
        Rfc9942ReceiptPayload::Attached(head.root()),
        signature,
    )
    .unwrap()
}

#[cfg(feature = "semantic-receipts")]
fn cbor_bstr(bytes: &[u8]) -> Vec<u8> {
    assert!(bytes.len() < 256);
    let mut out = vec![0x40 | bytes.len() as u8];
    out.extend_from_slice(bytes);
    out
}

#[cfg(feature = "semantic-receipts")]
fn signed_outer_with_receipts(
    collection: &Rfc9942ReceiptCollection,
    payload: &[u8],
    signer: &EcdsaKeyPair,
    rng: &SystemRandom,
) -> Rfc9942SignatureWithReceipts {
    let collection_bytes = collection.to_cbor();
    let protected = [0xa1, 0x01, 0x26];

    let mut unsigned_wire = vec![0xd2, 0x84, 0x43];
    unsigned_wire.extend_from_slice(&protected);
    unsigned_wire.extend_from_slice(&[0xa1, 0x19, 0x01, 0x8a]);
    unsigned_wire.extend_from_slice(&collection_bytes);
    unsigned_wire.extend_from_slice(&cbor_bstr(payload));
    unsigned_wire.extend_from_slice(&cbor_bstr(&[0; 64]));

    let unsigned = Rfc9942SignatureWithReceipts::from_cbor(&unsigned_wire).unwrap();
    let signature = signer
        .sign(rng, &unsigned.signature1_tbs(&[], None).unwrap())
        .unwrap()
        .as_ref()
        .to_vec();

    let mut signed_wire = vec![0xd2, 0x84, 0x43];
    signed_wire.extend_from_slice(&protected);
    signed_wire.extend_from_slice(&[0xa1, 0x19, 0x01, 0x8a]);
    signed_wire.extend_from_slice(&collection_bytes);
    signed_wire.extend_from_slice(&cbor_bstr(payload));
    signed_wire.extend_from_slice(&cbor_bstr(&signature));

    Rfc9942SignatureWithReceipts::from_cbor(&signed_wire).unwrap()
}

#[cfg(feature = "semantic-receipts")]
#[test]
fn verified_selection_projection_requires_exact_capability_and_collection() {
    let rng = SystemRandom::new();
    let signer = signing_key(&rng);
    let key = signer.public_key().as_ref().to_vec();

    let first_receipt = signed_inclusion_receipt(
        &[b"candidate".to_vec(), b"other-a".to_vec()],
        &signer,
        &rng,
    );
    let second_receipt = signed_inclusion_receipt(
        &[b"candidate".to_vec(), b"other-b".to_vec()],
        &signer,
        &rng,
    );

    let first_verified = first_receipt
        .verify_es256_inclusion_state(b"candidate", &key, &[], None)
        .unwrap();
    let second_verified = second_receipt
        .verify_es256_inclusion_state(b"candidate", &key, &[], None)
        .unwrap();

    assert_ne!(
        first_verified.capability_sha256(),
        second_verified.capability_sha256()
    );
    assert_ne!(first_verified.receipt_sha256(), second_verified.receipt_sha256());

    let collection = Rfc9942ReceiptCollection::new(vec![first_receipt, second_receipt]).unwrap();
    let decision = evaluate_priority_first_valid(&collection, |index, _| {
        if index == 0 {
            Ok(())
        } else {
            Err(symthaea_swarm::Rfc9942VdpError::NoMatchingProof)
        }
    });

    assert_eq!(decision.selected_index, Some(0));
    assert!(matches!(
        decision.candidates[1].status,
        symthaea_swarm::rfc9942_selection::ReceiptSelectionCandidateStatus::NotEvaluatedAfterSelection
    ));

    let outer_signer = signing_key(&rng);
    let outer_key = outer_signer.public_key().as_ref().to_vec();
    let outer = signed_outer_with_receipts(&collection, b"candidate", &outer_signer, &rng);
    let (verified_outer, witness) = outer
        .verify_es256_inclusion_priority_first_valid_receipt_selection_state(
            &key,
            &outer_key,
            &[],
            &[],
            None,
        )
        .unwrap();

    let context = ReceiptSelectionContext::from_verified_selection(&witness).unwrap();

    assert_eq!(
        witness.verified_capability_sha256(),
        verified_outer.receipt().capability_sha256()
    );
    assert_eq!(
        witness.verified_composition_capability_sha256(),
        verified_outer.capability_sha256()
    );
    assert_eq!(
        witness.selection_decision_sha256().unwrap(),
        decision.validated_digest().unwrap()
    );
    assert_eq!(
        context.verified_capability_sha256,
        witness.verified_capability_sha256()
    );
    assert_eq!(
        context.verified_composition_capability_sha256,
        witness.verified_composition_capability_sha256()
    );

    // A different valid outer signer yields a different composition capability,
    // even though the selected Receipt and collection are unchanged.
    let other_outer_signer = signing_key(&rng);
    let other_outer_key = other_outer_signer.public_key().as_ref().to_vec();
    let other_outer =
        signed_outer_with_receipts(&collection, b"candidate", &other_outer_signer, &rng);
    let (other_verified_outer, _) = other_outer
        .verify_es256_inclusion_priority_first_valid_receipt_selection_state(
            &key,
            &other_outer_key,
            &[],
            &[],
            None,
        )
        .unwrap();
    assert_ne!(
        verified_outer.capability_sha256(),
        other_verified_outer.capability_sha256()
    );

    let reversed = Rfc9942ReceiptCollection::new(
        collection.receipts().iter().cloned().rev().collect(),
    )
    .unwrap();
    assert_eq!(
        witness.decision().validate_against_collection(&reversed),
        Err(symthaea_swarm::rfc9942_selection::ReceiptSelectionDecisionError::CollectionDigestMismatch)
    );

    assert!(MAX_RECEIPT_SELECTION_CANDIDATES >= collection.len() as u32);
}

#[cfg(feature = "semantic-receipts")]
#[test]
fn unprotected_outer_metadata_is_not_authenticated_but_is_provenance_bound() {
    let rng = SystemRandom::new();
    let receipt_signer = signing_key(&rng);
    let outer_signer = signing_key(&rng);
    let receipt_key = receipt_signer.public_key().as_ref().to_vec();
    let outer_key = outer_signer.public_key().as_ref().to_vec();

    let collection = Rfc9942ReceiptCollection::new(vec![
        signed_inclusion_receipt(
            &[b"candidate".to_vec(), b"other".to_vec()],
            &receipt_signer,
            &rng,
        ),
    ])
    .unwrap();

    let outer = signed_outer_with_receipts(
        &collection,
        b"candidate",
        &outer_signer,
        &rng,
    );
    let baseline_wire = outer.to_cbor();
    let (baseline_verified, _) = outer
        .verify_es256_inclusion_priority_first_valid_receipt_selection_state(
            &receipt_key,
            &outer_key,
            &[],
            &[],
            None,
        )
        .unwrap();

    // Add an unrelated unprotected header without resigning. RFC 9052's
    // Sig_structure excludes the unprotected bucket, so the outer signature
    // must continue to verify. The provenance capability must nevertheless
    // distinguish the modified transport artifact.
    let collection_start = baseline_wire
        .windows(4)
        .position(|window| window == [0xa1, 0x19, 0x01, 0x8a])
        .expect("receipt header marker");

    // The existing unprotected map is {394: collection}; replace its map
    // header with a two-member map and insert an opaque extension before 394.
    let mut modified_wire = baseline_wire.clone();
    assert_eq!(modified_wire.get(collection_start), Some(&0xa1));
    modified_wire[collection_start] = 0xa2;
    modified_wire.splice(
        collection_start..collection_start,
        [0x19, 0x23, 0x28, 0x01],
    );

    let modified = Rfc9942SignatureWithReceipts::from_cbor(&modified_wire).unwrap();
    let (modified_verified, _) = modified
        .verify_es256_inclusion_priority_first_valid_receipt_selection_state(
            &receipt_key,
            &outer_key,
            &[],
            &[],
            None,
        )
        .unwrap();

    assert_ne!(
        baseline_verified.capability_sha256(),
        modified_verified.capability_sha256()
    );
    assert_ne!(
        baseline_verified.outer_signature_with_receipt_sha256(),
        modified_verified.outer_signature_with_receipt_sha256()
    );
    assert_ne!(modified_wire, baseline_wire);
}
#[cfg(feature = "semantic-receipts")]
#[test]
fn atomic_priority_selection_api_returns_bound_witness() {
    let rng = SystemRandom::new();
    let receipt_signer = signing_key(&rng);
    let outer_signer = signing_key(&rng);
    let receipt_key = receipt_signer.public_key().as_ref().to_vec();
    let outer_key = outer_signer.public_key().as_ref().to_vec();

    let first_receipt = signed_inclusion_receipt(
        &[b"candidate".to_vec(), b"other-a".to_vec()],
        &receipt_signer,
        &rng,
    );
    let second_receipt = signed_inclusion_receipt(
        &[b"candidate".to_vec(), b"other-b".to_vec()],
        &receipt_signer,
        &rng,
    );
    let collection =
        Rfc9942ReceiptCollection::new(vec![first_receipt, second_receipt]).unwrap();

    // The helper produces a real Signature_With_Receipt whose outer payload
    // is exactly the candidate bytes consumed by inner inclusion verification.
    let outer = signed_outer_with_receipts(&collection, b"candidate", &outer_signer, &rng);
    let (verified_outer, witness) = outer
        .verify_es256_inclusion_priority_first_valid_receipt_selection_state(
            &receipt_key,
            &outer_key,
            &[],
            &[],
            None,
        )
        .unwrap();

    assert_eq!(witness.decision().selected_index, Some(0));
    assert_eq!(
        witness.decision().selected_receipt_sha256,
        Some(witness.decision().candidates[0].receipt_sha256)
    );
    assert!(matches!(
        witness.decision().candidates[1].status,
        symthaea_swarm::rfc9942_selection::ReceiptSelectionCandidateStatus::NotEvaluatedAfterSelection
    ));
    assert_eq!(
        witness.verified_capability_sha256(),
        verified_outer.receipt().capability_sha256()
    );
    assert_eq!(
        witness.verified_composition_capability_sha256(),
        verified_outer.capability_sha256()
    );
    assert_eq!(
        witness.selection_decision_sha256().unwrap(),
        witness.decision().validated_digest().unwrap()
    );
}

#[cfg(feature = "semantic-receipts")]
#[test]
fn noncanonical_protected_algorithm_encoding_requires_exact_tbs() {
    let rng = SystemRandom::new();
    let receipt_signer = signing_key(&rng);
    let outer_signer = signing_key(&rng);
    let receipt_key = receipt_signer.public_key().as_ref().to_vec();
    let outer_key = outer_signer.public_key().as_ref().to_vec();

    let collection = Rfc9942ReceiptCollection::new(vec![
        signed_inclusion_receipt(
            &[b"candidate".to_vec(), b"other".to_vec()],
            &receipt_signer,
            &rng,
        ),
    ])
    .unwrap();
    let collection_bytes = collection.to_cbor();

    // -7 is canonically encoded as 0x26. The equivalent noncanonical
    // negative-integer encoding is 0x38 0x06. Protected bytes are part of
    // COSE Sig_structure, so each representation requires its own signature.
    let canonical_protected = [0xa1, 0x01, 0x26];
    let noncanonical_protected = [0xa1, 0x01, 0x38, 0x06];

    let sign_with_protected = |protected: &[u8]| {
        let mut unsigned_wire = vec![0xd2, 0x84, 0x43];
        unsigned_wire.extend_from_slice(protected);
        unsigned_wire.extend_from_slice(&[0xa1, 0x19, 0x01, 0x8a]);
        unsigned_wire.extend_from_slice(&collection_bytes);
        unsigned_wire.extend_from_slice(&cbor_bstr(b"candidate"));
        unsigned_wire.extend_from_slice(&cbor_bstr(&[0; 64]));
        let unsigned = Rfc9942SignatureWithReceipts::from_cbor(&unsigned_wire).unwrap();
        let signature = outer_signer
            .sign(rng.as_ref(), &unsigned.signature1_tbs(&[], None).unwrap())
            .unwrap()
            .as_ref()
            .to_vec();

        let mut signed_wire = vec![0xd2, 0x84, 0x43];
        signed_wire.extend_from_slice(protected);
        signed_wire.extend_from_slice(&[0xa1, 0x19, 0x01, 0x8a]);
        signed_wire.extend_from_slice(&collection_bytes);
        signed_wire.extend_from_slice(&cbor_bstr(b"candidate"));
        signed_wire.extend_from_slice(&cbor_bstr(&signature));
        Rfc9942SignatureWithReceipts::from_cbor(&signed_wire).unwrap()
    };

    let canonical = sign_with_protected(&canonical_protected);
    let noncanonical = sign_with_protected(&noncanonical_protected);

    let (canonical_verified, _) = canonical
        .verify_es256_inclusion_priority_first_valid_receipt_selection_state(
            &receipt_key,
            &outer_key,
            &[],
            &[],
            None,
        )
        .unwrap();
    let (noncanonical_verified, _) = noncanonical
        .verify_es256_inclusion_priority_first_valid_receipt_selection_state(
            &receipt_key,
            &outer_key,
            &[],
            &[],
            None,
        )
        .unwrap();

    assert_eq!(canonical_verified.outer_algorithm_id(), COSE_ES256_ALGORITHM_ID);
    assert_eq!(noncanonical_verified.outer_algorithm_id(), COSE_ES256_ALGORITHM_ID);
    assert_ne!(
        canonical_verified.outer_protected_header_sha256(),
        noncanonical_verified.outer_protected_header_sha256()
    );
    assert_ne!(
        canonical_verified.outer_signature_with_receipt_sha256(),
        noncanonical_verified.outer_signature_with_receipt_sha256()
    );
    assert_ne!(
        canonical_verified.capability_sha256(),
        noncanonical_verified.capability_sha256()
    );
}

#[cfg(feature = "semantic-receipts")]
#[test]
fn noncanonical_outer_wire_changes_only_wire_capability_identity() {
    let rng = SystemRandom::new();
    let receipt_signer = signing_key(&rng);
    let outer_signer = signing_key(&rng);
    let receipt_key = receipt_signer.public_key().as_ref().to_vec();
    let outer_key = outer_signer.public_key().as_ref().to_vec();

    let collection = Rfc9942ReceiptCollection::new(vec![
        signed_inclusion_receipt(
            &[b"candidate".to_vec(), b"other".to_vec()],
            &receipt_signer,
            &rng,
        ),
    ])
    .unwrap();
    let canonical_outer =
        signed_outer_with_receipts(&collection, b"candidate", &outer_signer, &rng);
    let canonical_wire = canonical_outer.to_cbor();
    assert_eq!(canonical_wire.get(1), Some(&0x84));

    // COSE verification authenticates Sig_structure inputs, not the enclosing
    // COSE_Sign1 array framing. Therefore an indefinite-length top-level array
    // can retain equivalent verified semantics while still being a distinct
    // wire artifact for provenance.
    let mut noncanonical_wire = canonical_wire.clone();
    noncanonical_wire[1] = 0x9f;
    noncanonical_wire.push(0xff);

    let noncanonical_outer =
        Rfc9942SignatureWithReceipts::from_cbor(&noncanonical_wire).unwrap();
    assert_eq!(noncanonical_outer.to_cbor(), noncanonical_wire);
    // Top-level CBOR framing is not part of the semantic fields compared by
    // this structural type; exact wire identity is carried separately.
    assert_eq!(canonical_outer, noncanonical_outer);

    let (canonical_verified, canonical_selection) = canonical_outer
        .verify_es256_inclusion_priority_first_valid_receipt_selection_state(
            &receipt_key,
            &outer_key,
            &[],
            &[],
            None,
        )
        .unwrap();
    let (noncanonical_verified, noncanonical_selection) = noncanonical_outer
        .verify_es256_inclusion_priority_first_valid_receipt_selection_state(
            &receipt_key,
            &outer_key,
            &[],
            &[],
            None,
        )
        .unwrap();

    assert_eq!(canonical_selection.decision(), noncanonical_selection.decision());
    assert_eq!(
        canonical_verified.receipt().proof(),
        noncanonical_verified.receipt().proof()
    );
    assert_ne!(
        canonical_verified.outer_signature_with_receipt_sha256(),
        noncanonical_verified.outer_signature_with_receipt_sha256()
    );
    assert_ne!(
        canonical_verified.capability_sha256(),
        noncanonical_verified.capability_sha256()
    );
}
#[cfg(feature = "semantic-receipts")]
#[test]
fn noncanonical_receipt_wire_identity_survives_verified_projection() {
    let rng = SystemRandom::new();
    let signer = signing_key(&rng);
    let key = signer.public_key().as_ref().to_vec();

    let canonical_receipt = signed_inclusion_receipt(
        &[b"candidate".to_vec(), b"other".to_vec()],
        &signer,
        &rng,
    );
    let canonical_wire = canonical_receipt.to_cbor();
    let mut noncanonical_wire = canonical_wire.clone();
    assert_eq!(noncanonical_wire.get(1), Some(&0x84));
    noncanonical_wire[1] = 0x9f;
    noncanonical_wire.push(0xff);

    let mut collection_wire = vec![0x81];
    collection_wire.extend_from_slice(&cbor_bstr(&noncanonical_wire));
    let collection = Rfc9942ReceiptCollection::from_cbor(&collection_wire).unwrap();

    let parsed = collection.receipts()[0].clone();
    assert_eq!(parsed.to_cbor(), noncanonical_wire);

    // The noncanonical top-level array is semantically equivalent for COSE
    // processing, but its exact wire identity is different. Both therefore
    // verify the same semantic receipt state while retaining distinct
    // provenance capabilities.
    let canonical_verified = canonical_receipt
        .verify_es256_inclusion_state(b"candidate", &key, &[], None)
        .unwrap();
    let verified = parsed
        .verify_es256_inclusion_state(b"candidate", &key, &[], None)
        .unwrap();
    assert_eq!(
        verified.receipt_sha256(),
        sha2::Sha256::digest(&noncanonical_wire).into()
    );
    assert_ne!(
        canonical_verified.receipt_sha256(),
        verified.receipt_sha256()
    );
    assert_ne!(
        canonical_verified.capability_sha256(),
        verified.capability_sha256()
    );
    assert_eq!(
        canonical_verified.proof(),
        verified.proof()
    );

    let outer_signer = signing_key(&rng);
    let outer_key = outer_signer.public_key().as_ref().to_vec();
    let outer = signed_outer_with_receipts(&collection, b"candidate", &outer_signer, &rng);
    let (_verified_outer, witness) = outer
        .verify_es256_inclusion_priority_first_valid_receipt_selection_state(
            &key,
            &outer_key,
            &[],
            &[],
            None,
        )
        .unwrap();
    let context = ReceiptSelectionContext::from_verified_selection(&witness).unwrap();
    assert_eq!(
        context.selected_receipt_sha256,
        sha2::Sha256::digest(&noncanonical_wire).into()
    );
}
