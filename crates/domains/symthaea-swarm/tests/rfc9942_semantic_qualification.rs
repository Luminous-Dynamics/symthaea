//! Focused RFC 9942 semantic qualification.
//!
//! This is an integration test on purpose: it exercises the public RFC 9942
//! boundary without compiling the crate's unrelated #[cfg(test)] modules.

#![cfg(feature = "semantic-receipts")]

use symthaea_swarm::semantic_evidence_vds::{
    Rfc9942ProofKind, Rfc9942ReceiptEnvelope, Rfc9942ReceiptPayload, Rfc9942SignatureWithReceipts,
    Rfc9162ConsistencyProof, Rfc9162InclusionProof, Rfc9942VdpError, Rfc9942Vdp,
    Rfc9942ReceiptCollection, Rfc9942VerifiedProof, Rfc9162Sha256Vds,
    Rfc9942Es256CoseKey, COSE_ES256_ALGORITHM_ID,
};
use ring::{rand::SystemRandom, signature::EcdsaKeyPair};
use sha2::Digest;

const RFC8392_PRIVATE_D: [u8; 32] = [
    0x6c, 0x13, 0x82, 0x76, 0x5a, 0xec, 0x53, 0x58,
    0xf1, 0x17, 0x73, 0x3d, 0x28, 0x1c, 0x1c, 0x7b,
    0xdc, 0x39, 0x88, 0x4d, 0x04, 0xa4, 0x5a, 0x1e,
    0x6c, 0x67, 0xc8, 0x58, 0xbc, 0x20, 0x6c, 0x19,
];
const RFC8392_PUBLIC_X: [u8; 32] = [
    0x14, 0x33, 0x29, 0xcc, 0xe7, 0x86, 0x8e, 0x41,
    0x69, 0x27, 0x59, 0x9c, 0xf6, 0x5a, 0x34, 0xf3,
    0xce, 0x2f, 0xfd, 0xa5, 0x5a, 0x7a, 0xec, 0xa6,
    0x9e, 0xd8, 0x91, 0x9a, 0x39, 0x4d, 0x42, 0xf0,
];
const RFC8392_PUBLIC_Y: [u8; 32] = [
    0x60, 0xf7, 0xf1, 0xa7, 0x80, 0xd8, 0xa7, 0x83,
    0xbf, 0xb7, 0xa2, 0xdd, 0x6b, 0x27, 0x96, 0xe8,
    0x12, 0x8d, 0xbc, 0xef, 0x9d, 0x3d, 0x16, 0x8d,
    0xb9, 0x52, 0x99, 0x71, 0xa3, 0x6e, 0x7b, 0x09,
];

fn rfc8392_public_key() -> [u8; 65] {
    let mut key = [0u8; 65];
    key[0] = 0x04;
    key[1..33].copy_from_slice(&RFC8392_PUBLIC_X);
    key[33..65].copy_from_slice(&RFC8392_PUBLIC_Y);
    key
}

#[test]
fn rfc9942_inclusion_and_consistency_preserve_required_verification_order() {
    let inclusion_proof =
        Rfc9162InclusionProof::new(2, 0, vec![[0x11; 32]]).to_cbor();
    let inclusion_vdp =
        Rfc9942Vdp::new(Rfc9942ProofKind::Inclusion, vec![inclusion_proof]).unwrap();
    let inclusion =
        Rfc9942ReceiptEnvelope::new(
            COSE_ES256_ALGORITHM_ID,
            inclusion_vdp,
            Rfc9942ReceiptPayload::Attached([0x22; 32]),
            vec![0xAA; 63],
        )
        .unwrap();

    // Both proof and signature are invalid. RFC 9942 requires inclusion proof
    // verification before signature verification, so the proof failure wins.
    let public_key = [0x04; 65];
    assert_eq!(
        inclusion.verify_es256_inclusion(
            b"candidate",
            &public_key,
            &[],
            None,
        ),
        Err(Rfc9942VdpError::NoMatchingProof)
    );

    let consistency_proof =
        Rfc9162ConsistencyProof::new(1, 2, vec![[0x33; 32]]).to_cbor();
    let consistency_vdp =
        Rfc9942Vdp::new(Rfc9942ProofKind::Consistency, vec![consistency_proof]).unwrap();
    let consistency =
        Rfc9942ReceiptEnvelope::new(
            COSE_ES256_ALGORITHM_ID,
            consistency_vdp,
            Rfc9942ReceiptPayload::Attached([0x44; 32]),
            vec![0xBB; 63],
        )
        .unwrap();

    // Both proof and signature are invalid. RFC 9942 requires the signature
    // check before consistency verification, so the signature failure wins.
    assert_eq!(
        consistency.verify_es256_consistency(
            symthaea_swarm::semantic_evidence_vds::VdsTreeHead::new(1, [0x55; 32]),
            &public_key,
            &[],
            None,
        ),
        Err(Rfc9942VdpError::InvalidEs256Signature)
    );
    assert_eq!(
        consistency.verify_es256_consistency_state(
            symthaea_swarm::semantic_evidence_vds::VdsTreeHead::new(1, [0x55; 32]),
            &public_key,
            &[],
            None,
        ),
        Err(Rfc9942VdpError::InvalidEs256Signature)
    );
}

#[test]
fn rfc9942_outer_receipts_are_ordered_and_placement_is_not_normalized() {
    fn bytes(b: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        match b.len() {
            0..=23 => out.push(0x40 + b.len() as u8),
            24..=255 => out.extend_from_slice(&[0x58, b.len() as u8]),
            _ => unreachable!(),
        }
        out.extend_from_slice(b);
        out
    }

    // RFC 9942 label 394 requires a canonical two-byte unsigned integer.
    fn int_394(out: &mut Vec<u8>) {
        out.extend_from_slice(&[0x19, 0x01, 0x8a]);
    }

    let proof = Rfc9162InclusionProof::new(2, 0, vec![[0x66; 32]]).to_cbor();
    let vdp = Rfc9942Vdp::new(Rfc9942ProofKind::Inclusion, vec![proof]).unwrap();
    let receipt = Rfc9942ReceiptEnvelope::new(
        COSE_ES256_ALGORITHM_ID,
        vdp,
        Rfc9942ReceiptPayload::Attached([0x77; 32]),
        vec![0x88; 64],
    )
    .unwrap();
    let collection =
        Rfc9942ReceiptCollection::new(vec![receipt]).unwrap().to_cbor();

    let mut protected = Vec::new();
    protected.push(0xa1);
    int_394(&mut protected);
    protected.extend_from_slice(&collection);

    let mut wire = Vec::new();
    wire.extend_from_slice(&[0xd2, 0x84]);
    wire.extend_from_slice(&bytes(&protected));
    wire.push(0xa0);
    wire.extend_from_slice(&bytes(b"payload"));
    wire.extend_from_slice(&bytes(&[0x99; 64]));

    let decoded = Rfc9942SignatureWithReceipts::from_cbor(&wire).unwrap();
    assert!(decoded.protected_receipts().is_some());
    assert!(decoded.unprotected_receipts().is_none());
    assert_eq!(decoded.to_cbor(), wire);
}

#[test]
fn rfc9162_inclusion_and_consistency_path_bounds_are_tree_size_derived() {
    let hash = [0x11; 32];

    // Raw RFC 9162 permits an empty singleton inclusion path.
    let singleton = [0x83, 0x01, 0x00, 0x80];
    assert!(Rfc9162InclusionProof::from_cbor(&singleton).is_ok());

    // RFC 9942's profile uses [+ bstr] for inclusion paths, so the
    // singleton proof is rejected when promoted into an RFC 9942 VDP.
    assert_eq!(
        Rfc9942Vdp::new(Rfc9942ProofKind::Inclusion, vec![singleton.to_vec()]),
        Err(Rfc9942VdpError::InvalidProof(
            symthaea_swarm::semantic_evidence_vds::Rfc9162ProofDecodeError::InvalidStructure
        ))
    );

    // For a two-leaf tree, ceil(log2(2)) == 1. A two-node path is impossible.
    let mut short_tree = Vec::new();
    short_tree.extend_from_slice(&[0x83, 0x01, 0x00, 0x82]);
    for _ in 0..2 {
        short_tree.push(0x58);
        short_tree.push(0x20);
        short_tree.extend_from_slice(&hash);
    }
    assert_eq!(
        Rfc9162InclusionProof::from_cbor(&short_tree),
        Err(symthaea_swarm::semantic_evidence_vds::Rfc9162ProofDecodeError::InvalidStructure)
    );

    // At the u64 ceiling, an inclusion path may reach 64 hashes.
    let mut max_inclusion = Vec::new();
    max_inclusion.extend_from_slice(&[
        0x83, 0x1b, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, // tree_size=u64::MAX
        0x00, 0x98, 0x40, // leaf_index=0, path length=64
    ]);
    for _ in 0..64 {
        max_inclusion.push(0x58);
        max_inclusion.push(0x20);
        max_inclusion.extend_from_slice(&hash);
    }
    assert!(Rfc9162InclusionProof::from_cbor(&max_inclusion).is_ok());

    // Consistency uses ceil(log2(second)) + 1. For second=2, three nodes are impossible.
    let mut short_consistency = Vec::new();
    short_consistency.extend_from_slice(&[0x83, 0x01, 0x02, 0x83]);
    for _ in 0..3 {
        short_consistency.push(0x58);
        short_consistency.push(0x20);
        short_consistency.extend_from_slice(&hash);
    }
    assert_eq!(
        Rfc9162ConsistencyProof::from_cbor(&short_consistency),
        Err(symthaea_swarm::semantic_evidence_vds::Rfc9162ProofDecodeError::InvalidStructure)
    );

    // At the u64 ceiling, RFC 9162's consistency bound reaches 65.
    let mut max_consistency = Vec::new();
    max_consistency.extend_from_slice(&[
        0x83, 0x01, 0x1b, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0x98, 0x41, // old=1, new=u64::MAX, path length=65
    ]);
    for _ in 0..65 {
        max_consistency.push(0x58);
        max_consistency.push(0x20);
        max_consistency.extend_from_slice(&hash);
    }
    assert!(Rfc9162ConsistencyProof::from_cbor(&max_consistency).is_ok());
}


#[test]
fn cose_extension_values_accept_well_formed_simple_items_and_round_trip_exactly() {
    for value in [&[0xe0][..], &[0xf3][..], &[0xf8, 0x20][..]] {
        let mut protected = Vec::new();
        protected.extend_from_slice(&[0xa1, 0x19, 0x03, 0xe7]);
        protected.extend_from_slice(value);

        let mut encoded = Vec::new();
        encoded.extend_from_slice(&[0xd2, 0x84]);
        encoded.push(0x45);
        encoded.extend_from_slice(&protected);
        encoded.push(0xa0);
        encoded.push(0xf6);
        encoded.extend_from_slice(&[0x41, 0xaa]);

        let decoded = Rfc9942SignatureWithReceipts::from_cbor(&encoded)
            .expect("well-formed generic COSE header value must be accepted");
        assert_eq!(decoded.to_cbor(), encoded);
    }
}

#[test]
fn rfc9942_receipt_decode_rejects_malformed_rfc9162_vdp_content() {
    // COSE_Sign1 protected = {1: -7, 395: 1}; unprotected = {396: {-1: [h'01']}}.
    // The inner inclusion proof is not an RFC 9162 proof array and must be
    // rejected during receipt decoding rather than being deferred to verify().
    let protected = [0xa2, 0x01, 0x26, 0x19, 0x01, 0x8b, 0x01];
    let encoded = [
        0xd2, 0x84, 0x47, protected[0], protected[1], protected[2], protected[3],
        protected[4], protected[5], protected[6],
        0xa1, 0x19, 0x01, 0x8c, 0xa1, 0x20, 0x81, 0x41, 0x01,
        0xf6, 0x40,
    ];
    assert_eq!(
        Rfc9942ReceiptEnvelope::from_cbor(&encoded),
        Err(Rfc9942VdpError::InvalidProof(
            symthaea_swarm::semantic_evidence_vds::Rfc9162ProofDecodeError::InvalidStructure
        ))
    );
}

#[test]
fn cose_extension_accepts_full_range_unsigned_integer_labels() {
    // label = 2^63, represented canonically as CBOR uint64.
    // RFC 9052 permits int labels; the adapter must not collapse or reject
    // this value merely because its internal signed representation is narrower.
    let protected = [
        0xa1, 0x1b, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x20,
    ];

    let mut encoded = Vec::new();
    encoded.extend_from_slice(&[0xd2, 0x84, 0x4b]);
    encoded.extend_from_slice(&protected);
    encoded.extend_from_slice(&[0xa0, 0xf6, 0x41, 0xaa]);

    let decoded = Rfc9942SignatureWithReceipts::from_cbor(&encoded)
        .expect("full-range unsigned COSE label must be accepted");
    assert_eq!(decoded.to_cbor(), encoded);
}

#[test]
fn cose_extension_accepts_full_range_negative_integer_labels() {
    // label = -2^64, encoded canonically as CBOR major type 1 with argument u64::MAX.
    let protected = [
        0xa1, 0x3b, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x20,
    ];

    let mut encoded = Vec::new();
    encoded.extend_from_slice(&[0xd2, 0x84, 0x4b]);
    encoded.extend_from_slice(&protected);
    encoded.extend_from_slice(&[0xa0, 0xf6, 0x41, 0xaa]);

    let decoded = Rfc9942SignatureWithReceipts::from_cbor(&encoded)
        .expect("full-range negative COSE label must be accepted");
    assert_eq!(decoded.to_cbor(), encoded);
}

#[test]
fn cose_extension_accepts_full_range_integer_values() {
    for value in [
        [
            0x1b, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        ],
        [
            0x3b, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        ],
    ] {
        // Protected map = { 999 => value }; total serialized map size is 14 bytes.
        let mut protected = Vec::with_capacity(14);
        protected.extend_from_slice(&[0xa1, 0x19, 0x03, 0xe7]);
        protected.extend_from_slice(&value);

        let mut encoded = Vec::new();
        encoded.extend_from_slice(&[0xd2, 0x84, 0x4e]);
        encoded.extend_from_slice(&protected);
        encoded.extend_from_slice(&[0xa0, 0xf6, 0x41, 0xaa]);

        let decoded = Rfc9942SignatureWithReceipts::from_cbor(&encoded)
            .expect("full-range generic integer value must be accepted");
        assert_eq!(decoded.to_cbor(), encoded);
    }
}

#[test]
fn rfc9942_semantic_state_cannot_confuse_valid_signature_with_wrong_entry() {
    fn signed_receipt(candidate: &[u8], other_entry: &[u8]) -> Rfc9942ReceiptEnvelope {
        let leaves = vec![candidate.to_vec(), other_entry.to_vec()];
        let vds = Rfc9162Sha256Vds;
        let head = vds.tree_head(&leaves);
        let proof = vds.inclusion_proof(&leaves, 0).unwrap().to_cbor();
        let vdp = Rfc9942Vdp::new(Rfc9942ProofKind::Inclusion, vec![proof]).unwrap();

        let unsigned = Rfc9942ReceiptEnvelope::new(
            COSE_ES256_ALGORITHM_ID,
            vdp.clone(),
            Rfc9942ReceiptPayload::Attached(head.root()),
            vec![0u8; 64],
        )
        .unwrap();

        let rng = SystemRandom::new();
        let signing_key = EcdsaKeyPair::from_private_key_and_public_key(
            &ring::signature::ECDSA_P256_SHA256_FIXED_SIGNING,
            &RFC8392_PRIVATE_D,
            &rfc8392_public_key(),
            &rng,
        )
        .unwrap();
        let tbs = unsigned.signature1_tbs(&[], None).unwrap();
        let signature = signing_key.sign(&rng, &tbs).unwrap().as_ref().to_vec();

        Rfc9942ReceiptEnvelope::new(
            COSE_ES256_ALGORITHM_ID,
            vdp,
            Rfc9942ReceiptPayload::Attached(head.root()),
            signature,
        )
        .unwrap()
    }

    let receipt = signed_receipt(b"candidate", b"other-entry");
    let key = rfc8392_public_key();

    // Signature verification alone proves only that the protected headers and
    // selected payload were signed by the key.
    receipt.verify_es256(&key, &[], None).unwrap();

    let state = receipt
        .verify_es256_inclusion_state(b"candidate", &key, &[], None)
        .unwrap();
    assert_eq!(state.algorithm_id(), COSE_ES256_ALGORITHM_ID);
    assert_eq!(state.vds_id(), 1);
    assert!(matches!(
        state.proof(),
        Rfc9942VerifiedProof::Inclusion { .. }
    ));
    assert_eq!(state.proof().proof_index(), 0);
    assert_eq!(state.proof().inclusion_head().unwrap().tree_size(), 2);
    let mut leaf_input = Vec::with_capacity(1 + b"candidate".len());
    leaf_input.push(0x00);
    leaf_input.extend_from_slice(b"candidate");
    let expected_leaf: [u8; 32] = sha2::Sha256::digest(&leaf_input).into();
    assert_eq!(state.proof().inclusion_candidate_leaf(), Some(expected_leaf));

    // The same validly signed Receipt must not be composable with a different
    // candidate entry. This closes the signature-success/semantic-proof gap.
    assert_eq!(
        receipt.verify_es256_inclusion_state(b"different", &key, &[], None),
        Err(Rfc9942VdpError::NoMatchingProof)
    );
}


#[test]
fn rfc9942_consistency_state_binds_signature_to_detached_root() {
    let leaves = vec![
        b"old-a".to_vec(),
        b"old-b".to_vec(),
        b"new-c".to_vec(),
        b"new-d".to_vec(),
    ];
    let vds = Rfc9162Sha256Vds;
    let older = vds.tree_head(&leaves[..2].to_vec());
    let newer = vds.tree_head(&leaves);
    let proof = vds.prove(&leaves, 2).unwrap().to_cbor();
    let vdp = Rfc9942Vdp::new(Rfc9942ProofKind::Consistency, vec![proof]).unwrap();

    let unsigned = Rfc9942ReceiptEnvelope::new(
        COSE_ES256_ALGORITHM_ID,
        vdp.clone(),
        Rfc9942ReceiptPayload::Detached,
        vec![0u8; 64],
    )
    .unwrap();

    let rng = SystemRandom::new();
    let signing_key = EcdsaKeyPair::from_private_key_and_public_key(
        &ring::signature::ECDSA_P256_SHA256_FIXED_SIGNING,
        &RFC8392_PRIVATE_D,
        &rfc8392_public_key(),
        &rng,
    )
    .unwrap();
    let root = newer.root();
    let tbs = unsigned.signature1_tbs(&[], Some(&root)).unwrap();
    let signature = signing_key.sign(&rng, &tbs).unwrap().as_ref().to_vec();

    let receipt = Rfc9942ReceiptEnvelope::new(
        COSE_ES256_ALGORITHM_ID,
        vdp,
        Rfc9942ReceiptPayload::Detached,
        signature,
    )
    .unwrap();
    let key = rfc8392_public_key();

    let state = receipt
        .verify_es256_consistency_state(older, &key, &[], Some(&root))
        .unwrap();
    assert_eq!(state.algorithm_id(), COSE_ES256_ALGORITHM_ID);
    assert_eq!(state.vds_id(), 1);
    assert_eq!(
        state.proof().consistency_heads(),
        Some((older, newer))
    );
    assert_eq!(state.proof().proof_index(), 0);

    // A detached root is part of the signed Sig_structure. Supplying a
    // different root therefore fails cryptographically before proof evaluation.
    let wrong_root = [0xA5; 32];
    assert_eq!(
        receipt.verify_es256_consistency_state(older, &key, &[], Some(&wrong_root)),
        Err(Rfc9942VdpError::InvalidEs256Signature)
    );
}



#[test]
fn rfc9942_es256_cose_key_inclusion_uses_detached_proof_derived_root() {
    let candidate = b"candidate";
    let leaves = vec![candidate.to_vec(), b"other-entry".to_vec()];
    let vds = Rfc9162Sha256Vds;
    let proof = vds.inclusion_proof(&leaves, 0).unwrap().to_cbor();
    let head = vds.tree_head(&leaves);

    let vdp = Rfc9942Vdp::new(Rfc9942ProofKind::Inclusion, vec![proof]).unwrap();

    let cose_key_wire = {
        let mut out = vec![0xa5, 0x01, 0x02, 0x03, 0x26, 0x20, 0x01, 0x21, 0x58, 0x20];
        out.extend_from_slice(&RFC8392_PUBLIC_X);
        out.extend_from_slice(&[0x22, 0x58, 0x20]);
        out.extend_from_slice(&RFC8392_PUBLIC_Y);
        out
    };
    let cose_key = Rfc9942Es256CoseKey::from_cbor(&cose_key_wire).unwrap();

    let unsigned = Rfc9942ReceiptEnvelope::new(
        COSE_ES256_ALGORITHM_ID,
        vdp.clone(),
        Rfc9942ReceiptPayload::Detached,
        vec![0u8; 64],
    )
    .unwrap();

    let rng = SystemRandom::new();
    let signer = EcdsaKeyPair::from_private_key_and_public_key(
        &ring::signature::ECDSA_P256_SHA256_FIXED_SIGNING,
        &RFC8392_PRIVATE_D,
        &rfc8392_public_key(),
        &rng,
    )
    .unwrap();
    let tbs = unsigned.signature1_tbs(&[], Some(head.root())).unwrap();
    let signature = signer.sign(&rng, &tbs).unwrap().as_ref().to_vec();

    let receipt = Rfc9942ReceiptEnvelope::new(
        COSE_ES256_ALGORITHM_ID,
        vdp,
        Rfc9942ReceiptPayload::Detached,
        signature,
    )
    .unwrap();

    // The detached payload need not be supplied by the caller for inclusion:
    // the semantic verifier derives the root from the candidate + proof and
    // then authenticates that exact derived root. The COSE_Key convenience
    // helper must remain on that same path.
    assert_eq!(
        receipt.verify_es256_cose_key_inclusion(candidate, &cose_key, &[], None),
        Ok(head)
    );
}


#[test]
fn rfc9942_es256_cose_key_consistency_preserves_signature_first_order() {
    // The COSE_Key convenience helper must delegate to the canonical semantic
    // verifier. Both the signature and consistency proof are deliberately
    // invalid; RFC 9942 requires the signature failure to win.
    let consistency_proof =
        Rfc9162ConsistencyProof::new(1, 2, vec![[0x33; 32]]).to_cbor();
    let vdp = Rfc9942Vdp::new(Rfc9942ProofKind::Consistency, vec![consistency_proof])
        .unwrap();
    let receipt = Rfc9942ReceiptEnvelope::new(
        COSE_ES256_ALGORITHM_ID,
        vdp,
        Rfc9942ReceiptPayload::Attached([0x44; 32]),
        vec![0xBB; 63],
    )
    .unwrap();

    let cose_key_wire = {
        let mut out = vec![0xa5, 0x01, 0x02, 0x03, 0x26, 0x20, 0x01, 0x21, 0x58, 0x20];
        out.extend_from_slice(&RFC8392_PUBLIC_X);
        out.extend_from_slice(&[0x22, 0x58, 0x20]);
        out.extend_from_slice(&RFC8392_PUBLIC_Y);
        out
    };
    let cose_key = Rfc9942Es256CoseKey::from_cbor(&cose_key_wire).unwrap();

    assert_eq!(
        receipt.verify_es256_cose_key_consistency(
            symthaea_swarm::semantic_evidence_vds::VdsTreeHead::new(1, [0x55; 32]),
            &cose_key,
            &[],
            None,
        ),
        Err(Rfc9942VdpError::InvalidEs256Signature)
    );
}

#[test]
fn rfc9942_outer_verification_binds_exact_payload_to_inner_inclusion() {
    fn bstr(bytes: &[u8]) -> Vec<u8> {
        assert!(bytes.len() < 24);
        let mut out = vec![0x40 | bytes.len() as u8];
        out.extend_from_slice(bytes);
        out
    }

    fn outer_wire(
        receipt: &Rfc9942ReceiptEnvelope,
        payload: &[u8],
        signature: &[u8],
        protect_receipts: bool,
    ) -> Vec<u8> {
        let collection = Rfc9942ReceiptCollection::new(vec![receipt.clone()]).unwrap().to_cbor();

        fn cbor_bstr(bytes: &[u8]) -> Vec<u8> {
            assert!(bytes.len() < 256);
            let mut out = if bytes.len() < 24 {
                vec![0x40 | bytes.len() as u8]
            } else {
                vec![0x58, bytes.len() as u8]
            };
            out.extend_from_slice(bytes);
            out
        }

        let mut protected_map = Vec::new();
        protected_map.push(if protect_receipts { 0xa2 } else { 0xa1 });
        protected_map.extend_from_slice(&[0x01, 0x26]); // { alg: -7 }
        if protect_receipts {
            protected_map.extend_from_slice(&[0x19, 0x01, 0x8a]);
            protected_map.extend_from_slice(&collection);
        }

        let mut unprotected = Vec::new();
        if protect_receipts {
            unprotected.push(0xa0);
        } else {
            unprotected.push(0xa1);
            unprotected.extend_from_slice(&[0x19, 0x01, 0x8a]);
            unprotected.extend_from_slice(&collection);
        }

        let mut out = Vec::new();
        out.extend_from_slice(&[0xd2, 0x84]);
        out.extend_from_slice(&cbor_bstr(&protected_map));
        out.extend_from_slice(&unprotected);
        out.extend_from_slice(&cbor_bstr(payload));
        out.extend_from_slice(&[0x58, signature.len() as u8]);
        out.extend_from_slice(signature);
        out
    }

    fn signed_outer(
        receipt: &Rfc9942ReceiptEnvelope,
        payload: &[u8],
        protect_receipts: bool,
    ) -> Rfc9942SignatureWithReceipts {
        let unsigned_wire = outer_wire(receipt, payload, &[0u8; 64], protect_receipts);
        let unsigned = Rfc9942SignatureWithReceipts::from_cbor(&unsigned_wire).unwrap();
        let rng = SystemRandom::new();
        let signing_key = EcdsaKeyPair::from_private_key_and_public_key(
            &ring::signature::ECDSA_P256_SHA256_FIXED_SIGNING,
            &RFC8392_PRIVATE_D,
            &rfc8392_public_key(),
            &rng,
        )
        .unwrap();
        let tbs = unsigned.signature1_tbs(&[], None).unwrap();
        let signature = signing_key.sign(&rng, &tbs).unwrap().as_ref().to_vec();
        Rfc9942SignatureWithReceipts::from_cbor(&outer_wire(
            receipt,
            payload,
            &signature,
            protect_receipts,
        ))
        .unwrap()
    }

    fn signed_receipt(candidate: &[u8]) -> Rfc9942ReceiptEnvelope {
        let leaves = vec![candidate.to_vec(), b"other-entry".to_vec()];
        let vds = Rfc9162Sha256Vds;
        let head = vds.tree_head(&leaves);
        let proof = vds.inclusion_proof(&leaves, 0).unwrap().to_cbor();
        let vdp = Rfc9942Vdp::new(Rfc9942ProofKind::Inclusion, vec![proof]).unwrap();

        let unsigned = Rfc9942ReceiptEnvelope::new(
            COSE_ES256_ALGORITHM_ID,
            vdp.clone(),
            Rfc9942ReceiptPayload::Attached(head.root()),
            vec![0u8; 64],
        )
        .unwrap();

        let rng = SystemRandom::new();
        let signing_key = EcdsaKeyPair::from_private_key_and_public_key(
            &ring::signature::ECDSA_P256_SHA256_FIXED_SIGNING,
            &RFC8392_PRIVATE_D,
            &rfc8392_public_key(),
            &rng,
        )
        .unwrap();
        let tbs = unsigned.signature1_tbs(&[], None).unwrap();
        let signature = signing_key.sign(&rng, &tbs).unwrap().as_ref().to_vec();

        Rfc9942ReceiptEnvelope::new(
            COSE_ES256_ALGORITHM_ID,
            vdp,
            Rfc9942ReceiptPayload::Attached(head.root()),
            signature,
        )
        .unwrap()
    }

    let receipt = signed_receipt(b"candidate");
    let key = rfc8392_public_key();

    let detached_receipt = {
        let candidate = b"candidate";
        let leaves = vec![candidate.to_vec(), b"other-entry".to_vec()];
        let vds = Rfc9162Sha256Vds;
        let head = vds.tree_head(&leaves);
        let proof = vds.inclusion_proof(&leaves, 0).unwrap().to_cbor();
        let vdp = Rfc9942Vdp::new(Rfc9942ProofKind::Inclusion, vec![proof]).unwrap();
        let unsigned = Rfc9942ReceiptEnvelope::new(
            COSE_ES256_ALGORITHM_ID,
            vdp.clone(),
            Rfc9942ReceiptPayload::Detached,
            vec![0u8; 64],
        ).unwrap();
        let rng = SystemRandom::new();
        let signer = EcdsaKeyPair::from_private_key_and_public_key(
            &ring::signature::ECDSA_P256_SHA256_FIXED_SIGNING,
            &RFC8392_PRIVATE_D,
            &key,
            &rng,
        ).unwrap();
        let root = head.root();
        let tbs = unsigned.signature1_tbs(&[], Some(&root)).unwrap();
        let signature = signer.sign(&rng, &tbs).unwrap().as_ref().to_vec();
        Rfc9942ReceiptEnvelope::new(
            COSE_ES256_ALGORITHM_ID,
            vdp,
            Rfc9942ReceiptPayload::Detached,
            signature,
        ).unwrap()
    };

    let detached_outer = signed_outer(&detached_receipt, b"candidate", false);
    let detached_state = detached_outer
        .verify_es256_inclusion_receipt_state(0, &key, &key, &[], &[], None)
        .unwrap();
    assert_eq!(
        detached_state.receipt().proof().inclusion_head().unwrap().tree_size(),
        2
    );

    let valid_outer = signed_outer(&receipt, b"candidate", false);
    let state = valid_outer
        .verify_es256_inclusion_receipt_state(
            0, &key, &key, &[], &[], None,
        )
        .unwrap();

    assert_eq!(state.outer_algorithm_id(), COSE_ES256_ALGORITHM_ID);
    assert_eq!(state.receipt_index(), 0);
    assert_eq!(
        state.receipt_placement(),
        symthaea_swarm::semantic_evidence_vds::Rfc9942ReceiptPlacement::Unprotected
    );
    let mut candidate_digest = Vec::new();
    candidate_digest.extend_from_slice(b"candidate");
    let expected_payload_sha256: [u8; 32] = sha2::Sha256::digest(&candidate_digest).into();
    assert_eq!(state.outer_payload_sha256(), expected_payload_sha256);

    // Sign the outer object over different payload bytes while retaining the
    // same valid inner Receipt. The outer signature is valid, but the Receipt
    // no longer proves the exact outer payload. The combined verifier must fail.
    let mismatched_outer = signed_outer(&receipt, b"different");
    mismatched_outer
        .verify_es256(&key, &[], None)
        .expect("outer signature over the mismatched payload is still valid");
    assert_eq!(
        mismatched_outer.verify_es256_inclusion_receipt_state(
            0, &key, &key, &[], &[], None,
        ),
        Err(Rfc9942VdpError::NoMatchingProof)
    );

    // The same proof must also work when the RFC 9942 receipts parameter is
    // carried in the protected header. This placement is covered by the outer
    // COSE Sig_structure, so replacing it invalidates the outer signature.
    let valid_protected = signed_outer(&receipt, b"candidate", true);
    let protected_state = valid_protected
        .verify_es256_inclusion_receipt_state(
            0, &key, &key, &[], &[], None,
        )
        .unwrap();
    assert_eq!(
        protected_state.receipt_placement(),
        symthaea_swarm::semantic_evidence_vds::Rfc9942ReceiptPlacement::Protected
    );

    let alternate_receipt = signed_receipt(b"candidate", b"alternate-tree-entry");
    let replaced_protected = Rfc9942SignatureWithReceipts::from_cbor(&outer_wire(
        &alternate_receipt,
        b"candidate",
        valid_protected.signature(),
        true,
    ))
    .unwrap();
    assert_eq!(
        replaced_protected.verify_es256_inclusion_receipt_state(
            0, &key, &key, &[], &[], None,
        ),
        Err(Rfc9942VdpError::InvalidEs256Signature)
    );

    // Selection semantics are explicit and deterministic: there is no implicit
    // fallback to a different receipt, and absence is a distinct failure.
    assert_eq!(
        valid_outer.verify_es256_inclusion_receipt_state(
            1, &key, &key, &[], &[], None,
        ),
        Err(Rfc9942VdpError::ReceiptIndexOutOfBounds)
    );
    let without_receipts = Rfc9942SignatureWithReceipts::new(
        symthaea_swarm::semantic_evidence_vds::Rfc9942SignaturePayload::Attached(
            b"candidate".to_vec(),
        ),
        vec![0u8; 64],
        None,
    );
    assert_eq!(
        without_receipts.verify_es256_inclusion_receipt_state(
            0, &key, &key, &[], &[], None,
        ),
        Err(Rfc9942VdpError::ReceiptsMissing)
    );
}


#[test]
fn detached_inclusion_state_derives_and_binds_root() {
    let candidate = b"detached-candidate";
    let leaves = vec![candidate.to_vec(), b"other-entry".to_vec()];
    let vds = Rfc9162Sha256Vds;
    let head = vds.tree_head(&leaves);
    let proof = vds.inclusion_proof(&leaves, 0).unwrap().to_cbor();
    let vdp = Rfc9942Vdp::new(Rfc9942ProofKind::Inclusion, vec![proof]).unwrap();
    let unsigned = Rfc9942ReceiptEnvelope::new(
        COSE_ES256_ALGORITHM_ID,
        vdp.clone(),
        Rfc9942ReceiptPayload::Detached,
        vec![0u8; 64],
    ).unwrap();

    let key = rfc8392_public_key();
    let rng = SystemRandom::new();
    let signer = EcdsaKeyPair::from_private_key_and_public_key(
        &ring::signature::ECDSA_P256_SHA256_FIXED_SIGNING,
        &RFC8392_PRIVATE_D,
        &key,
        &rng,
    ).unwrap();
    let root = head.root();
    let tbs = unsigned.signature1_tbs(&[], Some(&root)).unwrap();
    let sig = signer.sign(&rng, &tbs).unwrap().as_ref().to_vec();
    let receipt = Rfc9942ReceiptEnvelope::new(
        COSE_ES256_ALGORITHM_ID,
        vdp,
        Rfc9942ReceiptPayload::Detached,
        sig,
    ).unwrap();

    assert_eq!(
        receipt.verify_es256_inclusion_state(candidate, &key, &[], None)
            .unwrap().proof().inclusion_head(),
        Some(head)
    );
    assert!(
        receipt.verify_es256_inclusion_state(candidate, &key, &[], Some(&root)).is_ok()
    );
    assert_eq!(
        receipt.verify_es256_inclusion_state(candidate, &key, &[], Some(&[0xA5; 32])),
        Err(Rfc9942VdpError::NoMatchingProof)
    );

    // The legacy head-returning API now shares the proof-derived detached path:
    // no externally supplied Merkle root is needed.
    assert_eq!(
        receipt.verify_es256_inclusion(candidate, &key, &[], None),
        Ok(head)
    );
}

#[test]
fn rfc9942_verified_state_records_selected_proof_index() {
    let candidate = b"candidate";
    let leaves = vec![candidate.to_vec(), b"other-entry".to_vec()];
    let alternate = vec![b"alternate".to_vec(), b"tree-entry".to_vec()];
    let vds = Rfc9162Sha256Vds;

    let wrong_proof = vds.inclusion_proof(&alternate, 0).unwrap().to_cbor();
    let matching_proof = vds.inclusion_proof(&leaves, 0).unwrap().to_cbor();
    let head = vds.tree_head(&leaves);
    let vdp = Rfc9942Vdp::new(
        Rfc9942ProofKind::Inclusion,
        vec![wrong_proof, matching_proof],
    )
    .unwrap();

    let unsigned = Rfc9942ReceiptEnvelope::new(
        COSE_ES256_ALGORITHM_ID,
        vdp.clone(),
        Rfc9942ReceiptPayload::Attached(head.root()),
        vec![0u8; 64],
    )
    .unwrap();

    let key = rfc8392_public_key();
    let rng = SystemRandom::new();
    let signer = EcdsaKeyPair::from_private_key_and_public_key(
        &ring::signature::ECDSA_P256_SHA256_FIXED_SIGNING,
        &RFC8392_PRIVATE_D,
        &key,
        &rng,
    )
    .unwrap();
    let tbs = unsigned.signature1_tbs(&[], None).unwrap();
    let signature = signer.sign(&rng, &tbs).unwrap().as_ref().to_vec();

    let receipt = Rfc9942ReceiptEnvelope::new(
        COSE_ES256_ALGORITHM_ID,
        vdp,
        Rfc9942ReceiptPayload::Attached(head.root()),
        signature,
    )
    .unwrap();

    let state = receipt
        .verify_es256_inclusion_state(candidate, &key, &[], None)
        .unwrap();
    assert_eq!(state.proof().proof_index(), 1);
    assert_eq!(state.proof().inclusion_head(), Some(head));
}
