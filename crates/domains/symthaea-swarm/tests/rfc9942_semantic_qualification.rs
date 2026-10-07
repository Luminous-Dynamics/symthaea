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
    MAX_RFC9942_COSE_KEY_ENCODED_BYTES,
};
use ring::{rand::SystemRandom, signature::{EcdsaKeyPair, KeyPair}};
use sha2::Digest;

const RFC8392_PUBLIC_X: [u8; 32] = [
    0x14, 0x33, 0x29, 0xcc, 0xe7, 0x86, 0x8e, 0x41,
    0x69, 0x27, 0x59, 0x9c, 0xf6, 0x5a, 0x34, 0xf3,
    0xce, 0x2f, 0xfd, 0xa5, 0x5a, 0x7e, 0xca, 0x69,
    0xed, 0x89, 0x19, 0xa3, 0x94, 0xd4, 0x2f, 0x0f,
];
const RFC8392_PUBLIC_Y: [u8; 32] = [
    0x60, 0xf7, 0xf1, 0xa7, 0x80, 0xd8, 0xa7, 0x83,
    0xbf, 0xb7, 0xa2, 0xdd, 0x6b, 0x27, 0x96, 0xe8,
    0x12, 0x8d, 0xbb, 0xce, 0xf9, 0xd3, 0xd1, 0x68,
    0xdb, 0x95, 0x29, 0x97, 0x1a, 0x36, 0xe7, 0xb9,
];

fn rfc8392_public_key() -> [u8; 65] {
    let mut key = [0u8; 65];
    key[0] = 0x04;
    key[1..33].copy_from_slice(&RFC8392_PUBLIC_X);
    key[33..65].copy_from_slice(&RFC8392_PUBLIC_Y);
    key
}

const RFC8392_P256_PKCS8: &[u8] = &[
    0x30, 0x81, 0x87, 0x02, 0x01, 0x00, 0x30, 0x13,
    0x06, 0x07, 0x2a, 0x86, 0x48, 0xce, 0x3d, 0x02,
    0x01, 0x06, 0x08, 0x2a, 0x86, 0x48, 0xce, 0x3d,
    0x03, 0x01, 0x07, 0x04, 0x6d, 0x30, 0x6b, 0x02,
    0x01, 0x01, 0x04, 0x20, 0x6c, 0x13, 0x82, 0x76,
    0x5a, 0xec, 0x53, 0x58, 0xf1, 0x17, 0x73, 0x3d,
    0x28, 0x1c, 0x1c, 0x7b, 0xdc, 0x39, 0x88, 0x4d,
    0x04, 0xa4, 0x5a, 0x1e, 0x6c, 0x67, 0xc8, 0x58,
    0xbc, 0x20, 0x6c, 0x19, 0xa1, 0x44, 0x03, 0x42,
    0x00, 0x04, 0x14, 0x33, 0x29, 0xcc, 0xe7, 0x86,
    0x8e, 0x41, 0x69, 0x27, 0x59, 0x9c, 0xf6, 0x5a,
    0x34, 0xf3, 0xce, 0x2f, 0xfd, 0xa5, 0x5a, 0x7e,
    0xca, 0x69, 0xed, 0x89, 0x19, 0xa3, 0x94, 0xd4,
    0x2f, 0x0f, 0x60, 0xf7, 0xf1, 0xa7, 0x80, 0xd8,
    0xa7, 0x83, 0xbf, 0xb7, 0xa2, 0xdd, 0x6b, 0x27,
    0x96, 0xe8, 0x12, 0x8d, 0xbb, 0xce, 0xf9, 0xd3,
    0xd1, 0x68, 0xdb, 0x95, 0x29, 0x97, 0x1a, 0x36,
    0xe7, 0xb9,
];

fn rfc8392_signing_key(rng: &SystemRandom) -> EcdsaKeyPair {
    EcdsaKeyPair::from_pkcs8(
        &ring::signature::ECDSA_P256_SHA256_FIXED_SIGNING,
        RFC8392_P256_PKCS8,
        rng,
    )
    .expect("RFC 8392 P-256 PKCS#8 key must parse")
}

#[test]
fn rfc8392_fixture_binds_private_scalar_to_public_point() {
    let rng = SystemRandom::new();
    let signing_key = rfc8392_signing_key(&rng);
    assert_eq!(signing_key.public_key().as_ref(), &rfc8392_public_key());
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
            Rfc9942ReceiptPayload::Detached,
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
            Some(&[0x44; 32]),
        ),
        Err(Rfc9942VdpError::InvalidEs256Signature)
    );
    assert_eq!(
        consistency.verify_es256_consistency_state(
            symthaea_swarm::semantic_evidence_vds::VdsTreeHead::new(1, [0x55; 32]),
            &public_key,
            &[],
            Some(&[0x44; 32]),
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
fn rfc9942_unprotected_text_label_resource_limit_is_typed() {
    let mut wire = Vec::new();
    wire.extend_from_slice(&[0xd2, 0x84]); // COSE_Sign1
    wire.extend_from_slice(&[0x41, 0xa0]); // protected = {}
    wire.extend_from_slice(&[0xa1]); // one-entry unprotected map
    wire.push(0x79); // tstr, two-byte length
    wire.extend_from_slice(&257u16.to_be_bytes());
    wire.extend(std::iter::repeat_n(b'x', 257));
    wire.push(0x00); // arbitrary well-formed CBOR value
    wire.extend_from_slice(&[0xf6, 0x40]); // detached payload, empty signature

    assert_eq!(
        Rfc9942SignatureWithReceipts::from_cbor(&wire),
        Err(Rfc9942VdpError::ResourceLimitExceeded)
    );
}

#[test]
fn rfc9942_inclusion_path_semantic_bound_is_distinct_from_resource_cap() {
    let mut semantic_oversize = vec![0x83, 0x02, 0x00, 0x82];
    semantic_oversize.extend_from_slice(&[0x58, 0x20]);
    semantic_oversize.extend_from_slice(&[0u8; 32]);
    semantic_oversize.extend_from_slice(&[0x58, 0x20]);
    semantic_oversize.extend_from_slice(&[0u8; 32]);

    let mut semantic_vdp = vec![0xa1, 0x20, 0x81, 0x59,
        (semantic_oversize.len() >> 8) as u8, semantic_oversize.len() as u8];
    semantic_vdp.extend_from_slice(&semantic_oversize);
    assert!(matches!(
        Rfc9942Vdp::from_cbor(&semantic_vdp),
        Err(Rfc9942VdpError::InvalidProof(_))
    ));

    let mut resource_oversize = vec![0x83, 0x02, 0x00, 0x81, 0x5f];
    resource_oversize.extend(std::iter::repeat_n(0x40, 4097));
    resource_oversize.push(0xff);

    let len = resource_oversize.len();
    let mut resource_vdp = vec![0xa1, 0x20, 0x81, 0x59,
        (len >> 8) as u8, len as u8];
    resource_vdp.extend_from_slice(&resource_oversize);
    assert_eq!(
        Rfc9942Vdp::from_cbor(&resource_vdp),
        Err(Rfc9942VdpError::ResourceLimitExceeded)
    );
}

#[test]
fn rfc9942_proof_hash_chunk_resource_limit_is_typed() {
    for (proof_prefix, proof_label, label) in [
        (vec![0x83, 0x02, 0x00, 0x81], 0x20, "inclusion"),
        (vec![0x83, 0x01, 0x02, 0x81], 0x21, "consistency"),
    ] {
        let mut proof = proof_prefix;
        proof.push(0x5f);
        proof.extend(std::iter::repeat_n(0x40, 4097));
        proof.push(0xff);

        let len = proof.len();
        assert!(len <= u16::MAX as usize);

        let mut encoded = vec![0xa1];
        encoded.extend_from_slice(&[proof_label, 0x81, 0x59, (len >> 8) as u8, len as u8]);
        encoded.extend_from_slice(&proof);

        let result = Rfc9942Vdp::from_cbor(&encoded);
        assert_eq!(
            result,
            Err(Rfc9942VdpError::ResourceLimitExceeded),
            "{label} proof hash chunk exhaustion must remain a resource failure"
        );
    }
}

#[test]
fn rfc9942_vdp_resource_limit_is_not_collapsed_into_encoding_error() {
    let mut encoded = vec![0xa1, 0x20];
    encoded.extend(std::iter::repeat_n(0xc0, 17));
    encoded.push(0x01);

    // The outer VDP scanner hits its recursion ceiling before interpreting
    // the VDP value. That is a resource admission failure, not malformed CBOR.
    assert_eq!(
        Rfc9942Vdp::from_cbor(&encoded),
        Err(Rfc9942VdpError::ResourceLimitExceeded)
    );
}

#[test]
fn rfc9942_vdp_array_count_resource_limit_is_typed() {
    let encoded = {
        let mut value = vec![0xa1, 0x20, 0x99, 0x01, 0x01];
        value.extend(std::iter::repeat_n(0x40, 257));
        value
    };

    assert_eq!(
        Rfc9942Vdp::from_cbor(&encoded),
        Err(Rfc9942VdpError::ResourceLimitExceeded)
    );
}

#[test]
fn rfc9942_nested_receipt_resource_limit_is_preserved() {
    let proof = [
        0x83, 0x02, 0x00, 0x81,
        0x58, 0x20,
    ];
    let mut proof_bytes = proof.to_vec();
    proof_bytes.extend_from_slice(&[0u8; 32]);

    let mut vdp = vec![0xa1, 0x20, 0x81, 0x59,
        (proof_bytes.len() >> 8) as u8, proof_bytes.len() as u8];
    vdp.extend_from_slice(&proof_bytes);

    let mut nested_receipt = vec![0xd2, 0x84];
    nested_receipt.extend_from_slice(&[0x47, 0xa2, 0x01, 0x26, 0x19, 0x01, 0x8b, 0x01]);
    nested_receipt.extend_from_slice(&[0xa1, 0x19, 0x01, 0x8c]);
    nested_receipt.extend_from_slice(&vdp);
    nested_receipt.extend_from_slice(&[0xf6, 0x5a, 0x00, 0x01, 0x00, 0x01]);
    nested_receipt.extend_from_slice(&[0u8; 65537]);

    let nested_len = nested_receipt.len();
    assert!(nested_len <= u32::MAX as usize);

    let mut outer = vec![0xd2, 0x84, 0x41, 0xa0];
    outer.extend_from_slice(&[0xa1, 0x19, 0x01, 0x8a, 0x81, 0x5a]);
    outer.extend_from_slice(&(nested_len as u32).to_be_bytes());
    outer.extend_from_slice(&nested_receipt);
    outer.extend_from_slice(&[0xf6, 0x58, 0x40]);
    outer.extend_from_slice(&[0u8; 64]);

    assert_eq!(
        Rfc9942SignatureWithReceipts::from_cbor(&outer),
        Err(Rfc9942VdpError::ReceiptCollectionResourceLimitExceeded)
    );
}
#[test]
fn rfc9942_receipt_count_resource_limit_is_typed() {
    let mut encoded = vec![
        0xd2, 0x84, // COSE_Sign1
        0x41, 0xa0, // protected = {}
        0xa1, 0x19, 0x01, 0x8a, 0x91, // receipts (394) = [ ... ] x 17
    ];
    encoded.extend(std::iter::repeat_n(0x40, 17));
    encoded.extend_from_slice(&[0xf6, 0x40]); // detached payload, empty signature

    assert_eq!(
        Rfc9942SignatureWithReceipts::from_cbor(&encoded),
        Err(Rfc9942VdpError::ReceiptCollectionResourceLimitExceeded)
    );
}

#[test]
fn rfc9942_indefinite_bstr_chunk_limit_is_typed() {
    let mut encoded = vec![
        0xd2, 0x84, // COSE_Sign1
        0x41, 0xa0, // protected = {}
        0xa1, 0x18, 0x1e, 0x5f, // extension label 30 = indefinite bstr
    ];
    encoded.extend(std::iter::repeat_n(0x40, 4097));
    encoded.extend_from_slice(&[0xff, 0xf6, 0x40]); // break, detached payload, signature

    assert_eq!(
        Rfc9942SignatureWithReceipts::from_cbor(&encoded),
        Err(Rfc9942VdpError::ResourceLimitExceeded)
    );
}

#[test]
fn rfc9942_indefinite_tstr_chunk_limit_is_typed() {
    let mut encoded = vec![
        0xd2, 0x84, // COSE_Sign1
        0x41, 0xa0, // protected = {}
        0xa1, 0x18, 0x1e, 0x7f, // extension label 30 = indefinite tstr
    ];
    encoded.extend(std::iter::repeat_n(0x60, 4097));
    encoded.extend_from_slice(&[0xff, 0xf6, 0x40]); // break, detached payload, signature

    assert_eq!(
        Rfc9942SignatureWithReceipts::from_cbor(&encoded),
        Err(Rfc9942VdpError::ResourceLimitExceeded)
    );
}

#[test]
fn rfc9942_crit_rejects_unknown_critical_header() {
    // Protected = {2: [999], 999: {bstr(0): 0}}.
    // The extension itself is understood as opaque, but a critical extension
    // must be understood by this processing layer.
    let protected = [
        0xa2, 0x02, 0x81, 0x19, 0x03, 0xe7,
        0x19, 0x03, 0xe7, 0xa1, 0x41, 0x00, 0x00,
    ];
    let mut wire = vec![0xd2, 0x84, 0x4d];
    wire.extend_from_slice(&protected);
    wire.extend_from_slice(&[0xa0, 0xf6, 0x40, 0x40]);

    assert_eq!(
        Rfc9942SignatureWithReceipts::from_cbor(&wire),
        Err(Rfc9942VdpError::CriticalHeaderNotUnderstood)
    );
}


#[test]
fn rfc9942_crit_cannot_reference_unprotected_receipt_parameters() {
    // Build a structurally valid RFC 9942 Receipt for use in the outer
    // unprotected receipts bucket. The regression must reach crit processing;
    // an empty receipt array would correctly fail earlier as EmptyReceiptCollection.
    let proof = Rfc9162InclusionProof::new(2, 0, vec![[0u8; 32]]).to_cbor();
    let vdp = Rfc9942Vdp::new(Rfc9942ProofKind::Inclusion, vec![proof]).unwrap();
    let receipt = Rfc9942ReceiptEnvelope::new(
        COSE_ES256_ALGORITHM_ID,
        vdp.clone(),
        Rfc9942ReceiptPayload::Attached([0u8; 32]),
        vec![0u8; 64],
    )
    .unwrap();

    // Inner Receipt: crit names header 396 (vdp), but 396 is carried in the
    // unprotected bucket. That is a fatal COSE processing error.
    let protected = [
        0xa3, // { alg: -7, vds: 1, crit: [396] }
        0x01, 0x26,
        0x19, 0x01, 0x8b, 0x01,
        0x02, 0x81, 0x19, 0x01, 0x8c,
    ];
    let mut inner = vec![0xd2, 0x84, 0x4f];
    inner.extend_from_slice(&protected);
    inner.extend_from_slice(&[
        0xa1, 0x19, 0x01, 0x8c, // vdp
    ]);
    inner.extend_from_slice(&vdp.to_cbor());
    inner.extend_from_slice(&[0x58, 0x20]);
    inner.extend_from_slice(&[0u8; 32]);
    inner.extend_from_slice(&[0x58, 0x40]);
    inner.extend_from_slice(&[0u8; 64]);

    assert_eq!(
        Rfc9942ReceiptEnvelope::from_cbor(&inner),
        Err(Rfc9942VdpError::CriticalHeaderNotProtected)
    );

    // Outer Signature_With_Receipt: crit names 394 while the actual receipts
    // parameter is in the unprotected bucket. The structurally valid receipt
    // ensures the failure is specifically the cross-bucket crit violation.
    let collection = Rfc9942ReceiptCollection::new(vec![receipt]).unwrap().to_cbor();
    let outer_protected = [
        0xa2, // { alg: -7, crit: [394] }
        0x01, 0x26,
        0x02, 0x81, 0x19, 0x01, 0x8a,
    ];
    let mut outer = vec![0xd2, 0x84, 0x4f];
    outer.extend_from_slice(&outer_protected);
    outer.extend_from_slice(&[0xa1, 0x19, 0x01, 0x8a]);
    outer.extend_from_slice(&collection);
    outer.extend_from_slice(&[0x58, 0x20]);
    outer.extend_from_slice(&[0u8; 32]);
    outer.extend_from_slice(&[0x58, 0x40]);
    outer.extend_from_slice(&[0u8; 64]);

    assert_eq!(
        Rfc9942SignatureWithReceipts::from_cbor(&outer),
        Err(Rfc9942VdpError::CriticalHeaderNotProtected)
    );
}

#[test]
fn rfc9942_crit_rejects_unprotected_critical_header() {
    // Unprotected = {2: [999], 999: null}; crit may not be unprotected.
    let unprotected = [0xa2, 0x02, 0x81, 0x19, 0x03, 0xe7, 0x19, 0x03, 0xe7, 0xf6];

    let mut wire = vec![0xd2, 0x84, 0x41, 0xa0];
    wire.extend_from_slice(&unprotected);
    wire.push(0xf6);
    wire.extend_from_slice(&[0x40, 0x40]);

    assert_eq!(
        Rfc9942SignatureWithReceipts::from_cbor(&wire),
        Err(Rfc9942VdpError::CriticalHeaderNotProtected)
    );
}

#[test]
fn rfc9942_crit_rejects_duplicate_critical_labels() {
    // Protected = {2: [1, 1], 1: -7}.
    let protected = [0xa2, 0x02, 0x82, 0x01, 0x01, 0x01, 0x26];
    let mut wire = vec![0xd2, 0x84, 0x47];
    wire.extend_from_slice(&protected);
    wire.extend_from_slice(&[0xa0, 0xf6, 0x40, 0x40]);

    assert_eq!(
        Rfc9942SignatureWithReceipts::from_cbor(&wire),
        Err(Rfc9942VdpError::InvalidStructure)
    );
}

#[test]
fn cose_extension_values_accept_well_formed_simple_items_and_round_trip_exactly() {
    for value in [&[0xe0][..], &[0xf3][..], &[0xf8, 0x20][..]] {
        let mut protected = Vec::new();
        protected.extend_from_slice(&[0xa1, 0x19, 0x03, 0xe7]);
        protected.extend_from_slice(value);

        let mut encoded = Vec::new();
        encoded.extend_from_slice(&[0xd2, 0x84]);
        encoded.push(0x40 | protected.len() as u8);
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
        0xa1, 0x19, 0x01, 0x8c, 0xa1, 0x20, 0x81, 0x44, 0x83, 0x01, 0x00, 0x80,
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
        // Protected map = { 999 => value }; total serialized map size is 13 bytes.
        let mut protected = Vec::with_capacity(14);
        protected.extend_from_slice(&[0xa1, 0x19, 0x03, 0xe7]);
        protected.extend_from_slice(&value);

        let mut encoded = Vec::new();
        encoded.extend_from_slice(&[0xd2, 0x84, 0x4d]);
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
        let signing_key = rfc8392_signing_key(&rng);
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
    assert_eq!(
        state.payload_mode(),
        symthaea_swarm::semantic_evidence_vds::Rfc9942PayloadMode::Attached
    );
    let expected_receipt_payload_sha256: [u8; 32] =
        sha2::Sha256::digest(receipt.payload().attached_root().unwrap()).into();
    assert_eq!(state.payload_sha256(), expected_receipt_payload_sha256);
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
    let signing_key = rfc8392_signing_key(&rng);
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
        state.payload_mode(),
        symthaea_swarm::semantic_evidence_vds::Rfc9942PayloadMode::Detached
    );
    let expected_newer_root_sha256: [u8; 32] = sha2::Sha256::digest(root).into();
    assert_eq!(state.payload_sha256(), expected_newer_root_sha256);
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
fn rfc9942_ed25519_consistency_preserves_signature_first_order() {
    let proof =
        Rfc9162ConsistencyProof::new(1, 2, vec![[0x33; 32]]).to_cbor();
    let vdp = Rfc9942Vdp::new(Rfc9942ProofKind::Consistency, vec![proof]).unwrap();
    let receipt = Rfc9942ReceiptEnvelope::new(
        -8,
        vdp,
        Rfc9942ReceiptPayload::Detached,
        vec![0xBB; 63],
    )
    .unwrap();

    // Both the detached signature and consistency proof are invalid. The
    // semantic verifier must report signature failure before proof failure.
    assert_eq!(
        receipt.verify_ed25519_consistency(
            symthaea_swarm::semantic_evidence_vds::VdsTreeHead::new(1, [0x55; 32]),
            &[0u8; 32],
            &[],
            Some(&[0x44; 32]),
        ),
        Err(Rfc9942VdpError::InvalidEd25519Signature)
    );
}

#[test]
fn rfc9942_consistency_semantic_paths_reject_attached_payloads() {
    let proof =
        Rfc9162ConsistencyProof::new(1, 2, vec![[0x33; 32]]).to_cbor();
    let vdp = Rfc9942Vdp::new(Rfc9942ProofKind::Consistency, vec![proof]).unwrap();
    let older =
        symthaea_swarm::semantic_evidence_vds::VdsTreeHead::new(1, [0x55; 32]);

    let es256 = Rfc9942ReceiptEnvelope::new(
        COSE_ES256_ALGORITHM_ID,
        vdp.clone(),
        Rfc9942ReceiptPayload::Attached([0x44; 32]),
        vec![0xBB; 64],
    )
    .unwrap();
    assert_eq!(
        es256.verify_es256_consistency_state(older, &[0x04; 65], &[], None),
        Err(Rfc9942VdpError::InvalidStructure)
    );

    let ed25519 = Rfc9942ReceiptEnvelope::new(
        -8,
        vdp,
        Rfc9942ReceiptPayload::Attached([0x44; 32]),
        vec![0xBB; 64],
    )
    .unwrap();
    assert_eq!(
        ed25519.verify_ed25519_consistency(older, &[0u8; 32], &[], None),
        Err(Rfc9942VdpError::InvalidStructure)
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
    let signer = rfc8392_signing_key(&rng);
    let root = head.root();
    let tbs = unsigned.signature1_tbs(&[], Some(&root)).unwrap();
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
fn rfc9942_outer_detached_payload_binds_inner_inclusion_and_outer_signature() {
    let candidate = b"detached-outer-candidate";
    let leaves = vec![candidate.to_vec(), b"other-entry".to_vec()];
    let vds = Rfc9162Sha256Vds;
    let head = vds.tree_head(&leaves);
    let proof = vds.inclusion_proof(&leaves, 0).unwrap().to_cbor();
    let vdp = Rfc9942Vdp::new(Rfc9942ProofKind::Inclusion, vec![proof]).unwrap();

    // The inner RFC 9942 Receipt is detached: its signature authenticates the
    // proof-derived Merkle root, not an in-receipt payload.
    let unsigned_receipt = Rfc9942ReceiptEnvelope::new(
        COSE_ES256_ALGORITHM_ID,
        vdp.clone(),
        Rfc9942ReceiptPayload::Detached,
        vec![0u8; 64],
    )
    .unwrap();
    let key = rfc8392_public_key();
    let rng = SystemRandom::new();
    let signer = rfc8392_signing_key(&rng);
    let inner_tbs = unsigned_receipt.signature1_tbs(&[], Some(&head.root())).unwrap();
    let inner_signature = signer.sign(&rng, &inner_tbs).unwrap().as_ref().to_vec();
    let receipt = Rfc9942ReceiptEnvelope::new(
        COSE_ES256_ALGORITHM_ID,
        vdp,
        Rfc9942ReceiptPayload::Detached,
        inner_signature,
    )
    .unwrap();

    let collection = Rfc9942ReceiptCollection::new(vec![receipt]).unwrap().to_cbor();

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

    fn outer_wire(collection: &[u8], signature: &[u8]) -> Vec<u8> {
        let protected = [0xa1, 0x01, 0x26]; // { alg: -7 }
        let mut unprotected = Vec::new();
        unprotected.push(0xa1);
        unprotected.extend_from_slice(&[0x19, 0x01, 0x8a]); // receipts: 394
        unprotected.extend_from_slice(collection);

        let mut out = Vec::new();
        out.extend_from_slice(&[0xd2, 0x84]);
        out.extend_from_slice(&cbor_bstr(&protected));
        out.extend_from_slice(&unprotected);
        out.push(0xf6); // detached outer application payload
        out.extend_from_slice(&cbor_bstr(signature));
        out
    }

    let unsigned_outer = Rfc9942SignatureWithReceipts::from_cbor(
        &outer_wire(&collection, &[0u8; 64]),
    )
    .unwrap();
    let outer_tbs = unsigned_outer.signature1_tbs(&[], Some(candidate)).unwrap();
    let outer_signature = signer.sign(&rng, &outer_tbs).unwrap().as_ref().to_vec();
    let outer = Rfc9942SignatureWithReceipts::from_cbor(&outer_wire(
        &collection,
        &outer_signature,
    ))
    .unwrap();

    let state = outer
        .verify_es256_inclusion_receipt_state(
            0,
            &key,
            &key,
            &[],
            &[],
            Some(candidate),
        )
        .unwrap();
    assert_eq!(state.outer_algorithm_id(), COSE_ES256_ALGORITHM_ID);
    assert_eq!(
        state.receipt_placement(),
        symthaea_swarm::semantic_evidence_vds::Rfc9942ReceiptPlacement::Unprotected
    );
    assert_eq!(state.receipt_index(), 0);
    assert_eq!(
        state.receipt().proof().inclusion_head(),
        Some(head)
    );

    // The inner proof may derive the correct root, but the outer signature must
    // still authenticate the exact detached application payload.
    assert_eq!(
        outer.verify_es256_inclusion_receipt_state(
            0,
            &key,
            &key,
            &[],
            &[],
            Some(b"wrong-application-payload"),
        ),
        Err(Rfc9942VdpError::InvalidEs256Signature)
    );

    // Separately tamper the outer signature while keeping the detached payload
    // and inner Receipt unchanged. This must fail at the outer COSE boundary.
    let mut tampered = outer.to_cbor();
    let last = tampered.len() - 1;
    tampered[last] ^= 0x01;
    let tampered_outer = Rfc9942SignatureWithReceipts::from_cbor(&tampered).unwrap();
    assert_eq!(
        tampered_outer.verify_es256_inclusion_receipt_state(
            0,
            &key,
            &key,
            &[],
            &[],
            Some(candidate),
        ),
        Err(Rfc9942VdpError::InvalidEs256Signature)
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
        Rfc9942ReceiptPayload::Detached,
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
            Some(&[0x44; 32]),
        ),
        Err(Rfc9942VdpError::InvalidEs256Signature)
    );
}

#[test]
fn rfc9942_outer_verification_binds_exact_payload_to_inner_inclusion() {
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

    fn state_header_bytes(outer: &Rfc9942SignatureWithReceipts) -> Vec<u8> {
        outer.protected_header_bytes()
    }

    fn signed_outer(
        receipt: &Rfc9942ReceiptEnvelope,
        payload: &[u8],
        protect_receipts: bool,
    ) -> Rfc9942SignatureWithReceipts {
        let unsigned_wire = outer_wire(receipt, payload, &[0u8; 64], protect_receipts);
        let unsigned = Rfc9942SignatureWithReceipts::from_cbor(&unsigned_wire).unwrap();
        let rng = SystemRandom::new();
        let signing_key = rfc8392_signing_key(&rng);
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
        signed_receipt_with_other(candidate, b"other-entry")
    }

    fn signed_receipt_with_other(candidate: &[u8], other: &[u8]) -> Rfc9942ReceiptEnvelope {
        let leaves = vec![candidate.to_vec(), other.to_vec()];
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
        let signing_key = rfc8392_signing_key(&rng);
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
        let signer = rfc8392_signing_key(&rng);
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
        detached_state.outer_payload_mode(),
        symthaea_swarm::semantic_evidence_vds::Rfc9942PayloadMode::Attached
    );
    assert_eq!(
        detached_state.receipt().payload_mode(),
        symthaea_swarm::semantic_evidence_vds::Rfc9942PayloadMode::Detached
    );
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
    assert_eq!(
        state.outer_payload_mode(),
        symthaea_swarm::semantic_evidence_vds::Rfc9942PayloadMode::Attached
    );
    let expected_key_fingerprint: [u8; 32] = sha2::Sha256::digest(&key).into();
    let expected_empty_aad_fingerprint: [u8; 32] = sha2::Sha256::digest(&[]).into();
    let expected_outer_header_fingerprint: [u8; 32] =
        sha2::Sha256::digest(&state_header_bytes(&valid_outer)).into();
    let expected_receipt_header_fingerprint: [u8; 32] =
        sha2::Sha256::digest(&receipt.protected_header_bytes()).into();
    let expected_receipt_payload_sha256: [u8; 32] =
        sha2::Sha256::digest(receipt.payload().attached_root().unwrap()).into();
    assert_eq!(state.outer_verification_key_sha256(), expected_key_fingerprint);
    assert_eq!(
        state.outer_protected_header_sha256(),
        expected_outer_header_fingerprint
    );
    assert_eq!(state.outer_external_aad_sha256(), expected_empty_aad_fingerprint);
    assert_eq!(
        state.receipt().verification_key_sha256(),
        expected_key_fingerprint
    );
    assert_eq!(
        state.receipt().protected_header_sha256(),
        expected_receipt_header_fingerprint
    );
    assert_eq!(
        state.receipt().payload_sha256(),
        expected_receipt_payload_sha256
    );
    assert_eq!(
        state.receipt().external_aad_sha256(),
        expected_empty_aad_fingerprint
    );
    assert_eq!(state.receipt_index(), 0);
    assert_eq!(
        state.receipt_placement(),
        symthaea_swarm::semantic_evidence_vds::Rfc9942ReceiptPlacement::Unprotected
    );
    assert_eq!(state.receipt_index(), 0);
    let mut candidate_digest = Vec::new();
    candidate_digest.extend_from_slice(b"candidate");
    let expected_payload_sha256: [u8; 32] = sha2::Sha256::digest(&candidate_digest).into();
    assert_eq!(state.outer_payload_sha256(), expected_payload_sha256);

    // Sign the outer object over different payload bytes while retaining the
    // same valid inner Receipt. The outer signature is valid, but the Receipt
    // no longer proves the exact outer payload. The combined verifier must fail.
    let mismatched_outer = signed_outer(&receipt, b"different", false);
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
    assert_eq!(
        protected_state.receipt().protected_header_sha256(),
        expected_receipt_header_fingerprint
    );
    assert_ne!(
        state.outer_protected_header_sha256(),
        protected_state.outer_protected_header_sha256(),
        "moving receipts into the protected bucket must change the authenticated header provenance"
    );
    assert_eq!(protected_state.receipt_index(), 0);

    let alternate_receipt = signed_receipt_with_other(b"candidate", b"alternate-tree-entry");
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
fn rfc9942_detached_outer_payload_mode_is_preserved() {
    let receipt = signed_receipt(b"candidate");
    let collection = Rfc9942ReceiptCollection::new(vec![receipt]).unwrap();
    let key = rfc8392_public_key();
    let rng = SystemRandom::new();

    // The outer payload is detached, while the inner inclusion Receipt keeps
    // an attached Merkle root. The combined capability must preserve both
    // transport modes independently.
    let unsigned = Rfc9942SignatureWithReceipts::new(
        symthaea_swarm::semantic_evidence_vds::Rfc9942SignaturePayload::Detached,
        vec![0u8; 64],
        Some(collection.clone()),
    );
    let tbs = unsigned.signature1_tbs(&[], Some(b"candidate")).unwrap();
    let signer = rfc8392_signing_key(&rng);
    let signature = signer.sign(&rng, &tbs).unwrap().as_ref().to_vec();
    let outer = Rfc9942SignatureWithReceipts::new(
        symthaea_swarm::semantic_evidence_vds::Rfc9942SignaturePayload::Detached,
        signature,
        Some(collection),
    );

    let state = outer
        .verify_es256_inclusion_receipt_state(
            0,
            &key,
            &key,
            &[],
            &[],
            Some(b"candidate"),
        )
        .unwrap();

    assert_eq!(
        state.outer_payload_mode(),
        symthaea_swarm::semantic_evidence_vds::Rfc9942PayloadMode::Detached
    );
    assert_eq!(
        state.receipt().payload_mode(),
        symthaea_swarm::semantic_evidence_vds::Rfc9942PayloadMode::Attached
    );
    let expected_outer_payload_sha256: [u8; 32] =
        sha2::Sha256::digest(b"candidate").into();
    assert_eq!(state.outer_payload_sha256(), expected_outer_payload_sha256);
}

#[test]
fn rfc9942_outer_selection_errors_precede_authentication() {
    let empty = Rfc9942SignatureWithReceipts::new(
        symthaea_swarm::semantic_evidence_vds::Rfc9942SignaturePayload::Attached(
            b"candidate".to_vec(),
        ),
        vec![0u8; 64],
        None,
    );
    let key = rfc8392_public_key();

    assert_eq!(
        empty.verify_es256_inclusion_receipt_state(
            0, &key, &key, &[], &[], None,
        ),
        Err(Rfc9942VdpError::ReceiptsMissing)
    );

    // Selection is structural and deterministic. A nonexistent selected
    // receipt must report its index error without performing cryptographic
    // work, even when the supplied outer signature is invalid.
    let receipts = Rfc9942ReceiptCollection::new(vec![
        Rfc9942ReceiptEnvelope::new(
            COSE_ES256_ALGORITHM_ID,
            Rfc9942Vdp::new(
                Rfc9942ProofKind::Inclusion,
                vec![Rfc9162InclusionProof::new(2, 0, vec![[0u8; 32]]).to_cbor()],
            ).unwrap(),
            Rfc9942ReceiptPayload::Attached([0u8; 32]),
            vec![0u8; 64],
        ).unwrap(),
    ]).unwrap();

    let outer = Rfc9942SignatureWithReceipts::new(
        symthaea_swarm::semantic_evidence_vds::Rfc9942SignaturePayload::Attached(
            b"candidate".to_vec(),
        ),
        vec![0u8; 64],
        Some(receipts),
    );

    assert_eq!(
        outer.verify_es256_inclusion_receipt_state(
            1, &key, &key, &[], &[], None,
        ),
        Err(Rfc9942VdpError::ReceiptIndexOutOfBounds)
    );
}

#[test]
fn rfc9942_outer_signature_failure_short_circuits_inner_proof_work() {
    let candidate = b"candidate";
    let leaves = vec![candidate.to_vec(), b"other-entry".to_vec()];
    let vds = Rfc9162Sha256Vds;
    let head = vds.tree_head(&leaves);

    // Build an inner Receipt whose signature is valid for its bytes but whose
    // inclusion proof is deliberately wrong. This makes the inner proof path
    // independently fail while leaving its cryptographic signature valid.
    let mut invalid_proof = vds.inclusion_proof(&leaves, 0).unwrap().to_cbor();
    let last_proof_byte = invalid_proof.len() - 1;
    invalid_proof[last_proof_byte] ^= 0x01;
    let vdp = Rfc9942Vdp::new(Rfc9942ProofKind::Inclusion, vec![invalid_proof]).unwrap();

    let unsigned_receipt = Rfc9942ReceiptEnvelope::new(
        COSE_ES256_ALGORITHM_ID,
        vdp.clone(),
        Rfc9942ReceiptPayload::Attached(head.root()),
        vec![0u8; 64],
    )
    .unwrap();
    let rng = SystemRandom::new();
    let signer = rfc8392_signing_key(&rng);
    let receipt_tbs = unsigned_receipt.signature1_tbs(&[], None).unwrap();
    let receipt_signature = signer.sign(&rng, &receipt_tbs).unwrap().as_ref().to_vec();
    let invalid_receipt = Rfc9942ReceiptEnvelope::new(
        COSE_ES256_ALGORITHM_ID,
        vdp,
        Rfc9942ReceiptPayload::Attached(head.root()),
        receipt_signature,
    )
    .unwrap();

    let key = rfc8392_public_key();
    assert_eq!(
        invalid_receipt.verify_es256_inclusion(candidate, &key, &[], None),
        Err(Rfc9942VdpError::NoMatchingProof)
    );

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

    fn signed_outer(
        receipt: &Rfc9942ReceiptEnvelope,
        payload: &[u8],
    ) -> Rfc9942SignatureWithReceipts {
        let collection =
            Rfc9942ReceiptCollection::new(vec![receipt.clone()]).unwrap().to_cbor();
        let protected = [0xa1, 0x01, 0x26]; // { alg: -7 }
        let mut unprotected = Vec::new();
        unprotected.extend_from_slice(&[0xa1, 0x19, 0x01, 0x8a]);
        unprotected.extend_from_slice(&collection);

        let mut unsigned_wire = Vec::new();
        unsigned_wire.extend_from_slice(&[0xd2, 0x84]);
        unsigned_wire.extend_from_slice(&cbor_bstr(&protected));
        unsigned_wire.extend_from_slice(&unprotected);
        unsigned_wire.extend_from_slice(&cbor_bstr(payload));
        unsigned_wire.extend_from_slice(&[0x58, 0x40]);
        unsigned_wire.extend_from_slice(&[0u8; 64]);

        let unsigned =
            Rfc9942SignatureWithReceipts::from_cbor(&unsigned_wire).unwrap();
        let rng = SystemRandom::new();
        let signer = rfc8392_signing_key(&rng);
        let tbs = unsigned.signature1_tbs(&[], None).unwrap();
        let signature = signer.sign(&rng, &tbs).unwrap().as_ref().to_vec();

        let mut signed_wire = Vec::new();
        signed_wire.extend_from_slice(&[0xd2, 0x84]);
        signed_wire.extend_from_slice(&cbor_bstr(&protected));
        signed_wire.extend_from_slice(&unprotected);
        signed_wire.extend_from_slice(&cbor_bstr(payload));
        signed_wire.extend_from_slice(&cbor_bstr(&signature));
        Rfc9942SignatureWithReceipts::from_cbor(&signed_wire).unwrap()
    }

    // The outer signature is independently valid before tampering.
    let signed_outer = signed_outer(&invalid_receipt, candidate);
    signed_outer
        .verify_es256(&key, &[], None)
        .expect("outer signature must initially verify");

    // Now invalidate only the outer signature. The combined verifier should
    // reject at the authenticated outer boundary before touching the invalid
    // inner proof, rather than exposing the inner proof failure as precedence.
    let mut wire = signed_outer.to_cbor();
    let last_signature_byte = wire.len() - 1;
    wire[last_signature_byte] ^= 0x01;
    let both_invalid = Rfc9942SignatureWithReceipts::from_cbor(&wire).unwrap();

    assert_eq!(
        both_invalid.verify_es256_inclusion_receipt_state(
            0, &key, &key, &[], &[], None,
        ),
        Err(Rfc9942VdpError::InvalidEs256Signature)
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
    let signer = rfc8392_signing_key(&rng);
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
    let signer = rfc8392_signing_key(&rng);
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



#[test]
fn rfc9942_verified_inclusion_state_preserves_duplicate_leaf_position() {
    let candidate = b"duplicate";
    let leaves = vec![
        candidate.to_vec(),
        b"middle".to_vec(),
        candidate.to_vec(),
    ];
    let vds = Rfc9162Sha256Vds;
    let head = vds.tree_head(&leaves);
    let proof_at_zero = vds.inclusion_proof(&leaves, 0).unwrap().to_cbor();
    let proof_at_two = vds.inclusion_proof(&leaves, 2).unwrap().to_cbor();

    // The candidate bytes are identical at positions 0 and 2 and therefore
    // have the same candidate-leaf hash. The proof itself carries the log
    // position; verified state must preserve that position instead of
    // collapsing the two semantic locations into one leaf identity.
    let vdp_both = Rfc9942Vdp::new(
        Rfc9942ProofKind::Inclusion,
        vec![proof_at_zero.clone(), proof_at_two.clone()],
    )
    .unwrap();
    let unsigned = Rfc9942ReceiptEnvelope::new(
        COSE_ES256_ALGORITHM_ID,
        vdp_both.clone(),
        Rfc9942ReceiptPayload::Attached(head.root()),
        vec![0u8; 64],
    )
    .unwrap();
    let rng = SystemRandom::new();
    let signer = rfc8392_signing_key(&rng);
    let signature = signer
        .sign(&rng, &unsigned.signature1_tbs(&[], None).unwrap())
        .unwrap()
        .as_ref()
        .to_vec();
    let receipt = Rfc9942ReceiptEnvelope::new(
        COSE_ES256_ALGORITHM_ID,
        vdp_both,
        Rfc9942ReceiptPayload::Attached(head.root()),
        signature,
    )
    .unwrap();
    let key = rfc8392_public_key();

    let first_state = receipt
        .verify_es256_inclusion_state(candidate, &key, &[], None)
        .unwrap();
    let mut candidate_leaf_input = Vec::with_capacity(1 + candidate.len());
    candidate_leaf_input.push(0x00);
    candidate_leaf_input.extend_from_slice(candidate);
    let expected_candidate_leaf: [u8; 32] =
        sha2::Sha256::digest(&candidate_leaf_input).into();
    assert_eq!(first_state.proof().proof_index(), 0);
    assert_eq!(first_state.proof().inclusion_leaf_index(), Some(0));
    assert_eq!(
        first_state.proof().inclusion_candidate_leaf(),
        Some(expected_candidate_leaf)
    );

    // With only the second valid proof available, the same candidate bytes
    // and same tree head establish a different verified log position.
    let vdp_second = Rfc9942Vdp::new(
        Rfc9942ProofKind::Inclusion,
        vec![proof_at_two],
    )
    .unwrap();
    let unsigned_second = Rfc9942ReceiptEnvelope::new(
        COSE_ES256_ALGORITHM_ID,
        vdp_second.clone(),
        Rfc9942ReceiptPayload::Attached(head.root()),
        vec![0u8; 64],
    )
    .unwrap();
    let signature_second = signer
        .sign(&rng, &unsigned_second.signature1_tbs(&[], None).unwrap())
        .unwrap()
        .as_ref()
        .to_vec();
    let receipt_second = Rfc9942ReceiptEnvelope::new(
        COSE_ES256_ALGORITHM_ID,
        vdp_second,
        Rfc9942ReceiptPayload::Attached(head.root()),
        signature_second,
    )
    .unwrap();

    let second_state = receipt_second
        .verify_es256_inclusion_state(candidate, &key, &[], None)
        .unwrap();
    assert_eq!(second_state.proof().proof_index(), 0);
    assert_eq!(second_state.proof().inclusion_leaf_index(), Some(2));
    assert_eq!(
        second_state.proof().inclusion_candidate_leaf(),
        Some(expected_candidate_leaf)
    );
}

#[test]
fn rfc9942_consistency_state_records_selected_proof_index() {
    let leaves = vec![
        b"old-a".to_vec(),
        b"old-b".to_vec(),
        b"new-c".to_vec(),
        b"new-d".to_vec(),
    ];
    let vds = Rfc9162Sha256Vds;
    let older = vds.tree_head(&leaves[..2].to_vec());
    let newer = vds.tree_head(&leaves);

    // Both proofs describe the same old/new tree sizes. The first is kept
    // structurally valid but is cryptographically invalid; the second is the
    // valid proof. The semantic capability must report which proof actually
    // established the verified state rather than merely returning the first
    // matching-looking entry.
    let mut invalid_proof = vds.prove(&leaves, 2).unwrap().to_cbor();
    let last_hash_byte = invalid_proof.len() - 1;
    invalid_proof[last_hash_byte] ^= 0x01;
    let valid_proof = vds.prove(&leaves, 2).unwrap().to_cbor();
    let vdp = Rfc9942Vdp::new(
        Rfc9942ProofKind::Consistency,
        vec![invalid_proof, valid_proof],
    )
    .unwrap();

    let unsigned = Rfc9942ReceiptEnvelope::new(
        COSE_ES256_ALGORITHM_ID,
        vdp.clone(),
        Rfc9942ReceiptPayload::Detached,
        vec![0u8; 64],
    )
    .unwrap();

    let rng = SystemRandom::new();
    let signing_key = rfc8392_signing_key(&rng);
    let signature = signing_key
        .sign(&rng, &unsigned.signature1_tbs(&[], Some(&newer.root())).unwrap())
        .unwrap()
        .as_ref()
        .to_vec();

    let receipt = Rfc9942ReceiptEnvelope::new(
        COSE_ES256_ALGORITHM_ID,
        vdp,
        Rfc9942ReceiptPayload::Detached,
        signature,
    )
    .unwrap();
    let key = rfc8392_public_key();

    let state = receipt
        .verify_es256_consistency_state(older, &key, &[], Some(&newer.root()))
        .unwrap();

    assert_eq!(state.proof().proof_index(), 1);
    assert_eq!(state.proof().consistency_heads(), Some((older, newer)));

    let expected_payload_sha256: [u8; 32] = sha2::Sha256::digest(&newer.root()).into();
    assert_eq!(state.payload_sha256(), expected_payload_sha256);
}

#[test]
fn rfc9942_round_trip_preserves_unprotected_header_entry_order() {
    fn bstr(bytes: &[u8]) -> Vec<u8> {
        assert!(bytes.len() < 256);
        let mut out = if bytes.len() < 24 {
            vec![0x40 | bytes.len() as u8]
        } else {
            vec![0x58, bytes.len() as u8]
        };
        out.extend_from_slice(bytes);
        out
    }

    let proof = Rfc9162InclusionProof::new(2, 0, vec![[0x11; 32]]).to_cbor();
    let vdp = Rfc9942Vdp::new(Rfc9942ProofKind::Inclusion, vec![proof]).unwrap();
    let receipt = Rfc9942ReceiptEnvelope::new(
        COSE_ES256_ALGORITHM_ID,
        vdp,
        Rfc9942ReceiptPayload::Detached,
        vec![0xAA; 64],
    )
    .unwrap();
    let receipt_wire = {
        let vdp_bytes = receipt.vdp().to_cbor();
        let mut protected = Vec::new();
        protected.extend_from_slice(&[0xa2, 0x01, 0x26, 0x19, 0x01, 0x8b, 0x01]);
        let mut out = vec![0xd2, 0x84];
        out.extend_from_slice(&bstr(&protected));
        out.extend_from_slice(&[
            0xa2,
            0x19, 0x23, 0x28, 0x01,
            0x19, 0x01, 0x8c,
        ]);
        out.extend_from_slice(&vdp_bytes);
        out.extend_from_slice(&[0xf6, 0x58, 0x40]);
        out.extend_from_slice(&[0xAA; 64]);
        out
    };

    // The unrelated extension (9000: 1) intentionally precedes vdp (396).
    let parsed_receipt = Rfc9942ReceiptEnvelope::from_cbor(&receipt_wire).unwrap();
    assert_eq!(parsed_receipt.to_cbor(), receipt_wire);

    let collection = Rfc9942ReceiptCollection::new(vec![parsed_receipt]).unwrap().to_cbor();
    let outer_wire = {
        let protected = [0xa1, 0x01, 0x26];
        let mut unprotected = vec![
            0xa2,
            0x19, 0x23, 0x28, 0x01,
            0x19, 0x01, 0x8a,
        ];
        unprotected.extend_from_slice(&collection);
        let mut out = vec![0xd2, 0x84];
        out.extend_from_slice(&bstr(&protected));
        out.extend_from_slice(&unprotected);
        out.extend_from_slice(&[0x47, b'p', b'a', b'y', b'l', b'o', b'a', b'd']);
        out.extend_from_slice(&[0x58, 0x40]);
        out.extend_from_slice(&[0xBB; 64]);
        out
    };

    let parsed_outer = Rfc9942SignatureWithReceipts::from_cbor(&outer_wire).unwrap();
    assert_eq!(parsed_outer.to_cbor(), outer_wire);
}

#[test]
fn rfc9942_es256_binds_external_aad() {
    let proof = Rfc9162InclusionProof::new(2, 0, vec![[0x11; 32]]).to_cbor();
    let vdp = Rfc9942Vdp::new(Rfc9942ProofKind::Inclusion, vec![proof]).unwrap();
    let payload_root = [0x22; 32];

    let unsigned = Rfc9942ReceiptEnvelope::new(
        COSE_ES256_ALGORITHM_ID,
        vdp.clone(),
        Rfc9942ReceiptPayload::Attached(payload_root),
        vec![0u8; 64],
    )
    .unwrap();

    let rng = SystemRandom::new();
    let signer = rfc8392_signing_key(&rng);
    let aad = b"qualification-aad";
    let tbs = unsigned.signature1_tbs(aad, None).unwrap();
    let signature = signer.sign(&rng, &tbs).unwrap().as_ref().to_vec();

    let receipt = Rfc9942ReceiptEnvelope::new(
        COSE_ES256_ALGORITHM_ID,
        vdp,
        Rfc9942ReceiptPayload::Attached(payload_root),
        signature,
    )
    .unwrap();
    let key = rfc8392_public_key();

    receipt.verify_es256(&key, aad, None).unwrap();
    assert_eq!(
        receipt.verify_es256(&key, b"wrong-aad", None),
        Err(Rfc9942VdpError::InvalidEs256Signature)
    );
}


#[test]
fn rfc9942_vdp_malformed_map_shape_is_not_resource_exhaustion() {
    let encoded = [
        0xa2, // Two VDP members are structurally invalid but below the scan cap.
        0x20, 0x81, 0x40,
        0x21, 0x81, 0x40,
    ];

    assert_eq!(
        Rfc9942Vdp::from_cbor(&encoded),
        Err(Rfc9942VdpError::InvalidStructure)
    );
}

#[test]
fn rfc9942_vdp_map_resource_limit_remains_typed() {
    let mut encoded = vec![0xb8, 0x21]; // 33 map members > the 32-member scan cap.
    encoded.extend(std::iter::repeat_n(0x00, 66));

    assert_eq!(
        Rfc9942Vdp::from_cbor(&encoded),
        Err(Rfc9942VdpError::ResourceLimitExceeded)
    );
}


#[test]
fn rfc9942_protected_header_map_entry_resource_limit_is_typed() {
    let mut protected = vec![0xb8, 0x21];
    for label in 100..133u8 {
        protected.extend_from_slice(&[0x18, label, 0x00]);
    }
    assert_eq!(protected.len(), 101);

    let mut receipt = vec![0xd2, 0x84, 0x58, protected.len() as u8];
    receipt.extend_from_slice(&protected);
    receipt.extend_from_slice(&[0xa0, 0xf6, 0x58, 0x40]);
    receipt.extend_from_slice(&[0u8; 64]);

    assert_eq!(
        Rfc9942ReceiptEnvelope::from_cbor(&receipt),
        Err(Rfc9942VdpError::ResourceLimitExceeded)
    );

    let mut outer = vec![0xd2, 0x84, 0x58, protected.len() as u8];
    outer.extend_from_slice(&protected);
    outer.extend_from_slice(&[0xa0, 0xf6, 0x58, 0x40]);
    outer.extend_from_slice(&[0u8; 64]);

    assert_eq!(
        Rfc9942SignatureWithReceipts::from_cbor(&outer),
        Err(Rfc9942VdpError::ResourceLimitExceeded)
    );
}

#[test]
fn rfc9942_receipt_payload_chunk_resource_limit_is_typed() {
    let proof = {
        let mut value = vec![0x83, 0x02, 0x00, 0x81, 0x58, 0x20];
        value.extend_from_slice(&[0u8; 32]);
        value
    };
    let mut vdp = vec![0xa1, 0x20, 0x81, 0x58, proof.len() as u8];
    vdp.extend_from_slice(&proof);

    let mut encoded = vec![0xd2, 0x84, 0x47, 0xa2, 0x01, 0x26, 0x19, 0x01, 0x8b, 0x01];
    encoded.extend_from_slice(&[0xa1, 0x19, 0x01, 0x8c]);
    encoded.extend_from_slice(&vdp);
    encoded.push(0x5f);
    encoded.extend(std::iter::repeat_n(0x40, 4097));
    encoded.extend_from_slice(&[0xff, 0x58, 0x40]);
    encoded.extend_from_slice(&[0u8; 64]);

    assert_eq!(
        Rfc9942ReceiptEnvelope::from_cbor(&encoded),
        Err(Rfc9942VdpError::ResourceLimitExceeded)
    );
}

#[test]
fn rfc9942_receipt_protected_header_resource_limit_is_typed() {
    let mut encoded = vec![0xd2, 0x84, 0x59, 0x10, 0x01];
    encoded.extend(std::iter::repeat_n(0x00, 4097));

    assert_eq!(
        Rfc9942ReceiptEnvelope::from_cbor(&encoded),
        Err(Rfc9942VdpError::ResourceLimitExceeded)
    );
}

#[test]
fn rfc9942_outer_protected_header_resource_limit_is_typed() {
    let mut encoded = vec![0xd2, 0x84, 0x59, 0x10, 0x01];
    encoded.extend(std::iter::repeat_n(0x00, 4097));

    assert_eq!(
        Rfc9942SignatureWithReceipts::from_cbor(&encoded),
        Err(Rfc9942VdpError::ResourceLimitExceeded)
    );
}

#[test]
fn rfc9942_receipt_signature_resource_limit_is_typed() {
    // Protected = { alg: -7, vds: 1 }, with a structurally valid inclusion VDP.
    let protected = [0xa2, 0x01, 0x26, 0x19, 0x01, 0x8b, 0x01];
    let proof = [
        0x83, 0x02, 0x00, 0x81, // tree_size=2, leaf_index=0, one hash
        0x58, 0x20,
    ];
    let mut vdp = vec![0xa1, 0x20, 0x81, 0x58, (proof.len() + 32) as u8];
    vdp.extend_from_slice(&proof);
    vdp.extend_from_slice(&[0u8; 32]);
    let mut encoded = vec![0xd2, 0x84, 0x47];
    encoded.extend_from_slice(&protected);
    encoded.push(0xa1);
    encoded.extend_from_slice(&[0x19, 0x01, 0x8c]);
    encoded.extend_from_slice(&vdp);
    encoded.extend_from_slice(&[0xf6, 0x5a, 0x00, 0x01, 0x00, 0x01]);
    encoded.extend(std::iter::repeat_n(0x00, 65537));

    // The parser now reaches the signature field with all preceding structure
    // valid, so this result specifically proves the defensive signature limit.
    assert_eq!(
        Rfc9942ReceiptEnvelope::from_cbor(&encoded),
        Err(Rfc9942VdpError::ResourceLimitExceeded)
    );
}


#[test]
fn rfc9942_outer_signature_resource_limit_is_typed() {
    // Outer Signature_With_Receipt with a structurally valid four-element
    // COSE_Sign1 shape and a detached payload; only the signature bstr exceeds
    // the decoder's 64 KiB defensive bound.
    let mut encoded = vec![
        0xd2, 0x84,
        0x41, 0xa0, // protected = {}
        0xa0,       // unprotected = {}
        0xf6,       // detached payload
        0x5a, 0x00, 0x01, 0x00, 0x01,
    ];
    encoded.extend(std::iter::repeat_n(0x00, 65537));

    assert_eq!(
        Rfc9942SignatureWithReceipts::from_cbor(&encoded),
        Err(Rfc9942VdpError::ResourceLimitExceeded)
    );
}


#[test]
fn rfc9942_es256_key_key_ops_resource_limit_is_typed() {
    let mut encoded = vec![0xa1, 0x04, 0x98, 0x11];
    encoded.extend(std::iter::repeat_n(0x02, 17));

    assert_eq!(
        Rfc9942Es256CoseKey::from_cbor(&encoded),
        Err(Rfc9942VdpError::ResourceLimitExceeded)
    );
}


#[test]
fn rfc9942_es256_key_aggregate_map_below_cap_remains_semantic() {
    // Four opaque extension values are each individually within the 4 KiB
    // value bound and the complete map stays below the 16 KiB aggregate cap.
    // The map is semantically incomplete, so the failure must remain
    // InvalidEs256CoseKey rather than being mislabeled as resource exhaustion.
    let mut encoded = vec![0xa4];
    for label in 1000i64..1004 {
        encoded.extend_from_slice(&[0x19, (label >> 8) as u8, label as u8]);
        encoded.extend_from_slice(&[0x59, 0x0f, 0xf9]); // 4089-byte bstr
        encoded.extend(std::iter::repeat_n(0x00, 4089));
    }

    assert_eq!(encoded.len(), MAX_RFC9942_COSE_KEY_ENCODED_BYTES - 3);
    assert_eq!(
        Rfc9942Es256CoseKey::from_cbor(&encoded),
        Err(Rfc9942VdpError::InvalidEs256CoseKey)
    );
}

#[test]
fn rfc9942_es256_key_aggregate_map_resource_limit_is_typed() {
    // Each extension value is individually within the 4 KiB opaque-value
    // decoder bound, but four such values plus map framing exceed the 16 KiB
    // aggregate COSE_Key admission budget. This proves the aggregate budget
    // cannot be bypassed by splitting hostile material across entries.
    let mut encoded = vec![0xa4];
    for label in 1000i64..1004 {
        encoded.extend_from_slice(&[0x19, (label >> 8) as u8, label as u8]);
        encoded.extend_from_slice(&[0x59, 0x10, 0x00]);
        encoded.extend(std::iter::repeat_n(0x00, 4096));
    }

    assert!(
        encoded.len() > MAX_RFC9942_COSE_KEY_ENCODED_BYTES,
        "regression wire must exceed aggregate key budget"
    );
    assert_eq!(
        Rfc9942Es256CoseKey::from_cbor(&encoded),
        Err(Rfc9942VdpError::ResourceLimitExceeded)
    );
}

#[test]
fn rfc9942_receipt_crit_array_resource_limit_is_typed() {
    let mut protected = vec![0xa2, 0x01, 0x26, 0x02, 0x98, 0x11];
    for label in 0..17u8 { protected.push(label); }
    let mut encoded = vec![0xd2, 0x84, 0x58, protected.len() as u8];
    encoded.extend_from_slice(&protected);
    assert_eq!(Rfc9942ReceiptEnvelope::from_cbor(&encoded), Err(Rfc9942VdpError::ResourceLimitExceeded));
}

#[test]
fn rfc9942_outer_crit_array_resource_limit_is_typed() {
    let mut protected = vec![0xa2, 0x01, 0x26, 0x02, 0x98, 0x11];
    for label in 0..17u8 { protected.push(label); }
    let mut encoded = vec![0xd2, 0x84, 0x58, protected.len() as u8];
    encoded.extend_from_slice(&protected);
    assert_eq!(Rfc9942SignatureWithReceipts::from_cbor(&encoded), Err(Rfc9942VdpError::ResourceLimitExceeded));
}

#[test]
fn rfc9942_indefinite_crit_array_resource_limit_is_typed() {
    let mut protected = vec![0xa2, 0x01, 0x26, 0x02, 0x9f];
    for label in 0..17u8 {
        protected.push(label);
    }
    protected.push(0xff);

    for context in ["receipt", "outer"] {
        let mut encoded = vec![0xd2, 0x84, 0x58, protected.len() as u8];
        encoded.extend_from_slice(&protected);
        encoded.push(0xa0);
        encoded.push(0xf6);
        encoded.push(0x40);

        let result = if context == "receipt" {
            Rfc9942ReceiptEnvelope::from_cbor(&encoded).map(|_| ())
        } else {
            Rfc9942SignatureWithReceipts::from_cbor(&encoded).map(|_| ())
        };
        assert_eq!(
            result,
            Err(Rfc9942VdpError::ResourceLimitExceeded),
            "{context} indefinite crit array must preserve the resource boundary"
        );
    }
}

#[test]
fn rfc9942_protected_extension_recursion_resource_limit_is_typed() {
    let mut receipt_protected = vec![0xa3, 0x01, 0x26, 0x19, 0x01, 0x8b, 0x01, 0x18, 0x1e];
    receipt_protected.extend(std::iter::repeat_n(0xc0, 17));
    receipt_protected.push(0xf6);

    let mut receipt = vec![0xd2, 0x84, 0x58, receipt_protected.len() as u8];
    receipt.extend_from_slice(&receipt_protected);
    receipt.extend_from_slice(&[0xa0, 0xf6, 0x58, 0x40]);
    receipt.extend_from_slice(&[0u8; 64]);

    assert_eq!(
        Rfc9942ReceiptEnvelope::from_cbor(&receipt),
        Err(Rfc9942VdpError::ResourceLimitExceeded)
    );

    let mut outer_protected = vec![0xa2, 0x01, 0x26, 0x18, 0x1e];
    outer_protected.extend(std::iter::repeat_n(0xc0, 17));
    outer_protected.push(0xf6);

    let mut outer = vec![0xd2, 0x84, 0x58, outer_protected.len() as u8];
    outer.extend_from_slice(&outer_protected);
    outer.extend_from_slice(&[0xa0, 0xf6, 0x58, 0x40]);
    outer.extend_from_slice(&[0u8; 64]);

    assert_eq!(
        Rfc9942SignatureWithReceipts::from_cbor(&outer),
        Err(Rfc9942VdpError::ResourceLimitExceeded)
    );
}

#[test]
fn rfc9942_receipt_extension_recursion_resource_limit_is_typed() {
    let mut encoded = vec![0xd2, 0x84, 0x47, 0xa2, 0x01, 0x26, 0x19, 0x01, 0x8b, 0x01, 0xa1, 0x18, 0x1e];
    for _ in 0..17 { encoded.push(0xc0); }
    encoded.push(0xf6);
    assert_eq!(Rfc9942ReceiptEnvelope::from_cbor(&encoded), Err(Rfc9942VdpError::ResourceLimitExceeded));
}
