//! Focused RFC 9942 semantic qualification.
//!
//! This is an integration test on purpose: it exercises the public RFC 9942
//! boundary without compiling the crate's unrelated #[cfg(test)] modules.

#![cfg(feature = "semantic-receipts")]

use symthaea_swarm::semantic_evidence_vds::{
    Rfc9942ProofKind, Rfc9942ReceiptEnvelope, Rfc9942ReceiptPayload, Rfc9942SignatureWithReceipts,
    Rfc9162ConsistencyProof, Rfc9162InclusionProof, Rfc9942VdpError, Rfc9942Vdp,
    Rfc9942ReceiptCollection, COSE_ES256_ALGORITHM_ID,
};

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

