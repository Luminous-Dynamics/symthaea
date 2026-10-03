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
