//! COSE_Key interoperability and rejection corpus for ES256.
//!
//! Positive coordinates are the public EC2 key from RFC 9052 Appendix C.7.1.
//! Negative cases exercise the COSE_Key semantic boundary independently of
//! signature generation.

#![cfg(feature = "semantic-receipts")]

use symthaea_swarm::semantic_evidence_vds::{
    Rfc9942Es256CoseKey, Rfc9942ReceiptEnvelope, Rfc9942ReceiptPayload,
    Rfc9942SignaturePayload, Rfc9942SignatureWithReceipts, Rfc9942Vdp,
    Rfc9942ProofKind, Rfc9942VdpError, Rfc9162InclusionProof,
    COSE_ES256_ALGORITHM_ID, MAX_CBOR_BSTR_CHUNKS, MAX_CBOR_TSTR_CHUNKS,
};

const X: [u8; 32] = [
    0x65, 0xed, 0xa5, 0xa1, 0x25, 0x77, 0xc2, 0xba,
    0xe8, 0x29, 0x43, 0x7f, 0xe3, 0x38, 0x70, 0x1a,
    0x10, 0xaa, 0xa3, 0x75, 0xe1, 0xbb, 0x5b, 0x5d,
    0xe1, 0x08, 0xde, 0x43, 0x90, 0x85, 0x51, 0xd1,
];
const Y: [u8; 32] = [
    0x1e, 0x52, 0xed, 0x75, 0x70, 0x11, 0x63, 0xf7,
    0xf9, 0xe4, 0x0d, 0xdf, 0x9f, 0x34, 0x1b, 0x3d,
    0xc9, 0xba, 0x86, 0x0a, 0xf7, 0xe0, 0xca, 0x7c,
    0xa7, 0xe9, 0xee, 0xcd, 0x00, 0x84, 0xd1, 0x9c,
];

fn bstr(bytes: &[u8]) -> Vec<u8> {
    let mut out = match bytes.len() {
        0..=23 => vec![0x40 + bytes.len() as u8],
        24..=255 => vec![0x58, bytes.len() as u8],
        _ => panic!("fixture bstr too large"),
    };
    out.extend_from_slice(bytes);
    out
}
fn uint_field(label: u8, value: u8) -> Vec<u8> {
    vec![label, value]
}
fn neg1_field(value: u8) -> Vec<u8> {
    vec![0x20, value]
}
fn bstr_field(label: u8, bytes: &[u8]) -> Vec<u8> {
    let mut out = vec![label];
    out.extend_from_slice(&bstr(bytes));
    out
}
fn valid_fields() -> Vec<Vec<u8>> {
    vec![
        uint_field(1, 2),
        bstr_field(2, b"rfc9052-c7.1"),
        vec![0x03, 0x26],
        vec![0x04, 0x81, 0x02],
        neg1_field(1),
        bstr_field(0x21, &X),
        bstr_field(0x22, &Y),
    ]
}
fn key(fields: &[Vec<u8>]) -> Vec<u8> {
    assert!(fields.len() < 24);
    let mut out = vec![0xa0 + fields.len() as u8];
    for field in fields {
        out.extend_from_slice(field);
    }
    out
}
fn valid_key() -> Vec<u8> {
    key(&valid_fields())
}

fn indefinite_map(fields: &[Vec<u8>], include_break: bool) -> Vec<u8> {
    let mut out = vec![0xbf];
    for field in fields {
        out.extend_from_slice(field);
    }
    if include_break {
        out.push(0xff);
    }
    out
}

fn cose_key_with_indefinite_root(extra_fields: usize, include_break: bool) -> Vec<u8> {
    let mut fields = valid_fields();
    for label in 0..extra_fields {
        let label = 5u8.checked_add(label as u8).expect("test label must fit");
        let mut key = match label {
            0..=23 => vec![label],
            _ => vec![0x18, label],
        };
        key.push(0x00);
        fields.push(key);
    }
    indefinite_map(&fields, include_break)
}

fn indefinite_bstr_with_exact_chunk_cap(bytes: &[u8]) -> Vec<u8> {
    assert!(MAX_CBOR_BSTR_CHUNKS >= 1);
    let mut out = vec![0x5f];
    for _ in 0..MAX_CBOR_BSTR_CHUNKS - 1 {
        out.push(0x40);
    }
    match bytes.len() {
        0..=23 => out.push(0x40 + bytes.len() as u8),
        24..=255 => out.extend_from_slice(&[0x58, bytes.len() as u8]),
        _ => panic!("test bstr too large"),
    }
    out.extend_from_slice(bytes);
    out.push(0xff);
    out
}

fn indefinite_text_with_exact_chunk_cap(bytes: &[u8]) -> Vec<u8> {
    assert!(MAX_CBOR_TSTR_CHUNKS >= 1);
    std::str::from_utf8(bytes).expect("test text must be UTF-8");
    let mut out = vec![0x7f];
    for _ in 0..MAX_CBOR_TSTR_CHUNKS - 1 {
        out.push(0x60);
    }
    assert!(bytes.len() <= 23);
    out.push(0x60 + bytes.len() as u8);
    out.extend_from_slice(bytes);
    out.push(0xff);
    out
}


fn oversized_inclusion_proof_wire() -> Vec<u8> {
    let mut out = vec![0x83, 0x1b];
    out.extend_from_slice(&u64::MAX.to_be_bytes());
    out.extend_from_slice(&[0x00, 0x98, 0x40]);
    for _ in 0..64 {
        out.push(0x5f);
        for _ in 0..32 {
            out.extend_from_slice(&[0x41, 0x00]);
        }
        out.push(0xff);
    }
    out
}

#[test]
fn rfc9942_vdp_accepts_proof_bstr_above_generic_skip_value_cap() {
    let proof = oversized_inclusion_proof_wire();
    assert!(proof.len() > 4096);

    let vdp = Rfc9942Vdp::new(Rfc9942ProofKind::Inclusion, vec![proof.clone()])
        .expect("RFC9162 inclusion proof remains structurally valid above the generic opaque-value cap");
    let wire = vdp.to_cbor();
    let parsed = Rfc9942Vdp::from_cbor(&wire)
        .expect("RFC9942 VDP proof bstr must use its protocol-specific 8 KiB bound");
    assert_eq!(parsed.proofs()[0].len(), proof.len());
}

#[test]
fn rfc9942_receipt_collection_accepts_receipt_bstr_above_generic_skip_value_cap() {
    let proof = Rfc9162InclusionProof::new(2, 0, vec![[0u8; 32]]).to_cbor();
    let vdp = Rfc9942Vdp::new(Rfc9942ProofKind::Inclusion, vec![proof]).unwrap();
    let receipt = Rfc9942ReceiptEnvelope::new(
        COSE_ES256_ALGORITHM_ID,
        vdp,
        Rfc9942ReceiptPayload::Attached([0u8; 32]),
        vec![0u8; 4097],
    ).unwrap();

    let encoded = receipt.to_cbor();
    assert!(encoded.len() > 4096);

    let mut wire = vec![0x9f];
    wire.extend_from_slice(&bstr(&encoded));
    wire.push(0xff);

    let parsed = Rfc9942ReceiptCollection::from_cbor(&wire)
        .expect("RFC9942 receipt bstr must use its protocol-specific 4 MiB bound");
    assert_eq!(parsed.receipts()[0].signature().len(), 4097);
}

#[test]
fn cose_key_accepts_indefinite_map_root() {
    let parsed = Rfc9942Es256CoseKey::from_cbor(&cose_key_with_indefinite_root(0, true))
        .expect("RFC 8949 indefinite map must be accepted at the COSE_Key root");
    assert_eq!(parsed.kid(), Some(b"rfc9052-c7.1".as_slice()));
    assert_eq!(&parsed.public_key_sec1()[1..33], &X);
}

#[test]
fn cose_key_accepts_exact_indefinite_map_entry_cap_before_break() {
    let extra = 32 - valid_fields().len();
    let parsed = Rfc9942Es256CoseKey::from_cbor(&cose_key_with_indefinite_root(extra, true))
        .expect("the break after exactly 32 map entries is valid");
    assert_eq!(&parsed.public_key_sec1()[33..65], &Y);
}

#[test]
fn cose_key_accepts_unknown_label_with_nested_opaque_cbor_value() {
    let mut fields = valid_fields();
    let mut field = vec![0x18, 30];
    field.extend_from_slice(&[0xa1, 0x41, 0x00, 0x01]);
    fields.push(field);

    let parsed = Rfc9942Es256CoseKey::from_cbor(&key(&fields))
        .expect("an unknown COSE label must consume its complete arbitrary CBOR value");
    assert_eq!(parsed.kid(), Some(b"rfc9052-c7.1".as_slice()));
}

fn receipt_with_unknown_extension(protected_extension: bool) -> Vec<u8> {
    let proof = Rfc9162InclusionProof::new(2, 0, vec![[0u8; 32]]).to_cbor();
    let vdp = Rfc9942Vdp::new(Rfc9942ProofKind::Inclusion, vec![proof])
        .expect("fixture VDP must be valid")
        .to_cbor();

    let mut protected = if protected_extension {
        vec![0xa3, 0x01, 0x26, 0x19, 0x01, 0x8b, 0x01]
    } else {
        vec![0xa2, 0x01, 0x26, 0x19, 0x01, 0x8b, 0x01]
    };
    if protected_extension {
        protected.extend_from_slice(&[0x18, 0x1e, 0xa1, 0x41, 0x00, 0x01]);
    }

    let mut unprotected = if protected_extension {
        vec![0xa1, 0x19, 0x01, 0x8c]
    } else {
        vec![0xa2, 0x19, 0x01, 0x8c]
    };
    unprotected.extend_from_slice(&vdp);
    if !protected_extension {
        unprotected.extend_from_slice(&[0x18, 0x1e, 0xa1, 0x41, 0x00, 0x01]);
    }

    let mut out = vec![0xd2, 0x84];
    out.extend_from_slice(&bstr(&protected));
    out.extend_from_slice(&unprotected);
    out.extend_from_slice(&bstr(&[0u8; 32]));
    out.extend_from_slice(&bstr(&[0u8; 64]));
    out
}

fn outer_with_unknown_extension(protected_extension: bool) -> Vec<u8> {
    let protected = if protected_extension {
        vec![0xa1, 0x18, 0x1e, 0xa1, 0x41, 0x00, 0x01]
    } else {
        vec![0xa0]
    };
    let unprotected = if protected_extension {
        vec![0xa0]
    } else {
        vec![0xa1, 0x18, 0x1e, 0xa1, 0x41, 0x00, 0x01]
    };

    let mut out = vec![0xd2, 0x84];
    out.extend_from_slice(&bstr(&protected));
    out.extend_from_slice(&unprotected);
    out.push(0xf6);
    out.extend_from_slice(&bstr(&[0u8; 64]));
    out
}

#[test]
fn receipt_accepts_unknown_protected_extension_and_round_trips() {
    let bytes = receipt_with_unknown_extension(true);
    let parsed = Rfc9942ReceiptEnvelope::from_cbor(&bytes)
        .expect("unknown protected COSE extension must consume its arbitrary CBOR value");
    assert_eq!(parsed.to_cbor(), bytes);
}

#[test]
fn receipt_accepts_unknown_unprotected_extension_and_round_trips() {
    let bytes = receipt_with_unknown_extension(false);
    let parsed = Rfc9942ReceiptEnvelope::from_cbor(&bytes)
        .expect("unknown unprotected COSE extension must consume its arbitrary CBOR value");
    assert_eq!(parsed.to_cbor(), bytes);
}

#[test]
fn outer_cose_accepts_unknown_protected_extension_and_round_trips() {
    let bytes = outer_with_unknown_extension(true);
    let parsed = Rfc9942SignatureWithReceipts::from_cbor(&bytes)
        .expect("unknown protected outer COSE extension must consume its arbitrary CBOR value");
    assert_eq!(parsed.to_cbor(), bytes);
    assert_eq!(parsed.payload(), &Rfc9942SignaturePayload::Detached);
}

#[test]
fn outer_cose_accepts_unknown_unprotected_extension_and_round_trips() {
    let bytes = outer_with_unknown_extension(false);
    let parsed = Rfc9942SignatureWithReceipts::from_cbor(&bytes)
        .expect("unknown unprotected outer COSE extension must consume its arbitrary CBOR value");
    assert_eq!(parsed.to_cbor(), bytes);
}

#[test]
fn cose_key_rejects_indefinite_map_entry_count_above_cap() {
    let extra = 33 - valid_fields().len();
    assert_eq!(
        Rfc9942Es256CoseKey::from_cbor(&cose_key_with_indefinite_root(extra, true)),
        Err(Rfc9942VdpError::InvalidEs256CoseKey)
    );
}

#[test]
fn cose_key_rejects_unterminated_exact_indefinite_map_entry_cap() {
    let extra = 32 - valid_fields().len();
    assert_eq!(
        Rfc9942Es256CoseKey::from_cbor(&cose_key_with_indefinite_root(extra, false)),
        Err(Rfc9942VdpError::InvalidEncoding)
    );
}



#[test]
fn cose_key_accepts_exact_indefinite_bstr_chunk_cap_before_break() {
    let mut fields = valid_fields();
    let mut encoded = vec![0x21];
    encoded.extend_from_slice(&indefinite_bstr_with_exact_chunk_cap(&X));
    fields[5] = encoded;

    let parsed = Rfc9942Es256CoseKey::from_cbor(&key(&fields))
        .expect("the break after exactly MAX_CBOR_BSTR_CHUNKS chunks is valid");
    assert_eq!(&parsed.public_key_sec1()[1..33], &X);
}

#[test]
fn cose_key_accepts_exact_indefinite_tstr_chunk_cap_before_break() {
    let mut fields = valid_fields();
    let mut encoded = vec![0x01];
    encoded.extend_from_slice(&indefinite_text_with_exact_chunk_cap(b"EC2"));
    fields[0] = encoded;

    let parsed = Rfc9942Es256CoseKey::from_cbor(&key(&fields))
        .expect("the break after exactly MAX_CBOR_TSTR_CHUNKS chunks is valid");
    assert_eq!(&parsed.public_key_sec1()[1..33], &X);
}

#[test]
fn cose_key_rejects_bstr_chunk_count_above_exact_cap() {
    let mut fields = valid_fields();
    let mut encoded = vec![0x21, 0x5f];
    for _ in 0..=MAX_CBOR_BSTR_CHUNKS {
        encoded.push(0x40);
    }
    encoded.push(0xff);
    fields[5] = encoded;
    assert_eq!(
        Rfc9942Es256CoseKey::from_cbor(&key(&fields)),
        Err(Rfc9942VdpError::InvalidEs256CoseKey)
    );
}

#[test]
fn cose_key_rejects_tstr_chunk_count_above_exact_cap() {
    let mut fields = valid_fields();
    let mut encoded = vec![0x01, 0x7f];
    for _ in 0..=MAX_CBOR_TSTR_CHUNKS {
        encoded.push(0x60);
    }
    encoded.push(0xff);
    fields[0] = encoded;
    assert_eq!(
        Rfc9942Es256CoseKey::from_cbor(&key(&fields)),
        Err(Rfc9942VdpError::InvalidEs256CoseKey)
    );
}

#[test]
fn rfc9162_inclusion_path_accepts_indefinite_32_byte_hash_bstr() {
    let mut proof = vec![0x83, 0x02, 0x00, 0x81, 0x5f];
    proof.push(0x50);
    proof.extend_from_slice(&[0x11; 16]);
    proof.push(0x50);
    proof.extend_from_slice(&[0x11; 16]);
    proof.push(0xff);

    let decoded = symthaea_swarm::semantic_evidence_vds::Rfc9162InclusionProof::from_cbor(&proof)
        .expect("an indefinite-length bstr remains a valid 32-byte hash value");
    assert_eq!(decoded.tree_size, 2);
    assert_eq!(decoded.leaf_index, 0);
    assert_eq!(decoded.inclusion_path, vec![[0x11; 32]]);
}

#[test]
fn rfc9052_public_ec2_key_parses_and_preserves_kid() {
    let parsed = Rfc9942Es256CoseKey::from_cbor(&valid_key()).expect("RFC 9052 public key");
    assert_eq!(parsed.kid(), Some(b"rfc9052-c7.1".as_slice()));
    assert_eq!(parsed.public_key_sec1()[0], 0x04);
    assert_eq!(&parsed.public_key_sec1()[1..33], &X);
    assert_eq!(&parsed.public_key_sec1()[33..65], &Y);
}

#[test]
fn cose_key_requires_kty_and_curve_coordinates() {
    let fields = valid_fields();
    assert_eq!(
        Rfc9942Es256CoseKey::from_cbor(&key(&fields[1..])),
        Err(Rfc9942VdpError::InvalidEs256CoseKey)
    );
    let fields = vec![
        fields[0].clone(), fields[1].clone(), fields[2].clone(),
        fields[3].clone(), fields[5].clone(), fields[6].clone(),
    ];
    assert_eq!(
        Rfc9942Es256CoseKey::from_cbor(&key(&fields)),
        Err(Rfc9942VdpError::InvalidEs256CoseKey)
    );
}

#[test]
fn cose_key_rejects_wrong_curve() {
    let mut bytes = valid_key();
    let pos = bytes.windows(2).position(|w| w == [0x20, 0x01]).unwrap();
    bytes[pos + 1] = 0x02; // P-384
    assert_eq!(
        Rfc9942Es256CoseKey::from_cbor(&bytes),
        Err(Rfc9942VdpError::InvalidEs256CoseKey)
    );
}

#[test]
fn cose_key_rejects_mismatched_algorithm() {
    let mut bytes = valid_key();
    let alg_pos = bytes.windows(2).position(|w| w == [0x03, 0x26]).unwrap();
    bytes[alg_pos + 1] = 0x38;
    bytes.insert(
        bytes.windows(2).position(|w| w == [0x03, 0x38]).unwrap() + 2,
        0x22,
    );
    assert_eq!(
        Rfc9942Es256CoseKey::from_cbor(&bytes),
        Err(Rfc9942VdpError::Es256CoseKeyAlgorithmMismatch)
    );
}

#[test]
fn cose_key_rejects_key_without_verify_operation() {
    let mut bytes = valid_key();
    let pos = bytes.windows(3).position(|w| w == [0x04, 0x81, 0x02]).unwrap();
    bytes[pos + 2] = 0x01; // sign
    assert_eq!(
        Rfc9942Es256CoseKey::from_cbor(&bytes),
        Err(Rfc9942VdpError::Es256CoseKeyOperationNotPermitted)
    );
}

#[test]
fn cose_key_accepts_unknown_textual_key_ops_alongside_verify() {
    let mut fields = valid_fields();
    fields[3] = vec![0x04, 0x82, 0x02, 0x63, b'f', b'o', b'o'];
    let parsed = Rfc9942Es256CoseKey::from_cbor(&key(&fields))
        .expect("unknown textual key operation is extensible when verify is present");
    assert_eq!(parsed.public_key_sec1()[0], 0x04);
}

#[test]
fn cose_key_rejects_private_d_material() {
    let mut bytes = valid_key();
    bytes[0] = 0xa8;
    bytes.extend_from_slice(&[0x23, 0x58, 0x20]);
    bytes.extend_from_slice(&[0x42; 32]);
    assert_eq!(
        Rfc9942Es256CoseKey::from_cbor(&bytes),
        Err(Rfc9942VdpError::Es256PrivateKeyMaterial)
    );
}

#[test]
fn cose_key_rejects_wrong_coordinate_length() {
    let mut bytes = valid_key();
    let pos = bytes.windows(3).position(|w| w == [0x21, 0x58, 0x20]).unwrap();
    bytes[pos + 3] = 0x1f;
    bytes.remove(pos + 4);
    assert_eq!(
        Rfc9942Es256CoseKey::from_cbor(&bytes),
        Err(Rfc9942VdpError::InvalidEncoding)
    );
}

#[test]
fn cose_key_rejects_truncated_coordinate_bstr() {
    let mut bytes = valid_key();
    let pos = bytes.windows(3).position(|w| w == [0x21, 0x58, 0x20]).unwrap();
    bytes.truncate(pos + 3 + 10);
    assert_eq!(
        Rfc9942Es256CoseKey::from_cbor(&bytes),
        Err(Rfc9942VdpError::InvalidEncoding)
    );
}

#[test]
fn receipt_rejects_duplicate_semantic_protected_label_with_nonminimal_integer_encoding() {
    let proof = Rfc9162InclusionProof::new(2, 0, vec![[0u8; 32]]).to_cbor();
    let vdp = Rfc9942Vdp::new(Rfc9942ProofKind::Inclusion, vec![proof]).unwrap();

    // Protected = {1: -7, 1: -7 using non-minimal uint label encoding, 395: 1}.
    let protected = [
        0xa3, 0x01, 0x26, 0x18, 0x01, 0x26,
        0x19, 0x01, 0x8b, 0x01,
    ];

    let mut encoded = vec![0xd2, 0x84];
    encoded.extend_from_slice(&bstr(&protected));
    encoded.extend_from_slice(&[0xa1, 0x19, 0x01, 0x8c]);
    encoded.extend_from_slice(&vdp.to_cbor());
    encoded.extend_from_slice(&bstr(&[0u8; 32]));
    encoded.extend_from_slice(&bstr(&[0u8; 64]));

    assert_eq!(
        Rfc9942ReceiptEnvelope::from_cbor(&encoded),
        Err(Rfc9942VdpError::InvalidStructure)
    );
}

#[test]
fn outer_cose_rejects_duplicate_semantic_protected_label_with_nonminimal_integer_encoding() {
    // Protected = {1: -7, 1: -7 using non-minimal uint label encoding}.
    let protected = [0xa2, 0x01, 0x26, 0x18, 0x01, 0x26];

    let mut encoded = vec![0xd2, 0x84];
    encoded.extend_from_slice(&bstr(&protected));
    encoded.push(0xa0);
    encoded.push(0xf6);
    encoded.extend_from_slice(&bstr(&[0u8; 64]));

    assert_eq!(
        Rfc9942SignatureWithReceipts::from_cbor(&encoded),
        Err(Rfc9942VdpError::InvalidStructure)
    );
}

#[test]
fn cose_key_rejects_duplicate_labels() {
    let mut bytes = valid_key();
    bytes[0] = 0xa8;
    bytes.extend_from_slice(&[0x01, 0x02]);
    assert_eq!(
        Rfc9942Es256CoseKey::from_cbor(&bytes),
        Err(Rfc9942VdpError::InvalidEs256CoseKey)
    );
}

#[test]
fn cose_key_rejects_non_label_root_map_key() {
    let mut wire=vec![0xbf];
    wire.push(0x01); wire.push(0x02);
    wire.extend_from_slice(&[0x42,0xaa,0xbb,0x00]);
    wire.push(0xff);
    assert_eq!(
        Rfc9942Es256CoseKey::from_cbor(&wire),
        Err(Rfc9942VdpError::InvalidEncoding)
    );
}

#[test]
fn cose_key_rejects_duplicate_semantic_label_with_nonminimal_integer_encoding() {
    let mut wire=valid_key();
    assert_eq!(wire[0],0xa7);
    wire[0]=0xa8;
    wire.extend_from_slice(&[0x18,0x01,0x02]);
    assert_eq!(
        Rfc9942Es256CoseKey::from_cbor(&wire),
        Err(Rfc9942VdpError::InvalidEs256CoseKey)
    );
}

#[test]
fn cose_key_rejects_non_map_root_and_trailing_bytes() {
    assert_eq!(
        Rfc9942Es256CoseKey::from_cbor(&[0x80]),
        Err(Rfc9942VdpError::InvalidEncoding)
    );
    let mut bytes = valid_key();
    bytes.push(0x00);
    assert_eq!(
        Rfc9942Es256CoseKey::from_cbor(&bytes),
        Err(Rfc9942VdpError::InvalidEncoding)
    );
}

#[test]
fn cose_key_accepts_textual_ec2_p256_es256_and_verify() {
    let mut bytes = vec![0xa6];
    bytes.extend_from_slice(&[0x01, 0x63, b'E', b'C', b'2']);
    bytes.extend_from_slice(&[0x03, 0x65, b'E', b'S', b'2', b'5', b'6']);
    bytes.extend_from_slice(&[0x04, 0x81, 0x66, b'v', b'e', b'r', b'i', b'f', b'y']);
    bytes.extend_from_slice(&[0x20, 0x65, b'P', b'-', b'2', b'5', b'6']);
    bytes.extend_from_slice(&bstr_field(0x21, &X));
    bytes.extend_from_slice(&bstr_field(0x22, &Y));
    let parsed = Rfc9942Es256CoseKey::from_cbor(&bytes).expect("textual aliases");
    assert_eq!(&parsed.public_key_sec1()[1..33], &X);
    assert_eq!(&parsed.public_key_sec1()[33..65], &Y);
}


#[test]
fn cose_key_accepts_noncanonical_integer_but_rejects_indefinite_forms() {
    let mut bytes = valid_key();
    let pos = bytes.windows(2).position(|w| w == [0x01, 0x02]).unwrap();
    bytes.splice(pos..pos + 2, [0x01, 0x18, 0x02]);
    assert!(
        Rfc9942Es256CoseKey::from_cbor(&bytes).is_ok(),
        "valid non-minimal integer encoding must remain interoperable"
    );

    let mut indefinite = valid_key();
    indefinite[0] = 0xbf;
    assert_eq!(
        Rfc9942Es256CoseKey::from_cbor(&indefinite),
        Err(Rfc9942VdpError::InvalidEncoding)
    );

    let mut indefinite_bstr = valid_key();
    let pos = indefinite_bstr.windows(3).position(|w| w == [0x21, 0x58, 0x20]).unwrap();
    indefinite_bstr[pos + 1] = 0x5f;
    indefinite_bstr.remove(pos + 2);
    assert_eq!(
        Rfc9942Es256CoseKey::from_cbor(&indefinite_bstr),
        Err(Rfc9942VdpError::InvalidEncoding)
    );
}

#[test]
fn cose_key_accepts_indefinite_coordinate_bstr() {
    let mut bytes = valid_key();
    let pos = bytes.windows(3).position(|w| w == [0x21, 0x58, 0x20]).unwrap();
    let mut replacement = vec![0x21, 0x5f, 0x50];
    replacement.extend_from_slice(&X[..16]);
    replacement.extend_from_slice(&[0x50]);
    replacement.extend_from_slice(&X[16..]);
    replacement.push(0xff);
    bytes.splice(pos..pos + 35, replacement);

    let parsed = Rfc9942Es256CoseKey::from_cbor(&bytes)
        .expect("indefinite coordinate bstr must be accepted");
    assert_eq!(&parsed.public_key_sec1()[1..33], &X);
    assert_eq!(&parsed.public_key_sec1()[33..65], &Y);
}

#[test]
fn malformed_indefinite_coordinate_bstr_is_rejected() {
    let mut bytes = valid_key();
    let pos = bytes.windows(3).position(|w| w == [0x21, 0x58, 0x20]).unwrap();
    bytes.splice(pos..pos + 35, [0x21, 0x5f, 0x01, 0xff]);
    assert_eq!(
        Rfc9942Es256CoseKey::from_cbor(&bytes),
        Err(Rfc9942VdpError::InvalidEncoding)
    );
}

#[test]
fn syntactically_valid_but_invalid_p256_point_is_rejected_by_crypto_boundary() {
    let mut fields = valid_fields();
    fields[5] = bstr_field(0x21, &[0u8; 32]);
    fields[6] = bstr_field(0x22, &[0u8; 32]);
    let key = Rfc9942Es256CoseKey::from_cbor(&key(&fields))
        .expect("point shape is structurally valid");

    let proof = Rfc9162InclusionProof::new(2, 0, vec![[0u8; 32]]).to_cbor();
    let vdp = Rfc9942Vdp::new(Rfc9942ProofKind::Inclusion, vec![proof]).unwrap();
    let receipt = Rfc9942ReceiptEnvelope::new(
        COSE_ES256_ALGORITHM_ID,
        vdp,
        Rfc9942ReceiptPayload::Attached([0u8; 32]),
        vec![0u8; 64],
    )
    .unwrap();

    assert_eq!(
        receipt.verify_es256_cose_key(&key, &[], None),
        Err(Rfc9942VdpError::InvalidEs256Signature)
    );
}

