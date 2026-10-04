//! Cross-implementation ES256 COSE_Sign1 known-answer vectors.
//!
//! The positive vector is RFC 8392 Appendix A.3 (signed CWT), with the public
//! P-256 key from Appendix A.2.3. It is checked as fixed wire bytes so the
//! verification path is independent of Symthaea's signer implementation.

#![cfg(feature = "semantic-receipts")]

use symthaea_swarm::semantic_evidence_vds::{
    Rfc9942SignaturePayload, Rfc9942SignatureWithReceipts, Rfc9942VdpError,
    COSE_ES256_ALGORITHM_ID,
};

const RFC8392_SIGNED_CWT: &[u8] = &[
    0xd2, 0x84, 0x43, 0xa1, 0x01, 0x26, 0xa1, 0x04,
    0x52, 0x41, 0x73, 0x79, 0x6d, 0x6d, 0x65, 0x74,
    0x72, 0x69, 0x63, 0x45, 0x43, 0x44, 0x53, 0x41,
    0x32, 0x35, 0x36, 0x58, 0x50, 0xa7, 0x01, 0x75,
    0x63, 0x6f, 0x61, 0x70, 0x3a, 0x2f, 0x2f, 0x61,
    0x73, 0x2e, 0x65, 0x78, 0x61, 0x6d, 0x70, 0x6c,
    0x65, 0x2e, 0x63, 0x6f, 0x6d, 0x02, 0x65, 0x65,
    0x72, 0x69, 0x6b, 0x77, 0x03, 0x78, 0x18, 0x63,
    0x6f, 0x61, 0x70, 0x3a, 0x2f, 0x2f, 0x6c, 0x69,
    0x67, 0x68, 0x74, 0x2e, 0x65, 0x78, 0x61, 0x6d,
    0x70, 0x6c, 0x65, 0x2e, 0x63, 0x6f, 0x6d, 0x04,
    0x1a, 0x56, 0x12, 0xae, 0xb0, 0x05, 0x1a, 0x56,
    0x10, 0xd9, 0xf0, 0x06, 0x1a, 0x56, 0x10, 0xd9,
    0xf0, 0x07, 0x42, 0x0b, 0x71, 0x58, 0x40, 0x54,
    0x27, 0xc1, 0xff, 0x28, 0xd2, 0x3f, 0xba, 0xd1,
    0xf2, 0x9c, 0x4c, 0x7c, 0x6a, 0x55, 0x5e, 0x60,
    0x1d, 0x6f, 0xa2, 0x9f, 0x91, 0x79, 0xbc, 0x3d,
    0x74, 0x38, 0xba, 0xca, 0xca, 0x5a, 0xcd, 0x08,
    0xc8, 0xd4, 0xd4, 0xf9, 0x61, 0x31, 0x68, 0x0c,
    0x42, 0x9a, 0x01, 0xf8, 0x59, 0x51, 0xec, 0xee,
    0x74, 0x3a, 0x52, 0xb9, 0xb6, 0x36, 0x32, 0xc5,
    0x72, 0x09, 0x12, 0x0e, 0x1c, 0x9e, 0x30,
];

fn hex_bytes(s: &str) -> Vec<u8> {
    assert_eq!(s.len() % 2, 0);
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

const RFC8392_P256_X: [u8; 32] = [
    0x14, 0x33, 0x29, 0xcc, 0xe7, 0x86, 0x8e, 0x41,
    0x69, 0x27, 0x59, 0x9c, 0xf6, 0x5a, 0x34, 0xf3,
    0xce, 0x2f, 0xfd, 0xa5, 0x5a, 0x7e, 0xca, 0x69,
    0xed, 0x89, 0x19, 0xa3, 0x94, 0xd4, 0x2f, 0x0f,
];

const RFC8392_P256_Y: [u8; 32] = [
    0x60, 0xf7, 0xf1, 0xa7, 0x80, 0xd8, 0xa7, 0x83,
    0xbf, 0xb7, 0xa2, 0xdd, 0x6b, 0x27, 0x96, 0xe8,
    0x12, 0x8d, 0xbc, 0xef, 0x9d, 0x3d, 0x16, 0x8d,
    0xb9, 0x52, 0x99, 0x71, 0xa3, 0x6e, 0x7b, 0x09,
];

fn sec1_public_key() -> [u8; 65] {
    let mut key = [0u8; 65];
    key[0] = 0x04;
    key[1..33].copy_from_slice(&RFC8392_P256_X);
    key[33..].copy_from_slice(&RFC8392_P256_Y);
    key
}

fn rfc8392_detached_signed_cwt() -> Vec<u8> {
    const PAYLOAD_BSTR_START: usize = 27;
    const SIGNATURE_BSTR_START: usize = 109;

    let mut detached = Vec::with_capacity(RFC8392_SIGNED_CWT.len() - 81);
    detached.extend_from_slice(&RFC8392_SIGNED_CWT[..PAYLOAD_BSTR_START]);
    detached.push(0xf6);
    detached.extend_from_slice(&RFC8392_SIGNED_CWT[SIGNATURE_BSTR_START..]);
    detached
}

#[test]
fn rfc8392_es256_known_answer_verifies() {
    assert_eq!(RFC8392_SIGNED_CWT.len(), 175, "RFC 8392 Appendix A.3 wire fixture length changed");
    assert_eq!(&RFC8392_SIGNED_CWT[109..111], &[0x58, 0x40], "RFC 8392 signature must be a 64-byte bstr");
    assert_eq!(
        &RFC8392_SIGNED_CWT[111..],
        &hex_bytes("5427c1ff28d23fbad1f29c4c7c6a555e601d6fa29f9179bc3d7438bacaca5acd08c8d4d4f96131680c429a01f85951ecee743a52b9b63632c57209120e1c9e30"),
        "RFC 8392 signature bytes changed"
    );
    let message = Rfc9942SignatureWithReceipts::from_cbor(RFC8392_SIGNED_CWT)
        .expect("RFC 8392 signed CWT must parse");

    assert_eq!(
        message.protected_algorithm_id().unwrap(),
        COSE_ES256_ALGORITHM_ID
    );
    assert_eq!(
        message.payload(),
        &Rfc9942SignaturePayload::Attached(
            RFC8392_SIGNED_CWT[29..109].to_vec()
        )
    );

    message
        .verify_es256(&sec1_public_key(), &[], None)
        .expect("RFC 8392 ES256 known-answer vector must verify");
}

#[test]
fn rfc8392_es256_known_answer_rejects_signature_tampering() {
    let mut encoded = RFC8392_SIGNED_CWT.to_vec();
    let last = encoded.len() - 1;
    encoded[last] ^= 0x01;

    let message = Rfc9942SignatureWithReceipts::from_cbor(&encoded)
        .expect("tampered signature bytes remain structurally valid");

    assert_eq!(
        message.verify_es256(&sec1_public_key(), &[], None),
        Err(Rfc9942VdpError::InvalidEs256Signature)
    );
}

#[test]
fn rfc8392_es256_known_answer_binds_protected_algorithm() {
    let mut encoded = RFC8392_SIGNED_CWT.to_vec();
    // Protected header is the bstr 43 a1 01 26; replace -7 (0x26) with -8 (0x27).
    encoded[5] = 0x27;

    let message = Rfc9942SignatureWithReceipts::from_cbor(&encoded)
        .expect("algorithm substitution preserves COSE structure");

    assert_eq!(message.protected_algorithm_id().unwrap(), -8);
    assert_eq!(
        message.verify_es256(&sec1_public_key(), &[], None),
        Err(Rfc9942VdpError::UnsupportedSignatureAlgorithm(-8))
    );
}

#[test]
fn rfc8392_es256_known_answer_detached_payload_requires_and_binds_external_bytes() {
    const PAYLOAD_START: usize = 29;
    const PAYLOAD_END: usize = 109;

    let encoded = rfc8392_detached_signed_cwt();
    assert_eq!(encoded.len(), 94);

    let message = Rfc9942SignatureWithReceipts::from_cbor(&encoded)
        .expect("RFC 8392 detached variant must parse");
    assert_eq!(message.payload(), &Rfc9942SignaturePayload::Detached);

    assert_eq!(
        message.verify_es256(&sec1_public_key(), &[], None),
        Err(Rfc9942VdpError::DetachedPayloadRequired)
    );

    message
        .verify_es256(
            &sec1_public_key(),
            &[],
            Some(&RFC8392_SIGNED_CWT[PAYLOAD_START..PAYLOAD_END]),
        )
        .expect("RFC 8392 signature must verify with its externally supplied payload");

    assert_eq!(
        message.verify_es256(&sec1_public_key(), &[], Some(&[0x00; 80])),
        Err(Rfc9942VdpError::InvalidEs256Signature)
    );
}

#[test]
fn rfc8392_es256_known_answer_rejects_external_payload_for_attached_content() {
    const PAYLOAD_START: usize = 29;
    const PAYLOAD_END: usize = 109;

    let message = Rfc9942SignatureWithReceipts::from_cbor(RFC8392_SIGNED_CWT)
        .expect("RFC 8392 signed CWT must parse");

    assert_eq!(
        message.verify_es256(
            &sec1_public_key(),
            &[],
            Some(&RFC8392_SIGNED_CWT[PAYLOAD_START..PAYLOAD_END]),
        ),
        Err(Rfc9942VdpError::InvalidStructure)
    );
}
