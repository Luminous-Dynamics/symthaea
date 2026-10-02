//! Cross-implementation ES256 COSE_Sign1 known-answer vectors.
//!
//! The positive vector is copied from the public cose-js verification example
//! and is intentionally checked as a fixed byte-level COSE_Sign1 object rather
//! than regenerated with ring. This keeps the test independent of Symthaea's
//! signer implementation.

#![cfg(feature = "semantic-receipts")]

use symthaea_swarm::semantic_evidence_vds::{
    Rfc9942SignatureWithReceipts, Rfc9942VdpError, COSE_ES256_ALGORITHM_ID,
};

const COSE_JS_ES256_MESSAGE: &[u8] = &[
    0xd2, 0x84,
    0x43, 0xa1, 0x01, 0x26,
    0xa1, 0x04, 0x42, 0x31, 0x31,
    0x72, b'I', b'm', b'p', b'o', b'r', b't', b'a', b'n', b't', b' ',
    b'm', b'e', b's', b's', b'a', b'g', b'e', b'!',
    0x58, 0x40,
    0x4c, 0x2b, 0x6b, 0x66, 0xdf, 0xed, 0xc4, 0xcf,
    0xef, 0x0f, 0x22, 0x1c, 0xf7, 0xac, 0x7f, 0x95,
    0x08, 0x7a, 0x4c, 0x42, 0x45, 0xfe, 0xf0, 0x06,
    0x3a, 0x0f, 0xd4, 0x01, 0x4b, 0x67, 0x0f, 0x64,
    0x2d, 0x31, 0xe2, 0x6d, 0x38, 0x34, 0x5b, 0xb4,
    0xef, 0xcd, 0xc7, 0xde, 0xd3, 0x08, 0x3a, 0xb4,
    0xfe, 0x71, 0xb6, 0x2a, 0x23, 0xf7, 0x66, 0xd8,
    0x37, 0x85, 0xf0, 0x44, 0xb2, 0x05, 0x34, 0x4f,
    0x09,
];

const COSE_JS_ES256_X: [u8; 32] = [
    0x14, 0x33, 0x29, 0xcc, 0xe7, 0x86, 0x8e, 0x41,
    0x69, 0x27, 0x59, 0x9c, 0xf6, 0x5a, 0x34, 0xf3,
    0xce, 0x2f, 0xfd, 0xa5, 0x5a, 0x7a, 0xec, 0xa6,
    0x9e, 0xd8, 0x91, 0x9a, 0x39, 0x4d, 0x42, 0xf0,
];

const COSE_JS_ES256_Y: [u8; 32] = [
    0x60, 0xf7, 0xf1, 0xa7, 0x80, 0xd8, 0xa7, 0x83,
    0xbf, 0xb7, 0xa2, 0xdd, 0x6b, 0x27, 0x96, 0xe8,
    0x12, 0x8d, 0xbc, 0xef, 0x9d, 0x3d, 0x16, 0x8d,
    0xb9, 0x52, 0x99, 0x71, 0xa3, 0x6e, 0x7b, 0x9,
];

fn sec1_public_key() -> [u8; 65] {
    let mut key = [0u8; 65];
    key[0] = 0x04;
    key[1..33].copy_from_slice(&COSE_JS_ES256_X);
    key[33..].copy_from_slice(&COSE_JS_ES256_Y);
    key
}

#[test]
fn cose_js_es256_known_answer_verifies() {
    let message = Rfc9942SignatureWithReceipts::from_cbor(COSE_JS_ES256_MESSAGE)
        .expect("known-answer COSE_Sign1 must parse");

    assert_eq!(
        message.protected_algorithm_id().unwrap(),
        COSE_ES256_ALGORITHM_ID
    );
    assert_eq!(message.payload(), &symthaea_swarm::semantic_evidence_vds::Rfc9942SignaturePayload::Attached(
        b"Important message!".to_vec()
    ));

    message
        .verify_es256(&sec1_public_key(), &[], None)
        .expect("independent ES256 COSE_Sign1 vector must verify");
}

#[test]
fn cose_js_es256_known_answer_rejects_payload_substitution() {
    let message = Rfc9942SignatureWithReceipts::from_cbor(COSE_JS_ES256_MESSAGE)
        .expect("known-answer COSE_Sign1 must parse");

    let error = message
        .verify_es256(&sec1_public_key(), &[], Some(b"tampered payload"))
        .expect_err("attached payload must not accept a detached replacement");

    assert_eq!(error, Rfc9942VdpError::InvalidStructure);
}
