// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Cross-repository byte oracle for the mirrored Xenia authenticated-payload
//! receipt body. The expected bytes are neutral bincode-v1 wire material shared
//! with Xenia's qualification branch.

use symthaea_iot_transport_receipt::{
    XENIA_AUTHENTICATED_PAYLOAD_RECEIPT_DOMAIN, XENIA_AUTHENTICATED_PAYLOAD_RECEIPT_SCHEMA,
    XENIA_HYBRID_SIGNATURE_SUITE, XeniaAuthenticatedPayloadReceiptBodyV1,
    XeniaReceiptPeerRoleV1,
};

const EXPECTED_HEX: &str = include_str!("test-vectors/authenticated-payload-receipt-body-v1.hex");
const EXPECTED_CANONICAL_LEN: usize = 354;

fn sequence(start: u8) -> [u8; 32] {
    std::array::from_fn(|index| start.wrapping_add(index as u8))
}

fn neutral_body() -> XeniaAuthenticatedPayloadReceiptBodyV1 {
    XeniaAuthenticatedPayloadReceiptBodyV1 {
        schema: XENIA_AUTHENTICATED_PAYLOAD_RECEIPT_SCHEMA.to_owned(),
        attestor_id: "xenia-host-a".to_owned(),
        key_id: "transport-attestor-1".to_owned(),
        signature_algorithm: XENIA_HYBRID_SIGNATURE_SUITE.to_owned(),
        session_evidence_digest: sequence(0x01),
        peer_role: XeniaReceiptPeerRoleV1::Viewer,
        peer_identity_fingerprint: sequence(0x21),
        transcript_hash: sequence(0x41),
        session_context_hash: sequence(0x61),
        telemetry_enabled: true,
        input_control_enabled: false,
        payload_type: 0x70,
        payload_len: 0x1234,
        payload_digest: sequence(0x81),
        sealed_envelope_digest: sequence(0xA1),
        opened_at_unix_ms: 0x0102_0304_0506_0708,
        expires_at_unix_ms: 0x0102_0304_0506_17E9,
    }
}

fn decode_hex(input: &str) -> Vec<u8> {
    let input = input.trim();
    assert_eq!(input.len() % 2, 0, "wire vector hex must have even length");
    input
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| (hex_nibble(pair[0]) << 4) | hex_nibble(pair[1]))
        .collect()
}

fn hex_nibble(byte: u8) -> u8 {
    match byte {
        b'0'..=b'9' => byte - b'0',
        b'a'..=b'f' => byte - b'a' + 10,
        b'A'..=b'F' => byte - b'A' + 10,
        _ => panic!("non-hex byte in committed receipt-body vector"),
    }
}

#[test]
fn symthaea_mirror_matches_neutral_xenia_body_bytes_exactly() {
    let body = neutral_body();
    let expected = decode_hex(EXPECTED_HEX);
    assert_eq!(expected.len(), EXPECTED_CANONICAL_LEN);

    let actual = body.canonical_bytes().expect("neutral mirrored body is valid");
    assert_eq!(actual.len(), EXPECTED_CANONICAL_LEN);
    assert_eq!(actual, expected);
}

#[test]
fn neutral_wire_bytes_round_trip_to_symthaea_mirror_exactly() {
    let expected = decode_hex(EXPECTED_HEX);
    let decoded: XeniaAuthenticatedPayloadReceiptBodyV1 =
        bincode::deserialize(&expected).expect("neutral Xenia body must decode in Symthaea");

    assert_eq!(decoded, neutral_body());
    assert_eq!(
        bincode::serialize(&decoded).expect("mirrored neutral body must reserialize"),
        expected
    );
}

#[test]
fn mirrored_signing_digest_is_exact_domain_plus_neutral_wire_bytes() {
    let body = neutral_body();
    let expected = decode_hex(EXPECTED_HEX);
    let mut preimage =
        Vec::with_capacity(XENIA_AUTHENTICATED_PAYLOAD_RECEIPT_DOMAIN.len() + expected.len());
    preimage.extend_from_slice(XENIA_AUTHENTICATED_PAYLOAD_RECEIPT_DOMAIN);
    preimage.extend_from_slice(&expected);

    assert_eq!(
        body.signing_digest().expect("mirrored neutral body signing digest"),
        *blake3::hash(&preimage).as_bytes()
    );
}

#[test]
fn mirrored_consequential_field_mutation_changes_bytes_and_digest() {
    let body = neutral_body();
    let baseline_bytes = body.canonical_bytes().unwrap();
    let baseline_digest = body.signing_digest().unwrap();

    let mut changed = neutral_body();
    changed.input_control_enabled = true;

    assert_ne!(changed.canonical_bytes().unwrap(), baseline_bytes);
    assert_ne!(changed.signing_digest().unwrap(), baseline_digest);
}
