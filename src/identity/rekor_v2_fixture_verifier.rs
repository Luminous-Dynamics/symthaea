// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Offline, fixture-only verification of the Rekor v2/C2SP mechanics needed by
//! EUREKA-002 V2 external-high-water publication.
//!
//! This module is deliberately test-only. It verifies Sigstore's published
//! Rekor v2 example with a pinned public test key and demonstrates the exact
//! checkpoint-signature + RFC6962 inclusion mechanics that a later provider
//! adapter will need. It does NOT establish a production Sigstore TrustedRoot,
//! perform a live publication, prove independent witnessing/consistency, admit
//! an EUREKA authority root, or grant scientific execution authority.

use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde_json::Value;
use sha2::{Digest, Sha256};

const UPSTREAM_REPOSITORY: &str = "sigstore/rekor-tiles";
const UPSTREAM_COMMIT: &str = "7e4395ba93e6882cf89fc5b23ec53d4d2bea0778";
const UPSTREAM_CLIENTS_BLOB: &str = "48b5d79b9a9bee5c0c1d05c13eb9d4c3682219f7";
const UPSTREAM_PUBLIC_KEY_BLOB: &str = "0a1b8edcc4ce7e6ccda0e31e5d9fca2618758139";
const CHECKPOINT_KEY_NAME: &str = "rekor-local";
const ED25519_SIGNATURE_TYPE: u8 = 0x01;
const EXPECTED_FULL_CHECKPOINT_KEY_ID_B64: &str =
    "2AtEIMfG6Y41yK0tcwRTBS2tjhOrjKGIpDkHFgp65g0=";
const EXPECTED_ROOT_B64: &str = "m5JVGx4ESzbU2kFUPgYQ9adCA7e7mjnwQRluEmGJbHU=";
const EXPECTED_CANONICAL_BODY_SHA256: [u8; 32] = [
    0xa8, 0xb9, 0x88, 0xe0, 0x08, 0xb6, 0x88, 0x0d, 0x34, 0x37, 0xf1, 0x2b, 0x21, 0x60,
    0xae, 0x8d, 0x29, 0x87, 0x37, 0x5e, 0x15, 0x43, 0xc5, 0xa4, 0x0e, 0x13, 0xc1, 0xe5,
    0x37, 0x43, 0x01, 0x72,
];

// Raw RFC8032 Ed25519 public key extracted from the pinned upstream SPKI PEM:
// tests/testdata/pki/ed25519-pub-key.pem at UPSTREAM_COMMIT.
const PUBLIC_KEY: [u8; 32] = [
    0x44, 0x4b, 0xc9, 0xc8, 0xd6, 0x46, 0x8d, 0x7e, 0x81, 0xdc, 0x30, 0x08, 0xb8, 0x3d,
    0xc1, 0x4e, 0x0f, 0x6b, 0x23, 0x05, 0x74, 0xd0, 0x66, 0x3c, 0x5e, 0x0e, 0x45, 0x53,
    0xe2, 0x05, 0x0d, 0x44,
];

// Exact example response published in rekor-tiles/CLIENTS.md at UPSTREAM_COMMIT.
// The duplicated inclusionProof rootHash/treeSize are intentionally present but
// are not authority inputs. Trusted root/tree-size below come only from the
// successfully verified C2SP checkpoint, as Rekor v2 requires.
const REKOR_V2_RESPONSE_JSON: &str = r#"{
  "logIndex": "0",
  "logId": {
    "keyId": "2AtEIMfG6Y41yK0tcwRTBS2tjhOrjKGIpDkHFgp65g0="
  },
  "kindVersion": {
    "kind": "hashedrekord",
    "version": "0.0.2"
  },
  "integratedTime": "0",
  "inclusionPromise": null,
  "inclusionProof": {
    "logIndex": "0",
    "rootHash": "OWI5MjU1MWIxZTA0NGIzNmQ0ZGE0MTU0M2UwNjEwZjVhNzQyMDNiN2JiOWEzOWYwNDExOTZlMTI2MTg5NmM3NQ==",
    "treeSize": "1",
    "hashes": [],
    "checkpoint": {
      "envelope": "rekor-local\n1\nm5JVGx4ESzbU2kFUPgYQ9adCA7e7mjnwQRluEmGJbHU=\n\n— rekor-local 2AtEIIwnbtxrneJ7L1lQebfBRl7TxK84DTmx+kcZi7A25cBDgESI23f9ylThAlOireJ7U+H8eZF/4kJQcn9o5Qt8mQU=\n"
    }
  },
  "canonicalizedBody": "eyJhcGlWZXJzaW9uIjoiMC4wLjIiLCJraW5kIjoiaGFzaGVkcmVrb3JkIiwic3BlYyI6eyJoYXNoZWRSZWtvcmRWMF8wXzIiOnsiZGF0YSI6eyJhbGdvcml0aG0iOiJTSEEyXzI1NiIsImRpZ2VzdCI6ImR5ajRlZG5ZSGpONC96c2pqQmVlTGFoUzlzbHA5N1o2N0xUQVZ4anJqWHc9In0sInNpZ25hdHVyZSI6eyJjb250ZW50IjoiTUVRQ0lCK1lQYTlvM1NOMHNRNHVkdUdmK21aeHdGZk9oRlowQ2d5K3A3VnQxbzJTQWlBUEZESHFPQUpMWW12dENXT3NEeU5ZMUg0VjN6bTRORURZczNOeXZIaDFQZz09IiwidmVyaWZpZXIiOnsia2V5RGV0YWlscyI6IlBLSVhfRUNEU0FfUDI1Nl9TSEFfMjU2IiwicHVibGljS2V5Ijp7InJhd0J5dGVzIjoiTUZrd0V3WUhLb1pJemowQ0FRWUlLb1pJemowREFRY0RRZ0FFMnNsT2Y4ZVpjajJtb1cydDRVRmo3dkNMNlFwRHprRHFxU1VtbTRPSkNWdklhdUtMeG0wYUdzM1ZNUFBmYXVNUGFNdXRuMC9zM2pnMHJyb0Z4b2ljeWc9PSJ9fX19fX0="
}"#;

#[derive(Debug, Clone, PartialEq, Eq)]
struct VerifiedRekorV2MechanicsFixture {
    upstream_commit: &'static str,
    checkpoint_origin: String,
    checkpoint_tree_size: u64,
    checkpoint_root: [u8; 32],
    checkpoint_key_id: [u8; 32],
    log_index: u64,
    canonical_body_sha256: [u8; 32],
    fixture_only: bool,
    production_trusted_root: bool,
    externality_verified: bool,
    witnessed_consistency_verified: bool,
    execution_authority_granted: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum FixtureError {
    Json,
    MissingField,
    WrongFixtureShape,
    NonCanonicalInteger,
    Base64,
    SignedNote,
    WrongKeyName,
    KeyIdMismatch,
    Signature,
    Checkpoint,
    BodyKindVersion,
    Inclusion,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct VerifiedCheckpoint {
    origin: String,
    tree_size: u64,
    root: [u8; 32],
    full_key_id: [u8; 32],
}

fn verify_official_fixture(
    response_json: &str,
) -> Result<VerifiedRekorV2MechanicsFixture, FixtureError> {
    let response: Value = serde_json::from_str(response_json).map_err(|_| FixtureError::Json)?;

    let log_index = canonical_u64(json_str(&response, &["logIndex"])?)?;
    let full_log_id = decode_base64(json_str(&response, &["logId", "keyId"])?)?;
    if full_log_id.len() != 32 {
        return Err(FixtureError::WrongFixtureShape);
    }

    if json_str(&response, &["kindVersion", "kind"])? != "hashedrekord"
        || json_str(&response, &["kindVersion", "version"])? != "0.0.2"
        || !json_value(&response, &["inclusionPromise"])?.is_null()
    {
        return Err(FixtureError::WrongFixtureShape);
    }

    let hashes = json_value(&response, &["inclusionProof", "hashes"])?
        .as_array()
        .ok_or(FixtureError::WrongFixtureShape)?;

    let checkpoint_envelope = json_str(
        &response,
        &["inclusionProof", "checkpoint", "envelope"],
    )?;
    let checkpoint = verify_checkpoint(checkpoint_envelope, &PUBLIC_KEY)?;

    // Rekor v2 CLIENTS.md requires clients to obtain tree size/root from the
    // verified checkpoint and log index from the top-level entry. These
    // duplicated proof fields and integratedTime are intentionally parsed only
    // for fixture-shape observability; they do not influence the verified result.
    let _untrusted_duplicate_log_index =
        json_str(&response, &["inclusionProof", "logIndex"])?;
    let _untrusted_duplicate_root = json_str(&response, &["inclusionProof", "rootHash"])?;
    let _untrusted_duplicate_tree_size = json_str(&response, &["inclusionProof", "treeSize"])?;
    let _untrusted_integrated_time = json_str(&response, &["integratedTime"])?;

    let mut log_id_array = [0_u8; 32];
    log_id_array.copy_from_slice(&full_log_id);
    if log_id_array != checkpoint.full_key_id {
        return Err(FixtureError::KeyIdMismatch);
    }

    let canonical_body = decode_base64(json_str(&response, &["canonicalizedBody"])?)?;
    verify_canonical_body_shape(&canonical_body)?;

    if checkpoint.tree_size != 1 || log_index != 0 || !hashes.is_empty() {
        return Err(FixtureError::Inclusion);
    }

    let mut leaf_hasher = Sha256::new();
    leaf_hasher.update([0_u8]);
    leaf_hasher.update(&canonical_body);
    let leaf_hash: [u8; 32] = leaf_hasher.finalize().into();
    if leaf_hash != checkpoint.root {
        return Err(FixtureError::Inclusion);
    }

    let canonical_body_sha256: [u8; 32] = Sha256::digest(&canonical_body).into();
    Ok(VerifiedRekorV2MechanicsFixture {
        upstream_commit: UPSTREAM_COMMIT,
        checkpoint_origin: checkpoint.origin,
        checkpoint_tree_size: checkpoint.tree_size,
        checkpoint_root: checkpoint.root,
        checkpoint_key_id: checkpoint.full_key_id,
        log_index,
        canonical_body_sha256,
        fixture_only: true,
        production_trusted_root: false,
        externality_verified: false,
        witnessed_consistency_verified: false,
        execution_authority_granted: false,
    })
}

fn verify_checkpoint(
    envelope: &str,
    public_key: &[u8; 32],
) -> Result<VerifiedCheckpoint, FixtureError> {
    if !envelope.ends_with('\n') || envelope.bytes().any(|byte| byte < 0x20 && byte != b'\n') {
        return Err(FixtureError::SignedNote);
    }
    let separator = envelope.rfind("\n\n").ok_or(FixtureError::SignedNote)?;
    let note_text = &envelope[..separator + 1];
    let signature_text = &envelope[separator + 2..];
    if signature_text.is_empty() || !signature_text.ends_with('\n') {
        return Err(FixtureError::SignedNote);
    }

    let body_without_newline = note_text
        .strip_suffix('\n')
        .ok_or(FixtureError::SignedNote)?;
    let body_lines: Vec<&str> = body_without_newline.split('\n').collect();
    if body_lines.len() < 3 || body_lines.iter().any(|line| line.is_empty()) {
        return Err(FixtureError::Checkpoint);
    }
    let origin = body_lines[0];
    let tree_size = canonical_u64(body_lines[1])?;
    let root_bytes = decode_base64(body_lines[2])?;
    if root_bytes.len() != 32 {
        return Err(FixtureError::Checkpoint);
    }
    let mut root = [0_u8; 32];
    root.copy_from_slice(&root_bytes);

    let full_key_id = checkpoint_key_id(origin, public_key);
    let mut saw_known_signature = false;
    let mut signature_count = 0_usize;
    for line in signature_text[..signature_text.len() - 1].split('\n') {
        signature_count = signature_count
            .checked_add(1)
            .ok_or(FixtureError::SignedNote)?;
        if signature_count > 16 {
            return Err(FixtureError::SignedNote);
        }
        let rest = line.strip_prefix("— ").ok_or(FixtureError::SignedNote)?;
        let (key_name, signature_b64) = rest.split_once(' ').ok_or(FixtureError::SignedNote)?;
        if key_name.is_empty() || key_name.contains('+') || key_name.chars().any(char::is_whitespace) {
            return Err(FixtureError::SignedNote);
        }
        if key_name != origin {
            continue;
        }
        if key_name != CHECKPOINT_KEY_NAME {
            return Err(FixtureError::WrongKeyName);
        }

        let signed = decode_base64(signature_b64)?;
        if signed.len() != 68 || signed[..4] != full_key_id[..4] {
            return Err(FixtureError::KeyIdMismatch);
        }
        let signature_bytes: [u8; 64] = signed[4..]
            .try_into()
            .map_err(|_| FixtureError::Signature)?;
        let signature = Signature::from_bytes(&signature_bytes);
        let verifying_key =
            VerifyingKey::from_bytes(public_key).map_err(|_| FixtureError::Signature)?;
        verifying_key
            .verify(note_text.as_bytes(), &signature)
            .map_err(|_| FixtureError::Signature)?;
        saw_known_signature = true;
    }
    if !saw_known_signature {
        return Err(FixtureError::Signature);
    }

    Ok(VerifiedCheckpoint {
        origin: origin.to_owned(),
        tree_size,
        root,
        full_key_id,
    })
}

fn checkpoint_key_id(key_name: &str, public_key: &[u8; 32]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(key_name.as_bytes());
    hasher.update([b'\n', ED25519_SIGNATURE_TYPE]);
    hasher.update(public_key);
    hasher.finalize().into()
}

fn verify_canonical_body_shape(bytes: &[u8]) -> Result<(), FixtureError> {
    let body: Value = serde_json::from_slice(bytes).map_err(|_| FixtureError::Json)?;
    if json_str(&body, &["apiVersion"])? != "0.0.2"
        || json_str(&body, &["kind"])? != "hashedrekord"
        || json_str(
            &body,
            &["spec", "hashedRekordV0_0_2", "data", "algorithm"],
        )? != "SHA2_256"
    {
        return Err(FixtureError::BodyKindVersion);
    }
    let digest = decode_base64(json_str(
        &body,
        &["spec", "hashedRekordV0_0_2", "data", "digest"],
    )?)?;
    if digest.len() != 32 {
        return Err(FixtureError::BodyKindVersion);
    }
    Ok(())
}

fn json_value<'a>(root: &'a Value, path: &[&str]) -> Result<&'a Value, FixtureError> {
    let mut current = root;
    for key in path {
        current = current.get(*key).ok_or(FixtureError::MissingField)?;
    }
    Ok(current)
}

fn json_str<'a>(root: &'a Value, path: &[&str]) -> Result<&'a str, FixtureError> {
    json_value(root, path)?
        .as_str()
        .ok_or(FixtureError::WrongFixtureShape)
}

fn canonical_u64(text: &str) -> Result<u64, FixtureError> {
    if text.is_empty()
        || (text.len() > 1 && text.starts_with('0'))
        || !text.bytes().all(|byte| byte.is_ascii_digit())
    {
        return Err(FixtureError::NonCanonicalInteger);
    }
    text.parse().map_err(|_| FixtureError::NonCanonicalInteger)
}

fn decode_base64(input: &str) -> Result<Vec<u8>, FixtureError> {
    let bytes = input.as_bytes();
    if bytes.is_empty() {
        return Ok(Vec::new());
    }
    if !bytes.len().is_multiple_of(4) {
        return Err(FixtureError::Base64);
    }

    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    let chunks = bytes.len() / 4;
    for (index, chunk) in bytes.chunks_exact(4).enumerate() {
        let last = index + 1 == chunks;
        let a = base64_value(chunk[0])?;
        let b = base64_value(chunk[1])?;
        out.push((a << 2) | (b >> 4));

        match (chunk[2], chunk[3]) {
            (b'=', b'=') if last && b & 0x0f == 0 => {}
            (b'=', b'=') => return Err(FixtureError::Base64),
            (b'=', _) => return Err(FixtureError::Base64),
            (c, b'=') if last => {
                let c = base64_value(c)?;
                if c & 0x03 != 0 {
                    return Err(FixtureError::Base64);
                }
                out.push((b << 4) | (c >> 2));
            }
            (c, b'=') => {
                let _ = c;
                return Err(FixtureError::Base64);
            }
            (c, d) => {
                let c = base64_value(c)?;
                let d = base64_value(d)?;
                out.push((b << 4) | (c >> 2));
                out.push((c << 6) | d);
            }
        }
    }
    Ok(out)
}

fn base64_value(byte: u8) -> Result<u8, FixtureError> {
    match byte {
        b'A'..=b'Z' => Ok(byte - b'A'),
        b'a'..=b'z' => Ok(byte - b'a' + 26),
        b'0'..=b'9' => Ok(byte - b'0' + 52),
        b'+' => Ok(62),
        b'/' => Ok(63),
        _ => Err(FixtureError::Base64),
    }
}

#[test]
fn official_rekor_v2_fixture_verifies_offline_without_authority_escalation() {
    let verified = verify_official_fixture(REKOR_V2_RESPONSE_JSON).expect("official fixture");

    assert_eq!(verified.upstream_commit, UPSTREAM_COMMIT);
    assert_eq!(UPSTREAM_REPOSITORY, "sigstore/rekor-tiles");
    assert_eq!(UPSTREAM_CLIENTS_BLOB.len(), 40);
    assert_eq!(UPSTREAM_PUBLIC_KEY_BLOB.len(), 40);
    assert_eq!(verified.checkpoint_origin, CHECKPOINT_KEY_NAME);
    assert_eq!(verified.checkpoint_tree_size, 1);
    assert_eq!(verified.log_index, 0);
    assert_eq!(verified.checkpoint_root, decode_32(EXPECTED_ROOT_B64));
    assert_eq!(
        verified.checkpoint_key_id,
        decode_32(EXPECTED_FULL_CHECKPOINT_KEY_ID_B64)
    );
    assert_eq!(
        verified.canonical_body_sha256,
        EXPECTED_CANONICAL_BODY_SHA256
    );
    assert!(verified.fixture_only);
    assert!(!verified.production_trusted_root);
    assert!(!verified.externality_verified);
    assert!(!verified.witnessed_consistency_verified);
    assert!(!verified.execution_authority_granted);
}

#[test]
fn checkpoint_key_id_matches_both_full_log_id_and_truncated_note_id() {
    let full = checkpoint_key_id(CHECKPOINT_KEY_NAME, &PUBLIC_KEY);
    assert_eq!(full, decode_32(EXPECTED_FULL_CHECKPOINT_KEY_ID_B64));
    assert_eq!(&full[..4], &[0xd8, 0x0b, 0x44, 0x20]);
}

#[test]
fn canonical_body_leaf_hash_is_the_verified_one_leaf_checkpoint_root() {
    let response: Value = serde_json::from_str(REKOR_V2_RESPONSE_JSON).expect("fixture json");
    let body = decode_base64(json_str(&response, &["canonicalizedBody"]).expect("body"))
        .expect("base64");
    let mut hasher = Sha256::new();
    hasher.update([0_u8]);
    hasher.update(body);
    let leaf: [u8; 32] = hasher.finalize().into();
    assert_eq!(leaf, decode_32(EXPECTED_ROOT_B64));
}

#[test]
fn non_authoritative_rekor_duplicates_cannot_change_verified_checkpoint() {
    let baseline = verify_official_fixture(REKOR_V2_RESPONSE_JSON).expect("baseline");
    let mutated = REKOR_V2_RESPONSE_JSON
        .replacen(
            "\"integratedTime\": \"0\"",
            "\"integratedTime\": \"987654321\"",
            1,
        )
        .replacen(
            "\"rootHash\": \"OWI5MjU1MWIxZTA0NGIzNmQ0ZGE0MTU0M2UwNjEwZjVhNzQyMDNiN2JiOWEzOWYwNDExOTZlMTI2MTg5NmM3NQ==\"",
            "\"rootHash\": \"not-authoritative\"",
            1,
        )
        .replacen(
            "\"treeSize\": \"1\"",
            "\"treeSize\": \"999999\"",
            1,
        );
    let verified = verify_official_fixture(&mutated).expect("duplicates are ignored");
    assert_eq!(
        verified.checkpoint_tree_size,
        baseline.checkpoint_tree_size
    );
    assert_eq!(verified.checkpoint_root, baseline.checkpoint_root);
    assert_eq!(verified.log_index, baseline.log_index);
}

#[test]
fn mutated_canonical_body_is_rejected() {
    let mutated = REKOR_V2_RESPONSE_JSON.replacen(
        "eyJhcGlWZXJzaW9u",
        "fyJhcGlWZXJzaW9u",
        1,
    );
    assert!(verify_official_fixture(&mutated).is_err());
}

#[test]
fn wrong_full_log_id_is_rejected_even_with_valid_checkpoint_signature() {
    let mutated = REKOR_V2_RESPONSE_JSON.replacen(
        EXPECTED_FULL_CHECKPOINT_KEY_ID_B64,
        "3AtEIMfG6Y41yK0tcwRTBS2tjhOrjKGIpDkHFgp65g0=",
        1,
    );
    assert_eq!(
        verify_official_fixture(&mutated),
        Err(FixtureError::KeyIdMismatch)
    );
}

#[test]
fn checkpoint_signature_mutation_is_rejected() {
    let response: Value = serde_json::from_str(REKOR_V2_RESPONSE_JSON).expect("fixture json");
    let envelope = json_str(
        &response,
        &["inclusionProof", "checkpoint", "envelope"],
    )
    .expect("checkpoint");
    let mutated = envelope.replacen("I23f9ylTh", "I23f9ylTi", 1);
    assert_eq!(
        verify_checkpoint(&mutated, &PUBLIC_KEY),
        Err(FixtureError::Signature)
    );
}

#[test]
fn wrong_origin_and_noncanonical_tree_size_fail_closed() {
    let response: Value = serde_json::from_str(REKOR_V2_RESPONSE_JSON).expect("fixture json");
    let envelope = json_str(
        &response,
        &["inclusionProof", "checkpoint", "envelope"],
    )
    .expect("checkpoint");

    let wrong_origin = envelope.replacen("rekor-local\n1\n", "rekor-other\n1\n", 1);
    assert!(verify_checkpoint(&wrong_origin, &PUBLIC_KEY).is_err());

    let noncanonical_size = envelope.replacen("rekor-local\n1\n", "rekor-local\n01\n", 1);
    assert_eq!(
        verify_checkpoint(&noncanonical_size, &PUBLIC_KEY),
        Err(FixtureError::NonCanonicalInteger)
    );
}

#[test]
fn strict_base64_rejects_whitespace_bad_padding_and_noncanonical_pad_bits() {
    assert_eq!(decode_base64("TQ==").expect("canonical"), b"M");
    assert_eq!(decode_base64("TWE=").expect("canonical"), b"Ma");
    assert_eq!(decode_base64("T W E="), Err(FixtureError::Base64));
    assert_eq!(decode_base64("TQ="), Err(FixtureError::Base64));
    assert_eq!(decode_base64("TR=="), Err(FixtureError::Base64));
    assert_eq!(decode_base64("TWF="), Err(FixtureError::Base64));
}

#[test]
fn source_scope_excludes_network_private_signing_and_authority_escalation_surfaces() {
    let source = include_str!("rekor_v2_fixture_verifier.rs");
    let forbidden = [
        ["req", "west::"].concat(),
        ["std::", "net::"].concat(),
        ["Tcp", "Stream"].concat(),
        ["Command", "::new"].concat(),
        ["Signing", "Key"].concat(),
        ["BEGIN PRIVATE", " KEY"].concat(),
        ["externality_verified:", " true"].concat(),
        ["production_trusted_root:", " true"].concat(),
        ["witnessed_consistency_verified:", " true"].concat(),
        ["execution_authority_granted:", " true"].concat(),
    ];
    for forbidden in forbidden {
        assert!(
            !source.contains(&forbidden),
            "forbidden surface: {forbidden}"
        );
    }
}

fn decode_32(text: &str) -> [u8; 32] {
    let bytes = decode_base64(text).expect("constant base64");
    bytes.try_into().expect("32-byte constant")
}
