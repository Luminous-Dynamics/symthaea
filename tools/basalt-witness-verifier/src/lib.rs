use base64::{Engine as _, engine::general_purpose::STANDARD};
use dsse::{
    Ed25519Verifier, Envelope, MAX_SIGNATURES, Signature, VerifiedPayload, Verifier, verify,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashSet};

pub const CHECKPOINT_PAYLOAD_TYPE: &str =
    "application/vnd.luminous.basalt.witness-checkpoint.v1+json";
pub const HEAD_RESPONSE_PAYLOAD_TYPE: &str =
    "application/vnd.luminous.basalt.witness-head-response.v1+json";
pub const WITNESS_SET_PAYLOAD_TYPE: &str =
    "application/vnd.luminous.basalt.witness-set-metadata.v1+json";
pub const TRUST_ROOT_PAYLOAD_TYPE: &str =
    "application/vnd.luminous.basalt.witness-trust-root.v1+json";

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TrustStore {
    pub schema_version: u32,
    pub authority_id: String,
    pub witness_set_id: String,
    pub witness_set_version: u64,
    pub trust_root_version: u64,
    pub fault_tolerance: usize,
    pub member_count: usize,
    pub write_quorum: usize,
    pub read_quorum: usize,
    pub members: Vec<TrustMember>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TrustMember {
    pub member_id: String,
    pub keyid: String,
    pub algorithm: String,
    pub public_key: String,
    pub administrative_domain: String,
    pub status: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct StrictEnvelope {
    #[serde(rename = "payloadType")]
    payload_type: String,
    payload: String,
    signatures: Vec<StrictSignature>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct StrictSignature {
    keyid: String,
    sig: String,
}

#[derive(Serialize)]
struct KeyIdPreimage<'a> {
    keytype: &'static str,
    public: &'a str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuorumKind {
    Read,
    Write,
}

#[derive(Clone, Debug, Serialize)]
pub struct VerificationReport {
    pub report_version: u32,
    pub verifier_version: String,
    pub status: String,
    pub payload_type: String,
    pub trust_store_sha256: String,
    pub envelope_sha256: String,
    pub payload_sha256: String,
    pub accepted_key_ids: Vec<String>,
    pub threshold_required: usize,
    pub claim_ceiling: String,
}

#[derive(Clone, Debug)]
pub struct VerifiedWitnessPayload {
    pub payload_type: String,
    pub payload: Vec<u8>,
    pub json: Value,
    pub accepted_key_ids: Vec<String>,
    pub trust_store_sha256: String,
    pub payload_sha256: String,
}

#[derive(Clone, Debug)]
struct PreparedTrust {
    verifiers: Vec<Ed25519Verifier>,
    trust_store_sha256: String,
    threshold_read: usize,
    threshold_write: usize,
}

pub fn sha256_prefixed(bytes: &[u8]) -> String {
    format!("sha256:{}", hex_lower(&Sha256::digest(bytes)))
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    hex_lower(&Sha256::digest(bytes))
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

fn is_lower_hex_64(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn canonical_key_id(public_key_base64: &str) -> Result<String, String> {
    let preimage = KeyIdPreimage {
        keytype: "ed25519",
        public: public_key_base64,
    };
    let canonical = serde_json_canonicalizer::to_vec(&preimage)
        .map_err(|error| format!("cannot canonicalize key-id preimage: {error}"))?;
    Ok(sha256_hex(&canonical))
}

fn decode_canonical_base64(encoded: &str, label: &str) -> Result<Vec<u8>, String> {
    let decoded = STANDARD
        .decode(encoded)
        .map_err(|error| format!("{label} is not valid standard Base64: {error}"))?;
    if STANDARD.encode(&decoded) != encoded {
        return Err(format!("{label} is not canonical padded Base64"));
    }
    Ok(decoded)
}

fn prepare_trust_store(
    trust: &TrustStore,
    trust_store_sha256: String,
) -> Result<PreparedTrust, String> {
    if trust.schema_version != 1 {
        return Err("unsupported trust-store schema_version".to_string());
    }
    if trust.authority_id.is_empty()
        || trust.witness_set_id.is_empty()
        || trust.witness_set_version == 0
        || trust.trust_root_version == 0
    {
        return Err("trust-store identity/version fields are invalid".to_string());
    }

    let expected_member_count = trust
        .fault_tolerance
        .checked_mul(3)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| "fault-tolerance arithmetic overflow".to_string())?;
    let expected_quorum = trust
        .fault_tolerance
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| "quorum arithmetic overflow".to_string())?;
    if trust.member_count != expected_member_count {
        return Err("member_count must equal 3*f+1".to_string());
    }
    if trust.write_quorum != expected_quorum || trust.read_quorum != expected_quorum {
        return Err("read_quorum and write_quorum must both equal 2*f+1".to_string());
    }

    let mut seen_member_ids = HashSet::new();
    let mut seen_key_ids = HashSet::new();
    let mut seen_public_keys = HashSet::new();
    let mut active_domains = BTreeSet::new();
    let mut verifiers = Vec::new();

    for member in &trust.members {
        if member.member_id.is_empty() || !seen_member_ids.insert(member.member_id.as_str()) {
            return Err(format!(
                "empty or duplicate witness member_id: {}",
                member.member_id
            ));
        }
        if !is_lower_hex_64(&member.keyid) || !seen_key_ids.insert(member.keyid.as_str()) {
            return Err(format!(
                "malformed or duplicate witness keyid: {}",
                member.keyid
            ));
        }
        if member.algorithm != "ed25519" {
            return Err(format!(
                "unsupported witness key algorithm: {}",
                member.algorithm
            ));
        }
        let public_bytes = decode_canonical_base64(&member.public_key, "witness public_key")?;
        if public_bytes.len() != 32 {
            return Err(format!(
                "witness public key {} is not 32 bytes",
                member.member_id
            ));
        }
        if !seen_public_keys.insert(public_bytes.clone()) {
            return Err(
                "the same Ed25519 public key appears under multiple member identities".to_string(),
            );
        }
        let computed_key_id = canonical_key_id(&member.public_key)?;
        if member.keyid != computed_key_id {
            return Err(format!(
                "keyid does not match the public key for {}",
                member.member_id
            ));
        }

        match member.status.as_str() {
            "active" => {
                if member.administrative_domain.is_empty()
                    || !active_domains.insert(member.administrative_domain.as_str())
                {
                    return Err(format!(
                        "active witness members must have distinct administrative domains: {}",
                        member.member_id
                    ));
                }
                let verifier = Ed25519Verifier::from_bytes(&public_bytes)
                    .map_err(|error| {
                        format!("invalid Ed25519 public key {}: {error}", member.member_id)
                    })?
                    .with_key_id(member.keyid.clone());
                verifiers.push(verifier);
            }
            "revoked" => {}
            other => {
                return Err(format!("unsupported member status {other}"));
            }
        }
    }

    if verifiers.len() != trust.member_count {
        return Err(format!(
            "active witness count {} differs from member_count {}",
            verifiers.len(),
            trust.member_count
        ));
    }
    if verifiers.len() < trust.write_quorum || verifiers.len() < trust.read_quorum {
        return Err("active witness set cannot satisfy the configured quorum".to_string());
    }

    Ok(PreparedTrust {
        verifiers,
        trust_store_sha256,
        threshold_read: trust.read_quorum,
        threshold_write: trust.write_quorum,
    })
}

/// Loads a trust store only when its exact bytes match an independently pinned digest.
///
/// The expected digest must come from provisioning or another trust channel outside the
/// database and backup set being verified. Reading a trust file and trusting its own
/// embedded digest is not a trust bootstrap.
pub fn load_pinned_trust_store(
    trust_bytes: &[u8],
    expected_sha256: &str,
) -> Result<TrustStore, String> {
    if !expected_sha256.starts_with("sha256:") || !is_lower_hex_64(&expected_sha256[7..]) {
        return Err("expected trust-store digest must be sha256:<64 lowercase hex>".to_string());
    }
    let actual = sha256_prefixed(trust_bytes);
    if actual != expected_sha256 {
        return Err(format!(
            "pinned trust-store digest mismatch: expected {expected_sha256}, got {actual}"
        ));
    }
    let trust: TrustStore = serde_json::from_slice(trust_bytes).map_err(|error| {
        format!("trust store is invalid or contains duplicate/unknown fields: {error}")
    })?;
    // Validate immediately so malformed key material cannot be reused by later calls.
    prepare_trust_store(&trust, actual)?;
    Ok(trust)
}

/// Verifies a DSSE envelope against active keys from an already pinned trust store.
///
/// The payload is returned only after DSSE threshold verification and byte-for-byte
/// RFC 8785 canonical JSON verification. Envelope key IDs remain unauthenticated hints;
/// only the verifier's trusted key identities count toward the quorum.
pub fn verify_witness_envelope(
    envelope_bytes: &[u8],
    trust_store_bytes: &[u8],
    expected_trust_store_sha256: &str,
    expected_payload_type: &str,
    quorum_kind: QuorumKind,
) -> Result<VerifiedWitnessPayload, String> {
    let trust = load_pinned_trust_store(trust_store_bytes, expected_trust_store_sha256)?;
    let actual_trust_digest = sha256_prefixed(trust_store_bytes);
    let prepared = prepare_trust_store(&trust, actual_trust_digest.clone())?;

    // Deserialize a strict local mirror first: Serde rejects duplicate struct fields,
    // and deny_unknown_fields prevents permissive envelope reinterpretation.
    let strict: StrictEnvelope = serde_json::from_slice(envelope_bytes)
        .map_err(|error| format!("invalid DSSE envelope JSON: {error}"))?;
    if strict.payload_type != expected_payload_type {
        return Err(format!(
            "unexpected DSSE payload type: expected {expected_payload_type}, got {}",
            strict.payload_type
        ));
    }
    if strict.signatures.is_empty() || strict.signatures.len() > MAX_SIGNATURES {
        return Err(format!(
            "DSSE signature count must be between 1 and {MAX_SIGNATURES}"
        ));
    }
    for signature in &strict.signatures {
        if !is_lower_hex_64(&signature.keyid) {
            return Err(
                "DSSE signature keyid must be 64 lowercase hexadecimal characters".to_string(),
            );
        }
        if signature.sig.is_empty() {
            return Err("DSSE signature bytes must not be empty".to_string());
        }
    }

    let envelope = Envelope {
        payload: strict.payload,
        payload_type: strict.payload_type,
        signatures: strict
            .signatures
            .into_iter()
            .map(|signature| Signature {
                keyid: Some(signature.keyid),
                sig: signature.sig,
            })
            .collect(),
    };

    let verifier_refs: Vec<&dyn Verifier> = prepared
        .verifiers
        .iter()
        .map(|verifier| verifier as &dyn Verifier)
        .collect();
    let threshold = match quorum_kind {
        QuorumKind::Read => prepared.threshold_read,
        QuorumKind::Write => prepared.threshold_write,
    };
    let verified: VerifiedPayload = verify(&envelope, &verifier_refs, threshold)
        .map_err(|error| format!("DSSE threshold verification failed: {error}"))?;

    let json: Value = serde_json::from_slice(&verified.payload)
        .map_err(|error| format!("verified payload is not strict JSON: {error}"))?;
    let canonical = serde_json_canonicalizer::to_vec(&json).map_err(|error| {
        format!("verified payload cannot be canonicalized as RFC 8785 JCS: {error}")
    })?;
    if canonical != verified.payload {
        return Err("signed payload is not byte-for-byte RFC 8785 canonical JSON".to_string());
    }

    Ok(VerifiedWitnessPayload {
        payload_type: verified.payload_type,
        payload_sha256: sha256_prefixed(&verified.payload),
        payload: verified.payload,
        json,
        accepted_key_ids: verified.accepted_keys,
        trust_store_sha256: prepared.trust_store_sha256,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use dsse::{Ed25519Signer, sign, sign_with};
    use ed25519_dalek::SigningKey;

    const TEST_PAYLOAD_TYPE: &str = CHECKPOINT_PAYLOAD_TYPE;
    const TEST_PAYLOAD: &[u8] =
        br#"{"authority_id":"basalt-test","authority_revision":"7","record_type":"basalt.authority.witness-checkpoint","schema_version":1}"#;

    fn test_seed(index: u8) -> [u8; 32] {
        [index; 32]
    }

    fn build_trust(seeds: &[[u8; 32]]) -> TrustStore {
        let members = seeds
            .iter()
            .enumerate()
            .map(|(index, seed)| {
                let signing_key = SigningKey::from_bytes(seed);
                let public_bytes = signing_key.verifying_key().to_bytes();
                let public_key = STANDARD.encode(public_bytes);
                let keyid = canonical_key_id(&public_key).unwrap();
                TrustMember {
                    member_id: format!("member-{index}"),
                    keyid,
                    algorithm: "ed25519".to_string(),
                    public_key,
                    administrative_domain: format!("admin-domain-{index}"),
                    status: "active".to_string(),
                }
            })
            .collect::<Vec<_>>();
        let f = (seeds.len() - 1) / 3;
        let quorum = 2 * f + 1;
        TrustStore {
            schema_version: 1,
            authority_id: "basalt-test-authority".to_string(),
            witness_set_id: "basalt-test-witness-set".to_string(),
            witness_set_version: 1,
            trust_root_version: 1,
            fault_tolerance: f,
            member_count: seeds.len(),
            write_quorum: quorum,
            read_quorum: quorum,
            members,
        }
    }

    fn trust_bytes(trust: &TrustStore) -> Vec<u8> {
        serde_json::to_vec(trust).unwrap()
    }

    fn sign_payload(payload_type: &str, payload: &[u8], seeds: &[[u8; 32]]) -> Vec<u8> {
        let signers = seeds
            .iter()
            .map(|seed| {
                let signing_key = SigningKey::from_bytes(seed);
                let public_bytes = signing_key.verifying_key().to_bytes();
                let public_key = STANDARD.encode(public_bytes);
                let keyid = canonical_key_id(&public_key).unwrap();
                Ed25519Signer::from_bytes(seed).unwrap().with_key_id(keyid)
            })
            .collect::<Vec<_>>();
        let signer_refs: Vec<&dyn dsse::Signer> = signers
            .iter()
            .map(|signer| signer as &dyn dsse::Signer)
            .collect();
        let envelope = sign_with(payload_type, payload, &signer_refs).unwrap();
        envelope.to_json().unwrap()
    }

    #[test]
    fn pae_matches_independent_checked_in_vector() {
        let payload_type = TEST_PAYLOAD_TYPE;
        let payload =
            br#"{"record_type":"basalt.authority.witness-checkpoint","schema_version":1}"#;
        let pae_bytes = dsse::pae(payload_type, payload);
        assert_eq!(pae_bytes.len(), 144);
        assert_eq!(
            String::from_utf8(pae_bytes).unwrap(),
            "DSSEv1 58 application/vnd.luminous.basalt.witness-checkpoint.v1+json 72 {\"record_type\":\"basalt.authority.witness-checkpoint\",\"schema_version\":1}"
        );
    }

    #[test]
    fn valid_three_of_four_dsse_quorum_verifies_and_returns_exact_payload() {
        let seeds = [test_seed(7), test_seed(8), test_seed(9), test_seed(10)];
        let trust = build_trust(&seeds);
        let trust_bytes = trust_bytes(&trust);
        let pin = sha256_prefixed(&trust_bytes);
        let envelope = sign_payload(TEST_PAYLOAD_TYPE, TEST_PAYLOAD, &seeds[..3]);
        let verified = verify_witness_envelope(
            &envelope ,
            &trust_bytes,
            &pin,
            TEST_PAYLOAD_TYPE,
            QuorumKind::Write,
        )
        .unwrap();
        assert_eq!(verified.payload, TEST_PAYLOAD);
        assert_eq!(verified.accepted_key_ids.len(), 3);
    }

    #[test]
    fn two_of_four_is_below_threshold() {
        let seeds = [test_seed(7), test_seed(8), test_seed(9), test_seed(10)];
        let trust = build_trust(&seeds);
        let bytes = trust_bytes(&trust);
        let pin = sha256_prefixed(&bytes);
        let envelope = sign_payload(TEST_PAYLOAD_TYPE, TEST_PAYLOAD, &seeds[..2]);
        assert!(
            verify_witness_envelope(
                &envelope ,
                &bytes,
                &pin,
                TEST_PAYLOAD_TYPE,
                QuorumKind::Write
            )
            .is_err()
        );
    }

    #[test]
    fn duplicated_signature_from_one_member_does_not_satisfy_threshold() {
        let seeds = [test_seed(7), test_seed(8), test_seed(9), test_seed(10)];
        let trust = build_trust(&seeds);
        let bytes = trust_bytes(&trust);
        let pin = sha256_prefixed(&bytes);
        let mut envelope = sign(
            TEST_PAYLOAD_TYPE,
            TEST_PAYLOAD,
            &Ed25519Signer::from_bytes(&seeds[0])
                .unwrap()
                .with_key_id(trust.members[0].keyid.clone()),
        )
        .unwrap();
        let duplicate = envelope.signatures[0].clone();
        envelope.signatures = vec![duplicate.clone(), duplicate.clone(), duplicate];
        let raw = envelope.to_json().unwrap();
        assert!(
            verify_witness_envelope(
                &raw ,
                &bytes,
                &pin,
                TEST_PAYLOAD_TYPE,
                QuorumKind::Write
            )
            .is_err()
        );
    }

    #[test]
    fn wrong_payload_type_cannot_be_relabelled() {
        let seeds = [test_seed(7), test_seed(8), test_seed(9), test_seed(10)];
        let trust = build_trust(&seeds);
        let bytes = trust_bytes(&trust);
        let pin = sha256_prefixed(&bytes);
        let envelope = sign_payload(HEAD_RESPONSE_PAYLOAD_TYPE, TEST_PAYLOAD, &seeds[..3]);
        assert!(
            verify_witness_envelope(
                &envelope ,
                &bytes,
                &pin,
                TEST_PAYLOAD_TYPE,
                QuorumKind::Write
            )
            .is_err()
        );
    }

    #[test]
    fn payload_tampering_breaks_signature() {
        let seeds = [test_seed(7), test_seed(8), test_seed(9), test_seed(10)];
        let trust = build_trust(&seeds);
        let bytes = trust_bytes(&trust);
        let pin = sha256_prefixed(&bytes);
        let mut envelope: StrictEnvelope =
            serde_json::from_slice(&sign_payload(TEST_PAYLOAD_TYPE, TEST_PAYLOAD, &seeds[..3]))
                .unwrap();
        envelope.payload = STANDARD.encode(br#"{"authority_revision":"8"}"#);
        let raw = serde_json::to_vec(&serde_json::json!({
            "payloadType": envelope.payload_type, "payload": envelope.payload,
            "signatures": envelope.signatures.iter().map(|x| serde_json::json!({"keyid":x.keyid,"sig":x.sig})).collect::<Vec<_>>()
        })).unwrap();
        assert!(
            verify_witness_envelope(
                &raw ,
                &bytes,
                &pin,
                TEST_PAYLOAD_TYPE,
                QuorumKind::Write
            )
            .is_err()
        );
    }

    #[test]
    fn signed_but_noncanonical_payload_is_rejected_after_signature_verification() {
        let seeds = [test_seed(7), test_seed(8), test_seed(9), test_seed(10)];
        let trust = build_trust(&seeds);
        let bytes = trust_bytes(&trust);
        let pin = sha256_prefixed(&bytes);
        let noncanonical = br#"{ "schema_version": 1, "record_type": "basalt.authority.witness-checkpoint", "authority_id": "basalt-test", "authority_revision": "7" }"#;
        let envelope = sign_payload(TEST_PAYLOAD_TYPE, noncanonical, &seeds[..3]);
        assert!(
            verify_witness_envelope(
                &envelope ,
                &bytes,
                &pin,
                TEST_PAYLOAD_TYPE,
                QuorumKind::Write
            )
            .is_err()
        );
    }

    #[test]
    fn duplicate_json_envelope_field_is_rejected() {
        let seeds = [test_seed(7), test_seed(8), test_seed(9), test_seed(10)];
        let trust = build_trust(&seeds);
        let bytes = trust_bytes(&trust);
        let pin = sha256_prefixed(&bytes);
        let good = sign_payload(TEST_PAYLOAD_TYPE, TEST_PAYLOAD, &seeds[..3]);
        let good = String::from_utf8(good).unwrap();
        let duplicate = good.replacen(
            "\"payloadType\":",
            "\"payloadType\":\"attacker-type\",\"payloadType\":",
            1,
        );
        assert!(
            verify_witness_envelope(
                duplicate.as_bytes() ,
                &bytes,
                &pin,
                TEST_PAYLOAD_TYPE,
                QuorumKind::Write
            )
            .is_err()
        );
    }

    #[test]
    fn duplicate_object_member_inside_signed_payload_is_rejected_by_jcs_equality() {
        let seeds = [test_seed(7), test_seed(8), test_seed(9), test_seed(10)];
        let trust = build_trust(&seeds);
        let bytes = trust_bytes(&trust);
        let pin = sha256_prefixed(&bytes);
        let payload = br#"{"authority_revision":"7","authority_revision":"8"}"#;
        let envelope = sign_payload(TEST_PAYLOAD_TYPE, payload, &seeds[..3]);
        assert!(
            verify_witness_envelope(
                &envelope ,
                &bytes,
                &pin,
                TEST_PAYLOAD_TYPE,
                QuorumKind::Write
            )
            .is_err()
        );
    }

    #[test]
    fn revoked_member_cannot_contribute_to_quorum() {
        let seeds = [test_seed(7), test_seed(8), test_seed(9), test_seed(10)];
        let mut trust = build_trust(&seeds);
        trust.members[3].status = "revoked".to_string();
        let replacement_seed = test_seed(11);
        let replacement_key = SigningKey::from_bytes(&replacement_seed);
        let replacement_public = STANDARD.encode(replacement_key.verifying_key().to_bytes());
        trust.members.push(TrustMember {
            member_id: "member-replacement".to_string(),
            keyid: canonical_key_id(&replacement_public).unwrap(),
            algorithm: "ed25519".to_string(),
            public_key: replacement_public,
            administrative_domain: "admin-domain-replacement".to_string(),
            status: "active".to_string(),
        });
        // Four active keys remain in the new set; an envelope containing only two
        // active signatures plus the revoked member's signature must not pass 3-of-4.
        let bytes = trust_bytes(&trust);
        let pin = sha256_prefixed(&bytes);
        let signers = [
            Ed25519Signer::from_bytes(&seeds[0])
                .unwrap()
                .with_key_id(trust.members[0].keyid.clone()),
            Ed25519Signer::from_bytes(&seeds[1])
                .unwrap()
                .with_key_id(trust.members[1].keyid.clone()),
            Ed25519Signer::from_bytes(&seeds[3])
                .unwrap()
                .with_key_id(trust.members[3].keyid.clone()),
        ];
        let signer_refs: Vec<&dyn dsse::Signer> = signers
            .iter()
            .map(|signer| signer as &dyn dsse::Signer)
            .collect();
        let envelope = sign_with(TEST_PAYLOAD_TYPE, TEST_PAYLOAD, &signer_refs)
            .unwrap()
            .to_json()
            .unwrap();
        assert!(
            verify_witness_envelope(
                &envelope ,
                &bytes,
                &pin,
                TEST_PAYLOAD_TYPE,
                QuorumKind::Write
            )
            .is_err()
        );
    }

    #[test]
    fn duplicate_public_key_under_second_member_identity_is_rejected() {
        let seeds = [test_seed(7), test_seed(8), test_seed(9), test_seed(10)];
        let mut trust = build_trust(&seeds);
        trust.members[1].public_key = trust.members[0].public_key.clone();
        trust.members[1].keyid = trust.members[0].keyid.clone();
        let bytes = trust_bytes(&trust);
        let pin = sha256_prefixed(&bytes);
        assert!(load_pinned_trust_store(&bytes, &pin).is_err());
    }

    #[test]
    fn duplicate_administrative_domain_is_rejected() {
        let seeds = [test_seed(7), test_seed(8), test_seed(9), test_seed(10)];
        let mut trust = build_trust(&seeds);
        trust.members[1].administrative_domain = trust.members[0].administrative_domain.clone();
        let bytes = trust_bytes(&trust);
        let pin = sha256_prefixed(&bytes);
        assert!(load_pinned_trust_store(&bytes, &pin).is_err());
    }

    #[test]
    fn wrong_out_of_band_pin_is_rejected_before_parsing() {
        let seeds = [test_seed(7), test_seed(8), test_seed(9), test_seed(10)];
        let trust = build_trust(&seeds);
        let bytes = trust_bytes(&trust);
        assert!(
            load_pinned_trust_store(
                &bytes,
                "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            )
            .is_err()
        );
    }
}
