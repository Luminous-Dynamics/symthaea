use base64::{Engine as _, engine::general_purpose::STANDARD};
use dsse::{
    Ed25519Verifier, Envelope, MAX_SIGNATURES, Signature, VerifiedPayload, Verifier, verify,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashSet};

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
    /// One authenticated member response. Aggregators must still count distinct keys.
    Member,
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

fn is_prefixed_sha256(value: &str) -> bool {
    value.strip_prefix("sha256:").is_some_and(is_lower_hex_64)
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

    if quorum_kind == QuorumKind::Member && strict.signatures.len() != 1 {
        return Err("member response must contain exactly one signature".to_string());
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
        QuorumKind::Member => 1,
        QuorumKind::Read => prepared.threshold_read,
        QuorumKind::Write => prepared.threshold_write,
    };
    let verified: VerifiedPayload = verify(&envelope, &verifier_refs, threshold)
        .map_err(|error| format!("DSSE threshold verification failed: {error}"))?;

    if quorum_kind == QuorumKind::Member && verified.accepted_keys.len() != 1 {
        return Err("member response must authenticate exactly one trusted member key".to_string());
    }
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

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WitnessHeadResponse {
    pub record_type: String,
    pub schema_version: u32,
    pub challenge_nonce: String,
    pub authority_id: String,
    pub authority_incarnation_id: String,
    pub authority_epoch: String,
    pub witness_set_id: String,
    pub witness_set_version: String,
    pub trust_root_version: String,
    pub head_sequence: String,
    pub head_checkpoint_digest: String,
    pub head_authority_revision: String,
    pub head_state_digest: String,
    pub head_policy_digest: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WitnessCheckpoint {
    pub record_type: String,
    pub schema_version: u32,
    pub authority_id: String,
    pub authority_incarnation_id: String,
    pub authority_epoch: String,
    pub sequence: String,
    pub authority_revision: String,
    pub state_schema: String,
    pub state_digest: String,
    pub policy_revision: String,
    pub policy_digest: String,
    pub previous_checkpoint_digest: Option<String>,
    pub witness_set_id: String,
    pub witness_set_version: String,
    pub trust_root_version: String,
    pub transition_bundle_digest: Option<String>,
}

#[derive(Clone, Debug)]
pub struct WitnessHeadEvidence {
    pub response_envelope: Vec<u8>,
    pub checkpoint_envelope: Vec<u8>,
}

#[derive(Clone, Debug, Serialize)]
pub struct SelectedWitnessHead {
    pub status: String,
    pub authority_id: String,
    pub authority_incarnation_id: String,
    pub authority_epoch: String,
    pub sequence: String,
    pub authority_revision: String,
    pub checkpoint_digest: String,
    pub state_digest: String,
    pub policy_digest: String,
    pub transition_bundle_digest: Option<String>,
    pub distinct_member_responses: usize,
    pub read_quorum: usize,
    pub accepted_member_key_ids: Vec<String>,
    pub response_set_sha256: String,
    pub counterexamples: Vec<String>,
    pub claim_ceiling: String,
}

fn parse_canonical_u64(value: &str, label: &str) -> Result<u64, String> {
    let parsed = value
        .parse::<u64>()
        .map_err(|error| format!("{label} is not a u64 decimal string: {error}"))?;
    if parsed.to_string() != value {
        return Err(format!("{label} is not a canonical decimal string"));
    }
    Ok(parsed)
}

#[derive(Clone, Debug)]
struct AcceptedWitnessHead {
    response: WitnessHeadResponse,
    checkpoint_digest: String,
    checkpoint: WitnessCheckpoint,
}

fn checkpoint_digest_from_payload(payload: &[u8]) -> String {
    let mut preimage = b"BASALT-WITNESS-CHECKPOINT-V1\0".to_vec();
    preimage.extend_from_slice(payload);
    sha256_prefixed(&preimage)
}

fn rejected_head_report(status: &str, message: String, read_quorum: usize) -> SelectedWitnessHead {
    SelectedWitnessHead {
        status: status.to_string(),
        authority_id: String::new(),
        authority_incarnation_id: String::new(),
        authority_epoch: String::new(),
        sequence: String::new(),
        authority_revision: String::new(),
        checkpoint_digest: String::new(),
        state_digest: String::new(),
        policy_digest: String::new(),
        transition_bundle_digest: None,
        distinct_member_responses: 0,
        read_quorum,
        accepted_member_key_ids: Vec::new(),
        response_set_sha256: sha256_prefixed(b""),
        counterexamples: vec![message],
        claim_ceiling: "No witness head selected; freshness is unknown or rejected.".to_string(),
    }
}

/// Verifies a nonce-bound read quorum and selects the highest write-quorum-certified
/// checkpoint head, allowing honest replicas to lag. It does not verify the complete
/// predecessor chain to genesis or trust-root rotation.
pub fn verify_head_read_quorum(
    evidence: &[WitnessHeadEvidence],
    trust_store_bytes: &[u8],
    expected_trust_store_sha256: &str,
    expected_challenge_nonce: &str,
    expected_authority_incarnation_id: &str,
    expected_authority_epoch: &str,
) -> SelectedWitnessHead {
    let trust = match load_pinned_trust_store(trust_store_bytes, expected_trust_store_sha256) {
        Ok(value) => value,
        Err(error) => return rejected_head_report("rejected_trust_store", error, 0),
    };
    let quorum = trust.read_quorum;
    let nonce_is_valid = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(expected_challenge_nonce)
        .is_ok_and(|value| {
            value.len() == 32
                && base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(&value)
                    == expected_challenge_nonce
        });
    if !nonce_is_valid {
        return rejected_head_report(
            "invalid_challenge",
            "challenge nonce must be canonical unpadded Base64URL for exactly 32 bytes".to_string(),
            quorum,
        );
    }
    if parse_canonical_u64(expected_authority_epoch, "expected authority epoch").is_err()
        || expected_authority_incarnation_id.is_empty()
    {
        return rejected_head_report(
            "invalid_challenge",
            "expected authority incarnation/epoch is malformed".to_string(),
            quorum,
        );
    }

    if evidence.len() < quorum {
        return rejected_head_report(
            "anti_rollback_unknown",
            format!(
                "only {} member responses supplied; read quorum requires {quorum}",
                evidence.len()
            ),
            quorum,
        );
    }

    let mut accepted = BTreeMap::<String, AcceptedWitnessHead>::new();
    let mut reported_heads = BTreeMap::<String, String>::new();
    let mut certified_by_sequence = BTreeMap::<u64, String>::new();
    let mut response_entries = Vec::<(String, String, String)>::new();
    let mut response_rejections = Vec::<String>::new();

    macro_rules! reject_response {
        ($response_index:expr, $status:expr, $message:expr) => {{
            response_rejections.push(format!(
                "response {} {}: {}",
                $response_index,
                $status,
                $message
            ));
            continue;
        }};
    }

    for (index, item) in evidence.iter().enumerate() {
        let response = match verify_witness_envelope(
            &item.response_envelope,
            trust_store_bytes,
            expected_trust_store_sha256,
            HEAD_RESPONSE_PAYLOAD_TYPE,
            QuorumKind::Member,
        ) {
            Ok(value) => value,
            Err(error) => {
                reject_response!(index, "rejected_witness_response", format!("response {index} failed authentication: {error}"));
            }
        };
        let response_payload: WitnessHeadResponse =
            match serde_json::from_value(response.json.clone()) {
                Ok(value) => value,
                Err(error) => {
                    reject_response!(index, "rejected_witness_response", format!("response {index} payload shape invalid: {error}"));
                }
            };
        if response_payload.record_type != "basalt.authority.witness-head-response"
            || response_payload.schema_version != 1
            || response_payload.challenge_nonce != expected_challenge_nonce
            || response_payload.authority_id != trust.authority_id
            || response_payload.authority_incarnation_id != expected_authority_incarnation_id
            || response_payload.authority_epoch != expected_authority_epoch
            || response_payload.witness_set_id != trust.witness_set_id
            || response_payload.witness_set_version != trust.witness_set_version.to_string()
            || response_payload.trust_root_version != trust.trust_root_version.to_string()
        {
            reject_response!(index, "rejected_witness_response", format!(
                    "response {index} is not bound to the expected nonce/authority/witness set"
                ));
        }
        if let Err(error) = parse_canonical_u64(
            &response_payload.head_authority_revision,
            "head_authority_revision",
        ) {
            reject_response!(index, "rejected_witness_response", error);
        }
        if let Err(error) = parse_canonical_u64(&response_payload.head_sequence, "head_sequence") {
            reject_response!(index, "rejected_witness_response", error);
        }
        if !is_prefixed_sha256(&response_payload.head_checkpoint_digest)
            || !is_prefixed_sha256(&response_payload.head_state_digest)
            || !is_prefixed_sha256(&response_payload.head_policy_digest)
        {
            reject_response!(index, "rejected_witness_response", format!("response {index} contains a malformed digest"));
        }
        let Some(member_key_id) = response.accepted_key_ids.first().cloned() else {
            reject_response!(index, "rejected_witness_response", format!("response {index} has no authenticated member key"));
        };

        if let Some(previous_digest) = reported_heads.get(&member_key_id) {
            if previous_digest != &response_payload.head_checkpoint_digest {
                return rejected_head_report(
                    "rejected_witness_equivocation",
                    format!("member {member_key_id} signed different heads for one challenge"),
                    quorum,
                );
            }
        } else {
            reported_heads.insert(
                member_key_id.clone(),
                response_payload.head_checkpoint_digest.clone(),
            );
        }

        let checkpoint = match verify_witness_envelope(
            &item.checkpoint_envelope,
            trust_store_bytes,
            expected_trust_store_sha256,
            CHECKPOINT_PAYLOAD_TYPE,
            QuorumKind::Write,
        ) {
            Ok(value) => value,
            Err(error) => {
                reject_response!(index, "rejected_checkpoint_certificate", format!("response {index} checkpoint certificate failed: {error}"));
            }
        };
        let checkpoint_payload: WitnessCheckpoint =
            match serde_json::from_value(checkpoint.json.clone()) {
                Ok(value) => value,
                Err(error) => {
                    reject_response!(index, "rejected_checkpoint_certificate", format!("response {index} checkpoint payload shape invalid: {error}"));
                }
            };
        if checkpoint_payload.state_schema != "basalt.authority-state.v1" {
            reject_response!(index, "rejected_checkpoint_certificate", format!("response {index} checkpoint uses an unsupported state schema"));
        }
        let checkpoint_sequence =
            match parse_canonical_u64(&checkpoint_payload.sequence, "checkpoint sequence") {
                Ok(value) => value,
                Err(error) => {
                    reject_response!(index, "rejected_checkpoint_certificate", error);
                }
            };
        if parse_canonical_u64(
            &checkpoint_payload.authority_revision,
            "checkpoint authority revision",
        )
        .is_err()
            || parse_canonical_u64(
                &checkpoint_payload.policy_revision,
                "checkpoint policy revision",
            )
            .is_err()
            || checkpoint_payload
                .previous_checkpoint_digest
                .as_deref()
                .is_some_and(|value| !is_prefixed_sha256(value))
            || checkpoint_payload
                .transition_bundle_digest
                .as_deref()
                .is_some_and(|value| !is_prefixed_sha256(value))
        {
            reject_response!(index, "rejected_checkpoint_certificate", format!("response {index} checkpoint contains malformed revision/digest fields"));
        }
        if checkpoint_payload.record_type != "basalt.authority.witness-checkpoint"
            || checkpoint_payload.schema_version != 1
            || checkpoint_payload.authority_id != trust.authority_id
            || checkpoint_payload.authority_incarnation_id != expected_authority_incarnation_id
            || checkpoint_payload.authority_epoch != expected_authority_epoch
            || checkpoint_payload.witness_set_id != trust.witness_set_id
            || checkpoint_payload.witness_set_version != trust.witness_set_version.to_string()
            || checkpoint_payload.trust_root_version != trust.trust_root_version.to_string()
            || checkpoint_payload.sequence != response_payload.head_sequence
            || checkpoint_payload.authority_revision != response_payload.head_authority_revision
            || checkpoint_payload.state_digest != response_payload.head_state_digest
            || checkpoint_payload.policy_digest != response_payload.head_policy_digest
        {
            reject_response!(index, "rejected_checkpoint_certificate", format!("response {index} fields disagree with its write-quorum checkpoint"));
        }
        let computed_checkpoint_digest = checkpoint_digest_from_payload(&checkpoint.payload);
        if computed_checkpoint_digest != response_payload.head_checkpoint_digest {
            reject_response!(index, "rejected_checkpoint_certificate", format!("response {index} checkpoint digest does not bind canonical payload"));
        }

        if let Some(existing_digest) = certified_by_sequence.get(&checkpoint_sequence) {
            if existing_digest != &computed_checkpoint_digest {
                return rejected_head_report(
                    "rejected_witness_equivocation",
                    format!(
                        "two write-quorum certificates exist for sequence {checkpoint_sequence}"
                    ),
                    quorum,
                );
            }
        } else {
            certified_by_sequence.insert(checkpoint_sequence, computed_checkpoint_digest.clone());
        }

        if accepted.contains_key(&member_key_id) {
            // Exact duplicate responses from one member do not increase the quorum.
            continue;
        }

        let response_digest = sha256_prefixed(&item.response_envelope);
        response_entries.push((
            member_key_id.clone(),
            response_digest,
            computed_checkpoint_digest.clone(),
        ));
        accepted.insert(
            member_key_id,
            AcceptedWitnessHead {
                response: response_payload,
                checkpoint_digest: computed_checkpoint_digest,
                checkpoint: checkpoint_payload,
            },
        );
    }

    if accepted.len() < quorum {
        let mut report = rejected_head_report(
            "anti_rollback_unknown",
            format!(
                "{} distinct members authenticated; read quorum requires {quorum}",
                accepted.len()
            ),
            quorum,
        );
        report.counterexamples.extend(response_rejections);
        return report;
    }

    let Some((sequence, selected_digest)) = certified_by_sequence.iter().next_back() else {
        return rejected_head_report(
            "anti_rollback_unknown",
            "no certified checkpoint head found".to_string(),
            quorum,
        );
    };
    let Some(selected) = accepted.values().find(|entry| {
        entry.response.head_sequence == sequence.to_string()
            && &entry.checkpoint_digest == selected_digest
    }) else {
        return rejected_head_report(
            "anti_rollback_unknown",
            "selected checkpoint has no matching response evidence".to_string(),
            quorum,
        );
    };
    let selected_response = &selected.response;
    let selected_checkpoint = &selected.checkpoint;

    response_entries.sort_by(|left, right| left.0.cmp(&right.0));
    let response_set_json = match serde_json_canonicalizer::to_vec(&response_entries) {
        Ok(value) => value,
        Err(error) => {
            return rejected_head_report(
                "invalid_evidence",
                format!("cannot canonicalize response set: {error}"),
                quorum,
            );
        }
    };
    let accepted_keys = accepted.keys().cloned().collect::<Vec<_>>();
    SelectedWitnessHead {
        status: "fresh_head_selected".to_string(),
        authority_id: selected_response.authority_id.clone(),
        authority_incarnation_id: selected_response.authority_incarnation_id.clone(),
        authority_epoch: selected_response.authority_epoch.clone(),
        sequence: selected_response.head_sequence.clone(),
        authority_revision: selected_response.head_authority_revision.clone(),
        checkpoint_digest: selected_digest.clone(),
        state_digest: selected_response.head_state_digest.clone(),
        policy_digest: selected_response.head_policy_digest.clone(),
        transition_bundle_digest: selected_checkpoint.transition_bundle_digest.clone(),
        distinct_member_responses: accepted.len(),
        read_quorum: quorum,
        accepted_member_key_ids: accepted_keys,
        response_set_sha256: sha256_prefixed(&response_set_json),
        counterexamples: response_rejections,
        claim_ceiling: "Selected the highest nonce-bound write-quorum-certified checkpoint reported by a distinct read quorum. Invalid extra responses were excluded and recorded. Predecessor-chain-to-genesis, trust-root rotation, real member durability and physical anti-rollback are not verified.".to_string(),
    }
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

    fn test_nonce() -> String {
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode([42_u8; 32])
    }

    fn checkpoint_fixture(sequence: &str, state_byte: u8) -> WitnessCheckpoint {
        WitnessCheckpoint {
            record_type: "basalt.authority.witness-checkpoint".to_string(),
            schema_version: 1,
            authority_id: "basalt-test-authority".to_string(),
            authority_incarnation_id: "incarnation-1".to_string(),
            authority_epoch: "1".to_string(),
            sequence: sequence.to_string(),
            authority_revision: (sequence.parse::<u64>().unwrap() + 7).to_string(),
            state_schema: "basalt.authority-state.v1".to_string(),
            state_digest: format!("sha256:{}", format!("{state_byte:02x}").repeat(32)),
            policy_revision: "1".to_string(),
            policy_digest: format!("sha256:{}", "ab".repeat(32)),
            previous_checkpoint_digest: Some(format!("sha256:{}", "cd".repeat(32))),
            witness_set_id: "basalt-test-witness-set".to_string(),
            witness_set_version: "1".to_string(),
            trust_root_version: "1".to_string(),
            transition_bundle_digest: Some(format!("sha256:{}", "ef".repeat(32))),
        }
    }

    fn sign_checkpoint_fixture(checkpoint: &WitnessCheckpoint, seeds: &[[u8; 32]]) -> Vec<u8> {
        let canonical = serde_json_canonicalizer::to_vec(checkpoint).unwrap();
        sign_payload(CHECKPOINT_PAYLOAD_TYPE, &canonical, seeds)
    }

    fn sign_head_response_fixture(
        checkpoint: &WitnessCheckpoint,
        nonce: &str,
        seed: &[u8; 32],
    ) -> Vec<u8> {
        let checkpoint_canonical = serde_json_canonicalizer::to_vec(checkpoint).unwrap();
        let response = WitnessHeadResponse {
            record_type: "basalt.authority.witness-head-response".to_string(),
            schema_version: 1,
            challenge_nonce: nonce.to_string(),
            authority_id: checkpoint.authority_id.clone(),
            authority_incarnation_id: checkpoint.authority_incarnation_id.clone(),
            authority_epoch: checkpoint.authority_epoch.clone(),
            witness_set_id: checkpoint.witness_set_id.clone(),
            witness_set_version: checkpoint.witness_set_version.clone(),
            trust_root_version: checkpoint.trust_root_version.clone(),
            head_sequence: checkpoint.sequence.clone(),
            head_checkpoint_digest: checkpoint_digest_from_payload(&checkpoint_canonical),
            head_authority_revision: checkpoint.authority_revision.clone(),
            head_state_digest: checkpoint.state_digest.clone(),
            head_policy_digest: checkpoint.policy_digest.clone(),
        };
        let canonical = serde_json_canonicalizer::to_vec(&response).unwrap();
        sign_payload(
            HEAD_RESPONSE_PAYLOAD_TYPE,
            &canonical,
            std::slice::from_ref(seed),
        )
    }

    fn evidence_fixture(
        checkpoint: &WitnessCheckpoint,
        seed: &[u8; 32],
        nonce: &str,
        checkpoint_signers: &[[u8; 32]],
    ) -> WitnessHeadEvidence {
        WitnessHeadEvidence {
            response_envelope: sign_head_response_fixture(checkpoint, nonce, seed),
            checkpoint_envelope: sign_checkpoint_fixture(checkpoint, checkpoint_signers),
        }
    }

    fn read_fixture() -> (Vec<u8>, String, Vec<WitnessHeadEvidence>) {
        let seeds = [test_seed(7), test_seed(8), test_seed(9), test_seed(10)];
        let trust_bytes = trust_bytes(&build_trust(&seeds));
        let pin = sha256_prefixed(&trust_bytes);
        let nonce = test_nonce();
        let latest = checkpoint_fixture("5", 0x55);
        let lagging = checkpoint_fixture("4", 0x44);
        let evidence = vec![
            evidence_fixture(&latest, &seeds[0], &nonce, &seeds[..3]),
            evidence_fixture(&lagging, &seeds[1], &nonce, &seeds[..3]),
            evidence_fixture(&latest, &seeds[2], &nonce, &seeds[..3]),
        ];
        (trust_bytes, pin, evidence)
    }

    #[test]
    fn nonce_bound_read_selects_highest_certified_head_with_lagging_replicas() {
        let (trust_bytes, pin, evidence) = read_fixture();
        let selected = verify_head_read_quorum(
            &evidence,
            &trust_bytes,
            &pin,
            &test_nonce(),
            "incarnation-1",
            "1",
        );
        assert_eq!(selected.status, "fresh_head_selected");
        assert_eq!(selected.sequence, "5");
        assert_eq!(selected.read_quorum, 3);
        assert_eq!(selected.distinct_member_responses, 3);
        assert_eq!(selected.accepted_member_key_ids.len(), 3);
        assert!(selected.counterexamples.is_empty());
    }

    #[test]
    fn one_invalid_extra_response_does_not_veto_a_valid_read_quorum() {
        let (trust_bytes, pin, mut evidence) = read_fixture();
        let seeds = [test_seed(7), test_seed(8), test_seed(9), test_seed(10)];
        let nonce = test_nonce();
        let latest = checkpoint_fixture("5", 0x55);
        // This response is correctly signed but its key is absent from the pinned trust store.
        evidence.push(evidence_fixture(
            &latest,
            &test_seed(12),
            &nonce,
            &seeds[..3],
        ));

        let selected = verify_head_read_quorum(
            &evidence,
            &trust_bytes,
            &pin,
            &nonce,
            "incarnation-1",
            "1",
        );
        assert_eq!(selected.status, "fresh_head_selected");
        assert_eq!(selected.distinct_member_responses, 3);
        assert_eq!(selected.sequence, "5");
        assert_eq!(selected.counterexamples.len(), 1);
        assert!(selected.counterexamples[0].contains("failed authentication"));
    }

    #[test]
    fn invalid_extra_responses_are_reported_when_the_valid_quorum_is_short() {
        let (trust_bytes, pin, mut evidence) = read_fixture();
        evidence.truncate(2);
        let nonce = test_nonce();
        let latest = checkpoint_fixture("5", 0x55);
        evidence.push(evidence_fixture(
            &latest,
            &test_seed(12),
            &nonce,
            &[test_seed(7), test_seed(8), test_seed(9)],
        ));

        let selected = verify_head_read_quorum(
            &evidence,
            &trust_bytes,
            &pin,
            &nonce,
            "incarnation-1",
            "1",
        );
        assert_eq!(selected.status, "anti_rollback_unknown");
        assert_eq!(selected.distinct_member_responses, 0);
        assert!(selected.counterexamples.len() >= 1);
        assert!(selected.counterexamples[0].contains("failed authentication"));
    }

    #[test]
    fn read_quorum_shortfall_is_anti_rollback_unknown() {
        let (trust_bytes, pin, mut evidence) = read_fixture();
        evidence.truncate(2);
        let selected = verify_head_read_quorum(
            &evidence,
            &trust_bytes,
            &pin,
            &test_nonce(),
            "incarnation-1",
            "1",
        );
        assert_eq!(selected.status, "anti_rollback_unknown");
        assert_eq!(selected.distinct_member_responses, 0);
    }

    #[test]
    fn repeated_member_response_never_counts_twice() {
        let (trust_bytes, pin, mut evidence) = read_fixture();
        evidence[2] = evidence[0].clone();
        let selected = verify_head_read_quorum(
            &evidence,
            &trust_bytes,
            &pin,
            &test_nonce(),
            "incarnation-1",
            "1",
        );
        assert_eq!(selected.status, "anti_rollback_unknown");
        assert_eq!(selected.distinct_member_responses, 0);
    }

    #[test]
    fn same_sequence_two_write_certified_digests_is_equivocation() {
        let seeds = [test_seed(7), test_seed(8), test_seed(9), test_seed(10)];
        let trust_bytes = trust_bytes(&build_trust(&seeds));
        let pin = sha256_prefixed(&trust_bytes);
        let nonce = test_nonce();
        let first = checkpoint_fixture("5", 0x55);
        let second = checkpoint_fixture("5", 0x66);
        let evidence = vec![
            evidence_fixture(&first, &seeds[0], &nonce, &seeds[..3]),
            evidence_fixture(&second, &seeds[1], &nonce, &seeds[..3]),
            evidence_fixture(&first, &seeds[2], &nonce, &seeds[..3]),
        ];
        let selected =
            verify_head_read_quorum(&evidence, &trust_bytes, &pin, &nonce, "incarnation-1", "1");
        assert_eq!(selected.status, "rejected_witness_equivocation");
    }

    #[test]
    fn one_member_reporting_two_heads_for_the_same_challenge_is_equivocation() {
        let seeds = [test_seed(7), test_seed(8), test_seed(9), test_seed(10)];
        let trust_bytes = trust_bytes(&build_trust(&seeds));
        let pin = sha256_prefixed(&trust_bytes);
        let nonce = test_nonce();
        let latest = checkpoint_fixture("5", 0x55);
        let lagging = checkpoint_fixture("4", 0x44);
        let evidence = vec![
            evidence_fixture(&latest, &seeds[0], &nonce, &seeds[..3]),
            evidence_fixture(&lagging, &seeds[0], &nonce, &seeds[..3]),
            evidence_fixture(&latest, &seeds[1], &nonce, &seeds[..3]),
        ];
        let selected =
            verify_head_read_quorum(&evidence, &trust_bytes, &pin, &nonce, "incarnation-1", "1");
        assert_eq!(selected.status, "rejected_witness_equivocation");
    }

    #[test]
    fn response_replay_from_another_nonce_is_rejected() {
        let (trust_bytes, pin, mut evidence) = read_fixture();
        let latest = checkpoint_fixture("5", 0x55);
        evidence[0].response_envelope = sign_head_response_fixture(
            &latest,
            &base64::engine::general_purpose::URL_SAFE_NO_PAD.encode([43_u8; 32]),
            &test_seed(7),
        );
        let selected = verify_head_read_quorum(
            &evidence,
            &trust_bytes,
            &pin,
            &test_nonce(),
            "incarnation-1",
            "1",
        );
        assert_eq!(selected.status, "rejected_witness_response");
    }

    #[test]
    fn below_threshold_checkpoint_certificate_is_rejected() {
        let seeds = [test_seed(7), test_seed(8), test_seed(9), test_seed(10)];
        let trust_bytes = trust_bytes(&build_trust(&seeds));
        let pin = sha256_prefixed(&trust_bytes);
        let nonce = test_nonce();
        let latest = checkpoint_fixture("5", 0x55);
        let lagging = checkpoint_fixture("4", 0x44);
        let mut evidence = vec![
            evidence_fixture(&latest, &seeds[0], &nonce, &seeds[..2]),
            evidence_fixture(&lagging, &seeds[1], &nonce, &seeds[..3]),
            evidence_fixture(&latest, &seeds[2], &nonce, &seeds[..3]),
        ];
        evidence[0].checkpoint_envelope = sign_checkpoint_fixture(&latest, &seeds[..2]);
        let selected =
            verify_head_read_quorum(&evidence, &trust_bytes, &pin, &nonce, "incarnation-1", "1");
        assert_eq!(selected.status, "rejected_checkpoint_certificate");
    }

    #[test]
    fn response_and_checkpoint_payload_must_match() {
        let seeds = [test_seed(7), test_seed(8), test_seed(9), test_seed(10)];
        let trust_bytes = trust_bytes(&build_trust(&seeds));
        let pin = sha256_prefixed(&trust_bytes);
        let nonce = test_nonce();
        let latest = checkpoint_fixture("5", 0x55);
        let lagging = checkpoint_fixture("4", 0x44);
        let mut evidence = vec![
            evidence_fixture(&latest, &seeds[0], &nonce, &seeds[..3]),
            evidence_fixture(&lagging, &seeds[1], &nonce, &seeds[..3]),
            evidence_fixture(&latest, &seeds[2], &nonce, &seeds[..3]),
        ];
        evidence[0].checkpoint_envelope = sign_checkpoint_fixture(&lagging, &seeds[..3]);
        let selected =
            verify_head_read_quorum(&evidence, &trust_bytes, &pin, &nonce, "incarnation-1", "1");
        assert_eq!(selected.status, "rejected_checkpoint_certificate");
    }

    #[test]
    fn malformed_or_padded_challenge_nonce_is_rejected() {
        let (trust_bytes, pin, evidence) = read_fixture();
        let selected = verify_head_read_quorum(
            &evidence,
            &trust_bytes,
            &pin,
            "not-a-nonce",
            "incarnation-1",
            "1",
        );
        assert_eq!(selected.status, "invalid_challenge");
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
            &envelope,
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
                &envelope,
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
            verify_witness_envelope(&raw, &bytes, &pin, TEST_PAYLOAD_TYPE, QuorumKind::Write)
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
                &envelope,
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
            verify_witness_envelope(&raw, &bytes, &pin, TEST_PAYLOAD_TYPE, QuorumKind::Write)
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
                &envelope,
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
                duplicate.as_bytes(),
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
                &envelope,
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
                &envelope,
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
    fn envelope_verification_is_bound_to_the_exact_pinned_trust_store_bytes() {
        let signing_seeds = [test_seed(7), test_seed(8), test_seed(9), test_seed(10)];
        let unrelated_seeds = [test_seed(17), test_seed(18), test_seed(19), test_seed(20)];
        let signing_trust = build_trust(&signing_seeds);
        let unrelated_trust = build_trust(&unrelated_seeds);
        let signing_trust_bytes = trust_bytes(&signing_trust);
        let unrelated_trust_bytes = trust_bytes(&unrelated_trust);
        let signing_trust_pin = sha256_prefixed(&signing_trust_bytes);
        let envelope = sign_payload(TEST_PAYLOAD_TYPE, TEST_PAYLOAD, &signing_seeds[..3]);

        assert!(
            verify_witness_envelope(
                &envelope,
                &unrelated_trust_bytes,
                &signing_trust_pin,
                TEST_PAYLOAD_TYPE,
                QuorumKind::Write,
            )
            .is_err()
        );
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
