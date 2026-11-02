use base64::{Engine as _, engine::general_purpose::{STANDARD, URL_SAFE_NO_PAD}};
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

fn parse_strict_envelope(
    envelope_bytes: &[u8],
    expected_payload_type: &str,
) -> Result<Envelope, String> {
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
            return Err("DSSE signature keyid must be 64 lowercase hexadecimal characters".to_string());
        }
        if signature.sig.is_empty() {
            return Err("DSSE signature bytes must not be empty".to_string());
        }
    }
    Ok(Envelope {
        payload: strict.payload,
        payload_type: strict.payload_type,
        signatures: strict.signatures.into_iter().map(|signature| Signature {
            keyid: Some(signature.keyid),
            sig: signature.sig,
        }).collect(),
    })
}

fn verify_envelope_with_verifiers(
    envelope_bytes: &[u8],
    expected_payload_type: &str,
    verifiers: &[Ed25519Verifier],
    threshold: usize,
    trust_store_sha256: String,
) -> Result<VerifiedWitnessPayload, String> {
    if threshold == 0 || threshold > verifiers.len() {
        return Err(format!("signature threshold {threshold} is invalid for {} active keys", verifiers.len()));
    }
    let envelope = parse_strict_envelope(envelope_bytes, expected_payload_type)?;
    let verifier_refs: Vec<&dyn Verifier> = verifiers
        .iter()
        .map(|verifier| verifier as &dyn Verifier)
        .collect();
    let verified: VerifiedPayload = verify(&envelope, &verifier_refs, threshold)
        .map_err(|error| format!("DSSE threshold verification failed: {error}"))?;

    let json: Value = serde_json::from_slice(&verified.payload)
        .map_err(|error| format!("verified payload is not strict JSON: {error}"))?;
    let canonical = serde_json_canonicalizer::to_vec(&json)
        .map_err(|error| format!("verified payload cannot be canonicalized as RFC 8785 JCS: {error}"))?;
    if canonical != verified.payload {
        return Err("signed payload is not byte-for-byte RFC 8785 canonical JSON".to_string());
    }

    Ok(VerifiedWitnessPayload {
        payload_type: verified.payload_type,
        payload_sha256: sha256_prefixed(&verified.payload),
        payload: verified.payload,
        json,
        accepted_key_ids: verified.accepted_keys,
        trust_store_sha256,
    })
}

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
    let threshold = match quorum_kind {
        QuorumKind::Read => prepared.threshold_read,
        QuorumKind::Write => prepared.threshold_write,
    };
    verify_envelope_with_verifiers(
        envelope_bytes,
        expected_payload_type,
        &prepared.verifiers,
        threshold,
        prepared.trust_store_sha256,
    )
}


#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RootKeyRecord {
    pub keyid: String,
    pub algorithm: String,
    pub public_key: String,
    pub administrative_domain: String,
    pub status: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TrustRootPayload {
    pub record_type: String,
    pub schema_version: u32,
    pub authority_id: String,
    pub trust_root_version: String,
    pub previous_trust_root_digest: Option<String>,
    pub root_threshold: usize,
    pub root_keys: Vec<RootKeyRecord>,
    pub initial_witness_set_id: String,
    pub initial_witness_set_version: String,
    pub initial_witness_set_digest: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedTrustRoot {
    payload: TrustRootPayload,
    digest: String,
}

impl VerifiedTrustRoot {
    pub fn payload(&self) -> &TrustRootPayload {
        &self.payload
    }

    pub fn digest(&self) -> &str {
        &self.digest
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct WitnessSetMetadataPayload {
    pub record_type: String,
    pub schema_version: u32,
    pub authority_id: String,
    pub witness_set_id: String,
    pub witness_set_version: String,
    pub previous_witness_set_digest: Option<String>,
    pub trust_root_version: String,
    pub fault_tolerance: usize,
    pub member_count: usize,
    pub write_quorum: usize,
    pub read_quorum: usize,
    pub signature_algorithm_policy: String,
    pub members: Vec<TrustMember>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedWitnessSet {
    metadata: WitnessSetMetadataPayload,
    digest: String,
    trust_store: TrustStore,
}

impl VerifiedWitnessSet {
    pub fn metadata(&self) -> &WitnessSetMetadataPayload {
        &self.metadata
    }

    pub fn digest(&self) -> &str {
        &self.digest
    }
}

fn root_key_verifiers(keys: &[RootKeyRecord]) -> Result<Vec<Ed25519Verifier>, String> {
    let mut key_ids = HashSet::new();
    let mut public_keys = HashSet::new();
    let mut domains = BTreeSet::new();
    let mut verifiers = Vec::new();
    for key in keys {
        if !is_lower_hex_64(&key.keyid) || !key_ids.insert(key.keyid.as_str()) {
            return Err(format!("malformed or duplicate root key ID: {}", key.keyid));
        }
        if key.algorithm != "ed25519" {
            return Err(format!("unsupported root key algorithm: {}", key.algorithm));
        }
        let public = decode_canonical_base64(&key.public_key, "root public_key")?;
        if public.len() != 32 || !public_keys.insert(public.clone()) {
            return Err("root public keys must be unique 32-byte Ed25519 keys".to_string());
        }
        if canonical_key_id(&key.public_key)? != key.keyid {
            return Err(format!("root key ID does not bind public key {}", key.keyid));
        }
        match key.status.as_str() {
            "active" => {
                if key.administrative_domain.is_empty()
                    || !domains.insert(key.administrative_domain.as_str())
                {
                    return Err("active trust-root keys must have distinct administrative domains".to_string());
                }
                verifiers.push(
                    Ed25519Verifier::from_bytes(&public)
                        .map_err(|error| format!("invalid root Ed25519 key {}: {error}", key.keyid))?
                        .with_key_id(key.keyid.clone()),
                );
            }
            "revoked" => {}
            other => return Err(format!("unsupported root key status: {other}")),
        }
    }
    Ok(verifiers)
}

fn validate_trust_root_payload(root: &TrustRootPayload) -> Result<Vec<Ed25519Verifier>, String> {
    if root.record_type != "basalt.authority.witness-trust-root" || root.schema_version != 1 {
        return Err("unsupported trust-root record type or schema version".to_string());
    }
    if root.authority_id.is_empty() || root.initial_witness_set_id.is_empty() {
        return Err("trust-root authority and initial witness-set IDs must not be empty".to_string());
    }
    canonical_counter(&root.trust_root_version, "trust_root_version", false)?;
    canonical_counter(&root.initial_witness_set_version, "initial_witness_set_version", false)?;
    if root.root_threshold == 0 {
        return Err("trust-root threshold must be at least one".to_string());
    }
    if !valid_sha256(&root.initial_witness_set_digest) {
        return Err("trust root does not contain a valid initial witness-set digest".to_string());
    }
    match root.previous_trust_root_digest.as_deref() {
        None if root.trust_root_version != "1" => {
            return Err("only trust-root version one may omit its predecessor digest".to_string());
        }
        Some(digest) if !valid_sha256(digest) => {
            return Err("trust-root predecessor digest is malformed".to_string());
        }
        _ => {}
    }
    let verifiers = root_key_verifiers(&root.root_keys)?;
    if root.root_threshold > verifiers.len() {
        return Err("trust-root threshold exceeds active root key count".to_string());
    }
    Ok(verifiers)
}

pub fn trust_root_digest(root: &TrustRootPayload) -> Result<String, String> {
    let canonical = serde_json_canonicalizer::to_vec(root)
        .map_err(|error| format!("cannot canonicalize trust-root payload: {error}"))?;
    let mut hasher = Sha256::new();
    hasher.update(b"BASALT-WITNESS-TRUST-ROOT-V1\0");
    hasher.update(canonical);
    Ok(format!("sha256:{}", hex_lower(&hasher.finalize())))
}

/// Loads an offline trust root only when its canonical, domain-separated digest matches
/// the digest provisioned through an out-of-band trust channel.
pub fn load_pinned_trust_root(
    root_bytes: &[u8],
    expected_digest: &str,
) -> Result<VerifiedTrustRoot, String> {
    let root: TrustRootPayload = serde_json::from_slice(root_bytes)
        .map_err(|error| format!("invalid trust-root JSON: {error}"))?;
    let canonical = serde_json_canonicalizer::to_vec(&root)
        .map_err(|error| format!("cannot canonicalize trust-root payload: {error}"))?;
    if canonical != root_bytes {
        return Err("pinned trust-root file must be byte-for-byte RFC 8785 canonical JSON".to_string());
    }
    let _verifiers = validate_trust_root_payload(&root)?;
    let digest = trust_root_digest(&root)?;
    if digest != expected_digest {
        return Err(format!("out-of-band trust-root digest mismatch: expected {expected_digest}, got {digest}"));
    }
    Ok(VerifiedTrustRoot { payload: root, digest })
}

fn verify_root_signed_payload(
    envelope: &[u8],
    root: &TrustRootPayload,
) -> Result<VerifiedWitnessPayload, String> {
    let verifiers = validate_trust_root_payload(root)?;
    verify_envelope_with_verifiers(
        envelope,
        TRUST_ROOT_PAYLOAD_TYPE,
        &verifiers,
        root.root_threshold,
        trust_root_digest(root)?,
    )
}

/// Verifies a successor root under both the currently pinned root and the successor's
/// own threshold. This is the dual-threshold rotation rule; a valid old-root signature
/// by itself is not sufficient.
pub fn verify_trust_root_rotation(
    current: &VerifiedTrustRoot,
    successor_envelope: &[u8],
) -> Result<VerifiedTrustRoot, String> {
    let old_verified = verify_root_signed_payload(successor_envelope, &current.payload)?;
    let successor: TrustRootPayload = serde_json::from_slice(&old_verified.payload)
        .map_err(|error| format!("invalid signed successor trust root: {error}"))?;
    let next_verifiers = validate_trust_root_payload(&successor)?;
    if successor.authority_id != current.payload.authority_id {
        return Err("trust-root rotation changed the stable authority ID".to_string());
    }
    let current_version = canonical_counter(&current.payload.trust_root_version, "current trust_root_version", false)?;
    let next_version = canonical_counter(&successor.trust_root_version, "successor trust_root_version", false)?;
    if next_version != current_version.checked_add(1).ok_or_else(|| "trust-root version exhausted".to_string())? {
        return Err("successor trust-root version must increase by exactly one".to_string());
    }
    if successor.previous_trust_root_digest.as_deref() != Some(current.digest.as_str()) {
        return Err("successor trust root does not link to the currently pinned root digest".to_string());
    }
    if successor.initial_witness_set_id != current.payload.initial_witness_set_id
        || successor.initial_witness_set_version != current.payload.initial_witness_set_version
        || successor.initial_witness_set_digest != current.payload.initial_witness_set_digest
    {
        return Err("trust-root rotation cannot silently replace the pinned initial witness set".to_string());
    }
    let new_verified = verify_envelope_with_verifiers(
        successor_envelope,
        TRUST_ROOT_PAYLOAD_TYPE,
        &next_verifiers,
        successor.root_threshold,
        trust_root_digest(&successor)?,
    )?;
    if new_verified.payload != old_verified.payload {
        return Err("old-root and new-root signatures did not authenticate identical successor bytes".to_string());
    }
    Ok(VerifiedTrustRoot {
        digest: trust_root_digest(&successor)?,
        payload: successor,
    })
}

pub fn witness_set_digest(metadata: &WitnessSetMetadataPayload) -> Result<String, String> {
    let canonical = serde_json_canonicalizer::to_vec(metadata)
        .map_err(|error| format!("cannot canonicalize witness-set metadata: {error}"))?;
    let mut hasher = Sha256::new();
    hasher.update(b"BASALT-WITNESS-SET-METADATA-V1\0");
    hasher.update(canonical);
    Ok(format!("sha256:{}", hex_lower(&hasher.finalize())))
}

fn validate_witness_set_payload(
    metadata: &WitnessSetMetadataPayload,
    root: &TrustRootPayload,
) -> Result<TrustStore, String> {
    if metadata.record_type != "basalt.authority.witness-set-metadata" || metadata.schema_version != 1 {
        return Err("unsupported witness-set metadata record type or schema version".to_string());
    }
    if metadata.authority_id != root.authority_id
        || metadata.trust_root_version != root.trust_root_version
        || metadata.witness_set_id.is_empty()
    {
        return Err("witness-set metadata does not bind the active authority and trust-root version".to_string());
    }
    let trust = TrustStore {
        schema_version: 1,
        authority_id: metadata.authority_id.clone(),
        witness_set_id: metadata.witness_set_id.clone(),
        witness_set_version: canonical_counter(&metadata.witness_set_version, "witness_set_version", false)?,
        trust_root_version: canonical_counter(&metadata.trust_root_version, "trust_root_version", false)?,
        fault_tolerance: metadata.fault_tolerance,
        member_count: metadata.member_count,
        write_quorum: metadata.write_quorum,
        read_quorum: metadata.read_quorum,
        members: metadata.members.clone(),
    };
    if metadata.signature_algorithm_policy
        != "from-pinned-trust-metadata; never selected from an untrusted envelope"
    {
        return Err("witness-set signature algorithm policy is not the v1 pinned policy".to_string());
    }
    let set_digest = witness_set_digest(metadata)?;
    prepare_trust_store(&trust, set_digest)?;
    let root_domains: HashSet<&str> = root.root_keys.iter()
        .filter(|key| key.status == "active")
        .map(|key| key.administrative_domain.as_str())
        .collect();
    for member in metadata.members.iter().filter(|member| member.status == "active") {
        if root_domains.contains(member.administrative_domain.as_str()) {
            return Err("witness members and trust-root signers share an administrative failure domain".to_string());
        }
    }
    if metadata.previous_witness_set_digest.as_deref().is_some_and(|digest| !valid_sha256(digest)) {
        return Err("previous witness-set digest is malformed".to_string());
    }
    Ok(trust)
}

pub fn verify_witness_set_metadata(
    envelope: &[u8],
    root: &VerifiedTrustRoot,
    previous: Option<&VerifiedWitnessSet>,
) -> Result<VerifiedWitnessSet, String> {
    let verified = verify_root_signed_payload(envelope, &root.payload)?;
    let metadata: WitnessSetMetadataPayload = serde_json::from_slice(&verified.payload)
        .map_err(|error| format!("invalid signed witness-set metadata: {error}"))?;
    let trust_store = validate_witness_set_payload(&metadata, &root.payload)?;
    let digest = witness_set_digest(&metadata)?;
    if let Some(previous) = previous {
        let previous_metadata = previous.metadata();
        let expected_version = canonical_counter(&previous_metadata.witness_set_version, "previous witness_set_version", false)?
            .checked_add(1).ok_or_else(|| "witness-set version exhausted".to_string())?;
        if canonical_counter(&metadata.witness_set_version, "witness_set_version", false)? != expected_version
            || metadata.authority_id != previous_metadata.authority_id
            || metadata.previous_witness_set_digest.as_deref() != Some(previous.digest())
        {
            return Err("witness-set rotation must increment one version and link to the prior verified metadata digest".to_string());
        }
    } else if metadata.witness_set_id != root.payload.initial_witness_set_id
        || metadata.witness_set_version != root.payload.initial_witness_set_version
        || digest != root.payload.initial_witness_set_digest
        || metadata.previous_witness_set_digest.is_some()
    {
        return Err("initial witness set does not match the separately pinned trust-root metadata".to_string());
    }
    Ok(VerifiedWitnessSet {
        metadata,
        digest,
        trust_store,
    })
}

pub fn verify_checkpoint_chain_with_witness_set(
    envelopes: &[Vec<u8>],
    witness_set: &VerifiedWitnessSet,
) -> Result<VerifiedCheckpointChain, String> {
    let trust_bytes = serde_json::to_vec(&witness_set.trust_store)
        .map_err(|error| format!("cannot serialize verified witness set for chain verification: {error}"))?;
    verify_checkpoint_chain(envelopes, &trust_bytes, &sha256_prefixed(&trust_bytes))
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CheckpointPayload {
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

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct HeadResponsePayload {
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedCheckpoint {
    pub payload: CheckpointPayload,
    pub digest: String,
    pub accepted_key_ids: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedCheckpointChain {
    pub checkpoints: Vec<VerifiedCheckpoint>,
    pub head: VerifiedCheckpoint,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RestoreObservation {
    pub authority_id: String,
    pub authority_incarnation_id: String,
    pub authority_epoch: String,
    pub authority_revision: String,
    pub state_digest: String,
    pub policy_revision: String,
    pub policy_digest: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RestoreVerdict {
    Fresh,
    RejectedAntiRollback { reason: String },
}

fn canonical_counter(value: &str, field: &str, allow_zero: bool) -> Result<u64, String> {
    if value.is_empty()
        || (value.len() > 1 && value.starts_with('0'))
        || !value.bytes().all(|byte| byte.is_ascii_digit())
    {
        return Err(format!("{field} is not a canonical decimal string"));
    }
    let parsed = value.parse::<u64>()
        .map_err(|error| format!("{field} is outside the u64 range: {error}"))?;
    if !allow_zero && parsed == 0 {
        return Err(format!("{field} must be greater than zero"));
    }
    Ok(parsed)
}

fn valid_sha256(value: &str) -> bool {
    value.strip_prefix("sha256:").is_some_and(is_lower_hex_64)
}

pub fn checkpoint_digest(payload: &CheckpointPayload) -> Result<String, String> {
    let canonical = serde_json_canonicalizer::to_vec(payload)
        .map_err(|error| format!("cannot canonicalize checkpoint payload: {error}"))?;
    let mut hasher = Sha256::new();
    hasher.update(b"BASALT-WITNESS-CHECKPOINT-V1\0");
    hasher.update(canonical);
    Ok(format!("sha256:{}", hex_lower(&hasher.finalize())))
}

fn validate_checkpoint_payload(
    payload: &CheckpointPayload,
    trust: &TrustStore,
) -> Result<(), String> {
    if payload.record_type != "basalt.authority.witness-checkpoint" || payload.schema_version != 1 {
        return Err("unsupported witness checkpoint record type or schema version".to_string());
    }
    if payload.authority_id != trust.authority_id
        || payload.witness_set_id != trust.witness_set_id
        || payload.witness_set_version != trust.witness_set_version.to_string()
        || payload.trust_root_version != trust.trust_root_version.to_string()
    {
        return Err("checkpoint is not bound to the pinned authority, witness set and trust root".to_string());
    }
    if payload.authority_incarnation_id.is_empty() || payload.state_schema.is_empty() {
        return Err("checkpoint incarnation and state schema must not be empty".to_string());
    }
    canonical_counter(&payload.authority_epoch, "authority_epoch", true)?;
    canonical_counter(&payload.sequence, "sequence", true)?;
    canonical_counter(&payload.authority_revision, "authority_revision", true)?;
    canonical_counter(&payload.policy_revision, "policy_revision", true)?;
    if !valid_sha256(&payload.state_digest) || !valid_sha256(&payload.policy_digest) {
        return Err("checkpoint state or policy digest is malformed".to_string());
    }
    if payload.previous_checkpoint_digest.as_deref().is_some_and(|value| !valid_sha256(value))
        || payload.transition_bundle_digest.as_deref().is_some_and(|value| !valid_sha256(value))
    {
        return Err("checkpoint predecessor or transition-bundle digest is malformed".to_string());
    }
    Ok(())
}

/// Verifies a complete sequence-zero-anchored checkpoint list using a pinned witness set.
///
/// This validates signatures, sequence continuity, predecessor links, authority incarnation,
/// revision monotonicity, policy revision monotonicity and transition-bundle commitments.
/// It does not prove that the transition-bundle content is available or independently retained.
pub fn verify_checkpoint_chain(
    envelopes: &[Vec<u8>],
    trust_store_bytes: &[u8],
    expected_trust_store_sha256: &str,
) -> Result<VerifiedCheckpointChain, String> {
    if envelopes.is_empty() {
        return Err("checkpoint chain must include its sequence-zero genesis record".to_string());
    }
    if envelopes.len() > 100_000 {
        return Err("checkpoint chain exceeds the 100000-record safety limit".to_string());
    }
    let trust = load_pinned_trust_store(trust_store_bytes, expected_trust_store_sha256)?;
    let mut checkpoints = Vec::with_capacity(envelopes.len());
    let mut previous: Option<VerifiedCheckpoint> = None;

    for envelope in envelopes {
        let verified = verify_witness_envelope(
            envelope,
            trust_store_bytes,
            expected_trust_store_sha256,
            CHECKPOINT_PAYLOAD_TYPE,
            QuorumKind::Write,
        )?;
        let payload: CheckpointPayload = serde_json::from_slice(&verified.payload)
            .map_err(|error| format!("verified checkpoint payload has invalid shape: {error}"))?;
        validate_checkpoint_payload(&payload, &trust)?;
        let sequence = canonical_counter(&payload.sequence, "sequence", true)?;
        match previous.as_ref() {
            None => {
                if sequence != 0
                    || payload.previous_checkpoint_digest.is_some()
                    || payload.transition_bundle_digest.is_some()
                {
                    return Err("genesis checkpoint must be sequence zero with null predecessor and transition bundle".to_string());
                }
            }
            Some(prior) => {
                let prior_payload = &prior.payload;
                let expected_sequence = canonical_counter(&prior_payload.sequence, "prior sequence", true)?
                    .checked_add(1)
                    .ok_or_else(|| "witness sequence exhausted".to_string())?;
                if sequence != expected_sequence {
                    return Err(format!("checkpoint sequence gap: expected {expected_sequence}, got {sequence}"));
                }
                if payload.previous_checkpoint_digest.as_deref() != Some(prior.digest.as_str()) {
                    return Err("checkpoint predecessor digest does not match the verified prior checkpoint".to_string());
                }
                if payload.transition_bundle_digest.is_none() {
                    return Err("non-genesis checkpoint must commit a transition-bundle digest".to_string());
                }
                if payload.authority_incarnation_id != prior_payload.authority_incarnation_id
                    || payload.authority_epoch != prior_payload.authority_epoch
                {
                    return Err("authority incarnation or epoch changed without a separately authorized re-provisioning ceremony".to_string());
                }
                if canonical_counter(&payload.authority_revision, "authority_revision", true)?
                    <= canonical_counter(&prior_payload.authority_revision, "prior authority_revision", true)?
                {
                    return Err("authority revision did not increase strictly".to_string());
                }
                let prior_policy = canonical_counter(&prior_payload.policy_revision, "prior policy_revision", true)?;
                let next_policy = canonical_counter(&payload.policy_revision, "policy_revision", true)?;
                if next_policy < prior_policy {
                    return Err("policy revision rolled back".to_string());
                }
                if next_policy == prior_policy && payload.policy_digest != prior_payload.policy_digest {
                    return Err("policy digest changed without a policy revision increase".to_string());
                }
            }
        }

        let checkpoint = VerifiedCheckpoint {
            digest: checkpoint_digest(&payload)?,
            payload,
            accepted_key_ids: verified.accepted_key_ids,
        };
        previous = Some(checkpoint.clone());
        checkpoints.push(checkpoint);
    }

    let head = checkpoints.last().cloned()
        .ok_or_else(|| "checkpoint chain unexpectedly empty".to_string())?;
    Ok(VerifiedCheckpointChain { checkpoints, head })
}

/// Verifies a fresh, nonce-bound read-quorum response against an already verified chain head.
pub fn verify_challenge_bound_head(
    envelope: &[u8],
    trust_store_bytes: &[u8],
    expected_trust_store_sha256: &str,
    challenge_nonce: &[u8; 32],
    chain: &VerifiedCheckpointChain,
) -> Result<Vec<String>, String> {
    let verified = verify_witness_envelope(
        envelope,
        trust_store_bytes,
        expected_trust_store_sha256,
        HEAD_RESPONSE_PAYLOAD_TYPE,
        QuorumKind::Read,
    )?;
    let response: HeadResponsePayload = serde_json::from_slice(&verified.payload)
        .map_err(|error| format!("verified head response has invalid shape: {error}"))?;
    let nonce = URL_SAFE_NO_PAD.encode(challenge_nonce);
    if response.challenge_nonce != nonce {
        return Err("head response does not bind the verifier's fresh 32-byte nonce".to_string());
    }
    let head = &chain.head.payload;
    if response.record_type != "basalt.authority.witness-head-response" || response.schema_version != 1 {
        return Err("unsupported witness head-response type or schema version".to_string());
    }
    let matches = response.authority_id == head.authority_id
        && response.authority_incarnation_id == head.authority_incarnation_id
        && response.authority_epoch == head.authority_epoch
        && response.witness_set_id == head.witness_set_id
        && response.witness_set_version == head.witness_set_version
        && response.trust_root_version == head.trust_root_version
        && response.head_sequence == head.sequence
        && response.head_checkpoint_digest == chain.head.digest
        && response.head_authority_revision == head.authority_revision
        && response.head_state_digest == head.state_digest
        && response.head_policy_digest == head.policy_digest;
    if !matches {
        return Err("challenge-bound quorum head disagrees with the verified checkpoint chain".to_string());
    }
    Ok(verified.accepted_key_ids)
}


pub fn verify_challenge_bound_head_with_witness_set(
    envelope: &[u8],
    challenge_nonce: &[u8; 32],
    witness_set: &VerifiedWitnessSet,
    chain: &VerifiedCheckpointChain,
) -> Result<Vec<String>, String> {
    let trust_bytes = serde_json::to_vec(&witness_set.trust_store)
        .map_err(|error| format!("cannot serialize verified witness trust set: {error}"))?;
    verify_challenge_bound_head(
        envelope,
        &trust_bytes,
        &sha256_prefixed(&trust_bytes),
        challenge_nonce,
        chain,
    )
}

pub fn assess_restored_store(
    observed: &RestoreObservation,
    witnessed_head: &CheckpointPayload,
) -> Result<RestoreVerdict, String> {
    if observed.authority_id != witnessed_head.authority_id {
        return Ok(RestoreVerdict::RejectedAntiRollback {
            reason: "restored store authority ID differs from the witnessed authority".to_string(),
        });
    }
    if observed.authority_incarnation_id != witnessed_head.authority_incarnation_id
        || observed.authority_epoch != witnessed_head.authority_epoch
    {
        return Ok(RestoreVerdict::RejectedAntiRollback {
            reason: "restored store incarnation or epoch differs from the witnessed head".to_string(),
        });
    }
    let store_revision = canonical_counter(&observed.authority_revision, "store authority revision", true)?;
    let witness_revision = canonical_counter(&witnessed_head.authority_revision, "witness authority revision", true)?;
    if store_revision != witness_revision {
        return Ok(RestoreVerdict::RejectedAntiRollback {
            reason: format!("store revision {} differs from witness revision {}", observed.authority_revision, witnessed_head.authority_revision),
        });
    }
    if observed.state_digest != witnessed_head.state_digest {
        return Ok(RestoreVerdict::RejectedAntiRollback {
            reason: "restored state digest differs from witness head".to_string(),
        });
    }
    if observed.policy_revision != witnessed_head.policy_revision
        || observed.policy_digest != witnessed_head.policy_digest
    {
        return Ok(RestoreVerdict::RejectedAntiRollback {
            reason: "restored policy head differs from witness head".to_string(),
        });
    }
    Ok(RestoreVerdict::Fresh)
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


    fn root_key_records(seeds: &[[u8; 32]], domain_prefix: &str) -> Vec<RootKeyRecord> {
        seeds.iter().enumerate().map(|(index, seed)| {
            let signing_key = SigningKey::from_bytes(seed);
            let public_key = STANDARD.encode(signing_key.verifying_key().to_bytes());
            RootKeyRecord {
                keyid: canonical_key_id(&public_key).unwrap(),
                algorithm: "ed25519".to_string(),
                public_key,
                administrative_domain: format!("{domain_prefix}-{index}"),
                status: "active".to_string(),
            }
        }).collect()
    }

    fn witness_metadata_from_trust(
        trust: &TrustStore,
        previous: Option<String>,
    ) -> WitnessSetMetadataPayload {
        WitnessSetMetadataPayload {
            record_type: "basalt.authority.witness-set-metadata".to_string(),
            schema_version: 1,
            authority_id: trust.authority_id.clone(),
            witness_set_id: trust.witness_set_id.clone(),
            witness_set_version: trust.witness_set_version.to_string(),
            previous_witness_set_digest: previous,
            trust_root_version: trust.trust_root_version.to_string(),
            fault_tolerance: trust.fault_tolerance,
            member_count: trust.member_count,
            write_quorum: trust.write_quorum,
            read_quorum: trust.read_quorum,
            signature_algorithm_policy: "from-pinned-trust-metadata; never selected from an untrusted envelope".to_string(),
            members: trust.members.clone(),
        }
    }

    fn make_trust_root(
        root_keys: Vec<RootKeyRecord>,
        initial_set: &WitnessSetMetadataPayload,
    ) -> TrustRootPayload {
        TrustRootPayload {
            record_type: "basalt.authority.witness-trust-root".to_string(),
            schema_version: 1,
            authority_id: initial_set.authority_id.clone(),
            trust_root_version: "1".to_string(),
            previous_trust_root_digest: None,
            root_threshold: 3,
            root_keys,
            initial_witness_set_id: initial_set.witness_set_id.clone(),
            initial_witness_set_version: initial_set.witness_set_version.clone(),
            initial_witness_set_digest: witness_set_digest(initial_set).unwrap(),
        }
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

    fn checkpoint_fixture(
        sequence: &str,
        revision: &str,
        state_digest: &str,
        previous: Option<String>,
        bundle: Option<String>,
    ) -> CheckpointPayload {
        CheckpointPayload {
            record_type: "basalt.authority.witness-checkpoint".to_string(),
            schema_version: 1,
            authority_id: "basalt-test-authority".to_string(),
            authority_incarnation_id: "incarnation-1".to_string(),
            authority_epoch: "1".to_string(),
            sequence: sequence.to_string(),
            authority_revision: revision.to_string(),
            state_schema: "basalt.authority-state.v1".to_string(),
            state_digest: state_digest.to_string(),
            policy_revision: "1".to_string(),
            policy_digest: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string(),
            previous_checkpoint_digest: previous,
            witness_set_id: "basalt-test-witness-set".to_string(),
            witness_set_version: "1".to_string(),
            trust_root_version: "1".to_string(),
            transition_bundle_digest: bundle,
        }
    }

    fn signed_checkpoint(payload: &CheckpointPayload, seeds: &[[u8; 32]]) -> Vec<u8> {
        let canonical = serde_json_canonicalizer::to_vec(payload).unwrap();
        sign_payload(CHECKPOINT_PAYLOAD_TYPE, &canonical, seeds)
    }

    #[test]
    fn checkpoint_chain_and_challenge_bound_fresh_head_verify_end_to_end() {
        let seeds = [test_seed(31), test_seed(32), test_seed(33), test_seed(34)];
        let trust = build_trust(&seeds);
        let trust_bytes = trust_bytes(&trust);
        let pin = sha256_prefixed(&trust_bytes);
        let state0 = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
        let state1 = "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
        let genesis = checkpoint_fixture("0", "0", state0, None, None);
        let genesis_digest = checkpoint_digest(&genesis).unwrap();
        let next = checkpoint_fixture(
            "1",
            "1",
            state1,
            Some(genesis_digest),
            Some("sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd".to_string()),
        );
        let envelopes = vec![
            signed_checkpoint(&genesis, &seeds[..3]),
            signed_checkpoint(&next, &seeds[..3]),
        ];
        let chain = verify_checkpoint_chain(&envelopes, &trust_bytes, &pin).unwrap();
        assert_eq!(chain.checkpoints.len(), 2);
        assert_eq!(chain.head.payload, next);

        let nonce = [42u8; 32];
        let response = HeadResponsePayload {
            record_type: "basalt.authority.witness-head-response".to_string(),
            schema_version: 1,
            challenge_nonce: URL_SAFE_NO_PAD.encode(nonce),
            authority_id: next.authority_id.clone(),
            authority_incarnation_id: next.authority_incarnation_id.clone(),
            authority_epoch: next.authority_epoch.clone(),
            witness_set_id: next.witness_set_id.clone(),
            witness_set_version: next.witness_set_version.clone(),
            trust_root_version: next.trust_root_version.clone(),
            head_sequence: next.sequence.clone(),
            head_checkpoint_digest: chain.head.digest.clone(),
            head_authority_revision: next.authority_revision.clone(),
            head_state_digest: next.state_digest.clone(),
            head_policy_digest: next.policy_digest.clone(),
        };
        let response_bytes = serde_json_canonicalizer::to_vec(&response).unwrap();
        let response_envelope = sign_payload(HEAD_RESPONSE_PAYLOAD_TYPE, &response_bytes, &seeds[..3]);
        let accepted = verify_challenge_bound_head(
            &response_envelope,
            &trust_bytes,
            &pin,
            &nonce,
            &chain,
        )
        .unwrap();
        assert_eq!(accepted.len(), 3);

        let observation = RestoreObservation {
            authority_id: next.authority_id.clone(),
            authority_incarnation_id: next.authority_incarnation_id.clone(),
            authority_epoch: next.authority_epoch.clone(),
            authority_revision: next.authority_revision.clone(),
            state_digest: next.state_digest.clone(),
            policy_revision: next.policy_revision.clone(),
            policy_digest: next.policy_digest.clone(),
        };
        assert_eq!(
            assess_restored_store(&observation, &chain.head.payload).unwrap(),
            RestoreVerdict::Fresh
        );
    }

    #[test]
    fn old_snapshot_is_rejected_against_newer_witness_head() {
        let seeds = [test_seed(41), test_seed(42), test_seed(43), test_seed(44)];
        let trust = build_trust(&seeds);
        let trust_bytes = trust_bytes(&trust);
        let pin = sha256_prefixed(&trust_bytes);
        let state0 = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
        let state1 = "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
        let genesis = checkpoint_fixture("0", "0", state0, None, None);
        let genesis_digest = checkpoint_digest(&genesis).unwrap();
        let head = checkpoint_fixture(
            "1",
            "9",
            state1,
            Some(genesis_digest),
            Some("sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd".to_string()),
        );
        let chain = verify_checkpoint_chain(
            &[signed_checkpoint(&genesis, &seeds[..3]), signed_checkpoint(&head, &seeds[..3])],
            &trust_bytes,
            &pin,
        ).unwrap();

        let old = RestoreObservation {
            authority_id: head.authority_id.clone(),
            authority_incarnation_id: head.authority_incarnation_id.clone(),
            authority_epoch: head.authority_epoch.clone(),
            authority_revision: "2".to_string(),
            state_digest: state0.to_string(),
            policy_revision: "1".to_string(),
            policy_digest: head.policy_digest.clone(),
        };
        assert!(matches!(
            assess_restored_store(&old, &chain.head.payload).unwrap(),
            RestoreVerdict::RejectedAntiRollback { .. }
        ));
    }

    #[test]
    fn checkpoint_chain_rejects_a_broken_predecessor_even_with_valid_signatures() {
        let seeds = [test_seed(51), test_seed(52), test_seed(53), test_seed(54)];
        let trust = build_trust(&seeds);
        let trust_bytes = trust_bytes(&trust);
        let pin = sha256_prefixed(&trust_bytes);
        let state0 = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
        let state1 = "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
        let genesis = checkpoint_fixture("0", "0", state0, None, None);
        let broken = checkpoint_fixture(
            "1",
            "1",
            state1,
            Some("sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee".to_string()),
            Some("sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd".to_string()),
        );
        let envelopes = vec![
            signed_checkpoint(&genesis, &seeds[..3]),
            signed_checkpoint(&broken, &seeds[..3]),
        ];
        assert!(verify_checkpoint_chain(&envelopes, &trust_bytes, &pin).is_err());
    }

    #[test]
    fn challenge_bound_head_response_cannot_be_replayed_for_another_nonce() {
        let seeds = [test_seed(61), test_seed(62), test_seed(63), test_seed(64)];
        let trust = build_trust(&seeds);
        let trust_bytes = trust_bytes(&trust);
        let pin = sha256_prefixed(&trust_bytes);
        let state = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
        let checkpoint = checkpoint_fixture("0", "0", state, None, None);
        let chain = verify_checkpoint_chain(
            &[signed_checkpoint(&checkpoint, &seeds[..3])],
            &trust_bytes,
            &pin,
        ).unwrap();
        let nonce = [71u8; 32];
        let response = HeadResponsePayload {
            record_type: "basalt.authority.witness-head-response".to_string(),
            schema_version: 1,
            challenge_nonce: URL_SAFE_NO_PAD.encode(nonce),
            authority_id: checkpoint.authority_id.clone(),
            authority_incarnation_id: checkpoint.authority_incarnation_id.clone(),
            authority_epoch: checkpoint.authority_epoch.clone(),
            witness_set_id: checkpoint.witness_set_id.clone(),
            witness_set_version: checkpoint.witness_set_version.clone(),
            trust_root_version: checkpoint.trust_root_version.clone(),
            head_sequence: checkpoint.sequence.clone(),
            head_checkpoint_digest: chain.head.digest.clone(),
            head_authority_revision: checkpoint.authority_revision.clone(),
            head_state_digest: checkpoint.state_digest.clone(),
            head_policy_digest: checkpoint.policy_digest.clone(),
        };
        let response_bytes = serde_json_canonicalizer::to_vec(&response).unwrap();
        let envelope = sign_payload(HEAD_RESPONSE_PAYLOAD_TYPE, &response_bytes, &seeds[..3]);
        let mut replay_nonce = nonce;
        replay_nonce[0] ^= 1;
        assert!(verify_challenge_bound_head(
            &envelope,
            &trust_bytes,
            &pin,
            &replay_nonce,
            &chain,
        ).is_err());
    }


    #[test]
    fn checkpoint_chain_rejects_incarnation_substitution_and_policy_rollback() {
        let seeds = [test_seed(91), test_seed(92), test_seed(93), test_seed(94)];
        let trust = build_trust(&seeds);
        let trust_bytes = trust_bytes(&trust);
        let pin = sha256_prefixed(&trust_bytes);
        let state0 = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
        let state1 = "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
        let genesis = checkpoint_fixture("0", "0", state0, None, None);
        let digest = checkpoint_digest(&genesis).unwrap();

        let mut changed_incarnation = checkpoint_fixture(
            "1",
            "1",
            state1,
            Some(digest.clone()),
            Some("sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd".to_string()),
        );
        changed_incarnation.authority_incarnation_id = "incarnation-rollback".to_string();
        assert!(verify_checkpoint_chain(
            &[signed_checkpoint(&genesis, &seeds[..3]), signed_checkpoint(&changed_incarnation, &seeds[..3])],
            &trust_bytes,
            &pin,
        ).is_err());

        let mut policy_rollback = checkpoint_fixture(
            "1",
            "1",
            state1,
            Some(digest),
            Some("sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd".to_string()),
        );
        policy_rollback.policy_revision = "0".to_string();
        assert!(verify_checkpoint_chain(
            &[signed_checkpoint(&genesis, &seeds[..3]), signed_checkpoint(&policy_rollback, &seeds[..3])],
            &trust_bytes,
            &pin,
        ).is_err());
    }

    #[test]
    fn checkpoint_policy_digest_cannot_change_without_revision_advance() {
        let seeds = [test_seed(101), test_seed(102), test_seed(103), test_seed(104)];
        let trust = build_trust(&seeds);
        let trust_bytes = trust_bytes(&trust);
        let pin = sha256_prefixed(&trust_bytes);
        let state0 = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
        let state1 = "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
        let genesis = checkpoint_fixture("0", "0", state0, None, None);
        let digest = checkpoint_digest(&genesis).unwrap();
        let mut next = checkpoint_fixture(
            "1",
            "1",
            state1,
            Some(digest),
            Some("sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd".to_string()),
        );
        next.policy_digest = "sha256:9999999999999999999999999999999999999999999999999999999999999999".to_string();
        assert!(verify_checkpoint_chain(
            &[signed_checkpoint(&genesis, &seeds[..3]), signed_checkpoint(&next, &seeds[..3])],
            &trust_bytes,
            &pin,
        ).is_err());
    }

    #[test]
    fn checkpoint_sequence_gaps_and_missing_transition_bundles_are_rejected() {
        let seeds = [test_seed(81), test_seed(82), test_seed(83), test_seed(84)];
        let trust = build_trust(&seeds);
        let trust_bytes = trust_bytes(&trust);
        let pin = sha256_prefixed(&trust_bytes);
        let state0 = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
        let state1 = "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
        let genesis = checkpoint_fixture("0", "0", state0, None, None);
        let genesis_digest = checkpoint_digest(&genesis).unwrap();
        let gap = checkpoint_fixture(
            "2",
            "2",
            state1,
            Some(genesis_digest.clone()),
            Some("sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd".to_string()),
        );
        assert!(verify_checkpoint_chain(
            &[signed_checkpoint(&genesis, &seeds[..3]), signed_checkpoint(&gap, &seeds[..3])],
            &trust_bytes,
            &pin,
        ).is_err());

        let missing_bundle = checkpoint_fixture("1", "1", state1, Some(genesis_digest), None);
        assert!(verify_checkpoint_chain(
            &[signed_checkpoint(&genesis, &seeds[..3]), signed_checkpoint(&missing_bundle, &seeds[..3])],
            &trust_bytes,
            &pin,
        ).is_err());
    }


    #[test]
    fn offline_root_pin_authenticates_initial_witness_set_and_checkpoint_chain() {
        let root_seeds = [test_seed(111), test_seed(112), test_seed(113), test_seed(114)];
        let witness_seeds = [test_seed(121), test_seed(122), test_seed(123), test_seed(124)];
        let root_keys = root_key_records(&root_seeds, "root-zone");
        let trust = build_trust(&witness_seeds);
        let metadata = witness_metadata_from_trust(&trust, None);
        let root = make_trust_root(root_keys, &metadata);
        let root_digest = trust_root_digest(&root).unwrap();
        let root_bytes = serde_json_canonicalizer::to_vec(&root).unwrap();
        let pinned = load_pinned_trust_root(&root_bytes, &root_digest).unwrap();

        let set_payload = serde_json_canonicalizer::to_vec(&metadata).unwrap();
        let set_envelope = sign_payload(WITNESS_SET_PAYLOAD_TYPE, &set_payload, &root_seeds[..3]);
        let verified_set = verify_witness_set_metadata(&set_envelope, &pinned, None).unwrap();
        assert_eq!(verified_set.digest(), root.initial_witness_set_digest);

        let state0 = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
        let state1 = "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
        let genesis = checkpoint_fixture("0", "0", state0, None, None);
        let genesis_digest = checkpoint_digest(&genesis).unwrap();
        let next = checkpoint_fixture(
            "1",
            "1",
            state1,
            Some(genesis_digest),
            Some("sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd".to_string()),
        );
        let chain = verify_checkpoint_chain_with_witness_set(
            &[
                signed_checkpoint(&genesis, &witness_seeds[..3]),
                signed_checkpoint(&next, &witness_seeds[..3]),
            ],
            &verified_set,
        ).unwrap();
        assert_eq!(chain.head.payload.state_digest, state1);
    }

    #[test]
    fn trust_root_rotation_requires_both_old_and_new_key_thresholds() {
        let old_seeds = [test_seed(131), test_seed(132), test_seed(133), test_seed(134)];
        let new_seeds = [test_seed(141), test_seed(142), test_seed(143), test_seed(144)];
        let witness_seeds = [test_seed(151), test_seed(152), test_seed(153), test_seed(154)];
        let trust = build_trust(&witness_seeds);
        let metadata = witness_metadata_from_trust(&trust, None);
        let current = make_trust_root(root_key_records(&old_seeds, "old-root"), &metadata);
        let current_digest = trust_root_digest(&current).unwrap();
        let current_bytes = serde_json_canonicalizer::to_vec(&current).unwrap();
        let pinned = load_pinned_trust_root(&current_bytes, &current_digest).unwrap();

        let mut successor = current.clone();
        successor.trust_root_version = "2".to_string();
        successor.previous_trust_root_digest = Some(current_digest.clone());
        successor.root_keys = root_key_records(&new_seeds, "new-root");
        let successor_bytes = serde_json_canonicalizer::to_vec(&successor).unwrap();
        let mut dual_signers = old_seeds[..3].to_vec();
        dual_signers.extend_from_slice(&new_seeds[..3]);
        let dual_signed = sign_payload(TRUST_ROOT_PAYLOAD_TYPE, &successor_bytes, &dual_signers);
        let rotated = verify_trust_root_rotation(&pinned, &dual_signed).unwrap();
        assert_eq!(rotated.payload().trust_root_version, "2");
        assert_eq!(rotated.payload().previous_trust_root_digest.as_deref(), Some(current_digest.as_str()));

        let old_only = sign_payload(TRUST_ROOT_PAYLOAD_TYPE, &successor_bytes, &old_seeds[..3]);
        assert!(verify_trust_root_rotation(&pinned, &old_only).is_err());
    }

    #[test]
    fn witness_set_rotation_requires_a_linked_incremented_metadata_version() {
        let root_seeds = [test_seed(161), test_seed(162), test_seed(163), test_seed(164)];
        let first_witness_seeds = [test_seed(171), test_seed(172), test_seed(173), test_seed(174)];
        let next_witness_seeds = [test_seed(181), test_seed(182), test_seed(183), test_seed(184)];
        let first_trust = build_trust(&first_witness_seeds);
        let first_metadata = witness_metadata_from_trust(&first_trust, None);
        let root = make_trust_root(root_key_records(&root_seeds, "root-domain"), &first_metadata);
        let root_digest = trust_root_digest(&root).unwrap();
        let root_bytes = serde_json_canonicalizer::to_vec(&root).unwrap();
        let pinned = load_pinned_trust_root(&root_bytes, &root_digest).unwrap();

        let first_payload = serde_json_canonicalizer::to_vec(&first_metadata).unwrap();
        let first_envelope = sign_payload(WITNESS_SET_PAYLOAD_TYPE, &first_payload, &root_seeds[..3]);
        let verified_first = verify_witness_set_metadata(&first_envelope, &pinned, None).unwrap();

        let next_trust = build_trust(&next_witness_seeds);
        let mut next_metadata = witness_metadata_from_trust(
            &next_trust,
            Some(witness_set_digest(&first_metadata).unwrap()),
        );
        next_metadata.witness_set_id = "basalt-test-witness-set-2".to_string();
        next_metadata.witness_set_version = "2".to_string();
        let next_payload = serde_json_canonicalizer::to_vec(&next_metadata).unwrap();
        let next_envelope = sign_payload(WITNESS_SET_PAYLOAD_TYPE, &next_payload, &root_seeds[..3]);
        let next = verify_witness_set_metadata(&next_envelope, &pinned, Some(&verified_first)).unwrap();
        assert_eq!(next.metadata().witness_set_version, "2");

        next_metadata.previous_witness_set_digest = Some(
            "sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee".to_string()
        );
        let fork_payload = serde_json_canonicalizer::to_vec(&next_metadata).unwrap();
        let fork_envelope = sign_payload(WITNESS_SET_PAYLOAD_TYPE, &fork_payload, &root_seeds[..3]);
        assert!(verify_witness_set_metadata(&fork_envelope, &pinned, Some(&verified_first)).is_err());
    }

}
