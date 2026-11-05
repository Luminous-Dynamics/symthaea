//! Shared DSSE/Ed25519 evidence verification for connectomics tools.
//!
//! A verified receipt authenticates exact payload bytes under a caller-pinned
//! role-bound trust policy. It does not establish that the statement is true.

use base64::{Engine, engine::general_purpose::STANDARD};
use ed25519_dalek::{Signature, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const EXPECTED_PAYLOAD_TYPE: &str =
    "application/vnd.luminous.connectome-evidence+json;version=1";

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DsseEnvelope {
    pub payload: String,
    pub payload_type: String,
    pub signatures: Vec<DsseSignature>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DsseSignature {
    pub keyid: String,
    pub sig: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TrustPolicy {
    pub schema_version: u32,
    pub policy_id: String,
    pub minimum_signatures: usize,
    pub required_roles: Vec<String>,
    pub trusted_keys: Vec<TrustedKey>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TrustedKey {
    pub key_id: String,
    pub public_key_hex: String,
    pub role: String,
    pub valid_from_unix: i64,
    pub valid_until_unix: Option<i64>,
    pub revoked_at_unix: Option<i64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceStatement {
    pub schema_version: u32,
    pub statement_id: String,
    pub statement_type: String,
    pub claim_scope: String,
    pub dataset_revision_commit_sha: String,
    pub query_id: String,
    pub query_spec_sha256: String,
    pub subject_sha256: String,
    pub evidence_sha256: String,
    pub protocol_sha256: String,
    pub observer_record_id: String,
    pub issued_at_unix: i64,
    pub expires_at_unix: Option<i64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct VerifiedSigner {
    pub key_id: String,
    pub role: String,
}

#[derive(Debug, Serialize)]
pub struct VerificationReceipt {
    pub schema_version: u32,
    pub tool: &'static str,
    pub envelope_sha256: String,
    pub payload_sha256: String,
    pub trust_policy_sha256: String,
    pub verification_time_unix: i64,
    pub payload_type: String,
    pub statement_id: String,
    pub statement_type: String,
    pub claim_scope: String,
    pub dataset_revision_commit_sha: String,
    pub query_id: String,
    pub query_spec_sha256: String,
    pub subject_sha256: String,
    pub evidence_sha256: String,
    pub protocol_sha256: String,
    pub observer_record_id: String,
    pub issued_at_unix: i64,
    pub expires_at_unix: Option<i64>,
    pub required_roles: Vec<String>,
    pub covered_roles: Vec<String>,
    pub minimum_signatures: usize,
    pub verified_signer_count: usize,
    pub verified_signers: Vec<VerifiedSigner>,
    pub policy_accepted: bool,
    pub status: &'static str,
    pub signature_claim: &'static str,
    pub trust_policy_claim: &'static str,
    pub scientific_truth_claim: &'static str,
    pub scope_note: &'static str,
}

pub fn sha256(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

pub fn valid_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

pub fn dsse_pae(payload_type: &str, payload: &[u8]) -> Vec<u8> {
    let header = format!(
        "DSSEv1 {} {} {} ",
        payload_type.len(),
        payload_type,
        payload.len()
    );
    let mut pae = header.into_bytes();
    pae.extend_from_slice(payload);
    pae
}

fn validate_statement(
    statement: &EvidenceStatement,
    verification_time_unix: i64,
) -> Result<(), String> {
    if statement.schema_version != 1 {
        return Err("attestation payload schema_version must equal 1".into());
    }
    for (name, value) in [
        ("statement_id", statement.statement_id.as_str()),
        ("statement_type", statement.statement_type.as_str()),
        ("claim_scope", statement.claim_scope.as_str()),
        ("query_id", statement.query_id.as_str()),
        ("observer_record_id", statement.observer_record_id.as_str()),
    ] {
        if !nonempty(value) {
            return Err(format!("attestation {name} must be non-empty"));
        }
    }
    if !valid_hex(&statement.dataset_revision_commit_sha, 40)
        || !valid_hex(&statement.query_spec_sha256, 64)
        || !valid_hex(&statement.subject_sha256, 64)
        || !valid_hex(&statement.evidence_sha256, 64)
        || !valid_hex(&statement.protocol_sha256, 64)
    {
        return Err(
            "attestation contains malformed dataset/evidence/protocol SHA-256 binding".into(),
        );
    }
    if statement.issued_at_unix <= 0 || statement.issued_at_unix > verification_time_unix {
        return Err("attestation issue time is invalid or later than verification time".into());
    }
    if statement
        .expires_at_unix
        .is_some_and(|expires_at| expires_at <= verification_time_unix)
    {
        return Err("attestation has expired at the declared verification time".into());
    }
    Ok(())
}

fn validate_trust_policy(policy: &TrustPolicy) -> Result<BTreeMap<&str, &TrustedKey>, String> {
    if policy.schema_version != 1 || !nonempty(&policy.policy_id) {
        return Err("trust policy schema_version or policy_id is invalid".into());
    }
    if policy.minimum_signatures == 0 || policy.trusted_keys.is_empty() {
        return Err("trust policy requires at least one signer and a positive signature threshold".into());
    }
    let mut required_roles = BTreeSet::new();
    for role in &policy.required_roles {
        if !nonempty(role) || !required_roles.insert(role.as_str()) {
            return Err("trust policy required roles must be non-empty and unique".into());
        }
    }
    let mut keys = BTreeMap::new();
    let mut public_key_material = BTreeSet::new();
    for key in &policy.trusted_keys {
        if !nonempty(&key.key_id)
            || !nonempty(&key.role)
            || !valid_hex(&key.public_key_hex, 64)
            || key.valid_from_unix <= 0
            || key
                .valid_until_unix
                .is_some_and(|valid_until| valid_until < key.valid_from_unix)
            || key
                .revoked_at_unix
                .is_some_and(|revoked_at| revoked_at <= 0)
        {
            return Err(format!(
                "trust key {} has malformed identity, role, key, or validity",
                key.key_id
            ));
        }
        if !public_key_material.insert(key.public_key_hex.as_str()) {
            return Err(format!(
                "trust policy reuses Ed25519 public-key material under key ID {}",
                key.key_id
            ));
        }
        if keys.insert(key.key_id.as_str(), key).is_some() {
            return Err(format!("trust policy repeats key ID {}", key.key_id));
        }
    }
    if policy.minimum_signatures > keys.len() {
        return Err("trust policy signature threshold exceeds trusted-key count".into());
    }
    Ok(keys)
}

pub fn verify_envelope(
    envelope_bytes: &[u8],
    policy_bytes: &[u8],
    expected_policy_sha256: &str,
    verification_time_unix: i64,
) -> Result<VerificationReceipt, String> {
    if !valid_hex(expected_policy_sha256, 64) {
        return Err("expected trust-policy SHA-256 pin is malformed".into());
    }
    let actual_policy_sha256 = sha256(policy_bytes);
    if actual_policy_sha256 != expected_policy_sha256 {
        return Err("trust-policy SHA-256 does not match the caller-supplied pin".into());
    }
    let envelope: DsseEnvelope = serde_json::from_slice(envelope_bytes)
        .map_err(|error| format!("invalid DSSE envelope JSON: {error}"))?;
    let policy: TrustPolicy = serde_json::from_slice(policy_bytes)
        .map_err(|error| format!("invalid trust policy JSON: {error}"))?;
    if envelope.payload_type != EXPECTED_PAYLOAD_TYPE {
        return Err("DSSE payloadType does not match the connectome evidence media type".into());
    }
    if envelope.signatures.is_empty() {
        return Err("DSSE envelope contains no signatures".into());
    }

    let payload_bytes = STANDARD
        .decode(&envelope.payload)
        .map_err(|error| format!("DSSE payload is not valid standard base64: {error}"))?;
    let statement: EvidenceStatement = serde_json::from_slice(&payload_bytes)
        .map_err(|error| format!("invalid signed evidence payload JSON: {error}"))?;
    validate_statement(&statement, verification_time_unix)?;
    let key_map = validate_trust_policy(&policy)?;

    let pae = dsse_pae(&envelope.payload_type, &payload_bytes);
    let mut seen_key_ids = BTreeSet::new();
    let mut covered_roles = BTreeSet::new();
    let mut verified_signers = Vec::with_capacity(envelope.signatures.len());

    for entry in &envelope.signatures {
        if !nonempty(&entry.keyid) || !seen_key_ids.insert(entry.keyid.as_str()) {
            return Err(format!(
                "DSSE signature key ID is empty or duplicated: {}",
                entry.keyid
            ));
        }
        let key = key_map.get(entry.keyid.as_str()).ok_or_else(|| {
            format!(
                "DSSE signer {} is not in the pinned trust policy",
                entry.keyid
            )
        })?;
        if verification_time_unix < key.valid_from_unix
            || key
                .valid_until_unix
                .is_some_and(|valid_until| verification_time_unix > valid_until)
        {
            return Err(format!(
                "trusted signing key {} is outside its validity interval",
                key.key_id
            ));
        }
        if key
            .revoked_at_unix
            .is_some_and(|revoked_at| verification_time_unix >= revoked_at)
        {
            return Err(format!(
                "trusted signing key {} is revoked at verification time",
                key.key_id
            ));
        }
        if statement.issued_at_unix < key.valid_from_unix
            || key
                .valid_until_unix
                .is_some_and(|valid_until| statement.issued_at_unix > valid_until)
        {
            return Err(format!(
                "statement issue time is outside key {} validity interval",
                key.key_id
            ));
        }

        let public_key_bytes = hex::decode(&key.public_key_hex)
            .map_err(|error| format!("trusted key {} has invalid hex: {error}", key.key_id))?;
        let public_key_array: [u8; 32] = public_key_bytes
            .try_into()
            .map_err(|_| format!("trusted key {} is not 32 bytes", key.key_id))?;
        let verifying_key = VerifyingKey::from_bytes(&public_key_array).map_err(|error| {
            format!(
                "trusted key {} is not a valid Ed25519 public key: {error}",
                key.key_id
            )
        })?;
        let signature_bytes = STANDARD.decode(&entry.sig).map_err(|error| {
            format!(
                "signature from {} is not valid standard base64: {error}",
                key.key_id
            )
        })?;
        let signature = Signature::from_slice(&signature_bytes)
            .map_err(|error| format!("signature from {} is not 64 bytes: {error}", key.key_id))?;
        verifying_key
            .verify_strict(&pae, &signature)
            .map_err(|_| format!("DSSE signature verification failed for {}", key.key_id))?;

        covered_roles.insert(key.role.as_str());
        verified_signers.push(VerifiedSigner {
            key_id: key.key_id.clone(),
            role: key.role.clone(),
        });
    }

    if verified_signers.len() < policy.minimum_signatures {
        return Err(format!(
            "verified signer count {} is below policy threshold {}",
            verified_signers.len(),
            policy.minimum_signatures
        ));
    }
    let missing_roles = policy
        .required_roles
        .iter()
        .filter(|role| !covered_roles.contains(role.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    if !missing_roles.is_empty() {
        return Err(format!(
            "required trust-policy roles are missing: {}",
            missing_roles.join(", ")
        ));
    }

    verified_signers.sort_by(|left, right| left.key_id.cmp(&right.key_id));
    let mut covered_roles = covered_roles.into_iter().map(str::to_string).collect::<Vec<_>>();
    covered_roles.sort();

    Ok(VerificationReceipt {
        schema_version: 1,
        tool: "connectome-evidence-attestation-audit",
        envelope_sha256: sha256(envelope_bytes),
        payload_sha256: sha256(&payload_bytes),
        trust_policy_sha256: actual_policy_sha256,
        verification_time_unix,
        payload_type: envelope.payload_type,
        statement_id: statement.statement_id,
        statement_type: statement.statement_type,
        claim_scope: statement.claim_scope,
        dataset_revision_commit_sha: statement.dataset_revision_commit_sha,
        query_id: statement.query_id,
        query_spec_sha256: statement.query_spec_sha256,
        subject_sha256: statement.subject_sha256,
        evidence_sha256: statement.evidence_sha256,
        protocol_sha256: statement.protocol_sha256,
        observer_record_id: statement.observer_record_id,
        issued_at_unix: statement.issued_at_unix,
        expires_at_unix: statement.expires_at_unix,
        required_roles: policy.required_roles,
        covered_roles,
        minimum_signatures: policy.minimum_signatures,
        verified_signer_count: verified_signers.len(),
        verified_signers,
        policy_accepted: true,
        status: "SIGNED_EVIDENCE_VERIFIED_UNDER_PINNED_TRUST_POLICY",
        signature_claim: "ED25519_SIGNATURES_VALID_OVER_DSSE_PAE",
        trust_policy_claim: "TRUST_POLICY_MATCHED_CALLER_SUPPLIED_SHA256_PIN",
        scientific_truth_claim: "NOT_ESTABLISHED",
        scope_note: "This receipt establishes that the listed trusted keys signed the exact DSSE payload bytes under the pinned trust policy at the declared verification time. It does not establish signer honesty, observation accuracy, protocol validity, sample representativeness, evidence independence or biological truth. The trust-policy pin must be anchored out-of-band. The statement issue time is signer-asserted unless separately timestamped by an independently trusted transparency service.",
    })
}
