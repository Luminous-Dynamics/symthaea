use chrono::{DateTime, NaiveDateTime, Utc};
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{env, fs, path::Path, process};

const PROTOCOL: &str = "wcare37-attestation-v1";
const PROVENANCE_PROTOCOL: &str = "wcare36-reviewer-provenance-v1";
const DOMAIN: &str = "reviewer-evidence-attestation";

const PROVENANCE_STRENGTHS: &[&str] = &[
    "SelfDeclared",
    "OrganizerVerified",
    "ExternalVerified",
    "InstitutionalAttestation",
    "ModelSessionProvenance",
];
const RELATION_STRENGTHS: &[&str] = &[
    "SelfDeclared",
    "OrganizerAssessed",
    "ExternalVerified",
    "InstitutionalAttestation",
    "ModelAssessment",
];

#[derive(Clone)]
struct Artifact {
    bytes: Vec<u8>,
    json: Value,
    sha256: String,
}

#[derive(Default)]
struct Checks {
    signature_valid: bool,
    subject_binding_valid: bool,
    canonicalization_valid: bool,
    issuer_key_present: bool,
    key_valid_at_issue_time: bool,
    revocation_policy_satisfied: bool,
    attestation_current_at_evaluation: bool,
    scope_authorized: bool,
    claimed_strength_authorized: bool,
    issuer_trusted_for_claim: bool,
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn load(path: &Path) -> Result<Artifact, String> {
    let bytes = fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let json = serde_json::from_slice(&bytes)
        .map_err(|e| format!("parse JSON {}: {e}", path.display()))?;
    Ok(Artifact {
        sha256: sha256_hex(&bytes),
        bytes,
        json,
    })
}

fn field<'a>(value: &'a Value, name: &str) -> Result<&'a str, String> {
    value
        .get(name)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing_or_nonstring:{name}"))
}

fn is_lower_hex(value: &str, len: usize) -> bool {
    value.len() == len
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn is_token(value: &str) -> bool {
    !value.is_empty()
        && value.bytes().all(|b| {
            b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b':' | b'-')
        })
}

fn parse_utc(value: &str) -> Result<DateTime<Utc>, String> {
    if value.len() != 20 || !value.ends_with('Z') {
        return Err(format!("noncanonical_utc:{value}"));
    }
    let naive = NaiveDateTime::parse_from_str(value, "%Y-%m-%dT%H:%M:%SZ")
        .map_err(|_| format!("invalid_utc:{value}"))?;
    let dt = DateTime::<Utc>::from_naive_utc_and_offset(naive, Utc);
    if dt.format("%Y-%m-%dT%H:%M:%SZ").to_string() != value {
        return Err(format!("noncanonical_utc:{value}"));
    }
    Ok(dt)
}

fn canonical_message(envelope: &Value) -> Result<String, String> {
    let protocol = field(envelope, "protocol_version")?;
    if protocol != PROTOCOL {
        return Err("protocol_version_mismatch".into());
    }

    let w36 = field(envelope, "wcare36_result_sha256")?;
    let w35 = field(envelope, "wcare35_result_sha256")?;
    let kind = field(envelope, "subject_receipt_kind")?;
    let receipt = field(envelope, "subject_receipt_sha256")?;
    let reviewer = field(envelope, "reviewer_identity_commitment_sha256")?;
    let provenance = field(envelope, "provenance_strength_claim")?;
    let relation = field(envelope, "relation_evidence_strength_claim")?;
    let key_id = field(envelope, "issuer_key_id")?;
    let public_key = field(envelope, "issuer_public_key_ed25519_hex")?;
    let policy_id = field(envelope, "issuer_policy_id")?;
    let issued = field(envelope, "issued_at_utc")?;
    let expires = field(envelope, "expires_at_utc")?;
    let nonce = field(envelope, "nonce_sha256")?;
    let signature = field(envelope, "signature_ed25519_hex")?;

    for (name, value) in [("wcare36", w36), ("wcare35", w35), ("receipt", receipt), ("nonce", nonce)] {
        if !is_lower_hex(value, 64) {
            return Err(format!("invalid_64_hex:{name}"));
        }
    }
    if reviewer != "-" && !is_lower_hex(reviewer, 64) {
        return Err("invalid_reviewer_identity_commitment".into());
    }
    if !is_lower_hex(public_key, 64) {
        return Err("invalid_issuer_public_key".into());
    }
    if !is_lower_hex(signature, 128) {
        return Err("invalid_signature_hex".into());
    }
    if !is_token(key_id) || !is_token(policy_id) {
        return Err("invalid_token".into());
    }
    let issued_dt = parse_utc(issued)?;
    if expires != "-" {
        let expires_dt = parse_utc(expires)?;
        if expires_dt <= issued_dt {
            return Err("expiry_not_after_issue_time".into());
        }
    }

    match kind {
        "ReviewerProvenance" => {
            if reviewer == "-" || !PROVENANCE_STRENGTHS.contains(&provenance) || relation != "-" {
                return Err("invalid_provenance_scope_fields".into());
            }
        }
        "ReviewerRelation" => {
            if provenance != "-" || !RELATION_STRENGTHS.contains(&relation) {
                return Err("invalid_relation_scope_fields".into());
            }
        }
        _ => return Err("invalid_subject_receipt_kind".into()),
    }

    Ok(format!(
        "SYMTHAEA-WCARE37-ATTESTATION-V1\n\
protocol_version={protocol}\n\
wcare36_result_sha256={w36}\n\
wcare35_result_sha256={w35}\n\
subject_receipt_kind={kind}\n\
subject_receipt_sha256={receipt}\n\
reviewer_identity_commitment_sha256={reviewer}\n\
provenance_strength_claim={provenance}\n\
relation_evidence_strength_claim={relation}\n\
issuer_key_id={key_id}\n\
issuer_public_key_ed25519_hex={public_key}\n\
issuer_policy_id={policy_id}\n\
issued_at_utc={issued}\n\
expires_at_utc={expires}\n\
nonce_sha256={nonce}\n\
domain={DOMAIN}\n"
    ))
}

fn verify_signature(envelope: &Value, message: &[u8]) -> Result<bool, String> {
    let public_key = hex::decode(field(envelope, "issuer_public_key_ed25519_hex")?)
        .map_err(|_| "invalid_public_key_hex".to_string())?;
    let signature = hex::decode(field(envelope, "signature_ed25519_hex")?)
        .map_err(|_| "invalid_signature_hex".to_string())?;
    let public_key: [u8; 32] = public_key
        .try_into()
        .map_err(|_| "invalid_public_key_length".to_string())?;
    let signature: [u8; 64] = signature
        .try_into()
        .map_err(|_| "invalid_signature_length".to_string())?;
    let key = VerifyingKey::from_bytes(&public_key)
        .map_err(|_| "invalid_ed25519_public_key".to_string())?;
    let signature = Signature::from_bytes(&signature);
    Ok(key.verify(message, &signature).is_ok())
}

fn subject_binding(
    envelope: &Value,
    policy: &Artifact,
    w36: &Artifact,
    w35: &Artifact,
    subject: &Artifact,
) -> Result<bool, String> {
    if field(envelope, "wcare36_result_sha256")? != w36.sha256
        || field(envelope, "wcare35_result_sha256")? != w35.sha256
        || field(envelope, "subject_receipt_sha256")? != subject.sha256
        || field(policy.json.as_ref(), "wcare36_result_sha256")? != w36.sha256
        || field(policy.json.as_ref(), "wcare35_result_sha256")? != w35.sha256
    {
        return Ok(false);
    }
    if field(envelope, "issuer_policy_id")? != field(policy.json.as_ref(), "issuer_policy_id")? {
        return Ok(false);
    }
    if subject.json.get("protocol_version").and_then(Value::as_str) != Some(PROVENANCE_PROTOCOL) {
        return Ok(false);
    }
    if subject.json.get("wcare35_result_sha256").and_then(Value::as_str) != Some(&w35.sha256) {
        return Ok(false);
    }
    if w36.json.get("wcare35_result_sha256").and_then(Value::as_str) != Some(&w35.sha256) {
        return Ok(false);
    }

    match field(envelope, "subject_receipt_kind")? {
        "ReviewerProvenance" => Ok(
            subject.json.get("reviewer_identity_commitment_sha256").and_then(Value::as_str)
                == Some(field(envelope, "reviewer_identity_commitment_sha256")?)
                && subject.json.get("provenance_strength").and_then(Value::as_str)
                    == Some(field(envelope, "provenance_strength_claim")?),
        ),
        "ReviewerRelation" => Ok(
            subject.json.get("relation_evidence_strength").and_then(Value::as_str)
                == Some(field(envelope, "relation_evidence_strength_claim")?),
        ),
        _ => Ok(false),
    }
}

fn array_contains(value: &Value, field_name: &str, needle: &str) -> bool {
    value
        .get(field_name)
        .and_then(Value::as_array)
        .is_some_and(|items| items.iter().any(|item| item.as_str() == Some(needle)))
}

fn emit_result(
    envelope: &Artifact,
    policy: &Artifact,
    w36: &Artifact,
    w35: &Artifact,
    subject: &Artifact,
    canonical_message_sha256: Option<String>,
    issuer_key_id: Option<String>,
    checks: &Checks,
    disposition: &str,
    detail: &str,
) {
    let payload = json!({
        "protocol_version": PROTOCOL,
        "wcare36_result_sha256": w36.sha256,
        "wcare35_result_sha256": w35.sha256,
        "subject_receipt_sha256": subject.sha256,
        "attestation_envelope_sha256": envelope.sha256,
        "issuer_trust_policy_sha256": policy.sha256,
        "canonical_message_sha256": canonical_message_sha256,
        "issuer_key_id": issuer_key_id,
        "signature_valid": checks.signature_valid,
        "subject_binding_valid": checks.subject_binding_valid,
        "canonicalization_valid": checks.canonicalization_valid,
        "issuer_key_present": checks.issuer_key_present,
        "key_valid_at_issue_time": checks.key_valid_at_issue_time,
        "revocation_policy_satisfied": checks.revocation_policy_satisfied,
        "attestation_current_at_evaluation": checks.attestation_current_at_evaluation,
        "scope_authorized": checks.scope_authorized,
        "claimed_strength_authorized": checks.claimed_strength_authorized,
        "issuer_trusted_for_claim": checks.issuer_trusted_for_claim,
        "disposition": disposition,
        "detail": detail,
        "signature_proves_key_control_only": true,
        "reviewer_independence_established": false,
        "moral_correctness_established": false,
        "runtime_authority_granted": false
    });
    println!("{}", serde_json::to_string(&payload).expect("serialize result"));
}

fn run() -> Result<i32, String> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 7 {
        return Err(format!(
            "usage: {} <envelope.json> <policy.json> <wcare36-result.json> <wcare35-result.json> <subject-receipt.json> <evaluation-utc>",
            args.first().map(String::as_str).unwrap_or("wcare37-attestation-verifier")
        ));
    }

    let envelope = load(Path::new(&args[1]))?;
    let policy = load(Path::new(&args[2]))?;
    let w36 = load(Path::new(&args[3]))?;
    let w35 = load(Path::new(&args[4]))?;
    let subject = load(Path::new(&args[5]))?;
    let evaluation_time = parse_utc(&args[6])?;

    let mut checks = Checks::default();
    let issuer_key_id = envelope
        .json
        .get("issuer_key_id")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);

    let canonical = match canonical_message(&envelope.json) {
        Ok(value) => {
            checks.canonicalization_valid = true;
            value
        }
        Err(detail) => {
            emit_result(
                &envelope,
                &policy,
                &w36,
                &w35,
                &subject,
                None,
                issuer_key_id,
                &checks,
                "ATTESTATION_REJECTED",
                &detail,
            );
            return Ok(1);
        }
    };
    let canonical_sha = sha256_hex(canonical.as_bytes());

    checks.subject_binding_valid = subject_binding(&envelope.json, &policy, &w36, &w35, &subject)?;
    if !checks.subject_binding_valid {
        emit_result(
            &envelope,
            &policy,
            &w36,
            &w35,
            &subject,
            Some(canonical_sha),
            issuer_key_id,
            &checks,
            "ATTESTATION_REJECTED",
            "subject_or_policy_binding_mismatch",
        );
        return Ok(1);
    }

    checks.signature_valid = verify_signature(&envelope.json, canonical.as_bytes())?;
    if !checks.signature_valid {
        emit_result(
            &envelope,
            &policy,
            &w36,
            &w35,
            &subject,
            Some(canonical_sha),
            issuer_key_id,
            &checks,
            "ATTESTATION_REJECTED",
            "ed25519_signature_invalid",
        );
        return Ok(1);
    }

    let issued = parse_utc(field(&envelope.json, "issued_at_utc")?)?;
    let expires = field(&envelope.json, "expires_at_utc")?;
    checks.attestation_current_at_evaluation = if expires == "-" {
        true
    } else {
        evaluation_time < parse_utc(expires)?
    };
    if !checks.attestation_current_at_evaluation {
        emit_result(
            &envelope,
            &policy,
            &w36,
            &w35,
            &subject,
            Some(canonical_sha),
            issuer_key_id,
            &checks,
            "ATTESTATION_REJECTED",
            "attestation_expired_at_evaluation_time",
        );
        return Ok(1);
    }

    let policy_created = parse_utc(field(&policy.json, "policy_created_utc")?)?;
    let policy_preregistered = policy_created <= issued;
    let key_id = field(&envelope.json, "issuer_key_id")?;
    let public_key = field(&envelope.json, "issuer_public_key_ed25519_hex")?;
    let keys = policy
        .json
        .get("keys")
        .and_then(Value::as_array)
        .ok_or_else(|| "policy_keys_missing".to_string())?;
    let matching_ids: Vec<&Value> = keys
        .iter()
        .filter(|entry| entry.get("issuer_key_id").and_then(Value::as_str) == Some(key_id))
        .collect();
    if matching_ids.len() > 1 {
        emit_result(
            &envelope,
            &policy,
            &w36,
            &w35,
            &subject,
            Some(canonical_sha),
            issuer_key_id,
            &checks,
            "ATTESTATION_REJECTED",
            "duplicate_issuer_key_id_in_policy",
        );
        return Ok(1);
    }

    if let Some(entry) = matching_ids.first().copied() {
        checks.issuer_key_present = entry
            .get("public_key_ed25519_hex")
            .and_then(Value::as_str)
            == Some(public_key);
        if checks.issuer_key_present {
            let valid_from = parse_utc(field(entry, "valid_from_utc")?)?;
            let valid_until = field(entry, "valid_until_utc")?;
            checks.key_valid_at_issue_time = issued >= valid_from
                && (valid_until == "-" || issued < parse_utc(valid_until)?);
            let revoked = field(entry, "revocation_effective_utc")?;
            checks.revocation_policy_satisfied =
                revoked == "-" || issued < parse_utc(revoked)?;

            let scope = field(&envelope.json, "subject_receipt_kind")?;
            checks.scope_authorized = array_contains(entry, "allowed_attestation_scopes", scope);
            checks.claimed_strength_authorized = match scope {
                "ReviewerProvenance" => array_contains(
                    entry,
                    "allowed_provenance_strengths",
                    field(&envelope.json, "provenance_strength_claim")?,
                ),
                "ReviewerRelation" => array_contains(
                    entry,
                    "allowed_relation_evidence_strengths",
                    field(&envelope.json, "relation_evidence_strength_claim")?,
                ),
                _ => false,
            };
        }
    }

    checks.issuer_trusted_for_claim = policy_preregistered
        && checks.issuer_key_present
        && checks.key_valid_at_issue_time
        && checks.revocation_policy_satisfied
        && checks.scope_authorized
        && checks.claimed_strength_authorized;

    let (disposition, detail, code) = if checks.issuer_trusted_for_claim {
        ("ATTESTATION_ACCEPTED", "signature_and_issuer_policy_accepted", 0)
    } else {
        (
            "SIGNATURE_VALID_ISSUER_UNTRUSTED",
            if !policy_preregistered {
                "trust_policy_created_after_attestation_issue_time"
            } else {
                "signature_valid_but_issuer_not_authorized_for_claim"
            },
            2,
        )
    };

    emit_result(
        &envelope,
        &policy,
        &w36,
        &w35,
        &subject,
        Some(canonical_sha),
        issuer_key_id,
        &checks,
        disposition,
        detail,
    );
    Ok(code)
}

fn main() {
    match run() {
        Ok(code) => process::exit(code),
        Err(error) => {
            eprintln!("wcare37 verifier infrastructure/error: {error}");
            process::exit(3);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};

    fn base_envelope(public_key: &str) -> Value {
        json!({
            "protocol_version": PROTOCOL,
            "wcare36_result_sha256": "11".repeat(32),
            "wcare35_result_sha256": "22".repeat(32),
            "subject_receipt_kind": "ReviewerProvenance",
            "subject_receipt_sha256": "33".repeat(32),
            "reviewer_identity_commitment_sha256": "44".repeat(32),
            "provenance_strength_claim": "ExternalVerified",
            "relation_evidence_strength_claim": "-",
            "issuer_key_id": "issuer.example.v1",
            "issuer_public_key_ed25519_hex": public_key,
            "issuer_policy_id": "policy.example.v1",
            "issued_at_utc": "2026-09-13T16:00:00Z",
            "expires_at_utc": "2026-10-13T16:00:00Z",
            "nonce_sha256": "55".repeat(32),
            "signature_ed25519_hex": "00".repeat(64)
        })
    }

    #[test]
    fn canonical_message_is_stable_and_signature_verifies() {
        let signing_key = SigningKey::from_bytes(&[7u8; 32]);
        let public_key = hex::encode(signing_key.verifying_key().as_bytes());
        let mut envelope = base_envelope(&public_key);
        let message = canonical_message(&envelope).unwrap();
        assert!(message.ends_with("domain=reviewer-evidence-attestation\n"));
        let signature = signing_key.sign(message.as_bytes());
        envelope["signature_ed25519_hex"] = Value::String(hex::encode(signature.to_bytes()));
        assert!(verify_signature(&envelope, message.as_bytes()).unwrap());
    }

    #[test]
    fn relation_scope_uses_relation_vocabulary() {
        let signing_key = SigningKey::from_bytes(&[9u8; 32]);
        let public_key = hex::encode(signing_key.verifying_key().as_bytes());
        let mut envelope = base_envelope(&public_key);
        envelope["subject_receipt_kind"] = Value::String("ReviewerRelation".into());
        envelope["reviewer_identity_commitment_sha256"] = Value::String("-".into());
        envelope["provenance_strength_claim"] = Value::String("-".into());
        envelope["relation_evidence_strength_claim"] = Value::String("OrganizerAssessed".into());
        assert!(canonical_message(&envelope).is_ok());
        envelope["relation_evidence_strength_claim"] = Value::String("OrganizerVerified".into());
        assert!(canonical_message(&envelope).is_err());
    }

    #[test]
    fn expiry_must_follow_issue_time() {
        let signing_key = SigningKey::from_bytes(&[11u8; 32]);
        let public_key = hex::encode(signing_key.verifying_key().as_bytes());
        let mut envelope = base_envelope(&public_key);
        envelope["expires_at_utc"] = Value::String("2026-09-13T16:00:00Z".into());
        assert_eq!(
            canonical_message(&envelope).unwrap_err(),
            "expiry_not_after_issue_time"
        );
    }
}
