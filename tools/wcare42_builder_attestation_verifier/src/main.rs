use chrono::{DateTime, NaiveDateTime, Utc};
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{collections::HashSet, env, fs, path::Path, process};

const PROTOCOL: &str = "wcare41-authenticated-preregistration-v1";
const VERIFIER_PROTOCOL: &str = "wcare42-builder-attestation-verifier-v1";
const WCARE40_PROTOCOL: &str = "wcare40-execution-replication-v1";
const DOMAIN: &str = "builder-evidence-attestation";

const PROVENANCE_STRENGTHS: &[&str] = &[
    "SelfDeclared",
    "OrganizerVerified",
    "ExternalVerified",
    "InstitutionalAttestation",
];
const RELATION_STRENGTHS: &[&str] = &[
    "SelfDeclared",
    "OrganizerAssessed",
    "ExternalVerified",
    "InstitutionalAttestation",
];
const ENVELOPE_KEYS: &[&str] = &[
    "protocol_version",
    "wcare40_plan_sha256",
    "wcare40_result_sha256",
    "subject_kind",
    "subject_receipt_sha256",
    "provenance_strength_claim",
    "relation_evidence_strength_claim",
    "issuer_key_id",
    "issuer_public_key_ed25519_hex",
    "issuer_policy_sha256",
    "issued_at_utc",
    "expires_at_utc",
    "nonce_sha256",
    "domain",
    "signature_ed25519_hex",
];
const POLICY_KEYS: &[&str] = &[
    "protocol_version",
    "policy_id",
    "policy_created_utc",
    "issuer_keys",
    "notes",
];
const ISSUER_ENTRY_KEYS: &[&str] = &[
    "issuer_key_id",
    "issuer_public_key_ed25519_hex",
    "issuer_identity_commitment_sha256",
    "valid_from_utc",
    "valid_until_utc",
    "revocation_effective_utc",
    "allowed_subject_kinds",
    "allowed_provenance_strengths",
    "allowed_relation_evidence_strengths",
];

#[derive(Clone)]
struct Artifact {
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
        json,
    })
}

fn field<'a>(value: &'a Value, name: &str) -> Result<&'a str, String> {
    value
        .get(name)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing_or_nonstring:{name}"))
}

fn exact_object_keys(value: &Value, allowed: &[&str], label: &str) -> Result<(), String> {
    let object = value
        .as_object()
        .ok_or_else(|| format!("{label}_not_object"))?;
    for key in object.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(format!("{label}_unknown_field:{key}"));
        }
    }
    Ok(())
}

fn is_lower_hex(value: &str, len: usize) -> bool {
    value.len() == len
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn is_token(value: &str) -> bool {
    !value.is_empty()
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b':' | b'-'))
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
    exact_object_keys(envelope, ENVELOPE_KEYS, "envelope")?;
    if field(envelope, "protocol_version")? != PROTOCOL {
        return Err("protocol_version_mismatch".into());
    }
    if field(envelope, "domain")? != DOMAIN {
        return Err("domain_mismatch".into());
    }

    let plan = field(envelope, "wcare40_plan_sha256")?;
    let result = field(envelope, "wcare40_result_sha256")?;
    let kind = field(envelope, "subject_kind")?;
    let receipt = field(envelope, "subject_receipt_sha256")?;
    let provenance = field(envelope, "provenance_strength_claim")?;
    let relation = field(envelope, "relation_evidence_strength_claim")?;
    let key_id = field(envelope, "issuer_key_id")?;
    let public_key = field(envelope, "issuer_public_key_ed25519_hex")?;
    let policy_sha = field(envelope, "issuer_policy_sha256")?;
    let issued = field(envelope, "issued_at_utc")?;
    let expires = field(envelope, "expires_at_utc")?;
    let nonce = field(envelope, "nonce_sha256")?;
    let signature = field(envelope, "signature_ed25519_hex")?;

    for (name, value) in [("plan", plan), ("receipt", receipt), ("policy", policy_sha), ("nonce", nonce)] {
        if !is_lower_hex(value, 64) {
            return Err(format!("invalid_64_hex:{name}"));
        }
    }
    if result != "-" && !is_lower_hex(result, 64) {
        return Err("invalid_result_sha256".into());
    }
    if !is_lower_hex(public_key, 64) || !is_lower_hex(signature, 128) {
        return Err("invalid_signature_or_public_key_hex".into());
    }
    if !is_token(key_id) {
        return Err("invalid_issuer_key_id".into());
    }

    let issued_dt = parse_utc(issued)?;
    if expires != "-" {
        let expires_dt = parse_utc(expires)?;
        if expires_dt <= issued_dt {
            return Err("expiry_not_after_issue_time".into());
        }
    }

    match kind {
        "BuilderProvenance" => {
            if !PROVENANCE_STRENGTHS.contains(&provenance) || relation != "-" {
                return Err("invalid_builder_provenance_scope_fields".into());
            }
        }
        "BuilderRelation" => {
            if provenance != "-" || !RELATION_STRENGTHS.contains(&relation) {
                return Err("invalid_builder_relation_scope_fields".into());
            }
        }
        _ => return Err("invalid_subject_kind".into()),
    }

    Ok(format!(
        "SYMTHAEA-WCARE41-BUILDER-ATTESTATION-V1\n\
protocol_version={PROTOCOL}\n\
wcare40_plan_sha256={plan}\n\
wcare40_result_sha256={result}\n\
subject_kind={kind}\n\
subject_receipt_sha256={receipt}\n\
provenance_strength_claim={provenance}\n\
relation_evidence_strength_claim={relation}\n\
issuer_key_id={key_id}\n\
issuer_public_key_ed25519_hex={public_key}\n\
issuer_policy_sha256={policy_sha}\n\
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

fn array_strings(value: &Value, name: &str) -> Result<Vec<String>, String> {
    let array = value
        .get(name)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("missing_or_nonarray:{name}"))?;
    let mut seen = HashSet::new();
    let mut result = Vec::with_capacity(array.len());
    for item in array {
        let item = item
            .as_str()
            .ok_or_else(|| format!("nonstr_array_item:{name}"))?;
        if !seen.insert(item.to_string()) {
            return Err(format!("duplicate_array_item:{name}:{item}"));
        }
        result.push(item.to_string());
    }
    Ok(result)
}

fn validate_policy(policy: &Value) -> Result<Vec<&Value>, String> {
    exact_object_keys(policy, POLICY_KEYS, "policy")?;
    if field(policy, "protocol_version")? != PROTOCOL {
        return Err("policy_protocol_mismatch".into());
    }
    if !is_token(field(policy, "policy_id")?) {
        return Err("policy_id_invalid".into());
    }
    parse_utc(field(policy, "policy_created_utc")?)?;
    let keys = policy
        .get("issuer_keys")
        .and_then(Value::as_array)
        .ok_or_else(|| "policy_issuer_keys_missing".to_string())?;
    if keys.is_empty() {
        return Err("policy_issuer_keys_empty".into());
    }
    let mut ids = HashSet::new();
    for entry in keys {
        exact_object_keys(entry, ISSUER_ENTRY_KEYS, "issuer_entry")?;
        let id = field(entry, "issuer_key_id")?;
        if !is_token(id) || !ids.insert(id.to_string()) {
            return Err(format!("issuer_key_id_invalid_or_duplicate:{id}"));
        }
        if !is_lower_hex(field(entry, "issuer_public_key_ed25519_hex")?, 64)
            || !is_lower_hex(field(entry, "issuer_identity_commitment_sha256")?, 64)
        {
            return Err(format!("issuer_entry_digest_invalid:{id}"));
        }
        let valid_from = parse_utc(field(entry, "valid_from_utc")?)?;
        if let Some(valid_until) = entry.get("valid_until_utc") {
            if !valid_until.is_null() {
                let valid_until = parse_utc(valid_until.as_str().ok_or("valid_until_not_string")?)?;
                if valid_until <= valid_from {
                    return Err(format!("issuer_valid_interval_invalid:{id}"));
                }
            }
        }
        if let Some(revoked) = entry.get("revocation_effective_utc") {
            if !revoked.is_null() {
                parse_utc(revoked.as_str().ok_or("revocation_not_string")?)?;
            }
        }
        for scope in array_strings(entry, "allowed_subject_kinds")? {
            if !matches!(scope.as_str(), "BuilderProvenance" | "BuilderRelation") {
                return Err(format!("issuer_scope_invalid:{id}:{scope}"));
            }
        }
        for strength in array_strings(entry, "allowed_provenance_strengths")? {
            if !PROVENANCE_STRENGTHS.contains(&strength.as_str()) {
                return Err(format!("issuer_provenance_strength_invalid:{id}:{strength}"));
            }
        }
        for strength in array_strings(entry, "allowed_relation_evidence_strengths")? {
            if !RELATION_STRENGTHS.contains(&strength.as_str()) {
                return Err(format!("issuer_relation_strength_invalid:{id}:{strength}"));
            }
        }
    }
    Ok(keys.iter().collect())
}

fn subject_binding(
    envelope: &Value,
    policy: &Artifact,
    plan: &Artifact,
    result: &Artifact,
    subject: &Artifact,
) -> Result<(bool, bool), String> {
    if field(envelope, "wcare40_plan_sha256")? != plan.sha256.as_str()
        || field(envelope, "subject_receipt_sha256")? != subject.sha256.as_str()
        || field(envelope, "issuer_policy_sha256")? != policy.sha256.as_str()
    {
        return Ok((false, false));
    }
    if plan.json.get("protocol_version").and_then(Value::as_str) != Some(WCARE40_PROTOCOL)
        || result.json.get("protocol_version").and_then(Value::as_str) != Some(WCARE40_PROTOCOL)
        || subject.json.get("protocol_version").and_then(Value::as_str) != Some(WCARE40_PROTOCOL)
    {
        return Ok((false, false));
    }
    if subject.json.get("plan_sha256").and_then(Value::as_str) != Some(plan.sha256.as_str()) {
        return Ok((false, false));
    }
    if let Some(result_plan) = result.json.get("plan_sha256").and_then(Value::as_str) {
        if result_plan != plan.sha256.as_str() {
            return Ok((false, false));
        }
    }

    let result_claim = field(envelope, "wcare40_result_sha256")?;
    let result_bound = result_claim != "-";
    if result_bound && result_claim != result.sha256.as_str() {
        return Ok((false, true));
    }

    match field(envelope, "subject_kind")? {
        "BuilderProvenance" => Ok((
            subject.json.get("provenance_strength").and_then(Value::as_str)
                == Some(field(envelope, "provenance_strength_claim")?),
            result_bound,
        )),
        "BuilderRelation" => Ok((
            subject.json.get("relation_evidence_strength").and_then(Value::as_str)
                == Some(field(envelope, "relation_evidence_strength_claim")?),
            result_bound,
        )),
        _ => Ok((false, result_bound)),
    }
}

fn evaluate_trust(
    envelope: &Value,
    policy: &Value,
    evaluation: DateTime<Utc>,
    checks: &mut Checks,
) -> Result<(), String> {
    let keys = validate_policy(policy)?;
    let issued = parse_utc(field(envelope, "issued_at_utc")?)?;
    let policy_created = parse_utc(field(policy, "policy_created_utc")?)?;
    let expires = field(envelope, "expires_at_utc")?;
    checks.attestation_current_at_evaluation = if expires == "-" {
        evaluation >= issued
    } else {
        let expires = parse_utc(expires)?;
        evaluation >= issued && evaluation < expires
    };

    let key_id = field(envelope, "issuer_key_id")?;
    let public_key = field(envelope, "issuer_public_key_ed25519_hex")?;
    let kind = field(envelope, "subject_kind")?;
    let claim = if kind == "BuilderProvenance" {
        field(envelope, "provenance_strength_claim")?
    } else {
        field(envelope, "relation_evidence_strength_claim")?
    };

    let matching = keys.into_iter().find(|entry| {
        entry.get("issuer_key_id").and_then(Value::as_str) == Some(key_id)
            && entry
                .get("issuer_public_key_ed25519_hex")
                .and_then(Value::as_str)
                == Some(public_key)
    });
    let Some(entry) = matching else {
        return Ok(());
    };
    checks.issuer_key_present = true;

    let valid_from = parse_utc(field(entry, "valid_from_utc")?)?;
    let valid_until = entry.get("valid_until_utc");
    checks.key_valid_at_issue_time = issued >= valid_from
        && valid_until
            .and_then(Value::as_str)
            .map(|value| parse_utc(value).map(|dt| issued < dt))
            .transpose()?
            .unwrap_or(true)
        && policy_created <= issued;

    let revoked = entry.get("revocation_effective_utc");
    checks.revocation_policy_satisfied = revoked
        .and_then(Value::as_str)
        .map(|value| parse_utc(value).map(|dt| issued < dt))
        .transpose()?
        .unwrap_or(true);

    checks.scope_authorized = array_strings(entry, "allowed_subject_kinds")?
        .iter()
        .any(|item| item == kind);
    checks.claimed_strength_authorized = if kind == "BuilderProvenance" {
        array_strings(entry, "allowed_provenance_strengths")?
            .iter()
            .any(|item| item == claim)
    } else {
        array_strings(entry, "allowed_relation_evidence_strengths")?
            .iter()
            .any(|item| item == claim)
    };
    checks.issuer_trusted_for_claim = checks.issuer_key_present
        && checks.key_valid_at_issue_time
        && checks.revocation_policy_satisfied
        && checks.attestation_current_at_evaluation
        && checks.scope_authorized
        && checks.claimed_strength_authorized;
    Ok(())
}

fn emit(
    disposition: &str,
    detail: &str,
    envelope: Option<&Artifact>,
    policy: Option<&Artifact>,
    plan: Option<&Artifact>,
    result: Option<&Artifact>,
    subject: Option<&Artifact>,
    canonical_message_sha256: Option<&str>,
    result_bound: bool,
    checks: &Checks,
) -> i32 {
    let accepted = disposition == "ATTESTATION_ACCEPTED";
    let payload = json!({
        "authority": "MeasurementOnly",
        "verifier_protocol_version": VERIFIER_PROTOCOL,
        "wcare41_protocol_version": PROTOCOL,
        "disposition": disposition,
        "detail": detail,
        "envelope_sha256": envelope.map(|x| x.sha256.clone()),
        "issuer_trust_policy_sha256": policy.map(|x| x.sha256.clone()),
        "wcare40_plan_sha256": plan.map(|x| x.sha256.clone()),
        "wcare40_result_sha256": result.map(|x| x.sha256.clone()),
        "subject_receipt_sha256": subject.map(|x| x.sha256.clone()),
        "canonical_message_sha256": canonical_message_sha256,
        "issuer_key_id": envelope.and_then(|x| x.json.get("issuer_key_id")).and_then(Value::as_str),
        "subject_kind": envelope.and_then(|x| x.json.get("subject_kind")).and_then(Value::as_str),
        "result_bound_by_signature": result_bound,
        "checks": {
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
        },
        "attestation_accepted": accepted,
        "builder_authentication_established": false,
        "preregistration_temporal_precedence_established": false,
        "subject_correctness_established": false,
        "runtime_authority_granted": false,
    });
    println!("{}", serde_json::to_string(&payload).expect("serialize verifier result"));
    match disposition {
        "ATTESTATION_ACCEPTED" => 0,
        "SIGNATURE_VALID_ISSUER_UNTRUSTED" => 2,
        "ATTESTATION_REJECTED" => 4,
        _ => 3,
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 7 {
        let checks = Checks::default();
        process::exit(emit(
            "ATTESTATION_REJECTED",
            "usage: verifier ENVELOPE POLICY WCARE40_PLAN WCARE40_RESULT SUBJECT_RECEIPT EVALUATION_UTC",
            None,
            None,
            None,
            None,
            None,
            None,
            false,
            &checks,
        ));
    }

    let evaluation = match parse_utc(&args[6]) {
        Ok(value) => value,
        Err(error) => {
            let checks = Checks::default();
            process::exit(emit(
                "ATTESTATION_REJECTED",
                &error,
                None,
                None,
                None,
                None,
                None,
                None,
                false,
                &checks,
            ));
        }
    };

    let envelope = load(Path::new(&args[1]));
    let policy = load(Path::new(&args[2]));
    let plan = load(Path::new(&args[3]));
    let result = load(Path::new(&args[4]));
    let subject = load(Path::new(&args[5]));
    let (envelope, policy, plan, result, subject) = match (envelope, policy, plan, result, subject) {
        (Ok(a), Ok(b), Ok(c), Ok(d), Ok(e)) => (a, b, c, d, e),
        _ => {
            let checks = Checks::default();
            process::exit(emit(
                "INFRASTRUCTURE_INDETERMINATE",
                "artifact_read_or_parse_failed",
                None,
                None,
                None,
                None,
                None,
                None,
                false,
                &checks,
            ));
        }
    };

    let mut checks = Checks::default();
    let message = match canonical_message(&envelope.json) {
        Ok(value) => {
            checks.canonicalization_valid = true;
            value
        }
        Err(error) => {
            process::exit(emit(
                "ATTESTATION_REJECTED",
                &error,
                Some(&envelope),
                Some(&policy),
                Some(&plan),
                Some(&result),
                Some(&subject),
                None,
                false,
                &checks,
            ));
        }
    };
    let message_sha = sha256_hex(message.as_bytes());

    checks.signature_valid = verify_signature(&envelope.json, message.as_bytes()).unwrap_or(false);
    if !checks.signature_valid {
        process::exit(emit(
            "ATTESTATION_REJECTED",
            "signature_invalid",
            Some(&envelope),
            Some(&policy),
            Some(&plan),
            Some(&result),
            Some(&subject),
            Some(&message_sha),
            false,
            &checks,
        ));
    }

    let (binding, result_bound) = match subject_binding(&envelope.json, &policy, &plan, &result, &subject) {
        Ok(value) => value,
        Err(error) => {
            process::exit(emit(
                "ATTESTATION_REJECTED",
                &error,
                Some(&envelope),
                Some(&policy),
                Some(&plan),
                Some(&result),
                Some(&subject),
                Some(&message_sha),
                false,
                &checks,
            ));
        }
    };
    checks.subject_binding_valid = binding;
    if !binding {
        process::exit(emit(
            "ATTESTATION_REJECTED",
            "subject_binding_invalid",
            Some(&envelope),
            Some(&policy),
            Some(&plan),
            Some(&result),
            Some(&subject),
            Some(&message_sha),
            result_bound,
            &checks,
        ));
    }

    if let Err(error) = evaluate_trust(&envelope.json, &policy.json, evaluation, &mut checks) {
        process::exit(emit(
            "ATTESTATION_REJECTED",
            &error,
            Some(&envelope),
            Some(&policy),
            Some(&plan),
            Some(&result),
            Some(&subject),
            Some(&message_sha),
            result_bound,
            &checks,
        ));
    }

    let disposition = if checks.issuer_trusted_for_claim {
        "ATTESTATION_ACCEPTED"
    } else {
        "SIGNATURE_VALID_ISSUER_UNTRUSTED"
    };
    process::exit(emit(
        disposition,
        if checks.issuer_trusted_for_claim {
            "signature_subject_and_trust_policy_accepted"
        } else {
            "signature_and_subject_valid_but_issuer_not_authorized_for_claim"
        },
        Some(&envelope),
        Some(&policy),
        Some(&plan),
        Some(&result),
        Some(&subject),
        Some(&message_sha),
        result_bound,
        &checks,
    ));
}
