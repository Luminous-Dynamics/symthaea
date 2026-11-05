//! Typed, fail-closed validation for package signature evaluation statements.
//!
//! The contract deliberately separates package content integrity, signature mathematics,
//! signer key/certificate identity, trust-policy acceptance, application authorization and
//! runtime behavior. It contains no aggregate "trusted" or "safe" boolean.

use jsonschema::draft202012;
use serde::de::{self, Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use serde_json::{Map, Number, Value};
use sha2::{Digest, Sha256};
use std::fmt;

const REQUIRED_NON_CLAIMS: &[&str] = &[
    "publisher_benevolence",
    "human_or_organization_identity_attribution",
    "absence_of_malicious_code",
    "safe_runtime_behavior",
    "user_intent_alignment",
    "uninspected_component_semantics",
    "future_update_safety",
    "complete_platform_policy_compliance",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvaluationError {
    InvalidSchema(String),
    SchemaViolation(String),
    SemanticViolation(String),
    Canonicalization(String),
}

impl fmt::Display for EvaluationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSchema(detail) => write!(formatter, "invalid evaluation schema: {detail}"),
            Self::SchemaViolation(detail) => {
                write!(formatter, "evaluation schema violation: {detail}")
            }
            Self::SemanticViolation(detail) => {
                write!(formatter, "evaluation semantic violation: {detail}")
            }
            Self::Canonicalization(detail) => {
                write!(formatter, "evaluation canonicalization failed: {detail}")
            }
        }
    }
}

impl std::error::Error for EvaluationError {}

pub fn validate_schema_document(schema: &Value) -> Result<(), EvaluationError> {
    jsonschema::meta::validate(schema)
        .map_err(|error| EvaluationError::InvalidSchema(error.to_string()))?;
    draft202012::options()
        .build(schema)
        .map_err(|error| EvaluationError::InvalidSchema(error.to_string()))?;
    Ok(())
}

/// Reject duplicate property names before serde_json::Value can collapse them.
pub fn parse_unique_json(input: &[u8]) -> Result<Value, EvaluationError> {
    let mut deserializer = serde_json::Deserializer::from_slice(input);
    let value = UniqueValue::deserialize(&mut deserializer)
        .map_err(|error| EvaluationError::SchemaViolation(error.to_string()))?;
    deserializer
        .end()
        .map_err(|error| EvaluationError::SchemaViolation(error.to_string()))?;
    Ok(value.0)
}

/// Parse a raw statement, reject duplicate keys, then validate schema and cross-field invariants.
pub fn parse_and_validate(schema: &Value, input: &[u8]) -> Result<Value, EvaluationError> {
    let statement = parse_unique_json(input)?;
    validate_evaluation(schema, &statement)?;
    Ok(statement)
}

pub fn validate_evaluation(schema: &Value, statement: &Value) -> Result<(), EvaluationError> {
    let validator = draft202012::options()
        .build(schema)
        .map_err(|error| EvaluationError::InvalidSchema(error.to_string()))?;
    validator
        .validate(statement)
        .map_err(|error| EvaluationError::SchemaViolation(error.to_string()))?;
    validate_semantics(statement)
}

/// Deterministic RFC 8785 bytes, only after schema and semantic validation.
pub fn canonicalize_evaluation(
    schema: &Value,
    statement: &Value,
) -> Result<Vec<u8>, EvaluationError> {
    validate_evaluation(schema, statement)?;
    serde_json_canonicalizer::to_vec(statement)
        .map_err(|error| EvaluationError::Canonicalization(error.to_string()))
}

/// Hash of the canonical evaluation statement; never a signature or trust assertion.
pub fn evaluation_sha256(
    schema: &Value,
    statement: &Value,
) -> Result<String, EvaluationError> {
    let canonical = canonicalize_evaluation(schema, statement)?;
    Ok(format!("{:x}", Sha256::digest(canonical)))
}

struct UniqueValue(Value);

impl<'de> Deserialize<'de> for UniqueValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(UniqueValueVisitor)
    }
}

struct UniqueValueVisitor;

impl<'de> Visitor<'de> for UniqueValueVisitor {
    type Value = UniqueValue;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON value with unique property names")
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::Bool(value)))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::Number(Number::from(value))))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::Number(Number::from(value))))
    }

    fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Number::from_f64(value)
            .map(|number| UniqueValue(Value::Number(number)))
            .ok_or_else(|| E::custom("non-finite JSON numbers are not permitted"))
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::String(value.to_owned())))
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::String(value)))
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::Null))
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::Null))
    }

    fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        UniqueValue::deserialize(deserializer)
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(value) = sequence.next_element::<UniqueValue>()? {
            values.push(value.0);
        }
        Ok(UniqueValue(Value::Array(values)))
    }

    fn visit_map<A>(self, mut object: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut values = Map::new();
        while let Some(key) = object.next_key::<String>()? {
            if values.contains_key(&key) {
                return Err(de::Error::custom(format!(
                    "duplicate JSON object property {key:?}"
                )));
            }
            let value = object.next_value::<UniqueValue>()?;
            values.insert(key, value.0);
        }
        Ok(UniqueValue(Value::Object(values)))
    }
}

fn validate_semantics(statement: &Value) -> Result<(), EvaluationError> {
    let package = object_at(statement, &["package"])?;
    let content_receipt = statement
        .get("content_receipt")
        .ok_or_else(|| semantic("content_receipt is required"))?;
    let platform = object_at(statement, &["platform"])?;
    let policy = object_at(statement, &["policy_context"])?;
    let claims = object_at(statement, &["claim_results"])?;
    let signers = array_at(statement, &["signers"])?;
    let schemes = array_at(statement, &["scheme_results"])?;
    let authorization = object_at(statement, &["authorization"])?;
    let runtime_context = object_at(statement, &["runtime_context"])?;
    let attestation = object_at(statement, &["attestation"])?;
    let format = string_at(package, &["format"])?;
    let platform_kind = string_at(platform, &["kind"])?;

    if (format == "apk" && platform_kind != "android")
        || (format == "ipa" && platform_kind != "apple")
    {
        return Err(semantic(
            "package format and platform kind must describe the same platform",
        ));
    }

    let package_digest = string_at(package, &["package_sha256"])?;
    let _package_size = decimal_string_at(package, &["package_size_bytes"])?;
    match content_receipt {
        Value::Null => {
            if claim_status(claims, "content_integrity")? == "pass" {
                return Err(semantic(
                    "content_integrity=pass requires a linked member content receipt",
                ));
            }
        }
        Value::Object(receipt) => {
            let receipt_package_digest = receipt
                .get("package_sha256")
                .and_then(Value::as_str)
                .ok_or_else(|| semantic("content receipt package_sha256 must be a string"))?;
            if receipt_package_digest != package_digest {
                return Err(semantic(
                    "content receipt package digest must equal package.package_sha256",
                ));
            }
            if !receipt.contains_key("receipt_sha256") {
                return Err(semantic("content receipt digest is missing"));
            }
        }
        _ => return Err(semantic("content_receipt must be an object or null")),
    }

    for required in REQUIRED_NON_CLAIMS {
        if !array_at(statement, &["not_established"])?
            .iter()
            .any(|value| value.as_str() == Some(required))
        {
            return Err(semantic(format!(
                "not_established must explicitly retain {required}"
            )));
        }
    }

    for claim_name in [
        "content_integrity",
        "signature_cryptographic",
        "signer_identity",
        "trust_policy",
        "application_authorization",
        "runtime_behavior",
    ] {
        let claim = object_at(claims, &[claim_name])?;
        if claim_status(claims, claim_name)? == "pass" {
            let evidence_refs = array_at(claim, &["evidence_refs"])?;
            if evidence_refs.is_empty() {
                return Err(semantic(format!(
                    "{claim_name}=pass requires at least one evidence reference"
                )));
            }
            if string_at(claim, &["evaluator"])?.trim().is_empty() {
                return Err(semantic(format!(
                    "{claim_name}=pass requires an explicit evaluator"
                )));
            }
        }
    }

    if claim_status(claims, "content_integrity")? == "pass" && content_receipt.is_null() {
        return Err(semantic(
            "content_integrity=pass requires a package-digest-linked content receipt",
        ));
    }

    if claim_status(claims, "signature_cryptographic")? == "pass" {
        let has_verified_scheme = schemes.iter().any(|scheme| {
            scheme.get("status").and_then(Value::as_str) == Some("pass")
        });
        if !has_verified_scheme {
            return Err(semantic(
                "signature_cryptographic=pass requires at least one scheme-specific pass",
            ));
        }
        if signers.is_empty() {
            return Err(semantic(
                "signature_cryptographic=pass requires at least one signer key/certificate record",
            ));
        }
        for scheme in schemes.iter().filter(|scheme| {
            scheme.get("status").and_then(Value::as_str) == Some("pass")
        }) {
            let scheme_name = string_at(scheme, &["scheme"])?;
            if !signers.iter().any(|signer| {
                signer.get("scheme").and_then(Value::as_str) == Some(scheme_name)
            }) {
                return Err(semantic(format!(
                    "passing scheme {scheme_name} requires a signer record for that same scheme"
                )));
            }
        }
        for signer in signers {
            let key = signer
                .get("public_key_spki_sha256")
                .and_then(Value::as_str)
                .ok_or_else(|| semantic("signer SPKI digest is missing"))?;
            let cert = signer
                .get("certificate_der_sha256")
                .and_then(Value::as_str)
                .ok_or_else(|| semantic("signer certificate digest is missing"))?;
            if !is_sha256(key) || !is_sha256(cert) {
                return Err(semantic(
                    "signer key and certificate byte identities require exact SHA-256 digests",
                ));
            }
        }
    }

    if claim_status(claims, "signer_identity")? == "pass" {
        if signers.is_empty() {
            return Err(semantic(
                "signer_identity=pass requires exact SPKI and certificate DER digest records",
            ));
        }
        for signer in signers {
            if !is_sha256(string_at(signer, &["public_key_spki_sha256"])?)
                || !is_sha256(string_at(signer, &["certificate_der_sha256"])?)
            {
                return Err(semantic(
                    "signer identity pass requires exact public-key and certificate byte digests",
                ));
            }
        }
    }

    for scheme in schemes {
        let name = string_at(scheme, &["scheme"])?;
        let status = string_at(scheme, &["status"])?;
        let coverage = string_at(scheme, &["coverage_scope"])?;
        if format == "apk" && name == "apk-v1" && status == "pass" && coverage == "whole_package" {
            return Err(semantic(
                "APK v1/JAR signing cannot claim whole-package ZIP metadata coverage",
            ));
        }
        if name == "apk-v4" && status == "pass" {
            if scheme.get("companion_sha256").is_none_or(Value::is_null) {
                return Err(semantic("APK v4 pass requires a bound .idsig digest"));
            }
            let supporting = array_at(scheme, &["supporting_schemes"])?;
            if !supporting.iter().any(|value| {
                matches!(value.as_str(), Some("apk-v2" | "apk-v3"))
            }) {
                return Err(semantic(
                    "APK v4 pass requires a complementary v2 or v3 scheme result",
                ));
            }
            let supporting_pass = schemes.iter().any(|other| {
                matches!(
                    other.get("scheme").and_then(Value::as_str),
                    Some("apk-v2" | "apk-v3")
                ) && other.get("status").and_then(Value::as_str) == Some("pass")
            });
            if !supporting_pass {
                return Err(semantic(
                    "APK v4 pass requires a matching v2/v3 scheme result with status pass",
                ));
            }
        }
        if name == "apple-code-signing" && status == "pass" {
            for field in ["code_directory_sha256", "cms_signature_sha256"] {
                if scheme.get(field).is_none_or(Value::is_null) {
                    return Err(semantic(format!(
                        "Apple code-signing pass requires {field}"
                    )));
                }
            }
        }
    }

    if claim_status(claims, "trust_policy")? == "pass" {
        if string_at(policy, &["evaluation_status"])? != "evaluated" {
            return Err(semantic(
                "trust_policy=pass requires policy_context.evaluation_status=evaluated",
            ));
        }
        for field in [
            "policy_id",
            "policy_version",
            "trust_anchor_set_sha256",
            "checked_at",
            "target_platform_version",
        ] {
            if policy.get(field).is_none_or(Value::is_null) {
                return Err(semantic(format!(
                    "trust_policy=pass requires policy_context.{field}"
                )));
            }
        }
        if !is_sha256(string_at(policy, &["trust_anchor_set_sha256"])?)
            || claim_status(claims, "signer_identity")? != "pass"
        {
            return Err(semantic(
                "trust_policy=pass requires a trust-anchor digest and verified signer identity",
            ));
        }
        if string_at(policy, &["revocation_mode"])? == "not_checked" {
            return Err(semantic(
                "trust_policy=pass cannot silently omit revocation treatment",
            ));
        }
    }

    if claim_status(claims, "application_authorization")? == "pass" {
        if claim_status(claims, "signature_cryptographic")? != "pass"
            || claim_status(claims, "signer_identity")? != "pass"
        {
            return Err(semantic(
                "application authorization pass requires cryptographic signature and signer identity passes",
            ));
        }
        if format == "apk" {
            require_equal_identifier(
                authorization,
                "expected_application_id",
                "observed_application_id",
            )?;
            require_equal_identifier(
                authorization,
                "expected_signer_spki_sha256",
                "observed_signer_spki_sha256",
            )?;
        } else {
            require_equal_identifier(authorization, "expected_bundle_id", "observed_bundle_id")?;
            require_equal_identifier(authorization, "expected_team_id", "observed_team_id")?;
            require_equal_identifier(
                authorization,
                "expected_signer_spki_sha256",
                "observed_signer_spki_sha256",
            )?;
            if authorization
                .get("provisioning_profile_sha256")
                .is_none_or(Value::is_null)
                || string_at(authorization, &["entitlement_evaluation"])? != "pass"
            {
                return Err(semantic(
                    "Apple authorization pass requires a profile digest and passing entitlement evaluation",
                ));
            }
        }
    }

    if claim_status(claims, "runtime_behavior")? == "pass" {
        for field in ["test_suite_id", "execution_evidence_sha256", "platform_build", "device_class"] {
            if runtime_context.get(field).is_none_or(Value::is_null) {
                return Err(semantic(format!(
                    "runtime_behavior=pass requires runtime_context.{field}"
                )));
            }
        }
        if !is_sha256(string_at(runtime_context, &["execution_evidence_sha256"])?)
        {
            return Err(semantic(
                "runtime behavior evidence must have an exact SHA-256",
            ));
        }
    }

    match string_at(attestation, &["status"])? {
        "not_provided" => {
            if !attestation
                .get("envelope_reference")
                .is_some_and(Value::is_null)
                || !attestation
                    .get("envelope_sha256")
                    .is_some_and(Value::is_null)
            {
                return Err(semantic(
                    "not_provided attestation must keep envelope reference and digest null",
                ));
            }
        }
        "separate_dsse_reference" => {
            if attestation
                .get("envelope_reference")
                .and_then(Value::as_str)
                .is_none_or(str::is_empty)
                || !is_sha256(string_at(attestation, &["envelope_sha256"])?)
            {
                return Err(semantic(
                    "DSSE attestation reference requires a URI and envelope SHA-256",
                ));
            }
        }
        _ => return Err(semantic("unknown attestation status")),
    }

    // The schema deliberately has no aggregate "trusted" or "safe" field. Reject accidental
    // reintroduction even if the schema is later relaxed.
    if statement.get("trusted").is_some()
        || statement.get("safe").is_some()
        || statement.get("valid_package").is_some()
    {
        return Err(semantic(
            "aggregate trusted/safe/valid_package booleans are prohibited",
        ));
    }

    Ok(())
}

fn require_equal_identifier(
    authorization: &serde_json::Map<String, Value>,
    expected_field: &str,
    observed_field: &str,
) -> Result<(), EvaluationError> {
    let expected = authorization
        .get(expected_field)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| semantic(format!("{expected_field} is required for authorization pass")))?;
    let observed = authorization
        .get(observed_field)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| semantic(format!("{observed_field} is required for authorization pass")))?;
    if expected != observed {
        return Err(semantic(format!(
            "{expected_field} and {observed_field} must match for authorization pass"
        )));
    }
    Ok(())
}

fn claim_status<'a>(claims: &'a Value, name: &str) -> Result<&'a str, EvaluationError> {
    string_at(claims, &[name, "status"])
}

fn object_at<'a>(
    value: &'a Value,
    path: &[&str],
) -> Result<&'a serde_json::Map<String, Value>, EvaluationError> {
    value_at(value, path)?
        .as_object()
        .ok_or_else(|| semantic(format!("{} must be an object", path.join("."))))
}

fn array_at<'a>(value: &'a Value, path: &[&str]) -> Result<&'a Vec<Value>, EvaluationError> {
    value_at(value, path)?
        .as_array()
        .ok_or_else(|| semantic(format!("{} must be an array", path.join("."))))
}

fn string_at<'a>(value: &'a Value, path: &[&str]) -> Result<&'a str, EvaluationError> {
    value_at(value, path)?
        .as_str()
        .ok_or_else(|| semantic(format!("{} must be a string", path.join("."))))
}

fn decimal_string_at(value: &Value, path: &[&str]) -> Result<u64, EvaluationError> {
    let text = string_at(value, path)?;
    if text.is_empty()
        || (text.len() > 1 && text.starts_with('0'))
        || !text.bytes().all(|byte| byte.is_ascii_digit())
    {
        return Err(semantic(format!(
            "{} must be a canonical unsigned decimal string",
            path.join(".")
        )));
    }
    text.parse::<u64>()
        .map_err(|_| semantic(format!("{} exceeds u64", path.join("."))))
}

fn value_at<'a>(value: &'a Value, path: &[&str]) -> Result<&'a Value, EvaluationError> {
    let mut current = value;
    for component in path {
        current = current
            .get(*component)
            .ok_or_else(|| semantic(format!("missing {}", path.join("."))))?;
    }
    Ok(current)
}

fn is_sha256(text: &str) -> bool {
    text.len() == 64
        && text
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn semantic(message: impl Into<String>) -> EvaluationError {
    EvaluationError::SemanticViolation(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    const EVALUATION_SCHEMA: &str =
        include_str!("../../../docs/research/package-signature-evaluation-v0.schema.json");

    fn schema() -> Value {
        serde_json::from_str(EVALUATION_SCHEMA).unwrap()
    }

    fn claim(status: &str) -> Value {
        json!({
            "status": status,
            "evidence_refs": [],
            "evaluator": null,
            "limitations": []
        })
    }

    fn valid_base() -> Value {
        json!({
            "schema_version": "0.1.0",
            "package": {
                "format": "apk",
                "package_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "package_size_bytes": "18446744073709551615",
                "application_identifier": "org.example.app"
            },
            "content_receipt": null,
            "platform": {
                "kind": "android",
                "version": "16",
                "build": "BP1A.250905.012",
                "architecture": "arm64-v8a",
                "api_level": 36,
                "device_class": "emulator"
            },
            "evaluator": {
                "implementation": "REA package evaluation harness",
                "version": "0.1.0",
                "source_commit": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                "toolchain": "rustc 1.96.0",
                "target": "x86_64-unknown-linux-gnu",
                "cargo_lock_sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
            },
            "policy_context": {
                "evaluation_status": "not_evaluated",
                "policy_id": null,
                "policy_version": null,
                "trust_anchor_set_sha256": null,
                "revocation_mode": "not_checked",
                "checked_at": null,
                "network_mode": "not_used",
                "target_platform_version": null
            },
            "claim_results": {
                "content_integrity": claim("not_evaluated"),
                "signature_cryptographic": claim("unknown"),
                "signer_identity": claim("unknown"),
                "trust_policy": claim("not_evaluated"),
                "application_authorization": claim("not_evaluated"),
                "runtime_behavior": claim("not_evaluated")
            },
            "signers": [],
            "scheme_results": [],
            "authorization": {
                "expected_application_id": null,
                "observed_application_id": null,
                "expected_bundle_id": null,
                "observed_bundle_id": null,
                "expected_team_id": null,
                "observed_team_id": null,
                "expected_signer_spki_sha256": null,
                "observed_signer_spki_sha256": null,
                "provisioning_profile_sha256": null,
                "entitlement_evaluation": "not_evaluated",
                "device_distribution_policy": "not_evaluated"
            },
            "runtime_context": {
                "test_suite_id": null,
                "execution_evidence_sha256": null,
                "platform_build": null,
                "device_class": null
            },
            "attestation": {
                "status": "not_provided",
                "envelope_reference": null,
                "envelope_sha256": null
            },
            "not_established": [
                "publisher_benevolence",
                "human_or_organization_identity_attribution",
                "absence_of_malicious_code",
                "safe_runtime_behavior",
                "user_intent_alignment",
                "uninspected_component_semantics",
                "future_update_safety",
                "complete_platform_policy_compliance"
            ]
        })
    }

    fn pass_claim() -> Value {
        json!({
            "status": "pass",
            "evidence_refs": ["sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"],
            "evaluator": "test-oracle",
            "limitations": []
        })
    }

    fn add_android_signer_and_v2(statement: &mut Value) {
        statement["signers"] = json!([{
            "scheme": "apk-v2",
            "public_key_spki_sha256": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
            "certificate_der_sha256": "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
            "signature_algorithm": "rsa-pss-sha256",
            "digest_algorithm": "sha256",
            "subject_display": "CN=Example",
            "issuer_display": "CN=Example CA",
            "rotation_lineage_sha256": null
        }]);
        statement["scheme_results"] = json!([{
            "scheme":"apk-v2",
            "status":"pass",
            "coverage_scope":"protected_package_sections",
            "content_digest_sha256":"1111111111111111111111111111111111111111111111111111111111111111",
            "companion_sha256":null,
            "supporting_schemes":[],
            "min_sdk":24,
            "max_sdk":null,
            "code_directory_sha256":null,
            "cms_signature_sha256":null,
            "resource_envelope_sha256":null,
            "limitations":[]
        }]);
        statement["claim_results"]["signature_cryptographic"] = pass_claim();
        statement["claim_results"]["signer_identity"] = pass_claim();
    }

    #[test]
    fn schema_is_valid_and_minimal_unknown_evaluation_passes() {
        let schema = schema();
        validate_schema_document(&schema).unwrap();
        validate_evaluation(&schema, &valid_base()).unwrap();
    }

    #[test]
    fn exact_u64_package_size_is_decimal_string() {
        let schema = schema();
        let statement = valid_base();
        let canonical = canonicalize_evaluation(&schema, &statement).unwrap();
        let text = std::str::from_utf8(&canonical).unwrap();
        assert!(text.contains(r#""package_size_bytes":"18446744073709551615""#));
    }

    #[test]
    fn content_integrity_requires_package_digest_linked_receipt() {
        let schema = schema();
        let mut statement = valid_base();
        statement["claim_results"]["content_integrity"] = pass_claim();
        assert!(validate_evaluation(&schema, &statement).is_err());

        statement["content_receipt"] = json!({
            "receipt_sha256":"2222222222222222222222222222222222222222222222222222222222222222",
            "package_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "receipt_schema_version":"0.1.0"
        });
        validate_evaluation(&schema, &statement).unwrap();

        statement["content_receipt"]["package_sha256"] =
            json!("3333333333333333333333333333333333333333333333333333333333333333");
        assert!(validate_evaluation(&schema, &statement).is_err());
    }

    #[test]
    fn cryptographic_pass_requires_scheme_and_exact_signer_digests() {
        let schema = schema();
        let mut statement = valid_base();
        statement["claim_results"]["signature_cryptographic"] = pass_claim();
        assert!(validate_evaluation(&schema, &statement).is_err());

        add_android_signer_and_v2(&mut statement);
        validate_evaluation(&schema, &statement).unwrap();

        statement["signers"][0]["certificate_der_sha256"] = json!("not-a-digest");
        assert!(validate_evaluation(&schema, &statement).is_err());
    }

    #[test]
    fn valid_signature_by_unexpected_signer_is_not_trust_or_authorization() {
        let schema = schema();
        let mut statement = valid_base();
        add_android_signer_and_v2(&mut statement);
        statement["claim_results"]["trust_policy"] = json!({
            "status":"fail","evidence_refs":["trust-policy-result"],"evaluator":"android-policy","limitations":[]
        });
        statement["claim_results"]["application_authorization"] = json!({
            "status":"fail","evidence_refs":["expected-signer-mismatch"],"evaluator":"application-policy","limitations":[]
        });
        validate_evaluation(&schema, &statement).unwrap();
    }

    #[test]
    fn trust_pass_requires_named_policy_anchor_time_platform_and_revocation() {
        let schema = schema();
        let mut statement = valid_base();
        add_android_signer_and_v2(&mut statement);
        statement["policy_context"] = json!({
            "evaluation_status":"evaluated",
            "policy_id":"org.example.android-production",
            "policy_version":"2026-11",
            "trust_anchor_set_sha256":"4444444444444444444444444444444444444444444444444444444444444444",
            "revocation_mode":"platform_managed",
            "checked_at":"2026-11-05T00:00:00Z",
            "network_mode":"offline",
            "target_platform_version":"Android 16 / API 36"
        });
        statement["claim_results"]["trust_policy"] = pass_claim();
        validate_evaluation(&schema, &statement).unwrap();

        statement["policy_context"]["trust_anchor_set_sha256"] = Value::Null;
        assert!(validate_evaluation(&schema, &statement).is_err());

        statement["policy_context"]["trust_anchor_set_sha256"] =
            json!("4444444444444444444444444444444444444444444444444444444444444444");
        statement["policy_context"]["revocation_mode"] = json!("not_checked");
        assert!(validate_evaluation(&schema, &statement).is_err());
    }

    #[test]
    fn trust_pass_cannot_be_inferred_without_signer_identity() {
        let schema = schema();
        let mut statement = valid_base();
        statement["policy_context"] = json!({
            "evaluation_status":"evaluated",
            "policy_id":"policy",
            "policy_version":"1",
            "trust_anchor_set_sha256":"4444444444444444444444444444444444444444444444444444444444444444",
            "revocation_mode":"platform_managed",
            "checked_at":"2026-11-05T00:00:00Z",
            "network_mode":"offline",
            "target_platform_version":"Android 16"
        });
        statement["claim_results"]["trust_policy"] = pass_claim();
        assert!(validate_evaluation(&schema, &statement).is_err());
    }

    #[test]
    fn android_v4_requires_idsig_and_passing_v2_or_v3() {
        let schema = schema();
        let mut statement = valid_base();
        add_android_signer_and_v2(&mut statement);
        statement["scheme_results"][0]["scheme"] = json!("apk-v4");
        statement["scheme_results"][0]["coverage_scope"] = json!("merkle_tree");
        assert!(validate_evaluation(&schema, &statement).is_err());

        statement["scheme_results"][0]["companion_sha256"] =
            json!("5555555555555555555555555555555555555555555555555555555555555555");
        statement["scheme_results"][0]["supporting_schemes"] = json!(["apk-v2"]);
        assert!(validate_evaluation(&schema, &statement).is_err());

        statement["scheme_results"].push(json!({
            "scheme":"apk-v2","status":"pass","coverage_scope":"protected_package_sections",
            "content_digest_sha256":"1111111111111111111111111111111111111111111111111111111111111111",
            "companion_sha256":null,"supporting_schemes":[],"min_sdk":24,"max_sdk":null,
            "code_directory_sha256":null,"cms_signature_sha256":null,"resource_envelope_sha256":null,"limitations":[]
        }));
        validate_evaluation(&schema, &statement).unwrap();
    }

    #[test]
    fn apk_v1_cannot_claim_whole_package_coverage() {
        let schema = schema();
        let mut statement = valid_base();
        add_android_signer_and_v2(&mut statement);
        statement["scheme_results"][0]["scheme"] = json!("apk-v1");
        statement["scheme_results"][0]["coverage_scope"] = json!("whole_package");
        assert!(validate_evaluation(&schema, &statement).is_err());
    }

    #[test]
    fn apple_authorization_requires_matching_bundle_team_profile_and_entitlements() {
        let schema = schema();
        let mut statement = valid_base();
        statement["package"]["format"] = json!("ipa");
        statement["platform"]["kind"] = json!("apple");
        statement["signers"] = json!([{
            "scheme":"apple-code-signing",
            "public_key_spki_sha256":"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
            "certificate_der_sha256":"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
            "signature_algorithm":"ecdsa-p256-sha256","digest_algorithm":"sha256",
            "subject_display":"Apple Development: Example","issuer_display":"Apple WWDR","rotation_lineage_sha256":null
        }]);
        statement["scheme_results"] = json!([{
            "scheme":"apple-code-signing","status":"pass","coverage_scope":"code_directory_pages",
            "content_digest_sha256":null,"companion_sha256":null,"supporting_schemes":[],
            "min_sdk":null,"max_sdk":null,
            "code_directory_sha256":"6666666666666666666666666666666666666666666666666666666666666666",
            "cms_signature_sha256":"7777777777777777777777777777777777777777777777777777777777777777",
            "resource_envelope_sha256":"8888888888888888888888888888888888888888888888888888888888888888",
            "limitations":[]
        }]);
        statement["claim_results"]["signature_cryptographic"] = pass_claim();
        statement["claim_results"]["signer_identity"] = pass_claim();
        statement["claim_results"]["application_authorization"] = pass_claim();
        statement["authorization"] = json!({
            "expected_application_id":null,"observed_application_id":null,
            "expected_bundle_id":"org.example.app","observed_bundle_id":"org.example.app",
            "expected_team_id":"AB12CDE345","observed_team_id":"AB12CDE345",
            "expected_signer_spki_sha256":"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
            "observed_signer_spki_sha256":"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
            "provisioning_profile_sha256":"9999999999999999999999999999999999999999999999999999999999999999",
            "entitlement_evaluation":"pass","device_distribution_policy":"development-device-allowlist"
        });
        validate_evaluation(&schema, &statement).unwrap();

        statement["authorization"]["observed_team_id"] = json!("WRONGTEAM");
        assert!(validate_evaluation(&schema, &statement).is_err());
    }

    #[test]
    fn runtime_pass_requires_separate_platform_execution_evidence() {
        let schema = schema();
        let mut statement = valid_base();
        statement["claim_results"]["runtime_behavior"] = pass_claim();
        assert!(validate_evaluation(&schema, &statement).is_err());

        statement["runtime_context"] = json!({
            "test_suite_id":"mobile-runtime-regression-v1",
            "execution_evidence_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "platform_build":"Android 16 build BP1A",
            "device_class":"Pixel emulator"
        });
        validate_evaluation(&schema, &statement).unwrap();
    }

    #[test]
    fn aggregate_trusted_and_safe_booleans_are_rejected() {
        let schema = schema();
        let mut statement = valid_base();
        statement["trusted"] = json!(true);
        assert!(validate_evaluation(&schema, &statement).is_err());
        statement.as_object_mut().unwrap().remove("trusted");
        statement["safe"] = json!(true);
        assert!(validate_evaluation(&schema, &statement).is_err());
    }

    #[test]
    fn dsse_attestation_remains_separate_and_does_not_upgrade_claims() {
        let schema = schema();
        let mut statement = valid_base();
        statement["attestation"] = json!({
            "status":"separate_dsse_reference",
            "envelope_reference":"https://example.invalid/attestations/sha256/abcd",
            "envelope_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        });
        validate_evaluation(&schema, &statement).unwrap();
        assert_eq!(
            statement["claim_results"]["signature_cryptographic"]["status"],
            "unknown"
        );
    }

    #[test]
    fn raw_parser_rejects_duplicate_properties() {
        assert!(parse_unique_json(br#"{"status":"pass","status":"fail"}"#).is_err());
        assert!(parse_unique_json(br#"{"nested":{"id":"a","id":"b"}}"#).is_err());
        assert!(parse_unique_json(br#"{"status":"pass","evidence":[]}"#).is_ok());
    }

    #[test]
    fn required_non_claims_cannot_be_removed() {
        let schema = schema();
        let mut statement = valid_base();
        statement["not_established"]
            .as_array_mut()
            .unwrap()
            .retain(|value| value.as_str() != Some("publisher_benevolence"));
        assert!(validate_evaluation(&schema, &statement).is_err());
    }

    #[test]
    fn attestation_reference_requires_digest() {
        let schema = schema();
        let mut statement = valid_base();
        statement["attestation"] = json!({
            "status":"separate_dsse_reference",
            "envelope_reference":"https://example.invalid/attestation",
            "envelope_sha256":null
        });
        assert!(validate_evaluation(&schema, &statement).is_err());
    }

    #[test]
    fn evaluation_digest_is_deterministic() {
        let schema = schema();
        let statement = valid_base();
        assert_eq!(
            evaluation_sha256(&schema, &statement).unwrap(),
            evaluation_sha256(&schema, &statement).unwrap()
        );
    }
}
