use serde::de::{self, Error as _, MapAccess, SeqAccess, Visitor};
use serde::ser::{SerializeMap, SerializeSeq};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::env;
use std::fmt;
use std::fs;
use std::path::Path;

const MAX_PLAN_BYTES: usize = 1_048_576;
const MAX_SAFE_INTEGER: i64 = 9_007_199_254_740_991;
const DOMAIN_SEPARATOR: &[u8] = b"LUMINOUS/CLAIM-PLAN-IR/V1\0";

const SCHEMA: &str = include_str!("../../CLAIM_PLAN_IR_V1.schema.json");

#[derive(Debug, Clone)]
enum StrictJson {
    Null,
    Bool(bool),
    Integer(i64),
    String(String),
    Array(Vec<StrictJson>),
    Object(BTreeMap<String, StrictJson>),
}

// A purpose-built input representation. The map visitor rejects duplicate
// decoded keys before any ordinary map/value parser can discard one.
impl<'de> Deserialize<'de> for StrictJson {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(StrictJsonVisitor)
    }
}

struct StrictJsonVisitor;

impl<'de> Visitor<'de> for StrictJsonVisitor {
    type Value = StrictJson;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("strict JSON with unique object member names and exact safe integers")
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(StrictJson::Null)
    }

    fn visit_none<E>(self) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(StrictJson::Null)
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(StrictJson::Bool(value))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        if !(-MAX_SAFE_INTEGER..=MAX_SAFE_INTEGER).contains(&value) {
            return Err(E::custom("integer is outside the JCS exact safe-integer range"));
        }
        Ok(StrictJson::Integer(value))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        if value > MAX_SAFE_INTEGER as u64 {
            return Err(E::custom("integer is outside the JCS exact safe-integer range"));
        }
        Ok(StrictJson::Integer(value as i64))
    }

    fn visit_f64<E>(self, _value: f64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Err(E::custom(
            "floating-point JSON values are not admitted by Claim Plan IR v1; use a typed decimal string",
        ))
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(StrictJson::String(value.to_owned()))
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(StrictJson::String(value))
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(value) = seq.next_element::<StrictJson>()? {
            values.push(value);
        }
        Ok(StrictJson::Array(values))
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut values = BTreeMap::new();
        while let Some(key) = map.next_key::<String>()? {
            if values.contains_key(&key) {
                return Err(A::Error::custom(format!(
                    "duplicate decoded JSON member name: {key}"
                )));
            }
            let value = map.next_value::<StrictJson>()?;
            values.insert(key, value);
        }
        Ok(StrictJson::Object(values))
    }
}

// Serialize only after strict admission. JCS performs the final property sort
// and ECMAScript-compatible primitive serialization.
impl Serialize for StrictJson {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            StrictJson::Null => serializer.serialize_unit(),
            StrictJson::Bool(value) => serializer.serialize_bool(*value),
            StrictJson::Integer(value) => serializer.serialize_i64(*value),
            StrictJson::String(value) => serializer.serialize_str(value),
            StrictJson::Array(values) => {
                let mut seq = serializer.serialize_seq(Some(values.len()))?;
                for value in values {
                    seq.serialize_element(value)?;
                }
                seq.end()
            }
            StrictJson::Object(values) => {
                let mut map = serializer.serialize_map(Some(values.len()))?;
                for (key, value) in values {
                    map.serialize_entry(key, value)?;
                }
                map.end()
            }
        }
    }
}

fn parse_strict(raw: &str) -> Result<StrictJson, String> {
    if raw.len() > MAX_PLAN_BYTES {
        return Err(format!(
            "claim plan exceeds maximum input size of {MAX_PLAN_BYTES} bytes"
        ));
    }
    serde_json::from_str::<StrictJson>(raw).map_err(|error| format!("strict JSON admission: {error}"))
}

fn str_at<'a>(value: &'a Value, key: &str, what: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{what}.{key} must be a string"))
}

fn field_type<'a>(field: &'a Value) -> Result<&'a str, String> {
    field
        .pointer("/interpretation/value_type")
        .and_then(Value::as_str)
        .ok_or_else(|| "field interpretation.value_type is missing".to_owned())
}

fn validate_literal_for_field(field: &Value, literal: &Value) -> Result<(), String> {
    let id = str_at(field, "field_id", "field")?;
    match field_type(field)? {
        "unsigned_integer" => {
            let number = literal
                .as_u64()
                .ok_or_else(|| format!("literal for unsigned field {id} must be an integer"))?;
            if number > MAX_SAFE_INTEGER as u64 {
                return Err(format!("literal for field {id} exceeds the exact safe-integer range"));
            }
        }
        "signed_integer" | "timestamp_unix_seconds" => {
            let number = literal
                .as_i64()
                .ok_or_else(|| format!("literal for signed field {id} must be an integer"))?;
            if !(-MAX_SAFE_INTEGER..=MAX_SAFE_INTEGER).contains(&number) {
                return Err(format!("literal for field {id} exceeds the exact safe-integer range"));
            }
        }
        "utf8_string" | "decimal_string" | "timestamp_rfc3339" | "closed_enum" => {
            if !literal.is_string() {
                return Err(format!("literal for field {id} must be a string"));
            }
        }
        unsupported => return Err(format!("unsupported v1 interpretation type: {unsupported}")),
    }
    Ok(())
}

fn validate_predicate(
    expression: &Value,
    fields: &HashMap<String, Value>,
    referenced: &mut HashSet<String>,
) -> Result<(), String> {
    let kind = str_at(expression, "kind", "predicate expression")?;
    match kind {
        "all" | "any" => {
            let children = expression
                .get("children")
                .and_then(Value::as_array)
                .ok_or_else(|| format!("{kind} expression requires children"))?;
            if children.is_empty() {
                return Err(format!("{kind} expression must not be empty"));
            }
            for child in children {
                validate_predicate(child, fields, referenced)?;
            }
        }
        "not" => {
            let child = expression
                .get("child")
                .ok_or_else(|| "not expression requires child".to_owned())?;
            validate_predicate(child, fields, referenced)?;
        }
        "compare" => {
            let left = expression
                .get("left")
                .ok_or_else(|| "compare expression requires left operand".to_owned())?;
            if str_at(left, "kind", "left operand")? != "claim_value" {
                return Err("v1 compare left operand must be a typed claim_value".to_owned());
            }
            let id = str_at(left, "field_id", "claim_value")?;
            let field = fields
                .get(id)
                .ok_or_else(|| format!("predicate references unknown field_id: {id}"))?;
            referenced.insert(id.to_owned());

            let right = expression
                .get("right")
                .ok_or_else(|| "compare expression requires right operand".to_owned())?;
            if str_at(right, "kind", "right operand")? != "literal" {
                return Err("v1 compare right operand must be a literal".to_owned());
            }
            let literal = right
                .get("value")
                .ok_or_else(|| "literal expression requires value".to_owned())?;
            validate_literal_for_field(field, literal)?;

            let operator = str_at(expression, "operator", "compare expression")?;
            match field_type(field)? {
                "utf8_string" | "decimal_string" | "timestamp_rfc3339" | "closed_enum"
                    if !matches!(operator, "eq" | "neq") =>
                {
                    return Err(format!(
                        "operator {operator} is not admitted for string-like field {id}"
                    ));
                }
                _ => {}
            }
        }
        "freshness" => {
            let id = str_at(expression, "field_id", "freshness expression")?;
            let field = fields
                .get(id)
                .ok_or_else(|| format!("freshness expression references unknown field_id: {id}"))?;
            referenced.insert(id.to_owned());
            if field_type(field)? != "timestamp_rfc3339" {
                return Err(format!(
                    "freshness expression field {id} must use timestamp_rfc3339 interpretation"
                ));
            }
            if str_at(expression, "reference", "freshness expression")? != "request_issued_at" {
                return Err("v1 freshness reference must be request_issued_at".to_owned());
            }
        }
        unsupported => return Err(format!("unsupported predicate expression kind: {unsupported}")),
    }
    Ok(())
}

fn is_valid_json_pointer(pointer: &str) -> bool {
    if pointer.is_empty() {
        return true;
    }
    if !pointer.starts_with('/') {
        return false;
    }

    let bytes = pointer.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'~' {
            if index + 1 >= bytes.len() || !matches!(bytes[index + 1], b'0' | b'1') {
                return false;
            }
            index += 2;
        } else {
            index += 1;
        }
    }
    true
}

fn canonical_https_dns_host(origin: &str) -> Result<&str, String> {
    let authority = origin
        .strip_prefix("https://")
        .ok_or_else(|| "source origin must use HTTPS".to_owned())?;
    if authority.is_empty()
        || authority.contains('/')
        || authority.contains('?')
        || authority.contains('#')
        || authority.contains('@')
        || authority.contains('\\')
        || authority.chars().any(char::is_whitespace)
        || authority.contains('[')
        || authority.contains(']')
    {
        return Err("source.request.origin must be a canonical HTTPS DNS origin".to_owned());
    }

    let (host, port) = match authority.split_once(':') {
        Some((host, port)) => {
            if port.is_empty()
                || !port.bytes().all(|byte| byte.is_ascii_digit())
                || port.starts_with('0')
            {
                return Err("HTTPS origin port must be canonical decimal without leading zeroes".to_owned());
            }
            let parsed = port
                .parse::<u16>()
                .map_err(|_| "HTTPS origin port must be in the range 1..=65535".to_owned())?;
            if parsed == 443 {
                return Err("canonical HTTPS origin must omit the default port 443".to_owned());
            }
            (host, Some(parsed))
        }
        None => (authority, None),
    };
    let _ = port;

    if host.is_empty() || host.len() > 253 || !host.is_ascii() {
        return Err("HTTPS origin host must be an ASCII DNS A-label of at most 253 bytes".to_owned());
    }
    if host.bytes().any(|byte| byte.is_ascii_uppercase()) || host.ends_with('.') {
        return Err("HTTPS origin host must be lowercase and omit the trailing DNS dot".to_owned());
    }
    for label in host.split('.') {
        if label.is_empty() || label.len() > 63 {
            return Err("HTTPS origin contains an empty or overlong DNS label".to_owned());
        }
        let bytes = label.as_bytes();
        if !bytes[0].is_ascii_lowercase() && !bytes[0].is_ascii_digit() {
            return Err("DNS labels must begin with a lowercase alphanumeric character".to_owned());
        }
        if !bytes[bytes.len() - 1].is_ascii_lowercase()
            && !bytes[bytes.len() - 1].is_ascii_digit()
        {
            return Err("DNS labels must end with a lowercase alphanumeric character".to_owned());
        }
        if !bytes
            .iter()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || *byte == b'-')
        {
            return Err("DNS labels may contain only lowercase letters, digits and hyphens".to_owned());
        }
    }
    Ok(host)
}

fn validate_semantics(value: &Value) -> Result<usize, String> {
    let fields = value
        .get("fields")
        .and_then(Value::as_array)
        .ok_or_else(|| "fields must be an array".to_owned())?;
    let mut by_id = HashMap::new();
    let mut pointers = BTreeSet::new();
    let mut common_parser: Option<(String, String)> = None;
    let mut common_schema: Option<(String, String)> = None;

    for field in fields {
        let id = str_at(field, "field_id", "field")?.to_owned();
        if by_id.insert(id.clone(), field.clone()).is_some() {
            return Err(format!("duplicate field_id: {id}"));
        }
        let pointer = str_at(field, "json_pointer", "field")?;
        if !is_valid_json_pointer(pointer) {
            return Err(format!("field {id} has an invalid RFC6901 JSON Pointer escape or prefix"));
        }
        if !pointers.insert(pointer.to_owned()) {
            return Err(format!("duplicate JSON Pointer across field bindings: {pointer}"));
        }

        let parser = (
            str_at(field, "parser_profile_id", "field")?.to_owned(),
            str_at(field, "parser_profile_digest", "field")?.to_owned(),
        );
        let schema = (
            str_at(field, "schema_profile_id", "field")?.to_owned(),
            str_at(field, "schema_profile_digest", "field")?.to_owned(),
        );
        if common_parser.as_ref().is_some_and(|existing| existing != &parser) {
            return Err("all fields from one response must share one parser profile".to_owned());
        }
        if common_schema.as_ref().is_some_and(|existing| existing != &schema) {
            return Err("all fields from one response must share one schema profile".to_owned());
        }
        common_parser = Some(parser);
        common_schema = Some(schema);
    }

    let mut referenced = HashSet::new();
    let ast = value
        .pointer("/predicate/ast")
        .ok_or_else(|| "predicate.ast is missing".to_owned())?;
    validate_predicate(ast, &by_id, &mut referenced)?;
    for id in by_id.keys() {
        if !referenced.contains(id) {
            return Err(format!("field {id} is selected but unused by the predicate"));
        }
    }

    let status = value
        .pointer("/privacy/status_lookup")
        .ok_or_else(|| "privacy.status_lookup is missing".to_owned())?;
    let spec = str_at(status, "specification", "status_lookup")?;
    let maturity = str_at(status, "maturity", "status_lookup")?;
    match (spec, maturity) {
        ("bitstring-status-list-v1.0", "w3c-recommendation")
        | ("bitstring-status-list-v1.1-wd-2026-11-11", "w3c-working-draft")
        | ("other-qualified", "other-profile") => {}
        _ => return Err("status-list specification and maturity do not match".to_owned()),
    }

    let origin = str_at(
        value.pointer("/source/request").ok_or("source.request missing")?,
        "origin",
        "source.request",
    )?;
    let host = canonical_https_dns_host(origin)?;
    let path_template = str_at(
        value.pointer("/source/request").ok_or("source.request missing")?,
        "path_template",
        "source.request",
    )?;
    if !path_template.starts_with('/')
        || path_template.starts_with("//")
        || path_template.contains('?')
        || path_template.contains('#')
        || path_template.contains('\\')
        || path_template.chars().any(char::is_control)
    {
        return Err("source.request.path_template must be an origin-relative path without query, fragment or backslash".to_owned());
    }
    if value.pointer("/source/server/identity_kind").and_then(Value::as_str) == Some("dns_name") {
        let server_name = str_at(
            value.pointer("/source/server").ok_or("source.server missing")?,
            "identity_value",
            "source.server",
        )?;
        if server_name != host {
            return Err("canonical DNS server identity must exactly match the HTTPS origin host".to_owned());
        }
    }

    let request = value
        .pointer("/request")
        .ok_or_else(|| "request envelope is missing".to_owned())?;
    let issued_at = str_at(request, "issued_at", "request")?;
    let expires_at = str_at(request, "expires_at", "request")?;
    // The schema restricts both values to fixed-width UTC seconds. Lexical
    // order is therefore chronological order after the RFC3339 format check.
    if expires_at <= issued_at {
        return Err("request expires_at must be later than issued_at".to_owned());
    }

    let mode = str_at(
        value.pointer("/source").ok_or("source missing")?,
        "protocol_mode",
        "source",
    )?;
    let trust = value
        .get("trust")
        .ok_or_else(|| "trust profile is missing".to_owned())?;
    let witness_kind = str_at(trust, "session_witness_kind", "trust")?;
    let verification_scope = str_at(trust, "verification_scope", "trust")?;
    let artifacts = trust
        .get("evidence_artifact_kinds")
        .and_then(Value::as_array)
        .ok_or_else(|| "trust.evidence_artifact_kinds must be an array".to_owned())?
        .iter()
        .filter_map(Value::as_str)
        .collect::<HashSet<_>>();

    let (expected_witness, expected_scope, required_artifact): (&str, &str, Option<&str>) =
        match mode {
            "tlsnotary_mpc_direct" | "deco_three_party" | "dido_plus_three_party"
            | "zkpass_mpc" => ("direct_verifier", "same_session_participant", None),
            "tlsnotary_proxy" | "zkpass_proxy" => {
                ("proxy_verifier", "same_session_participant", None)
            }
            "tlsnotary_delegated_notary" => (
                "delegated_notary",
                "delegated_attestation_verifier",
                Some("signed_transcript_attestation"),
            ),
            "reclaim_split_tee_mpc" => (
                "tee_attestor",
                "tcb_attestation_verifier",
                Some("tee_quote"),
            ),
            "scribe_attestor_stark" => (
                "protocol_attestor",
                "delegated_attestation_verifier",
                Some("signed_value_commitment"),
            ),
            unsupported => return Err(format!("unsupported v1 source mode: {unsupported}")),
        };
    if witness_kind != expected_witness || verification_scope != expected_scope {
        return Err(format!(
            "trust profile overstates or substitutes source-mode witness/scope: {mode} requires {expected_witness} / {expected_scope}"
        ));
    }
    if let Some(required) = required_artifact {
        if !artifacts.contains(required) {
            return Err(format!(
                "trust profile for {mode} requires evidence artifact {required}"
            ));
        }
    }
    if mode == "scribe_attestor_stark" && !artifacts.contains("transparent_zk_proof") {
        return Err("Scribe trust profile also requires transparent_zk_proof".to_owned());
    }

    let strategy = str_at(
        value.pointer("/source/commitment").ok_or("source.commitment missing")?,
        "strategy",
        "source.commitment",
    )?;
    if mode == "scribe_attestor_stark" && strategy != "attestor_value_commitment" {
        return Err("Scribe profile requires attestor_value_commitment strategy".to_owned());
    }
    if mode.starts_with("tlsnotary_") && strategy == "attestor_value_commitment" {
        return Err("TLSNotary profile cannot be substituted with an attestor-value commitment".to_owned());
    }

    Ok(fields.len())
}

fn validate_plan(raw: &str) -> Result<(String, usize), String> {
    let strict = parse_strict(raw)?;
    let instance = serde_json::to_value(&strict)
        .map_err(|error| format!("strict JSON conversion: {error}"))?;
    let schema: Value = serde_json::from_str(SCHEMA)
        .map_err(|error| format!("embedded JSON Schema parse: {error}"))?;
    let validator = jsonschema::draft202012::options()
        .should_validate_formats(true)
        .build(&schema)
        .map_err(|error| format!("JSON Schema compilation: {error}"))?;
    validator
        .validate(&instance)
        .map_err(|error| format!("JSON Schema validation: {error}"))?;

    let field_count = validate_semantics(&instance)?;
    let canonical = serde_json_canonicalizer::to_vec(&strict)
        .map_err(|error| format!("RFC 8785 JCS canonicalization: {error}"))?;
    let mut digest = Sha256::new();
    digest.update(DOMAIN_SEPARATOR);
    digest.update(canonical);
    let plan_digest = hex::encode(digest.finalize());
    Ok((plan_digest, field_count))
}

fn run() -> Result<(), String> {
    let path = env::args_os()
        .nth(1)
        .ok_or_else(|| "usage: claim-plan-ir-preflight <claim-plan.json>".to_owned())?;
    let raw = fs::read_to_string(Path::new(&path))
        .map_err(|error| format!("cannot read plan: {error}"))?;
    let (digest, fields) = validate_plan(&raw)?;
    println!(
        "{}",
        serde_json::json!({
            "status": "structural_and_semantic_preflight_pass",
            "plan_digest": digest,
            "field_count": fields,
            "cryptographic_source_status": "not_verified",
            "proof_status": "not_verified",
            "issuer_status": "not_verified",
            "currentness_status": "not_qualified",
            "authority_status": "not_evaluated"
        })
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("claim-plan preflight rejected: {error}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const POSITIVE: &str = include_str!("../../fixtures/positive_zar_balance_plan_v1.json");
    const UNSAFE: &str = include_str!("../../fixtures/negative/unsafe_integer_literal.json");
    const MATURITY: &str = include_str!("../../fixtures/negative/status_version_maturity_mismatch.json");
    const ORIGIN: &str = include_str!("../../fixtures/negative/origin_contains_path.json");
    const ANTI_ROLLBACK: &str =
        include_str!("../../fixtures/negative/missing_status_anti_rollback.json");
    const DUPLICATE: &str =
        include_str!("../../fixtures/negative/duplicate_member_name.raw.json");
    const BAD_POINTER: &str =
        include_str!("../../fixtures/negative/invalid_json_pointer_escape.json");
    const EXPIRED_REQUEST: &str =
        include_str!("../../fixtures/negative/expiry_not_after_issued_at.json");
    const TRUST_SCOPE: &str =
        include_str!("../../fixtures/negative/trust_scope_overclaim.json");
    const BAD_PORT: &str =
        include_str!("../../fixtures/negative/https_origin_port_out_of_range.json");
    const DEFAULT_PORT: &str =
        include_str!("../../fixtures/negative/https_origin_explicit_default_port.json");
    const SERVER_MISMATCH: &str =
        include_str!("../../fixtures/negative/dns_server_identity_mismatch.json");

    #[test]
    fn jcs_uses_utf16_code_unit_order_for_object_keys() {
        // UTF-16 sorts U+1F600 (D83D DE00) before U+E000; UTF-8 byte
        // ordering and Rust's scalar-value ordering produce the opposite.
        let raw = r#"{"\uE000":"bmp","\uD83D\uDE00":"astral"}"#;
        let strict = parse_strict(raw).expect("valid JCS input");
        let canonical =
            serde_json_canonicalizer::to_string(&strict).expect("canonicalize with JCS");
        assert_eq!(canonical, r#"{"😀":"astral","":"bmp"}"#);
    }

    #[test]
    fn positive_three_field_plan_passes_preflight() {
        let (digest, count) = validate_plan(POSITIVE).expect("positive fixture should pass");
        assert_eq!(count, 3);
        assert_eq!(digest.len(), 64);
    }

    #[test]
    fn canonical_https_origin_omits_default_port_443() {
        let error = validate_plan(DEFAULT_PORT)
            .expect_err("explicit default port 443 must reject");
        assert!(error.contains("omit the default port 443"));
    }

    #[test]
    fn dns_server_identity_must_match_origin_host() {
        let error = validate_plan(SERVER_MISMATCH)
            .expect_err("DNS server identity mismatch must reject");
        assert!(error.contains("must exactly match the HTTPS origin host"));
    }

    #[test]
    fn https_origin_port_must_be_in_range() {
        let error = validate_plan(BAD_PORT)
            .expect_err("out-of-range HTTPS port must reject");
        assert!(error.contains("port must be in the range"));
    }

    #[test]
    fn delegated_notary_cannot_claim_direct_or_public_verification_scope() {
        let error = validate_plan(TRUST_SCOPE)
            .expect_err("delegated attestation cannot claim direct session scope");
        assert!(error.contains("trust profile overstates or substitutes"));
    }

    #[test]
    fn request_expiry_must_follow_issue_time() {
        let error = validate_plan(EXPIRED_REQUEST)
            .expect_err("expiry at/before issue time must reject");
        assert!(error.contains("expires_at must be later than issued_at"));
    }

    #[test]
    fn invalid_json_pointer_escape_rejects() {
        let error = validate_plan(BAD_POINTER)
            .expect_err("invalid pointer escape fixture must reject");
        assert!(error.contains("invalid RFC6901 JSON Pointer"));
    }

    #[test]
    fn duplicate_decoded_member_names_reject_before_schema() {
        let error = parse_strict(DUPLICATE).expect_err("duplicate member names must reject");
        assert!(error.contains("duplicate decoded JSON member name"));
    }

    #[test]
    fn non_safe_integer_rejects_before_canonicalization() {
        let error = validate_plan(UNSAFE).expect_err("unsafe integer must reject");
        assert!(error.contains("exact safe-integer range"));
    }

    #[test]
    fn status_specification_maturity_mismatch_rejects() {
        let error = validate_plan(MATURITY).expect_err("status maturity mismatch must reject");
        assert!(error.contains("specification and maturity"));
    }

    #[test]
    fn origin_with_application_path_rejects() {
        let error = validate_plan(ORIGIN).expect_err("origin path must reject");
        assert!(error.contains("source origin") || error.contains("origin"));
    }

    #[test]
    fn anti_rollback_requirement_is_mandatory() {
        let error = validate_plan(ANTI_ROLLBACK).expect_err("anti rollback omission must reject");
        assert!(error.contains("anti_rollback_required") || error.contains("required"));
    }

    #[test]
    fn predicate_cannot_reference_an_unselected_field() {
        let mut value: Value = serde_json::from_str(POSITIVE).expect("fixture JSON");
        value["predicate"]["ast"]["children"][0]["left"]["field_id"] =
            Value::String("not_selected".to_owned());
        let raw = serde_json::to_string(&value).expect("serialize mutation");
        let error = validate_plan(&raw).expect_err("unknown field reference must reject");
        assert!(error.contains("unknown field_id"));
    }

    #[test]
    fn freshness_requires_a_timestamp_field() {
        let mut value: Value = serde_json::from_str(POSITIVE).expect("fixture JSON");
        value["fields"][2]["interpretation"]["value_type"] =
            Value::String("utf8_string".to_owned());
        let raw = serde_json::to_string(&value).expect("serialize mutation");
        let error = validate_plan(&raw).expect_err("string field cannot satisfy freshness");
        assert!(error.contains("timestamp_rfc3339"));
    }

    #[test]
    fn all_selected_fields_must_participate_in_the_predicate() {
        let mut value: Value = serde_json::from_str(POSITIVE).expect("fixture JSON");
        value["predicate"]["ast"]["children"].as_array_mut().unwrap().pop();
        let raw = serde_json::to_string(&value).expect("serialize mutation");
        let error = validate_plan(&raw).expect_err("unused field must reject");
        assert!(error.contains("selected but unused"));
    }

    #[test]
    fn floating_point_input_rejects_before_schema() {
        let raw = POSITIVE.replace("100000", "100000.5");
        let error = validate_plan(&raw).expect_err("floating point must reject");
        assert!(error.contains("floating-point"));
    }
}
