//! Offline validation and RFC 8785 canonicalization for REA member-verification receipts.
//!
//! The JSON Schema provides structural validation. This module adds cross-field invariants
//! that are security-relevant and awkward to express in JSON Schema alone.

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64_STANDARD};
use jsonschema::draft202012;
use serde::de::{self, Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use serde_json::{Map, Number, Value};
use sha2::{Digest, Sha256};
use std::fmt;

const PROHIBITED_INFERENCES: &[&str] = &[
    "producer_authenticity",
    "producer_identity",
    "apk_signature_validity",
    "ipa_code_signature_validity",
    "whole_archive_completeness",
    "package_installability",
    "runtime_behavior",
    "masvs_control_pass",
    "semantic_truth",
    "absence_of_malicious_writer",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReceiptError {
    InvalidSchema(String),
    SchemaViolation(String),
    SemanticViolation(String),
    Canonicalization(String),
}

impl fmt::Display for ReceiptError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSchema(detail) => write!(formatter, "invalid receipt schema: {detail}"),
            Self::SchemaViolation(detail) => {
                write!(formatter, "receipt schema violation: {detail}")
            }
            Self::SemanticViolation(detail) => {
                write!(formatter, "receipt semantic violation: {detail}")
            }
            Self::Canonicalization(detail) => {
                write!(formatter, "receipt canonicalization failed: {detail}")
            }
        }
    }
}

impl std::error::Error for ReceiptError {}

/// Parse and validate the schema itself before compiling it into a reusable validator.
pub fn validate_schema_document(schema: &Value) -> Result<(), ReceiptError> {
    jsonschema::meta::validate(schema)
        .map_err(|error| ReceiptError::InvalidSchema(error.to_string()))?;
    draft202012::options()
        .build(schema)
        .map_err(|error| ReceiptError::InvalidSchema(error.to_string()))?;
    Ok(())
}

/// Parse raw JSON while rejecting duplicate property names before a Value can collapse them.
///
/// RFC 8785/JCS inherits the I-JSON requirement that object property names are unique. A normal
/// serde_json::Value parse would otherwise retain only one duplicate value and hide the input
/// ambiguity from downstream validation.
pub fn parse_unique_json(input: &[u8]) -> Result<Value, ReceiptError> {
    let mut deserializer = serde_json::Deserializer::from_slice(input);
    let unique = UniqueValue::deserialize(&mut deserializer)
        .map_err(|error| ReceiptError::SchemaViolation(error.to_string()))?;
    deserializer
        .end()
        .map_err(|error| ReceiptError::SchemaViolation(error.to_string()))?;
    Ok(unique.0)
}

/// Parse, reject duplicate names, and validate a raw receipt document in one operation.
pub fn parse_and_validate_receipt(schema: &Value, input: &[u8]) -> Result<Value, ReceiptError> {
    let receipt = parse_unique_json(input)?;
    validate_receipt(schema, &receipt)?;
    Ok(receipt)
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
        formatter.write_str("a JSON value with unique object property names")
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

/// Validate one receipt structurally and enforce cross-field integrity/claim-ceiling rules.
pub fn validate_receipt(schema: &Value, receipt: &Value) -> Result<(), ReceiptError> {
    let validator = draft202012::options()
        .build(schema)
        .map_err(|error| ReceiptError::InvalidSchema(error.to_string()))?;

    validator
        .validate(receipt)
        .map_err(|error| ReceiptError::SchemaViolation(error.to_string()))?;

    validate_semantics(receipt)
}

/// Produce RFC 8785 canonical JSON bytes only after schema and semantic validation.
///
/// Potentially 64-bit values are decimal strings in the receipt schema. The only JSON numbers
/// permitted by the current receipt schema are small bounded integers: ZIP flags, compression
/// method codes, and the bounded read-buffer length. Floating-point values and larger integers
/// are rejected before canonicalization.
pub fn canonicalize_receipt(schema: &Value, receipt: &Value) -> Result<Vec<u8>, ReceiptError> {
    validate_receipt(schema, receipt)?;
    canonicalize_jcs_value(receipt)
}

/// Parse raw JSON with duplicate-key rejection, validate it, then return the receipt and its
/// canonical JCS bytes. This is the preferred ingress path for untrusted receipt documents.
pub fn canonicalize_raw_receipt(
    schema: &Value,
    input: &[u8],
) -> Result<(Value, Vec<u8>), ReceiptError> {
    let receipt = parse_and_validate_receipt(schema, input)?;
    let canonical = canonicalize_receipt(schema, &receipt)?;
    Ok((receipt, canonical))
}

/// SHA-256 over a schema- and semantics-validated RFC 8785 canonical receipt.
/// The digest identifies the receipt statement; it does not authenticate the producer or make
/// the underlying observations true.
pub fn receipt_sha256(schema: &Value, receipt: &Value) -> Result<String, ReceiptError> {
    let canonical = canonicalize_receipt(schema, receipt)?;
    Ok(format!("{:x}", Sha256::digest(canonical)))
}

/// Preferred raw-input path for receipt digests: duplicate-key parsing, validation, JCS and hash.
pub fn receipt_sha256_from_raw(schema: &Value, input: &[u8]) -> Result<String, ReceiptError> {
    let (_, canonical) = canonicalize_raw_receipt(schema, input)?;
    Ok(format!("{:x}", Sha256::digest(canonical)))
}

fn canonicalize_jcs_value(value: &Value) -> Result<Vec<u8>, ReceiptError> {
    validate_jcs_number_subset(value)?;
    serde_json_canonicalizer::to_vec(value)
        .map_err(|error| ReceiptError::Canonicalization(error.to_string()))
}

fn validate_semantics(receipt: &Value) -> Result<(), ReceiptError> {
    let outcome = string_at(receipt, &["outcome"])?;
    let observations = object_at(receipt, &["observations"])?;
    let source = object_at(receipt, &["source"])?;
    let occurrence = object_at(receipt, &["occurrence"])?;
    let limits = object_at(receipt, &["limits"])?;
    let claim_scope = object_at(receipt, &["claim_scope"])?;
    let attestation = object_at(receipt, &["attestation"])?;

    let expected_asserted_claim = if outcome == "verified" {
        "member_content_integrity_only"
    } else {
        "none"
    };
    if string_at(claim_scope, &["asserted_claim"])? != expected_asserted_claim {
        return Err(ReceiptError::SemanticViolation(format!(
            "outcome {outcome:?} requires asserted_claim={expected_asserted_claim:?}"
        )));
    }

    let not_established = claim_scope
        .get("not_established")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            ReceiptError::SemanticViolation("claim_scope.not_established must be an array".into())
        })?;
    for required in PROHIBITED_INFERENCES {
        if !not_established
            .iter()
            .any(|value| value.as_str() == Some(required))
        {
            return Err(ReceiptError::SemanticViolation(format!(
                "claim ceiling must explicitly retain {required}"
            )));
        }
    }

    match string_at(attestation, &["status"])? {
        "not_provided" => {
            if !attestation
                .get("envelope_reference")
                .is_some_and(Value::is_null)
            {
                return Err(ReceiptError::SemanticViolation(
                    "not_provided attestation must have a null envelope_reference".into(),
                ));
            }
        }
        "separate_envelope_reference" => {
            if string_at(attestation, &["envelope_reference"])?
                .trim()
                .is_empty()
            {
                return Err(ReceiptError::SemanticViolation(
                    "separate attestation requires a non-empty envelope reference".into(),
                ));
            }
        }
        _ => {
            return Err(ReceiptError::SemanticViolation(
                "unknown attestation status".into(),
            ));
        }
    }

    if outcome != "verified" {
        if !observations
            .get("payload_sha256")
            .is_some_and(Value::is_null)
        {
            return Err(ReceiptError::SemanticViolation(
                "non-verified outcome must not publish a successful payload SHA-256".into(),
            ));
        }
        let failure_kind = observations
            .get("failure_kind")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                ReceiptError::SemanticViolation(
                    "non-verified outcome requires a non-null failure_kind".into(),
                )
            })?;
        let disposition_matches = match outcome {
            "rejected" => matches!(failure_kind, "io" | "invalid_archive"),
            "unsupported" => failure_kind == "unsupported",
            "source_changed" => failure_kind == "source_changed",
            "limit_exceeded" => failure_kind == "limit_exceeded",
            "abstained" => failure_kind == "abstained",
            _ => false,
        };
        if !disposition_matches {
            return Err(ReceiptError::SemanticViolation(format!(
                "failure_kind {failure_kind:?} does not match outcome {outcome:?}"
            )));
        }
        return Ok(());
    }

    if !observations.get("failure_kind").is_some_and(Value::is_null)
        || !observations
            .get("failure_detail")
            .is_some_and(Value::is_null)
    {
        return Err(ReceiptError::SemanticViolation(
            "verified outcome must not carry a failure kind or failure detail".into(),
        ));
    }

    let binding_strength = string_at(source, &["binding_strength"])?;
    if !matches!(
        binding_strength,
        "qualified_identity" | "content_digest" | "immutable_snapshot"
    ) {
        return Err(ReceiptError::SemanticViolation(
            "verified outcome requires qualified source identity, content digest, or immutable snapshot"
                .into(),
        ));
    }
    let identity_scheme = string_at(source, &["identity_scheme"])?;
    let scheme_matches = match binding_strength {
        "qualified_identity" => matches!(
            identity_scheme,
            "unix_dev_ino_mode_mtime_ctime" | "handle_file_id"
        ),
        "content_digest" => identity_scheme == "archive_sha256",
        "immutable_snapshot" => identity_scheme == "immutable_snapshot",
        _ => false,
    };
    if !scheme_matches {
        return Err(ReceiptError::SemanticViolation(
            "source identity scheme does not support the declared binding strength".into(),
        ));
    }

    for field in [
        "held_descriptor_matches",
        "selected_path_matches",
        "identity_revalidated_after_attempt",
    ] {
        if !bool_at(source, &[field])? {
            return Err(ReceiptError::SemanticViolation(format!(
                "verified outcome requires source.{field}=true"
            )));
        }
    }

    let archive_sha256 = string_at(source, &["archive_sha256"])?;
    if !is_sha256(archive_sha256) {
        return Err(ReceiptError::SemanticViolation(
            "verified outcome requires a whole-archive SHA-256 for replay binding".into(),
        ));
    }
    let archive_size = decimal_at(source, &["archive_size_bytes"])?;
    let max_archive = decimal_at(limits, &["max_archive_bytes"])?;
    if archive_size > max_archive {
        return Err(ReceiptError::SemanticViolation(
            "archive source size exceeds the configured archive-byte limit".into(),
        ));
    }

    for field in [
        "decoder_reached_eof",
        "decoder_crc32_valid",
        "decoder_size_valid",
        "independent_crc32_valid",
        "metadata_parity_valid",
        "source_revalidated",
    ] {
        if !bool_at(observations, &[field])? {
            return Err(ReceiptError::SemanticViolation(format!(
                "verified outcome requires observations.{field}=true"
            )));
        }
    }

    if string_at(occurrence, &["entry_kind"])? != "file" || bool_at(occurrence, &["encrypted"])? {
        return Err(ReceiptError::SemanticViolation(
            "verified outcome only supports unencrypted regular-file occurrences".into(),
        ));
    }

    let raw_name_base64 = string_at(occurrence, &["raw_filename_base64"])?;
    let raw_name = BASE64_STANDARD.decode(raw_name_base64).map_err(|error| {
        ReceiptError::SemanticViolation(format!("raw_filename_base64 is invalid: {error}"))
    })?;
    if BASE64_STANDARD.encode(raw_name) != raw_name_base64 {
        return Err(ReceiptError::SemanticViolation(
            "raw_filename_base64 must use canonical standard base64".into(),
        ));
    }
    if let Some(duplicate_count) = occurrence
        .get("duplicate_raw_name_count")
        .filter(|value| !value.is_null())
    {
        let count = duplicate_count.as_str().ok_or_else(|| {
            ReceiptError::SemanticViolation(
                "duplicate_raw_name_count must be a decimal string or null".into(),
            )
        })?;
        if count.parse::<u64>().unwrap_or(0) == 0 {
            return Err(ReceiptError::SemanticViolation(
                "duplicate_raw_name_count, when present, must be positive".into(),
            ));
        }
    }

    let occurrence_index = decimal_at(occurrence, &["central_directory_index"])?;
    let max_entries = decimal_at(limits, &["max_entries"])?;
    if occurrence_index >= max_entries {
        return Err(ReceiptError::SemanticViolation(
            "central-directory occurrence index exceeds the configured entry-count limit".into(),
        ));
    }
    let local_header_offset = decimal_at(occurrence, &["local_header_offset_absolute"])?;
    let data_offset = decimal_at(occurrence, &["data_offset_absolute"])?;
    let declared_compressed = decimal_at(occurrence, &["declared_compressed_bytes"])?;
    if local_header_offset > data_offset
        || data_offset > archive_size
        || declared_compressed > archive_size
    {
        return Err(ReceiptError::SemanticViolation(
            "member offsets or compressed length are inconsistent with the admitted archive size"
                .into(),
        ));
    }

    let compression_method = occurrence
        .get("compression_method")
        .and_then(Value::as_u64)
        .ok_or_else(|| {
            ReceiptError::SemanticViolation("compression_method must be an unsigned integer".into())
        })?;
    if !matches!(compression_method, 0 | 8) {
        return Err(ReceiptError::SemanticViolation(
            "verified outcome only supports Stored (0) and Deflate (8)".into(),
        ));
    }

    let declared_size = decimal_at(occurrence, &["declared_uncompressed_bytes"])?;
    let observed_size = decimal_at(observations, &["observed_uncompressed_bytes"])?;
    let member_limit = decimal_at(limits, &["max_member_output_bytes"])?;
    let entry_limit = decimal_at(limits, &["max_entry_declared_uncompressed_bytes"])?;
    let aggregate_declared_limit = decimal_at(limits, &["max_total_declared_uncompressed_bytes"])?;
    if observed_size != declared_size {
        return Err(ReceiptError::SemanticViolation(
            "observed decompressed byte count must equal declared uncompressed size".into(),
        ));
    }
    if declared_size > aggregate_declared_limit {
        return Err(ReceiptError::SemanticViolation(
            "member declared size exceeds the aggregate declared-output limit".into(),
        ));
    }
    if observed_size > member_limit || observed_size > entry_limit {
        return Err(ReceiptError::SemanticViolation(
            "observed decompressed bytes exceed a configured member/entry limit".into(),
        ));
    }

    let declared_crc = string_at(occurrence, &["declared_crc32"])?;
    let observed_crc = string_at(observations, &["observed_crc32"])?;
    if declared_crc != observed_crc {
        return Err(ReceiptError::SemanticViolation(
            "independent observed CRC-32 does not match the central-directory declaration".into(),
        ));
    }
    if !is_sha256(string_at(observations, &["payload_sha256"])?) {
        return Err(ReceiptError::SemanticViolation(
            "verified outcome requires a payload SHA-256".into(),
        ));
    }

    let read_buffer = limits
        .get("read_buffer_bytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| {
            ReceiptError::SemanticViolation("read_buffer_bytes must be an unsigned integer".into())
        })?;
    if !(1..=1_048_576).contains(&read_buffer) {
        return Err(ReceiptError::SemanticViolation(
            "read buffer must be in the range 1..=1 MiB".into(),
        ));
    }

    Ok(())
}

fn validate_jcs_number_subset(value: &Value) -> Result<(), ReceiptError> {
    match value {
        Value::Number(number) => {
            let Some(integer) = number.as_u64() else {
                return Err(ReceiptError::Canonicalization(
                    "receipt contains a negative, fractional, or non-finite JSON number".into(),
                ));
            };
            if integer > 1_048_576 {
                return Err(ReceiptError::Canonicalization(
                    "receipt JSON numbers must stay within the safe small-integer subset; encode large values as decimal strings"
                        .into(),
                ));
            }
        }
        Value::Array(values) => {
            for item in values {
                validate_jcs_number_subset(item)?;
            }
        }
        Value::Object(properties) => {
            for item in properties.values() {
                validate_jcs_number_subset(item)?;
            }
        }
        Value::Null | Value::Bool(_) | Value::String(_) => {}
    }
    Ok(())
}

fn object_at<'a>(value: &'a Value, path: &[&str]) -> Result<&'a Value, ReceiptError> {
    let child = value_at(value, path)?;
    if !child.is_object() {
        return Err(ReceiptError::SemanticViolation(format!(
            "{} must be an object",
            path.join(".")
        )));
    }
    Ok(child)
}

fn string_at<'a>(value: &'a Value, path: &[&str]) -> Result<&'a str, ReceiptError> {
    value_at(value, path)?.as_str().ok_or_else(|| {
        ReceiptError::SemanticViolation(format!("{} must be a string", path.join(".")))
    })
}

fn bool_at(value: &Value, path: &[&str]) -> Result<bool, ReceiptError> {
    value_at(value, path)?.as_bool().ok_or_else(|| {
        ReceiptError::SemanticViolation(format!("{} must be a boolean", path.join(".")))
    })
}

fn decimal_at(value: &Value, path: &[&str]) -> Result<u64, ReceiptError> {
    let text = string_at(value, path)?;
    if text.is_empty()
        || (text.len() > 1 && text.starts_with('0'))
        || !text.bytes().all(|byte| byte.is_ascii_digit())
    {
        return Err(ReceiptError::SemanticViolation(format!(
            "{} must be a canonical unsigned decimal string",
            path.join(".")
        )));
    }
    text.parse::<u64>().map_err(|_| {
        ReceiptError::SemanticViolation(format!(
            "{} exceeds the supported u64 range",
            path.join(".")
        ))
    })
}

fn value_at<'a>(value: &'a Value, path: &[&str]) -> Result<&'a Value, ReceiptError> {
    let mut current = value;
    for component in path {
        current = current.get(*component).ok_or_else(|| {
            ReceiptError::SemanticViolation(format!("missing {}", path.join(".")))
        })?;
    }
    Ok(current)
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const RECEIPT_SCHEMA: &str =
        include_str!("../../../docs/research/rea-zip-content-verification-receipt-v0.schema.json");

    fn schema() -> Value {
        serde_json::from_str(RECEIPT_SCHEMA).unwrap()
    }

    fn valid_verified_receipt() -> Value {
        json!({
            "schema_version": "0.1.0",
            "outcome": "verified",
            "source": {
                "archive_size_bytes": "18446744073709551615",
                "binding_strength": "qualified_identity",
                "identity_scheme": "unix_dev_ino_mode_mtime_ctime",
                "platform": {
                    "os": "linux",
                    "architecture": "x86_64",
                    "kernel_or_filesystem": "Linux 7.0 / ext4"
                },
                "held_descriptor_matches": true,
                "selected_path_matches": true,
                "identity_revalidated_after_attempt": true,
                "identity_commitment_sha256": null,
                "archive_sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
            },
            "occurrence": {
                "central_directory_index": "0",
                "raw_filename_base64": "YXNzZXQuYmlu",
                "decoded_filename": "asset.bin",
                "duplicate_raw_name_count": "1",
                "local_header_offset_absolute": "0",
                "data_offset_absolute": "39",
                "entry_kind": "file",
                "encrypted": false,
                "general_purpose_flags": 0,
                "compression_method": 0,
                "declared_compressed_bytes": "11",
                "declared_uncompressed_bytes": "11",
                "declared_crc32": "0d4a1185"
            },
            "verifier": {
                "implementation": "Symthaea REA reader",
                "source_commit": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "toolchain": "rustc 1.96.0",
                "target": "x86_64-unknown-linux-gnu",
                "cargo_lock_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                "sha256_implementation": "sha2 0.10.9",
                "independent_crc_implementation": "Symthaea incremental bitwise CRC-32/ISO-HDLC",
                "secondary_decoder": "zipcore 0.10.2 / miniz_oxide",
                "secondary_decoder_crc_implementation": "crc32fast 1.5.4",
                "differential_oracles": [
                    {
                        "name": "zip",
                        "version": "9.3.0 / zlib-rs 0.6.8",
                        "role": "independent Deflate decoder and structural oracle",
                        "used_in_this_attempt": false,
                        "dependency_commonality": "shares crc32fast ZIP CRC dependency with zipcore"
                    }
                ]
            },
            "limits": {
                "max_archive_bytes": "18446744073709551615",
                "max_central_directory_bytes": "67108864",
                "max_entries": "100000",
                "max_total_declared_uncompressed_bytes": "4294967296",
                "max_entry_declared_uncompressed_bytes": "2147483648",
                "max_member_output_bytes": "128",
                "read_buffer_bytes": 65536
            },
            "observations": {
                "observed_uncompressed_bytes": "11",
                "observed_crc32": "0d4a1185",
                "payload_sha256": "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9",
                "decoder_reached_eof": true,
                "decoder_crc32_valid": true,
                "decoder_size_valid": true,
                "independent_crc32_valid": true,
                "metadata_parity_valid": true,
                "source_revalidated": true,
                "failure_kind": null,
                "failure_detail": null
            },
            "claim_scope": {
                "asserted_claim": "member_content_integrity_only",
                "not_established": PROHIBITED_INFERENCES
            },
            "attestation": {
                "status": "not_provided",
                "envelope_reference": null
            }
        })
    }

    #[test]
    fn jcs_matches_rfc_8785_number_and_string_vector() {
        let input = r#"{"numbers":[333333333.33333329,1E30,4.50,2e-3,0.000000000000000000000000001],"string":"\u20ac$\u000F\u000aA'\u0042\u0022\u005c\\\"\/","literals":[null,true,false]}"#;
        let value = parse_unique_json(input.as_bytes()).unwrap();
        let canonical = serde_json_canonicalizer::to_string(&value).unwrap();
        assert_eq!(
            canonical,
            r#"{"literals":[null,true,false],"numbers":[333333333.3333333,1e+30,4.5,0.002,1e-27],"string":"€$\u000f\nA'B\"\\\\\"/"}"#
        );
    }

    #[test]
    fn jcs_matches_rfc_8785_utf16_property_sorting_vector() {
        let input = r#"{"\u20ac":"Euro Sign","\r":"Carriage Return","\ufb33":"Hebrew Letter Dalet With Dagesh","1":"One","\ud83d\ude00":"Emoji: Grinning Face","\u0080":"Control","\u00f6":"Latin Small Letter O With Diaeresis"}"#;
        let value = parse_unique_json(input.as_bytes()).unwrap();
        let canonical = serde_json_canonicalizer::to_string(&value).unwrap();
        assert_eq!(
            canonical,
            r#"{\"\\r\":\"Carriage Return\",\"1\":\"One\",\"\":\"Control\",\"ö\":\"Latin Small Letter O With Diaeresis\",\"€\":\"Euro Sign\",\"😀\":\"Emoji: Grinning Face\",\"דּ\":\"Hebrew Letter Dalet With Dagesh\"}"#
        );
    }

    #[test]
    fn raw_receipt_parser_rejects_duplicate_properties_at_any_depth() {
        let duplicate_root = br#"{"outcome":"verified","outcome":"rejected"}"#;
        assert!(parse_unique_json(duplicate_root).is_err());

        let duplicate_nested = br#"{"source":{"archive_sha256":"a","archive_sha256":"b"}}"#;
        assert!(parse_unique_json(duplicate_nested).is_err());

        let unique = br#"{"source":{"archive_sha256":"a"},"outcome":"verified"}"#;
        assert!(parse_unique_json(unique).is_ok());
    }

    #[test]
    fn schema_document_is_valid_and_verified_receipt_passes() {
        let schema = schema();
        validate_schema_document(&schema).unwrap();
        validate_receipt(&schema, &valid_verified_receipt()).unwrap();
    }

    #[test]
    fn canonical_receipt_preserves_u64_values_exactly_as_decimal_strings() {
        let schema = schema();
        let receipt = valid_verified_receipt();
        let canonical = canonicalize_receipt(&schema, &receipt).unwrap();
        let text = std::str::from_utf8(&canonical).unwrap();
        assert!(text.contains(r#""archive_size_bytes":"18446744073709551615""#));
        assert!(!text.contains("18446744073709551615.0"));
        assert_eq!(receipt_sha256(&schema, &receipt).unwrap().len(), 64);

        let raw = serde_json::to_vec(&receipt).unwrap();
        assert_eq!(receipt_sha256_from_raw(&schema, &raw).unwrap().len(), 64);
    }

    #[test]
    fn canonicalization_sorts_object_properties_without_reordering_arrays() {
        let value = json!({"z": 1, "a": {"y": 2, "b": 3}, "list": [2, 1]});
        let canonical = serde_json_canonicalizer::to_string(&value).unwrap();
        assert_eq!(canonical, r#"{"a":{"b":3,"y":2},"list":[2,1],"z":1}"#);
    }

    #[test]
    fn verified_receipt_rejects_missing_archive_replay_digest() {
        let schema = schema();
        let mut receipt = valid_verified_receipt();
        receipt["source"]["archive_sha256"] = Value::Null;
        assert!(validate_receipt(&schema, &receipt).is_err());
    }

    #[test]
    fn verified_receipt_rejects_crc_or_size_disagreement() {
        let schema = schema();
        let mut bad_crc = valid_verified_receipt();
        bad_crc["observations"]["observed_crc32"] = json!("00000000");
        assert!(validate_receipt(&schema, &bad_crc).is_err());

        let mut bad_size = valid_verified_receipt();
        bad_size["observations"]["observed_uncompressed_bytes"] = json!("10");
        assert!(validate_receipt(&schema, &bad_size).is_err());
    }

    #[test]
    fn verified_receipt_rejects_best_effort_identity_and_false_claim_ceiling() {
        let schema = schema();
        let mut weak_identity = valid_verified_receipt();
        weak_identity["source"]["binding_strength"] = json!("best_effort");
        assert!(validate_receipt(&schema, &weak_identity).is_err());

        let mut overclaim = valid_verified_receipt();
        overclaim["claim_scope"]["not_established"]
            .as_array_mut()
            .unwrap()
            .retain(|value| value.as_str() != Some("producer_authenticity"));
        assert!(validate_receipt(&schema, &overclaim).is_err());
    }

    #[test]
    fn nonverified_outcome_cannot_publish_successful_payload_digest() {
        let schema = schema();
        let mut rejected = valid_verified_receipt();
        rejected["outcome"] = json!("rejected");
        assert!(validate_receipt(&schema, &rejected).is_err());

        rejected["observations"]["payload_sha256"] = Value::Null;
        assert!(validate_receipt(&schema, &rejected).is_err());

        rejected["claim_scope"]["asserted_claim"] = json!("none");
        rejected["observations"]["failure_kind"] = json!("invalid_archive");
        rejected["observations"]["failure_detail"] = json!("seeded malformed input");
        assert!(validate_receipt(&schema, &rejected).is_ok());

        rejected["observations"]["failure_kind"] = json!("source_changed");
        assert!(validate_receipt(&schema, &rejected).is_err());
    }

    #[test]
    fn verified_receipt_rejects_noncanonical_base64_and_inconsistent_offsets() {
        let schema = schema();
        let mut bad_base64 = valid_verified_receipt();
        bad_base64["occurrence"]["raw_filename_base64"] = json!("Y Q==");
        assert!(validate_receipt(&schema, &bad_base64).is_err());

        let mut bad_offsets = valid_verified_receipt();
        bad_offsets["occurrence"]["data_offset_absolute"] = json!("0");
        assert!(validate_receipt(&schema, &bad_offsets).is_err());
    }

    #[test]
    fn verified_receipt_rejects_archive_and_occurrence_budget_exceedance() {
        let schema = schema();
        let mut oversized_archive = valid_verified_receipt();
        oversized_archive["limits"]["max_archive_bytes"] = json!("100");
        assert!(validate_receipt(&schema, &oversized_archive).is_err());

        let mut missing_entry_budget = valid_verified_receipt();
        missing_entry_budget["limits"]["max_entries"] = json!("0");
        assert!(validate_receipt(&schema, &missing_entry_budget).is_err());
    }

    #[test]
    fn raw_parse_and_validation_is_one_fail_closed_operation() {
        let schema = schema();
        let receipt = valid_verified_receipt();
        let raw = serde_json::to_vec(&receipt).unwrap();
        assert!(parse_and_validate_receipt(&schema, &raw).is_ok());

        let duplicate = br#"{"outcome":"verified","outcome":"rejected"}"#;
        assert!(parse_and_validate_receipt(&schema, duplicate).is_err());
    }

    #[test]
    fn canonicalization_rejects_floats_and_unsafe_json_integers() {
        assert!(canonicalize_jcs_value(&json!({"value": 1.25})).is_err());
        assert!(canonicalize_jcs_value(&json!({"value": 9007199254740992u64})).is_err());
    }

    #[test]
    fn raw_canonicalization_rejects_duplicate_keys_before_digesting() {
        let schema = schema();
        let duplicate = br#"{"outcome":"verified","outcome":"rejected"}"#;
        assert!(canonicalize_raw_receipt(&schema, duplicate).is_err());
        assert!(receipt_sha256_from_raw(&schema, duplicate).is_err());
    }
}
