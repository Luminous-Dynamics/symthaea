//! Independent v0.1 joint-receipt integrity verifier and MissionTrace consumer.
//!
//! This module verifies duplicate-key-free JSON, the v0.1 no-float/safe-integer
//! profile, RFC 8785 canonical payload bytes, the SHA-256 known-answer digest,
//! and the no-actuation-authority invariant. Full Draft 2020-12 schema
//! validation remains a separate gate in the parent interop workflow.

#![forbid(unsafe_code)]

use super::joint_metrology_adapter::{
    JointMetrologyEvidence, JointMetrologyReport, QualificationDisposition,
    UpstreamEnergyDisposition, qualify_joint_metrology,
};
use super::{AssuranceDisposition, EventStamp, ReplayDisposition, replay_mission};
use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::fmt;

const MAX_SAFE_JSON_INTEGER: u64 = 9_007_199_254_740_991;
const PROTOCOL_ID: &str = "LOCO-JOINT-RECEIPT-INTEROP-001";
const SCHEMA_VERSION: &str = "0.1.0";
const CANONICALIZATION: &str = "RFC8785-JCS-UTF8";
const DIGEST_SCOPE: &str = "top_level_without_integrity";

/// Strict JSON value for the v0.1 profile.
///
/// Object members are retained as pairs so canonical output can sort by UTF-16
/// code units exactly as required by JCS. Duplicate names are rejected while
/// parsing, before any object representation could overwrite them.
#[derive(Debug, Clone, PartialEq, Eq)]
enum JsonValue {
    Null,
    Bool(bool),
    Integer(i64),
    String(String),
    Array(Vec<JsonValue>),
    Object(Vec<(String, JsonValue)>),
}

struct StrictJsonVisitor;

impl<'de> Visitor<'de> for StrictJsonVisitor {
    type Value = JsonValue;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("strict v0.1 JSON with unique keys, safe integers and no floats")
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(JsonValue::Null)
    }

    fn visit_none<E>(self) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(JsonValue::Null)
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(JsonValue::Bool(value))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        if value.unsigned_abs() > MAX_SAFE_JSON_INTEGER {
            return Err(E::custom(
                "JSON integer exceeds the IEEE-754 safe-integer range",
            ));
        }
        Ok(JsonValue::Integer(value))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        if value > MAX_SAFE_JSON_INTEGER {
            return Err(E::custom(
                "JSON integer exceeds the IEEE-754 safe-integer range",
            ));
        }
        Ok(JsonValue::Integer(value as i64))
    }

    fn visit_f64<E>(self, _value: f64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Err(E::custom(
            "floating-point JSON numbers are forbidden by protocol v0.1",
        ))
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(JsonValue::String(value.to_owned()))
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(JsonValue::String(value))
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(value) = sequence.next_element::<JsonValue>()? {
            values.push(value);
        }
        Ok(JsonValue::Array(values))
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut seen = BTreeSet::new();
        let mut fields = Vec::new();
        while let Some(key) = map.next_key::<String>()? {
            if !seen.insert(key.clone()) {
                return Err(de::Error::custom(format!(
                    "duplicate JSON object key: {key}"
                )));
            }
            let value = map.next_value::<JsonValue>()?;
            fields.push((key, value));
        }
        Ok(JsonValue::Object(fields))
    }
}

impl<'de> Deserialize<'de> for JsonValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(StrictJsonVisitor)
    }
}

/// Validated protocol envelope summary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedReceipt {
    /// Protocol identity.
    pub protocol_id: String,
    /// Exact wire schema version.
    pub schema_version: String,
    /// Verified SHA-256 over the top-level object without the integrity member.
    pub payload_sha256: String,
    /// The protocol can never authorize actuation.
    pub actuation_authorized: bool,
}

/// Structural or integrity failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WireError {
    /// JSON syntax, duplicate-key, float or safe-integer parsing failure.
    InvalidJson(String),
    /// Root must be a JSON object.
    RootMustBeObject,
    /// Required top-level field is missing.
    MissingField(&'static str),
    /// A required field has the wrong JSON type or value.
    InvalidField(&'static str),
    /// Integrity object is malformed.
    InvalidIntegrityField(&'static str),
    /// Claimed and recomputed payload digests differ.
    DigestMismatch {
        expected: String,
        observed: String,
    },
    BindingMismatch {
        field: String,
        expected: String,
        observed: String,
    },
    UnsupportedProfile(String),
    InvalidInterval(&'static str),
}

impl fmt::Display for WireError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for WireError {}

fn parse_strict_json(input: &[u8]) -> Result<JsonValue, WireError> {
    let mut deserializer = serde_json::Deserializer::from_slice(input);
    let value = JsonValue::deserialize(&mut deserializer)
        .map_err(|error| WireError::InvalidJson(error.to_string()))?;
    deserializer
        .end()
        .map_err(|error| WireError::InvalidJson(error.to_string()))?;
    Ok(value)
}

fn get_member<'a>(object: &'a [(String, JsonValue)], name: &str) -> Option<&'a JsonValue> {
    object
        .iter()
        .find_map(|(key, value)| (key == name).then_some(value))
}

fn require_string<'a>(
    object: &'a [(String, JsonValue)],
    name: &'static str,
) -> Result<&'a str, WireError> {
    match get_member(object, name) {
        Some(JsonValue::String(value)) => Ok(value),
        Some(_) => Err(WireError::InvalidField(name)),
        None => Err(WireError::MissingField(name)),
    }
}

fn utf16_key_order(left: &str, right: &str) -> Ordering {
    left.encode_utf16().cmp(right.encode_utf16())
}

fn write_jcs_string(value: &str, output: &mut String) {
    output.push('"');
    for character in value.chars() {
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\u{0008}' => output.push_str("\\b"),
            '\u{0009}' => output.push_str("\\t"),
            '\u{000a}' => output.push_str("\\n"),
            '\u{000c}' => output.push_str("\\f"),
            '\u{000d}' => output.push_str("\\r"),
            character if character <= '\u{001f}' => {
                use fmt::Write as _;
                let _ = write!(output, "\\u{:04x}", character as u32);
            }
            character => output.push(character),
        }
    }
    output.push('"');
}

fn write_jcs_value(value: &JsonValue, output: &mut String) {
    match value {
        JsonValue::Null => output.push_str("null"),
        JsonValue::Bool(false) => output.push_str("false"),
        JsonValue::Bool(true) => output.push_str("true"),
        JsonValue::Integer(value) => output.push_str(&value.to_string()),
        JsonValue::String(value) => write_jcs_string(value, output),
        JsonValue::Array(values) => {
            output.push('[');
            for (index, value) in values.iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                write_jcs_value(value, output);
            }
            output.push(']');
        }
        JsonValue::Object(fields) => {
            let mut sorted = fields.iter().collect::<Vec<_>>();
            sorted.sort_by(|(left, _), (right, _)| utf16_key_order(left, right));
            output.push('{');
            for (index, (key, value)) in sorted.iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                write_jcs_string(key, output);
                output.push(':');
                write_jcs_value(value, output);
            }
            output.push('}');
        }
    }
}

/// Canonicalize strict v0.1 JSON according to the supported RFC 8785 profile.
///
/// This accepts objects, arrays, strings, booleans, null and safe JSON integers.
/// It rejects duplicate names, floats, non-safe integer numbers and trailing data.
pub fn canonicalize_json(input: &[u8]) -> Result<Vec<u8>, WireError> {
    let value = parse_strict_json(input)?;
    let mut output = String::new();
    write_jcs_value(&value, &mut output);
    Ok(output.into_bytes())
}

fn canonical_payload_bytes(value: &JsonValue) -> Result<Vec<u8>, WireError> {
    let JsonValue::Object(fields) = value else {
        return Err(WireError::RootMustBeObject);
    };
    if get_member(fields, "integrity").is_none() {
        return Err(WireError::MissingField("integrity"));
    }
    let payload = JsonValue::Object(
        fields
            .iter()
            .filter(|(key, _)| key != "integrity")
            .cloned()
            .collect(),
    );
    let mut output = String::new();
    write_jcs_value(&payload, &mut output);
    Ok(output.into_bytes())
}

/// Compute SHA-256 over the RFC 8785 canonical top-level object without integrity.
pub fn payload_sha256(input: &[u8]) -> Result<String, WireError> {
    let value = parse_strict_json(input)?;
    let payload = canonical_payload_bytes(&value)?;
    let digest = Sha256::digest(payload);
    Ok(lowercase_hex(&digest))
}

fn lowercase_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

fn integrity_digest(integrity: &[(String, JsonValue)]) -> Result<&str, WireError> {
    let digest = require_string(integrity, "payload_sha256")
        .map_err(|_| WireError::InvalidIntegrityField("payload_sha256"))?;
    if digest.len() != 64
        || !digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(WireError::InvalidIntegrityField("payload_sha256"));
    }
    Ok(digest)
}

/// Parse and verify the integrity envelope and no-actuation-authority invariant.
///
/// This is not a complete JSON Schema validator. The parent workflow continues
/// to validate the closed Draft 2020-12 schema and the entire mutation corpus.
pub fn validate_receipt(input: &[u8]) -> Result<ValidatedReceipt, WireError> {
    let value = parse_strict_json(input)?;
    let JsonValue::Object(fields) = &value else {
        return Err(WireError::RootMustBeObject);
    };

    let protocol_id = require_string(fields, "protocol_id")?;
    if protocol_id != PROTOCOL_ID {
        return Err(WireError::InvalidField("protocol_id"));
    }
    let schema_version = require_string(fields, "schema_version")?;
    if schema_version != SCHEMA_VERSION {
        return Err(WireError::InvalidField("schema_version"));
    }
    match get_member(fields, "actuation_authorized") {
        Some(JsonValue::Bool(false)) => {}
        Some(_) => return Err(WireError::InvalidField("actuation_authorized")),
        None => return Err(WireError::MissingField("actuation_authorized")),
    }

    let integrity = match get_member(fields, "integrity") {
        Some(JsonValue::Object(value)) => value,
        Some(_) => return Err(WireError::InvalidIntegrityField("integrity")),
        None => return Err(WireError::MissingField("integrity")),
    };
    if require_string(integrity, "canonicalization")
        .map_err(|_| WireError::InvalidIntegrityField("canonicalization"))?
        != CANONICALIZATION
    {
        return Err(WireError::InvalidIntegrityField("canonicalization"));
    }
    if require_string(integrity, "digest_scope")
        .map_err(|_| WireError::InvalidIntegrityField("digest_scope"))?
        != DIGEST_SCOPE
    {
        return Err(WireError::InvalidIntegrityField("digest_scope"));
    }
    let expected = integrity_digest(integrity)?.to_owned();
    let payload = canonical_payload_bytes(&value)?;
    let observed = lowercase_hex(&Sha256::digest(payload));
    if observed != expected {
        return Err(WireError::DigestMismatch { expected, observed });
    }

    Ok(ValidatedReceipt {
        protocol_id: protocol_id.to_owned(),
        schema_version: schema_version.to_owned(),
        payload_sha256: observed,
        actuation_authorized: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const GOLDEN_PACKET: &str = include_str!("../fixtures/joint_receipt_v0_1_complete.json");
    const GOLDEN_DIGEST: &str = "cb19e0327c2f47c895eae4e49a89c9f8f9c9918e080d52b201d98b1a794823b8";

    #[test]
    fn synthetic_golden_receipt_matches_rfc8785_sha256_known_answer() {
        let report = validate_receipt(GOLDEN_PACKET.as_bytes()).unwrap();
        assert_eq!(report.protocol_id, PROTOCOL_ID);
        assert_eq!(report.schema_version, SCHEMA_VERSION);
        assert_eq!(report.payload_sha256, GOLDEN_DIGEST);
        assert!(!report.actuation_authorized);
    }

    #[test]
    fn payload_digest_is_independent_of_integrity_member_contents() {
        let original = GOLDEN_PACKET;
        let changed_integrity = original.replace(
            GOLDEN_DIGEST,
            "0000000000000000000000000000000000000000000000000000000000000000",
        );
        assert_eq!(
            payload_sha256(original.as_bytes()).unwrap(),
            payload_sha256(changed_integrity.as_bytes()).unwrap()
        );
        assert!(matches!(
            validate_receipt(changed_integrity.as_bytes()),
            Err(WireError::DigestMismatch { .. })
        ));
    }

    #[test]
    fn payload_mutation_is_detected() {
        let changed = GOLDEN_PACKET.replace("robot-a1-synthetic-01", "robot-a1-synthetic-02");
        assert!(matches!(
            validate_receipt(changed.as_bytes()),
            Err(WireError::DigestMismatch { .. })
        ));
    }

    #[test]
    fn duplicate_top_level_json_key_is_rejected() {
        let duplicated = GOLDEN_PACKET.replacen(
            "{\"schema_version\":",
            "{\"protocol_id\":\"spoofed\",\"schema_version\":",
            1,
        );
        assert!(matches!(
            validate_receipt(duplicated.as_bytes()),
            Err(WireError::InvalidJson(_))
        ));
    }

    #[test]
    fn duplicate_nested_json_key_is_rejected() {
        let duplicated = GOLDEN_PACKET.replacen(
            "\"mission_id\":\"jm-golden-001\"",
            "\"mission_id\":\"jm-golden-001\",\"mission_id\":\"attacker-value\"",
            1,
        );
        assert!(matches!(
            validate_receipt(duplicated.as_bytes()),
            Err(WireError::InvalidJson(_))
        ));
    }

    #[test]
    fn floating_point_values_are_rejected_by_v0_1_profile() {
        assert!(matches!(
            canonicalize_json(br#"{"sample_rate":2000.0}"#),
            Err(WireError::InvalidJson(_))
        ));
        assert!(matches!(
            canonicalize_json(br#"{"sample_rate":2e3}"#),
            Err(WireError::InvalidJson(_))
        ));
    }

    #[test]
    fn numbers_outside_json_safe_integer_range_are_rejected() {
        assert!(matches!(
            canonicalize_json(br#"{"unsafe":9007199254740992}"#),
            Err(WireError::InvalidJson(_))
        ));
        assert!(matches!(
            canonicalize_json(br#"{"unsafe":-9007199254740992}"#),
            Err(WireError::InvalidJson(_))
        ));
    }

    #[test]
    fn object_key_order_is_canonical_and_utf16_ordered() {
        let first = canonicalize_json("{\"z\":1,\"a\":2,\"𐀀\":3,\"\":4}".as_bytes()).unwrap();
        let second = canonicalize_json("{\"\":4,\"𐀀\":3,\"a\":2,\"z\":1}".as_bytes()).unwrap();
        assert_eq!(first, second);
        assert_eq!(
            String::from_utf8(first).unwrap(),
            "{\"a\":2,\"z\":1,\"𐀀\":3,\"\":4}"
        );
    }

    #[test]
    fn jcs_string_escaping_uses_short_control_escapes() {
        let output = canonicalize_json("{\"value\":\"\\b\\t\\n\\f\\r\"}".as_bytes()).unwrap();
        assert_eq!(output, br#"{"value":"\b\t\n\f\r"}"#);
    }

    #[test]
    fn actuation_authority_true_is_never_accepted() {
        let changed = GOLDEN_PACKET.replace(
            "\"actuation_authorized\":false",
            "\"actuation_authorized\":true",
        );
        assert!(matches!(
            validate_receipt(changed.as_bytes()),
            Err(WireError::InvalidField("actuation_authorized"))
                | Err(WireError::DigestMismatch { .. })
        ));
    }

    #[test]
    fn protocol_version_mismatch_is_rejected() {
        let changed = GOLDEN_PACKET.replace(
            "\"schema_version\":\"0.1.0\"",
            "\"schema_version\":\"0.2.0\"",
        );
        assert!(matches!(
            validate_receipt(changed.as_bytes()),
            Err(WireError::InvalidField("schema_version")) | Err(WireError::DigestMismatch { .. })
        ));
    }

    #[test]
    fn malformed_or_uppercase_digest_is_rejected() {
        let changed = GOLDEN_PACKET.replace(
            GOLDEN_DIGEST,
            "CB19E0327C2F47C895EAE4E49A89C9F8F9C9918E080D52B201D98B1A794823B8",
        );
        assert!(matches!(
            validate_receipt(changed.as_bytes()),
            Err(WireError::InvalidIntegrityField("payload_sha256"))
        ));
    }
}

/// Independently supplied clock authority; the packet cannot establish this relation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClockAuthorityBinding {
    SameMonotonicClock { clock_id: String, boot_epoch: u64 },
    Unknown { reason_code: String },
}

/// Independently supplied identities not stored as fields on MissionTrace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JointReceiptAuthorityBindings {
    pub mission_trace_digest: String,
    pub assurance_receipt_digest: String,
    pub terminal_receipt_digest: String,
    pub clock_relation: ClockAuthorityBinding,
}

/// Result of independently validating and joining one v0.1 packet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JointReceiptWireReport {
    pub payload_sha256: String,
    pub joint_metrology: JointMetrologyReport,
    pub actuation_authorized: bool,
}

fn path_value<'a>(root: &'a JsonValue, path: &[&'static str]) -> Result<&'a JsonValue, WireError> {
    let mut current = root;
    for key in path {
        let JsonValue::Object(fields) = current else {
            return Err(WireError::InvalidField(path[0]));
        };
        current = fields
            .iter()
            .find_map(|(name, value)| (name == key).then_some(value))
            .ok_or(WireError::MissingField(path[path.len() - 1]))?;
    }
    Ok(current)
}

fn path_string(root: &JsonValue, path: &[&'static str]) -> Result<String, WireError> {
    match path_value(root, path)? {
        JsonValue::String(value) if !value.trim().is_empty() => Ok(value.clone()),
        _ => Err(WireError::InvalidField(path[path.len() - 1])),
    }
}

fn path_bool(root: &JsonValue, path: &[&'static str]) -> Result<bool, WireError> {
    match path_value(root, path)? {
        JsonValue::Bool(value) => Ok(*value),
        _ => Err(WireError::InvalidField(path[path.len() - 1])),
    }
}

fn path_u64_string(root: &JsonValue, path: &[&'static str]) -> Result<u64, WireError> {
    let value = path_string(root, path)?;
    if value.len() > 1 && value.starts_with('0') {
        return Err(WireError::InvalidField(path[path.len() - 1]));
    }
    value
        .parse::<u64>()
        .map_err(|_| WireError::InvalidField(path[path.len() - 1]))
}

fn path_u64_number(root: &JsonValue, path: &[&'static str]) -> Result<u64, WireError> {
    match path_value(root, path)? {
        JsonValue::Integer(value) if *value >= 0 => Ok(*value as u64),
        _ => Err(WireError::InvalidField(path[path.len() - 1])),
    }
}

fn path_array<'a>(
    root: &'a JsonValue,
    path: &[&'static str],
) -> Result<&'a [JsonValue], WireError> {
    match path_value(root, path)? {
        JsonValue::Array(values) => Ok(values),
        _ => Err(WireError::InvalidField(path[path.len() - 1])),
    }
}

fn bind_text(field: &'static str, observed: &str, expected: &str) -> Result<(), WireError> {
    if observed != expected {
        return Err(WireError::BindingMismatch {
            field: field.to_owned(),
            expected: expected.to_owned(),
            observed: observed.to_owned(),
        });
    }
    Ok(())
}

fn bind_u64(field: &'static str, observed: u64, expected: u64) -> Result<(), WireError> {
    if observed != expected {
        return Err(WireError::BindingMismatch {
            field: field.to_owned(),
            expected: expected.to_string(),
            observed: observed.to_string(),
        });
    }
    Ok(())
}

fn bind_event(
    root: &JsonValue,
    path: &[&'static str],
    stamp: &EventStamp,
) -> Result<(), WireError> {
    let mut id_path = path.to_vec();
    id_path.push("event_id");
    bind_text("event_id", &path_string(root, &id_path)?, &stamp.event_id)?;
    let mut sequence_path = path.to_vec();
    sequence_path.push("sequence");
    bind_u64(
        "event_sequence",
        path_u64_string(root, &sequence_path)?,
        stamp.sequence,
    )
}

fn bind_stamp(
    root: &JsonValue,
    path: &[&'static str],
    stamp: &EventStamp,
) -> Result<(), WireError> {
    let mut clock_path = path.to_vec();
    clock_path.push("clock_id");
    bind_text(
        "clock_id",
        &path_string(root, &clock_path)?,
        &stamp.clock_id,
    )?;
    let mut epoch_path = path.to_vec();
    epoch_path.push("boot_epoch");
    bind_u64(
        "boot_epoch",
        path_u64_string(root, &epoch_path)?,
        stamp.boot_epoch,
    )?;
    let mut lower_path = path.to_vec();
    lower_path.push("lower_ns");
    bind_u64(
        "timestamp.lower_ns",
        path_u64_string(root, &lower_path)?,
        stamp.timestamp.lower_ns,
    )?;
    let mut upper_path = path.to_vec();
    upper_path.push("upper_ns");
    bind_u64(
        "timestamp.upper_ns",
        path_u64_string(root, &upper_path)?,
        stamp.timestamp.upper_ns,
    )
}

fn bind_interval(
    root: &JsonValue,
    path: &[&'static str],
    lower_value: u64,
    upper_value: u64,
) -> Result<(), WireError> {
    let mut lower = path.to_vec();
    lower.push("lower_ns");
    bind_u64(
        "measurement.lower_ns",
        path_u64_string(root, &lower)?,
        lower_value,
    )?;
    let mut upper = path.to_vec();
    upper.push("upper_ns");
    bind_u64(
        "measurement.upper_ns",
        path_u64_string(root, &upper)?,
        upper_value,
    )
}

fn interval_difference(
    target: (u64, u64),
    source: (u64, u64),
    reason: &'static str,
) -> Result<(u64, u64), WireError> {
    if target.0 > target.1 || source.0 > source.1 {
        return Err(WireError::InvalidInterval(reason));
    }
    let lower = i128::from(target.0) - i128::from(source.1);
    let upper = i128::from(target.1) - i128::from(source.0);
    if lower < 0 || upper < lower {
        return Err(WireError::InvalidInterval(reason));
    }
    Ok((
        u64::try_from(lower).map_err(|_| WireError::InvalidInterval(reason))?,
        u64::try_from(upper).map_err(|_| WireError::InvalidInterval(reason))?,
    ))
}

fn verify_edges(root: &JsonValue) -> Result<(), WireError> {
    let stages: [&'static str; 8] = [
        "source_observation",
        "decision",
        "proposal",
        "assurance",
        "command",
        "actuator_application",
        "physical_transition",
        "terminal",
    ];
    let mut events = Vec::with_capacity(stages.len());
    for stage in stages {
        events.push((
            path_string(root, &["events", stage, "event_id"])?,
            path_u64_string(root, &["events", stage, "sequence"])?,
        ));
    }
    if events.windows(2).any(|pair| pair[0].1 >= pair[1].1) {
        return Err(WireError::InvalidField("event_sequence_order"));
    }
    let edges = path_array(root, &["causal_edges"])?;
    if edges.len() != events.len() - 1 {
        return Err(WireError::InvalidField("causal_edges_length"));
    }
    for (index, edge) in edges.iter().enumerate() {
        let wrapper = JsonValue::Object(vec![("edge".to_owned(), edge.clone())]);
        let from_id = path_string(&wrapper, &["edge", "from_event_id"])?;
        let to_id = path_string(&wrapper, &["edge", "to_event_id"])?;
        let from_sequence = path_u64_string(&wrapper, &["edge", "from_sequence"])?;
        let to_sequence = path_u64_string(&wrapper, &["edge", "to_sequence"])?;
        if from_id != events[index].0
            || to_id != events[index + 1].0
            || from_sequence != events[index].1
            || to_sequence != events[index + 1].1
        {
            return Err(WireError::BindingMismatch {
                field: format!("causal_edges[{index}]"),
                expected: format!("{:?}", (&events[index], &events[index + 1])),
                observed: format!("{:?}", (from_id, from_sequence, to_id, to_sequence)),
            });
        }
    }
    Ok(())
}

/// Validate JCS integrity and join packet claims to the exact supplied trace,
/// instrumentation and authority receipts. It never constructs a MissionTrace.
pub fn verify_and_join_joint_receipt(
    packet: &[u8],
    evidence: &JointMetrologyEvidence,
    authority: &JointReceiptAuthorityBindings,
) -> Result<JointReceiptWireReport, WireError> {
    for (field, value) in [
        (
            "mission_trace_digest",
            authority.mission_trace_digest.as_str(),
        ),
        (
            "assurance_receipt_digest",
            authority.assurance_receipt_digest.as_str(),
        ),
        (
            "terminal_receipt_digest",
            authority.terminal_receipt_digest.as_str(),
        ),
    ] {
        if value.trim().is_empty() {
            return Err(WireError::InvalidField(field));
        }
    }

    match &authority.clock_relation {
        ClockAuthorityBinding::SameMonotonicClock { clock_id, .. }
            if clock_id.trim().is_empty() =>
        {
            return Err(WireError::InvalidField("clock_relation.clock_id"));
        }
        ClockAuthorityBinding::Unknown { reason_code } if reason_code.trim().is_empty() => {
            return Err(WireError::InvalidField("clock_relation.reason_code"));
        }
        _ => {}
    }

    let verified = validate_receipt(packet)?;
    let mut deserializer = serde_json::Deserializer::from_slice(packet);
    let root = JsonValue::deserialize(&mut deserializer)
        .map_err(|error| WireError::InvalidJson(error.to_string()))?;
    deserializer
        .end()
        .map_err(|error| WireError::InvalidJson(error.to_string()))?;

    let profile = path_string(&root, &["profile"])?;
    if profile != "synthetic_known_answer" && profile != "physical_measurement" {
        return Err(WireError::UnsupportedProfile(profile));
    }
    if path_string(&root, &["producer", "repository_full_name"])? != "Luminous-Dynamics/symtropy"
        || path_string(&root, &["consumer_reference", "repository_full_name"])?
            != "Luminous-Dynamics/symthaea"
        || path_string(&root, &["consumer_reference", "branch"])?
            != "research/rob-causal-mission-replay-001"
        || path_string(&root, &["consumer_reference", "module"])? != "joint_metrology_adapter"
    {
        return Err(WireError::InvalidField("producer_consumer_identity"));
    }

    let mission = &evidence.mission;
    if mission.observations.len() != 1
        || mission.decisions.len() != 1
        || mission.chunks.len() != 1
        || mission.assurance.len() != 1
        || mission.applications.len() != 1
        || mission.physical_transitions.len() != 1
        || mission.terminal.is_none()
        || !mission.remote_requests.is_empty()
    {
        return Err(WireError::UnsupportedProfile(
            "v0.1 requires one observation, decision, proposal, assurance, application and transition".to_owned(),
        ));
    }
    let observation = &mission.observations[0];
    let decision = &mission.decisions[0];
    let proposal = &mission.chunks[0];
    let assurance = &mission.assurance[0];
    let application = &mission.applications[0];
    let transition = &mission.physical_transitions[0];
    let terminal = mission.terminal.as_ref().unwrap();
    let instruments = &evidence.instruments;

    for (field, observed, expected) in [
        (
            "scope.mission_id",
            path_string(&root, &["scope", "mission_id"])?,
            mission.scope.mission_id.as_str(),
        ),
        (
            "scope.robot_id",
            path_string(&root, &["scope", "robot_id"])?,
            mission.scope.robot_id.as_str(),
        ),
        (
            "scope.clock_id",
            path_string(&root, &["scope", "clock_id"])?,
            mission.scope.clock_id.as_str(),
        ),
        (
            "scope.scenario_manifest_digest",
            path_string(&root, &["scope", "scenario_manifest_digest"])?,
            mission.scope.scenario_manifest_digest.as_str(),
        ),
        (
            "scope.mission_trace_digest",
            path_string(&root, &["scope", "mission_trace_digest"])?,
            authority.mission_trace_digest.as_str(),
        ),
        (
            "scope.actuator_id",
            path_string(&root, &["scope", "actuator_id"])?,
            instruments.actuator_id.as_str(),
        ),
    ] {
        bind_text(field, &observed, expected)?;
    }
    bind_u64(
        "scope.boot_epoch",
        path_u64_string(&root, &["scope", "boot_epoch"])?,
        mission.scope.boot_epoch,
    )?;

    bind_event(&root, &["events", "source_observation"], &observation.stamp)?;
    bind_stamp(
        &root,
        &["events", "source_observation", "capture"],
        &observation.stamp,
    )?;
    bind_text(
        "source_observation.id",
        &path_string(&root, &["events", "source_observation", "observation_id"])?,
        &observation.observation_id,
    )?;
    bind_text(
        "source_observation.modality",
        &path_string(&root, &["events", "source_observation", "source_modality"])?,
        &observation.modality,
    )?;
    bind_u64(
        "source_observation.sequence",
        path_u64_string(&root, &["events", "source_observation", "source_sequence"])?,
        observation.source_sequence,
    )?;
    bind_text(
        "source_observation.sensor_calibration",
        &path_string(
            &root,
            &["events", "source_observation", "sensor_calibration_digest"],
        )?,
        &observation.sensor_digest,
    )?;

    bind_event(&root, &["events", "decision"], &decision.stamp)?;
    for (field, observed, expected) in [
        (
            "decision.id",
            path_string(&root, &["events", "decision", "decision_id"])?,
            decision.decision_id.as_str(),
        ),
        (
            "decision.model",
            path_string(&root, &["events", "decision", "model_digest"])?,
            decision.loss_model_digest.as_str(),
        ),
        (
            "decision.policy",
            path_string(&root, &["events", "decision", "policy_digest"])?,
            decision.policy_digest.as_str(),
        ),
        (
            "decision.action_domain",
            path_string(&root, &["events", "decision", "action_domain_digest"])?,
            decision.action_domain_digest.as_str(),
        ),
        (
            "decision.calibration",
            path_string(&root, &["events", "decision", "calibration_receipt_digest"])?,
            decision.calibration_digest.as_str(),
        ),
    ] {
        bind_text(field, &observed, expected)?;
    }
    bind_u64(
        "decision.regret",
        path_u64_string(&root, &["events", "decision", "regret_upper_units"])?,
        decision.regret_upper_units,
    )?;
    bind_u64(
        "decision.epsilon",
        path_u64_string(&root, &["events", "decision", "epsilon_task_loss_units"])?,
        decision.epsilon_task_loss_units,
    )?;
    bind_u64(
        "decision.valid_through",
        path_u64_string(&root, &["events", "decision", "valid_through_tick"])?,
        decision.valid_through_tick,
    )?;

    bind_event(&root, &["events", "proposal"], &proposal.stamp)?;
    bind_text(
        "proposal.id",
        &path_string(&root, &["events", "proposal", "proposal_id"])?,
        &proposal.proposal_id,
    )?;
    bind_text(
        "proposal.prefix_expected",
        &path_string(
            &root,
            &["events", "proposal", "executed_prefix_expected_digest"],
        )?,
        &proposal.expected_executed_prefix_digest,
    )?;
    bind_text(
        "proposal.prefix_observed",
        &path_string(
            &root,
            &["events", "proposal", "executed_prefix_observed_digest"],
        )?,
        &proposal.observed_executed_prefix_digest,
    )?;

    bind_event(&root, &["events", "assurance"], &assurance.stamp)?;
    let expected_assurance = match assurance.disposition {
        AssuranceDisposition::Admitted => "Qualified",
        AssuranceDisposition::Unknown => "Unknown",
        AssuranceDisposition::Rejected => "Rejected",
    };
    bind_text(
        "assurance.disposition",
        &path_string(&root, &["events", "assurance", "disposition"])?,
        expected_assurance,
    )?;
    bind_text(
        "assurance.receipt_digest",
        &path_string(&root, &["events", "assurance", "receipt_digest"])?,
        &authority.assurance_receipt_digest,
    )?;

    bind_text(
        "command.id",
        &path_string(&root, &["events", "command", "command_id"])?,
        &application.command_id,
    )?;
    let command_sequence = path_u64_string(&root, &["events", "command", "sequence"])?;
    if !(assurance.stamp.sequence < command_sequence
        && command_sequence < application.stamp.sequence)
    {
        return Err(WireError::InvalidField("command_sequence"));
    }

    bind_event(
        &root,
        &["events", "actuator_application"],
        &application.stamp,
    )?;
    bind_stamp(
        &root,
        &["events", "actuator_application", "stamp"],
        &application.stamp,
    )?;
    bind_text(
        "application.receipt",
        &path_string(
            &root,
            &[
                "events",
                "actuator_application",
                "application_receipt_digest",
            ],
        )?,
        &application.application_receipt_digest,
    )?;
    bind_text(
        "application.actuator",
        &path_string(&root, &["events", "actuator_application", "actuator_id"])?,
        &application.actuator_id,
    )?;
    bind_u64(
        "application.tick",
        path_u64_string(&root, &["events", "actuator_application", "tick"])?,
        application.stamp.tick,
    )?;

    bind_event(&root, &["events", "physical_transition"], &transition.stamp)?;
    bind_stamp(
        &root,
        &["events", "physical_transition", "stamp"],
        &transition.stamp,
    )?;
    bind_text(
        "physical.transition_id",
        &path_string(
            &root,
            &["events", "physical_transition", "physical_transition_id"],
        )?,
        &transition.transition_id,
    )?;
    bind_text(
        "physical.predecessor",
        &path_string(
            &root,
            &[
                "events",
                "physical_transition",
                "reserve_predecessor_digest",
            ],
        )?,
        &transition.predecessor_state_digest,
    )?;
    bind_text(
        "physical.result",
        &path_string(
            &root,
            &["events", "physical_transition", "reserve_result_digest"],
        )?,
        &transition.resulting_state_digest,
    )?;
    bind_text(
        "physical.thermal_snapshot",
        &path_string(
            &root,
            &["events", "physical_transition", "thermal_snapshot_digest"],
        )?,
        &transition.thermal_snapshot_digest,
    )?;

    bind_event(&root, &["events", "terminal"], &terminal.stamp)?;
    bind_stamp(&root, &["events", "terminal", "stamp"], &terminal.stamp)?;
    bind_text(
        "terminal.receipt",
        &path_string(&root, &["events", "terminal", "terminal_receipt_digest"])?,
        &authority.terminal_receipt_digest,
    )?;
    bind_text(
        "terminal.reserve",
        &path_string(&root, &["events", "terminal", "reserve_digest"])?,
        &terminal.reserve_state_digest,
    )?;
    bind_text(
        "terminal.safe_stop",
        &path_bool(&root, &["events", "terminal", "safe_stop_qualified"])?.to_string(),
        &(terminal.disposition == super::TerminalDisposition::SafeStopQualified).to_string(),
    )?;

    for (field, expected) in [
        (
            "actuator_configuration_digest",
            instruments.actuator_configuration_digest.as_str(),
        ),
        (
            "controller_configuration_digest",
            instruments.controller_configuration_digest.as_str(),
        ),
        ("policy_digest", instruments.policy_digest.as_str()),
        ("model_digest", instruments.model_digest.as_str()),
        (
            "action_domain_digest",
            instruments.action_domain_digest.as_str(),
        ),
        (
            "source_sensor_calibration_digest",
            instruments.sensor_calibration_digest.as_str(),
        ),
        (
            "torque_transducer_calibration_digest",
            instruments.torque_transducer_calibration_digest.as_str(),
        ),
        (
            "power_analyzer_calibration_digest",
            instruments.power_analyzer_calibration_digest.as_str(),
        ),
        (
            "thermal_sensor_calibration_digest",
            instruments.thermal_sensor_calibration_digest.as_str(),
        ),
        (
            "sampling_spec_digest",
            instruments.sampling_spec_digest.as_str(),
        ),
        (
            "energy_boundary_manifest_digest",
            instruments.energy_boundary_manifest_digest.as_str(),
        ),
        (
            "decision_calibration_receipt_digest",
            instruments.decision_calibration_receipt_digest.as_str(),
        ),
        (
            "actuator_application_receipt_digest",
            instruments.actuator_application_receipt_digest.as_str(),
        ),
        (
            "executed_prefix_expected_digest",
            instruments.executed_prefix_expected_digest.as_str(),
        ),
        (
            "executed_prefix_observed_digest",
            instruments.executed_prefix_observed_digest.as_str(),
        ),
    ] {
        bind_text(
            field,
            &path_string(&root, &["configuration", field])?,
            expected,
        )?;
    }

    let app_tick = path_u64_string(&root, &["decision", "application_tick"])?;
    bind_u64(
        "decision.application_tick",
        app_tick,
        application.stamp.tick,
    )?;
    bind_u64(
        "decision.regret_top",
        path_u64_string(&root, &["decision", "regret_upper_units"])?,
        decision.regret_upper_units,
    )?;
    bind_u64(
        "decision.epsilon_top",
        path_u64_string(&root, &["decision", "epsilon_task_loss_units"])?,
        decision.epsilon_task_loss_units,
    )?;
    bind_u64(
        "decision.valid_tick_top",
        path_u64_string(&root, &["decision", "valid_through_tick"])?,
        decision.valid_through_tick,
    )?;
    bind_text(
        "decision.calibration_top",
        &path_string(&root, &["decision", "calibration_receipt_digest"])?,
        &decision.calibration_digest,
    )?;
    let expected_decision = if app_tick > decision.valid_through_tick {
        "Expired"
    } else if decision.calibration_disposition == super::CalibrationDisposition::Qualified
        && decision.regret_upper_units <= decision.epsilon_task_loss_units
    {
        "Sufficient"
    } else {
        "Insufficient"
    };
    bind_text(
        "decision.disposition",
        &path_string(&root, &["decision", "disposition"])?,
        expected_decision,
    )?;

    // v0.1 single-clock profile: require the caller's clock identity before
    // any numeric age is accepted. Unknown mapping must carry null AoI.
    match &authority.clock_relation {
        ClockAuthorityBinding::SameMonotonicClock {
            clock_id,
            boot_epoch,
        } => {
            bind_text(
                "clock_relation.kind",
                &path_string(&root, &["clock_relation", "kind"])?,
                "same_monotonic_clock_verified",
            )?;
            bind_text(
                "clock_relation.id",
                &path_string(&root, &["clock_relation", "clock_id"])?,
                clock_id,
            )?;
            bind_u64(
                "clock_relation.epoch",
                path_u64_string(&root, &["clock_relation", "boot_epoch"])?,
                *boot_epoch,
            )?;
            if observation.stamp.clock_id != *clock_id
                || application.stamp.clock_id != *clock_id
                || observation.stamp.boot_epoch != *boot_epoch
                || application.stamp.boot_epoch != *boot_epoch
            {
                return Err(WireError::InvalidField("clock_authority_binding"));
            }
            let aoi = interval_difference(
                (
                    application.stamp.timestamp.lower_ns,
                    application.stamp.timestamp.upper_ns,
                ),
                (
                    observation.stamp.timestamp.lower_ns,
                    observation.stamp.timestamp.upper_ns,
                ),
                "source_to_actuator_aoi",
            )?;
            bind_text(
                "aoi.disposition",
                &path_string(
                    &root,
                    &["measurements", "source_to_actuator_aoi", "disposition"],
                )?,
                "Qualified",
            )?;
            bind_interval(
                &root,
                &["measurements", "source_to_actuator_aoi", "value"],
                aoi.0,
                aoi.1,
            )?;
        }
        ClockAuthorityBinding::Unknown { reason_code } => {
            bind_text(
                "clock_relation.kind",
                &path_string(&root, &["clock_relation", "kind"])?,
                "unknown",
            )?;
            bind_text(
                "clock_relation.reason_code",
                &path_string(&root, &["clock_relation", "reason_code"])?,
                reason_code,
            )?;
            bind_text(
                "aoi.disposition",
                &path_string(
                    &root,
                    &["measurements", "source_to_actuator_aoi", "disposition"],
                )?,
                "Unknown",
            )?;
            if !matches!(
                path_value(&root, &["measurements", "source_to_actuator_aoi", "value"])?,
                JsonValue::Null
            ) {
                return Err(WireError::InvalidField("aoi.value"));
            }
        }
    }

    let onset = evidence
        .motion_onset
        .as_ref()
        .ok_or(WireError::MissingField("motion_onset"))?;
    bind_text(
        "motion.command_id",
        &path_string(&root, &["events", "motion_onset", "command_id"])?,
        &onset.command_id,
    )?;
    bind_stamp(&root, &["events", "motion_onset", "stamp"], &onset.stamp)?;
    if onset.stamp.clock_id == application.stamp.clock_id
        && onset.stamp.boot_epoch == application.stamp.boot_epoch
    {
        let response = interval_difference(
            (
                onset.stamp.timestamp.lower_ns,
                onset.stamp.timestamp.upper_ns,
            ),
            (
                application.stamp.timestamp.lower_ns,
                application.stamp.timestamp.upper_ns,
            ),
            "command_to_motion",
        )?;
        bind_text(
            "response.disposition",
            &path_string(&root, &["measurements", "command_to_motion", "disposition"])?,
            "Qualified",
        )?;
        bind_interval(
            &root,
            &["measurements", "command_to_motion", "value"],
            response.0,
            response.1,
        )?;
    }

    verify_edges(&root)?;

    bind_text(
        "thermal.transition_id",
        &path_string(&root, &["thermal", "physical_transition_id"])?,
        &transition.transition_id,
    )?;
    bind_text(
        "thermal.calibration",
        &path_string(&root, &["thermal", "sensor_calibration_digest"])?,
        &instruments.thermal_sensor_calibration_digest,
    )?;
    bind_text(
        "thermal.result",
        &path_string(&root, &["thermal", "result_digest"])?,
        &transition.thermal_snapshot_digest,
    )?;
    bind_text(
        "thermal.predecessor",
        &path_string(&root, &["thermal", "expected_predecessor_digest"])?,
        &path_string(&root, &["thermal", "observed_predecessor_digest"])?,
    )?;
    if path_bool(&root, &["thermal", "sequence_reset"])? {
        return Err(WireError::InvalidField("thermal.sequence_reset"));
    }

    bind_text(
        "energy.boundary_manifest",
        &path_string(&root, &["energy", "boundary_manifest_digest"])?,
        &evidence.energy.boundary_manifest_digest,
    )?;
    let energy_kind = match evidence.energy.closure_disposition {
        UpstreamEnergyDisposition::Balanced => "Balanced",
        UpstreamEnergyDisposition::Unknown => "Unknown",
        UpstreamEnergyDisposition::Inconclusive => "Inconclusive",
        UpstreamEnergyDisposition::Invalid => "Invalid",
    };
    bind_text(
        "energy.disposition",
        &path_string(&root, &["energy", "upstream_disposition"])?,
        energy_kind,
    )?;
    bind_text(
        "energy.closure_digest",
        &path_string(&root, &["energy", "closure_receipt_digest"])?,
        &evidence.energy.closure_receipt_digest,
    )?;
    if path_bool(&root, &["energy", "boundary_complete"])? != evidence.energy.boundary_complete {
        return Err(WireError::InvalidField("energy.boundary_complete"));
    }
    bind_u64(
        "sampling.measured",
        path_u64_number(&root, &["sampling", "measured_rate_hz"])?,
        instruments.measured_sample_rate_hz,
    )?;
    bind_u64(
        "sampling.required",
        path_u64_number(&root, &["sampling", "required_rate_hz"])?,
        instruments.required_sample_rate_hz,
    )?;
    bind_text(
        "sampling.digest",
        &path_string(&root, &["sampling", "specification_digest"])?,
        &instruments.sampling_spec_digest,
    )?;
    if path_bool(&root, &["sampling", "specification_qualified"])?
        != instruments.sampling_spec_valid
    {
        return Err(WireError::InvalidField("sampling.specification_qualified"));
    }

    bind_text(
        "future_options.disposition",
        &path_string(&root, &["future_options", "disposition"])?,
        "NotEvaluated",
    )?;
    if !path_array(&root, &["future_options", "observation_classes"])?.is_empty() {
        return Err(WireError::UnsupportedProfile(
            "future-option proof is owned by the option-ledger consumer".to_owned(),
        ));
    }

    let joint_metrology = qualify_joint_metrology(evidence);
    let replay = replay_mission(mission);
    bind_text(
        "replay.causal_chain_complete",
        &path_bool(&root, &["replay", "causal_chain_complete"])?.to_string(),
        &(replay.disposition == ReplayDisposition::Closed).to_string(),
    )?;
    let closure = if joint_metrology.disposition == QualificationDisposition::Qualified {
        "Closed"
    } else {
        "Open"
    };
    bind_text(
        "replay.mission_closure",
        &path_string(&root, &["replay", "mission_closure", "disposition"])?,
        closure,
    )?;

    Ok(JointReceiptWireReport {
        payload_sha256: verified.payload_sha256,
        joint_metrology,
        actuation_authorized: false,
    })
}
