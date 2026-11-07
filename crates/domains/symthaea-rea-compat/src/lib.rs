#![forbid(unsafe_code)]

//! Rust-native admission and semantic-identity verification for REA Evidence.
//!
//! This crate separates raw-file identity, REA semantic Evidence identity,
//! schema validation, and producer authenticity. It does not authenticate the
//! producer or prove that an observation happened.
//!
//! Raw-file ingestion rejects duplicate object keys and has a size bound. That
//! is deliberately stricter than JavaScript JSON.parse, which resolves
//! duplicate keys before REA's object-level schema sees the object.

use serde::de::{MapAccess, SeqAccess, Visitor};
use serde::ser::{SerializeMap, SerializeSeq};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Number, Value};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fmt;

/// REA EvidenceBundle and residual-unknown graph verification.
pub mod bundle;

pub const MAX_EVIDENCE_JSON_BYTES: usize = 64 * 1024 * 1024;
pub const EVIDENCE_ID_PREFIX: &str = "ev_";

const EVIDENCE_REQUIRED_KEYS: &[&str] = &[
    "evidence_id",
    "subject",
    "provider",
    "analysis_profile",
    "predicate_type",
    "operation",
    "parameters",
    "raw_result",
    "normalized_result",
    "confidence",
    "authority",
    "environment",
    "limitations",
    "relationship_coverage",
    "locations",
    "evidence_links",
];

const EVIDENCE_OPTIONAL_KEYS: &[&str] = &["caller_input_path"];

const SUBJECT_FORMATS: &[&str] = &[
    "hopper",
    "analysis-database",
    "mach-o",
    "elf",
    "pe",
    "dos-mz",
    "dos-com",
    "raw-binary",
    "gamecube-dol",
    "zip",
    "ipa",
    "apk",
    "msix",
    "appx",
    "hap",
    "hsp",
    "app-pack",
    "asar",
    "ines",
    "godot-pck",
    "unityfs",
    "uimage",
    "dmg",
    "pkg",
    "plist",
    "javascript",
    "source-map",
    "directory",
    "file",
    "unknown",
    "mach-o-universal",
    "javascript-bundle",
    "entitlements",
    "pyinstaller",
    "node-sea",
    "dotnet-bundle",
    "bun-standalone",
    "intel-hex",
    "srec",
    "esp-image",
    "uf2",
    "dfuse",
    "esp-flash-image",
    "fit",
];

const ARCHITECTURES: &[&str] = &[
    "x86", "x86_64", "arm", "arm64", "arm64_32", "mips", "6502", "riscv", "powerpc", "m68k",
    "hexagon",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceErrorKind {
    InvalidJson,
    SchemaViolation,
    Canonicalization,
    IdentityMismatch,
    InputTooLarge,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceError {
    pub kind: EvidenceErrorKind,
    pub message: String,
}

impl EvidenceError {
    fn new(kind: EvidenceErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
}

impl fmt::Display for EvidenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for EvidenceError {}

#[derive(Debug, Clone, PartialEq)]
pub struct VerifiedEvidence {
    /// REA semantic identifier, prefixed with ev_.
    pub evidence_id: String,
    /// Unprefixed SHA-256 used by evidence_id.
    pub semantic_sha256: String,
    /// SHA-256 of the exact input JSON bytes, not the semantic projection.
    pub raw_file_sha256: Option<String>,
    /// Validated original record. No source fields are silently rewritten.
    pub value: Value,
}

fn schema_error(message: impl Into<String>) -> EvidenceError {
    EvidenceError::new(EvidenceErrorKind::SchemaViolation, message)
}

fn object<'a>(value: &'a Value, context: &str) -> Result<&'a Map<String, Value>, EvidenceError> {
    value
        .as_object()
        .ok_or_else(|| schema_error(format!("{context} must be a JSON object")))
}

fn strict_keys(
    map: &Map<String, Value>,
    required: &[&str],
    optional: &[&str],
    context: &str,
) -> Result<(), EvidenceError> {
    for key in required {
        if !map.contains_key(*key) {
            return Err(schema_error(format!("{context}.{key} is required")));
        }
    }
    for key in map.keys() {
        if !required.contains(&key.as_str()) && !optional.contains(&key.as_str()) {
            return Err(schema_error(format!(
                "{context} contains unknown field '{key}'"
            )));
        }
    }
    Ok(())
}

fn field<'a>(
    map: &'a Map<String, Value>,
    key: &str,
    context: &str,
) -> Result<&'a Value, EvidenceError> {
    map.get(key)
        .ok_or_else(|| schema_error(format!("{context}.{key} is required")))
}

fn nonempty_string<'a>(value: &'a Value, context: &str) -> Result<&'a str, EvidenceError> {
    let value = value
        .as_str()
        .ok_or_else(|| schema_error(format!("{context} must be a string")))?;
    if value.is_empty() {
        return Err(schema_error(format!("{context} must not be empty")));
    }
    Ok(value)
}

fn nul_free_string<'a>(value: &'a Value, context: &str) -> Result<&'a str, EvidenceError> {
    let value = value
        .as_str()
        .ok_or_else(|| schema_error(format!("{context} must be a string")))?;
    if value.contains('\0') {
        return Err(schema_error(format!("{context} must not contain NUL")));
    }
    Ok(value)
}

fn is_lower_hex_64(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_sha256(value: &Value, context: &str) -> Result<(), EvidenceError> {
    let digest = value
        .as_str()
        .ok_or_else(|| schema_error(format!("{context} must be a string")))?;
    if !is_lower_hex_64(digest) {
        return Err(schema_error(format!(
            "{context} must be 64 lowercase hexadecimal characters"
        )));
    }
    Ok(())
}

fn validate_prefixed_digest(
    value: &Value,
    prefix: &str,
    context: &str,
) -> Result<(), EvidenceError> {
    let value = value
        .as_str()
        .ok_or_else(|| schema_error(format!("{context} must be a string")))?;
    let Some(digest) = value.strip_prefix(prefix) else {
        return Err(schema_error(format!(
            "{context} must start with '{prefix}'"
        )));
    };
    if !is_lower_hex_64(digest) {
        return Err(schema_error(format!(
            "{context} must contain a 64-character lowercase SHA-256 digest"
        )));
    }
    Ok(())
}

fn validate_provider(
    value: &Value,
    context: &str,
    nullable_version: bool,
) -> Result<(), EvidenceError> {
    let map = object(value, context)?;
    strict_keys(map, &["id", "name", "version"], &[], context)?;
    nonempty_string(field(map, "id", context)?, &format!("{context}.id"))?;
    nonempty_string(field(map, "name", context)?, &format!("{context}.name"))?;
    let version = field(map, "version", context)?;
    if nullable_version {
        if !version.is_null() && version.as_str().is_none() {
            return Err(schema_error(format!(
                "{context}.version must be a string or null"
            )));
        }
        return Ok(());
    }
    nonempty_string(version, &format!("{context}.version"))?;
    Ok(())
}

fn validate_subject(value: &Value) -> Result<(), EvidenceError> {
    if value.is_null() {
        return Ok(());
    }

    let context = "subject";
    let map = object(value, context)?;
    strict_keys(
        map,
        &["name", "digest", "local_path", "format", "architecture"],
        &[],
        context,
    )?;
    nonempty_string(field(map, "name", context)?, "subject.name")?;
    nul_free_string(field(map, "local_path", context)?, "subject.local_path")?;

    let digest = object(field(map, "digest", context)?, "subject.digest")?;
    strict_keys(digest, &["sha256"], &[], "subject.digest")?;
    validate_sha256(
        field(digest, "sha256", "subject.digest")?,
        "subject.digest.sha256",
    )?;

    let format = nonempty_string(field(map, "format", context)?, "subject.format")?;
    if !SUBJECT_FORMATS.contains(&format) {
        return Err(schema_error(format!(
            "subject.format '{format}' is unsupported"
        )));
    }

    let architecture = field(map, "architecture", context)?;
    if !architecture.is_null() {
        let architecture = nonempty_string(architecture, "subject.architecture")?;
        if !ARCHITECTURES.contains(&architecture) {
            return Err(schema_error(format!(
                "subject.architecture '{architecture}' is unsupported"
            )));
        }
        if format == "gamecube-dol" && architecture != "powerpc" {
            return Err(schema_error(
                "gamecube-dol subjects must declare powerpc architecture",
            ));
        }
        if format != "raw-binary"
            && format != "gamecube-dol"
            && matches!(architecture, "6502" | "m68k")
        {
            return Err(schema_error(format!(
                "subject architecture '{architecture}' is only valid for raw-binary"
            )));
        }
    } else if format == "gamecube-dol" {
        return Err(schema_error(
            "gamecube-dol subjects must declare powerpc architecture",
        ));
    }

    Ok(())
}

fn providers_match(left: &Value, right: &Value) -> bool {
    ["id", "name", "version"]
        .into_iter()
        .all(|key| left.get(key) == right.get(key))
}

fn validate_analysis_profile(value: &Value, provider: &Value) -> Result<(), EvidenceError> {
    if value.is_null() {
        return Ok(());
    }

    let context = "analysis_profile";
    let map = object(value, context)?;
    strict_keys(map, &["provider", "parameters", "digest"], &[], context)?;

    let profile_provider = field(map, "provider", context)?;
    validate_provider(profile_provider, "analysis_profile.provider", false)?;

    let profile_parameters = field(map, "parameters", context)?;
    object(profile_parameters, "analysis_profile.parameters")?;
    validate_sha256(field(map, "digest", context)?, "analysis_profile.digest")?;

    if !providers_match(profile_provider, provider) {
        return Err(schema_error(
            "analysis_profile.provider must exactly match Evidence.provider",
        ));
    }

    let digest_input = serde_json::json!({
        "provider": profile_provider,
        "parameters": profile_parameters,
    });
    let expected = canonical_sha256(&digest_input)?;
    let actual = field(map, "digest", context)?
        .as_str()
        .expect("validated SHA-256 string");
    if expected != actual {
        return Err(EvidenceError::new(
            EvidenceErrorKind::IdentityMismatch,
            "analysis_profile.digest does not match its canonical provider/parameters commitment",
        ));
    }

    Ok(())
}

fn validate_environment(value: &Value) -> Result<(), EvidenceError> {
    if value.is_null() {
        return Ok(());
    }
    let context = "environment";
    let map = object(value, context)?;
    strict_keys(
        map,
        &["id", "platform", "architecture", "isolation"],
        &[],
        context,
    )?;
    for key in ["id", "platform", "architecture"] {
        nonempty_string(field(map, key, context)?, &format!("{context}.{key}"))?;
    }
    let isolation = nonempty_string(field(map, "isolation", context)?, "environment.isolation")?;
    if !["none", "process", "container", "virtual-machine"].contains(&isolation) {
        return Err(schema_error(format!(
            "environment.isolation '{isolation}' is unsupported"
        )));
    }
    Ok(())
}

fn validate_string_array(
    value: &Value,
    context: &str,
    nonempty: bool,
) -> Result<(), EvidenceError> {
    let array = value
        .as_array()
        .ok_or_else(|| schema_error(format!("{context} must be an array")))?;
    if nonempty && array.is_empty() {
        return Err(schema_error(format!("{context} must not be empty")));
    }
    for (index, item) in array.iter().enumerate() {
        if nonempty {
            nonempty_string(item, &format!("{context}[{index}]"))?;
        } else if item.as_str().is_none() {
            return Err(schema_error(format!("{context}[{index}] must be a string")));
        }
    }
    Ok(())
}

fn validate_limited_directions(value: &Value, context: &str) -> Result<(), EvidenceError> {
    let array = value
        .as_array()
        .ok_or_else(|| schema_error(format!("{context} must be an array")))?;
    let valid = matches!(array.as_slice(),
        [Value::String(one)] if one == "callers" || one == "callees"
    ) || matches!(array.as_slice(),
        [Value::String(left), Value::String(right)]
            if left == "callers" && right == "callees"
    );
    if !valid {
        return Err(schema_error(format!(
            "{context} must be [callers], [callees], or [callers, callees]"
        )));
    }
    Ok(())
}

fn validate_relationship_coverage(value: &Value) -> Result<(), EvidenceError> {
    if value.is_null() {
        return Ok(());
    }
    let context = "relationship_coverage";
    let map = object(value, context)?;
    let status = nonempty_string(
        field(map, "status", context)?,
        "relationship_coverage.status",
    )?;

    match status {
        "complete" => {
            strict_keys(map, &["status"], &["diagnostics"], context)?;
        }
        "partial" => {
            strict_keys(
                map,
                &["status", "limitations"],
                &["limited_directions", "diagnostics"],
                context,
            )?;
            validate_string_array(
                field(map, "limitations", context)?,
                "relationship_coverage.limitations",
                true,
            )?;
        }
        "unknown" => {
            strict_keys(
                map,
                &["status", "reason"],
                &["limited_directions", "diagnostics"],
                context,
            )?;
            nonempty_string(
                field(map, "reason", context)?,
                "relationship_coverage.reason",
            )?;
        }
        other => {
            return Err(schema_error(format!(
                "relationship_coverage.status '{other}' is unsupported"
            )));
        }
    }

    if let Some(diagnostics) = map.get("diagnostics") {
        validate_string_array(diagnostics, "relationship_coverage.diagnostics", true)?;
    }
    if let Some(directions) = map.get("limited_directions") {
        validate_limited_directions(directions, "relationship_coverage.limited_directions")?;
    }
    Ok(())
}

fn validate_nonnegative_safe_integer(value: &Value, context: &str) -> Result<(), EvidenceError> {
    const MAX_SAFE_INTEGER: f64 = 9_007_199_254_740_991.0;
    let number = value
        .as_f64()
        .ok_or_else(|| schema_error(format!("{context} must be a non-negative safe integer")))?;
    if !number.is_finite() || number < 0.0 || number.fract() != 0.0 || number > MAX_SAFE_INTEGER {
        return Err(schema_error(format!(
            "{context} must be a non-negative JavaScript safe integer"
        )));
    }
    Ok(())
}

fn validate_location(value: &Value, index: usize) -> Result<(), EvidenceError> {
    let context = format!("locations[{index}]");
    let map = object(value, &context)?;
    let kind = nonempty_string(field(map, "kind", &context)?, &format!("{context}.kind"))?;

    let (required, key): (&[&str], &str) = match kind {
        "address" => (&["kind", "address"], "address"),
        "address-range" => (&["kind", "start", "end"], ""),
        "artifact-path" => (&["kind", "path"], "path"),
        "file-offset" => (&["kind", "offset"], ""),
        "file-offset-range" => (&["kind", "start", "end"], ""),
        other => {
            return Err(schema_error(format!(
                "{context}.kind '{other}' is unsupported"
            )));
        }
    };
    strict_keys(map, required, &[], &context)?;

    if !key.is_empty() {
        nonempty_string(field(map, key, &context)?, &format!("{context}.{key}"))?;
    }
    match kind {
        "address-range" => {
            nonempty_string(field(map, "start", &context)?, &format!("{context}.start"))?;
            nonempty_string(field(map, "end", &context)?, &format!("{context}.end"))?;
        }
        "file-offset" => {
            validate_nonnegative_safe_integer(
                field(map, "offset", &context)?,
                &format!("{context}.offset"),
            )?;
        }
        "file-offset-range" => {
            for key in ["start", "end"] {
                validate_nonnegative_safe_integer(
                    field(map, key, &context)?,
                    &format!("{context}.{key}"),
                )?;
            }
        }
        _ => {}
    }
    Ok(())
}

/// Validate the strict provider-neutral REA Evidence envelope.
///
/// Operation-specific normalized_result schemas remain owned by their operation
/// and are intentionally not guessed by this generic compatibility layer.
pub fn validate_evidence_schema(value: &Value) -> Result<(), EvidenceError> {
    let context = "Evidence";
    let map = object(value, context)?;
    strict_keys(map, EVIDENCE_REQUIRED_KEYS, EVIDENCE_OPTIONAL_KEYS, context)?;

    validate_prefixed_digest(
        field(map, "evidence_id", context)?,
        EVIDENCE_ID_PREFIX,
        "evidence_id",
    )?;
    validate_subject(field(map, "subject", context)?)?;

    let provider = field(map, "provider", context)?;
    validate_provider(provider, "provider", true)?;
    validate_analysis_profile(field(map, "analysis_profile", context)?, provider)?;

    nonempty_string(field(map, "predicate_type", context)?, "predicate_type")?;
    nonempty_string(field(map, "operation", context)?, "operation")?;
    object(field(map, "parameters", context)?, "parameters")?;

    if let Some(path) = map.get("caller_input_path") {
        nul_free_string(path, "caller_input_path")?;
    }

    let confidence = nonempty_string(field(map, "confidence", context)?, "confidence")?;
    if !["observed", "derived", "inferred"].contains(&confidence) {
        return Err(schema_error(format!(
            "confidence '{confidence}' is unsupported"
        )));
    }

    let authority = nonempty_string(field(map, "authority", context)?, "authority")?;
    if ![
        "shipped-artifact",
        "controlled-replay",
        "historical-reference",
        "external-service",
        "analyst-inference",
    ]
    .contains(&authority)
    {
        return Err(schema_error(format!(
            "authority '{authority}' is unsupported"
        )));
    }

    validate_environment(field(map, "environment", context)?)?;
    validate_string_array(field(map, "limitations", context)?, "limitations", false)?;
    validate_relationship_coverage(field(map, "relationship_coverage", context)?)?;

    let locations = field(map, "locations", context)?
        .as_array()
        .ok_or_else(|| schema_error("locations must be an array"))?;
    for (index, location) in locations.iter().enumerate() {
        validate_location(location, index)?;
    }

    let evidence_links = field(map, "evidence_links", context)?
        .as_array()
        .ok_or_else(|| schema_error("evidence_links must be an array"))?;
    for (index, evidence_link) in evidence_links.iter().enumerate() {
        validate_prefixed_digest(
            evidence_link,
            EVIDENCE_ID_PREFIX,
            &format!("evidence_links[{index}]"),
        )?;
    }

    Ok(())
}

/// Produce REA's semantic projection, excluding local display/request metadata.
///
/// subject.name, subject.local_path and caller_input_path are omitted. Artifact
/// digest, format and architecture remain part of subject identity.
pub fn semantic_evidence_content(value: &Value) -> Result<Value, EvidenceError> {
    let map = object(value, "Evidence")?;
    let subject = field(map, "subject", "Evidence")?;
    let semantic_subject = if subject.is_null() {
        Value::Null
    } else {
        let subject = object(subject, "subject")?;
        let mut projected = Map::new();
        for key in ["digest", "format", "architecture"] {
            projected.insert(key.to_owned(), field(subject, key, "subject")?.clone());
        }
        Value::Object(projected)
    };

    let mut semantic = Map::new();
    semantic.insert("subject".to_owned(), semantic_subject);
    for key in [
        "provider",
        "analysis_profile",
        "predicate_type",
        "operation",
        "parameters",
        "raw_result",
        "normalized_result",
        "confidence",
        "authority",
        "environment",
        "limitations",
        "relationship_coverage",
        "locations",
        "evidence_links",
    ] {
        semantic.insert(key.to_owned(), field(map, key, "Evidence")?.clone());
    }

    Ok(Value::Object(semantic))
}

/// Serialize a JSON value using ECMAScript's binary64 JSON number semantics
/// and RFC 8785 JCS property ordering/primitive serialization.
///
/// serde_json::Value can retain integer tokens as exact i64/u64 values, while
/// JavaScript JSON.parse converts every JSON number to IEEE-754 binary64 before
/// JSON.stringify. The adapter therefore serializes every Number as f64 rather
/// than forwarding serde_json's integer serializer unchanged.
struct EcmaJsonValue<'a>(&'a Value);

impl Serialize for EcmaJsonValue<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self.0 {
            Value::Null => serializer.serialize_unit(),
            Value::Bool(value) => serializer.serialize_bool(*value),
            Value::Number(value) => {
                let number = value.as_f64().ok_or_else(|| {
                    <S::Error as serde::ser::Error>::custom(
                        "JSON number cannot be represented as finite IEEE-754 binary64",
                    )
                })?;
                if !number.is_finite() {
                    return Err(<S::Error as serde::ser::Error>::custom(
                        "JSON numbers must be finite",
                    ));
                }
                serializer.serialize_f64(number)
            }
            Value::String(value) => serializer.serialize_str(value),
            Value::Array(values) => {
                let mut sequence = serializer.serialize_seq(Some(values.len()))?;
                for value in values {
                    sequence.serialize_element(&EcmaJsonValue(value))?;
                }
                sequence.end()
            }
            Value::Object(values) => {
                let mut map = serializer.serialize_map(Some(values.len()))?;
                for (key, value) in values {
                    map.serialize_entry(key, &EcmaJsonValue(value))?;
                }
                map.end()
            }
        }
    }
}

/// Serialize a JSON value using ECMAScript binary64 numbers and RFC 8785 JCS.
pub fn canonical_jcs_bytes(value: &Value) -> Result<Vec<u8>, EvidenceError> {
    serde_json_canonicalizer::to_vec(&EcmaJsonValue(value)).map_err(|error| {
        EvidenceError::new(
            EvidenceErrorKind::Canonicalization,
            format!("canonical JSON serialization failed: {error}"),
        )
    })
}

/// SHA-256 of bytes as lowercase hexadecimal.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut result = String::with_capacity(64);
    for byte in digest {
        use fmt::Write as _;
        let _ = write!(&mut result, "{byte:02x}");
    }
    result
}

fn canonical_sha256(value: &Value) -> Result<String, EvidenceError> {
    Ok(sha256_hex(&canonical_jcs_bytes(value)?))
}

/// Compute the REA analysis-profile digest over exactly provider and parameters.
pub fn analysis_profile_digest(profile: &Value) -> Result<String, EvidenceError> {
    let map = object(profile, "analysis_profile")?;
    let provider = field(map, "provider", "analysis_profile")?;
    let parameters = field(map, "parameters", "analysis_profile")?;
    let value = serde_json::json!({
        "provider": provider,
        "parameters": parameters,
    });
    canonical_sha256(&value)
}

/// Compute REA's ev_-prefixed semantic identifier from an Evidence-shaped value.
///
/// Use verify_evidence_value for schema validation plus semantic verification.
pub fn compute_evidence_id(value: &Value) -> Result<String, EvidenceError> {
    let semantic = semantic_evidence_content(value)?;
    Ok(format!(
        "{EVIDENCE_ID_PREFIX}{}",
        canonical_sha256(&semantic)?
    ))
}

/// Validate the envelope and analysis-profile commitment, then verify Evidence ID.
/// This does not authenticate the producer or prove the underlying observation.
fn verify_evidence_identity(value: &Value) -> Result<(String, String), EvidenceError> {
    validate_evidence_schema(value)?;
    let expected = compute_evidence_id(value)?;
    let actual = field(object(value, "Evidence")?, "evidence_id", "Evidence")?
        .as_str()
        .expect("validated evidence_id string");
    if expected != actual {
        return Err(EvidenceError::new(
            EvidenceErrorKind::IdentityMismatch,
            format!("Evidence semantic ID mismatch: expected {expected}, received {actual}"),
        ));
    }
    let semantic_sha256 = expected
        .strip_prefix(EVIDENCE_ID_PREFIX)
        .expect("computed Evidence ID has prefix")
        .to_owned();
    Ok((expected, semantic_sha256))
}

pub fn verify_evidence_value(value: &Value) -> Result<VerifiedEvidence, EvidenceError> {
    let (evidence_id, semantic_sha256) = verify_evidence_identity(value)?;
    Ok(VerifiedEvidence {
        evidence_id,
        semantic_sha256,
        raw_file_sha256: None,
        value: value.clone(),
    })
}

/// Parse a saved Evidence JSON file with duplicate-key rejection and a bounded
/// input size, then validate its schema and semantic identifier.
///
/// Duplicate-key rejection and the 64 MiB bound are deliberately stricter than
/// upstream's JavaScript object-level schema boundary. For exact object-level
/// disposition parity on a parsed JSON value, call verify_evidence_value.
/// Parse bounded JSON while rejecting duplicate object keys.
///
/// This parser is shared by Evidence and EvidenceBundle import. Its duplicate-key
/// and input-size policy is stricter than upstream JavaScript JSON.parse.
pub(crate) fn parse_strict_json_bytes(
    input: &[u8],
    max_bytes: usize,
) -> Result<Value, EvidenceError> {
    if input.len() > max_bytes {
        return Err(EvidenceError::new(
            EvidenceErrorKind::InputTooLarge,
            format!(
                "JSON input is {} bytes; native importer limit is {} bytes",
                input.len(),
                max_bytes
            ),
        ));
    }

    let mut deserializer = serde_json::Deserializer::from_slice(input);
    let parsed = StrictJsonValue::deserialize(&mut deserializer).map_err(|error| {
        EvidenceError::new(
            EvidenceErrorKind::InvalidJson,
            format!("JSON parsing failed: {error}"),
        )
    })?;
    deserializer.end().map_err(|error| {
        EvidenceError::new(
            EvidenceErrorKind::InvalidJson,
            format!("JSON has trailing data: {error}"),
        )
    })?;

    Ok(parsed.0)
}

pub fn verify_evidence_json(input: &[u8]) -> Result<VerifiedEvidence, EvidenceError> {
    let parsed = parse_strict_json_bytes(input, MAX_EVIDENCE_JSON_BYTES)?;
    let (evidence_id, semantic_sha256) = verify_evidence_identity(&parsed)?;
    Ok(VerifiedEvidence {
        evidence_id,
        semantic_sha256,
        raw_file_sha256: Some(sha256_hex(input)),
        value: parsed,
    })
}

struct StrictJsonValue(Value);

impl<'de> Deserialize<'de> for StrictJsonValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(StrictJsonVisitor)
    }
}

struct StrictJsonVisitor;

impl<'de> Visitor<'de> for StrictJsonVisitor {
    type Value = StrictJsonValue;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON value with unique object keys")
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(StrictJsonValue(Value::Null))
    }

    fn visit_none<E>(self) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(StrictJsonValue(Value::Null))
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(StrictJsonValue(Value::Bool(value)))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(StrictJsonValue(Value::Number(Number::from(value))))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(StrictJsonValue(Value::Number(Number::from(value))))
    }

    fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        let number =
            Number::from_f64(value).ok_or_else(|| E::custom("JSON numbers must be finite"))?;
        Ok(StrictJsonValue(Value::Number(number)))
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(StrictJsonValue(Value::String(value.to_owned())))
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(StrictJsonValue(Value::String(value)))
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(value) = sequence.next_element::<StrictJsonValue>()? {
            values.push(value.0);
        }
        Ok(StrictJsonValue(Value::Array(values)))
    }

    fn visit_map<A>(self, mut object: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut values = Map::new();
        let mut keys = HashSet::new();
        while let Some((key, value)) = object.next_entry::<String, StrictJsonValue>()? {
            if !keys.insert(key.clone()) {
                return Err(serde::de::Error::custom(format!(
                    "duplicate JSON object key '{key}'"
                )));
            }
            values.insert(key, value.0);
        }
        Ok(StrictJsonValue(Value::Object(values)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const GOLDEN_EVIDENCE_ID: &str =
        "ev_a4e49bdca4cbf80c2ec8e4d7eba116af403bc33a4937ffb7fca00b6aab6590c4";
    const GOLDEN_PROFILE_DIGEST: &str =
        "ff7fbe668a874e8ad9cf87ca1bd96201dacd8bf83fbcbee44cd37b2a5c8c8db4";

    fn golden() -> Value {
        serde_json::from_str(include_str!("../tests/fixtures/rea-evidence-v1.json"))
            .expect("golden fixture is valid JSON")
    }

    #[test]
    fn golden_evidence_id_and_profile_digest_match_reference_vector() {
        let evidence = golden();
        assert_eq!(
            analysis_profile_digest(&evidence["analysis_profile"]).unwrap(),
            GOLDEN_PROFILE_DIGEST
        );
        let verified = verify_evidence_value(&evidence).unwrap();
        assert_eq!(verified.evidence_id, GOLDEN_EVIDENCE_ID);
        assert_eq!(verified.semantic_sha256, &GOLDEN_EVIDENCE_ID[3..]);
        assert_eq!(verified.raw_file_sha256, None);
    }

    #[test]
    fn provider_object_key_order_does_not_change_profile_binding_or_identity() {
        let mut evidence = golden();
        let profile_provider = evidence["analysis_profile"]["provider"]
            .as_object()
            .unwrap();
        let mut reordered = Map::new();
        reordered.insert("version".to_owned(), profile_provider["version"].clone());
        reordered.insert("name".to_owned(), profile_provider["name"].clone());
        reordered.insert("id".to_owned(), profile_provider["id"].clone());
        evidence["analysis_profile"]["provider"] = Value::Object(reordered);
        assert!(verify_evidence_value(&evidence).is_ok());
        assert_eq!(compute_evidence_id(&evidence).unwrap(), GOLDEN_EVIDENCE_ID);
    }

    #[test]
    fn display_paths_and_names_do_not_change_semantic_identity() {
        let mut evidence = golden();
        let original = compute_evidence_id(&evidence).unwrap();
        evidence["subject"]["name"] = json!("renamed-copy.apk");
        evidence["subject"]["local_path"] = json!("/other/location/copy.apk");
        evidence["caller_input_path"] = json!("/other/location/copy.apk");
        assert_eq!(compute_evidence_id(&evidence).unwrap(), original);
    }

    #[test]
    fn semantic_fields_change_the_evidence_identity() {
        let original = compute_evidence_id(&golden()).unwrap();

        let mut changed_result = golden();
        changed_result["normalized_result"]["entries"][0]["size"] = json!(43);
        assert_ne!(compute_evidence_id(&changed_result).unwrap(), original);

        let mut changed_limitations = golden();
        changed_limitations["limitations"] = json!(["scope limited"]);
        assert_ne!(compute_evidence_id(&changed_limitations).unwrap(), original);

        let mut changed_location = golden();
        changed_location["locations"][0]["path"] = json!("classes2.dex");
        assert_ne!(compute_evidence_id(&changed_location).unwrap(), original);

        let mut changed_provider = golden();
        changed_provider["provider"]["version"] = json!("1.2.4");
        assert_ne!(compute_evidence_id(&changed_provider).unwrap(), original);
    }

    #[test]
    fn rejects_stale_evidence_id_and_tampered_profile() {
        let mut evidence = golden();
        evidence["evidence_id"] =
            json!("ev_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
        assert_eq!(
            verify_evidence_value(&evidence).unwrap_err().kind,
            EvidenceErrorKind::IdentityMismatch
        );

        let mut profile = golden();
        profile["analysis_profile"]["parameters"]["mode"] = json!("dynamic");
        assert_eq!(
            verify_evidence_value(&profile).unwrap_err().kind,
            EvidenceErrorKind::IdentityMismatch
        );
    }

    #[test]
    fn optional_caller_path_is_nonsemantic_but_nullable_profile_is_required() {
        let mut evidence = golden();
        let original = compute_evidence_id(&evidence).unwrap();
        evidence
            .as_object_mut()
            .unwrap()
            .remove("caller_input_path");
        assert_eq!(compute_evidence_id(&evidence).unwrap(), original);
        assert!(verify_evidence_value(&evidence).is_ok());

        let mut missing_profile = golden();
        missing_profile
            .as_object_mut()
            .unwrap()
            .remove("analysis_profile");
        assert_eq!(
            validate_evidence_schema(&missing_profile).unwrap_err().kind,
            EvidenceErrorKind::SchemaViolation
        );
    }

    #[test]
    fn file_offsets_match_zod_safe_integer_bounds() {
        let mut evidence = golden();
        evidence["locations"] = json!([{"kind": "file-offset", "offset": 1.0}]);
        let new_id = compute_evidence_id(&evidence).unwrap();
        evidence["evidence_id"] = json!(new_id);
        assert!(verify_evidence_value(&evidence).is_ok());

        evidence["locations"] = json!([{"kind": "file-offset", "offset": 9007199254740991u64}]);
        let new_id = compute_evidence_id(&evidence).unwrap();
        evidence["evidence_id"] = json!(new_id);
        assert!(verify_evidence_value(&evidence).is_ok());

        evidence["locations"][0]["offset"] = json!(9007199254740992u64);
        assert_eq!(
            validate_evidence_schema(&evidence).unwrap_err().kind,
            EvidenceErrorKind::SchemaViolation
        );
    }

    #[test]
    fn rejects_unknown_top_level_fields_and_invalid_digests() {
        let mut unknown = golden();
        unknown["unreviewed_field"] = json!(true);
        assert_eq!(
            validate_evidence_schema(&unknown).unwrap_err().kind,
            EvidenceErrorKind::SchemaViolation
        );

        let mut bad_digest = golden();
        bad_digest["subject"]["digest"]["sha256"] = json!("ABC");
        assert_eq!(
            validate_evidence_schema(&bad_digest).unwrap_err().kind,
            EvidenceErrorKind::SchemaViolation
        );
    }

    #[test]
    fn raw_file_ingestion_rejects_duplicate_object_keys() {
        let input = br#"{"same":1,"same":2}"#;
        let error = verify_evidence_json(input).unwrap_err();
        assert_eq!(error.kind, EvidenceErrorKind::InvalidJson);
        assert!(error.message.contains("duplicate JSON object key"));
    }

    #[test]
    fn raw_file_hash_is_distinct_from_semantic_evidence_identity() {
        let raw = include_str!("../tests/fixtures/rea-evidence-v1.json").as_bytes();
        let verified = verify_evidence_json(raw).unwrap();
        assert_eq!(verified.evidence_id, GOLDEN_EVIDENCE_ID);
        let expected_raw_hash = sha256_hex(raw);
        assert_eq!(
            verified.raw_file_sha256.as_deref(),
            Some(expected_raw_hash.as_str())
        );
        assert_ne!(
            verified.raw_file_sha256.as_deref(),
            Some(verified.semantic_sha256.as_str())
        );
    }

    #[test]
    fn jcs_orders_keys_by_utf16_code_units_and_uses_ecmascript_number_formatting() {
        let value: Value = serde_json::from_str(
            r#"{"":1,"𐀀":2,"numbers":[-0,-0.0,333333333.33333329,1E30,4.50,2e-3,0.000000000000000000000000001]}"#,
        )
        .unwrap();
        let canonical = String::from_utf8(canonical_jcs_bytes(&value).unwrap()).unwrap();
        assert_eq!(
            canonical,
            r#"{"numbers":[0,0,333333333.3333333,1e+30,4.5,0.002,1e-27],"𐀀":2,"":1}"#
        );
    }

    #[test]
    fn canonicalization_converts_integer_tokens_to_javascript_binary64() {
        let value: Value = serde_json::from_str(
            r#"{"large":9007199254740993,"u64":18446744073709551615,"minus":-9007199254740993}"#,
        )
        .unwrap();
        let canonical = String::from_utf8(canonical_jcs_bytes(&value).unwrap()).unwrap();
        assert_eq!(
            canonical,
            r#"{"large":9007199254740992,"minus":-9007199254740992,"u64":18446744073709552000}"#
        );
    }

    #[test]
    fn preserves_explicit_unknown_relationship_coverage() {
        let mut evidence = golden();
        evidence["relationship_coverage"] = json!({
            "status": "unknown",
            "reason": "provider could not inspect relationship graph",
            "diagnostics": ["native provider unavailable"]
        });
        let new_id = compute_evidence_id(&evidence).unwrap();
        evidence["evidence_id"] = json!(new_id);
        assert!(verify_evidence_value(&evidence).is_ok());
        assert_eq!(
            evidence["relationship_coverage"]["status"],
            json!("unknown")
        );
    }

    #[test]
    fn accepts_null_subject() {
        let mut evidence = golden();
        evidence["subject"] = Value::Null;
        let new_id = compute_evidence_id(&evidence).unwrap();
        evidence["evidence_id"] = json!(new_id);
        assert!(verify_evidence_value(&evidence).is_ok());
    }

    #[test]
    fn accepts_empty_top_level_limitation_but_requires_nonempty_coverage_diagnostics() {
        let mut evidence = golden();
        evidence["limitations"] = json!([""]);
        let new_id = compute_evidence_id(&evidence).unwrap();
        evidence["evidence_id"] = json!(new_id);
        assert!(verify_evidence_value(&evidence).is_ok());

        evidence["relationship_coverage"] = json!({
            "status": "complete",
            "diagnostics": [""]
        });
        let new_id = compute_evidence_id(&evidence).unwrap();
        evidence["evidence_id"] = json!(new_id);
        assert_eq!(
            validate_evidence_schema(&evidence).unwrap_err().kind,
            EvidenceErrorKind::SchemaViolation
        );
    }

    #[test]
    fn matches_zod_nullable_provider_version_and_nul_free_path_semantics() {
        let mut evidence = golden();
        evidence["analysis_profile"] = Value::Null;
        evidence["provider"]["version"] = json!("");
        evidence["subject"]["local_path"] = json!("");
        evidence["caller_input_path"] = json!("");
        let new_id = compute_evidence_id(&evidence).unwrap();
        evidence["evidence_id"] = json!(new_id);
        assert!(verify_evidence_value(&evidence).is_ok());

        evidence["subject"]["local_path"] = json!("bad\0path");
        assert_eq!(
            validate_evidence_schema(&evidence).unwrap_err().kind,
            EvidenceErrorKind::SchemaViolation
        );
    }
    #[test]
    fn byte_and_value_admission_share_identity_and_semantics() {
        let raw = include_str!("../tests/fixtures/rea-evidence-v1.json").as_bytes();
        let from_bytes = verify_evidence_json(raw).unwrap();
        let from_value = verify_evidence_value(&from_bytes.value).unwrap();

        assert_eq!(from_bytes.evidence_id, from_value.evidence_id);
        assert_eq!(from_bytes.semantic_sha256, from_value.semantic_sha256);
        assert_eq!(from_bytes.value, from_value.value);
        assert_eq!(
            from_bytes.raw_file_sha256.as_deref(),
            Some(sha256_hex(raw).as_str())
        );
        assert!(from_value.raw_file_sha256.is_none());
    }
}
