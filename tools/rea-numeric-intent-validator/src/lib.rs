//! Standalone schema-bound numeric-intent validator for REA.
//!
//! This crate validates exact path-specific numeric representation contracts.
//! It does not modify REA Evidence v1 identity, infer domain intent, or assert
//! that source values are true. Raw JSON is audited before Value parsing.

use jsonschema::draft202012;
use rea_numeric_lexeme_audit::{NumericLexemeWarning, audit_json};
use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

pub const REPORT_SCHEMA: &str = "luminous.rea.numeric-intent-validation-report.v0";
pub const PROFILE_SCHEMA_BYTES: &str =
    include_str!("../schema/rea-numeric-intent-profile-v0.schema.json");
const DRAFT_2020_12: &str = "https://json-schema.org/draft/2020-12/schema";
const MAX_SAFE_INTEGER: &str = "9007199254740991";

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct NumericIntentProfile {
    schema: String,
    profile_id: String,
    profile_version: u32,
    value_schema: ValueSchemaBinding,
    numeric_paths: Vec<NumericPath>,
    profile_sha256: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct ValueSchemaBinding {
    schema_id: String,
    sha256: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct NumericPath {
    json_pointer: String,
    intent: String,
    #[serde(default)]
    quantity_kind_ref: Option<String>,
    #[serde(default)]
    unit_code_ref: Option<String>,
    contract: Value,
}

#[derive(Debug, Clone, Serialize)]
pub struct NumericPathResult {
    pub json_pointer: String,
    pub intent: String,
    pub status: String,
    pub precision_warning: bool,
    /// Exact source token status. Raw bytes remain bound by source_raw_sha256.
    pub raw_value_status: String,
    pub canonical_binary64: Option<String>,
    pub unit_quantity_reference_status: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct NumericIntentValidationReport {
    pub schema: String,
    pub profile_id: String,
    pub profile_sha256: String,
    pub value_schema_id: String,
    pub value_schema_sha256: String,
    pub source_raw_sha256: String,
    pub source_json_audit: String,
    pub numeric_precision_warnings: Vec<NumericLexemeWarning>,
    pub path_results: Vec<NumericPathResult>,
    pub overall_status: String,
}

#[derive(Debug, Clone)]
pub struct ValidationError {
    pub kind: &'static str,
    pub message: String,
}

impl ValidationError {
    fn new(kind: &'static str, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.kind, self.message)
    }
}

impl std::error::Error for ValidationError {}

type Result<T> = std::result::Result<T, ValidationError>;

/// Validate a profile, its bound Draft 2020-12 value schema, and one raw JSON
/// document. Every input is audited for duplicate keys and bounded traversal
/// before serde_json or JCS processing.
pub fn validate_inputs(
    profile_bytes: &[u8],
    value_schema_bytes: &[u8],
    document_bytes: &[u8],
) -> Result<NumericIntentValidationReport> {
    let _profile_audit = audit_json(profile_bytes)
        .map_err(|e| ValidationError::new("profile_raw_audit", e.to_string()))?;
    let _value_schema_audit = audit_json(value_schema_bytes)
        .map_err(|e| ValidationError::new("value_schema_raw_audit", e.to_string()))?;
    let document_audit = audit_json(document_bytes)
        .map_err(|e| ValidationError::new("document_raw_audit", e.to_string()))?;

    let profile_schema: Value = serde_json::from_str(PROFILE_SCHEMA_BYTES)
        .map_err(|e| ValidationError::new("embedded_profile_schema", e.to_string()))?;
    draft202012::meta::validate(&profile_schema).map_err(|e| {
        ValidationError::new(
            "profile_schema_meta_invalid",
            format!("embedded Draft 2020-12 profile schema is invalid: {e}"),
        )
    })?;

    let profile_value: Value = serde_json::from_slice(profile_bytes)
        .map_err(|e| ValidationError::new("profile_parse", e.to_string()))?;
    let profile_validator = draft202012::options()
        .build(&profile_schema)
        .map_err(|e| ValidationError::new("profile_schema_compile", e.to_string()))?;
    let profile_errors: Vec<String> = profile_validator
        .iter_errors(&profile_value)
        .map(|e| e.to_string())
        .collect();
    if !profile_errors.is_empty() {
        return Err(ValidationError::new(
            "profile_schema_rejected",
            profile_errors.join("; "),
        ));
    }

    let profile: NumericIntentProfile = serde_json::from_value(profile_value.clone())
        .map_err(|e| ValidationError::new("profile_deserialize", e.to_string()))?;
    if profile.schema != "luminous.rea.numeric-intent-profile.v0" || profile.profile_version != 0 {
        return Err(ValidationError::new(
            "unsupported_profile_version",
            "only luminous.rea.numeric-intent-profile.v0 is supported",
        ));
    }

    let profile_body = profile_body_without_digest(&profile_value)?;
    let computed_profile_digest = jcs_sha256(&profile_body)?;
    if computed_profile_digest != profile.profile_sha256 {
        return Err(ValidationError::new(
            "profile_digest_mismatch",
            format!(
                "declared profile_sha256 {} does not match JCS SHA-256 {}",
                profile.profile_sha256, computed_profile_digest
            ),
        ));
    }

    validate_profile_semantics(&profile.numeric_paths)?;

    let value_schema: Value = serde_json::from_slice(value_schema_bytes)
        .map_err(|e| ValidationError::new("value_schema_parse", e.to_string()))?;
    require_draft_2020_12(&value_schema, "value schema")?;
    reject_external_references(&value_schema)?;
    draft202012::meta::validate(&value_schema).map_err(|e| {
        ValidationError::new(
            "value_schema_meta_invalid",
            format!("domain value schema is not a valid Draft 2020-12 schema: {e}"),
        )
    })?;

    let supplied_schema_id = value_schema
        .get("$id")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            ValidationError::new(
                "value_schema_id_missing",
                "value schema must declare $id matching the profile schema_id",
            )
        })?;
    if supplied_schema_id != profile.value_schema.schema_id {
        return Err(ValidationError::new(
            "value_schema_id_mismatch",
            format!(
                "profile binds schema_id {} but supplied schema declares $id {}",
                profile.value_schema.schema_id, supplied_schema_id
            ),
        ));
    }

    let computed_value_schema_digest = jcs_sha256(&value_schema)?;
    if computed_value_schema_digest != profile.value_schema.sha256 {
        return Err(ValidationError::new(
            "value_schema_digest_mismatch",
            format!(
                "profile binds {} but supplied value schema has JCS SHA-256 {}",
                profile.value_schema.sha256, computed_value_schema_digest
            ),
        ));
    }

    let value_validator = draft202012::options()
        .build(&value_schema)
        .map_err(|e| ValidationError::new("value_schema_compile", e.to_string()))?;
    let document: Value = serde_json::from_slice(document_bytes)
        .map_err(|e| ValidationError::new("document_parse", e.to_string()))?;
    let document_errors: Vec<String> = value_validator
        .iter_errors(&document)
        .map(|e| e.to_string())
        .collect();
    if !document_errors.is_empty() {
        return Err(ValidationError::new(
            "document_schema_rejected",
            document_errors.join("; "),
        ));
    }

    let warnings_by_pointer: HashMap<String, NumericLexemeWarning> = document_audit
        .numeric_precision_warnings
        .iter()
        .cloned()
        .map(|warning| (warning.json_pointer.clone(), warning))
        .collect();

    let path_results: Vec<NumericPathResult> = profile
        .numeric_paths
        .iter()
        .map(|path| evaluate_path(path, &document, &warnings_by_pointer))
        .collect::<Result<Vec<_>>>()?;

    let values_conformant = path_results.iter().all(path_result_is_conformant);
    let has_unresolved_authority_references = path_results
        .iter()
        .any(|result| result.unit_quantity_reference_status.is_some());
    let overall_status = if !values_conformant {
        "nonconformant"
    } else if has_unresolved_authority_references {
        "conformant_with_unresolved_authority_references"
    } else {
        "conformant"
    };

    Ok(NumericIntentValidationReport {
        schema: REPORT_SCHEMA.to_owned(),
        profile_id: profile.profile_id,
        profile_sha256: computed_profile_digest,
        value_schema_id: profile.value_schema.schema_id,
        value_schema_sha256: computed_value_schema_digest,
        source_raw_sha256: document_audit.source_raw_sha256,
        source_json_audit: "passed_duplicate_key_and_resource_checks".to_owned(),
        numeric_precision_warnings: document_audit.numeric_precision_warnings,
        path_results,
        overall_status: overall_status.to_owned(),
    })
}

fn profile_body_without_digest(profile: &Value) -> Result<Value> {
    let mut body = profile.clone();
    let object = body
        .as_object_mut()
        .ok_or_else(|| ValidationError::new("profile_shape", "profile root must be an object"))?;
    if object.remove("profile_sha256").is_none() {
        return Err(ValidationError::new(
            "profile_shape",
            "profile_sha256 is required",
        ));
    }
    Ok(body)
}

fn require_draft_2020_12(schema: &Value, label: &str) -> Result<()> {
    if schema.get("$schema").and_then(Value::as_str) != Some(DRAFT_2020_12) {
        return Err(ValidationError::new(
            "unsupported_schema_dialect",
            format!("{label} must declare {DRAFT_2020_12}"),
        ));
    }
    Ok(())
}

/// v0 refuses remote and file-based references. A caller must supply a
/// self-contained schema bundle; validation must not fetch mutable resources.
fn reject_external_references(value: &Value) -> Result<()> {
    match value {
        Value::Object(object) => {
            for (key, child) in object {
                if key == "$ref" || key == "$dynamicRef" {
                    let reference = child.as_str().ok_or_else(|| {
                        ValidationError::new(
                            "invalid_schema_reference",
                            format!("{key} must be a string"),
                        )
                    })?;
                    if !reference.starts_with('#') {
                        return Err(ValidationError::new(
                            "external_schema_reference_forbidden",
                            format!(
                                "external schema reference is not permitted in v0: {reference}"
                            ),
                        ));
                    }
                }
                reject_external_references(child)?;
            }
        }
        Value::Array(items) => {
            for child in items {
                reject_external_references(child)?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn validate_profile_semantics(paths: &[NumericPath]) -> Result<()> {
    let mut seen = HashSet::new();
    for path in paths {
        if !is_valid_non_root_json_pointer(&path.json_pointer) {
            return Err(ValidationError::new(
                "invalid_json_pointer",
                format!("invalid or unsupported JSON Pointer: {}", path.json_pointer),
            ));
        }
        if !seen.insert(path.json_pointer.clone()) {
            return Err(ValidationError::new(
                "duplicate_profile_pointer",
                format!(
                    "numeric path is declared more than once: {}",
                    path.json_pointer
                ),
            ));
        }

        if path.quantity_kind_ref.is_some() != path.unit_code_ref.is_some() {
            return Err(ValidationError::new(
                "incomplete_quantity_unit_reference",
                format!(
                    "{} must declare quantity_kind_ref and unit_code_ref together",
                    path.json_pointer
                ),
            ));
        }

        match path.intent.as_str() {
            "binary64" | "opaque_numeric" => {}
            "safe_integer" => validate_safe_integer_contract(&path.contract, &path.json_pointer)?,
            "exact_integer_string" => {
                validate_exact_integer_contract(&path.contract, &path.json_pointer)?
            }
            "fixed_decimal_string" => {
                validate_fixed_decimal_contract(&path.contract, &path.json_pointer)?
            }
            other => {
                return Err(ValidationError::new(
                    "unsupported_numeric_intent",
                    format!("unsupported intent {other} at {}", path.json_pointer),
                ));
            }
        }
    }
    Ok(())
}

fn contract_string<'a>(contract: &'a Value, key: &str, pointer: &str) -> Result<&'a str> {
    contract.get(key).and_then(Value::as_str).ok_or_else(|| {
        ValidationError::new(
            "invalid_numeric_contract",
            format!("{pointer}: contract.{key} must be a string"),
        )
    })
}

fn validate_safe_integer_contract(contract: &Value, pointer: &str) -> Result<()> {
    let signedness = contract_string(contract, "signedness", pointer)?;
    let minimum = contract_string(contract, "min_value", pointer)?;
    let maximum = contract_string(contract, "max_value", pointer)?;
    let minimum = parse_integer(minimum, false).map_err(|message| {
        ValidationError::new("invalid_numeric_contract", format!("{pointer}: {message}"))
    })?;
    let maximum = parse_integer(maximum, false).map_err(|message| {
        ValidationError::new("invalid_numeric_contract", format!("{pointer}: {message}"))
    })?;

    if signedness == "unsigned" && minimum.starts_with('-') {
        return Err(ValidationError::new(
            "invalid_numeric_contract",
            format!("{pointer}: unsigned safe_integer cannot have a negative minimum"),
        ));
    }
    if compare_signed_decimal(&minimum, &maximum) == std::cmp::Ordering::Greater {
        return Err(ValidationError::new(
            "invalid_numeric_contract",
            format!("{pointer}: min_value exceeds max_value"),
        ));
    }
    let safe_min = format!("-{MAX_SAFE_INTEGER}");
    if compare_signed_decimal(&minimum, &safe_min) == std::cmp::Ordering::Less
        || compare_signed_decimal(&maximum, MAX_SAFE_INTEGER) == std::cmp::Ordering::Greater
    {
        return Err(ValidationError::new(
            "invalid_numeric_contract",
            format!("{pointer}: safe_integer bounds must remain inside ±{MAX_SAFE_INTEGER}"),
        ));
    }
    Ok(())
}

fn validate_exact_integer_contract(contract: &Value, pointer: &str) -> Result<()> {
    let signedness = contract_string(contract, "signedness", pointer)?;
    let minimum = contract_string(contract, "min_value", pointer)?;
    let maximum = contract_string(contract, "max_value", pointer)?;
    let max_digits = contract
        .get("max_digits")
        .and_then(Value::as_u64)
        .ok_or_else(|| {
            ValidationError::new(
                "invalid_numeric_contract",
                format!("{pointer}: max_digits must be an integer"),
            )
        })? as usize;

    let minimum = parse_integer(minimum, false).map_err(|message| {
        ValidationError::new("invalid_numeric_contract", format!("{pointer}: {message}"))
    })?;
    let maximum = parse_integer(maximum, false).map_err(|message| {
        ValidationError::new("invalid_numeric_contract", format!("{pointer}: {message}"))
    })?;
    if signedness == "unsigned" && minimum.starts_with('-') {
        return Err(ValidationError::new(
            "invalid_numeric_contract",
            format!("{pointer}: unsigned exact integer cannot have a negative minimum"),
        ));
    }
    if digit_count(&minimum) > max_digits || digit_count(&maximum) > max_digits {
        return Err(ValidationError::new(
            "invalid_numeric_contract",
            format!("{pointer}: declared bounds exceed max_digits"),
        ));
    }
    if compare_signed_decimal(&minimum, &maximum) == std::cmp::Ordering::Greater {
        return Err(ValidationError::new(
            "invalid_numeric_contract",
            format!("{pointer}: min_value exceeds max_value"),
        ));
    }
    Ok(())
}

fn validate_fixed_decimal_contract(contract: &Value, pointer: &str) -> Result<()> {
    let signedness = contract_string(contract, "signedness", pointer)?;
    let minimum = contract_string(contract, "min_value", pointer)?;
    let maximum = contract_string(contract, "max_value", pointer)?;
    let scale = contract
        .get("scale")
        .and_then(Value::as_u64)
        .ok_or_else(|| {
            ValidationError::new(
                "invalid_numeric_contract",
                format!("{pointer}: scale must be an integer"),
            )
        })? as usize;
    let max_integer_digits = contract
        .get("max_integer_digits")
        .and_then(Value::as_u64)
        .ok_or_else(|| {
            ValidationError::new(
                "invalid_numeric_contract",
                format!("{pointer}: max_integer_digits must be an integer"),
            )
        })? as usize;

    let minimum =
        parse_fixed_decimal(minimum, scale, max_integer_digits, signedness).map_err(|message| {
            ValidationError::new("invalid_numeric_contract", format!("{pointer}: {message}"))
        })?;
    let maximum =
        parse_fixed_decimal(maximum, scale, max_integer_digits, signedness).map_err(|message| {
            ValidationError::new("invalid_numeric_contract", format!("{pointer}: {message}"))
        })?;
    if compare_signed_decimal(&minimum, &maximum) == std::cmp::Ordering::Greater {
        return Err(ValidationError::new(
            "invalid_numeric_contract",
            format!("{pointer}: min_value exceeds max_value"),
        ));
    }
    Ok(())
}

fn evaluate_path(
    path: &NumericPath,
    document: &Value,
    warnings: &HashMap<String, NumericLexemeWarning>,
) -> Result<NumericPathResult> {
    let precision_warning = warnings.contains_key(&path.json_pointer);
    let value = document.pointer(&path.json_pointer);
    let (status, canonical_binary64) = match value {
        None => ("missing", None),
        Some(Value::Null) => ("type_mismatch", None),
        Some(value) => match path.intent.as_str() {
            "binary64" => {
                if !value.is_number() {
                    ("type_mismatch", None)
                } else {
                    let canonical = serde_json_canonicalizer::to_string(value).map_err(|e| {
                        ValidationError::new("binary64_canonicalization", e.to_string())
                    })?;
                    ("valid_binary64", Some(canonical))
                }
            }
            "safe_integer" => {
                if precision_warning {
                    ("unsafe_integer", None)
                } else if let Some(number) = value.as_number() {
                    if !number.is_i64() && !number.is_u64() {
                        ("lexical_form_mismatch", None)
                    } else {
                        match validate_integer_value(&number.to_string(), &path.contract, false) {
                            Ok(()) => ("valid_safe_integer", None),
                            Err(status) => (status, None),
                        }
                    }
                } else {
                    ("type_mismatch", None)
                }
            }
            "exact_integer_string" => {
                if let Some(text) = value.as_str() {
                    match validate_exact_integer_value(text, &path.contract) {
                        Ok(()) => ("valid_exact_integer_string", None),
                        Err(status) => (status, None),
                    }
                } else {
                    ("type_mismatch", None)
                }
            }
            "fixed_decimal_string" => {
                if let Some(text) = value.as_str() {
                    match validate_fixed_decimal_value(text, &path.contract) {
                        Ok(()) => ("valid_fixed_decimal_string", None),
                        Err(status) => (status, None),
                    }
                } else {
                    ("type_mismatch", None)
                }
            }
            "opaque_numeric" => {
                if value.is_number() {
                    ("opaque_numeric_no_arithmetic_authority", None)
                } else {
                    ("type_mismatch", None)
                }
            }
            _ => {
                return Err(ValidationError::new(
                    "unsupported_numeric_intent",
                    format!("unsupported intent at {}", path.json_pointer),
                ));
            }
        },
    };

    let unit_quantity_reference_status =
        if path.quantity_kind_ref.is_some() && path.unit_code_ref.is_some() {
            Some("unresolved_authority_reference_not_validated".to_owned())
        } else {
            None
        };
    let raw_value_status = match path.intent.as_str() {
        "exact_integer_string" | "fixed_decimal_string" => {
            "decoded_string_value_exact_source_bytes_committed"
        }
        "safe_integer" => "integer_value_checked_source_bytes_committed",
        "binary64" => "jcs_binary64_value_source_bytes_committed",
        "opaque_numeric" => "opaque_no_arithmetic_exact_lexeme_not_materialized",
        _ => "unsupported_numeric_intent",
    };

    Ok(NumericPathResult {
        json_pointer: path.json_pointer.clone(),
        intent: path.intent.clone(),
        status: status.to_owned(),
        precision_warning,
        raw_value_status: raw_value_status.to_owned(),
        canonical_binary64,
        unit_quantity_reference_status,
    })
}

fn path_result_is_conformant(result: &NumericPathResult) -> bool {
    matches!(
        result.status.as_str(),
        "valid_binary64"
            | "valid_safe_integer"
            | "valid_exact_integer_string"
            | "valid_fixed_decimal_string"
            | "opaque_numeric_no_arithmetic_authority"
    )
}

fn validate_integer_value(
    value: &str,
    contract: &Value,
    reject_negative_zero: bool,
) -> std::result::Result<(), &'static str> {
    let signedness = contract
        .get("signedness")
        .and_then(Value::as_str)
        .unwrap_or("");
    let minimum = contract
        .get("min_value")
        .and_then(Value::as_str)
        .unwrap_or("");
    let maximum = contract
        .get("max_value")
        .and_then(Value::as_str)
        .unwrap_or("");
    let parsed =
        parse_integer(value, reject_negative_zero).map_err(|_| "invalid_integer_grammar")?;
    let min = parse_integer(minimum, false).map_err(|_| "invalid_numeric_contract")?;
    let max = parse_integer(maximum, false).map_err(|_| "invalid_numeric_contract")?;
    if signedness == "unsigned" && parsed.starts_with('-') {
        return Err("signedness_mismatch");
    }
    if compare_signed_decimal(&parsed, &min) == std::cmp::Ordering::Less
        || compare_signed_decimal(&parsed, &max) == std::cmp::Ordering::Greater
    {
        return Err("range_violation");
    }
    Ok(())
}

fn validate_exact_integer_value(
    value: &str,
    contract: &Value,
) -> std::result::Result<(), &'static str> {
    let max_digits = contract
        .get("max_digits")
        .and_then(Value::as_u64)
        .unwrap_or(0) as usize;
    if digit_count(value) > max_digits {
        return Err("digit_limit_exceeded");
    }
    validate_integer_value(value, contract, true)
}

fn validate_fixed_decimal_value(
    value: &str,
    contract: &Value,
) -> std::result::Result<(), &'static str> {
    let signedness = contract
        .get("signedness")
        .and_then(Value::as_str)
        .unwrap_or("");
    let scale = contract.get("scale").and_then(Value::as_u64).unwrap_or(0) as usize;
    let max_integer_digits = contract
        .get("max_integer_digits")
        .and_then(Value::as_u64)
        .unwrap_or(0) as usize;
    let minimum = contract
        .get("min_value")
        .and_then(Value::as_str)
        .unwrap_or("");
    let maximum = contract
        .get("max_value")
        .and_then(Value::as_str)
        .unwrap_or("");
    let parsed = parse_fixed_decimal(value, scale, max_integer_digits, signedness)
        .map_err(|_| "invalid_decimal_grammar")?;
    let min = parse_fixed_decimal(minimum, scale, max_integer_digits, signedness)
        .map_err(|_| "invalid_numeric_contract")?;
    let max = parse_fixed_decimal(maximum, scale, max_integer_digits, signedness)
        .map_err(|_| "invalid_numeric_contract")?;
    if compare_signed_decimal(&parsed, &min) == std::cmp::Ordering::Less
        || compare_signed_decimal(&parsed, &max) == std::cmp::Ordering::Greater
    {
        return Err("range_violation");
    }
    Ok(())
}

fn parse_integer(value: &str, reject_negative_zero: bool) -> std::result::Result<String, String> {
    if value.is_empty() {
        return Err("integer must not be empty".to_owned());
    }
    let (negative, digits) = match value.strip_prefix('-') {
        Some(digits) => (true, digits),
        None => (false, value),
    };
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(format!("invalid exact integer grammar: {value}"));
    }
    if digits.len() > 1 && digits.starts_with('0') {
        return Err(format!("leading zero is forbidden: {value}"));
    }
    if negative && digits == "0" && reject_negative_zero {
        return Err("negative zero is forbidden".to_owned());
    }
    let normalized = strip_leading_zeroes(digits);
    if negative && normalized != "0" {
        Ok(format!("-{normalized}"))
    } else {
        Ok(normalized)
    }
}

fn parse_fixed_decimal(
    value: &str,
    scale: usize,
    max_integer_digits: usize,
    signedness: &str,
) -> std::result::Result<String, String> {
    if value.is_empty() {
        return Err("decimal must not be empty".to_owned());
    }
    let (negative, unsigned) = match value.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, value),
    };
    if negative && signedness == "unsigned" {
        return Err("negative decimal is forbidden for unsigned intent".to_owned());
    }
    let mut parts = unsigned.split('.');
    let integer = parts.next().unwrap_or("");
    let fractional = parts.next();
    if parts.next().is_some() {
        return Err("more than one decimal point".to_owned());
    }
    if integer.is_empty()
        || !integer.bytes().all(|byte| byte.is_ascii_digit())
        || (integer.len() > 1 && integer.starts_with('0'))
    {
        return Err("invalid decimal integer part".to_owned());
    }
    if integer.len() > max_integer_digits {
        return Err("integer part exceeds max_integer_digits".to_owned());
    }
    let fractional = fractional.unwrap_or("");
    if fractional.len() != scale || !fractional.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err("decimal scale mismatch".to_owned());
    }
    if scale == 0 && value.contains('.') {
        return Err("decimal point is forbidden at scale zero".to_owned());
    }
    let combined = format!("{integer}{fractional}");
    let normalized = strip_leading_zeroes(&combined);
    if negative && normalized == "0" {
        return Err("negative zero is forbidden".to_owned());
    }
    if negative {
        Ok(format!("-{normalized}"))
    } else {
        Ok(normalized)
    }
}

fn strip_leading_zeroes(value: &str) -> String {
    let stripped = value.trim_start_matches('0');
    if stripped.is_empty() {
        "0".to_owned()
    } else {
        stripped.to_owned()
    }
}

fn digit_count(value: &str) -> usize {
    value.strip_prefix('-').unwrap_or(value).len()
}

fn compare_signed_decimal(left: &str, right: &str) -> std::cmp::Ordering {
    let left_negative = left.starts_with('-');
    let right_negative = right.starts_with('-');
    match (left_negative, right_negative) {
        (true, false) => std::cmp::Ordering::Less,
        (false, true) => std::cmp::Ordering::Greater,
        (false, false) => compare_unsigned_decimal(left, right),
        (true, true) => compare_unsigned_decimal(
            right.strip_prefix('-').unwrap_or(right),
            left.strip_prefix('-').unwrap_or(left),
        ),
    }
}

fn compare_unsigned_decimal(left: &str, right: &str) -> std::cmp::Ordering {
    left.len().cmp(&right.len()).then_with(|| left.cmp(right))
}

fn is_valid_non_root_json_pointer(pointer: &str) -> bool {
    if !pointer.starts_with('/') {
        return false;
    }
    for segment in pointer.split('/').skip(1) {
        let bytes = segment.as_bytes();
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
    }
    true
}

fn jcs_sha256(value: &Value) -> Result<String> {
    let bytes = serde_json_canonicalizer::to_vec(value)
        .map_err(|e| ValidationError::new("jcs_canonicalization", e.to_string()))?;
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        write!(output, "{byte:02x}").expect("writing to String cannot fail");
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn profile_and_schema() -> (Vec<u8>, Vec<u8>) {
        let value_schema = json!({
            "$schema": DRAFT_2020_12,
            "$id": "urn:example:schema:test",
            "type": "object",
            "additionalProperties": false,
            "required": ["counter", "account_id", "price", "reading", "vendor_metric"],
            "properties": {
                "counter": {"type": "integer"},
                "account_id": {"type": "string"},
                "price": {"type": "string"},
                "reading": {"type": "number"},
                "vendor_metric": {"type": "number"}
            }
        });
        let schema_digest = jcs_sha256(&value_schema).unwrap();
        let mut profile = json!({
            "schema": "luminous.rea.numeric-intent-profile.v0",
            "profile_id": "test-profile",
            "profile_version": 0,
            "value_schema": {
                "schema_id": "urn:example:schema:test",
                "sha256": schema_digest
            },
            "numeric_paths": [
                {
                    "json_pointer": "/counter",
                    "intent": "safe_integer",
                    "contract": {
                        "signedness": "signed",
                        "min_value": "-9007199254740991",
                        "max_value": "9007199254740991",
                        "integer_form_only": true,
                        "safe_range_only": true
                    }
                },
                {
                    "json_pointer": "/account_id",
                    "intent": "exact_integer_string",
                    "contract": {
                        "signedness": "unsigned",
                        "max_digits": 32,
                        "min_value": "0",
                        "max_value": "18446744073709551615",
                        "leading_zero_policy": "reject",
                        "plus_sign_policy": "reject",
                        "negative_zero_policy": "reject"
                    }
                },
                {
                    "json_pointer": "/price",
                    "intent": "fixed_decimal_string",
                    "contract": {
                        "signedness": "unsigned",
                        "scale": 2,
                        "max_integer_digits": 12,
                        "min_value": "0.00",
                        "max_value": "999999999999.99",
                        "decimal_grammar": "base10_fixed_point_exact_scale",
                        "scale_policy": "exact",
                        "exponent_policy": "reject",
                        "rounding_policy": "reject_excess_scale",
                        "negative_zero_policy": "reject"
                    }
                },
                {
                    "json_pointer": "/reading",
                    "intent": "binary64",
                    "contract": {
                        "semantics": "rfc8785_ecmascript_binary64",
                        "non_finite_policy": "reject",
                        "negative_zero_policy": "jcs_canonicalizes_to_zero"
                    }
                },
                {
                    "json_pointer": "/vendor_metric",
                    "intent": "opaque_numeric",
                    "contract": {
                        "arithmetic_policy": "forbidden",
                        "ordering_policy": "forbidden",
                        "equivalence_policy": "none"
                    }
                }
            ],
            "profile_sha256": "0".repeat(64)
        });
        let digest = jcs_sha256(&profile_body_without_digest(&profile).unwrap()).unwrap();
        profile["profile_sha256"] = json!(digest);
        (
            serde_json::to_vec(&profile).unwrap(),
            serde_json::to_vec(&value_schema).unwrap(),
        )
    }

    #[test]
    fn embedded_profile_schema_is_valid_draft_2020_12() {
        let schema: Value = serde_json::from_str(PROFILE_SCHEMA_BYTES).unwrap();
        draft202012::meta::validate(&schema).unwrap();
    }

    #[test]
    fn numeric_intents_validate_without_coercing_exact_strings() {
        let (profile, schema) = profile_and_schema();
        let document = br#"{
            "counter": 42,
            "account_id": "9007199254740993",
            "price": "0.10",
            "reading": 42.0,
            "vendor_metric": 123456789012345678901234567890
        }"#;
        let report = validate_inputs(&profile, &schema, document).unwrap();
        assert_eq!(report.overall_status, "conformant");
        assert_eq!(report.path_results[0].status, "valid_safe_integer");
        assert_eq!(report.path_results[1].status, "valid_exact_integer_string");
        assert_eq!(report.path_results[2].status, "valid_fixed_decimal_string");
        assert_eq!(
            report.path_results[3].canonical_binary64.as_deref(),
            Some("42")
        );
        assert_eq!(
            report.path_results[4].status,
            "opaque_numeric_no_arithmetic_authority"
        );
        assert_eq!(
            report.path_results[4].raw_value_status,
            "opaque_no_arithmetic_exact_lexeme_not_materialized"
        );
        assert!(report.numeric_precision_warnings.iter().any(|warning| {
            warning.json_pointer == "/vendor_metric"
                && warning.integer_lexeme == "123456789012345678901234567890"
        }));
    }

    #[test]
    fn unsafe_integer_is_rejected_by_safe_integer_intent_but_not_by_jcs_identity() {
        let (profile, schema) = profile_and_schema();
        let document = br#"{
            "counter": 9007199254740992,
            "account_id": "9007199254740993",
            "price": "0.10",
            "reading": 42.0,
            "vendor_metric": 5
        }"#;
        let report = validate_inputs(&profile, &schema, document).unwrap();
        assert_eq!(report.path_results[0].status, "unsafe_integer");
        assert_eq!(report.overall_status, "nonconformant");
    }

    #[test]
    fn decimal_or_exponent_integer_spelling_is_not_a_safe_integer_token() {
        let (profile, schema) = profile_and_schema();
        let document = br#"{
            "counter": 42.0,
            "account_id": "9007199254740993",
            "price": "0.10",
            "reading": 42,
            "vendor_metric": 5
        }"#;
        let report = validate_inputs(&profile, &schema, document).unwrap();
        assert_eq!(report.path_results[0].status, "lexical_form_mismatch");
    }

    #[test]
    fn duplicate_profile_pointers_are_rejected_outside_json_schema() {
        let (profile, schema) = profile_and_schema();
        let mut value: Value = serde_json::from_slice(&profile).unwrap();
        let first = value["numeric_paths"][0].clone();
        value["numeric_paths"].as_array_mut().unwrap().push(first);
        let mut value_body = value.clone();
        value_body.as_object_mut().unwrap().remove("profile_sha256");
        value["profile_sha256"] = json!(jcs_sha256(&value_body).unwrap());
        let result = validate_inputs(
            &serde_json::to_vec(&value).unwrap(),
            &schema,
            br#"{"counter":42,"account_id":"5","price":"0.10","reading":1,"vendor_metric":2}"#,
        );
        assert_eq!(result.unwrap_err().kind, "duplicate_profile_pointer");
    }

    #[test]
    fn profile_digest_mutation_is_rejected() {
        let (profile, schema) = profile_and_schema();
        let mut value: Value = serde_json::from_slice(&profile).unwrap();
        value["profile_sha256"] = json!("0".repeat(64));
        let result = validate_inputs(
            &serde_json::to_vec(&value).unwrap(),
            &schema,
            br#"{"counter":42,"account_id":"5","price":"0.10","reading":1,"vendor_metric":2}"#,
        );
        assert_eq!(result.unwrap_err().kind, "profile_digest_mismatch");
    }

    #[test]
    fn duplicate_source_keys_are_rejected_before_value_parsing() {
        let (profile, schema) = profile_and_schema();
        let result = validate_inputs(
            &profile,
            &schema,
            br#"{"counter":42,"counter":43,"account_id":"5","price":"0.10","reading":1,"vendor_metric":2}"#,
        );
        assert_eq!(result.unwrap_err().kind, "document_raw_audit");
    }

    #[test]
    fn unresolved_unit_and_quantity_references_prevent_unqualified_conformance() {
        let (profile, schema) = profile_and_schema();
        let mut profile_value: Value = serde_json::from_slice(&profile).unwrap();
        profile_value["numeric_paths"][0]["quantity_kind_ref"] =
            json!("urn:example:quantity:length");
        profile_value["numeric_paths"][0]["unit_code_ref"] = json!("urn:example:unit:metre");
        let profile_body = profile_body_without_digest(&profile_value).unwrap();
        profile_value["profile_sha256"] = json!(jcs_sha256(&profile_body).unwrap());

        let report = validate_inputs(
            &serde_json::to_vec(&profile_value).unwrap(),
            &schema,
            br#"{"counter":42,"account_id":"5","price":"0.10","reading":1,"vendor_metric":2}"#,
        )
        .unwrap();

        assert_eq!(
            report.path_results[0]
                .unit_quantity_reference_status
                .as_deref(),
            Some("unresolved_authority_reference_not_validated")
        );
        assert_eq!(
            report.overall_status,
            "conformant_with_unresolved_authority_references"
        );
    }

    #[test]
    fn value_schema_id_must_match_the_profile_binding() {
        let (profile, schema) = profile_and_schema();
        let mut value_schema: Value = serde_json::from_slice(&schema).unwrap();
        value_schema["$id"] = json!("urn:example:schema:substituted");
        let result = validate_inputs(
            &profile,
            &serde_json::to_vec(&value_schema).unwrap(),
            br#"{"counter":42,"account_id":"5","price":"0.10","reading":1,"vendor_metric":2}"#,
        );
        assert_eq!(result.unwrap_err().kind, "value_schema_id_mismatch");
    }

    #[test]
    fn numeric_intent_design_fixture_is_well_formed_and_unique() {
        let fixture: Value =
            serde_json::from_str(include_str!("../tests/fixtures/rea-numeric-intent-v0.json"))
                .unwrap();
        let cases = fixture["cases"].as_array().unwrap();
        assert_eq!(cases.len(), 23);
        let ids: HashSet<String> = cases
            .iter()
            .map(|case| case["id"].as_str().unwrap().to_owned())
            .collect();
        assert_eq!(ids.len(), 23);
        assert!(cases.iter().all(|case| case["expected"].is_string()));
    }
}
