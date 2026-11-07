//! Lossless, bounded audit of integer-form JSON number tokens.
//!
//! This tool runs before JCS/ECMAScript binary64 conversion. It preserves the
//! exact raw decimal token spelling and RFC 6901 path. It does not rewrite
//! values, compute REA Evidence IDs, or decide whether a domain-specific
//! integer is valid.

use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::value::RawValue;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fmt;

pub const AUDIT_SCHEMA: &str = "luminous.rea.numeric-lexeme-audit.v0";
pub const MAX_INPUT_BYTES: usize = 64 * 1024 * 1024;
pub const MAX_NESTING_DEPTH: usize = 128;
pub const MAX_VISITED_VALUES: usize = 1_000_000;
pub const MAX_WARNINGS: usize = 100_000;
const MAX_SAFE_INTEGER_DECIMAL: &str = "9007199254740991";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NumericLexemeWarning {
    pub json_pointer: String,
    pub integer_lexeme: String,
    pub magnitude_class: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NumericLexemeAudit {
    pub schema: String,
    pub source_raw_sha256: String,
    pub visited_values: usize,
    pub numeric_precision_warnings: Vec<NumericLexemeWarning>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AuditLimits {
    pub max_input_bytes: usize,
    pub max_nesting_depth: usize,
    pub max_visited_values: usize,
    pub max_warnings: usize,
}

impl Default for AuditLimits {
    fn default() -> Self {
        Self {
            max_input_bytes: MAX_INPUT_BYTES,
            max_nesting_depth: MAX_NESTING_DEPTH,
            max_visited_values: MAX_VISITED_VALUES,
            max_warnings: MAX_WARNINGS,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuditErrorKind {
    InvalidJson,
    DuplicateObjectKey,
    InputTooLarge,
    NestingDepthExceeded,
    VisitedValueLimitExceeded,
    WarningLimitExceeded,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuditError {
    pub kind: AuditErrorKind,
    pub message: String,
}

impl AuditError {
    fn new(kind: AuditErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
}

impl fmt::Display for AuditError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.message)
    }
}

impl std::error::Error for AuditError {}

#[derive(Debug)]
struct WalkState<'limits> {
    limits: &'limits AuditLimits,
    visited_values: usize,
    warnings: Vec<NumericLexemeWarning>,
    deferred_error: Option<AuditError>,
}

impl WalkState<'_> {
    fn visit(&mut self, raw: &RawValue, pointer: &str, depth: usize) -> Result<(), AuditError> {
        if depth > self.limits.max_nesting_depth {
            return Err(AuditError::new(
                AuditErrorKind::NestingDepthExceeded,
                format!(
                    "JSON nesting depth {depth} exceeds limit {} at pointer '{pointer}'",
                    self.limits.max_nesting_depth
                ),
            ));
        }

        self.visited_values = self.visited_values.checked_add(1).ok_or_else(|| {
            AuditError::new(
                AuditErrorKind::VisitedValueLimitExceeded,
                "visited JSON value counter overflow",
            )
        })?;
        if self.visited_values > self.limits.max_visited_values {
            return Err(AuditError::new(
                AuditErrorKind::VisitedValueLimitExceeded,
                format!(
                    "visited JSON values exceed limit {}",
                    self.limits.max_visited_values
                ),
            ));
        }

        let token = raw.get();
        let first = token
            .as_bytes()
            .iter()
            .copied()
            .find(|byte| !matches!(byte, b' ' | b'\n' | b'\r' | b'\t'));

        match first {
            Some(b'{') => self.visit_object(token, pointer, depth),
            Some(b'[') => self.visit_array(token, pointer, depth),
            Some(b'-' | b'0'..=b'9') => self.visit_number(token, pointer),
            Some(b'"' | b't' | b'f' | b'n') => Ok(()),
            _ => Err(AuditError::new(
                AuditErrorKind::InvalidJson,
                format!("unexpected JSON token at pointer '{pointer}'"),
            )),
        }
    }

    fn visit_object(&mut self, token: &str, pointer: &str, depth: usize) -> Result<(), AuditError> {
        let mut deserializer = serde_json::Deserializer::from_str(token);
        let result = deserializer.deserialize_map(ObjectVisitor {
            state: self,
            pointer: pointer.to_owned(),
            depth,
        });

        if let Err(error) = result {
            return Err(self.deferred_error.take().unwrap_or_else(|| {
                AuditError::new(AuditErrorKind::InvalidJson, error.to_string())
            }));
        }
        deserializer
            .end()
            .map_err(|error| AuditError::new(AuditErrorKind::InvalidJson, error.to_string()))
    }

    fn visit_array(&mut self, token: &str, pointer: &str, depth: usize) -> Result<(), AuditError> {
        let mut deserializer = serde_json::Deserializer::from_str(token);
        let result = deserializer.deserialize_seq(ArrayVisitor {
            state: self,
            pointer: pointer.to_owned(),
            depth,
        });

        if let Err(error) = result {
            return Err(self.deferred_error.take().unwrap_or_else(|| {
                AuditError::new(AuditErrorKind::InvalidJson, error.to_string())
            }));
        }
        deserializer
            .end()
            .map_err(|error| AuditError::new(AuditErrorKind::InvalidJson, error.to_string()))
    }

    fn visit_number(&mut self, token: &str, pointer: &str) -> Result<(), AuditError> {
        // Decimal and exponent tokens remain JCS floating-point cases, not
        // lexical integer tokens.
        if token.contains('.') || token.contains('e') || token.contains('E') {
            return Ok(());
        }

        let digits = token.strip_prefix('-').unwrap_or(token);
        if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(AuditError::new(
                AuditErrorKind::InvalidJson,
                format!("invalid integer token at pointer '{pointer}'"),
            ));
        }

        let outside_safe_range = digits.len() > MAX_SAFE_INTEGER_DECIMAL.len()
            || (digits.len() == MAX_SAFE_INTEGER_DECIMAL.len()
                && digits > MAX_SAFE_INTEGER_DECIMAL);

        if outside_safe_range {
            if self.warnings.len() >= self.limits.max_warnings {
                return Err(AuditError::new(
                    AuditErrorKind::WarningLimitExceeded,
                    format!(
                        "numeric precision warnings exceed limit {}",
                        self.limits.max_warnings
                    ),
                ));
            }
            self.warnings.push(NumericLexemeWarning {
                json_pointer: pointer.to_owned(),
                integer_lexeme: token.to_owned(),
                magnitude_class: "outside_ecmascript_safe_integer".to_owned(),
            });
        }
        Ok(())
    }
}

struct ObjectVisitor<'state, 'limits> {
    state: &'state mut WalkState<'limits>,
    pointer: String,
    depth: usize,
}

impl<'de> Visitor<'de> for ObjectVisitor<'_, '_> {
    type Value = ();

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON object with unique member names")
    }

    fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
    where
        M: MapAccess<'de>,
    {
        let mut seen = HashSet::new();
        while let Some(key) = map.next_key::<String>()? {
            if !seen.insert(key.clone()) {
                let pointer = join_pointer(&self.pointer, &key);
                let error = AuditError::new(
                    AuditErrorKind::DuplicateObjectKey,
                    format!("duplicate JSON object key at pointer '{pointer}'"),
                );
                self.state.deferred_error = Some(error);
                return Err(de::Error::custom("duplicate JSON object key"));
            }

            let child = map.next_value::<&RawValue>()?;
            let child_pointer = join_pointer(&self.pointer, &key);
            if let Err(error) = self.state.visit(child, &child_pointer, self.depth + 1)
            {
                self.state.deferred_error = Some(error.clone());
                return Err(de::Error::custom(error));
            }
        }
        Ok(())
    }
}

struct ArrayVisitor<'state, 'limits> {
    state: &'state mut WalkState<'limits>,
    pointer: String,
    depth: usize,
}

impl<'de> Visitor<'de> for ArrayVisitor<'_, '_> {
    type Value = ();

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON array")
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut index = 0usize;
        while let Some(child) = sequence.next_element::<&RawValue>()? {
            let child_pointer = join_pointer(&self.pointer, &index.to_string());
            if let Err(error) = self.state.visit(child, &child_pointer, self.depth + 1)
            {
                self.state.deferred_error = Some(error.clone());
                return Err(de::Error::custom(error));
            }
            index = index
                .checked_add(1)
                .ok_or_else(|| de::Error::custom("JSON array index overflow"))?;
        }
        Ok(())
    }
}

fn join_pointer(parent: &str, token: &str) -> String {
    let escaped = token.replace('~', "~0").replace('/', "~1");
    format!("{parent}/{escaped}")
}

fn sha256_hex(input: &[u8]) -> String {
    let digest = Sha256::digest(input);
    let mut output = String::with_capacity(64);
    for byte in digest {
        use fmt::Write as _;
        let _ = write!(output, "{byte:02x}");
    }
    output
}

/// Audit raw JSON bytes before conversion to serde_json::Value or JCS binary64.
pub fn audit_json(input: &[u8]) -> Result<NumericLexemeAudit, AuditError> {
    audit_json_with_limits(input, &AuditLimits::default())
}

/// Testable bounded variant. Production callers should use audit_json unless
/// a stricter limit profile is explicitly part of their protocol.
pub fn audit_json_with_limits(
    input: &[u8],
    limits: &AuditLimits,
) -> Result<NumericLexemeAudit, AuditError> {
    if input.len() > limits.max_input_bytes {
        return Err(AuditError::new(
            AuditErrorKind::InputTooLarge,
            format!(
                "JSON input is {} bytes; limit is {}",
                input.len(),
                limits.max_input_bytes
            ),
        ));
    }

    let mut deserializer = serde_json::Deserializer::from_slice(input);
    let raw = <&RawValue>::deserialize(&mut deserializer)
        .map_err(|error| AuditError::new(AuditErrorKind::InvalidJson, error.to_string()))?;
    deserializer
        .end()
        .map_err(|error| AuditError::new(AuditErrorKind::InvalidJson, error.to_string()))?;

    let mut state = WalkState {
        limits,
        visited_values: 0,
        warnings: Vec::new(),
        deferred_error: None,
    };
    state.visit(raw, "", 0)?;
    state.warnings.sort_by(|left, right| {
        left.json_pointer
            .cmp(&right.json_pointer)
            .then_with(|| left.integer_lexeme.cmp(&right.integer_lexeme))
    });

    Ok(NumericLexemeAudit {
        schema: AUDIT_SCHEMA.to_owned(),
        source_raw_sha256: sha256_hex(input),
        visited_values: state.visited_values,
        numeric_precision_warnings: state.warnings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn warnings(input: &[u8]) -> Vec<NumericLexemeWarning> {
        audit_json(input).unwrap().numeric_precision_warnings
    }

    fn warning(pointer: &str, lexeme: &str) -> NumericLexemeWarning {
        NumericLexemeWarning {
            json_pointer: pointer.to_owned(),
            integer_lexeme: lexeme.to_owned(),
            magnitude_class: "outside_ecmascript_safe_integer".to_owned(),
        }
    }

    #[test]
    fn safe_integer_boundary_is_inclusive_and_first_unsafe_values_are_reported() {
        let input = br#"{
            "safe_pos": 9007199254740991,
            "safe_neg": -9007199254740991,
            "unsafe_pos": 9007199254740992,
            "unsafe_neg": -9007199254740992
        }"#;
        assert_eq!(
            warnings(input),
            vec![
                warning("/unsafe_neg", "-9007199254740992"),
                warning("/unsafe_pos", "9007199254740992"),
            ]
        );
    }

    #[test]
    fn preserves_u64_and_arbitrarily_long_integer_lexemes_exactly() {
        let huge = "9".repeat(120);
        let input =
            format!(r#"{{"u64_max":18446744073709551615,"huge":{huge},"negative_huge":-{huge}}}"#);
        let actual = audit_json(input.as_bytes()).unwrap();
        assert_eq!(
            actual.numeric_precision_warnings,
            vec![
                warning("/huge", &huge),
                warning("/negative_huge", &format!("-{huge}")),
                warning("/u64_max", "18446744073709551615"),
            ]
        );
    }

    #[test]
    fn nested_paths_escape_slash_and_tilde_and_preserve_array_indices() {
        let input = br#"{"outer/key":{"~id":[0,9007199254740992]}}"#;
        assert_eq!(
            warnings(input),
            vec![warning("/outer~1key/~0id/1", "9007199254740992")]
        );
    }

    #[test]
    fn decimal_exponent_and_string_controls_do_not_produce_integer_warnings() {
        let input = br#"{
            "decimal": 1.0,
            "exponent": 1e0,
            "tiny": 5e-324,
            "large_exp": 1e+30,
            "small_exp": 1e-27,
            "string": "9007199254740993",
            "text": "integer 18446744073709551615"
        }"#;
        assert!(warnings(input).is_empty());
    }

    #[test]
    fn duplicate_keys_are_rejected_recursively_after_escape_decoding() {
        let input = br#"{"outer":{"a":1,"\u0061":9007199254740992}}"#;
        let error = audit_json(input).unwrap_err();
        assert_eq!(error.kind, AuditErrorKind::DuplicateObjectKey);
    }

    #[test]
    fn malformed_and_trailing_json_are_rejected() {
        assert_eq!(
            audit_json(br#"{"n":01}"#).unwrap_err().kind,
            AuditErrorKind::InvalidJson
        );
        assert_eq!(
            audit_json(br#"{"n":1} {"n":2}"#).unwrap_err().kind,
            AuditErrorKind::InvalidJson
        );
        assert_eq!(
            audit_json(br#"{"n":NaN}"#).unwrap_err().kind,
            AuditErrorKind::InvalidJson
        );
    }

    #[test]
    fn root_integer_uses_the_empty_json_pointer() {
        assert_eq!(
            warnings(b"9007199254740992"),
            vec![warning("", "9007199254740992")]
        );
    }

    #[test]
    fn warning_order_is_stable_across_object_key_permutations() {
        let first = audit_json(br#"{"z":9007199254740993,"a":9007199254740992}"#).unwrap();
        let second = audit_json(br#"{"a":9007199254740992,"z":9007199254740993}"#).unwrap();

        assert_eq!(
            first.numeric_precision_warnings,
            second.numeric_precision_warnings
        );
        assert_ne!(first.source_raw_sha256, second.source_raw_sha256);
    }

    #[test]
    fn size_depth_value_and_warning_limits_fail_closed() {
        let small_input_limit = AuditLimits {
            max_input_bytes: 4,
            ..AuditLimits::default()
        };
        assert_eq!(
            audit_json_with_limits(b"12345", &small_input_limit)
                .unwrap_err()
                .kind,
            AuditErrorKind::InputTooLarge
        );

        let depth_limit = AuditLimits {
            max_nesting_depth: 1,
            ..AuditLimits::default()
        };
        assert_eq!(
            audit_json_with_limits(br#"{"a":{"b":1}}"#, &depth_limit)
                .unwrap_err()
                .kind,
            AuditErrorKind::NestingDepthExceeded
        );

        let value_limit = AuditLimits {
            max_visited_values: 2,
            ..AuditLimits::default()
        };
        assert_eq!(
            audit_json_with_limits(br#"{"a":1,"b":2}"#, &value_limit)
                .unwrap_err()
                .kind,
            AuditErrorKind::VisitedValueLimitExceeded
        );

        let warning_limit = AuditLimits {
            max_warnings: 1,
            ..AuditLimits::default()
        };
        assert_eq!(
            audit_json_with_limits(br#"[9007199254740992,9007199254740993]"#, &warning_limit)
                .unwrap_err()
                .kind,
            AuditErrorKind::WarningLimitExceeded
        );
    }
}
