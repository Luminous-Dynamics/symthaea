//! Bounded JSON field selection that preserves the exact raw JSON value byte span.
//!
//! This is a host-side parser/range-selection primitive for the PPAC/TLSN field
//! semantics experiment. It is not a TLSNotary verifier, transcript-opening
//! verifier, or zero-knowledge parser.
//!
//! The returned span is relative to the exact JSON body slice passed by the
//! caller. A separate qualified HTTP framing layer must establish whether that
//! body slice maps contiguously to a particular TLS transcript range.

pub mod http11;

use std::{
    collections::HashSet,
    fmt,
    ops::Range,
};

use serde::{
    Deserialize,
    de::{MapAccess, SeqAccess, Visitor},
};
use serde_json::value::RawValue;

/// Stable identifier for this parser profile.
pub const FIELD_BINDING_PARSER_PROFILE_V1: &str = "symthaea-json-raw-pointer-v1";

/// Resource limits applied before accepting a JSON field selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FieldSelectorLimits {
    /// Maximum exact JSON body size in bytes.
    pub max_document_bytes: usize,
    /// Maximum nesting depth. The root JSON value has depth zero.
    pub max_depth: usize,
    /// Maximum number of JSON values visited during ambiguity/resource audit.
    pub max_nodes: usize,
    /// Maximum members in any one object or array.
    pub max_container_items: usize,
    /// Maximum decoded UTF-8 key length in bytes.
    pub max_key_bytes: usize,
    /// Maximum raw scalar token length in bytes.
    pub max_scalar_bytes: usize,
}

impl Default for FieldSelectorLimits {
    fn default() -> Self {
        Self {
            max_document_bytes: 1 << 20,
            max_depth: 32,
            max_nodes: 100_000,
            max_container_items: 10_000,
            max_key_bytes: 4_096,
            max_scalar_bytes: 256 * 1024,
        }
    }
}

/// JSON value kind at the selected JSON Pointer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JsonValueKind {
    String,
    Number,
    Boolean,
    Null,
    Object,
    Array,
}

/// Result of selecting a field from one exact JSON body.
///
/// The body range covers the raw JSON value token, including quotes for
/// strings. It is a body-relative byte range, not a TLS transcript range.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JsonFieldSelectionV1 {
    /// Parser profile identifier.
    pub parser_profile_id: &'static str,
    /// Exact RFC 6901 JSON Pointer supplied by the verifier-side request.
    pub json_pointer: String,
    /// Kind of the selected JSON value.
    pub value_kind: JsonValueKind,
    /// Exact raw token byte range within the supplied JSON body.
    pub body_range: Range<usize>,
}

/// Errors from bounded JSON Pointer field selection.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum FieldSelectorError {
    #[error("JSON document exceeds the configured byte limit")]
    DocumentTooLarge,
    #[error("invalid JSON document")]
    InvalidJson,
    #[error("JSON Pointer must be empty or begin with '/'")]
    InvalidPointerSyntax,
    #[error("JSON Pointer contains an invalid escape")]
    InvalidPointerEscape,
    #[error("duplicate JSON object key is not admitted by this parser profile")]
    DuplicateKey,
    #[error("JSON object key exceeds the configured byte limit")]
    KeyTooLarge,
    #[error("JSON scalar token exceeds the configured byte limit")]
    ScalarTooLarge,
    #[error("JSON object or array exceeds the configured item limit")]
    ContainerTooLarge,
    #[error("JSON nesting depth exceeds the configured limit")]
    DepthLimitExceeded,
    #[error("JSON value count exceeds the configured node limit")]
    NodeLimitExceeded,
    #[error("JSON Pointer member does not exist")]
    MissingMember,
    #[error("JSON Pointer traverses a scalar value")]
    CannotTraverseScalar,
    #[error("JSON Pointer array index is not canonical")]
    InvalidArrayIndex,
    #[error("JSON Pointer array index is outside the selected array")]
    ArrayIndexOutOfBounds,
    #[error("selected raw value is not a span within the supplied document")]
    InvalidRawSpan,
    #[error("byte offset overflow")]
    OffsetOverflow,
    #[error("the disclosure view contains no undisclosed byte range")]
    MissingHiddenRange,
    #[error("the disclosure view contains multiple undisclosed byte ranges")]
    MultipleHiddenRanges,
    #[error("the one undisclosed range is empty or exceeds the selected scalar bound")]
    InvalidHiddenRange,
    #[error("the selected JSON field is not a string")]
    SelectedValueNotString,
    #[error("the undisclosed range is not exactly the selected JSON string contents")]
    HiddenRangeNotSelectedField,
}

#[derive(Debug)]
struct RawObject<'a> {
    entries: Vec<(String, &'a RawValue)>,
}

impl<'de> Deserialize<'de> for RawObject<'de> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct RawObjectVisitor;

        impl<'de> Visitor<'de> for RawObjectVisitor {
            type Value = RawObject<'de>;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a JSON object with borrowed raw member values")
            }

            fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
            where
                M: MapAccess<'de>,
            {
                let mut entries = Vec::new();
                while let Some((key, value)) = map.next_entry::<String, &'de RawValue>()? {
                    entries.push((key, value));
                }
                Ok(RawObject { entries })
            }
        }

        deserializer.deserialize_map(RawObjectVisitor)
    }
}

#[derive(Debug)]
struct RawArray<'a> {
    values: Vec<&'a RawValue>,
}

impl<'de> Deserialize<'de> for RawArray<'de> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct RawArrayVisitor;

        impl<'de> Visitor<'de> for RawArrayVisitor {
            type Value = RawArray<'de>;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a JSON array with borrowed raw values")
            }

            fn visit_seq<S>(self, mut seq: S) -> Result<Self::Value, S::Error>
            where
                S: SeqAccess<'de>,
            {
                let mut values = Vec::new();
                while let Some(value) = seq.next_element::<&'de RawValue>()? {
                    values.push(value);
                }
                Ok(RawArray { values })
            }
        }

        deserializer.deserialize_seq(RawArrayVisitor)
    }
}

/// Select an exact raw JSON value token by RFC 6901 JSON Pointer.
///
/// This function parses exactly one JSON body using serde_json, rejects
/// duplicate decoded object member names anywhere in the document, enforces
/// resource bounds, and preserves the selected token's exact byte span.
///
/// The caller must supply the exact JSON body bytes. HTTP transfer decoding,
/// content decompression, charset conversion, and mapping the body back to TLS
/// transcript offsets are separate evidence/derivation steps and are not
/// performed here.
pub fn select_json_field_v1(
    document: &[u8],
    json_pointer: &str,
    limits: FieldSelectorLimits,
) -> Result<JsonFieldSelectionV1, FieldSelectorError> {
    if document.len() > limits.max_document_bytes {
        return Err(FieldSelectorError::DocumentTooLarge);
    }

    let root: &RawValue =
        serde_json::from_slice(document).map_err(|_| FieldSelectorError::InvalidJson)?;

    // Fail closed on duplicate keys anywhere in the document. This avoids
    // first-wins/last-wins parser differentials, including escaped spellings
    // that decode to the same Unicode string.
    let mut audit = AuditState { nodes: 0, limits };
    audit_value(root, 0, &mut audit)?;

    let segments = parse_json_pointer(json_pointer)?;
    let mut selected = root;

    for segment in segments {
        match value_kind(selected)? {
            JsonValueKind::Object => {
                let object: RawObject<'_> = serde_json::from_str(selected.get())
                    .map_err(|_| FieldSelectorError::InvalidJson)?;
                selected = object
                    .entries
                    .iter()
                    .find(|(key, _)| key == &segment)
                    .map(|(_, value)| *value)
                    .ok_or(FieldSelectorError::MissingMember)?;
            }
            JsonValueKind::Array => {
                let index = parse_array_index(&segment)?;
                let array: RawArray<'_> = serde_json::from_str(selected.get())
                    .map_err(|_| FieldSelectorError::InvalidJson)?;
                selected = *array
                    .values
                    .get(index)
                    .ok_or(FieldSelectorError::ArrayIndexOutOfBounds)?;
            }
            JsonValueKind::String
            | JsonValueKind::Number
            | JsonValueKind::Boolean
            | JsonValueKind::Null => return Err(FieldSelectorError::CannotTraverseScalar),
        }
    }

    let body_range = borrowed_raw_span(document, selected.get())?;
    let selected_kind = value_kind(selected)?;

    Ok(JsonFieldSelectionV1 {
        parser_profile_id: FIELD_BINDING_PARSER_PROFILE_V1,
        json_pointer: json_pointer.to_owned(),
        value_kind: selected_kind,
        body_range,
    })
}

/// Result of binding one hidden string's contents to a verifier-selected JSON Pointer.
/// Both ranges are relative to the JSON body. body_range includes the quote bytes;
/// hidden_content_range excludes those quotes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HiddenJsonFieldBindingV1 {
    pub parser_profile_id: &'static str,
    pub json_pointer: String,
    pub body_range: Range<usize>,
    pub hidden_content_range: Range<usize>,
}

/// Bind the one undisclosed byte run to the string value selected by json_pointer.
///
/// Some(byte) means disclosed (the caller must already authenticate it); None
/// means undisclosed. Exactly one non-empty contiguous hidden run is permitted.
/// The function removes that run, parses the structural JSON, and requires the
/// selected pointer to resolve to the resulting empty-string token.
///
/// This establishes a parser-profile/path relation only. It does not authenticate
/// disclosed bytes or map body-relative offsets into TLS transcript offsets.
pub fn bind_single_hidden_string_field_v1(
    partial_document: &[Option<u8>],
    json_pointer: &str,
    limits: FieldSelectorLimits,
) -> Result<HiddenJsonFieldBindingV1, FieldSelectorError> {
    if partial_document.len() > limits.max_document_bytes {
        return Err(FieldSelectorError::DocumentTooLarge);
    }

    let mut hidden_start: Option<usize> = None;
    let mut hidden_end = 0usize;
    let mut in_hidden = false;
    let mut hidden_runs = 0usize;
    let mut skeleton = Vec::with_capacity(partial_document.len());

    for (index, byte) in partial_document.iter().enumerate() {
        match byte {
            Some(value) => {
                in_hidden = false;
                skeleton.push(*value);
            }
            None => {
                if !in_hidden {
                    hidden_runs = hidden_runs
                        .checked_add(1)
                        .ok_or(FieldSelectorError::MultipleHiddenRanges)?;
                    if hidden_start.is_none() {
                        hidden_start = Some(index);
                    }
                    in_hidden = true;
                }
                hidden_end = index + 1;
            }
        }
    }

    if hidden_runs == 0 {
        return Err(FieldSelectorError::MissingHiddenRange);
    }
    if hidden_runs != 1 {
        return Err(FieldSelectorError::MultipleHiddenRanges);
    }

    let hidden_start = hidden_start.ok_or(FieldSelectorError::MissingHiddenRange)?;
    let hidden_range = hidden_start..hidden_end;
    if hidden_range.is_empty() || hidden_range.len() > limits.max_scalar_bytes {
        return Err(FieldSelectorError::InvalidHiddenRange);
    }

    let selection = select_json_field_v1(&skeleton, json_pointer, limits)?;
    if selection.value_kind != JsonValueKind::String {
        return Err(FieldSelectorError::SelectedValueNotString);
    }

    let selected_token = skeleton
        .get(selection.body_range.clone())
        .ok_or(FieldSelectorError::InvalidRawSpan)?;
    if selected_token != b"\"\"" {
        return Err(FieldSelectorError::HiddenRangeNotSelectedField);
    }

    let expected_hidden_start = selection.body_range.start
        .checked_add(1)
        .ok_or(FieldSelectorError::OffsetOverflow)?;
    let expected_hidden_end = selection.body_range.end
        .checked_add(hidden_range.len())
        .and_then(|end| end.checked_sub(1))
        .ok_or(FieldSelectorError::OffsetOverflow)?;
    if hidden_range.start != expected_hidden_start || hidden_range.end != expected_hidden_end {
        return Err(FieldSelectorError::HiddenRangeNotSelectedField);
    }

    let original_end = selection.body_range.end
        .checked_add(hidden_range.len())
        .ok_or(FieldSelectorError::OffsetOverflow)?;
    Ok(HiddenJsonFieldBindingV1 {
        parser_profile_id: FIELD_BINDING_PARSER_PROFILE_V1,
        json_pointer: json_pointer.to_owned(),
        body_range: selection.body_range.start..original_end,
        hidden_content_range: hidden_range,
    })
}

struct AuditState {
    nodes: usize,
    limits: FieldSelectorLimits,
}

fn audit_value(
    value: &RawValue,
    depth: usize,
    state: &mut AuditState,
) -> Result<(), FieldSelectorError> {
    if depth > state.limits.max_depth {
        return Err(FieldSelectorError::DepthLimitExceeded);
    }

    state.nodes = state
        .nodes
        .checked_add(1)
        .ok_or(FieldSelectorError::NodeLimitExceeded)?;
    if state.nodes > state.limits.max_nodes {
        return Err(FieldSelectorError::NodeLimitExceeded);
    }

    match value_kind(value)? {
        JsonValueKind::Object => {
            let object: RawObject<'_> = serde_json::from_str(value.get())
                .map_err(|_| FieldSelectorError::InvalidJson)?;
            if object.entries.len() > state.limits.max_container_items {
                return Err(FieldSelectorError::ContainerTooLarge);
            }

            let mut keys = HashSet::with_capacity(object.entries.len());
            for (key, child) in object.entries {
                if key.len() > state.limits.max_key_bytes {
                    return Err(FieldSelectorError::KeyTooLarge);
                }
                if !keys.insert(key) {
                    return Err(FieldSelectorError::DuplicateKey);
                }
                audit_value(child, depth + 1, state)?;
            }
        }
        JsonValueKind::Array => {
            let array: RawArray<'_> = serde_json::from_str(value.get())
                .map_err(|_| FieldSelectorError::InvalidJson)?;
            if array.values.len() > state.limits.max_container_items {
                return Err(FieldSelectorError::ContainerTooLarge);
            }
            for child in array.values {
                audit_value(child, depth + 1, state)?;
            }
        }
        JsonValueKind::String
        | JsonValueKind::Number
        | JsonValueKind::Boolean
        | JsonValueKind::Null => {
            if value.get().len() > state.limits.max_scalar_bytes {
                return Err(FieldSelectorError::ScalarTooLarge);
            }
        }
    }

    Ok(())
}

fn value_kind(value: &RawValue) -> Result<JsonValueKind, FieldSelectorError> {
    let trimmed = value
        .get()
        .trim_start_matches(|ch| matches!(ch, ' ' | '\n' | '\r' | '\t'));
    let first = trimmed
        .as_bytes()
        .first()
        .copied()
        .ok_or(FieldSelectorError::InvalidJson)?;

    match first {
        b'{' => Ok(JsonValueKind::Object),
        b'[' => Ok(JsonValueKind::Array),
        b'"' => Ok(JsonValueKind::String),
        b't' | b'f' => Ok(JsonValueKind::Boolean),
        b'n' => Ok(JsonValueKind::Null),
        b'-' | b'0'..=b'9' => Ok(JsonValueKind::Number),
        _ => Err(FieldSelectorError::InvalidJson),
    }
}

fn parse_json_pointer(pointer: &str) -> Result<Vec<String>, FieldSelectorError> {
    if pointer.is_empty() {
        return Ok(Vec::new());
    }
    let Some(rest) = pointer.strip_prefix('/') else {
        return Err(FieldSelectorError::InvalidPointerSyntax);
    };

    rest.split('/').map(decode_pointer_segment).collect()
}

fn decode_pointer_segment(segment: &str) -> Result<String, FieldSelectorError> {
    let mut decoded = String::with_capacity(segment.len());
    let mut chars = segment.chars();

    while let Some(ch) = chars.next() {
        if ch != '~' {
            decoded.push(ch);
            continue;
        }

        match chars.next() {
            Some('0') => decoded.push('~'),
            Some('1') => decoded.push('/'),
            _ => return Err(FieldSelectorError::InvalidPointerEscape),
        }
    }

    Ok(decoded)
}

fn parse_array_index(segment: &str) -> Result<usize, FieldSelectorError> {
    if segment.is_empty()
        || (segment.len() > 1 && segment.starts_with('0'))
        || !segment.bytes().all(|byte| byte.is_ascii_digit())
    {
        return Err(FieldSelectorError::InvalidArrayIndex);
    }

    segment
        .parse::<usize>()
        .map_err(|_| FieldSelectorError::InvalidArrayIndex)
}

fn borrowed_raw_span(document: &[u8], raw_value: &str) -> Result<Range<usize>, FieldSelectorError> {
    let base = document.as_ptr() as usize;
    let selected = raw_value.as_ptr() as usize;
    let start = selected
        .checked_sub(base)
        .ok_or(FieldSelectorError::InvalidRawSpan)?;
    let end = start
        .checked_add(raw_value.len())
        .ok_or(FieldSelectorError::OffsetOverflow)?;
    if end > document.len() {
        return Err(FieldSelectorError::InvalidRawSpan);
    }

    Ok(start..end)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn select<'a>(document: &'a [u8], pointer: &str) -> JsonFieldSelectionV1 {
        select_json_field_v1(document, pointer, FieldSelectorLimits::default()).unwrap()
    }

    fn selected_bytes<'a>(document: &'a [u8], selection: &JsonFieldSelectionV1) -> &'a [u8] {
        &document[selection.body_range.clone()]
    }

    #[test]
    fn selects_exact_nested_value_span() {
        let document = br#"{"taxpayer":{"id":"opaque","date_of_birth":"1985-03-12"},"other_date":"2020-01-01"}"#;
        let selection = select(document, "/taxpayer/date_of_birth");

        assert_eq!(selection.value_kind, JsonValueKind::String);
        assert_eq!(selected_bytes(document, &selection), br#""1985-03-12""#);
        assert_eq!(selection.parser_profile_id, FIELD_BINDING_PARSER_PROFILE_V1);
    }

    #[test]
    fn selects_requested_field_not_similar_sibling() {
        let document = br#"{"date_of_birth":"1985-03-12","beneficiary_date_of_birth":"2010-01-01"}"#;
        let dob = select(document, "/date_of_birth");
        let beneficiary_dob = select(document, "/beneficiary_date_of_birth");

        assert_eq!(selected_bytes(document, &dob), br#""1985-03-12""#);
        assert_eq!(selected_bytes(document, &beneficiary_dob), br#""2010-01-01""#);
        assert_ne!(dob.body_range, beneficiary_dob.body_range);
    }

    #[test]
    fn rejects_duplicate_keys() {
        let document = br#"{"a":1,"a":2}"#;
        assert_eq!(
            select_json_field_v1(document, "/a", FieldSelectorLimits::default()),
            Err(FieldSelectorError::DuplicateKey)
        );
    }

    #[test]
    fn rejects_escaped_duplicate_keys_after_json_decoding() {
        let document = br#"{"dob":"1985-03-12","d\u006fb":"2010-01-01"}"#;
        assert_eq!(
            select_json_field_v1(document, "/dob", FieldSelectorLimits::default()),
            Err(FieldSelectorError::DuplicateKey)
        );
    }

    #[test]
    fn supports_rfc6901_slash_and_tilde_escapes() {
        let document = br#"{"a/b":{"~dob":"1985-03-12"}}"#;
        let selection = select(document, "/a~1b/~0dob");
        assert_eq!(selected_bytes(document, &selection), br#""1985-03-12""#);
    }

    #[test]
    fn selects_array_element_with_canonical_index() {
        let document = br#"{"people":[{"dob":"2000-01-01"},{"dob":"1985-03-12"}]}"#;
        let selection = select(document, "/people/1/dob");
        assert_eq!(selected_bytes(document, &selection), br#""1985-03-12""#);
    }

    #[test]
    fn rejects_noncanonical_array_index() {
        let document = br#"{"people":[{"dob":"2000-01-01"}]}"#;
        assert_eq!(
            select_json_field_v1(document, "/people/00/dob", FieldSelectorLimits::default()),
            Err(FieldSelectorError::InvalidArrayIndex)
        );
    }

    #[test]
    fn preserves_raw_number_lexeme() {
        let document = br#"{"amount":1.00,"scientific":1e2}"#;
        let amount = select(document, "/amount");
        let scientific = select(document, "/scientific");
        assert_eq!(selected_bytes(document, &amount), b"1.00");
        assert_eq!(selected_bytes(document, &scientific), b"1e2");
    }

    #[test]
    fn rejects_malformed_pointer_escape() {
        let document = br#"{"a":1}"#;
        assert_eq!(
            select_json_field_v1(document, "/~2", FieldSelectorLimits::default()),
            Err(FieldSelectorError::InvalidPointerEscape)
        );
    }

    #[test]
    fn rejects_trailing_non_json_bytes() {
        let document = br#"{"a":1} not-json"#;
        assert_eq!(
            select_json_field_v1(document, "/a", FieldSelectorLimits::default()),
            Err(FieldSelectorError::InvalidJson)
        );
    }


    #[test]
    fn preserves_exact_unicode_codepoint_matching_without_normalization() {
        let document = "{\"é\":\"composed\",\"é\":\"decomposed\"}".as_bytes();
        let composed = select(document, "/é");
        let decomposed = select(document, "/é");

        assert_eq!(selected_bytes(document, &composed), br#""composed""#);
        assert_eq!(selected_bytes(document, &decomposed), br#""decomposed""#);
        assert_ne!(composed.body_range, decomposed.body_range);
    }

    #[test]
    fn binds_one_hidden_string_to_exact_json_pointer() {
        let document = br#"{\"taxpayer\":{\"date_of_birth\":\"1985-03-12\"},\"other_date\":\"2020-01-01\"}"#;
        let selection = select(document, "/taxpayer/date_of_birth");
        let hidden = (selection.body_range.start + 1)..(selection.body_range.end - 1);
        let partial = document.iter().enumerate().map(|(i, b)| if hidden.contains(&i) { None } else { Some(*b) }).collect::<Vec<_>>();
        let binding = bind_single_hidden_string_field_v1(&partial, "/taxpayer/date_of_birth", FieldSelectorLimits::default()).unwrap();
        assert_eq!(binding.body_range, selection.body_range);
        assert_eq!(binding.hidden_content_range, hidden);
    }

    #[test]
    fn rejects_hidden_value_at_wrong_json_pointer() {
        let document = br#"{\"date_of_birth\":\"2010-01-01\",\"beneficiary_date_of_birth\":\"1985-03-12\"}"#;
        let selection = select(document, "/beneficiary_date_of_birth");
        let hidden = (selection.body_range.start + 1)..(selection.body_range.end - 1);
        let partial = document.iter().enumerate().map(|(i, b)| if hidden.contains(&i) { None } else { Some(*b) }).collect::<Vec<_>>();
        assert_eq!(bind_single_hidden_string_field_v1(&partial, "/date_of_birth", FieldSelectorLimits::default()), Err(FieldSelectorError::HiddenRangeNotSelectedField));
    }

    #[test]
    fn rejects_multiple_hidden_ranges() {
        let document = br#"{\"dob\":\"1985-03-12\",\"name\":\"Alice\"}"#;
        let partial = document.iter().enumerate().map(|(i, b)| if (8..12).contains(&i) || (25..28).contains(&i) { None } else { Some(*b) }).collect::<Vec<_>>();
        assert_eq!(bind_single_hidden_string_field_v1(&partial, "/dob", FieldSelectorLimits::default()), Err(FieldSelectorError::MultipleHiddenRanges));
    }

    #[test]
    fn rejects_no_hidden_range() {
        let document = br#"{\"dob\":\"1985-03-12\"}"#;
        let partial = document.iter().copied().map(Some).collect::<Vec<_>>();
        assert_eq!(bind_single_hidden_string_field_v1(&partial, "/dob", FieldSelectorLimits::default()), Err(FieldSelectorError::MissingHiddenRange));
    }

    #[test]
    fn rejects_hidden_range_that_includes_string_quotes() {
        let document = br#"{\"dob\":\"1985-03-12\"}"#;
        let selection = select(document, "/dob");
        let partial = document.iter().enumerate().map(|(i, b)| if selection.body_range.contains(&i) { None } else { Some(*b) }).collect::<Vec<_>>();
        assert!(bind_single_hidden_string_field_v1(&partial, "/dob", FieldSelectorLimits::default()).is_err());
    }

    #[test]
    fn hidden_string_is_not_confused_with_other_value_kind() {
        let document = br#"{\"dob\":\"1985-03-12\",\"age\":40}"#;
        let selection = select(document, "/dob");
        let hidden = (selection.body_range.start + 1)..(selection.body_range.end - 1);
        let partial = document.iter().enumerate().map(|(i, b)| if hidden.contains(&i) { None } else { Some(*b) }).collect::<Vec<_>>();
        assert_eq!(bind_single_hidden_string_field_v1(&partial, "/age", FieldSelectorLimits::default()), Err(FieldSelectorError::SelectedValueNotString));
    }

    #[test]
    fn enforces_document_size_bound() {
        let document = br#"{"a":"1234567890"}"#;
        let limits = FieldSelectorLimits { max_document_bytes: 4, ..FieldSelectorLimits::default() };
        assert_eq!(
            select_json_field_v1(document, "/a", limits),
            Err(FieldSelectorError::DocumentTooLarge)
        );
    }

    #[test]
    fn rejects_duplicate_keys_in_unselected_nested_subtrees() {
        let document = br#"{"target":"1985-03-12","unrelated":{"x":1,"x":2}}"#;
        assert_eq!(
            select_json_field_v1(document, "/target", FieldSelectorLimits::default()),
            Err(FieldSelectorError::DuplicateKey)
        );
    }
}
