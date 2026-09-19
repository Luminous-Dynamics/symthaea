use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};

const INPUT_SCHEMA: &str = "symthaea.rust-diagnostic-extraction-input.v1";
const OUTPUT_SCHEMA: &str = "symthaea.rust-diagnostic-extraction.v1";
const SET_HASH_DOMAIN: &[u8] = b"symthaea.rust-diagnostic-extraction.v1\0";
const DIAGNOSTIC_HASH_DOMAIN: &[u8] = b"symthaea.rust-diagnostic.v1\0";
const SUGGESTION_HASH_DOMAIN: &[u8] = b"symthaea.rust-suggestion-group.v1\0";

/// Binds diagnostic extraction to one exact Cargo build observation.
///
/// The caller supplies the observation ID and the exact stdout digest from
/// ENV-CARGO-001E. The transcript is re-hashed before parsing so diagnostics
/// from one run cannot be accidentally attached to another observation.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RustDiagnosticExtractionSpec {
    pub schema: String,
    pub observation_id: String,
    pub expected_stdout_sha256: String,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ExtractionState {
    Complete,
    Incomplete,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct RustDiagnosticExtraction {
    pub extraction_id: String,
    pub schema: &'static str,
    pub observation_id: String,
    pub stdout_sha256: String,
    pub state: ExtractionState,
    pub diagnostics: Vec<RustDiagnosticObservation>,
    pub machine_applicable_group_count: usize,
    pub machine_applicable_edit_count: usize,
    /// True only when at least one individually eligible MachineApplicable edit
    /// exists and the union of all such edits is pairwise non-overlapping.
    pub machine_applicable_batch_safe: bool,
    pub conflicts: Vec<RepairConflict>,
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct RustDiagnosticObservation {
    pub diagnostic_id: String,
    pub ordinal: usize,
    pub compiler_message_sha256: String,
    pub package_id: String,
    pub manifest_path: String,
    pub target_name: String,
    pub code: Option<String>,
    pub level: String,
    pub message: String,
    pub primary_spans: Vec<DiagnosticSpan>,
    pub related_spans: Vec<DiagnosticSpan>,
    pub suggestion_groups: Vec<SuggestionGroup>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct DiagnosticSpan {
    pub file_name: String,
    pub byte_start: u64,
    pub byte_end: u64,
    pub line_start: u64,
    pub line_end: u64,
    pub column_start: u64,
    pub column_end: u64,
    pub is_primary: bool,
    pub label: Option<String>,
    pub has_macro_expansion: bool,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct SuggestionGroup {
    pub suggestion_id: String,
    /// Child-index path within the rustc diagnostic tree; root is [].
    pub diagnostic_path: Vec<usize>,
    pub message: String,
    pub level: String,
    pub edits: Vec<SuggestedEdit>,
    pub auto_apply_eligible: bool,
    pub ineligibility_reasons: Vec<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct SuggestedEdit {
    pub span: DiagnosticSpan,
    pub replacement: String,
    pub applicability: SuggestionApplicability,
}

/// Forward-compatible projection of rustc's suggestion applicability.
/// Unknown future enum values are preserved and are never auto-applied.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum SuggestionApplicability {
    MachineApplicable,
    MaybeIncorrect,
    HasPlaceholders,
    Unspecified,
    Missing,
    Unknown(String),
}

impl SuggestionApplicability {
    fn parse(value: Option<&Value>) -> Result<Self, String> {
        match value {
            None | Some(Value::Null) => Ok(Self::Missing),
            Some(Value::String(value)) => Ok(match value.as_str() {
                "MachineApplicable" => Self::MachineApplicable,
                "MaybeIncorrect" => Self::MaybeIncorrect,
                "HasPlaceholders" => Self::HasPlaceholders,
                "Unspecified" => Self::Unspecified,
                other => Self::Unknown(other.to_string()),
            }),
            Some(_) => Err("suggestion_applicability is not a string/null".into()),
        }
    }

    fn machine_applicable(&self) -> bool {
        matches!(self, Self::MachineApplicable)
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct RepairConflict {
    pub first_suggestion_id: String,
    pub first_edit_index: usize,
    pub second_suggestion_id: String,
    pub second_edit_index: usize,
    pub file_name: String,
}

#[derive(Serialize)]
struct ExtractionIdentity<'a> {
    schema: &'static str,
    observation_id: &'a str,
    stdout_sha256: &'a str,
    diagnostic_ids: Vec<&'a str>,
}

#[derive(Serialize)]
struct DiagnosticIdentity<'a> {
    observation_id: &'a str,
    ordinal: usize,
    compiler_message_sha256: &'a str,
}

#[derive(Serialize)]
struct SuggestionIdentity<'a> {
    diagnostic_id: &'a str,
    diagnostic_path: &'a [usize],
    message: &'a str,
    level: &'a str,
    edits: &'a [SuggestedEdit],
}

pub fn run(
    spec_path: &Path,
    messages_path: &Path,
    output: Option<PathBuf>,
) -> anyhow::Result<()> {
    let spec_bytes = fs::read(spec_path)
        .with_context(|| format!("read Rust diagnostic spec {}", spec_path.display()))?;
    let spec: RustDiagnosticExtractionSpec = serde_json::from_slice(&spec_bytes)
        .with_context(|| format!("parse Rust diagnostic spec {}", spec_path.display()))?;
    let messages = fs::read(messages_path)
        .with_context(|| format!("read Cargo JSONL transcript {}", messages_path.display()))?;
    let extraction = extract_bytes(spec, &messages)?;

    let mut rendered = serde_json::to_string_pretty(&extraction)?;
    rendered.push('\n');
    if let Some(path) = output {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create diagnostic output directory {}", parent.display()))?;
        }
        fs::write(&path, rendered)
            .with_context(|| format!("write Rust diagnostic extraction {}", path.display()))?;
        println!("Rust diagnostic extraction written to {}", path.display());
    } else {
        print!("{rendered}");
    }
    Ok(())
}

pub fn extract_bytes(
    mut spec: RustDiagnosticExtractionSpec,
    messages: &[u8],
) -> anyhow::Result<RustDiagnosticExtraction> {
    validate_spec(&mut spec)?;
    let stdout_sha256 = sha256(messages);
    if stdout_sha256 != spec.expected_stdout_sha256 {
        bail!(
            "Cargo transcript identity mismatch: expected={}, computed={stdout_sha256}",
            spec.expected_stdout_sha256
        );
    }

    let mut diagnostics = Vec::new();
    let mut reasons = Vec::new();
    let mut ordinal = 0usize;

    for line in messages.split(|byte| *byte == b'\n') {
        let trimmed = trim_ascii(line);
        if trimmed.is_empty() || trimmed.first() != Some(&b'{') {
            continue;
        }
        let value: Value = match serde_json::from_slice(trimmed) {
            Ok(value) => value,
            Err(_) => continue, // ENV-CARGO-001E owns whole-stream completeness.
        };
        if value.get("reason").and_then(Value::as_str) != Some("compiler-message") {
            continue;
        }

        let line_hash = sha256(trimmed);
        match parse_compiler_message(&spec.observation_id, ordinal, &line_hash, &value) {
            Ok(diagnostic) => diagnostics.push(diagnostic),
            Err(reason) => reasons.push(format!("compiler-message[{ordinal}]: {reason}")),
        }
        ordinal += 1;
    }

    let conflicts = collect_conflicts(&diagnostics);
    let machine_applicable_group_count = diagnostics
        .iter()
        .flat_map(|diagnostic| &diagnostic.suggestion_groups)
        .filter(|group| group.auto_apply_eligible)
        .count();
    let machine_applicable_edit_count = diagnostics
        .iter()
        .flat_map(|diagnostic| &diagnostic.suggestion_groups)
        .filter(|group| group.auto_apply_eligible)
        .map(|group| group.edits.len())
        .sum();
    let machine_applicable_batch_safe = machine_applicable_edit_count > 0 && conflicts.is_empty();

    reasons.sort();
    reasons.dedup();
    let state = if reasons.is_empty() {
        ExtractionState::Complete
    } else {
        ExtractionState::Incomplete
    };

    let diagnostic_ids = diagnostics
        .iter()
        .map(|diagnostic| diagnostic.diagnostic_id.as_str())
        .collect();
    let extraction_identity = ExtractionIdentity {
        schema: OUTPUT_SCHEMA,
        observation_id: &spec.observation_id,
        stdout_sha256: &stdout_sha256,
        diagnostic_ids,
    };
    let extraction_bytes = serde_json::to_vec(&extraction_identity)
        .context("serialize Rust diagnostic extraction identity")?;

    Ok(RustDiagnosticExtraction {
        extraction_id: domain_sha256(SET_HASH_DOMAIN, &extraction_bytes),
        schema: OUTPUT_SCHEMA,
        observation_id: spec.observation_id,
        stdout_sha256,
        state,
        diagnostics,
        machine_applicable_group_count,
        machine_applicable_edit_count,
        machine_applicable_batch_safe,
        conflicts,
        reasons,
    })
}

fn parse_compiler_message(
    observation_id: &str,
    ordinal: usize,
    line_hash: &str,
    wrapper: &Value,
) -> Result<RustDiagnosticObservation, String> {
    let message = required_value(wrapper, "message")?;
    let diagnostic_identity = DiagnosticIdentity {
        observation_id,
        ordinal,
        compiler_message_sha256: line_hash,
    };
    let bytes = serde_json::to_vec(&diagnostic_identity)
        .map_err(|error| format!("serialize diagnostic identity: {error}"))?;
    let diagnostic_id = domain_sha256(DIAGNOSTIC_HASH_DOMAIN, &bytes);

    let mut primary_spans = Vec::new();
    let mut related_spans = Vec::new();
    for span in required_array(message, "spans")? {
        let parsed = parse_span(span)?;
        if parsed.is_primary {
            primary_spans.push(parsed);
        } else {
            related_spans.push(parsed);
        }
    }
    primary_spans.sort();
    primary_spans.dedup();
    related_spans.sort();
    related_spans.dedup();

    let mut suggestion_groups = Vec::new();
    collect_suggestion_groups(&diagnostic_id, message, &[], &mut suggestion_groups)?;
    suggestion_groups.sort_by(|a, b| a.suggestion_id.cmp(&b.suggestion_id));

    Ok(RustDiagnosticObservation {
        diagnostic_id,
        ordinal,
        compiler_message_sha256: line_hash.to_string(),
        package_id: required_string(wrapper, "package_id")?,
        manifest_path: required_string(wrapper, "manifest_path")?,
        target_name: required_value(wrapper, "target")?
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| "compiler-message.target.name missing/non-string".to_string())?
            .to_string(),
        code: diagnostic_code(message)?,
        level: required_string(message, "level")?,
        message: required_string(message, "message")?,
        primary_spans,
        related_spans,
        suggestion_groups,
    })
}

fn collect_suggestion_groups(
    diagnostic_id: &str,
    node: &Value,
    path: &[usize],
    output: &mut Vec<SuggestionGroup>,
) -> Result<(), String> {
    let mut edits = Vec::new();
    for span in required_array(node, "spans")? {
        let replacement = match span.get("suggested_replacement") {
            None | Some(Value::Null) => continue,
            Some(Value::String(value)) => value.clone(),
            Some(_) => return Err("suggested_replacement is not a string/null".into()),
        };
        edits.push(SuggestedEdit {
            span: parse_span(span)?,
            replacement,
            applicability: SuggestionApplicability::parse(span.get("suggestion_applicability"))?,
        });
    }

    if !edits.is_empty() {
        edits.sort_by(|a, b| {
            (
                &a.span.file_name,
                a.span.byte_start,
                a.span.byte_end,
                &a.replacement,
            )
                .cmp(&(
                    &b.span.file_name,
                    b.span.byte_start,
                    b.span.byte_end,
                    &b.replacement,
                ))
        });
        let mut ineligibility_reasons = Vec::new();
        if edits.iter().any(|edit| !edit.applicability.machine_applicable()) {
            ineligibility_reasons.push("not every edit is MachineApplicable".into());
        }
        if edits.iter().any(|edit| edit.span.has_macro_expansion) {
            ineligibility_reasons.push("suggestion touches a macro expansion".into());
        }
        if edits.iter().any(|edit| edit.span.byte_end < edit.span.byte_start) {
            ineligibility_reasons.push("suggestion contains an invalid byte range".into());
        }
        if internal_edits_overlap(&edits) {
            ineligibility_reasons.push("suggestion contains overlapping edits".into());
        }
        ineligibility_reasons.sort();
        ineligibility_reasons.dedup();

        let message = required_string(node, "message")?;
        let level = required_string(node, "level")?;
        let identity = SuggestionIdentity {
            diagnostic_id,
            diagnostic_path: path,
            message: &message,
            level: &level,
            edits: &edits,
        };
        let bytes = serde_json::to_vec(&identity)
            .map_err(|error| format!("serialize suggestion identity: {error}"))?;
        output.push(SuggestionGroup {
            suggestion_id: domain_sha256(SUGGESTION_HASH_DOMAIN, &bytes),
            diagnostic_path: path.to_vec(),
            message,
            level,
            edits,
            auto_apply_eligible: ineligibility_reasons.is_empty(),
            ineligibility_reasons,
        });
    }

    for (index, child) in required_array(node, "children")?.iter().enumerate() {
        let mut child_path = path.to_vec();
        child_path.push(index);
        collect_suggestion_groups(diagnostic_id, child, &child_path, output)?;
    }
    Ok(())
}

fn collect_conflicts(diagnostics: &[RustDiagnosticObservation]) -> Vec<RepairConflict> {
    struct EditRef<'a> {
        suggestion_id: &'a str,
        edit_index: usize,
        span: &'a DiagnosticSpan,
    }

    let mut edits = Vec::new();
    for diagnostic in diagnostics {
        for group in &diagnostic.suggestion_groups {
            if !group.auto_apply_eligible {
                continue;
            }
            for (edit_index, edit) in group.edits.iter().enumerate() {
                edits.push(EditRef {
                    suggestion_id: &group.suggestion_id,
                    edit_index,
                    span: &edit.span,
                });
            }
        }
    }

    let mut conflicts = Vec::new();
    for first_index in 0..edits.len() {
        for second_index in first_index + 1..edits.len() {
            let first = &edits[first_index];
            let second = &edits[second_index];
            if spans_conflict(first.span, second.span) {
                conflicts.push(RepairConflict {
                    first_suggestion_id: first.suggestion_id.to_string(),
                    first_edit_index: first.edit_index,
                    second_suggestion_id: second.suggestion_id.to_string(),
                    second_edit_index: second.edit_index,
                    file_name: first.span.file_name.clone(),
                });
            }
        }
    }
    conflicts.sort_by(|a, b| {
        (
            &a.file_name,
            &a.first_suggestion_id,
            a.first_edit_index,
            &a.second_suggestion_id,
            a.second_edit_index,
        )
            .cmp(&(
                &b.file_name,
                &b.first_suggestion_id,
                b.first_edit_index,
                &b.second_suggestion_id,
                b.second_edit_index,
            ))
    });
    conflicts
}

fn internal_edits_overlap(edits: &[SuggestedEdit]) -> bool {
    edits.iter().enumerate().any(|(index, first)| {
        edits[index + 1..]
            .iter()
            .any(|second| spans_conflict(&first.span, &second.span))
    })
}

fn spans_conflict(first: &DiagnosticSpan, second: &DiagnosticSpan) -> bool {
    if first.file_name != second.file_name {
        return false;
    }
    let (a_start, a_end) = (first.byte_start, first.byte_end);
    let (b_start, b_end) = (second.byte_start, second.byte_end);

    if a_start == a_end && b_start == b_end {
        return a_start == b_start;
    }
    if a_start == a_end {
        return a_start >= b_start && a_start <= b_end;
    }
    if b_start == b_end {
        return b_start >= a_start && b_start <= a_end;
    }
    a_start < b_end && b_start < a_end
}

fn parse_span(value: &Value) -> Result<DiagnosticSpan, String> {
    Ok(DiagnosticSpan {
        file_name: required_string(value, "file_name")?,
        byte_start: required_u64(value, "byte_start")?,
        byte_end: required_u64(value, "byte_end")?,
        line_start: required_u64(value, "line_start")?,
        line_end: required_u64(value, "line_end")?,
        column_start: required_u64(value, "column_start")?,
        column_end: required_u64(value, "column_end")?,
        is_primary: required_bool(value, "is_primary")?,
        label: optional_string(value, "label")?,
        has_macro_expansion: !matches!(value.get("expansion"), None | Some(Value::Null)),
    })
}

fn diagnostic_code(value: &Value) -> Result<Option<String>, String> {
    match value.get("code") {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Object(code)) => match code.get("code") {
            None | Some(Value::Null) => Ok(None),
            Some(Value::String(code)) => Ok(Some(code.clone())),
            Some(_) => Err("diagnostic code.code is not a string/null".into()),
        },
        Some(_) => Err("diagnostic code is not an object/null".into()),
    }
}

fn required_value<'a>(value: &'a Value, key: &str) -> Result<&'a Value, String> {
    value.get(key).ok_or_else(|| format!("missing field {key}"))
}

fn required_array<'a>(value: &'a Value, key: &str) -> Result<&'a Vec<Value>, String> {
    value
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("missing/non-array field {key}"))
}

fn required_string(value: &Value, key: &str) -> Result<String, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| format!("missing/non-string field {key}"))
}

fn optional_string(value: &Value, key: &str) -> Result<Option<String>, String> {
    match value.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => Ok(Some(value.clone())),
        Some(_) => Err(format!("field {key} is not a string/null")),
    }
}

fn required_u64(value: &Value, key: &str) -> Result<u64, String> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("missing/non-u64 field {key}"))
}

fn required_bool(value: &Value, key: &str) -> Result<bool, String> {
    value
        .get(key)
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("missing/non-bool field {key}"))
}

fn validate_spec(spec: &mut RustDiagnosticExtractionSpec) -> anyhow::Result<()> {
    if spec.schema != INPUT_SCHEMA {
        bail!("unsupported Rust diagnostic extraction schema: {}", spec.schema);
    }
    validate_sha256("observation_id", &spec.observation_id)?;
    validate_sha256("expected_stdout_sha256", &spec.expected_stdout_sha256)?;
    spec.observation_id.make_ascii_lowercase();
    spec.expected_stdout_sha256.make_ascii_lowercase();
    Ok(())
}

fn validate_sha256(name: &str, digest: &str) -> anyhow::Result<()> {
    if digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("{name} must be a 64-character SHA-256 hex digest");
    }
    Ok(())
}

fn trim_ascii(mut bytes: &[u8]) -> &[u8] {
    while bytes.first().is_some_and(|byte| byte.is_ascii_whitespace()) {
        bytes = &bytes[1..];
    }
    while bytes.last().is_some_and(|byte| byte.is_ascii_whitespace()) {
        bytes = &bytes[..bytes.len() - 1];
    }
    bytes
}

fn domain_sha256(domain: &[u8], bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(bytes);
    hex_lower(&hasher.finalize())
}

fn sha256(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex_lower(&hasher.finalize())
}

fn hex_lower(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        write!(&mut out, "{byte:02x}").expect("writing to String cannot fail");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec_for(messages: &[u8]) -> RustDiagnosticExtractionSpec {
        RustDiagnosticExtractionSpec {
            schema: INPUT_SCHEMA.into(),
            observation_id: "11".repeat(32),
            expected_stdout_sha256: sha256(messages),
        }
    }

    fn target() -> &'static str {
        r#"{"name":"demo"}"#
    }

    fn span(
        start: u64,
        end: u64,
        replacement: Option<&str>,
        applicability: Option<&str>,
        expansion: bool,
    ) -> String {
        let replacement = replacement
            .map(|value| format!("\"{value}\""))
            .unwrap_or_else(|| "null".into());
        let applicability = applicability
            .map(|value| format!("\"{value}\""))
            .unwrap_or_else(|| "null".into());
        let expansion = if expansion { "{}" } else { "null" };
        format!(
            r#"{{"file_name":"src/lib.rs","byte_start":{start},"byte_end":{end},"line_start":1,"line_end":1,"column_start":1,"column_end":2,"is_primary":true,"text":[],"label":null,"suggested_replacement":{replacement},"suggestion_applicability":{applicability},"expansion":{expansion}}}"#
        )
    }

    fn compiler_message(root_spans: &str, children: &str) -> String {
        format!(
            r#"{{"reason":"compiler-message","package_id":"path+file:///repo#demo@0.1.0","manifest_path":"/repo/Cargo.toml","target":{},"message":{{"message":"diagnostic","code":{{"code":"E0000","explanation":null}},"level":"error","spans":[{root_spans}],"children":[{children}],"rendered":null}}}}"#,
            target()
        )
    }

    #[test]
    fn extracts_machine_applicable_root_suggestion() {
        let line = compiler_message(
            &span(10, 13, Some("new"), Some("MachineApplicable"), false),
            "",
        );
        let transcript = format!("{line}\n{{\"reason\":\"build-finished\",\"success\":false}}\n");
        let result = extract_bytes(spec_for(transcript.as_bytes()), transcript.as_bytes()).unwrap();
        assert_eq!(result.state, ExtractionState::Complete);
        assert_eq!(result.machine_applicable_group_count, 1);
        assert_eq!(result.machine_applicable_edit_count, 1);
        assert!(result.machine_applicable_batch_safe);
        let group = &result.diagnostics[0].suggestion_groups[0];
        assert!(group.auto_apply_eligible);
        assert_eq!(group.edits[0].replacement, "new");
    }

    #[test]
    fn child_help_suggestion_is_preserved_with_path() {
        let child = format!(
            r#"{{"message":"prefix with underscore","code":null,"level":"help","spans":[{}],"children":[],"rendered":null}}"#,
            span(4, 5, Some("_x"), Some("MachineApplicable"), false)
        );
        let line = compiler_message("", &child);
        let transcript = format!("{line}\n");
        let result = extract_bytes(spec_for(transcript.as_bytes()), transcript.as_bytes()).unwrap();
        let group = &result.diagnostics[0].suggestion_groups[0];
        assert_eq!(group.diagnostic_path, vec![0]);
        assert!(group.auto_apply_eligible);
    }

    #[test]
    fn maybe_incorrect_and_macro_expansion_never_auto_apply() {
        let children = format!(
            r#"{{"message":"uncertain","code":null,"level":"help","spans":[{}],"children":[],"rendered":null}},{{"message":"macro","code":null,"level":"help","spans":[{}],"children":[],"rendered":null}}"#,
            span(4, 5, Some("a"), Some("MaybeIncorrect"), false),
            span(8, 9, Some("b"), Some("MachineApplicable"), true)
        );
        let line = compiler_message("", &children);
        let transcript = format!("{line}\n");
        let result = extract_bytes(spec_for(transcript.as_bytes()), transcript.as_bytes()).unwrap();
        assert_eq!(result.machine_applicable_group_count, 0);
        assert!(!result.machine_applicable_batch_safe);
    }

    #[test]
    fn unknown_future_applicability_is_preserved_not_promoted() {
        let line = compiler_message(
            &span(1, 2, Some("x"), Some("FutureApplicability"), false),
            "",
        );
        let transcript = format!("{line}\n");
        let result = extract_bytes(spec_for(transcript.as_bytes()), transcript.as_bytes()).unwrap();
        assert!(matches!(
            result.diagnostics[0].suggestion_groups[0].edits[0].applicability,
            SuggestionApplicability::Unknown(_)
        ));
        assert!(!result.diagnostics[0].suggestion_groups[0].auto_apply_eligible);
    }

    #[test]
    fn overlapping_machine_suggestions_make_batch_unsafe() {
        let children = format!(
            r#"{{"message":"one","code":null,"level":"help","spans":[{}],"children":[],"rendered":null}},{{"message":"two","code":null,"level":"help","spans":[{}],"children":[],"rendered":null}}"#,
            span(4, 8, Some("a"), Some("MachineApplicable"), false),
            span(7, 10, Some("b"), Some("MachineApplicable"), false)
        );
        let line = compiler_message("", &children);
        let transcript = format!("{line}\n");
        let result = extract_bytes(spec_for(transcript.as_bytes()), transcript.as_bytes()).unwrap();
        assert_eq!(result.machine_applicable_group_count, 2);
        assert_eq!(result.conflicts.len(), 1);
        assert!(!result.machine_applicable_batch_safe);
    }

    #[test]
    fn transcript_must_match_bound_cargo_observation() {
        let messages = b"{\"reason\":\"build-finished\",\"success\":true}\n";
        let mut spec = spec_for(messages);
        spec.expected_stdout_sha256 = "22".repeat(32);
        assert!(extract_bytes(spec, messages).is_err());
    }
}
