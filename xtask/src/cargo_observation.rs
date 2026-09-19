use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};

const INPUT_SCHEMA: &str = "symthaea.cargo-build-observation-input.v1";
const OBSERVATION_SCHEMA: &str = "symthaea.cargo-build-observation.v1";
const OBSERVATION_HASH_DOMAIN: &[u8] = b"symthaea.cargo-build-observation.v1\0";

/// Non-stdout facts supplied by the command wrapper that captured a Cargo run.
/// The Cargo JSONL transcript itself is read separately and hashed byte-for-byte.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CargoObservationSpec {
    pub schema: String,
    pub context_id: String,
    pub invocation_id: String,
    pub cargo_version: String,
    pub rustc_version: String,
    pub exit_code: i32,
    #[serde(default)]
    pub stderr_sha256: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ObservationState {
    CompleteMessageSurface,
    IncompleteMessageSurface,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct CargoBuildObservation {
    pub observation_id: String,
    pub schema: &'static str,
    pub context_id: String,
    pub invocation_id: String,
    pub cargo_version: String,
    pub rustc_version: String,
    pub exit_code: i32,
    pub stdout_sha256: String,
    pub stderr_sha256: Option<String>,
    pub state: ObservationState,
    pub build_finished: Option<bool>,
    pub compiler_artifacts: Vec<CompilerArtifactObservation>,
    pub build_scripts: Vec<BuildScriptObservation>,
    pub diagnostics: Vec<DiagnosticObservation>,
    pub unknown_messages: Vec<UnknownMessageObservation>,
    pub opaque_stdout_line_count: usize,
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct CompilerArtifactObservation {
    pub package_id: String,
    pub manifest_path: String,
    pub target: CargoTargetObservation,
    pub profile: CargoProfileObservation,
    pub features: Vec<String>,
    pub filenames: Vec<String>,
    pub executable: Option<String>,
    pub fresh: bool,
    pub message_sha256: String,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct CargoTargetObservation {
    pub name: String,
    pub kind: Vec<String>,
    pub crate_types: Vec<String>,
    pub src_path: String,
    pub edition: String,
    pub required_features: Vec<String>,
    pub doc: Option<bool>,
    pub doctest: Option<bool>,
    pub test: Option<bool>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct CargoProfileObservation {
    pub opt_level: Option<String>,
    pub debuginfo: Option<Value>,
    pub debug_assertions: Option<bool>,
    pub overflow_checks: Option<bool>,
    pub test: Option<bool>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct BuildScriptObservation {
    pub package_id: String,
    /// Linker ordering can be significant, so this is deliberately not sorted.
    pub linked_libs: Vec<String>,
    /// Search-path ordering can be significant, so this is deliberately not sorted.
    pub linked_paths: Vec<String>,
    pub cfgs: Vec<String>,
    /// Preserve emission order and duplicate keys; consumers may interpret last-wins semantics.
    pub env: Vec<[String; 2]>,
    pub out_dir: String,
    pub message_sha256: String,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct DiagnosticObservation {
    pub package_id: String,
    pub manifest_path: String,
    pub target: CargoTargetObservation,
    pub level: String,
    pub code: Option<String>,
    pub message: String,
    pub rendered_sha256: Option<String>,
    pub message_sha256: String,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct UnknownMessageObservation {
    pub reason: Option<String>,
    pub message_sha256: String,
}

#[derive(Serialize)]
struct ObservationIdentity<'a> {
    schema: &'static str,
    context_id: &'a str,
    invocation_id: &'a str,
    cargo_version: &'a str,
    rustc_version: &'a str,
    exit_code: i32,
    stdout_sha256: &'a str,
    stderr_sha256: &'a Option<String>,
}

pub fn run(
    spec_path: &Path,
    messages_path: &Path,
    output: Option<PathBuf>,
) -> anyhow::Result<()> {
    let spec_bytes = fs::read(spec_path)
        .with_context(|| format!("read Cargo observation spec {}", spec_path.display()))?;
    let spec: CargoObservationSpec = serde_json::from_slice(&spec_bytes)
        .with_context(|| format!("parse Cargo observation spec {}", spec_path.display()))?;
    let messages = fs::read(messages_path)
        .with_context(|| format!("read Cargo JSONL transcript {}", messages_path.display()))?;
    let observation = observe_bytes(spec, &messages)?;

    let mut rendered = serde_json::to_string_pretty(&observation)?;
    rendered.push('\n');
    if let Some(path) = output {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create observation output directory {}", parent.display()))?;
        }
        fs::write(&path, rendered)
            .with_context(|| format!("write Cargo observation {}", path.display()))?;
        println!("Cargo build observation written to {}", path.display());
    } else {
        print!("{rendered}");
    }
    Ok(())
}

pub fn observe_bytes(
    mut spec: CargoObservationSpec,
    messages: &[u8],
) -> anyhow::Result<CargoBuildObservation> {
    validate_spec(&mut spec)?;
    let stdout_sha256 = sha256(messages);

    let identity = ObservationIdentity {
        schema: OBSERVATION_SCHEMA,
        context_id: &spec.context_id,
        invocation_id: &spec.invocation_id,
        cargo_version: &spec.cargo_version,
        rustc_version: &spec.rustc_version,
        exit_code: spec.exit_code,
        stdout_sha256: &stdout_sha256,
        stderr_sha256: &spec.stderr_sha256,
    };
    let identity_bytes = serde_json::to_vec(&identity).context("serialize Cargo observation identity")?;
    let observation_id = domain_sha256(OBSERVATION_HASH_DOMAIN, &identity_bytes);

    let mut artifacts = Vec::new();
    let mut build_scripts = Vec::new();
    let mut diagnostics = Vec::new();
    let mut unknown_messages = Vec::new();
    let mut reasons = Vec::new();
    let mut opaque_stdout_line_count = 0usize;
    let mut build_finished_values = Vec::new();

    for line in messages.split(|byte| *byte == b'\n') {
        let trimmed = trim_ascii(line);
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.first() != Some(&b'{') {
            opaque_stdout_line_count += 1;
            continue;
        }

        let line_hash = sha256(trimmed);
        let value: Value = match serde_json::from_slice(trimmed) {
            Ok(value) => value,
            Err(error) => {
                reasons.push(format!(
                    "JSON-looking stdout line could not be parsed: {error}"
                ));
                unknown_messages.push(UnknownMessageObservation {
                    reason: None,
                    message_sha256: line_hash,
                });
                continue;
            }
        };

        let reason = value.get("reason").and_then(Value::as_str);
        let parsed = match reason {
            Some("compiler-artifact") => parse_artifact(&value, line_hash)
                .map(|record| artifacts.push(record)),
            Some("build-script-executed") => parse_build_script(&value, line_hash)
                .map(|record| build_scripts.push(record)),
            Some("compiler-message") => parse_diagnostic(&value, line_hash)
                .map(|record| diagnostics.push(record)),
            Some("build-finished") => required_bool(&value, "success")
                .map(|success| build_finished_values.push(success)),
            other => {
                unknown_messages.push(UnknownMessageObservation {
                    reason: other.map(str::to_string),
                    message_sha256: line_hash,
                });
                Err(format!(
                    "unsupported Cargo JSON message reason: {}",
                    other.unwrap_or("<missing>")
                ))
            }
        };

        if let Err(reason) = parsed {
            reasons.push(reason);
        }
    }

    if build_finished_values.is_empty() {
        reasons.push("Cargo transcript did not contain build-finished".into());
    } else if build_finished_values.len() > 1 {
        reasons.push(format!(
            "Cargo transcript contained {} build-finished messages",
            build_finished_values.len()
        ));
    }
    let build_finished = if build_finished_values.len() == 1 {
        Some(build_finished_values[0])
    } else {
        None
    };

    canonicalize_records(&mut artifacts, &mut build_scripts, &mut diagnostics);
    reasons.sort();
    reasons.dedup();
    unknown_messages.sort_by(|a, b| {
        (&a.reason, &a.message_sha256).cmp(&(&b.reason, &b.message_sha256))
    });

    let state = if reasons.is_empty() {
        ObservationState::CompleteMessageSurface
    } else {
        ObservationState::IncompleteMessageSurface
    };

    Ok(CargoBuildObservation {
        observation_id,
        schema: OBSERVATION_SCHEMA,
        context_id: spec.context_id,
        invocation_id: spec.invocation_id,
        cargo_version: spec.cargo_version,
        rustc_version: spec.rustc_version,
        exit_code: spec.exit_code,
        stdout_sha256,
        stderr_sha256: spec.stderr_sha256,
        state,
        build_finished,
        compiler_artifacts: artifacts,
        build_scripts,
        diagnostics,
        unknown_messages,
        opaque_stdout_line_count,
        reasons,
    })
}

fn validate_spec(spec: &mut CargoObservationSpec) -> anyhow::Result<()> {
    if spec.schema != INPUT_SCHEMA {
        bail!("unsupported Cargo observation input schema: {}", spec.schema);
    }
    validate_sha256("context_id", &spec.context_id)?;
    validate_sha256("invocation_id", &spec.invocation_id)?;
    spec.context_id.make_ascii_lowercase();
    spec.invocation_id.make_ascii_lowercase();
    if spec.cargo_version.trim().is_empty() || spec.rustc_version.trim().is_empty() {
        bail!("cargo_version and rustc_version must not be empty");
    }
    if let Some(digest) = &mut spec.stderr_sha256 {
        validate_sha256("stderr_sha256", digest)?;
        digest.make_ascii_lowercase();
    }
    Ok(())
}

fn parse_artifact(value: &Value, message_sha256: String) -> Result<CompilerArtifactObservation, String> {
    let mut features = required_strings(value, "features")?;
    let mut filenames = required_strings(value, "filenames")?;
    canonicalize_set(&mut features);
    canonicalize_set(&mut filenames);

    Ok(CompilerArtifactObservation {
        package_id: required_string(value, "package_id")?,
        manifest_path: required_string(value, "manifest_path")?,
        target: parse_target(required_value(value, "target")?)?,
        profile: parse_profile(required_value(value, "profile")?)?,
        features,
        filenames,
        executable: optional_string(value, "executable")?,
        fresh: required_bool(value, "fresh")?,
        message_sha256,
    })
}

fn parse_build_script(value: &Value, message_sha256: String) -> Result<BuildScriptObservation, String> {
    let mut cfgs = required_strings(value, "cfgs")?;
    canonicalize_set(&mut cfgs);
    Ok(BuildScriptObservation {
        package_id: required_string(value, "package_id")?,
        linked_libs: required_strings(value, "linked_libs")?,
        linked_paths: required_strings(value, "linked_paths")?,
        cfgs,
        env: required_env(value, "env")?,
        out_dir: required_string(value, "out_dir")?,
        message_sha256,
    })
}

fn parse_diagnostic(value: &Value, message_sha256: String) -> Result<DiagnosticObservation, String> {
    let message = required_value(value, "message")?;
    let rendered_sha256 = match message.get("rendered") {
        Some(Value::String(rendered)) => Some(sha256(rendered.as_bytes())),
        Some(Value::Null) | None => None,
        Some(_) => return Err("compiler-message.message.rendered is not a string/null".into()),
    };
    let code = match message.get("code") {
        Some(Value::Object(code)) => code
            .get("code")
            .and_then(Value::as_str)
            .map(str::to_string),
        Some(Value::Null) | None => None,
        Some(_) => return Err("compiler-message.message.code is not an object/null".into()),
    };

    Ok(DiagnosticObservation {
        package_id: required_string(value, "package_id")?,
        manifest_path: required_string(value, "manifest_path")?,
        target: parse_target(required_value(value, "target")?)?,
        level: required_string(message, "level")?,
        code,
        message: required_string(message, "message")?,
        rendered_sha256,
        message_sha256,
    })
}

fn parse_target(value: &Value) -> Result<CargoTargetObservation, String> {
    let mut kind = required_strings(value, "kind")?;
    let mut crate_types = required_strings(value, "crate_types")?;
    let mut required_features = optional_strings(value, "required-features")?;
    canonicalize_set(&mut kind);
    canonicalize_set(&mut crate_types);
    canonicalize_set(&mut required_features);

    Ok(CargoTargetObservation {
        name: required_string(value, "name")?,
        kind,
        crate_types,
        src_path: required_string(value, "src_path")?,
        edition: required_string(value, "edition")?,
        required_features,
        doc: optional_bool(value, "doc")?,
        doctest: optional_bool(value, "doctest")?,
        test: optional_bool(value, "test")?,
    })
}

fn parse_profile(value: &Value) -> Result<CargoProfileObservation, String> {
    Ok(CargoProfileObservation {
        opt_level: value
            .get("opt_level")
            .and_then(Value::as_str)
            .map(str::to_string),
        debuginfo: value.get("debuginfo").cloned().filter(|value| !value.is_null()),
        debug_assertions: optional_bool(value, "debug_assertions")?,
        overflow_checks: optional_bool(value, "overflow_checks")?,
        test: optional_bool(value, "test")?,
    })
}

fn required_env(value: &Value, key: &str) -> Result<Vec<[String; 2]>, String> {
    let items = value
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("missing/non-array field {key}"))?;
    let mut result = Vec::with_capacity(items.len());
    for item in items {
        let pair = item
            .as_array()
            .ok_or_else(|| format!("{key} entry is not an array"))?;
        if pair.len() != 2 {
            return Err(format!("{key} entry does not contain exactly two values"));
        }
        let k = pair[0]
            .as_str()
            .ok_or_else(|| format!("{key} key is not a string"))?;
        let v = pair[1]
            .as_str()
            .ok_or_else(|| format!("{key} value is not a string"))?;
        result.push([k.to_string(), v.to_string()]);
    }
    Ok(result)
}

fn required_value<'a>(value: &'a Value, key: &str) -> Result<&'a Value, String> {
    value.get(key).ok_or_else(|| format!("missing field {key}"))
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

fn required_bool(value: &Value, key: &str) -> Result<bool, String> {
    value
        .get(key)
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("missing/non-bool field {key}"))
}

fn optional_bool(value: &Value, key: &str) -> Result<Option<bool>, String> {
    match value.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Bool(value)) => Ok(Some(*value)),
        Some(_) => Err(format!("field {key} is not a bool/null")),
    }
}

fn required_strings(value: &Value, key: &str) -> Result<Vec<String>, String> {
    let items = value
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("missing/non-array field {key}"))?;
    items
        .iter()
        .map(|item| {
            item.as_str()
                .map(str::to_string)
                .ok_or_else(|| format!("{key} contains a non-string value"))
        })
        .collect()
}

fn optional_strings(value: &Value, key: &str) -> Result<Vec<String>, String> {
    match value.get(key) {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(_) => required_strings(value, key),
    }
}

fn canonicalize_records(
    artifacts: &mut Vec<CompilerArtifactObservation>,
    build_scripts: &mut Vec<BuildScriptObservation>,
    diagnostics: &mut Vec<DiagnosticObservation>,
) {
    artifacts.sort_by(|a, b| {
        (
            &a.package_id,
            &a.target.name,
            &a.target.kind,
            &a.features,
            &a.message_sha256,
        )
            .cmp(&(
                &b.package_id,
                &b.target.name,
                &b.target.kind,
                &b.features,
                &b.message_sha256,
            ))
    });
    build_scripts.sort_by(|a, b| {
        (&a.package_id, &a.message_sha256).cmp(&(&b.package_id, &b.message_sha256))
    });
    diagnostics.sort_by(|a, b| {
        (
            &a.package_id,
            &a.target.name,
            &a.level,
            &a.code,
            &a.message,
            &a.message_sha256,
        )
            .cmp(&(
                &b.package_id,
                &b.target.name,
                &b.level,
                &b.code,
                &b.message,
                &b.message_sha256,
            ))
    });
}

fn canonicalize_set(values: &mut Vec<String>) {
    values.sort();
    values.dedup();
}

fn trim_ascii(mut bytes: &[u8]) -> &[u8] {
    while bytes.first().is_some_and(u8::is_ascii_whitespace) {
        bytes = &bytes[1..];
    }
    while bytes.last().is_some_and(u8::is_ascii_whitespace) {
        bytes = &bytes[..bytes.len() - 1];
    }
    bytes
}

fn validate_sha256(name: &str, digest: &str) -> anyhow::Result<()> {
    if digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("{name} must be a 64-character SHA-256 hex digest");
    }
    Ok(())
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

    fn spec() -> CargoObservationSpec {
        CargoObservationSpec {
            schema: INPUT_SCHEMA.into(),
            context_id: "11".repeat(32),
            invocation_id: "22".repeat(32),
            cargo_version: "cargo 1.96.0".into(),
            rustc_version: "rustc 1.96.0".into(),
            exit_code: 0,
            stderr_sha256: Some("33".repeat(32)),
        }
    }

    fn target_json() -> &'static str {
        r#"{"kind":["lib"],"crate_types":["lib"],"name":"demo","src_path":"/repo/src/lib.rs","edition":"2024","doc":true,"doctest":true,"test":true}"#
    }

    #[test]
    fn captures_artifact_features_build_script_native_edges_and_diagnostic() {
        let transcript = format!(
            "{{\"reason\":\"compiler-artifact\",\"package_id\":\"path+file:///repo#demo@0.1.0\",\"manifest_path\":\"/repo/Cargo.toml\",\"target\":{},\"profile\":{{\"opt_level\":\"0\",\"debuginfo\":2,\"debug_assertions\":true,\"overflow_checks\":true,\"test\":false}},\"features\":[\"z\",\"a\"],\"filenames\":[\"/repo/target/debug/libdemo.rlib\"],\"executable\":null,\"fresh\":false}}\n\
{{\"reason\":\"build-script-executed\",\"package_id\":\"registry+https://example.invalid#index@1.0.0\",\"linked_libs\":[\"foo\",\"static=bar\"],\"linked_paths\":[\"native=/opt/lib\"],\"cfgs\":[\"cfg_b\",\"cfg_a\"],\"env\":[[\"NATIVE_HOME\",\"/opt/native\"]],\"out_dir\":\"/repo/target/debug/build/pkg/out\"}}\n\
{{\"reason\":\"compiler-message\",\"package_id\":\"path+file:///repo#demo@0.1.0\",\"manifest_path\":\"/repo/Cargo.toml\",\"target\":{},\"message\":{{\"message\":\"unused variable\",\"code\":{{\"code\":\"unused_variables\",\"explanation\":null}},\"level\":\"warning\",\"rendered\":\"warning: unused variable\\n\"}}}}\n\
{{\"reason\":\"build-finished\",\"success\":true}}\n",
            target_json(),
            target_json()
        );

        let observation = observe_bytes(spec(), transcript.as_bytes()).unwrap();
        assert_eq!(observation.state, ObservationState::CompleteMessageSurface);
        assert_eq!(observation.build_finished, Some(true));
        assert_eq!(observation.compiler_artifacts.len(), 1);
        assert_eq!(
            observation.compiler_artifacts[0].features,
            vec!["a".to_string(), "z".to_string()]
        );
        assert_eq!(observation.build_scripts.len(), 1);
        assert_eq!(
            observation.build_scripts[0].linked_libs,
            vec!["foo".to_string(), "static=bar".to_string()]
        );
        assert_eq!(observation.diagnostics[0].code.as_deref(), Some("unused_variables"));
    }

    #[test]
    fn opaque_program_output_is_bound_but_does_not_make_cargo_surface_unknown() {
        let transcript = b"program says hello\n{\"reason\":\"build-finished\",\"success\":true}\n";
        let observation = observe_bytes(spec(), transcript).unwrap();
        assert_eq!(observation.state, ObservationState::CompleteMessageSurface);
        assert_eq!(observation.opaque_stdout_line_count, 1);
    }

    #[test]
    fn unknown_json_reason_fails_toward_incomplete() {
        let transcript = b"{\"reason\":\"future-cargo-message\",\"value\":1}\n{\"reason\":\"build-finished\",\"success\":true}\n";
        let observation = observe_bytes(spec(), transcript).unwrap();
        assert_eq!(observation.state, ObservationState::IncompleteMessageSurface);
        assert_eq!(observation.unknown_messages.len(), 1);
    }

    #[test]
    fn missing_build_finished_is_incomplete() {
        let transcript = b"not cargo json\n";
        let observation = observe_bytes(spec(), transcript).unwrap();
        assert_eq!(observation.state, ObservationState::IncompleteMessageSurface);
        assert!(observation
            .reasons
            .iter()
            .any(|reason| reason.contains("did not contain build-finished")));
    }

    #[test]
    fn exact_transcript_bytes_are_identity_significant() {
        let a = observe_bytes(
            spec(),
            b"{\"reason\":\"build-finished\",\"success\":true}\n",
        )
        .unwrap();
        let b = observe_bytes(
            spec(),
            b"{\"reason\":\"build-finished\",\"success\":true}\r\n",
        )
        .unwrap();
        assert_ne!(a.stdout_sha256, b.stdout_sha256);
        assert_ne!(a.observation_id, b.observation_id);
    }

    #[test]
    fn invalid_context_identity_is_rejected() {
        let mut value = spec();
        value.context_id = "not-a-digest".into();
        assert!(observe_bytes(value, b"").is_err());
    }
}
