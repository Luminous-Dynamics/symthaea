use anyhow::{Context, bail};
use serde::Serialize;
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};

use crate::cargo_observation::{CargoBuildObservation, CargoObservationSpec, observe_bytes};

const INPUT_SCHEMA: &str = "symthaea.cargo-build-observation-input.v1";
const OBSERVATION_SCHEMA: &str = "symthaea.cargo-build-observation.v1";
const VERIFY_SCHEMA: &str = "symthaea.cargo-build-observation-verification.v1";

#[derive(Debug, Serialize)]
pub struct CargoObservationVerification {
    pub schema: &'static str,
    pub observation_id: String,
    pub stdout_sha256: String,
    pub semantic_receipt_match: bool,
}

pub fn run(
    observation_path: &Path,
    messages_path: &Path,
    output: Option<PathBuf>,
) -> anyhow::Result<()> {
    let receipt_bytes = fs::read(observation_path)
        .with_context(|| format!("read Cargo observation receipt {}", observation_path.display()))?;
    let messages = fs::read(messages_path)
        .with_context(|| format!("read Cargo JSONL transcript {}", messages_path.display()))?;
    let verified = verify_bytes(&receipt_bytes, &messages)?;

    let report = CargoObservationVerification {
        schema: VERIFY_SCHEMA,
        observation_id: verified.observation_id,
        stdout_sha256: verified.stdout_sha256,
        semantic_receipt_match: true,
    };
    write_json(&report, output)
}

pub(crate) fn verify_bytes(
    receipt_bytes: &[u8],
    messages: &[u8],
) -> anyhow::Result<CargoBuildObservation> {
    let stored: Value =
        serde_json::from_slice(receipt_bytes).context("parse stored Cargo observation receipt")?;
    let object = stored
        .as_object()
        .context("stored Cargo observation receipt is not a JSON object")?;

    let schema = object
        .get("schema")
        .and_then(Value::as_str)
        .context("stored Cargo observation missing/non-string schema")?;
    if schema != OBSERVATION_SCHEMA {
        bail!("unsupported Cargo observation schema: {schema}");
    }

    let context_id = required_string(object, "context_id")?;
    let invocation_id = required_string(object, "invocation_id")?;
    let cargo_version = required_string(object, "cargo_version")?;
    let rustc_version = required_string(object, "rustc_version")?;
    let exit_code_i64 = object
        .get("exit_code")
        .and_then(Value::as_i64)
        .context("stored Cargo observation missing/non-integer exit_code")?;
    let exit_code = i32::try_from(exit_code_i64)
        .context("stored Cargo observation exit_code does not fit i32")?;
    let stderr_sha256 = match object.get("stderr_sha256") {
        None | Some(Value::Null) => None,
        Some(Value::String(value)) => Some(value.clone()),
        Some(_) => bail!("stored Cargo observation stderr_sha256 is not string/null"),
    };

    let spec = CargoObservationSpec {
        schema: INPUT_SCHEMA.into(),
        context_id,
        invocation_id,
        cargo_version,
        rustc_version,
        exit_code,
        stderr_sha256,
    };
    let derived = observe_bytes(spec, messages)?;
    let derived_value =
        serde_json::to_value(&derived).context("serialize re-derived Cargo observation")?;

    if derived_value != stored {
        bail!(
            "stored Cargo observation does not match the canonical projection re-derived from the exact transcript"
        );
    }
    Ok(derived)
}

fn required_string(
    object: &serde_json::Map<String, Value>,
    key: &str,
) -> anyhow::Result<String> {
    object
        .get(key)
        .and_then(Value::as_str)
        .map(str::to_string)
        .with_context(|| format!("stored Cargo observation missing/non-string {key}"))
}

fn write_json<T: Serialize>(value: &T, output: Option<PathBuf>) -> anyhow::Result<()> {
    let mut rendered = serde_json::to_string_pretty(value)?;
    rendered.push('\n');
    if let Some(path) = output {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!("create Cargo observation verification directory {}", parent.display())
            })?;
        }
        fs::write(&path, rendered).with_context(|| {
            format!("write Cargo observation verification {}", path.display())
        })?;
        println!("Cargo observation verification written to {}", path.display());
    } else {
        print!("{rendered}");
    }
    Ok(())
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

    fn transcript() -> &'static [u8] {
        b"{\"reason\":\"build-finished\",\"success\":true}\n"
    }

    #[test]
    fn exact_receipt_rederives_from_transcript() {
        let observation = observe_bytes(spec(), transcript()).unwrap();
        let receipt = serde_json::to_vec_pretty(&observation).unwrap();
        let verified = verify_bytes(&receipt, transcript()).unwrap();
        assert_eq!(verified.observation_id, observation.observation_id);
    }

    #[test]
    fn projection_tampering_is_rejected_even_if_observation_id_is_retained() {
        let observation = observe_bytes(spec(), transcript()).unwrap();
        let mut stored = serde_json::to_value(&observation).unwrap();
        stored["opaque_stdout_line_count"] = Value::from(99_u64);
        let receipt = serde_json::to_vec(&stored).unwrap();
        assert!(verify_bytes(&receipt, transcript()).is_err());
    }

    #[test]
    fn transcript_substitution_is_rejected() {
        let observation = observe_bytes(spec(), transcript()).unwrap();
        let receipt = serde_json::to_vec(&observation).unwrap();
        let other = b"program output\n{\"reason\":\"build-finished\",\"success\":true}\n";
        assert!(verify_bytes(&receipt, other).is_err());
    }

    #[test]
    fn identity_field_tampering_is_rejected() {
        let observation = observe_bytes(spec(), transcript()).unwrap();
        let mut stored = serde_json::to_value(&observation).unwrap();
        stored["context_id"] = Value::String("44".repeat(32));
        let receipt = serde_json::to_vec(&stored).unwrap();
        assert!(verify_bytes(&receipt, transcript()).is_err());
    }
}
