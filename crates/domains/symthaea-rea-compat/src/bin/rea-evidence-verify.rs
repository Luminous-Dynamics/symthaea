//! Offline command-line admission and semantic verification for saved REA Evidence.
//!
//! This command verifies only internal schema/profile/Evidence identity
//! consistency. It does not authenticate the provider or establish observation
//! truth, hardware identity, runtime integrity, or security compliance.

use serde_json::{Value, json};
use std::env;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use symthaea_rea_compat::{
    EvidenceError, MAX_EVIDENCE_JSON_BYTES, verify_evidence_json,
};

const REPORT_SCHEMA: &str = "luminous.rea-evidence-verification-report.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputFormat {
    Json,
    Text,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Options {
    input: PathBuf,
    output: OutputFormat,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum CliError {
    Usage(String),
    Io(String),
}

fn usage() -> &'static str {
    "Usage: rea-evidence-verify [--json|--text] <evidence.json>\n\
     \n\
     Verify a saved REA Evidence record offline. The input is bounded to 64 MiB.\n\
     A valid result establishes internal semantic consistency only; it does not\n\
     authenticate the producer or prove that the observation occurred.\n"
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Option<Options>, CliError> {
    let mut input = None;
    let mut output = OutputFormat::Json;
    let mut iter = args.into_iter();

    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--help" | "-h" => return Ok(None),
            "--json" => output = OutputFormat::Json,
            "--text" => output = OutputFormat::Text,
            "--" => {
                if input.is_some() {
                    return Err(CliError::Usage("only one input path is accepted".to_owned()));
                }
                let Some(path) = iter.next() else {
                    return Err(CliError::Usage("missing input path after --".to_owned()));
                };
                input = Some(PathBuf::from(path));
                if iter.next().is_some() {
                    return Err(CliError::Usage("unexpected argument after input path".to_owned()));
                }
                break;
            }
            value if value.starts_with('-') => {
                return Err(CliError::Usage(format!("unknown option: {value}")));
            }
            value => {
                if input.replace(PathBuf::from(value)).is_some() {
                    return Err(CliError::Usage("only one input path is accepted".to_owned()));
                }
            }
        }
    }

    let Some(input) = input else {
        return Err(CliError::Usage("missing input path".to_owned()));
    };
    Ok(Some(Options { input, output }))
}

fn read_bounded(path: &Path) -> Result<Vec<u8>, CliError> {
    let file = File::open(path).map_err(|error| {
        CliError::Io(format!("could not open input file: {error}"))
    })?;
    let metadata = file.metadata().map_err(|error| {
        CliError::Io(format!("could not inspect input file: {error}"))
    })?;
    let limit = MAX_EVIDENCE_JSON_BYTES as u64;
    if metadata.len() > limit {
        return Err(CliError::Io(format!(
            "input file is {} bytes; limit is {} bytes",
            metadata.len(),
            MAX_EVIDENCE_JSON_BYTES
        )));
    }

    // Bound the read independently of metadata so a file that grows after
    // stat cannot cause an unbounded allocation.
    let mut bytes = Vec::new();
    file.take(limit + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| CliError::Io(format!("could not read input file: {error}")))?;
    if bytes.len() > MAX_EVIDENCE_JSON_BYTES {
        return Err(CliError::Io(format!(
            "input exceeded the {} byte read limit",
            MAX_EVIDENCE_JSON_BYTES
        )));
    }
    Ok(bytes)
}

fn claim_ceiling() -> Value {
    json!({
        "producer_authenticated": false,
        "observation_truth_established": false,
        "hardware_attestation_verified": false,
        "runtime_integrity_verified": false,
        "security_compliance_established": false
    })
}

fn verify_bytes(input: &[u8]) -> Result<Value, EvidenceError> {
    let verified = verify_evidence_json(input)?;
    let value = &verified.value;
    let subject_sha256 = value
        .get("subject")
        .and_then(|subject| subject.get("digest"))
        .and_then(|digest| digest.get("sha256"))
        .and_then(Value::as_str);
    let provider_id = value
        .get("provider")
        .and_then(|provider| provider.get("id"))
        .and_then(Value::as_str);
    let provider_version = value
        .get("provider")
        .and_then(|provider| provider.get("version"))
        .cloned()
        .unwrap_or(Value::Null);
    let profile_digest = value
        .get("analysis_profile")
        .and_then(|profile| profile.get("digest"))
        .and_then(Value::as_str);
    let coverage = value
        .get("relationship_coverage")
        .and_then(|coverage| coverage.get("status"))
        .and_then(Value::as_str);

    Ok(json!({
        "schema": REPORT_SCHEMA,
        "result": "internally_consistent",
        "evidence": {
            "evidence_id": verified.evidence_id,
            "semantic_sha256": verified.semantic_sha256,
            "raw_file_sha256": verified.raw_file_sha256,
            "subject_sha256": subject_sha256,
            "provider_id": provider_id,
            "provider_version": provider_version,
            "analysis_profile_digest": profile_digest,
            "predicate_type": value.get("predicate_type"),
            "operation": value.get("operation"),
            "confidence": value.get("confidence"),
            "authority": value.get("authority"),
            "relationship_coverage": coverage
        },
        "claim_ceiling": claim_ceiling()
    }))
}

fn rejected_report(message: &str, kind: &str) -> Value {
    json!({
        "schema": REPORT_SCHEMA,
        "result": "rejected",
        "error": {
            "kind": kind,
            "message": message
        },
        "claim_ceiling": claim_ceiling()
    })
}

fn render(value: &Value, output: OutputFormat) -> String {
    match output {
        OutputFormat::Json => serde_json::to_string_pretty(value)
            .unwrap_or_else(|_| "{\"result\":\"report_serialization_failed\"}".to_owned()),
        OutputFormat::Text => {
            let mut lines = Vec::new();
            lines.push(format!(
                "result: {}",
                value.get("result").and_then(Value::as_str).unwrap_or("unknown")
            ));
            if let Some(evidence) = value.get("evidence") {
                for key in [
                    "evidence_id",
                    "semantic_sha256",
                    "raw_file_sha256",
                    "subject_sha256",
                    "provider_id",
                    "provider_version",
                    "analysis_profile_digest",
                    "operation",
                    "confidence",
                    "authority",
                    "relationship_coverage",
                ] {
                    if let Some(item) = evidence.get(key) {
                        lines.push(format!("{key}: {}", item));
                    }
                }
            }
            if let Some(error) = value.get("error") {
                lines.push(format!(
                    "error: {}: {}",
                    error.get("kind").and_then(Value::as_str).unwrap_or("unknown"),
                    error.get("message").and_then(Value::as_str).unwrap_or("no details")
                ));
            }
            lines.push("producer_authenticated: false".to_owned());
            lines.push("observation_truth_established: false".to_owned());
            lines.join("\n")
        }
    }
}

fn run(options: Options) -> ExitCode {
    let bytes = match read_bounded(&options.input) {
        Ok(bytes) => bytes,
        Err(error) => {
            let report = match error {
                CliError::Io(message) => rejected_report(&message, "input_io_or_size"),
                CliError::Usage(message) => rejected_report(&message, "usage"),
            };
            println!("{}", render(&report, options.output));
            return ExitCode::from(2);
        }
    };

    match verify_bytes(&bytes) {
        Ok(report) => {
            println!("{}", render(&report, options.output));
            ExitCode::SUCCESS
        }
        Err(error) => {
            let report = rejected_report(&error.message, &format!("{:?}", error.kind));
            println!("{}", render(&report, options.output));
            ExitCode::from(1)
        }
    }
}

fn main() -> ExitCode {
    match parse_args(env::args().skip(1)) {
        Ok(None) => {
            print!("{}", usage());
            ExitCode::SUCCESS
        }
        Ok(Some(options)) => run(options),
        Err(CliError::Usage(message)) => {
            eprintln!("{message}\n{}", usage());
            ExitCode::from(2)
        }
        Err(CliError::Io(message)) => {
            eprintln!("{message}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fixture() -> &'static [u8] {
        include_bytes!("../../tests/fixtures/rea-evidence-v1.json")
    }

    #[test]
    fn valid_fixture_report_separates_semantic_and_raw_file_digests() {
        let report = verify_bytes(fixture()).unwrap();
        assert_eq!(report["result"], "internally_consistent");
        assert_eq!(report["evidence"]["evidence_id"], "ev_a4e49bdca4cbf80c2ec8e4d7eba116af403bc33a4937ffb7fca00b6aab6590c4");
        assert_ne!(
            report["evidence"]["raw_file_sha256"],
            report["evidence"]["semantic_sha256"]
        );
        assert_eq!(report["claim_ceiling"]["producer_authenticated"], false);
        assert_eq!(report["claim_ceiling"]["observation_truth_established"], false);
    }

    #[test]
    fn report_does_not_emit_local_subject_paths_or_raw_observation_payloads() {
        let report = verify_bytes(fixture()).unwrap();
        let text = report.to_string();
        assert!(!text.contains("/tmp/sample.apk"));
        assert!(!text.contains("classes.dex"));
    }

    #[test]
    fn tampered_semantic_field_produces_rejection_report() {
        let mut value: Value = serde_json::from_slice(fixture()).unwrap();
        value["normalized_result"]["entries"][0]["size"] = json!(99);
        let bytes = serde_json::to_vec(&value).unwrap();
        let error = verify_evidence_json_for_report(&bytes).unwrap_err();
        assert_eq!(error.kind, "IdentityMismatch");
        assert!(error.message.contains("semantic ID mismatch"));
    }

    fn verify_evidence_json_for_report(bytes: &[u8]) -> Result<Value, ReportError> {
        match verify_bytes(bytes) {
            Ok(report) => Ok(report),
            Err(error) => Err(ReportError {
                kind: format!("{:?}", error.kind),
                message: error.message,
            }),
        }
    }

    #[derive(Debug)]
    struct ReportError {
        kind: String,
        message: String,
    }

    #[test]
    fn parser_rejects_unknown_options_and_multiple_input_paths() {
        assert!(matches!(
            parse_args(["--unsafe".to_owned(), "x.json".to_owned()]),
            Err(CliError::Usage(_))
        ));
        assert!(matches!(
            parse_args(["a.json".to_owned(), "b.json".to_owned()]),
            Err(CliError::Usage(_))
        ));
    }

    #[test]
    fn parser_defaults_to_json_and_accepts_explicit_text() {
        assert_eq!(
            parse_args(["evidence.json".to_owned()]).unwrap(),
            Some(Options {
                input: PathBuf::from("evidence.json"),
                output: OutputFormat::Json,
            })
        );
        assert_eq!(
            parse_args(["--text".to_owned(), "evidence.json".to_owned()])
                .unwrap()
                .unwrap()
                .output,
            OutputFormat::Text
        );
    }

    #[test]
    fn help_does_not_require_an_input_path() {
        assert_eq!(
            parse_args(["--help".to_owned()]).unwrap(),
            None
        );
    }
}
