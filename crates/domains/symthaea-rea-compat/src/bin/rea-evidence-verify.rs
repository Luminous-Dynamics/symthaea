#![forbid(unsafe_code)]

//! Offline verifier for saved REA Evidence JSON records.
//!
//! This binary performs bounded local file reading, strict raw JSON admission,
//! analysis-profile commitment verification, and semantic Evidence ID checking.
//! It never authenticates the producer or verifies that the described observation
//! actually occurred.

use serde_json::{Value, json};
use std::env;
use std::ffi::OsStr;
use std::fs::File;
use std::io::{self, Read};
use std::path::PathBuf;
use std::process::ExitCode;
use symthaea_rea_compat::{
    EvidenceError, EvidenceErrorKind, MAX_EVIDENCE_JSON_BYTES, VerifiedEvidence,
    verify_evidence_json,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputFormat {
    Text,
    Json,
}

#[derive(Debug, PartialEq, Eq)]
struct Arguments {
    format: OutputFormat,
    path: PathBuf,
}

#[derive(Debug, PartialEq, Eq)]
enum ArgumentError {
    Help,
    Invalid(String),
}

#[derive(Debug)]
enum CliError {
    Io(String),
    Evidence(EvidenceError),
}

impl CliError {
    fn category(&self) -> &'static str {
        match self {
            Self::Io(_) => "input_io_error",
            Self::Evidence(error) => match error.kind {
                EvidenceErrorKind::InvalidJson => "invalid_json",
                EvidenceErrorKind::SchemaViolation => "schema_violation",
                EvidenceErrorKind::Canonicalization => "canonicalization_error",
                EvidenceErrorKind::IdentityMismatch => "identity_mismatch",
                EvidenceErrorKind::InputTooLarge => "input_too_large",
            },
        }
    }

    fn message(&self) -> &str {
        match self {
            Self::Io(message) => message,
            Self::Evidence(error) => &error.message,
        }
    }

    fn exit_code(&self) -> u8 {
        match self {
            Self::Io(_) => 66,
            Self::Evidence(_) => 65,
        }
    }
}

fn parse_arguments<I>(arguments: I) -> Result<Arguments, ArgumentError>
where
    I: IntoIterator<Item = std::ffi::OsString>,
{
    let mut format = OutputFormat::Text;
    let mut path = None;
    let mut literal_paths = false;

    for argument in arguments {
        if !literal_paths && argument == OsStr::new("--") {
            literal_paths = true;
            continue;
        }
        if !literal_paths && argument == OsStr::new("--help") {
            return Err(ArgumentError::Help);
        }
        if !literal_paths && argument == OsStr::new("--json") {
            format = OutputFormat::Json;
            continue;
        }
        if !literal_paths && argument == OsStr::new("--text") {
            format = OutputFormat::Text;
            continue;
        }
        if !literal_paths
            && argument
                .to_str()
                .is_some_and(|value| value.starts_with('-'))
        {
            return Err(ArgumentError::Invalid(format!(
                "unrecognized option: {}",
                argument.to_string_lossy()
            )));
        }
        if path.is_some() {
            return Err(ArgumentError::Invalid(
                "exactly one Evidence JSON path must be supplied".to_owned(),
            ));
        }
        path = Some(PathBuf::from(argument));
    }

    let path = path.ok_or_else(|| {
        ArgumentError::Invalid("missing Evidence JSON path; use --help for usage".to_owned())
    })?;

    Ok(Arguments { format, path })
}

fn read_bounded<R: Read>(reader: R, max_bytes: usize) -> Result<Vec<u8>, CliError> {
    let cap = u64::try_from(max_bytes)
        .unwrap_or(u64::MAX)
        .saturating_add(1);
    let mut limited = reader.take(cap);
    let mut bytes = Vec::new();
    limited
        .read_to_end(&mut bytes)
        .map_err(|error| CliError::Io(format!("failed to read Evidence input: {error}")))?;
    if bytes.len() > max_bytes {
        return Err(CliError::Evidence(EvidenceError {
            kind: EvidenceErrorKind::InputTooLarge,
            message: format!(
                "Evidence JSON exceeds the {} byte input limit",
                max_bytes
            ),
        }));
    }
    Ok(bytes)
}

fn verify_path(path: &PathBuf) -> Result<(VerifiedEvidence, usize), CliError> {
    let file = File::open(path).map_err(|error| {
        CliError::Io(format!("cannot open Evidence input: {error}"))
    })?;
    let bytes = read_bounded(file, MAX_EVIDENCE_JSON_BYTES)?;
    let byte_len = bytes.len();
    let verified = verify_evidence_json(&bytes).map_err(CliError::Evidence)?;
    Ok((verified, byte_len))
}

fn coverage_status(value: &Value) -> Value {
    value
        .get("relationship_coverage")
        .and_then(|coverage| coverage.get("status"))
        .cloned()
        .unwrap_or(Value::Null)
}

fn success_json(verified: &VerifiedEvidence, input_bytes: usize) -> Value {
    json!({
        "schema": "luminous.rea-evidence-verification-result.v1",
        "status": "semantic_identity_verified",
        "evidence_id": verified.evidence_id,
        "semantic_sha256": verified.semantic_sha256,
        "raw_file_sha256": verified.raw_file_sha256,
        "input_bytes": input_bytes,
        "subject_present": !verified.value["subject"].is_null(),
        "analysis_profile_present": !verified.value["analysis_profile"].is_null(),
        "relationship_coverage": coverage_status(&verified.value),
        "claim_ceiling": {
            "producer_authenticated": false,
            "observation_truth_verified": false,
            "hardware_attestation_verified": false,
            "runtime_integrity_verified": false,
            "security_compliance_verified": false
        }
    })
}

fn error_json(error: &CliError) -> Value {
    json!({
        "schema": "luminous.rea-evidence-verification-result.v1",
        "status": "rejected",
        "error": {
            "category": error.category(),
            "message": error.message()
        },
        "claim_ceiling": {
            "verification_succeeded": false,
            "producer_authenticated": false,
            "observation_truth_verified": false,
            "hardware_attestation_verified": false,
            "runtime_integrity_verified": false,
            "security_compliance_verified": false
        }
    })
}

fn success_text(verified: &VerifiedEvidence, input_bytes: usize) -> String {
    format!(
        concat!(
            "REA Evidence semantic identity verified\n",
            "Evidence ID: {}\n",
            "Semantic SHA-256: {}\n",
            "Raw-file SHA-256: {}\n",
            "Input bytes: {}\n",
            "Relationship coverage: {}\n",
            "Producer authentication: not performed\n",
            "Observation truth: not established\n",
            "Hardware attestation: not verified\n",
            "Runtime integrity: not verified\n",
            "Security compliance: not certified\n"
        ),
        verified.evidence_id,
        verified.semantic_sha256,
        verified.raw_file_sha256.as_deref().unwrap_or("unavailable"),
        input_bytes,
        coverage_status(&verified.value)
            .as_str()
            .unwrap_or("unknown/not supplied")
    )
}

fn usage() -> &'static str {
    concat!(
        "Usage: rea-evidence-verify [--json|--text] <evidence.json>\n",
        "\n",
        "Verify a saved REA Evidence record offline.\n",
        "This checks schema, analysis-profile commitment, and semantic Evidence ID.\n",
        "It does not authenticate the producer or establish observation truth.\n",
        "\n",
        "Options:\n",
        "  --json   Emit machine-readable JSON\n",
        "  --text   Emit human-readable text (default)\n",
        "  --help   Show this help\n"
    )
}

fn run(arguments: impl IntoIterator<Item = std::ffi::OsString>) -> ExitCode {
    let parsed = match parse_arguments(arguments) {
        Ok(arguments) => arguments,
        Err(ArgumentError::Help) => {
            print!("{}", usage());
            return ExitCode::SUCCESS;
        }
        Err(ArgumentError::Invalid(message)) => {
            eprintln!("usage error: {message}");
            eprint!("{}", usage());
            return ExitCode::from(64);
        }
    };

    match verify_path(&parsed.path) {
        Ok((verified, input_bytes)) => {
            match parsed.format {
                OutputFormat::Json => {
                    println!("{}", success_json(&verified, input_bytes));
                }
                OutputFormat::Text => {
                    print!("{}", success_text(&verified, input_bytes));
                }
            }
            ExitCode::SUCCESS
        }
        Err(error) => {
            match parsed.format {
                OutputFormat::Json => println!("{}", error_json(&error)),
                OutputFormat::Text => {
                    eprintln!("REA Evidence verification rejected");
                    eprintln!("Category: {}", error.category());
                    eprintln!("Reason: {}", error.message());
                }
            }
            ExitCode::from(error.exit_code())
        }
    }
}

fn main() -> ExitCode {
    run(env::args_os().skip(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requires_exactly_one_path() {
        assert!(matches!(
            parse_arguments(Vec::<std::ffi::OsString>::new()),
            Err(ArgumentError::Invalid(_))
        ));
        let args = ["one.json", "two.json"]
            .into_iter()
            .map(std::ffi::OsString::from);
        assert!(matches!(
            parse_arguments(args),
            Err(ArgumentError::Invalid(_))
        ));
    }

    #[test]
    fn supports_explicit_json_text_and_literal_path_modes() {
        let json_args = ["--json", "evidence.json"]
            .into_iter()
            .map(std::ffi::OsString::from);
        assert_eq!(
            parse_arguments(json_args).unwrap(),
            Arguments {
                format: OutputFormat::Json,
                path: PathBuf::from("evidence.json")
            }
        );

        let literal_args = ["--", "--json"]
            .into_iter()
            .map(std::ffi::OsString::from);
        assert_eq!(
            parse_arguments(literal_args).unwrap(),
            Arguments {
                format: OutputFormat::Text,
                path: PathBuf::from("--json")
            }
        );
    }

    #[test]
    fn bounded_reader_stops_at_limit_plus_one() {
        let input = vec![b'x'; 11];
        let error = read_bounded(input.as_slice(), 10).unwrap_err();
        assert_eq!(error.category(), "input_too_large");

        let accepted = read_bounded(&b"12345"[..], 5).unwrap();
        assert_eq!(accepted, b"12345");
    }

    #[test]
    fn json_success_output_does_not_disclose_local_paths_or_result_payloads() {
        let fixture: Value =
            serde_json::from_str(include_str!("../tests/fixtures/rea-evidence-v1.json"))
                .unwrap();
        let evidence = symthaea_rea_compat::verify_evidence_value(&fixture).unwrap();
        let rendered = success_json(&evidence, 123).to_string();
        for forbidden in ["/tmp/sample.apk", "classes.dex", "normalized_result", "raw_result"] {
            assert!(!rendered.contains(forbidden));
        }
        assert_eq!(
            success_json(&evidence, 123)["status"],
            Value::String("semantic_identity_verified".to_owned())
        );
    }
}
