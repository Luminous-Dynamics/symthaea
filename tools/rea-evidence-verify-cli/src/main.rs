use std::env;
use std::ffi::OsString;
use std::fs::File;
use std::io::{self, Read};
use std::process::ExitCode;

use serde_json::{Value, json};
use symthaea_rea_compat::{
    EvidenceError, EvidenceErrorKind, MAX_EVIDENCE_JSON_BYTES, VerifiedEvidence,
    verify_evidence_json,
};

const EXIT_USAGE: u8 = 2;
const EXIT_IO: u8 = 3;
const EXIT_INVALID_JSON: u8 = 4;
const EXIT_SCHEMA: u8 = 5;
const EXIT_IDENTITY: u8 = 6;
const EXIT_TOO_LARGE: u8 = 7;
const EXIT_CANONICALIZATION: u8 = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputFormat {
    Json,
    Text,
}

#[derive(Debug, PartialEq, Eq)]
struct Arguments {
    format: OutputFormat,
    input: OsString,
}

fn parse_arguments<I>(arguments: I) -> Result<Arguments, String>
where
    I: IntoIterator<Item = OsString>,
{
    let mut format = OutputFormat::Text;
    let mut input: Option<OsString> = None;
    let mut saw_format = false;

    for argument in arguments {
        if argument == "--json" {
            if saw_format {
                return Err("choose exactly one output format".to_owned());
            }
            format = OutputFormat::Json;
            saw_format = true;
        } else if argument == "--text" {
            if saw_format {
                return Err("choose exactly one output format".to_owned());
            }
            format = OutputFormat::Text;
            saw_format = true;
        } else if argument == "--help" || argument == "-h" {
            return Err("usage: rea-evidence-verify [--json|--text] <evidence.json>".to_owned());
        } else if argument.to_string_lossy().starts_with('-') {
            return Err("unknown option".to_owned());
        } else if input.replace(argument).is_some() {
            return Err("exactly one Evidence file is required".to_owned());
        }
    }

    let input = input
        .ok_or_else(|| "usage: rea-evidence-verify [--json|--text] <evidence.json>".to_owned())?;

    Ok(Arguments { format, input })
}

fn read_bounded<R: Read>(mut reader: R, maximum: usize) -> io::Result<Vec<u8>> {
    let mut bytes = Vec::with_capacity(maximum.min(64 * 1024));
    let mut limited = (&mut reader).take((maximum as u64).saturating_add(1));
    limited.read_to_end(&mut bytes)?;
    Ok(bytes)
}

fn error_code(kind: EvidenceErrorKind) -> &'static str {
    match kind {
        EvidenceErrorKind::InvalidJson => "invalid_json",
        EvidenceErrorKind::SchemaViolation => "schema_violation",
        EvidenceErrorKind::Canonicalization => "canonicalization_error",
        EvidenceErrorKind::IdentityMismatch => "identity_mismatch",
        EvidenceErrorKind::InputTooLarge => "input_too_large",
    }
}

fn exit_code(kind: EvidenceErrorKind) -> u8 {
    match kind {
        EvidenceErrorKind::InvalidJson => EXIT_INVALID_JSON,
        EvidenceErrorKind::SchemaViolation => EXIT_SCHEMA,
        EvidenceErrorKind::Canonicalization => EXIT_CANONICALIZATION,
        EvidenceErrorKind::IdentityMismatch => EXIT_IDENTITY,
        EvidenceErrorKind::InputTooLarge => EXIT_TOO_LARGE,
    }
}

fn provider_summary(value: &Value) -> Value {
    match value.get("provider").and_then(Value::as_object) {
        Some(provider) => json!({
            "id": provider.get("id").and_then(Value::as_str),
            "name": provider.get("name").and_then(Value::as_str),
            "version": provider.get("version").cloned().unwrap_or(Value::Null),
        }),
        None => Value::Null,
    }
}

fn subject_summary(value: &Value) -> Value {
    match value.get("subject") {
        Some(Value::Object(subject)) => json!({
            "format": subject.get("format").and_then(Value::as_str),
            "architecture": subject.get("architecture").cloned().unwrap_or(Value::Null),
            "sha256": subject
                .get("digest")
                .and_then(|digest| digest.get("sha256"))
                .and_then(Value::as_str),
        }),
        _ => Value::Null,
    }
}

fn coverage_summary(value: &Value) -> Value {
    match value.get("relationship_coverage") {
        Some(Value::Object(coverage)) => json!({
            "status": coverage.get("status").and_then(Value::as_str),
            "limitations_count": coverage
                .get("limitations")
                .and_then(Value::as_array)
                .map(Vec::len)
                .unwrap_or(0),
            "diagnostics_count": coverage
                .get("diagnostics")
                .and_then(Value::as_array)
                .map(Vec::len)
                .unwrap_or(0),
        }),
        Some(Value::Null) | None => Value::Null,
        _ => Value::Null,
    }
}

fn verification_report(verified: &VerifiedEvidence) -> Value {
    let profile_digest = verified
        .value
        .get("analysis_profile")
        .filter(|profile| !profile.is_null())
        .and_then(|profile| symthaea_rea_compat::analysis_profile_digest(profile).ok());

    json!({
        "schema": "luminous.rea-evidence-verification-report.v1",
        "result": "semantic_consistency_verified",
        "evidence_id": verified.evidence_id,
        "semantic_sha256": verified.semantic_sha256,
        "raw_file_sha256": verified.raw_file_sha256,
        "analysis_profile_digest": profile_digest,
        "provider": provider_summary(&verified.value),
        "subject": subject_summary(&verified.value),
        "confidence": verified.value.get("confidence").and_then(Value::as_str),
        "authority": verified.value.get("authority").and_then(Value::as_str),
        "relationship_coverage": coverage_summary(&verified.value),
        "claim_ceiling": {
            "producer_authenticated": false,
            "observation_truth_established": false,
            "hardware_attestation_verified": false,
            "runtime_integrity_verified": false,
            "security_compliance_verified": false,
            "meaning": "This report verifies local schema and semantic-digest consistency only."
        }
    })
}

fn text_report(report: &Value) -> String {
    let provider = report["provider"]["id"].as_str().unwrap_or("unknown");
    let format = report["subject"]["format"].as_str().unwrap_or("no-subject");
    let architecture = report["subject"]["architecture"]
        .as_str()
        .unwrap_or("unspecified");
    let coverage = report["relationship_coverage"]["status"]
        .as_str()
        .unwrap_or("not-declared");

    format!(
        concat!(
            "REA Evidence: verified semantic consistency\n",
            "Evidence ID: {}\n",
            "Semantic SHA-256: {}\n",
            "Raw file SHA-256: {}\n",
            "Analysis profile SHA-256: {}\n",
            "Provider: {}\n",
            "Subject: {} ({})\n",
            "Relationship coverage: {}\n",
            "\n",
            "Claim ceiling: producer authenticity, observation truth, hardware attestation, ",
            "runtime integrity and security compliance are NOT established.\n"
        ),
        report["evidence_id"].as_str().unwrap_or(""),
        report["semantic_sha256"].as_str().unwrap_or(""),
        report["raw_file_sha256"].as_str().unwrap_or(""),
        report["analysis_profile_digest"]
            .as_str()
            .unwrap_or("not-present"),
        provider,
        format,
        architecture,
        coverage
    )
}

fn print_error(format: OutputFormat, code: &str, message: &str) {
    match format {
        OutputFormat::Json => {
            let output = json!({
                "schema": "luminous.rea-evidence-verification-report.v1",
                "result": "rejected",
                "error": {
                    "code": code,
                    "message": message
                }
            });
            eprintln!(
                "{}",
                serde_json::to_string(&output).unwrap_or_else(|_| {
                    "{\"result\":\"rejected\",\"error\":{\"code\":\"internal_error\"}}".to_owned()
                })
            );
        }
        OutputFormat::Text => {
            eprintln!("{code}: {message}");
        }
    }
}

fn run(arguments: Arguments) -> u8 {
    let file = match File::open(&arguments.input) {
        Ok(file) => file,
        Err(_) => {
            print_error(arguments.format, "io_error", "unable to read Evidence file");
            return EXIT_IO;
        }
    };

    let bytes = match read_bounded(file, MAX_EVIDENCE_JSON_BYTES) {
        Ok(bytes) => bytes,
        Err(_) => {
            print_error(arguments.format, "io_error", "unable to read Evidence file");
            return EXIT_IO;
        }
    };

    if bytes.len() > MAX_EVIDENCE_JSON_BYTES {
        print_error(
            arguments.format,
            "input_too_large",
            "Evidence file exceeds the 64 MiB input limit",
        );
        return EXIT_TOO_LARGE;
    }

    let verified = match verify_evidence_json(&bytes) {
        Ok(verified) => verified,
        Err(error) => {
            print_error(
                arguments.format,
                error_code(error.kind),
                safe_error_message(&error),
            );
            return exit_code(error.kind);
        }
    };

    let report = verification_report(&verified);
    match arguments.format {
        OutputFormat::Json => {
            println!(
                "{}",
                serde_json::to_string_pretty(&report)
                    .unwrap_or_else(|_| "{\"result\":\"verified\"}".to_owned())
            );
        }
        OutputFormat::Text => {
            print!("{}", text_report(&report));
        }
    }
    0
}

fn safe_error_message(error: &EvidenceError) -> &str {
    match error.kind {
        EvidenceErrorKind::InvalidJson => "Evidence JSON is malformed or contains duplicate keys",
        EvidenceErrorKind::SchemaViolation => "Evidence does not satisfy the supported schema",
        EvidenceErrorKind::Canonicalization => "Evidence cannot be canonicalized",
        EvidenceErrorKind::IdentityMismatch => {
            "Evidence semantic identity does not match its content"
        }
        EvidenceErrorKind::InputTooLarge => "Evidence file exceeds the 64 MiB input limit",
    }
}

fn main() -> ExitCode {
    let parsed = parse_arguments(env::args_os().skip(1));
    let arguments = match parsed {
        Ok(arguments) => arguments,
        Err(message) => {
            eprintln!("{message}");
            return ExitCode::from(EXIT_USAGE);
        }
    };

    ExitCode::from(run(arguments))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn fixture() -> VerifiedEvidence {
        let raw = include_bytes!(
            "../../../crates/domains/symthaea-rea-compat/tests/fixtures/rea-evidence-v1.json"
        );
        verify_evidence_json(raw).unwrap()
    }

    #[test]
    fn parser_accepts_one_format_and_one_path() {
        let args =
            parse_arguments([OsString::from("--json"), OsString::from("evidence.json")]).unwrap();
        assert_eq!(args.format, OutputFormat::Json);
        assert_eq!(args.input, OsString::from("evidence.json"));
    }

    #[test]
    fn parser_rejects_duplicate_format_and_multiple_paths() {
        assert!(
            parse_arguments([
                OsString::from("--json"),
                OsString::from("--text"),
                OsString::from("evidence.json"),
            ])
            .is_err()
        );

        assert!(
            parse_arguments([OsString::from("one.json"), OsString::from("two.json"),]).is_err()
        );
    }

    #[test]
    fn bounded_reader_never_reads_past_limit_plus_one() {
        let input = Cursor::new(b"0123456789abcdef".to_vec());
        let bytes = read_bounded(input, 8).unwrap();
        assert_eq!(bytes, b"012345678");
    }

    #[test]
    fn report_excludes_paths_and_raw_normalized_payloads() {
        let report = verification_report(&fixture());
        let serialized = serde_json::to_string(&report).unwrap();
        assert!(!serialized.contains("/tmp/sample.apk"));
        assert!(!serialized.contains("classes.dex"));
        assert!(!serialized.contains("raw_result"));
        assert!(!serialized.contains("normalized_result"));
        assert!(!serialized.contains("local_path"));
        assert!(report["claim_ceiling"]["producer_authenticated"] == false);
    }

    #[test]
    fn text_report_separates_raw_and_semantic_digests_and_claim_ceiling() {
        let report = verification_report(&fixture());
        let text = text_report(&report);
        assert!(text.contains("Evidence ID: ev_"));
        assert!(text.contains("Semantic SHA-256:"));
        assert!(text.contains("Raw file SHA-256:"));
        assert!(text.contains("NOT established"));
    }

    #[test]
    fn error_codes_are_stable_and_distinct() {
        assert_eq!(error_code(EvidenceErrorKind::InvalidJson), "invalid_json");
        assert_eq!(
            error_code(EvidenceErrorKind::SchemaViolation),
            "schema_violation"
        );
        assert_eq!(
            error_code(EvidenceErrorKind::IdentityMismatch),
            "identity_mismatch"
        );
        assert_eq!(
            error_code(EvidenceErrorKind::InputTooLarge),
            "input_too_large"
        );
        assert_eq!(
            exit_code(EvidenceErrorKind::IdentityMismatch),
            EXIT_IDENTITY
        );
    }
}
