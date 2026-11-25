use claim_evidence_integrity_benchmark::{self_test, validate_pack};
use std::process::ExitCode;

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let first = args.next();
    if first.as_deref() == Some("--self-test") {
        return match self_test() {
            Ok(report) => {
                println!(
                    "PASS: {} fixtures, {} evidence artifacts, {} verified SHA-256 digests",
                    report.fixture_count, report.evidence_count, report.verified_digest_count
                );
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("FAIL: {error}");
                ExitCode::FAILURE
            }
        };
    }

    let path = first.unwrap_or_else(|| "fixtures/claim-evidence-integrity-v0.1.json".to_owned());
    let input = match std::fs::read_to_string(&path) {
        Ok(input) => input,
        Err(error) => {
            eprintln!("cannot read {path}: {error}");
            return ExitCode::FAILURE;
        }
    };

    match validate_pack(&input) {
        Ok(report) => {
            println!(
                "PASS: {} fixtures, {} evidence artifacts, {} verified SHA-256 digests",
                report.fixture_count, report.evidence_count, report.verified_digest_count
            );
            ExitCode::SUCCESS
        }
        Err(issues) => {
            for issue in issues {
                eprintln!("{}: {}", issue.path, issue.message);
            }
            ExitCode::FAILURE
        }
    }
}
