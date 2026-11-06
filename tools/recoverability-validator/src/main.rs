use science_bench_refresh_validator::{
    compute_recoverability_manifest_commitment, verify_recoverability_manifest,
};
use serde_json::json;
use std::env;
use std::fs;
use std::process::ExitCode;

fn usage(program: &str) {
    eprintln!(
        "Usage:\n  {program} commit-recoverability-manifest MANIFEST_FIELDS.json\n  {program} audit-recoverability-manifest MANIFEST.json"
    );
}

fn read_file(path: &str) -> Result<Vec<u8>, ExitCode> {
    fs::read(path).map_err(|error| {
        eprintln!("cannot read {path}: {error}");
        ExitCode::from(2)
    })
}

fn run(args: &[String]) -> ExitCode {
    if args.len() != 2 {
        usage("science-bench-refresh-validator");
        return ExitCode::from(2);
    }

    match args[0].as_str() {
        "commit-recoverability-manifest" => {
            let bytes = match read_file(&args[1]) {
                Ok(bytes) => bytes,
                Err(code) => return code,
            };
            match compute_recoverability_manifest_commitment(&bytes) {
                Ok(commitment) => {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&json!({
                            "schema": "symthaea.science_bench_refresh_recoverability_commitment.v1",
                            "manifest_commitment_sha256": commitment
                        }))
                        .expect("commitment report serializes")
                    );
                    ExitCode::SUCCESS
                }
                Err(errors) => {
                    for error in errors {
                        eprintln!("{error}");
                    }
                    ExitCode::from(2)
                }
            }
        }
        "audit-recoverability-manifest" => {
            let bytes = match read_file(&args[1]) {
                Ok(bytes) => bytes,
                Err(code) => return code,
            };
            let report = match verify_recoverability_manifest(&bytes) {
                Ok(report) => report,
                Err(errors) => {
                    for error in errors {
                        eprintln!("{error}");
                    }
                    return ExitCode::from(2);
                }
            };
            println!(
                "{}",
                serde_json::to_string_pretty(&report).expect("audit report serializes")
            );
            if report.valid {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(11)
            }
        }
        _ => {
            usage("science-bench-refresh-validator");
            ExitCode::from(2)
        }
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    run(&args)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_arguments_are_rejected() {
        assert_eq!(run(&[]), ExitCode::from(2));
    }

    #[test]
    fn unknown_command_is_rejected() {
        assert_eq!(
            run(&["unexpected".to_string(), "file.json".to_string()]),
            ExitCode::from(2)
        );
    }
}
