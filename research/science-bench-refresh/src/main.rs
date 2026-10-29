use science_bench_refresh_validator::{
    audit_policy_input_overlap, compute_artifact_commitment, compute_protocol_commitment,
    generate_task_set_bytes, sha256_hex, validate_manifest_bytes, verify_regeneration,
    verify_reveal,
};
use serde_json::json;
use std::env;
use std::fs;
use std::process::ExitCode;

fn usage(program: &str) {
    eprintln!(
        concat!(
            "usage:\n",
            "  {program} MANIFEST.json [--require-scoring|--require-sealed]\n",
            "  {program} audit-overlap LEFT_LABEL LEFT_TASK_SET.json RIGHT_LABEL RIGHT_TASK_SET.json\n",
            "  {program} protocol-commit MANIFEST.json\n",
            "  {program} artifact-commit MANIFEST.json seed|task-set ARTIFACT_FILE\n",
            "  {program} verify-reveal MANIFEST.json SEED_FILE TASK_SET_FILE\n",
            "  {program} generate-task-set MANIFEST.json SEED_FILE\n",
            "    GENERATOR_CONFIG.json OUTPUT.json\n",
            "  {program} verify-regeneration MANIFEST.json SEED_FILE\n",
            "    GENERATOR_CONFIG.json TASK_SET.json"
        )
    );
}

fn read(path: &str, label: &str) -> Result<Vec<u8>, ExitCode> {
    fs::read(path).map_err(|error| {
        eprintln!("cannot read {label} {path}: {error}");
        ExitCode::from(2)
    })
}

fn run(args: &[String]) -> ExitCode {
    if args.is_empty() {
        return ExitCode::from(2);
    }

    match args[0].as_str() {
        "audit-overlap" => {
            if args.len() != 5 {
                usage("science-bench-refresh-validator");
                return ExitCode::from(2);
            }
            let left_bytes = match read(&args[2], "left task set") {
                Ok(bytes) => bytes,
                Err(code) => return code,
            };
            let right_bytes = match read(&args[4], "right task set") {
                Ok(bytes) => bytes,
                Err(code) => return code,
            };
            let report = match audit_policy_input_overlap(
                &args[1],
                &left_bytes,
                &args[3],
                &right_bytes,
            ) {
                Ok(report) => report,
                Err(error) => {
                    eprintln!("{error}");
                    return ExitCode::from(2);
                }
            };
            println!(
                "{}",
                serde_json::to_string_pretty(&report)
                    .expect("overlap audit report serializes")
            );
            if report.exact_overlap_pair_count > 0 {
                ExitCode::from(6)
            } else if report.near_duplicate_candidate_pair_count > 0 {
                // Similarity is not a confirmed leak, but qualification must
                // pause until a reviewer resolves all numeric-near candidates.
                ExitCode::from(7)
            } else {
                ExitCode::SUCCESS
            }
        }
        "protocol-commit" => {
            if args.len() != 2 {
                usage("science-bench-refresh-validator");
                return ExitCode::from(2);
            }
            let bytes = match read(&args[1], "manifest") {
                Ok(bytes) => bytes,
                Err(code) => return code,
            };
            match compute_protocol_commitment(&bytes) {
                Ok(commitment) => {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&json!({
                            "schema": "symthaea.science_bench_refresh_protocol_commitment.v1",
                            "manifest_sha256": sha256_hex(&bytes),
                            "protocol_commitment_sha256": commitment
                        }))
                        .expect("commitment report serializes")
                    );
                    ExitCode::SUCCESS
                }
                Err(error) => {
                    eprintln!("{error}");
                    ExitCode::from(2)
                }
            }
        }
        "artifact-commit" => {
            if args.len() != 4 {
                usage("science-bench-refresh-validator");
                return ExitCode::from(2);
            }
            let manifest_bytes = match read(&args[1], "manifest") {
                Ok(bytes) => bytes,
                Err(code) => return code,
            };
            let artifact_bytes = match read(&args[3], "artifact") {
                Ok(bytes) => bytes,
                Err(code) => return code,
            };
            match compute_artifact_commitment(&manifest_bytes, &args[2], &artifact_bytes) {
                Ok(commitment) => {
                    let protocol = match compute_protocol_commitment(&manifest_bytes) {
                        Ok(value) => value,
                        Err(error) => {
                            eprintln!("{error}");
                            return ExitCode::from(2);
                        }
                    };
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&json!({
                            "schema": "symthaea.science_bench_refresh_artifact_commitment.v1",
                            "kind": args[2],
                            "protocol_commitment_sha256": protocol,
                            "artifact_bytes": artifact_bytes.len(),
                            "artifact_commitment_sha256": commitment
                        }))
                        .expect("artifact commitment report serializes")
                    );
                    ExitCode::SUCCESS
                }
                Err(error) => {
                    eprintln!("{error}");
                    ExitCode::from(2)
                }
            }
        }
        "generate-task-set" => {
            if args.len() != 5 {
                usage("science-bench-refresh-validator");
                return ExitCode::from(2);
            }
            let manifest_bytes = match read(&args[1], "manifest") {
                Ok(bytes) => bytes,
                Err(code) => return code,
            };
            let seed_bytes = match read(&args[2], "revealed seed") {
                Ok(bytes) => bytes,
                Err(code) => return code,
            };
            let config_bytes = match read(&args[3], "generator configuration") {
                Ok(bytes) => bytes,
                Err(code) => return code,
            };
            let task_set_bytes =
                match generate_task_set_bytes(&manifest_bytes, &seed_bytes, &config_bytes) {
                    Ok(bytes) => bytes,
                    Err(error) => {
                        eprintln!("{error}");
                        return ExitCode::from(2);
                    }
                };
            if let Err(error) = fs::write(&args[4], &task_set_bytes) {
                eprintln!("cannot write generated task set {}: {error}", args[4]);
                return ExitCode::from(2);
            }
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "schema": "symthaea.science_bench_refresh_generation_result.v1",
                    "output_path": args[4],
                    "task_set_bytes": task_set_bytes.len(),
                    "task_set_sha256": sha256_hex(&task_set_bytes)
                }))
                .expect("generation report serializes")
            );
            ExitCode::SUCCESS
        }
        "verify-regeneration" => {
            if args.len() != 5 {
                usage("science-bench-refresh-validator");
                return ExitCode::from(2);
            }
            let manifest_bytes = match read(&args[1], "manifest") {
                Ok(bytes) => bytes,
                Err(code) => return code,
            };
            let seed_bytes = match read(&args[2], "revealed seed") {
                Ok(bytes) => bytes,
                Err(code) => return code,
            };
            let config_bytes = match read(&args[3], "generator configuration") {
                Ok(bytes) => bytes,
                Err(code) => return code,
            };
            let task_set_bytes = match read(&args[4], "committed task set") {
                Ok(bytes) => bytes,
                Err(code) => return code,
            };
            let report = match verify_regeneration(
                &manifest_bytes,
                &seed_bytes,
                &config_bytes,
                &task_set_bytes,
            ) {
                Ok(report) => report,
                Err(error) => {
                    eprintln!("{error}");
                    return ExitCode::from(2);
                }
            };
            println!(
                "{}",
                serde_json::to_string_pretty(&report)
                    .expect("regeneration report serializes")
            );
            if report.verified {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(5)
            }
        }

        "verify-reveal" => {
            if args.len() != 4 {
                usage("science-bench-refresh-validator");
                return ExitCode::from(2);
            }
            let manifest_bytes = match read(&args[1], "manifest") {
                Ok(bytes) => bytes,
                Err(code) => return code,
            };
            let seed_bytes = match read(&args[2], "revealed seed") {
                Ok(bytes) => bytes,
                Err(code) => return code,
            };
            let task_set_bytes = match read(&args[3], "revealed task-set artifact") {
                Ok(bytes) => bytes,
                Err(code) => return code,
            };
            let report = match verify_reveal(&manifest_bytes, &seed_bytes, &task_set_bytes) {
                Ok(report) => report,
                Err(error) => {
                    eprintln!("{error}");
                    return ExitCode::from(2);
                }
            };
            println!(
                "{}",
                serde_json::to_string_pretty(&report)
                    .expect("reveal verification report serializes")
            );
            if report.verified {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(5)
            }
        }
        _ => run_manifest_mode(args),
    }
}

fn run_manifest_mode(args: &[String]) -> ExitCode {
    if args.is_empty() || args.len() > 2 {
        usage("science-bench-refresh-validator");
        return ExitCode::from(2);
    }

    let mode = if args.len() == 2 {
        match args[1].as_str() {
            "--require-scoring" => "scoring",
            "--require-sealed" => "sealed",
            other => {
                eprintln!("unknown option: {other}");
                usage("science-bench-refresh-validator");
                return ExitCode::from(2);
            }
        }
    } else {
        "report"
    };

    let bytes = match read(&args[0], "manifest") {
        Ok(bytes) => bytes,
        Err(code) => return code,
    };

    let report = match validate_manifest_bytes(&bytes) {
        Ok(report) => report,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::from(2);
        }
    };

    println!(
        "{}",
        serde_json::to_string_pretty(&report).expect("validation report serializes")
    );

    if !report.valid {
        return ExitCode::from(2);
    }
    if mode == "scoring" && !report.manifest_declares_scoring_eligible {
        return ExitCode::from(3);
    }
    if mode == "sealed" && !report.manifest_declares_sealed_generalization_profile {
        return ExitCode::from(4);
    }
    ExitCode::SUCCESS
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    run(&args)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_args_are_rejected() {
        assert_eq!(run(&[]), ExitCode::from(2));
    }
}
