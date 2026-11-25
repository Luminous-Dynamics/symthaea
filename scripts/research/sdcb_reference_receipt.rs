//! Emit a deterministic-structure JSON receipt for SDCB reference controls.
//!
//! Build from the repository root:
//!   rustc --edition=2024 scripts/research/sdcb_reference_receipt.rs -o /tmp/sdcb-reference-receipt
//!
//! Run:
//!   /tmp/sdcb-reference-receipt --source-revision "$(git rev-parse HEAD)" --compiler-version "$(rustc --version)"
//!
//! This receipt covers analytic reference controls only. It is not a benchmark
//! score, a model evaluation, or evidence about RH, cognition, or consciousness.

#[path = "sdcb_reference.rs"]
mod reference;

use reference::{
    adjacent_gap_ratios, classify_discrete_stability, continuous_time_hurwitz_2x2,
    deterministic_codeword, hamming_distance, input_regressor_rank_2d,
    majority_bundle, observability_rank_2x2, prime_sieve, select_discriminating_sensor,
    spectral_radius_2x2, symmetric_eigenvalues_2x2, xor_bind, xor_unbind,
    DiscreteStability, MatrixRank,
};
use std::env;
use std::process::{Command, ExitCode};
use std::time::{SystemTime, UNIX_EPOCH};

struct CaseResult {
    id: &'static str,
    passed: bool,
}

fn json_escape(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len() + 2);
    for character in value.chars() {
        match character {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            character if character.is_control() => {
                use std::fmt::Write;
                let _ = write!(escaped, "\\u{:04x}", character as u32);
            }
            character => escaped.push(character),
        }
    }
    escaped
}

fn parse_args() -> Result<(String, String), String> {
    let mut source_revision = None;
    let mut compiler_version = None;
    let mut args = env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--source-revision" => {
                source_revision = Some(
                    args.next()
                        .ok_or_else(|| "--source-revision requires a value".to_string())?,
                );
            }
            "--compiler-version" => {
                compiler_version = Some(
                    args.next()
                        .ok_or_else(|| "--compiler-version requires a value".to_string())?,
                );
            }
            "--help" | "-h" => {
                println!(
                    "usage: sdcb-reference-receipt --source-revision <40-hex-sha> --compiler-version <rustc-version>"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }

    let revision = source_revision
        .ok_or_else(|| "--source-revision is required".to_string())?;
    if revision.len() != 40 || !revision.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err("source revision must be a 40-character hexadecimal commit SHA".to_string());
    }
    let compiler = compiler_version
        .ok_or_else(|| "--compiler-version is required".to_string())?;
    if compiler.trim().is_empty() {
        return Err("compiler version must not be empty".to_string());
    }
    Ok((revision.to_ascii_lowercase(), compiler))
}


fn verify_git_provenance(source_revision: &str) -> Result<String, String> {
    let head = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .map_err(|error| format!("could not execute git rev-parse: {error}"))?;
    if !head.status.success() {
        return Err("git rev-parse HEAD failed; run from the repository root".to_string());
    }
    let head = String::from_utf8(head.stdout)
        .map_err(|_| "git rev-parse returned non-UTF-8 output".to_string())?
        .trim()
        .to_ascii_lowercase();
    if head.len() != 40 || !head.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err("git rev-parse did not return a 40-character commit SHA".to_string());
    }
    if head != source_revision {
        return Err(format!(
            "provided source revision does not match repository HEAD: provided={source_revision}, observed={head}"
        ));
    }

    let status = Command::new("git")
        .args(["status", "--porcelain", "--untracked-files=all"])
        .output()
        .map_err(|error| format!("could not execute git status: {error}"))?;
    if !status.status.success() {
        return Err("git status --porcelain failed".to_string());
    }
    if !status.stdout.is_empty() {
        let dirty = String::from_utf8_lossy(&status.stdout);
        return Err(format!(
            "working tree is not clean; refusing to emit a source-pinned receipt: {dirty}"
        ));
    }

    Ok(head)
}

fn build_cases() -> Vec<CaseResult> {
    let mut cases = Vec::new();

    let eigenvalues = symmetric_eigenvalues_2x2(2.0, 1.0, 2.0);
    cases.push(CaseResult {
        id: "symmetric_eigenvalues_2x2",
        passed: eigenvalues
            .map(|values| (values[0] - 1.0).abs() < 1e-12 && (values[1] - 3.0).abs() < 1e-12)
            .unwrap_or(false),
    });

    cases.push(CaseResult {
        id: "discrete_stable_control",
        passed: classify_discrete_stability(0.5, 0.0, 0.0, 0.25, 1e-12)
            == Ok(DiscreteStability::AsymptoticallyStable),
    });
    cases.push(CaseResult {
        id: "discrete_marginal_control",
        passed: classify_discrete_stability(1.0, 0.0, 0.0, 0.5, 1e-12)
            == Ok(DiscreteStability::Marginal),
    });
    cases.push(CaseResult {
        id: "discrete_unstable_control",
        passed: classify_discrete_stability(1.25, 0.0, 0.0, 0.5, 1e-12)
            == Ok(DiscreteStability::Unstable),
    });

    let complex_radius = spectral_radius_2x2(0.0, -0.9, 0.9, 0.0);
    cases.push(CaseResult {
        id: "complex_pair_spectral_radius",
        passed: complex_radius.map(|value| (value - 0.9).abs() < 1e-12).unwrap_or(false),
    });
    cases.push(CaseResult {
        id: "continuous_hurwitz_control",
        passed: continuous_time_hurwitz_2x2(-1.0, 0.0, 0.0, -2.0) == Ok(true)
            && continuous_time_hurwitz_2x2(1.0, 0.0, 0.0, -2.0) == Ok(false),
    });

    let gaps = adjacent_gap_ratios(&[0.0, 1.0, 3.0, 6.0]);
    cases.push(CaseResult {
        id: "adjacent_gap_ratio_reference",
        passed: gaps
            .as_ref()
            .map(|values| {
                values.len() == 2
                    && (values[0] - 0.5).abs() < 1e-12
                    && (values[1] - (2.0 / 3.0)).abs() < 1e-12
            })
            .unwrap_or(false),
    });
    let gaps_scaled = adjacent_gap_ratios(&[11.0, 13.0, 17.0, 23.0]);
    cases.push(CaseResult {
        id: "adjacent_gap_affine_invariance",
        passed: match (gaps, gaps_scaled) {
            (Ok(left), Ok(right)) => left
                .iter()
                .zip(right)
                .all(|(a, b)| (a - b).abs() < 1e-12),
            _ => false,
        },
    });
    cases.push(CaseResult {
        id: "invalid_gap_input_rejected",
        passed: adjacent_gap_ratios(&[0.0, 1.0, 1.0]).is_err()
            && adjacent_gap_ratios(&[0.0, 2.0, 1.0]).is_err(),
    });

    cases.push(CaseResult {
        id: "small_prime_sieve",
        passed: prime_sieve(30) == vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29],
    });

    let left = [1, 0, 1, 1, 0, 0, 1, 0];
    let right = [0, 1, 1, 0, 0, 1, 0, 1];
    let bound = xor_bind(&left, &right);
    cases.push(CaseResult {
        id: "hdc_xor_binding_roundtrip",
        passed: bound
            .map(|bound| {
                xor_unbind(&bound, &right) == Ok(left.to_vec())
                    && xor_unbind(&bound, &left) == Ok(right.to_vec())
            })
            .unwrap_or(false),
    });
    let a = [1, 0, 1, 0, 0, 1];
    let b = [1, 1, 0, 0, 1, 1];
    let c = [0, 1, 1, 1, 1, 0];
    cases.push(CaseResult {
        id: "hdc_majority_bundle",
        passed: majority_bundle(&[&a, &b, &c]) == Ok(vec![1, 1, 1, 0, 1, 1]),
    });
    cases.push(CaseResult {
        id: "hdc_hamming_distance",
        passed: hamming_distance(&[1, 0, 1], &[0, 0, 1]) == Ok(1),
    });
    let code_a = deterministic_codeword(42, 256);
    let code_b = deterministic_codeword(42, 256);
    let code_c = deterministic_codeword(43, 256);
    cases.push(CaseResult {
        id: "seeded_codeword_reproducibility",
        passed: code_a.is_ok() && code_a == code_b && code_a != code_c,
    });

    cases.push(CaseResult {
        id: "observability_hidden_mode_unobservable",
        passed: observability_rank_2x2(0.5, 0.0, 0.0, 0.2, 1.0, 0.0, 1e-12)
            == Ok(MatrixRank::One),
    });
    cases.push(CaseResult {
        id: "observability_full_rank_positive_control",
        passed: observability_rank_2x2(0.0, 1.0, -1.0, 0.0, 1.0, 0.0, 1e-12)
            == Ok(MatrixRank::Two),
    });
    cases.push(CaseResult {
        id: "observability_unit_scale_invariance",
        passed: observability_rank_2x2(0.0, 1.0, -1.0, 0.0, 1.0, 0.0, 1e-12)
            == observability_rank_2x2(0.0, 1.0, -1.0, 0.0, 1e-8, 0.0, 1e-12),
    });
    cases.push(CaseResult {
        id: "input_excitation_rank_deficient",
        passed: input_regressor_rank_2d(&[[1.0, 0.0], [2.0, 0.0]], 1e-12)
            == Ok(MatrixRank::One),
    });
    cases.push(CaseResult {
        id: "input_excitation_full_rank_positive_control",
        passed: input_regressor_rank_2d(&[[1.0, 0.0], [0.0, 1.0]], 1e-12)
            == Ok(MatrixRank::Two),
    });
    cases.push(CaseResult {
        id: "input_excitation_unit_scale_invariance",
        passed: input_regressor_rank_2d(&[[1.0, 0.0], [0.0, 1.0]], 1e-12)
            == input_regressor_rank_2d(&[[1e-8, 0.0], [0.0, 1e-8]], 1e-12),
    });

    let unobservable_a =
        reference::diagonal_observation_trace(0.5, 0.2, 1.0, 1.0, 1.0, 0.0, 8);
    let unobservable_b =
        reference::diagonal_observation_trace(0.5, 0.9, 1.0, 1.0, 1.0, 0.0, 8);
    cases.push(CaseResult {
        id: "same_observations_different_hidden_dynamics",
        passed: unobservable_a.is_ok() && unobservable_a == unobservable_b,
    });

    let sensor_choice = select_discriminating_sensor(
        0.5,
        &[0.2, 0.9],
        [1.0, 1.0],
        &[[1.0, 0.0], [0.0, 5.0]],
        8,
        1e-12,
    );
    cases.push(CaseResult {
        id: "active_sensor_selects_discriminating_measurement",
        passed: sensor_choice
            .map(|choice| choice.map(|selection| selection.candidate_index) == Some(1))
            .unwrap_or(false),
    });
    let no_sensor = select_discriminating_sensor(
        0.5,
        &[0.2, 0.9],
        [1.0, 1.0],
        &[[1.0, 0.0]],
        8,
        1e-12,
    );
    cases.push(CaseResult {
        id: "active_sensor_abstains_without_discrimination",
        passed: no_sensor == Ok(None),
    });
    let sensor_gain_invariant = select_discriminating_sensor(
        0.5,
        &[0.2, 0.9],
        [1.0, 1.0],
        &[[0.0, 1.0], [0.0, 10.0]],
        8,
        1e-12,
    );
    cases.push(CaseResult {
        id: "active_sensor_gain_normalization",
        passed: sensor_gain_invariant
            .map(|selection| selection.map(|choice| choice.candidate_index) == Some(0))
            .unwrap_or(false),
    });

    cases
}

fn main() -> ExitCode {
    let (source_revision, compiler_version) = match parse_args() {
        Ok(values) => values,
        Err(error) => {
            eprintln!("SDCB_RECEIPT_ARGUMENT_ERROR: {error}");
            return ExitCode::from(2);
        }
    };
    let observed_repository_head = match verify_git_provenance(&source_revision) {
        Ok(head) => head,
        Err(error) => {
            eprintln!("SDCB_RECEIPT_PROVENANCE_ERROR: {error}");
            return ExitCode::from(2);
        }
    };
    let rustflags = env::var("RUSTFLAGS").unwrap_or_default();

    let created_unix_seconds = match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(duration) => duration.as_secs(),
        Err(error) => {
            eprintln!("system clock precedes Unix epoch: {error}");
            return ExitCode::from(2);
        }
    };

    let cases = build_cases();
    let total_cases = cases.len();
    let passed_cases = cases.iter().filter(|case| case.passed).count();
    let failed_cases = total_cases - passed_cases;
    let overall_status = if failed_cases == 0 { "passed" } else { "failed" };

    println!("{{");
    println!("  \"schema_version\": \"1.0.0\",");
    println!("  \"receipt_type\": \"sdcb_reference_control_run_v1\",");
    println!("  \"source_revision\": \"{}\",", json_escape(&source_revision));
    println!("  \"source_revision_provenance\": \"observed_git_head_matches_clean_worktree\",");
    println!("  \"repository_head\": \"{}\",", observed_repository_head);
    println!("  \"git_working_tree_clean\": true,");
    println!("  \"build_command\": \"rustc --edition=2024 scripts/research/sdcb_reference_receipt.rs\",");
    println!("  \"rustflags\": \"{}\",", json_escape(&rustflags));
    println!("  \"compiler_version\": \"{}\",", json_escape(&compiler_version));
    println!("  \"operating_system\": \"{}\",", json_escape(env::consts::OS));
    println!("  \"architecture\": \"{}\",", json_escape(env::consts::ARCH));
    println!("  \"created_unix_seconds\": {created_unix_seconds},");
    println!("  \"total_cases\": {total_cases},");
    println!("  \"passed_cases\": {passed_cases},");
    println!("  \"failed_cases\": {failed_cases},");
    println!("  \"overall_status\": \"{overall_status}\",");
    println!("  \"claim_ceiling\": \"analytic_reference_controls_only\",");
    println!("  \"cases\": [");
    for (index, case) in cases.iter().enumerate() {
        let comma = if index + 1 == total_cases { "" } else { "," };
        let status = if case.passed { "passed" } else { "failed" };
        println!(
            "    {{\"case_id\": \"{}\", \"status\": \"{}\"}}{}",
            case.id, status, comma
        );
    }
    println!("  ]");
    println!("}}");

    if failed_cases == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}
