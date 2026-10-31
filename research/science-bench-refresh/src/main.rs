use science_bench_refresh_validator::counterfactual_evidence::{
    compute_evidence_intervention_plan_commitment, verify_evidence_intervention_plan,
};
use science_bench_refresh_validator::evidence_behavior_contracts::{
    compute_evidence_behavior_contract_plan_commitment,
    compute_evidence_behavior_contract_report_commitment, verify_evidence_behavior_contract_plan,
    verify_evidence_behavior_contract_report,
};
use science_bench_refresh_validator::evidence_outcomes::{
    compute_evidence_outcome_report_commitment, verify_evidence_outcome_report,
};
use science_bench_refresh_validator::exact_protected_matcher::{
    ProtectedSearchArtifactIndexCommitmentInput, compute_protected_index_commitment,
    match_search_exposure_exact, verify_search_exposure_exact_match_report,
};
use science_bench_refresh_validator::judge_factorial::{
    analyze_judge_factorial, compute_judge_factorial_plan_commitment, verify_judge_factorial_plan,
};
use science_bench_refresh_validator::judge_robustness::{
    compute_judge_robustness_plan_commitment, compute_judge_robustness_report_commitment,
    verify_judge_robustness_plan, verify_judge_robustness_report,
};
use science_bench_refresh_validator::intervention_quality::{
    analyze_intervention_quality, compute_intervention_quality_plan_commitment,
};
use science_bench_refresh_validator::intervention_target_delta::{
    analyze_intervention_target_delta, compute_target_delta_plan_commitment,
};
use science_bench_refresh_validator::leakage_audit::audit_policy_input_overlap;
use science_bench_refresh_validator::lineage_audit::audit_lineage_overlap;
use science_bench_refresh_validator::paired_outcome_analysis::analyze_paired_evidence_outcomes;
use science_bench_refresh_validator::reference_assurance::{
    compute_reference_package_commitment, verify_reference_assurance_package,
};
use science_bench_refresh_validator::search_exposure::verify_search_exposure_receipt;
use science_bench_refresh_validator::search_exposure_match::verify_search_exposure_match_report;
use science_bench_refresh_validator::{
    compute_artifact_commitment, compute_protocol_commitment, generate_task_set_bytes, sha256_hex,
    validate_manifest_bytes, verify_regeneration, verify_reveal,
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
            "  {program} audit-lineage LEFT_TASK_SET.json LEFT_LEDGER.json\n",
            "    RIGHT_TASK_SET.json RIGHT_LEDGER.json\n",
            "  {program} verify-search-receipt RECEIPT.json\n",
            "  {program} verify-search-matches RECEIPT.json MATCH_REPORT.json\n",
            "  {program} commit-evidence-intervention-plan PLAN_FIELDS.json\n",
            "  {program} verify-evidence-intervention-plan PLAN.json\n",
            "  {program} commit-evidence-outcomes PLAN.json OUTCOME_FIELDS.json\n",
            "  {program} verify-evidence-outcomes PLAN.json OUTCOME_REPORT.json\n",
            "  {program} commit-evidence-behavior-contract-plan PLAN_FIELDS.json\n",
            "  {program} verify-evidence-behavior-contract-plan PLAN.json\n",
            "  {program} commit-evidence-behavior-contract-report PLAN.json REPORT_FIELDS.json\n",
            "  {program} verify-evidence-behavior-contract-report PLAN.json REPORT.json\n",
            "  {program} commit-judge-robustness-plan PLAN_FIELDS.json\n",
            "  {program} verify-judge-robustness-plan PLAN.json\n",
            "  {program} commit-judge-robustness-report PLAN.json REPORT_FIELDS.json\n",
            "  {program} verify-judge-robustness-report PLAN.json REPORT.json\n",
            "  {program} commit-intervention-quality-plan PLAN_FIELDS.json\n",
            "  {program} analyze-intervention-quality PLAN.json RECEIPT.json\n",
            "  {program} commit-intervention-target-delta-plan PLAN_FIELDS.json QUALITY_PLAN.json\n",
            "  {program} analyze-intervention-target-delta QUALITY_PLAN.json QUALITY_RECEIPT.json TARGET_PLAN.json TARGET_RECEIPT.json\n",
            "  {program} commit-judge-factorial-plan PLAN_FIELDS.json\n",
            "  {program} verify-judge-factorial-plan PLAN.json\n",
            "  {program} analyze-judge-factorial PLAN.json RECEIPT.json\n",
            "  {program} commit-reference-assurance-package-fields PACKAGE_FIELDS.json\n",
            "  {program} verify-reference-assurance-package PACKAGE.json\n",
            "  {program} analyze-paired-evidence-outcomes PLAN.json OUTCOME_REPORT.json ANALYSIS_PLAN.json\n",
            "  {program} commit-protected-index INDEX_FIELDS.json\n",
            "  {program} match-search-exact RECEIPT.json PROTECTED_INDEX.json\n",
            "  {program} verify-search-exact-matches RECEIPT.json\n",
            "    PROTECTED_INDEX.json MATCH_REPORT.json\n",
            "  {program} protocol-commit MANIFEST.json\n",
            "  {program} artifact-commit MANIFEST.json seed|task-set ARTIFACT_FILE\n",
            "  {program} verify-reveal MANIFEST.json SEED_FILE TASK_SET_FILE\n",
            "  {program} generate-task-set MANIFEST.json SEED_FILE\n",
            "    GENERATOR_CONFIG.json OUTPUT.json\n",
            "  {program} verify-regeneration MANIFEST.json SEED_FILE\n",
            "    GENERATOR_CONFIG.json TASK_SET.json"
        ),
        program = program
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
        "commit-judge-factorial-plan" => {
            if args.len() != 2 {
                usage("science-bench-refresh-validator");
                return ExitCode::from(2);
            }
            let fields_bytes = match read(&args[1], "judge factorial plan fields") {
                Ok(bytes) => bytes,
                Err(code) => return code,
            };
            match compute_judge_factorial_plan_commitment(&fields_bytes) {
                Ok(commitment) => {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&json!({
                            "schema": "symthaea.science_bench_refresh_judge_factorial_plan_commitment.v1",
                            "plan_commitment_sha256": commitment
                        }))
                        .expect("judge factorial plan commitment serializes")
                    );
                    ExitCode::SUCCESS
                }
                Err(error) => {
                    eprintln!("{error}");
                    ExitCode::from(2)
                }
            }
        }
        "verify-judge-factorial-plan" => {
            if args.len() != 2 {
                usage("science-bench-refresh-validator");
                return ExitCode::from(2);
            }
            let plan_bytes = match read(&args[1], "judge factorial plan") {
                Ok(bytes) => bytes,
                Err(code) => return code,
            };
            match verify_judge_factorial_plan(&plan_bytes) {
                Ok((plan, plan_sha256)) => {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&json!({
                            "schema": "symthaea.science_bench_refresh_judge_factorial_plan_verification.v1",
                            "valid": true,
                            "plan_sha256": plan_sha256,
                            "plan_commitment_sha256": plan.plan_commitment_sha256,
                            "analyzer_source_revision": plan.analyzer_source_revision,
                            "study_id": plan.study_id,
                            "planned_block_count": plan.blocks.len(),
                            "planned_condition_count": plan.blocks.iter().map(|block| block.conditions.len()).sum::<usize>(),
                            "repeats_per_condition": plan.repeats_per_condition,
                            "factor_count": 5,
                            "factorial_cells_per_repeat": 32,
                            "scoring_eligible": false,
                            "external_anchor_required": true
                        }))
                        .expect("judge factorial plan verification serializes")
                    );
                    ExitCode::SUCCESS
                }
                Err(error) => {
                    eprintln!("{error}");
                    ExitCode::from(2)
                }
            }
        }
        "analyze-judge-factorial" => {
            if args.len() != 3 {
                usage("science-bench-refresh-validator");
                return ExitCode::from(2);
            }
            let plan_bytes = match read(&args[1], "judge factorial plan") {
                Ok(bytes) => bytes,
                Err(code) => return code,
            };
            let receipt_bytes = match read(&args[2], "judge factorial receipt") {
                Ok(bytes) => bytes,
                Err(code) => return code,
            };
            match analyze_judge_factorial(&plan_bytes, &receipt_bytes) {
                Ok(report) => {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&report)
                            .expect("judge factorial analysis report serializes")
                    );
                    ExitCode::SUCCESS
                }
                Err(error) => {
                    eprintln!("{error}");
                    ExitCode::from(2)
                }
            }
        }
        "commit-intervention-target-delta-plan" => {
            if args.len() != 3 {
                usage("science-bench-refresh-validator");
                return ExitCode::from(2);
            }
            let fields_bytes = match read(&args[1], "target-delta plan fields") {
                Ok(bytes) => bytes,
                Err(code) => return code,
            };
            let quality_plan_bytes = match read(&args[2], "parent intervention quality plan") {
                Ok(bytes) => bytes,
                Err(code) => return code,
            };
            match compute_target_delta_plan_commitment(&fields_bytes, &quality_plan_bytes) {
                Ok(commitment) => {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&json!({
                            "schema": "symthaea.science_bench_refresh_intervention_target_delta_plan_commitment.v1",
                            "plan_commitment_sha256": commitment
                        }))
                        .expect("target-delta plan commitment serializes")
                    );
                    ExitCode::SUCCESS
                }
                Err(error) => {
                    eprintln!("{error}");
                    ExitCode::from(2)
                }
            }
        }
        "analyze-intervention-target-delta" => {
            if args.len() != 5 {
                usage("science-bench-refresh-validator");
                return ExitCode::from(2);
            }
            let quality_plan_bytes = match read(&args[1], "parent intervention quality plan") {
                Ok(bytes) => bytes,
                Err(code) => return code,
            };
            let quality_receipt_bytes = match read(&args[2], "parent intervention quality receipt") {
                Ok(bytes) => bytes,
                Err(code) => return code,
            };
            let target_plan_bytes = match read(&args[3], "target-delta plan") {
                Ok(bytes) => bytes,
                Err(code) => return code,
            };
            let target_receipt_bytes = match read(&args[4], "target-delta receipt") {
                Ok(bytes) => bytes,
                Err(code) => return code,
            };
            match analyze_intervention_target_delta(
                &quality_plan_bytes,
                &quality_receipt_bytes,
                &target_plan_bytes,
                &target_receipt_bytes,
            ) {
                Ok(report) => {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&report)
                            .expect("target-delta report serializes")
                    );
                    ExitCode::SUCCESS
                }
                Err(error) => {
                    eprintln!("{error}");
                    ExitCode::from(2)
                }
            }
        }
        "commit-intervention-quality-plan" => {
            if args.len() != 2 {
                usage("science-bench-refresh-validator");
                return ExitCode::from(2);
            }
            let fields_bytes = match read(&args[1], "intervention quality plan fields") {
                Ok(bytes) => bytes,
                Err(code) => return code,
            };
            match compute_intervention_quality_plan_commitment(&fields_bytes) {
                Ok(commitment) => {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&json!({
                            "schema": "symthaea.science_bench_refresh_intervention_quality_plan_commitment.v1",
                            "plan_commitment_sha256": commitment
                        }))
                        .expect("intervention quality plan commitment serializes")
                    );
                    ExitCode::SUCCESS
                }
                Err(error) => {
                    eprintln!("{error}");
                    ExitCode::from(2)
                }
            }
        }
        "analyze-intervention-quality" => {
            if args.len() != 3 {
                usage("science-bench-refresh-validator");
                return ExitCode::from(2);
            }
            let plan_bytes = match read(&args[1], "intervention quality plan") {
                Ok(bytes) => bytes,
                Err(code) => return code,
            };
            let receipt_bytes = match read(&args[2], "intervention quality receipt") {
                Ok(bytes) => bytes,
                Err(code) => return code,
            };
            match analyze_intervention_quality(&plan_bytes, &receipt_bytes) {
                Ok(report) => {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&report)
                            .expect("intervention quality analysis report serializes")
                    );
                    ExitCode::SUCCESS
                }
                Err(error) => {
                    eprintln!("{error}");
                    ExitCode::from(2)
                }
            }
        }
        "commit-judge-robustness-plan" => {
            if args.len() != 2 {
                usage("science-bench-refresh-validator");
                return ExitCode::from(2);
            }
            let fields_bytes = match read(&args[1], "judge robustness plan fields") {
                Ok(bytes) => bytes,
                Err(code) => return code,
            };
            match compute_judge_robustness_plan_commitment(&fields_bytes) {
                Ok(commitment) => {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&json!({
                            "schema": "symthaea.science_bench_refresh_judge_plan_commitment.v1",
                            "plan_commitment_sha256": commitment
                        }))
                        .expect("judge robustness plan commitment serializes")
                    );
                    ExitCode::SUCCESS
                }
                Err(error) => {
                    eprintln!("{error}");
                    ExitCode::from(2)
                }
            }
        }
        "verify-judge-robustness-plan" => {
            if args.len() != 2 {
                usage("science-bench-refresh-validator");
                return ExitCode::from(2);
            }
            let plan_bytes = match read(&args[1], "judge robustness plan") {
                Ok(bytes) => bytes,
                Err(code) => return code,
            };
            let report = match verify_judge_robustness_plan(&plan_bytes) {
                Ok(report) => report,
                Err(error) => {
                    eprintln!("{error}");
                    return ExitCode::from(2);
                }
            };
            println!(
                "{}",
                serde_json::to_string_pretty(&report)
                    .expect("judge robustness plan report serializes")
            );
            if report.valid {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(11)
            }
        }
        "commit-judge-robustness-report" => {
            if args.len() != 3 {
                usage("science-bench-refresh-validator");
                return ExitCode::from(2);
            }
            let plan_bytes = match read(&args[1], "judge robustness plan") {
                Ok(bytes) => bytes,
                Err(code) => return code,
            };
            let fields_bytes = match read(&args[2], "judge robustness report fields") {
                Ok(bytes) => bytes,
                Err(code) => return code,
            };
            match compute_judge_robustness_report_commitment(&plan_bytes, &fields_bytes) {
                Ok(commitment) => {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&json!({
                            "schema": "symthaea.science_bench_refresh_judge_report_commitment.v1",
                            "report_commitment_sha256": commitment
                        }))
                        .expect("judge robustness report commitment serializes")
                    );
                    ExitCode::SUCCESS
                }
                Err(error) => {
                    eprintln!("{error}");
                    ExitCode::from(2)
                }
            }
        }
        "verify-judge-robustness-report" => {
            if args.len() != 3 {
                usage("science-bench-refresh-validator");
                return ExitCode::from(2);
            }
            let plan_bytes = match read(&args[1], "judge robustness plan") {
                Ok(bytes) => bytes,
                Err(code) => return code,
            };
            let report_bytes = match read(&args[2], "judge robustness report") {
                Ok(bytes) => bytes,
                Err(code) => return code,
            };
            let report = match verify_judge_robustness_report(&plan_bytes, &report_bytes) {
                Ok(report) => report,
                Err(error) => {
                    eprintln!("{error}");
                    return ExitCode::from(2);
                }
            };
            println!(
                "{}",
                serde_json::to_string_pretty(&report).expect("judge robustness report serializes")
            );
            if report.valid {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(11)
            }
        }
        "commit-evidence-behavior-contract-plan" => {
            if args.len() != 2 {
                usage("science-bench-refresh-validator");
                return ExitCode::from(2);
            }
            let fields_bytes = match read(&args[1], "evidence behavior contract plan fields") {
                Ok(bytes) => bytes,
                Err(code) => return code,
            };
            match compute_evidence_behavior_contract_plan_commitment(&fields_bytes) {
                Ok(commitment) => {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&json!({
                            "schema": "symthaea.science_bench_refresh_evidence_behavior_contract_plan_commitment.v1",
                            "plan_commitment_sha256": commitment
                        }))
                        .expect("evidence behavior contract plan commitment serializes")
                    );
                    ExitCode::SUCCESS
                }
                Err(error) => {
                    eprintln!("{error}");
                    ExitCode::from(2)
                }
            }
        }
        "verify-evidence-behavior-contract-plan" => {
            if args.len() != 2 {
                usage("science-bench-refresh-validator");
                return ExitCode::from(2);
            }
            let plan_bytes = match read(&args[1], "evidence behavior contract plan") {
                Ok(bytes) => bytes,
                Err(code) => return code,
            };
            let report = match verify_evidence_behavior_contract_plan(&plan_bytes) {
                Ok(report) => report,
                Err(error) => {
                    eprintln!("{error}");
                    return ExitCode::from(2);
                }
            };
            println!(
                "{}",
                serde_json::to_string_pretty(&report)
                    .expect("evidence behavior contract plan report serializes")
            );
            if report.valid {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(11)
            }
        }
        "commit-evidence-behavior-contract-report" => {
            if args.len() != 3 {
                usage("science-bench-refresh-validator");
                return ExitCode::from(2);
            }
            let plan_bytes = match read(&args[1], "evidence behavior contract plan") {
                Ok(bytes) => bytes,
                Err(code) => return code,
            };
            let fields_bytes = match read(&args[2], "evidence behavior contract report fields") {
                Ok(bytes) => bytes,
                Err(code) => return code,
            };
            match compute_evidence_behavior_contract_report_commitment(&plan_bytes, &fields_bytes) {
                Ok(commitment) => {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&json!({
                            "schema": "symthaea.science_bench_refresh_evidence_behavior_contract_report_commitment.v1",
                            "report_commitment_sha256": commitment
                        }))
                        .expect("evidence behavior contract report commitment serializes")
                    );
                    ExitCode::SUCCESS
                }
                Err(error) => {
                    eprintln!("{error}");
                    ExitCode::from(2)
                }
            }
        }
        "verify-evidence-behavior-contract-report" => {
            if args.len() != 3 {
                usage("science-bench-refresh-validator");
                return ExitCode::from(2);
            }
            let plan_bytes = match read(&args[1], "evidence behavior contract plan") {
                Ok(bytes) => bytes,
                Err(code) => return code,
            };
            let report_bytes = match read(&args[2], "evidence behavior contract report") {
                Ok(bytes) => bytes,
                Err(code) => return code,
            };
            let report = match verify_evidence_behavior_contract_report(&plan_bytes, &report_bytes)
            {
                Ok(report) => report,
                Err(error) => {
                    eprintln!("{error}");
                    return ExitCode::from(2);
                }
            };
            println!(
                "{}",
                serde_json::to_string_pretty(&report)
                    .expect("evidence behavior contract report serializes")
            );
            if report.valid {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(11)
            }
        }
        "analyze-paired-evidence-outcomes" => {
            if args.len() != 4 {
                usage("science-bench-refresh-validator");
                return ExitCode::from(2);
            }
            let plan_bytes = match read(&args[1], "counterfactual evidence plan") {
                Ok(bytes) => bytes,
                Err(code) => return code,
            };
            let outcome_bytes = match read(&args[2], "evidence outcome report") {
                Ok(bytes) => bytes,
                Err(code) => return code,
            };
            let analysis_bytes = match read(&args[3], "paired analysis plan") {
                Ok(bytes) => bytes,
                Err(code) => return code,
            };
            let verification = match analyze_paired_evidence_outcomes(
                &plan_bytes,
                &outcome_bytes,
                &analysis_bytes,
            ) {
                Ok(report) => report,
                Err(error) => {
                    eprintln!("{error}");
                    return ExitCode::from(2);
                }
            };
            println!(
                "{}",
                serde_json::to_string_pretty(&verification)
                    .expect("paired analysis verification serializes")
            );
            if verification.valid {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(11)
            }
        }
        "commit-evidence-outcomes" => {
            if args.len() != 3 {
                usage("science-bench-refresh-validator");
                return ExitCode::from(2);
            }
            let plan_bytes = match read(&args[1], "counterfactual evidence plan") {
                Ok(bytes) => bytes,
                Err(code) => return code,
            };
            let fields_bytes = match read(&args[2], "evidence outcome fields") {
                Ok(bytes) => bytes,
                Err(code) => return code,
            };
            match compute_evidence_outcome_report_commitment(&plan_bytes, &fields_bytes) {
                Ok(commitment) => {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&json!({
                            "schema": "symthaea.science_bench_refresh_evidence_outcome_commitment.v1",
                            "report_commitment_sha256": commitment
                        }))
                        .expect("evidence outcome commitment serializes")
                    );
                    ExitCode::SUCCESS
                }
                Err(error) => {
                    eprintln!("{error}");
                    ExitCode::from(2)
                }
            }
        }
        "verify-evidence-outcomes" => {
            if args.len() != 3 {
                usage("science-bench-refresh-validator");
                return ExitCode::from(2);
            }
            let plan_bytes = match read(&args[1], "counterfactual evidence plan") {
                Ok(bytes) => bytes,
                Err(code) => return code,
            };
            let report_bytes = match read(&args[2], "evidence outcome report") {
                Ok(bytes) => bytes,
                Err(code) => return code,
            };
            let report = match verify_evidence_outcome_report(&plan_bytes, &report_bytes) {
                Ok(report) => report,
                Err(error) => {
                    eprintln!("{error}");
                    return ExitCode::from(2);
                }
            };
            println!(
                "{}",
                serde_json::to_string_pretty(&report)
                    .expect("evidence outcome verification report serializes")
            );
            if report.valid {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(11)
            }
        }
        "commit-evidence-intervention-plan" => {
            if args.len() != 2 {
                usage("science-bench-refresh-validator");
                return ExitCode::from(2);
            }
            let fields_bytes = match read(&args[1], "counterfactual evidence plan fields") {
                Ok(bytes) => bytes,
                Err(code) => return code,
            };
            match compute_evidence_intervention_plan_commitment(&fields_bytes) {
                Ok(commitment) => {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&json!({
                            "schema": "symthaea.science_bench_refresh_counterfactual_evidence_plan_commitment.v1",
                            "plan_commitment_sha256": commitment
                        }))
                        .expect("evidence intervention plan commitment serializes")
                    );
                    ExitCode::SUCCESS
                }
                Err(error) => {
                    eprintln!("{error}");
                    ExitCode::from(2)
                }
            }
        }
        "verify-evidence-intervention-plan" => {
            if args.len() != 2 {
                usage("science-bench-refresh-validator");
                return ExitCode::from(2);
            }
            let plan_bytes = match read(&args[1], "counterfactual evidence plan") {
                Ok(bytes) => bytes,
                Err(code) => return code,
            };
            let report = match verify_evidence_intervention_plan(&plan_bytes) {
                Ok(report) => report,
                Err(error) => {
                    eprintln!("{error}");
                    return ExitCode::from(2);
                }
            };
            println!(
                "{}",
                serde_json::to_string_pretty(&report)
                    .expect("evidence intervention plan report serializes")
            );
            if report.valid {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(11)
            }
        }
        "commit-protected-index" => {
            if args.len() != 2 {
                usage("science-bench-refresh-validator");
                return ExitCode::from(2);
            }
            let input_bytes = match read(&args[1], "protected index fields") {
                Ok(bytes) => bytes,
                Err(code) => return code,
            };
            let input: ProtectedSearchArtifactIndexCommitmentInput =
                match serde_json::from_slice(&input_bytes) {
                    Ok(input) => input,
                    Err(error) => {
                        eprintln!("protected index fields do not match v1 shape: {error}");
                        return ExitCode::from(2);
                    }
                };
            let commitment = match compute_protected_index_commitment(&input) {
                Ok(commitment) => commitment,
                Err(error) => {
                    eprintln!("{error}");
                    return ExitCode::from(2);
                }
            };
            println!(
                "{}",
                serde_json::json!({
                    "schema": "symthaea.science_bench_refresh_protected_search_artifact_index_commitment.v1",
                    "generation_id": input.generation_id,
                    "index_commitment_sha256": commitment
                })
            );
            ExitCode::SUCCESS
        }
        "match-search-exact" => {
            if args.len() != 3 {
                usage("science-bench-refresh-validator");
                return ExitCode::from(2);
            }
            let receipt_bytes = match read(&args[1], "search exposure receipt") {
                Ok(bytes) => bytes,
                Err(code) => return code,
            };
            let protected_index_bytes = match read(&args[2], "protected artifact index") {
                Ok(bytes) => bytes,
                Err(code) => return code,
            };
            let report = match match_search_exposure_exact(&receipt_bytes, &protected_index_bytes) {
                Ok(report) => report,
                Err(error) => {
                    eprintln!("{error}");
                    return ExitCode::from(2);
                }
            };
            let detected = report.records.iter().any(|record| {
                matches!(
                    record.classification.as_str(),
                    "benchmark_metadata" | "question_context" | "explicit_answer"
                )
            });
            println!(
                "{}",
                serde_json::to_string_pretty(&report)
                    .expect("exact search match report serializes")
            );
            if detected {
                ExitCode::from(12)
            } else {
                ExitCode::SUCCESS
            }
        }
        "verify-search-exact-matches" => {
            if args.len() != 4 {
                usage("science-bench-refresh-validator");
                return ExitCode::from(2);
            }
            let receipt_bytes = match read(&args[1], "search exposure receipt") {
                Ok(bytes) => bytes,
                Err(code) => return code,
            };
            let protected_index_bytes = match read(&args[2], "protected artifact index") {
                Ok(bytes) => bytes,
                Err(code) => return code,
            };
            let match_bytes = match read(&args[3], "exact search match report") {
                Ok(bytes) => bytes,
                Err(code) => return code,
            };
            let report = match verify_search_exposure_exact_match_report(
                &receipt_bytes,
                &protected_index_bytes,
                &match_bytes,
            ) {
                Ok(report) => report,
                Err(error) => {
                    eprintln!("{error}");
                    return ExitCode::from(11);
                }
            };
            println!(
                "{}",
                serde_json::to_string_pretty(&report)
                    .expect("exact search match verification report serializes")
            );
            if !report.integrity_valid {
                ExitCode::from(11)
            } else if report.contaminated_event_count > 0 {
                ExitCode::from(12)
            } else if !report.receipt_policy_compliant {
                ExitCode::from(9)
            } else if report.review_required_event_count > 0 {
                ExitCode::from(13)
            } else {
                ExitCode::SUCCESS
            }
        }
        "verify-search-matches" => {
            if args.len() != 3 {
                usage("science-bench-refresh-validator");
                return ExitCode::from(2);
            }
            let receipt_bytes = match read(&args[1], "search exposure receipt") {
                Ok(bytes) => bytes,
                Err(code) => return code,
            };
            let match_bytes = match read(&args[2], "evaluator-side search match report") {
                Ok(bytes) => bytes,
                Err(code) => return code,
            };
            let receipt_report = match verify_search_exposure_receipt(&receipt_bytes) {
                Ok(report) => report,
                Err(error) => {
                    eprintln!("{error}");
                    return ExitCode::from(2);
                }
            };
            if !receipt_report.integrity_valid {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&receipt_report)
                        .expect("receipt verification report serializes")
                );
                return ExitCode::from(10);
            }
            let report = match verify_search_exposure_match_report(&receipt_bytes, &match_bytes) {
                Ok(report) => report,
                Err(error) => {
                    eprintln!("{error}");
                    return ExitCode::from(2);
                }
            };
            println!(
                "{}",
                serde_json::to_string_pretty(&report)
                    .expect("search exposure match report serializes")
            );
            if !report.integrity_valid {
                ExitCode::from(11)
            } else if report.contaminated_event_count > 0 {
                ExitCode::from(12)
            } else if !report.receipt_policy_compliant {
                ExitCode::from(9)
            } else if report.review_required_event_count > 0 {
                ExitCode::from(13)
            } else {
                ExitCode::SUCCESS
            }
        }
        "verify-search-receipt" => {
            if args.len() != 2 {
                usage("science-bench-refresh-validator");
                return ExitCode::from(2);
            }
            let receipt_bytes = match read(&args[1], "search exposure receipt") {
                Ok(bytes) => bytes,
                Err(code) => return code,
            };
            let report = match verify_search_exposure_receipt(&receipt_bytes) {
                Ok(report) => report,
                Err(error) => {
                    eprintln!("{error}");
                    return ExitCode::from(2);
                }
            };
            println!(
                "{}",
                serde_json::to_string_pretty(&report)
                    .expect("search exposure receipt report serializes")
            );
            if !report.integrity_valid {
                ExitCode::from(10)
            } else if !report.policy_compliant {
                ExitCode::from(9)
            } else {
                ExitCode::SUCCESS
            }
        }
        "audit-lineage" => {
            if args.len() != 5 {
                usage("science-bench-refresh-validator");
                return ExitCode::from(2);
            }
            let left_task_set = match read(&args[1], "left task set") {
                Ok(bytes) => bytes,
                Err(code) => return code,
            };
            let left_ledger = match read(&args[2], "left lineage ledger") {
                Ok(bytes) => bytes,
                Err(code) => return code,
            };
            let right_task_set = match read(&args[3], "right task set") {
                Ok(bytes) => bytes,
                Err(code) => return code,
            };
            let right_ledger = match read(&args[4], "right lineage ledger") {
                Ok(bytes) => bytes,
                Err(code) => return code,
            };
            let report = match audit_lineage_overlap(
                &left_task_set,
                &left_ledger,
                &right_task_set,
                &right_ledger,
            ) {
                Ok(report) => report,
                Err(error) => {
                    eprintln!("{error}");
                    return ExitCode::from(2);
                }
            };
            println!(
                "{}",
                serde_json::to_string_pretty(&report).expect("lineage audit report serializes")
            );
            if report.shared_lineage_root_count > 0
                || report.left_exposed_refresh_task_count > 0
                || report.right_exposed_refresh_task_count > 0
            {
                ExitCode::from(8)
            } else {
                ExitCode::SUCCESS
            }
        }
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
            let report =
                match audit_policy_input_overlap(&args[1], &left_bytes, &args[3], &right_bytes) {
                    Ok(report) => report,
                    Err(error) => {
                        eprintln!("{error}");
                        return ExitCode::from(2);
                    }
                };
            println!(
                "{}",
                serde_json::to_string_pretty(&report).expect("overlap audit report serializes")
            );
            if report.shared_task_id_count > 0 || report.exact_overlap_pair_count > 0 {
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
                serde_json::to_string_pretty(&report).expect("regeneration report serializes")
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
        "commit-reference-assurance-package-fields" => {
            if args.len() != 2 {
                usage("science-bench-refresh-validator");
                return ExitCode::from(2);
            }
            let fields_bytes = match read(&args[1], "reference assurance package fields") {
                Ok(bytes) => bytes,
                Err(code) => return code,
            };
            match compute_reference_package_commitment(&fields_bytes) {
                Ok(commitment) => {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&json!({
                            "schema": "symthaea.science_bench_refresh_reference_package_commitment.v1",
                            "package_commitment_sha256": commitment
                        }))
                        .expect("reference package commitment serializes")
                    );
                    ExitCode::SUCCESS
                }
                Err(error) => {
                    eprintln!("{error}");
                    ExitCode::from(2)
                }
            }
        }
        "verify-reference-assurance-package" => {
            if args.len() != 2 {
                usage("science-bench-refresh-validator");
                return ExitCode::from(2);
            }
            let package_bytes = match read(&args[1], "reference assurance package") {
                Ok(bytes) => bytes,
                Err(code) => return code,
            };
            let report = match verify_reference_assurance_package(&package_bytes) {
                Ok(report) => report,
                Err(error) => {
                    eprintln!("{error}");
                    return ExitCode::from(2);
                }
            };
            println!(
                "{}",
                serde_json::to_string_pretty(&report)
                    .expect("reference assurance report serializes")
            );
            if report.valid {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(11)
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
