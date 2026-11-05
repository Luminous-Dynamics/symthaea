use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    path::PathBuf,
    process::ExitCode,
};

const SNAPSHOT_SCHEMA: &str = "pitb.aegis-snapshot.v0";
const PROGRAM_SCHEMA: &str = "pitb.aegis-program-pair.v0";
const SPEC_SCHEMA: &str = "pitb.aegis-specification.v0";
const SAMPLE_SCHEMA: &str = "pitb.aegis-sample-entry.v0";
const ATTEMPT_SCHEMA: &str = "pitb.aegis-attempt.v0";
const REPORT_SCHEMA: &str = "pitb.aegis-integrity-report.v0";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct PaperReference {
    attempted_source_problems: usize,
    validated_rust_lean_program_pairs: usize,
    candidate_specifications: usize,
    evaluation_sample_size: usize,
    evaluation_program_count: usize,
    evaluation_attempted_count: usize,
    evaluation_statement_proofs: usize,
    evaluation_unresolved_count: usize,
    per_statement_budget_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Snapshot {
    schema_version: String,
    benchmark_id: String,
    benchmark_version: String,
    paper_arxiv_id: String,
    paper_version: String,
    paper_submitted_date: String,
    snapshot_id: String,
    dataset_artifact_status: String,
    dataset_artifact_url: Option<String>,
    dataset_archive_sha256: Option<String>,
    dataset_license_status: String,
    dataset_license_identifier: Option<String>,
    program_pair_count: Option<usize>,
    candidate_specification_count: Option<usize>,
    sample_entry_count: Option<usize>,
    paper_reference: PaperReference,
    task_manifest_sha256: Option<String>,
    sample_manifest_sha256: Option<String>,
    environment_policy_sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct ProgramPair {
    schema_version: String,
    program_id: String,
    source_problem_id: String,
    implementation_kind: String,
    rust_source_sha256: String,
    rust_toolchain_sha256: String,
    rust_compiler_version: String,
    reference_case_count: usize,
    rust_reference_tests_passed: usize,
    rust_test_receipt_sha256: String,
    charon_version: String,
    charon_source_commit: String,
    charon_config_sha256: String,
    charon_extraction_receipt_sha256: String,
    aeneas_version: String,
    aeneas_source_commit: String,
    aeneas_config_sha256: String,
    aeneas_translation_receipt_sha256: String,
    lean_toolchain: String,
    lean_environment_sha256: String,
    lean_program_sha256: String,
    lean_elaboration_receipt_sha256: String,
    differential_case_count: usize,
    differential_cases_matched: usize,
    differential_receipt_sha256: String,
    validation_status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Specification {
    schema_version: String,
    specification_id: String,
    program_id: String,
    theorem_name: String,
    statement_sha256: String,
    frozen_prefix_sha256: String,
    natural_language_description_sha256: String,
    requested_polarity: String,
    category: String,
    difficulty: String,
    plausible_screen_status: String,
    specification_status: String,
    specification_receipt_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct SampleEntry {
    schema_version: String,
    sample_entry_id: String,
    specification_id: String,
    program_id: String,
    selected: bool,
    exclusion_reason: String,
    category: String,
    difficulty: String,
    stratum_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Attempt {
    schema_version: String,
    attempt_id: String,
    specification_id: String,
    program_id: String,
    prover_id: String,
    prover_version: String,
    model_checkpoint_sha256: Option<String>,
    wall_budget_seconds: u64,
    elapsed_seconds: u64,
    input_tokens: u64,
    output_tokens: u64,
    api_cost_microusd: u64,
    outcome: String,
    theorem_statement_sha256: String,
    frozen_prefix_sha256: String,
    target_unchanged: bool,
    kernel_check_passed: bool,
    axiom_policy_passed: bool,
    no_sorry: bool,
    proof_artifact_sha256: Option<String>,
    solution_patch_sha256: Option<String>,
    comparator_receipt_sha256: Option<String>,
    diagnostic_receipt_sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct StratumSummary {
    dimension: String,
    stratum: String,
    selected_count: usize,
    attempted_count: usize,
    proved_statement_count: usize,
    proved_negation_count: usize,
    unresolved_count: usize,
    resource_exhausted_count: usize,
    harness_error_count: usize,
    resolved_count: usize,
    resolution_rate_over_selected: Option<f64>,
    original_statement_proof_rate_over_selected: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct ProgramSummary {
    program_count_total: usize,
    program_count_validated: usize,
    program_count_rejected: usize,
    program_count_in_sample: usize,
    max_sample_specs_per_program: usize,
}

#[derive(Debug, Clone, Serialize)]
struct OutcomeSummary {
    selected_specification_count: usize,
    attempted_specification_count: usize,
    not_attempted_count: usize,
    proved_statement_count: usize,
    proved_negation_count: usize,
    unresolved_count: usize,
    resource_exhausted_count: usize,
    harness_error_count: usize,
    resolved_count: usize,
    requested_polarity_correct_count: usize,
    requested_polarity_opposite_count: usize,
    resolution_rate_over_selected: Option<f64>,
    original_statement_proof_rate_over_selected: Option<f64>,
    requested_polarity_accuracy_over_resolved: Option<f64>,
    total_wall_seconds: u64,
    total_input_tokens: u64,
    total_output_tokens: u64,
    total_api_cost_microusd: u64,
    p50_elapsed_seconds: Option<f64>,
    p95_elapsed_seconds: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct Report {
    schema_version: String,
    snapshot_id: String,
    snapshot_mode: String,
    input_programs_jsonl_sha256: String,
    input_specifications_jsonl_sha256: String,
    input_sample_jsonl_sha256: String,
    input_attempts_jsonl_sha256: String,
    program_summary: ProgramSummary,
    outcome_summary: OutcomeSummary,
    strata: Vec<StratumSummary>,
    prover_id: Option<String>,
    prover_version: Option<String>,
    artifact_admission: String,
    implementation_correspondence_boundary: String,
    proof_trust_boundary: String,
    interpretation_boundary: String,
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn is_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn digest(value: &str, label: &str) -> Result<(), String> {
    if is_hex(value, 64) {
        Ok(())
    } else {
        Err(format!("{label} must be 64 lowercase hexadecimal characters"))
    }
}

fn commit(value: &str, label: &str) -> Result<(), String> {
    if is_hex(value, 40) {
        Ok(())
    } else {
        Err(format!("{label} must be a 40-character lowercase Git commit"))
    }
}

fn non_empty(value: &str, label: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        Err(format!("{label} cannot be empty"))
    } else {
        Ok(())
    }
}

fn ratio(n: usize, d: usize) -> Option<f64> {
    if d == 0 { None } else { Some(n as f64 / d as f64) }
}

fn validate_snapshot(snapshot: &Snapshot, allow_synthetic: bool) -> Result<(), String> {
    if snapshot.schema_version != SNAPSHOT_SCHEMA {
        return Err("unsupported Aegis snapshot schema".into());
    }
    if snapshot.benchmark_id != "Aegis" || snapshot.paper_arxiv_id != "2610.38174" {
        return Err("snapshot does not identify the pinned Aegis paper".into());
    }
    non_empty(&snapshot.benchmark_version, "snapshot.benchmark_version")?;
    non_empty(&snapshot.paper_version, "snapshot.paper_version")?;
    non_empty(&snapshot.paper_submitted_date, "snapshot.paper_submitted_date")?;
    non_empty(&snapshot.snapshot_id, "snapshot.snapshot_id")?;
    if snapshot.paper_reference.attempted_source_problems != 2356
        || snapshot.paper_reference.validated_rust_lean_program_pairs != 2045
        || snapshot.paper_reference.candidate_specifications != 11533
        || snapshot.paper_reference.evaluation_sample_size != 2000
        || snapshot.paper_reference.evaluation_program_count != 1257
        || snapshot.paper_reference.evaluation_attempted_count != 1997
        || snapshot.paper_reference.evaluation_statement_proofs != 1054
        || snapshot.paper_reference.evaluation_unresolved_count != 943
        || snapshot.paper_reference.per_statement_budget_seconds != 1800
    {
        return Err("paper reference counts differ from the pinned arXiv v1 record".into());
    }
    match snapshot.dataset_artifact_status.as_str() {
        "not_located" => {
            if snapshot.dataset_artifact_url.is_some()
                || snapshot.dataset_archive_sha256.is_some()
                || snapshot.program_pair_count.is_some()
                || snapshot.candidate_specification_count.is_some()
                || snapshot.sample_entry_count.is_some()
            {
                return Err("not_located artifact cannot claim archive identity or local dataset counts".into());
            }
            if snapshot.dataset_license_status != "not_confirmed"
                || snapshot.dataset_license_identifier.is_some()
            {
                return Err("unlocated artifact must not imply its license is confirmed".into());
            }
            if !allow_synthetic {
                return Err("Aegis task archive has not been located; measured dataset verification is unavailable".into());
            }
        }
        "available" => {
            let url = snapshot.dataset_artifact_url.as_deref().ok_or("available artifact requires dataset_artifact_url")?;
            non_empty(url, "snapshot.dataset_artifact_url")?;
            digest(snapshot.dataset_archive_sha256.as_deref().ok_or("available artifact requires archive SHA-256")?, "snapshot.dataset_archive_sha256")?;
            if snapshot.dataset_license_status != "confirmed" {
                return Err("Aegis artifact license must be confirmed before ingest".into());
            }
            non_empty(snapshot.dataset_license_identifier.as_deref().ok_or("confirmed license identifier required")?, "snapshot.dataset_license_identifier")?;
            for (name, value) in [
                ("program_pair_count", snapshot.program_pair_count),
                ("candidate_specification_count", snapshot.candidate_specification_count),
                ("sample_entry_count", snapshot.sample_entry_count),
            ] {
                if value.is_none() || value == Some(0) {
                    return Err(format!("available artifact requires a positive {name}"));
                }
            }
            for (name, value) in [
                ("task_manifest_sha256", snapshot.task_manifest_sha256.as_deref()),
                ("sample_manifest_sha256", snapshot.sample_manifest_sha256.as_deref()),
                ("environment_policy_sha256", snapshot.environment_policy_sha256.as_deref()),
            ] {
                digest(value.ok_or_else(|| format!("available artifact requires {name}"))?, &format!("snapshot.{name}"))?;
            }
        }
        "synthetic_fixture" if allow_synthetic => {
            if snapshot.dataset_license_status != "synthetic_only" {
                return Err("synthetic fixture must carry synthetic_only license status".into());
            }
        }
        _ => return Err("unsupported dataset_artifact_status".into()),
    }
    Ok(())
}

fn validate_program(program: &ProgramPair, index: usize) -> Result<(), String> {
    let label = format!("program pair row {index}");
    if program.schema_version != PROGRAM_SCHEMA {
        return Err(format!("{label} has unsupported schema"));
    }
    for (name, value) in [
        ("program_id", program.program_id.as_str()),
        ("source_problem_id", program.source_problem_id.as_str()),
        ("rust_compiler_version", program.rust_compiler_version.as_str()),
        ("charon_version", program.charon_version.as_str()),
        ("aeneas_version", program.aeneas_version.as_str()),
        ("lean_toolchain", program.lean_toolchain.as_str()),
    ] {
        non_empty(value, &format!("{label}.{name}"))?;
    }
    if program.implementation_kind != "safe_rust_pure_function" {
        return Err(format!("{label} has unsupported implementation_kind"));
    }
    for (name, value) in [
        ("rust_source_sha256", &program.rust_source_sha256),
        ("rust_toolchain_sha256", &program.rust_toolchain_sha256),
        ("charon_config_sha256", &program.charon_config_sha256),
        ("charon_extraction_receipt_sha256", &program.charon_extraction_receipt_sha256),
        ("aeneas_config_sha256", &program.aeneas_config_sha256),
        ("aeneas_translation_receipt_sha256", &program.aeneas_translation_receipt_sha256),
        ("lean_environment_sha256", &program.lean_environment_sha256),
        ("lean_program_sha256", &program.lean_program_sha256),
        ("lean_elaboration_receipt_sha256", &program.lean_elaboration_receipt_sha256),
        ("rust_test_receipt_sha256", &program.rust_test_receipt_sha256),
        ("differential_receipt_sha256", &program.differential_receipt_sha256),
    ] {
        digest(value, &format!("{label}.{name}"))?;
    }
    commit(&program.charon_source_commit, &format!("{label}.charon_source_commit"))?;
    commit(&program.aeneas_source_commit, &format!("{label}.aeneas_source_commit"))?;
    if program.reference_case_count == 0
        || program.rust_reference_tests_passed > program.reference_case_count
        || program.differential_case_count > program.reference_case_count
        || program.differential_cases_matched > program.differential_case_count
    {
        return Err(format!("{label} has inconsistent reference/differential case counts"));
    }
    match program.validation_status.as_str() {
        "validated" => {
            if program.rust_reference_tests_passed != program.reference_case_count
                || program.differential_case_count != program.reference_case_count
                || program.differential_cases_matched != program.differential_case_count
            {
                return Err(format!("{label} validated status contradicts test/differential counts"));
            }
        }
        "rejected" => {}
        _ => return Err(format!("{label} has unsupported validation_status")),
    }
    Ok(())
}

fn validate_specification(spec: &Specification, index: usize) -> Result<(), String> {
    let label = format!("specification row {index}");
    if spec.schema_version != SPEC_SCHEMA {
        return Err(format!("{label} has unsupported schema"));
    }
    for (name, value) in [
        ("specification_id", spec.specification_id.as_str()),
        ("program_id", spec.program_id.as_str()),
        ("theorem_name", spec.theorem_name.as_str()),
    ] {
        non_empty(value, &format!("{label}.{name}"))?;
    }
    for (name, value) in [
        ("statement_sha256", &spec.statement_sha256),
        ("frozen_prefix_sha256", &spec.frozen_prefix_sha256),
        ("natural_language_description_sha256", &spec.natural_language_description_sha256),
        ("specification_receipt_sha256", &spec.specification_receipt_sha256),
    ] {
        digest(value, &format!("{label}.{name}"))?;
    }
    if !matches!(spec.requested_polarity.as_str(), "hold" | "fail") {
        return Err(format!("{label} has unsupported requested_polarity"));
    }
    if !matches!(
        spec.category.as_str(),
        "edge_case" | "explicit_contract" | "functional_correctness" | "totality"
            | "output_bound" | "output_shape" | "relational" | "other"
    ) {
        return Err(format!("{label} has unsupported category"));
    }
    if !matches!(spec.difficulty.as_str(), "easy" | "medium" | "hard" | "unknown") {
        return Err(format!("{label} has unsupported difficulty"));
    }
    if !matches!(
        spec.plausible_screen_status.as_str(),
        "counterexample_found" | "no_counterexample_found" | "not_testable" | "not_run"
    ) {
        return Err(format!("{label} has unsupported Plausible screen status"));
    }
    if !matches!(spec.specification_status.as_str(), "retained" | "excluded_duplicate" | "excluded_trivial" | "excluded_vacuous" | "elaboration_error") {
        return Err(format!("{label} has unsupported specification_status"));
    }
    Ok(())
}

fn validate_sample(entry: &SampleEntry, spec: &Specification, index: usize) -> Result<(), String> {
    let label = format!("sample entry row {index}");
    if entry.schema_version != SAMPLE_SCHEMA {
        return Err(format!("{label} has unsupported schema"));
    }
    non_empty(&entry.sample_entry_id, &format!("{label}.sample_entry_id"))?;
    non_empty(&entry.stratum_id, &format!("{label}.stratum_id"))?;
    if entry.specification_id != spec.specification_id || entry.program_id != spec.program_id
        || entry.category != spec.category || entry.difficulty != spec.difficulty
    {
        return Err(format!("{label} has a stale or mismatched specification join"));
    }
    if entry.selected {
        if entry.exclusion_reason != "none" {
            return Err(format!("{label} selected row cannot carry an exclusion reason"));
        }
        if spec.specification_status != "retained" {
            return Err(format!("{label} selected a specification that was filtered upstream"));
        }
        if spec.requested_polarity != "hold" {
            return Err(format!("{label} selected a requested-to-fail specification; published Aegis sample excludes that category"));
        }
        if spec.plausible_screen_status == "counterexample_found" {
            return Err(format!("{label} selected a statement already refuted by the Plausible screen"));
        }
    } else {
        let valid_reason = match entry.exclusion_reason.as_str() {
            "plausible_refuted" => spec.plausible_screen_status == "counterexample_found",
            "requested_fail" => spec.requested_polarity == "fail",
            "filtered_specification" => spec.specification_status != "retained",
            "other" => true,
            _ => false,
        };
        if !valid_reason {
            return Err(format!("{label} exclusion_reason contradicts its source specification"));
        }
    }
    Ok(())
}

fn validate_attempt(attempt: &Attempt, spec: &Specification, program: &ProgramPair, index: usize) -> Result<(), String> {
    let label = format!("attempt row {index}");
    if attempt.schema_version != ATTEMPT_SCHEMA {
        return Err(format!("{label} has unsupported schema"));
    }
    for (name, value) in [
        ("attempt_id", attempt.attempt_id.as_str()),
        ("prover_id", attempt.prover_id.as_str()),
        ("prover_version", attempt.prover_version.as_str()),
    ] {
        non_empty(value, &format!("{label}.{name}"))?;
    }
    for (name, value) in [
        ("theorem_statement_sha256", &attempt.theorem_statement_sha256),
        ("frozen_prefix_sha256", &attempt.frozen_prefix_sha256),
    ] {
        digest(value, &format!("{label}.{name}"))?;
    }
    if attempt.model_checkpoint_sha256.is_some() {
        digest(attempt.model_checkpoint_sha256.as_deref().unwrap(), &format!("{label}.model_checkpoint_sha256"))?;
    }
    if attempt.wall_budget_seconds == 0 || attempt.elapsed_seconds > attempt.wall_budget_seconds {
        return Err(format!("{label} has invalid elapsed/budget accounting"));
    }
    if attempt.specification_id != spec.specification_id || attempt.program_id != program.program_id {
        return Err(format!("{label} specification/program binding mismatch"));
    }
    if attempt.theorem_statement_sha256 != spec.statement_sha256
        || attempt.frozen_prefix_sha256 != spec.frozen_prefix_sha256
    {
        return Err(format!("{label} does not bind the frozen statement/prefix"));
    }
    if !matches!(attempt.outcome.as_str(), "proved_statement" | "proved_negation" | "unresolved" | "resource_exhausted" | "harness_error") {
        return Err(format!("{label} has unsupported outcome"));
    }
    if attempt.outcome == "proved_statement" || attempt.outcome == "proved_negation" {
        if !attempt.target_unchanged
            || !attempt.kernel_check_passed
            || !attempt.axiom_policy_passed
            || !attempt.no_sorry
        {
            return Err(format!("{label} accepted outcome lacks frozen target, kernel, axiom-policy or no-sorry gate"));
        }
        for (name, value) in [
            ("proof_artifact_sha256", attempt.proof_artifact_sha256.as_deref()),
            ("solution_patch_sha256", attempt.solution_patch_sha256.as_deref()),
            ("comparator_receipt_sha256", attempt.comparator_receipt_sha256.as_deref()),
        ] {
            digest(value.ok_or_else(|| format!("{label} accepted outcome missing {name}"))?, &format!("{label}.{name}"))?;
        }
    } else if attempt.kernel_check_passed || attempt.axiom_policy_passed || attempt.no_sorry {
        return Err(format!("{label} unresolved/error outcome cannot claim all acceptance gates passed"));
    }
    if attempt.comparator_receipt_sha256.is_some() {
        digest(attempt.comparator_receipt_sha256.as_deref().unwrap(), &format!("{label}.comparator_receipt_sha256"))?;
    }
    if attempt.diagnostic_receipt_sha256.is_some() {
        digest(attempt.diagnostic_receipt_sha256.as_deref().unwrap(), &format!("{label}.diagnostic_receipt_sha256"))?;
    }
    Ok(())
}

fn parse_jsonl<T: for<'de> Deserialize<'de>>(bytes: &[u8], label: &str) -> Result<Vec<T>, String> {
    let text = std::str::from_utf8(bytes).map_err(|e| format!("{label} is not UTF-8: {e}"))?;
    let mut rows = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() { continue; }
        let row = serde_json::from_str(line).map_err(|e| format!("{label} line {}: {e}", i + 1))?;
        rows.push(row);
    }
    if rows.is_empty() {
        return Err(format!("{label} must contain at least one row"));
    }
    Ok(rows)
}

fn percentile(values: &[u64], q: f64) -> Option<f64> {
    if values.is_empty() { return None; }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted.get(idx).copied().map(|x| x as f64)
}

fn validate_all(
    snapshot: &Snapshot,
    programs: &[ProgramPair],
    specifications: &[Specification],
    sample: &[SampleEntry],
    attempts: &[Attempt],
    input_digests: [String; 4],
    allow_synthetic: bool,
) -> Result<Report, String> {
    validate_snapshot(snapshot, allow_synthetic)?;
    for (name, d) in [
        ("programs", input_digests[0].as_str()),
        ("specifications", input_digests[1].as_str()),
        ("sample", input_digests[2].as_str()),
        ("attempts", input_digests[3].as_str()),
    ] {
        digest(d, &format!("input {name} JSONL digest"))?;
    }
    let mut program_map = BTreeMap::<String, &ProgramPair>::new();
    for (i, program) in programs.iter().enumerate() {
        validate_program(program, i + 1)?;
        if program_map.insert(program.program_id.clone(), program).is_some() {
            return Err(format!("duplicate program_id {}", program.program_id));
        }
    }
    let mut spec_map = BTreeMap::<String, &Specification>::new();
    for (i, spec) in specifications.iter().enumerate() {
        validate_specification(spec, i + 1)?;
        if !program_map.contains_key(&spec.program_id) {
            return Err(format!("specification {} references unknown program {}", spec.specification_id, spec.program_id));
        }
        if spec_map.insert(spec.specification_id.clone(), spec).is_some() {
            return Err(format!("duplicate specification_id {}", spec.specification_id));
        }
    }
    let mut sample_map = BTreeMap::<String, &SampleEntry>::new();
    for (i, entry) in sample.iter().enumerate() {
        let spec = spec_map.get(&entry.specification_id).ok_or_else(|| format!("sample row {} references unknown spec", i + 1))?;
        validate_sample(entry, spec, i + 1)?;
        if sample_map.insert(entry.specification_id.clone(), entry).is_some() {
            return Err(format!("duplicate sample entry for spec {}", entry.specification_id));
        }
    }
    let selected: BTreeSet<String> = sample_map.iter()
        .filter(|(_, row)| row.selected)
        .map(|(id, _)| id.clone())
        .collect();
    if selected.is_empty() {
        return Err("sample contains no selected task".into());
    }
    let mut attempt_ids = BTreeSet::new();
    let mut attempt_per_spec = BTreeSet::new();
    let mut prover_versions = BTreeSet::new();
    let mut by_spec = BTreeMap::<String, &Attempt>::new();
    for (i, attempt) in attempts.iter().enumerate() {
        let spec = spec_map.get(&attempt.specification_id)
            .ok_or_else(|| format!("attempt {} references unknown spec", i + 1))?;
        let sample_entry = sample_map.get(&attempt.specification_id)
            .ok_or_else(|| format!("attempt {} references a spec missing from the sample ledger", i + 1))?;
        if !sample_entry.selected {
            return Err(format!("attempt {} targets a task excluded from the evaluation sample", i + 1));
        }
        let program = program_map.get(&attempt.program_id).ok_or_else(|| format!("attempt {} references unknown program", i + 1))?;
        if program.validation_status != "validated" {
            return Err(format!("attempt {} targets a Rust-Lean program pair not validated upstream", i + 1));
        }
        validate_attempt(attempt, spec, program, i + 1)?;
        if !attempt_ids.insert(attempt.attempt_id.clone()) {
            return Err(format!("duplicate attempt_id {}", attempt.attempt_id));
        }
        if !attempt_per_spec.insert(attempt.specification_id.clone()) {
            return Err(format!("duplicate attempt for specification {}; use a separate run ledger for repeats", attempt.specification_id));
        }
        prover_versions.insert((attempt.prover_id.clone(), attempt.prover_version.clone()));
        by_spec.insert(attempt.specification_id.clone(), attempt);
    }
    if prover_versions.len() > 1 {
        return Err("one scorecard run must contain exactly one prover_id/version pair".into());
    }
    let mut counts = OutcomeSummary {
        selected_specification_count: selected.len(),
        attempted_specification_count: attempts.len(),
        not_attempted_count: selected.len().saturating_sub(attempts.len()),
        proved_statement_count: 0,
        proved_negation_count: 0,
        unresolved_count: 0,
        resource_exhausted_count: 0,
        harness_error_count: 0,
        resolved_count: 0,
        requested_polarity_correct_count: 0,
        requested_polarity_opposite_count: 0,
        resolution_rate_over_selected: None,
        original_statement_proof_rate_over_selected: None,
        requested_polarity_accuracy_over_resolved: None,
        total_wall_seconds: 0,
        total_input_tokens: 0,
        total_output_tokens: 0,
        total_api_cost_microusd: 0,
        p50_elapsed_seconds: None,
        p95_elapsed_seconds: None,
    };
    let mut elapsed = Vec::new();
    for attempt in attempts {
        match attempt.outcome.as_str() {
            "proved_statement" => {
                counts.proved_statement_count += 1;
                let spec = spec_map[&attempt.specification_id];
                if spec.requested_polarity == "hold" {
                    counts.requested_polarity_correct_count += 1;
                } else {
                    counts.requested_polarity_opposite_count += 1;
                }
            }
            "proved_negation" => {
                counts.proved_negation_count += 1;
                let spec = spec_map[&attempt.specification_id];
                if spec.requested_polarity == "fail" {
                    counts.requested_polarity_correct_count += 1;
                } else {
                    counts.requested_polarity_opposite_count += 1;
                }
            }
            "unresolved" => counts.unresolved_count += 1,
            "resource_exhausted" => counts.resource_exhausted_count += 1,
            "harness_error" => counts.harness_error_count += 1,
            _ => unreachable!(),
        }
        counts.total_wall_seconds += attempt.elapsed_seconds;
        counts.total_input_tokens += attempt.input_tokens;
        counts.total_output_tokens += attempt.output_tokens;
        counts.total_api_cost_microusd += attempt.api_cost_microusd;
        elapsed.push(attempt.elapsed_seconds);
    }
    counts.resolved_count = counts.proved_statement_count + counts.proved_negation_count;
    counts.resolution_rate_over_selected = ratio(counts.resolved_count, counts.selected_specification_count);
    counts.original_statement_proof_rate_over_selected = ratio(counts.proved_statement_count, counts.selected_specification_count);
    counts.requested_polarity_accuracy_over_resolved = ratio(counts.requested_polarity_correct_count, counts.resolved_count);
    counts.p50_elapsed_seconds = percentile(&elapsed, 0.50);
    counts.p95_elapsed_seconds = percentile(&elapsed, 0.95);

    let mut program_sample_counts = BTreeMap::<String, usize>::new();
    let mut program_sample_set = BTreeSet::new();
    for spec_id in &selected {
        let spec = spec_map[spec_id.as_str()];
        *program_sample_counts.entry(spec.program_id.clone()).or_default() += 1;
        program_sample_set.insert(spec.program_id.clone());
    }
    let max_sample_specs_per_program = program_sample_counts.values().copied().max().unwrap_or(0);
    let validated_programs = programs.iter().filter(|p| p.validation_status == "validated").count();
    let rejected_programs = programs.len() - validated_programs;

    let mut stratum_acc = BTreeMap::<(String, String), [usize; 7]>::new();
    for spec_id in &selected {
        let spec = spec_map[spec_id.as_str()];
        for (dimension, value) in [
            ("category", spec.category.as_str()),
            ("difficulty", spec.difficulty.as_str()),
        ] {
            let acc = stratum_acc.entry((dimension.to_owned(), value.to_owned())).or_default();
            acc[0] += 1; // selected
            if let Some(attempt) = by_spec.get(spec_id) {
                acc[1] += 1; // attempted
                match attempt.outcome.as_str() {
                    "proved_statement" => acc[2] += 1,
                    "proved_negation" => acc[3] += 1,
                    "unresolved" => acc[4] += 1,
                    "resource_exhausted" => acc[5] += 1,
                    "harness_error" => acc[6] += 1,
                    _ => unreachable!(),
                }
            }
        }
    }
    let mut strata = Vec::new();
    for ((dimension, stratum), acc) in stratum_acc {
        let resolved = acc[2] + acc[3];
        strata.push(StratumSummary {
            dimension,
            stratum,
            selected_count: acc[0],
            attempted_count: acc[1],
            proved_statement_count: acc[2],
            proved_negation_count: acc[3],
            unresolved_count: acc[4],
            resource_exhausted_count: acc[5],
            harness_error_count: acc[6],
            resolved_count: resolved,
            resolution_rate_over_selected: ratio(resolved, acc[0]),
            original_statement_proof_rate_over_selected: ratio(acc[2], acc[0]),
        });
    }
    let prover = prover_versions.iter().next().cloned();
    let artifact_admission = match snapshot.dataset_artifact_status.as_str() {
        "available" => "dataset_artifact_pinned_and_license_confirmed",
        "synthetic_fixture" => "synthetic_contract_only",
        _ => "paper_metadata_only_dataset_not_located",
    };
    Ok(Report {
        schema_version: REPORT_SCHEMA.into(),
        snapshot_id: snapshot.snapshot_id.clone(),
        snapshot_mode: snapshot.dataset_artifact_status.clone(),
        input_programs_jsonl_sha256: input_digests[0].clone(),
        input_specifications_jsonl_sha256: input_digests[1].clone(),
        input_sample_jsonl_sha256: input_digests[2].clone(),
        input_attempts_jsonl_sha256: input_digests[3].clone(),
        program_summary: ProgramSummary {
            program_count_total: programs.len(),
            program_count_validated: validated_programs,
            program_count_rejected: rejected_programs,
            program_count_in_sample: program_sample_set.len(),
            max_sample_specs_per_program,
        },
        outcome_summary: counts,
        strata,
        prover_id: prover.as_ref().map(|x| x.0.clone()),
        prover_version: prover.as_ref().map(|x| x.1.clone()),
        artifact_admission: artifact_admission.into(),
        implementation_correspondence_boundary: "Differential validation covers retained finite reference inputs only. It is not a proof of Rust-Lean semantic equivalence for all inputs; source bytes, Charon extraction and Aeneas translation receipts must be independently opened and authenticated.".into(),
        proof_trust_boundary: "A resolved outcome requires a frozen target/prefix, Lean kernel success, allowlisted axiom policy, no sorry, a proof artifact, a solution patch and an independent comparator receipt. This validator checks digest syntax and joins; it does not open those bytes or rerun Lean.".into(),
        interpretation_boundary: "A proof of P or a proof of not-P both resolve the frozen proposition. Requested-polarity agreement is scored separately; proving the opposite polarity is a specification-intent mismatch, not a successful confirmation of the requested property. Unattempted tasks remain in the selected denominator.".into(),
    })
}

fn fake_digest(text: &str) -> String { sha256(text.as_bytes()) }

fn fixture_snapshot() -> Snapshot {
    Snapshot {
        schema_version: SNAPSHOT_SCHEMA.into(),
        benchmark_id: "Aegis".into(),
        benchmark_version: "paper-v1-2026-10-28".into(),
        paper_arxiv_id: "2610.38174".into(),
        paper_version: "v1".into(),
        paper_submitted_date: "2026-10-28".into(),
        snapshot_id: "synthetic-aegis-contract-v0".into(),
        dataset_artifact_status: "synthetic_fixture".into(),
        dataset_artifact_url: None,
        dataset_archive_sha256: None,
        dataset_license_status: "synthetic_only".into(),
        dataset_license_identifier: None,
        program_pair_count: Some(2),
        candidate_specification_count: Some(4),
        sample_entry_count: Some(4),
        paper_reference: PaperReference {
            attempted_source_problems: 2356,
            validated_rust_lean_program_pairs: 2045,
            candidate_specifications: 11533,
            evaluation_sample_size: 2000,
            evaluation_program_count: 1257,
            evaluation_attempted_count: 1997,
            evaluation_statement_proofs: 1054,
            evaluation_unresolved_count: 943,
            per_statement_budget_seconds: 1800,
        },
        task_manifest_sha256: Some(fake_digest("synthetic tasks")),
        sample_manifest_sha256: Some(fake_digest("synthetic sample")),
        environment_policy_sha256: Some(fake_digest("synthetic environment")),
    }
}

fn fixture_program(id: &str) -> ProgramPair {
    ProgramPair {
        schema_version: PROGRAM_SCHEMA.into(),
        program_id: id.into(),
        source_problem_id: format!("leetcode-{id}"),
        implementation_kind: "safe_rust_pure_function".into(),
        rust_source_sha256: fake_digest(&format!("{id}-rust")),
        rust_toolchain_sha256: fake_digest("rust-toolchain"),
        rust_compiler_version: "rustc 1.91.0".into(),
        reference_case_count: 3,
        rust_reference_tests_passed: 3,
        rust_test_receipt_sha256: fake_digest(&format!("{id}-rust-tests")),
        charon_version: "charon-pinned".into(),
        charon_source_commit: "1".repeat(40),
        charon_config_sha256: fake_digest("charon-config"),
        charon_extraction_receipt_sha256: fake_digest(&format!("{id}-charon")),
        aeneas_version: "aeneas-pinned".into(),
        aeneas_source_commit: "2".repeat(40),
        aeneas_config_sha256: fake_digest("aeneas-config"),
        aeneas_translation_receipt_sha256: fake_digest(&format!("{id}-aeneas")),
        lean_toolchain: "leanprover/lean4:v4.32.2".into(),
        lean_environment_sha256: fake_digest("lean-environment"),
        lean_program_sha256: fake_digest(&format!("{id}-lean")),
        lean_elaboration_receipt_sha256: fake_digest(&format!("{id}-lean-elab")),
        differential_case_count: 3,
        differential_cases_matched: 3,
        differential_receipt_sha256: fake_digest(&format!("{id}-diff")),
        validation_status: "validated".into(),
    }
}

fn fixture_spec(id: &str, program_id: &str, polarity: &str, category: &str, difficulty: &str, plausible: &str) -> Specification {
    Specification {
        schema_version: SPEC_SCHEMA.into(),
        specification_id: id.into(),
        program_id: program_id.into(),
        theorem_name: format!("Aegis.{id}"),
        statement_sha256: fake_digest(&format!("{id}-statement")),
        frozen_prefix_sha256: fake_digest(&format!("{id}-prefix")),
        natural_language_description_sha256: fake_digest(&format!("{id}-description")),
        requested_polarity: polarity.into(),
        category: category.into(),
        difficulty: difficulty.into(),
        plausible_screen_status: plausible.into(),
        specification_status: "retained".into(),
        specification_receipt_sha256: fake_digest(&format!("{id}-spec")),
    }
}

fn fixture_sample(id: &str, spec: &Specification, selected: bool, reason: &str) -> SampleEntry {
    SampleEntry {
        schema_version: SAMPLE_SCHEMA.into(),
        sample_entry_id: format!("sample-{id}"),
        specification_id: spec.specification_id.clone(),
        program_id: spec.program_id.clone(),
        selected,
        exclusion_reason: reason.into(),
        category: spec.category.clone(),
        difficulty: spec.difficulty.clone(),
        stratum_id: format!("{}-{}", spec.category, spec.difficulty),
    }
}

fn fixture_attempt(id: &str, spec: &Specification, program: &ProgramPair, outcome: &str, seconds: u64) -> Attempt {
    let accepted = outcome == "proved_statement" || outcome == "proved_negation";
    Attempt {
        schema_version: ATTEMPT_SCHEMA.into(),
        attempt_id: format!("attempt-{id}"),
        specification_id: spec.specification_id.clone(),
        program_id: program.program_id.clone(),
        prover_id: "synthetic-prover".into(),
        prover_version: "contract-test".into(),
        model_checkpoint_sha256: None,
        wall_budget_seconds: 1800,
        elapsed_seconds: seconds,
        input_tokens: 100,
        output_tokens: 30,
        api_cost_microusd: 1000,
        outcome: outcome.into(),
        theorem_statement_sha256: spec.statement_sha256.clone(),
        frozen_prefix_sha256: spec.frozen_prefix_sha256.clone(),
        target_unchanged: accepted,
        kernel_check_passed: accepted,
        axiom_policy_passed: accepted,
        no_sorry: accepted,
        proof_artifact_sha256: if accepted { Some(fake_digest(&format!("{id}-proof"))) } else { None },
        solution_patch_sha256: if accepted { Some(fake_digest(&format!("{id}-patch"))) } else { None },
        comparator_receipt_sha256: if accepted { Some(fake_digest(&format!("{id}-comparator"))) } else { None },
        diagnostic_receipt_sha256: if accepted { None } else { Some(fake_digest(&format!("{id}-diagnostic"))) },
    }
}

fn jsonl<T: Serialize>(rows: &[T]) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    for row in rows {
        bytes.extend(serde_json::to_vec(row).map_err(|e| e.to_string())?);
        bytes.push(b'\n');
    }
    Ok(bytes)
}

fn write_fixture(dir: &PathBuf) -> Result<Report, String> {
    fs::create_dir_all(dir).map_err(|e| format!("create fixture dir: {e}"))?;
    let snapshot = fixture_snapshot();
    let programs = vec![fixture_program("program-a"), fixture_program("program-b")];
    let specs = vec![
        fixture_spec("spec-edge", "program-a", "hold", "edge_case", "easy", "no_counterexample_found"),
        fixture_spec("spec-contract", "program-a", "hold", "explicit_contract", "hard", "no_counterexample_found"),
        fixture_spec("spec-fail", "program-b", "fail", "explicit_contract", "medium", "not_run"),
        fixture_spec("spec-screened", "program-b", "hold", "edge_case", "medium", "counterexample_found"),
    ];
    let sample = vec![
        fixture_sample("edge", &specs[0], true, "none"),
        fixture_sample("contract", &specs[1], true, "none"),
        fixture_sample("fail", &specs[2], false, "requested_fail"),
        fixture_sample("screened", &specs[3], false, "plausible_refuted"),
    ];
    let attempts = vec![
        fixture_attempt("edge", &specs[0], &programs[0], "proved_statement", 40),
        fixture_attempt("contract", &specs[1], &programs[0], "proved_negation", 90),
    ];
    let p = jsonl(&programs)?;
    let s = jsonl(&specs)?;
    let m = jsonl(&sample)?;
    let a = jsonl(&attempts)?;
    fs::write(dir.join("snapshot.json"), serde_json::to_vec_pretty(&snapshot).map_err(|e| e.to_string())?)
        .map_err(|e| format!("write snapshot: {e}"))?;
    fs::write(dir.join("programs.jsonl"), &p).map_err(|e| format!("write programs: {e}"))?;
    fs::write(dir.join("specifications.jsonl"), &s).map_err(|e| format!("write specifications: {e}"))?;
    fs::write(dir.join("sample.jsonl"), &m).map_err(|e| format!("write sample: {e}"))?;
    fs::write(dir.join("attempts.jsonl"), &a).map_err(|e| format!("write attempts: {e}"))?;
    let report = validate_all(&snapshot, &programs, &specs, &sample, &attempts, [sha256(&p), sha256(&s), sha256(&m), sha256(&a)], true)?;
    fs::write(dir.join("report.json"), serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?)
        .map_err(|e| format!("write report: {e}"))?;
    Ok(report)
}

fn run_self_tests() -> Result<(), String> {
    let dir = env::temp_dir().join(format!("pitb-aegis-self-test-{}", std::process::id()));
    let report = write_fixture(&dir)?;
    let counts = &report.outcome_summary;
    if counts.selected_specification_count != 2
        || counts.attempted_specification_count != 2
        || counts.proved_statement_count != 1
        || counts.proved_negation_count != 1
        || counts.resolved_count != 2
        || counts.requested_polarity_correct_count != 1
        || counts.requested_polarity_opposite_count != 1
    {
        return Err(format!("Aegis polarity-aware scorecard control failed: {counts:?}"));
    }
    println!("SELF_TEST_PASS proof and refutation both resolve the frozen proposition");
    println!("SELF_TEST_PASS requested-polarity mismatch is scored separately from prover resolution");

    let snapshot: Snapshot = serde_json::from_slice(&fs::read(dir.join("snapshot.json")).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let mut bad = snapshot.clone();
    bad.dataset_artifact_status = "available".into();
    bad.dataset_artifact_url = None;
    if validate_snapshot(&bad, true).is_ok() {
        return Err("unlocated Aegis archive cannot be silently labeled available".into());
    }
    println!("SELF_TEST_PASS unknown dataset release is not promoted to available");

    let specs: Vec<Specification> = parse_jsonl(&fs::read(dir.join("specifications.jsonl")).map_err(|e| e.to_string())?, "fixture specs")?;
    let programs: Vec<ProgramPair> = parse_jsonl(&fs::read(dir.join("programs.jsonl")).map_err(|e| e.to_string())?, "fixture programs")?;
    let sample: Vec<SampleEntry> = parse_jsonl(&fs::read(dir.join("sample.jsonl")).map_err(|e| e.to_string())?, "fixture sample")?;
    let attempts: Vec<Attempt> = parse_jsonl(&fs::read(dir.join("attempts.jsonl")).map_err(|e| e.to_string())?, "fixture attempts")?;
    let digests = [
        sha256(&fs::read(dir.join("programs.jsonl")).map_err(|e| e.to_string())?),
        sha256(&fs::read(dir.join("specifications.jsonl")).map_err(|e| e.to_string())?),
        sha256(&fs::read(dir.join("sample.jsonl")).map_err(|e| e.to_string())?),
        sha256(&fs::read(dir.join("attempts.jsonl")).map_err(|e| e.to_string())?),
    ];
    let mut bad_attempts = attempts.clone();
    bad_attempts[0].theorem_statement_sha256 = fake_digest("rewritten theorem");
    if validate_all(&snapshot, &programs, &specs, &sample, &bad_attempts, digests.clone(), true).is_ok() {
        return Err("modified frozen statement was accepted".into());
    }
    println!("SELF_TEST_PASS theorem statement mutation is rejected");

    let mut bad_attempts = attempts.clone();
    bad_attempts[0].no_sorry = false;
    if validate_all(&snapshot, &programs, &specs, &sample, &bad_attempts, digests.clone(), true).is_ok() {
        return Err("proof containing sorry was accepted".into());
    }
    println!("SELF_TEST_PASS sorry-bearing proof artifact is rejected");

    let mut bad_programs = programs.clone();
    bad_programs[0].differential_cases_matched = 2;
    if validate_all(&snapshot, &bad_programs, &specs, &sample, &attempts, digests.clone(), true).is_ok() {
        return Err("Rust-Lean differential mismatch was accepted as validated".into());
    }
    println!("SELF_TEST_PASS Rust-Lean differential mismatch is rejected");

    let mut bad_sample = sample.clone();
    bad_sample[2].selected = true;
    bad_sample[2].exclusion_reason = "none".into();
    if validate_all(&snapshot, &programs, &specs, &bad_sample, &attempts, digests.clone(), true).is_ok() {
        return Err("requested-to-fail spec was admitted to the paper's proof-only sample".into());
    }
    println!("SELF_TEST_PASS requested-to-fail sample exclusion is enforced");

    let mut bad_sample = sample.clone();
    bad_sample[3].selected = true;
    bad_sample[3].exclusion_reason = "none".into();
    if validate_all(&snapshot, &programs, &specs, &bad_sample, &attempts, digests.clone(), true).is_ok() {
        return Err("Plausible-refuted task was admitted to the paper's proof-only sample".into());
    }
    println!("SELF_TEST_PASS Plausible-refuted sample exclusion is enforced");

    let mut bad_programs = programs.clone();
    bad_programs[0].reference_case_count = 0;
    if validate_all(&snapshot, &bad_programs, &specs, &sample, &attempts, digests.clone(), true).is_ok() {
        return Err("zero-case differential evidence was accepted".into());
    }
    println!("SELF_TEST_PASS zero-case source/translation evidence is rejected");
    let _ = fs::remove_dir_all(&dir);
    Ok(())
}

fn execute() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    if args.len() == 2 && args[1] == "--self-test" {
        run_self_tests()?;
        println!("PITB_AEGIS_INTEGRITY_SELF_TEST_PASS");
        return Ok(());
    }
    if args.len() == 3 && args[1] == "--self-test-fixture-dir" {
        let report = write_fixture(&PathBuf::from(&args[2]))?;
        if report.snapshot_mode != "synthetic_fixture" {
            return Err("synthetic fixture mode was not preserved".into());
        }
        println!("PITB_AEGIS_INTEGRITY_FIXTURE_OK");
        return Ok(());
    }
    if args.len() != 8 || args[1] != "--verify" {
        return Err(format!(
            "usage: {} --verify <snapshot.json> <programs.jsonl> <specifications.jsonl> <sample.jsonl> <attempts.jsonl> <report.json> OR --self-test-fixture-dir <directory>",
            args.first().map(String::as_str).unwrap_or("pitb-aegis-integrity")
        ));
    }
    let snapshot_bytes = fs::read(&args[2]).map_err(|e| format!("read Aegis snapshot: {e}"))?;
    let snapshot: Snapshot = serde_json::from_slice(&snapshot_bytes).map_err(|e| format!("parse Aegis snapshot: {e}"))?;
    let program_bytes = fs::read(&args[3]).map_err(|e| format!("read program pairs: {e}"))?;
    let spec_bytes = fs::read(&args[4]).map_err(|e| format!("read specifications: {e}"))?;
    let sample_bytes = fs::read(&args[5]).map_err(|e| format!("read sample ledger: {e}"))?;
    let attempt_bytes = fs::read(&args[6]).map_err(|e| format!("read attempts: {e}"))?;
    let programs: Vec<ProgramPair> = parse_jsonl(&program_bytes, "program pairs")?;
    let specifications: Vec<Specification> = parse_jsonl(&spec_bytes, "specifications")?;
    let sample: Vec<SampleEntry> = parse_jsonl(&sample_bytes, "sample ledger")?;
    let attempts: Vec<Attempt> = parse_jsonl(&attempt_bytes, "attempt ledger")?;
    let report = validate_all(
        &snapshot,
        &programs,
        &specifications,
        &sample,
        &attempts,
        [sha256(&program_bytes), sha256(&spec_bytes), sha256(&sample_bytes), sha256(&attempt_bytes)],
        false,
    )?;
    fs::write(&args[7], serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?)
        .map_err(|e| format!("write Aegis report: {e}"))?;
    println!("PITB_AEGIS_INTEGRITY_OK");
    Ok(())
}

fn main() -> ExitCode {
    match execute() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("INVALID: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registered_aegis_integrity_controls_pass() {
        run_self_tests().expect("Aegis integrity controls must pass");
    }
}
