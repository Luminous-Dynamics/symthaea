use jsonschema::validator_for;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::process::ExitCode;

const SCHEMA_VERSION: &str = "sc-soft-mode-triage.v0.1";
const CLAIM_CEILING: &str = "synthetic-soft-mode-disposition-and-recall-semantics-only";
const SCHEMA_PATH: &str = "data/schemas/sc_soft_mode_triage_v0_1.schema.json";
const FIXTURE_PATH: &str = "data/research/sc_soft_mode_triage_fixture_v0_1.json";

#[derive(Debug, Clone)]
struct PhononRun {
    run_id: String,
    convergence_profile_id: String,
    functional_id: String,
    pseudopotential_set_id: String,
    q_mesh: [u32; 3],
    supercell: [u32; 3],
    electronic_cutoff_ev: f64,
    displacement_amplitude_angstrom: f64,
    converged: bool,
    minimum_frequency_cm1: f64,
    frequency_uncertainty_cm1: f64,
    max_force_residual_mev_per_angstrom: f64,
}

#[derive(Debug, Clone)]
struct Candidate {
    candidate_id: String,
    material_family_id: String,
    condition_profile_id: String,
    reference_case_id: Option<String>,
    runs: Vec<PhononRun>,
    truth_label: String,
}

#[derive(Debug, Clone)]
struct Thresholds {
    imaginary_tolerance_cm1: f64,
    minimum_converged_runs: usize,
    maximum_force_residual_mev_per_angstrom: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Disposition {
    HarmonicStable,
    MarginalSoftMode,
    RobustHarmonicInstability,
    UnderconvergedUnknown,
}

impl Disposition {
    fn as_str(&self) -> &'static str {
        match self {
            Self::HarmonicStable => "HARMONIC_STABLE_FOR_DECLARED_METHOD",
            Self::MarginalSoftMode => "MARGINAL_SOFT_MODE_UNRESOLVED",
            Self::RobustHarmonicInstability => "ROBUST_HARMONIC_INSTABILITY",
            Self::UnderconvergedUnknown => "UNDERCONVERGED_OR_UNKNOWN",
        }
    }

    fn next_action(&self) -> &'static str {
        match self {
            Self::HarmonicStable => "CONTINUE_ELECTRON_PHONON_EVALUATION",
            Self::MarginalSoftMode => "ESCALATE_CONVERGENCE_OR_ANHARMONIC_REVIEW",
            Self::RobustHarmonicInstability => "FOLLOW_LOWER_SYMMETRY_DISTORTION_PATH",
            Self::UnderconvergedUnknown => "ABSTAIN_AND_ACQUIRE_REQUIRED_EVIDENCE",
        }
    }

    fn queue_status(&self) -> &'static str {
        match self {
            Self::HarmonicStable => "CONTINUE",
            Self::MarginalSoftMode => "RETAIN_FOR_REEVALUATION",
            Self::RobustHarmonicInstability => "ROUTE_TO_DISTORTED_PHASE_SEARCH",
            Self::UnderconvergedUnknown => "HOLD_UNKNOWN",
        }
    }
}

fn fail(path: &str, message: impl AsRef<str>) -> String {
    format!("{path}: {}", message.as_ref())
}

fn object<'a>(value: &'a Value, path: &str) -> Result<&'a Map<String, Value>, String> {
    value.as_object().ok_or_else(|| fail(path, "must be an object"))
}

fn array<'a>(value: &'a Value, path: &str) -> Result<&'a Vec<Value>, String> {
    value.as_array().ok_or_else(|| fail(path, "must be an array"))
}

fn exact_keys(value: &Value, required: &[&str], path: &str) -> Result<(), String> {
    let map = object(value, path)?;
    for key in required {
        if !map.contains_key(*key) {
            return Err(fail(path, format!("missing property {key}")));
        }
    }
    if let Some(key) = map.keys().find(|key| !required.contains(&key.as_str())) {
        return Err(fail(path, format!("unknown property {key}")));
    }
    Ok(())
}

fn text_field<'a>(value: &'a Value, key: &str, path: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|text| !text.trim().is_empty())
        .ok_or_else(|| fail(path, format!("{key} must be a non-empty string")))
}

fn number_field(value: &Value, key: &str, path: &str) -> Result<f64, String> {
    let number = value
        .get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| fail(path, format!("{key} must be finite")))?;
    if !number.is_finite() {
        return Err(fail(path, format!("{key} must be finite")));
    }
    Ok(number)
}

fn positive_number(value: &Value, key: &str, path: &str) -> Result<f64, String> {
    let number = number_field(value, key, path)?;
    if number <= 0.0 {
        return Err(fail(path, format!("{key} must be positive")));
    }
    Ok(number)
}

fn nonnegative_number(value: &Value, key: &str, path: &str) -> Result<f64, String> {
    let number = number_field(value, key, path)?;
    if number < 0.0 {
        return Err(fail(path, format!("{key} must be non-negative")));
    }
    Ok(number)
}

fn usize_field(value: &Value, key: &str, path: &str) -> Result<usize, String> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|number| usize::try_from(number).ok())
        .ok_or_else(|| fail(path, format!("{key} must be a non-negative integer")))
}

fn bool_field(value: &Value, key: &str, path: &str) -> Result<bool, String> {
    value
        .get(key)
        .and_then(Value::as_bool)
        .ok_or_else(|| fail(path, format!("{key} must be boolean")))
}

fn positive_integer_triplet(value: &Value, key: &str, path: &str) -> Result<[u32; 3], String> {
    let values = array(
        value.get(key).ok_or_else(|| fail(path, format!("missing property {key}")))?,
        &format!("{path}.{key}"),
    )?;
    if values.len() != 3 {
        return Err(fail(path, format!("{key} must contain exactly three positive integers")));
    }
    let mut result = [0_u32; 3];
    for (index, item) in values.iter().enumerate() {
        let raw = item
            .as_u64()
            .ok_or_else(|| fail(path, format!("{key}[{index}] must be a positive integer")))?;
        result[index] = u32::try_from(raw)
            .ok()
            .filter(|number| *number > 0)
            .ok_or_else(|| fail(path, format!("{key}[{index}] is outside the valid range")))?;
    }
    Ok(result)
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn validate_sha256(value: &str, path: &str) -> Result<(), String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(fail(path, "must be lowercase 64-character SHA-256 hex"));
    }
    Ok(())
}

fn parse_thresholds(value: &Value) -> Result<Thresholds, String> {
    const KEYS: &[&str] = &[
        "imaginary_tolerance_cm1",
        "minimum_converged_runs",
        "maximum_force_residual_mev_per_angstrom",
    ];
    exact_keys(value, KEYS, "$.thresholds")?;
    let minimum_converged_runs = usize_field(value, "minimum_converged_runs", "$.thresholds")?;
    if minimum_converged_runs < 2 {
        return Err(fail("$.thresholds", "minimum_converged_runs must be at least two"));
    }
    Ok(Thresholds {
        imaginary_tolerance_cm1: positive_number(
            value,
            "imaginary_tolerance_cm1",
            "$.thresholds",
        )?,
        minimum_converged_runs,
        maximum_force_residual_mev_per_angstrom: positive_number(
            value,
            "maximum_force_residual_mev_per_angstrom",
            "$.thresholds",
        )?,
    })
}

fn parse_run(value: &Value, path: &str) -> Result<PhononRun, String> {
    const KEYS: &[&str] = &[
        "run_id",
        "convergence_profile_id",
        "method_scope",
        "functional_id",
        "pseudopotential_set_id",
        "q_mesh",
        "supercell",
        "electronic_cutoff_ev",
        "displacement_amplitude_angstrom",
        "converged",
        "minimum_frequency_cm1",
        "frequency_uncertainty_cm1",
        "max_force_residual_mev_per_angstrom",
    ];
    exact_keys(value, KEYS, path)?;
    let method_scope = text_field(value, "method_scope", path)?.to_owned();
    if method_scope != "harmonic_0k_same_structure" {
        return Err(fail(
            path,
            "v0.1 only compares runs with method_scope=harmonic_0k_same_structure",
        ));
    }
    Ok(PhononRun {
        run_id: text_field(value, "run_id", path)?.to_owned(),
        convergence_profile_id: text_field(value, "convergence_profile_id", path)?.to_owned(),
        functional_id: text_field(value, "functional_id", path)?.to_owned(),
        pseudopotential_set_id: text_field(value, "pseudopotential_set_id", path)?.to_owned(),
        q_mesh: positive_integer_triplet(value, "q_mesh", path)?,
        supercell: positive_integer_triplet(value, "supercell", path)?,
        electronic_cutoff_ev: positive_number(value, "electronic_cutoff_ev", path)?,
        displacement_amplitude_angstrom: positive_number(
            value,
            "displacement_amplitude_angstrom",
            path,
        )?,
        converged: bool_field(value, "converged", path)?,
        minimum_frequency_cm1: number_field(value, "minimum_frequency_cm1", path)?,
        frequency_uncertainty_cm1: nonnegative_number(
            value,
            "frequency_uncertainty_cm1",
            path,
        )?,
        max_force_residual_mev_per_angstrom: nonnegative_number(
            value,
            "max_force_residual_mev_per_angstrom",
            path,
        )?,
    })
}

fn parse_candidate(value: &Value, path: &str) -> Result<Candidate, String> {
    const KEYS: &[&str] = &[
        "candidate_id",
        "material_family_id",
        "condition_profile_id",
        "structure_sha256",
        "reference_case_id",
        "phonon_runs",
        "evaluation_truth",
    ];
    exact_keys(value, KEYS, path)?;
    let structure_sha256 = text_field(value, "structure_sha256", path)?.to_owned();
    validate_sha256(&structure_sha256, &format!("{path}.structure_sha256"))?;
    let reference_case_id = value
        .get("reference_case_id")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let run_values = array(
        value
            .get("phonon_runs")
            .ok_or_else(|| fail(path, "missing phonon_runs"))?,
        &format!("{path}.phonon_runs"),
    )?;
    let mut run_ids = BTreeSet::new();
    let mut convergence_profile_ids = BTreeSet::new();
    let mut runs = Vec::with_capacity(run_values.len());
    let mut reference_method: Option<(String, String)> = None;
    for (index, run) in run_values.iter().enumerate() {
        let parsed = parse_run(run, &format!("{path}.phonon_runs[{index}]"))?;
        if !run_ids.insert(parsed.run_id.clone()) {
            return Err(fail(path, "duplicate phonon run_id"));
        }
        if !convergence_profile_ids.insert(parsed.convergence_profile_id.clone()) {
            return Err(fail(path, "duplicate convergence_profile_id"));
        }
        let method = (
            parsed.functional_id.clone(),
            parsed.pseudopotential_set_id.clone(),
        );
        if let Some(expected) = &reference_method {
            if expected != &method {
                return Err(fail(
                    path,
                    "convergence runs must use the same functional and pseudopotential set",
                ));
            }
        } else {
            reference_method = Some(method);
        }
        runs.push(parsed);
    }
    let truth = text_field(value, "evaluation_truth", path)?.to_owned();
    if ![
        "synthetic_known_positive",
        "synthetic_known_negative",
        "synthetic_unknown",
    ]
    .contains(&truth.as_str())
    {
        return Err(fail(path, "unsupported synthetic evaluation_truth"));
    }
    Ok(Candidate {
        candidate_id: text_field(value, "candidate_id", path)?.to_owned(),
        material_family_id: text_field(value, "material_family_id", path)?.to_owned(),
        condition_profile_id: text_field(value, "condition_profile_id", path)?.to_owned(),
        reference_case_id,
        runs,
        truth_label: truth,
    })
}

fn disposition(candidate: &Candidate, thresholds: &Thresholds) -> Disposition {
    if candidate.runs.len() < thresholds.minimum_converged_runs {
        return Disposition::UnderconvergedUnknown;
    }
    if candidate.runs.iter().any(|run| {
        !run.converged
            || run.max_force_residual_mev_per_angstrom
                > thresholds.maximum_force_residual_mev_per_angstrom
    }) {
        return Disposition::UnderconvergedUnknown;
    }
    let distinct_convergence_profiles = candidate
        .runs
        .iter()
        .map(|run| run.convergence_profile_id.as_str())
        .collect::<BTreeSet<_>>();
    let distinct_mesh_supercell_settings = candidate
        .runs
        .iter()
        .map(|run| (run.q_mesh, run.supercell))
        .collect::<BTreeSet<_>>();
    if distinct_convergence_profiles.len() < thresholds.minimum_converged_runs
        || distinct_mesh_supercell_settings.len() < thresholds.minimum_converged_runs
    {
        return Disposition::UnderconvergedUnknown;
    }

    let mut all_stable = true;
    let mut all_robustly_imaginary = true;
    for run in &candidate.runs {
        let lower = run.minimum_frequency_cm1 - run.frequency_uncertainty_cm1;
        let upper = run.minimum_frequency_cm1 + run.frequency_uncertainty_cm1;
        if lower < -thresholds.imaginary_tolerance_cm1 {
            all_stable = false;
        }
        if upper >= -thresholds.imaginary_tolerance_cm1 {
            all_robustly_imaginary = false;
        }
    }
    if all_stable {
        Disposition::HarmonicStable
    } else if all_robustly_imaginary {
        Disposition::RobustHarmonicInstability
    } else {
        Disposition::MarginalSoftMode
    }
}

fn parse_source_reference_cases(values: &[Value]) -> Result<BTreeSet<String>, String> {
    let mut ids = BTreeSet::new();
    for (index, case) in values.iter().enumerate() {
        let path = format!("$.source_reference_cases[{index}]");
        exact_keys(
            case,
            &[
                "reference_case_id",
                "material_formula",
                "source_doi",
                "source_url",
                "evidence_class",
                "reported_soft_mode_status",
                "reported_experimental_tc_k",
                "reported_experimental_evidence",
                "raw_phonon_artifact_status",
                "claim_ceiling",
            ],
            &path,
        )?;
        let id = text_field(case, "reference_case_id", &path)?.to_owned();
        if !ids.insert(id) {
            return Err(fail(&path, "duplicate reference_case_id"));
        }
        if text_field(case, "evidence_class", &path)? != "source_reported_literature" {
            return Err(fail(&path, "source reference must remain source_reported_literature"));
        }
        if text_field(case, "raw_phonon_artifact_status", &path)?
            != "not_admitted_no_numeric_frequency_values"
        {
            return Err(fail(
                &path,
                "source metadata must not fabricate numeric phonon values",
            ));
        }
        if text_field(case, "claim_ceiling", &path)? != "source-reported-positive-control-only" {
            return Err(fail(&path, "source reference claim ceiling mismatch"));
        }
        positive_number(case, "reported_experimental_tc_k", &path)?;
        text_field(case, "reported_experimental_evidence", &path)?;
        text_field(case, "reported_soft_mode_status", &path)?;
        text_field(case, "source_doi", &path)?;
        text_field(case, "source_url", &path)?;
        text_field(case, "material_formula", &path)?;
    }
    if ids.is_empty() {
        return Err(fail(
            "$.source_reference_cases",
            "must contain at least one source-linked positive control",
        ));
    }
    Ok(ids)
}

fn hard_cut_rejects(candidate: &Candidate, threshold: f64) -> bool {
    candidate
        .runs
        .iter()
        .any(|run| run.minimum_frequency_cm1 < -threshold)
}

fn audit(value: &Value) -> Result<Value, String> {
    exact_keys(
        value,
        &[
            "schema_version",
            "evidence_class",
            "claim_ceiling",
            "thresholds",
            "source_reference_cases",
            "synthetic_candidates",
        ],
        "$",
    )?;
    if text_field(value, "schema_version", "$")? != SCHEMA_VERSION {
        return Err(fail("$.schema_version", "unsupported schema version"));
    }
    if text_field(value, "evidence_class", "$")? != "synthetic" {
        return Err(fail("$.evidence_class", "auditor accepts synthetic fixture only"));
    }
    if text_field(value, "claim_ceiling", "$")? != CLAIM_CEILING {
        return Err(fail("$.claim_ceiling", "claim ceiling mismatch"));
    }
    let thresholds = parse_thresholds(&value["thresholds"])?;
    let source_ids = parse_source_reference_cases(array(
        &value["source_reference_cases"],
        "$.source_reference_cases",
    )?)?;

    let candidate_values = array(&value["synthetic_candidates"], "$.synthetic_candidates")?;
    if candidate_values.len() < 4 {
        return Err(fail(
            "$.synthetic_candidates",
            "must contain at least four synthetic controls",
        ));
    }
    let mut candidate_ids = BTreeSet::new();
    let mut candidates = Vec::with_capacity(candidate_values.len());
    for (index, candidate_value) in candidate_values.iter().enumerate() {
        let candidate = parse_candidate(
            candidate_value,
            &format!("$.synthetic_candidates[{index}]"),
        )?;
        if !candidate_ids.insert(candidate.candidate_id.clone()) {
            return Err(fail("$.synthetic_candidates", "duplicate candidate_id"));
        }
        if let Some(reference_id) = &candidate.reference_case_id {
            if !source_ids.contains(reference_id) {
                return Err(fail(
                    "$.synthetic_candidates",
                    "candidate references an unknown source reference case",
                ));
            }
        }
        candidates.push(candidate);
    }

    let mut rows = Vec::with_capacity(candidates.len());
    let mut hard_false_negatives = 0usize;
    let mut triage_preserved_positives = 0usize;
    let mut unsupported_stability_claims = 0usize;
    let mut disposition_counts = BTreeMap::<String, usize>::new();
    for candidate in &candidates {
        let result = disposition(candidate, &thresholds);
        let hard_rejected = hard_cut_rejects(candidate, thresholds.imaginary_tolerance_cm1);
        let known_positive = candidate.truth_label == "synthetic_known_positive";
        if hard_rejected && known_positive {
            hard_false_negatives += 1;
        }
        if known_positive && result != Disposition::RobustHarmonicInstability {
            triage_preserved_positives += 1;
        }
        if result == Disposition::HarmonicStable
            && candidate.runs.iter().any(|run| {
                run.minimum_frequency_cm1 - run.frequency_uncertainty_cm1
                    < -thresholds.imaginary_tolerance_cm1
            })
        {
            unsupported_stability_claims += 1;
        }
        *disposition_counts
            .entry(result.as_str().to_owned())
            .or_insert(0) += 1;
        rows.push(json!({
            "candidate_id": candidate.candidate_id,
            "material_family_id": candidate.material_family_id,
            "condition_profile_id": candidate.condition_profile_id,
            "reference_case_id": candidate.reference_case_id,
            "synthetic_evaluation_truth": candidate.truth_label,
            "run_count": candidate.runs.len(),
            "hard_any_imaginary_gate_rejects": hard_rejected,
            "disposition": result.as_str(),
            "next_action": result.next_action(),
            "queue_status": result.queue_status()
        }));
    }
    let known_positive_count = candidates
        .iter()
        .filter(|candidate| candidate.truth_label == "synthetic_known_positive")
        .count();
    let hard_gate_positive_recall = if known_positive_count == 0 {
        Value::Null
    } else {
        json!((known_positive_count - hard_false_negatives) as f64 / known_positive_count as f64)
    };
    let triage_positive_recall = if known_positive_count == 0 {
        Value::Null
    } else {
        json!(triage_preserved_positives as f64 / known_positive_count as f64)
    };
    Ok(json!({
        "schema_version": "sc-soft-mode-triage-receipt.v0.1",
        "source_reference_case_count": source_ids.len(),
        "synthetic_candidate_count": candidates.len(),
        "known_positive_count": known_positive_count,
        "hard_any_imaginary_gate_false_negatives": hard_false_negatives,
        "hard_any_imaginary_gate_positive_recall": hard_gate_positive_recall,
        "triage_policy_preserved_positive_count": triage_preserved_positives,
        "triage_policy_positive_preservation_fraction": triage_positive_recall,
        "unsupported_harmonic_stability_claim_count": unsupported_stability_claims,
        "disposition_counts": disposition_counts,
        "candidate_dispositions": rows,
        "claim_ceiling": CLAIM_CEILING
    }))
}

fn run() -> Result<(), String> {
    let mut schema_path = SCHEMA_PATH.to_owned();
    let mut fixture_path = FIXTURE_PATH.to_owned();
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        let value = args.next().ok_or_else(|| format!("missing value for {arg}"))?;
        match arg.as_str() {
            "--schema" => schema_path = value,
            "--fixture" => fixture_path = value,
            other => return Err(format!("unsupported argument {other}")),
        }
    }
    let schema_bytes = fs::read(&schema_path)
        .map_err(|error| format!("cannot read schema {schema_path}: {error}"))?;
    let fixture_bytes = fs::read(&fixture_path)
        .map_err(|error| format!("cannot read fixture {fixture_path}: {error}"))?;
    let schema: Value = serde_json::from_slice(&schema_bytes)
        .map_err(|error| format!("invalid schema JSON: {error}"))?;
    let fixture: Value = serde_json::from_slice(&fixture_bytes)
        .map_err(|error| format!("invalid fixture JSON: {error}"))?;
    let validator = validator_for(&schema)
        .map_err(|error| format!("schema compilation failed: {error}"))?;
    if let Some(error) = validator.iter_errors(&fixture).next() {
        return Err(format!(
            "fixture schema validation failed: {} at {}",
            error,
            error.instance_path()
        ));
    }
    let mut receipt = audit(&fixture)?;
    receipt["schema_sha256"] = json!(sha256_hex(&schema_bytes));
    receipt["fixture_sha256"] = json!(sha256_hex(&fixture_bytes));
    println!(
        "{}",
        serde_json::to_string_pretty(&receipt)
            .map_err(|error| format!("receipt serialization failed: {error}"))?
    );
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("sc_soft_mode_triage_audit: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Value {
        serde_json::from_str(include_str!(
            "../../../../data/research/sc_soft_mode_triage_fixture_v0_1.json"
        ))
        .expect("soft-mode fixture must parse")
    }

    #[test]
    fn synthetic_fixture_distinguishes_all_four_dispositions() {
        let receipt = audit(&fixture()).expect("fixture should validate");
        assert_eq!(receipt["synthetic_candidate_count"], 5);
        let rows = receipt["candidate_dispositions"].as_array().unwrap();
        assert_eq!(rows[0]["disposition"], "MARGINAL_SOFT_MODE_UNRESOLVED");
        assert_eq!(rows[1]["disposition"], "HARMONIC_STABLE_FOR_DECLARED_METHOD");
        assert_eq!(rows[2]["disposition"], "ROBUST_HARMONIC_INSTABILITY");
        assert_eq!(rows[3]["disposition"], "UNDERCONVERGED_OR_UNKNOWN");
    }

    #[test]
    fn hard_any_imaginary_gate_misses_synthetic_positive_control_but_triage_retains_it() {
        let receipt = audit(&fixture()).expect("fixture should validate");
        assert_eq!(receipt["hard_any_imaginary_gate_false_negatives"], 1);
        assert_eq!(receipt["triage_policy_preserved_positive_count"], 1);
        assert_eq!(receipt["unsupported_harmonic_stability_claim_count"], 0);
    }

    #[test]
    fn source_linked_luru3b2_case_has_no_fabricated_phonon_frequency() {
        let value = fixture();
        let source = &value["source_reference_cases"][0];
        assert_eq!(source["material_formula"], "LuRu3B2");
        assert_eq!(
            source["raw_phonon_artifact_status"],
            "not_admitted_no_numeric_frequency_values"
        );
        assert!(source.get("minimum_frequency_cm1").is_none());
    }

    #[test]
    fn changing_synthetic_truth_does_not_change_physical_disposition() {
        let mut value = fixture();
        let before = audit(&value).expect("fixture should validate");
        value["synthetic_candidates"][0]["evaluation_truth"] =
            json!("synthetic_known_negative");
        let after = audit(&value).expect("mutated fixture remains structurally valid");
        assert_eq!(
            before["candidate_dispositions"][0]["disposition"],
            after["candidate_dispositions"][0]["disposition"]
        );
    }

    #[test]
    fn rejects_threshold_below_zero_or_zero() {
        let mut value = fixture();
        value["thresholds"]["imaginary_tolerance_cm1"] = json!(0.0);
        assert!(
            audit(&value)
                .unwrap_err()
                .contains("must be positive")
        );
    }

    #[test]
    fn rejects_only_one_converged_run_as_insufficient_design_threshold() {
        let mut value = fixture();
        value["thresholds"]["minimum_converged_runs"] = json!(1);
        assert!(
            audit(&value)
                .unwrap_err()
                .contains("must be at least two")
        );
    }

    #[test]
    fn underconverged_run_is_never_harmonic_stable() {
        let mut value = fixture();
        value["synthetic_candidates"][1]["phonon_runs"][0]["converged"] = json!(false);
        let receipt = audit(&value).expect("fixture remains valid");
        assert_eq!(
            receipt["candidate_dispositions"][1]["disposition"],
            "UNDERCONVERGED_OR_UNKNOWN"
        );
    }

    #[test]
    fn method_scope_mismatch_is_rejected_not_pooled() {
        let mut value = fixture();
        value["synthetic_candidates"][0]["phonon_runs"][1]["method_scope"] =
            json!("finite_temperature_anharmonic");
        assert!(
            audit(&value)
                .unwrap_err()
                .contains("only compares runs")
        );
    }

    #[test]
    fn candidate_structure_digest_must_be_sha256() {
        let mut value = fixture();
        value["synthetic_candidates"][0]["structure_sha256"] = json!("bad");
        assert!(
            audit(&value)
                .unwrap_err()
                .contains("SHA-256")
        );
    }

    #[test]
    fn rejects_duplicate_candidate_ids() {
        let mut value = fixture();
        let first = value["synthetic_candidates"][0]["candidate_id"].clone();
        value["synthetic_candidates"][1]["candidate_id"] = first;
        assert!(
            audit(&value)
                .unwrap_err()
                .contains("duplicate candidate_id")
        );
    }

    #[test]
    fn rejects_duplicate_run_ids_within_candidate() {
        let mut value = fixture();
        let first = value["synthetic_candidates"][0]["phonon_runs"][0]["run_id"].clone();
        value["synthetic_candidates"][0]["phonon_runs"][1]["run_id"] = first;
        assert!(
            audit(&value)
                .unwrap_err()
                .contains("duplicate phonon run_id")
        );
    }

    #[test]
    fn convergence_replicates_must_span_distinct_q_mesh_and_supercell_settings() {
        let mut value = fixture();
        value["synthetic_candidates"][1]["phonon_runs"][1]["q_mesh"] =
            value["synthetic_candidates"][1]["phonon_runs"][0]["q_mesh"].clone();
        value["synthetic_candidates"][1]["phonon_runs"][1]["supercell"] =
            value["synthetic_candidates"][1]["phonon_runs"][0]["supercell"].clone();
        let receipt = audit(&value).expect("duplicate settings remain a valid but underconverged record");
        assert_eq!(
            receipt["candidate_dispositions"][1]["disposition"],
            "UNDERCONVERGED_OR_UNKNOWN"
        );
    }

    #[test]
    fn convergence_profile_ids_must_be_unique() {
        let mut value = fixture();
        let first = value["synthetic_candidates"][0]["phonon_runs"][0]
            ["convergence_profile_id"]
            .clone();
        value["synthetic_candidates"][0]["phonon_runs"][1]["convergence_profile_id"] = first;
        assert!(
            audit(&value)
                .unwrap_err()
                .contains("duplicate convergence_profile_id")
        );
    }

    #[test]
    fn convergence_runs_cannot_pool_different_functionals_or_pseudopotentials() {
        let mut value = fixture();
        value["synthetic_candidates"][0]["phonon_runs"][1]["functional_id"] =
            json!("OTHER-FUNCTIONAL");
        assert!(
            audit(&value)
                .unwrap_err()
                .contains("same functional and pseudopotential set")
        );
    }

    #[test]
    fn q_mesh_and_supercell_triplets_must_be_positive() {
        let mut value = fixture();
        value["synthetic_candidates"][0]["phonon_runs"][0]["q_mesh"] =
            json!([6, 0, 6]);
        assert!(
            audit(&value)
                .unwrap_err()
                .contains("q_mesh[1] is outside the valid range")
        );
    }

    #[test]
    fn robust_instability_routes_to_distortion_path_not_global_candidate_rejection() {
        let receipt = audit(&fixture()).expect("fixture should validate");
        let rows = receipt["candidate_dispositions"].as_array().unwrap();
        assert_eq!(rows[2]["queue_status"], "ROUTE_TO_DISTORTED_PHASE_SEARCH");
        assert_eq!(rows[2]["next_action"], "FOLLOW_LOWER_SYMMETRY_DISTORTION_PATH");
    }
}
