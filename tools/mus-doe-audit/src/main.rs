use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

const DEFAULT_SCHEMA: &str = "schemas/instrument-engineering/mus_doe_split_corpus_v0_1.schema.json";
const DEFAULT_CORPUS: &str = "fixtures/instrument-engineering/mus_doe_split_corpus_v0_1.json";

const REQUIRED_FAULTS: &[&str] = &[
    "physical_unit_cross_partition",
    "build_batch_cross_partition",
    "builder_cross_partition",
    "environment_cross_partition",
    "measurement_session_cross_partition",
    "material_lot_cross_partition",
    "locked_test_outcome_used",
    "inadmissible_experiment_selected",
    "random_seed_missing",
    "posthoc_policy_change",
];

#[derive(Debug, Serialize)]
struct CaseResult {
    case_id: String,
    case_type: String,
    expected_disposition: String,
    computed_disposition: String,
}

#[derive(Debug, Serialize)]
struct AuditReport {
    report_version: &'static str,
    tool_version: &'static str,
    schema_draft: &'static str,
    report_status: &'static str,
    claim_ceiling: &'static str,
    schema_sha256: String,
    corpus_sha256: String,
    case_count: usize,
    positive_control_count: usize,
    negative_control_count: usize,
    findings: Vec<String>,
    case_results: Vec<CaseResult>,
}

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

fn str_at<'a>(v: &'a Value, key: &str) -> &'a str {
    v.get(key).and_then(Value::as_str).unwrap_or("")
}

fn strings_at(v: &Value, key: &str) -> Vec<String> {
    v.get(key).and_then(Value::as_array)
        .map(|xs| xs.iter().filter_map(Value::as_str).map(str::to_owned).collect())
        .unwrap_or_default()
}

fn actual_group_faults(case: &Value) -> BTreeSet<String> {
    let mut faults = BTreeSet::new();
    let axes = strings_at(case, "independence_axes");
    let observations = match case.get("observations").and_then(Value::as_array) {
        Some(rows) => rows,
        None => return faults,
    };
    let mut map_axis_to_fault = BTreeMap::new();
    map_axis_to_fault.insert("physical_unit_id", "physical_unit_cross_partition");
    map_axis_to_fault.insert("build_batch_id", "build_batch_cross_partition");
    map_axis_to_fault.insert("builder_id", "builder_cross_partition");
    map_axis_to_fault.insert("environment_id", "environment_cross_partition");
    map_axis_to_fault.insert("measurement_session_id", "measurement_session_cross_partition");
    map_axis_to_fault.insert("material_lot_id", "material_lot_cross_partition");

    for axis in axes {
        let Some(fault_name) = map_axis_to_fault.get(axis.as_str()) else {
            continue;
        };
        let mut partitions_by_group: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for observation in observations {
            let partition = str_at(observation, "partition").to_owned();
            let group_id = observation.get("group_ids")
                .and_then(|g| g.get(axis.as_str()))
                .and_then(Value::as_str);
            if let Some(group_id) = group_id {
                partitions_by_group.entry(group_id.to_owned()).or_default().insert(partition);
            }
        }
        if partitions_by_group.values().any(|partitions| partitions.len() > 1) {
            faults.insert((*fault_name).to_owned());
        }
    }
    faults
}

fn actual_policy_faults(case: &Value) -> BTreeSet<String> {
    let mut faults = actual_group_faults(case);
    let Some(selection) = case.get("selection") else { return faults };

    let consumed_test = strings_at(selection, "data_partitions_consumed")
        .iter().any(|p| p == "locked_test");
    if selection.get("locked_test_outcomes_used").and_then(Value::as_bool) == Some(true) || consumed_test {
        faults.insert("locked_test_outcome_used".to_owned());
    }
    if selection.get("selected_candidates_admissible").and_then(Value::as_bool) == Some(false) {
        faults.insert("inadmissible_experiment_selected".to_owned());
    }
    if str_at(selection, "policy") == "seeded_random"
        && !selection.get("random_seed").is_some_and(|seed| !seed.is_null())
    {
        faults.insert("random_seed_missing".to_owned());
    }
    if selection.get("posthoc_policy_change").and_then(Value::as_bool) == Some(true) {
        faults.insert("posthoc_policy_change".to_owned());
    }
    faults
}

fn disposition_for_faults(faults: &BTreeSet<String>) -> &'static str {
    if faults.is_empty() {
        return "VALID_SYNTHETIC_SPLIT";
    }
    if faults.contains("locked_test_outcome_used") {
        return "REJECT_TEST_CONTAMINATION";
    }
    if faults.contains("inadmissible_experiment_selected") {
        return "REJECT_INADMISSIBLE_SELECTION";
    }
    if faults.contains("random_seed_missing") {
        return "REJECT_NONDETERMINISTIC_REPLAY";
    }
    if faults.contains("posthoc_policy_change") {
        return "REJECT_POSTHOC_CHANGE";
    }
    "REJECT_GROUP_LEAKAGE"
}

fn semantic_findings(corpus: &Value) -> (Vec<String>, Vec<CaseResult>) {
    let mut findings = Vec::new();
    let Some(cases) = corpus.get("cases").and_then(Value::as_array) else {
        return (vec!["$.cases: missing or not an array".to_owned()], Vec::new());
    };

    let mut seen_ids = BTreeSet::new();
    let mut seen_fault_coverage = BTreeSet::new();
    let mut case_results = Vec::new();

    for (index, case) in cases.iter().enumerate() {
        let path = format!("$.cases[{index}]");
        let id = str_at(case, "case_id");
        if !seen_ids.insert(id.to_owned()) {
            findings.push(format!("{path}.case_id: duplicate identity '{id}'"));
        }

        let case_type = str_at(case, "case_type");
        let fault_entries = strings_at(case, "fault_modes");
        let declared: BTreeSet<String> = fault_entries.iter()
            .filter(|fault| fault.as_str() != "none").cloned().collect();
        seen_fault_coverage.extend(declared.iter().cloned());
        let active_entry_count = fault_entries.iter().filter(|fault| fault.as_str() != "none").count();
        if declared.len() != active_entry_count
            || (fault_entries.iter().any(|fault| fault == "none") && fault_entries.len() != 1)
        {
            findings.push(format!("{path}.fault_modes: duplicate or conflicting fault declaration"));
        }

        let actual = actual_policy_faults(case);
        let expected_disposition = str_at(case, "expected_disposition");
        let computed_disposition = disposition_for_faults(&actual);

        if declared != actual {
            let declared_text = declared.iter().cloned().collect::<Vec<_>>().join(",");
            let actual_text = actual.iter().cloned().collect::<Vec<_>>().join(",");
            findings.push(format!("{path}.fault_modes: declared [{declared_text}] differs from computed [{actual_text}]"));
        }
        if expected_disposition != computed_disposition {
            findings.push(format!("{path}.expected_disposition: expected '{computed_disposition}' from supplied policy fields, got '{expected_disposition}'"));
        }

        if case_type == "positive_control" {
            if !actual.is_empty() || expected_disposition != "VALID_SYNTHETIC_SPLIT" {
                findings.push(format!("{path}: positive control must be a valid split with no policy faults"));
            }
        } else if case_type == "negative_control" {
            if declared.len() != 1 {
                findings.push(format!("{path}: each negative control must isolate exactly one fault"));
            }
            if actual.is_empty() {
                findings.push(format!("{path}: negative control declares no effective policy violation"));
            }
            if expected_disposition == "VALID_SYNTHETIC_SPLIT" {
                findings.push(format!("{path}: negative control cannot be marked valid"));
            }
        }

        case_results.push(CaseResult {
            case_id: id.to_owned(),
            case_type: case_type.to_owned(),
            expected_disposition: expected_disposition.to_owned(),
            computed_disposition: computed_disposition.to_owned(),
        });
    }

    let expected: BTreeSet<String> = REQUIRED_FAULTS.iter().map(|s| (*s).to_owned()).collect();
    for missing in expected.difference(&seen_fault_coverage) {
        findings.push(format!("$.cases: missing negative-control coverage for '{missing}'"));
    }
    for extra in seen_fault_coverage.difference(&expected) {
        findings.push(format!("$.cases: unrecognized fault mode '{extra}'"));
    }

    (findings, case_results)
}

fn audit(schema_path: &PathBuf, corpus_path: &PathBuf) -> Result<AuditReport, String> {
    let schema_bytes = fs::read(schema_path)
        .map_err(|e| format!("cannot read schema {}: {e}", schema_path.display()))?;
    let corpus_bytes = fs::read(corpus_path)
        .map_err(|e| format!("cannot read corpus {}: {e}", corpus_path.display()))?;
    let schema: Value = serde_json::from_slice(&schema_bytes)
        .map_err(|e| format!("schema JSON parse failed: {e}"))?;
    let corpus: Value = serde_json::from_slice(&corpus_bytes)
        .map_err(|e| format!("corpus JSON parse failed: {e}"))?;

    jsonschema::meta::validate(&schema)
        .map_err(|e| format!("schema meta-validation failed: {e}"))?;
    let validator = jsonschema::validator_for(&schema)
        .map_err(|e| format!("schema compilation failed: {e}"))?;
    let mut findings = validator.iter_errors(&corpus)
        .map(|e| format!("schema validation at {}: {e}", e.instance_path()))
        .collect::<Vec<_>>();
    let (semantic, case_results) = semantic_findings(&corpus);
    findings.extend(semantic);

    let cases = corpus.get("cases").and_then(Value::as_array).cloned().unwrap_or_default();
    Ok(AuditReport {
        report_version: "mus-doe-audit-report/0.1",
        tool_version: env!("CARGO_PKG_VERSION"),
        schema_draft: "https://json-schema.org/draft/2020-12/schema",
        report_status: if findings.is_empty() { "CONSISTENT_SYNTHETIC_POLICY_ONLY" } else { "REJECTED" },
        claim_ceiling: "SYNTHETIC_SPLIT_AND_SELECTION_POLICY_INTEGRITY_ONLY",
        schema_sha256: sha256(&schema_bytes),
        corpus_sha256: sha256(&corpus_bytes),
        case_count: cases.len(),
        positive_control_count: cases.iter().filter(|c| str_at(c, "case_type") == "positive_control").count(),
        negative_control_count: cases.iter().filter(|c| str_at(c, "case_type") == "negative_control").count(),
        findings,
        case_results,
    })
}

fn parse_args() -> Result<(PathBuf, PathBuf), String> {
    let mut schema = PathBuf::from(DEFAULT_SCHEMA);
    let mut corpus = PathBuf::from(DEFAULT_CORPUS);
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--schema" => schema = args.next().ok_or("--schema requires a path")?.into(),
            "--corpus" => corpus = args.next().ok_or("--corpus requires a path")?.into(),
            "--help" | "-h" => {
                println!("mus-doe-audit [--schema PATH] [--corpus PATH]\nAudits the MUS-DOE synthetic split/selection-policy corpus.");
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    Ok((schema, corpus))
}

fn main() -> ExitCode {
    let (schema_path, corpus_path) = match parse_args() {
        Ok(paths) => paths,
        Err(error) => {
            eprintln!("mus-doe-audit: {error}");
            return ExitCode::from(2);
        }
    };
    let report = match audit(&schema_path, &corpus_path) {
        Ok(report) => report,
        Err(error) => {
            eprintln!("mus-doe-audit: {error}");
            return ExitCode::from(2);
        }
    };
    let rejected = !report.findings.is_empty();
    println!("{}", serde_json::to_string_pretty(&report).unwrap_or_else(|_| json!({"report_status":"REJECTED","findings":["report serialization failed"]}).to_string()));
    if rejected { ExitCode::from(1) } else { ExitCode::SUCCESS }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corpus() -> Value {
        serde_json::from_str(include_str!("../../../fixtures/instrument-engineering/mus_doe_split_corpus_v0_1.json")).expect("corpus parses")
    }

    #[test]
    fn checked_in_corpus_has_expected_semantics() {
        let (findings, _) = semantic_findings(&corpus());
        assert!(findings.is_empty(), "{}", findings.join("\n"));
    }

    #[test]
    fn duplicate_case_id_is_rejected() {
        let mut data = corpus();
        let first = data["cases"][0].clone();
        data["cases"].as_array_mut().unwrap().push(first);
        assert!(semantic_findings(&data).0.iter().any(|f| f.contains("duplicate identity")));
    }

    #[test]
    fn every_declared_independence_axis_detects_cross_partition_leakage() {
        let axes = [
            ("physical_unit_id", "physical_unit_cross_partition"),
            ("build_batch_id", "build_batch_cross_partition"),
            ("builder_id", "builder_cross_partition"),
            ("environment_id", "environment_cross_partition"),
            ("measurement_session_id", "measurement_session_cross_partition"),
            ("material_lot_id", "material_lot_cross_partition"),
        ];
        for (axis, expected_fault) in axes {
            let mut data = corpus();
            data["cases"][0]["observations"][4]["group_ids"][axis] =
                data["cases"][0]["observations"][0]["group_ids"][axis].clone();
            data["cases"][0]["independence_axes"] = json!([axis]);
            data["cases"][0]["fault_modes"] = json!(["none"]);
            let findings = semantic_findings(&data).0;
            assert!(
                findings.iter().any(|f| f.contains(expected_fault)),
                "axis {axis} did not produce {expected_fault}: {}",
                findings.join("\\n")
            );
        }
    }

    #[test]
    fn locked_test_outcome_is_rejected_even_if_flag_is_false() {
        let mut data = corpus();
        data["cases"][0]["selection"]["data_partitions_consumed"] = json!(["fit", "locked_test"]);
        data["cases"][0]["selection"]["locked_test_outcomes_used"] = json!(false);
        assert!(semantic_findings(&data).0.iter().any(|f| f.contains("locked_test_outcome_used")));
    }

    #[test]
    fn seedless_random_selection_is_rejected() {
        let mut data = corpus();
        data["cases"][0]["selection"]["policy"] = json!("seeded_random");
        data["cases"][0]["selection"]["random_seed"] = Value::Null;
        assert!(semantic_findings(&data).0.iter().any(|f| f.contains("random_seed_missing")));
    }

    #[test]
    fn inadmissible_selected_candidate_is_rejected() {
        let mut data = corpus();
        data["cases"][0]["selection"]["selected_candidates_admissible"] = json!(false);
        assert!(semantic_findings(&data).0.iter().any(|f| f.contains("inadmissible_experiment_selected")));
    }

    #[test]
    fn posthoc_policy_change_is_rejected() {
        let mut data = corpus();
        data["cases"][0]["selection"]["posthoc_policy_change"] = json!(true);
        assert!(semantic_findings(&data).0.iter().any(|f| f.contains("posthoc_policy_change")));
    }

    #[test]
    fn declared_fault_without_matching_evidence_is_rejected() {
        let mut data = corpus();
        data["cases"][0]["fault_modes"] = json!(["physical_unit_cross_partition"]);
        assert!(semantic_findings(&data).0.iter().any(|f| f.contains("differs from computed")));
    }
}
