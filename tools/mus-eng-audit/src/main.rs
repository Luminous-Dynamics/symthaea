use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

const DEFAULT_SCHEMA: &str = "schemas/instrument-engineering/mus_eng_corpus_v0_1.schema.json";
const DEFAULT_CORPUS: &str = "fixtures/instrument-engineering/mus_eng_corpus_v0_1.json";

#[derive(Debug, Serialize)]
struct CaseResult {
    case_id: String,
    case_type: String,
    candidate_claim: String,
    declared_disposition: String,
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
    negative_control_count: usize,
    negative_controls_marked_passing: usize,
    case_results: Vec<CaseResult>,
    findings: Vec<String>,
}

fn sha256(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn string_at<'a>(value: &'a Value, key: &str) -> &'a str {
    value.get(key).and_then(Value::as_str).unwrap_or("")
}

fn strings_at(value: &Value, key: &str) -> Vec<String> {
    value
        .get(key)
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

fn required_evidence(claim: &str) -> &'static [&'static str] {
    match claim {
        "digital_contract_integrity" => &["requirements", "design_manifest", "bom"],
        "analytical_prediction" => &["requirements", "design_manifest", "model_report"],
        "numerical_prediction" => &[
            "requirements",
            "design_manifest",
            "model_report",
            "numerical_solver_receipt",
        ],
        "surrogate_prediction" => &["requirements", "design_manifest", "model_report"],
        "physical_measurement" => &[
            "requirements",
            "design_manifest",
            "calibration_record",
            "raw_measurement",
        ],
        "human_perceptual_claim" => &[
            "requirements",
            "human_study_protocol",
            "human_study_results",
        ],
        "release_readiness" => &[
            "design_manifest",
            "bom",
            "as_built_manifest",
            "license_manifest",
        ],
        "repair_success" => &["design_manifest", "as_built_manifest", "post_repair_test"],
        "reproducibility_claim" => &[
            "requirements",
            "design_manifest",
            "repeated_trials",
            "complete_trial_denominator",
        ],
        _ => &[],
    }
}

fn expected_disposition_for_faults(faults: &[String]) -> Option<&'static str> {
    if faults.is_empty() || faults.iter().all(|fault| fault == "none") {
        return None;
    }
    if faults
        .iter()
        .any(|fault| fault == "unresolved_asset_rights")
    {
        Some("BLOCK_RELEASE")
    } else {
        Some("REJECT_CLAIM")
    }
}

fn expected_ceiling_for_single_fault(fault: &str) -> Option<&'static str> {
    match fault {
        "simulation_promoted_to_measurement" | "expired_calibration" | "stale_design_revision" => {
            Some("physical_claim_rejected")
        }
        "unbound_as_built_substitution" | "cherry_picked_trials" => Some("none"),
        "unresolved_asset_rights" => Some("release_blocked"),
        "missing_post_repair_test" => Some("repair_claim_rejected"),
        "human_claim_without_protocol" => Some("human_claim_rejected"),
        "none" => None,
        _ => None,
    }
}

fn semantic_findings(corpus: &Value) -> Vec<String> {
    let mut findings = Vec::new();
    let Some(cases) = corpus.get("cases").and_then(Value::as_array) else {
        findings.push("$.cases: missing or not an array".to_owned());
        return findings;
    };

    let mut ids = BTreeSet::new();
    for (index, case) in cases.iter().enumerate() {
        let prefix = format!("$.cases[{index}]");
        let id = string_at(case, "case_id").to_owned();
        let case_type = string_at(case, "case_type");
        let claim = string_at(case, "candidate_claim");
        let disposition = string_at(case, "expected_disposition");
        let ceiling = string_at(case, "evidence_ceiling");
        let available = strings_at(case, "available_evidence");
        let missing = strings_at(case, "missing_evidence");
        let faults = strings_at(case, "faults");
        let available_set: BTreeSet<&str> = available.iter().map(String::as_str).collect();
        let missing_set: BTreeSet<&str> = missing.iter().map(String::as_str).collect();

        if !ids.insert(id.clone()) {
            findings.push(format!("{prefix}.case_id: duplicate identity {id}"));
        }

        for item in &available {
            if missing_set.contains(item.as_str()) {
                findings.push(format!(
                    "{prefix}: evidence '{item}' is both available and missing"
                ));
            }
        }

        if faults.iter().any(|fault| fault == "none") && faults.len() != 1 {
            findings.push(format!(
                "{prefix}.faults: 'none' cannot coexist with another fault"
            ));
        }

        if case_type == "negative_control" && disposition == "CONSISTENT_UNQUALIFIED" {
            findings.push(format!("{prefix}: negative control is marked as passing"));
        }

        if case_type == "positive_control" {
            if faults.len() != 1 || faults[0] != "none" {
                findings.push(format!(
                    "{prefix}: positive control must declare exactly the 'none' fault"
                ));
            }
            if disposition != "CONSISTENT_UNQUALIFIED" {
                findings.push(format!(
                    "{prefix}: positive control must remain internally consistent and unqualified"
                ));
            }
            let expected_ceiling = match claim {
                "digital_contract_integrity" => Some("schema_only"),
                "analytical_prediction" | "numerical_prediction" | "surrogate_prediction" => {
                    Some("model_only")
                }
                _ => None,
            };
            if expected_ceiling.is_some_and(|expected| expected != ceiling) {
                findings.push(format!("{prefix}: evidence ceiling '{ceiling}' exceeds or differs from the expected prediction-only ceiling"));
            }
        }

        let required = required_evidence(claim);
        let absent_required: Vec<&str> = required
            .iter()
            .copied()
            .filter(|field| !available_set.contains(field))
            .collect();
        for evidence in &absent_required {
            if !missing_set.contains(evidence) {
                findings.push(format!("{prefix}: required evidence '{evidence}' is neither available nor declared missing"));
            }
        }
        if !absent_required.is_empty() && disposition == "CONSISTENT_UNQUALIFIED" {
            findings.push(format!(
                "{prefix}: claim is marked consistent despite absent required evidence: {}",
                absent_required.join(", ")
            ));
        }

        if let Some(expected) = expected_disposition_for_faults(&faults) {
            if disposition != expected {
                findings.push(format!(
                    "{prefix}: faults require disposition {expected}, got {disposition}"
                ));
            }
        }

        let active_faults: Vec<&str> = faults
            .iter()
            .map(String::as_str)
            .filter(|fault| *fault != "none")
            .collect();
        if active_faults.len() == 1 {
            if let Some(expected_ceiling) = expected_ceiling_for_single_fault(active_faults[0]) {
                if ceiling != expected_ceiling {
                    findings.push(format!(
                        "{prefix}: fault '{}' requires evidence ceiling '{}', got '{}'",
                        active_faults[0], expected_ceiling, ceiling
                    ));
                }
            }
        }

        if faults.is_empty() && case_type == "negative_control" {
            findings.push(format!("{prefix}: negative control has no declared fault"));
        }
    }

    findings
}

fn audit(schema_path: &PathBuf, corpus_path: &PathBuf) -> Result<AuditReport, String> {
    let schema_bytes = fs::read(schema_path)
        .map_err(|error| format!("cannot read schema {}: {error}", schema_path.display()))?;
    let corpus_bytes = fs::read(corpus_path)
        .map_err(|error| format!("cannot read corpus {}: {error}", corpus_path.display()))?;
    let schema: Value = serde_json::from_slice(&schema_bytes)
        .map_err(|error| format!("schema JSON parse failed: {error}"))?;
    let corpus: Value = serde_json::from_slice(&corpus_bytes)
        .map_err(|error| format!("corpus JSON parse failed: {error}"))?;

    jsonschema::meta::validate(&schema)
        .map_err(|error| format!("schema meta-validation failed: {error}"))?;
    let validator = jsonschema::validator_for(&schema)
        .map_err(|error| format!("schema compilation failed: {error}"))?;

    let mut findings = Vec::new();
    for error in validator.iter_errors(&corpus) {
        findings.push(format!("schema {}: {error}", error.instance_path()));
    }
    findings.extend(semantic_findings(&corpus));

    let cases = corpus
        .get("cases")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let negative_control_count = cases
        .iter()
        .filter(|case| string_at(case, "case_type") == "negative_control")
        .count();
    let negative_controls_marked_passing = cases
        .iter()
        .filter(|case| {
            string_at(case, "case_type") == "negative_control"
                && string_at(case, "expected_disposition") == "CONSISTENT_UNQUALIFIED"
        })
        .count();
    let case_results = cases
        .iter()
        .map(|case| CaseResult {
            case_id: string_at(case, "case_id").to_owned(),
            case_type: string_at(case, "case_type").to_owned(),
            candidate_claim: string_at(case, "candidate_claim").to_owned(),
            declared_disposition: string_at(case, "expected_disposition").to_owned(),
        })
        .collect();

    Ok(AuditReport {
        report_version: "mus-eng-audit-report/0.1",
        tool_version: env!("CARGO_PKG_VERSION"),
        schema_draft: "https://json-schema.org/draft/2020-12/schema",
        report_status: if findings.is_empty() {
            "CONSISTENT_UNQUALIFIED"
        } else {
            "REJECTED"
        },
        claim_ceiling: "SYNTHETIC_FIXTURE_CONSISTENCY_ONLY",
        schema_sha256: sha256(&schema_bytes),
        corpus_sha256: sha256(&corpus_bytes),
        case_count: cases.len(),
        negative_control_count,
        negative_controls_marked_passing,
        case_results,
        findings,
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
                println!(
                    "mus-eng-audit [--schema PATH] [--corpus PATH]\nValidates the MUS-ENG schema, synthetic corpus, and policy invariants."
                );
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
            eprintln!("mus-eng-audit: {error}");
            return ExitCode::from(2);
        }
    };
    let report = match audit(&schema_path, &corpus_path) {
        Ok(report) => report,
        Err(error) => {
            eprintln!("mus-eng-audit: {error}");
            return ExitCode::from(2);
        }
    };
    let rejected = !report.findings.is_empty();
    println!(
        "{}",
        serde_json::to_string_pretty(&report).unwrap_or_else(|_| {
            json!({"report_status":"REJECTED","findings":["report serialization failed"]})
                .to_string()
        })
    );
    if rejected {
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn corpus() -> Value {
        serde_json::from_str(include_str!(
            "../../../fixtures/instrument-engineering/mus_eng_corpus_v0_1.json"
        ))
        .expect("fixture corpus parses")
    }

    #[test]
    fn checked_in_schema_and_corpus_validate() {
        let schema: Value = serde_json::from_str(include_str!(
            "../../../schemas/instrument-engineering/mus_eng_corpus_v0_1.schema.json"
        ))
        .expect("schema parses");
        let instance = corpus();
        jsonschema::meta::validate(&schema).expect("schema conforms to its draft meta-schema");
        let validator = jsonschema::validator_for(&schema).expect("schema compiles");
        assert!(validator.is_valid(&instance), "corpus should match schema");
        assert!(
            semantic_findings(&instance).is_empty(),
            "{}",
            semantic_findings(&instance).join("\n")
        );
    }

    #[test]
    fn duplicate_case_identity_is_rejected() {
        let mut data = corpus();
        let first = data["cases"][0].clone();
        data["cases"].as_array_mut().unwrap().push(first);
        assert!(
            semantic_findings(&data)
                .iter()
                .any(|finding| finding.contains("duplicate identity"))
        );
    }

    #[test]
    fn evidence_cannot_be_both_available_and_missing() {
        let mut data = corpus();
        data["cases"][0]["missing_evidence"]
            .as_array_mut()
            .unwrap()
            .push(json!("bom"));
        assert!(
            semantic_findings(&data)
                .iter()
                .any(|finding| finding.contains("both available and missing"))
        );
    }

    #[test]
    fn negative_control_cannot_be_marked_consistent() {
        let mut data = corpus();
        data["cases"][2]["expected_disposition"] = json!("CONSISTENT_UNQUALIFIED");
        assert!(
            semantic_findings(&data)
                .iter()
                .any(|finding| finding.contains("negative control is marked as passing"))
        );
    }

    #[test]
    fn fault_requires_fail_closed_disposition_and_ceiling() {
        let mut data = corpus();
        data["cases"][3]["expected_disposition"] = json!("CONSISTENT_UNQUALIFIED");
        data["cases"][3]["evidence_ceiling"] = json!("schema_only");
        let findings = semantic_findings(&data);
        assert!(
            findings
                .iter()
                .any(|finding| finding.contains("faults require disposition"))
        );
        assert!(
            findings
                .iter()
                .any(|finding| finding.contains("requires evidence ceiling"))
        );
    }

    #[test]
    fn undeclared_missing_evidence_is_rejected() {
        let mut data = corpus();
        data["cases"][1]["missing_evidence"] = json!([]);
        data["cases"][1]["available_evidence"] = json!(["requirements", "design_manifest"]);
        assert!(
            semantic_findings(&data)
                .iter()
                .any(|finding| finding.contains("neither available nor declared missing"))
        );
    }

    #[test]
    fn human_claim_requires_protocol_and_results() {
        let mut data = corpus();
        data["cases"][9]["expected_disposition"] = json!("CONSISTENT_UNQUALIFIED");
        assert!(
            semantic_findings(&data)
                .iter()
                .any(|finding| finding.contains("human_study_protocol"))
        );
    }
}
