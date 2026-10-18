use serde_json::Value;
use std::env;
use std::fs;
use std::process::ExitCode;

const DEFAULT_BENCHMARKS: &str = "docs/research/cropfoundry/cropfoundry-benchmark-v0.json";
const DEFAULT_SOURCES: &str = "docs/research/cropfoundry/data-source-manifest-v0.json";

fn string_at<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(Value::as_str)
}

fn nonempty_string(value: &Value, key: &str) -> bool {
    string_at(value, key).is_some_and(|text| !text.trim().is_empty())
}

fn errors_or_ok(errors: Vec<String>) -> Result<(), Vec<String>> {
    if errors.is_empty() { Ok(()) } else { Err(errors) }
}

/// Validate proposal structure and fail-closed status semantics.
/// This does not execute a scientific benchmark or establish crop performance.
pub fn validate_benchmark_contract(root: &Value) -> Result<usize, Vec<String>> {
    let mut errors = Vec::new();

    if string_at(root, "suite") != Some("CropFoundry") {
        errors.push("suite must be exactly CropFoundry".to_owned());
    }
    if string_at(root, "status") != Some("proposed_not_run") {
        errors.push("suite status must remain proposed_not_run until scored evidence exists".to_owned());
    }
    if !nonempty_string(root, "suite_version") {
        errors.push("suite_version must be a non-empty string".to_owned());
    }

    let Some(benchmarks) = root.get("benchmarks").and_then(Value::as_array) else {
        errors.push("benchmarks must be an array".to_owned());
        return Err(errors);
    };
    if benchmarks.is_empty() {
        errors.push("benchmarks must not be empty".to_owned());
    }

    let allowed_statuses = [
        "proposed_not_run",
        "not_evaluable_until_physical_trials",
        "not_evaluable_until_local_cost_and_partner_data",
    ];
    let mut ids = std::collections::BTreeSet::new();

    for (index, benchmark) in benchmarks.iter().enumerate() {
        let prefix = format!("benchmarks[{index}]");
        let Some(id) = string_at(benchmark, "id") else {
            errors.push(format!("{prefix}.id must be a string"));
            continue;
        };
        if id.trim().is_empty() {
            errors.push(format!("{prefix}.id must not be empty"));
        }
        if !ids.insert(id.to_owned()) {
            errors.push(format!("duplicate benchmark id {id}"));
        }

        for key in ["category", "name", "question", "primary_metric", "proposed_target", "split_protocol"] {
            if !nonempty_string(benchmark, key) {
                errors.push(format!("{prefix}.{key} must be a non-empty string"));
            }
        }

        match string_at(benchmark, "status") {
            Some(status) if allowed_statuses.contains(&status) => {}
            Some(status) => errors.push(format!(
                "{prefix}.status {status} is not an allowed non-passing proposal status"
            )),
            None => errors.push(format!("{prefix}.status must be a string")),
        }
    }

    errors_or_ok(errors).map(|()| benchmarks.len())
}

/// Validate source permission/provenance semantics. A public URL is not blanket
/// authorization to download, redistribute, or train on an artifact.
pub fn validate_source_manifest(root: &Value) -> Result<usize, Vec<String>> {
    let mut errors = Vec::new();

    if string_at(root, "manifest") != Some("CropFoundryDataSourceInventory") {
        errors.push("manifest must be exactly CropFoundryDataSourceInventory".to_owned());
    }
    if !nonempty_string(root, "version") {
        errors.push("version must be a non-empty string".to_owned());
    }
    if string_at(root, "status") != Some("discovery_only") {
        errors.push("manifest status must remain discovery_only until acquisition is reviewed".to_owned());
    }

    let Some(sources) = root.get("sources").and_then(Value::as_array) else {
        errors.push("sources must be an array".to_owned());
        return Err(errors);
    };
    if sources.is_empty() {
        errors.push("sources must not be empty".to_owned());
    }

    const INGESTION_STATES: &[&str] = &[
        "quarantined",
        "context_only",
        "discovery_only",
        "integration_only",
        "ingestion_authorized",
    ];
    let mut ids = std::collections::BTreeSet::new();

    for (index, source) in sources.iter().enumerate() {
        let prefix = format!("sources[{index}]");
        let Some(id) = string_at(source, "id") else {
            errors.push(format!("{prefix}.id must be a string"));
            continue;
        };
        if id.trim().is_empty() {
            errors.push(format!("{prefix}.id must not be empty"));
        }
        if !ids.insert(id.to_owned()) {
            errors.push(format!("duplicate source id {id}"));
        }

        for key in ["title", "source_uri", "source_type", "license_status", "access_status"] {
            if !nonempty_string(source, key) {
                errors.push(format!("{prefix}.{key} must be a non-empty string"));
            }
        }

        let ingestion_status = string_at(source, "ingestion_status");
        if !ingestion_status.is_some_and(|status| INGESTION_STATES.contains(&status)) {
            errors.push(format!("{prefix}.ingestion_status is missing or unsupported"));
        }

        let training_authorized = source
            .get("training_authorized")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let permitted_uses: Vec<&str> = source
            .get("permitted_current_use")
            .and_then(Value::as_array)
            .map(|items| items.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        let declares_training_use = permitted_uses.contains(&"model_training")
            || permitted_uses.contains(&"training")
            || permitted_uses.contains(&"fine_tuning");

        if !training_authorized {
            if ingestion_status == Some("ingestion_authorized") {
                errors.push(format!(
                    "{prefix} claims ingestion_authorized while training_authorized is false"
                ));
            }
            if declares_training_use {
                errors.push(format!(
                    "{prefix} permits training while training_authorized is false"
                ));
            }
        }

        if training_authorized {
            if ingestion_status != Some("ingestion_authorized") {
                errors.push(format!(
                    "{prefix} is training-authorized but ingestion_status is not ingestion_authorized"
                ));
            }
            if string_at(source, "license_status") != Some("verified") {
                errors.push(format!("{prefix} requires license_status verified for training"));
            }
            if string_at(source, "access_status") != Some("authorized") {
                errors.push(format!("{prefix} requires access_status authorized for training"));
            }
            if !declares_training_use {
                errors.push(format!(
                    "{prefix} requires explicit training use in permitted_current_use"
                ));
            }
            for key in [
                "retrieval_timestamp",
                "license_evidence_uri",
                "access_terms_evidence_uri",
                "intended_use",
            ] {
                if !nonempty_string(source, key) {
                    errors.push(format!(
                        "{prefix}.{key} is required before training can be authorized"
                    ));
                }
            }
            if source.get("dataset_structure_reviewed").and_then(Value::as_bool) != Some(true) {
                errors.push(format!(
                    "{prefix}.dataset_structure_reviewed must be true before training can be authorized"
                ));
            }
            match string_at(source, "content_sha256") {
                Some(digest)
                    if digest.len() == 64
                        && digest.bytes().all(|byte| byte.is_ascii_hexdigit()) => {}
                _ => errors.push(format!(
                    "{prefix}.content_sha256 must be a 64-character hexadecimal SHA-256 before training can be authorized"
                )),
            }
        }
    }

    errors_or_ok(errors).map(|()| sources.len())
}

fn validate_file(path: &str, kind: &str) -> Result<Value, String> {
    let input = fs::read_to_string(path)
        .map_err(|error| format!("cannot read {kind} {path}: {error}"))?;
    serde_json::from_str(&input)
        .map_err(|error| format!("cannot parse {kind} {path} as JSON: {error}"))
}

fn print_errors(kind: &str, result: Result<usize, Vec<String>>) -> bool {
    match result {
        Ok(count) => {
            println!("PASS {kind}: {count} entries; contract semantics validated");
            true
        }
        Err(errors) => {
            eprintln!("FAIL {kind}:");
            for error in errors {
                eprintln!("  - {error}");
            }
            false
        }
    }
}

fn main() -> ExitCode {
    let mut benchmarks_path = DEFAULT_BENCHMARKS.to_owned();
    let mut sources_path = DEFAULT_SOURCES.to_owned();
    let mut arguments = env::args().skip(1);

    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--benchmarks" => match arguments.next() {
                Some(path) => benchmarks_path = path,
                None => {
                    eprintln!("--benchmarks requires a file path");
                    return ExitCode::from(2);
                }
            },
            "--sources" => match arguments.next() {
                Some(path) => sources_path = path,
                None => {
                    eprintln!("--sources requires a file path");
                    return ExitCode::from(2);
                }
            },
            "--help" | "-h" => {
                println!(
                    "Usage: cropfoundry-validator [--benchmarks PATH] [--sources PATH]\n\
                     Validates proposal semantics only; it does not run crop models or certify data rights."
                );
                return ExitCode::SUCCESS;
            }
            other => {
                eprintln!("unknown argument {other}; use --help");
                return ExitCode::from(2);
            }
        }
    }

    let benchmarks = match validate_file(&benchmarks_path, "benchmark contract") {
        Ok(value) => value,
        Err(error) => {
            eprintln!("FAIL {error}");
            return ExitCode::FAILURE;
        }
    };
    let sources = match validate_file(&sources_path, "data-source manifest") {
        Ok(value) => value,
        Err(error) => {
            eprintln!("FAIL {error}");
            return ExitCode::FAILURE;
        }
    };

    let benchmarks_ok = print_errors(
        "benchmark contract",
        validate_benchmark_contract(&benchmarks),
    );
    let sources_ok = print_errors(
        "data-source manifest",
        validate_source_manifest(&sources),
    );

    if benchmarks_ok && sources_ok {
        println!("PASS CropFoundry proposal contract; no scientific results were evaluated");
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const BENCHMARK_FIXTURE: &str =
        include_str!("../../../../docs/research/cropfoundry/cropfoundry-benchmark-v0.json");
    const SOURCE_FIXTURE: &str =
        include_str!("../../../../docs/research/cropfoundry/data-source-manifest-v0.json");

    #[test]
    fn committed_benchmark_contract_is_well_formed_and_non_passing() {
        let fixture: Value = serde_json::from_str(BENCHMARK_FIXTURE).unwrap();
        let count = validate_benchmark_contract(&fixture).unwrap();
        assert_eq!(count, 12);
    }

    #[test]
    fn duplicate_benchmark_ids_fail_closed() {
        let mut fixture: Value = serde_json::from_str(BENCHMARK_FIXTURE).unwrap();
        let first_id = fixture["benchmarks"][0]["id"].clone();
        fixture["benchmarks"][1]["id"] = first_id;
        let errors = validate_benchmark_contract(&fixture).unwrap_err();
        assert!(errors.iter().any(|error| error.contains("duplicate benchmark id")));
    }

    #[test]
    fn achieved_status_cannot_be_smuggled_into_proposal() {
        let mut fixture: Value = serde_json::from_str(BENCHMARK_FIXTURE).unwrap();
        fixture["benchmarks"][0]["status"] = json!("passed");
        assert!(validate_benchmark_contract(&fixture).is_err());

        fixture["benchmarks"][0]["status"] = json!("proposed_not_run");
        fixture["status"] = json!("passed");
        assert!(validate_benchmark_contract(&fixture).is_err());
    }

    #[test]
    fn missing_target_fails_closed() {
        let mut fixture: Value = serde_json::from_str(BENCHMARK_FIXTURE).unwrap();
        fixture["benchmarks"][0]["proposed_target"] = Value::Null;
        let errors = validate_benchmark_contract(&fixture).unwrap_err();
        assert!(errors.iter().any(|error| error.contains("proposed_target")));
    }

    #[test]
    fn current_source_manifest_keeps_every_source_non_trainable() {
        let fixture: Value = serde_json::from_str(SOURCE_FIXTURE).unwrap();
        let count = validate_source_manifest(&fixture).unwrap();
        assert_eq!(count, 9);
        assert!(fixture["sources"]
            .as_array()
            .unwrap()
            .iter()
            .all(|source| source["training_authorized"] == false));
    }

    #[test]
    fn unresolved_license_cannot_be_training_authorized() {
        let mut fixture: Value = serde_json::from_str(SOURCE_FIXTURE).unwrap();
        let source = &mut fixture["sources"][0];
        source["training_authorized"] = json!(true);
        source["permitted_current_use"] = json!(["model_training"]);
        source["ingestion_status"] = json!("ingestion_authorized");
        source["access_status"] = json!("authorized");
        source["retrieval_timestamp"] = json!("2026-10-18T00:00:00Z");
        source["license_evidence_uri"] = json!("https://example.invalid/license");
        source["access_terms_evidence_uri"] = json!("https://example.invalid/terms");
        source["intended_use"] = json!("benchmark training");
        source["dataset_structure_reviewed"] = json!(true);
        source["content_sha256"] =
            json!("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");

        let errors = validate_source_manifest(&fixture).unwrap_err();
        assert!(errors.iter().any(|error| error.contains("license_status verified")));
    }

    #[test]
    fn training_permission_is_rejected_when_authorization_flag_is_false() {
        let mut fixture: Value = serde_json::from_str(SOURCE_FIXTURE).unwrap();
        fixture["sources"][0]["permitted_current_use"] = json!(["public_metadata_review", "training"]);
        let errors = validate_source_manifest(&fixture).unwrap_err();
        assert!(errors.iter().any(|error| error.contains("permits training")));
    }

    #[test]
    fn duplicate_source_ids_fail_closed() {
        let mut fixture: Value = serde_json::from_str(SOURCE_FIXTURE).unwrap();
        let first_id = fixture["sources"][0]["id"].clone();
        fixture["sources"][1]["id"] = first_id;
        let errors = validate_source_manifest(&fixture).unwrap_err();
        assert!(errors.iter().any(|error| error.contains("duplicate source id")));
    }
}
