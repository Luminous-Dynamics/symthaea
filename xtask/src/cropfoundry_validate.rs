use anyhow::{Context, Result, anyhow};
use serde_json::Value;
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

const BENCHMARK_RELATIVE_PATH: &str = "docs/research/cropfoundry/cropfoundry-benchmark-v0.json";
const SOURCE_RELATIVE_PATH: &str = "docs/research/cropfoundry/data-source-manifest-v0.json";

fn string_at<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(Value::as_str)
}

fn nonempty_string(value: &Value, key: &str) -> bool {
    string_at(value, key).is_some_and(|text| !text.trim().is_empty())
}

fn errors_or_ok(errors: Vec<String>) -> std::result::Result<(), Vec<String>> {
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Validate proposal structure and fail-closed status semantics.
/// This does not execute a scientific benchmark or establish crop performance.
fn validate_benchmark_contract(root: &Value) -> std::result::Result<usize, Vec<String>> {
    let mut errors = Vec::new();

    if string_at(root, "suite") != Some("CropFoundry") {
        errors.push("suite must be exactly CropFoundry".to_owned());
    }
    if string_at(root, "status") != Some("proposed_not_run") {
        errors.push(
            "suite status must remain proposed_not_run until scored evidence exists".to_owned(),
        );
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
        "not_evaluable_until_genotype_and_pedigree_data",
        "not_evaluable_until_multi_crop_harmonized_data",
        "not_evaluable_until_interaction_data",
        "not_evaluable_until_imagery_snapshot_and_splits_pinned",
        "not_evaluable_until_experimental_design_documented",
        "not_evaluable_until_time_series_metadata_pinned",
        "not_evaluable_until_paired_stress_sensor_data_pinned",
        "not_evaluable_until_calibration_reference_and_site_metadata_pinned",
        "not_evaluable_until_postharvest_batch_storage_and_quality_data_pinned",
        "not_evaluable_until_complete_multi_nutrient_batch_and_matched_resource_ledger",
        "not_evaluable_until_nutrient_assays_retention_factors_and_diet_requirements_pinned",
        "not_evaluable_until_trial_protocol_variance_and_cluster_metadata_pinned",
        "not_evaluable_until_joint_uncertainty_sets_and_preference_protocol_pinned",
    ];
    let mut ids = BTreeSet::new();

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

        for key in [
            "category",
            "name",
            "question",
            "primary_metric",
            "proposed_target",
            "split_protocol",
        ] {
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

    let Some(invariants) = root.get("invariants").and_then(Value::as_array) else {
        errors.push("invariants must be an array".to_owned());
        return Err(errors);
    };
    if invariants.is_empty() {
        errors.push("invariants must not be empty".to_owned());
    }
    let mut invariant_ids = BTreeSet::new();
    for (index, invariant) in invariants.iter().enumerate() {
        let prefix = format!("invariants[{index}]");
        match string_at(invariant, "id") {
            Some(id) if !id.trim().is_empty() => {
                if !invariant_ids.insert(id.to_owned()) {
                    errors.push(format!("duplicate invariant id {id}"));
                }
            }
            _ => errors.push(format!("{prefix}.id must be a non-empty string")),
        }
        for key in ["rule", "fail_behavior"] {
            if !nonempty_string(invariant, key) {
                errors.push(format!("{prefix}.{key} must be a non-empty string"));
            }
        }
    }

    errors_or_ok(errors).map(|()| benchmarks.len())
}

/// Validate source permission/provenance semantics. A public URL is not blanket
/// authorization to download, redistribute, or train on an artifact.
fn validate_source_manifest(root: &Value) -> std::result::Result<usize, Vec<String>> {
    let mut errors = Vec::new();

    if string_at(root, "manifest") != Some("CropFoundryDataSourceInventory") {
        errors.push("manifest must be exactly CropFoundryDataSourceInventory".to_owned());
    }
    if !nonempty_string(root, "version") {
        errors.push("version must be a non-empty string".to_owned());
    }
    let manifest_status = string_at(root, "status");
    const MANIFEST_STATES: &[&str] = &[
        "discovery_only",
        "source_rights_reviewed",
        "controlled_subset_authorized",
    ];
    if !manifest_status.is_some_and(|status| MANIFEST_STATES.contains(&status)) {
        errors.push("manifest status is missing or unsupported".to_owned());
    }
    let declared_no_training = match root
        .get("policy")
        .and_then(|policy| policy.get("no_source_is_training_authorized"))
        .and_then(Value::as_bool)
    {
        Some(value) => value,
        None => {
            errors.push(
                "policy.no_source_is_training_authorized must be an explicit boolean".to_owned(),
            );
            true
        }
    };

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
    let mut ids = BTreeSet::new();
    let mut any_training_authorized = false;

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

        for key in [
            "title",
            "source_uri",
            "source_type",
            "license_status",
            "access_status",
        ] {
            if !nonempty_string(source, key) {
                errors.push(format!("{prefix}.{key} must be a non-empty string"));
            }
        }

        let ingestion_status = string_at(source, "ingestion_status");
        if !ingestion_status.is_some_and(|status| INGESTION_STATES.contains(&status)) {
            errors.push(format!(
                "{prefix}.ingestion_status is missing or unsupported"
            ));
        }

        let training_authorized = match source.get("training_authorized").and_then(Value::as_bool) {
            Some(value) => value,
            None => {
                errors.push(format!(
                    "{prefix}.training_authorized must be an explicit boolean"
                ));
                false
            }
        };
        let permitted_uses_array = source
            .get("permitted_current_use")
            .and_then(Value::as_array);
        if !permitted_uses_array
            .is_some_and(|items| !items.is_empty() && items.iter().all(Value::is_string))
        {
            errors.push(format!(
                "{prefix}.permitted_current_use must be a non-empty array of strings"
            ));
        }
        let permitted_uses: Vec<&str> = permitted_uses_array
            .map(|items| items.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        let declares_training_use = permitted_uses.contains(&"model_training")
            || permitted_uses.contains(&"training")
            || permitted_uses.contains(&"fine_tuning");
        any_training_authorized |= training_authorized;
        if declared_no_training && training_authorized {
            errors.push(format!(
                "{prefix} is training-authorized while the manifest declares no training authorization"
            ));
        }
        if manifest_status == Some("discovery_only") && training_authorized {
            errors.push(format!(
                "{prefix} cannot be training-authorized while the manifest status is discovery_only"
            ));
        }

        if !training_authorized && declares_training_use {
            errors.push(format!(
                "{prefix} permits training while training_authorized is false"
            ));
        }

        if training_authorized {
            if ingestion_status != Some("ingestion_authorized") {
                errors.push(format!(
                    "{prefix} is training-authorized but ingestion_status is not ingestion_authorized"
                ));
            }
            if string_at(source, "license_status") != Some("verified") {
                errors.push(format!(
                    "{prefix} requires license_status verified for training"
                ));
            }
            if string_at(source, "access_status") != Some("authorized") {
                errors.push(format!(
                    "{prefix} requires access_status authorized for training"
                ));
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
            if source
                .get("dataset_structure_reviewed")
                .and_then(Value::as_bool)
                != Some(true)
            {
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

    if declared_no_training == any_training_authorized {
        errors.push(
            "policy.no_source_is_training_authorized must equal whether all source records are non-trainable"
                .to_owned(),
        );
    }
    if any_training_authorized && manifest_status != Some("controlled_subset_authorized") {
        errors.push(
            "training-authorized sources require manifest status controlled_subset_authorized"
                .to_owned(),
        );
    }
    if manifest_status == Some("controlled_subset_authorized") && !any_training_authorized {
        errors.push(
            "controlled_subset_authorized requires at least one source with explicit authorization"
                .to_owned(),
        );
    }

    errors_or_ok(errors).map(|()| sources.len())
}

fn read_json(path: &Path, description: &str) -> Result<Value> {
    let input = fs::read_to_string(path)
        .with_context(|| format!("cannot read {description}: {}", path.display()))?;
    serde_json::from_str(&input)
        .with_context(|| format!("cannot parse {description} as JSON: {}", path.display()))
}

fn render_errors(description: &str, errors: Vec<String>) -> anyhow::Error {
    anyhow!("{description} rejected:\n - {}", errors.join("\n - "))
}

/// Validate the two proposal files in the selected workspace root.
pub fn run(root: &Path) -> Result<()> {
    let benchmark_path = root.join(BENCHMARK_RELATIVE_PATH);
    let source_path = root.join(SOURCE_RELATIVE_PATH);
    let benchmarks = read_json(&benchmark_path, "CropFoundry benchmark contract")?;
    let sources = read_json(&source_path, "CropFoundry data-source manifest")?;

    let benchmark_count = validate_benchmark_contract(&benchmarks)
        .map_err(|errors| render_errors("benchmark contract", errors))?;
    let source_count = validate_source_manifest(&sources)
        .map_err(|errors| render_errors("data-source manifest", errors))?;

    println!("PASS benchmark contract: {benchmark_count} entries; proposal semantics validated");
    println!(
        "PASS data-source manifest: {source_count} entries; training authorization remains fail-closed"
    );
    println!("PASS CropFoundry proposal contract; no scientific results were evaluated");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    const BENCHMARK_FIXTURE: &str =
        include_str!("../../docs/research/cropfoundry/cropfoundry-benchmark-v0.json");
    const SOURCE_FIXTURE: &str =
        include_str!("../../docs/research/cropfoundry/data-source-manifest-v0.json");

    #[test]
    fn committed_benchmark_contract_is_well_formed_and_non_passing() {
        let fixture: Value = serde_json::from_str(BENCHMARK_FIXTURE).unwrap();
        let count = validate_benchmark_contract(&fixture).unwrap();
        assert_eq!(count, 27);
    }

    #[test]
    fn duplicate_benchmark_ids_fail_closed() {
        let mut fixture: Value = serde_json::from_str(BENCHMARK_FIXTURE).unwrap();
        let first_id = fixture["benchmarks"][0]["id"].clone();
        fixture["benchmarks"][1]["id"] = first_id;
        let errors = validate_benchmark_contract(&fixture).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.contains("duplicate benchmark id"))
        );
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
    fn duplicate_invariant_ids_fail_closed() {
        let mut fixture: Value = serde_json::from_str(BENCHMARK_FIXTURE).unwrap();
        let first_id = fixture["invariants"][0]["id"].clone();
        fixture["invariants"][1]["id"] = first_id;
        let errors = validate_benchmark_contract(&fixture).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.contains("duplicate invariant id"))
        );
    }

    #[test]
    fn current_source_manifest_keeps_every_source_non_trainable() {
        let fixture: Value = serde_json::from_str(SOURCE_FIXTURE).unwrap();
        let count = validate_source_manifest(&fixture).unwrap();
        assert_eq!(count, 16);
        assert!(
            fixture["sources"]
                .as_array()
                .unwrap()
                .iter()
                .all(|source| source["training_authorized"] == false)
        );
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
        assert!(
            errors
                .iter()
                .any(|error| error.contains("license_status verified"))
        );
    }

    #[test]
    fn training_permission_is_rejected_when_authorization_flag_is_false() {
        let mut fixture: Value = serde_json::from_str(SOURCE_FIXTURE).unwrap();
        fixture["sources"][0]["permitted_current_use"] =
            json!(["public_metadata_review", "training"]);
        let errors = validate_source_manifest(&fixture).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.contains("permits training"))
        );
    }

    #[test]
    fn missing_training_authorization_flag_fails_closed() {
        let mut fixture: Value = serde_json::from_str(SOURCE_FIXTURE).unwrap();
        fixture["sources"][0]
            .as_object_mut()
            .unwrap()
            .remove("training_authorized");
        let errors = validate_source_manifest(&fixture).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.contains("training_authorized must be an explicit boolean"))
        );
    }

    #[test]
    fn discovery_only_manifest_cannot_authorize_training() {
        let mut fixture: Value = serde_json::from_str(SOURCE_FIXTURE).unwrap();
        fixture["sources"][0]["training_authorized"] = json!(true);
        let errors = validate_source_manifest(&fixture).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.contains("manifest status is discovery_only"))
        );
    }

    #[test]
    fn manifest_global_training_flag_must_match_source_records() {
        let mut fixture: Value = serde_json::from_str(SOURCE_FIXTURE).unwrap();
        fixture["policy"]["no_source_is_training_authorized"] = json!(false);
        let errors = validate_source_manifest(&fixture).unwrap_err();
        assert!(errors.iter().any(|error| {
            error.contains("must equal whether all source records are non-trainable")
        }));
    }

    #[test]
    fn fully_reviewed_subset_can_be_authorized_without_weakening_gates() {
        let mut fixture: Value = serde_json::from_str(SOURCE_FIXTURE).unwrap();
        fixture["status"] = json!("controlled_subset_authorized");
        fixture["policy"]["no_source_is_training_authorized"] = json!(false);

        let source = &mut fixture["sources"][0];
        source["training_authorized"] = json!(true);
        source["permitted_current_use"] = json!(["model_training"]);
        source["ingestion_status"] = json!("ingestion_authorized");
        source["license_status"] = json!("verified");
        source["access_status"] = json!("authorized");
        source["retrieval_timestamp"] = json!("2026-10-18T00:00:00Z");
        source["license_evidence_uri"] = json!("https://example.invalid/license");
        source["access_terms_evidence_uri"] = json!("https://example.invalid/terms");
        source["intended_use"] = json!("benchmark training");
        source["dataset_structure_reviewed"] = json!(true);
        source["content_sha256"] =
            json!("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");

        assert_eq!(validate_source_manifest(&fixture).unwrap(), 16);
    }

    #[test]
    fn ingestion_for_schema_review_does_not_imply_training_authority() {
        let mut fixture: Value = serde_json::from_str(SOURCE_FIXTURE).unwrap();
        fixture["sources"][0]["ingestion_status"] = json!("ingestion_authorized");
        fixture["sources"][0]["permitted_current_use"] =
            json!(["public_metadata_review", "schema_audit"]);
        fixture["sources"][0]["training_authorized"] = json!(false);
        assert_eq!(validate_source_manifest(&fixture).unwrap(), 16);
    }

    #[test]
    fn duplicate_source_ids_fail_closed() {
        let mut fixture: Value = serde_json::from_str(SOURCE_FIXTURE).unwrap();
        let first_id = fixture["sources"][0]["id"].clone();
        fixture["sources"][1]["id"] = first_id;
        let errors = validate_source_manifest(&fixture).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.contains("duplicate source id"))
        );
    }
}
