// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Synthetic-only, fail-closed validator for grouped plasma-ML evaluation
//! splits. This checks provenance and leakage boundaries; it does not train
//! a model, reproduce literature metrics, or predict a physical outcome.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

const EXPECTED_EVIDENCE_CLASS: &str = "SYNTHETIC";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Disposition {
    Pass,
    Blocked,
    Rejected,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Fixture {
    schema_version: String,
    benchmark_id: String,
    evidence_class: String,
    synthetic_only: bool,
    published_model_reproduction: bool,
    physical_measurements: bool,
    claim_ceiling: String,
    group_dimensions: Vec<String>,
    split_policy: String,
    augmentation_policy: String,
    preprocessing_policy: String,
    scenarios: Vec<Scenario>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Scenario {
    scenario_id: String,
    holdout_dimension: String,
    observations: Vec<Observation>,
    assignments: Vec<Assignment>,
    augmentations: Vec<Augmentation>,
    preprocessing_fit_observation_ids: Vec<String>,
    expected_disposition: Disposition,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Observation {
    observation_id: String,
    source_paper_id: Option<String>,
    lab_id: Option<String>,
    device_id: Option<String>,
    material_family: Option<String>,
    material_grade: Option<String>,
    specimen_id: Option<String>,
    raw_observation_digest: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Assignment {
    observation_id: String,
    fold: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Augmentation {
    generated_row_id: String,
    parent_observation_ids: Vec<String>,
    assigned_fold: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ScenarioResult {
    pub scenario_id: String,
    pub disposition: Disposition,
    pub expected_disposition: Disposition,
    pub reasons: Vec<String>,
    pub empirical_observation_count: usize,
    pub training_observation_count: usize,
    pub test_observation_count: usize,
    pub training_augmentation_count: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct AuditReport {
    pub schema_version: String,
    pub benchmark_id: String,
    pub evidence_class: String,
    pub synthetic_only: bool,
    pub published_model_reproduction: bool,
    pub physical_measurements: bool,
    pub scenario_count: usize,
    pub matched_expectations: usize,
    pub mismatched_expectations: usize,
    pub results: Vec<ScenarioResult>,
    pub claim_ceiling: String,
}

pub fn run_audit(fixture_path: &Path) -> Result<AuditReport> {
    let bytes = fs::read(fixture_path)
        .with_context(|| format!("read plasma ML fixture {}", fixture_path.display()))?;
    let report = audit_fixture_json(
        std::str::from_utf8(&bytes).context("plasma ML fixture is not UTF-8")?,
    )
    .map_err(anyhow::Error::msg)?;
    Ok(report)
}

pub fn audit_fixture_json(input: &str) -> std::result::Result<AuditReport, String> {
    let fixture: Fixture =
        serde_json::from_str(input).map_err(|error| format!("invalid ML fixture JSON: {error}"))?;

    if fixture.schema_version != "0.1.0" {
        return Err("unsupported plasma ML fixture schema_version".to_owned());
    }
    if fixture.benchmark_id != "PLASMA-ML-GENERALIZATION-001" {
        return Err("unexpected plasma ML benchmark_id".to_owned());
    }
    if fixture.evidence_class != EXPECTED_EVIDENCE_CLASS
        || !fixture.synthetic_only
        || fixture.published_model_reproduction
        || fixture.physical_measurements
    {
        return Err(
            "fixture must remain synthetic-only with no reproduction or physical measurement claim"
                .to_owned(),
        );
    }
    if fixture.claim_ceiling.trim().is_empty() {
        return Err("empty ML fixture claim ceiling".to_owned());
    }
    if fixture.split_policy != "GROUP_DISJOINT"
        || fixture.augmentation_policy != "TRAIN_FOLD_ONLY"
        || fixture.preprocessing_policy != "FIT_TRAIN_ONLY"
    {
        return Err("unsupported split/augmentation/preprocessing policy".to_owned());
    }
    let declared_dimensions: BTreeSet<_> =
        fixture.group_dimensions.iter().map(String::as_str).collect();
    let required_dimensions = [
        "source_paper_id",
        "lab_id",
        "device_id",
        "material_family",
    ];
    for required in required_dimensions {
        if !declared_dimensions.contains(required) {
            return Err(format!("missing required group dimension {required}"));
        }
    }
    if fixture.scenarios.is_empty() {
        return Err("fixture contains no split scenarios".to_owned());
    }

    let mut scenario_ids = BTreeSet::new();
    for scenario in &fixture.scenarios {
        if scenario.scenario_id.trim().is_empty()
            || !scenario_ids.insert(scenario.scenario_id.as_str())
        {
            return Err("empty or duplicate scenario ID".to_owned());
        }
    }

    let results: Vec<_> = fixture
        .scenarios
        .iter()
        .map(evaluate_scenario)
        .collect();
    let matched_expectations = results
        .iter()
        .filter(|result| result.disposition == result.expected_disposition)
        .count();
    let scenario_count = results.len();

    Ok(AuditReport {
        schema_version: fixture.schema_version,
        benchmark_id: fixture.benchmark_id,
        evidence_class: fixture.evidence_class,
        synthetic_only: fixture.synthetic_only,
        published_model_reproduction: fixture.published_model_reproduction,
        physical_measurements: fixture.physical_measurements,
        scenario_count,
        matched_expectations,
        mismatched_expectations: scenario_count - matched_expectations,
        results,
        claim_ceiling: fixture.claim_ceiling,
    })
}

fn group_value<'a>(observation: &'a Observation, dimension: &str) -> Option<&'a str> {
    match dimension {
        "source_paper_id" => observation.source_paper_id.as_deref(),
        "lab_id" => observation.lab_id.as_deref(),
        "device_id" => observation.device_id.as_deref(),
        "material_family" => observation.material_family.as_deref(),
        _ => None,
    }
}

fn evaluate_scenario(scenario: &Scenario) -> ScenarioResult {
    let mut rejected = Vec::<String>::new();
    let mut blocked = Vec::<String>::new();
    let mut training_count = 0usize;
    let mut test_count = 0usize;
    let mut augmentation_count = 0usize;

    let mut observation_ids = BTreeSet::<String>::new();
    for observation in &scenario.observations {
        if observation.observation_id.trim().is_empty()
            || !observation_ids.insert(observation.observation_id.clone())
        {
            rejected.push("observation identity is empty or duplicated".to_owned());
        }
    }
    if scenario.observations.is_empty() {
        blocked.push("scenario has no empirical observations".to_owned());
    }

    let required_dimension_present = matches!(
        scenario.holdout_dimension.as_str(),
        "source_paper_id" | "lab_id" | "device_id" | "material_family"
    );
    if !required_dimension_present {
        rejected.push("holdout dimension is unsupported".to_owned());
    }

    let mut assignments = BTreeMap::<String, String>::new();
    for assignment in &scenario.assignments {
        if !observation_ids.contains(&assignment.observation_id) {
            rejected.push(format!(
                "split assignment references unknown observation {}",
                assignment.observation_id
            ));
        }
        if !matches!(assignment.fold.as_str(), "TRAIN" | "TEST") {
            rejected.push(format!(
                "observation {} has unsupported fold {}",
                assignment.observation_id, assignment.fold
            ));
        }
        if assignments
            .insert(assignment.observation_id.clone(), assignment.fold.clone())
            .is_some()
        {
            rejected.push(format!(
                "duplicate split assignment for {}",
                assignment.observation_id
            ));
        }
    }
    for observation_id in &observation_ids {
        if !assignments.contains_key(observation_id) {
            blocked.push(format!(
                "observation {observation_id} has no split assignment"
            ));
        }
    }

    for observation in &scenario.observations {
        let Some(fold) = assignments.get(&observation.observation_id) else {
            continue;
        };
        match fold.as_str() {
            "TRAIN" => training_count += 1,
            "TEST" => test_count += 1,
            _ => {}
        }
        if required_dimension_present {
            match group_value(observation, &scenario.holdout_dimension) {
                Some(value) if !value.trim().is_empty() => {}
                _ => blocked.push(format!(
                    "observation {} lacks required holdout identity {}",
                    observation.observation_id, scenario.holdout_dimension
                )),
            }
        }
    }
    if training_count == 0 || test_count == 0 {
        blocked.push("split must contain at least one TRAIN and one TEST observation".to_owned());
    }

    // Grouped holdout: an entire source paper, lab, device or material family
    // must reside in exactly one partition.
    let mut group_folds = BTreeMap::<String, String>::new();
    for observation in &scenario.observations {
        let Some(fold) = assignments.get(&observation.observation_id) else {
            continue;
        };
        if let Some(group) = group_value(observation, &scenario.holdout_dimension) {
            if !group.trim().is_empty() {
                if let Some(previous_fold) = group_folds.insert(group.to_owned(), fold.clone()) {
                    if previous_fold != fold.as_str() {
                        rejected.push(format!(
                            "group {group} crosses TRAIN/TEST for {}",
                            scenario.holdout_dimension
                        ));
                    }
                }
            }
        }
    }

    // Specimen identity and raw-row digest may not be repeated across folds.
    let mut specimen_folds = BTreeMap::<String, String>::new();
    let mut digest_folds = BTreeMap::<String, String>::new();
    for observation in &scenario.observations {
        let Some(fold) = assignments.get(&observation.observation_id) else {
            continue;
        };
        match observation.specimen_id.as_deref() {
            Some(specimen) if !specimen.trim().is_empty() => {
                if let Some(previous_fold) = specimen_folds.insert(specimen.to_owned(), fold.clone()) {
                    if previous_fold != *fold {
                        rejected.push(format!(
                            "specimen identity {specimen} crosses TRAIN/TEST"
                        ));
                    }
                }
            }
            _ => blocked.push(format!(
                "observation {} lacks specimen identity",
                observation.observation_id
            )),
        }
        match observation.raw_observation_digest.as_deref() {
            Some(digest) if !digest.trim().is_empty() => {
                if let Some(previous_fold) = digest_folds.insert(digest.to_owned(), fold.clone()) {
                    if previous_fold != *fold {
                        rejected.push(format!(
                            "raw observation digest {digest} crosses TRAIN/TEST"
                        ));
                    }
                }
            }
            _ => blocked.push(format!(
                "observation {} lacks raw observation digest",
                observation.observation_id
            )),
        }
        if observation.material_grade.as_deref().is_none_or(str::is_empty) {
            blocked.push(format!(
                "observation {} lacks material-grade identity",
                observation.observation_id
            ));
        }
    }

    // Augmentation can only be generated from empirical TRAIN rows and used
    // inside that same training fold. A held-out parent is a leakage path.
    let mut generated_ids = BTreeSet::<String>::new();
    for augmentation in &scenario.augmentations {
        if augmentation.generated_row_id.trim().is_empty()
            || observation_ids.contains(&augmentation.generated_row_id)
            || !generated_ids.insert(augmentation.generated_row_id.clone())
        {
            rejected.push("generated augmentation identity is empty, duplicated or collides with raw row".to_owned());
        }
        if augmentation.assigned_fold != "TRAIN" {
            rejected.push(format!(
                "synthetic row {} is assigned outside TRAIN",
                augmentation.generated_row_id
            ));
        }
        if augmentation.parent_observation_ids.is_empty() {
            blocked.push(format!(
                "synthetic row {} has no parent observations",
                augmentation.generated_row_id
            ));
        }
        for parent_id in &augmentation.parent_observation_ids {
            match assignments.get(parent_id) {
                Some(fold) if fold == "TRAIN" => {}
                Some(_) => rejected.push(format!(
                    "synthetic row {} uses held-out parent {}",
                    augmentation.generated_row_id, parent_id
                )),
                None => rejected.push(format!(
                    "synthetic row {} references missing parent {}",
                    augmentation.generated_row_id, parent_id
                )),
            }
        }
        if augmentation.assigned_fold == "TRAIN" {
            augmentation_count += 1;
        }
    }

    // Fitted preprocessing state must be derived exclusively from raw TRAIN
    // observations. This includes scaling, imputation, feature selection and
    // any data-derived transform.
    let mut fit_ids = BTreeSet::<String>::new();
    if scenario.preprocessing_fit_observation_ids.is_empty() {
        blocked.push("preprocessing fit lineage is empty".to_owned());
    }
    for fit_id in &scenario.preprocessing_fit_observation_ids {
        if !fit_ids.insert(fit_id.clone()) {
            rejected.push(format!("duplicate preprocessing fit row {fit_id}"));
        }
        match assignments.get(fit_id) {
            Some(fold) if fold == "TRAIN" => {}
            Some(_) => rejected.push(format!(
                "preprocessing fit includes held-out observation {fit_id}"
            )),
            None => rejected.push(format!(
                "preprocessing fit references unknown observation {fit_id}"
            )),
        }
    }
    let expected_training_ids: BTreeSet<_> = assignments
        .iter()
        .filter(|(_, fold)| fold.as_str() == "TRAIN")
        .map(|(observation_id, _)| observation_id.clone())
        .collect();
    if fit_ids != expected_training_ids {
        blocked.push(
            "preprocessing fit lineage does not exactly cover the declared TRAIN population"
                .to_owned(),
        );
    }

    let (disposition, reasons) = if !rejected.is_empty() {
        (Disposition::Rejected, rejected)
    } else if !blocked.is_empty() {
        (Disposition::Blocked, blocked)
    } else {
        (Disposition::Pass, Vec::new())
    };

    ScenarioResult {
        scenario_id: scenario.scenario_id.clone(),
        disposition,
        expected_disposition: scenario.expected_disposition,
        reasons,
        empirical_observation_count: scenario.observations.len(),
        training_observation_count: training_count,
        test_observation_count: test_count,
        training_augmentation_count: augmentation_count,
    }
}

pub fn print_report(report: &AuditReport, as_json: bool) -> Result<()> {
    if as_json {
        println!("{}", serde_json::to_string_pretty(report)?);
        return Ok(());
    }
    println!("Benchmark: {}", report.benchmark_id);
    println!("Evidence class: {}", report.evidence_class);
    println!("Claim ceiling: {}", report.claim_ceiling);
    for result in &report.results {
        println!(
            "  {:?} {} train={} test={} augmentations={} expected={:?}",
            result.disposition,
            result.scenario_id,
            result.training_observation_count,
            result.test_observation_count,
            result.training_augmentation_count,
            result.expected_disposition
        );
        for reason in &result.reasons {
            println!("    - {reason}");
        }
    }
    println!(
        "Expected dispositions: {}/{} matched",
        report.matched_expectations, report.scenario_count
    );
    if report.mismatched_expectations > 0 {
        anyhow::bail!("plasma ML fixture has {} disposition mismatches", report.mismatched_expectations);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str =
        include_str!("../../data/research/plasma_ml_generalization_fixture_v0_1.json");

    fn fixture_value() -> serde_json::Value {
        serde_json::from_str(FIXTURE).expect("synthetic plasma ML fixture parses")
    }

    fn run(value: &serde_json::Value) -> AuditReport {
        audit_fixture_json(&value.to_string()).expect("mutated fixture remains valid JSON")
    }

    fn scenario_mut<'a>(
        value: &'a mut serde_json::Value,
        scenario_id: &str,
    ) -> &'a mut serde_json::Value {
        value["scenarios"]
            .as_array_mut()
            .expect("scenarios array")
            .iter_mut()
            .find(|scenario| scenario["scenario_id"].as_str() == Some(scenario_id))
            .expect("scenario exists")
    }

    fn result_for<'a>(report: &'a AuditReport, scenario_id: &str) -> &'a ScenarioResult {
        report.results.iter().find(|result| result.scenario_id == scenario_id).expect("result exists")
    }

    #[test]
    fn frozen_fixture_matches_all_expected_dispositions() {
        let report = audit_fixture_json(FIXTURE).expect("fixture audit");
        assert_eq!(report.scenario_count, 12);
        assert_eq!(report.mismatched_expectations, 0);
        assert_eq!(report.matched_expectations, 12);
        assert!(report.synthetic_only);
        assert!(!report.published_model_reproduction);
        assert!(!report.physical_measurements);
    }

    #[test]
    fn source_lab_device_and_material_grouped_holdouts_pass() {
        let report = audit_fixture_json(FIXTURE).expect("fixture audit");
        for id in [
            "source-heldout-clean",
            "device-heldout-clean",
            "lab-heldout-clean",
            "material-family-heldout-clean",
        ] {
            assert_eq!(result_for(&report, id).disposition, Disposition::Pass);
        }
    }

    #[test]
    fn augmentation_with_test_parent_is_rejected() {
        let report = audit_fixture_json(FIXTURE).expect("fixture audit");
        assert_eq!(
            result_for(&report, "augmentation-before-split-leakage").disposition,
            Disposition::Rejected
        );
    }

    #[test]
    fn preprocessing_fit_that_includes_test_row_is_rejected() {
        let report = audit_fixture_json(FIXTURE).expect("fixture audit");
        assert_eq!(
            result_for(&report, "preprocessor-fit-includes-test").disposition,
            Disposition::Rejected
        );
    }

    #[test]
    fn specimen_identity_crossing_split_is_rejected() {
        let report = audit_fixture_json(FIXTURE).expect("fixture audit");
        assert_eq!(
            result_for(&report, "specimen-duplicated-across-folds").disposition,
            Disposition::Rejected
        );
    }

    #[test]
    fn raw_digest_crossing_split_is_rejected() {
        let report = audit_fixture_json(FIXTURE).expect("fixture audit");
        assert_eq!(
            result_for(&report, "raw-record-digest-duplicated-across-folds").disposition,
            Disposition::Rejected
        );
    }

    #[test]
    fn missing_holdout_identity_is_blocked_not_imputed() {
        let report = audit_fixture_json(FIXTURE).expect("fixture audit");
        assert_eq!(
            result_for(&report, "missing-source-identity").disposition,
            Disposition::Blocked
        );
    }

    #[test]
    fn missing_augmentation_parent_is_rejected() {
        let report = audit_fixture_json(FIXTURE).expect("fixture audit");
        assert_eq!(
            result_for(&report, "augmentation-parent-missing").disposition,
            Disposition::Rejected
        );
    }

    #[test]
    fn synthetic_row_in_test_fold_is_rejected() {
        let report = audit_fixture_json(FIXTURE).expect("fixture audit");
        assert_eq!(
            result_for(&report, "synthetic-row-assigned-to-test").disposition,
            Disposition::Rejected
        );
    }

    #[test]
    fn empty_preprocessing_fit_lineage_is_blocked() {
        let report = audit_fixture_json(FIXTURE).expect("fixture audit");
        assert_eq!(
            result_for(&report, "incomplete-preprocessing-fit-lineage").disposition,
            Disposition::Blocked
        );
    }

    #[test]
    fn partial_preprocessing_fit_population_is_blocked() {
        let report = audit_fixture_json(FIXTURE).expect("fixture audit");
        assert_eq!(
            result_for(&report, "preprocessing-fit-omits-train-row").disposition,
            Disposition::Blocked
        );
    }


    #[test]
    fn missing_assignment_blocks_scenario() {
        let mut value = fixture_value();
        let scenario = scenario_mut(&mut value, "source-heldout-clean");
        scenario["assignments"].as_array_mut().unwrap().pop();
        let report = run(&value);
        assert_eq!(
            result_for(&report, "source-heldout-clean").disposition,
            Disposition::Blocked
        );
    }

    #[test]
    fn group_must_not_cross_training_and_test() {
        let mut value = fixture_value();
        let scenario = scenario_mut(&mut value, "source-heldout-clean");
        scenario["assignments"][0]["fold"] = serde_json::json!("TEST");
        let report = run(&value);
        assert_eq!(
            result_for(&report, "source-heldout-clean").disposition,
            Disposition::Rejected
        );
    }

    #[test]
    fn augmentation_must_be_train_fold_only_even_with_train_parents() {
        let mut value = fixture_value();
        let scenario = scenario_mut(&mut value, "source-heldout-clean");
        scenario["augmentations"][0]["assigned_fold"] = serde_json::json!("TEST");
        let report = run(&value);
        assert_eq!(
            result_for(&report, "source-heldout-clean").disposition,
            Disposition::Rejected
        );
    }
}
