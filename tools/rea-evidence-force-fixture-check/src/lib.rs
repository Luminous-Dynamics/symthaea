//! Structural and semantic-consistency checks for the REA force fixture corpus.
//!
//! This validates the authored dataset as a development artifact. It does not
//! adjudicate the evidence/claim labels or evaluate a system under test.

use jsonschema::draft202012;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

const SCHEMA: &str = include_str!("../../../docs/research/rea-evidence-force-pairs-v0.schema.json");
const FIXTURE: &str = include_str!("../../../docs/research/rea-evidence-force-pairs-v0.json");

const AXES: [&str; 8] = [
    "relation",
    "modality",
    "scope",
    "temporal_validity",
    "numeric_specificity",
    "attribution",
    "source_lineage",
    "population_completeness",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixtureError(pub String);

impl std::fmt::Display for FixtureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for FixtureError {}

fn string_at<'a>(value: &'a Value, path: &[&str]) -> Result<&'a str, FixtureError> {
    let mut current = value;
    for segment in path {
        current = current
            .get(*segment)
            .ok_or_else(|| FixtureError(format!("missing JSON path segment: {segment}")))?;
    }
    current
        .as_str()
        .ok_or_else(|| FixtureError(format!("JSON path is not a string: {}", path.join("."))))
}

pub fn validate_fixture(value: &Value) -> Result<(), FixtureError> {
    if string_at(value, &["status"])? != "design_fixture_only"
        || string_at(value, &["source_policy"])? != "original_synthetic_text_only"
        || string_at(value, &["use"])? != "development_and_schema_conformance"
        || value.get("not_human_adjudicated").and_then(Value::as_bool) != Some(true)
    {
        return Err(FixtureError(
            "fixture must retain its design-only / synthetic / non-adjudicated status".into(),
        ));
    }

    let pairs = value
        .get("primary_monotonicity_pairs")
        .and_then(Value::as_array)
        .ok_or_else(|| FixtureError("primary_monotonicity_pairs must be an array".into()))?;
    let controls = value
        .get("positive_evidence_growth_controls")
        .and_then(Value::as_array)
        .ok_or_else(|| FixtureError("positive_evidence_growth_controls must be an array".into()))?;

    if pairs.len() != 24 {
        return Err(FixtureError(format!(
            "expected 24 primary pairs, found {}",
            pairs.len()
        )));
    }
    if controls.len() != 8 {
        return Err(FixtureError(format!(
            "expected 8 evidence-growth controls, found {}",
            controls.len()
        )));
    }

    let mut pair_ids = BTreeSet::new();
    let mut evidence_ids = BTreeSet::new();
    let mut source_families = BTreeSet::new();
    let mut pair_axis_counts = BTreeMap::<String, usize>::new();

    for pair in pairs {
        let id = string_at(pair, &["pair_id"])?;
        if !pair_ids.insert(id.to_owned()) {
            return Err(FixtureError(format!("duplicate pair_id: {id}")));
        }
        let axis = string_at(pair, &["axis"])?;
        if !AXES.contains(&axis) {
            return Err(FixtureError(format!("unknown pair axis {axis} in {id}")));
        }
        *pair_axis_counts.entry(axis.to_owned()).or_default() += 1;

        if string_at(pair, &["split"])? != "development"
            || string_at(pair, &["expected_pair_relation"])? != "calibrated_strictly_more_warranted"
            || string_at(pair, &["calibrated_claim", "expected_warrant"])? != "adequate_as_written"
            || string_at(pair, &["force_raised_claim", "expected_warrant"])? != "insufficient"
            || string_at(pair, &["single_axis_mutation", "dimension"])? != axis
            || string_at(pair, &["single_axis_mutation", "operation"])? != "replace_span"
        {
            return Err(FixtureError(format!("pair contract mismatch for {id}")));
        }

        let evidence_id = string_at(pair, &["evidence", "evidence_id"])?;
        if !evidence_ids.insert(evidence_id.to_owned()) {
            return Err(FixtureError(format!(
                "duplicate evidence_id across contrast pairs: {evidence_id}"
            )));
        }
        if string_at(pair, &["evidence", "source_kind"])? != "synthetic" {
            return Err(FixtureError(format!(
                "non-synthetic evidence source in development fixture {id}"
            )));
        }

        let family = string_at(pair, &["source_family_id"])?;
        if !source_families.insert(family.to_owned()) {
            return Err(FixtureError(format!(
                "source family reused across contrast items: {family}"
            )));
        }

        let from = string_at(pair, &["single_axis_mutation", "from"])?;
        let to = string_at(pair, &["single_axis_mutation", "to"])?;
        let calibrated = string_at(pair, &["calibrated_claim", "text"])?;
        let raised = string_at(pair, &["force_raised_claim", "text"])?;
        if from.is_empty() || to.is_empty() || !calibrated.contains(from) || !raised.contains(to) {
            return Err(FixtureError(format!(
                "declared force mutation spans are not present in their claims: {id}"
            )));
        }

        if string_at(pair, &["adjudication", "status"])? != "synthetic_design_expectation"
            || pair
                .pointer("/adjudication/human_review_required")
                .and_then(Value::as_bool)
                != Some(true)
        {
            return Err(FixtureError(format!(
                "pair must retain synthetic expectation + human-review-required status: {id}"
            )));
        }
    }

    for axis in AXES {
        let count = pair_axis_counts.get(axis).copied().unwrap_or_default();
        if count != 3 {
            return Err(FixtureError(format!(
                "expected exactly 3 contrast pairs for axis {axis}, found {count}"
            )));
        }
    }

    let mut control_ids = BTreeSet::new();
    let mut control_axis_counts = BTreeMap::<String, usize>::new();
    for control in controls {
        let id = string_at(control, &["control_id"])?;
        if !control_ids.insert(id.to_owned()) {
            return Err(FixtureError(format!("duplicate control_id: {id}")));
        }
        let axis = string_at(control, &["axis"])?;
        if !AXES.contains(&axis) {
            return Err(FixtureError(format!("unknown control axis {axis} in {id}")));
        }
        *control_axis_counts.entry(axis.to_owned()).or_default() += 1;
        if string_at(control, &["split"])? != "development"
            || string_at(control, &["expected_transition", "after"])? != "adequate_as_written"
            || string_at(control, &["adjudication", "status"])? != "synthetic_design_expectation"
            || control
                .pointer("/adjudication/human_review_required")
                .and_then(Value::as_bool)
                != Some(true)
        {
            return Err(FixtureError(format!(
                "evidence-growth control contract mismatch: {id}"
            )));
        }
        let before = string_at(control, &["expected_transition", "before"])?;
        if !matches!(before, "adequate_only_with_qualification" | "insufficient") {
            return Err(FixtureError(format!(
                "unexpected evidence-growth precondition {before} in {id}"
            )));
        }
        if string_at(control, &["evidence_before", "source_kind"])? != "synthetic"
            || string_at(control, &["evidence_after", "source_kind"])? != "synthetic"
        {
            return Err(FixtureError(format!(
                "evidence-growth controls must be synthetic: {id}"
            )));
        }
    }

    for axis in AXES {
        let count = control_axis_counts.get(axis).copied().unwrap_or_default();
        if count != 1 {
            return Err(FixtureError(format!(
                "expected exactly 1 evidence-growth control for axis {axis}, found {count}"
            )));
        }
    }

    Ok(())
}

/// Validate both the schema itself and the data instance against it.
pub fn validate_embedded_corpus() -> Result<(), FixtureError> {
    let schema: Value = serde_json::from_str(SCHEMA)
        .map_err(|e| FixtureError(format!("schema JSON parse failed: {e}")))?;
    draft202012::meta::validate(&schema)
        .map_err(|e| FixtureError(format!("Draft 2020-12 meta-schema rejection: {e}")))?;

    let fixture: Value = serde_json::from_str(FIXTURE)
        .map_err(|e| FixtureError(format!("fixture JSON parse failed: {e}")))?;
    let validator = draft202012::options()
        .build(&schema)
        .map_err(|e| FixtureError(format!("schema compilation failed: {e}")))?;
    let errors: Vec<String> = validator
        .iter_errors(&fixture)
        .map(|e| e.to_string())
        .collect();
    if !errors.is_empty() {
        return Err(FixtureError(format!(
            "fixture failed JSON Schema validation: {}",
            errors.join("; ")
        )));
    }
    validate_fixture(&fixture)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Value {
        serde_json::from_str(FIXTURE).expect("checked-in fixture must parse")
    }

    #[test]
    fn draft_2020_12_schema_is_meta_valid_and_fixture_conforms() {
        validate_embedded_corpus().unwrap();
    }

    #[test]
    fn primary_axes_have_three_unique_pairs_each() {
        validate_fixture(&fixture()).unwrap();
    }

    #[test]
    fn duplicate_pair_id_is_rejected() {
        let mut value = fixture();
        let duplicate = value["primary_monotonicity_pairs"][0].clone();
        value["primary_monotonicity_pairs"]
            .as_array_mut()
            .unwrap()
            .push(duplicate);
        assert!(validate_fixture(&value).is_err());
    }

    #[test]
    fn axis_mutation_mismatch_is_rejected() {
        let mut value = fixture();
        value["primary_monotonicity_pairs"][0]["single_axis_mutation"]["dimension"] =
            Value::String("scope".to_owned());
        assert!(validate_fixture(&value).is_err());
    }

    #[test]
    fn absent_mutation_span_is_rejected() {
        let mut value = fixture();
        value["primary_monotonicity_pairs"][0]["single_axis_mutation"]["from"] =
            Value::String("not in the calibrated claim".to_owned());
        assert!(validate_fixture(&value).is_err());
    }

    #[test]
    fn non_synthetic_or_human_review_downgraded_case_is_rejected() {
        let mut value = fixture();
        value["primary_monotonicity_pairs"][0]["evidence"]["source_kind"] =
            Value::String("external".to_owned());
        assert!(validate_fixture(&value).is_err());

        let mut value = fixture();
        value["primary_monotonicity_pairs"][0]["adjudication"]["human_review_required"] =
            Value::Bool(false);
        assert!(validate_fixture(&value).is_err());
    }

    #[test]
    fn duplicate_source_family_is_rejected() {
        let mut value = fixture();
        let first_family = value["primary_monotonicity_pairs"][0]["source_family_id"].clone();
        value["primary_monotonicity_pairs"][1]["source_family_id"] = first_family;
        assert!(validate_fixture(&value).is_err());
    }

    #[test]
    fn evidence_growth_control_cannot_be_hedge_only() {
        let mut value = fixture();
        value["positive_evidence_growth_controls"][0]["expected_transition"]["after"] =
            Value::String("insufficient".to_owned());
        assert!(validate_fixture(&value).is_err());
    }
}
