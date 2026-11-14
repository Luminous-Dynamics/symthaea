use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

fn string<'a>(case: &'a Value, key: &str) -> Result<&'a str> {
    case.get(key)
        .and_then(Value::as_str)
        .with_context(|| format!("missing or invalid string field: {key}"))
}

fn number(case: &Value, key: &str) -> Result<f64> {
    case.get(key)
        .and_then(Value::as_f64)
        .with_context(|| format!("missing or invalid numeric field: {key}"))
}

fn boolean(case: &Value, key: &str) -> Result<bool> {
    case.get(key)
        .and_then(Value::as_bool)
        .with_context(|| format!("missing or invalid boolean field: {key}"))
}

fn overlaps(left: &[String], right: &[String]) -> bool {
    left.iter().any(|item| right.iter().any(|other| item == other))
}

fn string_array(case: &Value, key: &str) -> Result<Vec<String>> {
    let values = case
        .get(key)
        .and_then(Value::as_array)
        .with_context(|| format!("missing or invalid array field: {key}"))?;

    values
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_owned)
                .with_context(|| format!("non-string entry in {key}"))
        })
        .collect()
}

fn finer_claim(claim: &str) -> bool {
    matches!(claim, "point" | "area" | "field")
}

fn evaluate_case(case: &Value) -> Result<&'static str> {
    // Identity and evidence-class checks come before any numerical interpretation.
    let _id = string(case, "id")?;
    let support = string(case, "support_class")?;
    let claim = string(case, "claim_level")?;
    let evidence = string(case, "evidence_class")?;
    let requires_physical = boolean(case, "requires_physical")?;

    if !matches!(support, "POINT" | "AREA" | "SPECIMEN" | "FIELD" | "INFERRED") {
        return Ok("REJECTED");
    }
    if !matches!(claim, "point" | "area" | "specimen" | "field") {
        return Ok("REJECTED");
    }
    if !matches!(evidence, "PHYSICAL" | "SYNTHETIC" | "INFERRED") {
        return Ok("REJECTED");
    }
    if (requires_physical && evidence != "PHYSICAL")
        || (support == "INFERRED" && evidence != "INFERRED")
        || (evidence == "INFERRED" && requires_physical)
    {
        return Ok("REJECTED");
    }

    // A specimen-level destructive outcome cannot be copied into local labels.
    // Inverse reconstruction is held at ABSTAIN until both the operator and
    // independent local ground truth have been validated.
    if support == "SPECIMEN" && finer_claim(claim) {
        let operator_mode = string(case, "aggregate_operator_mode")?;
        if operator_mode == "INVERSE_RECONSTRUCTION" {
            if !boolean(case, "aggregate_operator_validated")?
                || !boolean(case, "independent_local_truth")?
            {
                return Ok("BLOCKED");
            }
        } else {
            return Ok("REJECTED");
        }
    }
    if support == "AREA" && claim == "point" {
        return Ok("REJECTED");
    }
    if support == "AREA" && claim == "specimen" {
        let mode = string(case, "aggregate_operator_mode")?;
        let validated = boolean(case, "aggregate_operator_validated")?;
        let coverage = case
            .get("aggregate_coverage_fraction")
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        if mode != "AREA_TO_SPECIMEN" || !validated || coverage < 1.0 {
            return Ok("BLOCKED");
        }
    }
    if support == "POINT" && matches!(claim, "area" | "field") {
        return Ok("BLOCKED");
    }
    if support == "FIELD" && claim == "point" {
        return Ok("BLOCKED");
    }
    if support == "FIELD" && claim == "field" {
        if string(case, "field_scope_id")?.is_empty()
            || string(case, "field_sampling_plan_id")?.is_empty()
            || string(case, "field_interpolation_method")?.is_empty()
            || number(case, "field_spatial_resolution_mm")? <= 0.0
            || number(case, "field_unmapped_region_count")? > 0.0
        {
            return Ok("BLOCKED");
        }
    }
    if support == "INFERRED" && claim != "field" {
        return Ok("REJECTED");
    }

    let source_ids = string_array(case, "source_observation_ids")?;
    let support_ids = string_array(case, "support_ids")?;
    let claimed = number(case, "claimed_independent_count")?;
    let unique_sources: BTreeSet<&str> = source_ids.iter().map(String::as_str).collect();
    let unique_supports: BTreeSet<&str> = support_ids.iter().map(String::as_str).collect();
    if source_ids.is_empty()
        || support_ids.is_empty()
        || source_ids.len() != support_ids.len()
        || claimed < 1.0
        || claimed != source_ids.len() as f64
        || unique_sources.len() != source_ids.len()
        || unique_supports.len() != support_ids.len()
    {
        return Ok("REJECTED");
    }

    // Geometry identity is exact: an old transform cannot be applied to a new mesh.
    if string(case, "geometry_revision")? != string(case, "exposure_geometry_revision")? {
        return Ok("REJECTED");
    }
    if string(case, "source_frame")?.is_empty()
        || string(case, "target_frame")?.is_empty()
        || string(case, "exposure_frame")? != string(case, "target_frame")?
        || string(case, "transform_id")?.is_empty()
        || string(case, "transform_calibration_id")?.is_empty()
    {
        return Ok("BLOCKED");
    }

    // A shared causal group is not independent evidence without a selected-set
    // covariance receipt. Distinct row IDs do not establish independence.
    if boolean(case, "shared_common_cause")?
        && claimed > 1.0
        && !boolean(case, "covariance_receipt_present")?
    {
        return Ok("REJECTED");
    }

    let train = string_array(case, "train_groups")?;
    let test = string_array(case, "test_groups")?;
    if overlaps(&train, &test) {
        return Ok("REJECTED");
    }

    if boolean(case, "acceptance_claim")?
        && number(case, "critical_failed_regions")? > 0.0
    {
        return Ok("REJECTED");
    }
    if number(case, "critical_unknown_regions")? > 0.0 {
        return Ok("BLOCKED");
    }

    // Registration error must be bounded in physical units and smaller than
    // both the admissible calibration bound and the actual decision margin.
    let registration = number(case, "registration_bound_mm")?;
    let allowed = number(case, "allowed_registration_bound_mm")?;
    let margin = number(case, "decision_margin_mm")?;
    if registration < 0.0 || allowed <= 0.0 || margin <= 0.0 {
        return Ok("BLOCKED");
    }
    if registration > allowed || registration >= margin {
        return Ok("BLOCKED");
    }

    // Compare temporal intervals after expanding both by their declared clock
    // uncertainty. Disjoint intervals cannot be paired as the same treatment.
    let obs_start = number(case, "observation_start_ms")?;
    let obs_end = number(case, "observation_end_ms")?;
    let exp_start = number(case, "exposure_start_ms")?;
    let exp_end = number(case, "exposure_end_ms")?;
    let time_uncertainty = number(case, "time_uncertainty_ms")?;
    if obs_end < obs_start || exp_end < exp_start || time_uncertainty < 0.0 {
        return Ok("BLOCKED");
    }
    if obs_end + time_uncertainty < exp_start - time_uncertainty
        || exp_end + time_uncertainty < obs_start - time_uncertainty
    {
        return Ok("BLOCKED");
    }

    if matches!(support, "POINT" | "AREA") {
        let footprint = case.get("footprint_id").and_then(Value::as_str);
        if footprint.is_none_or(str::is_empty)
            || number(case, "support_area_mm2")? <= 0.0
            || !boolean(case, "has_weighting")?
        {
            return Ok("BLOCKED");
        }
    }

    // Synthetic evidence can pass a schema/contract audit only when it is not
    // being used to satisfy a physical measurement obligation.
    if evidence == "SYNTHETIC" && requires_physical {
        return Ok("REJECTED");
    }

    Ok("PASS")
}

pub fn audit_json(contents: &str) -> Result<BTreeMap<String, usize>> {
    let root: Value = serde_json::from_str(contents).context("parse plasma region-truth fixture")?;
    if root.get("benchmark_id").and_then(Value::as_str) != Some("PLASMA-REGION-TRUTH-001") {
        bail!("unexpected benchmark_id");
    }
    if root.get("physical_validation").and_then(Value::as_bool) != Some(false) {
        bail!("fixture must explicitly deny physical validation");
    }
    if root.get("product_release_authority").and_then(Value::as_bool) != Some(false) {
        bail!("fixture must explicitly deny product-release authority");
    }

    let cases = root
        .get("cases")
        .and_then(Value::as_array)
        .context("fixture has no cases array")?;
    if cases.is_empty() {
        bail!("fixture must contain at least one case");
    }

    let mut counts = BTreeMap::new();
    for case in cases {
        let id = string(case, "id")?;
        let expected = string(case, "expected")?;
        let actual = evaluate_case(case)?;
        if actual != expected {
            bail!("{id}: expected {expected}, got {actual}");
        }
        *counts.entry(actual.to_owned()).or_insert(0) += 1;
    }
    Ok(counts)
}

pub fn run_audit(path: &Path) -> Result<()> {
    let contents = fs::read_to_string(path)
        .with_context(|| format!("read fixture {}", path.display()))?;
    let counts = audit_json(&contents)?;
    let total: usize = counts.values().sum();
    println!("PLASMA-REGION-TRUTH-001: {total}/{total} expected dispositions matched");
    for (disposition, count) in counts {
        println!("  {disposition}: {count}");
    }
    println!("Synthetic contract audit only; no physical qualification or product release.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str =
        include_str!("../../data/research/plasma_region_truth_fixture_v0_1.json");

    #[test]
    fn frozen_fixture_matches_all_expected_dispositions() {
        let counts = audit_json(FIXTURE).expect("fixture should pass its known-answer audit");
        assert_eq!(counts.get("PASS"), Some(&5));
        assert_eq!(counts.get("BLOCKED"), Some(&5));
        assert_eq!(counts.get("REJECTED"), Some(&8));
    }

    #[test]
    fn specimen_level_result_cannot_be_smeared_to_pixels() {
        let root: Value = serde_json::from_str(FIXTURE).expect("valid fixture");
        let case = root["cases"]
            .as_array()
            .expect("cases")
            .iter()
            .find(|case| case["id"] == "SPECIMEN_TO_PIXEL_SMEAR")
            .expect("known case");
        assert_eq!(evaluate_case(case).expect("evaluate"), "REJECTED");
    }

    #[test]
    fn unresolved_registration_abstains() {
        let root: Value = serde_json::from_str(FIXTURE).expect("valid fixture");
        let case = root["cases"]
            .as_array()
            .expect("cases")
            .iter()
            .find(|case| case["id"] == "REGISTRATION_BOUNDARY")
            .expect("known case");
        assert_eq!(evaluate_case(case).expect("evaluate"), "BLOCKED");
    }

    #[test]
    fn duplicate_observation_and_support_ids_are_derived_and_rejected() {
        let root: Value = serde_json::from_str(FIXTURE).expect("valid fixture");
        let case = root["cases"]
            .as_array()
            .expect("cases")
            .iter()
            .find(|case| case["id"] == "DUPLICATE_SUPPORT")
            .expect("known case");
        assert_eq!(evaluate_case(case).expect("evaluate"), "REJECTED");
    }


    #[test]
    fn point_field_and_specimen_supports_are_admitted_at_their_declared_level() {
        let root: Value = serde_json::from_str(FIXTURE).expect("valid fixture");
        let cases = root["cases"].as_array().expect("cases");
        for id in ["POINT_SUPPORT", "FIELD_SUPPORT", "SPECIMEN_SUPPORT"] {
            let case = cases
                .iter()
                .find(|case| case["id"] == id)
                .expect("known case");
            assert_eq!(evaluate_case(case).expect("evaluate"), "PASS");
        }
    }

    #[test]
    fn area_to_specimen_claim_requires_a_validated_full_support_operator() {
        let root: Value = serde_json::from_str(FIXTURE).expect("valid fixture");
        let case = root["cases"]
            .as_array()
            .expect("cases")
            .iter()
            .find(|case| case["id"] == "AREA_TO_SPECIMEN_NO_AGGREGATION")
            .expect("known case");
        assert_eq!(evaluate_case(case).expect("evaluate"), "BLOCKED");
    }

    #[test]
    fn synthetic_pass_never_claims_physical_validation() {
        let counts = audit_json(FIXTURE).expect("fixture audit");
        assert_eq!(counts.get("PASS"), Some(&2));
    }
}
