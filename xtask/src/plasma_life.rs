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

fn nonempty(case: &Value, key: &str) -> bool {
    case.get(key)
        .and_then(Value::as_str)
        .is_some_and(|value| !value.trim().is_empty())
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn failure_evidence_valid(case: &Value) -> Result<bool> {
    Ok(
        boolean(case, "failure_mode_adjudicated")?
            && nonempty(case, "failure_mode_evidence_id")
            && string(case, "failure_mode")? != "UNCLASSIFIED",
    )
}

fn evaluate_case(case: &Value) -> Result<&'static str> {
    let _id = string(case, "id")?;
    if string(case, "data_class")? != "SYNTHETIC" {
        return Ok("REJECTED");
    }

    let claim = string(case, "claim_kind")?;
    let clock = string(case, "clock_kind")?;
    let analysis_clock = string(case, "analysis_clock_kind")?;
    let modality = string(case, "measurement_modality")?;
    let terminal = string(case, "terminal_status")?;

    if !matches!(claim, "STRENGTH_AT_AGE" | "SURVIVAL")
        || !matches!(clock, "PRE_BOND_DELAY" | "POST_BOND_AGE" | "UNDER_LOAD_SERVICE")
        || !matches!(
            modality,
            "DESTRUCTIVE_CROSS_SECTION" | "SUSTAINED_LOAD" | "PERIODIC_INSPECTION"
        )
    {
        return Ok("REJECTED");
    }
    if clock != analysis_clock {
        return Ok("REJECTED");
    }

    let specimen_ids = string_array(case, "physical_specimen_ids")?;
    let unique_specimens: BTreeSet<&str> =
        specimen_ids.iter().map(String::as_str).collect();
    let claimed_count = case
        .get("claimed_independent_specimen_count")
        .and_then(Value::as_u64)
        .unwrap_or(0) as usize;
    if specimen_ids.is_empty()
        || specimen_ids.iter().any(|id| id.trim().is_empty())
        || unique_specimens.len() != specimen_ids.len()
        || claimed_count != specimen_ids.len()
    {
        return Ok("REJECTED");
    }

    if !nonempty(case, "material_profile_id")
        || !nonempty(case, "adhesive_profile_id")
        || !nonempty(case, "protocol_id")
        || !nonempty(case, "calibration_id")
    {
        return Ok("BLOCKED");
    }
    let raw_digest = string(case, "raw_observation_digest")?;
    if !valid_digest(raw_digest) {
        return Ok("BLOCKED");
    }

    let material_profiles = case
        .get("cohort_material_profile_count")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let adhesive_profiles = case
        .get("cohort_adhesive_profile_count")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    if material_profiles == 0 || adhesive_profiles == 0 {
        return Ok("BLOCKED");
    }
    if (material_profiles > 1 || adhesive_profiles > 1)
        && !boolean(case, "stratified_cohort")?
        && !nonempty(case, "cohort_transfer_receipt_id")
    {
        return Ok("REJECTED");
    }

    // A claimed survival estimand must be based on a true life-test modality.
    // Destructive strength-at-age observations remain cross-sectional outcomes.
    if claim == "STRENGTH_AT_AGE" {
        if clock == "UNDER_LOAD_SERVICE"
            || modality != "DESTRUCTIVE_CROSS_SECTION"
            || terminal != "STRENGTH_AT_AGE"
        {
            return Ok("REJECTED");
        }
        let age = number(case, "measurement_age_days")?;
        let strength = number(case, "strength_at_age_mpa")?;
        if age <= 0.0 || strength <= 0.0 {
            return Ok("BLOCKED");
        }
    } else {
        if clock != "UNDER_LOAD_SERVICE"
            || !matches!(modality, "SUSTAINED_LOAD" | "PERIODIC_INSPECTION")
        {
            return Ok("REJECTED");
        }
        if matches!(terminal, "UNKNOWN" | "INVALID") {
            return Ok("BLOCKED");
        }
        if !matches!(
            terminal,
            "EVENT" | "RIGHT_CENSORED" | "INTERVAL_CENSORED" | "COMPETING_EVENT"
        ) {
            return Ok("REJECTED");
        }

        match terminal {
            "EVENT" | "COMPETING_EVENT" => {
                let event_time = number(case, "event_time_days")?;
                if event_time <= 0.0 {
                    return Ok("BLOCKED");
                }
                if !failure_evidence_valid(case)? {
                    return Ok("BLOCKED");
                }
                if terminal == "COMPETING_EVENT" && modality == "DESTRUCTIVE_CROSS_SECTION" {
                    return Ok("REJECTED");
                }
            }
            "RIGHT_CENSORED" => {
                let last_at_risk = number(case, "last_at_risk_days")?;
                if last_at_risk <= 0.0 || !nonempty(case, "censor_reason") {
                    return Ok("BLOCKED");
                }
                let mechanism = string(case, "censoring_mechanism")?;
                let reason = string(case, "censor_reason")?;
                if reason != "ADMINISTRATIVE_END"
                    || mechanism != "DECLARED_ADMINISTRATIVE"
                    || !nonempty(case, "censoring_design_receipt_id")
                {
                    return Ok("BLOCKED");
                }
            }
            "INTERVAL_CENSORED" => {
                if modality != "PERIODIC_INSPECTION" {
                    return Ok("REJECTED");
                }
                let last_good = number(case, "last_known_good_days")?;
                let first_failed = number(case, "first_known_failed_days")?;
                if last_good < 0.0
                    || first_failed <= last_good
                    || !nonempty(case, "inspection_record_id")
                    || !failure_evidence_valid(case)?
                {
                    return Ok("BLOCKED");
                }
            }
            _ => return Ok("REJECTED"),
        }

        // A quantile is not identified by a cohort with no observed failures.
        let event_count = case
            .get("cohort_event_count")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let quantile_claimed = case
            .get("claimed_life_quantile")
            .and_then(Value::as_str)
            .is_some_and(|value| !value.is_empty());
        if quantile_claimed {
            // S0 has no qualified life-quantile estimator. A zero-event cohort
            // cannot identify a finite median/q90; a nonzero-event cohort still
            // needs a future source-bound, independently qualified curve receipt.
            if event_count == 0 {
                return Ok("REJECTED");
            }
            return Ok("BLOCKED");
        }
    }

    let accelerated = boolean(case, "accelerated_claim")?;
    if accelerated {
        if boolean(case, "mechanism_changed")? {
            return Ok("REJECTED");
        }
        if !nonempty(case, "acceleration_model_id")
            || !nonempty(case, "reference_condition_id")
        {
            return Ok("BLOCKED");
        }
        if !nonempty(case, "condition_history_id") {
            return Ok("BLOCKED");
        }
        let evidence = string_array(case, "mechanism_evidence_ids")?;
        if evidence.len() < 2 || evidence.iter().any(|value| value.trim().is_empty()) {
            return Ok("BLOCKED");
        }

        let min_stress = number(case, "observed_stress_min_c")?;
        let max_stress = number(case, "observed_stress_max_c")?;
        let predicted_stress = number(case, "predicted_stress_c")?;
        let observed_time = number(case, "observed_time_max_days")?;
        let prediction_horizon = number(case, "prediction_horizon_days")?;
        if !min_stress.is_finite()
            || !max_stress.is_finite()
            || !predicted_stress.is_finite()
            || !observed_time.is_finite()
            || !prediction_horizon.is_finite()
            || min_stress >= max_stress
            || observed_time <= 0.0
            || prediction_horizon <= 0.0
        {
            return Ok("BLOCKED");
        }
        // S0 only admits interpolation within the observed stress and time envelope.
        if predicted_stress < min_stress
            || predicted_stress > max_stress
            || prediction_horizon > observed_time
        {
            return Ok("BLOCKED");
        }
    }

    Ok("PASS")
}

pub fn audit_json(contents: &str) -> Result<BTreeMap<String, usize>> {
    let root: Value =
        serde_json::from_str(contents).context("parse plasma life benchmark fixture")?;
    if root.get("benchmark_id").and_then(Value::as_str) != Some("PLASMA-LIFE-001") {
        bail!("unexpected benchmark_id");
    }
    if root.get("physical_validation").and_then(Value::as_bool) != Some(false) {
        bail!("fixture must explicitly deny physical validation");
    }
    if root
        .get("service_life_claim_authority")
        .and_then(Value::as_bool)
        != Some(false)
    {
        bail!("fixture must explicitly deny service-life claim authority");
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
    println!("PLASMA-LIFE-001: {total}/{total} expected dispositions matched");
    for (disposition, count) in counts {
        println!("  {disposition}: {count}");
    }
    println!("Synthetic evidence-contract audit only; no bond-life estimate or qualification.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = include_str!("../../data/research/plasma_life_fixture_v0_1.json");

    #[test]
    fn frozen_fixture_matches_all_expected_dispositions() {
        let counts = audit_json(FIXTURE).expect("fixture should pass its known-answer audit");
        assert_eq!(counts.get("PASS"), Some(&5));
        assert_eq!(counts.get("BLOCKED"), Some(&5));
        assert_eq!(counts.get("REJECTED"), Some(&8));
    }

    #[test]
    fn destructive_cross_section_is_not_a_survival_event() {
        let root: Value = serde_json::from_str(FIXTURE).expect("valid fixture");
        let case = root["cases"]
            .as_array()
            .expect("cases")
            .iter()
            .find(|case| case["id"] == "DESTRUCTIVE_STRENGTH_AS_SURVIVAL")
            .expect("known case");
        assert_eq!(evaluate_case(case).expect("evaluate"), "REJECTED");
    }

    #[test]
    fn treatment_delay_and_service_time_cannot_be_pooled() {
        let root: Value = serde_json::from_str(FIXTURE).expect("valid fixture");
        let case = root["cases"]
            .as_array()
            .expect("cases")
            .iter()
            .find(|case| case["id"] == "TREATMENT_DELAY_AS_SERVICE_AGE")
            .expect("known case");
        assert_eq!(evaluate_case(case).expect("evaluate"), "REJECTED");
    }

    #[test]
    fn interval_censoring_preserves_both_observation_bounds() {
        let root: Value = serde_json::from_str(FIXTURE).expect("valid fixture");
        let case = root["cases"]
            .as_array()
            .expect("cases")
            .iter()
            .find(|case| case["id"] == "INTERVAL_CENSORED_INSPECTION")
            .expect("known case");
        assert_eq!(evaluate_case(case).expect("evaluate"), "PASS");
    }

    #[test]
    fn unknown_and_policy_related_censoring_abstain() {
        let root: Value = serde_json::from_str(FIXTURE).expect("valid fixture");
        let cases = root["cases"].as_array().expect("cases");
        for id in ["UNKNOWN_OUTCOME_AS_CENSOR", "INFORMATIVE_POLICY_CENSOR"] {
            let case = cases
                .iter()
                .find(|case| case["id"] == id)
                .expect("known case");
            assert_eq!(evaluate_case(case).expect("evaluate"), "BLOCKED");
        }
    }

    #[test]
    fn acceleration_requires_mechanism_evidence_and_in_range_prediction() {
        let root: Value = serde_json::from_str(FIXTURE).expect("valid fixture");
        let cases = root["cases"].as_array().expect("cases");
        for (id, expected) in [
            ("EXACT_EVENT", "PASS"),
            ("ACCELERATED_OUTSIDE_RANGE", "BLOCKED"),
            ("MECHANISM_SHIFT", "REJECTED"),
            ("NO_MECHANISM_EVIDENCE", "BLOCKED"),
            ("STRESS_HISTORY_GAP", "BLOCKED"),
        ] {
            let case = cases
                .iter()
                .find(|case| case["id"] == id)
                .expect("known case");
            assert_eq!(evaluate_case(case).expect("evaluate"), expected);
        }
    }

    #[test]
    fn specimen_duplicates_are_rejected() {
        let root: Value = serde_json::from_str(FIXTURE).expect("valid fixture");
        let case = root["cases"]
            .as_array()
            .expect("cases")
            .iter()
            .find(|case| case["id"] == "DUPLICATE_SPECIMEN")
            .expect("known case");
        assert_eq!(evaluate_case(case).expect("evaluate"), "REJECTED");
    }
}
