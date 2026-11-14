use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::collections::BTreeMap;
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

fn evaluate_case(case: &Value) -> Result<&'static str> {
    let _id = string(case, "id")?;
    if string(case, "evidence_class")? != "SYNTHETIC" {
        return Ok("REJECTED");
    }

    // Exact article identity and canonical profile identity precede any
    // interpretation of service, release or recovery evidence.
    if !nonempty(case, "source_specimen_id")
        || !nonempty(case, "observed_specimen_id")
        || string(case, "source_specimen_id")? != string(case, "observed_specimen_id")?
    {
        return Ok("REJECTED");
    }
    if !nonempty(case, "material_profile_id") || !nonempty(case, "adhesive_profile_id") {
        return Ok("BLOCKED");
    }

    // Plasma/adhesive compatibility is an independently evidenced property.
    if boolean(case, "plasma_pretreated")? {
        let compatibility = string(case, "plasma_compatibility_result")?;
        if compatibility == "FAIL" {
            return Ok("REJECTED");
        }
        if compatibility != "PASS" || !nonempty(case, "plasma_compatibility_receipt_id") {
            return Ok("BLOCKED");
        }
    }

    let service_claim = boolean(case, "service_claim")?;
    let release_claim = boolean(case, "release_claim")?;
    let rebond_claim = boolean(case, "rebond_claim")?;

    // Service performance must be demonstrated before a lifecycle release claim.
    if service_claim {
        if boolean(case, "premature_release")? {
            return Ok("REJECTED");
        }
        if !nonempty(case, "service_test_receipt_id")
            || !nonempty(case, "aging_test_receipt_id")
        {
            return Ok("BLOCKED");
        }
        let initial = number(case, "service_strength_mpa")?;
        let initial_floor = number(case, "service_strength_floor_mpa")?;
        let aged = number(case, "aged_strength_mpa")?;
        let aged_floor = number(case, "aged_strength_floor_mpa")?;
        if initial_floor <= 0.0 || aged_floor <= 0.0 || initial <= 0.0 || aged <= 0.0 {
            return Ok("BLOCKED");
        }
        if initial < initial_floor || aged < aged_floor {
            return Ok("REJECTED");
        }
    }

    // A release claim cannot skip the independently qualified service phase.
    if release_claim && !service_claim {
        return Ok("REJECTED");
    }

    // Release is a separate, trigger-qualified and spatially selective event.
    if release_claim {
        let declared = string(case, "declared_trigger_modality")?;
        let actual = string(case, "actual_trigger_modality")?;
        let supported_trigger = |value: &str| {
            matches!(
                value,
                "THERMAL"
                    | "JOULE_HEAT"
                    | "INDUCTION"
                    | "PHOTOTHERMAL_UV"
                    | "CHEMICAL_SOLVOLYSIS"
                    | "MAGNETIC_HYBRID"
            )
        };
        if !supported_trigger(declared) || !supported_trigger(actual) {
            return Ok("REJECTED");
        }
        if declared != actual {
            return Ok("REJECTED");
        }
        if !nonempty(case, "trigger_history_id")
            || !nonempty(case, "trigger_calibration_id")
            || !valid_digest(string(case, "trigger_observation_digest")?)
        {
            return Ok("BLOCKED");
        }

        let energy = number(case, "trigger_energy_j")?;
        let energy_limit = number(case, "trigger_energy_limit_j")?;
        let duration = number(case, "trigger_duration_s")?;
        let duration_limit = number(case, "trigger_duration_limit_s")?;
        let temperature = number(case, "trigger_temperature_c")?;
        let temperature_limit = number(case, "trigger_temperature_limit_c")?;
        if energy <= 0.0
            || energy_limit <= 0.0
            || duration <= 0.0
            || duration_limit <= 0.0
            || temperature <= 0.0
            || temperature_limit <= 0.0
        {
            return Ok("BLOCKED");
        }
        if energy > energy_limit
            || duration > duration_limit
            || temperature > temperature_limit
        {
            return Ok("BLOCKED");
        }

        let target = string(case, "target_joint_id")?;
        let protected = string_array(case, "protected_joint_ids")?;
        let released = string_array(case, "released_joint_ids")?;
        let protected_unique: std::collections::BTreeSet<&str> =
            protected.iter().map(String::as_str).collect();
        let released_unique: std::collections::BTreeSet<&str> =
            released.iter().map(String::as_str).collect();
        if protected_unique.len() != protected.len()
            || released_unique.len() != released.len()
            || protected.iter().any(|joint| joint == target)
        {
            return Ok("REJECTED");
        }
        if !released.iter().any(|joint| joint == target) {
            return Ok("BLOCKED");
        }
        if protected.iter().any(|joint| released.contains(joint)) {
            return Ok("REJECTED");
        }
        if !nonempty(case, "release_observation_id")
            || !nonempty(case, "release_selectivity_receipt_id")
        {
            return Ok("BLOCKED");
        }

        // A successful separation is not a substrate-recovery result.
        if boolean(case, "substrate_damage")? {
            return Ok("REJECTED");
        }
        if !nonempty(case, "substrate_damage_evidence_id") {
            return Ok("BLOCKED");
        }
        let residue_status = string(case, "residue_status")?;
        if residue_status != "MEASURED_WITHIN_LIMIT" {
            return Ok("BLOCKED");
        }
        let residual = number(case, "residual_mass_mg")?;
        let residual_limit = number(case, "residue_limit_mg")?;
        if residual < 0.0 || residual_limit < 0.0 {
            return Ok("BLOCKED");
        }
        if residual > residual_limit {
            return Ok("REJECTED");
        }
        if !nonempty(case, "recovery_receipt_id") {
            return Ok("BLOCKED");
        }
    }

    // Rebonding requires a new process/adhesive/test lineage after a release.
    if rebond_claim {
        if !release_claim {
            return Ok("REJECTED");
        }
        let cycle = case
            .get("rebond_cycle_number")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        if cycle == 0
            || !nonempty(case, "rebond_process_lineage_id")
            || !nonempty(case, "rebond_adhesive_profile_id")
            || !nonempty(case, "rebond_test_receipt_id")
        {
            return Ok("BLOCKED");
        }
        let rebond_strength = number(case, "rebond_strength_mpa")?;
        let rebond_floor = number(case, "rebond_strength_floor_mpa")?;
        if rebond_strength <= 0.0 || rebond_floor <= 0.0 {
            return Ok("BLOCKED");
        }
        if rebond_strength < rebond_floor {
            return Ok("REJECTED");
        }
    }

    Ok("PASS")
}

pub fn audit_json(contents: &str) -> Result<BTreeMap<String, usize>> {
    let root: Value =
        serde_json::from_str(contents).context("parse plasma circular-bond fixture")?;
    if root.get("benchmark_id").and_then(Value::as_str) != Some("PLASMA-CIRCULAR-BOND-001") {
        bail!("unexpected benchmark_id");
    }
    if root.get("physical_validation").and_then(Value::as_bool) != Some(false) {
        bail!("fixture must explicitly deny physical validation");
    }
    if root
        .get("equipment_control_authority")
        .and_then(Value::as_bool)
        != Some(false)
    {
        bail!("fixture must explicitly deny equipment control authority");
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
    println!("PLASMA-CIRCULAR-BOND-001: {total}/{total} expected dispositions matched");
    for (disposition, count) in counts {
        println!("  {disposition}: {count}");
    }
    println!("Synthetic lifecycle audit only; no chemistry qualification or equipment authority.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str =
        include_str!("../../data/research/plasma_circular_bond_fixture_v0_1.json");

    #[test]
    fn frozen_fixture_matches_all_expected_dispositions() {
        let counts = audit_json(FIXTURE).expect("fixture should pass its known-answer audit");
        assert_eq!(counts.get("PASS"), Some(&3));
        assert_eq!(counts.get("BLOCKED"), Some(&5));
        assert_eq!(counts.get("REJECTED"), Some(&7));
    }

    #[test]
    fn premature_release_fails_the_service_gate() {
        let root: Value = serde_json::from_str(FIXTURE).expect("valid fixture");
        let case = root["cases"]
            .as_array()
            .expect("cases")
            .iter()
            .find(|case| case["id"] == "PREMATURE_RELEASE")
            .expect("known case");
        assert_eq!(evaluate_case(case).expect("evaluate"), "REJECTED");
    }

    #[test]
    fn trigger_outside_envelope_is_blocked() {
        let root: Value = serde_json::from_str(FIXTURE).expect("valid fixture");
        let case = root["cases"]
            .as_array()
            .expect("cases")
            .iter()
            .find(|case| case["id"] == "TRIGGER_OUTSIDE_ENVELOPE")
            .expect("known case");
        assert_eq!(evaluate_case(case).expect("evaluate"), "BLOCKED");
    }

    #[test]
    fn collateral_release_is_rejected() {
        let root: Value = serde_json::from_str(FIXTURE).expect("valid fixture");
        let case = root["cases"]
            .as_array()
            .expect("cases")
            .iter()
            .find(|case| case["id"] == "COLLATERAL_RELEASE")
            .expect("known case");
        assert_eq!(evaluate_case(case).expect("evaluate"), "REJECTED");
    }

    #[test]
    fn substrate_damage_and_residue_are_distinct_gates() {
        let root: Value = serde_json::from_str(FIXTURE).expect("valid fixture");
        let cases = root["cases"].as_array().expect("cases");
        for (id, expected) in [
            ("SUBSTRATE_DAMAGE", "REJECTED"),
            ("RESIDUE_UNQUALIFIED", "BLOCKED"),
        ] {
            let case = cases
                .iter()
                .find(|case| case["id"] == id)
                .expect("known case");
            assert_eq!(evaluate_case(case).expect("evaluate"), expected);
        }
    }

    #[test]
    fn rebond_claim_requires_new_lineage_and_service_floor() {
        let root: Value = serde_json::from_str(FIXTURE).expect("valid fixture");
        let cases = root["cases"].as_array().expect("cases");
        for (id, expected) in [
            ("REBOND_CYCLE_UNVERIFIED", "BLOCKED"),
            ("REBOND_STRENGTH_DECAY", "REJECTED"),
            ("COMPLETE_SYNTHETIC", "PASS"),
        ] {
            let case = cases
                .iter()
                .find(|case| case["id"] == id)
                .expect("known case");
            assert_eq!(evaluate_case(case).expect("evaluate"), expected);
        }
    }

    #[test]
    fn plasma_compatibility_and_trigger_modality_are_independent_gates() {
        let root: Value = serde_json::from_str(FIXTURE).expect("valid fixture");
        let cases = root["cases"].as_array().expect("cases");
        for (id, expected) in [
            ("PLASMA_COMPATIBILITY_UNTESTED", "BLOCKED"),
            ("WRONG_TRIGGER_MODALITY", "REJECTED"),
            ("MANUAL_FORCE_ONLY", "REJECTED"),
        ] {
            let case = cases
                .iter()
                .find(|case| case["id"] == id)
                .expect("known case");
            assert_eq!(evaluate_case(case).expect("evaluate"), expected);
        }
    }


    #[test]
    fn release_requires_service_qualification_and_supported_trigger_type() {
        let root: Value = serde_json::from_str(FIXTURE).expect("valid fixture");
        let cases = root["cases"].as_array().expect("cases");
        let selective = cases
            .iter()
            .find(|case| case["id"] == "SELECTIVE_RELEASE")
            .expect("known case");
        let mut no_service = selective.clone();
        no_service["service_claim"] = Value::Bool(false);
        assert_eq!(evaluate_case(&no_service).expect("evaluate"), "REJECTED");

        let mut unsupported = selective.clone();
        unsupported["declared_trigger_modality"] = Value::String("UNKNOWN_TRIGGER".to_owned());
        unsupported["actual_trigger_modality"] = Value::String("UNKNOWN_TRIGGER".to_owned());
        assert_eq!(evaluate_case(&unsupported).expect("evaluate"), "REJECTED");
    }

    #[test]
    fn article_identity_must_survive_release() {
        let root: Value = serde_json::from_str(FIXTURE).expect("valid fixture");
        let case = root["cases"]
            .as_array()
            .expect("cases")
            .iter()
            .find(|case| case["id"] == "MATERIAL_IDENTITY_MISMATCH")
            .expect("known case");
        assert_eq!(evaluate_case(case).expect("evaluate"), "REJECTED");
    }
}
