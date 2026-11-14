//! Decision-qualified measurement-system capability audit (S0).
//! This checks a frozen evidence contract; it does not estimate physical metrology capability.
use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::fs;
use std::path::Path;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Disposition { Pass, Blocked, Rejected }

impl Disposition {
    fn as_str(self) -> &'static str {
        match self { Self::Pass => "PASS", Self::Blocked => "BLOCKED", Self::Rejected => "REJECTED" }
    }
}

pub fn run(fixture_path: &Path) -> Result<()> {
    let bytes = fs::read(fixture_path).with_context(|| format!("reading {}", fixture_path.display()))?;
    let fixture: Value = serde_json::from_slice(&bytes).context("parsing measurement-capability fixture")?;
    let cases = fixture.get("cases").and_then(Value::as_array).context("fixture must contain cases[]")?;
    let policy = fixture.get("policy").context("policy missing")?;
    let mut counts = [0usize; 3];
    for case in cases {
        let id = case.get("id").and_then(Value::as_str).context("case id missing")?;
        let expected = case.get("expected_disposition").and_then(Value::as_str).context("expected disposition missing")?;
        let actual = audit_case(case.get("receipt").unwrap_or(&Value::Null), policy);
        if actual.as_str() != expected {
            bail!("case {id}: expected {expected}, got {}", actual.as_str());
        }
        match actual { Disposition::Pass => counts[0] += 1, Disposition::Blocked => counts[1] += 1, Disposition::Rejected => counts[2] += 1 }
    }
    println!("plasma-measurement-capability-audit: {} cases; PASS={}, BLOCKED={}, REJECTED={}", cases.len(), counts[0], counts[1], counts[2]);
    println!("S0 synthetic evidence-contract audit only; no physical measurement capability is implied.");
    Ok(())
}

fn audit_case(receipt: &Value, policy: &Value) -> Disposition {
    let required = [
        "measurand", "unit", "method_id", "edition", "geometry", "material_scope",
        "instrument_id", "fixture_id", "software_digest", "calibration_id",
        "calibration_current", "traceability_valid", "study_design_identifiable",
        "expanded_uncertainty", "decision_margin", "validity_start", "validity_end",
        "invalidation_events", "source_sha256",
    ];
    for key in required {
        if receipt.get(key).is_none_or(Value::is_null) { return Disposition::Blocked; }
    }
    if receipt["measurand"].as_str() != Some("lap_shear_strength") { return Disposition::Rejected; }
    if receipt["unit"].as_str() != Some("MPa")
        || receipt["method_id"].as_str() != Some("ISO_4587")
        || receipt["geometry"].as_str() != Some("single_lap_rigid_to_rigid")
        || receipt["edition"].as_str() != Some("ISO 4587:2003")
        || receipt["material_scope"].as_str() != Some("rigid_polymer_pair")
        || receipt["instrument_id"].as_str() != Some("UTM-01")
        || receipt["fixture_id"].as_str() != Some("LAP-FIXTURE-01") {
        return Disposition::Blocked;
    }
    if !valid_sha256(receipt["software_digest"].as_str().unwrap_or(""))
        || !valid_sha256(receipt["source_sha256"].as_str().unwrap_or("")) {
        return Disposition::Rejected;
    }
    if receipt["software_digest"].as_str() != policy["qualified_software_digest"].as_str() {
        return Disposition::Blocked;
    }
    if receipt["calibration_id"].as_str().is_none_or(str::is_empty)
        || receipt["calibration_current"].as_bool() != Some(true)
        || receipt["traceability_valid"].as_bool() != Some(true)
        || receipt["study_design_identifiable"].as_bool() != Some(true) {
        return Disposition::Blocked;
    }
    let Some(events) = receipt["invalidation_events"].as_array() else { return Disposition::Blocked; };
    if !events.is_empty() { return Disposition::Blocked; }
    let (Some(start), Some(end)) = (receipt["validity_start"].as_str(), receipt["validity_end"].as_str()) else { return Disposition::Blocked; };
    if !valid_date(start) || !valid_date(end) || start > end { return Disposition::Rejected; }
    if end < "2026-11-14" { return Disposition::Blocked; }
    let (Some(uncertainty), Some(margin)) = (receipt["expanded_uncertainty"].as_f64(), receipt["decision_margin"].as_f64()) else { return Disposition::Blocked; };
    if !uncertainty.is_finite() || !margin.is_finite() || uncertainty < 0.0 || margin <= 0.0 { return Disposition::Rejected; }
    let Some(max_ratio) = policy["max_expanded_uncertainty_to_decision_margin_ratio"].as_f64() else { return Disposition::Blocked; };
    if !max_ratio.is_finite() || !(0.0..=1.0).contains(&max_ratio) { return Disposition::Rejected; }
    if uncertainty / margin <= max_ratio { Disposition::Pass } else { Disposition::Blocked }
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn valid_date(value: &str) -> bool {
    let bytes = value.as_bytes();
    bytes.len() == 10 && bytes[4] == b'-' && bytes[7] == b'-'
        && bytes.iter().enumerate().all(|(i, byte)| i == 4 || i == 7 || byte.is_ascii_digit())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn fixture() -> Value {
        serde_json::from_str(include_str!("../../data/research/plasma_measurement_capability_fixture_v0_1.json")).expect("fixture parses")
    }

    #[test]
    fn all_fixture_dispositions_match() {
        let data = fixture();
        for case in data["cases"].as_array().expect("cases") {
            assert_eq!(audit_case(&case["receipt"], &data["policy"]).as_str(), case["expected_disposition"].as_str().expect("expected"), "{}", case["id"]);
        }
    }

    #[test]
    fn capability_must_resolve_the_decision_margin() {
        let data = fixture();
        let case = data["cases"].as_array().unwrap().iter().find(|row| row["id"] == "uncertainty-above-target").unwrap();
        assert_eq!(audit_case(&case["receipt"], &data["policy"]), Disposition::Blocked);
    }

    #[test]
    fn current_calibration_without_traceability_is_not_enough() {
        let data = fixture();
        let case = data["cases"].as_array().unwrap().iter().find(|row| row["id"] == "no-traceability").unwrap();
        assert_eq!(audit_case(&case["receipt"], &data["policy"]), Disposition::Blocked);
    }

    #[test]
    fn fixture_and_instrument_are_part_of_the_scope() {
        let data = fixture();
        let case = data["cases"].as_array().unwrap().iter().find(|row| row["id"] == "wrong-fixture").unwrap();
        assert_eq!(audit_case(&case["receipt"], &data["policy"]), Disposition::Blocked);
    }

    #[test]
    fn invalidation_event_blocks_old_receipt() {
        let data = fixture();
        let case = data["cases"].as_array().unwrap().iter().find(|row| row["id"] == "known-invalidation-event").unwrap();
        assert_eq!(audit_case(&case["receipt"], &data["policy"]), Disposition::Blocked);
    }

    #[test]
    fn uncertainty_exactly_at_declared_limit_passes() {
        let data = fixture();
        let case = data["cases"].as_array().unwrap().iter().find(|row| row["id"] == "uncertainty-exactly-at-target").unwrap();
        assert_eq!(audit_case(&case["receipt"], &data["policy"]), Disposition::Pass);
    }

    #[test]
    fn wrong_measurand_is_rejected_not_relabelled() {
        let data = fixture();
        let case = data["cases"].as_array().unwrap().iter().find(|row| row["id"] == "wrong-measurand").unwrap();
        assert_eq!(audit_case(&case["receipt"], &data["policy"]), Disposition::Rejected);
    }

    #[test]
    fn expired_capability_receipt_blocks() {
        let data = fixture();
        let case = data["cases"].as_array().unwrap().iter().find(|row| row["id"] == "calibration-valid-but-receipt-expired").unwrap();
        assert_eq!(audit_case(&case["receipt"], &data["policy"]), Disposition::Blocked);
    }
}
