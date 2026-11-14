//! Evidence-carrying qualification transfer audit for plasma-assisted bonding.
//! S0 checks a frozen contract; it does not estimate physical outcomes or authorize changes.
use anyhow::{bail, Context, Result};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs;
use std::path::Path;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Disposition { Pass, Blocked, Rejected }

impl Disposition {
    fn as_str(self) -> &'static str {
        match self { Self::Pass => "PASS", Self::Blocked => "BLOCKED", Self::Rejected => "REJECTED" }
    }
}

pub fn run(fixture_path: &Path, composition_fixture_path: &Path) -> Result<()> {
    let bytes = fs::read(fixture_path).with_context(|| format!("reading {}", fixture_path.display()))?;
    let fixture: Value = serde_json::from_slice(&bytes).context("parsing plasma-transfer fixture")?;
    if fixture.get("schema_version").and_then(Value::as_str) != Some("plasma-transfer-fixture-v0.2") {
        bail!("unsupported plasma-transfer fixture schema");
    }
    let cases = fixture.get("cases").and_then(Value::as_array).context("fixture must contain cases[]")?;
    let (mut pass, mut blocked, mut rejected) = (0usize, 0usize, 0usize);
    for case in cases {
        let id = case.get("id").and_then(Value::as_str).context("case id missing")?;
        let expected = case.get("expected_disposition").and_then(Value::as_str).context("expected disposition missing")?;
        let actual = audit_case_composed(case, &fixture, composition_fixture_path)?;
        if actual.as_str() != expected { bail!("case {id}: expected {expected}, got {}", actual.as_str()); }
        match actual { Disposition::Pass => pass += 1, Disposition::Blocked => blocked += 1, Disposition::Rejected => rejected += 1 }
    }
    println!("plasma-transfer-audit: {} composed cases; PASS={pass}, BLOCKED={blocked}, REJECTED={rejected}", cases.len());
    println!("Transfer PASS requires the legacy factor/interval gates and per-bridge method-qualified apparent-shear/design evidence.");
    println!("Age-retention and failure-mode interval gates remain legacy records; they are not yet fully typed endpoint compositions.");
    println!("S0 structural evidence audit only; no physical qualification is implied.");
    Ok(())
}

fn audit_case_composed(
    case: &Value,
    fixture: &Value,
    composition_fixture_path: &Path,
) -> Result<Disposition> {
    let legacy = audit_case(case, fixture);
    if legacy != Disposition::Pass {
        return Ok(legacy);
    }

    let Some(baseline) = case.get("baseline_profile").and_then(Value::as_object) else {
        return Ok(Disposition::Blocked);
    };
    let Some(case_id) = case.get("id").and_then(Value::as_str) else {
        return Ok(Disposition::Blocked);
    };
    let Some(scope_registry) = fixture.get("material_scope_registry").and_then(Value::as_object) else {
        return Ok(Disposition::Blocked);
    };
    let Some(substrate_grade) = baseline.get("substrate_grade").and_then(Value::as_str) else {
        return Ok(Disposition::Blocked);
    };
    let Some(scope) = scope_registry.get(substrate_grade).and_then(Value::as_object) else {
        return Ok(Disposition::Blocked);
    };
    let Some(material_pair_class) = scope.get("material_pair_class").and_then(Value::as_str) else {
        return Ok(Disposition::Blocked);
    };
    let Some(geometry) = scope.get("geometry").and_then(Value::as_str) else {
        return Ok(Disposition::Blocked);
    };
    let baseline_profile_sha256 = profile_digest(baseline)?;
    let required_factors = string_list(fixture.get("required_factors")).ok_or_else(|| anyhow::anyhow!("required_factors missing"))?;
    let candidate = case.get("candidate_profile").and_then(Value::as_object).ok_or_else(|| anyhow::anyhow!("candidate_profile missing after legacy pass"))?;
    let changed: Vec<String> = required_factors.iter().filter(|factor| baseline.get(*factor) != candidate.get(*factor)).cloned().collect();

    let mut seen_composition_ids = HashSet::new();
    if changed.is_empty() {
        let Some(composition_id) = case.get("baseline_composition_case_id").and_then(Value::as_str) else {
            return Ok(Disposition::Blocked);
        };
        if !seen_composition_ids.insert(composition_id.to_owned()) {
            return Ok(Disposition::Rejected);
        }
        let decision = crate::plasma_evidence_composition::validate_transfer_case(
            composition_fixture_path,
            composition_id,
            case_id,
            &format!("{case_id}::UNCHANGED_PROFILE"),
            &baseline_profile_sha256,
            &baseline_profile_sha256,
            material_pair_class,
            geometry,
            &[],
        )?;
        return Ok(map_composition_disposition(decision.disposition));
    }

    let bridges = case.get("bridges").and_then(Value::as_array).map(Vec::as_slice).unwrap_or(&[]);
    for bridge in bridges {
        let Some(dimensions) = bridge.get("dimensions").and_then(Value::as_object) else {
            return Ok(Disposition::Blocked);
        };
        let mut keys: Vec<String> = dimensions.keys().cloned().collect();
        keys.sort();
        let bridge_id = format!("{case_id}::{}", keys.join("+"));
        let Some(composition_id) = bridge.get("composition_case_id").and_then(Value::as_str) else {
            return Ok(Disposition::Blocked);
        };
        if !seen_composition_ids.insert(composition_id.to_owned()) {
            return Ok(Disposition::Rejected);
        }

        // A one-factor bridge is bound to a profile in which only that factor changed.
        // Multi-factor bridges are bound to the exact joint vector they purport to qualify.
        let mut bridge_profile = baseline.clone();
        for key in &keys {
            let Some(to) = dimensions.get(key).and_then(Value::as_object).and_then(|mapping| mapping.get("to")) else {
                return Ok(Disposition::Blocked);
            };
            bridge_profile.insert(key.clone(), to.clone());
        }
        let Some(candidate_grade) = bridge_profile.get("substrate_grade").and_then(Value::as_str) else {
            return Ok(Disposition::Blocked);
        };
        let Some(candidate_scope) = scope_registry.get(candidate_grade).and_then(Value::as_object) else {
            return Ok(Disposition::Blocked);
        };
        if candidate_scope.get("material_pair_class").and_then(Value::as_str) != Some(material_pair_class)
            || candidate_scope.get("geometry").and_then(Value::as_str) != Some(geometry)
        {
            return Ok(Disposition::Rejected);
        }
        let candidate_profile_sha256 = profile_digest(&bridge_profile)?;
        let decision = crate::plasma_evidence_composition::validate_transfer_case(
            composition_fixture_path,
            composition_id,
            case_id,
            &bridge_id,
            &baseline_profile_sha256,
            &candidate_profile_sha256,
            material_pair_class,
            geometry,
            &keys,
        )?;
        let mapped = map_composition_disposition(decision.disposition);
        if mapped != Disposition::Pass {
            return Ok(mapped);
        }
    }

    Ok(Disposition::Pass)
}

fn profile_digest(profile: &serde_json::Map<String, Value>) -> Result<String> {
    let bytes = serde_json::to_vec(&Value::Object(profile.clone()))
        .context("serializing exact transfer profile for digest")?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn map_composition_disposition(
    disposition: crate::plasma_evidence_composition::Disposition,
) -> Disposition {
    match disposition {
        crate::plasma_evidence_composition::Disposition::Pass => Disposition::Pass,
        crate::plasma_evidence_composition::Disposition::Blocked => Disposition::Blocked,
        crate::plasma_evidence_composition::Disposition::Rejected => Disposition::Rejected,
    }
}

fn audit_case(case: &Value, fixture: &Value) -> Disposition {
    let Some(baseline) = case.get("baseline_profile").and_then(Value::as_object) else { return Disposition::Blocked; };
    let Some(candidate) = case.get("candidate_profile").and_then(Value::as_object) else { return Disposition::Blocked; };
    let Some(required_factors) = string_list(fixture.get("required_factors")) else { return Disposition::Blocked; };
    let Some(required_endpoints) = string_list(fixture.get("required_endpoints")) else { return Disposition::Blocked; };
    let mut required_set = HashSet::new();
    let mut changed = Vec::<String>::new();
    for factor in &required_factors {
        if !required_set.insert(factor.as_str()) { return Disposition::Rejected; }
        let (Some(old), Some(new)) = (baseline.get(factor), candidate.get(factor)) else { return Disposition::Blocked; };
        if old.is_null() || new.is_null() || old.is_array() || old.is_object() || new.is_array() || new.is_object() { return Disposition::Blocked; }
        if old != new { changed.push(factor.clone()); }
    }
    let all_keys: HashSet<&String> = baseline.keys().chain(candidate.keys()).collect();
    for key in all_keys {
        if !required_set.contains(key.as_str()) && baseline.get(key) != candidate.get(key) { return Disposition::Blocked; }
    }
    let baseline_check = validate_baseline_qualification(case, baseline, &required_endpoints);
    if baseline_check != Disposition::Pass { return baseline_check; }
    let Some(scope) = fixture.get("qualification_scope").and_then(Value::as_object) else { return Disposition::Blocked; };
    for factor in &required_factors {
        let Some(factor_scope) = scope.get(factor) else { return Disposition::Blocked; };
        let Some(baseline_value) = baseline.get(factor) else { return Disposition::Blocked; };
        match scope_contains(baseline_value, factor_scope) {
            Some(true) => {}
            Some(false) => return Disposition::Rejected,
            None => return Disposition::Blocked,
        }
    }
    let Some(baseline_receipt) = case.get("baseline_qualification").and_then(Value::as_object) else { return Disposition::Blocked; };
    let Some(baseline_artifact) = baseline_receipt.get("artifact_sha256").and_then(Value::as_str) else { return Disposition::Blocked; };
    let mut seen_hashes = HashSet::<String>::new();
    if !seen_hashes.insert(baseline_artifact.to_ascii_lowercase()) { return Disposition::Rejected; }
    let bridges = case.get("bridges").and_then(Value::as_array).map(Vec::as_slice).unwrap_or(&[]);
    if changed.is_empty() { return if bridges.is_empty() { Disposition::Pass } else { Disposition::Rejected }; }
    for factor in &changed {
        let Some(factor_scope) = scope.get(factor) else { return Disposition::Blocked; };
        let Some(candidate_value) = candidate.get(factor) else { return Disposition::Blocked; };
        match scope_contains(candidate_value, factor_scope) {
            Some(true) => {}
            Some(false) => return Disposition::Rejected,
            None => return Disposition::Blocked,
        }
    }
    let Some(min_confidence) = fixture.get("minimum_confidence_level").and_then(Value::as_f64) else { return Disposition::Blocked; };
    if !(0.0..=1.0).contains(&min_confidence) { return Disposition::Rejected; }

    let mut seen_sets = HashSet::<String>::new();
    let mut individual = HashSet::<String>::new();
    let mut joint = false;
    let mut full_set = changed.clone();
    full_set.sort();

    for bridge in bridges {
        let Some(dimensions) = bridge.get("dimensions").and_then(Value::as_object) else { return Disposition::Blocked; };
        if dimensions.is_empty() { return Disposition::Rejected; }
        let mut keys: Vec<String> = dimensions.keys().cloned().collect();
        keys.sort();
        if keys.iter().any(|key| !changed.contains(key)) { return Disposition::Rejected; }
        let set_key = keys.join("|");
        if !seen_sets.insert(set_key.clone()) { return Disposition::Rejected; }
        for factor in &keys {
            let Some(mapping) = dimensions.get(factor).and_then(Value::as_object) else { return Disposition::Blocked; };
            let (Some(from), Some(to), Some(old), Some(new)) = (mapping.get("from"), mapping.get("to"), baseline.get(factor), candidate.get(factor)) else { return Disposition::Blocked; };
            if from != old || to != new { return Disposition::Rejected; }
        }
        let checked = validate_evidence(bridge, &required_endpoints, min_confidence, &mut seen_hashes);
        if checked != Disposition::Pass { return checked; }
        if keys.len() == 1 { individual.insert(keys[0].clone()); }
        else if keys == full_set { joint = true; }
        // Partial multi-factor bridges do not satisfy the full-vector interaction requirement.
    }
    if changed.iter().any(|factor| !individual.contains(factor)) { return Disposition::Blocked; }
    if changed.len() >= 2 && !joint { return Disposition::Blocked; }
    Disposition::Pass
}

fn validate_baseline_qualification(case: &Value, baseline: &serde_json::Map<String, Value>, required_endpoints: &[String]) -> Disposition {
    let Some(receipt) = case.get("baseline_qualification").and_then(Value::as_object) else { return Disposition::Blocked; };
    for key in ["qualification_id", "protocol_id", "independence_group"] {
        let Some(value) = receipt.get(key).and_then(Value::as_str) else { return Disposition::Blocked; };
        if value.trim().is_empty() { return Disposition::Blocked; }
    }
    let Some(profile_digest) = receipt.get("profile_sha256").and_then(Value::as_str) else { return Disposition::Blocked; };
    let Ok(profile_bytes) = serde_json::to_vec(&Value::Object(baseline.clone())) else { return Disposition::Blocked; };
    let actual_digest = format!("{:x}", Sha256::digest(profile_bytes));
    if profile_digest != actual_digest { return Disposition::Rejected; }
    let Some(artifact_digest) = receipt.get("artifact_sha256").and_then(Value::as_str) else { return Disposition::Blocked; };
    if !valid_sha256(artifact_digest) { return Disposition::Rejected; }
    let Some(specimen_ids) = string_list(receipt.get("specimen_ids")) else { return Disposition::Blocked; };
    if specimen_ids.is_empty() { return Disposition::Blocked; }
    if !unique_nonempty(&specimen_ids) { return Disposition::Rejected; }
    let Some(qualified_endpoints) = string_list(receipt.get("qualified_endpoints")) else { return Disposition::Blocked; };
    if !unique_nonempty(&qualified_endpoints) { return Disposition::Rejected; }
    if required_endpoints.iter().any(|endpoint| !qualified_endpoints.contains(endpoint)) { return Disposition::Blocked; }
    Disposition::Pass
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn scope_contains(candidate: &Value, scope: &Value) -> Option<bool> {
    if let Some(values) = scope.get("allowed_values").and_then(Value::as_array) {
        return Some(values.iter().any(|value| value == candidate));
    }
    let min = scope.get("min").and_then(Value::as_f64)?;
    let max = scope.get("max").and_then(Value::as_f64)?;
    let value = candidate.as_f64()?;
    if !min.is_finite() || !max.is_finite() || min > max || !value.is_finite() { return Some(false); }
    Some((min..=max).contains(&value))
}

fn string_list(value: Option<&Value>) -> Option<Vec<String>> {
    value?.as_array()?.iter().map(|item| item.as_str().map(str::to_owned)).collect()
}

fn validate_evidence(bridge: &Value, required_endpoints: &[String], min_confidence: f64, seen_hashes: &mut HashSet<String>) -> Disposition {
    let Some(evidence) = bridge.get("evidence").and_then(Value::as_object) else { return Disposition::Blocked; };
    let Some(protocol) = evidence.get("protocol_id").and_then(Value::as_str) else { return Disposition::Blocked; };
    if protocol.trim().is_empty() { return Disposition::Blocked; }
    let Some(hash) = evidence.get("artifact_sha256").and_then(Value::as_str) else { return Disposition::Blocked; };
    if hash.len() != 64 || !hash.bytes().all(|byte| byte.is_ascii_hexdigit()) { return Disposition::Rejected; }
    if !seen_hashes.insert(hash.to_ascii_lowercase()) { return Disposition::Rejected; }

    let (Some(base_ids), Some(candidate_ids), Some(base_groups), Some(candidate_groups)) = (
        string_list(evidence.get("baseline_specimen_ids")), string_list(evidence.get("candidate_specimen_ids")),
        string_list(evidence.get("baseline_independence_groups")), string_list(evidence.get("candidate_independence_groups"))
    ) else { return Disposition::Blocked; };
    if base_ids.is_empty() || candidate_ids.is_empty() || base_groups.is_empty() || candidate_groups.is_empty() { return Disposition::Blocked; }
    if !unique_nonempty(&base_ids) || !unique_nonempty(&candidate_ids) || !unique_nonempty(&base_groups) || !unique_nonempty(&candidate_groups)
        || intersects(&base_ids, &candidate_ids) || intersects(&base_groups, &candidate_groups) { return Disposition::Rejected; }

    let Some(intervals) = evidence.get("intervals").and_then(Value::as_array) else { return Disposition::Blocked; };
    let mut observed = HashSet::<String>::new();
    for interval in intervals {
        let Some(endpoint) = interval.get("endpoint").and_then(Value::as_str) else { return Disposition::Blocked; };
        if !observed.insert(endpoint.to_owned()) { return Disposition::Rejected; }
        let (Some(margin), Some(lower), Some(upper), Some(confidence)) = (
            interval.get("margin").and_then(Value::as_f64),
            interval.get("lower_abs_difference_bound").and_then(Value::as_f64),
            interval.get("upper_abs_difference_bound").and_then(Value::as_f64),
            interval.get("confidence_level").and_then(Value::as_f64)
        ) else { return Disposition::Blocked; };
        if !margin.is_finite() || !lower.is_finite() || !upper.is_finite() || !confidence.is_finite()
            || margin <= 0.0 || lower < 0.0 || upper < lower || !(0.0..=1.0).contains(&confidence) { return Disposition::Rejected; }
        if lower > margin { return Disposition::Rejected; }
        if upper > margin || confidence < min_confidence { return Disposition::Blocked; }
    }
    if required_endpoints.iter().any(|endpoint| !observed.contains(endpoint)) { return Disposition::Blocked; }
    Disposition::Pass
}

fn unique_nonempty(values: &[String]) -> bool {
    let mut seen = HashSet::new();
    values.iter().all(|value| !value.trim().is_empty() && seen.insert(value))
}

fn intersects(left: &[String], right: &[String]) -> bool {
    let right_set: HashSet<&str> = right.iter().map(String::as_str).collect();
    left.iter().any(|value| right_set.contains(value.as_str()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn fixture() -> Value {
        serde_json::from_str(include_str!("../../data/research/plasma_transfer_fixture_v0_2.json")).expect("fixture parses")
    }

    fn composition_fixture_path() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent().expect("xtask parent exists")
            .join("data/research/plasma_transfer_composition_fixture_v0_1.json")
    }

    fn composed(data: &Value, id: &str) -> Disposition {
        audit_case_composed(case(data, id), data, &composition_fixture_path())
            .expect("composed transfer fixture audits")
    }
    fn case<'a>(data: &'a Value, id: &str) -> &'a Value {
        data["cases"].as_array().expect("cases").iter().find(|row| row["id"].as_str() == Some(id)).expect("case exists")
    }

    #[test] fn all_fixture_dispositions_match() {
        let data = fixture();
        for row in data["cases"].as_array().expect("cases") {
            assert_eq!(audit_case_composed(row, &data, &composition_fixture_path()).expect("composed fixture audit").as_str(), row["expected_disposition"].as_str().expect("expected"));
        }
    }
    #[test] fn unchanged_profile_passes() {
        let data = fixture();
        assert_eq!(composed(&data, "unchanged-profile"), Disposition::Pass);
    }
    #[test] fn absent_single_factor_bridge_blocks() {
        let data = fixture();
        assert_eq!(audit_case(case(&data, "material-lot-no-bridge"), &data), Disposition::Blocked);
    }
    #[test] fn multiple_changes_need_exact_joint_evidence() {
        let data = fixture();
        assert_eq!(audit_case(case(&data, "supplier-and-head-without-joint"), &data), Disposition::Blocked);
    }
    #[test] fn three_factor_joint_bridge_passes() {
        let data = fixture();
        assert_eq!(audit_case(case(&data, "three-factor-exact-joint-bridge"), &data), Disposition::Pass);
    }
    #[test] fn out_of_scope_value_rejects() {
        let data = fixture();
        assert_eq!(audit_case(case(&data, "supplier-outside-qualified-scope"), &data), Disposition::Rejected);
    }
    #[test] fn crossing_interval_blocks() {
        let data = fixture();
        assert_eq!(audit_case(case(&data, "interval-crosses-equivalence-margin"), &data), Disposition::Blocked);
    }
    #[test] fn duplicate_artifact_rejects() {
        let data = fixture();
        assert_eq!(audit_case(case(&data, "duplicate-bridge-artifact"), &data), Disposition::Rejected);
    }
    #[test] fn baseline_outside_scope_rejects() {
        let data = fixture();
        assert_eq!(audit_case(case(&data, "baseline-outside-qualified-scope"), &data), Disposition::Rejected);
    }
    #[test] fn bridge_cannot_reuse_baseline_receipt_artifact() {
        let data = fixture();
        assert_eq!(audit_case(case(&data, "bridge-reuses-baseline-qualification-artifact"), &data), Disposition::Rejected);
    }
    #[test] fn baseline_profile_hash_mismatch_rejects() {
        let data = fixture();
        assert_eq!(audit_case(case(&data, "baseline-profile-hash-mismatch"), &data), Disposition::Rejected);
    }
    #[test] fn missing_baseline_qualification_blocks() {
        let data = fixture();
        assert_eq!(audit_case(case(&data, "missing-baseline-qualification-receipt"), &data), Disposition::Blocked);
    }

    #[test] fn interval_only_bridge_cannot_pass_without_composition() {
        let data = fixture();
        assert_eq!(composed(&data, "pass-interval-without-composition"), Disposition::Blocked);
    }

    #[test] fn composition_from_another_transfer_is_rejected() {
        let data = fixture();
        assert_eq!(composed(&data, "composition-case-from-another-transfer"), Disposition::Rejected);
    }

    #[test] fn composition_receipt_cannot_be_reused_across_bridges() {
        let data = fixture();
        assert_eq!(composed(&data, "same-composition-reused-across-bridges"), Disposition::Rejected);
    }

    #[test] fn joint_design_cannot_substitute_for_single_factor_bridge() {
        let data = fixture();
        assert_eq!(composed(&data, "joint-design-substituted-for-single-factor"), Disposition::Rejected);
    }

    #[test] fn unchanged_profile_requires_current_baseline_composition() {
        let data = fixture();
        assert_eq!(composed(&data, "unchanged-profile-without-baseline-composition"), Disposition::Blocked);
    }
}
