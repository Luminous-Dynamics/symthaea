use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;

const POLICY_BYTES: &[u8] =
    include_bytes!("../tests/fixtures/hardsec/mobile-policy-v1/pixel-8-pro-research-policy.json");
const PROFILE_BYTES: &[u8] = include_bytes!(
    "../tests/fixtures/hardsec/mobile-device-v1/pixel-8-pro-capability-fixture.json"
);

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn claim_value<'a>(profile: &'a Value, claim_path: &str) -> Option<&'a Value> {
    match claim_path {
        "boot_integrity.verified_boot_state" => {
            Some(&profile["boot_integrity"]["verified_boot_state"])
        }
        "boot_integrity.attestation_security_level" => {
            Some(&profile["boot_integrity"]["attestation_security_level"])
        }
        "boot_integrity.attestation_challenge_freshness" => {
            Some(&profile["boot_integrity"]["attestation_challenge_freshness"])
        }
        "boot_integrity.certificate_chain_status" => {
            Some(&profile["boot_integrity"]["certificate_chain_status"])
        }
        "hardware_protections.memory_tagging.runtime_state" => {
            Some(&profile["hardware_protections"]["memory_tagging"]["runtime_state"])
        }
        "pmu_acquisition.observer_effect_status" => {
            Some(&profile["pmu_acquisition"]["observer_effect_status"])
        }
        _ => None,
    }
}

fn evidence_ids_for_claim<'a>(profile: &'a Value, claim_path: &str) -> Option<&'a Value> {
    match claim_path {
        "boot_integrity.verified_boot_state"
        | "boot_integrity.attestation_security_level"
        | "boot_integrity.attestation_challenge_freshness"
        | "boot_integrity.certificate_chain_status" => {
            Some(&profile["boot_integrity"]["evidence_ids"])
        }
        "hardware_protections.memory_tagging.runtime_state" => {
            Some(&profile["hardware_protections"]["memory_tagging"]["evidence_ids"])
        }
        "pmu_acquisition.observer_effect_status" => {
            Some(&profile["pmu_acquisition"]["evidence_ids"])
        }
        _ => None,
    }
}

fn validate_evidence_ledger(profile: &Value) -> Result<BTreeMap<String, String>, String> {
    let entries = profile["evidence"]
        .as_array()
        .ok_or_else(|| "profile evidence must be an array".to_owned())?;
    let mut ledger = BTreeMap::new();

    for entry in entries {
        let id = entry["evidence_id"]
            .as_str()
            .filter(|id| !id.is_empty())
            .ok_or_else(|| "evidence entry has no non-empty evidence_id".to_owned())?;
        let source = entry["source_type"]
            .as_str()
            .filter(|source| !source.is_empty())
            .ok_or_else(|| format!("evidence {id} has no source_type"))?;

        if ledger.insert(id.to_owned(), source.to_owned()).is_some() {
            return Err(format!("duplicate evidence_id: {id}"));
        }
    }
    Ok(ledger)
}

fn validate_policy(policy: &Value, profile: &Value) -> Result<(), String> {
    if policy["schema_version"] != "hardsec-mobile-policy-v1" {
        return Err("unsupported policy schema_version".to_owned());
    }
    if profile["schema_version"] != "hardsec-mobile-device-profile-v1" {
        return Err("unsupported profile schema_version".to_owned());
    }
    if policy["target"]["platform"] != profile["device"]["platform"] {
        return Err("policy target platform does not match profile".to_owned());
    }
    if policy["target"]["product_family"] != profile["device"]["product_family"] {
        return Err("policy target product_family does not match profile".to_owned());
    }

    let controls = policy["controls"]
        .as_array()
        .filter(|controls| !controls.is_empty())
        .ok_or_else(|| "policy controls must be a non-empty array".to_owned())?;
    let mut ids = BTreeSet::new();

    for control in controls {
        let id = control["control_id"]
            .as_str()
            .filter(|id| !id.is_empty())
            .ok_or_else(|| "control is missing a non-empty control_id".to_owned())?;
        if !ids.insert(id.to_owned()) {
            return Err(format!("duplicate control_id: {id}"));
        }

        let claim_path = control["claim_path"]
            .as_str()
            .ok_or_else(|| format!("control {id} has no claim_path"))?;
        if claim_value(profile, claim_path).is_none()
            || evidence_ids_for_claim(profile, claim_path).is_none()
        {
            return Err(format!("control {id} uses an unsupported claim_path"));
        }

        if control["on_unknown"] != "abstain" {
            return Err(format!("control {id} must abstain on unknown state"));
        }
        if control["on_unsupported"] != "incompatible" {
            return Err(format!(
                "control {id} must mark unsupported targets incompatible"
            ));
        }

        if control["desired_values"]
            .as_array()
            .is_none_or(|values| values.is_empty() || values.iter().any(|v| !v.is_string()))
        {
            return Err(format!("control {id} has invalid desired_values"));
        }
        if control["accepted_evidence_source_types"]
            .as_array()
            .is_none_or(|sources| {
                sources.is_empty()
                    || sources.iter().any(|value| {
                        !matches!(
                            value.as_str(),
                            Some(
                                "hardware_attestation_verified"
                                    | "privileged_device_tool"
                                    | "physical_lab_measurement"
                            )
                        )
                    })
            })
        {
            return Err(format!(
                "control {id} has invalid accepted_evidence_source_types"
            ));
        }
        if control["verification_action"]
            .as_str()
            .is_none_or(str::is_empty)
        {
            return Err(format!("control {id} has no verification_action"));
        }
    }
    Ok(())
}

fn evaluate_control(
    control: &Value,
    profile: &Value,
    ledger: &BTreeMap<String, String>,
) -> Result<Value, String> {
    let id = control["control_id"].as_str().unwrap_or("invalid-control");
    let claim_path = control["claim_path"].as_str().unwrap_or("");
    let observed = claim_value(profile, claim_path)
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let desired = control["desired_values"].as_array().unwrap();
    let accepted = control["accepted_evidence_source_types"]
        .as_array()
        .unwrap();
    let ids_value = evidence_ids_for_claim(profile, claim_path).unwrap();
    let ids = ids_value
        .as_array()
        .ok_or_else(|| format!("{id} evidence_ids must be an array"))?;

    let mut linked_sources = Vec::new();
    for id_value in ids {
        let evidence_id = id_value
            .as_str()
            .filter(|value| !value.is_empty())
            .ok_or_else(|| format!("{id} has an invalid evidence ID"))?;
        let source = ledger
            .get(evidence_id)
            .ok_or_else(|| format!("{id} references unknown evidence ID {evidence_id}"))?;
        linked_sources.push(json!({
            "evidence_id": evidence_id,
            "source_type": source
        }));
    }

    let desired_match = desired.iter().any(|value| value.as_str() == Some(observed));
    let evidence_domain = profile["evidence_domain"].as_str().unwrap_or("unknown");
    let accepted_linked_evidence = evidence_domain != "synthetic_fixture"
        && linked_sources.iter().any(|source| {
            accepted
                .iter()
                .any(|allowed| allowed.as_str() == source["source_type"].as_str())
        });

    let (status, action, reason) = if matches!(
        observed,
        "unknown" | "not_measured" | "not_available" | "null"
    ) {
        (
            "evidence_incomplete",
            "abstain",
            "the required state has not been measured",
        )
    } else if matches!(observed, "unsupported" | "not_applicable") {
        (
            "incompatible",
            "reject_policy_for_target",
            "the target does not provide the required capability",
        )
    } else if desired_match && !accepted_linked_evidence {
        (
            "unqualified_evidence",
            "abstain",
            "the observed value lacks claim-scoped evidence from an accepted source",
        )
    } else if desired_match {
        (
            "independent_verifier_required",
            "abstain",
            "a linked evidence source label is not an authenticated verification receipt; an independent verifier must qualify the exact claim, evidence, and profile digest",
        )
    } else {
        (
            "policy_violated",
            "deny_or_reconfigure",
            "the observed claim does not match any desired value",
        )
    };

    let linked_evidence_ids = linked_sources
        .iter()
        .filter_map(|source| source["evidence_id"].as_str())
        .collect::<Vec<_>>();

    Ok(json!({
        "obligation_id": format!("verify:{id}"),
        "control_id": id,
        "claim_path": claim_path,
        "enforcement_layer": control["enforcement_layer"],
        "desired_values": desired,
        "observed_value": observed,
        "linked_evidence": linked_sources,
        "accepted_evidence_source_types": accepted,
        "verification_action": control["verification_action"],
        "status": status,
        "action": action,
        "reason": reason,
        "evidence_ids": linked_evidence_ids
    }))
}

fn compile_policy(
    policy: &Value,
    profile: &Value,
    policy_bytes: &[u8],
    profile_bytes: &[u8],
) -> Result<Value, String> {
    validate_policy(policy, profile)?;
    let ledger = validate_evidence_ledger(profile)?;
    let controls = policy["controls"].as_array().unwrap();
    let obligations = controls
        .iter()
        .map(|control| evaluate_control(control, profile, &ledger))
        .collect::<Result<Vec<_>, _>>()?;

    let incomplete_count = obligations
        .iter()
        .filter(|obligation| obligation["status"] == "evidence_incomplete")
        .count();

    Ok(json!({
        "schema_version": "hardsec-mobile-policy-receipt-v1",
        "result": "obligations_compiled_verification_not_performed",
        "claim_scope": "deterministic policy obligation evaluation only; not proof of device security",
        "policy_id": policy["policy_id"],
        "policy_sha256": sha256_hex(policy_bytes),
        "profile_id": profile["profile_id"],
        "profile_sha256": sha256_hex(profile_bytes),
        "device_class": {
            "platform": profile["device"]["platform"],
            "product_family": profile["device"]["product_family"],
            "soc_family": profile["device"]["soc_family"]
        },
        "obligation_count": obligations.len(),
        "satisfied_count": 0,
        "outstanding_count": obligations.len(),
        "evidence_incomplete_count": incomplete_count,
        "obligations": obligations,
        "claim_ceiling": {
            "device_security_proved": false,
            "runtime_integrity_verified": false,
            "physical_attack_resistance_established": false,
            "limitations": [
                "The compiler evaluates declared profile claims; it does not configure the operating system.",
                "This compiler never marks a security control satisfied; it emits verification obligations only.",
                "Evidence source-type labels are untrusted metadata, not authenticated proof of their own authenticity.",
                "Independent verifiers must authenticate source, freshness, scope, quality and the exact profile/evidence digests.",
                "The supplied Pixel 8 Pro input is a synthetic capability fixture, not a physical-device capture."
            ]
        }
    }))
}

fn main() {
    let mut policy_bytes = POLICY_BYTES.to_vec();
    let mut profile_bytes = PROFILE_BYTES.to_vec();
    let args = env::args().skip(1).collect::<Vec<_>>();
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--policy" if index + 1 < args.len() => {
                policy_bytes = fs::read(&args[index + 1]).expect("unable to read policy file");
                index += 2;
            }
            "--profile" if index + 1 < args.len() => {
                profile_bytes = fs::read(&args[index + 1]).expect("unable to read profile file");
                index += 2;
            }
            _ => {
                eprintln!("usage: hardsec_mobile_policy_compile [--policy PATH] [--profile PATH]");
                std::process::exit(2);
            }
        }
    }

    let policy: Value = match serde_json::from_slice(&policy_bytes) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("invalid policy JSON: {error}");
            std::process::exit(2);
        }
    };
    let profile: Value = match serde_json::from_slice(&profile_bytes) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("invalid profile JSON: {error}");
            std::process::exit(2);
        }
    };

    match compile_policy(&policy, &profile, &policy_bytes, &profile_bytes) {
        Ok(receipt) => println!("{}", serde_json::to_string_pretty(&receipt).unwrap()),
        Err(error) => {
            eprintln!("policy compilation failed: {error}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_policy() -> Value {
        serde_json::from_slice(POLICY_BYTES).unwrap()
    }

    fn fixture_profile() -> Value {
        serde_json::from_slice(PROFILE_BYTES).unwrap()
    }

    #[test]
    fn synthetic_pixel_fixture_abstains_on_unmeasured_controls() {
        let policy_bytes = POLICY_BYTES;
        let profile_bytes = PROFILE_BYTES;
        let receipt = compile_policy(
            &fixture_policy(),
            &fixture_profile(),
            policy_bytes,
            profile_bytes,
        )
        .unwrap();

        assert_eq!(
            receipt["result"],
            "obligations_compiled_verification_not_performed"
        );
        assert_eq!(receipt["satisfied_count"], 0);
        assert_eq!(receipt["obligation_count"], 6);
        assert_eq!(receipt["evidence_incomplete_count"], 6);
        assert_eq!(receipt["claim_ceiling"]["device_security_proved"], false);
    }

    #[test]
    fn vendor_documentation_cannot_qualify_runtime_mte_enablement() {
        let mut profile = fixture_profile();
        profile["hardware_protections"]["memory_tagging"]["runtime_state"] =
            Value::String("enabled_verified".to_owned());

        let policy = fixture_policy();
        let ledger = validate_evidence_ledger(&profile).unwrap();
        let mte = policy["controls"]
            .as_array()
            .unwrap()
            .iter()
            .find(|control| control["control_id"] == "memory.mte-runtime")
            .unwrap();
        let receipt = evaluate_control(mte, &profile, &ledger).unwrap();

        assert_eq!(receipt["status"], "unqualified_evidence");
        assert_eq!(receipt["action"], "abstain");
    }

    #[test]
    fn accepted_source_label_still_requires_an_independent_verifier() {
        let mut profile = fixture_profile();
        profile["evidence_domain"] = Value::String("physical_device_lab".to_owned());
        profile["hardware_protections"]["memory_tagging"]["runtime_state"] =
            Value::String("enabled_verified".to_owned());
        profile["evidence"].as_array_mut().unwrap().push(json!({
            "evidence_id": "mte-runtime-lab-measurement",
            "source_type": "physical_lab_measurement"
        }));
        profile["hardware_protections"]["memory_tagging"]["evidence_ids"] =
            json!(["mte-runtime-lab-measurement"]);

        let policy = fixture_policy();
        let ledger = validate_evidence_ledger(&profile).unwrap();
        let mte = policy["controls"]
            .as_array()
            .unwrap()
            .iter()
            .find(|control| control["control_id"] == "memory.mte-runtime")
            .unwrap();
        let receipt = evaluate_control(mte, &profile, &ledger).unwrap();

        assert_eq!(receipt["status"], "independent_verifier_required");
        assert_eq!(receipt["action"], "abstain");
    }

    #[test]
    fn mismatched_profile_target_is_rejected() {
        let policy = fixture_policy();
        let mut profile = fixture_profile();
        profile["device"]["product_family"] = Value::String("Other device".to_owned());
        assert!(validate_policy(&policy, &profile).is_err());
    }

    #[test]
    fn duplicate_evidence_ids_are_rejected() {
        let mut profile = fixture_profile();
        let duplicate = profile["evidence"][0].clone();
        profile["evidence"].as_array_mut().unwrap().push(duplicate);
        assert!(validate_evidence_ledger(&profile).is_err());
    }

    #[test]
    fn unknown_claim_path_is_rejected_instead_of_dynamically_resolved() {
        let mut policy = fixture_policy();
        policy["controls"][0]["claim_path"] = Value::String("device.serial_number".to_owned());
        assert!(validate_policy(&policy, &fixture_profile()).is_err());
    }
}
