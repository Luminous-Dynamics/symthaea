use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::env;
use std::fs;

const PROFILE_BYTES: &[u8] = include_bytes!(
    "../tests/fixtures/hardsec/mobile-device-v1/pixel-8-pro-capability-fixture.json"
);

const FORBIDDEN_KEYS_NORMALIZED: &[&str] = &[
    "imei",
    "meid",
    "imsi",
    "iccid",
    "serial",
    "serialnumber",
    "hardwareuniqueid",
    "uniqueid",
    "rawattestationcertificate",
    "attestationcertificateder",
    "phonenumber",
    "subscriberid",
    "latitude",
    "longitude",
    "preciselocation",
    "androidid",
    "deviceid",
    "deviceidentifier",
    "macaddress",
    "wifimacaddress",
    "bluetoothaddress",
    "bssid",
    "ssid",
    "ipaddress",
];

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn find_forbidden_key(value: &Value, path: &str) -> Option<String> {
    match value {
        Value::Object(object) => {
            for (key, child) in object {
                let normalized: String = key
                    .to_ascii_lowercase()
                    .chars()
                    .filter(|character| character.is_ascii_alphanumeric())
                    .collect();
                if FORBIDDEN_KEYS_NORMALIZED.contains(&normalized.as_str()) {
                    return Some(format!("{path}.{key}"));
                }
                if let Some(found) = find_forbidden_key(child, &format!("{path}.{key}")) {
                    return Some(found);
                }
            }
            None
        }
        Value::Array(values) => values
            .iter()
            .enumerate()
            .find_map(|(index, child)| find_forbidden_key(child, &format!("{path}[{index}]"))),
        _ => None,
    }
}

fn evidence_sources(profile: &Value) -> Vec<&str> {
    profile["evidence"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|item| item["source_type"].as_str())
        .collect()
}

fn evidence_sources_for_ids<'a>(profile: &'a Value, evidence_ids: &Value) -> Vec<&'a str> {
    let Some(ids) = evidence_ids.as_array() else {
        return Vec::new();
    };
    let Some(evidence) = profile["evidence"].as_array() else {
        return Vec::new();
    };

    ids.iter()
        .filter_map(Value::as_str)
        .filter_map(|id| evidence.iter().find(|item| item["evidence_id"] == id))
        .filter_map(|item| item["source_type"].as_str())
        .collect()
}

fn has_linked_evidence_source(
    profile: &Value,
    evidence_ids: &Value,
    allowed_sources: &[&str],
) -> bool {
    evidence_sources_for_ids(profile, evidence_ids)
        .iter()
        .any(|source| allowed_sources.contains(source))
}

fn validate_evidence_links(profile: &Value) -> Result<(), String> {
    let evidence = profile["evidence"]
        .as_array()
        .ok_or_else(|| "evidence ledger must be an array".to_owned())?;
    let mut known_ids = BTreeSet::new();
    for item in evidence {
        let id = item["evidence_id"]
            .as_str()
            .filter(|value| !value.is_empty())
            .ok_or_else(|| "evidence item is missing a non-empty evidence_id".to_owned())?;
        if !known_ids.insert(id.to_owned()) {
            return Err(format!("duplicate evidence_id in evidence ledger: {id}"));
        }
    }

    let linked_sections = [
        ("boot_integrity", &profile["boot_integrity"]["evidence_ids"]),
        ("pmu_acquisition", &profile["pmu_acquisition"]["evidence_ids"]),
        (
            "memory_tagging",
            &profile["hardware_protections"]["memory_tagging"]["evidence_ids"],
        ),
        (
            "pointer_authentication",
            &profile["hardware_protections"]["pointer_authentication"]["evidence_ids"],
        ),
        (
            "branch_target_protection",
            &profile["hardware_protections"]["branch_target_protection"]["evidence_ids"],
        ),
        (
            "secure_element",
            &profile["hardware_protections"]["secure_element"]["evidence_ids"],
        ),
        ("tee", &profile["hardware_protections"]["tee"]["evidence_ids"]),
    ];

    for (section, value) in linked_sections {
        let ids = value
            .as_array()
            .ok_or_else(|| format!("{section}.evidence_ids must be an array"))?;
        let mut section_ids = BTreeSet::new();
        for id_value in ids {
            let id = id_value
                .as_str()
                .filter(|value| !value.is_empty())
                .ok_or_else(|| format!("{section}.evidence_ids contains an invalid ID"))?;
            if !section_ids.insert(id.to_owned()) {
                return Err(format!("{section}.evidence_ids contains duplicate ID: {id}"));
            }
            if !known_ids.contains(id) {
                return Err(format!("{section}.evidence_ids references unknown evidence ID: {id}"));
            }
        }
    }
    Ok(())
}

fn validate_protection_claim(profile: &Value, protection_name: &str) -> Result<(), String> {
    let protection = &profile["hardware_protections"][protection_name];
    let evidence_ids = &protection["evidence_ids"];
    let support = protection["hardware_support"].as_str().unwrap_or("unknown");
    let runtime = protection["runtime_state"].as_str().unwrap_or("unknown");

    match support {
        "vendor_documented"
            if !has_linked_evidence_source(profile, evidence_ids, &["vendor_documentation"]) =>
        {
            return Err(format!(
                "{protection_name} vendor-documented capability requires linked vendor_documentation evidence"
            ));
        }
        "attested"
            if !has_linked_evidence_source(
                profile,
                evidence_ids,
                &["hardware_attestation_verified"],
            ) =>
        {
            return Err(format!(
                "{protection_name} attested capability requires linked hardware_attestation_verified evidence"
            ));
        }
        "on_device_reported" | "unsupported"
            if !has_linked_evidence_source(
                profile,
                evidence_ids,
                &[
                    "on_device_property",
                    "privileged_device_tool",
                    "physical_lab_measurement",
                ],
            ) =>
        {
            return Err(format!(
                "{protection_name} on-device capability result requires linked device evidence"
            ));
        }
        _ => {}
    }

    if matches!(runtime, "enabled_verified" | "disabled_verified")
        && !has_linked_evidence_source(
            profile,
            evidence_ids,
            &[
                "hardware_attestation_verified",
                "on_device_property",
                "privileged_device_tool",
                "physical_lab_measurement",
            ],
        )
    {
        return Err(format!(
            "{protection_name} verified runtime state cannot be inferred from vendor documentation alone; linked runtime evidence is required"
        ));
    }
    Ok(())
}

fn validate_profile(profile: &Value) -> Result<(), String> {
    if profile["schema_version"] != "hardsec-mobile-device-profile-v1" {
        return Err("unsupported mobile profile schema_version".to_owned());
    }

    if let Some(path) = find_forbidden_key(profile, "$") {
        return Err(format!(
            "privacy-sensitive field is forbidden in profile: {path}"
        ));
    }

    let sources = evidence_sources(profile);
    if sources.is_empty() {
        return Err("profile must have at least one evidence source".to_owned());
    }
    validate_evidence_links(profile)?;

    let boot = &profile["boot_integrity"];
    let boot_attestation = has_linked_evidence_source(
        profile,
        &boot["evidence_ids"],
        &["hardware_attestation_verified"],
    );
    if boot["reported_bootloader_lock_state"] != "unknown"
        && boot["reported_bootloader_lock_state"] != "not_measured"
        && !has_linked_evidence_source(
            profile,
            &boot["evidence_ids"],
            &["on_device_property", "privileged_device_tool"],
        )
    {
        return Err("reported bootloader state requires linked on-device property evidence".to_owned());
    }
    if boot["reported_verified_boot_state"] != "unknown"
        && boot["reported_verified_boot_state"] != "not_measured"
        && !has_linked_evidence_source(
            profile,
            &boot["evidence_ids"],
            &["on_device_property", "privileged_device_tool"],
        )
    {
        return Err("reported Verified Boot state requires linked on-device property evidence".to_owned());
    }
    if (boot["bootloader_locked"] == "locked"
        || boot["bootloader_locked"] == "unlocked"
        || boot["verified_boot_state"] == "verified"
        || boot["verified_boot_state"] == "self_signed"
        || boot["verified_boot_state"] == "unverified"
        || boot["verified_boot_state"] == "failed"
        || boot["attestation_security_level"] == "software"
        || boot["attestation_security_level"] == "trusted_environment"
        || boot["attestation_security_level"] == "strongbox"
        || boot["attestation_security_level"] == "secure_enclave"
        || boot["patch_level_binding_status"] == "hardware_bound_verified")
        && !boot_attestation
    {
        return Err("cryptographic boot or attestation claim requires linked hardware_attestation_verified evidence".to_owned());
    }

    if boot["patch_level_binding_status"] == "software_reported"
        && !has_linked_evidence_source(
            profile,
            &boot["evidence_ids"],
            &["on_device_property", "privileged_device_tool"],
        )
    {
        return Err("software-reported patch level requires linked on-device property evidence".to_owned());
    }

    if profile["pmu_acquisition"]["available_events"]
        .as_array()
        .is_some_and(|events| !events.is_empty())
        && !has_linked_evidence_source(
            profile,
            &profile["pmu_acquisition"]["evidence_ids"],
            &["on_device_property", "privileged_device_tool", "physical_lab_measurement"],
        )
    {
        return Err("PMU event inventory requires linked on-device event-list evidence".to_owned());
    }

    for name in [
        "memory_tagging",
        "pointer_authentication",
        "branch_target_protection",
        "secure_element",
        "tee",
    ] {
        validate_protection_claim(profile, name)?;
    }

    let is_synthetic = profile["evidence_domain"] == "synthetic_fixture";
    if is_synthetic {
        if profile["claim_ceiling"]["runtime_integrity_verified"] != false
            || profile["claim_ceiling"]["pmu_measurement_qualified"] != false
            || profile["claim_ceiling"]["physical_attack_resistance_established"] != false
        {
            return Err(
                "synthetic fixtures cannot claim runtime, PMU, or physical qualification"
                    .to_owned(),
            );
        }
        if profile["boot_integrity"]["verified_boot_state"] != "not_measured"
            || profile["boot_integrity"]["attestation_security_level"] != "not_measured"
            || profile["pmu_acquisition"]["access_status"] != "not_measured"
        {
            return Err(
                "synthetic fixture cannot imply device boot, attestation, or PMU measurements"
                    .to_owned(),
            );
        }
    }

    if profile["boot_integrity"]["certificate_chain_status"] == "trusted"
        && !has_linked_evidence_source(
            profile,
            &profile["boot_integrity"]["evidence_ids"],
            &["hardware_attestation_verified"],
        )
    {
        return Err(
            "trusted attestation chain requires a hardware_attestation_verified evidence source"
                .to_owned(),
        );
    }

    if profile["boot_integrity"]["attestation_challenge_freshness"] == "fresh_verified"
        && !has_linked_evidence_source(
            profile,
            &profile["boot_integrity"]["evidence_ids"],
            &["hardware_attestation_verified"],
        )
    {
        return Err(
            "fresh attestation challenge requires a hardware_attestation_verified evidence source"
                .to_owned(),
        );
    }

    if profile["pmu_acquisition"]["read_overhead_status"] == "qualified"
        && !has_linked_evidence_source(
            profile,
            &profile["pmu_acquisition"]["evidence_ids"],
            &["physical_lab_measurement"],
        )
    {
        return Err(
            "qualified PMU read overhead requires physical_lab_measurement evidence".to_owned(),
        );
    }

    if profile["pmu_acquisition"]["observer_effect_status"] == "within_budget"
        && !has_linked_evidence_source(
            profile,
            &profile["pmu_acquisition"]["evidence_ids"],
            &["physical_lab_measurement"],
        )
    {
        return Err(
            "within-budget observer effect requires physical_lab_measurement evidence".to_owned(),
        );
    }

    if profile["pmu_acquisition"]["access_status"] == "permitted"
        && !has_linked_evidence_source(
            profile,
            &profile["pmu_acquisition"]["evidence_ids"],
            &["privileged_device_tool", "physical_lab_measurement"],
        )
    {
        return Err(
            "PMU access claim requires privileged device-tool or physical measurement evidence"
                .to_owned(),
        );
    }

    Ok(())
}

fn report(profile: &Value, profile_bytes: &[u8]) -> Value {
    json!({
        "schema_version": "hardsec-mobile-device-profile-audit-v1",
        "result": "profile_shape_and_privacy_audit_passed",
        "claim_scope": "profile_shape_privacy_and_evidence_source_contract_only",
        "profile_id": profile["profile_id"],
        "profile_sha256": sha256_hex(profile_bytes),
        "evidence_domain": profile["evidence_domain"],
        "device_class": {
            "platform": profile["device"]["platform"],
            "product_family": profile["device"]["product_family"],
            "soc_family": profile["device"]["soc_family"],
            "isa_family": profile["device"]["isa_family"]
        },
        "capability_vs_runtime": {
            "mte_hardware_support": profile["hardware_protections"]["memory_tagging"]["hardware_support"],
            "mte_runtime_state": profile["hardware_protections"]["memory_tagging"]["runtime_state"],
            "secure_element_hardware_support": profile["hardware_protections"]["secure_element"]["hardware_support"],
            "secure_element_runtime_state": profile["hardware_protections"]["secure_element"]["runtime_state"],
            "boot_state": profile["boot_integrity"]["verified_boot_state"],
            "reported_bootloader_lock_state": profile["boot_integrity"]["reported_bootloader_lock_state"],
            "reported_verified_boot_state": profile["boot_integrity"]["reported_verified_boot_state"],
            "attestation_security_level": profile["boot_integrity"]["attestation_security_level"]
        },
        "measurement": {
            "pmu_access_status": profile["pmu_acquisition"]["access_status"],
            "pmu_event_count": profile["pmu_acquisition"]["available_events"].as_array().map_or(0, Vec::len),
            "observer_effect_status": profile["pmu_acquisition"]["observer_effect_status"],
            "thermal_status": profile["environment"]["thermal_status"]
        },
        "privacy": {
            "forbidden_identifier_fields_found": false,
            "raw_attestation_material_exported": false,
            "stable_device_identifier_exported": false
        },
        "qualification": {
            "profile_contract_valid": true,
            "runtime_integrity_verified": profile["claim_ceiling"]["runtime_integrity_verified"],
            "pmu_measurement_qualified": profile["claim_ceiling"]["pmu_measurement_qualified"],
            "physical_attack_resistance_established": profile["claim_ceiling"]["physical_attack_resistance_established"],
            "evidence_sources": profile["evidence"].as_array().map_or(0, Vec::len),
            "device_was_queried": profile["evidence_domain"] == "physical_device_lab"
                || profile["evidence_domain"] == "physical_device_capture"
        },
        "limitations": profile["claim_ceiling"]["limitations"]
    })
}

fn load_profile_bytes() -> Result<Vec<u8>, String> {
    let mut arguments = env::args().skip(1);
    match arguments.next() {
        None => Ok(PROFILE_BYTES.to_vec()),
        Some(flag) if flag == "--profile" => {
            let path = arguments
                .next()
                .ok_or_else(|| "--profile requires a file path".to_owned())?;
            if arguments.next().is_some() {
                return Err("unexpected extra command-line argument".to_owned());
            }
            fs::read(path).map_err(|_| "profile file could not be read".to_owned())
        }
        Some(_) => Err("supported argument: --profile <path>".to_owned()),
    }
}

fn main() {
    let profile_bytes = match load_profile_bytes() {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("HARDSEC mobile profile audit failed: {error}");
            std::process::exit(2);
        }
    };
    let profile: Value = match serde_json::from_slice(&profile_bytes) {
        Ok(profile) => profile,
        Err(_) => {
            eprintln!("HARDSEC mobile profile audit failed: invalid JSON profile");
            std::process::exit(2);
        }
    };
    if let Err(error) = validate_profile(&profile) {
        eprintln!("HARDSEC mobile profile audit failed: {error}");
        std::process::exit(2);
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&report(&profile, &profile_bytes))
            .expect("serialize mobile profile audit receipt")
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Value {
        serde_json::from_slice(PROFILE_BYTES).unwrap()
    }

    #[test]
    fn synthetic_pixel_profile_does_not_promote_documented_support_to_runtime_state() {
        let profile = fixture();
        validate_profile(&profile).unwrap();
        assert_eq!(profile["device"]["product_family"], "Google Pixel 8 Pro");
        assert_eq!(
            profile["hardware_protections"]["memory_tagging"]["hardware_support"],
            "vendor_documented"
        );
        assert_eq!(
            profile["hardware_protections"]["memory_tagging"]["runtime_state"],
            "not_measured"
        );
        assert_eq!(
            profile["boot_integrity"]["verified_boot_state"],
            "not_measured"
        );
        assert_eq!(profile["pmu_acquisition"]["access_status"], "not_measured");
    }

    #[test]
    fn privacy_sensitive_identifiers_are_rejected_recursively() {
        let mut profile = fixture();
        profile["device"]["serial_number"] = json!("synthetic-serial");
        let error = validate_profile(&profile).unwrap_err();
        assert!(error.contains("privacy-sensitive field"));
    }

    #[test]
    fn vendor_documentation_cannot_claim_android_mte_is_runtime_enabled() {
        let mut profile = fixture();
        profile["hardware_protections"]["memory_tagging"]["runtime_state"] =
            json!("enabled_verified");
        let error = validate_profile(&profile).unwrap_err();
        assert!(error.contains("cannot be inferred from vendor documentation"));
    }

    #[test]
    fn stale_or_unverified_attestation_cannot_be_silently_promoted_to_trusted() {
        let mut profile = fixture();
        profile["boot_integrity"]["certificate_chain_status"] = json!("trusted");
        let error = validate_profile(&profile).unwrap_err();
        assert!(error.contains("trusted attestation chain requires"));
    }

    #[test]
    fn pmu_measurement_qualification_requires_physical_measurement_evidence() {
        let mut profile = fixture();
        profile["pmu_acquisition"]["observer_effect_status"] = json!("within_budget");
        let error = validate_profile(&profile).unwrap_err();
        assert!(error.contains("within-budget observer effect requires"));
    }

    #[test]
    fn unrelated_runtime_evidence_cannot_qualify_mte_enablement() {
        let mut profile = fixture();
        profile["hardware_protections"]["memory_tagging"]["runtime_state"] =
            json!("enabled_verified");
        profile["evidence"].as_array_mut().unwrap().push(json!({
            "evidence_id": "unrelated-on-device-property",
            "source_type": "on_device_property",
            "source_uri": "https://example.invalid/unrelated",
            "input_sha256": "d".repeat(64),
            "parser_sha256": "b".repeat(64),
            "captured_at": null,
            "claim_scope": "unrelated fixture evidence"
        }));
        let error = validate_profile(&profile).unwrap_err();
        assert!(error.contains("cannot be inferred from vendor documentation"));
    }

    #[test]
    fn unrelated_lab_measurement_cannot_qualify_pmu_observer_effect() {
        let mut profile = fixture();
        profile["pmu_acquisition"]["observer_effect_status"] = json!("within_budget");
        profile["evidence"].as_array_mut().unwrap().push(json!({
            "evidence_id": "unrelated-lab-measurement",
            "source_type": "physical_lab_measurement",
            "source_uri": "https://example.invalid/unrelated",
            "input_sha256": "e".repeat(64),
            "parser_sha256": "b".repeat(64),
            "captured_at": null,
            "claim_scope": "unrelated fixture measurement"
        }));
        let error = validate_profile(&profile).unwrap_err();
        assert!(error.contains("within-budget observer effect requires"));
    }

    #[test]
    fn android_id_and_radio_hardware_identifiers_are_rejected_recursively() {
        for key in ["androidId", "bluetooth_address", "wifiMacAddress", "bssid"] {
            let mut profile = fixture();
            profile["device"][key] = json!("synthetic-private-id");
            let error = validate_profile(&profile).unwrap_err();
            assert!(error.contains("privacy-sensitive field"));
        }
    }

    #[test]
    fn unresolved_evidence_ids_are_rejected() {
        let mut profile = fixture();
        profile["boot_integrity"]["evidence_ids"] = json!(["missing-evidence"]);
        let error = validate_profile(&profile).unwrap_err();
        assert!(error.contains("references unknown evidence ID"));
    }

    #[test]
    fn duplicate_evidence_ids_are_rejected() {
        let mut profile = fixture();
        let duplicate = profile["evidence"][0].clone();
        profile["evidence"].as_array_mut().unwrap().push(duplicate);
        let error = validate_profile(&profile).unwrap_err();
        assert!(error.contains("duplicate evidence_id"));
    }

    #[test]
    fn vendor_documentation_cannot_be_reused_as_on_device_mte_evidence() {
        let mut profile = fixture();
        profile["hardware_protections"]["memory_tagging"]["hardware_support"] =
            json!("on_device_reported");
        let error = validate_profile(&profile).unwrap_err();
        assert!(error.contains("on-device capability result requires linked device evidence"));
    }

    #[test]
    fn reported_boot_state_requires_linked_property_evidence() {
        let mut profile = fixture();
        profile["boot_integrity"]["reported_verified_boot_state"] = json!("green_reported");
        let error = validate_profile(&profile).unwrap_err();
        assert!(error.contains("reported Verified Boot state requires linked on-device"));
    }

    #[test]
    fn synthetic_fixture_cannot_claim_device_runtime_or_physical_qualification() {
        let mut profile = fixture();
        profile["claim_ceiling"]["runtime_integrity_verified"] = json!(true);
        let error = validate_profile(&profile).unwrap_err();
        assert!(error.contains("synthetic fixtures cannot claim"));
    }

    #[test]
    fn receipt_binds_exact_profile_fixture_bytes() {
        let profile = fixture();
        let value = report(&profile, PROFILE_BYTES);
        assert_eq!(value["profile_sha256"], sha256_hex(PROFILE_BYTES));
        assert_eq!(value["qualification"]["device_was_queried"], false);
        assert_eq!(value["privacy"]["forbidden_identifier_fields_found"], false);
        assert_eq!(value["qualification"]["pmu_measurement_qualified"], false);
    }
}
