use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::process::Command;

const PARSER_ID: &[u8] = b"hardsec-028-android-read-only-inventory-parser-v1";

const PROPERTY_KEYS: &[&str] = &[
    "ro.product.model",
    "ro.hardware",
    "ro.product.cpu.abi",
    "ro.build.version.release",
    "ro.build.version.sdk",
    "ro.build.version.security_patch",
    "ro.vendor.build.security_patch",
    "ro.boot.flash.locked",
    "ro.boot.vbmeta.device_state",
    "ro.boot.verifiedbootstate",
    "ro.build.fingerprint",
];

const CPU_FEATURE_ALLOWLIST: &[&str] = &["mte", "paca", "pacg", "bti"];

type Properties = BTreeMap<&'static str, String>;

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn parser_sha256() -> String {
    sha256_hex(PARSER_ID)
}

fn clean_property(value: &str) -> Option<String> {
    let value = value.trim_matches(|character: char| character == '\r' || character == '\n');
    let value = value.trim();
    (!value.is_empty()).then(|| value.to_owned())
}

fn reported_bootloader_state(properties: &Properties) -> &'static str {
    let flash = properties.get("ro.boot.flash.locked").map(String::as_str);
    let vbmeta = properties
        .get("ro.boot.vbmeta.device_state")
        .map(|value| value.to_ascii_lowercase());

    let flash_state = match flash {
        Some("1") => Some("locked_reported"),
        Some("0") => Some("unlocked_reported"),
        _ => None,
    };
    let vbmeta_state = match vbmeta.as_deref() {
        Some("locked") => Some("locked_reported"),
        Some("unlocked") => Some("unlocked_reported"),
        _ => None,
    };

    match (flash_state, vbmeta_state) {
        (Some(left), Some(right)) if left != right => "conflicting_reports",
        (Some(left), _) | (_, Some(left)) => left,
        (None, None) => "unknown",
    }
}

fn reported_verified_boot_state(properties: &Properties) -> &'static str {
    match properties
        .get("ro.boot.verifiedbootstate")
        .map(|value| value.to_ascii_lowercase())
        .as_deref()
    {
        Some("green") => "green_reported",
        Some("yellow") => "yellow_reported",
        Some("orange") => "orange_reported",
        Some("red") => "red_reported",
        _ => "unknown",
    }
}

// Return sorted per-CPU allowlisted feature sets. The raw /proc/cpuinfo is
// intentionally never retained in the profile or emitted to stdout.
fn cpu_feature_rows(cpuinfo: &str) -> Vec<BTreeSet<String>> {
    let mut rows = Vec::new();
    for line in cpuinfo.lines() {
        let Some((key, values)) = line.split_once(':') else {
            continue;
        };
        if key.trim() != "Features" {
            continue;
        }
        let features: BTreeSet<String> = values
            .split_whitespace()
            .map(str::to_ascii_lowercase)
            .filter(|feature| CPU_FEATURE_ALLOWLIST.contains(&feature.as_str()))
            .collect();
        rows.push(features);
    }
    rows
}

fn feature_support(rows: &[BTreeSet<String>], required: &[&str]) -> &'static str {
    if rows.is_empty() {
        return "not_measured";
    }
    let present_on_all = rows
        .iter()
        .all(|row| required.iter().all(|feature| row.contains(*feature)));
    if present_on_all {
        return "on_device_reported";
    }
    let present_on_any = rows
        .iter()
        .any(|row| required.iter().all(|feature| row.contains(*feature)));
    if present_on_any {
        "unknown"
    } else {
        "unsupported"
    }
}

fn parse_simpleperf_event_names(output: &str) -> Vec<String> {
    output
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            if line.is_empty() || line.to_ascii_lowercase().starts_with("list of ") {
                return None;
            }
            let candidate = line.split_whitespace().next()?;
            if candidate.len() > 128
                || !candidate.chars().all(|character| {
                    character.is_ascii_alphanumeric()
                        || matches!(character, '_' | '-' | '.' | ':' | '/')
                })
            {
                return None;
            }
            Some(candidate.to_owned())
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn property_evidence_id(key: &str) -> String {
    format!("adbprop-{}", key.replace('.', "-"))
}

fn push_evidence(
    evidence: &mut Vec<Value>,
    evidence_id: String,
    source_uri: String,
    safe_input: &[u8],
    claim_scope: String,
) {
    evidence.push(json!({
        "evidence_id": evidence_id,
        "source_type": "on_device_property",
        "source_uri": source_uri,
        "input_sha256": sha256_hex(safe_input),
        "parser_sha256": parser_sha256(),
        "captured_at": null,
        "claim_scope": claim_scope
    }));
}

fn capture_profile(
    properties: &Properties,
    cpuinfo: Option<&str>,
    simpleperf_output: Option<&str>,
) -> Value {
    let mut evidence = Vec::new();
    let mut property_ids = BTreeMap::new();

    for key in PROPERTY_KEYS {
        let raw = properties.get(key).map(String::as_str).unwrap_or("");
        let evidence_id = property_evidence_id(key);
        property_ids.insert(*key, evidence_id.clone());
        let claim_scope = if raw.is_empty() {
            format!("ADB getprop {key} returned no value; no runtime or attestation claim")
        } else if *key == "ro.build.fingerprint" {
            "ADB build fingerprint is hashed and discarded; digest only".to_owned()
        } else if key.starts_with("ro.boot.") {
            format!("ADB-reported boot property {key}; not cryptographic attestation")
        } else {
            format!("whitelisted ADB getprop {key}; software-reported device metadata")
        };
        push_evidence(
            &mut evidence,
            evidence_id,
            format!("urn:hardsec:android:property:{key}"),
            raw.as_bytes(),
            claim_scope,
        );
    }

    let model = properties
        .get("ro.product.model")
        .and_then(|value| clean_property(value))
        .unwrap_or_else(|| "Unknown Android device".to_owned());
    let soc_family = properties
        .get("ro.hardware")
        .and_then(|value| clean_property(value));
    let abi = properties
        .get("ro.product.cpu.abi")
        .and_then(|value| clean_property(value));
    let isa_family = match abi.as_deref() {
        Some("arm64-v8a") | Some("aarch64") => "aarch64",
        Some(value) if !value.is_empty() => "other",
        _ => "not_measured",
    };
    let os_version = properties
        .get("ro.build.version.release")
        .and_then(|value| clean_property(value));
    let os_api_level = properties
        .get("ro.build.version.sdk")
        .and_then(|value| clean_property(value));
    let security_patch_level = properties
        .get("ro.build.version.security_patch")
        .and_then(|value| clean_property(value));
    let vendor_security_patch_level = properties
        .get("ro.vendor.build.security_patch")
        .and_then(|value| clean_property(value));
    let build_fingerprint_sha256 = properties
        .get("ro.build.fingerprint")
        .and_then(|value| clean_property(value))
        .map(|value| sha256_hex(value.as_bytes()));

    let boot_evidence_ids = [
        "ro.boot.flash.locked",
        "ro.boot.vbmeta.device_state",
        "ro.boot.verifiedbootstate",
        "ro.build.version.security_patch",
        "ro.vendor.build.security_patch",
    ]
    .iter()
    .filter_map(|key| property_ids.get(key).cloned())
    .collect::<Vec<_>>();

    let cpu_rows = cpuinfo.map(cpu_feature_rows).unwrap_or_default();
    let mut cpu_feature_evidence_ids = Vec::new();
    if cpuinfo.is_some() {
        let canonical = cpu_rows
            .iter()
            .map(|row| row.iter().cloned().collect::<Vec<_>>().join(","))
            .collect::<Vec<_>>()
            .join("\n");
        let evidence_id = "cpuinfo-allowlisted-features".to_owned();
        push_evidence(
            &mut evidence,
            evidence_id.clone(),
            "urn:hardsec:android:proc:cpuinfo:allowlisted-features".to_owned(),
            canonical.as_bytes(),
            "only allowlisted CPU feature tokens retained; raw /proc/cpuinfo discarded"
                .to_owned(),
        );
        cpu_feature_evidence_ids.push(evidence_id);
    }

    let simpleperf_events = simpleperf_output.map(parse_simpleperf_event_names);
    let mut pmu_evidence_ids = Vec::new();
    if let Some(events) = &simpleperf_events {
        let canonical = events.join("\n");
        let evidence_id = "simpleperf-event-list".to_owned();
        push_evidence(
            &mut evidence,
            evidence_id.clone(),
            "urn:hardsec:android:simpleperf:list".to_owned(),
            canonical.as_bytes(),
            "simpleperf list event-name inventory only; no counter read or access qualification"
                .to_owned(),
        );
        pmu_evidence_ids.push(evidence_id);
    }

    let patch_level_binding_status = if security_patch_level.is_some()
        || vendor_security_patch_level.is_some()
    {
        "software_reported"
    } else {
        "not_measured"
    };

    json!({
        "schema_version": "hardsec-mobile-device-profile-v1",
        "profile_id": "android-read-only-inventory-v1",
        "evidence_domain": "physical_device_capture",
        "device": {
            "platform": "android",
            "product_family": model,
            "soc_family": soc_family,
            "isa_family": isa_family,
            "os_version": os_version,
            "os_api_level": os_api_level,
            "security_patch_level": security_patch_level,
            "vendor_security_patch_level": vendor_security_patch_level,
            "build_fingerprint_sha256": build_fingerprint_sha256
        },
        "boot_integrity": {
            "bootloader_locked": "not_measured",
            "verified_boot_state": "not_measured",
            "rollback_protection_state": "not_measured",
            "attestation_security_level": "not_measured",
            "attestation_challenge_freshness": "not_measured",
            "certificate_chain_status": "not_measured",
            "patch_level_binding_status": patch_level_binding_status,
            "reported_bootloader_lock_state": reported_bootloader_state(properties),
            "reported_verified_boot_state": reported_verified_boot_state(properties),
            "evidence_ids": boot_evidence_ids
        },
        "hardware_protections": {
            "memory_tagging": {
                "hardware_support": feature_support(&cpu_rows, &["mte"]),
                "runtime_state": "not_measured",
                "mode": null,
                "protected_scope": null,
                "evidence_ids": cpu_feature_evidence_ids
            },
            "pointer_authentication": {
                "hardware_support": feature_support(&cpu_rows, &["paca", "pacg"]),
                "runtime_state": "not_measured",
                "mode": null,
                "protected_scope": null,
                "evidence_ids": cpu_feature_evidence_ids
            },
            "branch_target_protection": {
                "hardware_support": feature_support(&cpu_rows, &["bti"]),
                "runtime_state": "not_measured",
                "mode": null,
                "protected_scope": null,
                "evidence_ids": cpu_feature_evidence_ids
            },
            "secure_element": {
                "hardware_support": "unknown",
                "runtime_state": "not_measured",
                "mode": null,
                "protected_scope": null,
                "evidence_ids": []
            },
            "tee": {
                "hardware_support": "unknown",
                "runtime_state": "not_measured",
                "mode": null,
                "protected_scope": null,
                "evidence_ids": []
            }
        },
        "pmu_acquisition": {
            "interface": if simpleperf_events.is_some() { "simpleperf" } else { "unknown" },
            "access_status": "not_measured",
            "available_events": simpleperf_events.unwrap_or_default(),
            "privilege_context": "shell",
            "multiplexing_status": "not_measured",
            "time_enabled_ns": null,
            "time_running_ns": null,
            "read_overhead_status": "not_measured",
            "sample_loss_status": "not_measured",
            "observer_effect_status": "not_measured",
            "evidence_ids": pmu_evidence_ids
        },
        "environment": {
            "thermal_status": "not_measured",
            "power_state": "not_measured",
            "charging_state": "not_measured",
            "radio_activity_class": "not_measured",
            "screen_state": "not_measured"
        },
        "evidence": evidence,
        "claim_ceiling": {
            "runtime_integrity_verified": false,
            "pmu_measurement_qualified": false,
            "physical_attack_resistance_established": false,
            "limitations": [
                "ADB getprop values are software-reported and are not cryptographic hardware attestation.",
                "Reported boot fields do not verify the boot chain or certificate chain.",
                "CPU features are on-device-reported capabilities; MTE/PAC/BTI process enablement and scope remain not measured.",
                "simpleperf list reports event names only; PMU access, multiplexing, overhead, sample loss and observer effect remain not measured.",
                "Thermal, battery, radio and screen state are not captured by this inventory.",
                "No device serial, IMEI, stable identifier, raw build fingerprint, raw CPU info, raw attestation material or user data is exported."
            ]
        }
    })
}

fn run_adb(adb_path: &str, shell_args: &[&str]) -> Result<String, String> {
    let output = Command::new(adb_path)
        .arg("shell")
        .args(shell_args)
        .output()
        .map_err(|_| "ADB executable could not be started".to_owned())?;
    if !output.status.success() {
        // Do not echo stderr: ADB may include the connected device serial.
        return Err("ADB shell command failed; raw command output was suppressed".to_owned());
    }
    String::from_utf8(output.stdout)
        .map_err(|_| "ADB command returned non-UTF-8 output".to_owned())
}

fn collect_from_runner<F>(mut runner: F) -> Result<Value, String>
where
    F: FnMut(&[&str]) -> Result<String, String>,
{
    let mut properties = Properties::new();
    for key in PROPERTY_KEYS {
        let value = runner(&["getprop", key])
            .map_err(|_| "ADB property capture failed; raw device output was suppressed".to_owned())?;
        properties.insert(key, value.trim().to_owned());
    }

    let cpuinfo = runner(&["cat", "/proc/cpuinfo"]).ok();
    let simpleperf = runner(&["simpleperf", "list"]).ok();

    Ok(capture_profile(
        &properties,
        cpuinfo.as_deref(),
        simpleperf.as_deref(),
    ))
}

fn collect_from_adb(adb_path: &str) -> Result<Value, String> {
    collect_from_runner(|shell_args| run_adb(adb_path, shell_args))
}

fn main() {
    let adb_path = env::var("HARDSEC_ADB_PATH").unwrap_or_else(|_| "adb".to_owned());
    match collect_from_adb(&adb_path) {
        Ok(profile) => {
            println!(
                "{}",
                serde_json::to_string_pretty(&profile)
                    .expect("serialize Android read-only inventory profile")
            );
        }
        Err(error) => {
            eprintln!("HARDSEC-028 Android inventory failed: {error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_properties() -> Properties {
        let mut values = Properties::new();
        values.insert("ro.product.model", "Pixel 8 Pro".to_owned());
        values.insert("ro.hardware", "tensor".to_owned());
        values.insert("ro.product.cpu.abi", "arm64-v8a".to_owned());
        values.insert("ro.build.version.release", "16".to_owned());
        values.insert("ro.build.version.sdk", "36".to_owned());
        values.insert("ro.build.version.security_patch", "2026-10-05".to_owned());
        values.insert("ro.vendor.build.security_patch", "2026-10-05".to_owned());
        values.insert("ro.boot.flash.locked", "1".to_owned());
        values.insert("ro.boot.vbmeta.device_state", "locked".to_owned());
        values.insert("ro.boot.verifiedbootstate", "green".to_owned());
        values.insert("ro.build.fingerprint", "google/husky/husky:16/fixture/123:user/release-keys".to_owned());
        values
    }

    #[test]
    fn injected_runner_sees_only_fixed_read_only_adb_commands() {
        let properties = fixture_properties();
        let mut calls: Vec<Vec<String>> = Vec::new();
        let profile = collect_from_runner(|arguments| {
            calls.push(arguments.iter().map(|argument| (*argument).to_owned()).collect());
            if arguments.len() == 2 && arguments[0] == "getprop" {
                Ok(properties
                    .get(arguments[1])
                    .cloned()
                    .unwrap_or_default())
            } else if arguments.len() == 2
                && arguments[0] == "cat"
                && arguments[1] == "/proc/cpuinfo"
            {
                Ok("Features : fp asimd mte paca pacg bti\n".to_owned())
            } else if arguments.len() == 2
                && arguments[0] == "simpleperf"
                && arguments[1] == "list"
            {
                Ok("List of hardware events:\ncpu-cycles\ninstructions\n".to_owned())
            } else {
                Err("unexpected ADB command".to_owned())
            }
        })
        .unwrap();

        assert_eq!(calls.len(), PROPERTY_KEYS.len() + 2);
        assert!(calls.iter().all(|call| call.first().is_some_and(|command| {
            command == "getprop" || (command == "cat" && call.get(1).is_some_and(|value| value == "/proc/cpuinfo"))
                || (command == "simpleperf" && call.get(1).is_some_and(|value| value == "list"))
        })));
        assert!(!calls.iter().any(|call| call.join(" ").contains("serial")));
        assert!(!calls.iter().any(|call| call.join(" ").contains("devices")));
        assert_eq!(profile["boot_integrity"]["bootloader_locked"], "not_measured");
        assert_eq!(profile["pmu_acquisition"]["access_status"], "not_measured");
    }

    #[test]
    fn ambiguous_device_selection_failure_does_not_echo_raw_adb_error() {
        let result = collect_from_runner(|_| {
            Err("more than one device/emulator: SERIAL-PRIVATE-123".to_owned())
        });
        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(!error.to_ascii_lowercase().contains("serial-private"));
    }

    #[test]
    fn reported_boot_properties_never_promote_attestation_fields() {
        let profile = capture_profile(
            &fixture_properties(),
            Some("Features : fp asimd aes mte paca pacg bti\nFeatures : fp asimd aes mte paca pacg bti\n"),
            Some("List of hardware events:\ncpu-cycles\ninstructions\ncache-misses\n"),
        );
        assert_eq!(profile["boot_integrity"]["reported_bootloader_lock_state"], "locked_reported");
        assert_eq!(profile["boot_integrity"]["reported_verified_boot_state"], "green_reported");
        assert_eq!(profile["boot_integrity"]["bootloader_locked"], "not_measured");
        assert_eq!(profile["boot_integrity"]["verified_boot_state"], "not_measured");
        assert_eq!(profile["boot_integrity"]["attestation_security_level"], "not_measured");
        assert_eq!(profile["boot_integrity"]["certificate_chain_status"], "not_measured");
    }

    #[test]
    fn conflicting_bootloader_properties_are_reported_as_conflict() {
        let mut properties = fixture_properties();
        properties.insert("ro.boot.vbmeta.device_state", "unlocked".to_owned());
        let profile = capture_profile(&properties, None, None);
        assert_eq!(profile["boot_integrity"]["reported_bootloader_lock_state"], "conflicting_reports");
        assert_eq!(profile["boot_integrity"]["bootloader_locked"], "not_measured");
    }

    #[test]
    fn missing_properties_remain_unknown_or_not_measured() {
        let profile = capture_profile(&Properties::new(), None, None);
        assert_eq!(profile["device"]["product_family"], "Unknown Android device");
        assert_eq!(profile["device"]["os_version"], Value::Null);
        assert_eq!(profile["boot_integrity"]["reported_bootloader_lock_state"], "unknown");
        assert_eq!(profile["boot_integrity"]["reported_verified_boot_state"], "unknown");
        assert_eq!(profile["pmu_acquisition"]["access_status"], "not_measured");
        assert_eq!(profile["pmu_acquisition"]["available_events"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn heterogeneous_cpu_feature_rows_do_not_promote_partial_feature_support() {
        let rows = cpu_feature_rows(
            "Processor : 0\nFeatures : fp asimd mte paca pacg bti\nProcessor : 1\nFeatures : fp asimd paca pacg\n",
        );
        assert_eq!(feature_support(&rows, &["mte"]), "unknown");
        assert_eq!(feature_support(&rows, &["paca", "pacg"]), "on_device_reported");
        assert_eq!(feature_support(&rows, &["bti"]), "unknown");
    }

    #[test]
    fn simpleperf_event_list_does_not_claim_pmu_access_or_measurement_quality() {
        let profile = capture_profile(
            &fixture_properties(),
            None,
            Some("List of hardware events:\ncpu-cycles\ninstructions\nbranch-misses\n"),
        );
        assert_eq!(profile["pmu_acquisition"]["interface"], "simpleperf");
        assert_eq!(profile["pmu_acquisition"]["available_events"], json!(["branch-misses", "cpu-cycles", "instructions"]));
        assert_eq!(profile["pmu_acquisition"]["access_status"], "not_measured");
        assert_eq!(profile["pmu_acquisition"]["multiplexing_status"], "not_measured");
        assert_eq!(profile["pmu_acquisition"]["observer_effect_status"], "not_measured");
        assert_eq!(profile["claim_ceiling"]["pmu_measurement_qualified"], false);
    }

    #[test]
    fn raw_build_fingerprint_and_cpuinfo_are_not_exported() {
        let properties = fixture_properties();
        let raw_fingerprint = properties["ro.build.fingerprint"].clone();
        let cpuinfo = "Processor : 0\nFeatures : fp asimd mte paca pacg bti\n";
        let profile = capture_profile(&properties, Some(cpuinfo), None);
        let output = serde_json::to_string(&profile).unwrap();
        assert!(!output.contains(&raw_fingerprint));
        assert!(!output.contains("Processor"));
        assert!(!output.contains("Features"));
        assert!(profile["device"]["build_fingerprint_sha256"].as_str().unwrap().len() == 64);
    }

    #[test]
    fn profile_contains_no_stable_identifiers_or_raw_attestation_fields() {
        let profile = capture_profile(&fixture_properties(), None, None);
        let output = serde_json::to_string(&profile).unwrap().to_ascii_lowercase();
        for forbidden in ["imei", "meid", "imsi", "iccid", "serial_number", "android_id", "attestation_certificate_der"] {
            assert!(!output.contains(forbidden));
        }
    }

    #[test]
    fn captured_profile_keeps_attestation_and_measurement_claims_not_measured() {
        let profile = capture_profile(
            &fixture_properties(),
            Some("Features : fp asimd mte paca pacg bti\n"),
            Some("List of hardware events:\ncpu-cycles\n"),
        );
        assert_eq!(profile["evidence_domain"], "physical_device_capture");
        assert_eq!(profile["device"]["product_family"], "Pixel 8 Pro");
        assert_eq!(profile["boot_integrity"]["reported_verified_boot_state"], "green_reported");
        assert_eq!(profile["boot_integrity"]["verified_boot_state"], "not_measured");
        assert_eq!(profile["boot_integrity"]["attestation_security_level"], "not_measured");
        assert_eq!(
            profile["hardware_protections"]["memory_tagging"]["hardware_support"],
            "on_device_reported"
        );
        assert_eq!(
            profile["hardware_protections"]["memory_tagging"]["runtime_state"],
            "not_measured"
        );
        assert_eq!(profile["pmu_acquisition"]["interface"], "simpleperf");
        assert_eq!(profile["pmu_acquisition"]["access_status"], "not_measured");
        assert_eq!(profile["claim_ceiling"]["runtime_integrity_verified"], false);
        assert_eq!(profile["claim_ceiling"]["pmu_measurement_qualified"], false);
    }

    #[test]
    fn profile_hash_evidence_is_bound_to_allowlisted_input_values() {
        let profile = capture_profile(&fixture_properties(), None, None);
        let evidence = profile["evidence"].as_array().unwrap();
        assert!(evidence.iter().any(|entry| {
            entry["evidence_id"] == "adbprop-ro-product-model"
                && entry["input_sha256"] == sha256_hex(b"Pixel 8 Pro")
        }));
        assert!(evidence.iter().any(|entry| {
            entry["evidence_id"] == "adbprop-ro-build-fingerprint"
                && entry["claim_scope"] == "ADB build fingerprint is hashed and discarded; digest only"
        }));
        assert!(evidence.iter().all(|entry| entry["source_type"] == "on_device_property"));
    }
}
