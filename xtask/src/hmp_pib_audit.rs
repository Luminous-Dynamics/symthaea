//! Deterministic structural and rights audit for PIB synthetic interaction fixtures.
//!
//! This validates fixture integrity only. It does not score agents, measure human diversity,
//! establish cultural competence, or validate the benchmark empirically.

use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

const REQUIRED_TRACKS: &[&str] = &[
    "PIB-REP",
    "PIB-WITHIN",
    "PIB-BETWEEN",
    "PIB-DYNAMIC",
    "PIB-MINORITY",
    "PIB-CONSENSUS",
    "PIB-DECISION",
    "PIB-JUDGE",
    "PIB-RIGHTS",
];

const VALID_OPERATIONS: &[&str] = &[
    "read", "display", "cite", "index", "retrieve", "summarize",
    "translate", "adapt", "train", "redistribute", "export", "preserve",
];

fn required_string<'a>(value: &'a Value, key: &str, at: &str) -> Result<&'a str> {
    let field = value.get(key).with_context(|| format!("{at}.{key} is required"))?;
    let text = field.as_str().with_context(|| format!("{at}.{key} must be a string"))?;
    if text.trim().is_empty() {
        bail!("{at}.{key} must not be empty");
    }
    Ok(text)
}

fn non_empty_string_array<'a>(value: &'a Value, key: &str, at: &str) -> Result<Vec<&'a str>> {
    let values = value.get(key).and_then(Value::as_array)
        .with_context(|| format!("{at}.{key} must be an array"))?;
    if values.is_empty() {
        bail!("{at}.{key} must not be empty");
    }
    values.iter().enumerate().map(|(index, item)| {
        let text = item.as_str()
            .with_context(|| format!("{at}.{key}[{index}] must be a string"))?;
        if text.trim().is_empty() {
            bail!("{at}.{key}[{index}] must not be empty");
        }
        Ok(text)
    }).collect()
}

/// Validate synthetic PIB fixture shape, coverage and operation-specific negative controls.
pub fn audit_fixture(fixture: &Value) -> Result<usize> {
    let benchmark_id = required_string(fixture, "benchmark_id", "benchmark")?;
    if !benchmark_id.starts_with("PIB-SYNTHETIC-") {
        bail!("benchmark.benchmark_id must start with PIB-SYNTHETIC-");
    }
    if required_string(fixture, "version", "benchmark")? != "0.1.0" {
        bail!("benchmark.version must be 0.1.0");
    }
    if required_string(fixture, "status", "benchmark")? != "synthetic_fixture_only" {
        bail!("benchmark.status must be synthetic_fixture_only");
    }
    required_string(fixture, "purpose", "benchmark")?;
    required_string(fixture, "scoring_warning", "benchmark")?;

    let cases = fixture.get("cases").and_then(Value::as_array)
        .context("benchmark.cases must be a non-empty array")?;
    if cases.is_empty() {
        bail!("benchmark.cases must not be empty");
    }

    let mut ids = BTreeSet::new();
    let mut tracks = BTreeSet::new();

    for (index, case) in cases.iter().enumerate() {
        let at = format!("benchmark.cases[{index}]");
        let case_id = required_string(case, "case_id", &at)?;
        if !case_id.starts_with("PIB-") || !ids.insert(case_id.to_owned()) {
            bail!("invalid or duplicate PIB case_id: {case_id}");
        }

        let track = required_string(case, "track", &at)?;
        if !REQUIRED_TRACKS.contains(&track) {
            bail!("{at}.track is not a registered PIB v0.1 track: {track}");
        }
        tracks.insert(track.to_owned());

        required_string(case, "scenario", &at)?;
        required_string(case, "interaction", &at)?;
        required_string(case, "prompt", &at)?;
        non_empty_string_array(case, "expected_behaviors", &at)?;
        non_empty_string_array(case, "forbidden_behaviors", &at)?;
        non_empty_string_array(case, "hard_fail_conditions", &at)?;

        let perspectives = case.get("initial_perspectives").and_then(Value::as_array)
            .with_context(|| format!("{at}.initial_perspectives must be a non-empty array"))?;
        if perspectives.is_empty() {
            bail!("{at}.initial_perspectives must not be empty");
        }
        let mut agents = BTreeSet::new();
        for (pindex, perspective) in perspectives.iter().enumerate() {
            let p_at = format!("{at}.initial_perspectives[{pindex}]");
            let agent = required_string(perspective, "agent", &p_at)?;
            if !agents.insert(agent.to_owned()) {
                bail!("{p_at}.agent is duplicated within the episode");
            }
            required_string(perspective, "scope", &p_at)?;
            required_string(perspective, "position", &p_at)?;
        }

        if track == "PIB-RIGHTS" {
            let rights = case.get("permission")
                .with_context(|| format!("{at}.permission is required for PIB-RIGHTS"))?;
            let access_class = required_string(rights, "access_class", &at)?;
            if !matches!(access_class, "restricted" | "community_controlled") {
                bail!("{at}.permission.access_class must be restricted or community_controlled");
            }

            let status = required_string(rights, "permission_status", &at)?;
            if !matches!(status, "pending" | "denied" | "withdrawn" | "not_requested" | "granted") {
                bail!("{at}.permission.permission_status is unknown");
            }

            let permitted_values = rights.get("permitted_uses").and_then(Value::as_array)
                .with_context(|| format!("{at}.permission.permitted_uses must be an array"))?;
            let mut permitted = BTreeSet::new();
            for item in permitted_values {
                let operation = item.as_str().context("permission.permitted_uses entries must be strings")?;
                if !VALID_OPERATIONS.contains(&operation) || !permitted.insert(operation.to_owned()) {
                    bail!("{at}.permission.permitted_uses contains an unknown or duplicate operation");
                }
            }
            if status != "granted" && !permitted.is_empty() {
                bail!("{at}: non-granted permission must have no permitted operations");
            }
            required_string(rights, "authority_reference", &at)?;
            if status == "granted" {
                required_string(rights, "permission_reference", &at)?;
            }

            let prohibited_values = non_empty_string_array(rights, "prohibited_uses", &at)?;
            let prohibited: BTreeSet<&str> = prohibited_values.into_iter().collect();
            let requested = non_empty_string_array(rights, "requested_operations", &at)?;
            for operation in requested {
                if !VALID_OPERATIONS.contains(&operation) {
                    bail!("{at}.permission.requested_operations contains unknown operation {operation}");
                }
                if permitted.contains(operation) || !prohibited.contains(operation) {
                    bail!("{at}: requested operation {operation} must be denied and prohibited");
                }
            }
        }
    }

    for track in REQUIRED_TRACKS {
        if !tracks.contains(*track) {
            bail!("PIB fixture is missing required track {track}");
        }
    }

    Ok(cases.len())
}

fn resolve(root: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() { path.to_path_buf() } else { root.join(path) }
}

pub fn run(root: &Path, file: &Path) -> Result<()> {
    let file_path = resolve(root, file);
    let fixture: Value = serde_json::from_slice(
        &std::fs::read(&file_path).with_context(|| format!("reading PIB fixture {}", file_path.display()))?
    ).with_context(|| format!("parsing PIB fixture {}", file_path.display()))?;
    let case_count = audit_fixture(&fixture)?;

    println!(
        "PIB synthetic fixture audit: PASS ({}; {} cases)",
        required_string(&fixture, "benchmark_id", "benchmark")?,
        case_count
    );
    println!("Registered track coverage: PASS ({} tracks)", REQUIRED_TRACKS.len());
    println!("Rights negative controls: PASS");
    println!("Note: fixture integrity only; no model scoring or human-validation claim.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fixture() -> Value {
        serde_json::from_str(include_str!(
            "../../research-fixtures/plurality-integrity/synthetic-adversarial-v0.1.0.json"
        )).expect("PIB fixture must be valid JSON")
    }

    #[test]
    fn pib_fixture_passes_integrity_audit() {
        assert_eq!(audit_fixture(&fixture()).expect("fixture should pass"), 11);
    }

    #[test]
    fn duplicate_episode_id_is_rejected() {
        let mut value = fixture();
        value["cases"][1]["case_id"] = value["cases"][0]["case_id"].clone();
        assert!(audit_fixture(&value).is_err());
    }

    #[test]
    fn missing_registered_track_is_rejected() {
        let mut value = fixture();
        value["cases"].as_array_mut().expect("cases").retain(
            |case| case["track"].as_str() != Some("PIB-BETWEEN")
        );
        assert!(audit_fixture(&value).is_err());
    }

    #[test]
    fn missing_attribution_scope_is_rejected() {
        let mut value = fixture();
        value["cases"][0]["initial_perspectives"][0].as_object_mut()
            .expect("perspective").remove("scope");
        assert!(audit_fixture(&value).is_err());
    }

    #[test]
    fn duplicate_agent_in_episode_is_rejected() {
        let mut value = fixture();
        value["cases"][0]["initial_perspectives"][1]["agent"] =
            value["cases"][0]["initial_perspectives"][0]["agent"].clone();
        assert!(audit_fixture(&value).is_err());
    }

    #[test]
    fn non_granted_permission_cannot_allow_any_operation() {
        let mut value = fixture();
        value["cases"][9]["permission"]["permitted_uses"] = json!(["cite"]);
        assert!(audit_fixture(&value).is_err());
    }

    #[test]
    fn requested_protected_operation_must_be_explicitly_denied() {
        let mut value = fixture();
        value["cases"][9]["permission"]["prohibited_uses"] =
            json!(["read", "summarize", "translate", "train", "redistribute", "export"]);
        assert!(audit_fixture(&value).is_err());
    }

    #[test]
    fn rights_fixture_requires_governance_reference() {
        let mut value = fixture();
        value["cases"][9]["permission"]["authority_reference"] = Value::Null;
        assert!(audit_fixture(&value).is_err());
    }

    #[test]
    fn empty_expected_behaviors_are_rejected() {
        let mut value = fixture();
        value["cases"][0]["expected_behaviors"] = json!([]);
        assert!(audit_fixture(&value).is_err());
    }

    #[test]
    fn fixture_must_remain_synthetic_only() {
        let mut value = fixture();
        value["status"] = json!("validated_dataset");
        assert!(audit_fixture(&value).is_err());
    }
}
