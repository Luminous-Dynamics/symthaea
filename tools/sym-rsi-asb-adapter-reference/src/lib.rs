//! Strict offline contract validation for the pinned AS-Bench adapter manifest.
//!
//! This crate checks compatibility metadata and declared gate mappings only.
//! It does not execute campaigns, verify upstream signatures, evaluate results,
//! access hidden truth, issue receipts, or authorize benchmark promotion.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

const MANIFEST_VERSION: &str = "sym-rsi-asb-compatibility-manifest-v0";
const REQUIRED_STATUS: &str = "PINNED_UPSTREAM_COMPATIBILITY_TARGET_NOT_YET_RUN";
const UPSTREAM_REPOSITORY: &str = "Yibo-Wen/as-bench";
const UPSTREAM_COMMIT: &str = "b82ee550344e6055197b5eef3db47a44082cae4d";
const EXPECTED_COMBINATION: &str = "all_discovery_and_learning_gates_must_pass";

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub manifest_version: String,
    pub status: String,
    pub generated_utc_date: String,
    pub compatibility: Compatibility,
    pub upstream: Upstream,
    pub adapter_scope: AdapterScope,
    pub campaigns: Vec<Campaign>,
    pub invariants: Vec<String>,
    pub ownership: BTreeMap<String, u32>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Compatibility {
    pub mode: String,
    pub official_benchmark_claim: bool,
    pub upstream_gate_mutation_permitted: bool,
    pub note: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Upstream {
    pub repository: String,
    pub url: String,
    pub commit: String,
    pub commit_url: String,
    pub task_schema_version: String,
    pub lab_api_version: u32,
    pub pinned_files: Vec<PinnedFile>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PinnedFile {
    pub path: String,
    pub git_blob_sha: String,
    pub purpose: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdapterScope {
    pub first_backend: String,
    pub second_backend: String,
    pub excluded_backends: Vec<String>,
    pub excluded_claims: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Campaign {
    pub id: String,
    pub task_path: String,
    pub backend: String,
    pub domain: String,
    pub source: String,
    pub source_doi: String,
    pub preprint_doi: Option<String>,
    pub dataset_doi: Option<String>,
    pub data_license: String,
    pub design_space: DesignSpace,
    pub gates: Gates,
    pub reference_shipped_split: Value,
    pub observation_semantics: Value,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DesignSpace {
    pub orderable_designs: u64,
    pub rounds: u64,
    pub batch_size: u64,
    pub total_measurement_budget: u64,
    pub held_out_designs: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Gates {
    pub discovery: Vec<Gate>,
    pub learning: Vec<Gate>,
    pub combination: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Gate {
    pub metric: String,
    pub operator: String,
    pub threshold: f64,
    pub held_out_designs: Option<u64>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ValidationReport {
    pub valid: bool,
    pub campaign_count: usize,
    pub pinned_file_count: usize,
    pub issues: Vec<String>,
    pub limitations: Vec<String>,
}

fn is_hex_sha(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn value_u64(value: &Value, key: &str) -> Option<u64> {
    value.get(key).and_then(Value::as_u64)
}

fn value_bool(value: &Value, key: &str) -> Option<bool> {
    value.get(key).and_then(Value::as_bool)
}

fn gate_matches(
    gate: &Gate,
    metric: &str,
    operator: &str,
    threshold: f64,
    held_out_designs: Option<u64>,
) -> bool {
    gate.metric == metric
        && gate.operator == operator
        && (gate.threshold - threshold).abs() <= 1e-12
        && gate.held_out_designs == held_out_designs
}

fn validate_campaign(campaign: &Campaign, issues: &mut Vec<String>) {
    if campaign.gates.combination != EXPECTED_COMBINATION {
        issues.push(format!("{}: discovery and learning must both gate success", campaign.id));
    }
    match campaign.design_space.rounds.checked_mul(campaign.design_space.batch_size) {
        Some(total) if total == campaign.design_space.total_measurement_budget => {}
        _ => issues.push(format!("{}: total budget must equal rounds × batch size", campaign.id)),
    }
    if campaign.design_space.total_measurement_budget > campaign.design_space.orderable_designs {
        issues.push(format!("{}: budget exceeds orderable design pool", campaign.id));
    }
    if campaign.gates.discovery.is_empty() || campaign.gates.learning.is_empty() {
        issues.push(format!("{}: both discovery and learning gates are required", campaign.id));
    }
    for gate in campaign.gates.discovery.iter().chain(&campaign.gates.learning) {
        if gate.metric.trim().is_empty()
            || gate.operator != ">="
            || !gate.threshold.is_finite()
        {
            issues.push(format!("{}: invalid or unsupported gate contract", campaign.id));
        }
    }

    match campaign.id.as_str() {
        "nuclease-active-learning" => {
            if campaign.backend != "replay"
                || campaign.task_path != "tasks/biology/deepmind/nuclease-active-learning"
                || campaign.design_space.orderable_designs != 20_000
                || campaign.design_space.rounds != 4
                || campaign.design_space.batch_size != 96
                || campaign.design_space.total_measurement_budget != 384
                || campaign.design_space.held_out_designs != Some(5_000)
                || campaign.gates.discovery.len() != 2
                || campaign.gates.learning.len() != 1
                || !gate_matches(&campaign.gates.discovery[0], "discovery_hits", ">=", 190.0, None)
                || !gate_matches(&campaign.gates.discovery[1], "distinct_hit_families", ">=", 175.0, None)
                || !gate_matches(&campaign.gates.learning[0], "ndcg_at_100", ">=", 0.42, Some(5_000))
            {
                issues.push("nuclease-active-learning: pinned replay task contract changed".to_owned());
            }
            if campaign.data_license != "CC-BY-4.0"
                || value_u64(&campaign.reference_shipped_split, "discovery_hits") != Some(278)
                || value_u64(&campaign.reference_shipped_split, "distinct_hit_families") != Some(215)
                || campaign.reference_shipped_split.get("ndcg_at_100").and_then(Value::as_f64)
                    .is_none_or(|value| (value - 0.488).abs() > 1e-12)
                || campaign.reference_shipped_split.get("role").and_then(Value::as_str)
                    != Some("upstream_reference_solution_result_not_a_Symthaea_result")
                || campaign.observation_semantics.get("measurement_source").and_then(Value::as_str)
                    != Some("published replay table")
            {
                issues.push("nuclease-active-learning: reference/provenance declaration changed".to_owned());
            }
        }
        "propylene-active-learning" => {
            if campaign.backend != "twin"
                || campaign.task_path != "tasks/chemistry/sargent-lab/propylene-active-learning"
                || campaign.design_space.orderable_designs != 736
                || campaign.design_space.rounds != 3
                || campaign.design_space.batch_size != 20
                || campaign.design_space.total_measurement_budget != 60
                || campaign.design_space.held_out_designs != Some(244)
                || campaign.gates.discovery.len() != 1
                || campaign.gates.learning.len() != 1
                || !gate_matches(&campaign.gates.discovery[0], "discovery_hits", ">=", 7.0, None)
                || !gate_matches(&campaign.gates.learning[0], "ndcg_at_20", ">=", 0.52, Some(244))
            {
                issues.push("propylene-active-learning: pinned twin task contract changed".to_owned());
            }
            if campaign.data_license != "CC-BY-4.0"
                || value_u64(&campaign.reference_shipped_split, "discovery_hits") != Some(8)
                || campaign.reference_shipped_split.get("ndcg_at_20").and_then(Value::as_f64)
                    .is_none_or(|value| (value - 0.626).abs() > 1e-12)
                || campaign.reference_shipped_split.get("role").and_then(Value::as_str)
                    != Some("upstream_reference_solution_result_not_a_Symthaea_result")
                || value_u64(&campaign.observation_semantics, "exact_published_measurements") != Some(96)
                || value_u64(&campaign.observation_semantics, "surrogate_model_outputs") != Some(884)
                || value_bool(&campaign.observation_semantics, "twin_is_noiseless_mean_output") != Some(true)
                || value_bool(&campaign.observation_semantics, "live_laboratory_measurements") != Some(false)
            {
                issues.push("propylene-active-learning: twin measurement provenance changed".to_owned());
            }
        }
        other => issues.push(format!("unknown campaign in pinned v0 contract: {other}")),
    }
}

pub fn validate_manifest(manifest: &Manifest) -> ValidationReport {
    let mut issues = Vec::new();
    let mut paths = BTreeSet::new();
    let mut campaign_ids = BTreeSet::new();

    if manifest.manifest_version != MANIFEST_VERSION {
        issues.push("unsupported manifest_version".to_owned());
    }
    if manifest.status != REQUIRED_STATUS {
        issues.push("status must remain NOT_YET_RUN until adapter conformance is evidenced".to_owned());
    }
    if manifest.generated_utc_date != "2026-11-07" {
        issues.push("unexpected manifest date; create a new version rather than silently retiming v0".to_owned());
    }
    if manifest.compatibility.mode != "read_only_upstream_replay_and_twin"
        || manifest.compatibility.official_benchmark_claim
        || manifest.compatibility.upstream_gate_mutation_permitted
        || manifest.compatibility.note.trim().is_empty()
    {
        issues.push("compatibility claim boundary is missing or weakened".to_owned());
    }
    if manifest.upstream.repository != UPSTREAM_REPOSITORY
        || manifest.upstream.commit != UPSTREAM_COMMIT
        || !is_hex_sha(&manifest.upstream.commit)
        || manifest.upstream.task_schema_version != "1.4"
        || manifest.upstream.lab_api_version != 1
    {
        issues.push("upstream repository, commit, task schema or Lab API pin changed".to_owned());
    }
    if manifest.adapter_scope.first_backend != "replay"
        || manifest.adapter_scope.second_backend != "twin"
        || !manifest.adapter_scope.excluded_backends.iter().any(|item| item == "live")
        || manifest.adapter_scope.excluded_claims.is_empty()
    {
        issues.push("adapter scope must begin with replay/twin and explicitly exclude live claims".to_owned());
    }

    const REQUIRED_PINS: [(&str, &str); 8] = [
        ("README.md", "f5430803960bdbe8e4c2fb0f2bd3dca25dbe52a9"),
        ("docs/ARCHITECTURE.md", "c056f367ec6f92ab91057465ed21b417e4ce0645"),
        ("runtime/api/openapi.yaml", "db5c272a5ffd13ac79d0ed9682a90ed6ffdd27dc"),
        ("task-template.toml", "4e60ee6d88d8ddc83c6b677163e0d944a67e6560"),
        ("tasks/biology/deepmind/nuclease-active-learning/task.toml", "f5cd2ab065844cad9209697aa2e559d5caa84542"),
        ("tasks/biology/deepmind/nuclease-active-learning/README.md", "2eb8d04d8c8e89efc2bbe4b85a33690fb3ff68e7"),
        ("tasks/chemistry/sargent-lab/propylene-active-learning/task.toml", "31a26d7baf312520d1eabd803fa40912ca43e4b9"),
        ("tasks/chemistry/sargent-lab/propylene-active-learning/README.md", "05513bae497d1c66fa9dd39253bf6bbaa6bb054d"),
    ];
    for pin in &manifest.upstream.pinned_files {
        if pin.path.trim().is_empty() || pin.purpose.trim().is_empty() || !is_hex_sha(&pin.git_blob_sha) {
            issues.push(format!("invalid upstream file pin: {}", pin.path));
        }
        if !paths.insert(pin.path.as_str()) {
            issues.push(format!("duplicate upstream file path: {}", pin.path));
        }
    }
    if manifest.upstream.pinned_files.len() != REQUIRED_PINS.len() {
        issues.push(format!("expected {} upstream pinned files", REQUIRED_PINS.len()));
    }
    for (path, sha) in REQUIRED_PINS {
        if !manifest.upstream.pinned_files.iter().any(|pin| pin.path == path && pin.git_blob_sha == sha) {
            issues.push(format!("missing or changed upstream pin: {path}"));
        }
    }

    if manifest.campaigns.len() != 2 {
        issues.push("v0 requires exactly one replay and one twin campaign".to_owned());
    }
    for campaign in &manifest.campaigns {
        if !campaign_ids.insert(campaign.id.as_str()) {
            issues.push(format!("duplicate campaign id: {}", campaign.id));
        }
        validate_campaign(campaign, &mut issues);
    }
    for required in ["nuclease-active-learning", "propylene-active-learning"] {
        if !campaign_ids.contains(required) {
            issues.push(format!("required campaign missing: {required}"));
        }
    }
    if manifest.invariants.len() < 10 || manifest.invariants.iter().any(|item| item.trim().is_empty()) {
        issues.push("required adapter invariants are incomplete".to_owned());
    }
    for owner in [10877, 9230, 8721, 9332, 10756, 10717, 10630, 10497] {
        if !manifest.ownership.values().any(|value| *value == owner) {
            issues.push(format!("existing owner #{owner} is not represented"));
        }
    }

    issues.sort();
    issues.dedup();
    ValidationReport {
        valid: issues.is_empty(),
        campaign_count: manifest.campaigns.len(),
        pinned_file_count: manifest.upstream.pinned_files.len(),
        issues,
        limitations: vec![
            "A structurally valid manifest does not prove the upstream Git object is authentic or still hosted.".to_owned(),
            "This validator checks declared compatibility metadata; it does not execute Lab API calls or campaign verifiers.".to_owned(),
            "Reference scores are upstream reference-solution results, not Symthaea performance.".to_owned(),
            "No live laboratory, scientific discovery, benchmark promotion or R4 claim is qualified by this tool.".to_owned(),
        ],
    }
}

pub fn parse_and_validate(contents: &str) -> Result<ValidationReport, serde_json::Error> {
    let manifest: Manifest = serde_json::from_str(contents)?;
    Ok(validate_manifest(&manifest))
}

#[cfg(test)]
mod tests {
    use super::*;

    const MANIFEST: &str =
        include_str!("../../../docs/research/adapters/SYM_RSI_ASB_COMPATIBILITY_MANIFEST_V0_2026-11-07.json");

    fn valid_manifest() -> Manifest {
        serde_json::from_str(MANIFEST).expect("checked-in compatibility manifest parses")
    }

    #[test]
    fn accepts_the_pinned_two_campaign_manifest() {
        let report = validate_manifest(&valid_manifest());
        assert!(report.valid, "{:?}", report.issues);
        assert_eq!(report.campaign_count, 2);
        assert_eq!(report.pinned_file_count, 8);
    }

    #[test]
    fn rejects_a_changed_upstream_commit() {
        let mut value: Value = serde_json::from_str(MANIFEST).expect("manifest JSON parses");
        value["upstream"]["commit"] = Value::String("0".repeat(40));
        let manifest: Manifest = serde_json::from_value(value).expect("mutated manifest parses");
        let report = validate_manifest(&manifest);
        assert!(report.issues.iter().any(|issue| issue.contains("upstream repository, commit")));
    }

    #[test]
    fn rejects_a_weakened_discovery_gate() {
        let mut value: Value = serde_json::from_str(MANIFEST).expect("manifest JSON parses");
        value["campaigns"][0]["gates"]["discovery"][0]["threshold"] = Value::from(1);
        let manifest: Manifest = serde_json::from_value(value).expect("mutated manifest parses");
        let report = validate_manifest(&manifest);
        assert!(report.issues.iter().any(|issue| issue.contains("pinned replay task contract changed")));
    }

    #[test]
    fn rejects_a_twin_mislabeled_as_live_measurement() {
        let mut value: Value = serde_json::from_str(MANIFEST).expect("manifest JSON parses");
        value["campaigns"][1]["observation_semantics"]["live_laboratory_measurements"] = Value::Bool(true);
        let manifest: Manifest = serde_json::from_value(value).expect("mutated manifest parses");
        let report = validate_manifest(&manifest);
        assert!(report.issues.iter().any(|issue| issue.contains("twin measurement provenance changed")));
    }

    #[test]
    fn rejects_budget_mismatch() {
        let mut value: Value = serde_json::from_str(MANIFEST).expect("manifest JSON parses");
        value["campaigns"][0]["design_space"]["total_measurement_budget"] = Value::from(383);
        let manifest: Manifest = serde_json::from_value(value).expect("mutated manifest parses");
        let report = validate_manifest(&manifest);
        assert!(report.issues.iter().any(|issue| issue.contains("rounds × batch size")));
    }
}
