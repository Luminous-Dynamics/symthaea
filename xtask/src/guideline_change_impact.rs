//! Synthetic guideline source-to-runtime change-impact validation.
//!
//! This research-only prototype parses an explicitly synthetic requirement fixture,
//! verifies its source artifact digests, checks the candidate requirement mapping,
//! and rejects incomplete or inconsistent change-impact records. It is not a
//! clinical guideline compiler or a clinical-validity oracle.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs,
    path::{Component, Path, PathBuf},
};

const CONTRACT_VERSION: &str = "first-aid-change-impact-v0";
const SYNTHETIC_STATUS: &str = "synthetic_contract_tests_only";

#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
struct ChangeImpactContract {
    schema_version: String,
    status: String,
    clinical_use_authorized: bool,
    release_candidate: bool,
    from_revision_id: String,
    to_revision_id: String,
    source_revisions: Vec<SourceRevision>,
    requirements: Vec<RequirementMapping>,
    change_records: Vec<ChangeRecord>,
    notes: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
struct SourceRevision {
    revision_id: String,
    source_path: String,
    source_sha256: String,
    jurisdiction: String,
    intended_use: String,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
struct RequirementMapping {
    requirement_id: String,
    source_revision_id: String,
    source_locator: String,
    criticality: String,
    disposition: String,
    bundle_target: Option<String>,
    test_ids: Vec<String>,
    reviewer_record_id: Option<String>,
    exclusion_rationale: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
struct ChangeRecord {
    requirement_id: String,
    change_kind: String,
    disposition: String,
    case_ids: Vec<String>,
    reviewer_record_id: Option<String>,
    rationale: Option<String>,
}

#[derive(Debug, Serialize)]
struct ChangeImpactReceipt {
    receipt_version: &'static str,
    status: &'static str,
    contract_sha256: String,
    source_revision_sha: String,
    source_artifact_sha256: BTreeMap<String, String>,
    validator_source_sha256: String,
    entrypoint_source_sha256: String,
    executable_sha256: String,
    runtime_os: String,
    runtime_arch: String,
    from_revision_id: String,
    to_revision_id: String,
    baseline_requirement_count: usize,
    candidate_requirement_count: usize,
    mapped_requirement_count: usize,
    change_record_count: usize,
    claim_ceiling: &'static str,
}

#[derive(Debug)]
struct ValidatedChangeImpact {
    source_artifact_sha256: BTreeMap<String, String>,
    from_revision_id: String,
    to_revision_id: String,
    baseline_requirement_count: usize,
    candidate_requirement_count: usize,
    mapped_requirement_count: usize,
    change_record_count: usize,
}

pub fn run(
    root: &Path,
    contract_arg: &Path,
    out_arg: Option<&Path>,
) -> Result<()> {
    let contract_path = resolve(root, contract_arg);
    let contract_bytes = fs::read(&contract_path)
        .with_context(|| format!("read change-impact contract {}", contract_path.display()))?;
    let contract: ChangeImpactContract =
        serde_json::from_slice(&contract_bytes).context("parse change-impact contract")?;
    let validated = validate_contract(&contract, root)?;

    let validator_source = fs::read(root.join("xtask/src/guideline_change_impact.rs"))
        .context("read change-impact validator source for receipt identity")?;
    let entrypoint_source = fs::read(root.join("xtask/src/main.rs"))
        .context("read xtask entrypoint source for receipt identity")?;
    let executable_path = std::env::current_exe().context("resolve validator executable")?;
    let executable_bytes = fs::read(&executable_path)
        .with_context(|| format!("read validator executable {}", executable_path.display()))?;

    let receipt = ChangeImpactReceipt {
        receipt_version: "first-aid-change-impact-receipt-v0",
        status: "synthetic_change_impact_contract_passed",
        contract_sha256: sha256_hex(&contract_bytes),
        source_revision_sha: std::env::var("FIRST_AID_SOURCE_REVISION")
            .unwrap_or_else(|_| "not-provided".to_string()),
        source_artifact_sha256: validated.source_artifact_sha256,
        validator_source_sha256: sha256_hex(&validator_source),
        entrypoint_source_sha256: sha256_hex(&entrypoint_source),
        executable_sha256: sha256_hex(&executable_bytes),
        runtime_os: std::env::consts::OS.to_string(),
        runtime_arch: std::env::consts::ARCH.to_string(),
        from_revision_id: validated.from_revision_id,
        to_revision_id: validated.to_revision_id,
        baseline_requirement_count: validated.baseline_requirement_count,
        candidate_requirement_count: validated.candidate_requirement_count,
        mapped_requirement_count: validated.mapped_requirement_count,
        change_record_count: validated.change_record_count,
        claim_ceiling: "synthetic source-change mapping integrity only; no clinical validation",
    };
    let encoded = serde_json::to_string_pretty(&receipt)?;

    if let Some(out_arg) = out_arg {
        let out_path = resolve(root, out_arg);
        if let Some(parent) = out_path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)
                .with_context(|| format!("create receipt directory {}", parent.display()))?;
        }
        fs::write(&out_path, format!("{encoded}\n"))
            .with_context(|| format!("write change-impact receipt {}", out_path.display()))?;
        println!("Receipt written: {}", out_path.display());
    }
    println!("{encoded}");
    Ok(())
}

fn resolve(root: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn is_requirement_id(id: &str) -> bool {
    let Some(suffix) = id.strip_prefix("REQ-") else {
        return false;
    };
    suffix.len() == 3 && suffix.bytes().all(|byte| byte.is_ascii_digit())
}

fn unique_nonempty(values: &[String], field: &str, owner: &str) -> Result<()> {
    let mut seen = HashSet::new();
    for value in values {
        if value.trim().is_empty() {
            bail!("{owner}: {field} contains an empty value");
        }
        if !seen.insert(value) {
            bail!("{owner}: duplicate value {value:?} in {field}");
        }
    }
    Ok(())
}

fn source_path_is_allowed(path: &str) -> bool {
    if !path.starts_with("docs/research/first-aid/fixtures/") {
        return false;
    }
    let parsed = Path::new(path);
    if parsed.is_absolute() {
        return false;
    }
    parsed.components().all(|part| {
        !matches!(part, Component::ParentDir | Component::RootDir | Component::Prefix(_))
    })
}

fn parse_synthetic_requirements(source: &str, revision_id: &str) -> Result<BTreeMap<String, String>> {
    let mut requirements = BTreeMap::new();
    for (index, line) in source.lines().enumerate() {
        if !line.starts_with("REQ-") {
            continue;
        }
        let (id, text) = line
            .split_once(':')
            .with_context(|| format!("{revision_id}: line {} has malformed requirement", index + 1))?;
        if !is_requirement_id(id) {
            bail!("{revision_id}: line {} has invalid requirement ID {id:?}", index + 1);
        }
        let text = text.trim();
        if text.is_empty() {
            bail!("{revision_id}: line {} has an empty requirement", index + 1);
        }
        if requirements.insert(id.to_string(), text.to_string()).is_some() {
            bail!("{revision_id}: duplicate source requirement ID {id}");
        }
    }
    if requirements.is_empty() {
        bail!("{revision_id}: source fixture contains no synthetic requirements");
    }
    Ok(requirements)
}

fn validate_contract(contract: &ChangeImpactContract, root: &Path) -> Result<ValidatedChangeImpact> {
    if contract.schema_version != CONTRACT_VERSION {
        bail!("unsupported change-impact schema: {}", contract.schema_version);
    }
    if contract.status != SYNTHETIC_STATUS {
        bail!("change-impact fixture must remain synthetic-only");
    }
    if contract.clinical_use_authorized {
        bail!("synthetic change-impact fixture cannot authorize clinical use");
    }
    if contract.from_revision_id.trim().is_empty()
        || contract.to_revision_id.trim().is_empty()
        || contract.from_revision_id == contract.to_revision_id
    {
        bail!("from/to source revision IDs must be nonempty and different");
    }
    if contract.source_revisions.len() < 2 {
        bail!("change-impact validation requires two distinct source revisions");
    }

    let mut revisions = HashMap::new();
    let mut source_artifact_sha256 = BTreeMap::new();
    let mut source_requirements = HashMap::new();
    for revision in &contract.source_revisions {
        if revision.revision_id.trim().is_empty() {
            bail!("source revision ID must not be empty");
        }
        if revision.jurisdiction != "SYNTHETIC"
            || revision.intended_use != "validator-contract-test-only"
        {
            bail!(
                "{}: v0 accepts only explicitly synthetic source fixtures",
                revision.revision_id
            );
        }
        if !source_path_is_allowed(&revision.source_path) {
            bail!(
                "{}: source path is outside the allowed synthetic fixture directory",
                revision.revision_id
            );
        }
        if revisions.contains_key(&revision.revision_id) {
            bail!("duplicate source revision ID: {}", revision.revision_id);
        }

        let bytes = fs::read(root.join(&revision.source_path))
            .with_context(|| format!("read synthetic source {}", revision.source_path))?;
        let actual_sha = sha256_hex(&bytes);
        if actual_sha != revision.source_sha256 {
            bail!(
                "{}: source digest mismatch (declared {}, actual {})",
                revision.revision_id,
                revision.source_sha256,
                actual_sha
            );
        }
        let text = std::str::from_utf8(&bytes)
            .with_context(|| format!("{} is not valid UTF-8", revision.source_path))?;
        let requirements = parse_synthetic_requirements(text, &revision.revision_id)?;
        source_artifact_sha256.insert(revision.revision_id.clone(), actual_sha);
        source_requirements.insert(revision.revision_id.clone(), requirements);
        revisions.insert(revision.revision_id.clone(), revision);
    }

    if !revisions.contains_key(&contract.from_revision_id) {
        bail!("unknown from_revision_id: {}", contract.from_revision_id);
    }
    if !revisions.contains_key(&contract.to_revision_id) {
        bail!("unknown to_revision_id: {}", contract.to_revision_id);
    }

    let from = source_requirements
        .get(&contract.from_revision_id)
        .context("from revision requirement parse missing")?;
    let to = source_requirements
        .get(&contract.to_revision_id)
        .context("to revision requirement parse missing")?;

    // Diff requirement IDs and exact requirement text in the candidate source.
    let mut expected_changes: BTreeMap<String, String> = BTreeMap::new();
    for (id, text) in to {
        match from.get(id) {
            None => {
                expected_changes.insert(id.clone(), "added".to_string());
            }
            Some(old_text) if old_text != text => {
                expected_changes.insert(id.clone(), "changed".to_string());
            }
            Some(_) => {}
        }
    }
    for id in from.keys() {
        if !to.contains_key(id) {
            expected_changes.insert(id.clone(), "removed".to_string());
        }
    }

    let mut mappings: BTreeMap<&str, &RequirementMapping> = BTreeMap::new();
    for mapping in &contract.requirements {
        if !is_requirement_id(&mapping.requirement_id) {
            bail!("invalid requirement mapping ID: {}", mapping.requirement_id);
        }
        if mappings.insert(&mapping.requirement_id, mapping).is_some() {
            bail!("duplicate requirement mapping ID: {}", mapping.requirement_id);
        }
        if mapping.source_revision_id != contract.to_revision_id {
            bail!(
                "{}: mapping must bind to candidate revision {}",
                mapping.requirement_id,
                contract.to_revision_id
            );
        }
        if mapping.source_locator.trim().is_empty() {
            bail!("{}: source locator must not be empty", mapping.requirement_id);
        }
        if !to.contains_key(&mapping.requirement_id) {
            bail!(
                "{}: mapping references no requirement in candidate source",
                mapping.requirement_id
            );
        }
        if !matches!(mapping.criticality.as_str(), "safety_critical" | "standard") {
            bail!("{}: unknown criticality", mapping.requirement_id);
        }
        if !matches!(
            mapping.disposition.as_str(),
            "accepted" | "excluded" | "unresolved"
        ) {
            bail!("{}: unknown requirement disposition", mapping.requirement_id);
        }

        if !mapping.test_ids.is_empty() {
            unique_nonempty(&mapping.test_ids, "test_ids", &mapping.requirement_id)?;
        }
        match mapping.disposition.as_str() {
            "accepted" => {
                if mapping.bundle_target.as_deref().unwrap_or("").trim().is_empty() {
                    bail!("{}: accepted mapping requires a bundle target", mapping.requirement_id);
                }
                if mapping.reviewer_record_id.as_deref().unwrap_or("").trim().is_empty() {
                    bail!("{}: accepted mapping requires a review record", mapping.requirement_id);
                }
                if mapping.test_ids.is_empty() {
                    bail!("{}: accepted mapping requires at least one test ID", mapping.requirement_id);
                }
                if mapping.criticality == "safety_critical" && mapping.test_ids.len() < 2 {
                    bail!(
                        "{}: safety-critical mapping requires at least two independently identified test IDs",
                        mapping.requirement_id
                    );
                }
            }
            "excluded" => {
                if mapping.reviewer_record_id.as_deref().unwrap_or("").trim().is_empty()
                    || mapping.exclusion_rationale.as_deref().unwrap_or("").trim().is_empty()
                {
                    bail!(
                        "{}: exclusion requires a reviewer record and explicit rationale",
                        mapping.requirement_id
                    );
                }
                if mapping.bundle_target.is_some() {
                    bail!("{}: excluded mapping must not have a bundle target", mapping.requirement_id);
                }
                if !mapping.test_ids.is_empty() {
                    bail!("{}: excluded mapping must not claim accepted test coverage", mapping.requirement_id);
                }
            }
            "unresolved" => {
                if contract.release_candidate {
                    bail!(
                        "{}: unresolved mapping blocks release-candidate validation",
                        mapping.requirement_id
                    );
                }
            }
            _ => unreachable!("disposition enum validated above"),
        }
    }

    let candidate_ids: HashSet<&str> = to.keys().map(String::as_str).collect();
    let mapped_ids: HashSet<&str> = mappings.keys().copied().collect();
    if candidate_ids != mapped_ids {
        bail!(
            "candidate source/mapping coverage mismatch: source has {} requirement IDs, mappings have {}",
            candidate_ids.len(),
            mapped_ids.len()
        );
    }

    let mut change_ids = HashSet::new();
    for record in &contract.change_records {
        if !is_requirement_id(&record.requirement_id) {
            bail!("invalid change-record requirement ID: {}", record.requirement_id);
        }
        if !change_ids.insert(record.requirement_id.as_str()) {
            bail!("duplicate change-impact record: {}", record.requirement_id);
        }
        let expected_kind = expected_changes
            .get(&record.requirement_id)
            .with_context(|| format!("change record {} is not a source diff", record.requirement_id))?;
        if expected_kind != &record.change_kind {
            bail!(
                "{}: declared change kind {} differs from source diff {}",
                record.requirement_id,
                record.change_kind,
                expected_kind
            );
        }
        if !record.case_ids.is_empty() {
            unique_nonempty(&record.case_ids, "case_ids", &record.requirement_id)?;
        }
        if record.reviewer_record_id.as_deref().unwrap_or("").trim().is_empty() {
            bail!("{}: change record requires a review record", record.requirement_id);
        }

        match expected_kind.as_str() {
            "changed" | "added" => {
                let mapping = mappings
                    .get(record.requirement_id.as_str())
                    .with_context(|| format!("{} has no candidate mapping", record.requirement_id))?;
                match mapping.disposition.as_str() {
                    "accepted" => {
                        if record.disposition != "retested" {
                            bail!("{}: accepted changes must be retested", record.requirement_id);
                        }
                        if mapping.criticality == "safety_critical" && record.case_ids.len() < 2 {
                            bail!(
                                "{}: changed safety-critical requirement needs at least two impact cases",
                                record.requirement_id
                            );
                        }
                        for case_id in &record.case_ids {
                            if !mapping.test_ids.contains(case_id) {
                                bail!(
                                    "{}: change case {case_id} is not in the requirement's test IDs",
                                    record.requirement_id
                                );
                            }
                        }
                    }
                    "excluded" => {
                        if record.disposition != "explicit_exclusion"
                            || record.rationale.as_deref().unwrap_or("").trim().is_empty()
                        {
                            bail!(
                                "{}: excluded source change requires an explicit exclusion record",
                                record.requirement_id
                            );
                        }
                    }
                    "unresolved" => {
                        bail!("{}: unresolved changed requirement cannot be released", record.requirement_id);
                    }
                    _ => unreachable!("disposition enum validated above"),
                }
            }
            "removed" => {
                if mappings.contains_key(record.requirement_id.as_str()) {
                    bail!(
                        "{}: removed source requirement must not have a candidate mapping",
                        record.requirement_id
                    );
                }
                if record.disposition != "explicit_removal_review"
                    || record.rationale.as_deref().unwrap_or("").trim().is_empty()
                {
                    bail!(
                        "{}: removed requirement needs explicit reviewer decision and rationale",
                        record.requirement_id
                    );
                }
            }
            _ => bail!("unsupported source change kind: {expected_kind}"),
        }
    }

    let declared_change_ids: HashSet<&str> = expected_changes.keys().map(String::as_str).collect();
    if change_ids != declared_change_ids {
        bail!(
            "change-impact coverage mismatch: source diff has {} changed IDs, receipts have {}",
            declared_change_ids.len(),
            change_ids.len()
        );
    }

    let _ = &contract.notes;
    Ok(ValidatedChangeImpact {
        source_artifact_sha256,
        from_revision_id: contract.from_revision_id.clone(),
        to_revision_id: contract.to_revision_id.clone(),
        baseline_requirement_count: from.len(),
        candidate_requirement_count: to.len(),
        mapped_requirement_count: contract.requirements.len(),
        change_record_count: contract.change_records.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONTRACT_JSON: &str =
        include_str!("../../docs/research/first-aid/guideline-change-impact-v0.json");

    fn root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("xtask is one level below workspace root")
            .to_path_buf()
    }

    fn fixture() -> ChangeImpactContract {
        serde_json::from_str(CONTRACT_JSON).expect("checked-in change-impact fixture parses")
    }

    fn assert_rejected(contract: &ChangeImpactContract) {
        assert!(validate_contract(contract, &root()).is_err());
    }

    #[test]
    fn checked_in_source_diff_and_mapping_are_consistent() {
        let validated = validate_contract(&fixture(), &root()).expect("fixture should validate");
        assert_eq!(validated.source_artifact_sha256.len(), 2);
        assert_eq!(validated.baseline_requirement_count, 2);
        assert_eq!(validated.candidate_requirement_count, 3);
        assert_eq!(validated.mapped_requirement_count, 3);
        assert_eq!(validated.change_record_count, 2);
    }

    #[test]
    fn mutated_source_digest_is_rejected() {
        let mut contract = fixture();
        contract.source_revisions[1].source_sha256 = "0".repeat(64);
        assert_rejected(&contract);
    }

    #[test]
    fn omitted_source_change_record_is_rejected() {
        let mut contract = fixture();
        contract.change_records.retain(|record| record.requirement_id != "REQ-003");
        assert_rejected(&contract);
    }

    #[test]
    fn wrong_change_kind_is_rejected() {
        let mut contract = fixture();
        contract
            .change_records
            .iter_mut()
            .find(|record| record.requirement_id == "REQ-002")
            .expect("REQ-002 change record exists")
            .change_kind = "added".to_string();
        assert_rejected(&contract);
    }

    #[test]
    fn unresolved_mapping_blocks_release_candidate() {
        let mut contract = fixture();
        contract
            .requirements
            .iter_mut()
            .find(|mapping| mapping.requirement_id == "REQ-002")
            .expect("REQ-002 mapping exists")
            .disposition = "unresolved".to_string();
        assert_rejected(&contract);
    }

    #[test]
    fn safety_critical_mapping_needs_independent_test_ids() {
        let mut contract = fixture();
        let mapping = contract
            .requirements
            .iter_mut()
            .find(|mapping| mapping.requirement_id == "REQ-002")
            .expect("REQ-002 mapping exists");
        mapping.test_ids = vec!["KAT-002".to_string()];
        assert_rejected(&contract);
    }

    #[test]
    fn duplicate_requirement_mapping_id_is_rejected() {
        let mut contract = fixture();
        contract.requirements[1].requirement_id = contract.requirements[0].requirement_id.clone();
        assert_rejected(&contract);
    }

    #[test]
    fn clinical_authority_flag_is_rejected() {
        let mut contract = fixture();
        contract.clinical_use_authorized = true;
        assert_rejected(&contract);
    }

    #[test]
    fn candidate_requirement_without_mapping_is_rejected() {
        let mut contract = fixture();
        contract.requirements.retain(|mapping| mapping.requirement_id != "REQ-003");
        assert_rejected(&contract);
    }

    #[test]
    fn source_path_traversal_is_rejected() {
        let mut contract = fixture();
        contract.source_revisions[0].source_path = "../Cargo.toml".to_string();
        assert_rejected(&contract);
    }

    #[test]
    fn duplicate_source_requirement_ids_are_rejected() {
        let source = "SYNTHETIC FIXTURE\nREQ-001: first\nREQ-001: second\n";
        assert!(parse_synthetic_requirements(source, "SYN-TEST").is_err());
    }

    #[test]
    fn missing_change_review_record_is_rejected() {
        let mut contract = fixture();
        contract.change_records[0].reviewer_record_id = None;
        assert_rejected(&contract);
    }
}
