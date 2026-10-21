//! Synthetic guideline source-to-runtime change-impact validation.
//!
//! This research-only prototype parses an explicitly synthetic requirement fixture,
//! verifies its source artifact digests, checks the candidate requirement mapping,
//! and rejects incomplete or inconsistent change-impact records. It is not a
//! clinical guideline compiler or a clinical-validity oracle.

use anyhow::{Context, Result, bail};
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
    oracle_fixture_path: String,
    oracle_fixture_sha256: String,
    oracle_class: String,
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
    source_requirement_sha256: String,
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

#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
struct IndependentOracle {
    schema_version: String,
    status: String,
    clinical_use_authorized: bool,
    oracle_class: String,
    note: String,
    cases: Vec<OracleCase>,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
struct OracleCase {
    case_id: String,
    requirement_id: String,
    case_kind: String,
    input_condition: String,
    expected_predicate: String,
    expected_outcome: String,
    fault_injection: Option<String>,
    oracle_review_record_id: String,
}

#[derive(Debug, Serialize)]
struct ChangeImpactReceipt {
    receipt_version: &'static str,
    status: &'static str,
    contract_sha256: String,
    source_revision_sha: String,
    source_artifact_sha256: BTreeMap<String, String>,
    oracle_sha256: String,
    oracle_case_count: usize,
    known_answer_case_count: usize,
    fault_mutation_case_count: usize,
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
    oracle_sha256: String,
    oracle_case_count: usize,
    known_answer_case_count: usize,
    fault_mutation_case_count: usize,
}

pub fn run(root: &Path, contract_arg: &Path, out_arg: Option<&Path>) -> Result<()> {
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
        oracle_sha256: validated.oracle_sha256,
        oracle_case_count: validated.oracle_case_count,
        known_answer_case_count: validated.known_answer_case_count,
        fault_mutation_case_count: validated.fault_mutation_case_count,
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

fn is_numbered_id(id: &str, prefix: &str) -> bool {
    let Some(suffix) = id.strip_prefix(prefix) else {
        return false;
    };
    suffix.len() == 3 && suffix.bytes().all(|byte| byte.is_ascii_digit())
}

fn is_requirement_id(id: &str) -> bool {
    is_numbered_id(id, "REQ-")
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
        !matches!(
            part,
            Component::ParentDir | Component::RootDir | Component::Prefix(_)
        )
    })
}

fn parse_synthetic_requirements(
    source: &str,
    revision_id: &str,
) -> Result<BTreeMap<String, String>> {
    let mut requirements = BTreeMap::new();
    for (index, line) in source.lines().enumerate() {
        if !line.starts_with("REQ-") {
            continue;
        }
        let (id, text) = line.split_once(':').with_context(|| {
            format!(
                "{revision_id}: line {} has malformed requirement",
                index + 1
            )
        })?;
        if !is_requirement_id(id) {
            bail!(
                "{revision_id}: line {} has invalid requirement ID {id:?}",
                index + 1
            );
        }
        let text = text.trim();
        if text.is_empty() {
            bail!("{revision_id}: line {} has an empty requirement", index + 1);
        }
        if requirements
            .insert(id.to_string(), text.to_string())
            .is_some()
        {
            bail!("{revision_id}: duplicate source requirement ID {id}");
        }
    }
    if requirements.is_empty() {
        bail!("{revision_id}: source fixture contains no synthetic requirements");
    }
    Ok(requirements)
}

fn validate_contract(
    contract: &ChangeImpactContract,
    root: &Path,
) -> Result<ValidatedChangeImpact> {
    if contract.schema_version != CONTRACT_VERSION {
        bail!(
            "unsupported change-impact schema: {}",
            contract.schema_version
        );
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
    if contract.oracle_class != "separate_synthetic_expected_outcomes" {
        bail!("change-impact fixture must declare a separate synthetic expected-outcome oracle");
    }
    if !source_path_is_allowed(&contract.oracle_fixture_path) {
        bail!("oracle fixture path is outside the allowed synthetic fixture directory");
    }
    let oracle_bytes = fs::read(root.join(&contract.oracle_fixture_path))
        .with_context(|| format!("read synthetic oracle {}", contract.oracle_fixture_path))?;
    let oracle_sha256 = sha256_hex(&oracle_bytes);
    if oracle_sha256 != contract.oracle_fixture_sha256 {
        bail!(
            "oracle digest mismatch (declared {}, actual {})",
            contract.oracle_fixture_sha256,
            oracle_sha256
        );
    }
    let oracle: IndependentOracle =
        serde_json::from_slice(&oracle_bytes).context("parse independent synthetic oracle")?;
    if oracle.schema_version != "synthetic-first-aid-change-oracle-v0"
        || oracle.status != SYNTHETIC_STATUS
        || oracle.clinical_use_authorized
        || oracle.oracle_class != contract.oracle_class
        || oracle.note.trim().is_empty()
    {
        bail!("oracle metadata is unsupported or attempts to authorize clinical use");
    }
    if oracle.cases.is_empty() {
        bail!("independent synthetic oracle must contain cases");
    }
    let mut oracle_cases: HashMap<String, OracleCase> = HashMap::new();
    for case in &oracle.cases {
        let prefix = match case.case_kind.as_str() {
            "known_answer" => "KAT-",
            "fault_mutation" => "MUT-",
            other => bail!("{}: unsupported oracle case kind {other}", case.case_id),
        };
        if !is_numbered_id(&case.case_id, prefix)
            || !is_requirement_id(&case.requirement_id)
            || case.input_condition.trim().is_empty()
            || case.expected_predicate.trim().is_empty()
            || case.expected_outcome.trim().is_empty()
            || case.oracle_review_record_id.trim().is_empty()
        {
            bail!("{}: incomplete or malformed oracle case", case.case_id);
        }
        if case.case_kind == "known_answer" && case.fault_injection.is_some() {
            bail!(
                "{}: known-answer case must not declare a fault injection",
                case.case_id
            );
        }
        if case.case_kind == "fault_mutation"
            && case
                .fault_injection
                .as_deref()
                .unwrap_or("")
                .trim()
                .is_empty()
        {
            bail!(
                "{}: mutation case must declare a fault injection",
                case.case_id
            );
        }
        if oracle_cases
            .insert(case.case_id.clone(), case.clone())
            .is_some()
        {
            bail!("duplicate oracle case ID: {}", case.case_id);
        }
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
        if revisions
            .insert(revision.revision_id.clone(), revision)
            .is_some()
        {
            bail!("duplicate source revision ID: {}", revision.revision_id);
        }
        source_artifact_sha256.insert(revision.revision_id.clone(), actual_sha);
        source_requirements.insert(revision.revision_id.clone(), requirements);
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
            bail!(
                "duplicate requirement mapping ID: {}",
                mapping.requirement_id
            );
        }
        if mapping.source_revision_id != contract.to_revision_id {
            bail!(
                "{}: mapping must bind to candidate revision {}",
                mapping.requirement_id,
                contract.to_revision_id
            );
        }
        let candidate_text = to.get(&mapping.requirement_id).with_context(|| {
            format!(
                "{}: mapping references no candidate requirement",
                mapping.requirement_id
            )
        })?;
        let expected_locator = format!(
            "synthetic://{}/{}",
            contract.to_revision_id, mapping.requirement_id
        );
        if mapping.source_locator != expected_locator {
            bail!(
                "{}: source locator does not bind to candidate revision and requirement ID",
                mapping.requirement_id
            );
        }
        let expected_requirement_sha = sha256_hex(candidate_text.as_bytes());
        if mapping.source_requirement_sha256 != expected_requirement_sha {
            bail!(
                "{}: source requirement digest mismatch",
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
            bail!(
                "{}: unknown requirement disposition",
                mapping.requirement_id
            );
        }

        if !mapping.test_ids.is_empty() {
            unique_nonempty(&mapping.test_ids, "test_ids", &mapping.requirement_id)?;
        }
        let mut known_answer_count = 0usize;
        let mut fault_mutation_count = 0usize;
        for test_id in &mapping.test_ids {
            let oracle_case = oracle_cases.get(test_id).with_context(|| {
                format!(
                    "{}: test ID {test_id} is missing from independent oracle",
                    mapping.requirement_id
                )
            })?;
            if oracle_case.requirement_id != mapping.requirement_id {
                bail!(
                    "{}: test ID {test_id} belongs to requirement {}, not this mapping",
                    mapping.requirement_id,
                    oracle_case.requirement_id
                );
            }
            match oracle_case.case_kind.as_str() {
                "known_answer" => known_answer_count += 1,
                "fault_mutation" => fault_mutation_count += 1,
                _ => unreachable!("oracle case kind was validated above"),
            }
        }
        match mapping.disposition.as_str() {
            "accepted" => {
                if mapping
                    .bundle_target
                    .as_deref()
                    .unwrap_or("")
                    .trim()
                    .is_empty()
                {
                    bail!(
                        "{}: accepted mapping requires a bundle target",
                        mapping.requirement_id
                    );
                }
                if mapping
                    .reviewer_record_id
                    .as_deref()
                    .unwrap_or("")
                    .trim()
                    .is_empty()
                {
                    bail!(
                        "{}: accepted mapping requires a review record",
                        mapping.requirement_id
                    );
                }
                if mapping.test_ids.is_empty() {
                    bail!(
                        "{}: accepted mapping requires at least one test ID",
                        mapping.requirement_id
                    );
                }
                if mapping.criticality == "safety_critical"
                    && (known_answer_count == 0 || fault_mutation_count == 0)
                {
                    bail!(
                        "{}: safety-critical mapping needs known-answer and fault-mutation cases",
                        mapping.requirement_id
                    );
                }
            }
            "excluded" => {
                if mapping
                    .reviewer_record_id
                    .as_deref()
                    .unwrap_or("")
                    .trim()
                    .is_empty()
                    || mapping
                        .exclusion_rationale
                        .as_deref()
                        .unwrap_or("")
                        .trim()
                        .is_empty()
                {
                    bail!(
                        "{}: exclusion requires a reviewer record and explicit rationale",
                        mapping.requirement_id
                    );
                }
                if mapping.bundle_target.is_some() {
                    bail!(
                        "{}: excluded mapping must not have a bundle target",
                        mapping.requirement_id
                    );
                }
                if !mapping.test_ids.is_empty() {
                    bail!(
                        "{}: excluded mapping must not claim accepted test coverage",
                        mapping.requirement_id
                    );
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
            "candidate mapping coverage mismatch: source IDs {}, mapping IDs {}",
            candidate_ids.len(),
            mapped_ids.len()
        );
    }

    let mut change_ids = HashSet::new();
    for record in &contract.change_records {
        if !is_requirement_id(&record.requirement_id) {
            bail!(
                "invalid change-record requirement ID: {}",
                record.requirement_id
            );
        }
        if !change_ids.insert(record.requirement_id.as_str()) {
            bail!("duplicate change-impact record: {}", record.requirement_id);
        }
        let expected_kind = expected_changes
            .get(&record.requirement_id)
            .with_context(|| {
                format!(
                    "change record {} is not a source diff",
                    record.requirement_id
                )
            })?;
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
        if record
            .reviewer_record_id
            .as_deref()
            .unwrap_or("")
            .trim()
            .is_empty()
        {
            bail!(
                "{}: change record requires a review record",
                record.requirement_id
            );
        }

        match expected_kind.as_str() {
            "changed" | "added" => {
                let mapping = mappings
                    .get(record.requirement_id.as_str())
                    .with_context(|| {
                        format!("{} has no candidate mapping", record.requirement_id)
                    })?;
                match mapping.disposition.as_str() {
                    "accepted" => {
                        if record.disposition != "retested" {
                            bail!(
                                "{}: accepted changes must be retested",
                                record.requirement_id
                            );
                        }
                        if mapping.criticality == "safety_critical" && record.case_ids.len() < 2 {
                            bail!(
                                "{}: changed safety-critical requirement needs two impact cases",
                                record.requirement_id
                            );
                        }
                        for case_id in &record.case_ids {
                            if !mapping.test_ids.contains(case_id) {
                                bail!(
                                    "{}: change case {case_id} is absent from requirement test IDs",
                                    record.requirement_id
                                );
                            }
                            let oracle_case = oracle_cases.get(case_id).with_context(|| {
                                format!("change case {case_id} missing from oracle")
                            })?;
                            if oracle_case.requirement_id != record.requirement_id {
                                bail!(
                                    "{}: change case {case_id} is bound to a different requirement",
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
                        bail!(
                            "{}: unresolved changed requirement cannot be released",
                            record.requirement_id
                        );
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
        oracle_sha256,
        oracle_case_count: oracle.cases.len(),
        known_answer_case_count: oracle
            .cases
            .iter()
            .filter(|case| case.case_kind == "known_answer")
            .count(),
        fault_mutation_case_count: oracle
            .cases
            .iter()
            .filter(|case| case.case_kind == "fault_mutation")
            .count(),
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
        assert_eq!(validated.oracle_case_count, 6);
        assert_eq!(validated.known_answer_case_count, 3);
        assert_eq!(validated.fault_mutation_case_count, 3);
        assert_eq!(
            validated.oracle_sha256,
            "28ebb0721856de30088c40dc0aab385078c5b5e7401646ecae8285f29dbe5bef"
        );
    }

    #[test]
    fn mutated_source_digest_is_rejected() {
        let mut contract = fixture();
        contract.source_revisions[1].source_sha256 = "0".repeat(64);
        assert_rejected(&contract);
    }

    #[test]
    fn tampered_oracle_digest_is_rejected() {
        let mut contract = fixture();
        contract.oracle_fixture_sha256 = "0".repeat(64);
        assert_rejected(&contract);
    }

    #[test]
    fn mapping_test_id_must_exist_in_the_oracle() {
        let mut contract = fixture();
        contract.requirements[0].test_ids[0] = "KAT-999".to_string();
        assert_rejected(&contract);
    }

    #[test]
    fn mapping_test_must_belong_to_the_same_requirement() {
        let mut contract = fixture();
        contract.requirements[0].test_ids[0] = "KAT-002".to_string();
        assert_rejected(&contract);
    }

    #[test]
    fn oracle_class_cannot_authorize_clinical_use() {
        let mut contract = fixture();
        contract.oracle_class = "clinical_authority".to_string();
        assert_rejected(&contract);
    }

    #[test]
    fn omitted_source_change_record_is_rejected() {
        let mut contract = fixture();
        contract
            .change_records
            .retain(|record| record.requirement_id != "REQ-003");
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
    fn requirement_text_digest_mismatch_is_rejected() {
        let mut contract = fixture();
        contract.requirements[1].source_requirement_sha256 = "0".repeat(64);
        assert_rejected(&contract);
    }

    #[test]
    fn explicit_exclusion_is_allowed_only_with_rationale() {
        let mut contract = fixture();
        let mapping = contract
            .requirements
            .iter_mut()
            .find(|mapping| mapping.requirement_id == "REQ-003")
            .expect("REQ-003 mapping exists");
        mapping.disposition = "excluded".to_string();
        mapping.bundle_target = None;
        mapping.test_ids.clear();
        mapping.reviewer_record_id = Some("SYNTHETIC-REVIEW-EXCLUSION".to_string());
        mapping.exclusion_rationale =
            Some("Excluded in this software-only fixture by design".to_string());

        let record = contract
            .change_records
            .iter_mut()
            .find(|record| record.requirement_id == "REQ-003")
            .expect("REQ-003 change record exists");
        record.disposition = "explicit_exclusion".to_string();
        record.case_ids.clear();
        record.rationale = Some("Reviewed software-only exclusion".to_string());

        assert!(validate_contract(&contract, &root()).is_ok());
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
        contract
            .requirements
            .retain(|mapping| mapping.requirement_id != "REQ-003");
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
