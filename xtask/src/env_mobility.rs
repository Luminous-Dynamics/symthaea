use anyhow::{ensure, Context, Result};
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

const FEASIBILITY_PATH: &str = "docs/research/fixtures/env-mobility-feasibility-v1.json";
const REPAIRABILITY_PATH: &str = "docs/research/fixtures/env-mobility-repairability-v1.json";

const FEASIBILITY_FIELDS: &[&str] = &[
    "site_authority",
    "ecological_constraint",
    "environment_envelope",
    "sensor_calibration",
    "coordinate_reference",
    "energy_reserve",
    "recovery_plan",
    "payload_compatibility",
    "local_failsafe",
    "communications",
    "offline_policy",
    "data_rights",
    "observation_source",
    "requested_claim",
    "source_status",
    "configuration_match",
    "reference_quality",
    "baseline",
    "common_mode_assumption",
    "recovery_history",
];

const REPAIRABILITY_FIELDS: &[&str] = &[
    "intent",
    "source_format",
    "bom_status",
    "component_qualification",
    "article_history",
    "repair_access",
    "routine_tools",
    "post_repair_check",
    "calibration",
    "configuration_record",
    "fault_history",
    "serviceability_target",
];

const CASE_FIELDS: &[&str] = &[
    "case_id",
    "mission_family",
    "platform_family",
    "scenario",
    "facts",
    "expected_disposition",
    "expected_reason_codes",
];

const REPAIR_CASE_FIELDS: &[&str] = &[
    "case_id",
    "component_family",
    "criticality_profile",
    "scenario",
    "facts",
    "expected_disposition",
    "expected_reason_codes",
];

const FEASIBILITY_CORPUS_FIELDS: &[&str] = &[
    "corpus_id",
    "schema_version",
    "status",
    "generated_on",
    "source_class",
    "claim_ceiling",
    "oracle_rule_order",
    "disposition_vocabulary",
    "expected_outcomes_are_independent_test_labels",
    "implementation_binding",
    "cases",
];

const REPAIRABILITY_CORPUS_FIELDS: &[&str] = &[
    "corpus_id",
    "schema_version",
    "status",
    "generated_on",
    "source_class",
    "claim_ceiling",
    "disposition_vocabulary",
    "rule_order",
    "independent_verifier_requirement",
    "cases",
];

#[derive(Debug, Clone, Serialize)]
struct CaseResult {
    corpus: String,
    case_id: String,
    expected_disposition: String,
    expected_reason_codes: Vec<String>,
    derived_disposition: String,
    derived_reason_codes: Vec<String>,
    matches: bool,
    protected_gate_false_accept: bool,
}

#[derive(Debug, Serialize)]
struct CorpusSummary {
    corpus_id: String,
    schema_version: String,
    path: String,
    raw_sha256: String,
    case_count: usize,
    unique_case_ids: bool,
    expected_dispositions: BTreeMap<String, usize>,
    derived_dispositions: BTreeMap<String, usize>,
    mismatch_count: usize,
}

#[derive(Debug, Serialize)]
pub struct QualificationReport {
    report_schema: &'static str,
    claim_ceiling: &'static str,
    feasibility: CorpusSummary,
    repairability: CorpusSummary,
    total_cases: usize,
    pub total_mismatches: usize,
    pub protected_gate_false_accepts: usize,
    cases: Vec<CaseResult>,
}

#[derive(Debug)]
struct Decision {
    disposition: String,
    reason_codes: Vec<String>,
}

impl Decision {
    fn one(disposition: &str, reason: &str) -> Self {
        Self {
            disposition: disposition.to_owned(),
            reason_codes: vec![reason.to_owned()],
        }
    }
}

fn workspace_root() -> Result<PathBuf> {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(Path::to_path_buf)
        .context("xtask must be located one level below the workspace root")
}

fn string_at<'a>(value: &'a Value, key: &str, context: &str) -> Result<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .with_context(|| format!("{context}: required string field {key:?} is missing or invalid"))
}

fn enum_at(value: &Value, key: &str, allowed: &[&str], context: &str) -> Result<String> {
    let candidate = string_at(value, key, context)?;
    ensure!(
        allowed.contains(&candidate),
        "{context}: unsupported value {candidate:?} for {key:?}; allowed: {allowed:?}"
    );
    Ok(candidate.to_owned())
}

fn array_at<'a>(value: &'a Value, key: &str, context: &str) -> Result<&'a Vec<Value>> {
    value
        .get(key)
        .and_then(Value::as_array)
        .with_context(|| format!("{context}: required array field {key:?} is missing or invalid"))
}

fn ensure_exact_keys(value: &Value, allowed: &[&str], context: &str) -> Result<()> {
    let object = value
        .as_object()
        .with_context(|| format!("{context}: expected a JSON object"))?;
    let mut actual: Vec<&str> = object.keys().map(String::as_str).collect();
    actual.sort_unstable();
    let mut expected = allowed.to_vec();
    expected.sort_unstable();
    ensure!(
        actual == expected,
        "{context}: key set differs from schema; expected {expected:?}, got {actual:?}"
    );
    Ok(())
}

fn add_count(map: &mut BTreeMap<String, usize>, value: &str) {
    *map.entry(value.to_owned()).or_insert(0) += 1;
}

fn require_disposition(value: &str, allowed: &[String], context: &str) -> Result<()> {
    ensure!(
        allowed.iter().any(|item| item.as_str() == value),
        "{context}: disposition {value:?} is absent from the corpus vocabulary"
    );
    Ok(())
}

fn validate_reason_codes(value: &Value, context: &str) -> Result<Vec<String>> {
    let items = array_at(value, "expected_reason_codes", context)?;
    ensure!(!items.is_empty(), "{context}: expected_reason_codes must not be empty");
    let reasons: Vec<String> = items
        .iter()
        .enumerate()
        .map(|(index, item)| {
            item.as_str()
                .map(str::to_owned)
                .with_context(|| format!("{context}: reason code at index {index} must be a string"))
        })
        .collect::<Result<_>>()?;
    let mut seen = HashSet::with_capacity(reasons.len());
    ensure!(
        reasons.iter().all(|reason| seen.insert(reason.clone())),
        "{context}: duplicate expected reason codes"
    );
    Ok(reasons)
}

fn derive_feasibility(facts: &Value, context: &str) -> Result<Decision> {
    ensure_exact_keys(facts, FEASIBILITY_FIELDS, context)?;
    let site = enum_at(facts, "site_authority", &["permitted", "prohibited", "unknown"], context)?;
    let ecology = enum_at(facts, "ecological_constraint", &["within_limit", "violated", "unknown"], context)?;
    let envelope = enum_at(facts, "environment_envelope", &["within", "outside", "unknown"], context)?;
    let calibration = enum_at(facts, "sensor_calibration", &["current", "stale", "missing", "failed"], context)?;
    let coordinate = enum_at(facts, "coordinate_reference", &["known", "ambiguous", "invalid"], context)?;
    let energy = enum_at(facts, "energy_reserve", &["sufficient", "insufficient", "uncertain"], context)?;
    let recovery = enum_at(facts, "recovery_plan", &["viable", "unavailable", "unknown"], context)?;
    let payload = enum_at(facts, "payload_compatibility", &["compatible", "incompatible", "unknown"], context)?;
    let failsafe = enum_at(facts, "local_failsafe", &["present_tested", "absent", "unknown"], context)?;
    let communications = enum_at(facts, "communications", &["available", "partition"], context)?;
    let offline = enum_at(facts, "offline_policy", &["valid", "missing", "unknown"], context)?;
    let rights = enum_at(facts, "data_rights", &["permitted", "prohibited", "unknown"], context)?;
    let source = enum_at(facts, "observation_source", &["measured", "derived", "estimated"], context)?;
    let claim = enum_at(facts, "requested_claim", &["measured", "derived", "estimated"], context)?;
    let source_status = enum_at(facts, "source_status", &["valid", "timeout_as_zero", "stale", "inconsistent"], context)?;
    let configuration = enum_at(facts, "configuration_match", &["exact", "mismatch", "unknown"], context)?;
    let reference_quality = enum_at(facts, "reference_quality", &["adequate", "insufficient", "unknown"], context)?;
    let baseline = enum_at(facts, "baseline", &["robot_needed", "fixed_sensor_sufficient", "manual_preferred"], context)?;
    let common_mode = enum_at(facts, "common_mode_assumption", &["not_claimed", "independence_verified", "shared_but_claimed_independent", "unknown"], context)?;
    let recovery_history = enum_at(facts, "recovery_history", &["clean", "failed_attempts_reported", "failed_attempts_hidden", "unknown"], context)?;

    // Known hard violations take precedence over missing evidence.
    if site == "prohibited" {
        return Ok(Decision::one("reject", "SITE_PERMISSION_DENIED"));
    }
    if rights == "prohibited" {
        return Ok(Decision::one("reject", "DATA_RIGHTS_PROHIBIT_CLAIM"));
    }
    if ecology == "violated" {
        return Ok(Decision::one("reject", "ECOLOGICAL_LIMIT_EXCEEDED"));
    }
    if envelope == "outside" {
        return Ok(Decision::one("reject", "OUTSIDE_DECLARED_ENVELOPE"));
    }
    if coordinate == "invalid" {
        return Ok(Decision::one("reject", "INVALID_COORDINATE_REFERENCE"));
    }
    if energy == "insufficient" {
        return Ok(Decision::one("reject", "ENERGY_RESERVE_INSUFFICIENT"));
    }
    if recovery == "unavailable" {
        return Ok(Decision::one("reject", "NO_VIABLE_RECOVERY"));
    }
    if payload == "incompatible" {
        return Ok(Decision::one("reject", "PAYLOAD_INCOMPATIBLE"));
    }
    if communications == "partition" && failsafe == "absent" {
        return Ok(Decision::one("reject", "LOCAL_FAILSAFE_ABSENT"));
    }
    if source == "estimated" && claim == "measured" {
        return Ok(Decision::one("reject", "ESTIMATE_LAUNDERED_AS_MEASUREMENT"));
    }
    if source_status == "timeout_as_zero" {
        return Ok(Decision::one("reject", "SENSOR_TIMEOUT_COERCED_TO_ZERO"));
    }
    if configuration == "mismatch" {
        return Ok(Decision::one("reject", "CONFIGURATION_EVIDENCE_MISMATCH"));
    }
    if common_mode == "shared_but_claimed_independent" {
        return Ok(Decision::one("reject", "COMMON_MODE_REDUNDANCY_LAUNDERED"));
    }
    if recovery_history == "failed_attempts_hidden" {
        return Ok(Decision::one("reject", "FAILED_RECOVERY_ATTEMPTS_HIDDEN"));
    }

    // Unknown, stale or invalid evidence is not permissive.
    if site == "unknown" {
        return Ok(Decision::one("abstain", "SITE_PERMISSION_UNKNOWN"));
    }
    if ecology == "unknown" {
        return Ok(Decision::one("abstain", "ECOLOGICAL_LIMIT_UNKNOWN"));
    }
    if envelope == "unknown" {
        return Ok(Decision::one("abstain", "ENVIRONMENT_ENVELOPE_UNKNOWN"));
    }
    if calibration != "current" {
        return Ok(Decision::one("abstain", "CALIBRATION_NOT_CURRENT"));
    }
    if coordinate == "ambiguous" {
        return Ok(Decision::one("abstain", "COORDINATE_REFERENCE_AMBIGUOUS"));
    }
    if energy == "uncertain" {
        return Ok(Decision::one("abstain", "ENERGY_RESERVE_UNCERTAIN"));
    }
    if recovery == "unknown" {
        return Ok(Decision::one("abstain", "RECOVERY_PLAN_UNKNOWN"));
    }
    if payload == "unknown" {
        return Ok(Decision::one("abstain", "PAYLOAD_COMPATIBILITY_UNKNOWN"));
    }
    if failsafe == "unknown" {
        return Ok(Decision::one("abstain", "LOCAL_FAILSAFE_UNVERIFIED"));
    }
    if communications == "partition" && offline != "valid" {
        return Ok(Decision::one("abstain", "OFFLINE_POLICY_NOT_ESTABLISHED"));
    }
    if rights == "unknown" {
        return Ok(Decision::one("abstain", "DATA_RIGHTS_UNKNOWN"));
    }
    if source_status != "valid" {
        return Ok(Decision::one("abstain", "SOURCE_DATA_INVALID_OR_STALE"));
    }
    if configuration == "unknown" {
        return Ok(Decision::one("abstain", "CONFIGURATION_MATCH_UNKNOWN"));
    }
    if common_mode == "unknown" {
        return Ok(Decision::one("abstain", "COMMON_MODE_INDEPENDENCE_UNKNOWN"));
    }
    if recovery_history == "unknown" {
        return Ok(Decision::one("abstain", "RECOVERY_HISTORY_UNKNOWN"));
    }
    if reference_quality == "unknown" {
        return Ok(Decision::one("abstain", "REFERENCE_QUALITY_UNKNOWN"));
    }
    if reference_quality == "insufficient" {
        return Ok(Decision::one("abstain", "REFERENCE_QUALITY_INSUFFICIENT"));
    }

    if baseline == "fixed_sensor_sufficient" {
        return Ok(Decision::one("prefer_baseline", "LOWER_COMPLEXITY_BASELINE_SUFFICIENT"));
    }
    if baseline == "manual_preferred" {
        return Ok(Decision::one("prefer_baseline", "MANUAL_BASELINE_PREFERRED"));
    }
    if communications == "partition" {
        return Ok(Decision::one("admit_degraded", "STORE_AND_FORWARD_WITH_LOCAL_FAILSAFE"));
    }
    Ok(Decision::one("admit", "ALL_DECLARED_SYNTHETIC_GATES_SATISFIED"))
}

fn derive_repairability(case: &Value, context: &str) -> Result<Decision> {
    let criticality = enum_at(
        case,
        "criticality_profile",
        &[
            "routine_replaceable",
            "safety_relevant",
            "electrical_interface",
            "measurement_critical",
            "environment_exposed",
            "pressure_boundary",
            "energy_storage_safety_relevant",
        ],
        context,
    )?;
    let facts = case
        .get("facts")
        .with_context(|| format!("{context}: facts object is missing"))?;
    ensure_exact_keys(facts, REPAIRABILITY_FIELDS, context)?;

    let intent = enum_at(facts, "intent", &["repair", "local_fabrication"], context)?;
    let source = enum_at(facts, "source_format", &["editable", "intermediate_only", "missing"], context)?;
    let bom = enum_at(facts, "bom_status", &["exact", "approved_substitution", "unapproved_substitution", "unknown"], context)?;
    let qualification = enum_at(facts, "component_qualification", &["qualified_for_profile", "bench_only", "failed", "unknown"], context)?;
    let history = enum_at(facts, "article_history", &["known", "unknown", "quarantined"], context)?;
    let access = enum_at(facts, "repair_access", &["non_destructive", "destructive", "blocked", "unknown"], context)?;
    let tools = enum_at(facts, "routine_tools", &["within_profile", "specialist_required", "unknown"], context)?;
    let post_check = enum_at(facts, "post_repair_check", &["complete", "missing", "failed"], context)?;
    let calibration = enum_at(facts, "calibration", &["current", "stale", "missing", "failed"], context)?;
    let configuration = enum_at(facts, "configuration_record", &["updated_exact", "stale", "missing"], context)?;
    let fault = enum_at(facts, "fault_history", &["resolved", "recurring_unresolved", "failed_repair_reintroduced", "unknown"], context)?;
    let serviceability = enum_at(
        facts,
        "serviceability_target",
        &["field_replaceable", "specialist_service_only", "qualified_module_swap_only"],
        context,
    )?;

    if history == "quarantined" {
        return Ok(Decision::one("quarantine", "ARTICLE_ALREADY_QUARANTINED"));
    }
    if bom == "unapproved_substitution" {
        return Ok(Decision::one("quarantine", "UNAPPROVED_SUBSTITUTION"));
    }
    if qualification == "failed" {
        return Ok(Decision::one("quarantine", "COMPONENT_QUALIFICATION_FAILED"));
    }
    if history == "unknown" && (criticality == "safety_relevant" || criticality == "pressure_boundary") {
        let reason = if criticality == "pressure_boundary" {
            "PRESSURE_BOUNDARY_HISTORY_UNKNOWN"
        } else {
            "SAFETY_RELEVANT_HISTORY_UNKNOWN"
        };
        return Ok(Decision::one("quarantine", reason));
    }
    if fault == "failed_repair_reintroduced" {
        return Ok(Decision::one("quarantine", "FAILED_REPAIR_REINTRODUCED_WITHOUT_RECHECK"));
    }
    if fault == "recurring_unresolved" {
        return Ok(Decision::one("quarantine", "RECURRING_UNRESOLVED_FAULT"));
    }
    if post_check == "failed" {
        return Ok(Decision::one("quarantine", "POST_REPAIR_CHECK_FAILED"));
    }
    if calibration == "failed" {
        return Ok(Decision::one("quarantine", "CALIBRATION_CHECK_FAILED"));
    }
    if criticality == "pressure_boundary"
        && intent == "local_fabrication"
        && qualification != "qualified_for_profile"
    {
        return Ok(Decision::one("quarantine", "PRESSURE_BOUNDARY_NOT_QUALIFIED_FOR_PROFILE"));
    }
    if criticality == "energy_storage_safety_relevant"
        && intent == "local_fabrication"
        && qualification != "qualified_for_profile"
    {
        return Ok(Decision::one("quarantine", "BATTERY_ASSEMBLY_NOT_QUALIFIED"));
    }

    if intent == "local_fabrication" && source != "editable" {
        return Ok(Decision::one("not_locally_reproducible", "PREFERRED_EDITABLE_SOURCE_UNAVAILABLE"));
    }
    if serviceability == "field_replaceable"
        && (access == "destructive" || access == "blocked" || tools == "specialist_required")
    {
        return Ok(Decision::one("redesign_for_serviceability", "ROUTINE_REPAIR_REQUIRES_DESTRUCTIVE_ACCESS"));
    }
    if configuration != "updated_exact" {
        return Ok(Decision::one("hold_for_retest", "CONFIGURATION_RECORD_STALE"));
    }
    if post_check != "complete" {
        return Ok(Decision::one("hold_for_retest", "POST_REPAIR_CHECK_INCOMPLETE"));
    }
    if calibration != "current" {
        return Ok(Decision::one("hold_for_retest", "CALIBRATION_NOT_CURRENT"));
    }
    if bom == "unknown" {
        return Ok(Decision::one("hold_for_evidence", "SUBSTITUTION_STATUS_UNKNOWN"));
    }
    if qualification == "unknown" || history == "unknown" || access == "unknown" || tools == "unknown" || fault == "unknown" {
        return Ok(Decision::one("hold_for_evidence", "REPAIR_EVIDENCE_INCOMPLETE"));
    }
    Ok(Decision::one("release_to_acceptance", "ALL_DECLARED_REPAIR_GATES_SATISFIED"))
}

fn is_admitting(corpus: &str, disposition: &str) -> bool {
    match corpus {
        "feasibility" => matches!(disposition, "admit" | "admit_degraded" | "prefer_baseline"),
        "repairability" => disposition == "release_to_acceptance",
        _ => false,
    }
}

fn load_and_qualify(
    root: &Path,
    relative_path: &str,
    expected_schema: &str,
    expected_count: usize,
    corpus_kind: &str,
) -> Result<(CorpusSummary, Vec<CaseResult>)> {
    let path = root.join(relative_path);
    let bytes = fs::read(&path).with_context(|| format!("read corpus {}", path.display()))?;
    let raw_sha256 = format!("{:x}", Sha256::digest(&bytes));
    let parsed: Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse corpus {}", path.display()))?;
    let context = relative_path.to_owned();

    ensure_exact_keys(
        &parsed,
        if corpus_kind == "feasibility" { FEASIBILITY_CORPUS_FIELDS } else { REPAIRABILITY_CORPUS_FIELDS },
        &context,
    )?;
    ensure!(
        string_at(&parsed, "schema_version", &context)? == expected_schema,
        "{context}: unsupported schema version"
    );
    ensure!(
        string_at(&parsed, "status", &context)? == "synthetic_frozen_candidate",
        "{context}: this verifier accepts only the frozen synthetic-candidate status"
    );
    for key in ["corpus_id", "generated_on", "source_class", "claim_ceiling"] {
        let field = string_at(&parsed, key, &context)?;
        ensure!(!field.trim().is_empty(), "{context}: metadata field {key:?} must not be empty");
    }
    let rules_key = if corpus_kind == "feasibility" { "oracle_rule_order" } else { "rule_order" };
    let rules = array_at(&parsed, rules_key, &context)?;
    ensure!(!rules.is_empty(), "{context}: decision rule list must not be empty");
    for (index, rule) in rules.iter().enumerate() {
        ensure!(
            rule.as_str().is_some_and(|text| !text.trim().is_empty()),
            "{context}: rule {index} must be a non-empty string"
        );
    }
    if corpus_kind == "feasibility" {
        let binding = string_at(&parsed, "implementation_binding", &context)?;
        ensure!(!binding.trim().is_empty(), "{context}: implementation_binding must not be empty");
    } else {
        let requirement = string_at(&parsed, "independent_verifier_requirement", &context)?;
        ensure!(!requirement.trim().is_empty(), "{context}: independent_verifier_requirement must not be empty");
    }

    let vocabulary: Vec<String> = array_at(&parsed, "disposition_vocabulary", &context)?
        .iter()
        .enumerate()
        .map(|(index, item)| {
            item.as_str().map(str::to_owned).with_context(|| {
                format!("{context}: disposition vocabulary item {index} is not a string")
            })
        })
        .collect::<Result<_>>()?;
    ensure!(!vocabulary.is_empty(), "{context}: empty disposition vocabulary");
    let mut vocabulary_ids = HashSet::with_capacity(vocabulary.len());
    ensure!(
        vocabulary.iter().all(|item| vocabulary_ids.insert(item.clone())),
        "{context}: duplicate disposition vocabulary entries"
    );

    if corpus_kind == "feasibility" {
        ensure!(
            parsed.get("expected_outcomes_are_independent_test_labels").and_then(Value::as_bool) == Some(true),
            "{context}: expected labels must remain independent test labels"
        );
    }

    let cases = array_at(&parsed, "cases", &context)?;
    ensure!(
        cases.len() == expected_count,
        "{context}: expected {expected_count} cases, found {}",
        cases.len()
    );

    let mut ids = HashSet::with_capacity(cases.len());
    let mut results = Vec::with_capacity(cases.len());
    let mut expected_counts = BTreeMap::new();
    let mut derived_counts = BTreeMap::new();

    for (index, case) in cases.iter().enumerate() {
        let case_context = format!("{relative_path} case[{index}]");
        ensure_exact_keys(
            case,
            if corpus_kind == "feasibility" { CASE_FIELDS } else { REPAIR_CASE_FIELDS },
            &case_context,
        )?;
        let case_id = string_at(case, "case_id", &case_context)?.to_owned();
        ensure!(!case_id.trim().is_empty(), "{case_context}: empty case_id");
        ensure!(ids.insert(case_id.clone()), "{case_context}: duplicate case_id {case_id:?}");
        let scenario = string_at(case, "scenario", &case_context)?;
        ensure!(!scenario.trim().is_empty(), "{case_context}: scenario must not be empty");
        if corpus_kind == "feasibility" {
            for key in ["mission_family", "platform_family"] {
                let value = string_at(case, key, &case_context)?;
                ensure!(!value.trim().is_empty(), "{case_context}: field {key:?} must not be empty");
            }
        } else {
            let value = string_at(case, "component_family", &case_context)?;
            ensure!(!value.trim().is_empty(), "{case_context}: component_family must not be empty");
        }
        let facts = case.get("facts").context("case facts missing")?;

        let decision = if corpus_kind == "feasibility" {
            derive_feasibility(facts, &case_context)?
        } else {
            derive_repairability(case, &case_context)?
        };

        let expected_disposition = string_at(case, "expected_disposition", &case_context)?.to_owned();
        require_disposition(&expected_disposition, &vocabulary, &case_context)?;
        require_disposition(&decision.disposition, &vocabulary, &case_context)?;
        let expected_reason_codes = validate_reason_codes(case, &case_context)?;
        let matches = expected_disposition == decision.disposition
            && expected_reason_codes == decision.reason_codes;

        let protected_gate_false_accept = expected_disposition != decision.disposition
            && !is_admitting(corpus_kind, &expected_disposition)
            && is_admitting(corpus_kind, &decision.disposition);

        add_count(&mut expected_counts, &expected_disposition);
        add_count(&mut derived_counts, &decision.disposition);
        results.push(CaseResult {
            corpus: corpus_kind.to_owned(),
            case_id,
            expected_disposition,
            expected_reason_codes,
            derived_disposition: decision.disposition,
            derived_reason_codes: decision.reason_codes,
            matches,
            protected_gate_false_accept,
        });
    }

    let mismatch_count = results.iter().filter(|item| !item.matches).count();
    Ok((
        CorpusSummary {
            corpus_id: string_at(&parsed, "corpus_id", &context)?.to_owned(),
            schema_version: expected_schema.to_owned(),
            path: relative_path.to_owned(),
            raw_sha256,
            case_count: cases.len(),
            unique_case_ids: ids.len() == cases.len(),
            expected_dispositions: expected_counts,
            derived_dispositions: derived_counts,
            mismatch_count,
        },
        results,
    ))
}

pub fn run() -> Result<QualificationReport> {
    let root = workspace_root()?;
    let (feasibility, mut feasibility_cases) =
        load_and_qualify(&root, FEASIBILITY_PATH, "env-mobility-case-v1", 48, "feasibility")?;
    let (repairability, repairability_cases) =
        load_and_qualify(&root, REPAIRABILITY_PATH, "env-mobility-repair-case-v1", 28, "repairability")?;

    let mut cases = Vec::with_capacity(feasibility_cases.len() + repairability_cases.len());
    cases.append(&mut feasibility_cases);
    cases.append(&mut repairability_cases);
    let total_mismatches = cases.iter().filter(|item| !item.matches).count();
    let protected_gate_false_accepts = cases
        .iter()
        .filter(|item| item.protected_gate_false_accept)
        .count();

    Ok(QualificationReport {
        report_schema: "env-mobility-qualifier-report-v1",
        claim_ceiling: "Synthetic fixture conformance only; no real-world feasibility, repair qualification, manufacturing yield, regulatory status or safety is established.",
        feasibility,
        repairability,
        total_cases: cases.len(),
        total_mismatches,
        protected_gate_false_accepts,
        cases,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn read_first_facts(relative_path: &str) -> Value {
        let root = workspace_root().expect("workspace root");
        let source: Value =
            serde_json::from_slice(&fs::read(root.join(relative_path)).expect("read fixture"))
                .expect("parse fixture");
        source["cases"][0]["facts"].clone()
    }

    #[test]
    fn frozen_corpora_match_independent_decision_order() {
        let report = run().expect("qualify frozen corpora");
        assert_eq!(report.feasibility.case_count, 48);
        assert_eq!(report.repairability.case_count, 28);
        assert!(report.feasibility.unique_case_ids);
        assert!(report.repairability.unique_case_ids);
        assert_eq!(report.total_cases, 76);
        assert_eq!(report.total_mismatches, 0, "mismatches: {:?}", report.cases.iter().filter(|c| !c.matches).collect::<Vec<_>>());
        assert_eq!(report.protected_gate_false_accepts, 0);
    }

    #[test]
    fn mission_hard_boundary_mutations_fail_closed() {
        let context = "unit test mission mutation";
        let mut facts = read_first_facts(FEASIBILITY_PATH);
        facts["site_authority"] = json!("prohibited");
        assert_eq!(derive_feasibility(&facts, context).unwrap().disposition, "reject");

        let mut facts = read_first_facts(FEASIBILITY_PATH);
        facts["sensor_calibration"] = json!("stale");
        assert_eq!(derive_feasibility(&facts, context).unwrap().disposition, "abstain");

        let mut facts = read_first_facts(FEASIBILITY_PATH);
        facts["energy_reserve"] = json!("insufficient");
        assert_eq!(derive_feasibility(&facts, context).unwrap().disposition, "reject");

        let mut facts = read_first_facts(FEASIBILITY_PATH);
        facts["observation_source"] = json!("estimated");
        facts["requested_claim"] = json!("measured");
        assert_eq!(derive_feasibility(&facts, context).unwrap().disposition, "reject");

        let mut facts = read_first_facts(FEASIBILITY_PATH);
        facts["common_mode_assumption"] = json!("shared_but_claimed_independent");
        assert_eq!(derive_feasibility(&facts, context).unwrap().disposition, "reject");

        let mut facts = read_first_facts(FEASIBILITY_PATH);
        facts["recovery_history"] = json!("failed_attempts_hidden");
        assert_eq!(derive_feasibility(&facts, context).unwrap().disposition, "reject");

        let mut facts = read_first_facts(FEASIBILITY_PATH);
        facts["source_status"] = json!("stale");
        assert_eq!(derive_feasibility(&facts, context).unwrap().disposition, "abstain");

        let mut facts = read_first_facts(FEASIBILITY_PATH);
        facts["site_authority"] = json!("not-a-valid-state");
        assert!(derive_feasibility(&facts, context).is_err());
    }

    #[test]
    fn repair_mutations_do_not_inherit_acceptance() {
        let context = "unit test repair mutation";
        let mut facts = read_first_facts(REPAIRABILITY_PATH);
        facts["bom_status"] = json!("unapproved_substitution");
        let case = json!({
            "criticality_profile": "routine_replaceable",
            "facts": facts
        });
        assert_eq!(derive_repairability(&case, context).unwrap().disposition, "quarantine");

        let mut facts = read_first_facts(REPAIRABILITY_PATH);
        facts["configuration_record"] = json!("stale");
        let case = json!({
            "criticality_profile": "routine_replaceable",
            "facts": facts
        });
        assert_eq!(derive_repairability(&case, context).unwrap().disposition, "hold_for_retest");

        let mut facts = read_first_facts(REPAIRABILITY_PATH);
        facts["repair_access"] = json!("destructive");
        let case = json!({
            "criticality_profile": "routine_replaceable",
            "facts": facts
        });
        assert_eq!(derive_repairability(&case, context).unwrap().disposition, "redesign_for_serviceability");

        let mut facts = read_first_facts(REPAIRABILITY_PATH);
        facts["fault_history"] = json!("failed_repair_reintroduced");
        let case = json!({
            "criticality_profile": "routine_replaceable",
            "facts": facts
        });
        assert_eq!(derive_repairability(&case, context).unwrap().disposition, "quarantine");
    }
}
