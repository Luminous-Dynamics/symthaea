//! Fail-closed admission checks for EGDDM ShiftBench task manifests.
//!
//! This checks declared two-sided task validity, blind reconstruction,
//! candidate/evaluator separation, clean-room regrading and gold-label custody.
//! It does not hash external artifacts, inspect patches, execute graders, or
//! authenticate adjudicators. It is an admission-contract checker, not proof
//! that a declared gate actually happened.

use serde::{Deserialize, Serialize};
use serde_json::Value;

const VERSION: &str = "0.1.0";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdmissionReport {
    pub valid: bool,
    pub task_id: Option<String>,
    pub scenario_pair_id: Option<String>,
    pub diagnostics: Vec<String>,
    pub claim_ceiling: String,
}

impl AdmissionReport {
    fn new() -> Self {
        Self {
            valid: false,
            task_id: None,
            scenario_pair_id: None,
            diagnostics: Vec::new(),
            claim_ceiling: "declared admission contract only; no patch, test, grader, digest preimage, identity, or custody artifact was independently inspected".to_owned(),
        }
    }
    fn error(&mut self, path: &str, message: impl AsRef<str>) {
        self.diagnostics.push(format!("{path}: {}", message.as_ref()));
    }
    fn finish(mut self) -> Self {
        self.diagnostics.sort();
        self.diagnostics.dedup();
        self.valid = self.diagnostics.is_empty();
        self
    }
}

fn at<'a>(value: &'a Value, path: &[&str]) -> Option<&'a Value> {
    path.iter().try_fold(value, |node, key| node.get(*key))
}
fn str_at<'a>(value: &'a Value, path: &[&str]) -> Option<&'a str> {
    at(value, path).and_then(Value::as_str)
}
fn bool_at(value: &Value, path: &[&str]) -> Option<bool> {
    at(value, path).and_then(Value::as_bool)
}
fn valid_hex(value: &str, len: usize) -> bool {
    value.len() == len && value.bytes().all(|b| b.is_ascii_hexdigit())
}
fn valid_digest(value: Option<&str>) -> bool {
    value.is_some_and(|digest| valid_hex(digest, 64))
}
fn valid_oid(value: &str) -> bool {
    valid_hex(value, 40) || valid_hex(value, 64)
}
fn required_string(value: &Value, path: &[&str], label: &str, report: &mut AdmissionReport) -> Option<String> {
    match str_at(value, path).filter(|s| !s.trim().is_empty()) {
        Some(text) => Some(text.to_owned()),
        None => {
            report.error(label, "must be a non-empty string");
            None
        }
    }
}
fn digest_field(value: &Value, path: &[&str], label: &str, report: &mut AdmissionReport) {
    if !valid_digest(str_at(value, path)) {
        report.error(label, "must be a 64-character SHA-256 hexadecimal digest");
    }
}

const FAMILIES: &[&str] = &[
    "requirement_addition_or_reinterpretation",
    "requirement_removal_or_narrowing",
    "dependency_api_breaking_change",
    "toolchain_or_build_system_upgrade",
    "configuration_world_drift",
    "runtime_or_external_service_change",
    "session_tool_or_execution_context_change",
    "authority_or_policy_revocation",
    "evidence_invalidation_or_contradiction",
    "test_or_acceptance_contract_revision",
    "common_cause_environment_change",
    "cross_repository_contract_or_schema_change",
];

pub fn validate_shiftbench_admission_json(input: &str) -> Result<AdmissionReport, String> {
    let manifest: Value = serde_json::from_str(input)
        .map_err(|error| format!("admission manifest JSON parse failed: {error}"))?;
    let mut report = AdmissionReport::new();

    if str_at(&manifest, &["schema_version"]) != Some(VERSION) {
        report.error("schema_version", "expected 0.1.0");
    }
    let task_id = required_string(&manifest, &["task_id"], "task_id", &mut report);
    report.task_id = task_id;
    let scenario_pair_id = required_string(
        &manifest,
        &["scenario_pair_id"],
        "scenario_pair_id",
        &mut report,
    );
    report.scenario_pair_id = scenario_pair_id;

    let family = str_at(&manifest, &["intervention", "family"]).unwrap_or("");
    if !FAMILIES.contains(&family) {
        report.error("intervention.family", "unsupported intervention family");
    }
    if !["early", "middle", "late"].contains(&str_at(&manifest, &["intervention", "position"]).unwrap_or("")) {
        report.error("intervention.position", "must be early, middle, or late");
    }
    if !["target_intervention", "sham", "positive_control", "negative_control", "common_cause_control"]
        .contains(&str_at(&manifest, &["intervention", "control_class"]).unwrap_or(""))
    {
        report.error("intervention.control_class", "unsupported control class");
    }

    for path in [
        &["task_seed_digest"][..],
        &["task_bundle_digest"][..],
        &["acceptance_contract_digest"][..],
    ] {
        digest_field(&manifest, path, &path.join("."), &mut report);
    }

    let repository = str_at(&manifest, &["repository_snapshot", "repository_full_name"]).unwrap_or("");
    let revision = str_at(&manifest, &["repository_snapshot", "revision_oid"]).unwrap_or("");
    let tree = str_at(&manifest, &["repository_snapshot", "tree_oid"]).unwrap_or("");
    if repository.split('/').count() != 2 || repository.split('/').any(str::is_empty) {
        report.error("repository_snapshot.repository_full_name", "must be owner/repository");
    }
    if !valid_oid(revision) {
        report.error("repository_snapshot.revision_oid", "must be a 40- or 64-character object ID");
    }
    if !valid_oid(tree) || (!revision.is_empty() && tree.len() != revision.len()) {
        report.error("repository_snapshot.tree_oid", "must be an object ID matching revision length");
    }
    digest_field(&manifest, &["repository_snapshot", "configuration_world_digest"], "repository_snapshot.configuration_world_digest", &mut report);

    let reference_result = str_at(&manifest, &["two_sided_validity", "reference_patch", "result"]);
    if reference_result != Some("pass") {
        report.error("two_sided_validity.reference_patch.result", "reference patch must pass");
    }
    if bool_at(&manifest, &["two_sided_validity", "reference_patch", "clean_regrade"]) != Some(true) {
        report.error("two_sided_validity.reference_patch.clean_regrade", "reference patch must pass a pristine clean regrade");
    }
    digest_field(&manifest, &["two_sided_validity", "reference_patch", "patch_digest"], "two_sided_validity.reference_patch.patch_digest", &mut report);
    digest_field(&manifest, &["two_sided_validity", "reference_patch", "evidence_digest"], "two_sided_validity.reference_patch.evidence_digest", &mut report);

    let empty_result = str_at(&manifest, &["two_sided_validity", "empty_patch_control", "result"]);
    if empty_result != Some("fail") {
        report.error("two_sided_validity.empty_patch_control.result", "empty/no-op patch must fail");
    }
    if bool_at(&manifest, &["two_sided_validity", "empty_patch_control", "clean_regrade"]) != Some(true) {
        report.error("two_sided_validity.empty_patch_control.clean_regrade", "empty patch must be graded in a pristine environment");
    }
    digest_field(&manifest, &["two_sided_validity", "empty_patch_control", "patch_digest"], "two_sided_validity.empty_patch_control.patch_digest", &mut report);
    digest_field(&manifest, &["two_sided_validity", "empty_patch_control", "evidence_digest"], "two_sided_validity.empty_patch_control.evidence_digest", &mut report);

    let reference_grader = str_at(&manifest, &["two_sided_validity", "reference_patch", "grader_environment_digest"]);
    let empty_grader = str_at(&manifest, &["two_sided_validity", "empty_patch_control", "grader_environment_digest"]);
    if !valid_digest(reference_grader) || !valid_digest(empty_grader) {
        report.error("two_sided_validity.grader_environment_digest", "both controls require a SHA-256 grader environment digest");
    } else if reference_grader != empty_grader {
        report.error("two_sided_validity.grader_environment_digest", "reference and empty-patch controls must use the same grader environment");
    }

    if bool_at(&manifest, &["blind_reconstruction", "completed"]) != Some(true) {
        report.error("blind_reconstruction.completed", "independent blind reconstruction must be complete");
    }
    if bool_at(&manifest, &["blind_reconstruction", "disagreement_resolved"]) != Some(true) {
        report.error("blind_reconstruction.disagreement_resolved", "adjudicator disagreement must be resolved before admission");
    }
    let author = str_at(&manifest, &["blind_reconstruction", "primary_author_id"]).unwrap_or("");
    if author.is_empty() {
        report.error("blind_reconstruction.primary_author_id", "must be a non-empty pseudonymous identity");
    }
    let adjudicators = manifest.pointer("/blind_reconstruction/adjudicator_ids").and_then(Value::as_array);
    match adjudicators {
        Some(ids) if ids.len() >= 2 => {
            let mut unique = std::collections::HashSet::new();
            for id in ids {
                let Some(id) = id.as_str().filter(|s| !s.trim().is_empty()) else {
                    report.error("blind_reconstruction.adjudicator_ids", "all adjudicator IDs must be non-empty strings");
                    continue;
                };
                if id == author {
                    report.error("blind_reconstruction.adjudicator_ids", "primary task author cannot be an independent adjudicator");
                }
                if !unique.insert(id) {
                    report.error("blind_reconstruction.adjudicator_ids", "adjudicator IDs must be unique");
                }
            }
        }
        _ => report.error("blind_reconstruction.adjudicator_ids", "at least two independent adjudicators are required"),
    }

    digest_field(&manifest, &["isolation", "runner_image_digest"], "isolation.runner_image_digest", &mut report);
    digest_field(&manifest, &["isolation", "network_policy_digest"], "isolation.network_policy_digest", &mut report);
    for (path, label) in [
        (&["isolation", "model_endpoint_only"][..], "isolation.model_endpoint_only"),
        (&["isolation", "candidate_evaluator_separated"][..], "isolation.candidate_evaluator_separated"),
        (&["isolation", "pristine_regrade"][..], "isolation.pristine_regrade"),
    ] {
        if bool_at(&manifest, path) != Some(true) {
            report.error(label, "must be true");
        }
    }
    if bool_at(&manifest, &["isolation", "candidate_can_read_gold"]) != Some(false) {
        report.error("isolation.candidate_can_read_gold", "must be explicitly false");
    }

    digest_field(&manifest, &["gold_custody", "gold_label_manifest_digest"], "gold_custody.gold_label_manifest_digest", &mut report);
    let _custodian_id = required_string(
        &manifest,
        &["gold_custody", "custodian_id"],
        "gold_custody.custodian_id",
        &mut report,
    );
    if bool_at(&manifest, &["gold_custody", "candidate_readable"]) != Some(false) {
        report.error("gold_custody.candidate_readable", "must be explicitly false");
    }
    if bool_at(&manifest, &["gold_custody", "evaluator_readable"]) != Some(true) {
        report.error("gold_custody.evaluator_readable", "must be explicitly true");
    }
    if bool_at(&manifest, &["gold_custody", "labels_separate_from_candidate_bundle"]) != Some(true) {
        report.error("gold_custody.labels_separate_from_candidate_bundle", "must be explicitly true");
    }

    Ok(report.finish())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const VALID: &str = include_str!("../../../docs/research/fixtures/egddm-shiftbench-task-admission-v0.1.synthetic.json");

    fn check(value: &Value) -> AdmissionReport {
        validate_shiftbench_admission_json(&value.to_string()).unwrap()
    }

    #[test]
    fn accepts_complete_two_sided_blind_and_custody_manifest() {
        let report = validate_shiftbench_admission_json(VALID).unwrap();
        assert!(report.valid, "{:?}", report.diagnostics);
    }

    #[test]
    fn rejects_reference_patch_failure() {
        let mut v: Value = serde_json::from_str(VALID).unwrap();
        v["two_sided_validity"]["reference_patch"]["result"] = json!("fail");
        assert!(!check(&v).valid);
    }

    #[test]
    fn rejects_empty_patch_success() {
        let mut v: Value = serde_json::from_str(VALID).unwrap();
        v["two_sided_validity"]["empty_patch_control"]["result"] = json!("pass");
        assert!(!check(&v).valid);
    }

    #[test]
    fn rejects_missing_reference_clean_regrade() {
        let mut v: Value = serde_json::from_str(VALID).unwrap();
        v["two_sided_validity"]["reference_patch"]["clean_regrade"] = json!(false);
        assert!(!check(&v).valid);
    }

    #[test]
    fn rejects_different_grader_environments_for_two_sided_gate() {
        let mut v: Value = serde_json::from_str(VALID).unwrap();
        v["two_sided_validity"]["empty_patch_control"]["grader_environment_digest"] = json!("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
        assert!(!check(&v).valid);
    }

    #[test]
    fn rejects_author_as_adjudicator() {
        let mut v: Value = serde_json::from_str(VALID).unwrap();
        v["blind_reconstruction"]["adjudicator_ids"][0] = json!("task-author-01");
        assert!(!check(&v).valid);
    }

    #[test]
    fn rejects_single_adjudicator() {
        let mut v: Value = serde_json::from_str(VALID).unwrap();
        v["blind_reconstruction"]["adjudicator_ids"] = json!(["reviewer-01"]);
        assert!(!check(&v).valid);
    }

    #[test]
    fn rejects_duplicate_adjudicators() {
        let mut v: Value = serde_json::from_str(VALID).unwrap();
        v["blind_reconstruction"]["adjudicator_ids"] = json!(["reviewer-01", "reviewer-01"]);
        assert!(!check(&v).valid);
    }

    #[test]
    fn rejects_unresolved_adjudicator_disagreement() {
        let mut v: Value = serde_json::from_str(VALID).unwrap();
        v["blind_reconstruction"]["disagreement_resolved"] = json!(false);
        assert!(!check(&v).valid);
    }

    #[test]
    fn rejects_gold_label_exposure() {
        let mut v: Value = serde_json::from_str(VALID).unwrap();
        v["gold_custody"]["candidate_readable"] = json!(true);
        assert!(!check(&v).valid);
    }

    #[test]
    fn rejects_gold_labels_inside_candidate_bundle() {
        let mut v: Value = serde_json::from_str(VALID).unwrap();
        v["gold_custody"]["labels_separate_from_candidate_bundle"] = json!(false);
        assert!(!check(&v).valid);
    }

    #[test]
    fn rejects_shared_candidate_and_evaluator_environment() {
        let mut v: Value = serde_json::from_str(VALID).unwrap();
        v["isolation"]["candidate_evaluator_separated"] = json!(false);
        assert!(!check(&v).valid);
    }

    #[test]
    fn rejects_unrestricted_network() {
        let mut v: Value = serde_json::from_str(VALID).unwrap();
        v["isolation"]["model_endpoint_only"] = json!(false);
        assert!(!check(&v).valid);
    }

    #[test]
    fn rejects_non_pristine_regrade() {
        let mut v: Value = serde_json::from_str(VALID).unwrap();
        v["isolation"]["pristine_regrade"] = json!(false);
        assert!(!check(&v).valid);
    }

    #[test]
    fn rejects_invalid_intervention_family() {
        let mut v: Value = serde_json::from_str(VALID).unwrap();
        v["intervention"]["family"] = json!("invented_family");
        assert!(!check(&v).valid);
    }

    #[test]
    fn rejects_invalid_intervention_position() {
        let mut v: Value = serde_json::from_str(VALID).unwrap();
        v["intervention"]["position"] = json!("whenever");
        assert!(!check(&v).valid);
    }

    #[test]
    fn rejects_invalid_revision_object_id() {
        let mut v: Value = serde_json::from_str(VALID).unwrap();
        v["repository_snapshot"]["revision_oid"] = json!("not-an-object-id");
        assert!(!check(&v).valid);
    }

    #[test]
    fn rejects_invalid_digest() {
        let mut v: Value = serde_json::from_str(VALID).unwrap();
        v["task_bundle_digest"] = json!("short");
        assert!(!check(&v).valid);
    }

    #[test]
    fn claim_ceiling_does_not_promote_declarations_to_proof() {
        let report = validate_shiftbench_admission_json(VALID).unwrap();
        assert!(report.claim_ceiling.contains("no patch"));
        assert!(report.claim_ceiling.contains("not independently inspected"));
    }
}
