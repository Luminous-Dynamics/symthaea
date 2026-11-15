//! Fail-closed semantic and externally scoped checks for EGDDM in-toto statements.
//!
//! This module is deliberately not a signature, transparency, workload-identity,
//! provenance-resolution, or replay-ledger verifier. It checks the enclosing
//! Statement shape, receipt semantics, linked subject digests, chronology, and
//! exact equality with a separately supplied verifier policy. It never upgrades
//! an unsigned statement into trusted execution evidence.

use chrono::{DateTime, FixedOffset};
use serde::{Deserialize, Serialize};
use serde_json::Value;

const STATEMENT_TYPE: &str = "https://in-toto.io/Statement/v1";
const PREDICATE_TYPE: &str =
    "https://luminousdynamics.io/attestations/egddm-execution-receipt/v0.1";
const SLSA_PROVENANCE_TYPE: &str = "https://slsa.dev/provenance/v1";
const TEST_RESULT_TYPE: &str = "https://in-toto.io/attestation/test-result/v0.1";

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpectedReceiptContext {
    pub repository_full_name: String,
    pub revision_oid: String,
    pub tree_oid: String,
    pub configuration_world_digest: String,
    pub obligation_id: String,
    pub check_id: String,
    pub claim_kind: String,
    #[serde(default)]
    pub expected_outcome: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReceiptValidationReport {
    pub valid: bool,
    pub receipt_id: Option<String>,
    pub claim_kind: Option<String>,
    pub producer_role: Option<String>,
    pub diagnostics: Vec<String>,
    pub claim_ceiling: String,
}

impl ReceiptValidationReport {
    fn new() -> Self {
        Self {
            valid: false,
            receipt_id: None,
            claim_kind: None,
            producer_role: None,
            diagnostics: Vec::new(),
            claim_ceiling: "semantic and external-scope checks only; no signature, issuer trust, linked-statement resolution, transparency inclusion, or consume-once replay state was verified".to_owned(),
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

fn string_at<'a>(value: &'a Value, path: &[&str]) -> Option<&'a str> {
    at(value, path).and_then(Value::as_str)
}

fn valid_hex(value: &str, length: usize) -> bool {
    value.len() == length && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn valid_oid(value: &str) -> bool {
    valid_hex(value, 40) || valid_hex(value, 64)
}

fn digest_value(value: &Value) -> Option<&str> {
    if value.get("algorithm").and_then(Value::as_str)? != "sha256" {
        return None;
    }
    let digest = value.get("value").and_then(Value::as_str)?;
    valid_hex(digest, 64).then_some(digest)
}

fn parse_time(value: &str) -> Result<DateTime<FixedOffset>, String> {
    DateTime::parse_from_rfc3339(value).map_err(|_| format!("not RFC 3339: {value:?}"))
}

fn check_time(
    value: &Value,
    path: &[&str],
    label: &str,
    report: &mut ReceiptValidationReport,
) -> Option<DateTime<FixedOffset>> {
    match string_at(value, path).and_then(|raw| parse_time(raw).ok()) {
        Some(parsed) => Some(parsed),
        None => {
            report.error(label, "missing or invalid RFC 3339 timestamp");
            None
        }
    }
}

fn expected_role(claim_kind: &str) -> Option<&'static str> {
    match claim_kind {
        "agent_observation" => Some("agent"),
        "runner_execution" => Some("runner"),
        "independent_regrade" => Some("independent_grader"),
        "authority_decision" => Some("authority_service"),
        _ => None,
    }
}

fn validate_linked_statement_reference(
    execution: &Value,
    key: &str,
    expected_predicate_type: &str,
    outer_subject_digests: &[String],
    report: &mut ReceiptValidationReport,
) {
    let path = format!("predicate.execution.{key}");
    let Some(reference) = execution.get(key) else {
        report.error(&path, "required linked statement reference is missing");
        return;
    };
    let predicate_type = reference.get("predicate_type").and_then(Value::as_str);
    if predicate_type != Some(expected_predicate_type) {
        report.error(
            &format!("{path}.predicate_type"),
            format!("expected {expected_predicate_type}"),
        );
    }
    if digest_value(&reference["statement_digest"]).is_none() {
        report.error(&format!("{path}.statement_digest"), "expected a SHA-256 digest object");
    }
    match digest_value(&reference["subject_digest"]) {
        Some(linked_subject) if outer_subject_digests.iter().any(|d| d == linked_subject) => {}
        Some(_) => report.error(
            &format!("{path}.subject_digest"),
            "linked statement subject does not match any enclosing Statement subject",
        ),
        None => report.error(
            &format!("{path}.subject_digest"),
            "expected a SHA-256 digest object",
        ),
    }
}

pub fn verify_execution_receipt_json(
    statement_json: &str,
    policy_json: &str,
) -> Result<ReceiptValidationReport, String> {
    let statement: Value = serde_json::from_str(statement_json)
        .map_err(|error| format!("statement JSON parse failed: {error}"))?;
    let policy: ExpectedReceiptContext = serde_json::from_str(policy_json)
        .map_err(|error| format!("expected-context policy parse failed: {error}"))?;
    let mut report = ReceiptValidationReport::new();

    if statement.get("_type").and_then(Value::as_str) != Some(STATEMENT_TYPE) {
        report.error("_type", format!("expected {STATEMENT_TYPE}"));
    }
    if statement.get("predicateType").and_then(Value::as_str) != Some(PREDICATE_TYPE) {
        report.error("predicateType", format!("expected {PREDICATE_TYPE}"));
    }

    let Some(subjects) = statement.get("subject").and_then(Value::as_array) else {
        report.error("subject", "must be a non-empty array");
        return Ok(report.finish());
    };
    if subjects.is_empty() {
        report.error("subject", "must be a non-empty array");
    }
    let mut outer_subject_digests = Vec::new();
    for (index, subject) in subjects.iter().enumerate() {
        let path = format!("subject[{index}]");
        if subject.get("name").and_then(Value::as_str).is_none_or(str::is_empty) {
            report.error(&format!("{path}.name"), "must be a non-empty string");
        }
        let Some(digest) = subject.get("digest").and_then(|d| d.get("sha256")).and_then(Value::as_str) else {
            report.error(&format!("{path}.digest.sha256"), "missing SHA-256 subject digest");
            continue;
        };
        if !valid_hex(digest, 64) {
            report.error(&format!("{path}.digest.sha256"), "must be 64 hexadecimal characters");
        } else {
            outer_subject_digests.push(digest.to_ascii_lowercase());
        }
    }
    if outer_subject_digests.is_empty() {
        report.error("subject", "at least one valid SHA-256 subject digest is required");
    }

    let Some(predicate) = statement.get("predicate") else {
        report.error("predicate", "missing predicate object");
        return Ok(report.finish());
    };
    if string_at(predicate, &["predicate_version"]) != Some("0.1.0") {
        report.error("predicate.predicate_version", "expected 0.1.0");
    }
    let receipt_id = string_at(predicate, &["receipt_id"]);
    if receipt_id.is_none_or(str::is_empty) {
        report.error("predicate.receipt_id", "must be a non-empty string");
    } else {
        report.receipt_id = receipt_id.map(str::to_owned);
    }

    let claim_kind = string_at(predicate, &["claim_kind"]).unwrap_or("");
    report.claim_kind = (!claim_kind.is_empty()).then(|| claim_kind.to_owned());
    let Some(required_role) = expected_role(claim_kind) else {
        report.error("predicate.claim_kind", "unsupported claim kind");
        return Ok(report.finish());
    };

    let role = string_at(predicate, &["producer", "role"]);
    report.producer_role = role.map(str::to_owned);
    if role != Some(required_role) {
        report.error(
            "predicate.producer.role",
            format!("claim kind {claim_kind} requires role {required_role}"),
        );
    }
    for field in ["workload_identity", "identity_provider"] {
        if string_at(predicate, &["producer", field]).is_none_or(str::is_empty) {
            report.error(&format!("predicate.producer.{field}"), "must be a non-empty string");
        }
    }
    if let Some(issuer) = string_at(predicate, &["producer", "issuer_uri"]) {
        if issuer.trim().is_empty() {
            report.error("predicate.producer.issuer_uri", "must not be empty when supplied");
        }
    }

    let repository = string_at(predicate, &["subject_scope", "repository_full_name"]).unwrap_or("");
    let revision = string_at(predicate, &["subject_scope", "revision_oid"]).unwrap_or("");
    let tree = string_at(predicate, &["subject_scope", "tree_oid"]).unwrap_or("");
    let configuration = digest_value(
        at(predicate, &["subject_scope", "configuration_world_digest"])
            .unwrap_or(&Value::Null),
    );
    if repository.split('/').count() != 2 || repository.split('/').any(str::is_empty) {
        report.error("predicate.subject_scope.repository_full_name", "must be owner/repository");
    }
    if !valid_oid(revision) {
        report.error("predicate.subject_scope.revision_oid", "must be a 40- or 64-character object ID");
    }
    if !valid_oid(tree) || (!revision.is_empty() && tree.len() != revision.len()) {
        report.error("predicate.subject_scope.tree_oid", "must be a valid object ID matching revision length");
    }
    if configuration.is_none() {
        report.error(
            "predicate.subject_scope.configuration_world_digest",
            "must be a SHA-256 digest object",
        );
    }

    if repository != policy.repository_full_name {
        report.error("policy.repository_full_name", "receipt repository differs from expected policy");
    }
    if revision != policy.revision_oid {
        report.error("policy.revision_oid", "receipt revision differs from expected policy");
    }
    if tree != policy.tree_oid {
        report.error("policy.tree_oid", "receipt tree differs from expected policy");
    }
    if configuration != Some(policy.configuration_world_digest.as_str()) {
        report.error(
            "policy.configuration_world_digest",
            "receipt configuration differs from expected policy",
        );
    }

    let obligation_id = string_at(predicate, &["obligation", "obligation_id"]).unwrap_or("");
    let check_id = string_at(predicate, &["obligation", "check_id"]).unwrap_or("");
    let check_kind = string_at(predicate, &["obligation", "check_kind"]).unwrap_or("");
    let outcome = string_at(predicate, &["obligation", "outcome"]).unwrap_or("");
    if obligation_id.is_empty() || check_id.is_empty() {
        report.error("predicate.obligation", "obligation_id and check_id must be non-empty");
    }
    if !["test", "build", "review", "runtime", "policy", "security", "compatibility", "other"]
        .contains(&check_kind)
    {
        report.error("predicate.obligation.check_kind", "unsupported check kind");
    }
    if !["pass", "fail", "unknown", "not_run"].contains(&outcome) {
        report.error("predicate.obligation.outcome", "unsupported outcome");
    }
    if obligation_id != policy.obligation_id {
        report.error("policy.obligation_id", "receipt obligation differs from expected policy");
    }
    if check_id != policy.check_id {
        report.error("policy.check_id", "receipt check differs from expected policy");
    }
    if claim_kind != policy.claim_kind {
        report.error("policy.claim_kind", "receipt claim kind differs from expected policy");
    }
    if policy.expected_outcome.as_deref().is_some_and(|expected| expected != outcome) {
        report.error("policy.expected_outcome", "receipt outcome differs from required outcome");
    }

    let execution = predicate.get("execution").unwrap_or(&Value::Null);
    let execution_required = matches!(claim_kind, "runner_execution" | "independent_regrade");
    if execution_required && !execution.is_object() {
        report.error("predicate.execution", "execution object required for runner and independent-grader claims");
    } else if execution.is_object() {
        for field in [
            "attempt_id",
            "invocation_id",
            "runner_image_digest",
            "toolchain_digest",
            "harness_digest",
            "command_digest",
            "test_input_digest",
            "isolation_profile_digest",
            "network_policy_digest",
            "environment_digest",
            "workflow_run_id",
        ] {
            if string_at(execution, &[field]).is_none_or(str::is_empty)
                && digest_value(execution.get(field).unwrap_or(&Value::Null)).is_none()
            {
                report.error(&format!("predicate.execution.{field}"), "required non-empty string or SHA-256 digest");
            }
        }
        if execution.get("workflow_attempt").and_then(Value::as_u64).is_none_or(|attempt| attempt == 0) {
            report.error("predicate.execution.workflow_attempt", "must be a positive integer");
        }
    }

    if claim_kind == "runner_execution" && check_kind == "build" {
        validate_linked_statement_reference(
            execution,
            "slsa_provenance",
            SLSA_PROVENANCE_TYPE,
            &outer_subject_digests,
            &mut report,
        );
    }
    if matches!(claim_kind, "runner_execution" | "independent_regrade")
        && matches!(check_kind, "test" | "compatibility")
    {
        validate_linked_statement_reference(
            execution,
            "test_result",
            TEST_RESULT_TYPE,
            &outer_subject_digests,
            &mut report,
        );
    }

    let started = check_time(predicate, &["observation", "started_at"], "predicate.observation.started_at", &mut report);
    let completed = check_time(predicate, &["observation", "completed_at"], "predicate.observation.completed_at", &mut report);
    let observed = check_time(predicate, &["observation", "observed_at"], "predicate.observation.observed_at", &mut report);
    let available = check_time(predicate, &["observation", "available_at"], "predicate.observation.available_at", &mut report);
    if let (Some(started), Some(completed), Some(observed), Some(available)) =
        (started, completed, observed, available)
    {
        if started > completed || completed > observed || observed > available {
            report.error(
                "predicate.observation",
                "timestamps must satisfy started_at <= completed_at <= observed_at <= available_at",
            );
        }
    }

    let valid_from = check_time(predicate, &["authority", "valid_from"], "predicate.authority.valid_from", &mut report);
    let valid_until = check_time(predicate, &["authority", "valid_until"], "predicate.authority.valid_until", &mut report);
    if string_at(predicate, &["authority", "decision_id"]).is_none_or(str::is_empty)
        || string_at(predicate, &["authority", "permitted_scope"]).is_none_or(str::is_empty)
        || digest_value(at(predicate, &["authority", "policy_digest"]).unwrap_or(&Value::Null)).is_none()
    {
        report.error("predicate.authority", "decision_id, permitted_scope and a SHA-256 policy_digest are required");
    }
    if let (Some(from), Some(until)) = (valid_from, valid_until) {
        if from > until {
            report.error("predicate.authority", "valid_from must not be later than valid_until");
        }
        if let (Some(started), Some(completed)) = (
            string_at(predicate, &["observation", "started_at"]).and_then(|v| parse_time(v).ok()),
            string_at(predicate, &["observation", "completed_at"]).and_then(|v| parse_time(v).ok()),
        ) {
            if started < from || completed > until {
                report.error("predicate.authority", "execution interval falls outside authority validity");
            }
        }
    }

    let challenge_issued = check_time(predicate, &["anti_replay", "issued_at"], "predicate.anti_replay.issued_at", &mut report);
    let challenge_expires = check_time(predicate, &["anti_replay", "expires_at"], "predicate.anti_replay.expires_at", &mut report);
    if string_at(predicate, &["anti_replay", "challenge_id"]).is_none_or(str::is_empty)
        || string_at(predicate, &["anti_replay", "challenge_issuer"]).is_none_or(str::is_empty)
        || digest_value(at(predicate, &["anti_replay", "nonce_digest"]).unwrap_or(&Value::Null)).is_none()
        || predicate.pointer("/anti_replay/single_use").and_then(Value::as_bool) != Some(true)
    {
        report.error(
            "predicate.anti_replay",
            "challenge identity, issuer, nonce digest and single_use=true are required",
        );
    }
    if let (Some(issued), Some(expires)) = (challenge_issued, challenge_expires) {
        if issued > expires {
            report.error("predicate.anti_replay", "issued_at must not be later than expires_at");
        }
        if let Some(started) = string_at(predicate, &["observation", "started_at"]).and_then(|v| parse_time(v).ok()) {
            if started < issued || started > expires {
                report.error("predicate.anti_replay", "execution start falls outside challenge validity");
            }
        }
        if let Some(completed) = string_at(predicate, &["observation", "completed_at"]).and_then(|v| parse_time(v).ok()) {
            if completed > expires {
                report.error("predicate.anti_replay", "execution completed after challenge expiry");
            }
        }
    }

    Ok(report.finish())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const RUNNER_TEST: &str = include_str!("../../../docs/research/fixtures/egddm-execution-receipt-predicate-v0.1.synthetic.json");
    const RUNNER_BUILD: &str = include_str!("../../../docs/research/fixtures/egddm-build-execution-statement-v0.1.synthetic.json");
    const INDEPENDENT: &str = include_str!("../../../docs/research/fixtures/egddm-independent-regrade-statement-v0.1.synthetic.json");
    const AGENT: &str = include_str!("../../../docs/research/fixtures/egddm-agent-observation-statement-v0.1.synthetic.json");
    const AUTHORITY: &str = include_str!("../../../docs/research/fixtures/egddm-authority-decision-statement-v0.1.synthetic.json");

    fn policy_for(statement: &str) -> String {
        let value: Value = serde_json::from_str(statement).unwrap();
        let predicate = &value["predicate"];
        serde_json::to_string(&json!({
            "repository_full_name": predicate["subject_scope"]["repository_full_name"],
            "revision_oid": predicate["subject_scope"]["revision_oid"],
            "tree_oid": predicate["subject_scope"]["tree_oid"],
            "configuration_world_digest": predicate["subject_scope"]["configuration_world_digest"]["value"],
            "obligation_id": predicate["obligation"]["obligation_id"],
            "check_id": predicate["obligation"]["check_id"],
            "claim_kind": predicate["claim_kind"],
            "expected_outcome": predicate["obligation"]["outcome"]
        })).unwrap()
    }

    fn verify(statement: &str) -> ReceiptValidationReport {
        verify_execution_receipt_json(statement, &policy_for(statement)).unwrap()
    }

    #[test]
    fn accepts_all_five_synthetic_claim_kinds_with_exact_context() {
        for statement in [RUNNER_TEST, RUNNER_BUILD, INDEPENDENT, AGENT, AUTHORITY] {
            let report = verify(statement);
            assert!(report.valid, "{:?}", report.diagnostics);
        }
    }

    #[test]
    fn rejects_wrong_statement_type() {
        let mut value: Value = serde_json::from_str(RUNNER_TEST).unwrap();
        value["_type"] = json!("https://example.invalid/not-a-statement");
        assert!(!verify(&value.to_string()).valid);
    }

    #[test]
    fn rejects_wrong_predicate_type() {
        let mut value: Value = serde_json::from_str(RUNNER_TEST).unwrap();
        value["predicateType"] = json!("https://example.invalid/not-egddm");
        assert!(!verify(&value.to_string()).valid);
    }

    #[test]
    fn rejects_claim_role_substitution() {
        let mut value: Value = serde_json::from_str(RUNNER_TEST).unwrap();
        value["predicate"]["producer"]["role"] = json!("agent");
        assert!(!verify(&value.to_string()).valid);
    }

    #[test]
    fn rejects_repository_scope_mismatch() {
        let mut value: Value = serde_json::from_str(RUNNER_TEST).unwrap();
        value["predicate"]["subject_scope"]["repository_full_name"] = json!("attacker/other");
        assert!(!verify(&value.to_string()).valid);
    }

    #[test]
    fn rejects_revision_scope_mismatch() {
        let mut value: Value = serde_json::from_str(RUNNER_TEST).unwrap();
        value["predicate"]["subject_scope"]["revision_oid"] = json!("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb");
        assert!(!verify(&value.to_string()).valid);
    }

    #[test]
    fn rejects_tree_scope_mismatch() {
        let mut value: Value = serde_json::from_str(RUNNER_TEST).unwrap();
        value["predicate"]["subject_scope"]["tree_oid"] = json!("cccccccccccccccccccccccccccccccccccccccc");
        assert!(!verify(&value.to_string()).valid);
    }

    #[test]
    fn rejects_configuration_world_mismatch() {
        let mut value: Value = serde_json::from_str(RUNNER_TEST).unwrap();
        value["predicate"]["subject_scope"]["configuration_world_digest"]["value"] = json!("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
        assert!(!verify(&value.to_string()).valid);
    }

    #[test]
    fn rejects_obligation_substitution() {
        let mut value: Value = serde_json::from_str(RUNNER_TEST).unwrap();
        value["predicate"]["obligation"]["obligation_id"] = json!("different-obligation");
        assert!(!verify(&value.to_string()).valid);
    }

    #[test]
    fn rejects_check_substitution() {
        let mut value: Value = serde_json::from_str(RUNNER_TEST).unwrap();
        value["predicate"]["obligation"]["check_id"] = json!("different-check");
        assert!(!verify(&value.to_string()).valid);
    }

    #[test]
    fn rejects_outcome_not_allowed_by_policy() {
        let mut policy: Value = serde_json::from_str(&policy_for(RUNNER_TEST)).unwrap();
        policy["expected_outcome"] = json!("fail");
        let report = verify_execution_receipt_json(RUNNER_TEST, &policy.to_string()).unwrap();
        assert!(!report.valid);
    }

    #[test]
    fn rejects_missing_runner_execution() {
        let mut value: Value = serde_json::from_str(RUNNER_TEST).unwrap();
        value["predicate"].as_object_mut().unwrap().remove("execution");
        assert!(!verify(&value.to_string()).valid);
    }

    #[test]
    fn rejects_missing_test_result_link() {
        let mut value: Value = serde_json::from_str(RUNNER_TEST).unwrap();
        value["predicate"]["execution"].as_object_mut().unwrap().remove("test_result");
        assert!(!verify(&value.to_string()).valid);
    }

    #[test]
    fn rejects_missing_slsa_build_provenance_link() {
        let mut value: Value = serde_json::from_str(RUNNER_BUILD).unwrap();
        value["predicate"]["execution"].as_object_mut().unwrap().remove("slsa_provenance");
        assert!(!verify(&value.to_string()).valid);
    }

    #[test]
    fn rejects_linked_statement_subject_mismatch() {
        let mut value: Value = serde_json::from_str(RUNNER_TEST).unwrap();
        value["predicate"]["execution"]["test_result"]["subject_digest"]["value"] = json!("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
        assert!(!verify(&value.to_string()).valid);
    }

    #[test]
    fn rejects_wrong_linked_test_result_predicate_type() {
        let mut value: Value = serde_json::from_str(RUNNER_TEST).unwrap();
        value["predicate"]["execution"]["test_result"]["predicate_type"] = json!(SLSA_PROVENANCE_TYPE);
        assert!(!verify(&value.to_string()).valid);
    }

    #[test]
    fn rejects_future_observation_order() {
        let mut value: Value = serde_json::from_str(RUNNER_TEST).unwrap();
        value["predicate"]["observation"]["observed_at"] = json!("2026-11-15T07:59:00Z");
        assert!(!verify(&value.to_string()).valid);
    }

    #[test]
    fn rejects_authority_window_expired_before_execution() {
        let mut value: Value = serde_json::from_str(RUNNER_TEST).unwrap();
        value["predicate"]["authority"]["valid_until"] = json!("2026-11-15T07:59:59Z");
        assert!(!verify(&value.to_string()).valid);
    }

    #[test]
    fn rejects_execution_outside_challenge_window() {
        let mut value: Value = serde_json::from_str(RUNNER_TEST).unwrap();
        value["predicate"]["anti_replay"]["expires_at"] = json!("2026-11-15T07:59:59Z");
        assert!(!verify(&value.to_string()).valid);
    }

    #[test]
    fn rejects_non_single_use_challenge() {
        let mut value: Value = serde_json::from_str(RUNNER_TEST).unwrap();
        value["predicate"]["anti_replay"]["single_use"] = json!(false);
        assert!(!verify(&value.to_string()).valid);
    }

    #[test]
    fn does_not_claim_cryptographic_or_replay_verification() {
        let report = verify(RUNNER_TEST);
        assert!(report.claim_ceiling.contains("no signature"));
        assert!(report.claim_ceiling.contains("consume-once replay state"));
    }
}
