//! Semantic conformance checks for the Luminous Evidence Profile (LEP).
//!
//! This crate checks the meaning and internal consistency of a parsed in-toto
//! Statement carrying the LEP predicate. It is deliberately not a replacement
//! for JSON Schema validation, DSSE/in-toto signature verification, artifact
//! resolution, signer authorization, or scientific validation.
//!
//! A passing report means only that the supplied JSON has passed the checks
//! implemented here. It does not prove that a run occurred or that its claim
//! is true. Callers must combine this with the versioned schema, a trusted
//! attestation verifier, artifact verification, and domain-specific checks.

use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeSet;

pub const STATEMENT_TYPE: &str = "https://in-toto.io/Statement/v1";
pub const PREDICATE_TYPE: &str = "https://luminousdynamics.io/attestations/evidence/v0.1";
pub const SCHEMA_VERSION: &str = "0.1.0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Violation {
    pub code: String,
    pub path: String,
    pub message: String,
}

impl Violation {
    fn new(code: &str, path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.to_owned(),
            path: path.into(),
            message: message.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ValidationReport {
    pub valid: bool,
    pub violations: Vec<Violation>,
}

impl ValidationReport {
    fn from_violations(violations: Vec<Violation>) -> Self {
        Self {
            valid: violations.is_empty(),
            violations,
        }
    }

    pub fn invalid_json(message: impl Into<String>) -> Self {
        Self::from_violations(vec![Violation::new("INVALID_JSON", "$", message)])
    }
}

/// Validate a parsed in-toto Statement carrying the LEP predicate.
///
/// Object key order is ignored by JSON value equality. Array order remains
/// significant, so the target and checkedTarget must have exactly equivalent
/// structured values, including the order of input artifacts.
pub fn verify_statement(statement: &Value) -> ValidationReport {
    let mut violations = Vec::new();

    let Some(root) = statement.as_object() else {
        return ValidationReport::from_violations(vec![Violation::new(
            "INVALID_STATEMENT",
            "$",
            "The in-toto Statement must be a JSON object.",
        )]);
    };

    if root.get("_type").and_then(Value::as_str) != Some(STATEMENT_TYPE) {
        push(
            &mut violations,
            "STATEMENT_TYPE_UNSUPPORTED",
            "$._type",
            "Expected the in-toto Statement v1 type URI.",
        );
    }

    validate_subjects(root.get("subject"), &mut violations);

    if root.get("predicateType").and_then(Value::as_str) != Some(PREDICATE_TYPE) {
        push(
            &mut violations,
            "PREDICATE_TYPE_UNSUPPORTED",
            "$.predicateType",
            "Expected the LEP v0.1 predicate type URI.",
        );
    }

    let Some(predicate) = root.get("predicate").and_then(Value::as_object) else {
        push(
            &mut violations,
            "MISSING_PREDICATE",
            "$.predicate",
            "The Statement must contain an LEP predicate object.",
        );
        return ValidationReport::from_violations(violations);
    };

    if predicate.get("schemaVersion").and_then(Value::as_str) != Some(SCHEMA_VERSION) {
        push(
            &mut violations,
            "SCHEMA_VERSION_UNSUPPORTED",
            "$.predicate.schemaVersion",
            "Unsupported or missing LEP predicate version.",
        );
    }

    validate_claim(predicate.get("claim"), &mut violations);

    let target = predicate.get("target");
    match target {
        None | Some(Value::Null) => push(
            &mut violations,
            "MISSING_TARGET",
            "$.predicate.target",
            "Evidence must identify the source revision, configuration digest, and input artifacts.",
        ),
        Some(value) => validate_target(value, "$.predicate.target", &mut violations),
    }

    validate_execution(predicate.get("execution"), &mut violations);
    validate_artifacts(predicate.get("evidence"), "$.predicate.evidence", true, &mut violations);
    validate_limitations(predicate.get("limitations"), &mut violations);
    validate_extensions(predicate.get("extensions"), &mut violations);

    validate_verification(
        predicate.get("verification"),
        target,
        predicate.get("execution"),
        &mut violations,
    );

    ValidationReport::from_violations(violations)
}

fn push(
    out: &mut Vec<Violation>,
    code: &str,
    path: impl Into<String>,
    message: impl Into<String>,
) {
    out.push(Violation::new(code, path, message));
}

fn validate_subjects(value: Option<&Value>, out: &mut Vec<Violation>) {
    let Some(subjects) = value.and_then(Value::as_array) else {
        push(out, "INVALID_SUBJECTS", "$.subject", "subject must be an array.");
        return;
    };
    if subjects.is_empty() {
        push(out, "EMPTY_SUBJECTS", "$.subject", "At least one subject is required.");
    }

    for (index, subject) in subjects.iter().enumerate() {
        let path = format!("$.subject[{index}]");
        let Some(object) = subject.as_object() else {
            push(out, "INVALID_SUBJECT", path, "Each subject must be an object.");
            continue;
        };
        if object.get("name").and_then(Value::as_str).is_none_or(str::is_empty) {
            push(out, "INVALID_SUBJECT_NAME", format!("{path}.name"), "Subject name must be non-empty.");
        }
        validate_digest_set(object.get("digest"), &format!("{path}.digest"), out);
    }
}

fn validate_claim(value: Option<&Value>, out: &mut Vec<Violation>) {
    let path = "$.predicate.claim";
    let Some(claim) = value.and_then(Value::as_object) else {
        push(out, "MISSING_CLAIM", path, "The claim object is required.");
        return;
    };

    for (field, code) in [
        ("id", "INVALID_CLAIM_ID"),
        ("statement", "INVALID_CLAIM_STATEMENT"),
    ] {
        if claim.get(field).and_then(Value::as_str).is_none_or(str::is_empty) {
            push(out, code, format!("{path}.{field}"), "Required string must be non-empty.");
        }
    }

    let kinds = [
        "normative",
        "implementation",
        "test_result",
        "measurement",
        "simulation",
        "inference",
        "hypothesis",
        "aspiration",
    ];
    if !claim.get("kind").and_then(Value::as_str).is_some_and(|kind| kinds.contains(&kind)) {
        push(out, "INVALID_CLAIM_KIND", format!("{path}.kind"), "Unsupported or missing claim kind.");
    }

    let Some(scope) = claim.get("scope").and_then(Value::as_array) else {
        push(out, "INVALID_CLAIM_SCOPE", format!("{path}.scope"), "scope must be a non-empty array.");
        return;
    };
    if scope.is_empty() || scope.iter().any(|v| v.as_str().is_none_or(str::is_empty)) {
        push(out, "INVALID_CLAIM_SCOPE", format!("{path}.scope"), "scope must contain non-empty strings.");
    }
    let mut seen = BTreeSet::new();
    for entry in scope.iter().filter_map(Value::as_str) {
        if !seen.insert(entry) {
            push(out, "DUPLICATE_CLAIM_SCOPE", format!("{path}.scope"), "scope entries must be unique.");
            break;
        }
    }
}

fn validate_target(value: &Value, path: &str, out: &mut Vec<Violation>) {
    let Some(target) = value.as_object() else {
        push(out, "INVALID_TARGET", path, "target must be an object.");
        return;
    };

    let Some(source) = target.get("source").and_then(Value::as_object) else {
        push(out, "INVALID_SOURCE_IDENTITY", format!("{path}.source"), "Source identity is required.");
        return;
    };
    let repository = source.get("repository").and_then(Value::as_str);
    if repository.is_none_or(|s| !s.starts_with("https://") || s.chars().any(char::is_whitespace)) {
        push(
            out,
            "INVALID_SOURCE_REPOSITORY",
            format!("{path}.source.repository"),
            "Repository must be an HTTPS URI without whitespace.",
        );
    }

    let revision = source.get("revision").and_then(Value::as_str);
    if !revision.is_some_and(is_pinned_revision) {
        push(
            out,
            "SOURCE_REVISION_NOT_IMMUTABLE",
            format!("{path}.source.revision"),
            "Source revision must be a full lowercase 40- or 64-character hexadecimal commit identifier, not a branch or abbreviated ref.",
        );
    }

    validate_digest_set(target.get("configurationDigest"), &format!("{path}.configurationDigest"), out);
    validate_artifacts(target.get("inputs"), &format!("{path}.inputs"), false, out);
}

fn is_pinned_revision(value: &str) -> bool {
    matches!(value.len(), 40 | 64)
        && value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn validate_execution(value: Option<&Value>, out: &mut Vec<Violation>) {
    let path = "$.predicate.execution";
    let Some(execution) = value.and_then(Value::as_object) else {
        push(out, "MISSING_EXECUTION", path, "The execution object is required.");
        return;
    };
    let status = execution.get("status").and_then(Value::as_str);
    let result = execution.get("result").and_then(Value::as_str);

    if execution.get("runId").is_none() {
        push(out, "MISSING_RUN_ID", format!("{path}.runId"), "runId is required; use null when no run occurred.");
    } else if execution.get("runId").is_some_and(|v| !v.is_null() && v.as_str().is_none()) {
        push(out, "INVALID_RUN_ID", format!("{path}.runId"), "runId must be a string or null.");
    }

    match (status, result) {
        (Some("not_run" | "queued" | "running" | "cancelled"), Some("not_run")) => {}
        (Some("completed"), Some("passed" | "failed" | "inconclusive")) => {}
        (Some("failed"), Some("failed" | "inconclusive" | "not_run")) => {}
        _ => push(
            out,
            "EXECUTION_STATUS_RESULT_CONFLICT",
            path,
            "Execution status and result are missing, unsupported, or contradictory.",
        ),
    }

    if status == Some("completed") && execution.get("runId").and_then(Value::as_str).is_none_or(str::is_empty) {
        push(out, "COMPLETED_RUN_ID_REQUIRED", format!("{path}.runId"), "Completed execution requires a non-empty runId.");
    }
}

fn validate_verification(
    value: Option<&Value>,
    target: Option<&Value>,
    execution: Option<&Value>,
    out: &mut Vec<Violation>,
) {
    let path = "$.predicate.verification";
    let Some(verification) = value.and_then(Value::as_object) else {
        push(out, "MISSING_VERIFICATION", path, "The verification object is required.");
        return;
    };

    let outcome = verification.get("outcome").and_then(Value::as_str);
    let verifier = verification.get("verifier");
    let checked_target = verification.get("checkedTarget");
    let checks = verification.get("checks").and_then(Value::as_array);

    if !matches!(outcome, Some("not_run" | "passed" | "failed" | "inconclusive")) {
        push(out, "INVALID_VERIFICATION_OUTCOME", format!("{path}.outcome"), "Unsupported or missing verification outcome.");
    }

    if outcome == Some("not_run") {
        if !verifier.is_some_and(Value::is_null) {
            push(out, "UNRUN_VERIFICATION_HAS_VERIFIER", format!("{path}.verifier"), "An unrun verification must have a null verifier.");
        }
        if !checked_target.is_some_and(Value::is_null) {
            push(out, "UNRUN_VERIFICATION_HAS_TARGET", format!("{path}.checkedTarget"), "An unrun verification must have a null checkedTarget.");
        }
        if checks.is_none_or(|entries| !entries.is_empty()) {
            push(out, "UNRUN_VERIFICATION_HAS_CHECKS", format!("{path}.checks"), "An unrun verification cannot claim checks.");
        }
        return;
    }

    let Some(verifier_obj) = verifier.and_then(Value::as_object) else {
        push(out, "VERIFIER_IDENTITY_REQUIRED", format!("{path}.verifier"), "Executed verification requires verifier identity.");
        return;
    };
    for field in ["id", "version"] {
        if verifier_obj.get(field).and_then(Value::as_str).is_none_or(str::is_empty) {
            push(out, "VERIFIER_IDENTITY_REQUIRED", format!("{path}.verifier.{field}"), "Verifier id and version must be non-empty.");
        }
    }

    let Some(expected_target) = target.filter(|v| v.is_object()) else {
        push(out, "LEP_TARGET_MISMATCH", format!("{path}.checkedTarget"), "Cannot verify a claim without a valid declared target.");
        return;
    };
    let Some(actual_target) = checked_target.filter(|v| v.is_object()) else {
        push(out, "LEP_TARGET_MISMATCH", format!("{path}.checkedTarget"), "An executed verification must identify the target it checked.");
        return;
    };
    if expected_target != actual_target {
        push(
            out,
            "LEP_TARGET_MISMATCH",
            format!("{path}.checkedTarget"),
            "The verifier's target does not exactly match the predicate's declared target.",
        );
    }

    let Some(entries) = checks else {
        push(out, "INVALID_VERIFICATION_CHECKS", format!("{path}.checks"), "checks must be an array.");
        return;
    };
    let mut ids = BTreeSet::new();
    for (index, check) in entries.iter().enumerate() {
        let check_path = format!("{path}.checks[{index}]");
        let Some(check_obj) = check.as_object() else {
            push(out, "INVALID_CHECK", check_path, "Each check must be an object.");
            continue;
        };
        let id = check_obj.get("id").and_then(Value::as_str);
        if id.is_none_or(str::is_empty) {
            push(out, "INVALID_CHECK_ID", format!("{check_path}.id"), "Check id must be non-empty.");
        } else if !ids.insert(id.unwrap_or_default()) {
            push(out, "DUPLICATE_CHECK_ID", format!("{check_path}.id"), "Check ids must be unique within one verification.");
        }
        let check_outcome = check_obj.get("outcome").and_then(Value::as_str);
        if !matches!(check_outcome, Some("passed" | "failed" | "inconclusive" | "not_run")) {
            push(out, "INVALID_CHECK_OUTCOME", format!("{check_path}.outcome"), "Unsupported or missing check outcome.");
        }
        if outcome == Some("passed") && check_outcome != Some("passed") {
            push(out, "PASSED_VERIFICATION_HAS_NONPASSING_CHECK", format!("{check_path}.outcome"), "A passed verification may contain only passing checks.");
        }
    }

    if outcome == Some("passed") {
        if entries.is_empty() {
            push(out, "PASSED_VERIFICATION_HAS_NO_CHECKS", format!("{path}.checks"), "A passed verification requires at least one recorded check.");
        }
        let execution_obj = execution.and_then(Value::as_object);
        if execution_obj.and_then(|v| v.get("status")).and_then(Value::as_str) != Some("completed")
            || execution_obj.and_then(|v| v.get("result")).and_then(Value::as_str) != Some("passed")
        {
            push(
                out,
                "PASSED_VERIFICATION_WITHOUT_PASSED_EXECUTION",
                path,
                "A passed verification requires a completed, passed execution of the declared target.",
            );
        }
    }
}

fn validate_artifacts(
    value: Option<&Value>,
    path: &str,
    required: bool,
    out: &mut Vec<Violation>,
) {
    let Some(artifacts) = value.and_then(Value::as_array) else {
        push(
            out,
            if required { "INVALID_EVIDENCE_ARTIFACTS" } else { "INVALID_TARGET_INPUTS" },
            path,
            "Artifact collections must be arrays.",
        );
        return;
    };

    for (index, artifact) in artifacts.iter().enumerate() {
        let artifact_path = format!("{path}[{index}]");
        let Some(object) = artifact.as_object() else {
            push(out, "INVALID_ARTIFACT", artifact_path, "Each artifact must be an object.");
            continue;
        };
        if object.get("name").and_then(Value::as_str).is_none_or(str::is_empty) {
            push(out, "INVALID_ARTIFACT_NAME", format!("{artifact_path}.name"), "Artifact name must be non-empty.");
        }
        let kinds = [
            "source",
            "configuration",
            "dataset_manifest",
            "raw_result",
            "test_report",
            "measurement",
            "simulation_trace",
            "log",
            "certificate",
            "other",
        ];
        if !object.get("kind").and_then(Value::as_str).is_some_and(|kind| kinds.contains(&kind)) {
            push(out, "INVALID_ARTIFACT_KIND", format!("{artifact_path}.kind"), "Unsupported or missing artifact kind.");
        }
        validate_digest_set(object.get("digest"), &format!("{artifact_path}.digest"), out);
    }
}

fn validate_digest_set(value: Option<&Value>, path: &str, out: &mut Vec<Violation>) {
    let Some(digests) = value.and_then(Value::as_object) else {
        push(out, "INVALID_DIGEST_SET", path, "A digest set object is required.");
        return;
    };
    if digests.is_empty() {
        push(out, "EMPTY_DIGEST_SET", path, "At least one supported digest is required.");
        return;
    }

    for (algorithm, digest) in digests {
        let expected_length = match algorithm.as_str() {
            "sha256" | "blake3" => 64,
            "sha512" => 128,
            _ => {
                push(out, "UNSUPPORTED_DIGEST_ALGORITHM", format!("{path}.{algorithm}"), "LEP v0.1 supports only sha256, sha512, and blake3.");
                continue;
            }
        };
        let Some(hex) = digest.as_str() else {
            push(out, "INVALID_DIGEST_VALUE", format!("{path}.{algorithm}"), "Digest value must be a lowercase hexadecimal string.");
            continue;
        };
        if hex.len() != expected_length || !hex.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
            push(
                out,
                "INVALID_DIGEST_VALUE",
                format!("{path}.{algorithm}"),
                format!("Expected {expected_length} lowercase hexadecimal characters."),
            );
        }
    }
}

fn validate_limitations(value: Option<&Value>, out: &mut Vec<Violation>) {
    let Some(limitations) = value.and_then(Value::as_array) else {
        push(out, "INVALID_LIMITATIONS", "$.predicate.limitations", "limitations must be an array.");
        return;
    };
    if limitations.iter().any(|v| v.as_str().is_none_or(str::is_empty)) {
        push(out, "INVALID_LIMITATIONS", "$.predicate.limitations", "Each limitation must be a non-empty string.");
    }
}

fn validate_extensions(value: Option<&Value>, out: &mut Vec<Violation>) {
    let Some(extensions) = value else {
        return;
    };
    let Some(object) = extensions.as_object() else {
        push(out, "INVALID_EXTENSIONS", "$.predicate.extensions", "extensions must be an object.");
        return;
    };
    for key in object.keys() {
        if !key.starts_with("https://") || key.chars().any(char::is_whitespace) {
            push(
                out,
                "EXTENSION_NOT_NAMESPACED",
                format!("$.predicate.extensions.{key}"),
                "Extension keys must be URI-namespaced using HTTPS.",
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn minimal_statement() -> Value {
        let target = json!({
            "source": {
                "repository": "https://github.com/Luminous-Dynamics/symthaea",
                "revision": "754589fa14a63070cf75d8f9ec90078e673d626c"
            },
            "configurationDigest": {
                "sha256": "b5d8a9c4e3f2106789abcdef0123456789abcdef0123456789abcdef01234567"
            },
            "inputs": []
        });

        json!({
            "_type": STATEMENT_TYPE,
            "subject": [{
                "name": "synthetic-report.json",
                "digest": {"sha256": "a4d9f7a8e44c1d3c6f4a9f83a25c8e20f3a6d1c9b2e4f7a8d5c0e1f3a6b9d2e4"}
            }],
            "predicateType": PREDICATE_TYPE,
            "predicate": {
                "schemaVersion": SCHEMA_VERSION,
                "claim": {
                    "id": "LEP-TEST-001",
                    "kind": "test_result",
                    "statement": "Synthetic conformance test fixture; not a production claim.",
                    "scope": ["semantic-conformance"]
                },
                "target": target,
                "execution": {
                    "status": "completed",
                    "result": "passed",
                    "runId": "synthetic-test-run"
                },
                "verification": {
                    "outcome": "passed",
                    "verifier": {"id": "lep-test", "version": "0.1.0"},
                    "checkedTarget": target,
                    "checks": [{"id": "fixture", "outcome": "passed"}]
                },
                "evidence": [],
                "limitations": ["Synthetic data only."]
            }
        })
    }

    fn has_code(report: &ValidationReport, code: &str) -> bool {
        report.violations.iter().any(|violation| violation.code == code)
    }

    #[test]
    fn valid_synthetic_statement_is_accepted() {
        let report = verify_statement(&minimal_statement());
        assert!(report.valid, "{:?}", report.violations);
    }

    #[test]
    fn queued_execution_cannot_be_passed() {
        let mut statement = minimal_statement();
        statement["predicate"]["execution"]["status"] = json!("queued");
        statement["predicate"]["execution"]["result"] = json!("passed");
        let report = verify_statement(&statement);
        assert!(!report.valid);
        assert!(has_code(&report, "EXECUTION_STATUS_RESULT_CONFLICT"));
        assert!(has_code(&report, "PASSED_VERIFICATION_WITHOUT_PASSED_EXECUTION"));
    }

    #[test]
    fn stale_checked_target_is_rejected() {
        let mut statement = minimal_statement();
        statement["predicate"]["verification"]["checkedTarget"]["source"]["revision"] = json!("deadbeef1234567");
        let report = verify_statement(&statement);
        assert!(!report.valid);
        assert!(has_code(&report, "LEP_TARGET_MISMATCH"));
    }

    #[test]
    fn incomplete_hash_is_rejected() {
        let mut statement = minimal_statement();
        statement["predicate"]["target"]["configurationDigest"]["sha256"] = json!("deadbeef");
        statement["predicate"]["verification"]["checkedTarget"]["configurationDigest"]["sha256"] = json!("deadbeef");
        let report = verify_statement(&statement);
        assert!(!report.valid);
        assert!(has_code(&report, "INVALID_DIGEST_VALUE"));
    }

    #[test]
    fn consistent_unrun_verification_is_accepted() {
        let mut statement = minimal_statement();
        statement["predicate"]["execution"]["status"] = json!("not_run");
        statement["predicate"]["execution"]["result"] = json!("not_run");
        statement["predicate"]["execution"]["runId"] = Value::Null;
        statement["predicate"]["verification"]["outcome"] = json!("not_run");
        statement["predicate"]["verification"]["verifier"] = Value::Null;
        statement["predicate"]["verification"]["checkedTarget"] = Value::Null;
        statement["predicate"]["verification"]["checks"] = json!([]);
        let report = verify_statement(&statement);
        assert!(report.valid, "{:?}", report.violations);
    }

    #[test]
    fn unrun_verification_cannot_claim_a_verifier_or_checks() {
        let mut statement = minimal_statement();
        statement["predicate"]["execution"]["status"] = json!("not_run");
        statement["predicate"]["execution"]["result"] = json!("not_run");
        statement["predicate"]["execution"]["runId"] = Value::Null;
        statement["predicate"]["verification"]["outcome"] = json!("not_run");
        let report = verify_statement(&statement);
        assert!(!report.valid);
        assert!(has_code(&report, "UNRUN_VERIFICATION_HAS_VERIFIER"));
        assert!(has_code(&report, "UNRUN_VERIFICATION_HAS_TARGET"));
        assert!(has_code(&report, "UNRUN_VERIFICATION_HAS_CHECKS"));
    }

    #[test]
    fn repository_stale_target_fixture_is_rejected() {
        let statement: Value = serde_json::from_str(include_str!(
            "../../../../docs/research/fixtures/luminous-evidence-v0.1/semantic-stale-target.json"
        )).expect("stale-target fixture must be valid JSON");
        let report = verify_statement(&statement);
        assert!(!report.valid);
        assert!(has_code(&report, "LEP_TARGET_MISMATCH"), "{:?}", report.violations);
    }

    #[test]
    fn repository_valid_fixture_is_accepted() {
        let statement: Value = serde_json::from_str(include_str!(
            "../../../../docs/research/fixtures/luminous-evidence-v0.1/valid-passed.json"
        )).expect("valid fixture must be valid JSON");
        let report = verify_statement(&statement);
        assert!(report.valid, "{:?}", report.violations);
    }

    #[test]
    fn repository_queued_but_passed_fixture_is_rejected() {
        let statement: Value = serde_json::from_str(include_str!(
            "../../../../docs/research/fixtures/luminous-evidence-v0.1/invalid-queued-but-passed.json"
        )).expect("queued-but-passed fixture must be valid JSON");
        let report = verify_statement(&statement);
        assert!(!report.valid);
        assert!(has_code(&report, "EXECUTION_STATUS_RESULT_CONFLICT"), "{:?}", report.violations);
    }

    #[test]
    fn repository_missing_target_fixture_is_rejected() {
        let statement: Value = serde_json::from_str(include_str!(
            "../../../../docs/research/fixtures/luminous-evidence-v0.1/invalid-missing-target.json"
        )).expect("missing-target fixture must be valid JSON");
        let report = verify_statement(&statement);
        assert!(!report.valid);
        assert!(has_code(&report, "MISSING_TARGET"), "{:?}", report.violations);
    }

    #[test]
    fn mutable_branch_name_is_not_a_pinned_revision() {
        let mut statement = minimal_statement();
        statement["predicate"]["target"]["source"]["revision"] = json!("main");
        statement["predicate"]["verification"]["checkedTarget"]["source"]["revision"] = json!("main");
        let report = verify_statement(&statement);
        assert!(!report.valid);
        assert!(has_code(&report, "SOURCE_REVISION_NOT_IMMUTABLE"));
    }

    #[test]
    fn duplicate_check_ids_are_rejected() {
        let mut statement = minimal_statement();
        statement["predicate"]["verification"]["checks"] = json!([
            {"id": "same", "outcome": "passed"},
            {"id": "same", "outcome": "passed"}
        ]);
        let report = verify_statement(&statement);
        assert!(!report.valid);
        assert!(has_code(&report, "DUPLICATE_CHECK_ID"));
    }
}
