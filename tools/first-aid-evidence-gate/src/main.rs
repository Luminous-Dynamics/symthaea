use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::{env, fs, process};

const DEFAULT_REGISTRY: &str = "docs/research/first-aid/clinical-evidence-sources-v0.json";
const AUDITOR_SOURCE: &[u8] = include_bytes!("main.rs");

#[derive(Debug, PartialEq, Eq)]
struct AuditSummary {
    source_count: usize,
    claim_count: usize,
    approved_claim_count: usize,
}

fn field_str<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(Value::as_str)
}

fn require_text(value: &Value, key: &str, context: &str, errors: &mut Vec<String>) {
    if field_str(value, key).is_none_or(|text| text.trim().is_empty()) {
        errors.push(format!("{context}: required non-empty string '{key}'"));
    }
}

fn check_bool(
    value: &Value,
    key: &str,
    expected: bool,
    context: &str,
    errors: &mut Vec<String>,
) {
    if value.get(key).and_then(Value::as_bool) != Some(expected) {
        errors.push(format!("{context}: '{key}' must be {expected}"));
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<Vec<_>>()
        .join("")
}

fn validate_registry(root: &Value) -> Result<AuditSummary, Vec<String>> {
    let mut errors = Vec::new();

    if !root.is_object() {
        return Err(vec!["registry root must be a JSON object".to_owned()]);
    }
    if field_str(root, "schema_version") != Some("first-aid-clinical-evidence-registry-v0") {
        errors.push("schema_version is missing or unsupported".to_owned());
    }
    if field_str(root, "evidence_class") != Some("source_metadata_only") {
        errors.push("evidence_class must remain source_metadata_only".to_owned());
    }
    check_bool(root, "clinical_use_authorized", false, "registry", &mut errors);
    require_text(root, "registry_version", "registry", &mut errors);
    require_text(root, "checked_at", "registry", &mut errors);

    let null_value = Value::Null;
    let policy = root.get("claim_admission_policy").unwrap_or(&null_value);
    if !policy.is_object() {
        errors.push("claim_admission_policy must be an object".to_owned());
    }
    for key in [
        "separate_qualified_adjudicator_required",
        "exact_source_locator_required",
        "population_setting_jurisdiction_required",
        "localization_review_required",
        "review_date_and_source_update_trigger_required",
    ] {
        check_bool(policy, key, true, "claim_admission_policy", &mut errors);
    }
    check_bool(
        policy,
        "patient_data_allowed",
        false,
        "claim_admission_policy",
        &mut errors,
    );
    check_bool(
        policy,
        "source_text_reproduction_authorized",
        false,
        "claim_admission_policy",
        &mut errors,
    );
    if policy
        .get("minimum_independent_reviewers")
        .and_then(Value::as_u64)
        != Some(2)
    {
        errors.push(
            "claim_admission_policy.minimum_independent_reviewers must equal 2".to_owned(),
        );
    }
    if field_str(policy, "approved_claim_status")
        != Some("approved_for_reference_evaluation_only")
    {
        errors.push("claim_admission_policy.approved_claim_status is unsupported".to_owned());
    }

    let sources: &[Value] = match root.get("sources").and_then(Value::as_array) {
        Some(sources) if !sources.is_empty() => sources.as_slice(),
        _ => {
            errors.push("sources must be a non-empty array".to_owned());
            &[]
        }
    };
    let mut source_ids = BTreeSet::new();
    for (index, source) in sources.iter().enumerate() {
        let context = format!("sources[{index}]");
        for key in [
            "source_id",
            "publisher",
            "title",
            "edition",
            "publication_or_update_date",
            "url",
            "scope",
            "source_role",
            "jurisdiction_relevance",
            "localization_status",
            "claim_use_status",
        ] {
            require_text(source, key, &context, &mut errors);
        }

        if let Some(id) = field_str(source, "source_id") {
            if !source_ids.insert(id.to_owned()) {
                errors.push(format!("{context}: duplicate source_id '{id}'"));
            }
        }
        if field_str(source, "url").is_none_or(|url| !url.starts_with("https://")) {
            errors.push(format!("{context}: source URL must use HTTPS"));
        }
        if field_str(source, "claim_use_status")
            != Some("source_metadata_only_not_adjudicated")
        {
            errors.push(format!(
                "{context}: source metadata cannot itself adjudicate a clinical claim"
            ));
        }
        if field_str(source, "localization_status") != Some("local_adaptation_required") {
            errors.push(format!(
                "{context}: source must explicitly require local adaptation"
            ));
        }
        check_bool(source, "text_reproduced", false, &context, &mut errors);
    }

    let claims: &[Value] = match root.get("claims").and_then(Value::as_array) {
        Some(claims) => claims.as_slice(),
        None => {
            errors.push("claims must be an array (use [] when none are admitted)".to_owned());
            &[]
        }
    };
    let mut claim_ids = BTreeSet::new();
    let mut approved_claim_count = 0usize;
    for (index, claim) in claims.iter().enumerate() {
        let context = format!("claims[{index}]");
        for key in [
            "claim_id",
            "topic",
            "population",
            "setting",
            "target_jurisdiction",
            "locale",
            "source_edition",
            "source_locator",
            "limitations",
            "claim_status",
        ] {
            require_text(claim, key, &context, &mut errors);
        }
        if let Some(id) = field_str(claim, "claim_id") {
            if !claim_ids.insert(id.to_owned()) {
                errors.push(format!("{context}: duplicate claim_id '{id}'"));
            }
        }
        let claim_source_ids = match claim.get("source_ids").and_then(Value::as_array) {
            Some(ids) if !ids.is_empty() => ids,
            _ => {
                errors.push(format!("{context}: source_ids must be a non-empty array"));
                &[]
            }
        };
        for source_id in claim_source_ids.iter().filter_map(Value::as_str) {
            if !source_ids.contains(source_id) {
                errors.push(format!("{context}: unknown source_id '{source_id}'"));
            }
        }

        let status = field_str(claim, "claim_status").unwrap_or_default();
        if ![
            "unreviewed",
            "needs_review",
            "approved_for_reference_evaluation_only",
        ]
        .contains(&status)
        {
            errors.push(format!("{context}: unsupported claim_status '{status}'"));
        }
        if status == "approved_for_reference_evaluation_only" {
            approved_claim_count += 1;
            if field_str(claim, "localization_status")
                != Some("reviewed_for_target_jurisdiction")
            {
                errors.push(format!(
                    "{context}: approved claim lacks target-jurisdiction localization review"
                ));
            }
            require_text(claim, "review_date", &context, &mut errors);
            require_text(claim, "source_update_trigger", &context, &mut errors);

            let reviewers: &[Value] = match claim.get("reviewers").and_then(Value::as_array) {
                Some(reviewers) if reviewers.len() >= 2 => reviewers.as_slice(),
                _ => {
                    errors.push(format!(
                        "{context}: approval requires at least two independent reviewers"
                    ));
                    &[]
                }
            };
            let mut reviewer_ids = BTreeSet::new();
            let mut approving_reviewers = 0usize;
            for reviewer in reviewers {
                for key in ["reviewer_id", "qualification", "decision"] {
                    require_text(reviewer, key, &context, &mut errors);
                }
                if let Some(id) = field_str(reviewer, "reviewer_id") {
                    reviewer_ids.insert(id.to_owned());
                }
                if reviewer.get("independent").and_then(Value::as_bool) != Some(true) {
                    errors.push(format!("{context}: reviewer must attest independence"));
                }
                if field_str(reviewer, "decision") == Some("approve") {
                    approving_reviewers += 1;
                }
            }
            if reviewer_ids.len() < 2 || approving_reviewers < 2 {
                errors.push(format!(
                    "{context}: two distinct independent approving reviewer records are required"
                ));
            }

            let null_adjudicator = Value::Null;
            let adjudicator = claim.get("adjudicator").unwrap_or(&null_adjudicator);
            require_text(adjudicator, "qualification", &context, &mut errors);
            if adjudicator.get("independent").and_then(Value::as_bool) != Some(true)
                || field_str(adjudicator, "decision") != Some("approve")
            {
                errors.push(format!(
                    "{context}: approval requires a separate independent approving adjudicator"
                ));
            }
        }
    }

    if errors.is_empty() {
        Ok(AuditSummary {
            source_count: sources.len(),
            claim_count: claims.len(),
            approved_claim_count,
        })
    } else {
        Err(errors)
    }
}

fn main() {
    let path = env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_REGISTRY.to_owned());
    let bytes = match fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("registry read failed for '{path}': {error}");
            process::exit(2);
        }
    };
    let root: Value = match serde_json::from_slice(&bytes) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("registry JSON rejected: {error}");
            process::exit(2);
        }
    };
    let summary = match validate_registry(&root) {
        Ok(summary) => summary,
        Err(errors) => {
            for error in errors {
                eprintln!("REJECT: {error}");
            }
            process::exit(2);
        }
    };

    let disposition = if summary.approved_claim_count == 0 {
        "BLOCKED_NO_ADJUDICATED_CLAIMS"
    } else {
        "ELIGIBLE_FOR_REFERENCE_EVALUATION_ONLY"
    };
    let receipt = json!({
        "receipt_schema": "first-aid-evidence-gate-receipt-v0",
        "registry_path": path,
        "registry_version": field_str(&root, "registry_version").unwrap_or("unknown"),
        "registry_sha256": sha256_hex(&bytes),
        "auditor_source_sha256": sha256_hex(AUDITOR_SOURCE),
        "source_count": summary.source_count,
        "claim_count": summary.claim_count,
        "approved_claim_count": summary.approved_claim_count,
        "claim_disposition": disposition,
        "clinical_use_authorized": false,
        "evidence_class": "metadata_process_validation_only",
        "claim_ceiling": "Registry integrity only. No clinical correctness, patient benefit, safety, regulatory, or deployment claim."
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&receipt).expect("receipt serialization cannot fail")
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registry() -> Value {
        serde_json::from_slice(include_bytes!(
            "../../../docs/research/first-aid/clinical-evidence-sources-v0.json"
        ))
        .expect("committed source registry parses")
    }

    #[test]
    fn current_registry_is_integral_but_blocks_all_clinical_claims() {
        let result = validate_registry(&registry()).expect("metadata registry should validate");
        assert_eq!(result.source_count, 3);
        assert_eq!(result.claim_count, 0);
        assert_eq!(result.approved_claim_count, 0);
        assert_eq!(sha256_hex(AUDITOR_SOURCE).len(), 64);
    }

    #[test]
    fn clinical_use_authorization_is_always_rejected() {
        let mut input = registry();
        input["clinical_use_authorized"] = Value::Bool(true);
        let errors =
            validate_registry(&input).expect_err("clinical use must not be authorized here");
        assert!(errors
            .iter()
            .any(|error| error.contains("clinical_use_authorized")));
    }

    #[test]
    fn duplicate_source_ids_are_rejected() {
        let mut input = registry();
        let first = input["sources"][0].clone();
        input["sources"]
            .as_array_mut()
            .expect("sources array")
            .push(first);
        let errors = validate_registry(&input).expect_err("duplicate source must fail closed");
        assert!(errors.iter().any(|error| error.contains("duplicate source_id")));
    }

    #[test]
    fn claim_cannot_be_admitted_without_jurisdiction_and_review_records() {
        let mut input = registry();
        input["claims"] = json!([{
            "claim_id": "TEST-001",
            "topic": "synthetic_topic_only",
            "population": "must_be_reviewed",
            "setting": "must_be_reviewed",
            "target_jurisdiction": "ZA",
            "locale": "en-ZA",
            "source_edition": "2025",
            "source_locator": "not_a_real_clinical_claim",
            "limitations": "synthetic gate test",
            "claim_status": "approved_for_reference_evaluation_only",
            "source_ids": ["IFRC-IFA-2025"],
            "localization_status": "not_reviewed"
        }]);
        let errors = validate_registry(&input)
            .expect_err("approval without independent review must fail");
        assert!(errors.iter().any(|error| error.contains("localization review")));
        assert!(errors
            .iter()
            .any(|error| error.contains("at least two independent reviewers")));
        assert!(errors.iter().any(|error| error.contains("adjudicator")));
    }

    #[test]
    fn unknown_source_references_are_rejected() {
        let mut input = registry();
        input["claims"] = json!([{
            "claim_id": "TEST-002",
            "topic": "synthetic_topic_only",
            "population": "must_be_reviewed",
            "setting": "must_be_reviewed",
            "target_jurisdiction": "ZA",
            "locale": "en-ZA",
            "source_edition": "2025",
            "source_locator": "not_a_real_clinical_claim",
            "limitations": "synthetic gate test",
            "claim_status": "unreviewed",
            "source_ids": ["MISSING-SOURCE"]
        }]);
        let errors = validate_registry(&input).expect_err("unknown source must fail");
        assert!(errors.iter().any(|error| error.contains("unknown source_id")));
    }

    #[test]
    fn malformed_or_missing_claim_array_is_rejected() {
        let mut input = registry();
        input.as_object_mut().expect("root object").remove("claims");
        let errors = validate_registry(&input).expect_err("missing claim array must be rejected");
        assert!(errors.iter().any(|error| error.contains("claims must be an array")));
    }
}
