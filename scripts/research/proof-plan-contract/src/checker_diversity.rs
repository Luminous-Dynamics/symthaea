//! Source-grounded checker-lineage and same-export diversity validation.
//!
//! This module validates registry facts and run-record consistency. It does not
//! execute a checker, establish checker soundness, or qualify a Lean theorem.
//! Unknown pairwise relationships remain unknown and never count as independence.

use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashMap, HashSet};

const CHECKER_IDS: [&str; 6] = [
    "lean_official_kernel",
    "lean4lean",
    "nanoda",
    "con_leche",
    "con_ron",
    "leanchecker_paranoid",
];

const SUMMARY_FIELDS: [&str; 15] = [
    "requested_count",
    "executed_count",
    "accepted_count",
    "rejected_count",
    "declined_count",
    "error_count",
    "timeout_count",
    "resource_exhausted_count",
    "distinct_implementation_families",
    "distinct_algorithm_lineages",
    "distinct_languages",
    "known_term_representation_families",
    "known_parser_families",
    "independent_external_algorithm_lineages",
    "unknown_relationship_count",
];

fn object_string<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(Value::as_str)
}

fn array<'a>(value: &'a Value, key: &str) -> Option<&'a Vec<Value>> {
    value.get(key).and_then(Value::as_array)
}

fn nonempty_string(value: &Value, key: &str) -> bool {
    object_string(value, key).is_some_and(|s| !s.trim().is_empty())
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn sha256_hex(input: &[u8]) -> String {
    let digest = Sha256::digest(input);
    let mut result = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write;
        let _ = write!(result, "{byte:02x}");
    }
    result
}

fn push(errors: &mut Vec<String>, message: impl Into<String>) {
    errors.push(message.into());
}

fn pair_key(a: &str, b: &str) -> String {
    if a <= b {
        format!("{a}::{b}")
    } else {
        format!("{b}::{a}")
    }
}

fn checker_map<'a>(registry: &'a Value) -> Result<HashMap<&'a str, &'a Value>, Vec<String>> {
    let mut errors = Vec::new();
    let Some(checkers) = array(registry, "checkers") else {
        return Err(vec!["registry.checkers must be an array".to_owned()]);
    };
    let mut map = HashMap::new();
    for (index, checker) in checkers.iter().enumerate() {
        let Some(id) = object_string(checker, "checker_id") else {
            push(&mut errors, format!("registry.checkers[{index}].checker_id is missing"));
            continue;
        };
        if map.insert(id, checker).is_some() {
            push(&mut errors, format!("duplicate checker_id: {id}"));
        }
    }
    if errors.is_empty() {
        Ok(map)
    } else {
        Err(errors)
    }
}

pub fn validate_registry(registry: &Value) -> Vec<String> {
    let mut errors = Vec::new();

    if object_string(registry, "schema_version") != Some("1.0.0") {
        push(&mut errors, "unsupported registry schema_version");
    }
    if object_string(registry, "artifact_id") != Some("LEAN-CHECKER-DIVERSITY-REGISTRY-001") {
        push(&mut errors, "registry artifact_id mismatch");
    }
    if object_string(registry, "status")
        != Some("source_grounded_ecosystem_registry_not_a_run_receipt")
    {
        push(&mut errors, "registry status must not imply an observed checker run");
    }

    let map = match checker_map(registry) {
        Ok(map) => map,
        Err(mut map_errors) => {
            errors.append(&mut map_errors);
            return errors;
        }
    };

    let ids: BTreeSet<&str> = map.keys().copied().collect();
    let expected: BTreeSet<&str> = CHECKER_IDS.into_iter().collect();
    if ids != expected {
        push(
            &mut errors,
            "registry must contain exactly the six documented Lean comparator checker paths",
        );
    }

    let bundle: BTreeSet<&str> = registry
        .get("checker_set")
        .and_then(|v| v.get("comparator_paranoid_bundle"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect();
    if bundle != expected {
        push(
            &mut errors,
            "checker_set.comparator_paranoid_bundle must match the six registered checker IDs",
        );
    }

    for checker_id in CHECKER_IDS {
        let Some(checker) = map.get(checker_id) else {
            continue;
        };
        for field in [
            "implementation_family_id",
            "algorithm_lineage_id",
            "independence_class",
            "assurance_basis",
            "revision_policy",
        ] {
            if !nonempty_string(checker, field) {
                push(
                    &mut errors,
                    format!("checker {checker_id} requires nonempty {field}"),
                );
            }
        }
        if !matches!(
            object_string(checker, "implementation_language"),
            Some("cpp" | "lean" | "rust" | "unknown")
        ) {
            push(
                &mut errors,
                format!("checker {checker_id} has an invalid implementation_language"),
            );
        }
        for field in ["representation_family_id", "parser_family_id"] {
            if let Some(value) = checker.get(field) {
                if !value.is_null()
                    && value
                        .as_str()
                        .is_none_or(|s| s.trim().is_empty())
                {
                    push(
                        &mut errors,
                        format!("checker {checker_id} has invalid optional {field}"),
                    );
                }
            } else {
                push(
                    &mut errors,
                    format!("checker {checker_id} is missing {field}; unknown must be explicit null"),
                );
            }
        }
        if array(checker, "evidence_refs").is_none_or(|refs| refs.is_empty()) {
            push(
                &mut errors,
                format!("checker {checker_id} requires source evidence references"),
            );
        }
        if array(checker, "residual_trust").is_none_or(|risks| risks.is_empty()) {
            push(
                &mut errors,
                format!("checker {checker_id} requires residual trust assumptions"),
            );
        }
    }

    // Pinned lineage constraints prevent the most tempting double counts.
    let lineage = |id: &str| {
        map.get(id)
            .and_then(|checker| object_string(checker, "algorithm_lineage_id"))
    };
    if lineage("lean4lean") != lineage("lean_official_kernel") {
        push(&mut errors, "lean4lean must remain grouped with the official kernel lineage");
    }
    if lineage("leanchecker_paranoid") != lineage("lean_official_kernel") {
        push(
            &mut errors,
            "leanchecker-paranoid must remain grouped with the official kernel lineage",
        );
    }
    if lineage("con_ron") != lineage("con_leche") {
        push(
            &mut errors,
            "con-ron must remain grouped with con-leche's algorithm lineage",
        );
    }
    for id in ["nanoda", "con_leche"] {
        if object_string(map[id], "independence_class") != Some("independent_implementation") {
            push(&mut errors, format!("{id} must be recorded as an independently implemented external checker"));
        }
    }
    if object_string(map["con_ron"], "independence_class")
        != Some("verified_port_shared_algorithm")
    {
        push(&mut errors, "con-ron must be recorded as a verified port sharing con-leche's algorithm");
    }
    if object_string(map["lean4lean"], "independence_class")
        != Some("derived_from_official")
    {
        push(&mut errors, "lean4lean must be recorded as derived from the official kernel");
    }
    if object_string(map["leanchecker_paranoid"], "independence_class")
        != Some("hardening_variant")
    {
        push(&mut errors, "leanchecker-paranoid must be recorded as a hardening variant");
    }

    let relationships = match array(registry, "relationships") {
        Some(relations) => relations,
        None => {
            push(&mut errors, "registry.relationships must be an array");
            return errors;
        }
    };
    let expected_pairs = ids.len() * (ids.len() - 1) / 2;
    if relationships.len() != expected_pairs {
        push(
            &mut errors,
            format!("registry must explicitly classify all {expected_pairs} checker pairs"),
        );
    }
    let mut observed_pairs = HashSet::new();
    let known_ids: HashSet<&str> = ids.iter().copied().collect();
    for (index, relationship) in relationships.iter().enumerate() {
        let from = object_string(relationship, "from");
        let to = object_string(relationship, "to");
        let (Some(from), Some(to)) = (from, to) else {
            push(
                &mut errors,
                format!("relationship[{index}] is missing from/to checker IDs"),
            );
            continue;
        };
        if from == to || !known_ids.contains(from) || !known_ids.contains(to) {
            push(
                &mut errors,
                format!("relationship[{index}] references an invalid checker pair"),
            );
            continue;
        }
        if !observed_pairs.insert(pair_key(from, to)) {
            push(
                &mut errors,
                format!("duplicate pairwise checker relationship: {from}/{to}"),
            );
        }
        if object_string(relationship, "relation") == Some("unknown") {
            if object_string(relationship, "independence_effect") != Some("unknown_not_counted") {
                push(
                    &mut errors,
                    format!("unknown pair {from}/{to} must have independence_effect=unknown_not_counted"),
                );
            }
            if array(relationship, "evidence_refs").is_some_and(|refs| !refs.is_empty()) {
                push(
                    &mut errors,
                    format!("unknown pair {from}/{to} must not invent supporting evidence"),
                );
            }
        } else if array(relationship, "evidence_refs").is_none_or(|refs| refs.is_empty()) {
            push(
                &mut errors,
                format!("known pairwise relation {from}/{to} requires evidence_refs"),
            );
        }
    }

    if observed_pairs.len() != expected_pairs {
        push(
            &mut errors,
            "pairwise checker relationship graph is incomplete",
        );
    }
    if array(registry, "common_mode_domains").is_none_or(|domains| domains.is_empty()) {
        push(&mut errors, "registry must enumerate common-mode risk domains");
    }

    errors
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiversitySummary {
    pub requested_count: usize,
    pub executed_count: usize,
    pub accepted_count: usize,
    pub rejected_count: usize,
    pub declined_count: usize,
    pub error_count: usize,
    pub timeout_count: usize,
    pub resource_exhausted_count: usize,
    pub distinct_implementation_families: usize,
    pub distinct_algorithm_lineages: usize,
    pub distinct_languages: usize,
    pub known_term_representation_families: usize,
    pub known_parser_families: usize,
    pub independent_external_algorithm_lineages: usize,
    pub unknown_relationship_count: usize,
}

fn summarize(
    registry: &Value,
    profile: &Value,
) -> Result<DiversitySummary, Vec<String>> {
    let mut errors = Vec::new();
    let map = checker_map(registry)?;
    let requested: Vec<&str> = array(profile, "requested_checker_ids")
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect();
    let runs = array(profile, "checker_runs").ok_or_else(|| {
        vec!["profile.checker_runs must be an array".to_owned()]
    })?;

    let mut run_ids = HashSet::new();
    for run in runs {
        if let Some(id) = object_string(run, "checker_id") {
            if !run_ids.insert(id) {
                push(&mut errors, format!("duplicate checker run for {id}"));
            }
        }
    }
    let requested_set: HashSet<&str> = requested.iter().copied().collect();
    let run_set: HashSet<&str> = run_ids.iter().copied().collect();
    if requested_set.len() != requested.len() {
        push(&mut errors, "requested_checker_ids contains duplicates");
    }
    if requested_set != run_set {
        push(
            &mut errors,
            "checker_runs must contain exactly one record for every requested checker",
        );
    }

    let mut executed_count = 0usize;
    let mut accepted_count = 0usize;
    let mut rejected_count = 0usize;
    let mut declined_count = 0usize;
    let mut error_count = 0usize;
    let mut timeout_count = 0usize;
    let mut resource_exhausted_count = 0usize;

    let mut implementation_families = BTreeSet::new();
    let mut algorithm_lineages = BTreeSet::new();
    let mut languages = BTreeSet::new();
    let mut representations = BTreeSet::new();
    let mut parsers = BTreeSet::new();
    let mut independent_lineages = BTreeSet::new();

    let export = profile
        .get("subject")
        .and_then(|subject| object_string(subject, "export_sha256"));
    if export.is_none_or(|digest| !is_sha256(digest)) {
        push(&mut errors, "subject.export_sha256 must be a lowercase SHA-256 digest");
    }

    for run in runs {
        let Some(checker_id) = object_string(run, "checker_id") else {
            push(&mut errors, "checker run is missing checker_id");
            continue;
        };
        let Some(checker) = map.get(checker_id) else {
            push(&mut errors, format!("checker run references unregistered checker {checker_id}"));
            continue;
        };

        if object_string(run, "input_export_sha256") != export {
            push(
                &mut errors,
                format!("checker {checker_id} did not bind the exact subject export digest"),
            );
        }
        if !nonempty_string(run, "checker_version")
            || !nonempty_string(run, "implementation_revision")
            || !nonempty_string(run, "binary_sha256")
        {
            push(
                &mut errors,
                format!("checker {checker_id} is missing pinned version/revision/binary identity"),
            );
        }
        if let Some(binary_sha) = object_string(run, "binary_sha256") {
            if !is_sha256(binary_sha) {
                push(&mut errors, format!("checker {checker_id} has invalid binary_sha256"));
            }
        }
        let result = object_string(run, "result").unwrap_or("invalid");
        if result != "not_run"
            && object_string(run, "raw_receipt_sha256")
                .is_none_or(|digest| !is_sha256(digest))
        {
            push(
                &mut errors,
                format!("checker {checker_id} requires a raw receipt digest"),
            );
        }
        if result != "not_run" {
            executed_count += 1;
        }
        match result {
            "accepted" => {
                accepted_count += 1;
                implementation_families.insert(
                    object_string(checker, "implementation_family_id").unwrap_or("unknown"),
                );
                algorithm_lineages.insert(
                    object_string(checker, "algorithm_lineage_id").unwrap_or("unknown"),
                );
                languages.insert(
                    object_string(checker, "implementation_language").unwrap_or("unknown"),
                );
                if let Some(representation) = object_string(checker, "representation_family_id") {
                    representations.insert(representation);
                }
                if let Some(parser) = object_string(checker, "parser_family_id") {
                    parsers.insert(parser);
                }
                if object_string(checker, "independence_class")
                    == Some("independent_implementation")
                {
                    independent_lineages.insert(
                        object_string(checker, "algorithm_lineage_id").unwrap_or("unknown"),
                    );
                }
            }
            "rejected" => rejected_count += 1,
            "declined" => declined_count += 1,
            "error" => error_count += 1,
            "timeout" => timeout_count += 1,
            "resource_exhausted" => resource_exhausted_count += 1,
            "not_run" => {}
            _ => push(
                &mut errors,
                format!("checker {checker_id} has an unknown result state"),
            ),
        }
    }

    let unknown_relationship_count = array(registry, "relationships")
        .into_iter()
        .flatten()
        .filter(|rel| object_string(rel, "relation") == Some("unknown"))
        .count();

    if !errors.is_empty() {
        return Err(errors);
    }

    Ok(DiversitySummary {
        requested_count: requested.len(),
        executed_count,
        accepted_count,
        rejected_count,
        declined_count,
        error_count,
        timeout_count,
        resource_exhausted_count,
        distinct_implementation_families: implementation_families.len(),
        distinct_algorithm_lineages: algorithm_lineages.len(),
        distinct_languages: languages.len(),
        known_term_representation_families: representations.len(),
        known_parser_families: parsers.len(),
        independent_external_algorithm_lineages: independent_lineages.len(),
        unknown_relationship_count,
    })
}

fn summary_value(summary: &DiversitySummary, key: &str) -> Option<usize> {
    match key {
        "requested_count" => Some(summary.requested_count),
        "executed_count" => Some(summary.executed_count),
        "accepted_count" => Some(summary.accepted_count),
        "rejected_count" => Some(summary.rejected_count),
        "declined_count" => Some(summary.declined_count),
        "error_count" => Some(summary.error_count),
        "timeout_count" => Some(summary.timeout_count),
        "resource_exhausted_count" => Some(summary.resource_exhausted_count),
        "distinct_implementation_families" => Some(summary.distinct_implementation_families),
        "distinct_algorithm_lineages" => Some(summary.distinct_algorithm_lineages),
        "distinct_languages" => Some(summary.distinct_languages),
        "known_term_representation_families" => Some(summary.known_term_representation_families),
        "known_parser_families" => Some(summary.known_parser_families),
        "independent_external_algorithm_lineages" => {
            Some(summary.independent_external_algorithm_lineages)
        }
        "unknown_relationship_count" => Some(summary.unknown_relationship_count),
        _ => None,
    }
}

pub fn validate_profile(
    registry: &Value,
    profile: &Value,
    registry_input_bytes: &[u8],
) -> Vec<String> {
    let mut errors = validate_registry(registry);
    if !errors.is_empty() {
        return errors;
    }

    if object_string(profile, "schema_version") != Some("1.0.0")
        || object_string(profile, "artifact_id") != Some("LEAN-CHECKER-DIVERSITY-RUN-001")
    {
        push(&mut errors, "invalid checker diversity run profile identity");
    }

    let expected_registry_sha = sha256_hex(registry_input_bytes);
    if object_string(profile, "registry_sha256") != Some(expected_registry_sha.as_str()) {
        push(
            &mut errors,
            "profile.registry_sha256 does not bind the exact registry bytes",
        );
    }

    match object_string(profile, "profile_status") {
        Some("synthetic_fixture_not_execution" | "observed_exact_export_run") => {}
        _ => push(&mut errors, "invalid profile_status"),
    }

    match object_string(profile, "profile_disposition") {
        Some("draft" | "all_required_accepted" | "rejected" | "inconclusive") => {}
        _ => push(&mut errors, "invalid profile_disposition"),
    }

    let summary = match summarize(registry, profile) {
        Ok(summary) => summary,
        Err(mut summary_errors) => {
            errors.append(&mut summary_errors);
            return errors;
        }
    };

    let recorded = profile.get("diversity_summary").unwrap_or(&Value::Null);
    for key in SUMMARY_FIELDS {
        let expected = summary_value(&summary, key).unwrap_or_default() as u64;
        let actual = recorded.get(key).and_then(Value::as_u64);
        if actual != Some(expected) {
            push(
                &mut errors,
                format!("diversity_summary.{key} mismatch: expected {expected}, got {actual:?}"),
            );
        }
    }

    let requirements = profile.get("diversity_requirements").unwrap_or(&Value::Null);
    let min_algorithm = requirements
        .get("minimum_algorithm_lineages")
        .and_then(Value::as_u64)
        .unwrap_or(u64::MAX) as usize;
    let min_independent = requirements
        .get("minimum_independent_external_algorithm_lineages")
        .and_then(Value::as_u64)
        .unwrap_or(u64::MAX) as usize;
    let min_languages = requirements
        .get("minimum_languages")
        .and_then(Value::as_u64)
        .unwrap_or(u64::MAX) as usize;
    let min_implementations = requirements
        .get("minimum_implementation_families")
        .and_then(Value::as_u64)
        .unwrap_or(u64::MAX) as usize;
    if requirements.get("require_all_requested_accept").and_then(Value::as_bool) != Some(true) {
        push(
            &mut errors,
            "require_all_requested_accept must be true for a checker concordance profile",
        );
    }

    let requested_count = summary.requested_count;
    let accepted_all = summary.accepted_count == requested_count
        && summary.executed_count == requested_count
        && summary.rejected_count == 0
        && summary.declined_count == 0
        && summary.error_count == 0
        && summary.timeout_count == 0
        && summary.resource_exhausted_count == 0;

    let has_rejection = summary.rejected_count > 0;
    let diversity_satisfied = summary.distinct_algorithm_lineages >= min_algorithm
        && summary.independent_external_algorithm_lineages >= min_independent
        && summary.distinct_languages >= min_languages
        && summary.distinct_implementation_families >= min_implementations;

    let orchestrator_success = profile
        .get("orchestrator")
        .and_then(|v| object_string(v, "exit_status"))
        == Some("success");

    let expected_disposition = match object_string(profile, "profile_status") {
        Some("synthetic_fixture_not_execution") => "draft",
        Some("observed_exact_export_run") if has_rejection => "rejected",
        Some("observed_exact_export_run")
            if accepted_all && diversity_satisfied && orchestrator_success =>
        {
            "all_required_accepted"
        }
        Some("observed_exact_export_run") => "inconclusive",
        _ => "draft",
    };
    if object_string(profile, "profile_disposition") != Some(expected_disposition) {
        push(
            &mut errors,
            format!(
                "profile_disposition must be {expected_disposition} for the recorded checker outcomes"
            ),
        );
    }

    if expected_disposition == "all_required_accepted" && !diversity_satisfied {
        push(&mut errors, "checker concordance cannot pass below the declared diversity requirements");
    }

    errors
}

pub fn validate_json_inputs(
    registry_input: &str,
    profile_input: &str,
) -> Result<DiversitySummary, Vec<String>> {
    let registry: Value = serde_json::from_str(registry_input)
        .map_err(|error| vec![format!("invalid checker registry JSON: {error}")])?;
    let profile: Value = serde_json::from_str(profile_input)
        .map_err(|error| vec![format!("invalid checker run profile JSON: {error}")])?;
    let errors = validate_profile(&registry, &profile, registry_input.as_bytes());
    if errors.is_empty() {
        summarize(&registry, &profile)
    } else {
        Err(errors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const REGISTRY: &str =
        include_str!("../../../../docs/research/LEAN_CHECKER_DIVERSITY_REGISTRY_V1.json");
    const SYNTHETIC_PROFILE: &str =
        include_str!("../../../../docs/research/LEAN_CHECKER_DIVERSITY_RUN_SYNTHETIC_V1.json");

    fn values() -> (Value, Value) {
        (
            serde_json::from_str(REGISTRY).expect("registry JSON"),
            serde_json::from_str(SYNTHETIC_PROFILE).expect("profile JSON"),
        )
    }

    #[test]
    fn validates_source_grounded_registry() {
        let (registry, _) = values();
        assert!(validate_registry(&registry).is_empty());
    }

    #[test]
    fn validates_synthetic_profile_without_claiming_execution() {
        let (registry, profile) = values();
        let bytes = REGISTRY.as_bytes();
        assert!(validate_profile(&registry, &profile, bytes).is_empty());
        assert_eq!(profile["profile_disposition"], "draft");
    }

    #[test]
    fn computes_three_algorithm_lineages_not_six() {
        let result = validate_json_inputs(REGISTRY, SYNTHETIC_PROFILE).expect("valid fixture");
        assert_eq!(result.requested_count, 6);
        assert_eq!(result.distinct_implementation_families, 6);
        assert_eq!(result.distinct_algorithm_lineages, 3);
        assert_eq!(result.independent_external_algorithm_lineages, 2);
        assert_eq!(result.distinct_languages, 3);
        assert_eq!(result.known_term_representation_families, 3);
        assert_eq!(result.known_parser_families, 1);
        assert_eq!(result.unknown_relationship_count, 8);
    }

    #[test]
    fn rejects_export_digest_mismatch() {
        let (registry, mut profile) = values();
        profile["checker_runs"][0]["input_export_sha256"] = json!("b".repeat(64));
        let bytes = REGISTRY.as_bytes();
        let errors = validate_profile(&registry, &profile, bytes);
        assert!(errors.iter().any(|error| error.contains("exact subject export")));
    }

    #[test]
    fn rejects_registry_digest_drift() {
        let (registry, mut profile) = values();
        profile["registry_sha256"] = json!("f".repeat(64));
        let errors = validate_profile(&registry, &profile, REGISTRY.as_bytes());
        assert!(errors.iter().any(|error| error.contains("exact registry bytes")));
    }

    #[test]
    fn rejects_optimistic_algorithm_diversity_count() {
        let (registry, mut profile) = values();
        profile["diversity_summary"]["distinct_algorithm_lineages"] = json!(6);
        let errors = validate_profile(&registry, &profile, REGISTRY.as_bytes());
        assert!(errors.iter().any(|error| error.contains("distinct_algorithm_lineages mismatch")));
    }

    #[test]
    fn treats_decline_as_inconclusive_not_rejection() {
        let (registry, mut profile) = values();
        profile["profile_status"] = json!("observed_exact_export_run");
        profile["profile_disposition"] = json!("inconclusive");
        profile["orchestrator"]["exit_status"] = json!("failure");
        profile["checker_runs"][1]["result"] = json!("declined");
        profile["diversity_summary"]["accepted_count"] = json!(5);
        profile["diversity_summary"]["declined_count"] = json!(1);
        // The failed checker no longer contributes to observed diversity.
        profile["diversity_summary"]["distinct_implementation_families"] = json!(5);
        profile["diversity_summary"]["distinct_algorithm_lineages"] = json!(3);
        profile["diversity_summary"]["distinct_languages"] = json!(3);
        profile["diversity_summary"]["known_term_representation_families"] = json!(3);
        profile["diversity_summary"]["known_parser_families"] = json!(1);
        profile["diversity_summary"]["independent_external_algorithm_lineages"] = json!(2);
        let errors = validate_profile(&registry, &profile, REGISTRY.as_bytes());
        assert!(errors.is_empty(), "{errors:?}");
    }

    #[test]
    fn treats_timeout_as_inconclusive_not_rejection() {
        let (registry, mut profile) = values();
        profile["profile_status"] = json!("observed_exact_export_run");
        profile["profile_disposition"] = json!("inconclusive");
        profile["orchestrator"]["exit_status"] = json!("failure");
        profile["checker_runs"][2]["result"] = json!("timeout");
        profile["diversity_summary"]["accepted_count"] = json!(5);
        profile["diversity_summary"]["timeout_count"] = json!(1);
        profile["diversity_summary"]["distinct_implementation_families"] = json!(5);
        profile["diversity_summary"]["distinct_algorithm_lineages"] = json!(3);
        profile["diversity_summary"]["distinct_languages"] = json!(3);
        profile["diversity_summary"]["known_term_representation_families"] = json!(3);
        profile["diversity_summary"]["known_parser_families"] = json!(1);
        profile["diversity_summary"]["independent_external_algorithm_lineages"] = json!(2);
        let errors = validate_profile(&registry, &profile, REGISTRY.as_bytes());
        assert!(errors.is_empty(), "{errors:?}");
    }

    #[test]
    fn a_checker_rejection_is_not_overridden_by_other_acceptances() {
        let (registry, mut profile) = values();
        profile["profile_status"] = json!("observed_exact_export_run");
        profile["profile_disposition"] = json!("rejected");
        profile["orchestrator"]["exit_status"] = json!("failure");
        profile["checker_runs"][3]["result"] = json!("rejected");
        profile["diversity_summary"]["accepted_count"] = json!(5);
        profile["diversity_summary"]["rejected_count"] = json!(1);
        profile["diversity_summary"]["distinct_implementation_families"] = json!(5);
        profile["diversity_summary"]["distinct_algorithm_lineages"] = json!(3);
        profile["diversity_summary"]["distinct_languages"] = json!(3);
        profile["diversity_summary"]["known_term_representation_families"] = json!(3);
        profile["diversity_summary"]["known_parser_families"] = json!(1);
        profile["diversity_summary"]["independent_external_algorithm_lineages"] = json!(1);
        let errors = validate_profile(&registry, &profile, REGISTRY.as_bytes());
        assert!(errors.is_empty(), "{errors:?}");
    }
}
