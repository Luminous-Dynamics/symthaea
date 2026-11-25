//! Cross-field semantic validator for LeanTrustClosureCapsuleV1.
//!
//! This validates assurance-record consistency. It does not run Lean, verify a
//! proof term, establish DFS completeness, or replace a kernel checker.

use serde_json::Value;
use std::collections::HashSet;

const AXES: [&str; 7] = [
    "statement_fidelity",
    "logical_dependency_closure",
    "official_kernel_replay",
    "independent_checker_replay",
    "construction_containment",
    "environment_reproducibility",
    "computational_trust",
];

fn at<'a>(value: &'a Value, path: &[&str]) -> Option<&'a Value> {
    path.iter().try_fold(value, |current, key| current.get(*key))
}

fn string_at<'a>(value: &'a Value, path: &[&str]) -> Option<&'a str> {
    at(value, path).and_then(Value::as_str)
}

fn bool_at(value: &Value, path: &[&str]) -> Option<bool> {
    at(value, path).and_then(Value::as_bool)
}

fn array_at<'a>(value: &'a Value, path: &[&str]) -> Option<&'a Vec<Value>> {
    at(value, path).and_then(Value::as_array)
}

fn is_hex(value: &str, length: usize) -> bool {
    value.len() == length && value.bytes().all(|b| b.is_ascii_hexdigit())
}

fn axis_status<'a>(value: &'a Value, axis: &str) -> Option<&'a str> {
    string_at(value, &["assurance_axes", axis, "status"])
}

fn is_satisfied(value: &Value, axis: &str) -> bool {
    axis_status(value, axis) == Some("satisfied")
}

fn push_missing(errors: &mut Vec<String>, path: &str) {
    errors.push(format!("{path} is missing or has the wrong type"));
}

fn check_digest(value: &Value, path: &[&str], label: &str, errors: &mut Vec<String>) {
    match string_at(value, path) {
        Some(digest) if is_hex(digest, 64) => {}
        _ => push_missing(errors, label),
    }
}

fn check_subject_and_environment(value: &Value, errors: &mut Vec<String>) {
    let repo = string_at(value, &["subject", "repository"]);
    if repo.is_none_or(|s| !s.contains('/') || s.starts_with('/') || s.ends_with('/')) {
        push_missing(errors, "subject.repository");
    }
    match string_at(value, &["subject", "commit"]) {
        Some(commit) if is_hex(commit, 40) => {}
        _ => push_missing(errors, "subject.commit"),
    }
    check_digest(value, &["subject", "source", "sha256"], "subject.source.sha256", errors);
    check_digest(
        value,
        &["subject", "challenge", "artifact", "sha256"],
        "subject.challenge.artifact.sha256",
        errors,
    );
    check_digest(
        value,
        &["subject", "challenge", "target_identity_sha256"],
        "subject.challenge.target_identity_sha256",
        errors,
    );

    if string_at(value, &["environment", "lean_toolchain"]).is_none() {
        push_missing(errors, "environment.lean_toolchain");
    }
    match string_at(value, &["environment", "lean_core_revision"]) {
        Some(revision) if is_hex(revision, 40) => {}
        _ => push_missing(errors, "environment.lean_core_revision"),
    }
    check_digest(
        value,
        &["environment", "lake_manifest_sha256"],
        "environment.lake_manifest_sha256",
        errors,
    );
    let imports = array_at(value, &["environment", "imports"]);
    if let Some(imports) = imports {
        for (index, import) in imports.iter().enumerate() {
            if string_at(import, &["module"]).is_none() {
                push_missing(errors, &format!("environment.imports[{index}].module"));
            }
            check_digest(
                import,
                &["artifact_sha256"],
                &format!("environment.imports[{index}].artifact_sha256"),
                errors,
            );
        }
    } else {
        push_missing(errors, "environment.imports");
    }
}

fn check_logical_closure(value: &Value, errors: &mut Vec<String>) {
    let closure_status = string_at(value, &["logical_dependency_audit", "status"]);
    let closure_axis = is_satisfied(value, "logical_dependency_closure");

    let algorithm = &["logical_dependency_audit", "algorithm"];
    if bool_at(value, &[algorithm[0], algorithm[1], "fresh_per_root"]) != Some(true) {
        errors.push("logical audit must use fresh_per_root=true".to_owned());
    }
    if bool_at(value, &[algorithm[0], algorithm[1], "visited_set_only"]) != Some(true) {
        errors.push("logical audit must use visited_set_only=true".to_owned());
    }
    if bool_at(value, &[algorithm[0], algorithm[1], "uses_cached_axiom_summary"]) != Some(false) {
        errors.push("logical audit must not use cached axiom summaries as its closure authority".to_owned());
    }

    let unresolved = array_at(value, &["logical_dependency_audit", "unresolved"]);
    let unresolved_nonempty = unresolved.is_none_or(|items| !items.is_empty());
    let admissions = array_at(value, &["logical_dependency_audit", "admissions"]);
    let admissions_nonempty = admissions.is_none_or(|items| !items.is_empty());
    let roots = array_at(value, &["logical_dependency_audit", "roots"]);
    let roots_complete = roots.is_some_and(|items| {
        !items.is_empty()
            && items
                .iter()
                .all(|root| string_at(root, &["status"]) == Some("complete"))
    });

    if closure_axis {
        if closure_status != Some("complete_within_coverage") {
            errors.push("logical closure cannot be satisfied unless audit status is complete_within_coverage".to_owned());
        }
        if unresolved_nonempty {
            errors.push("logical closure cannot be satisfied with unresolved or missing dependencies".to_owned());
        }
        if admissions_nonempty {
            errors.push("logical closure cannot be satisfied when admissions are present".to_owned());
        }
        if !roots_complete {
            errors.push("logical closure requires every declared root to be complete".to_owned());
        }
        if string_at(value, &["logical_dependency_audit", "root_order_permutation", "result"])
            != Some("identical")
            || bool_at(value, &["logical_dependency_audit", "root_order_permutation", "tested"])
                != Some(true)
        {
            errors.push("logical closure requires a tested, identical root-order permutation result".to_owned());
        }

        if let Some(nodes) = array_at(value, &["logical_dependency_audit", "nodes"]) {
            for (index, node) in nodes.iter().enumerate() {
                if string_at(node, &["kind"]) == Some("unknown")
                    || string_at(node, &["value_availability"]) == Some("unavailable")
                {
                    errors.push(format!("logical closure contains unknown or unavailable node at index {index}"));
                }
            }
        } else {
            push_missing(errors, "logical_dependency_audit.nodes");
        }

        if let Some(axioms) = array_at(value, &["logical_dependency_audit", "axioms"]) {
            let allowed_foundational: HashSet<&str> = array_at(value, &["policy", "allowed_foundational_axioms"])
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .collect();
            let allowed_custom: HashSet<&str> = array_at(value, &["policy", "allowed_custom_axioms"])
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .collect();
            for axiom in axioms {
                let name = string_at(axiom, &["name"]).unwrap_or("");
                let class = string_at(axiom, &["classification"]);
                let disposition = string_at(axiom, &["policy_disposition"]);
                match class {
                    Some("foundational") if !allowed_foundational.contains(name) => {
                        errors.push(format!("foundational axiom {name} is absent from the explicit policy allowlist"));
                    }
                    Some("custom") if !allowed_custom.contains(name) => {
                        errors.push(format!("custom axiom {name} is absent from the explicit policy allowlist"));
                    }
                    Some("unknown") | None => {
                        errors.push(format!("axiom {name} has unknown classification"));
                    }
                    _ => {}
                }
                if disposition != Some("allow") {
                    errors.push(format!("axiom {name} is not explicitly allowed by policy"));
                }
            }
        } else {
            push_missing(errors, "logical_dependency_audit.axioms");
        }
    }

    if !closure_axis && closure_status == Some("complete_within_coverage") && unresolved_nonempty {
        errors.push("audit claims complete closure while unresolved dependencies remain".to_owned());
    }
}

fn check_statement_fidelity(value: &Value, errors: &mut Vec<String>) {
    if !is_satisfied(value, "statement_fidelity") {
        return;
    }
    let review = string_at(value, &["subject", "challenge", "review_status"]);
    if !matches!(review, Some("trusted" | "reviewed")) {
        errors.push("statement fidelity requires a trusted or reviewed challenge".to_owned());
    }

    let challenge_digest = string_at(value, &["subject", "challenge", "target_identity_sha256"]);
    let root_digest = string_at(value, &["subject", "root_declaration", "type_sha256"]);
    let comparison = string_at(value, &["replays", "statement_comparison", "result"]);
    let challenge_replay = string_at(value, &["replays", "statement_comparison", "challenge_target_sha256"]);
    let solution_replay = string_at(value, &["replays", "statement_comparison", "solution_target_sha256"]);

    if comparison != Some("match")
        || challenge_digest.is_none()
        || challenge_digest != root_digest
        || challenge_digest != challenge_replay
        || challenge_digest != solution_replay
    {
        errors.push("statement fidelity requires exact challenge/root/comparator target identity".to_owned());
    }
}

fn check_replays(value: &Value, errors: &mut Vec<String>) {
    let export = string_at(value, &["replays", "export_sha256"]);
    if let Some(export) = export {
        if !is_hex(export, 64) {
            push_missing(errors, "replays.export_sha256");
        }
    } else {
        push_missing(errors, "replays.export_sha256");
    }

    if is_satisfied(value, "official_kernel_replay") {
        let result = string_at(value, &["replays", "official_kernel", "result"]);
        let replay_export = string_at(value, &["replays", "official_kernel", "export_sha256"]);
        if string_at(value, &["replays", "official_kernel", "status"]) != Some("recorded")
            || result != Some("accepted")
            || replay_export != export
        {
            errors.push("official kernel satisfaction requires acceptance of the exact capsule export".to_owned());
        }
        check_digest(
            value,
            &["replays", "official_kernel", "raw_log_sha256"],
            "replays.official_kernel.raw_log_sha256",
            errors,
        );
    }

    if is_satisfied(value, "independent_checker_replay") {
        let official_name = string_at(value, &["replays", "official_kernel", "checker"]);
        let checkers = array_at(value, &["replays", "independent_checkers"]);
        let has_independent_acceptance = checkers.is_some_and(|items| {
            items.iter().any(|checker| {
                string_at(checker, &["result"]) == Some("accepted")
                    && bool_at(checker, &["independent_from_official_kernel"]) == Some(true)
                    && string_at(checker, &["export_sha256"]) == export
                    && string_at(checker, &["checker"]).is_some_and(|name| Some(name) != official_name)
                    && string_at(checker, &["implementation_revision"]).is_some()
            })
        });
        if !has_independent_acceptance {
            errors.push("independent checker satisfaction requires a distinct implementation accepting the exact same export".to_owned());
        }
    }
}

fn check_construction(value: &Value, errors: &mut Vec<String>) {
    if !is_satisfied(value, "construction_containment") {
        return;
    }
    if string_at(value, &["construction_provenance", "status"]) != Some("recorded")
        || string_at(value, &["construction_provenance", "sandbox", "status"]) != Some("pass")
    {
        errors.push("construction containment requires recorded provenance and a passing sandbox".to_owned());
    }
    check_digest(
        value,
        &["construction_provenance", "sandbox", "policy_sha256"],
        "construction_provenance.sandbox.policy_sha256",
        errors,
    );

    if let Some(components) = array_at(value, &["construction_provenance", "components"]) {
        for (index, component) in components.iter().enumerate() {
            let stage = string_at(component, &["execution_stage"]);
            if matches!(stage, Some("elaboration" | "build" | "export"))
                && bool_at(component, &["sandboxed"]) != Some(true)
            {
                errors.push(format!("proof-construction component {index} was not recorded as sandboxed"));
            }
        }
    } else {
        push_missing(errors, "construction_provenance.components");
    }
}

fn check_computational_trust(value: &Value, errors: &mut Vec<String>) {
    if !is_satisfied(value, "computational_trust") {
        return;
    }
    if string_at(value, &["computational_trust", "status"]) != Some("classified") {
        errors.push("computational trust cannot be satisfied unless observations are classified".to_owned());
    }
    let observations = array_at(value, &["computational_trust", "observations"]);
    let Some(observations) = observations else {
        push_missing(errors, "computational_trust.observations");
        return;
    };
    if observations.is_empty() {
        errors.push("computational trust requires explicit observations, including none_observed".to_owned());
    }
    let policy = string_at(value, &["policy", "native_computation_policy"]);
    for (index, observation) in observations.iter().enumerate() {
        let kind = string_at(observation, &["kind"]);
        let disposition = string_at(observation, &["policy_disposition"]);
        if kind == Some("unknown") || disposition != Some("allow") {
            errors.push(format!("computational trust observation {index} is unknown or not explicitly allowed"));
        }
        if policy == Some("reject")
            && !matches!(kind, Some("none_observed"))
        {
            errors.push(format!("native computation observation {index} conflicts with reject policy"));
        }
        if kind == Some("none_observed") && policy == Some("not_assessed") {
            errors.push("none_observed is not an adequate classification when native computation policy is not_assessed".to_owned());
        }
    }
    if policy == Some("not_assessed") {
        errors.push("computational trust cannot be satisfied under a not_assessed policy".to_owned());
    }
}

pub fn validate_capsule(value: &Value) -> Vec<String> {
    let mut errors = Vec::new();

    if string_at(value, &["schema_version"]) != Some("1.0.0") {
        errors.push("schema_version must be 1.0.0".to_owned());
    }
    if string_at(value, &["artifact_id"]) != Some("LEAN-TRUST-CLOSURE-ASSURANCE-CAPSULE-001") {
        errors.push("artifact_id does not identify LeanTrustClosureCapsuleV1".to_owned());
    }
    if bool_at(value, &["policy", "reject_admissions"]) != Some(true) {
        errors.push("policy.reject_admissions must be true for v1".to_owned());
    }
    if string_at(value, &["policy", "unresolved_policy"])
        != Some("unknown_or_rejected_never_empty")
    {
        errors.push("unresolved_policy must fail closed".to_owned());
    }

    check_subject_and_environment(value, &mut errors);
    check_logical_closure(value, &mut errors);
    check_statement_fidelity(value, &mut errors);
    check_replays(value, &mut errors);
    check_construction(value, &mut errors);
    check_computational_trust(value, &mut errors);

    for axis in AXES {
        if axis_status(value, axis).is_none() {
            errors.push(format!("assurance_axes.{axis}.status is missing or invalid"));
        }
        if string_at(value, &["assurance_axes", axis, "rationale"])
            .is_none_or(|rationale| rationale.trim().is_empty())
        {
            errors.push(format!("assurance_axes.{axis}.rationale must be nonempty"));
        }
    }

    match string_at(value, &["disposition"]) {
        Some("qualified_under_policy") => {
            for axis in AXES {
                if !matches!(axis_status(value, axis), Some("satisfied" | "not_applicable")) {
                    errors.push(format!("qualified_under_policy is forbidden while {axis} is not satisfied"));
                }
                if axis_status(value, axis) == Some("not_applicable")
                    && string_at(value, &["assurance_axes", axis, "rationale"])
                        .is_none_or(|reason| reason.trim().is_empty())
                {
                    errors.push(format!("not_applicable axis {axis} requires an explicit rationale"));
                }
            }
        }
        Some("rejected") => {
            if !AXES.iter().any(|axis| axis_status(value, axis) == Some("failed")) {
                errors.push("rejected disposition requires at least one failed assurance axis".to_owned());
            }
        }
        Some("inconclusive") => {
            if !AXES.iter().any(|axis| axis_status(value, axis) == Some("unknown")) {
                errors.push("inconclusive disposition requires at least one unknown assurance axis".to_owned());
            }
        }
        Some("draft") => {}
        _ => errors.push("disposition is missing or invalid".to_owned()),
    }

    errors
}

pub fn validate_json(input: &str) -> Result<Value, Vec<String>> {
    let value: Value = serde_json::from_str(input)
        .map_err(|error| vec![format!("invalid trust-closure capsule JSON: {error}")])?;
    let errors = validate_capsule(&value);
    if errors.is_empty() {
        Ok(value)
    } else {
        Err(errors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn baseline() -> Value {
        let digest = "a".repeat(64);
        let commit = "b".repeat(40);
        json!({
            "schema_version":"1.0.0",
            "artifact_id":"LEAN-TRUST-CLOSURE-ASSURANCE-CAPSULE-001",
            "disposition":"qualified_under_policy",
            "policy":{
                "policy_id":"TEST-POLICY",
                "version":"1",
                "allowed_foundational_axioms":[],
                "allowed_custom_axioms":[],
                "reject_admissions":true,
                "unresolved_policy":"unknown_or_rejected_never_empty",
                "native_computation_policy":"reject"
            },
            "subject":{
                "repository":"example/repo",
                "commit":commit,
                "source":{"path":"Example.lean","sha256":digest},
                "root_declaration":{
                    "fully_qualified_name":"Example.target",
                    "declaration_kind":"theorem",
                    "type_sha256":digest,
                    "value_availability":"available"
                },
                "challenge":{
                    "artifact":{"path":"Challenge.lean","sha256":digest},
                    "target_identity_sha256":digest,
                    "serialization_version":"lean-expr-v1",
                    "review_status":"trusted"
                }
            },
            "environment":{
                "lean_toolchain":"leanprover/lean4:v4.37.0-rc1",
                "lean_core_revision":commit,
                "lake_manifest_sha256":digest,
                "imports":[{"module":"Init","artifact_sha256":digest}]
            },
            "logical_dependency_audit":{
                "status":"complete_within_coverage",
                "algorithm":{
                    "name":"fresh-root-dfs",
                    "version":"1",
                    "fresh_per_root":true,
                    "visited_set_only":true,
                    "uses_cached_axiom_summary":false,
                    "imported_values_visible":true
                },
                "roots":[{"name":"Example.target","closure_sha256":digest,"status":"complete"}],
                "nodes":[{"name":"Example.target","kind":"theorem","value_availability":"available","edges":[]}],
                "axioms":[],
                "admissions":[],
                "unresolved":[],
                "coverage":{"lean_versions_tested":["4.37.0-rc1"],"declaration_kinds_tested":["theorem"],"limitations":[]},
                "closure_sha256":digest,
                "root_order_permutation":{"tested":true,"result":"identical"}
            },
            "computational_trust":{
                "status":"classified",
                "observations":[{"kind":"none_observed","marker":"none","lean_version":"4.37.0-rc1","policy_disposition":"allow"}]
            },
            "construction_provenance":{
                "status":"recorded",
                "sandbox":{"status":"pass","profile":"restricted-build","policy_sha256":digest},
                "components":[{"kind":"tactic","name":"intro","execution_stage":"elaboration","sandboxed":true}]
            },
            "replays":{
                "export_sha256":digest,
                "official_kernel":{
                    "status":"recorded","checker":"lean-kernel","version":"4.37.0-rc1","result":"accepted",
                    "export_sha256":digest,"raw_log_sha256":digest,"exit_code":0
                },
                "independent_checkers":[{
                    "checker":"nanoda","implementation_revision":"0da9f98",
                    "independent_from_official_kernel":true,"result":"accepted",
                    "export_sha256":digest,"axiom_policy_id":"TEST-POLICY","raw_log_sha256":digest
                }],
                "statement_comparison":{
                    "status":"recorded","challenge_target_sha256":digest,
                    "solution_target_sha256":digest,"result":"match",
                    "comparator_version":"test","raw_log_sha256":digest
                }
            },
            "assurance_axes":{
                "statement_fidelity":{"status":"satisfied","rationale":"trusted challenge matched","evidence_digests":[digest]},
                "logical_dependency_closure":{"status":"satisfied","rationale":"fresh closure complete","evidence_digests":[digest]},
                "official_kernel_replay":{"status":"satisfied","rationale":"kernel accepted exact export","evidence_digests":[digest]},
                "independent_checker_replay":{"status":"satisfied","rationale":"distinct checker accepted exact export","evidence_digests":[digest]},
                "construction_containment":{"status":"satisfied","rationale":"build isolated","evidence_digests":[digest]},
                "environment_reproducibility":{"status":"satisfied","rationale":"inputs pinned","evidence_digests":[digest]},
                "computational_trust":{"status":"satisfied","rationale":"no native computation observed","evidence_digests":[digest]}
            },
            "claim_ceiling":["test only"]
        })
    }

    #[test]
    fn accepts_consistent_capsule() {
        let capsule = baseline();
        assert!(validate_capsule(&capsule).is_empty());
    }

    #[test]
    fn rejects_qualification_with_unknown_axis() {
        let mut capsule = baseline();
        capsule["assurance_axes"]["independent_checker_replay"]["status"] = json!("unknown");
        let errors = validate_capsule(&capsule);
        assert!(errors.iter().any(|error| error.contains("qualified_under_policy")));
    }

    #[test]
    fn unresolved_dependency_cannot_be_called_closed() {
        let mut capsule = baseline();
        capsule["logical_dependency_audit"]["unresolved"] = json!([{"name":"Missing.X","reason":"not found"}]);
        let errors = validate_capsule(&capsule);
        assert!(errors.iter().any(|error| error.contains("unresolved")));
    }

    #[test]
    fn same_kernel_twice_is_not_independent_replay() {
        let mut capsule = baseline();
        capsule["replays"]["independent_checkers"][0]["checker"] = json!("lean-kernel");
        capsule["replays"]["independent_checkers"][0]["independent_from_official_kernel"] = json!(false);
        let errors = validate_capsule(&capsule);
        assert!(errors.iter().any(|error| error.contains("distinct implementation")));
    }

    #[test]
    fn independent_checker_must_check_identical_export() {
        let mut capsule = baseline();
        capsule["replays"]["independent_checkers"][0]["export_sha256"] = json!("c".repeat(64));
        let errors = validate_capsule(&capsule);
        assert!(errors.iter().any(|error| error.contains("distinct implementation")));
    }

    #[test]
    fn target_substitution_blocks_statement_fidelity() {
        let mut capsule = baseline();
        capsule["replays"]["statement_comparison"]["solution_target_sha256"] = json!("c".repeat(64));
        let errors = validate_capsule(&capsule);
        assert!(errors.iter().any(|error| error.contains("statement fidelity")));
    }

    #[test]
    fn native_computation_rejected_by_reject_policy() {
        let mut capsule = baseline();
        capsule["computational_trust"]["observations"][0] = json!({
            "kind":"native_decide","marker":"decide_native_1","lean_version":"4.37.0-rc1",
            "policy_disposition":"reject"
        });
        let errors = validate_capsule(&capsule);
        assert!(errors.iter().any(|error| error.contains("not explicitly allowed")
            || error.contains("reject policy")));
    }

    #[test]
    fn missing_import_digest_is_rejected() {
        let mut capsule = baseline();
        capsule["environment"]["imports"][0]["artifact_sha256"] = json!("bad");
        let errors = validate_capsule(&capsule);
        assert!(errors.iter().any(|error| error.contains("artifact_sha256")));
    }

    #[test]
    fn invalid_json_is_rejected() {
        assert!(validate_json("{not-json").is_err());
    }
}
