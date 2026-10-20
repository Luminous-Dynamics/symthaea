use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::path::Path;

const TRANSFER_STATUSES: &[&str] = &[
    "untriaged",
    "candidate",
    "preconditions_satisfied",
    "counterexample_found",
    "verified_transfer",
    "rejected",
    "not_attempted",
];
const COMPATIBILITY: &[&str] = &["compatible", "incompatible", "partial", "unknown"];
const MISMATCH_CODES: &[&str] = &[
    "object_type",
    "premise_unavailable",
    "inequality_direction",
    "quantifier_scope",
    "precondition_mismatch",
    "domain_mismatch",
    "proof_dependency_mismatch",
    "predicate_mismatch",
    "unknown",
];
const AXIOM_POLICIES: &[&str] = &[
    "not_applicable",
    "declared_classical",
    "constructive_policy",
    "custom",
    "unknown",
];
const MOVE_KINDS: &[&str] = &[
    "inequality_transfer",
    "representation_reduction",
    "witness_construction",
    "induction",
    "normalization",
    "decomposition",
    "duality",
    "invariant_composition",
    "other",
];
const RELATIONS: &[&str] = &[
    "upper_bound",
    "strict_upper_bound",
    "lower_bound",
    "strict_lower_bound",
    "equality",
    "inequality",
    "existence",
    "nonexistence",
    "implication",
    "equivalence",
    "subset",
    "superset",
    "counterexample_existence",
    "other",
];

#[derive(Debug)]
struct Obligation<'a> {
    domain: &'a str,
    object_type: &'a str,
    predicate: &'a str,
    relation: &'a str,
}

fn parse_obligation<'a>(contract: &'a Value, field: &str) -> Result<Obligation<'a>> {
    let value = contract
        .get(field)
        .with_context(|| format!("directional_contract missing {field}"))?;
    let domain = required_string(value, "domain")?;
    let object_type = required_string(value, "object_type")?;
    let predicate = required_string(value, "predicate")?;
    let relation = required_string(value, "relation")?;
    if !RELATIONS.contains(&relation) {
        bail!("{field}.relation has unsupported value {relation:?}");
    }
    if domain.trim().is_empty()
        || object_type.trim().is_empty()
        || predicate.trim().is_empty()
        || required_string(value, "quantifier_scope")?.trim().is_empty()
    {
        bail!("{field} domain/object_type/predicate/quantifier_scope must be non-empty");
    }
    let preconditions = value
        .get("preconditions")
        .and_then(Value::as_array)
        .with_context(|| format!("{field}.preconditions must be an array"))?;
    if preconditions.iter().any(|item| {
        item.as_str()
            .is_none_or(|text| text.trim().is_empty())
    }) {
        bail!("{field}.preconditions must contain only non-empty strings");
    }
    Ok(Obligation {
        domain,
        object_type,
        predicate,
        relation,
    })
}

fn opposite_bound_direction(source: &str, target: &str) -> bool {
    matches!(
        (source, target),
        ("upper_bound", "lower_bound")
            | ("lower_bound", "upper_bound")
            | ("strict_upper_bound", "strict_lower_bound")
            | ("strict_lower_bound", "strict_upper_bound")
    )
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        bail!("usage: {} <insight-record.json>", args.first().map(String::as_str).unwrap_or("validate_transfer_contract"));
    }

    let path = Path::new(&args[1]);
    let text = fs::read_to_string(path)
        .with_context(|| format!("cannot read insight record {}", path.display()))?;
    let record: Value = serde_json::from_str(&text)
        .with_context(|| format!("insight record {} is not valid JSON", path.display()))?;
    let candidate_count = validate_record(&record)?;
    println!(
        "PASS: {} transfer candidate contract(s) are internally consistent in {}",
        candidate_count,
        path.display()
    );
    Ok(())
}

fn validate_record(record: &Value) -> Result<usize> {
    let candidates = record
        .pointer("/insight/transfer_candidates")
        .and_then(Value::as_array)
        .context("insight.transfer_candidates must be an array")?;

    for (index, candidate) in candidates.iter().enumerate() {
        validate_candidate(candidate)
            .with_context(|| format!("transfer_candidates[{index}] is invalid"))?;
    }
    Ok(candidates.len())
}

fn validate_candidate(candidate: &Value) -> Result<()> {
    let status = required_string(candidate, "status")?;
    if !TRANSFER_STATUSES.contains(&status) {
        bail!("unsupported transfer status {status:?}");
    }

    let contract = candidate
        .get("directional_contract")
        .and_then(Value::as_object)
        .context("directional_contract must be an object")?;
    let contract_value = Value::Object(contract.clone());
    let compatibility = contract
        .get("compatibility")
        .and_then(Value::as_str)
        .context("directional_contract.compatibility must be a string")?;
    if !COMPATIBILITY.contains(&compatibility) {
        bail!("unsupported compatibility classification {compatibility:?}");
    }

    let source_requirement = contract
        .get("source_move_requires")
        .and_then(Value::as_str)
        .context("directional_contract.source_move_requires must be a string")?;
    let target_requirement = contract
        .get("target_goal_requires")
        .and_then(Value::as_str)
        .context("directional_contract.target_goal_requires must be a string")?;
    if source_requirement.trim().is_empty() || target_requirement.trim().is_empty() {
        bail!("source move and target goal requirements must be non-empty");
    }

    let move_kind = required_string(&contract_value, "move_kind")?;
    if !MOVE_KINDS.contains(&move_kind) {
        bail!("unsupported proof move kind {move_kind:?}");
    }
    let source_obligation = parse_obligation(&contract_value, "source_obligation")?;
    let target_obligation = parse_obligation(&contract_value, "target_obligation")?;

    let mismatch_values = contract
        .get("mismatch_codes")
        .and_then(Value::as_array)
        .context("directional_contract.mismatch_codes must be an array")?;
    let mut mismatch_codes = BTreeSet::new();
    for value in mismatch_values {
        let code = value
            .as_str()
            .context("mismatch code must be a string")?;
        if !MISMATCH_CODES.contains(&code) {
            bail!("unsupported mismatch code {code:?}");
        }
        if !mismatch_codes.insert(code) {
            bail!("duplicate mismatch code {code:?}");
        }
    }

    match compatibility {
        "compatible" if !mismatch_codes.is_empty() => {
            bail!("compatible contract must not declare mismatch codes");
        }
        "incompatible" | "partial"
            if mismatch_codes.is_empty()
                || mismatch_codes.iter().all(|code| *code == "unknown") =>
        {
            bail!("{compatibility} contract needs at least one specific mismatch code");
        }
        "unknown" if !mismatch_codes.contains("unknown") => {
            bail!("unknown compatibility must include the unknown mismatch code");
        }
        _ => {}
    }

    if move_kind == "representation_reduction" {
        if source_obligation.domain != target_obligation.domain {
            if compatibility == "compatible" || !mismatch_codes.contains("domain_mismatch") {
                bail!("representation reduction across domains requires a domain mismatch disposition");
            }
        }
        if source_obligation.object_type != target_obligation.object_type {
            if compatibility == "compatible" || !mismatch_codes.contains("object_type") {
                bail!("representation reduction across object types requires an object_type mismatch");
            }
        }
    }
    if move_kind == "inequality_transfer" {
        if source_obligation.predicate != target_obligation.predicate {
            if compatibility == "compatible" || !mismatch_codes.contains("predicate_mismatch") {
                bail!("inequality transfer across predicates requires a predicate_mismatch disposition");
            }
        } else if opposite_bound_direction(source_obligation.relation, target_obligation.relation)
            && (compatibility == "compatible" || !mismatch_codes.contains("inequality_direction"))
        {
            bail!("opposite source/target inequality directions require an inequality_direction mismatch");
        }
    }

    if status == "preconditions_satisfied" && compatibility != "compatible" {
        bail!("preconditions_satisfied requires a compatible directional contract");
    }
    if status == "rejected" && !matches!(compatibility, "incompatible" | "partial") {
        bail!("rejected transfer requires incompatible or partial compatibility");
    }

    let transfer_verification = candidate.get("transfer_verification");
    if let Some(evidence) = transfer_verification {
        let target_status = required_string(evidence, "status")?;
        if target_status == "passed" && status != "verified_transfer" {
            bail!("passed target evidence cannot appear on a non-verified transfer");
        }
    }

    if status == "verified_transfer" {
        if compatibility != "compatible" || !mismatch_codes.is_empty() {
            bail!("verified_transfer requires a compatible contract with no mismatch codes");
        }
        let target_id = candidate
            .get("target_theorem_id")
            .and_then(Value::as_str)
            .context("verified_transfer requires target_theorem_id")?;
        if target_id.trim().is_empty() {
            bail!("verified_transfer target_theorem_id must be non-empty");
        }
        validate_target_verification(
            transfer_verification.context("verified_transfer requires target_verification evidence")?,
        )?;
    }

    Ok(())
}

fn validate_target_verification(evidence: &Value) -> Result<()> {
    if required_string(evidence, "status")? != "passed" {
        bail!("verified transfer requires target verification status passed");
    }
    if evidence.get("exact_statement_checked").and_then(Value::as_bool) != Some(true) {
        bail!("target verification must check the exact statement");
    }

    require_lower_hex(evidence, "target_source_commit", 40)?;
    require_lower_hex(evidence, "target_artifact_sha256", 64)?;
    require_lower_hex(evidence, "target_statement_digest", 64)?;
    require_lower_hex(evidence, "receipt_sha256", 64)?;
    for field in ["target_artifact_path", "checker", "checker_version"] {
        if required_string(evidence, field)?.trim().is_empty() {
            bail!("target verification {field} must be non-empty");
        }
    }

    let axiom_policy = required_string(evidence, "axiom_policy")?;
    if !AXIOM_POLICIES.contains(&axiom_policy) {
        bail!("unsupported target axiom policy {axiom_policy:?}");
    }

    let correspondence = evidence
        .get("target_statement_correspondence")
        .context("target verification missing statement correspondence")?;
    if required_string(correspondence, "status")? != "matches_intended_claim" {
        bail!("verified transfer requires a reviewed target statement matching the intended claim");
    }
    let reviewer_count = correspondence
        .get("independent_reviewer_count")
        .and_then(Value::as_u64)
        .context("target statement correspondence needs independent_reviewer_count")?;
    if reviewer_count == 0 {
        bail!("verified transfer needs at least one independent statement-correspondence reviewer");
    }

    let successor_status = required_string(evidence, "successor_test_status")?;
    if !matches!(successor_status, "passed" | "not_applicable") {
        bail!("verified transfer needs passed or explicitly not_applicable successor testing");
    }
    let successor_tests = evidence
        .get("successor_tests")
        .and_then(Value::as_array)
        .context("target verification successor_tests must be an array")?;
    if successor_status == "passed"
        && !successor_tests.iter().any(|test| {
            test.get("outcome").and_then(Value::as_str) == Some("passed")
                && test.get("depends_on_candidate").and_then(Value::as_bool) == Some(true)
        })
    {
        bail!("passed successor status requires a passed test that depends on the candidate");
    }
    if successor_status == "not_applicable" {
        let notes = evidence
            .get("notes")
            .and_then(Value::as_array)
            .context("not_applicable successor status requires notes")?;
        if !notes.iter().any(|note| {
            note.as_str()
                .is_some_and(|text| text.to_ascii_lowercase().contains("no applicable successor"))
        }) {
            bail!("not_applicable successor status must explain why no successor test applies");
        }
    }

    if required_string(evidence, "mutation_control_status")? != "passed" {
        bail!("verified transfer requires passed mutation controls");
    }
    let mutation_controls = evidence
        .get("mutation_controls")
        .and_then(Value::as_array)
        .context("target verification mutation_controls must be an array")?;
    if mutation_controls.is_empty() {
        bail!("verified transfer requires at least one mutation-control receipt");
    }
    let mut mutation_ids = BTreeSet::new();
    for mutation in mutation_controls {
        let mutation_id = required_string(mutation, "mutation_id")?;
        if mutation_id.trim().is_empty() || !mutation_ids.insert(mutation_id) {
            bail!("mutation-control IDs must be non-empty and unique");
        }
        if required_string(mutation, "status")? != "detected_as_expected"
            || mutation.get("expected_detection").and_then(Value::as_bool) != Some(true)
            || mutation.get("observed_detection").and_then(Value::as_bool) != Some(true)
        {
            bail!("verified transfer requires every mutation control to be detected as expected");
        }
    }
    Ok(())
}

fn required_string<'a>(value: &'a Value, field: &str) -> Result<&'a str> {
    value
        .get(field)
        .and_then(Value::as_str)
        .with_context(|| format!("{field} must be a string"))
}

fn require_lower_hex(value: &Value, field: &str, length: usize) -> Result<()> {
    let text = required_string(value, field)?;
    if text.len() != length
        || !text
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        bail!("{field} must be {length} lowercase hexadecimal characters");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{validate_candidate, validate_record};
    use serde_json::{Value, json};

    fn compatible_candidate() -> Value {
        json!({
            "target_family_id": "target-001",
            "target_theorem_id": null,
            "rationale": "Synthetic schema and contract fixture only",
            "status": "candidate",
            "directional_contract": {
                "compatibility": "compatible",
                "mismatch_codes": [],
                "move_kind": "other",
                "source_obligation": {
                    "domain": "synthetic-only",
                    "object_type": "synthetic-object",
                    "predicate": "synthetic-source",
                    "relation": "equality",
                    "quantifier_scope": "synthetic fixture",
                    "preconditions": []
                },
                "target_obligation": {
                    "domain": "synthetic-only",
                    "object_type": "synthetic-object",
                    "predicate": "synthetic-target",
                    "relation": "equality",
                    "quantifier_scope": "synthetic fixture",
                    "preconditions": []
                },
                "source_move_requires": "A synthetic precondition",
                "target_goal_requires": "A synthetic goal",
                "notes": ["No mathematical transfer is asserted"]
            }
        })
    }

    fn record_with(candidate: Value) -> Value {
        json!({"insight": {"transfer_candidates": [candidate]}})
    }

    #[test]
    fn accepts_candidate_with_consistent_compatible_contract() {
        let candidate = compatible_candidate();
        assert!(validate_candidate(&candidate).is_ok());
    }

    #[test]
    fn rejects_missing_directional_contract() {
        let mut candidate = compatible_candidate();
        candidate.as_object_mut().unwrap().remove("directional_contract");
        assert!(validate_candidate(&candidate).is_err());
    }

    #[test]
    fn rejects_compatible_contract_with_mismatch_codes() {
        let mut candidate = compatible_candidate();
        candidate["directional_contract"]["mismatch_codes"] = json!(["inequality_direction"]);
        assert!(validate_candidate(&candidate).is_err());
    }

    #[test]
    fn rejects_rejected_status_without_incompatible_contract() {
        let mut candidate = compatible_candidate();
        candidate["status"] = json!("rejected");
        assert!(validate_candidate(&candidate).is_err());
    }

    #[test]
    fn rejects_incompatible_contract_labeled_verified() {
        let mut candidate = compatible_candidate();
        candidate["status"] = json!("verified_transfer");
        candidate["target_theorem_id"] = json!("OAI.Synthetic.target");
        candidate["directional_contract"]["compatibility"] = json!("incompatible");
        candidate["directional_contract"]["mismatch_codes"] = json!(["inequality_direction"]);
        assert!(validate_candidate(&candidate).is_err());
    }

    #[test]
    fn rejects_unknown_compatibility_without_unknown_mismatch() {
        let mut candidate = compatible_candidate();
        candidate["directional_contract"]["compatibility"] = json!("unknown");
        assert!(validate_candidate(&candidate).is_err());
    }

    #[test]
    fn rejects_opposite_inequality_direction_when_marked_compatible() {
        let mut candidate = compatible_candidate();
        let contract = &mut candidate["directional_contract"];
        contract["move_kind"] = json!("inequality_transfer");
        contract["source_obligation"]["predicate"] = json!("independence_number");
        contract["source_obligation"]["relation"] = json!("upper_bound");
        contract["target_obligation"]["predicate"] = json!("independence_number");
        contract["target_obligation"]["relation"] = json!("lower_bound");
        assert!(validate_candidate(&candidate).is_err());

        contract["compatibility"] = json!("incompatible");
        contract["mismatch_codes"] = json!(["inequality_direction"]);
        candidate["status"] = json!("rejected");
        assert!(validate_candidate(&candidate).is_ok());
    }

    #[test]
    fn rejects_representation_reduction_across_domains_if_marked_compatible() {
        let mut candidate = compatible_candidate();
        let contract = &mut candidate["directional_contract"];
        contract["move_kind"] = json!("representation_reduction");
        contract["source_obligation"]["domain"] = json!("geometry");
        contract["target_obligation"]["domain"] = json!("graph theory");
        assert!(validate_candidate(&candidate).is_err());
    }

    #[test]
    fn verified_transfer_requires_target_side_receipt() {
        let mut candidate = compatible_candidate();
        candidate["status"] = json!("verified_transfer");
        candidate["target_theorem_id"] = json!("OAI.Synthetic.target");
        assert!(validate_candidate(&candidate).is_err());
    }

    #[test]
    fn accepts_verified_transfer_with_target_side_evidence() {
        let mut candidate = compatible_candidate();
        candidate["status"] = json!("verified_transfer");
        candidate["target_theorem_id"] = json!("OAI.Synthetic.target");
        candidate["transfer_verification"] = json!({
            "status": "passed",
            "target_source_commit": "a".repeat(40),
            "target_artifact_path": "lean/Synthetic/Target.lean",
            "target_artifact_sha256": "b".repeat(64),
            "target_statement_digest": "c".repeat(64),
            "checker": "Lean 4 kernel",
            "checker_version": "4.34.1",
            "exact_statement_checked": true,
            "axiom_policy": "declared_classical",
            "receipt_sha256": "d".repeat(64),
            "target_statement_correspondence": {
                "status": "matches_intended_claim",
                "independent_reviewer_count": 1
            },
            "successor_test_status": "passed",
            "successor_tests": [{
                "test_id": "synthetic-dependent-successor",
                "source_path": "lean/Synthetic/Target.lean",
                "test_declaration": "Synthetic.successor",
                "outcome": "passed",
                "depends_on_candidate": true
            }],
            "mutation_control_status": "passed",
            "mutation_controls": [{
                "mutation_id": "synthetic-mutated-target",
                "description": "Synthetic target statement mutation",
                "expected_detection": true,
                "observed_detection": true,
                "status": "detected_as_expected"
            }],
            "notes": ["Synthetic fixture only"]
        });
        assert!(validate_candidate(&candidate).is_ok());
    }

    #[test]
    fn record_requires_transfer_candidate_array() {
        assert!(validate_record(&json!({"insight": {}})).is_err());
    }
}
