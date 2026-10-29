// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Structural validator and runtime projection for paired counterfactual memory fixtures.
//!
//! This module validates the fixture contract only. It does not implement semantic
//! entailment or decide whether an answer is supported. Evaluator-only labels live in a
//! separate fixture file and are deliberately not accepted by the runtime projection.

use chrono::DateTime;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeSet;

pub const COUNTERFACTUAL_FIXTURE_SCHEMA: &str = "BROCA-COUNTERFACTUAL-EVIDENCE-AUDIT-V1";
pub const COUNTERFACTUAL_LABEL_SCHEMA: &str = "BROCA-COUNTERFACTUAL-EVIDENCE-LABELS-V1";

const ALLOWED_FAMILIES: &[&str] = &[
    "polarity_negation",
    "temporal_validity",
    "entity_binding",
    "scope_premise",
    "source_dependence",
    "missing_requirement",
    "ordered_transition",
    "correction_update",
    "qualifier_context",
    "memory_influence",
];

const ALLOWED_DECISION_RELATIONS: &[&str] = &[
    "claim_must_flip_or_be_rejected",
    "must_use_current_evidence_only",
    "must_not_cross_scope",
    "must_not_treat_duplicate_source_as_independent_support",
    "must_abstain_or_refine_if_requirement_missing",
    "must_follow_observed_transition",
    "must_recompute_support_with_context",
    "must_use_latest_applicable_correction",
    "irrelevant_memory_should_not_change_answer",
];

const FORBIDDEN_RUNTIME_KEYS: &[&str] = &[
    "answer",
    "goldanswer",
    "referenceanswer",
    "correctanswer",
    "answerkey",
    "groundtruth",
    "evaluator",
    "evaluationfunction",
    "answerability",
    "expectedclaim",
    "supportingevidenceids",
    "expecteddecision",
    "goldlabel",
    "questionid",
    "benchmarkquestionid",
];

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CounterfactualPairFixture {
    pub schema_version: String,
    pub pair_id: String,
    pub family: String,
    pub query: String,
    pub as_of: String,
    pub cases: Vec<CounterfactualCaseFixture>,
    pub mutation: CounterfactualMutationContract,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CounterfactualCaseFixture {
    pub case_id: String,
    pub evidence: Vec<CounterfactualEvidenceFixture>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CounterfactualEvidenceFixture {
    pub evidence_id: String,
    pub source_id: String,
    pub source_family: String,
    pub entity_id: String,
    pub scope: String,
    pub observed_at: String,
    pub valid_until: Option<String>,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CounterfactualMutationContract {
    pub kind: String,
    pub changed_fields: Vec<String>,
    pub expected_decision_relation: String,
}

/// The only case projection intended for a memory backend or answer reader.
/// Pair IDs, mutation descriptions, and expected decision relations are excluded.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CounterfactualRuntimeCase {
    pub query: String,
    pub as_of: String,
    pub evidence: Vec<CounterfactualEvidenceFixture>,
}

impl CounterfactualPairFixture {
    pub fn runtime_case(&self, index: usize) -> Option<CounterfactualRuntimeCase> {
        self.cases.get(index).map(|case| CounterfactualRuntimeCase {
            query: self.query.clone(),
            as_of: self.as_of.clone(),
            evidence: case.evidence.clone(),
        })
    }
}

/// Parse and structurally validate JSONL fixtures. No gold/evaluator file is read.
///
/// The declared mutation must match the actual field-level diff between paired evidence
/// sets. If a list length changes, the whole list is the mutation path (for example,
/// `evidence`); fixed-length lists report precise paths (for example, `evidence[0].text`).
pub fn validate_counterfactual_fixture_jsonl(
    input: &str,
) -> Result<Vec<CounterfactualPairFixture>, String> {
    let mut pairs = Vec::new();
    let mut pair_ids = BTreeSet::new();

    for (line_index, line) in input.lines().enumerate() {
        let line_number = line_index + 1;
        if line.trim().is_empty() {
            return Err(format!("fixture line {line_number} is blank"));
        }
        let value: Value = serde_json::from_str(line)
            .map_err(|error| format!("fixture line {line_number} is invalid JSON: {error}"))?;
        reject_forbidden_runtime_keys(&value)
            .map_err(|error| format!("fixture line {line_number} rejected: {error}"))?;
        let pair: CounterfactualPairFixture = serde_json::from_value(value)
            .map_err(|error| format!("fixture line {line_number} violates the pinned schema: {error}"))?;
        validate_pair(&pair)
            .map_err(|error| format!("fixture line {line_number} rejected: {error}"))?;
        if !pair_ids.insert(pair.pair_id.clone()) {
            return Err(format!("duplicate pair_id {:?}", pair.pair_id));
        }
        pairs.push(pair);
    }

    if pairs.is_empty() {
        return Err("fixture corpus must contain at least one pair".into());
    }
    Ok(pairs)
}

fn validate_pair(pair: &CounterfactualPairFixture) -> Result<(), String> {
    if pair.schema_version != COUNTERFACTUAL_FIXTURE_SCHEMA {
        return Err(format!("unsupported schema_version {:?}", pair.schema_version));
    }
    require_trimmed("pair_id", &pair.pair_id)?;
    require_trimmed("query", &pair.query)?;
    validate_timestamp("as_of", &pair.as_of)?;

    if !ALLOWED_FAMILIES.contains(&pair.family.as_str()) {
        return Err(format!("unsupported mutation family {:?}", pair.family));
    }
    let expected_kind = expected_mutation_kind(&pair.family)
        .ok_or_else(|| format!("no pinned mutation kind for family {:?}", pair.family))?;
    if pair.mutation.kind != expected_kind {
        return Err(format!(
            "mutation kind {:?} does not match family {:?}; expected {:?}",
            pair.mutation.kind, pair.family, expected_kind
        ));
    }
    require_trimmed("mutation.kind", &pair.mutation.kind)?;
    if !ALLOWED_DECISION_RELATIONS.contains(&pair.mutation.expected_decision_relation.as_str()) {
        return Err(format!(
            "unsupported expected decision relation {:?}",
            pair.mutation.expected_decision_relation
        ));
    }
    let expected_relation = expected_decision_relation(&pair.family)
        .ok_or_else(|| format!("no pinned decision relation for family {:?}", pair.family))?;
    if pair.mutation.expected_decision_relation != expected_relation {
        return Err(format!(
            "decision relation {:?} does not match family {:?}; expected {:?}",
            pair.mutation.expected_decision_relation, pair.family, expected_relation
        ));
    }
    if pair.cases.len() != 2 {
        return Err("each pair must contain exactly two cases".into());
    }

    let left = &pair.cases[0];
    let right = &pair.cases[1];
    require_trimmed("case[0].case_id", &left.case_id)?;
    require_trimmed("case[1].case_id", &right.case_id)?;
    if left.case_id == right.case_id {
        return Err("paired case IDs must be distinct".into());
    }

    validate_evidence_set(&left.evidence, "case[0]")?;
    validate_evidence_set(&right.evidence, "case[1]")?;

    let left_value = serde_json::to_value(&left.evidence)
        .map_err(|error| format!("cannot serialize left evidence set: {error}"))?;
    let right_value = serde_json::to_value(&right.evidence)
        .map_err(|error| format!("cannot serialize right evidence set: {error}"))?;
    let mut actual_changes = Vec::new();
    diff_paths(&left_value, &right_value, "evidence", &mut actual_changes);
    actual_changes.sort();
    actual_changes.dedup();

    if actual_changes.is_empty() {
        return Err("paired evidence sets are identical; a counterfactual mutation is required".into());
    }

    let mut declared_changes = pair.mutation.changed_fields.clone();
    if declared_changes.iter().any(|field| field.trim().is_empty() || field.trim() != field) {
        return Err("mutation changed_fields must be non-empty, trimmed paths".into());
    }
    declared_changes.sort();
    declared_changes.dedup();
    if declared_changes.len() != pair.mutation.changed_fields.len() {
        return Err("mutation changed_fields must not contain duplicates".into());
    }
    if actual_changes != declared_changes {
        return Err(format!(
            "declared changed_fields {:?} do not match actual evidence diff {:?}",
            declared_changes, actual_changes
        ));
    }
    let expected_path = expected_mutation_path(&pair.family)
        .ok_or_else(|| format!("no pinned changed-field path for family {:?}", pair.family))?;
    if actual_changes.len() != 1 || actual_changes[0].as_str() != expected_path {
        return Err(format!(
            "mutation family {:?} requires changed path {:?}, observed {:?}",
            pair.family, expected_path, actual_changes
        ));
    }
    Ok(())
}

fn expected_mutation_kind(family: &str) -> Option<&'static str> {
    match family {
        "polarity_negation" => Some("polarity_flip"),
        "temporal_validity" => Some("validity_expiration"),
        "entity_binding" => Some("entity_swap"),
        "scope_premise" => Some("scope_swap"),
        "source_dependence" => Some("duplicate_source_projection"),
        "missing_requirement" => Some("remove_required_outcome"),
        "ordered_transition" => Some("terminal_state_flip"),
        "correction_update" => Some("add_later_correction"),
        "qualifier_context" => Some("qualifier_meaning_flip"),
        "memory_influence" => Some("add_irrelevant_memory_atom"),
        _ => None,
    }
}

fn expected_decision_relation(family: &str) -> Option<&'static str> {
    match family {
        "polarity_negation" => Some("claim_must_flip_or_be_rejected"),
        "temporal_validity" => Some("must_use_current_evidence_only"),
        "entity_binding" | "qualifier_context" => Some("must_recompute_support_with_context"),
        "scope_premise" => Some("must_not_cross_scope"),
        "source_dependence" => Some("must_not_treat_duplicate_source_as_independent_support"),
        "missing_requirement" => Some("must_abstain_or_refine_if_requirement_missing"),
        "ordered_transition" => Some("must_follow_observed_transition"),
        "correction_update" => Some("must_use_latest_applicable_correction"),
        "memory_influence" => Some("irrelevant_memory_should_not_change_answer"),
        _ => None,
    }
}

fn expected_mutation_path(family: &str) -> Option<&'static str> {
    match family {
        "polarity_negation" | "ordered_transition" | "qualifier_context" => {
            Some("evidence[0].text")
        }
        "temporal_validity" => Some("evidence[0].valid_until"),
        "entity_binding" => Some("evidence[0].entity_id"),
        "scope_premise" => Some("evidence[0].scope"),
        "source_dependence" | "missing_requirement" | "correction_update" | "memory_influence" => Some("evidence"),
        _ => None,
    }
}

fn validate_evidence_set(
    evidence: &[CounterfactualEvidenceFixture],
    context: &str,
) -> Result<(), String> {
    if evidence.is_empty() {
        return Err(format!("{context} must contain at least one evidence record"));
    }
    let mut ids = BTreeSet::new();
    for (index, item) in evidence.iter().enumerate() {
        let prefix = format!("{context}.evidence[{index}]");
        for (field, value) in [
            ("evidence_id", item.evidence_id.as_str()),
            ("source_id", item.source_id.as_str()),
            ("source_family", item.source_family.as_str()),
            ("entity_id", item.entity_id.as_str()),
            ("scope", item.scope.as_str()),
            ("text", item.text.as_str()),
        ] {
            require_trimmed(&format!("{prefix}.{field}"), value)?;
        }
        if !ids.insert(item.evidence_id.as_str()) {
            return Err(format!("{context} has duplicate evidence_id {:?}", item.evidence_id));
        }
        validate_timestamp(&format!("{prefix}.observed_at"), &item.observed_at)?;
        if let Some(valid_until) = &item.valid_until {
            validate_timestamp(&format!("{prefix}.valid_until"), valid_until)?;
        }
    }
    Ok(())
}

fn require_trimmed(field: &str, value: &str) -> Result<(), String> {
    if value.trim().is_empty() || value.trim() != value {
        return Err(format!("{field} must be non-empty and trimmed"));
    }
    Ok(())
}

fn validate_timestamp(field: &str, value: &str) -> Result<(), String> {
    DateTime::parse_from_rfc3339(value)
        .map(|_| ())
        .map_err(|error| format!("{field} must be RFC3339: {error}"))
}

fn diff_paths(left: &Value, right: &Value, path: &str, output: &mut Vec<String>) {
    if left == right {
        return;
    }
    match (left, right) {
        (Value::Object(left_object), Value::Object(right_object)) => {
            let left_keys: BTreeSet<_> = left_object.keys().collect();
            let right_keys: BTreeSet<_> = right_object.keys().collect();
            if left_keys != right_keys {
                output.push(if path.is_empty() { "<root>".into() } else { path.into() });
                return;
            }
            for key in left_object.keys() {
                let child = if path.is_empty() {
                    key.clone()
                } else {
                    format!("{path}.{key}")
                };
                let left_value = left_object.get(key).expect("matching object keys were checked");
                let right_value = right_object.get(key).expect("matching object keys were checked");
                diff_paths(left_value, right_value, &child, output);
            }
        }
        (Value::Array(left_array), Value::Array(right_array)) => {
            if left_array.len() != right_array.len() {
                output.push(if path.is_empty() { "<root>".into() } else { path.into() });
                return;
            }
            for (index, (left_item, right_item)) in left_array.iter().zip(right_array).enumerate() {
                diff_paths(left_item, right_item, &format!("{path}[{index}]"), output);
            }
        }
        _ => output.push(if path.is_empty() { "<root>".into() } else { path.into() }),
    }
}

fn reject_forbidden_runtime_keys(value: &Value) -> Result<(), String> {
    match value {
        Value::Object(object) => {
            for (key, value) in object {
                let normalized: String = key
                    .chars()
                    .filter(|character| character.is_ascii_alphanumeric())
                    .flat_map(|character| character.to_lowercase())
                    .collect();
                if FORBIDDEN_RUNTIME_KEYS.contains(&normalized.as_str()) {
                    return Err(format!("evaluator-only key {key:?} is forbidden in the input fixture"));
                }
                reject_forbidden_runtime_keys(value)?;
            }
        }
        Value::Array(values) => {
            for value in values {
                reject_forbidden_runtime_keys(value)?;
            }
        }
        _ => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURES: &str = include_str!("../tests/fixtures/counterfactual-evidence-audit-v1.jsonl");
    const LABELS: &str = include_str!("../tests/fixtures/evaluator-only/counterfactual-evidence-audit-labels-v1.jsonl");

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct GoldLabels {
        schema_version: String,
        pair_id: String,
        cases: Vec<GoldCaseLabel>,
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct GoldCaseLabel {
        case_id: String,
        answerability: String,
        expected_decision: String,
        expected_claim: String,
        supporting_evidence_ids: Vec<String>,
        #[serde(default)]
        atom_use_targets: Vec<GoldAtomUseTarget>,
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct GoldAtomUseTarget {
        evidence_id: String,
        target_level: String,
    }

    #[test]
    fn counterfactual_fixture_corpus_obeys_the_pinned_pair_contract() {
        let pairs = validate_counterfactual_fixture_jsonl(FIXTURES).expect("fixtures must validate");
        assert!(pairs.len() >= 8, "retain a nontrivial adversarial fixture suite");
        let families: BTreeSet<_> = pairs.iter().map(|pair| pair.family.as_str()).collect();
        assert_eq!(families.len(), 10, "cover every pinned counterfactual failure family");
    }

    #[test]
    fn runtime_projection_excludes_pair_and_evaluator_metadata() {
        let pairs = validate_counterfactual_fixture_jsonl(FIXTURES).expect("fixtures must validate");
        let runtime = pairs[0].runtime_case(0).expect("first case exists");
        let serialized = serde_json::to_value(runtime).unwrap();
        let serialized_text = serialized.to_string();
        assert!(!serialized_text.contains("pair_id"));
        assert!(!serialized_text.contains("expected_decision_relation"));
        assert!(!serialized_text.contains("answerability"));
        assert!(!serialized_text.contains("expected_claim"));
        assert!(!serialized_text.contains("supporting_evidence_ids"));
    }

    #[test]
    fn evaluator_labels_are_separate_and_cover_each_input_case_once() {
        let pairs = validate_counterfactual_fixture_jsonl(FIXTURES).expect("fixtures must validate");
        let mut expected_ids = BTreeSet::new();
        for pair in &pairs {
            for case in &pair.cases {
                assert!(expected_ids.insert(case.case_id.as_str()), "input case IDs must be unique");
            }
        }

        let pair_by_id: std::collections::BTreeMap<_, _> = pairs
            .iter()
            .map(|pair| (pair.pair_id.as_str(), pair))
            .collect();
        let mut seen_ids = BTreeSet::new();
        let mut label_pair_ids = BTreeSet::new();
        for (line_index, line) in LABELS.lines().enumerate() {
            let labels: GoldLabels = serde_json::from_str(line)
                .unwrap_or_else(|error| panic!("gold label line {} invalid: {error}", line_index + 1));
            assert_eq!(labels.schema_version, COUNTERFACTUAL_LABEL_SCHEMA);
            assert!(label_pair_ids.insert(labels.pair_id.clone()), "duplicate gold pair ID");
            assert_eq!(labels.cases.len(), 2, "each label pair must have two cases");
            let input_pair = pair_by_id
                .get(labels.pair_id.as_str())
                .unwrap_or_else(|| panic!("orphan gold pair label {:?}", labels.pair_id));
            let expected_pair_case_ids: BTreeSet<_> =
                input_pair.cases.iter().map(|case| case.case_id.as_str()).collect();
            let labeled_pair_case_ids: BTreeSet<_> =
                labels.cases.iter().map(|label| label.case_id.as_str()).collect();
            assert_eq!(labeled_pair_case_ids, expected_pair_case_ids, "gold case labels must match their own pair");
            for label in &labels.cases {
                assert!(expected_ids.contains(label.case_id.as_str()), "orphan gold case label");
                assert!(seen_ids.insert(label.case_id.clone()), "duplicate gold case label");
                assert!(["answerable", "unanswerable"].contains(&label.answerability.as_str()));
                assert!(["answer", "abstain", "refine"].contains(&label.expected_decision.as_str()));
                assert!(!label.expected_claim.trim().is_empty());
                assert!(!label.supporting_evidence_ids.iter().any(|id| id.trim().is_empty()));
                let source_case = input_pair
                    .cases
                    .iter()
                    .find(|case| case.case_id == label.case_id)
                    .expect("gold case belongs to this input pair");
                let available_evidence_ids: BTreeSet<_> = source_case
                    .evidence
                    .iter()
                    .map(|evidence| evidence.evidence_id.as_str())
                    .collect();
                assert!(
                    label.supporting_evidence_ids
                        .iter()
                        .all(|id| available_evidence_ids.contains(id.as_str())),
                    "gold support references evidence absent from case {}",
                    label.case_id
                );
                assert!(
                    label.atom_use_targets.iter().all(|target| {
                        available_evidence_ids.contains(target.evidence_id.as_str())
                            && ["ignore", "bound", "control"].contains(&target.target_level.as_str())
                    }),
                    "atom-use targets must reference case evidence and use Ignore/Bound/Control"
                );
                if input_pair.family == "memory_influence" {
                    assert_eq!(
                        label.atom_use_targets.len(),
                        source_case.evidence.len(),
                        "memory-influence fixtures must target every evidence atom"
                    );
                }
            }
        }
        assert_eq!(seen_ids.len(), expected_ids.len(), "every case must have exactly one gold label");
        assert_eq!(label_pair_ids.len(), pairs.len(), "every input pair must have a gold label pair");
    }

    #[test]
    fn memory_influence_labels_mark_decisive_and_irrelevant_atoms_differently() {
        let pairs = validate_counterfactual_fixture_jsonl(FIXTURES).expect("fixtures must validate");
        let pair = pairs
            .iter()
            .find(|pair| pair.family == "memory_influence")
            .expect("memory influence pair exists");
        let labels: Vec<GoldLabels> = LABELS
            .lines()
            .map(|line| serde_json::from_str(line).expect("gold labels are valid"))
            .collect();
        let label_pair = labels
            .iter()
            .find(|labels| labels.pair_id == pair.pair_id)
            .expect("matching influence labels exist");
        assert_eq!(
            label_pair.cases[0].expected_claim,
            label_pair.cases[1].expected_claim,
            "adding an irrelevant atom must not change the expected material claim"
        );
        assert_eq!(
            label_pair.cases[0].expected_decision,
            label_pair.cases[1].expected_decision,
            "adding an irrelevant atom must not change answer/abstain disposition"
        );
        let left_control = label_pair.cases[0]
            .atom_use_targets
            .iter()
            .find(|target| target.evidence_id == "retry-config-1")
            .expect("decisive atom has a target");
        assert_eq!(left_control.target_level, "control");
        let left_bound = label_pair.cases[0]
            .atom_use_targets
            .iter()
            .find(|target| target.evidence_id == "retry-applicability-2")
            .expect("bounded constraint has a target");
        assert_eq!(left_bound.target_level, "bound");
        let right_control = label_pair.cases[1]
            .atom_use_targets
            .iter()
            .find(|target| target.evidence_id == "retry-config-1")
            .expect("decisive atom retains its target");
        assert_eq!(right_control.target_level, "control");
        let right_bound = label_pair.cases[1]
            .atom_use_targets
            .iter()
            .find(|target| target.evidence_id == "retry-applicability-2")
            .expect("bounded constraint retains its target");
        assert_eq!(right_bound.target_level, "bound");
        let distractor = label_pair.cases[1]
            .atom_use_targets
            .iter()
            .find(|target| target.evidence_id == "notification-note-2")
            .expect("added memory atom has a target");
        assert_eq!(distractor.target_level, "ignore");
    }

    #[test]
    fn validator_rejects_undeclared_mutations_and_gold_label_leaks() {
        let pairs = validate_counterfactual_fixture_jsonl(FIXTURES).unwrap();
        let mut pair_value = serde_json::to_value(&pairs[0]).unwrap();
        // The declared mutation is a text change; adding a second scope change must fail.
        pair_value["cases"][1]["evidence"][0]["scope"] = Value::String("prod-west".into());
        let mutated = format!("{}\n", pair_value);
        assert!(validate_counterfactual_fixture_jsonl(&mutated).is_err());

        let mut wrong_relation = serde_json::to_value(&pairs[0]).unwrap();
        wrong_relation["mutation"]["expected_decision_relation"] = Value::String("must_not_cross_scope".into());
        assert!(validate_counterfactual_fixture_jsonl(&format!("{}\n", wrong_relation)).is_err());

        let mut leaking = serde_json::to_value(&pairs[0]).unwrap();
        leaking["answerability"] = Value::String("answerable".into());
        assert!(validate_counterfactual_fixture_jsonl(&format!("{}\n", leaking)).is_err());
    }
}
