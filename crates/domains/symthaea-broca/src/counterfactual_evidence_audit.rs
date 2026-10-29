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
    require_trimmed("mutation.kind", &pair.mutation.kind)?;
    if !ALLOWED_DECISION_RELATIONS.contains(&pair.mutation.expected_decision_relation.as_str()) {
        return Err(format!(
            "unsupported expected decision relation {:?}",
            pair.mutation.expected_decision_relation
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
    diff_paths(&left_value, &right_value, "", &mut actual_changes);
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
    Ok(())
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
                diff_paths(&left_object[key], &right_object[key], &child, output);
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
                    .flat_map(char::to_lowercase)
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
    }

    #[test]
    fn counterfactual_fixture_corpus_obeys_the_pinned_pair_contract() {
        let pairs = validate_counterfactual_fixture_jsonl(FIXTURES).expect("fixtures must validate");
        assert!(pairs.len() >= 8, "retain a nontrivial adversarial fixture suite");
        let families: BTreeSet<_> = pairs.iter().map(|pair| pair.family.as_str()).collect();
        assert!(families.len() >= 8, "cover distinct counterfactual failure families");
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

        let mut seen_ids = BTreeSet::new();
        let mut label_pair_ids = BTreeSet::new();
        for (line_index, line) in LABELS.lines().enumerate() {
            let labels: GoldLabels = serde_json::from_str(line)
                .unwrap_or_else(|error| panic!("gold label line {} invalid: {error}", line_index + 1));
            assert_eq!(labels.schema_version, COUNTERFACTUAL_LABEL_SCHEMA);
            assert!(label_pair_ids.insert(labels.pair_id.clone()), "duplicate gold pair ID");
            assert_eq!(labels.cases.len(), 2, "each label pair must have two cases");
            for label in &labels.cases {
                assert!(expected_ids.contains(label.case_id.as_str()), "orphan gold case label");
                assert!(seen_ids.insert(label.case_id.clone()), "duplicate gold case label");
                assert!(["answerable", "unanswerable"].contains(&label.answerability.as_str()));
                assert!(["answer", "abstain", "refine"].contains(&label.expected_decision.as_str()));
                assert!(!label.expected_claim.trim().is_empty());
                assert!(!label.supporting_evidence_ids.iter().any(|id| id.trim().is_empty()));
            }
        }
        assert_eq!(seen_ids.len(), expected_ids.len(), "every case must have exactly one gold label");
        assert_eq!(label_pair_ids.len(), pairs.len(), "every input pair must have a gold label pair");
    }

    #[test]
    fn validator_rejects_undeclared_mutations_and_gold_label_leaks() {
        let pairs = validate_counterfactual_fixture_jsonl(FIXTURES).unwrap();
        let mut pair_value = serde_json::to_value(&pairs[0]).unwrap();
        pair_value["cases"][1]["evidence"][0]["text"] = Value::String("undeclared mutation".into());
        let mutated = format!("{}\n", pair_value);
        assert!(validate_counterfactual_fixture_jsonl(&mutated).is_err());

        let mut leaking = serde_json::to_value(&pairs[0]).unwrap();
        leaking["answerability"] = Value::String("answerable".into());
        assert!(validate_counterfactual_fixture_jsonl(&format!("{}\n", leaking)).is_err());
    }
}
