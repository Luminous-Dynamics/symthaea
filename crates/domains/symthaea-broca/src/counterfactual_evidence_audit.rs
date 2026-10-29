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
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

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


#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluatorLabelPair {
    pub schema_version: String,
    pub pair_id: String,
    pub cases: Vec<EvaluatorCaseLabel>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluatorCaseLabel {
    pub case_id: String,
    pub answerability: String,
    pub expected_decision: String,
    pub expected_claim: String,
    pub supporting_evidence_ids: Vec<String>,
    #[serde(default)]
    pub atom_use_targets: Vec<AtomUseTarget>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AtomUseTarget {
    pub evidence_id: String,
    pub target_level: String,
    #[serde(default)]
    pub must_change_fields: Vec<String>,
    #[serde(default)]
    pub allowed_change_fields: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AtomAblationObservation {
    pub schema_version: String,
    pub run_id: String,
    pub case_id: String,
    pub evidence_id: String,
    pub reader_revision: String,
    pub fixture_sha256: String,
    pub prompt_sha256: String,
    pub tokenizer_sha256: String,
    pub system_commit_sha: String,
    pub full_context_tokens: usize,
    pub ablated_context_tokens: usize,
    pub full_fields: BTreeMap<String, String>,
    pub ablated_fields: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AtomAblationJudgment {
    pub case_id: String,
    pub evidence_id: String,
    pub target_level: String,
    pub actual_changed_fields: Vec<String>,
    pub missing_required_changes: Vec<String>,
    pub disallowed_changes: Vec<String>,
    pub overuse: bool,
    pub underuse: bool,
    pub context_tokens_saved: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AtomAblationScoreSummary {
    pub schema_version: String,
    pub run_id: String,
    pub reader_revision: String,
    pub fixture_sha256: String,
    pub prompt_sha256: String,
    pub tokenizer_sha256: String,
    pub system_commit_sha: String,
    pub observation_count: usize,
    pub ignore_count: usize,
    pub ignore_overuse_count: usize,
    pub ignore_overuse_rate: Option<f64>,
    pub bound_count: usize,
    pub bound_overuse_count: usize,
    pub bound_overuse_rate: Option<f64>,
    pub bound_underuse_count: usize,
    pub bound_underuse_rate: Option<f64>,
    pub control_count: usize,
    pub control_overuse_count: usize,
    pub control_overuse_rate: Option<f64>,
    pub control_underuse_count: usize,
    pub control_underuse_rate: Option<f64>,
    pub context_tokens_saved_total: i64,
    pub judgments: Vec<AtomAblationJudgment>,
}

/// Offline evaluation only: compare structured material outputs with and without each
/// labelled atom. This function is not part of the production inference path.
pub fn score_atom_ablation_jsonl(
    observations_jsonl: &str,
    input_fixture_jsonl: &str,
    evaluator_labels_jsonl: &str,
) -> Result<AtomAblationScoreSummary, String> {
    const OBSERVATION_SCHEMA: &str = "BROCA-MEMORY-ATOM-ABLATION-OBSERVATIONS-V1";
    let fixtures = validate_counterfactual_fixture_jsonl(input_fixture_jsonl)?;
    let expected_fixture_sha256 = hex::encode(Sha256::digest(input_fixture_jsonl.as_bytes()));
    let mut fixture_evidence_keys = BTreeSet::<(String, String)>::new();
    let mut fixture_case_ids = BTreeSet::<String>::new();
    let mut fixture_pair_cases = BTreeMap::<String, BTreeSet<String>>::new();
    for pair in &fixtures {
        fixture_pair_cases.insert(
            pair.pair_id.clone(),
            pair.cases.iter().map(|case| case.case_id.clone()).collect(),
        );
        for case in &pair.cases {
            fixture_case_ids.insert(case.case_id.clone());
            for evidence in &case.evidence {
                fixture_evidence_keys.insert((case.case_id.clone(), evidence.evidence_id.clone()));
            }
        }
    }
    let mut observations = Vec::new();
    let mut observation_keys = BTreeSet::new();

    for (line_index, line) in observations_jsonl.lines().enumerate() {
        let line_number = line_index + 1;
        if line.trim().is_empty() {
            return Err(format!("observation line {line_number} is blank"));
        }
        let observation: AtomAblationObservation = serde_json::from_str(line)
            .map_err(|error| format!("observation line {line_number} invalid: {error}"))?;
        if observation.schema_version != OBSERVATION_SCHEMA {
            return Err(format!("unsupported observation schema on line {line_number}"));
        }
        require_trimmed("run_id", &observation.run_id)?;
        require_trimmed("case_id", &observation.case_id)?;
        require_trimmed("evidence_id", &observation.evidence_id)?;
        require_trimmed("reader_revision", &observation.reader_revision)?;
        validate_hex_digest("fixture_sha256", &observation.fixture_sha256, 64)?;
        if observation.fixture_sha256 != expected_fixture_sha256 {
            return Err(format!("fixture digest mismatch on observation line {line_number}"));
        }
        validate_hex_digest("prompt_sha256", &observation.prompt_sha256, 64)?;
        validate_hex_digest("tokenizer_sha256", &observation.tokenizer_sha256, 64)?;
        validate_hex_digest("system_commit_sha", &observation.system_commit_sha, 40)?;
        validate_material_fields(&observation.full_fields, line_number, "full_fields")?;
        validate_material_fields(&observation.ablated_fields, line_number, "ablated_fields")?;
        if !observation_keys.insert((observation.case_id.clone(), observation.evidence_id.clone())) {
            return Err(format!(
                "duplicate ablation observation for case {:?} / evidence {:?}",
                observation.case_id, observation.evidence_id
            ));
        }
        observations.push(observation);
    }
    if observations.is_empty() {
        return Err("at least one ablation observation is required".into());
    }

    let mut label_pairs = BTreeSet::new();
    let mut targets = BTreeMap::<(String, String), AtomUseTarget>::new();
    for (line_index, line) in evaluator_labels_jsonl.lines().enumerate() {
        let line_number = line_index + 1;
        if line.trim().is_empty() {
            return Err(format!("evaluator-label line {line_number} is blank"));
        }
        let pair: EvaluatorLabelPair = serde_json::from_str(line)
            .map_err(|error| format!("evaluator-label line {line_number} invalid: {error}"))?;
        if pair.schema_version != COUNTERFACTUAL_LABEL_SCHEMA {
            return Err(format!("unsupported evaluator label schema on line {line_number}"));
        }
        require_trimmed("label pair_id", &pair.pair_id)?;
        if !label_pairs.insert(pair.pair_id.clone()) {
            return Err(format!("duplicate evaluator label pair {:?}", pair.pair_id));
        }
        let expected_pair_cases = fixture_pair_cases
            .get(&pair.pair_id)
            .ok_or_else(|| format!("evaluator label references unknown fixture pair {:?}", pair.pair_id))?;
        let mut case_ids = BTreeSet::new();
        for case in pair.cases {
            require_trimmed("label case_id", &case.case_id)?;
            if !case_ids.insert(case.case_id.clone()) {
                return Err(format!("duplicate labeled case {:?}", case.case_id));
            }
            if !expected_pair_cases.contains(&case.case_id) {
                return Err(format!(
                    "case {:?} is not a member of its labelled fixture pair {:?}",
                    case.case_id, pair.pair_id
                ));
            }
            for target in case.atom_use_targets {
                validate_atom_target(&target)?;
                let key = (case.case_id.clone(), target.evidence_id.clone());
                if !fixture_evidence_keys.contains(&key) {
                    return Err(format!("evaluator target references absent fixture evidence {key:?}"));
                }
                if targets.insert(key.clone(), target).is_some() {
                    return Err(format!("duplicate atom-use target for {key:?}"));
                }
            }
        }
        if &case_ids != expected_pair_cases {
            return Err(format!(
                "evaluator labels must exactly cover fixture pair {:?}; expected {:?}, got {:?}",
                pair.pair_id, expected_pair_cases, case_ids
            ));
        }
    }
    if targets.is_empty() {
        return Err("evaluator labels contain no atom-use targets".into());
    }

    let observation_key_set: BTreeSet<_> = observation_keys.iter().cloned().collect();
    let target_key_set: BTreeSet<_> = targets.keys().cloned().collect();
    if observation_key_set != target_key_set {
        let missing: Vec<_> = target_key_set.difference(&observation_key_set).cloned().collect();
        let extra: Vec<_> = observation_key_set.difference(&target_key_set).cloned().collect();
        return Err(format!(
            "ablation observations must exactly match labelled atom targets; missing={missing:?}, extra={extra:?}"
        ));
    }

    let run_id = observations[0].run_id.clone();
    let reader_revision = observations[0].reader_revision.clone();
    let fixture_sha256 = observations[0].fixture_sha256.clone();
    let prompt_sha256 = observations[0].prompt_sha256.clone();
    let tokenizer_sha256 = observations[0].tokenizer_sha256.clone();
    let system_commit_sha = observations[0].system_commit_sha.clone();
    if observations.iter().any(|item| {
        item.run_id != run_id
            || item.reader_revision != reader_revision
            || item.fixture_sha256 != fixture_sha256
            || item.prompt_sha256 != prompt_sha256
            || item.tokenizer_sha256 != tokenizer_sha256
            || item.system_commit_sha != system_commit_sha
    }) {
        return Err("all observations in one score run must pin identical run, reader, fixture, prompt, tokenizer, and code revisions".into());
    }

    let mut judgments = Vec::with_capacity(observations.len());
    for observation in observations {
        let key = (observation.case_id.clone(), observation.evidence_id.clone());
        let target = targets.get(&key).expect("observation/target key equality validated");
        let actual = changed_material_fields(&observation.full_fields, &observation.ablated_fields);
        let required: BTreeSet<_> = target.must_change_fields.iter().cloned().collect();
        let allowed: BTreeSet<_> = target.allowed_change_fields.iter().cloned().collect();
        let missing_required_changes: Vec<_> = required.difference(&actual).cloned().collect();
        let disallowed_changes: Vec<_> = actual.difference(&allowed).cloned().collect();
        let full_tokens = i64::try_from(observation.full_context_tokens)
            .map_err(|_| "full_context_tokens exceeds signed receipt range".to_string())?;
        let ablated_tokens = i64::try_from(observation.ablated_context_tokens)
            .map_err(|_| "ablated_context_tokens exceeds signed receipt range".to_string())?;
        judgments.push(AtomAblationJudgment {
            case_id: observation.case_id,
            evidence_id: observation.evidence_id,
            target_level: target.target_level.clone(),
            actual_changed_fields: actual.into_iter().collect(),
            overuse: !disallowed_changes.is_empty(),
            underuse: !missing_required_changes.is_empty(),
            missing_required_changes,
            disallowed_changes,
            context_tokens_saved: full_tokens - ablated_tokens,
        });
    }
    judgments.sort_by(|left, right| {
        left.case_id.cmp(&right.case_id).then_with(|| left.evidence_id.cmp(&right.evidence_id))
    });

    let count_level = |level: &str| judgments.iter().filter(|item| item.target_level == level).count();
    let ignore_overuse_count = judgments.iter()
        .filter(|item| item.target_level == "ignore" && item.overuse).count();
    let bound_overuse_count = judgments.iter()
        .filter(|item| item.target_level == "bound" && item.overuse).count();
    let bound_underuse_count = judgments.iter()
        .filter(|item| item.target_level == "bound" && item.underuse).count();
    let control_overuse_count = judgments.iter()
        .filter(|item| item.target_level == "control" && item.overuse).count();
    let control_underuse_count = judgments.iter()
        .filter(|item| item.target_level == "control" && item.underuse).count();
    let rate = |numerator: usize, denominator: usize| {
        (denominator > 0).then_some(numerator as f64 / denominator as f64)
    };
    let ignore_count = count_level("ignore");
    let bound_count = count_level("bound");
    let control_count = count_level("control");
    let context_tokens_saved_total = judgments.iter().map(|item| item.context_tokens_saved).sum();

    Ok(AtomAblationScoreSummary {
        schema_version: "BROCA-MEMORY-ATOM-ABLATION-SCORE-V1".into(),
        run_id,
        reader_revision,
        fixture_sha256,
        prompt_sha256,
        tokenizer_sha256,
        system_commit_sha,
        observation_count: judgments.len(),
        ignore_count,
        ignore_overuse_count,
        ignore_overuse_rate: rate(ignore_overuse_count, ignore_count),
        bound_count,
        bound_overuse_count,
        bound_overuse_rate: rate(bound_overuse_count, bound_count),
        bound_underuse_count,
        bound_underuse_rate: rate(bound_underuse_count, bound_count),
        control_count,
        control_overuse_count,
        control_overuse_rate: rate(control_overuse_count, control_count),
        control_underuse_count,
        control_underuse_rate: rate(control_underuse_count, control_count),
        context_tokens_saved_total,
        judgments,
    })
}

fn validate_hex_digest(field: &str, value: &str, expected_len: usize) -> Result<(), String> {
    if value.len() != expected_len
        || !value.bytes().all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(format!("{field} must be {expected_len} lowercase hexadecimal characters"));
    }
    Ok(())
}

fn validate_material_fields(
    fields: &BTreeMap<String, String>,
    line_number: usize,
    field_name: &str,
) -> Result<(), String> {
    for (field, value) in fields {
        require_trimmed(&format!("{field_name} key"), field)?;
        require_trimmed(&format!("{field_name}[{field:?}] value"), value)?;
        if !["claim.", "action.", "decision."].iter().any(|prefix| field.starts_with(prefix)) {
            return Err(format!("non-material output key {field:?} in {field_name} on line {line_number}"));
        }
    }
    Ok(())
}

fn validate_atom_target(target: &AtomUseTarget) -> Result<(), String> {
    require_trimmed("atom target evidence_id", &target.evidence_id)?;
    match target.target_level.as_str() {
        "ignore" if !target.must_change_fields.is_empty() || !target.allowed_change_fields.is_empty() => {
            return Err("ignore targets must have empty must_change_fields and allowed_change_fields".into());
        }
        "ignore" | "bound" | "control" => {}
        _ => return Err(format!("unsupported atom-use target level {:?}", target.target_level)),
    }
    let must: BTreeSet<_> = target.must_change_fields.iter().collect();
    let allowed: BTreeSet<_> = target.allowed_change_fields.iter().collect();
    if must.len() != target.must_change_fields.len() || allowed.len() != target.allowed_change_fields.len() {
        return Err("atom target field lists must not contain duplicates".into());
    }
    for field in target.must_change_fields.iter().chain(target.allowed_change_fields.iter()) {
        require_trimmed("atom target material field", field)?;
        if !["claim.", "action.", "decision."].iter().any(|prefix| field.starts_with(prefix)) {
            return Err(format!("atom target names non-material field {field:?}"));
        }
    }
    if !must.is_subset(&allowed) {
        return Err("must_change_fields must be a subset of allowed_change_fields".into());
    }
    if target.target_level == "control" && target.must_change_fields.is_empty() {
        return Err("control targets require at least one must_change_field".into());
    }
    Ok(())
}

fn changed_material_fields(
    full: &BTreeMap<String, String>,
    ablated: &BTreeMap<String, String>,
) -> BTreeSet<String> {
    let keys: BTreeSet<_> = full.keys().chain(ablated.keys()).cloned().collect();
    keys.into_iter().filter(|key| full.get(key) != ablated.get(key)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURES: &str = include_str!("../tests/fixtures/counterfactual-evidence-audit-v1.jsonl");
    const LABELS: &str = include_str!("../tests/fixtures/evaluator-only/counterfactual-evidence-audit-labels-v1.jsonl");


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
            let labels: EvaluatorLabelPair = serde_json::from_str(line)
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
        let labels: Vec<EvaluatorLabelPair> = LABELS
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

    fn synthetic_ablation_rows(fixture_sha256: &str) -> Vec<Value> {
        use serde_json::json;

        fn observation(
            fixture_sha256: &str,
            case_id: &str,
            evidence_id: &str,
            full_fields: Value,
            ablated_fields: Value,
            full_context_tokens: usize,
            ablated_context_tokens: usize,
        ) -> Value {
            serde_json::json!({
                "schema_version": "BROCA-MEMORY-ATOM-ABLATION-OBSERVATIONS-V1",
                "run_id": "synthetic-scorer-test-v1",
                "case_id": case_id,
                "evidence_id": evidence_id,
                "reader_revision": "synthetic-reader-v1",
                "fixture_sha256": fixture_sha256,
                "prompt_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                "tokenizer_sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
                "system_commit_sha": "dddddddddddddddddddddddddddddddddddddddd",
                "full_context_tokens": full_context_tokens,
                "ablated_context_tokens": ablated_context_tokens,
                "full_fields": full_fields,
                "ablated_fields": ablated_fields
            })
        }

        let full = json!({
            "claim.retry_limit.value": "4",
            "claim.retry_limit.scope": "transient_network_errors"
        });
        let scope_bounded = json!({
            "claim.retry_limit.value": "4",
            "claim.retry_limit.scope": "unknown_scope"
        });
        let no_value = json!({
            "claim.retry_limit.scope": "transient_network_errors"
        });
        vec![
            observation(fixture_sha256, "memory-influence-0001-a", "retry-config-1", full.clone(), no_value.clone(), 50, 40),
            observation(fixture_sha256, "memory-influence-0001-a", "retry-applicability-2", full.clone(), scope_bounded.clone(), 50, 42),
            observation(fixture_sha256, "memory-influence-0001-b", "retry-config-1", full.clone(), no_value, 60, 50),
            observation(fixture_sha256, "memory-influence-0001-b", "retry-applicability-2", full.clone(), scope_bounded, 60, 52),
            observation(fixture_sha256, "memory-influence-0001-b", "notification-note-2", full.clone(), full, 60, 55),
        ]
    }

    #[test]
    fn atom_ablation_scorer_reports_ignore_bound_and_control_separately() {
        let fixture_sha256 = hex::encode(Sha256::digest(FIXTURES.as_bytes()));
        let rows = synthetic_ablation_rows(&fixture_sha256);
        let observations = rows.iter().map(Value::to_string).collect::<Vec<_>>().join("\n");
        let result = score_atom_ablation_jsonl(&observations, FIXTURES, LABELS)
            .expect("well-formed synthetic ablation run must score");
        assert_eq!(result.observation_count, 5);
        assert_eq!(result.ignore_count, 1);
        assert_eq!(result.ignore_overuse_count, 0);
        assert_eq!(result.bound_count, 2);
        assert_eq!(result.bound_overuse_count, 0);
        assert_eq!(result.bound_underuse_count, 0);
        assert_eq!(result.control_count, 2);
        assert_eq!(result.control_underuse_count, 0);
        assert_eq!(result.control_overuse_count, 0);
        assert_eq!(result.context_tokens_saved_total, 41);
        assert_eq!(result.ignore_overuse_rate, Some(0.0));
        assert_eq!(result.control_underuse_rate, Some(0.0));
    }

    #[test]
    fn atom_ablation_scorer_detects_overuse_and_underuse() {
        let fixture_sha256 = hex::encode(Sha256::digest(FIXTURES.as_bytes()));
        let mut rows = synthetic_ablation_rows(&fixture_sha256);
        // The Ignore atom now appears to change a material claim: over-use.
        rows[4]["ablated_fields"] = serde_json::json!({
            "claim.retry_limit.value": "5",
            "claim.retry_limit.scope": "transient_network_errors"
        });
        // Removing a Control atom causes no required claim change: under-use.
        let unchanged_control = rows[2]["full_fields"].clone();
        rows[2]["ablated_fields"] = unchanged_control;
        // The Bound atom should affect only the retry-limit scope field; no change is under-use.
        let unchanged_bound = rows[3]["full_fields"].clone();
        rows[3]["ablated_fields"] = unchanged_bound;
        let observations = rows.iter().map(Value::to_string).collect::<Vec<_>>().join("\n");
        let result = score_atom_ablation_jsonl(&observations, FIXTURES, LABELS)
            .expect("valid but behaviorally faulty synthetic observations still score");
        assert_eq!(result.ignore_overuse_count, 1);
        assert_eq!(result.control_underuse_count, 1);
        assert_eq!(result.bound_underuse_count, 1);
    }

    #[test]
    fn atom_ablation_scorer_binds_fixture_digest_and_allows_signed_token_delta() {
        let fixture_sha256 = hex::encode(Sha256::digest(FIXTURES.as_bytes()));
        let mut rows = synthetic_ablation_rows(&fixture_sha256);
        // BPE token counts can change at serialization boundaries; record signed savings
        // instead of assuming the ablated context must always have fewer tokens.
        rows[4]["ablated_context_tokens"] = serde_json::json!(65);
        let observations = rows.iter().map(Value::to_string).collect::<Vec<_>>().join("\n");
        let result = score_atom_ablation_jsonl(&observations, FIXTURES, LABELS)
            .expect("valid observations may have a negative token delta");
        assert_eq!(result.context_tokens_saved_total, 31);

        rows[0]["fixture_sha256"] =
            serde_json::json!("eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee");
        let wrong_digest = rows.iter().map(Value::to_string).collect::<Vec<_>>().join("\n");
        assert!(score_atom_ablation_jsonl(&wrong_digest, FIXTURES, LABELS).is_err());
    }

    #[test]
    fn atom_ablation_scorer_rejects_missing_or_unlabelled_observations() {
        let fixture_sha256 = hex::encode(Sha256::digest(FIXTURES.as_bytes()));
        let rows = synthetic_ablation_rows(&fixture_sha256);
        let observations = rows.iter().take(4).map(Value::to_string).collect::<Vec<_>>().join("\n");
        assert!(score_atom_ablation_jsonl(&observations, FIXTURES, LABELS).is_err());

        let observations = rows.iter().map(Value::to_string).collect::<Vec<_>>().join("\n");
        let extra = serde_json::json!({
            "schema_version": "BROCA-MEMORY-ATOM-ABLATION-OBSERVATIONS-V1",
            "run_id": "synthetic-scorer-test-v1",
            "case_id": "memory-influence-0001-b",
            "evidence_id": "unlabelled-atom",
            "reader_revision": "synthetic-reader-v1",
            "fixture_sha256": fixture_sha256,
            "prompt_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "tokenizer_sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
            "system_commit_sha": "dddddddddddddddddddddddddddddddddddddddd",
            "full_context_tokens": 10,
            "ablated_context_tokens": 5,
            "full_fields": {"claim.retry_limit.value": "4"},
            "ablated_fields": {}
        });
        let observations = format!("{}\n{}", observations, extra);
        assert!(score_atom_ablation_jsonl(&observations, FIXTURES, LABELS).is_err());

        let mismatched_pair_labels = LABELS.replace(
            r#""pair_id":"memory-influence-0001""#,
            r#""pair_id":"unknown-pair""#,
        );
        assert!(score_atom_ablation_jsonl(&observations, FIXTURES, &mismatched_pair_labels).is_err());
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
