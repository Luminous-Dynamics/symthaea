// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Narrow deterministic replay boundary for persisted disposition evaluations.
//!
//! This crate replays an explicit disposition policy over explicit predicate
//! facts and an explicit scientific-lineage context. A positive witness proves
//! only that one persisted *evaluation* exactly matches that deterministic
//! replay.
//!
//! It deliberately does not reconstruct evidence admission, candidate
//! accounting, reason topology, predicate derivation, provenance legitimacy,
//! evidence completeness, currentness, truth, consensus, recommendation, or
//! action authority. A later SCI layer must reconstruct and qualify those
//! upstream semantics before it may mint a full scientific-disposition replay
//! witness.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

pub const DISPOSITION_EVALUATION_REPLAY_PROFILE_V1: &str =
    "symthaea.science.disposition-evaluation-replay.v1";

/// Categorical policy input. Variant declaration order has no epistemic semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PredicateValueV1 {
    Satisfied,
    NotSatisfied,
    Unknown,
    NotApplicable,
    Blocked,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PredicateFactV1 {
    pub predicate_id: String,
    pub value: PredicateValueV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuleRequirementV1 {
    pub predicate_id: String,
    pub expected: PredicateValueV1,
}

/// Smaller priorities have higher precedence. Priorities must be unique.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DispositionRuleV1 {
    pub rule_id: String,
    pub priority: u32,
    pub requirements: Vec<RuleRequirementV1>,
    pub output: String,
}

/// Explicit policy data, not a qualified policy capability.
///
/// `artifact_id` binds provenance identity in addition to semantic content.
/// This crate does not authenticate that identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DispositionPolicyV1 {
    pub profile_id: String,
    pub artifact_id: String,
    pub rules: Vec<DispositionRuleV1>,
    pub fallback_output: String,
}

/// Exact scientific lineage supplied by an upstream SCI layer.
///
/// Every field is an opaque identity here. Replay binds it exactly but does
/// not prove the referenced state is authentic, complete, current, or
/// scientifically authoritative.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DispositionEvaluationContextV1 {
    pub evidence_view_id: String,
    pub evidence_view_snapshot_id: String,
    pub source_lifecycle_generation_id: String,
    pub external_argument_generation_id: String,
    pub dependency_graph_generation_id: String,
    pub target_compatibility_generation_id: String,
    pub triangulation_generation_id: String,
    pub reason_topology_snapshot_id: String,
    pub predicate_derivation_profile_id: String,
    pub predicate_derivation_artifact_id: String,
    pub predicate_derivation_execution_lineage_id: String,
    pub information_cutoff_id: String,
}

/// Explicit inputs to the policy-evaluation replay boundary.
///
/// `reason_topology_ids` are retained for explanation/access while
/// `reason_topology_snapshot_id` in the context binds the exact graph/root.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DispositionEvaluationInputsV1 {
    pub proposition_id: String,
    pub scientific_use_id: String,
    pub context: DispositionEvaluationContextV1,
    pub reason_topology_ids: Vec<String>,
    pub predicates: Vec<PredicateFactV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequirementTraceV1 {
    pub predicate_id: String,
    pub expected: PredicateValueV1,
    pub observed: Option<PredicateValueV1>,
    pub satisfied: bool,
}

/// Every rule is retained, including rules that lose to higher precedence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuleTraceV1 {
    pub rule_id: String,
    pub priority: u32,
    pub matched: bool,
    pub requirements: Vec<RequirementTraceV1>,
}

/// Complete evaluation material understood by this replay profile.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DispositionEvaluationMaterialV1 {
    pub proposition_id: String,
    pub scientific_use_id: String,
    pub context: DispositionEvaluationContextV1,
    pub policy: DispositionPolicyV1,
    pub reason_topology_ids: Vec<String>,
    pub predicate_facts: Vec<PredicateFactV1>,
    pub primary_disposition: String,
    pub policy_trace: Vec<RuleTraceV1>,
}

/// Ordinary persisted data. Deserialization does not recreate the positive
/// evaluation-replay witness. `assessment_id` is caller-shaped record metadata
/// and is deliberately not promoted into the positive replay capability.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PersistedDispositionEvaluationRecordV1 {
    pub replay_profile_id: String,
    pub assessment_id: String,
    pub material: DispositionEvaluationMaterialV1,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplayErrorV1 {
    EmptyField {
        field: &'static str,
    },
    DuplicatePredicate {
        predicate_id: String,
    },
    DuplicateReason {
        reason_id: String,
    },
    DuplicateRule {
        rule_id: String,
    },
    DuplicatePriority {
        priority: u32,
    },
    DuplicateRequirement {
        rule_id: String,
        predicate_id: String,
    },
    ReplayProfileMismatch {
        found: String,
    },
    PropositionMismatch,
    ScientificUseMismatch,
    EvaluationContextMismatch,
    PolicyMismatch,
    ReasonTopologyMismatch,
    PredicateSnapshotMismatch,
    PrimaryDispositionMismatch,
    PolicyTraceMismatch,
}

impl fmt::Display for ReplayErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyField { field } => write!(f, "empty required field: {field}"),
            Self::DuplicatePredicate { predicate_id } => {
                write!(f, "duplicate predicate identity: {predicate_id}")
            }
            Self::DuplicateReason { reason_id } => {
                write!(f, "duplicate reason identity: {reason_id}")
            }
            Self::DuplicateRule { rule_id } => write!(f, "duplicate rule identity: {rule_id}"),
            Self::DuplicatePriority { priority } => {
                write!(f, "duplicate rule priority: {priority}")
            }
            Self::DuplicateRequirement {
                rule_id,
                predicate_id,
            } => write!(f, "duplicate requirement {predicate_id} in rule {rule_id}"),
            Self::ReplayProfileMismatch { found } => {
                write!(f, "unexpected replay profile: {found}")
            }
            Self::PropositionMismatch => write!(f, "persisted proposition does not match replay"),
            Self::ScientificUseMismatch => {
                write!(f, "persisted scientific use does not match replay")
            }
            Self::EvaluationContextMismatch => {
                write!(f, "persisted evaluation context does not match replay")
            }
            Self::PolicyMismatch => write!(f, "persisted policy does not match replay"),
            Self::ReasonTopologyMismatch => {
                write!(f, "persisted reason topology does not match replay")
            }
            Self::PredicateSnapshotMismatch => {
                write!(f, "persisted predicate snapshot does not match replay")
            }
            Self::PrimaryDispositionMismatch => {
                write!(f, "persisted primary disposition does not match replay")
            }
            Self::PolicyTraceMismatch => write!(f, "persisted policy trace does not match replay"),
        }
    }
}

impl Error for ReplayErrorV1 {}

fn non_empty(value: &str, field: &'static str) -> Result<(), ReplayErrorV1> {
    if value.trim().is_empty() {
        Err(ReplayErrorV1::EmptyField { field })
    } else {
        Ok(())
    }
}

fn canonical_context(
    context: &DispositionEvaluationContextV1,
) -> Result<DispositionEvaluationContextV1, ReplayErrorV1> {
    for (field, value) in [
        ("evidence_view_id", context.evidence_view_id.as_str()),
        (
            "evidence_view_snapshot_id",
            context.evidence_view_snapshot_id.as_str(),
        ),
        (
            "source_lifecycle_generation_id",
            context.source_lifecycle_generation_id.as_str(),
        ),
        (
            "external_argument_generation_id",
            context.external_argument_generation_id.as_str(),
        ),
        (
            "dependency_graph_generation_id",
            context.dependency_graph_generation_id.as_str(),
        ),
        (
            "target_compatibility_generation_id",
            context.target_compatibility_generation_id.as_str(),
        ),
        (
            "triangulation_generation_id",
            context.triangulation_generation_id.as_str(),
        ),
        (
            "reason_topology_snapshot_id",
            context.reason_topology_snapshot_id.as_str(),
        ),
        (
            "predicate_derivation_profile_id",
            context.predicate_derivation_profile_id.as_str(),
        ),
        (
            "predicate_derivation_artifact_id",
            context.predicate_derivation_artifact_id.as_str(),
        ),
        (
            "predicate_derivation_execution_lineage_id",
            context.predicate_derivation_execution_lineage_id.as_str(),
        ),
        (
            "information_cutoff_id",
            context.information_cutoff_id.as_str(),
        ),
    ] {
        non_empty(value, field)?;
    }
    Ok(context.clone())
}

fn canonical_predicates(
    predicates: &[PredicateFactV1],
) -> Result<(BTreeMap<String, PredicateValueV1>, Vec<PredicateFactV1>), ReplayErrorV1> {
    let mut map = BTreeMap::new();
    for fact in predicates {
        non_empty(&fact.predicate_id, "predicate_id")?;
        if map.insert(fact.predicate_id.clone(), fact.value).is_some() {
            return Err(ReplayErrorV1::DuplicatePredicate {
                predicate_id: fact.predicate_id.clone(),
            });
        }
    }
    let canonical = map
        .iter()
        .map(|(predicate_id, value)| PredicateFactV1 {
            predicate_id: predicate_id.clone(),
            value: *value,
        })
        .collect();
    Ok((map, canonical))
}

fn canonical_reasons(reasons: &[String]) -> Result<Vec<String>, ReplayErrorV1> {
    let mut seen = BTreeSet::new();
    for reason in reasons {
        non_empty(reason, "reason_topology_id")?;
        if !seen.insert(reason.clone()) {
            return Err(ReplayErrorV1::DuplicateReason {
                reason_id: reason.clone(),
            });
        }
    }
    Ok(seen.into_iter().collect())
}

fn canonical_policy(policy: &DispositionPolicyV1) -> Result<DispositionPolicyV1, ReplayErrorV1> {
    non_empty(&policy.profile_id, "policy_profile_id")?;
    non_empty(&policy.artifact_id, "policy_artifact_id")?;
    non_empty(&policy.fallback_output, "fallback_output")?;

    let mut rule_ids = BTreeSet::new();
    let mut priorities = BTreeSet::new();
    let mut rules = Vec::with_capacity(policy.rules.len());

    for rule in &policy.rules {
        non_empty(&rule.rule_id, "rule_id")?;
        non_empty(&rule.output, "rule_output")?;
        if !rule_ids.insert(rule.rule_id.clone()) {
            return Err(ReplayErrorV1::DuplicateRule {
                rule_id: rule.rule_id.clone(),
            });
        }
        if !priorities.insert(rule.priority) {
            return Err(ReplayErrorV1::DuplicatePriority {
                priority: rule.priority,
            });
        }

        let mut requirement_ids = BTreeSet::new();
        let mut requirements = rule.requirements.clone();
        requirements.sort_by_key(|requirement| requirement.predicate_id.clone());
        for requirement in &requirements {
            non_empty(&requirement.predicate_id, "requirement_predicate_id")?;
            if !requirement_ids.insert(requirement.predicate_id.clone()) {
                return Err(ReplayErrorV1::DuplicateRequirement {
                    rule_id: rule.rule_id.clone(),
                    predicate_id: requirement.predicate_id.clone(),
                });
            }
        }

        rules.push(DispositionRuleV1 {
            rule_id: rule.rule_id.clone(),
            priority: rule.priority,
            requirements,
            output: rule.output.clone(),
        });
    }
    rules.sort_by_key(|rule| rule.priority);

    Ok(DispositionPolicyV1 {
        profile_id: policy.profile_id.clone(),
        artifact_id: policy.artifact_id.clone(),
        rules,
        fallback_output: policy.fallback_output.clone(),
    })
}

/// Pure deterministic policy evaluation. The returned material is ordinary
/// data, not a qualified scientific witness.
pub fn evaluate_disposition_material_v1(
    inputs: &DispositionEvaluationInputsV1,
    policy: &DispositionPolicyV1,
) -> Result<DispositionEvaluationMaterialV1, ReplayErrorV1> {
    non_empty(&inputs.proposition_id, "proposition_id")?;
    non_empty(&inputs.scientific_use_id, "scientific_use_id")?;

    let context = canonical_context(&inputs.context)?;
    let (predicate_map, predicate_facts) = canonical_predicates(&inputs.predicates)?;
    let reason_topology_ids = canonical_reasons(&inputs.reason_topology_ids)?;
    let policy = canonical_policy(policy)?;
    let mut policy_trace = Vec::with_capacity(policy.rules.len());
    let mut selected = None;

    for rule in &policy.rules {
        let mut requirements = Vec::with_capacity(rule.requirements.len());
        let mut matched = true;
        for requirement in &rule.requirements {
            let observed = predicate_map.get(&requirement.predicate_id).copied();
            let satisfied = observed == Some(requirement.expected);
            matched &= satisfied;
            requirements.push(RequirementTraceV1 {
                predicate_id: requirement.predicate_id.clone(),
                expected: requirement.expected,
                observed,
                satisfied,
            });
        }
        if selected.is_none() && matched {
            selected = Some(rule.output.clone());
        }
        policy_trace.push(RuleTraceV1 {
            rule_id: rule.rule_id.clone(),
            priority: rule.priority,
            matched,
            requirements,
        });
    }

    let primary_disposition = selected.unwrap_or_else(|| policy.fallback_output.clone());
    Ok(DispositionEvaluationMaterialV1 {
        proposition_id: inputs.proposition_id.clone(),
        scientific_use_id: inputs.scientific_use_id.clone(),
        context,
        policy,
        reason_topology_ids,
        predicate_facts,
        primary_disposition,
        policy_trace,
    })
}

/// Deterministically build ordinary persisted evaluation data. Callers can
/// still forge or mutate the result, so this helper grants no positive replay
/// authority.
pub fn build_persisted_evaluation_record_v1(
    assessment_id: impl Into<String>,
    inputs: &DispositionEvaluationInputsV1,
    policy: &DispositionPolicyV1,
) -> Result<PersistedDispositionEvaluationRecordV1, ReplayErrorV1> {
    let assessment_id = assessment_id.into();
    non_empty(&assessment_id, "assessment_id")?;
    Ok(PersistedDispositionEvaluationRecordV1 {
        replay_profile_id: DISPOSITION_EVALUATION_REPLAY_PROFILE_V1.to_string(),
        assessment_id,
        material: evaluate_disposition_material_v1(inputs, policy)?,
    })
}

/// Verifier-owned positive runtime capability for **policy-evaluation replay**.
///
/// This is intentionally not the future full scientific-disposition replay
/// capability. It proves deterministic equality only for material understood
/// by this crate. Caller-shaped persisted record labels are not carried into
/// this capability.
#[derive(Debug, PartialEq, Eq)]
pub struct ReplayVerifiedDispositionEvaluationV1 {
    replayed_material: DispositionEvaluationMaterialV1,
}

impl ReplayVerifiedDispositionEvaluationV1 {
    pub fn material(&self) -> &DispositionEvaluationMaterialV1 {
        &self.replayed_material
    }

    pub fn context(&self) -> &DispositionEvaluationContextV1 {
        &self.replayed_material.context
    }

    pub fn primary_disposition(&self) -> &str {
        &self.replayed_material.primary_disposition
    }

    pub fn reason_topology_ids(&self) -> &[String] {
        &self.replayed_material.reason_topology_ids
    }

    pub fn policy_trace(&self) -> &[RuleTraceV1] {
        &self.replayed_material.policy_trace
    }
}

/// Recompute every evaluation field understood by this profile before
/// constructing the positive runtime capability. `assessment_id` is checked
/// only as persisted record structure; it is not replay-derived scientific
/// identity and is therefore not retained by the positive witness.
pub fn verify_persisted_disposition_evaluation_v1(
    record: &PersistedDispositionEvaluationRecordV1,
    inputs: &DispositionEvaluationInputsV1,
    policy: &DispositionPolicyV1,
) -> Result<ReplayVerifiedDispositionEvaluationV1, ReplayErrorV1> {
    if record.replay_profile_id != DISPOSITION_EVALUATION_REPLAY_PROFILE_V1 {
        return Err(ReplayErrorV1::ReplayProfileMismatch {
            found: record.replay_profile_id.clone(),
        });
    }
    non_empty(&record.assessment_id, "assessment_id")?;
    let replayed = evaluate_disposition_material_v1(inputs, policy)?;

    if record.material.proposition_id != replayed.proposition_id {
        return Err(ReplayErrorV1::PropositionMismatch);
    }
    if record.material.scientific_use_id != replayed.scientific_use_id {
        return Err(ReplayErrorV1::ScientificUseMismatch);
    }
    if record.material.context != replayed.context {
        return Err(ReplayErrorV1::EvaluationContextMismatch);
    }
    if record.material.policy != replayed.policy {
        return Err(ReplayErrorV1::PolicyMismatch);
    }
    if record.material.reason_topology_ids != replayed.reason_topology_ids {
        return Err(ReplayErrorV1::ReasonTopologyMismatch);
    }
    if record.material.predicate_facts != replayed.predicate_facts {
        return Err(ReplayErrorV1::PredicateSnapshotMismatch);
    }
    if record.material.primary_disposition != replayed.primary_disposition {
        return Err(ReplayErrorV1::PrimaryDispositionMismatch);
    }
    if record.material.policy_trace != replayed.policy_trace {
        return Err(ReplayErrorV1::PolicyTraceMismatch);
    }

    Ok(ReplayVerifiedDispositionEvaluationV1 {
        replayed_material: replayed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context() -> DispositionEvaluationContextV1 {
        DispositionEvaluationContextV1 {
            evidence_view_id: "view:fixture".into(),
            evidence_view_snapshot_id: "view-snapshot:17".into(),
            source_lifecycle_generation_id: "lifecycle:4".into(),
            external_argument_generation_id: "argument:9".into(),
            dependency_graph_generation_id: "dependency:6".into(),
            target_compatibility_generation_id: "compatibility:3".into(),
            triangulation_generation_id: "triangulation:8".into(),
            reason_topology_snapshot_id: "reason-topology:12".into(),
            predicate_derivation_profile_id: "predicate-profile:v1".into(),
            predicate_derivation_artifact_id: "predicate-artifact:fixture".into(),
            predicate_derivation_execution_lineage_id: "predicate-execution:fixture".into(),
            information_cutoff_id: "cutoff:2026-01-01".into(),
        }
    }

    fn policy() -> DispositionPolicyV1 {
        DispositionPolicyV1 {
            profile_id: "fixture-policy-v1".into(),
            artifact_id: "fixture-policy-artifact:1".into(),
            rules: vec![
                DispositionRuleV1 {
                    rule_id: "blocked".into(),
                    priority: 10,
                    requirements: vec![RuleRequirementV1 {
                        predicate_id: "active-defeater".into(),
                        expected: PredicateValueV1::Satisfied,
                    }],
                    output: "BlockedByQualifiedDefeater".into(),
                },
                DispositionRuleV1 {
                    rule_id: "contested".into(),
                    priority: 20,
                    requirements: vec![
                        RuleRequirementV1 {
                            predicate_id: "support-exists".into(),
                            expected: PredicateValueV1::Satisfied,
                        },
                        RuleRequirementV1 {
                            predicate_id: "opposition-exists".into(),
                            expected: PredicateValueV1::Satisfied,
                        },
                    ],
                    output: "Contested".into(),
                },
                DispositionRuleV1 {
                    rule_id: "supported".into(),
                    priority: 30,
                    requirements: vec![RuleRequirementV1 {
                        predicate_id: "support-exists".into(),
                        expected: PredicateValueV1::Satisfied,
                    }],
                    output: "SupportedWithinFixtureScope".into(),
                },
            ],
            fallback_output: "Underdetermined".into(),
        }
    }

    fn inputs() -> DispositionEvaluationInputsV1 {
        DispositionEvaluationInputsV1 {
            proposition_id: "prop:fixture:1".into(),
            scientific_use_id: "use:fixture-review".into(),
            context: context(),
            reason_topology_ids: vec!["reason:b".into(), "reason:a".into()],
            predicates: vec![
                PredicateFactV1 {
                    predicate_id: "opposition-exists".into(),
                    value: PredicateValueV1::NotSatisfied,
                },
                PredicateFactV1 {
                    predicate_id: "support-exists".into(),
                    value: PredicateValueV1::Satisfied,
                },
                PredicateFactV1 {
                    predicate_id: "active-defeater".into(),
                    value: PredicateValueV1::NotSatisfied,
                },
            ],
        }
    }

    #[test]
    fn serialized_record_requires_replay_to_gain_evaluation_witness() {
        let policy = policy();
        let inputs = inputs();
        let record =
            build_persisted_evaluation_record_v1("assessment:1", &inputs, &policy).unwrap();
        let bytes = serde_json::to_string(&record).unwrap();
        let decoded: PersistedDispositionEvaluationRecordV1 = serde_json::from_str(&bytes).unwrap();
        let verified =
            verify_persisted_disposition_evaluation_v1(&decoded, &inputs, &policy).unwrap();

        assert_eq!(
            verified.primary_disposition(),
            "SupportedWithinFixtureScope"
        );
        assert_eq!(verified.context(), &inputs.context);
    }

    #[test]
    fn caller_assessment_id_does_not_enter_verified_capability() {
        let policy = policy();
        let inputs = inputs();
        let first = build_persisted_evaluation_record_v1("assessment:a", &inputs, &policy).unwrap();
        let second =
            build_persisted_evaluation_record_v1("assessment:b", &inputs, &policy).unwrap();

        let first_verified =
            verify_persisted_disposition_evaluation_v1(&first, &inputs, &policy).unwrap();
        let second_verified =
            verify_persisted_disposition_evaluation_v1(&second, &inputs, &policy).unwrap();

        assert_ne!(first.assessment_id, second.assessment_id);
        assert_eq!(first_verified, second_verified);
    }

    #[test]
    fn forged_primary_disposition_is_rejected() {
        let policy = policy();
        let inputs = inputs();
        let mut record =
            build_persisted_evaluation_record_v1("assessment:1", &inputs, &policy).unwrap();
        record.material.primary_disposition = "RefutedWithinFixtureScope".into();
        assert_eq!(
            verify_persisted_disposition_evaluation_v1(&record, &inputs, &policy),
            Err(ReplayErrorV1::PrimaryDispositionMismatch)
        );
    }

    #[test]
    fn same_label_with_different_reason_topology_is_rejected() {
        let policy = policy();
        let original = inputs();
        let record =
            build_persisted_evaluation_record_v1("assessment:1", &original, &policy).unwrap();
        let mut changed = original;
        changed.reason_topology_ids = vec!["reason:replacement".into()];
        let material = evaluate_disposition_material_v1(&changed, &policy).unwrap();
        assert_eq!(
            material.primary_disposition,
            record.material.primary_disposition
        );
        assert_eq!(
            verify_persisted_disposition_evaluation_v1(&record, &changed, &policy),
            Err(ReplayErrorV1::ReasonTopologyMismatch)
        );
    }

    #[test]
    fn same_label_with_changed_predicate_snapshot_is_rejected() {
        let policy = policy();
        let original = inputs();
        let record =
            build_persisted_evaluation_record_v1("assessment:1", &original, &policy).unwrap();
        let mut changed = original;
        changed.predicates.push(PredicateFactV1 {
            predicate_id: "unused-but-material".into(),
            value: PredicateValueV1::Unknown,
        });
        let material = evaluate_disposition_material_v1(&changed, &policy).unwrap();
        assert_eq!(
            material.primary_disposition,
            record.material.primary_disposition
        );
        assert_eq!(
            verify_persisted_disposition_evaluation_v1(&record, &changed, &policy),
            Err(ReplayErrorV1::PredicateSnapshotMismatch)
        );
    }

    #[test]
    fn context_drift_is_material_even_when_disposition_is_unchanged() {
        let policy = policy();
        let original = inputs();
        let record =
            build_persisted_evaluation_record_v1("assessment:1", &original, &policy).unwrap();

        let mutations: [fn(&mut DispositionEvaluationContextV1); 8] = [
            |ctx| ctx.evidence_view_snapshot_id = "view-snapshot:18".into(),
            |ctx| ctx.source_lifecycle_generation_id = "lifecycle:5".into(),
            |ctx| ctx.dependency_graph_generation_id = "dependency:7".into(),
            |ctx| ctx.triangulation_generation_id = "triangulation:9".into(),
            |ctx| ctx.reason_topology_snapshot_id = "reason-topology:13".into(),
            |ctx| ctx.predicate_derivation_profile_id = "predicate-profile:v2".into(),
            |ctx| ctx.predicate_derivation_artifact_id = "predicate-artifact:replacement".into(),
            |ctx| ctx.information_cutoff_id = "cutoff:2027-01-01".into(),
        ];

        for mutate in mutations {
            let mut changed = original.clone();
            mutate(&mut changed.context);
            let material = evaluate_disposition_material_v1(&changed, &policy).unwrap();
            assert_eq!(
                material.primary_disposition,
                record.material.primary_disposition
            );
            assert_eq!(
                verify_persisted_disposition_evaluation_v1(&record, &changed, &policy),
                Err(ReplayErrorV1::EvaluationContextMismatch)
            );
        }
    }

    #[test]
    fn complete_policy_snapshot_blocks_unexecuted_semantic_drift() {
        let original_policy = policy();
        let inputs = inputs();
        let record =
            build_persisted_evaluation_record_v1("assessment:1", &inputs, &original_policy)
                .unwrap();
        let mut changed_policy = original_policy;
        changed_policy.rules[0].output = "DifferentBlockedDisposition".into();
        changed_policy.fallback_output = "DifferentFallback".into();
        let changed = evaluate_disposition_material_v1(&inputs, &changed_policy).unwrap();

        assert_eq!(
            changed.primary_disposition,
            record.material.primary_disposition
        );
        assert_eq!(changed.policy_trace, record.material.policy_trace);
        assert_eq!(
            verify_persisted_disposition_evaluation_v1(&record, &inputs, &changed_policy),
            Err(ReplayErrorV1::PolicyMismatch)
        );
    }

    #[test]
    fn policy_artifact_identity_is_material_even_when_semantics_match() {
        let original_policy = policy();
        let inputs = inputs();
        let record =
            build_persisted_evaluation_record_v1("assessment:1", &inputs, &original_policy)
                .unwrap();
        let mut changed_policy = original_policy;
        changed_policy.artifact_id = "fixture-policy-artifact:2".into();
        let changed = evaluate_disposition_material_v1(&inputs, &changed_policy).unwrap();

        assert_eq!(
            changed.primary_disposition,
            record.material.primary_disposition
        );
        assert_eq!(changed.policy_trace, record.material.policy_trace);
        assert_eq!(
            verify_persisted_disposition_evaluation_v1(&record, &inputs, &changed_policy),
            Err(ReplayErrorV1::PolicyMismatch)
        );
    }

    #[test]
    fn canonical_input_order_does_not_change_material_identity() {
        let policy = policy();
        let inputs = inputs();
        let canonical = evaluate_disposition_material_v1(&inputs, &policy).unwrap();

        let mut reordered_inputs = inputs;
        reordered_inputs.reason_topology_ids.reverse();
        reordered_inputs.predicates.reverse();
        let mut reordered_policy = policy;
        reordered_policy.rules.reverse();
        for rule in &mut reordered_policy.rules {
            rule.requirements.reverse();
        }

        let reordered =
            evaluate_disposition_material_v1(&reordered_inputs, &reordered_policy).unwrap();
        assert_eq!(canonical, reordered);
    }

    #[test]
    fn trace_keeps_lower_precedence_matches() {
        let policy = policy();
        let mut inputs = inputs();
        for fact in &mut inputs.predicates {
            if fact.predicate_id == "active-defeater" || fact.predicate_id == "opposition-exists" {
                fact.value = PredicateValueV1::Satisfied;
            }
        }
        let material = evaluate_disposition_material_v1(&inputs, &policy).unwrap();

        assert_eq!(material.primary_disposition, "BlockedByQualifiedDefeater");
        assert_eq!(material.policy_trace.len(), 3);
        assert!(material.policy_trace.iter().all(|trace| trace.matched));
    }

    #[test]
    fn duplicate_priority_is_not_resolved_by_container_order() {
        let mut policy = policy();
        policy.rules[1].priority = policy.rules[0].priority;
        assert_eq!(
            evaluate_disposition_material_v1(&inputs(), &policy),
            Err(ReplayErrorV1::DuplicatePriority { priority: 10 })
        );
    }

    #[test]
    fn duplicate_predicate_and_reason_identities_fail_closed() {
        let policy = policy();
        let mut duplicate_predicate = inputs();
        duplicate_predicate.predicates.push(PredicateFactV1 {
            predicate_id: "support-exists".into(),
            value: PredicateValueV1::Satisfied,
        });
        assert_eq!(
            evaluate_disposition_material_v1(&duplicate_predicate, &policy),
            Err(ReplayErrorV1::DuplicatePredicate {
                predicate_id: "support-exists".into()
            })
        );

        let mut duplicate_reason = inputs();
        duplicate_reason.reason_topology_ids.push("reason:a".into());
        assert_eq!(
            evaluate_disposition_material_v1(&duplicate_reason, &policy),
            Err(ReplayErrorV1::DuplicateReason {
                reason_id: "reason:a".into()
            })
        );
    }

    #[test]
    fn unknown_remains_explicit_and_does_not_satisfy_positive_requirement() {
        let policy = policy();
        let mut inputs = inputs();
        for fact in &mut inputs.predicates {
            if fact.predicate_id == "support-exists" {
                fact.value = PredicateValueV1::Unknown;
            }
        }
        let material = evaluate_disposition_material_v1(&inputs, &policy).unwrap();
        let supported = material
            .policy_trace
            .iter()
            .find(|trace| trace.rule_id == "supported")
            .unwrap();

        assert_eq!(material.primary_disposition, "Underdetermined");
        assert_eq!(
            supported.requirements[0].observed,
            Some(PredicateValueV1::Unknown)
        );
        assert!(!supported.requirements[0].satisfied);
    }

    #[test]
    fn missing_predicate_is_distinct_from_explicit_unknown() {
        let policy = policy();
        let mut missing = inputs();
        missing
            .predicates
            .retain(|fact| fact.predicate_id != "support-exists");
        let missing_material = evaluate_disposition_material_v1(&missing, &policy).unwrap();
        let missing_trace = missing_material
            .policy_trace
            .iter()
            .find(|trace| trace.rule_id == "supported")
            .unwrap();
        assert_eq!(missing_trace.requirements[0].observed, None);

        let mut unknown = inputs();
        for fact in &mut unknown.predicates {
            if fact.predicate_id == "support-exists" {
                fact.value = PredicateValueV1::Unknown;
            }
        }
        let unknown_material = evaluate_disposition_material_v1(&unknown, &policy).unwrap();
        let unknown_trace = unknown_material
            .policy_trace
            .iter()
            .find(|trace| trace.rule_id == "supported")
            .unwrap();
        assert_eq!(
            unknown_trace.requirements[0].observed,
            Some(PredicateValueV1::Unknown)
        );
    }

    #[test]
    fn policy_profile_change_is_material_even_with_identical_rules() {
        let original_policy = policy();
        let inputs = inputs();
        let record =
            build_persisted_evaluation_record_v1("assessment:1", &inputs, &original_policy)
                .unwrap();
        let mut changed_policy = original_policy;
        changed_policy.profile_id = "fixture-policy-v2".into();
        assert_eq!(
            verify_persisted_disposition_evaluation_v1(&record, &inputs, &changed_policy),
            Err(ReplayErrorV1::PolicyMismatch)
        );
    }

    #[test]
    fn forged_policy_trace_is_rejected_even_when_label_is_unchanged() {
        let policy = policy();
        let inputs = inputs();
        let mut record =
            build_persisted_evaluation_record_v1("assessment:1", &inputs, &policy).unwrap();
        record.material.policy_trace[0].matched = true;
        assert_eq!(
            verify_persisted_disposition_evaluation_v1(&record, &inputs, &policy),
            Err(ReplayErrorV1::PolicyTraceMismatch)
        );
    }

    #[test]
    fn empty_context_field_fails_closed() {
        let policy = policy();
        let mut inputs = inputs();
        inputs
            .context
            .predicate_derivation_execution_lineage_id
            .clear();
        assert_eq!(
            evaluate_disposition_material_v1(&inputs, &policy),
            Err(ReplayErrorV1::EmptyField {
                field: "predicate_derivation_execution_lineage_id"
            })
        );
    }

    #[test]
    fn replay_profile_mismatch_never_mints_witness() {
        let policy = policy();
        let inputs = inputs();
        let mut record =
            build_persisted_evaluation_record_v1("assessment:1", &inputs, &policy).unwrap();
        record.replay_profile_id = "symthaea.science.disposition-evaluation-replay.v2".into();
        assert_eq!(
            verify_persisted_disposition_evaluation_v1(&record, &inputs, &policy),
            Err(ReplayErrorV1::ReplayProfileMismatch {
                found: "symthaea.science.disposition-evaluation-replay.v2".into()
            })
        );
    }
}
