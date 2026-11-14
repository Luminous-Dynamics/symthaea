//! Semantic guardrails for personal garment-design briefs.
//!
//! This crate deliberately does not parse JSON, generate patterns, estimate
//! physical material properties, simulate garments, or authorize production.
//! It validates decoded domain values and provides a fail-closed handoff gate.

#![forbid(unsafe_code)]

pub mod preset_stack;

use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HardConstraint {
    pub id: String,
    pub field: String,
    pub rule: ConstraintRule,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ConstraintRule {
    Equals(String),
    NotEquals(String),
    Minimum { value: i64, unit: String },
    Maximum { value: i64, unit: String },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ObjectiveWeight {
    pub id: String,
    /// Relative preference weight in basis points, 0..=10_000.
    /// Weights need not sum to 10_000; downstream rankers normalize them.
    pub weight_bps: u16,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SuggestionDisposition {
    Accepted,
    Rejected,
    Edited,
    Deferred,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SuggestionDecision {
    pub id: String,
    pub source_brief_revision: u64,
    pub source_brief_sha256: String,
    pub disposition: SuggestionDisposition,
    pub actor_id: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DesignBrief {
    pub id: String,
    pub revision: u64,
    pub sha256: String,
    pub hard_constraints: Vec<HardConstraint>,
    pub objectives: Vec<ObjectiveWeight>,
    pub suggestion_decisions: Vec<SuggestionDecision>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ValidationIssue {
    pub code: &'static str,
    pub path: String,
    pub detail: String,
}

impl ValidationIssue {
    fn new(code: &'static str, path: impl Into<String>, detail: impl Into<String>) -> Self {
        Self { code, path: path.into(), detail: detail.into() }
    }
}

/// Empty output means structural/semantic checks passed only. It does not
/// establish fit, comfort, safety, material performance or production readiness.
pub fn validate_brief(brief: &DesignBrief) -> Vec<ValidationIssue> {
    let mut issues = Vec::new();

    if brief.id.trim().is_empty() {
        issues.push(ValidationIssue::new("empty_brief_id", "id", "brief id must be non-empty"));
    }
    if brief.revision == 0 {
        issues.push(ValidationIssue::new("zero_revision", "revision", "revision must start at 1"));
    }
    if !is_sha256_hex(&brief.sha256) {
        issues.push(ValidationIssue::new(
            "invalid_brief_digest", "sha256", "expected exactly 64 hexadecimal characters",
        ));
    }

    check_unique_ids(
        brief.hard_constraints.iter().map(|v| v.id.as_str()),
        "hard_constraints",
        &mut issues,
    );
    check_unique_ids(brief.objectives.iter().map(|v| v.id.as_str()), "objectives", &mut issues);
    check_unique_ids(
        brief.suggestion_decisions.iter().map(|v| v.id.as_str()),
        "suggestion_decisions",
        &mut issues,
    );

    for (index, constraint) in brief.hard_constraints.iter().enumerate() {
        if constraint.id.trim().is_empty() {
            issues.push(ValidationIssue::new(
                "empty_constraint_id",
                format!("hard_constraints[{index}].id"),
                "constraint id must be non-empty",
            ));
        }
        if constraint.field.trim().is_empty() {
            issues.push(ValidationIssue::new(
                "empty_constraint_field",
                format!("hard_constraints[{index}].field"),
                "field must be non-empty",
            ));
        }
        if let ConstraintRule::Minimum { unit, .. }
        | ConstraintRule::Maximum { unit, .. } = &constraint.rule
        {
            if unit.trim().is_empty() {
                issues.push(ValidationIssue::new(
                    "empty_constraint_unit", format!("hard_constraints[{index}].rule"),
                    "numeric constraints require an explicit unit",
                ));
            }
        }
    }

    for (index, objective) in brief.objectives.iter().enumerate() {
        if objective.id.trim().is_empty() {
            issues.push(ValidationIssue::new(
                "empty_objective_id",
                format!("objectives[{index}].id"),
                "objective id must be non-empty",
            ));
        }
        if objective.weight_bps > 10_000 {
            issues.push(ValidationIssue::new(
                "objective_weight_out_of_range", format!("objectives[{index}].weight_bps"),
                "weight must be between 0 and 10,000 basis points",
            ));
        }
    }
    if brief.objectives.iter().all(|v| v.weight_bps == 0) {
        issues.push(ValidationIssue::new(
            "no_active_objective",
            "objectives",
            "at least one objective must have a positive weight",
        ));
    }

    validate_constraint_consistency(&brief.hard_constraints, &mut issues);

    for (index, decision) in brief.suggestion_decisions.iter().enumerate() {
        let path = format!("suggestion_decisions[{index}]");
        if decision.id.trim().is_empty() {
            issues.push(ValidationIssue::new(
                "empty_decision_id", format!("{path}.id"), "decision id must be non-empty",
            ));
        }
        if decision.source_brief_revision != brief.revision
            || !decision.source_brief_sha256.eq_ignore_ascii_case(&brief.sha256)
        {
            issues.push(ValidationIssue::new(
                "stale_suggestion_decision", path.clone(),
                "decision must bind to this exact brief revision and digest",
            ));
        }
        if decision.disposition != SuggestionDisposition::Deferred
            && decision.actor_id.as_deref().unwrap_or_default().trim().is_empty()
        {
            issues.push(ValidationIssue::new(
                "missing_decision_actor", format!("{path}.actor_id"),
                "accepted, rejected, and edited suggestions require an actor reference",
            ));
        }
    }

    issues
}

fn check_unique_ids<'a>(
    ids: impl Iterator<Item = &'a str>,
    path: &str,
    issues: &mut Vec<ValidationIssue>,
) {
    let mut seen = BTreeSet::new();
    for (index, id) in ids.enumerate() {
        if !seen.insert(id.to_owned()) {
            issues.push(ValidationIssue::new(
                "duplicate_id", format!("{path}[{index}].id"), format!("duplicate id `{id}`"),
            ));
        }
    }
}

fn validate_constraint_consistency(
    constraints: &[HardConstraint],
    issues: &mut Vec<ValidationIssue>,
) {
    let mut by_field: BTreeMap<&str, Vec<&ConstraintRule>> = BTreeMap::new();
    for constraint in constraints {
        by_field.entry(constraint.field.as_str()).or_default().push(&constraint.rule);
    }

    for (field, rules) in by_field {
        let mut required = BTreeSet::new();
        let mut forbidden = BTreeSet::new();
        let mut mins: BTreeMap<&str, i64> = BTreeMap::new();
        let mut maxs: BTreeMap<&str, i64> = BTreeMap::new();
        let mut units = BTreeSet::new();

        for rule in &rules {
            match rule {
                ConstraintRule::Equals(value) => { required.insert(value.as_str()); }
                ConstraintRule::NotEquals(value) => { forbidden.insert(value.as_str()); }
                ConstraintRule::Minimum { value, unit } => {
                    units.insert(unit.as_str());
                    mins.entry(unit.as_str())
                        .and_modify(|current| *current = (*current).max(*value))
                        .or_insert(*value);
                }
                ConstraintRule::Maximum { value, unit } => {
                    units.insert(unit.as_str());
                    maxs.entry(unit.as_str())
                        .and_modify(|current| *current = (*current).min(*value))
                        .or_insert(*value);
                }
            }
        }

        if required.len() > 1 {
            issues.push(ValidationIssue::new(
                "conflicting_required_values", format!("hard_constraints[{field}]"),
                "the same field has more than one required categorical value",
            ));
        }
        if required.iter().any(|value| forbidden.contains(value)) {
            issues.push(ValidationIssue::new(
                "required_value_forbidden", format!("hard_constraints[{field}]"),
                "a value is both required and forbidden",
            ));
        }
        if units.len() > 1 {
            issues.push(ValidationIssue::new(
                "incompatible_constraint_units", format!("hard_constraints[{field}]"),
                "numeric bounds on the same field use different units",
            ));
        }
        for (unit, min_value) in &mins {
            if let Some(max_value) = maxs.get(unit) {
                if min_value > max_value {
                    issues.push(ValidationIssue::new(
                        "contradictory_numeric_bounds", format!("hard_constraints[{field}]"),
                        format!("minimum {min_value} {unit} exceeds maximum {max_value} {unit}"),
                    ));
                }
            }
        }
    }
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.chars().all(|character| character.is_ascii_hexdigit())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GateStatus { Pass, Fail, Unknown }

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConstraintObservation {
    pub constraint_id: String,
    pub status: GateStatus,
    /// Reference only. The owning subsystem must independently verify the artifact.
    pub evidence_ref: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReleaseDisposition {
    ReadyForEngineeringHandoff,
    Blocked { failed_constraint_ids: Vec<String> },
    Unresolved { constraint_ids: Vec<String> },
    Invalid { issues: Vec<ValidationIssue> },
}

/// This is not production authorization, fit approval or physical validation.
/// Missing/unknown gates do not pass. A passing gate needs an evidence reference.
pub fn evaluate_handoff(
    brief: &DesignBrief,
    observations: &[ConstraintObservation],
) -> ReleaseDisposition {
    let issues = validate_brief(brief);
    if !issues.is_empty() { return ReleaseDisposition::Invalid { issues }; }

    let known: BTreeSet<&str> = brief.hard_constraints.iter().map(|v| v.id.as_str()).collect();
    let mut seen = BTreeSet::new();
    let mut invalid = Vec::new();

    for (index, observation) in observations.iter().enumerate() {
        let id = observation.constraint_id.as_str();
        if !known.contains(id) {
            invalid.push(ValidationIssue::new(
                "observation_for_unknown_constraint",
                format!("observations[{index}].constraint_id"),
                format!("`{id}` is not declared by this brief"),
            ));
        } else if !seen.insert(id) {
            invalid.push(ValidationIssue::new(
                "duplicate_constraint_observation", format!("observations[{index}].constraint_id"),
                format!("constraint `{id}` has more than one observation"),
            ));
        }
        if observation.status == GateStatus::Pass
            && observation.evidence_ref.as_deref().unwrap_or_default().trim().is_empty()
        {
            invalid.push(ValidationIssue::new(
                "pass_without_evidence_reference", format!("observations[{index}].evidence_ref"),
                "a passing hard gate requires an evidence reference",
            ));
        }
    }
    if !invalid.is_empty() { return ReleaseDisposition::Invalid { issues: invalid }; }

    let mut failed = Vec::new();
    let mut unresolved = Vec::new();
    for constraint in &brief.hard_constraints {
        match observations.iter().find(|v| v.constraint_id == constraint.id).map(|v| v.status) {
            Some(GateStatus::Pass) => {}
            Some(GateStatus::Fail) => failed.push(constraint.id.clone()),
            Some(GateStatus::Unknown) | None => unresolved.push(constraint.id.clone()),
        }
    }

    if !failed.is_empty() {
        ReleaseDisposition::Blocked { failed_constraint_ids: failed }
    } else if !unresolved.is_empty() {
        ReleaseDisposition::Unresolved { constraint_ids: unresolved }
    } else {
        ReleaseDisposition::ReadyForEngineeringHandoff
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_brief() -> DesignBrief {
        DesignBrief {
            id: "brief-001".to_owned(),
            revision: 1,
            sha256: "a".repeat(64),
            hard_constraints: vec![
                HardConstraint {
                    id: "closure-no-metal".to_owned(),
                    field: "closure.material".to_owned(),
                    rule: ConstraintRule::NotEquals("metal".to_owned()),
                },
                HardConstraint {
                    id: "price-cap".to_owned(),
                    field: "price.cents".to_owned(),
                    rule: ConstraintRule::Maximum { value: 8000, unit: "USD-cent".to_owned() },
                },
            ],
            objectives: vec![
                ObjectiveWeight { id: "comfort".to_owned(), weight_bps: 7000 },
                ObjectiveWeight { id: "repairability".to_owned(), weight_bps: 3000 },
            ],
            suggestion_decisions: Vec::new(),
        }
    }

    #[test]
    fn valid_brief_has_no_semantic_issues() {
        assert!(validate_brief(&valid_brief()).is_empty());
    }

    #[test]
    fn contradictory_numeric_bounds_are_rejected() {
        let mut brief = valid_brief();
        brief.hard_constraints.push(HardConstraint {
            id: "price-floor".to_owned(),
            field: "price.cents".to_owned(),
            rule: ConstraintRule::Minimum { value: 9000, unit: "USD-cent".to_owned() },
        });
        assert!(validate_brief(&brief).iter().any(|v| v.code == "contradictory_numeric_bounds"));
    }

    #[test]
    fn required_value_cannot_also_be_forbidden() {
        let mut brief = valid_brief();
        brief.hard_constraints.push(HardConstraint {
            id: "closure-require-metal".to_owned(),
            field: "closure.material".to_owned(),
            rule: ConstraintRule::Equals("metal".to_owned()),
        });
        assert!(validate_brief(&brief).iter().any(|v| v.code == "required_value_forbidden"));
    }

    #[test]
    fn stale_suggestion_decision_cannot_be_reused_after_revision() {
        let mut brief = valid_brief();
        brief.revision = 2;
        brief.suggestion_decisions.push(SuggestionDecision {
            id: "decision-1".to_owned(),
            source_brief_revision: 1,
            source_brief_sha256: "a".repeat(64),
            disposition: SuggestionDisposition::Accepted,
            actor_id: Some("user-local-1".to_owned()),
        });
        assert!(validate_brief(&brief).iter().any(|v| v.code == "stale_suggestion_decision"));
    }

    #[test]
    fn accepted_suggestion_requires_actor_reference() {
        let mut brief = valid_brief();
        brief.suggestion_decisions.push(SuggestionDecision {
            id: "decision-1".to_owned(),
            source_brief_revision: 1,
            source_brief_sha256: "a".repeat(64),
            disposition: SuggestionDisposition::Accepted,
            actor_id: None,
        });
        assert!(validate_brief(&brief).iter().any(|v| v.code == "missing_decision_actor"));
    }

    #[test]
    fn unknown_hard_gate_is_unresolved() {
        let brief = valid_brief();
        let observations = vec![ConstraintObservation {
            constraint_id: "closure-no-metal".to_owned(),
            status: GateStatus::Pass,
            evidence_ref: Some("fixture:closure".to_owned()),
        }];
        assert_eq!(evaluate_handoff(&brief, &observations), ReleaseDisposition::Unresolved {
            constraint_ids: vec!["price-cap".to_owned()],
        });
    }

    #[test]
    fn failed_hard_gate_blocks_handoff() {
        let brief = valid_brief();
        let observations = vec![
            ConstraintObservation {
                constraint_id: "closure-no-metal".to_owned(),
                status: GateStatus::Fail,
                evidence_ref: Some("fixture:failure".to_owned()),
            },
            ConstraintObservation {
                constraint_id: "price-cap".to_owned(),
                status: GateStatus::Pass,
                evidence_ref: Some("fixture:pass".to_owned()),
            },
        ];
        assert_eq!(evaluate_handoff(&brief, &observations), ReleaseDisposition::Blocked {
            failed_constraint_ids: vec!["closure-no-metal".to_owned()],
        });
    }

    #[test]
    fn every_hard_gate_must_pass_for_handoff() {
        let brief = valid_brief();
        let observations = vec![
            ConstraintObservation {
                constraint_id: "closure-no-metal".to_owned(),
                status: GateStatus::Pass,
                evidence_ref: Some("fixture:pass-1".to_owned()),
            },
            ConstraintObservation {
                constraint_id: "price-cap".to_owned(),
                status: GateStatus::Pass,
                evidence_ref: Some("fixture:pass-2".to_owned()),
            },
        ];
        assert_eq!(
            evaluate_handoff(&brief, &observations),
            ReleaseDisposition::ReadyForEngineeringHandoff
        );
    }

    #[test]
    fn duplicate_or_unknown_observation_ids_are_invalid() {
        let brief = valid_brief();
        let observations = vec![
            ConstraintObservation {
                constraint_id: "closure-no-metal".to_owned(),
                status: GateStatus::Pass,
                evidence_ref: Some("fixture:pass".to_owned()),
            },
            ConstraintObservation {
                constraint_id: "closure-no-metal".to_owned(),
                status: GateStatus::Pass,
                evidence_ref: Some("fixture:duplicate".to_owned()),
            },
        ];
        assert!(matches!(
            evaluate_handoff(&brief, &observations),
            ReleaseDisposition::Invalid { .. }
        ));
    }

    #[test]
    fn a_passing_gate_requires_an_evidence_reference() {
        let brief = valid_brief();
        let observations = vec![
            ConstraintObservation {
                constraint_id: "closure-no-metal".to_owned(),
                status: GateStatus::Pass,
                evidence_ref: None,
            },
            ConstraintObservation {
                constraint_id: "price-cap".to_owned(),
                status: GateStatus::Pass,
                evidence_ref: Some("fixture:price".to_owned()),
            },
        ];
        assert!(matches!(
            evaluate_handoff(&brief, &observations),
            ReleaseDisposition::Invalid { .. }
        ));
    }

    #[test]
    fn zero_weight_objective_set_is_rejected() {
        let mut brief = valid_brief();
        for objective in &mut brief.objectives { objective.weight_bps = 0; }
        assert!(validate_brief(&brief).iter().any(|v| v.code == "no_active_objective"));
    }
}
