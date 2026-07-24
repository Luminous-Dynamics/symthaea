// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Deterministic linting for formal rule packs.
//!
//! Validation findings never change legal outcomes. They surface structural
//! risks for review before a pack is approved for a semantic profile.

use crate::rules::{FormalRule, RuleKind, RulePack};

/// Review significance of a validation finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Severity {
    Information,
    Warning,
    Error,
}

/// One stable machine-readable validation finding.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ValidationIssue {
    pub severity: Severity,
    pub code: &'static str,
    pub path: String,
    pub message: String,
}

/// Canonically ordered report for a rule pack.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ValidationReport {
    pub issues: Vec<ValidationIssue>,
}

impl ValidationReport {
    pub fn has_errors(&self) -> bool {
        self.issues
            .iter()
            .any(|issue| issue.severity == Severity::Error)
    }

    pub fn warnings(&self) -> impl Iterator<Item = &ValidationIssue> {
        self.issues
            .iter()
            .filter(|issue| issue.severity == Severity::Warning)
    }
}

/// Audit a validated rule pack for higher-level semantic hazards.
pub fn validate_rule_pack(pack: &RulePack) -> ValidationReport {
    let rules: Vec<&FormalRule> = pack.rules().collect();
    let mut issues = Vec::new();

    for rule in &rules {
        let path = format!("rules/{}", rule.id);
        if rule.source.is_none() {
            issues.push(issue(
                Severity::Warning,
                "LR001_MISSING_SOURCE",
                &path,
                "formal rule has no source provision binding",
            ));
        }
        if rule.premises.is_empty() && rule.exceptions.is_empty() {
            issues.push(issue(
                Severity::Information,
                "LR002_UNCONDITIONAL_RULE",
                &path,
                "rule is unconditional; confirm that this is intentional",
            ));
        }
        if rule.exceptions.contains(&rule.conclusion) {
            issues.push(issue(
                Severity::Warning,
                "LR003_SELF_EXCEPTION",
                &path,
                "rule conclusion is also an exception and may make the rule unstable in recursive engines",
            ));
        }
        if rule.kind == RuleKind::Defeater
            && !pack
                .priority()
                .relations()
                .iter()
                .any(|relation| relation.stronger == rule.id)
        {
            issues.push(issue(
                Severity::Warning,
                "LR004_INERT_DEFEATER",
                &path,
                "defeater has no outgoing superiority edge and cannot defeat an opposing non-strict rule",
            ));
        }
    }

    for left_index in 0..rules.len() {
        for right_index in (left_index + 1)..rules.len() {
            let left = rules[left_index];
            let right = rules[right_index];
            if same_body(left, right) && left.conclusion == right.conclusion {
                issues.push(issue(
                    Severity::Warning,
                    "LR005_DUPLICATE_RULE_BODY",
                    &format!("rules/{},rules/{}", left.id, right.id),
                    "distinct rule ids have identical kind, body, exceptions, and conclusion",
                ));
            }
            if left.kind == RuleKind::Strict
                && right.kind == RuleKind::Strict
                && left.premises == right.premises
                && left.exceptions == right.exceptions
                && left.conclusion == right.conclusion.opposite()
            {
                issues.push(issue(
                    Severity::Error,
                    "LR006_CONTRADICTORY_STRICT_RULES",
                    &format!("rules/{},rules/{}", left.id, right.id),
                    "strict rules with the same body establish opposite conclusions",
                ));
            }
        }
    }

    issues.sort_unstable();
    ValidationReport { issues }
}

fn same_body(left: &FormalRule, right: &FormalRule) -> bool {
    left.kind == right.kind
        && left.premises == right.premises
        && left.exceptions == right.exceptions
        && left.conclusion == right.conclusion
}

fn issue(
    severity: Severity,
    code: &'static str,
    path: &str,
    message: &str,
) -> ValidationIssue {
    ValidationIssue {
        severity,
        code,
        path: path.to_string(),
        message: message.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Atom, Literal, RuleId, RulePackId};
    use crate::rules::{FormalRule, RuleKind, RulePack};

    fn literal(value: &str) -> Literal {
        Literal::Positive(Atom::new(value).unwrap())
    }

    fn rule(id: &str, kind: RuleKind, conclusion: Literal) -> FormalRule {
        FormalRule::new(
            RuleId::new(id).unwrap(),
            kind,
            [literal("condition")],
            conclusion,
        )
        .unwrap()
    }

    #[test]
    fn contradictory_strict_rules_are_errors() {
        let positive = literal("liable");
        let negative = positive.opposite();
        let pack = RulePack::new(
            RulePackId::new("strict-conflict").unwrap(),
            [
                rule("strict-positive", RuleKind::Strict, positive),
                rule("strict-negative", RuleKind::Strict, negative),
            ],
            [],
        )
        .unwrap();
        let report = validate_rule_pack(&pack);

        assert!(report.has_errors());
        assert!(report
            .issues
            .iter()
            .any(|issue| issue.code == "LR006_CONTRADICTORY_STRICT_RULES"));
    }

    #[test]
    fn missing_sources_and_inert_defeaters_are_warnings() {
        let pack = RulePack::new(
            RulePackId::new("warnings").unwrap(),
            [rule("blocker", RuleKind::Defeater, literal("enter"))],
            [],
        )
        .unwrap();
        let report = validate_rule_pack(&pack);
        let codes: Vec<&str> = report.warnings().map(|issue| issue.code).collect();

        assert!(codes.contains(&"LR001_MISSING_SOURCE"));
        assert!(codes.contains(&"LR004_INERT_DEFEATER"));
    }

    #[test]
    fn report_order_is_independent_of_rule_insertion() {
        let left = rule("left", RuleKind::Defeasible, literal("enter"));
        let right = rule("right", RuleKind::Defeasible, literal("enter"));
        let forward = RulePack::new(
            RulePackId::new("forward").unwrap(),
            [left.clone(), right.clone()],
            [],
        )
        .unwrap();
        let reverse = RulePack::new(
            RulePackId::new("reverse").unwrap(),
            [right, left],
            [],
        )
        .unwrap();

        assert_eq!(
            validate_rule_pack(&forward).issues,
            validate_rule_pack(&reverse).issues
        );
    }
}
