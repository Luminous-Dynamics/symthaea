// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Typed formal rules and versioned rule packs.
//!
//! This module is deliberately representational. It gives inference engines a
//! validated common language without silently choosing skeptical, credulous,
//! ambiguity-blocking, or ambiguity-propagating semantics.

use crate::model::{Literal, RuleId, RulePackId, SourceRef};
use crate::priority::{PriorityError, Superiority, SuperiorityGraph};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

/// Operational role of a formal rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RuleKind {
    /// An indefeasible implication within the selected formalization.
    Strict,
    /// A conclusion that may be defeated by contrary stronger support.
    Defeasible,
    /// A rule that can block contrary support but cannot establish its own
    /// conclusion as an independently supported legal result.
    Defeater,
}

/// One validated typed rule.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FormalRule {
    pub id: RuleId,
    pub kind: RuleKind,
    pub premises: Vec<Literal>,
    pub exceptions: Vec<Literal>,
    pub conclusion: Literal,
    pub source: Option<SourceRef>,
}

/// Invalid structure in a formal rule or rule pack.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RulePackError {
    ContradictoryPremises { rule: RuleId, literal: Literal },
    SelfBlockingException { rule: RuleId, literal: Literal },
    DuplicateRuleId { rule: RuleId },
    Priority(PriorityError),
}

impl fmt::Display for RulePackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RulePackError::ContradictoryPremises { rule, literal } => write!(
                f,
                "rule {rule} requires both a literal and its opposite: {literal:?}"
            ),
            RulePackError::SelfBlockingException { rule, literal } => write!(
                f,
                "rule {rule} contains the same literal as premise and exception: {literal:?}"
            ),
            RulePackError::DuplicateRuleId { rule } => {
                write!(f, "rule pack contains duplicate rule id {rule}")
            }
            RulePackError::Priority(error) => write!(f, "{error}"),
        }
    }
}

impl Error for RulePackError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            RulePackError::Priority(error) => Some(error),
            _ => None,
        }
    }
}

impl From<PriorityError> for RulePackError {
    fn from(value: PriorityError) -> Self {
        RulePackError::Priority(value)
    }
}

impl FormalRule {
    pub fn new(
        id: RuleId,
        kind: RuleKind,
        premises: impl IntoIterator<Item = Literal>,
        conclusion: Literal,
    ) -> Result<Self, RulePackError> {
        let premises = canonical_literals(premises);
        validate_premises(&id, &premises)?;
        Ok(Self {
            id,
            kind,
            premises,
            exceptions: Vec::new(),
            conclusion,
            source: None,
        })
    }

    pub fn with_exceptions(
        mut self,
        exceptions: impl IntoIterator<Item = Literal>,
    ) -> Result<Self, RulePackError> {
        let exceptions = canonical_literals(exceptions);
        for exception in &exceptions {
            if self.premises.contains(exception) {
                return Err(RulePackError::SelfBlockingException {
                    rule: self.id,
                    literal: exception.clone(),
                });
            }
        }
        self.exceptions = exceptions;
        Ok(self)
    }

    pub fn with_source(mut self, source: SourceRef) -> Self {
        self.source = Some(source);
        self
    }

    /// Whether every premise and no explicit exception is present.
    pub fn is_applicable(&self, facts: &BTreeSet<Literal>) -> bool {
        self.premises.iter().all(|premise| facts.contains(premise))
            && self
                .exceptions
                .iter()
                .all(|exception| !facts.contains(exception))
    }
}

fn canonical_literals(values: impl IntoIterator<Item = Literal>) -> Vec<Literal> {
    values.into_iter().collect::<BTreeSet<_>>().into_iter().collect()
}

fn validate_premises(rule: &RuleId, premises: &[Literal]) -> Result<(), RulePackError> {
    for literal in premises {
        if premises.binary_search(&literal.opposite()).is_ok() {
            return Err(RulePackError::ContradictoryPremises {
                rule: rule.clone(),
                literal: literal.clone(),
            });
        }
    }
    Ok(())
}

/// A versioned set of uniquely named rules and an explicit priority graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RulePack {
    pub id: RulePackId,
    rules: BTreeMap<RuleId, FormalRule>,
    priority: SuperiorityGraph,
}

impl RulePack {
    pub fn new(
        id: RulePackId,
        rules: impl IntoIterator<Item = FormalRule>,
        relations: impl IntoIterator<Item = Superiority>,
    ) -> Result<Self, RulePackError> {
        let mut indexed = BTreeMap::new();
        for rule in rules {
            let rule_id = rule.id.clone();
            if indexed.insert(rule_id.clone(), rule).is_some() {
                return Err(RulePackError::DuplicateRuleId { rule: rule_id });
            }
        }
        let priority = SuperiorityGraph::new(indexed.keys().cloned(), relations)?;
        Ok(Self {
            id,
            rules: indexed,
            priority,
        })
    }

    pub fn rule(&self, id: &RuleId) -> Option<&FormalRule> {
        self.rules.get(id)
    }

    pub fn rules(&self) -> impl Iterator<Item = &FormalRule> {
        self.rules.values()
    }

    pub fn rule_ids(&self) -> impl Iterator<Item = &RuleId> {
        self.rules.keys()
    }

    pub fn priority(&self) -> &SuperiorityGraph {
        &self.priority
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Atom;
    use crate::priority::PriorityBasis;

    fn positive(value: &str) -> Literal {
        Literal::Positive(Atom::new(value).unwrap())
    }

    fn negative(value: &str) -> Literal {
        Literal::Negative(Atom::new(value).unwrap())
    }

    fn id(value: &str) -> RuleId {
        RuleId::new(value).unwrap()
    }

    #[test]
    fn rule_canonicalizes_duplicate_premises_and_exceptions() {
        let rule = FormalRule::new(
            id("adult-capacity"),
            RuleKind::Defeasible,
            [positive("adult"), positive("adult")],
            positive("capacity"),
        )
        .unwrap()
        .with_exceptions([positive("incapacitated"), positive("incapacitated")])
        .unwrap();

        assert_eq!(rule.premises, vec![positive("adult")]);
        assert_eq!(rule.exceptions, vec![positive("incapacitated")]);
    }

    #[test]
    fn contradictory_and_self_blocking_rules_are_rejected() {
        assert!(matches!(
            FormalRule::new(
                id("impossible"),
                RuleKind::Strict,
                [positive("liable"), negative("liable")],
                positive("judgment")
            ),
            Err(RulePackError::ContradictoryPremises { .. })
        ));

        let result = FormalRule::new(
            id("self-blocked"),
            RuleKind::Defeasible,
            [positive("adult")],
            positive("capacity"),
        )
        .unwrap()
        .with_exceptions([positive("adult")]);
        assert!(matches!(
            result,
            Err(RulePackError::SelfBlockingException { .. })
        ));
    }

    #[test]
    fn pack_validates_unique_ids_and_priority_references() {
        let general = FormalRule::new(
            id("general"),
            RuleKind::Defeasible,
            [positive("resident")],
            positive("must-register"),
        )
        .unwrap();
        let specific = FormalRule::new(
            id("specific"),
            RuleKind::Defeasible,
            [positive("diplomat")],
            negative("must-register"),
        )
        .unwrap();
        let pack = RulePack::new(
            RulePackId::new("registration-v1").unwrap(),
            [general, specific],
            [Superiority::new(
                id("specific"),
                id("general"),
                PriorityBasis::MoreSpecific,
            )],
        )
        .unwrap();

        assert!(pack.priority().outranks(&id("specific"), &id("general")));
        assert_eq!(pack.rules().count(), 2);
    }
}
