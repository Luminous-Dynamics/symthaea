// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Skeptical conflict resolution for already-applicable typed rules.
//!
//! This module resolves direct support for a literal and its explicit opposite.
//! It is intentionally not a recursive inference engine: callers first choose
//! or derive a fact set, then ask which applicable rules survive opposition.

use crate::model::{Literal, RuleId};
use crate::rules::{FormalRule, RuleKind, RulePack};
use std::collections::{BTreeMap, BTreeSet};

/// Query-facing four-valued result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LegalStatus {
    Supported,
    Refuted,
    Both,
    Undetermined,
}

/// Why one applicable rule was defeated by another.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DefeatBasis {
    StrictOverDefeasible,
    ExplicitPriority,
}

/// One deterministic defeat relation found during conflict resolution.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RuleDefeat {
    pub winner: RuleId,
    pub loser: RuleId,
    pub basis: DefeatBasis,
}

/// Complete direct-resolution evidence for one queried literal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiteralResolution {
    pub query: Literal,
    pub status: LegalStatus,
    pub undefeated_support: Vec<RuleId>,
    pub undefeated_opposition: Vec<RuleId>,
    pub blocking_defeaters: Vec<RuleId>,
    pub defeats: Vec<RuleDefeat>,
}

impl LiteralResolution {
    pub fn is_skeptically_supported(&self) -> bool {
        self.status == LegalStatus::Supported
    }
}

/// Resolve direct, applicable support for `query` under the rule pack's
/// superiority graph.
///
/// Strict support defeats contrary non-strict support and cannot itself be
/// defeated by a non-strict rule. Among non-strict rules, defeat requires an
/// explicit path in the superiority graph. Defeaters can defeat contrary rules
/// but never establish their own conclusion.
pub fn resolve_literal(
    pack: &RulePack,
    facts: &BTreeSet<Literal>,
    query: &Literal,
) -> LiteralResolution {
    let opposite = query.opposite();
    let positive: Vec<&FormalRule> = pack
        .rules()
        .filter(|rule| &rule.conclusion == query && rule.is_applicable(facts))
        .collect();
    let negative: Vec<&FormalRule> = pack
        .rules()
        .filter(|rule| rule.conclusion == opposite && rule.is_applicable(facts))
        .collect();

    let mut memo = BTreeMap::new();
    let mut active = BTreeSet::new();
    let defeated_positive: BTreeSet<RuleId> = positive
        .iter()
        .filter(|rule| {
            is_defeated(
                rule,
                Side::Positive,
                &positive,
                &negative,
                pack,
                &mut memo,
                &mut active,
            )
        })
        .map(|rule| rule.id.clone())
        .collect();
    let defeated_negative: BTreeSet<RuleId> = negative
        .iter()
        .filter(|rule| {
            is_defeated(
                rule,
                Side::Negative,
                &positive,
                &negative,
                pack,
                &mut memo,
                &mut active,
            )
        })
        .map(|rule| rule.id.clone())
        .collect();

    let undefeated_support: Vec<RuleId> = positive
        .iter()
        .filter(|rule| rule.kind != RuleKind::Defeater && !defeated_positive.contains(&rule.id))
        .map(|rule| rule.id.clone())
        .collect();
    let undefeated_opposition: Vec<RuleId> = negative
        .iter()
        .filter(|rule| rule.kind != RuleKind::Defeater && !defeated_negative.contains(&rule.id))
        .map(|rule| rule.id.clone())
        .collect();

    let mut defeats = BTreeSet::new();
    for attacker in positive
        .iter()
        .filter(|rule| !defeated_positive.contains(&rule.id))
    {
        for defender in negative
            .iter()
            .filter(|rule| defeated_negative.contains(&rule.id))
        {
            if let Some(basis) = defeats_rule(attacker, defender, pack) {
                defeats.insert(RuleDefeat {
                    winner: attacker.id.clone(),
                    loser: defender.id.clone(),
                    basis,
                });
            }
        }
    }
    for attacker in negative
        .iter()
        .filter(|rule| !defeated_negative.contains(&rule.id))
    {
        for defender in positive
            .iter()
            .filter(|rule| defeated_positive.contains(&rule.id))
        {
            if let Some(basis) = defeats_rule(attacker, defender, pack) {
                defeats.insert(RuleDefeat {
                    winner: attacker.id.clone(),
                    loser: defender.id.clone(),
                    basis,
                });
            }
        }
    }

    let blocking_defeaters: Vec<RuleId> = positive
        .iter()
        .chain(negative.iter())
        .filter(|rule| rule.kind == RuleKind::Defeater)
        .filter(|rule| {
            !defeated_positive.contains(&rule.id) && !defeated_negative.contains(&rule.id)
        })
        .filter(|rule| defeats.iter().any(|defeat| defeat.winner == rule.id))
        .map(|rule| rule.id.clone())
        .collect();

    let status = match (
        undefeated_support.is_empty(),
        undefeated_opposition.is_empty(),
    ) {
        (false, true) => LegalStatus::Supported,
        (true, false) => LegalStatus::Refuted,
        (false, false) => LegalStatus::Both,
        (true, true) => LegalStatus::Undetermined,
    };

    LiteralResolution {
        query: query.clone(),
        status,
        undefeated_support,
        undefeated_opposition,
        blocking_defeaters,
        defeats: defeats.into_iter().collect(),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Side {
    Positive,
    Negative,
}

impl Side {
    fn opposite(self) -> Self {
        match self {
            Side::Positive => Side::Negative,
            Side::Negative => Side::Positive,
        }
    }
}

fn is_defeated(
    rule: &FormalRule,
    side: Side,
    positive: &[&FormalRule],
    negative: &[&FormalRule],
    pack: &RulePack,
    memo: &mut BTreeMap<(RuleId, Side), bool>,
    active: &mut BTreeSet<(RuleId, Side)>,
) -> bool {
    let key = (rule.id.clone(), side);
    if let Some(result) = memo.get(&key) {
        return *result;
    }
    if !active.insert(key.clone()) {
        // A validated acyclic superiority graph and strictness ordering should
        // make this unreachable. Failing open here preserves unresolved support
        // rather than manufacturing a defeat from an unexpected attack cycle.
        return false;
    }

    let attackers = match side {
        Side::Positive => negative,
        Side::Negative => positive,
    };
    let defeated = attackers.iter().any(|attacker| {
        defeats_rule(attacker, rule, pack).is_some()
            && !is_defeated(
                attacker,
                side.opposite(),
                positive,
                negative,
                pack,
                memo,
                active,
            )
    });

    active.remove(&key);
    memo.insert(key, defeated);
    defeated
}

fn defeats_rule(
    attacker: &FormalRule,
    defender: &FormalRule,
    pack: &RulePack,
) -> Option<DefeatBasis> {
    if defender.kind == RuleKind::Strict {
        return None;
    }
    if attacker.kind == RuleKind::Strict {
        return Some(DefeatBasis::StrictOverDefeasible);
    }
    pack.priority()
        .outranks(&attacker.id, &defender.id)
        .then_some(DefeatBasis::ExplicitPriority)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Atom, RulePackId};
    use crate::priority::{PriorityBasis, Superiority};
    use crate::rules::{FormalRule, RuleKind, RulePack};

    fn positive(value: &str) -> Literal {
        Literal::Positive(Atom::new(value).unwrap())
    }

    fn negative(value: &str) -> Literal {
        Literal::Negative(Atom::new(value).unwrap())
    }

    fn rule(id: &str, kind: RuleKind, premise: &str, conclusion: Literal) -> FormalRule {
        FormalRule::new(
            RuleId::new(id).unwrap(),
            kind,
            [positive(premise)],
            conclusion,
        )
        .unwrap()
    }

    #[test]
    fn specific_priority_resolves_direct_conflict() {
        let general = rule(
            "general-registration",
            RuleKind::Defeasible,
            "resident",
            positive("must-register"),
        );
        let diplomat = rule(
            "diplomat-exemption",
            RuleKind::Defeasible,
            "resident",
            negative("must-register"),
        );
        let pack = RulePack::new(
            RulePackId::new("registration").unwrap(),
            [general, diplomat],
            [Superiority::new(
                RuleId::new("diplomat-exemption").unwrap(),
                RuleId::new("general-registration").unwrap(),
                PriorityBasis::MoreSpecific,
            )],
        )
        .unwrap();
        let facts = [positive("resident")].into_iter().collect();
        let result = resolve_literal(&pack, &facts, &positive("must-register"));

        assert_eq!(result.status, LegalStatus::Refuted);
        assert_eq!(
            result.undefeated_opposition,
            vec![RuleId::new("diplomat-exemption").unwrap()]
        );
    }

    #[test]
    fn incomparable_support_is_reported_as_both() {
        let allow = rule(
            "allow",
            RuleKind::Defeasible,
            "condition",
            positive("enter"),
        );
        let deny = rule("deny", RuleKind::Defeasible, "condition", negative("enter"));
        let pack = RulePack::new(RulePackId::new("entry").unwrap(), [allow, deny], []).unwrap();
        let facts = [positive("condition")].into_iter().collect();

        assert_eq!(
            resolve_literal(&pack, &facts, &positive("enter")).status,
            LegalStatus::Both
        );
    }

    #[test]
    fn strict_support_defeats_non_strict_opposition() {
        let strict = rule(
            "constitutional-right",
            RuleKind::Strict,
            "citizen",
            positive("appeal"),
        );
        let contrary = rule(
            "agency-denial",
            RuleKind::Defeasible,
            "citizen",
            negative("appeal"),
        );
        let pack =
            RulePack::new(RulePackId::new("appeal").unwrap(), [strict, contrary], []).unwrap();
        let facts = [positive("citizen")].into_iter().collect();

        assert_eq!(
            resolve_literal(&pack, &facts, &positive("appeal")).status,
            LegalStatus::Supported
        );
    }

    #[test]
    fn defeated_attacker_cannot_produce_a_zombie_defeat() {
        let positive_blocker = rule(
            "top-blocker",
            RuleKind::Defeater,
            "condition",
            positive("enter"),
        );
        let negative_rule = rule(
            "middle-denial",
            RuleKind::Defeasible,
            "condition",
            negative("enter"),
        );
        let positive_rule = rule(
            "lower-permission",
            RuleKind::Defeasible,
            "condition",
            positive("enter"),
        );
        let pack = RulePack::new(
            RulePackId::new("grounded-defeat").unwrap(),
            [positive_blocker, negative_rule, positive_rule],
            [
                Superiority::new(
                    RuleId::new("top-blocker").unwrap(),
                    RuleId::new("middle-denial").unwrap(),
                    PriorityBasis::Procedural,
                ),
                Superiority::new(
                    RuleId::new("middle-denial").unwrap(),
                    RuleId::new("lower-permission").unwrap(),
                    PriorityBasis::Procedural,
                ),
            ],
        )
        .unwrap();
        let facts = [positive("condition")].into_iter().collect();
        let result = resolve_literal(&pack, &facts, &positive("enter"));

        assert_eq!(result.status, LegalStatus::Supported);
        assert_eq!(
            result.undefeated_support,
            vec![RuleId::new("lower-permission").unwrap()]
        );
    }

    #[test]
    fn defeater_blocks_without_establishing_its_conclusion() {
        let deny = rule("deny", RuleKind::Defeasible, "condition", negative("enter"));
        let blocker = rule(
            "block-denial",
            RuleKind::Defeater,
            "condition",
            positive("enter"),
        );
        let pack = RulePack::new(
            RulePackId::new("entry-defeater").unwrap(),
            [deny, blocker],
            [Superiority::new(
                RuleId::new("block-denial").unwrap(),
                RuleId::new("deny").unwrap(),
                PriorityBasis::Procedural,
            )],
        )
        .unwrap();
        let facts = [positive("condition")].into_iter().collect();
        let result = resolve_literal(&pack, &facts, &positive("enter"));

        assert_eq!(result.status, LegalStatus::Undetermined);
        assert_eq!(
            result.blocking_defeaters,
            vec![RuleId::new("block-denial").unwrap()]
        );
    }
}
