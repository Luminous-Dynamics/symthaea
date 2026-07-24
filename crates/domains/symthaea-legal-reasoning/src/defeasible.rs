// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Defeasible rules: defaults that fire unless an exception applies.
//!
//! Rules are evaluated with a deterministic, locally stratified semantics.
//! Any atom used as an exception must be resolved in a lower stratum than the
//! rule conclusion. This prevents a later-derived exception from leaving a
//! stale conclusion behind and makes derivation independent of rule ordering.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

/// A defeasible rule: if every condition holds and no exception holds, conclude.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Rule {
    pub conditions: Vec<String>,
    pub exceptions: Vec<String>,
    pub conclusion: String,
}

impl Rule {
    pub fn new(conditions: &[&str], exceptions: &[&str], conclusion: &str) -> Rule {
        Rule {
            conditions: conditions.iter().map(|s| s.to_string()).collect(),
            exceptions: exceptions.iter().map(|s| s.to_string()).collect(),
            conclusion: conclusion.to_string(),
        }
    }

    fn fires(&self, facts: &BTreeSet<String>) -> bool {
        self.conditions.iter().all(|c| facts.contains(c))
            && !self.exceptions.iter().any(|e| facts.contains(e))
    }
}

/// An invalid defeasible theory that cannot be assigned local strata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DerivationError {
    /// The theory contains a dependency cycle through one or more exceptions.
    /// Such theories require a stronger explicitly selected semantics rather
    /// than ordinary stratified evaluation.
    NonStratified { atoms: Vec<String> },
}

impl fmt::Display for DerivationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DerivationError::NonStratified { atoms } => write!(
                f,
                "defeasible theory is not locally stratified; exception cycle involves: {}",
                atoms.join(", ")
            ),
        }
    }
}

impl Error for DerivationError {}

/// One canonical derivation layer for a newly established conclusion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DerivationStep {
    pub conclusion: String,
    pub stratum: usize,
    /// Every rule that independently supported the conclusion when it first
    /// entered the fact set, in canonical structural order.
    pub supporting_rules: Vec<Rule>,
}

/// A deterministic derivation result with replayable explanation steps.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Derivation {
    pub initial_facts: BTreeSet<String>,
    pub facts: BTreeSet<String>,
    pub steps: Vec<DerivationStep>,
}

impl Derivation {
    pub fn entails(&self, conclusion: &str) -> bool {
        self.facts.contains(conclusion)
    }

    pub fn is_initial_fact(&self, conclusion: &str) -> bool {
        self.initial_facts.contains(conclusion)
    }

    pub fn supporting_step(&self, conclusion: &str) -> Option<&DerivationStep> {
        self.steps.iter().find(|step| step.conclusion == conclusion)
    }
}

/// A rule that could have established a requested conclusion, together with
/// the reasons it did not fire in the final derivation state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockedRule {
    pub rule: Rule,
    pub missing_conditions: Vec<String>,
    pub active_exceptions: Vec<String>,
}

fn compute_strata(rules: &[Rule]) -> Result<BTreeMap<String, usize>, DerivationError> {
    let mut strata = BTreeMap::new();
    for rule in rules {
        strata.entry(rule.conclusion.clone()).or_insert(0);
        for atom in rule.conditions.iter().chain(rule.exceptions.iter()) {
            strata.entry(atom.clone()).or_insert(0);
        }
    }

    // Difference constraints are monotone. If they are still increasing after
    // |atoms| complete relaxations, a strict (exception) edge participates in
    // a cycle and no finite local stratification exists.
    let limit = strata.len().max(1);
    for round in 0..=limit {
        let mut changed = BTreeSet::new();

        for rule in rules {
            let condition_floor = rule
                .conditions
                .iter()
                .filter_map(|atom| strata.get(atom).copied())
                .max()
                .unwrap_or(0);
            let exception_floor = rule
                .exceptions
                .iter()
                .filter_map(|atom| strata.get(atom).copied())
                .map(|level| level.saturating_add(1))
                .max()
                .unwrap_or(0);
            let required = condition_floor.max(exception_floor);
            let current = strata.get(&rule.conclusion).copied().unwrap_or(0);

            if required > current {
                strata.insert(rule.conclusion.clone(), required);
                changed.insert(rule.conclusion.clone());
            }
        }

        if changed.is_empty() {
            return Ok(strata);
        }

        if round == limit {
            return Err(DerivationError::NonStratified {
                atoms: changed.into_iter().collect(),
            });
        }
    }

    unreachable!("stratification loop always returns")
}

/// Derive all conclusions and a canonical proof trace under deterministic,
/// locally stratified semantics.
pub fn try_derive_with_trace(
    rules: &[Rule],
    initial_facts: &[&str],
) -> Result<Derivation, DerivationError> {
    let strata = compute_strata(rules)?;
    let initial_facts: BTreeSet<String> =
        initial_facts.iter().map(|s| s.to_string()).collect();
    let mut facts = initial_facts.clone();
    let mut ordered_rules: Vec<&Rule> = rules.iter().collect();
    ordered_rules.sort_unstable();
    let mut steps = Vec::new();

    let max_stratum = strata.values().copied().max().unwrap_or(0);
    for current_stratum in 0..=max_stratum {
        loop {
            let mut support: BTreeMap<String, Vec<Rule>> = BTreeMap::new();
            for rule in ordered_rules.iter().copied().filter(|rule| {
                strata.get(&rule.conclusion).copied().unwrap_or(0) == current_stratum
                    && rule.fires(&facts)
                    && !facts.contains(&rule.conclusion)
            }) {
                support
                    .entry(rule.conclusion.clone())
                    .or_default()
                    .push((*rule).clone());
            }

            if support.is_empty() {
                break;
            }

            for (conclusion, supporting_rules) in support {
                facts.insert(conclusion.clone());
                steps.push(DerivationStep {
                    conclusion,
                    stratum: current_stratum,
                    supporting_rules,
                });
            }
        }
    }

    Ok(Derivation {
        initial_facts,
        facts,
        steps,
    })
}

/// Derive all conclusions under deterministic locally stratified semantics.
///
/// Rules in lower strata are closed to a fixpoint before any higher-stratum
/// default is considered. Therefore exceptions are final when a dependent
/// default fires, and permuting the input rule slice cannot change the result.
pub fn try_derive(
    rules: &[Rule],
    initial_facts: &[&str],
) -> Result<BTreeSet<String>, DerivationError> {
    Ok(try_derive_with_trace(rules, initial_facts)?.facts)
}

/// Explain why `conclusion` was not derived by listing every directly relevant
/// rule and its missing conditions or active exceptions.
///
/// Returns an empty vector when the conclusion is already an initial or derived
/// fact, or when no rule in the theory concludes it.
pub fn try_why_not(
    rules: &[Rule],
    initial_facts: &[&str],
    conclusion: &str,
) -> Result<Vec<BlockedRule>, DerivationError> {
    let derivation = try_derive_with_trace(rules, initial_facts)?;
    if derivation.entails(conclusion) {
        return Ok(Vec::new());
    }

    let mut relevant: Vec<&Rule> = rules
        .iter()
        .filter(|rule| rule.conclusion == conclusion)
        .collect();
    relevant.sort_unstable();

    Ok(relevant
        .into_iter()
        .map(|rule| BlockedRule {
            rule: (*rule).clone(),
            missing_conditions: rule
                .conditions
                .iter()
                .filter(|condition| !derivation.facts.contains(*condition))
                .cloned()
                .collect(),
            active_exceptions: rule
                .exceptions
                .iter()
                .filter(|exception| derivation.facts.contains(*exception))
                .cloned()
                .collect(),
        })
        .collect())
}

/// Forward-chain a locally stratified rule theory to a fixpoint.
///
/// This compatibility wrapper panics for a non-stratified theory. New code
/// should prefer [`try_derive`] and handle [`DerivationError`] explicitly.
#[track_caller]
pub fn derive(rules: &[Rule], initial_facts: &[&str]) -> BTreeSet<String> {
    try_derive(rules, initial_facts)
        .unwrap_or_else(|error| panic!("cannot derive from invalid theory: {error}"))
}

/// Whether a specific conclusion is derivable in a valid stratified theory.
pub fn try_entails(
    rules: &[Rule],
    facts: &[&str],
    conclusion: &str,
) -> Result<bool, DerivationError> {
    Ok(try_derive(rules, facts)?.contains(conclusion))
}

/// Whether a specific conclusion is derivable.
///
/// This compatibility wrapper panics for a non-stratified theory. New code
/// should prefer [`try_entails`].
#[track_caller]
pub fn entails(rules: &[Rule], facts: &[&str], conclusion: &str) -> bool {
    try_entails(rules, facts, conclusion)
        .unwrap_or_else(|error| panic!("cannot query invalid theory: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zoology() -> Vec<Rule> {
        vec![
            Rule::new(&["bird"], &["penguin"], "flies"),
            Rule::new(&["mammal"], &[], "warm_blooded"),
            Rule::new(&["bird"], &[], "warm_blooded"),
        ]
    }

    #[test]
    fn default_fires_without_exception() {
        assert!(entails(&zoology(), &["bird"], "flies"));
        assert!(entails(&zoology(), &["bird"], "warm_blooded"));
    }

    #[test]
    fn exception_defeats_the_default() {
        let facts = derive(&zoology(), &["bird", "penguin"]);
        assert!(!facts.contains("flies"));
        assert!(facts.contains("warm_blooded"));
    }

    #[test]
    fn conclusions_chain() {
        let rules = vec![
            Rule::new(&["citizen", "adult"], &["disenfranchised"], "may_vote"),
            Rule::new(&["may_vote"], &[], "counts_toward_quorum"),
        ];
        assert!(entails(
            &rules,
            &["citizen", "adult"],
            "counts_toward_quorum"
        ));
        assert!(!entails(
            &rules,
            &["citizen", "adult", "disenfranchised"],
            "counts_toward_quorum"
        ));
    }

    #[test]
    fn later_derived_exception_retracts_default_independent_of_order() {
        let default = Rule::new(&["bird"], &["penguin"], "flies");
        let exception = Rule::new(&["bird"], &[], "penguin");

        let forward = try_derive(&[default.clone(), exception.clone()], &["bird"]).unwrap();
        let reverse = try_derive(&[exception, default], &["bird"]).unwrap();

        assert_eq!(forward, reverse);
        assert!(forward.contains("penguin"));
        assert!(!forward.contains("flies"));
    }

    #[test]
    fn positive_cycles_are_valid_and_close_to_a_fixpoint() {
        let rules = vec![
            Rule::new(&["a"], &[], "b"),
            Rule::new(&["b"], &[], "a"),
        ];
        let facts = try_derive(&rules, &["a"]).unwrap();
        assert!(facts.contains("a"));
        assert!(facts.contains("b"));
    }

    #[test]
    fn exception_cycles_are_rejected() {
        let rules = vec![
            Rule::new(&[], &["b"], "a"),
            Rule::new(&[], &["a"], "b"),
        ];
        assert!(matches!(
            try_derive(&rules, &[]),
            Err(DerivationError::NonStratified { .. })
        ));
    }

    #[test]
    fn duplicate_and_irrelevant_rules_do_not_change_the_query() {
        let base = vec![Rule::new(&["adult"], &[], "competent")];
        let expanded = vec![
            Rule::new(&["adult"], &[], "competent"),
            Rule::new(&["adult"], &[], "competent"),
            Rule::new(&["ship"], &[], "afloat"),
        ];
        assert_eq!(
            try_entails(&base, &["adult"], "competent"),
            try_entails(&expanded, &["adult"], "competent")
        );
    }

    #[test]
    fn trace_is_canonical_across_rule_order() {
        let adult = Rule::new(&["age_18"], &[], "adult");
        let capacity = Rule::new(&["adult"], &["incapacitated"], "has_capacity");

        let forward =
            try_derive_with_trace(&[adult.clone(), capacity.clone()], &["age_18"]).unwrap();
        let reverse = try_derive_with_trace(&[capacity, adult], &["age_18"]).unwrap();

        assert_eq!(forward, reverse);
        assert_eq!(forward.steps.len(), 2);
        assert_eq!(forward.steps[0].conclusion, "adult");
        assert_eq!(forward.steps[1].conclusion, "has_capacity");
    }

    #[test]
    fn why_not_reports_missing_conditions_and_active_exceptions() {
        let rules = vec![Rule::new(
            &["citizen", "adult"],
            &["disenfranchised"],
            "may_vote",
        )];
        let blocked = try_why_not(
            &rules,
            &["citizen", "disenfranchised"],
            "may_vote",
        )
        .unwrap();

        assert_eq!(blocked.len(), 1);
        assert_eq!(blocked[0].missing_conditions, vec!["adult"]);
        assert_eq!(blocked[0].active_exceptions, vec!["disenfranchised"]);
    }
}
