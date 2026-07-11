// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Defeasible rules: defaults that fire unless an exception applies.
//!
//! This is how statutes-with-exemptions and common-sense legal defaults behave:
//! "birds fly" *unless* the bird is a penguin.

use std::collections::BTreeSet;

/// A defeasible rule: if every condition holds and no exception holds, conclude.
#[derive(Debug, Clone, PartialEq, Eq)]
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

/// Forward-chain the rules over the initial facts to a fixpoint, respecting
/// exceptions. Returns all derivable conclusions plus the initial facts.
pub fn derive(rules: &[Rule], initial_facts: &[&str]) -> BTreeSet<String> {
    let mut facts: BTreeSet<String> = initial_facts.iter().map(|s| s.to_string()).collect();
    loop {
        let mut added = false;
        for rule in rules {
            if rule.fires(&facts) && !facts.contains(&rule.conclusion) {
                facts.insert(rule.conclusion.clone());
                added = true;
            }
        }
        if !added {
            break;
        }
    }
    facts
}

/// Whether a specific conclusion is derivable.
pub fn entails(rules: &[Rule], facts: &[&str], conclusion: &str) -> bool {
    derive(rules, facts).contains(conclusion)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zoology() -> Vec<Rule> {
        vec![
            Rule::new(&["bird"], &["penguin"], "flies"), // birds fly, unless penguins
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
        // A penguin is a bird but does not fly (exception fires) — yet is still
        // warm-blooded (that rule has no exception).
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
        // A disenfranchised citizen doesn't vote, so the chain doesn't fire.
        assert!(!entails(
            &rules,
            &["citizen", "adult", "disenfranchised"],
            "counts_toward_quorum"
        ));
    }
}
