// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Explicit and auditable rule-superiority relations.
//!
//! The kernel never invents priority merely from source order. A caller may
//! supply a validated acyclic superiority graph and record the legal basis for
//! each edge. Higher authority, specificity, and later enactment are therefore
//! policy inputs rather than hidden universal assumptions.

use crate::model::RuleId;
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

/// Why one formal rule is declared stronger than another.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PriorityBasis {
    Explicit,
    HigherAuthority,
    MoreSpecific,
    LaterInTime,
    Procedural,
    Other(String),
}

/// One directed superiority edge: `stronger` defeats `weaker` when they clash.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Superiority {
    pub stronger: RuleId,
    pub weaker: RuleId,
    pub basis: PriorityBasis,
}

impl Superiority {
    pub fn new(stronger: RuleId, weaker: RuleId, basis: PriorityBasis) -> Self {
        Self {
            stronger,
            weaker,
            basis,
        }
    }
}

/// Validation failure for a superiority graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PriorityError {
    SelfPriority { rule: RuleId },
    UnknownRule { rule: RuleId },
    Cycle { rules: Vec<RuleId> },
}

impl fmt::Display for PriorityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PriorityError::SelfPriority { rule } => {
                write!(f, "rule {rule} cannot outrank itself")
            }
            PriorityError::UnknownRule { rule } => {
                write!(f, "priority relation references unknown rule {rule}")
            }
            PriorityError::Cycle { rules } => {
                let joined = rules
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(f, "rule-superiority graph contains a cycle: {joined}")
            }
        }
    }
}

impl Error for PriorityError {}

/// A validated, deterministic, acyclic rule-superiority graph.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SuperiorityGraph {
    rules: BTreeSet<RuleId>,
    edges: BTreeMap<RuleId, BTreeMap<RuleId, BTreeSet<PriorityBasis>>>,
}

impl SuperiorityGraph {
    /// Validate and construct a graph over the supplied rule universe.
    pub fn new(
        rules: impl IntoIterator<Item = RuleId>,
        relations: impl IntoIterator<Item = Superiority>,
    ) -> Result<Self, PriorityError> {
        let rules: BTreeSet<RuleId> = rules.into_iter().collect();
        let mut edges: BTreeMap<RuleId, BTreeMap<RuleId, BTreeSet<PriorityBasis>>> =
            BTreeMap::new();

        for relation in relations {
            if relation.stronger == relation.weaker {
                return Err(PriorityError::SelfPriority {
                    rule: relation.stronger,
                });
            }
            if !rules.contains(&relation.stronger) {
                return Err(PriorityError::UnknownRule {
                    rule: relation.stronger,
                });
            }
            if !rules.contains(&relation.weaker) {
                return Err(PriorityError::UnknownRule {
                    rule: relation.weaker,
                });
            }
            edges
                .entry(relation.stronger)
                .or_default()
                .entry(relation.weaker)
                .or_default()
                .insert(relation.basis);
        }

        let graph = Self { rules, edges };
        if let Some(cycle) = graph.find_cycle() {
            return Err(PriorityError::Cycle { rules: cycle });
        }
        Ok(graph)
    }

    pub fn rules(&self) -> impl Iterator<Item = &RuleId> {
        self.rules.iter()
    }

    /// Direct legal bases recorded for one priority edge.
    pub fn direct_bases(
        &self,
        stronger: &RuleId,
        weaker: &RuleId,
    ) -> Option<&BTreeSet<PriorityBasis>> {
        self.edges.get(stronger)?.get(weaker)
    }

    /// Whether `stronger` transitively outranks `weaker`.
    pub fn outranks(&self, stronger: &RuleId, weaker: &RuleId) -> bool {
        if stronger == weaker {
            return false;
        }
        let mut frontier = vec![stronger.clone()];
        let mut visited = BTreeSet::new();
        while let Some(current) = frontier.pop() {
            if !visited.insert(current.clone()) {
                continue;
            }
            if let Some(next) = self.edges.get(&current) {
                for candidate in next.keys() {
                    if candidate == weaker {
                        return true;
                    }
                    frontier.push(candidate.clone());
                }
            }
        }
        false
    }

    /// Canonical direct relations, preserving all distinct legal bases.
    pub fn relations(&self) -> Vec<Superiority> {
        let mut relations = Vec::new();
        for (stronger, weaker_map) in &self.edges {
            for (weaker, bases) in weaker_map {
                for basis in bases {
                    relations.push(Superiority::new(
                        stronger.clone(),
                        weaker.clone(),
                        basis.clone(),
                    ));
                }
            }
        }
        relations
    }

    fn find_cycle(&self) -> Option<Vec<RuleId>> {
        fn visit(
            node: &RuleId,
            graph: &SuperiorityGraph,
            active: &mut Vec<RuleId>,
            complete: &mut BTreeSet<RuleId>,
        ) -> Option<Vec<RuleId>> {
            if let Some(start) = active.iter().position(|candidate| candidate == node) {
                let mut cycle = active[start..].to_vec();
                cycle.push(node.clone());
                return Some(cycle);
            }
            if complete.contains(node) {
                return None;
            }

            active.push(node.clone());
            if let Some(next) = graph.edges.get(node) {
                for candidate in next.keys() {
                    if let Some(cycle) = visit(candidate, graph, active, complete) {
                        return Some(cycle);
                    }
                }
            }
            active.pop();
            complete.insert(node.clone());
            None
        }

        let mut complete = BTreeSet::new();
        for rule in &self.rules {
            let mut active = Vec::new();
            if let Some(cycle) = visit(rule, self, &mut active, &mut complete) {
                return Some(cycle);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(value: &str) -> RuleId {
        RuleId::new(value).unwrap()
    }

    #[test]
    fn priority_is_transitive_but_not_reflexive() {
        let graph = SuperiorityGraph::new(
            [id("constitution"), id("statute"), id("regulation")],
            [
                Superiority::new(
                    id("constitution"),
                    id("statute"),
                    PriorityBasis::HigherAuthority,
                ),
                Superiority::new(
                    id("statute"),
                    id("regulation"),
                    PriorityBasis::HigherAuthority,
                ),
            ],
        )
        .unwrap();

        assert!(graph.outranks(&id("constitution"), &id("regulation")));
        assert!(!graph.outranks(&id("statute"), &id("statute")));
        assert!(!graph.outranks(&id("regulation"), &id("constitution")));
    }

    #[test]
    fn duplicate_edges_preserve_distinct_bases_canonically() {
        let graph = SuperiorityGraph::new(
            [id("specific"), id("general")],
            [
                Superiority::new(id("specific"), id("general"), PriorityBasis::MoreSpecific),
                Superiority::new(id("specific"), id("general"), PriorityBasis::LaterInTime),
            ],
        )
        .unwrap();

        assert_eq!(graph.relations().len(), 2);
        assert_eq!(
            graph
                .direct_bases(&id("specific"), &id("general"))
                .unwrap()
                .len(),
            2
        );
    }

    #[test]
    fn unknown_self_and_cyclic_relations_fail_closed() {
        assert!(matches!(
            SuperiorityGraph::new(
                [id("a")],
                [Superiority::new(id("a"), id("a"), PriorityBasis::Explicit)]
            ),
            Err(PriorityError::SelfPriority { .. })
        ));
        assert!(matches!(
            SuperiorityGraph::new(
                [id("a")],
                [Superiority::new(id("a"), id("b"), PriorityBasis::Explicit)]
            ),
            Err(PriorityError::UnknownRule { .. })
        ));
        assert!(matches!(
            SuperiorityGraph::new(
                [id("a"), id("b")],
                [
                    Superiority::new(id("a"), id("b"), PriorityBasis::Explicit),
                    Superiority::new(id("b"), id("a"), PriorityBasis::Explicit),
                ]
            ),
            Err(PriorityError::Cycle { .. })
        ));
    }
}
