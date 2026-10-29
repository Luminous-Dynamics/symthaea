// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Reference truth-maintenance graph for evolving agent memory.
//!
//! This is a small, deterministic research API, not a production persistence
//! layer. Updates use clone-validate-swap so a failed operation does not leave
//! a partially mutated in-memory graph. Production integration must replace
//! whole-state cloning with a transactional store and prove crash/replay parity.

use chrono::NaiveDate;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "kind", content = "id", rename_all = "snake_case")]
pub enum DependencyRef {
    SourceEvent(String),
    DerivedFact(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DerivedFactInput {
    pub fact_id: String,
    pub subject: String,
    pub attribute: String,
    pub value: String,
    /// Half-open interval start, formatted YYYY-MM-DD.
    pub valid_from: String,
    /// Optional exclusive end, formatted YYYY-MM-DD.
    pub valid_to: Option<String>,
    pub rule_id: String,
    pub rule_version: String,
    pub dependencies: BTreeSet<DependencyRef>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DerivedStatus {
    Current,
    Invalidated {
        because: DependencyRef,
        reason: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DerivedFactRecord {
    pub fact: DerivedFactInput,
    pub status: DerivedStatus,
}

#[derive(Debug, Clone, Default)]
pub struct DerivationGraph {
    source_events: BTreeSet<String>,
    retracted_source_events: BTreeSet<String>,
    facts: BTreeMap<String, DerivedFactRecord>,
    reverse_dependencies: BTreeMap<DependencyRef, BTreeSet<String>>,
}

fn non_empty(value: &str, field: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        return Err(format!("{field} must not be empty"));
    }
    Ok(())
}

fn validate_fact_input(fact: &DerivedFactInput) -> Result<(), String> {
    for (field, value) in [
        ("fact_id", fact.fact_id.as_str()),
        ("subject", fact.subject.as_str()),
        ("attribute", fact.attribute.as_str()),
        ("value", fact.value.as_str()),
        ("rule_id", fact.rule_id.as_str()),
        ("rule_version", fact.rule_version.as_str()),
    ] {
        non_empty(value, field)?;
    }
    if fact.dependencies.is_empty() {
        return Err(format!("derived fact {} must have at least one dependency", fact.fact_id));
    }
    let start = NaiveDate::parse_from_str(&fact.valid_from, "%Y-%m-%d")
        .map_err(|error| format!("valid_from must be YYYY-MM-DD: {error}"))?;
    if let Some(end) = fact.valid_to.as_deref() {
        let end = NaiveDate::parse_from_str(end, "%Y-%m-%d")
            .map_err(|error| format!("valid_to must be YYYY-MM-DD: {error}"))?;
        if end <= start {
            return Err("valid_to must be after valid_from".into());
        }
    }
    Ok(())
}

impl DerivationGraph {
    /// Register an immutable source-event identity. Reuse of IDs is rejected;
    /// retractions never delete the original event from the audit set.
    pub fn register_source_event(&mut self, event_id: impl Into<String>) -> Result<(), String> {
        let event_id = event_id.into();
        non_empty(&event_id, "source event ID")?;
        if self.source_events.contains(&event_id) {
            return Err(format!("duplicate source event ID {event_id:?}"));
        }

        let mut candidate = self.clone();
        candidate.source_events.insert(event_id);
        candidate.verify_consistency()?;
        *self = candidate;
        Ok(())
    }

    /// Add one derived fact whose dependencies must already exist and be
    /// current. Requiring dependencies to exist first makes the graph a DAG by
    /// construction; verify_consistency independently checks for cycles.
    pub fn derive(&mut self, fact: DerivedFactInput) -> Result<(), String> {
        validate_fact_input(&fact)?;
        if self.facts.contains_key(&fact.fact_id) {
            return Err(format!("duplicate derived fact ID {:?}", fact.fact_id));
        }

        for dependency in &fact.dependencies {
            match dependency {
                DependencyRef::SourceEvent(event_id) => {
                    if !self.source_events.contains(event_id) {
                        return Err(format!("unknown source dependency {event_id:?}"));
                    }
                    if self.retracted_source_events.contains(event_id) {
                        return Err(format!("cannot derive from retracted source event {event_id:?}"));
                    }
                }
                DependencyRef::DerivedFact(parent_id) => {
                    if parent_id == &fact.fact_id {
                        return Err(format!("derived fact {:?} cannot depend on itself", fact.fact_id));
                    }
                    let Some(record) = self.facts.get(parent_id) else {
                        return Err(format!("unknown derived dependency {parent_id:?}"));
                    };
                    if record.status != DerivedStatus::Current {
                        return Err(format!("cannot derive from invalidated fact {parent_id:?}"));
                    }
                }
            }
        }

        let id = fact.fact_id.clone();
        let mut candidate = self.clone();
        for dependency in &fact.dependencies {
            candidate
                .reverse_dependencies
                .entry(dependency.clone())
                .or_default()
                .insert(id.clone());
        }
        candidate.facts.insert(
            id,
            DerivedFactRecord {
                fact,
                status: DerivedStatus::Current,
            },
        );
        candidate.verify_consistency()?;
        *self = candidate;
        Ok(())
    }

    /// Retract a source event and invalidate every downstream derived fact.
    /// Returns the affected fact IDs in deterministic lexicographic order.
    /// Replaying the same retraction is idempotent and returns an empty list.
    pub fn invalidate_source_event(
        &mut self,
        event_id: &str,
        reason: &str,
    ) -> Result<Vec<String>, String> {
        non_empty(reason, "invalidation reason")?;
        if !self.source_events.contains(event_id) {
            return Err(format!("cannot retract unknown source event {event_id:?}"));
        }
        if self.retracted_source_events.contains(event_id) {
            return Ok(Vec::new());
        }
        self.verify_consistency()?;

        let mut candidate = self.clone();
        candidate.retracted_source_events.insert(event_id.to_owned());
        let mut queue = VecDeque::new();
        queue.push_back(DependencyRef::SourceEvent(event_id.to_owned()));
        let mut affected = BTreeSet::new();

        while let Some(dependency) = queue.pop_front() {
            let dependents: Vec<String> = candidate
                .reverse_dependencies
                .get(&dependency)
                .map(|ids| ids.iter().cloned().collect())
                .unwrap_or_default();

            for fact_id in dependents {
                let Some(record) = candidate.facts.get_mut(&fact_id) else {
                    return Err(format!("reverse index references missing fact {fact_id:?}"));
                };
                if record.status != DerivedStatus::Current {
                    continue;
                }
                record.status = DerivedStatus::Invalidated {
                    because: dependency.clone(),
                    reason: reason.to_owned(),
                };
                affected.insert(fact_id.clone());
                queue.push_back(DependencyRef::DerivedFact(fact_id));
            }
        }

        candidate.verify_consistency()?;
        *self = candidate;
        Ok(affected.into_iter().collect())
    }

    pub fn source_is_retracted(&self, event_id: &str) -> bool {
        self.retracted_source_events.contains(event_id)
    }

    pub fn fact(&self, fact_id: &str) -> Option<&DerivedFactRecord> {
        self.facts.get(fact_id)
    }

    /// Return a fact only when its derivation is still valid.
    pub fn current_fact(&self, fact_id: &str) -> Option<&DerivedFactInput> {
        self.facts
            .get(fact_id)
            .filter(|record| record.status == DerivedStatus::Current)
            .map(|record| &record.fact)
    }

    pub fn fact_count(&self) -> usize {
        self.facts.len()
    }

    pub fn current_fact_count(&self) -> usize {
        self.facts
            .values()
            .filter(|record| record.status == DerivedStatus::Current)
            .count()
    }

    pub fn invalidated_fact_count(&self) -> usize {
        self.facts
            .values()
            .filter(|record| matches!(&record.status, DerivedStatus::Invalidated { .. }))
            .count()
    }

    pub fn source_event_count(&self) -> usize {
        self.source_events.len()
    }

    /// Rebuild the reverse index from the canonical forward dependency sets.
    /// This is a reference operation for testing index-rebuild parity.
    pub fn rebuild_reverse_index(&mut self) -> Result<(), String> {
        let mut candidate = self.clone();
        candidate.reverse_dependencies.clear();
        for (fact_id, record) in &candidate.facts {
            for dependency in &record.fact.dependencies {
                candidate
                    .reverse_dependencies
                    .entry(dependency.clone())
                    .or_default()
                    .insert(fact_id.clone());
            }
        }
        candidate.verify_consistency()?;
        *self = candidate;
        Ok(())
    }

    /// Check forward/reverse index parity, referential integrity, valid current
    /// dependencies and acyclicity. This runs before state hashes are emitted.
    pub fn verify_consistency(&self) -> Result<(), String> {
        let mut expected_reverse: BTreeMap<DependencyRef, BTreeSet<String>> = BTreeMap::new();

        for (fact_id, record) in &self.facts {
            if fact_id != &record.fact.fact_id {
                return Err(format!("fact index mismatch for {fact_id:?}"));
            }
            validate_fact_input(&record.fact)?;
            if record.fact.dependencies.contains(&DependencyRef::DerivedFact(fact_id.clone())) {
                return Err(format!("derived fact {fact_id:?} has a self-dependency"));
            }

            for dependency in &record.fact.dependencies {
                match dependency {
                    DependencyRef::SourceEvent(event_id) => {
                        if !self.source_events.contains(event_id) {
                            return Err(format!("fact {fact_id:?} references unknown source {event_id:?}"));
                        }
                        if record.status == DerivedStatus::Current
                            && self.retracted_source_events.contains(event_id)
                        {
                            return Err(format!(
                                "current fact {fact_id:?} depends on retracted source {event_id:?}"
                            ));
                        }
                    }
                    DependencyRef::DerivedFact(parent_id) => {
                        let Some(parent) = self.facts.get(parent_id) else {
                            return Err(format!("fact {fact_id:?} references unknown derived fact {parent_id:?}"));
                        };
                        if record.status == DerivedStatus::Current
                            && parent.status != DerivedStatus::Current
                        {
                            return Err(format!(
                                "current fact {fact_id:?} depends on invalidated fact {parent_id:?}"
                            ));
                        }
                    }
                }
                expected_reverse
                    .entry(dependency.clone())
                    .or_default()
                    .insert(fact_id.clone());
            }
        }

        if expected_reverse != self.reverse_dependencies {
            return Err("forward/reverse dependency indexes differ".into());
        }

        // 0/unseen, 1/visiting, 2/complete.
        fn visit(
            fact_id: &str,
            facts: &BTreeMap<String, DerivedFactRecord>,
            marks: &mut BTreeMap<String, u8>,
        ) -> Result<(), String> {
            match marks.get(fact_id).copied() {
                Some(1) => return Err(format!("cycle detected at derived fact {fact_id:?}")),
                Some(2) => return Ok(()),
                _ => {}
            }
            let record = facts
                .get(fact_id)
                .ok_or_else(|| format!("missing derived fact {fact_id:?} during cycle check"))?;
            marks.insert(fact_id.to_owned(), 1);
            for dependency in &record.fact.dependencies {
                if let DependencyRef::DerivedFact(parent_id) = dependency {
                    visit(parent_id, facts, marks)?;
                }
            }
            marks.insert(fact_id.to_owned(), 2);
            Ok(())
        }

        let mut marks = BTreeMap::new();
        for fact_id in self.facts.keys() {
            visit(fact_id, &self.facts, &mut marks)?;
        }
        Ok(())
    }

    /// Deterministic digest of logical state plus its materialized reverse
    /// index. The digest is for parity checks, not authentication or anti-rollback.
    pub fn state_sha256(&self) -> Result<String, String> {
        self.verify_consistency()?;
        let reverse_index: Vec<(&DependencyRef, &BTreeSet<String>)> =
            self.reverse_dependencies.iter().collect();
        let bytes = serde_json::to_vec(&(
            &self.source_events,
            &self.retracted_source_events,
            &self.facts,
            reverse_index,
        ))
        .map_err(|error| format!("cannot serialize derivation state: {error}"))?;
        Ok(format!("{:x}", Sha256::digest(bytes)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_dependencies(items: impl IntoIterator<Item = DependencyRef>) -> BTreeSet<DependencyRef> {
        items.into_iter().collect()
    }

    fn fact(
        id: &str,
        deps: impl IntoIterator<Item = DependencyRef>,
    ) -> DerivedFactInput {
        DerivedFactInput {
            fact_id: id.into(),
            subject: "build-42".into(),
            attribute: "ready".into(),
            value: "true".into(),
            valid_from: "2026-01-01".into(),
            valid_to: None,
            rule_id: "all-prerequisites-satisfied".into(),
            rule_version: "v1".into(),
            dependencies: make_dependencies(deps),
        }
    }

    #[test]
    fn invalidating_leaf_propagates_through_transitive_descendants() {
        let mut graph = DerivationGraph::default();
        graph.register_source_event("E1").unwrap();
        graph.register_source_event("E2").unwrap();
        graph
            .derive(fact(
                "D1",
                [DependencyRef::SourceEvent("E1".into()), DependencyRef::SourceEvent("E2".into())],
            ))
            .unwrap();
        graph
            .derive(fact("D2", [DependencyRef::DerivedFact("D1".into())]))
            .unwrap();
        graph
            .derive(fact("D3", [DependencyRef::DerivedFact("D2".into())]))
            .unwrap();
        graph
            .derive(fact("D4", [DependencyRef::SourceEvent("E2".into())]))
            .unwrap();

        let affected = graph.invalidate_source_event("E1", "source correction").unwrap();
        assert_eq!(affected, vec!["D1", "D2", "D3"]);
        assert_eq!(graph.current_fact_count(), 1);
        assert_eq!(graph.invalidated_fact_count(), 3);
        assert!(graph.current_fact("D4").is_some());
        assert!(graph.current_fact("D2").is_none());
        graph.verify_consistency().unwrap();
    }

    #[test]
    fn invalidation_is_idempotent_and_keeps_audit_records() {
        let mut graph = DerivationGraph::default();
        graph.register_source_event("E1").unwrap();
        graph.derive(fact("D1", [DependencyRef::SourceEvent("E1".into())])).unwrap();

        assert_eq!(graph.invalidate_source_event("E1", "retracted").unwrap(), vec!["D1"]);
        assert!(graph.invalidate_source_event("E1", "replayed retraction").unwrap().is_empty());
        assert!(graph.source_is_retracted("E1"));
        assert_eq!(graph.fact_count(), 1);
        assert!(matches!(
            &graph.fact("D1").unwrap().status,
            DerivedStatus::Invalidated { .. }
        ));
    }

    #[test]
    fn rejects_derivation_from_retracted_source() {
        let mut graph = DerivationGraph::default();
        graph.register_source_event("E1").unwrap();
        graph.invalidate_source_event("E1", "bad evidence").unwrap();
        assert!(graph
            .derive(fact("D1", [DependencyRef::SourceEvent("E1".into())]))
            .unwrap_err()
            .contains("retracted source"));
    }

    #[test]
    fn rejects_derivation_from_invalidated_parent() {
        let mut graph = DerivationGraph::default();
        graph.register_source_event("E1").unwrap();
        graph.derive(fact("D1", [DependencyRef::SourceEvent("E1".into())])).unwrap();
        graph.derive(fact("D2", [DependencyRef::DerivedFact("D1".into())])).unwrap();
        graph.invalidate_source_event("E1", "correction").unwrap();
        assert!(graph
            .derive(fact("D3", [DependencyRef::DerivedFact("D1".into())]))
            .unwrap_err()
            .contains("invalidated fact"));
    }

    #[test]
    fn rejects_unknown_and_self_dependencies() {
        let mut graph = DerivationGraph::default();
        graph.register_source_event("E1").unwrap();
        assert!(graph
            .derive(fact("D1", [DependencyRef::SourceEvent("absent".into())]))
            .unwrap_err()
            .contains("unknown source dependency"));
        assert!(graph
            .derive(fact("D1", [DependencyRef::DerivedFact("D1".into())]))
            .unwrap_err()
            .contains("self"));
    }

    #[test]
    fn source_and_fact_ids_are_checked_for_duplicates() {
        let mut graph = DerivationGraph::default();
        graph.register_source_event("E1").unwrap();
        assert!(graph.register_source_event("E1").unwrap_err().contains("duplicate"));
        graph.derive(fact("D1", [DependencyRef::SourceEvent("E1".into())])).unwrap();
        assert!(graph
            .derive(fact("D1", [DependencyRef::SourceEvent("E1".into())]))
            .unwrap_err()
            .contains("duplicate derived fact"));
    }

    #[test]
    fn invalid_input_does_not_partially_mutate_graph() {
        let mut graph = DerivationGraph::default();
        graph.register_source_event("E1").unwrap();
        let before = graph.state_sha256().unwrap();
        let invalid = DerivedFactInput {
            fact_id: "D1".into(),
            subject: "build-42".into(),
            attribute: "ready".into(),
            value: "true".into(),
            valid_from: "2026-01-01".into(),
            valid_to: Some("2026-01-01".into()),
            rule_id: "rule".into(),
            rule_version: "v1".into(),
            dependencies: make_dependencies([DependencyRef::SourceEvent("E1".into())]),
        };
        assert!(graph.derive(invalid).is_err());
        assert_eq!(graph.state_sha256().unwrap(), before);
    }

    #[test]
    fn reverse_index_rebuild_preserves_state_digest() {
        let mut graph = DerivationGraph::default();
        graph.register_source_event("E1").unwrap();
        graph.register_source_event("E2").unwrap();
        graph.derive(fact("D1", [DependencyRef::SourceEvent("E1".into())])).unwrap();
        graph.derive(fact("D2", [DependencyRef::DerivedFact("D1".into()), DependencyRef::SourceEvent("E2".into())])).unwrap();
        let before = graph.state_sha256().unwrap();
        // Corrupt the materialized index to make this a real rebuild test.
        graph.reverse_dependencies.clear();
        assert!(graph.state_sha256().is_err());
        graph.rebuild_reverse_index().unwrap();
        assert_eq!(graph.state_sha256().unwrap(), before);
    }

    #[test]
    fn rule_version_is_part_of_the_derived_state_identity() {
        let mut graph = DerivationGraph::default();
        graph.register_source_event("E1").unwrap();
        graph.derive(fact("D1", [DependencyRef::SourceEvent("E1".into())])).unwrap();
        let first = graph.state_sha256().unwrap();
        // Rule versions are immutable in a graph: a new rule version requires
        // a new derived fact ID rather than silently rewriting history.
        let mut next = fact("D1-v2", [DependencyRef::SourceEvent("E1".into())]);
        next.rule_version = "v2".into();
        graph.derive(next).unwrap();
        assert_ne!(graph.state_sha256().unwrap(), first);
    }

    #[test]
    fn invalidation_order_and_output_are_deterministic() {
        fn build(reverse: bool) -> DerivationGraph {
            let mut graph = DerivationGraph::default();
            graph.register_source_event("E1").unwrap();
            graph.register_source_event("E2").unwrap();
            if reverse {
                graph.derive(fact("D2", [DependencyRef::SourceEvent("E2".into())])).unwrap();
                graph.derive(fact("D1", [DependencyRef::SourceEvent("E1".into())])).unwrap();
            } else {
                graph.derive(fact("D1", [DependencyRef::SourceEvent("E1".into())])).unwrap();
                graph.derive(fact("D2", [DependencyRef::SourceEvent("E2".into())])).unwrap();
            }
            graph
        }
        let mut left = build(false);
        let mut right = build(true);
        assert_eq!(left.state_sha256().unwrap(), right.state_sha256().unwrap());
        assert_eq!(left.invalidate_source_event("E1", "retraction").unwrap(),
                   right.invalidate_source_event("E1", "retraction").unwrap());
        assert_eq!(left.state_sha256().unwrap(), right.state_sha256().unwrap());
    }

    #[test]
    fn invalidation_reason_must_be_present() {
        let mut graph = DerivationGraph::default();
        graph.register_source_event("E1").unwrap();
        let before = graph.state_sha256().unwrap();
        assert!(graph.invalidate_source_event("E1", "  ").is_err());
        assert_eq!(graph.state_sha256().unwrap(), before);
    }

    #[test]
    fn invalid_intervals_are_rejected() {
        let mut graph = DerivationGraph::default();
        graph.register_source_event("E1").unwrap();
        let mut invalid = fact("D1", [DependencyRef::SourceEvent("E1".into())]);
        invalid.valid_from = "not-a-date".into();
        assert!(graph.derive(invalid).unwrap_err().contains("valid_from"));
    }
}
