// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Alternative-support truth-maintenance reference graph.
//!
//! This module deliberately separates propositions (claims) from justifications
//! (individual proofs). Dependencies inside one justification are conjunctive;
//! different justifications for one proposition are alternatives. A proposition
//! remains supported while at least one admissible justification supports it.
//!
//! This is a valid-time-only, in-memory reference implementation. Trusted
//! transaction-time, source authorization, durable storage and worker
//! integration are intentionally outside this first prototype.

use chrono::NaiveDate;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "kind", content = "id", rename_all = "snake_case")]
pub enum SupportDependency {
    SourceEvent(String),
    Proposition(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ValidInterval {
    /// Inclusive lower bound, formatted YYYY-MM-DD.
    pub valid_from: String,
    /// Optional exclusive upper bound, formatted YYYY-MM-DD.
    pub valid_to: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ParsedInterval {
    start: NaiveDate,
    end: Option<NaiveDate>,
}

impl ValidInterval {
    fn parse(&self) -> Result<ParsedInterval, String> {
        let start = NaiveDate::parse_from_str(&self.valid_from, "%Y-%m-%d")
            .map_err(|error| format!("valid_from must be YYYY-MM-DD: {error}"))?;
        let end = self
            .valid_to
            .as_deref()
            .map(|value| {
                NaiveDate::parse_from_str(value, "%Y-%m-%d")
                    .map_err(|error| format!("valid_to must be YYYY-MM-DD: {error}"))
            })
            .transpose()?;
        if end.is_some_and(|end| end <= start) {
            return Err("valid_to must be after valid_from".into());
        }
        Ok(ParsedInterval { start, end })
    }

    fn from_parsed(interval: ParsedInterval) -> Self {
        Self {
            valid_from: interval.start.format("%Y-%m-%d").to_string(),
            valid_to: interval.end.map(|end| end.format("%Y-%m-%d").to_string()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PropositionInput {
    pub proposition_id: String,
    pub subject: String,
    pub attribute: String,
    pub value: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JustificationInput {
    pub justification_id: String,
    pub proposition_id: String,
    pub rule_id: String,
    pub rule_version: String,
    pub valid_from: String,
    pub valid_to: Option<String>,
    /// Dependencies are ANDed within this one justification. Separate
    /// justifications for the same proposition are OR alternatives.
    pub dependencies: BTreeSet<SupportDependency>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceEventRecord {
    pub valid_interval: ValidInterval,
    pub retracted: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SupportGraph {
    propositions: BTreeMap<String, PropositionInput>,
    source_events: BTreeMap<String, SourceEventRecord>,
    justifications: BTreeMap<String, JustificationInput>,
    retired_rule_versions: BTreeSet<(String, String)>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SupportSnapshot {
    /// Union of all valid support intervals for each proposition.
    pub proposition_intervals: BTreeMap<String, Vec<ValidInterval>>,
    /// Valid support intervals contributed by each individual justification.
    /// Empty means the proof currently has no complete admissible witness.
    pub justification_intervals: BTreeMap<String, Vec<ValidInterval>>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SupportMutation {
    /// Propositions whose supported interval union changed.
    pub affected_propositions: Vec<String>,
    /// Justifications whose support interval changed.
    pub affected_justifications: Vec<String>,
}

fn non_empty(value: &str, field: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        return Err(format!("{field} must not be empty"));
    }
    Ok(())
}

fn validate_proposition(proposition: &PropositionInput) -> Result<(), String> {
    non_empty(&proposition.proposition_id, "proposition_id")?;
    non_empty(&proposition.subject, "subject")?;
    non_empty(&proposition.attribute, "attribute")?;
    non_empty(&proposition.value, "value")
}

fn validate_justification(justification: &JustificationInput) -> Result<(), String> {
    non_empty(&justification.justification_id, "justification_id")?;
    non_empty(&justification.proposition_id, "proposition_id")?;
    non_empty(&justification.rule_id, "rule_id")?;
    non_empty(&justification.rule_version, "rule_version")?;
    if justification.dependencies.is_empty() {
        return Err(format!(
            "justification {} must have at least one dependency",
            justification.justification_id
        ));
    }
    ValidInterval {
        valid_from: justification.valid_from.clone(),
        valid_to: justification.valid_to.clone(),
    }
    .parse()?;
    Ok(())
}

fn min_end(left: Option<NaiveDate>, right: Option<NaiveDate>) -> Option<NaiveDate> {
    match (left, right) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

fn max_end(left: Option<NaiveDate>, right: Option<NaiveDate>) -> Option<NaiveDate> {
    match (left, right) {
        (Some(a), Some(b)) => Some(a.max(b)),
        (Some(_), None) | (None, Some(_)) | (None, None) => None,
    }
}

fn intersects(left: ParsedInterval, right: ParsedInterval) -> Option<ParsedInterval> {
    let start = left.start.max(right.start);
    let end = min_end(left.end, right.end);
    if end.is_some_and(|end| end <= start) {
        None
    } else {
        Some(ParsedInterval { start, end })
    }
}

fn normalize_intervals(mut intervals: Vec<ParsedInterval>) -> Vec<ParsedInterval> {
    intervals.sort_by_key(|interval| interval.start);
    let mut normalized: Vec<ParsedInterval> = Vec::new();

    for next in intervals {
        let Some(previous) = normalized.last_mut() else {
            normalized.push(next);
            continue;
        };

        // Adjacent half-open intervals are continuous: [a,b) U [b,c) = [a,c).
        let overlaps_or_touches = previous.end.is_none_or(|end| next.start <= end);
        if overlaps_or_touches {
            previous.end = max_end(previous.end, next.end);
        } else {
            normalized.push(next);
        }
    }

    normalized
}

fn intersect_interval_sets(
    left: &[ParsedInterval],
    right: &[ParsedInterval],
) -> Vec<ParsedInterval> {
    let mut intersections = Vec::new();
    for a in left {
        for b in right {
            if let Some(overlap) = intersects(*a, *b) {
                intersections.push(overlap);
            }
        }
    }
    normalize_intervals(intersections)
}

fn union_interval_sets(
    left: &[ParsedInterval],
    right: &[ParsedInterval],
) -> Vec<ParsedInterval> {
    normalize_intervals(left.iter().chain(right.iter()).copied().collect())
}

impl SupportSnapshot {
    pub fn is_supported_at(&self, proposition_id: &str, valid_at: &str) -> Result<bool, String> {
        let date = NaiveDate::parse_from_str(valid_at, "%Y-%m-%d")
            .map_err(|error| format!("valid_at must be YYYY-MM-DD: {error}"))?;
        let Some(intervals) = self.proposition_intervals.get(proposition_id) else {
            return Ok(false);
        };
        for interval in intervals {
            let parsed = interval.parse()?;
            if parsed.start <= date && parsed.end.is_none_or(|end| date < end) {
                return Ok(true);
            }
        }
        Ok(false)
    }
}

impl SupportGraph {
    pub fn register_source_event(
        &mut self,
        event_id: impl Into<String>,
        valid_interval: ValidInterval,
    ) -> Result<(), String> {
        let event_id = event_id.into();
        non_empty(&event_id, "source event ID")?;
        valid_interval.parse()?;
        if self.source_events.contains_key(&event_id) {
            return Err(format!("duplicate source event ID {event_id:?}"));
        }

        let mut candidate = self.clone();
        candidate.source_events.insert(
            event_id,
            SourceEventRecord {
                valid_interval,
                retracted: false,
            },
        );
        candidate.verify_consistency()?;
        *self = candidate;
        Ok(())
    }

    pub fn register_proposition(&mut self, proposition: PropositionInput) -> Result<(), String> {
        validate_proposition(&proposition)?;
        if self.propositions.contains_key(&proposition.proposition_id) {
            return Err(format!(
                "duplicate proposition ID {:?}",
                proposition.proposition_id
            ));
        }

        let mut candidate = self.clone();
        candidate
            .propositions
            .insert(proposition.proposition_id.clone(), proposition);
        candidate.verify_consistency()?;
        *self = candidate;
        Ok(())
    }

    pub fn add_justification(&mut self, justification: JustificationInput) -> Result<(), String> {
        validate_justification(&justification)?;
        if self
            .justifications
            .contains_key(&justification.justification_id)
        {
            return Err(format!(
                "duplicate justification ID {:?}",
                justification.justification_id
            ));
        }
        if self.retired_rule_versions.contains(&(
            justification.rule_id.clone(),
            justification.rule_version.clone(),
        )) {
            return Err(format!(
                "rule version {}@{} is retired",
                justification.rule_id, justification.rule_version
            ));
        }

        let mut candidate = self.clone();
        candidate.justifications.insert(
            justification.justification_id.clone(),
            justification,
        );
        candidate.verify_consistency()?;
        *self = candidate;
        Ok(())
    }

    /// Compute support by monotone fixed-point iteration over an acyclic
    /// proposition dependency graph. The graph verifier rejects cycles so
    /// support cannot be sustained by an ungrounded circular justification.
    pub fn support_snapshot(&self) -> Result<SupportSnapshot, String> {
        self.verify_consistency()?;

        let mut proposition_intervals: BTreeMap<String, Vec<ParsedInterval>> = self
            .propositions
            .keys()
            .map(|id| (id.clone(), Vec::new()))
            .collect();

        let mut final_justification_intervals = BTreeMap::new();
        let max_rounds = self.propositions.len().saturating_add(2);

        for _ in 0..max_rounds {
            let mut next_proposition_intervals: BTreeMap<String, Vec<ParsedInterval>> = self
                .propositions
                .keys()
                .map(|id| (id.clone(), Vec::new()))
                .collect();
            let mut justification_intervals = BTreeMap::new();

            for (justification_id, justification) in &self.justifications {
                let rule_key = (
                    justification.rule_id.clone(),
                    justification.rule_version.clone(),
                );
                if self.retired_rule_versions.contains(&rule_key) {
                    justification_intervals.insert(justification_id.clone(), Vec::new());
                    continue;
                }

                let own_interval = ValidInterval {
                    valid_from: justification.valid_from.clone(),
                    valid_to: justification.valid_to.clone(),
                }
                .parse()?;
                let mut support = vec![own_interval];

                for dependency in &justification.dependencies {
                    let dependency_intervals = match dependency {
                        SupportDependency::SourceEvent(event_id) => {
                            let source = self
                                .source_events
                                .get(event_id)
                                .ok_or_else(|| format!("missing source event {event_id:?}"))?;
                            if source.retracted {
                                Vec::new()
                            } else {
                                vec![source.valid_interval.parse()?]
                            }
                        }
                        SupportDependency::Proposition(proposition_id) => {
                            proposition_intervals
                                .get(proposition_id)
                                .cloned()
                                .ok_or_else(|| {
                                    format!("missing proposition {proposition_id:?}")
                                })?
                        }
                    };

                    support = intersect_interval_sets(&support, &dependency_intervals);
                    if support.is_empty() {
                        break;
                    }
                }

                let support = normalize_intervals(support);
                let proposition_support = next_proposition_intervals
                    .get_mut(&justification.proposition_id)
                    .ok_or_else(|| {
                        format!(
                            "missing output proposition {:?}",
                            justification.proposition_id
                        )
                    })?;
                *proposition_support = union_interval_sets(proposition_support, &support);
                justification_intervals.insert(
                    justification_id.clone(),
                    support,
                );
            }

            let stable = next_proposition_intervals == proposition_intervals;
            proposition_intervals = next_proposition_intervals;
            final_justification_intervals = justification_intervals;
            if stable {
                let proposition_intervals = proposition_intervals
                    .into_iter()
                    .map(|(id, intervals)| {
                        (
                            id,
                            intervals
                                .into_iter()
                                .map(ValidInterval::from_parsed)
                                .collect(),
                        )
                    })
                    .collect();
                let justification_intervals = final_justification_intervals
                    .into_iter()
                    .map(|(id, intervals)| {
                        (
                            id,
                            intervals
                                .into_iter()
                                .map(ValidInterval::from_parsed)
                                .collect(),
                        )
                    })
                    .collect();
                return Ok(SupportSnapshot {
                    proposition_intervals,
                    justification_intervals,
                });
            }
        }

        Err("support fixed-point did not converge within the acyclic graph bound".into())
    }

    /// Retraction changes source eligibility but keeps the original record.
    /// Alternative justifications may preserve a proposition's support.
    pub fn retract_source_event(
        &mut self,
        event_id: &str,
    ) -> Result<SupportMutation, String> {
        let source = self
            .source_events
            .get(event_id)
            .ok_or_else(|| format!("cannot retract unknown source event {event_id:?}"))?;
        let _ = source;
        let before = self.support_snapshot()?;
        if self.source_events.get(event_id).is_some_and(|source| source.retracted) {
            return Ok(SupportMutation::default());
        }

        let mut candidate = self.clone();
        candidate
            .source_events
            .get_mut(event_id)
            .expect("validated source event exists")
            .retracted = true;
        candidate.verify_consistency()?;
        let after = candidate.support_snapshot()?;
        let mutation = support_difference(&before, &after);
        *self = candidate;
        Ok(mutation)
    }

    /// Retire one rule version. Other versions and alternative rules remain
    /// eligible; dependent propositions survive if another proof supports them.
    pub fn invalidate_rule_version(
        &mut self,
        rule_id: &str,
        rule_version: &str,
    ) -> Result<SupportMutation, String> {
        non_empty(rule_id, "rule ID")?;
        non_empty(rule_version, "rule version")?;
        let before = self.support_snapshot()?;
        let key = (rule_id.to_owned(), rule_version.to_owned());
        if self.retired_rule_versions.contains(&key) {
            return Ok(SupportMutation::default());
        }

        let mut candidate = self.clone();
        candidate.retired_rule_versions.insert(key);
        candidate.verify_consistency()?;
        let after = candidate.support_snapshot()?;
        let mutation = support_difference(&before, &after);
        *self = candidate;
        Ok(mutation)
    }

    pub fn conflicting_propositions_at(
        &self,
        subject: &str,
        attribute: &str,
        valid_at: &str,
    ) -> Result<Vec<String>, String> {
        let snapshot = self.support_snapshot()?;
        let mut supported: Vec<(String, String)> = Vec::new();
        for (id, proposition) in &self.propositions {
            if proposition.subject == subject && proposition.attribute == attribute
                && snapshot.is_supported_at(id, valid_at)?
            {
                supported.push((id.clone(), proposition.value.clone()));
            }
        }
        let distinct_values: BTreeSet<String> =
            supported.iter().map(|(_, value)| value.clone()).collect();
        if distinct_values.len() > 1 {
            Ok(supported.into_iter().map(|(id, _)| id).collect())
        } else {
            Ok(Vec::new())
        }
    }

    pub fn proposition(&self, proposition_id: &str) -> Option<&PropositionInput> {
        self.propositions.get(proposition_id)
    }

    pub fn justification(&self, justification_id: &str) -> Option<&JustificationInput> {
        self.justifications.get(justification_id)
    }

    pub fn verify_consistency(&self) -> Result<(), String> {
        for (event_id, source) in &self.source_events {
            non_empty(event_id, "source event ID")?;
            source.valid_interval.parse()?;
        }

        for (id, proposition) in &self.propositions {
            validate_proposition(proposition)?;
            if id != &proposition.proposition_id {
                return Err(format!("proposition index mismatch for {id:?}"));
            }
        }

        for (id, justification) in &self.justifications {
            validate_justification(justification)?;
            if id != &justification.justification_id {
                return Err(format!("justification index mismatch for {id:?}"));
            }
            if !self.propositions.contains_key(&justification.proposition_id) {
                return Err(format!(
                    "justification {id:?} references unknown output proposition {:?}",
                    justification.proposition_id
                ));
            }
            for dependency in &justification.dependencies {
                match dependency {
                    SupportDependency::SourceEvent(event_id) => {
                        if !self.source_events.contains_key(event_id) {
                            return Err(format!(
                                "justification {id:?} references unknown source {event_id:?}"
                            ));
                        }
                    }
                    SupportDependency::Proposition(proposition_id) => {
                        if !self.propositions.contains_key(proposition_id) {
                            return Err(format!(
                                "justification {id:?} references unknown proposition {proposition_id:?}"
                            ));
                        }
                        if proposition_id == &justification.proposition_id {
                            return Err(format!(
                                "justification {id:?} has a self-supporting proposition dependency"
                            ));
                        }
                    }
                }
            }
        }

        // Detect cycles across the proposition dependency graph. A proposition
        // may have many justifications; every dependency edge is included.
        fn visit(
            proposition_id: &str,
            graph: &SupportGraph,
            marks: &mut BTreeMap<String, u8>,
        ) -> Result<(), String> {
            match marks.get(proposition_id).copied() {
                Some(1) => {
                    return Err(format!(
                        "cycle detected in alternative-support graph at {proposition_id:?}"
                    ));
                }
                Some(2) => return Ok(()),
                _ => {}
            }

            marks.insert(proposition_id.to_owned(), 1);
            for justification in graph
                .justifications
                .values()
                .filter(|item| item.proposition_id == proposition_id)
            {
                for dependency in &justification.dependencies {
                    if let SupportDependency::Proposition(parent_id) = dependency {
                        visit(parent_id, graph, marks)?;
                    }
                }
            }
            marks.insert(proposition_id.to_owned(), 2);
            Ok(())
        }

        let mut marks = BTreeMap::new();
        for proposition_id in self.propositions.keys() {
            visit(proposition_id, self, &mut marks)?;
        }
        Ok(())
    }

    /// Deterministic digest of canonical inputs, retractions, retired rule
    /// versions, and the materialized support snapshot. It is not a signature.
    pub fn state_sha256(&self) -> Result<String, String> {
        self.verify_consistency()?;
        let snapshot = self.support_snapshot()?;
        let bytes = serde_json::to_vec(&(
            &self.propositions,
            &self.source_events,
            &self.justifications,
            &self.retired_rule_versions,
            snapshot,
        ))
        .map_err(|error| format!("cannot serialize support graph: {error}"))?;
        Ok(format!("{:x}", Sha256::digest(bytes)))
    }
}

fn support_difference(
    before: &SupportSnapshot,
    after: &SupportSnapshot,
) -> SupportMutation {
    let proposition_ids: BTreeSet<String> = before
        .proposition_intervals
        .keys()
        .chain(after.proposition_intervals.keys())
        .cloned()
        .collect();
    let affected_propositions = proposition_ids
        .into_iter()
        .filter(|id| {
            before.proposition_intervals.get(id) != after.proposition_intervals.get(id)
        })
        .collect();

    let justification_ids: BTreeSet<String> = before
        .justification_intervals
        .keys()
        .chain(after.justification_intervals.keys())
        .cloned()
        .collect();
    let affected_justifications = justification_ids
        .into_iter()
        .filter(|id| {
            before.justification_intervals.get(id) != after.justification_intervals.get(id)
        })
        .collect();

    SupportMutation {
        affected_propositions,
        affected_justifications,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn interval(start: &str, end: Option<&str>) -> ValidInterval {
        ValidInterval {
            valid_from: start.into(),
            valid_to: end.map(str::to_owned),
        }
    }

    fn proposition(id: &str, subject: &str, attribute: &str, value: &str) -> PropositionInput {
        PropositionInput {
            proposition_id: id.into(),
            subject: subject.into(),
            attribute: attribute.into(),
            value: value.into(),
        }
    }

    fn justification(
        id: &str,
        output: &str,
        start: &str,
        end: Option<&str>,
        dependencies: impl IntoIterator<Item = SupportDependency>,
    ) -> JustificationInput {
        JustificationInput {
            justification_id: id.into(),
            proposition_id: output.into(),
            rule_id: "rule".into(),
            rule_version: "v1".into(),
            valid_from: start.into(),
            valid_to: end.map(str::to_owned),
            dependencies: dependencies.into_iter().collect(),
        }
    }

    fn source(graph: &mut SupportGraph, id: &str, start: &str, end: Option<&str>) {
        graph.register_source_event(id, interval(start, end)).unwrap();
    }

    #[test]
    fn retracting_one_alternative_proof_preserves_proposition_and_dependents() {
        let mut graph = SupportGraph::default();
        source(&mut graph, "E1", "2026-01-01", None);
        source(&mut graph, "E2", "2026-01-01", None);
        graph.register_proposition(proposition("P-ready", "build-42", "ready", "true")).unwrap();
        graph.register_proposition(proposition("P-release", "build-42", "release_allowed", "true")).unwrap();

        graph.add_justification(justification(
            "J1", "P-ready", "2026-01-01", None,
            [SupportDependency::SourceEvent("E1".into())],
        )).unwrap();
        graph.add_justification(justification(
            "J2", "P-ready", "2026-01-01", None,
            [SupportDependency::SourceEvent("E2".into())],
        )).unwrap();
        graph.add_justification(justification(
            "J3", "P-release", "2026-01-01", None,
            [SupportDependency::Proposition("P-ready".into())],
        )).unwrap();

        let mutation = graph.retract_source_event("E1").unwrap();
        assert!(mutation.affected_propositions.is_empty());
        assert_eq!(mutation.affected_justifications, vec!["J1"]);
        let snapshot = graph.support_snapshot().unwrap();
        assert!(snapshot.is_supported_at("P-ready", "2026-01-15").unwrap());
        assert!(snapshot.is_supported_at("P-release", "2026-01-15").unwrap());
        assert!(snapshot.justification_intervals["J1"].is_empty());
        assert!(!snapshot.justification_intervals["J2"].is_empty());
        assert!(!snapshot.justification_intervals["J3"].is_empty());
    }

    #[test]
    fn retracting_last_alternative_support_invalidates_dependent_state() {
        let mut graph = SupportGraph::default();
        source(&mut graph, "E1", "2026-01-01", None);
        source(&mut graph, "E2", "2026-01-01", None);
        graph.register_proposition(proposition("P-ready", "build-42", "ready", "true")).unwrap();
        graph.register_proposition(proposition("P-release", "build-42", "release_allowed", "true")).unwrap();
        graph.add_justification(justification(
            "J1", "P-ready", "2026-01-01", None,
            [SupportDependency::SourceEvent("E1".into())],
        )).unwrap();
        graph.add_justification(justification(
            "J2", "P-ready", "2026-01-01", None,
            [SupportDependency::SourceEvent("E2".into())],
        )).unwrap();
        graph.add_justification(justification(
            "J3", "P-release", "2026-01-01", None,
            [SupportDependency::Proposition("P-ready".into())],
        )).unwrap();

        graph.retract_source_event("E1").unwrap();
        let mutation = graph.retract_source_event("E2").unwrap();
        assert_eq!(mutation.affected_propositions, vec!["P-ready", "P-release"]);
        let snapshot = graph.support_snapshot().unwrap();
        assert!(!snapshot.is_supported_at("P-ready", "2026-01-15").unwrap());
        assert!(!snapshot.is_supported_at("P-release", "2026-01-15").unwrap());
    }

    #[test]
    fn dependencies_are_conjunctive_within_a_justification_and_alternative_across_proofs() {
        let mut graph = SupportGraph::default();
        for id in ["E1", "E2", "E3"] {
            source(&mut graph, id, "2026-01-01", None);
        }
        graph.register_proposition(proposition("P-build", "build-42", "ready", "true")).unwrap();
        graph.add_justification(justification(
            "J1", "P-build", "2026-01-01", None,
            [
                SupportDependency::SourceEvent("E1".into()),
                SupportDependency::SourceEvent("E2".into()),
            ],
        )).unwrap();
        graph.add_justification(justification(
            "J2", "P-build", "2026-01-01", None,
            [SupportDependency::SourceEvent("E3".into())],
        )).unwrap();

        graph.retract_source_event("E1").unwrap();
        let snapshot = graph.support_snapshot().unwrap();
        assert!(snapshot.is_supported_at("P-build", "2026-01-15").unwrap());
        assert!(snapshot.justification_intervals["J1"].is_empty());
        assert!(!snapshot.justification_intervals["J2"].is_empty());
    }

    #[test]
    fn retiring_one_rule_version_preserves_an_alternative_rule() {
        let mut graph = SupportGraph::default();
        source(&mut graph, "E1", "2026-01-01", None);
        source(&mut graph, "E2", "2026-01-01", None);
        graph.register_proposition(proposition("P-compliant", "release-42", "compliant", "true")).unwrap();

        let mut old = justification(
            "J-old", "P-compliant", "2026-01-01", None,
            [SupportDependency::SourceEvent("E1".into())],
        );
        old.rule_id = "compliance-policy".into();
        old.rule_version = "v1".into();
        graph.add_justification(old).unwrap();

        let mut other = justification(
            "J-current", "P-compliant", "2026-01-01", None,
            [SupportDependency::SourceEvent("E2".into())],
        );
        other.rule_id = "independent-attestation".into();
        other.rule_version = "v3".into();
        graph.add_justification(other).unwrap();

        let mutation = graph.invalidate_rule_version("compliance-policy", "v1").unwrap();
        assert!(mutation.affected_propositions.is_empty());
        assert_eq!(mutation.affected_justifications, vec!["J-old"]);
        assert!(graph.support_snapshot().unwrap().is_supported_at("P-compliant", "2026-01-15").unwrap());
    }

    #[test]
    fn opposing_supported_values_remain_an_explicit_conflict() {
        let mut graph = SupportGraph::default();
        source(&mut graph, "E1", "2026-01-01", None);
        source(&mut graph, "E2", "2026-01-01", None);
        graph.register_proposition(proposition("P-true", "build-42", "ready", "true")).unwrap();
        graph.register_proposition(proposition("P-false", "build-42", "ready", "false")).unwrap();
        graph.add_justification(justification(
            "J-true", "P-true", "2026-01-01", None,
            [SupportDependency::SourceEvent("E1".into())],
        )).unwrap();
        graph.add_justification(justification(
            "J-false", "P-false", "2026-01-01", None,
            [SupportDependency::SourceEvent("E2".into())],
        )).unwrap();

        assert_eq!(
            graph.conflicting_propositions_at("build-42", "ready", "2026-01-15").unwrap(),
            vec!["P-false", "P-true"]
        );
    }

    #[test]
    fn adjacent_justification_intervals_form_a_contiguous_support_union() {
        let mut graph = SupportGraph::default();
        source(&mut graph, "E-jan", "2026-01-01", Some("2026-02-01"));
        source(&mut graph, "E-feb", "2026-02-01", Some("2026-03-01"));
        graph.register_proposition(proposition("P-up", "service-x", "available", "true")).unwrap();
        graph.add_justification(justification(
            "J-jan", "P-up", "2026-01-01", Some("2026-02-01"),
            [SupportDependency::SourceEvent("E-jan".into())],
        )).unwrap();
        graph.add_justification(justification(
            "J-feb", "P-up", "2026-02-01", Some("2026-03-01"),
            [SupportDependency::SourceEvent("E-feb".into())],
        )).unwrap();

        let snapshot = graph.support_snapshot().unwrap();
        assert_eq!(
            snapshot.proposition_intervals["P-up"],
            vec![interval("2026-01-01", Some("2026-03-01"))]
        );
        assert!(!snapshot.is_supported_at("P-up", "2026-03-01").unwrap());
        assert!(snapshot.is_supported_at("P-up", "2026-02-01").unwrap());
    }

    #[test]
    fn parent_support_intervals_intersect_child_justification_interval() {
        let mut graph = SupportGraph::default();
        source(&mut graph, "E-parent", "2026-01-01", Some("2026-02-01"));
        source(&mut graph, "E-child", "2026-01-15", Some("2026-03-01"));
        graph.register_proposition(proposition("P-parent", "x", "parent", "true")).unwrap();
        graph.register_proposition(proposition("P-child", "x", "child", "true")).unwrap();
        graph.add_justification(justification(
            "J-parent", "P-parent", "2026-01-01", None,
            [SupportDependency::SourceEvent("E-parent".into())],
        )).unwrap();
        graph.add_justification(justification(
            "J-child", "P-child", "2026-01-01", None,
            [
                SupportDependency::SourceEvent("E-child".into()),
                SupportDependency::Proposition("P-parent".into()),
            ],
        )).unwrap();

        let snapshot = graph.support_snapshot().unwrap();
        assert_eq!(
            snapshot.justification_intervals["J-child"],
            vec![interval("2026-01-15", Some("2026-02-01"))]
        );
    }

    #[test]
    fn rejects_proposition_dependency_cycles_without_mutating_state() {
        let mut graph = SupportGraph::default();
        source(&mut graph, "E1", "2026-01-01", None);
        graph.register_proposition(proposition("P-a", "x", "a", "true")).unwrap();
        graph.register_proposition(proposition("P-b", "x", "b", "true")).unwrap();
        graph.add_justification(justification(
            "J-a-root", "P-a", "2026-01-01", None,
            [SupportDependency::SourceEvent("E1".into())],
        )).unwrap();
        graph.add_justification(justification(
            "J-b-from-a", "P-b", "2026-01-01", None,
            [SupportDependency::Proposition("P-a".into())],
        )).unwrap();

        let before = graph.state_sha256().unwrap();
        let error = graph.add_justification(justification(
            "J-a-from-b", "P-a", "2026-01-01", None,
            [SupportDependency::Proposition("P-b".into())],
        )).unwrap_err();
        assert!(error.contains("cycle detected"));
        assert_eq!(graph.state_sha256().unwrap(), before);
    }

    #[test]
    fn invalid_retraction_is_idempotent_and_state_digest_is_deterministic() {
        fn build(reverse: bool) -> SupportGraph {
            let mut graph = SupportGraph::default();
            let mut ids = vec!["E1", "E2"];
            if reverse { ids.reverse(); }
            for id in ids {
                source(&mut graph, id, "2026-01-01", None);
            }
            graph.register_proposition(proposition("P", "x", "ready", "true")).unwrap();
            for id in ["J1", "J2"] {
                let event = if id == "J1" { "E1" } else { "E2" };
                graph.add_justification(justification(
                    id, "P", "2026-01-01", None,
                    [SupportDependency::SourceEvent(event.into())],
                )).unwrap();
            }
            graph
        }

        let mut left = build(false);
        let mut right = build(true);
        assert_eq!(left.state_sha256().unwrap(), right.state_sha256().unwrap());
        left.retract_source_event("E1").unwrap();
        right.retract_source_event("E1").unwrap();
        assert_eq!(left.state_sha256().unwrap(), right.state_sha256().unwrap());
        assert_eq!(left.retract_source_event("E1").unwrap(), SupportMutation::default());
    }
}
