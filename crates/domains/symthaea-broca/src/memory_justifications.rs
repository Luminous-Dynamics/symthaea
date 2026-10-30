// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Reference semantics for alternative sufficient justifications in evolving memory.
//!
//! Unlike a single dependency set (which is conjunctive), one conclusion may have
//! multiple justification sets. Dependencies within a set are AND; the sets are OR.
//! This module is a deterministic research oracle, not a persistence adapter.
//!
//! A conclusion is Supported when at least one complete justification is supported.
//! Otherwise it is Unsupported; this state is deliberately not interpreted as False.
//! No negation is inferred merely because evidence is absent. Any closed-world
//! negation rule must be explicit and provide its own scoped completeness witness.

use chrono::NaiveDate;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "kind", content = "id", rename_all = "snake_case")]
pub enum SupportRef {
    SourceEvent(String),
    DerivedFact(String),
}

/// One conjunctive proof for a conclusion. Alternative values in
/// `DerivedConclusionInput::justifications` are disjunctive alternatives.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JustificationInput {
    pub justification_id: String,
    pub dependencies: BTreeSet<SupportRef>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DerivedConclusionInput {
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
    /// OR across justification sets; AND within each set.
    pub justifications: Vec<JustificationInput>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum SupportStatus {
    Supported {
        /// Every currently complete proof, in stable lexical order.
        active_justification_ids: BTreeSet<String>,
    },
    /// No complete eligible proof currently supports this conclusion.
    /// This is unknown/unsupported, not a derivation of the proposition's negation.
    Unsupported,
}

impl SupportStatus {
    pub fn is_supported(&self) -> bool {
        matches!(self, Self::Supported { .. })
    }

    pub fn active_justification_ids(&self) -> BTreeSet<String> {
        match self {
            Self::Supported {
                active_justification_ids,
            } => active_justification_ids.clone(),
            Self::Unsupported => BTreeSet::new(),
        }
    }
}

/// Query-time support state under valid-time semantics.
///
/// `OutsideValidity` is distinct from `Unsupported`: a conclusion may have
/// structurally valid proofs but not apply at the requested date. Neither state
/// means that the proposition is false.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum AsOfSupportStatus {
    Supported {
        active_justification_ids: BTreeSet<String>,
    },
    Unsupported,
    OutsideValidity,
}

impl AsOfSupportStatus {
    pub fn is_supported(&self) -> bool {
        matches!(self, Self::Supported { .. })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DerivedConclusionRecord {
    pub conclusion: DerivedConclusionInput,
    pub status: SupportStatus,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RetractionReport {
    pub source_event_id: String,
    /// Facts that moved from Supported to Unsupported after the retraction.
    pub newly_unsupported_fact_ids: Vec<String>,
    /// Previously supported facts that remain supported through other justifications.
    pub retained_with_alternative_support: Vec<String>,
    /// Justifications that ceased to be complete, grouped by conclusion.
    pub inactive_justifications_by_fact: BTreeMap<String, Vec<String>>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct JustificationGraph {
    source_events: BTreeSet<String>,
    retracted_source_events: BTreeSet<String>,
    facts: BTreeMap<String, DerivedConclusionRecord>,
    /// Insertion order is a topological order: a justification may reference
    /// only source events or derived conclusions registered before its own fact.
    fact_order: Vec<String>,
}

fn non_empty(value: &str, field: &str) -> Result<(), String> {
    if value.trim().is_empty() || value.trim() != value {
        return Err(format!("{field} must be non-empty and trimmed"));
    }
    Ok(())
}

fn validate_date_interval(start: &str, end: Option<&str>) -> Result<(), String> {
    let start_date = NaiveDate::parse_from_str(start, "%Y-%m-%d")
        .map_err(|error| format!("valid_from must be YYYY-MM-DD: {error}"))?;
    if let Some(end) = end {
        let end_date = NaiveDate::parse_from_str(end, "%Y-%m-%d")
            .map_err(|error| format!("valid_to must be YYYY-MM-DD: {error}"))?;
        if end_date <= start_date {
            return Err("valid_to must be after valid_from".into());
        }
    }
    Ok(())
}

fn validate_conclusion_input(input: &DerivedConclusionInput) -> Result<(), String> {
    for (field, value) in [
        ("fact_id", input.fact_id.as_str()),
        ("subject", input.subject.as_str()),
        ("attribute", input.attribute.as_str()),
        ("value", input.value.as_str()),
        ("rule_id", input.rule_id.as_str()),
        ("rule_version", input.rule_version.as_str()),
    ] {
        non_empty(value, field)?;
    }
    validate_date_interval(&input.valid_from, input.valid_to.as_deref())?;
    if input.justifications.is_empty() {
        return Err("a derived conclusion must have at least one justification".into());
    }

    let mut justification_ids = BTreeSet::new();
    for justification in &input.justifications {
        non_empty(&justification.justification_id, "justification_id")?;
        if !justification_ids.insert(justification.justification_id.as_str()) {
            return Err(format!(
                "duplicate justification ID {:?} for fact {:?}",
                justification.justification_id, input.fact_id
            ));
        }
        if justification.dependencies.is_empty() {
            return Err(format!(
                "justification {:?} for fact {:?} must contain at least one dependency",
                justification.justification_id, input.fact_id
            ));
        }
        for dependency in &justification.dependencies {
            let (kind, id) = match dependency {
                SupportRef::SourceEvent(id) => ("source event ID", id.as_str()),
                SupportRef::DerivedFact(id) => ("derived fact ID", id.as_str()),
            };
            non_empty(id, kind)?;
        }
    }
    Ok(())
}

impl JustificationGraph {
    /// Register an immutable source event identity. Retraction preserves the ID
    /// in the audit set rather than deleting history.
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

    /// Insert a conclusion with one or more alternative justification sets.
    ///
    /// Every source/derived dependency must already exist and be Supported on
    /// insertion. Derived dependencies must precede the new fact, so this API
    /// constructs a DAG by construction. No empty proof is accepted as a way
    /// to derive a conclusion from absence.
    pub fn derive(&mut self, conclusion: DerivedConclusionInput) -> Result<(), String> {
        validate_conclusion_input(&conclusion)?;
        if self.facts.contains_key(&conclusion.fact_id) {
            return Err(format!("duplicate derived fact ID {:?}", conclusion.fact_id));
        }

        for justification in &conclusion.justifications {
            for dependency in &justification.dependencies {
                match dependency {
                    SupportRef::SourceEvent(event_id) => {
                        if !self.source_events.contains(event_id) {
                            return Err(format!("unknown source dependency {event_id:?}"));
                        }
                        if self.retracted_source_events.contains(event_id) {
                            return Err(format!("cannot derive from retracted source event {event_id:?}"));
                        }
                    }
                    SupportRef::DerivedFact(parent_id) => {
                        if parent_id == &conclusion.fact_id {
                            return Err(format!("derived fact {:?} cannot depend on itself", conclusion.fact_id));
                        }
                        let Some(parent) = self.facts.get(parent_id) else {
                            return Err(format!("unknown derived dependency {parent_id:?}"));
                        };
                        if !parent.status.is_supported() {
                            return Err(format!("cannot derive from unsupported fact {parent_id:?}"));
                        }
                    }
                }
            }
        }

        let id = conclusion.fact_id.clone();
        let mut candidate = self.clone();
        candidate.facts.insert(
            id.clone(),
            DerivedConclusionRecord {
                conclusion,
                // Recomputed below from the candidate's actual source and parent status.
                status: SupportStatus::Unsupported,
            },
        );
        candidate.fact_order.push(id);
        candidate.recompute_statuses()?;
        candidate.verify_consistency()?;
        *self = candidate;
        Ok(())
    }

    /// Retract a source event and recompute all conclusion support in topological
    /// order. Conclusions with any surviving complete proof remain Supported.
    /// Replaying the same retraction is idempotent.
    pub fn invalidate_source_event(
        &mut self,
        event_id: &str,
    ) -> Result<RetractionReport, String> {
        non_empty(event_id, "source event ID")?;
        if !self.source_events.contains(event_id) {
            return Err(format!("cannot retract unknown source event {event_id:?}"));
        }
        self.verify_consistency()?;
        if self.retracted_source_events.contains(event_id) {
            return Ok(RetractionReport {
                source_event_id: event_id.to_owned(),
                newly_unsupported_fact_ids: Vec::new(),
                retained_with_alternative_support: Vec::new(),
                inactive_justifications_by_fact: BTreeMap::new(),
            });
        }

        let before: BTreeMap<String, SupportStatus> = self
            .facts
            .iter()
            .map(|(id, record)| (id.clone(), record.status.clone()))
            .collect();
        let mut candidate = self.clone();
        candidate.retracted_source_events.insert(event_id.to_owned());
        candidate.recompute_statuses()?;
        candidate.verify_consistency()?;

        let mut newly_unsupported_fact_ids = Vec::new();
        let mut retained_with_alternative_support = Vec::new();
        let mut inactive_justifications_by_fact = BTreeMap::new();

        for fact_id in &candidate.fact_order {
            let old_status = before.get(fact_id).expect("fact existed before candidate clone");
            let new_status = &candidate.facts.get(fact_id).expect("fact order is consistent").status;
            if !old_status.is_supported() {
                continue;
            }

            let old_active = old_status.active_justification_ids();
            let new_active = new_status.active_justification_ids();
            let lost: Vec<String> = old_active.difference(&new_active).cloned().collect();
            if !lost.is_empty() {
                inactive_justifications_by_fact.insert(fact_id.clone(), lost);
            }
            match (old_status.is_supported(), new_status.is_supported()) {
                (true, false) => newly_unsupported_fact_ids.push(fact_id.clone()),
                (true, true) if !old_active.is_subset(&new_active) => {
                    retained_with_alternative_support.push(fact_id.clone());
                }
                _ => {}
            }
        }

        *self = candidate;
        Ok(RetractionReport {
            source_event_id: event_id.to_owned(),
            newly_unsupported_fact_ids,
            retained_with_alternative_support,
            inactive_justifications_by_fact,
        })
    }

    pub fn fact(&self, fact_id: &str) -> Option<&DerivedConclusionRecord> {
        self.facts.get(fact_id)
    }

    /// Return only supported conclusions. Unsupported records remain inspectable
    /// through `fact` for audit and must not be treated as false assertions.
    pub fn current_fact(&self, fact_id: &str) -> Option<&DerivedConclusionRecord> {
        self.facts.get(fact_id).filter(|record| record.status.is_supported())
    }

    /// Recompute query-time support for every conclusion using valid-time intervals.
    ///
    /// Intervals are half-open: valid_from <= as_of < valid_to. A conclusion outside
    /// its own interval is OutsideValidity; a conclusion whose interval applies but
    /// whose justifications are all ineligible is Unsupported. Derived dependencies
    /// must themselves be Supported at the same as-of date.
    ///
    /// Source-event IDs in this research model are timeless until retracted. This
    /// method therefore does not claim bitemporal source-event semantics; source
    /// valid-time and transaction-time metadata require a richer source-event record.
    pub fn support_statuses_at(
        &self,
        as_of: &str,
    ) -> Result<BTreeMap<String, AsOfSupportStatus>, String> {
        let as_of_date = NaiveDate::parse_from_str(as_of, "%Y-%m-%d")
            .map_err(|error| format!("as_of must be YYYY-MM-DD: {error}"))?;
        self.verify_consistency()?;

        let mut statuses = BTreeMap::<String, AsOfSupportStatus>::new();
        for fact_id in &self.fact_order {
            let record = self
                .facts
                .get(fact_id)
                .ok_or_else(|| format!("fact order references missing fact {fact_id:?}"))?;
            let start = NaiveDate::parse_from_str(&record.conclusion.valid_from, "%Y-%m-%d")
                .map_err(|error| format!("fact {fact_id:?} has invalid valid_from: {error}"))?;
            let end = record
                .conclusion
                .valid_to
                .as_deref()
                .map(|value| NaiveDate::parse_from_str(value, "%Y-%m-%d"))
                .transpose()
                .map_err(|error| format!("fact {fact_id:?} has invalid valid_to: {error}"))?;

            if as_of_date < start || end.is_some_and(|end_date| as_of_date >= end_date) {
                statuses.insert(fact_id.clone(), AsOfSupportStatus::OutsideValidity);
                continue;
            }

            let active: BTreeSet<String> = record
                .conclusion
                .justifications
                .iter()
                .filter(|justification| {
                    justification.dependencies.iter().all(|dependency| match dependency {
                        SupportRef::SourceEvent(source_id) => {
                            self.source_events.contains(source_id)
                                && !self.retracted_source_events.contains(source_id)
                        }
                        SupportRef::DerivedFact(parent_id) => statuses
                            .get(parent_id)
                            .is_some_and(AsOfSupportStatus::is_supported),
                    })
                })
                .map(|justification| justification.justification_id.clone())
                .collect();

            let status = if active.is_empty() {
                AsOfSupportStatus::Unsupported
            } else {
                AsOfSupportStatus::Supported {
                    active_justification_ids: active,
                }
            };
            statuses.insert(fact_id.clone(), status);
        }
        Ok(statuses)
    }

    /// Return one conclusion's query-time support state.
    pub fn support_status_at(
        &self,
        fact_id: &str,
        as_of: &str,
    ) -> Result<AsOfSupportStatus, String> {
        let statuses = self.support_statuses_at(as_of)?;
        statuses
            .get(fact_id)
            .cloned()
            .ok_or_else(|| format!("unknown derived fact {fact_id:?}"))
    }

    /// Return a conclusion only when it is supported and within its valid-time
    /// interval at the requested date.
    pub fn current_fact_at(
        &self,
        fact_id: &str,
        as_of: &str,
    ) -> Result<Option<&DerivedConclusionRecord>, String> {
        let status = self.support_status_at(fact_id, as_of)?;
        if status.is_supported() {
            Ok(self.facts.get(fact_id))
        } else {
            Ok(None)
        }
    }

    pub fn source_is_retracted(&self, event_id: &str) -> bool {
        self.retracted_source_events.contains(event_id)
    }

    pub fn fact_count(&self) -> usize {
        self.facts.len()
    }

    pub fn current_fact_count(&self) -> usize {
        self.facts.values().filter(|record| record.status.is_supported()).count()
    }

    pub fn unsupported_fact_count(&self) -> usize {
        self.facts.values().filter(|record| !record.status.is_supported()).count()
    }

    pub fn state_sha256(&self) -> Result<String, String> {
        self.verify_consistency()?;
        let bytes = serde_json::to_vec(self)
            .map_err(|error| format!("cannot serialize justification graph: {error}"))?;
        Ok(hex::encode(Sha256::digest(bytes)))
    }

    fn dependency_is_supported(&self, dependency: &SupportRef) -> bool {
        match dependency {
            SupportRef::SourceEvent(event_id) => {
                self.source_events.contains(event_id) && !self.retracted_source_events.contains(event_id)
            }
            SupportRef::DerivedFact(fact_id) => self
                .facts
                .get(fact_id)
                .is_some_and(|record| record.status.is_supported()),
        }
    }

    fn recompute_statuses(&mut self) -> Result<(), String> {
        let order = self.fact_order.clone();
        for fact_id in order {
            let justifications = self
                .facts
                .get(&fact_id)
                .ok_or_else(|| format!("fact order references missing fact {fact_id:?}"))?
                .conclusion
                .justifications
                .clone();
            let active: BTreeSet<String> = justifications
                .iter()
                .filter(|justification| {
                    justification
                        .dependencies
                        .iter()
                        .all(|dependency| self.dependency_is_supported(dependency))
                })
                .map(|justification| justification.justification_id.clone())
                .collect();
            let status = if active.is_empty() {
                SupportStatus::Unsupported
            } else {
                SupportStatus::Supported {
                    active_justification_ids: active,
                }
            };
            self.facts
                .get_mut(&fact_id)
                .expect("fact was checked above")
                .status = status;
        }
        Ok(())
    }

    fn verify_consistency(&self) -> Result<(), String> {
        if !self.retracted_source_events.is_subset(&self.source_events) {
            return Err("retracted source set contains an unknown source event".into());
        }
        let mut seen = BTreeSet::new();
        if self.fact_order.len() != self.facts.len() {
            return Err("fact order length does not match fact table".into());
        }

        for fact_id in &self.fact_order {
            if !seen.insert(fact_id.clone()) {
                return Err(format!("duplicate fact ID in insertion order {fact_id:?}"));
            }
            let record = self
                .facts
                .get(fact_id)
                .ok_or_else(|| format!("fact order references missing fact {fact_id:?}"))?;
            validate_conclusion_input(&record.conclusion)?;
            if record.conclusion.fact_id != *fact_id {
                return Err(format!("fact table key does not match conclusion ID {fact_id:?}"));
            }
            for justification in &record.conclusion.justifications {
                for dependency in &justification.dependencies {
                    match dependency {
                        SupportRef::SourceEvent(source_id) => {
                            if !self.source_events.contains(source_id) {
                                return Err(format!("unknown source dependency {source_id:?}"));
                            }
                        }
                        SupportRef::DerivedFact(parent_id) => {
                            if parent_id == fact_id || !seen.contains(parent_id) {
                                return Err(format!(
                                    "derived dependency {parent_id:?} must precede dependent fact {fact_id:?}"
                                ));
                            }
                        }
                    }
                }
            }
        }

        let mut recomputed = self.clone();
        recomputed.recompute_statuses()?;
        for fact_id in &self.fact_order {
            if self.facts.get(fact_id).map(|record| &record.status)
                != recomputed.facts.get(fact_id).map(|record| &record.status)
            {
                return Err(format!("stored support status differs from recomputed status for {fact_id:?}"));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn justification(
        id: &str,
        dependencies: impl IntoIterator<Item = SupportRef>,
    ) -> JustificationInput {
        JustificationInput {
            justification_id: id.into(),
            dependencies: dependencies.into_iter().collect(),
        }
    }

    fn conclusion(
        id: &str,
        justifications: impl IntoIterator<Item = JustificationInput>,
    ) -> DerivedConclusionInput {
        DerivedConclusionInput {
            fact_id: id.into(),
            subject: "deploy-42".into(),
            attribute: "ready".into(),
            value: "true".into(),
            valid_from: "2026-01-01".into(),
            valid_to: None,
            rule_id: "all-prerequisites-satisfied".into(),
            rule_version: "v1".into(),
            justifications: justifications.into_iter().collect(),
        }
    }

    fn source(id: &str) -> SupportRef {
        SupportRef::SourceEvent(id.into())
    }

    fn derived(id: &str) -> SupportRef {
        SupportRef::DerivedFact(id.into())
    }

    #[test]
    fn alternative_support_survives_until_the_last_justification_is_removed() {
        let mut graph = JustificationGraph::default();
        graph.register_source_event("A").unwrap();
        graph.register_source_event("B").unwrap();
        graph
            .derive(conclusion(
                "C",
                [
                    justification("from-A", [source("A")]),
                    justification("from-B", [source("B")]),
                ],
            ))
            .unwrap();
        graph.derive(conclusion("D", [justification("from-C", [derived("C")])])).unwrap();

        let first = graph.invalidate_source_event("A").unwrap();
        assert!(first.newly_unsupported_fact_ids.is_empty());
        assert_eq!(first.retained_with_alternative_support, vec!["C"]);
        assert_eq!(first.inactive_justifications_by_fact.get("C"), Some(&vec!["from-A".to_owned()]));
        assert!(graph.current_fact("C").is_some());
        assert!(graph.current_fact("D").is_some());
        assert_eq!(
            graph.fact("C").unwrap().status.active_justification_ids(),
            BTreeSet::from(["from-B".to_owned()])
        );

        let second = graph.invalidate_source_event("B").unwrap();
        assert_eq!(second.newly_unsupported_fact_ids, vec!["C", "D"]);
        assert!(graph.current_fact("C").is_none());
        assert!(graph.current_fact("D").is_none());
        assert_eq!(graph.unsupported_fact_count(), 2);
        graph.verify_consistency().unwrap();
    }

    #[test]
    fn mixed_conjunctive_and_alternative_support_is_not_over_invalidated() {
        let mut graph = JustificationGraph::default();
        for id in ["A", "B", "D"] {
            graph.register_source_event(id).unwrap();
        }
        graph
            .derive(conclusion(
                "C",
                [
                    justification("A-and-B", [source("A"), source("B")]),
                    justification("D-alone", [source("D")]),
                ],
            ))
            .unwrap();

        let first = graph.invalidate_source_event("A").unwrap();
        assert!(first.newly_unsupported_fact_ids.is_empty());
        assert_eq!(first.retained_with_alternative_support, vec!["C"]);
        assert_eq!(
            graph.fact("C").unwrap().status.active_justification_ids(),
            BTreeSet::from(["D-alone".to_owned()])
        );

        let second = graph.invalidate_source_event("D").unwrap();
        assert_eq!(second.newly_unsupported_fact_ids, vec!["C"]);
        assert!(graph.current_fact("C").is_none());
        assert_eq!(graph.fact("C").unwrap().status, SupportStatus::Unsupported);
        // B remains present, but A-and-B is an incomplete (conjunctive) proof.
        assert!(!graph.source_is_retracted("B"));
    }

    #[test]
    fn exhaustive_three_event_truth_table_matches_independent_support_oracle() {
        // Independent oracle: C is supported iff (A AND B) OR D.
        // Exercise every possible retraction subset, including all proofs removed.
        for retracted_mask in 0u8..8 {
            let mut graph = JustificationGraph::default();
            for event_id in ["A", "B", "D"] {
                graph.register_source_event(event_id).unwrap();
            }
            graph
                .derive(conclusion(
                    "C",
                    [
                        justification("A-and-B", [source("A"), source("B")]),
                        justification("D-alone", [source("D")]),
                    ],
                ))
                .unwrap();
            graph
                .derive(conclusion("child", [justification("from-C", [derived("C")])]))
                .unwrap();

            for (bit, event_id) in ["A", "B", "D"].iter().enumerate() {
                if (retracted_mask & (1 << bit)) != 0 {
                    graph.invalidate_source_event(event_id).unwrap();
                }
            }

            let a = (retracted_mask & 0b001) == 0;
            let b = (retracted_mask & 0b010) == 0;
            let d = (retracted_mask & 0b100) == 0;
            let expected_c_support = (a && b) || d;
            assert_eq!(
                graph.current_fact("C").is_some(),
                expected_c_support,
                "wrong C support state for retraction mask {retracted_mask:03b}"
            );
            assert_eq!(
                graph.current_fact("child").is_some(),
                expected_c_support,
                "dependent child did not track C for retraction mask {retracted_mask:03b}"
            );
            graph.verify_consistency().unwrap();
        }
    }

    #[test]
    fn as_of_validity_uses_half_open_interval_boundaries() {
        let mut graph = JustificationGraph::default();
        graph.register_source_event("A").unwrap();
        let mut time_bounded = conclusion(
            "C",
            [justification("from-A", [source("A")])],
        );
        time_bounded.valid_from = "2026-02-01".into();
        time_bounded.valid_to = Some("2026-03-01".into());
        graph.derive(time_bounded).unwrap();

        assert!(matches!(
            graph.support_status_at("C", "2026-01-31").unwrap(),
            AsOfSupportStatus::OutsideValidity
        ));
        assert!(matches!(
            graph.support_status_at("C", "2026-02-01").unwrap(),
            AsOfSupportStatus::Supported { .. }
        ));
        assert!(graph.current_fact_at("C", "2026-02-28").unwrap().is_some());
        assert!(matches!(
            graph.support_status_at("C", "2026-03-01").unwrap(),
            AsOfSupportStatus::OutsideValidity
        ));
        assert!(graph.current_fact_at("C", "2026-03-01").unwrap().is_none());
    }

    #[test]
    fn expired_parent_makes_descendant_unsupported_at_query_time() {
        let mut graph = JustificationGraph::default();
        graph.register_source_event("A").unwrap();
        let mut parent = conclusion(
            "parent",
            [justification("from-A", [source("A")])],
        );
        parent.valid_to = Some("2026-03-01".into());
        graph.derive(parent).unwrap();
        graph
            .derive(conclusion(
                "child",
                [justification("from-parent", [derived("parent")])],
            ))
            .unwrap();

        assert!(graph.current_fact_at("child", "2026-02-28").unwrap().is_some());
        assert!(matches!(
            graph.support_status_at("parent", "2026-03-01").unwrap(),
            AsOfSupportStatus::OutsideValidity
        ));
        assert!(matches!(
            graph.support_status_at("child", "2026-03-01").unwrap(),
            AsOfSupportStatus::Unsupported
        ));
        assert!(graph.current_fact_at("child", "2026-03-01").unwrap().is_none());
        // A time-bounded query does not mutate the timeless audit status.
        assert!(graph.current_fact("child").is_some());
    }

    #[test]
    fn alternative_proof_preserves_query_time_support_when_other_proof_expires() {
        let mut graph = JustificationGraph::default();
        graph.register_source_event("A").unwrap();
        graph.register_source_event("B").unwrap();
        let mut first_proof = conclusion(
            "proof-A",
            [justification("from-A", [source("A")])],
        );
        first_proof.valid_to = Some("2026-03-01".into());
        graph.derive(first_proof).unwrap();
        graph
            .derive(conclusion(
                "proof-B",
                [justification("from-B", [source("B")])],
            ))
            .unwrap();
        graph
            .derive(conclusion(
                "C",
                [
                    justification("via-A", [derived("proof-A")]),
                    justification("via-B", [derived("proof-B")]),
                ],
            ))
            .unwrap();

        assert_eq!(
            graph.support_status_at("C", "2026-02-28").unwrap(),
            AsOfSupportStatus::Supported {
                active_justification_ids: BTreeSet::from([
                    "via-A".to_owned(),
                    "via-B".to_owned()
                ])
            }
        );
        assert_eq!(
            graph.support_status_at("C", "2026-03-01").unwrap(),
            AsOfSupportStatus::Supported {
                active_justification_ids: BTreeSet::from(["via-B".to_owned()])
            }
        );
    }

    #[test]
    fn as_of_query_rejects_invalid_date_and_unknown_fact() {
        let mut graph = JustificationGraph::default();
        graph.register_source_event("A").unwrap();
        graph
            .derive(conclusion(
                "C",
                [justification("from-A", [source("A")])],
            ))
            .unwrap();
        assert!(graph.support_status_at("C", "not-a-date").is_err());
        assert!(graph.support_status_at("MISSING", "2026-02-01").is_err());
    }

    #[test]
    fn dependencies_inside_one_justification_are_conjunctive() {
        let mut graph = JustificationGraph::default();
        graph.register_source_event("A").unwrap();
        graph.register_source_event("B").unwrap();
        graph
            .derive(conclusion(
                "C",
                [justification("A-and-B", [source("A"), source("B")])],
            ))
            .unwrap();

        let report = graph.invalidate_source_event("A").unwrap();
        assert_eq!(report.newly_unsupported_fact_ids, vec!["C"]);
        assert!(graph.current_fact("C").is_none());
    }

    #[test]
    fn no_empty_justification_can_turn_absence_into_a_conclusion() {
        let mut graph = JustificationGraph::default();
        let before = graph.state_sha256().unwrap();
        let no_evidence = conclusion(
            "not-P",
            [JustificationInput {
                justification_id: "absence".into(),
                dependencies: BTreeSet::new(),
            }],
        );
        assert!(graph.derive(no_evidence).unwrap_err().contains("must contain at least one dependency"));
        assert_eq!(graph.state_sha256().unwrap(), before);
        assert!(graph.current_fact("not-P").is_none());
        // This reference model contains no implicit closed-world rule. A domain-specific
        // completeness witness and explicit rule are required to derive negation.
    }

    #[test]
    fn invalidation_is_idempotent_and_state_digest_is_deterministic() {
        fn build() -> JustificationGraph {
            let mut graph = JustificationGraph::default();
            graph.register_source_event("B").unwrap();
            graph.register_source_event("A").unwrap();
            graph.derive(conclusion("C", [
                justification("by-A", [source("A")]),
                justification("by-B", [source("B")]),
            ])).unwrap();
            graph
        }

        let mut left = build();
        let right = build();
        assert_eq!(left.state_sha256().unwrap(), right.state_sha256().unwrap());
        assert_eq!(left.invalidate_source_event("A").unwrap().newly_unsupported_fact_ids, Vec::<String>::new());
        let after_first = left.state_sha256().unwrap();
        let replay = left.invalidate_source_event("A").unwrap();
        assert!(replay.newly_unsupported_fact_ids.is_empty());
        assert!(replay.inactive_justifications_by_fact.is_empty());
        assert_eq!(left.state_sha256().unwrap(), after_first);
    }

    #[test]
    fn invalid_inputs_do_not_partially_mutate_the_graph() {
        let mut graph = JustificationGraph::default();
        graph.register_source_event("A").unwrap();
        let before = graph.state_sha256().unwrap();
        let invalid = conclusion(
            "C",
            [justification("unknown-parent", [derived("MISSING")])],
        );
        assert!(graph.derive(invalid).is_err());
        assert_eq!(graph.state_sha256().unwrap(), before);
    }
}
