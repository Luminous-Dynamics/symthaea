// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evidence-bound change chronology for IT incident reasoning.
//!
//! A change timeline is not a causal graph. It answers a narrower question:
//! which observed changes could plausibly fall before/around a failure window,
//! while preserving uncertainty in source clocks and observation time.

use crate::system_state::{EntityId, ObservationId, StateValueV1, SystemStateGraphV1};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ChangeId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChangeKindV1 {
    Configuration,
    Deployment,
    PackageUpgrade,
    Firmware,
    IdentityPolicy,
    Certificate,
    Dns,
    Routing,
    Firewall,
    ServiceLifecycle,
    Hardware,
    Topology,
    CloudControlPlane,
    Custom(String),
}

/// Time model for an event whose source clock may be imperfect.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChangeClockV1 {
    /// Source-declared change time.
    pub occurred_at_unix_ms: u64,
    /// Symmetric uncertainty around `occurred_at_unix_ms` when known.
    pub clock_uncertainty_ms: Option<u64>,
    /// Local collection time. This is retained separately and is not silently
    /// substituted for source event time.
    pub observed_at_unix_ms: u64,
}

impl ChangeClockV1 {
    pub fn earliest_possible_unix_ms(&self) -> u64 {
        self.occurred_at_unix_ms
            .saturating_sub(self.clock_uncertainty_ms.unwrap_or(0))
    }

    pub fn latest_possible_unix_ms(&self) -> u64 {
        self.occurred_at_unix_ms
            .saturating_add(self.clock_uncertainty_ms.unwrap_or(0))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SystemChangeV1 {
    pub id: ChangeId,
    pub subject: EntityId,
    pub kind: ChangeKindV1,
    pub clock: ChangeClockV1,
    /// Human-readable normalized summary. Raw secrets/log bodies should not be
    /// placed here without the owning adapter's privacy policy.
    pub summary: String,
    pub actor: Option<String>,
    pub before_digest: Option<String>,
    pub after_digest: Option<String>,
    #[serde(default)]
    pub attributes: BTreeMap<String, StateValueV1>,
    /// Exact observations supporting this change record.
    pub evidence: BTreeSet<ObservationId>,
}

impl SystemChangeV1 {
    pub fn validate(&self, graph: &SystemStateGraphV1) -> Result<(), ChangeTimelineError> {
        if self.id.0.trim().is_empty() {
            return Err(ChangeTimelineError::EmptyIdentifier("change"));
        }
        if self.subject.0.trim().is_empty() {
            return Err(ChangeTimelineError::EmptyIdentifier("subject"));
        }
        if self.summary.trim().is_empty() {
            return Err(ChangeTimelineError::EmptySummary);
        }
        if self.evidence.is_empty() {
            return Err(ChangeTimelineError::MissingEvidence(self.id.clone()));
        }
        if graph.entity(&self.subject).is_none() {
            return Err(ChangeTimelineError::UnknownSubject(self.subject.clone()));
        }
        for evidence_id in &self.evidence {
            if graph.observation(evidence_id).is_none() {
                return Err(ChangeTimelineError::UnknownObservation(evidence_id.clone()));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TemporalRelationV1 {
    DefinitelyBefore,
    DefinitelyAfter,
    OverlapsOrUncertain,
}

pub fn temporal_relation(left: &ChangeClockV1, right: &ChangeClockV1) -> TemporalRelationV1 {
    if left.latest_possible_unix_ms() < right.earliest_possible_unix_ms() {
        TemporalRelationV1::DefinitelyBefore
    } else if left.earliest_possible_unix_ms() > right.latest_possible_unix_ms() {
        TemporalRelationV1::DefinitelyAfter
    } else {
        TemporalRelationV1::OverlapsOrUncertain
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FailureWindowV1 {
    pub center_unix_ms: u64,
    /// Uncertainty around when the failure actually began.
    pub uncertainty_ms: u64,
    /// How far back to inspect for candidate precursor changes.
    pub lookback_ms: u64,
}

impl FailureWindowV1 {
    pub fn earliest_failure_unix_ms(&self) -> u64 {
        self.center_unix_ms.saturating_sub(self.uncertainty_ms)
    }

    pub fn latest_failure_unix_ms(&self) -> u64 {
        self.center_unix_ms.saturating_add(self.uncertainty_ms)
    }

    pub fn search_start_unix_ms(&self) -> u64 {
        self.earliest_failure_unix_ms().saturating_sub(self.lookback_ms)
    }
}

/// Independent immutable-change registry. The state graph remains the source of
/// current projected state; this type retains event chronology for incident work.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ChangeTimelineV1 {
    changes: BTreeMap<ChangeId, SystemChangeV1>,
}

impl ChangeTimelineV1 {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, id: &ChangeId) -> Option<&SystemChangeV1> {
        self.changes.get(id)
    }

    pub fn changes(&self) -> impl Iterator<Item = &SystemChangeV1> {
        self.changes.values()
    }

    /// Record a change once. Exact replay is idempotent; identity rebinding is rejected.
    pub fn record(
        &mut self,
        graph: &SystemStateGraphV1,
        change: SystemChangeV1,
    ) -> Result<bool, ChangeTimelineError> {
        change.validate(graph)?;
        if let Some(existing) = self.changes.get(&change.id) {
            if existing == &change {
                return Ok(false);
            }
            return Err(ChangeTimelineError::IdentityConflict(change.id));
        }
        self.changes.insert(change.id.clone(), change);
        Ok(true)
    }

    /// Return changes whose uncertainty intervals overlap the conservative
    /// precursor window. Inclusion is intentionally permissive: overlap means
    /// "candidate", never "cause".
    pub fn candidates_before_failure<'a>(
        &'a self,
        window: &FailureWindowV1,
    ) -> Vec<&'a SystemChangeV1> {
        let start = window.search_start_unix_ms();
        let end = window.latest_failure_unix_ms();
        let mut candidates: Vec<&SystemChangeV1> = self
            .changes
            .values()
            .filter(|change| {
                change.clock.latest_possible_unix_ms() >= start
                    && change.clock.earliest_possible_unix_ms() <= end
            })
            .collect();

        candidates.sort_by_key(|change| (
            change.clock.earliest_possible_unix_ms(),
            change.clock.latest_possible_unix_ms(),
            change.id.0.clone(),
        ));
        candidates
    }

    /// Changes that are definitely earlier than the supplied change after
    /// accounting for both clock uncertainty intervals.
    pub fn definitely_before<'a>(&'a self, target: &SystemChangeV1) -> Vec<&'a SystemChangeV1> {
        self.changes
            .values()
            .filter(|candidate| {
                candidate.id != target.id
                    && temporal_relation(&candidate.clock, &target.clock)
                        == TemporalRelationV1::DefinitelyBefore
            })
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ChangeTimelineError {
    EmptyIdentifier(&'static str),
    EmptySummary,
    MissingEvidence(ChangeId),
    UnknownSubject(EntityId),
    UnknownObservation(ObservationId),
    IdentityConflict(ChangeId),
}

impl fmt::Display for ChangeTimelineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyIdentifier(field) => write!(f, "empty {field} identifier"),
            Self::EmptySummary => write!(f, "change summary is empty"),
            Self::MissingEvidence(id) => write!(f, "change {:?} has no evidence", id.0),
            Self::UnknownSubject(id) => write!(f, "unknown change subject {:?}", id.0),
            Self::UnknownObservation(id) => write!(f, "unknown observation {:?}", id.0),
            Self::IdentityConflict(id) => write!(f, "change identity {:?} was rebound", id.0),
        }
    }
}

impl Error for ChangeTimelineError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system_state::{
        EntityKindV1, ObservationClockV1, ObservationProvenanceV1,
        ObservationSourceKindV1, SystemObservationV1,
    };

    fn graph_with_host_and_observation() -> SystemStateGraphV1 {
        let subject = EntityId("host:alpha".into());
        let evidence_id = ObservationId("obs:1".into());
        let mut graph = SystemStateGraphV1::new();
        graph
            .record_observation(SystemObservationV1 {
                id: evidence_id.clone(),
                subject: subject.clone(),
                provenance: ObservationProvenanceV1 {
                    source_id: "host:alpha:eventlog".into(),
                    source_kind: ObservationSourceKindV1::WindowsEventLog,
                    collector: "test".into(),
                    collector_version: None,
                    schema_version: None,
                    artifact_digest: None,
                },
                clock: ObservationClockV1 {
                    event_time_unix_ms: Some(1_000),
                    observed_at_unix_ms: 1_010,
                    ingested_at_unix_ms: Some(1_020),
                    max_age_ms: Some(60_000),
                    clock_uncertainty_ms: Some(5),
                },
                confidence: 1.0,
                facts: BTreeMap::new(),
            })
            .unwrap();
        graph
            .upsert_entity(
                subject,
                EntityKindV1::Host,
                BTreeMap::new(),
                &evidence_id,
            )
            .unwrap();
        graph
    }

    fn change(id: &str, at: u64, uncertainty: u64) -> SystemChangeV1 {
        SystemChangeV1 {
            id: ChangeId(id.into()),
            subject: EntityId("host:alpha".into()),
            kind: ChangeKindV1::Configuration,
            clock: ChangeClockV1 {
                occurred_at_unix_ms: at,
                clock_uncertainty_ms: Some(uncertainty),
                observed_at_unix_ms: at.saturating_add(10),
            },
            summary: "configuration changed".into(),
            actor: None,
            before_digest: Some("before".into()),
            after_digest: Some("after".into()),
            attributes: BTreeMap::new(),
            evidence: BTreeSet::from([ObservationId("obs:1".into())]),
        }
    }

    #[test]
    fn clock_uncertainty_prevents_false_ordering() {
        let a = ChangeClockV1 {
            occurred_at_unix_ms: 1_000,
            clock_uncertainty_ms: Some(100),
            observed_at_unix_ms: 1_100,
        };
        let b = ChangeClockV1 {
            occurred_at_unix_ms: 1_050,
            clock_uncertainty_ms: Some(100),
            observed_at_unix_ms: 1_200,
        };
        assert_eq!(temporal_relation(&a, &b), TemporalRelationV1::OverlapsOrUncertain);
    }

    #[test]
    fn exact_replay_is_idempotent() {
        let graph = graph_with_host_and_observation();
        let item = change("change:1", 1_500, 10);
        let mut timeline = ChangeTimelineV1::new();
        assert!(timeline.record(&graph, item.clone()).unwrap());
        assert!(!timeline.record(&graph, item).unwrap());
    }

    #[test]
    fn identity_rebinding_is_rejected() {
        let graph = graph_with_host_and_observation();
        let mut timeline = ChangeTimelineV1::new();
        timeline.record(&graph, change("change:1", 1_500, 10)).unwrap();
        let err = timeline
            .record(&graph, change("change:1", 2_500, 10))
            .unwrap_err();
        assert!(matches!(err, ChangeTimelineError::IdentityConflict(_)));
    }

    #[test]
    fn precursor_window_returns_overlap_candidates_not_causal_claims() {
        let graph = graph_with_host_and_observation();
        let mut timeline = ChangeTimelineV1::new();
        timeline.record(&graph, change("too-old", 100, 0)).unwrap();
        timeline.record(&graph, change("candidate", 1_800, 30)).unwrap();
        timeline.record(&graph, change("around-failure", 2_020, 50)).unwrap();
        timeline.record(&graph, change("too-late", 3_000, 0)).unwrap();

        let candidates = timeline.candidates_before_failure(&FailureWindowV1 {
            center_unix_ms: 2_000,
            uncertainty_ms: 100,
            lookback_ms: 500,
        });
        let ids: Vec<&str> = candidates.iter().map(|c| c.id.0.as_str()).collect();
        assert_eq!(ids, vec!["candidate", "around-failure"]);
    }

    #[test]
    fn missing_evidence_is_rejected() {
        let graph = graph_with_host_and_observation();
        let mut item = change("change:1", 1_500, 10);
        item.evidence = BTreeSet::from([ObservationId("obs:missing".into())]);
        let err = item.validate(&graph).unwrap_err();
        assert!(matches!(err, ChangeTimelineError::UnknownObservation(_)));
    }
}
