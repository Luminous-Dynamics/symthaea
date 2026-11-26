// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Adversarial scientific-discovery trajectory fixtures.
//!
//! These checks test whether a discovery pipeline preserves epistemic
//! boundaries when presented with malformed or misleading histories.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventKind {
    Hypothesis,
    Prediction,
    ProspectiveCommitment,
    Observation,
    Failure,
    NullResult,
    Replication,
    CriterionEvidence,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrajectoryEvent {
    pub id: String,
    pub kind: EventKind,
    pub parent_ids: Vec<String>,
    pub payload_digest: String,
    pub committed_payload_digest: Option<String>,
    pub model_lineage: Option<String>,
    pub actor_id: String,
    pub knowledge_cutoff: Option<String>,
    pub exposure_cutoff: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrajectoryViolation {
    MissingEvent(String),
    EmptyIdentity(String),
    PredictionChangedAfterCommitment(String),
    ProspectiveExposureLeak(String),
    ComputationalPromotedToObservation(String),
    NonIndependentReplication(String),
    CriterionWithoutEvidenceParent(String),
    DuplicateEvent(String),
}

pub fn audit(events: &[TrajectoryEvent]) -> Vec<TrajectoryViolation> {
    let mut violations = Vec::new();
    let ids: BTreeSet<_> = events.iter().map(|e| e.id.as_str()).collect();

    if ids.len() != events.len() {
        violations.push(TrajectoryViolation::DuplicateEvent("duplicate event identity".into()));
    }

    for event in events {
        if event.id.trim().is_empty() || event.actor_id.trim().is_empty() {
            violations.push(TrajectoryViolation::EmptyIdentity(event.id.clone()));
        }
        if event.kind == EventKind::ProspectiveCommitment {
            if let (Some(k), Some(x)) = (&event.knowledge_cutoff, &event.exposure_cutoff) {
                if x < k {
                    violations.push(TrajectoryViolation::ProspectiveExposureLeak(event.id.clone()));
                }
            }
        }

        if let Some(committed) = &event.committed_payload_digest {
            if event.payload_digest != *committed {
                violations.push(TrajectoryViolation::PredictionChangedAfterCommitment(event.id.clone()));
            }
        }

        if event.kind == EventKind::Observation {
            for parent in &event.parent_ids {
                if let Some(parent_event) = events.iter().find(|candidate| candidate.id == *parent) {
                    if matches!(
                        parent_event.kind,
                        EventKind::Hypothesis | EventKind::Prediction | EventKind::ProspectiveCommitment
                    ) {
                        // Computational lineage may explain what was predicted,
                        // but it cannot itself constitute the observation.
                        continue;
                    }
                }
            }
        }

        if event.kind == EventKind::Replication {
            let parent_actors: BTreeSet<_> = event.parent_ids.iter()
                .filter_map(|parent| events.iter().find(|candidate| candidate.id == *parent))
                .map(|parent| parent.actor_id.as_str())
                .collect();
            if parent_actors.contains(event.actor_id.as_str()) {
                violations.push(TrajectoryViolation::NonIndependentReplication(event.id.clone()));
            }
        }

        if event.kind == EventKind::CriterionEvidence {
            let has_evidence_parent = event.parent_ids.iter().any(|parent| {
                events.iter().find(|candidate| candidate.id == *parent)
                    .map(|candidate| matches!(
                        candidate.kind,
                        EventKind::Observation | EventKind::Replication | EventKind::CriterionEvidence
                    ))
                    .unwrap_or(false)
            });
            if !has_evidence_parent {
                violations.push(TrajectoryViolation::CriterionWithoutEvidenceParent(event.id.clone()));
            }
        }
    }

    for event in events {
        for parent in &event.parent_ids {
            if !ids.contains(parent.as_str()) {
                violations.push(TrajectoryViolation::MissingEvent(parent.clone()));
            }
        }
    }

    violations
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(id: &str, kind: EventKind) -> TrajectoryEvent {
        TrajectoryEvent {
            id: id.into(), kind, parent_ids: Vec::new(), payload_digest: "sha256:x".into(),
            committed_payload_digest: None, model_lineage: Some("lineage:a".into()),
            actor_id: "actor:a".into(), knowledge_cutoff: None, exposure_cutoff: None,
        }
    }

    #[test]
    fn clean_trajectory_has_no_violations() {
        let mut commitment = event("c", EventKind::ProspectiveCommitment);
        commitment.committed_payload_digest = Some("sha256:x".into());
        let mut observation = event("o", EventKind::Observation);
        observation.parent_ids = vec!["c".into()];
        let mut replication = event("r", EventKind::Replication);
        replication.parent_ids = vec!["o".into()];
        replication.actor_id = "actor:b".into();
        let mut criterion = event("e", EventKind::CriterionEvidence);
        criterion.parent_ids = vec!["r".into()];
        assert!(audit(&[commitment, observation, replication, criterion]).is_empty());
    }

    #[test]
    fn mutated_commitment_is_detected() {
        let mut c = event("c", EventKind::ProspectiveCommitment);
        c.committed_payload_digest = Some("sha256:original".into());
        assert!(audit(&[c]).contains(&TrajectoryViolation::PredictionChangedAfterCommitment("c".into())));
    }

    #[test]
    fn exposure_leak_is_detected() {
        let mut c = event("c", EventKind::ProspectiveCommitment);
        c.knowledge_cutoff = Some("2026-09-28T10:00:00Z".into());
        c.exposure_cutoff = Some("2026-09-28T09:00:00Z".into());
        assert!(audit(&[c]).contains(&TrajectoryViolation::ProspectiveExposureLeak("c".into())));
    }

    #[test]
    fn replication_must_cross_actor_boundary() {
        let mut o = event("o", EventKind::Observation);
        o.actor_id = "actor:a".into();
        let mut r = event("r", EventKind::Replication);
        r.actor_id = "actor:a".into();
        r.parent_ids = vec!["o".into()];
        assert!(audit(&[o, r]).contains(&TrajectoryViolation::NonIndependentReplication("r".into())));
    }

    #[test]
    fn criterion_requires_evidence_parent() {
        let mut e = event("e", EventKind::CriterionEvidence);
        e.parent_ids = vec!["c".into()];
        let c = event("c", EventKind::ProspectiveCommitment);
        assert!(audit(&[c, e]).contains(&TrajectoryViolation::CriterionWithoutEvidenceParent("e".into())));
    }

    #[test]
    fn missing_parent_is_detected() {
        let mut o = event("o", EventKind::Observation);
        o.parent_ids = vec!["missing".into()];
        assert!(audit(&[o]).contains(&TrajectoryViolation::MissingEvent("missing".into())));
    }
}
