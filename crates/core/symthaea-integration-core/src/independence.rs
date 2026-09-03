// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Conservative cohort-level source-independence assessment.
//!
//! Multiple monitoring products frequently re-export the same underlying
//! measurement. Counting reports is therefore not the same as counting
//! independent evidence. This module collapses observations that explicitly
//! share lineage and reports bounded answers rather than inventing certainty
//! where provenance is incomplete. Adapter-supplied independence-group labels
//! are descriptive provenance only: distinct labels do not prove independent
//! origins without a separately qualified independence authority.

use crate::{EntityRef, LineageRelationship, ObservationEnvelope};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// A contradictory provenance declaration discovered inside one collapsed
/// shared-origin component.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndependenceMetadataConflict {
    pub observation_ids: Vec<String>,
    pub declared_groups: Vec<String>,
}

/// Conservative bounds for one semantic evidence cohort.
///
/// `independent_lower_bound` is a proof bound. In v0.1, observation-local
/// metadata can prove that reports share an origin, but it cannot prove that
/// different declared groups are genuinely independent. Therefore a non-empty
/// cohort has a lower bound of one until a separately qualified independence
/// authority is introduced. `independent_upper_bound` counts shared-origin
/// components: the most independent the reports could be without contradicting
/// known lineage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndependenceAssessment {
    pub entity: EntityRef,
    pub signal: String,
    pub reports: usize,
    pub shared_origin_components: usize,
    /// Number of distinct non-conflicting adapter-declared group labels. This is
    /// descriptive only and does not raise `independent_lower_bound`.
    pub declared_independent_groups: usize,
    pub unresolved_components: usize,
    pub independent_lower_bound: usize,
    pub independent_upper_bound: usize,
    pub metadata_conflicts: Vec<IndependenceMetadataConflict>,
}

impl IndependenceAssessment {
    /// True only when the proof bounds collapse to one answer and provenance is
    /// not internally contradictory. Distinct self-declared group labels alone
    /// are insufficient to make the assessment fully resolved.
    pub fn fully_resolved(&self) -> bool {
        self.independent_lower_bound == self.independent_upper_bound
            && self.metadata_conflicts.is_empty()
    }

    /// Fraction of raw reports that survive as distinct known/possible origins.
    /// This is descriptive only; it is not a probability or confidence score.
    pub fn maximum_origin_fraction(&self) -> f64 {
        if self.reports == 0 {
            0.0
        } else {
            self.independent_upper_bound as f64 / self.reports as f64
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum IndependenceAssessmentError {
    #[error("evidence cohort is empty")]
    EmptyCohort,
    #[error("cohort mixes entities: expected {expected}, found {found}")]
    MixedEntity { expected: String, found: String },
    #[error("cohort mixes signals: expected `{expected}`, found `{found}`")]
    MixedSignal { expected: String, found: String },
}

/// Assess a set of observations that purport to measure the same entity/signal.
///
/// The caller chooses the temporal cohort/window. This function deliberately
/// refuses to compare different entities or signals, because provenance
/// independence is meaningful only after semantic alignment.
pub fn assess_independence(
    observations: &[ObservationEnvelope],
) -> Result<IndependenceAssessment, IndependenceAssessmentError> {
    let first = observations
        .first()
        .ok_or(IndependenceAssessmentError::EmptyCohort)?;
    let entity = first.entity.clone();
    let signal = first.signal.clone();

    for observation in observations.iter().skip(1) {
        if observation.entity != entity {
            return Err(IndependenceAssessmentError::MixedEntity {
                expected: entity.canonical_key(),
                found: observation.entity.canonical_key(),
            });
        }
        if observation.signal != signal {
            return Err(IndependenceAssessmentError::MixedSignal {
                expected: signal.clone(),
                found: observation.signal.clone(),
            });
        }
    }

    let mut sets = DisjointSet::new(observations.len());
    for left in 0..observations.len() {
        for right in (left + 1)..observations.len() {
            if matches!(
                observations[left].lineage_relationship(&observations[right]),
                LineageRelationship::SameObservation | LineageRelationship::SharedOrigin
            ) {
                sets.union(left, right);
            }
        }
    }

    let mut components: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for index in 0..observations.len() {
        let root = sets.find(index);
        components.entry(root).or_default().push(index);
    }

    let mut valid_declared_groups = BTreeSet::new();
    let mut unresolved_components = 0usize;
    let mut metadata_conflicts = Vec::new();

    for indices in components.values() {
        let groups: BTreeSet<String> = indices
            .iter()
            .filter_map(|index| observations[*index].lineage.independence_group.clone())
            .collect();

        match groups.len() {
            0 => unresolved_components += 1,
            1 => {
                if let Some(group) = groups.iter().next() {
                    valid_declared_groups.insert(group.clone());
                }
            }
            _ => {
                metadata_conflicts.push(IndependenceMetadataConflict {
                    observation_ids: indices
                        .iter()
                        .map(|index| observations[*index].observation_id.to_string())
                        .collect(),
                    declared_groups: groups.into_iter().collect(),
                });
            }
        }
    }

    // A non-empty observed cohort proves at least one evidence origin. Shared-
    // origin metadata can collapse the upper bound, but adapter-declared group
    // names cannot prove positive independence and therefore never raise the
    // lower bound in v0.1.
    let independent_lower_bound = 1;
    let independent_upper_bound = components.len();

    Ok(IndependenceAssessment {
        entity,
        signal,
        reports: observations.len(),
        shared_origin_components: components.len(),
        declared_independent_groups: valid_declared_groups.len(),
        unresolved_components,
        independent_lower_bound,
        independent_upper_bound,
        metadata_conflicts,
    })
}

#[derive(Debug)]
struct DisjointSet {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl DisjointSet {
    fn new(len: usize) -> Self {
        Self {
            parent: (0..len).collect(),
            rank: vec![0; len],
        }
    }

    fn find(&mut self, index: usize) -> usize {
        if self.parent[index] != index {
            let root = self.find(self.parent[index]);
            self.parent[index] = root;
        }
        self.parent[index]
    }

    fn union(&mut self, left: usize, right: usize) {
        let left_root = self.find(left);
        let right_root = self.find(right);
        if left_root == right_root {
            return;
        }

        match self.rank[left_root].cmp(&self.rank[right_root]) {
            std::cmp::Ordering::Less => self.parent[left_root] = right_root,
            std::cmp::Ordering::Greater => self.parent[right_root] = left_root,
            std::cmp::Ordering::Equal => {
                self.parent[right_root] = left_root;
                self.rank[left_root] = self.rank[left_root].saturating_add(1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ObservationId, ObservationKind, ObservationLineage, ObservationQuality, ObservationSource,
        ObservationValue,
    };

    fn observation(
        id: &str,
        lineage: &str,
        group: Option<&str>,
        upstream: Option<&str>,
    ) -> ObservationEnvelope {
        ObservationEnvelope::new(
            ObservationId::new(id),
            1_000,
            1_010,
            EntityRef::new("site:lab", "host", "node-1"),
            ObservationKind::Metric,
            "system.cpu.utilization",
            ObservationValue::Number {
                value: 0.5,
                unit: Some("1".into()),
            },
            ObservationSource {
                integration_id: format!("source-{id}"),
                collector_id: None,
                upstream_origin: upstream.map(str::to_string),
                measurement_method: "fixture".into(),
                tenant: None,
            },
            ObservationQuality::observed(0.9),
            ObservationLineage {
                lineage_id: lineage.into(),
                parent_ids: vec![],
                independence_group: group.map(str::to_string),
                transforms: vec![],
            },
        )
    }

    #[test]
    fn three_reexports_of_one_measurement_count_as_one_possible_origin() {
        let observations = vec![
            observation("a", "same", None, None),
            observation("b", "same", None, None),
            observation("c", "same", None, None),
        ];
        let assessment = assess_independence(&observations).unwrap();
        assert_eq!(assessment.reports, 3);
        assert_eq!(assessment.independent_lower_bound, 1);
        assert_eq!(assessment.independent_upper_bound, 1);
        assert!(assessment.fully_resolved());
    }

    #[test]
    fn distinct_declared_groups_do_not_prove_independence() {
        let observations = vec![
            observation("a", "lineage-a", Some("bmc"), None),
            observation("b", "lineage-b", Some("kernel"), None),
        ];
        let assessment = assess_independence(&observations).unwrap();
        assert_eq!(assessment.declared_independent_groups, 2);
        assert_eq!(assessment.independent_lower_bound, 1);
        assert_eq!(assessment.independent_upper_bound, 2);
        assert!(!assessment.fully_resolved());
    }

    #[test]
    fn arbitrary_unique_group_names_cannot_manufacture_independent_evidence() {
        let observations = vec![
            observation("a", "lineage-a", Some("self-declared-a"), None),
            observation("b", "lineage-b", Some("self-declared-b"), None),
            observation("c", "lineage-c", Some("self-declared-c"), None),
        ];
        let assessment = assess_independence(&observations).unwrap();
        assert_eq!(assessment.declared_independent_groups, 3);
        assert_eq!(assessment.independent_lower_bound, 1);
        assert_eq!(assessment.independent_upper_bound, 3);
        assert!(!assessment.fully_resolved());
    }

    #[test]
    fn unknown_relationship_never_inflates_the_lower_bound() {
        let observations = vec![
            observation("a", "lineage-a", Some("kernel"), None),
            observation("b", "lineage-b", None, None),
        ];
        let assessment = assess_independence(&observations).unwrap();
        assert_eq!(assessment.independent_lower_bound, 1);
        assert_eq!(assessment.independent_upper_bound, 2);
        assert_eq!(assessment.unresolved_components, 1);
        assert!(!assessment.fully_resolved());
    }

    #[test]
    fn shared_upstream_overrides_conflicting_independence_claims() {
        let observations = vec![
            observation("a", "lineage-a", Some("vendor-a"), Some("procfs:node-1")),
            observation("b", "lineage-b", Some("vendor-b"), Some("procfs:node-1")),
        ];
        let assessment = assess_independence(&observations).unwrap();
        assert_eq!(assessment.independent_lower_bound, 1);
        assert_eq!(assessment.independent_upper_bound, 1);
        assert_eq!(assessment.metadata_conflicts.len(), 1);
        assert!(!assessment.fully_resolved());
    }

    #[test]
    fn mixed_signals_are_rejected_instead_of_compared() {
        let a = observation("a", "lineage-a", None, None);
        let mut b = observation("b", "lineage-b", None, None);
        b.signal = "system.memory.utilization".into();
        assert!(matches!(
            assess_independence(&[a, b]),
            Err(IndependenceAssessmentError::MixedSignal { .. })
        ));
    }
}
