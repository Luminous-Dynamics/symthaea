// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Narrow evaluator projection of an ingress-verified MACE dependency sidecar.
//!
//! This is not a persistent evidence graph or an independent trust authority.
//! Source authenticity and lineage completeness come from the ingress receipt
//! verifier; this module only validates the sidecar binding and computes the
//! event-removal closure required by the counterfactual evaluator.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use broca_mace_ingress::{DependencyLineageStatusV1, EvidenceDependencySidecarV1};
use broca_mace_policy::Experience;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum SidecarIndexError {
    IntegrityFailure,
    CardinalityMismatch { memory: usize, records: usize },
    DuplicateMemoryEvent(u32),
    MissingCandidateEvent(u32),
    EvidenceBindingMismatch(u32),
    DependencyCycle,
}

#[derive(Clone, Debug)]
pub(crate) struct ReceiptDependencyProjection {
    children_by_parent: BTreeMap<u32, BTreeSet<u32>>,
    event_ids: BTreeSet<u32>,
    incomplete_event_ids: BTreeSet<u32>,
    unresolved_parent_reference_count: usize,
    sidecar_sha256: String,
}

impl ReceiptDependencyProjection {
    pub(crate) fn build(
        memory: &[Experience],
        sidecar: &EvidenceDependencySidecarV1,
    ) -> Result<Self, SidecarIndexError> {
        if !sidecar.verify_integrity() {
            return Err(SidecarIndexError::IntegrityFailure);
        }
        if sidecar.records().len() != memory.len() {
            return Err(SidecarIndexError::CardinalityMismatch {
                memory: memory.len(),
                records: sidecar.records().len(),
            });
        }

        let mut candidate_by_id = BTreeMap::<u32, &Experience>::new();
        for item in memory {
            if candidate_by_id.insert(item.event_id, item).is_some() {
                return Err(SidecarIndexError::DuplicateMemoryEvent(item.event_id));
            }
        }
        let event_ids: BTreeSet<u32> = candidate_by_id.keys().copied().collect();
        let mut children_by_parent: BTreeMap<u32, BTreeSet<u32>> =
            event_ids.iter().map(|id| (*id, BTreeSet::new())).collect();
        let mut indegree: BTreeMap<u32, usize> =
            event_ids.iter().map(|id| (*id, 0_usize)).collect();
        let mut incomplete_event_ids = BTreeSet::new();
        let mut unresolved_parent_reference_count = 0_usize;

        for record in sidecar.records() {
            let item = candidate_by_id
                .get(&record.event_id)
                .ok_or(SidecarIndexError::MissingCandidateEvent(record.event_id))?;
            if item.action != record.action
                || item.scope != record.scope
                || item.source_family != record.source_family
                || item.source_root_digest.as_bytes() != record.source_root_digest.as_bytes()
            {
                return Err(SidecarIndexError::EvidenceBindingMismatch(record.event_id));
            }

            if record.dependency_lineage_status != DependencyLineageStatusV1::Complete {
                incomplete_event_ids.insert(record.event_id);
            }

            for parent_id in &record.derived_from_event_ids {
                if !event_ids.contains(parent_id) {
                    // A declared parent can be outside the selected memory
                    // pool. Preserve the unresolved dependency, never assume
                    // the remaining child is independent.
                    unresolved_parent_reference_count += 1;
                    incomplete_event_ids.insert(record.event_id);
                    continue;
                }
                if children_by_parent
                    .get_mut(parent_id)
                    .expect("every candidate event has a child set")
                    .insert(record.event_id)
                {
                    *indegree
                        .get_mut(&record.event_id)
                        .expect("every candidate event has an indegree") += 1;
                }
            }
        }

        // Derived-view dependencies must form a DAG for deterministic closure.
        let mut ready: BTreeSet<u32> = indegree
            .iter()
            .filter_map(|(id, degree)| (*degree == 0).then_some(*id))
            .collect();
        let mut visited = 0_usize;
        while let Some(id) = ready.pop_first() {
            visited += 1;
            if let Some(children) = children_by_parent.get(&id) {
                for child in children {
                    let degree = indegree
                        .get_mut(child)
                        .expect("every child has an indegree");
                    *degree -= 1;
                    if *degree == 0 {
                        ready.insert(*child);
                    }
                }
            }
        }
        if visited != event_ids.len() {
            return Err(SidecarIndexError::DependencyCycle);
        }

        Ok(Self {
            children_by_parent,
            event_ids,
            incomplete_event_ids,
            unresolved_parent_reference_count,
            sidecar_sha256: sidecar.canonical_digest().to_owned(),
        })
    }

    /// Remove each selected root event and all known derived descendants.
    pub(crate) fn dependent_closure(
        &self,
        seeds: impl IntoIterator<Item = u32>,
    ) -> Vec<u32> {
        let mut closure = BTreeSet::new();
        let mut queue = VecDeque::new();
        for id in seeds {
            if self.event_ids.contains(&id) && closure.insert(id) {
                queue.push_back(id);
            }
        }
        while let Some(id) = queue.pop_front() {
            if let Some(children) = self.children_by_parent.get(&id) {
                for child in children {
                    if closure.insert(*child) {
                        queue.push_back(*child);
                    }
                }
            }
        }
        closure.into_iter().collect()
    }

    pub(crate) fn lineage_incomplete_event_count(&self) -> usize {
        self.incomplete_event_ids.len()
    }

    pub(crate) fn unresolved_parent_reference_count(&self) -> usize {
        self.unresolved_parent_reference_count
    }

    pub(crate) fn sidecar_sha256(&self) -> &str {
        &self.sidecar_sha256
    }
}
