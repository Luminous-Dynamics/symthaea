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
    parents_by_child: BTreeMap<u32, BTreeSet<u32>>,
    event_ids: BTreeSet<u32>,
    incomplete_event_ids: BTreeSet<u32>,
    unresolved_parent_reference_count: usize,
    sidecar_sha256: String,
    ingress_receipts_verified: bool,
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
        let mut parents_by_child: BTreeMap<u32, BTreeSet<u32>> =
            event_ids.iter().map(|id| (*id, BTreeSet::new())).collect();
        let mut indegree: BTreeMap<u32, usize> =
            event_ids.iter().map(|id| (*id, 0_usize)).collect();
        let ingress_receipts_verified = sidecar.has_verified_ingress_provenance();
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

            if record.dependency_lineage_status != DependencyLineageStatusV1::Complete
                || !ingress_receipts_verified
            {
                // A self-consistent JSON digest does not authenticate a receipt
                // issuer. Keep dependency edges for diagnostic knockouts, but
                // never qualify deserialized sidecars as complete lineage.
                incomplete_event_ids.insert(record.event_id);
            }

            for parent_id in &record.derived_from_event_ids {
                // Preserve the declared parent identity even when it is not
                // present in the selected pool. It remains useful for detecting
                // a control that descends from a knocked-out target.
                parents_by_child
                    .get_mut(&record.event_id)
                    .expect("every candidate event has a parent set")
                    .insert(*parent_id);
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
            parents_by_child,
            event_ids,
            incomplete_event_ids,
            unresolved_parent_reference_count,
            sidecar_sha256: sidecar.canonical_digest().to_owned(),
            ingress_receipts_verified,
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

    /// True only when every event has complete declared ancestry and no
    /// parent reference outside the selected candidate pool.
    pub(crate) fn all_lineage_complete(&self, event_ids: &[u32]) -> bool {
        event_ids.iter().all(|id| {
            self.event_ids.contains(id) && !self.incomplete_event_ids.contains(id)
        })
    }

    /// Reject a control whose own declared ancestry intersects the target
    /// knockout, even when provider/root display identities differ.
    pub(crate) fn depends_on_any(
        &self,
        event_id: u32,
        target_ids: &BTreeSet<u32>,
    ) -> bool {
        let mut visited = BTreeSet::new();
        let mut stack = self
            .parents_by_child
            .get(&event_id)
            .map(|parents| parents.iter().copied().collect::<Vec<_>>())
            .unwrap_or_default();
        while let Some(parent) = stack.pop() {
            if !visited.insert(parent) {
                continue;
            }
            if target_ids.contains(&parent) {
                return true;
            }
            if let Some(ancestors) = self.parents_by_child.get(&parent) {
                stack.extend(ancestors.iter().copied());
            }
        }
        false
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

    pub(crate) fn ingress_receipts_verified(&self) -> bool {
        self.ingress_receipts_verified
    }
}
