// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Assumption-scoped monotone capability activation.
//!
//! A closed dependency graph cannot by itself establish operational state. This
//! module therefore requires the caller to state two distinct assumptions:
//!
//! - which capabilities are already treated as available; and
//! - which unavailable capabilities are treated as activatable once their exact
//!   prerequisite expression is satisfied.
//!
//! Only caller-declared activatable capabilities may enter the derived closure.
//! A prerequisite-free definition is **not** automatically treated as available.
//!
//! Core theorem:
//!
//! `ActivationAssumption != Observation != VerifiedAvailability != Authority`.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    CapabilityGraphSnapshotId, CapabilityId, CapabilityRequirementV1, ValidatedCapabilityGraphV1,
};

/// Stable schema for one assumption set.
pub const CAPABILITY_ACTIVATION_ASSUMPTIONS_SCHEMA_V1: &str =
    "symthaea-continuity-capability-activation-assumptions-v1";

const ASSUMPTIONS_DOMAIN: &[u8] = b"symthaea.continuity.capability-activation-assumptions.v1\0";

/// Exact identity of one canonical assumption set against one exact graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct CapabilityActivationAssumptionsId([u8; 32]);

impl CapabilityActivationAssumptionsId {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Serializable candidate describing the assumptions under which activation
/// closure may be computed.
///
/// `available` means only "the caller asks this analysis to treat these
/// capabilities as available." It does not encode how that belief was obtained.
/// `activatable` means only "the caller asks this analysis to assume this
/// capability can become available when its declared prerequisites are met."
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityActivationAssumptionsV1 {
    schema_version: String,
    source_snapshot_id: CapabilityGraphSnapshotId,
    available: Vec<CapabilityId>,
    activatable: Vec<CapabilityId>,
    assumptions_id: CapabilityActivationAssumptionsId,
}

impl CapabilityActivationAssumptionsV1 {
    /// Construct a canonical assumption set bound to one validated graph.
    pub fn new(
        graph: &ValidatedCapabilityGraphV1,
        mut available: Vec<CapabilityId>,
        mut activatable: Vec<CapabilityId>,
    ) -> Result<Self, CapabilityActivationError> {
        canonicalize_ids(&mut available);
        canonicalize_ids(&mut activatable);

        // An already-available capability does not need a second activation
        // assumption. Removing overlap gives one canonical representation.
        let available_set: BTreeSet<_> = available.iter().copied().collect();
        activatable.retain(|capability_id| !available_set.contains(capability_id));

        validate_known(graph, &available)?;
        validate_known(graph, &activatable)?;

        let source_snapshot_id = graph.id();
        let assumptions_id = CapabilityActivationAssumptionsId(hash_assumptions(
            source_snapshot_id,
            &available,
            &activatable,
        ));

        Ok(Self {
            schema_version: CAPABILITY_ACTIVATION_ASSUMPTIONS_SCHEMA_V1.to_owned(),
            source_snapshot_id,
            available,
            activatable,
            assumptions_id,
        })
    }

    pub fn id(&self) -> CapabilityActivationAssumptionsId {
        self.assumptions_id
    }

    pub fn source_snapshot_id(&self) -> CapabilityGraphSnapshotId {
        self.source_snapshot_id
    }

    pub fn available(&self) -> &[CapabilityId] {
        &self.available
    }

    pub fn activatable(&self) -> &[CapabilityId] {
        &self.activatable
    }

    /// Revalidate transported assumptions against the exact graph they name.
    pub fn validate(
        &self,
        graph: &ValidatedCapabilityGraphV1,
    ) -> Result<ValidatedCapabilityActivationAssumptionsV1, CapabilityActivationError> {
        if self.schema_version != CAPABILITY_ACTIVATION_ASSUMPTIONS_SCHEMA_V1 {
            return Err(CapabilityActivationError::UnsupportedSchema(
                self.schema_version.clone(),
            ));
        }
        if self.source_snapshot_id != graph.id() {
            return Err(CapabilityActivationError::SourceSnapshotMismatch {
                declared: self.source_snapshot_id,
                actual: graph.id(),
            });
        }
        if !strictly_sorted(&self.available) {
            return Err(CapabilityActivationError::NonCanonicalAvailable);
        }
        if !strictly_sorted(&self.activatable) {
            return Err(CapabilityActivationError::NonCanonicalActivatable);
        }

        let available_set: BTreeSet<_> = self.available.iter().copied().collect();
        if self
            .activatable
            .iter()
            .any(|capability_id| available_set.contains(capability_id))
        {
            return Err(CapabilityActivationError::OverlappingAssumptions);
        }

        validate_known(graph, &self.available)?;
        validate_known(graph, &self.activatable)?;

        let expected = CapabilityActivationAssumptionsId(hash_assumptions(
            self.source_snapshot_id,
            &self.available,
            &self.activatable,
        ));
        if expected != self.assumptions_id {
            return Err(CapabilityActivationError::AssumptionIdentityMismatch);
        }

        Ok(ValidatedCapabilityActivationAssumptionsV1 {
            inner: self.clone(),
        })
    }
}

/// Non-Serde validated assumption wrapper accepted by closure analysis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedCapabilityActivationAssumptionsV1 {
    inner: CapabilityActivationAssumptionsV1,
}

impl ValidatedCapabilityActivationAssumptionsV1 {
    pub fn id(&self) -> CapabilityActivationAssumptionsId {
        self.inner.id()
    }

    pub fn source_snapshot_id(&self) -> CapabilityGraphSnapshotId {
        self.inner.source_snapshot_id()
    }

    pub fn available(&self) -> &[CapabilityId] {
        self.inner.available()
    }

    pub fn activatable(&self) -> &[CapabilityId] {
        self.inner.activatable()
    }

    pub fn as_raw(&self) -> &CapabilityActivationAssumptionsV1 {
        &self.inner
    }
}

/// One simultaneous activation round.
///
/// Every member in a round had its prerequisite expression satisfied by the
/// state at the **start** of that round. This prevents canonical iteration order
/// from silently changing activation depth.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityActivationRoundV1 {
    ordinal: usize,
    activated: Vec<CapabilityId>,
}

impl CapabilityActivationRoundV1 {
    /// One-based deterministic activation round.
    pub fn ordinal(&self) -> usize {
        self.ordinal
    }

    pub fn activated(&self) -> &[CapabilityId] {
        &self.activated
    }
}

/// Why one activatable capability remains blocked after the monotone closure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockedCapabilityActivationV1 {
    capability_id: CapabilityId,
    unsatisfied: UnsatisfiedCapabilityRequirementV1,
}

impl BlockedCapabilityActivationV1 {
    pub fn capability_id(&self) -> CapabilityId {
        self.capability_id
    }

    pub fn unsatisfied(&self) -> &UnsatisfiedCapabilityRequirementV1 {
        &self.unsatisfied
    }
}

/// Exact residual prerequisite expression under a particular derived closure.
///
/// Satisfied branches are removed from `AllOf`; an `AnyOf` is returned only if
/// every alternative remains unsatisfied. This is an explanation of the input
/// model, not evidence that the missing capabilities exist or can be restored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnsatisfiedCapabilityRequirementV1 {
    Leaf {
        capability_id: CapabilityId,
    },
    AllOf {
        requirements: Vec<UnsatisfiedCapabilityRequirementV1>,
    },
    AnyOf {
        requirements: Vec<UnsatisfiedCapabilityRequirementV1>,
    },
}

/// Derived monotone closure under one exact graph and one validated assumption
/// set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityActivationClosureV1 {
    source_snapshot_id: CapabilityGraphSnapshotId,
    assumptions_id: CapabilityActivationAssumptionsId,
    initially_available: Vec<CapabilityId>,
    activation_rounds: Vec<CapabilityActivationRoundV1>,
    available_after_closure: Vec<CapabilityId>,
    blocked_activatable: Vec<BlockedCapabilityActivationV1>,
}

impl CapabilityActivationClosureV1 {
    pub fn source_snapshot_id(&self) -> CapabilityGraphSnapshotId {
        self.source_snapshot_id
    }

    pub fn assumptions_id(&self) -> CapabilityActivationAssumptionsId {
        self.assumptions_id
    }

    pub fn initially_available(&self) -> &[CapabilityId] {
        &self.initially_available
    }

    pub fn activation_rounds(&self) -> &[CapabilityActivationRoundV1] {
        &self.activation_rounds
    }

    pub fn available_after_closure(&self) -> &[CapabilityId] {
        &self.available_after_closure
    }

    pub fn blocked_activatable(&self) -> &[BlockedCapabilityActivationV1] {
        &self.blocked_activatable
    }

    pub fn is_available_after_closure(&self, capability_id: CapabilityId) -> bool {
        self.available_after_closure
            .binary_search(&capability_id)
            .is_ok()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CapabilityActivationError {
    #[error("unsupported capability activation assumptions schema: {0}")]
    UnsupportedSchema(String),
    #[error("activation assumptions declare graph {declared:?} but analysis uses {actual:?}")]
    SourceSnapshotMismatch {
        declared: CapabilityGraphSnapshotId,
        actual: CapabilityGraphSnapshotId,
    },
    #[error("available capability assumptions must be in strict canonical identity order")]
    NonCanonicalAvailable,
    #[error("activatable capability assumptions must be in strict canonical identity order")]
    NonCanonicalActivatable,
    #[error("available and activatable assumption sets must be disjoint")]
    OverlappingAssumptions,
    #[error("activation assumptions reference capability absent from source graph: {0:?}")]
    UnknownCapability(CapabilityId),
    #[error("stored activation-assumption identity does not match canonical fields")]
    AssumptionIdentityMismatch,
    #[error(
        "validated activation assumption references capability unexpectedly absent from graph: {0:?}"
    )]
    MissingDefinition(CapabilityId),
    #[error("blocked activatable capability has no unsatisfied requirement: {0:?}")]
    InconsistentBlockedState(CapabilityId),
}

/// Compute monotone activation closure under explicitly validated assumptions.
///
/// This function does not read clocks, observations, hardware, files, Mycelix,
/// or authority state. It performs only deterministic Boolean evaluation of the
/// capability prerequisite language.
pub fn derive_capability_activation_closure(
    graph: &ValidatedCapabilityGraphV1,
    assumptions: &ValidatedCapabilityActivationAssumptionsV1,
) -> Result<CapabilityActivationClosureV1, CapabilityActivationError> {
    if assumptions.source_snapshot_id() != graph.id() {
        return Err(CapabilityActivationError::SourceSnapshotMismatch {
            declared: assumptions.source_snapshot_id(),
            actual: graph.id(),
        });
    }

    let initially_available = assumptions.available().to_vec();
    let mut available: BTreeSet<CapabilityId> = initially_available.iter().copied().collect();
    let mut remaining: BTreeSet<CapabilityId> = assumptions.activatable().iter().copied().collect();
    let mut activation_rounds = Vec::new();

    loop {
        let mut next_round = Vec::new();
        for &capability_id in &remaining {
            let definition = graph
                .definition(capability_id)
                .ok_or(CapabilityActivationError::MissingDefinition(capability_id))?;
            let satisfied = definition
                .requirements()
                .is_none_or(|requirement| requirement_satisfied(requirement, &available));
            if satisfied {
                next_round.push(capability_id);
            }
        }

        if next_round.is_empty() {
            break;
        }

        for capability_id in &next_round {
            available.insert(*capability_id);
            remaining.remove(capability_id);
        }
        activation_rounds.push(CapabilityActivationRoundV1 {
            ordinal: activation_rounds.len() + 1,
            activated: next_round,
        });
    }

    let available_after_closure: Vec<CapabilityId> = available.iter().copied().collect();
    let mut blocked_activatable = Vec::with_capacity(remaining.len());
    for capability_id in remaining {
        let definition = graph
            .definition(capability_id)
            .ok_or(CapabilityActivationError::MissingDefinition(capability_id))?;
        let Some(requirement) = definition.requirements() else {
            return Err(CapabilityActivationError::InconsistentBlockedState(
                capability_id,
            ));
        };
        let Some(unsatisfied) = explain_unsatisfied(requirement, &available) else {
            return Err(CapabilityActivationError::InconsistentBlockedState(
                capability_id,
            ));
        };
        blocked_activatable.push(BlockedCapabilityActivationV1 {
            capability_id,
            unsatisfied,
        });
    }

    Ok(CapabilityActivationClosureV1 {
        source_snapshot_id: graph.id(),
        assumptions_id: assumptions.id(),
        initially_available,
        activation_rounds,
        available_after_closure,
        blocked_activatable,
    })
}

fn requirement_satisfied(
    requirement: &CapabilityRequirementV1,
    available: &BTreeSet<CapabilityId>,
) -> bool {
    match requirement {
        CapabilityRequirementV1::Leaf { capability_id } => available.contains(capability_id),
        CapabilityRequirementV1::AllOf { requirements } => requirements
            .iter()
            .all(|requirement| requirement_satisfied(requirement, available)),
        CapabilityRequirementV1::AnyOf { requirements } => requirements
            .iter()
            .any(|requirement| requirement_satisfied(requirement, available)),
    }
}

fn explain_unsatisfied(
    requirement: &CapabilityRequirementV1,
    available: &BTreeSet<CapabilityId>,
) -> Option<UnsatisfiedCapabilityRequirementV1> {
    match requirement {
        CapabilityRequirementV1::Leaf { capability_id } => (!available.contains(capability_id))
            .then_some(UnsatisfiedCapabilityRequirementV1::Leaf {
                capability_id: *capability_id,
            }),
        CapabilityRequirementV1::AllOf { requirements } => {
            let unsatisfied: Vec<_> = requirements
                .iter()
                .filter_map(|requirement| explain_unsatisfied(requirement, available))
                .collect();
            (!unsatisfied.is_empty()).then_some(UnsatisfiedCapabilityRequirementV1::AllOf {
                requirements: unsatisfied,
            })
        }
        CapabilityRequirementV1::AnyOf { requirements } => {
            let mut unsatisfied = Vec::with_capacity(requirements.len());
            for requirement in requirements {
                let child = explain_unsatisfied(requirement, available)?;
                unsatisfied.push(child);
            }
            Some(UnsatisfiedCapabilityRequirementV1::AnyOf {
                requirements: unsatisfied,
            })
        }
    }
}

fn validate_known(
    graph: &ValidatedCapabilityGraphV1,
    capability_ids: &[CapabilityId],
) -> Result<(), CapabilityActivationError> {
    for &capability_id in capability_ids {
        if graph.definition(capability_id).is_none() {
            return Err(CapabilityActivationError::UnknownCapability(capability_id));
        }
    }
    Ok(())
}

fn canonicalize_ids(capability_ids: &mut Vec<CapabilityId>) {
    capability_ids.sort_unstable();
    capability_ids.dedup();
}

fn strictly_sorted(capability_ids: &[CapabilityId]) -> bool {
    capability_ids.windows(2).all(|pair| pair[0] < pair[1])
}

fn hash_assumptions(
    source_snapshot_id: CapabilityGraphSnapshotId,
    available: &[CapabilityId],
    activatable: &[CapabilityId],
) -> [u8; 32] {
    let mut bytes = Vec::with_capacity(96 + 32 * (available.len() + activatable.len()));
    put_str(&mut bytes, CAPABILITY_ACTIVATION_ASSUMPTIONS_SCHEMA_V1);
    bytes.extend_from_slice(source_snapshot_id.as_bytes());
    put_len(&mut bytes, available.len());
    for capability_id in available {
        bytes.extend_from_slice(capability_id.as_bytes());
    }
    put_len(&mut bytes, activatable.len());
    for capability_id in activatable {
        bytes.extend_from_slice(capability_id.as_bytes());
    }

    let mut hasher = blake3::Hasher::new();
    hasher.update(ASSUMPTIONS_DOMAIN);
    hasher.update(&bytes);
    *hasher.finalize().as_bytes()
}

fn put_len(out: &mut Vec<u8>, len: usize) {
    out.extend_from_slice(&(len as u64).to_le_bytes());
}

fn put_str(out: &mut Vec<u8>, value: &str) {
    put_len(out, value.len());
    out.extend_from_slice(value.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CapabilityDefinitionV1, CapabilityGraphSnapshotV1};

    fn id(name: &str) -> CapabilityId {
        CapabilityId::new("org.example", name).unwrap()
    }

    fn leaf(name: &str) -> CapabilityDefinitionV1 {
        CapabilityDefinitionV1::new("org.example", name, None).unwrap()
    }

    fn definition(name: &str, requirements: CapabilityRequirementV1) -> CapabilityDefinitionV1 {
        CapabilityDefinitionV1::new("org.example", name, Some(requirements)).unwrap()
    }

    fn graph(definitions: Vec<CapabilityDefinitionV1>) -> ValidatedCapabilityGraphV1 {
        CapabilityGraphSnapshotV1::new(definitions)
            .unwrap()
            .validate()
            .unwrap()
    }

    fn assumptions(
        graph: &ValidatedCapabilityGraphV1,
        available: Vec<CapabilityId>,
        activatable: Vec<CapabilityId>,
    ) -> ValidatedCapabilityActivationAssumptionsV1 {
        CapabilityActivationAssumptionsV1::new(graph, available, activatable)
            .unwrap()
            .validate(graph)
            .unwrap()
    }

    #[test]
    fn assumption_constructor_canonicalizes_deduplicates_and_overlap() {
        let a = leaf("a");
        let b = leaf("b");
        let a_id = a.id();
        let b_id = b.id();
        let graph = graph(vec![b, a]);

        let left = CapabilityActivationAssumptionsV1::new(
            &graph,
            vec![a_id, a_id],
            vec![a_id, b_id, b_id],
        )
        .unwrap();
        let right = CapabilityActivationAssumptionsV1::new(&graph, vec![a_id], vec![b_id]).unwrap();

        assert_eq!(left, right);
        assert_eq!(left.id(), right.id());
        assert_eq!(left.available(), &[a_id]);
        assert_eq!(left.activatable(), &[b_id]);
    }

    #[test]
    fn unknown_assumption_capability_is_rejected() {
        let graph = graph(vec![leaf("a")]);
        assert_eq!(
            CapabilityActivationAssumptionsV1::new(&graph, vec![id("missing")], vec![]),
            Err(CapabilityActivationError::UnknownCapability(id("missing")))
        );
    }

    #[test]
    fn assumptions_are_bound_to_exact_graph_snapshot() {
        let graph_a = graph(vec![leaf("a")]);
        let graph_b = graph(vec![leaf("a"), leaf("b")]);
        let raw = CapabilityActivationAssumptionsV1::new(
            &graph_a,
            vec![graph_a.definitions()[0].id()],
            vec![],
        )
        .unwrap();

        assert!(matches!(
            raw.validate(&graph_b),
            Err(CapabilityActivationError::SourceSnapshotMismatch { .. })
        ));
    }

    #[test]
    fn prerequisite_free_capability_is_not_implicitly_available() {
        let a = leaf("a");
        let a_id = a.id();
        let graph = graph(vec![a]);
        let assumptions = assumptions(&graph, vec![], vec![]);
        let closure = derive_capability_activation_closure(&graph, &assumptions).unwrap();

        assert!(!closure.is_available_after_closure(a_id));
        assert!(closure.activation_rounds().is_empty());
    }

    #[test]
    fn explicitly_activatable_prerequisite_free_capability_activates() {
        let a = leaf("a");
        let a_id = a.id();
        let graph = graph(vec![a]);
        let assumptions = assumptions(&graph, vec![], vec![a_id]);
        let closure = derive_capability_activation_closure(&graph, &assumptions).unwrap();

        assert!(closure.is_available_after_closure(a_id));
        assert_eq!(closure.activation_rounds().len(), 1);
        assert_eq!(closure.activation_rounds()[0].activated(), &[a_id]);
    }

    #[test]
    fn all_of_requires_every_prerequisite() {
        let a = leaf("a");
        let b = leaf("b");
        let a_id = a.id();
        let b_id = b.id();
        let target = definition(
            "target",
            CapabilityRequirementV1::all_of(vec![
                CapabilityRequirementV1::leaf(a_id),
                CapabilityRequirementV1::leaf(b_id),
            ])
            .unwrap(),
        );
        let target_id = target.id();
        let graph = graph(vec![target, b, a]);
        let assumptions = assumptions(&graph, vec![a_id], vec![target_id]);
        let closure = derive_capability_activation_closure(&graph, &assumptions).unwrap();

        assert!(!closure.is_available_after_closure(target_id));
        let blocked = &closure.blocked_activatable()[0];
        assert_eq!(blocked.capability_id(), target_id);
        assert_eq!(
            blocked.unsatisfied(),
            &UnsatisfiedCapabilityRequirementV1::AllOf {
                requirements: vec![UnsatisfiedCapabilityRequirementV1::Leaf {
                    capability_id: b_id,
                }],
            }
        );
    }

    #[test]
    fn any_of_one_satisfied_alternative_is_enough() {
        let a = leaf("a");
        let b = leaf("b");
        let a_id = a.id();
        let b_id = b.id();
        let target = definition(
            "target",
            CapabilityRequirementV1::any_of(vec![
                CapabilityRequirementV1::leaf(a_id),
                CapabilityRequirementV1::leaf(b_id),
            ])
            .unwrap(),
        );
        let target_id = target.id();
        let graph = graph(vec![target, b, a]);
        let assumptions = assumptions(&graph, vec![b_id], vec![target_id]);
        let closure = derive_capability_activation_closure(&graph, &assumptions).unwrap();

        assert!(closure.is_available_after_closure(target_id));
    }

    #[test]
    fn blocked_any_of_preserves_all_missing_alternatives() {
        let a = leaf("a");
        let b = leaf("b");
        let a_id = a.id();
        let b_id = b.id();
        let target = definition(
            "target",
            CapabilityRequirementV1::any_of(vec![
                CapabilityRequirementV1::leaf(a_id),
                CapabilityRequirementV1::leaf(b_id),
            ])
            .unwrap(),
        );
        let target_id = target.id();
        let graph = graph(vec![target, b, a]);
        let assumptions = assumptions(&graph, vec![], vec![target_id]);
        let closure = derive_capability_activation_closure(&graph, &assumptions).unwrap();

        let UnsatisfiedCapabilityRequirementV1::AnyOf { requirements } =
            closure.blocked_activatable()[0].unsatisfied()
        else {
            panic!("expected AnyOf residual");
        };
        let missing: BTreeSet<_> = requirements
            .iter()
            .map(|requirement| match requirement {
                UnsatisfiedCapabilityRequirementV1::Leaf { capability_id } => *capability_id,
                _ => panic!("expected leaf alternative"),
            })
            .collect();
        assert_eq!(missing, BTreeSet::from([a_id, b_id]));
    }

    #[test]
    fn transitive_activation_uses_simultaneous_rounds() {
        let a = leaf("a");
        let a_id = a.id();
        let b = definition("b", CapabilityRequirementV1::leaf(a_id));
        let b_id = b.id();
        let c = definition("c", CapabilityRequirementV1::leaf(b_id));
        let c_id = c.id();
        let graph = graph(vec![c, a, b]);
        let assumptions = assumptions(&graph, vec![a_id], vec![c_id, b_id]);
        let closure = derive_capability_activation_closure(&graph, &assumptions).unwrap();

        assert_eq!(closure.activation_rounds().len(), 2);
        assert_eq!(closure.activation_rounds()[0].activated(), &[b_id]);
        assert_eq!(closure.activation_rounds()[1].activated(), &[c_id]);
        assert!(closure.is_available_after_closure(c_id));
    }

    #[test]
    fn unseeded_mutual_cycle_does_not_spontaneously_bootstrap() {
        let a_id = id("a");
        let b_id = id("b");
        let a = definition("a", CapabilityRequirementV1::leaf(b_id));
        let b = definition("b", CapabilityRequirementV1::leaf(a_id));
        let graph = graph(vec![b, a]);
        let assumptions = assumptions(&graph, vec![], vec![a_id, b_id]);
        let closure = derive_capability_activation_closure(&graph, &assumptions).unwrap();

        assert!(closure.activation_rounds().is_empty());
        assert_eq!(closure.blocked_activatable().len(), 2);
        assert!(!closure.is_available_after_closure(a_id));
        assert!(!closure.is_available_after_closure(b_id));
    }

    #[test]
    fn explicit_seed_can_break_a_cycle_without_scc_magic() {
        let a_id = id("a");
        let b_id = id("b");
        let a = definition("a", CapabilityRequirementV1::leaf(b_id));
        let b = definition("b", CapabilityRequirementV1::leaf(a_id));
        let graph = graph(vec![b, a]);
        let assumptions = assumptions(&graph, vec![a_id], vec![b_id]);
        let closure = derive_capability_activation_closure(&graph, &assumptions).unwrap();

        assert!(closure.is_available_after_closure(a_id));
        assert!(closure.is_available_after_closure(b_id));
        assert_eq!(closure.activation_rounds().len(), 1);
        assert_eq!(closure.activation_rounds()[0].activated(), &[b_id]);
    }
}
