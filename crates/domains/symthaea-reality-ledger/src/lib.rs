// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Symthaea Reality Ledger v1.
//!
//! The ledger keeps a hard distinction between:
//! - physical-grounded observations;
//! - committed digital worlds;
//! - counterfactual branches;
//! - replays;
//! - dreams;
//! - imported worlds; and
//! - unresolved provenance.
//!
//! The crate is deliberately host-neutral. Bevy/Symtropy, robotics, dream
//! engines, futures simulations and other hosts may supply records, but this
//! crate does not grant mutation authority and does not decide metaphysical or
//! phenomenological questions.

pub mod commit;
pub mod context;
pub mod ledger;
pub mod memory;
pub mod types;

pub use commit::{CounterfactualCommitReceipt, RealityCommitError};
pub use context::{RealityContextError, RealityContextStack};
pub use ledger::{RealityLedger, RealityLedgerError, RealityRecord, RealityRecordKind};
pub use memory::{
    MemoryAdmissionClass, MemoryAdmissionError, MemoryAdmissionReceipt, assess_memory_admission,
};
pub use types::{
    EvidenceSource, RealityLayer, RealityRecordId, RealityTypeError, WorldDescriptor, WorldId,
    WorldLineageId, WorldOrigin, WorldParentRef, WorldRelation,
};

#[cfg(test)]
mod integration_tests {
    use super::*;

    fn studio() -> WorldDescriptor {
        WorldDescriptor {
            world_id: WorldId("studio".into()),
            lineage_id: WorldLineageId("studio-lineage".into()),
            layer: RealityLayer::DigitalCommitted,
            origin: WorldOrigin::DigitalHost { host_kind: "bevy".into() },
            parent: None,
            generation_depth: 0,
            creator_id: "symtropy".into(),
        }
    }

    fn counterfactual(parent: &WorldDescriptor, id: &str) -> WorldDescriptor {
        WorldDescriptor {
            world_id: WorldId(id.into()),
            lineage_id: WorldLineageId(format!("{id}-lineage")),
            layer: RealityLayer::Counterfactual,
            origin: WorldOrigin::CounterfactualBranch,
            parent: Some(WorldParentRef {
                world_id: parent.world_id.clone(),
                lineage_id: parent.lineage_id.clone(),
                relation: WorldRelation::CounterfactualOf,
            }),
            generation_depth: parent.generation_depth + 1,
            creator_id: "ghost-director".into(),
        }
    }

    #[test]
    fn imagined_event_remains_hypothetical_even_when_recalled() {
        let studio = studio();
        let ghost = counterfactual(&studio, "ghost-a");
        let record = RealityRecord {
            record_id: RealityRecordId("ghost-observation".into()),
            sequence: 0,
            world: ghost,
            kind: RealityRecordKind::MemoryCandidate,
            source: EvidenceSource::CounterfactualSimulation { engine_id: "ghost-render".into() },
            revision_id: Some("ghost-rev".into()),
            frame: Some(42),
            content_digest: "pixels-hash".into(),
            previous_record_digest: None,
        };
        let memory = assess_memory_admission(&record).unwrap();
        assert_eq!(memory.class, MemoryAdmissionClass::HypotheticalOnly);
        assert!(!memory.may_claim_happened_in_current_world);
        assert!(!memory.may_claim_physically_observed);
    }

    #[test]
    fn nested_ghost_world_never_changes_parent_descriptor() {
        let studio = studio();
        let ghost = counterfactual(&studio, "ghost-a");
        let mut stack = RealityContextStack::new(studio.clone(), 4).unwrap();
        stack.enter_child(ghost).unwrap();
        assert_eq!(stack.root(), &studio);
        assert_eq!(stack.current().layer, RealityLayer::Counterfactual);
        stack.leave_current().unwrap();
        assert_eq!(stack.current(), &studio);
    }
}
