// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Conservative checkpoint-fork semantics.
//!
//! Two distinct genesis checkpoints have no shared predecessor and therefore do
//! not, by themselves, prove a fork. They may be two legitimate independent
//! evidence lineages. A deterministic fork claim requires competing successor
//! roots under an actual shared predecessor.

use crate::{CheckpointFork, EvidenceCheckpoint};

/// Detect direct checkpoint forks only when competing children share an actual
/// predecessor root.
///
/// This intentionally suppresses `(index = 0, previous = None)` collisions:
/// without a shared anchor/predecessor there is no evidence that two genesis
/// roots belong to the same lineage. If an application needs to associate
/// independent genesis roots with one administrative lineage, that association
/// must come from a separately authenticated lineage/witness policy rather than
/// being inferred by this hash layer.
pub fn detect_checkpoint_forks(checkpoints: &[EvidenceCheckpoint]) -> Vec<CheckpointFork> {
    crate::checkpoint::detect_checkpoint_forks(checkpoints)
        .into_iter()
        .filter(|fork| fork.previous_checkpoint_root.is_some())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_cogsec::Digest32;

    fn checkpoint(index: u64, previous: Option<u8>, root: u8) -> EvidenceCheckpoint {
        EvidenceCheckpoint {
            schema_version: crate::EVIDENCE_CHECKPOINT_SCHEMA_V1,
            checkpoint_index: index,
            previous_checkpoint_root: previous.map(|byte| Digest32([byte; 32])),
            snapshot_root: Digest32([root.wrapping_add(20); 32]),
            ledger_epoch: 1,
            last_assigned_sequence: index,
            retained_event_count: 0,
            effect_binding_count: 0,
            checkpoint_root: Digest32([root; 32]),
        }
    }

    #[test]
    fn independent_genesis_roots_are_not_called_a_fork() {
        let left = checkpoint(0, None, 1);
        let right = checkpoint(0, None, 2);
        assert!(detect_checkpoint_forks(&[left, right]).is_empty());
    }

    #[test]
    fn competing_children_of_same_predecessor_are_a_fork() {
        let left = checkpoint(4, Some(7), 8);
        let right = checkpoint(4, Some(7), 9);
        let forks = detect_checkpoint_forks(&[left, right]);
        assert_eq!(forks.len(), 1);
        assert_eq!(forks[0].checkpoint_index, 4);
        assert_eq!(forks[0].previous_checkpoint_root, Some(Digest32([7; 32])));
        assert_eq!(forks[0].competing_roots.len(), 2);
    }
}
