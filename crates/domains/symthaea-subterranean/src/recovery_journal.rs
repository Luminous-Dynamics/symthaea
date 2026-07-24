// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Two-slot operational checkpoint journal with generation ordering and
//! externally pluggable integrity digests.

use crate::operational_checkpoint::{
    OPERATIONAL_CHECKPOINT_SCHEMA_VERSION, SubterraneanOperationalCheckpoint,
};
use crate::update_control::ArtifactDigest;
use serde::{Deserialize, Serialize};

pub trait JournalDigestProvider {
    fn digest(
        &self,
        generation: u64,
        checkpoint: &SubterraneanOperationalCheckpoint,
    ) -> ArtifactDigest;
}

/// Reproducible test digest. Production deployments should substitute a
/// cryptographic provider rooted in their secure boot/update trust domain.
#[derive(Debug, Clone, Copy, Default)]
pub struct DeterministicJournalDigest;

impl JournalDigestProvider for DeterministicJournalDigest {
    fn digest(
        &self,
        generation: u64,
        checkpoint: &SubterraneanOperationalCheckpoint,
    ) -> ArtifactDigest {
        let bytes = format!("{checkpoint:?}").into_bytes();
        let mut lanes = [
            generation ^ 0x6a09_e667_f3bc_c908,
            generation.rotate_left(13) ^ 0xbb67_ae85_84ca_a73b,
            generation.rotate_left(29) ^ 0x3c6e_f372_fe94_f82b,
            generation.rotate_left(47) ^ 0xa54f_f53a_5f1d_36f1,
        ];
        for (index, byte) in bytes.into_iter().enumerate() {
            let lane = index % lanes.len();
            lanes[lane] ^= (byte as u64).wrapping_add((index as u64).rotate_left(7));
            lanes[lane] = lanes[lane]
                .wrapping_mul(0x1000_0000_01b3)
                .rotate_left(((index + lane) % 61 + 1) as u32);
        }
        let mut output = [0u8; 32];
        for (index, lane) in lanes.into_iter().enumerate() {
            output[index * 8..(index + 1) * 8].copy_from_slice(&lane.to_le_bytes());
        }
        ArtifactDigest(output)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JournalSlot {
    pub generation: u64,
    pub checkpoint: SubterraneanOperationalCheckpoint,
    pub digest: ArtifactDigest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoveryJournalError {
    InvalidSchema,
    GenerationRegression,
    InvalidDigest,
    NoValidSlot,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryJournal {
    slots: [Option<JournalSlot>; 2],
    next_slot: usize,
    latest_generation: u64,
    rejected_slots: u64,
}

impl RecoveryJournal {
    pub fn new() -> Self {
        Self {
            slots: [None, None],
            next_slot: 0,
            latest_generation: 0,
            rejected_slots: 0,
        }
    }

    pub fn latest_generation(&self) -> u64 {
        self.latest_generation
    }

    pub fn rejected_slots(&self) -> u64 {
        self.rejected_slots
    }

    pub fn slots(&self) -> [Option<JournalSlot>; 2] {
        self.slots.clone()
    }

    pub fn write(
        &mut self,
        provider: &impl JournalDigestProvider,
        generation: u64,
        checkpoint: SubterraneanOperationalCheckpoint,
    ) -> Result<(), RecoveryJournalError> {
        if checkpoint.schema_version != OPERATIONAL_CHECKPOINT_SCHEMA_VERSION {
            return Err(RecoveryJournalError::InvalidSchema);
        }
        if generation <= self.latest_generation {
            return Err(RecoveryJournalError::GenerationRegression);
        }
        let digest = provider.digest(generation, &checkpoint);
        if !digest.is_valid() {
            return Err(RecoveryJournalError::InvalidDigest);
        }
        self.slots[self.next_slot] = Some(JournalSlot {
            generation,
            checkpoint,
            digest,
        });
        self.next_slot = (self.next_slot + 1) % self.slots.len();
        self.latest_generation = generation;
        Ok(())
    }

    fn valid_slot(provider: &impl JournalDigestProvider, slot: &JournalSlot) -> bool {
        slot.checkpoint.schema_version == OPERATIONAL_CHECKPOINT_SCHEMA_VERSION
            && slot.digest.is_valid()
            && provider.digest(slot.generation, &slot.checkpoint) == slot.digest
    }

    pub fn latest_valid(
        &mut self,
        provider: &impl JournalDigestProvider,
    ) -> Result<JournalSlot, RecoveryJournalError> {
        let mut valid = self
            .slots
            .iter()
            .filter_map(Option::as_ref)
            .filter(|slot| {
                let is_valid = Self::valid_slot(provider, slot);
                if !is_valid {
                    self.rejected_slots = self.rejected_slots.saturating_add(1);
                }
                is_valid
            })
            .cloned()
            .collect::<Vec<_>>();
        valid.sort_by_key(|slot| slot.generation);
        valid.pop().ok_or(RecoveryJournalError::NoValidSlot)
    }

    #[cfg(test)]
    fn corrupt_latest_digest(&mut self) {
        if let Some(slot) = self
            .slots
            .iter_mut()
            .filter_map(Option::as_mut)
            .max_by_key(|slot| slot.generation)
        {
            slot.digest.0[0] ^= 0xff;
        }
    }
}

impl Default for RecoveryJournal {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embodiment::SubterraneanEmbodiment;
    use symthaea_core::genesis::GenesisSeed;

    #[test]
    fn corrupt_newest_slot_falls_back_to_previous_generation() {
        let provider = DeterministicJournalDigest;
        let embodiment = SubterraneanEmbodiment::new(&GenesisSeed::from_phrase("journal"));
        let checkpoint = embodiment.operational_checkpoint();
        let mut journal = RecoveryJournal::new();
        journal
            .write(&provider, 1, checkpoint.clone())
            .expect("first checkpoint should be accepted");
        journal
            .write(&provider, 2, checkpoint)
            .expect("second checkpoint should be accepted");
        journal.corrupt_latest_digest();
        let restored = journal
            .latest_valid(&provider)
            .expect("older valid slot should remain recoverable");
        assert_eq!(restored.generation, 1);
    }

    #[test]
    fn generation_must_increase_monotonically() {
        let provider = DeterministicJournalDigest;
        let embodiment = SubterraneanEmbodiment::new(&GenesisSeed::from_phrase("generation"));
        let checkpoint = embodiment.operational_checkpoint();
        let mut journal = RecoveryJournal::new();
        journal
            .write(&provider, 4, checkpoint.clone())
            .expect("initial write should succeed");
        assert_eq!(
            journal.write(&provider, 4, checkpoint),
            Err(RecoveryJournalError::GenerationRegression)
        );
    }
}
