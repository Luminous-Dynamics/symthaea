// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Versioned persistence contract for [`crate::EncounterScheduler`] at a stable pairing boundary.
//!
//! Pairing is causal state. Random mode depends on the scheduler's private RNG stream; fixed-partner
//! mode additionally carries a persistent partner map. Restoring organisms without restoring these
//! values changes who interacts with whom even when every organism is otherwise identical.
//!
//! The live scheduler may construct this raw persistence shape, but deserialization still does not
//! create restore authority: callers must validate the raw snapshot before recreating a scheduler.
//!
//! A subtle production invariant matters here: the fixed-partner map is **not globally symmetric**.
//! When a partner dies, the surviving agent can be rematched while the dead agent's stale mapping
//! remains in the private map. Therefore a validator that required every `a -> b` to have `b -> a`
//! would reject legitimate live state. Higher population-level validation may later reason about
//! symmetry among the *currently living* subset, but this contract preserves the full scheduler map.

use serde::{Deserialize, Serialize};

use crate::{AgentId, PairingMode};

/// Serializable v1 representation of [`PairingMode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EncounterPairingModeSnapshotV1 {
    Random,
    FixedPartners,
}

impl EncounterPairingModeSnapshotV1 {
    pub(crate) fn from_mode(mode: PairingMode) -> Self {
        match mode {
            PairingMode::Random => Self::Random,
            PairingMode::FixedPartners => Self::FixedPartners,
        }
    }

    pub fn pairing_mode(self) -> PairingMode {
        match self {
            Self::Random => PairingMode::Random,
            Self::FixedPartners => PairingMode::FixedPartners,
        }
    }
}

/// Canonically ordered persistence entry for the scheduler's private fixed-partner map.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EncounterFixedPartnerEntrySnapshotV1 {
    agent_id: AgentId,
    partner_id: AgentId,
}

impl EncounterFixedPartnerEntrySnapshotV1 {
    pub(crate) fn new(agent_id: AgentId, partner_id: AgentId) -> Self {
        Self {
            agent_id,
            partner_id,
        }
    }

    pub fn agent_id(&self) -> AgentId {
        self.agent_id
    }

    pub fn partner_id(&self) -> AgentId {
        self.partner_id
    }
}

/// Raw serializable scheduler state. Fixed entries must be strictly sorted by `agent_id.raw()`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncounterSchedulerSnapshotV1 {
    mode: EncounterPairingModeSnapshotV1,
    fixed_partners: Vec<EncounterFixedPartnerEntrySnapshotV1>,
    rng_state: u64,
}

/// Non-serializable capability produced only after scheduler-state validation.
#[derive(Debug, Clone)]
pub struct ValidatedEncounterSchedulerSnapshotV1 {
    mode: EncounterPairingModeSnapshotV1,
    fixed_partners: Vec<EncounterFixedPartnerEntrySnapshotV1>,
    rng_state: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EncounterSchedulerSnapshotErrorV1 {
    ZeroRngState,
    RandomModeHasFixedPartners,
    FixedPartnersNotStrictlySorted,
    ReservedIdentity { field: &'static str },
    SelfPartner { agent_id: AgentId },
}

impl EncounterSchedulerSnapshotV1 {
    pub(crate) fn from_live_parts(
        mode: PairingMode,
        fixed_partners: Vec<EncounterFixedPartnerEntrySnapshotV1>,
        rng_state: u64,
    ) -> Self {
        Self {
            mode: EncounterPairingModeSnapshotV1::from_mode(mode),
            fixed_partners,
            rng_state,
        }
    }

    /// Consume raw persistence and produce scheduler restore authority only if the v1 structural
    /// invariants hold.
    pub fn validate(
        self,
    ) -> Result<ValidatedEncounterSchedulerSnapshotV1, EncounterSchedulerSnapshotErrorV1> {
        if self.rng_state == 0 {
            return Err(EncounterSchedulerSnapshotErrorV1::ZeroRngState);
        }
        validate_fixed_partner_entries(self.mode, &self.fixed_partners)?;
        Ok(ValidatedEncounterSchedulerSnapshotV1 {
            mode: self.mode,
            fixed_partners: self.fixed_partners,
            rng_state: self.rng_state,
        })
    }
}

impl ValidatedEncounterSchedulerSnapshotV1 {
    pub fn mode(&self) -> PairingMode {
        self.mode.pairing_mode()
    }

    pub fn rng_state(&self) -> u64 {
        self.rng_state
    }

    pub fn fixed_partners(&self) -> &[EncounterFixedPartnerEntrySnapshotV1] {
        &self.fixed_partners
    }
}

fn validate_fixed_partner_entries(
    mode: EncounterPairingModeSnapshotV1,
    entries: &[EncounterFixedPartnerEntrySnapshotV1],
) -> Result<(), EncounterSchedulerSnapshotErrorV1> {
    if mode == EncounterPairingModeSnapshotV1::Random && !entries.is_empty() {
        return Err(EncounterSchedulerSnapshotErrorV1::RandomModeHasFixedPartners);
    }

    let mut previous = None;
    for entry in entries {
        if entry.agent_id == AgentId::UNALLOCATED {
            return Err(EncounterSchedulerSnapshotErrorV1::ReservedIdentity {
                field: "fixed_partners.agent_id",
            });
        }
        if entry.partner_id == AgentId::UNALLOCATED {
            return Err(EncounterSchedulerSnapshotErrorV1::ReservedIdentity {
                field: "fixed_partners.partner_id",
            });
        }
        if entry.agent_id == entry.partner_id {
            return Err(EncounterSchedulerSnapshotErrorV1::SelfPartner {
                agent_id: entry.agent_id,
            });
        }

        let raw = entry.agent_id.raw();
        if previous.is_some_and(|prior| raw <= prior) {
            return Err(EncounterSchedulerSnapshotErrorV1::FixedPartnersNotStrictlySorted);
        }
        previous = Some(raw);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn allocated_ids(n: usize) -> Vec<AgentId> {
        let mut allocator = crate::AgentIdAllocator::new();
        (0..n).map(|_| allocator.allocate()).collect()
    }

    #[test]
    fn pairing_mode_snapshot_round_trips_without_reinterpretation() {
        assert_eq!(
            EncounterPairingModeSnapshotV1::from_mode(PairingMode::Random).pairing_mode(),
            PairingMode::Random
        );
        assert_eq!(
            EncounterPairingModeSnapshotV1::from_mode(PairingMode::FixedPartners).pairing_mode(),
            PairingMode::FixedPartners
        );
    }

    #[test]
    fn random_mode_requires_an_empty_fixed_partner_map() {
        let ids = allocated_ids(2);
        let raw = EncounterSchedulerSnapshotV1 {
            mode: EncounterPairingModeSnapshotV1::Random,
            fixed_partners: vec![EncounterFixedPartnerEntrySnapshotV1::new(ids[0], ids[1])],
            rng_state: 7,
        };
        assert_eq!(
            raw.validate().unwrap_err(),
            EncounterSchedulerSnapshotErrorV1::RandomModeHasFixedPartners
        );
    }

    #[test]
    fn zero_rng_state_fails_closed() {
        let raw = EncounterSchedulerSnapshotV1 {
            mode: EncounterPairingModeSnapshotV1::Random,
            fixed_partners: Vec::new(),
            rng_state: 0,
        };
        assert_eq!(
            raw.validate().unwrap_err(),
            EncounterSchedulerSnapshotErrorV1::ZeroRngState
        );
    }

    #[test]
    fn fixed_partner_map_requires_canonical_key_order() {
        let ids = allocated_ids(4);
        let entries = vec![
            EncounterFixedPartnerEntrySnapshotV1::new(ids[2], ids[3]),
            EncounterFixedPartnerEntrySnapshotV1::new(ids[0], ids[1]),
        ];
        assert_eq!(
            validate_fixed_partner_entries(EncounterPairingModeSnapshotV1::FixedPartners, &entries),
            Err(EncounterSchedulerSnapshotErrorV1::FixedPartnersNotStrictlySorted)
        );
    }

    #[test]
    fn stale_asymmetric_fixed_entries_are_legitimate_scheduler_history() {
        let ids = allocated_ids(4);
        // Historical state after an original 0<->1 pairing, death/rematch of 0, and a new
        // live 1<->2 relationship can legitimately retain a stale 0->1 entry. Global symmetry
        // is therefore intentionally not a snapshot invariant.
        let entries = vec![
            EncounterFixedPartnerEntrySnapshotV1::new(ids[0], ids[1]),
            EncounterFixedPartnerEntrySnapshotV1::new(ids[1], ids[2]),
            EncounterFixedPartnerEntrySnapshotV1::new(ids[2], ids[1]),
        ];
        validate_fixed_partner_entries(EncounterPairingModeSnapshotV1::FixedPartners, &entries)
            .expect("stale asymmetric history is valid fixed-partner state");
    }

    #[test]
    fn self_partner_is_never_canonical_scheduler_state() {
        let ids = allocated_ids(1);
        let entries = [EncounterFixedPartnerEntrySnapshotV1::new(ids[0], ids[0])];
        assert_eq!(
            validate_fixed_partner_entries(EncounterPairingModeSnapshotV1::FixedPartners, &entries),
            Err(EncounterSchedulerSnapshotErrorV1::SelfPartner { agent_id: ids[0] })
        );
    }
}
