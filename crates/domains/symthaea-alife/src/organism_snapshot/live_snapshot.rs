// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Live [`crate::Organism`] bridge for the validated v1 organism snapshot contract.
//!
//! This is a child of `organism_snapshot` so it can construct the raw persistence capsule and
//! consume the validated capability without widening either type's field visibility. The live
//! organism itself already exposes its complete owned state; nested FEP/Markov restoration stays
//! gated by their own validated capabilities.

use std::collections::HashMap;

use symthaea_fep::markov_blanket::MarkovBoundaryOperator;
use symthaea_fep::ActiveInferenceAgent;

use super::{
    OrganismConfigSnapshotV1, OrganismLedgerEntrySnapshotV1, OrganismSnapshotV1,
    ValidatedOrganismSnapshotV1,
};
use crate::Organism;

impl Organism {
    /// Capture every organism-owned causal field at a stable between-step boundary.
    ///
    /// Ledger entries are sorted only in the persistence representation. The live `HashMap` has
    /// no causal iteration contract; sorting gives equal factual ledgers one canonical byte form
    /// without altering population vector order or any live execution order.
    pub fn snapshot_v1(&self) -> OrganismSnapshotV1 {
        let mut ledger = self
            .ledger
            .iter()
            .map(|(&partner_id, &record)| OrganismLedgerEntrySnapshotV1::new(partner_id, record))
            .collect::<Vec<_>>();
        ledger.sort_by_key(|entry| entry.partner_id().raw());

        OrganismSnapshotV1 {
            id: self.id,
            agent: self.agent.snapshot_v1(),
            boundary: self.boundary.snapshot_v1(),
            energy_bits: self.energy.to_bits(),
            config: OrganismConfigSnapshotV1::from_config(self.cfg),
            last_resource_observed_bits: self.last_resource_observed.to_bits(),
            ledger,
            lineage_id: self.lineage_id,
            generation: self.generation,
        }
    }

    /// Restore a population-owned organism only from the non-serializable capability produced by
    /// [`OrganismSnapshotV1::validate`]. No constructor defaults or fresh RNG seeds are consulted.
    pub fn from_validated_snapshot_v1(snapshot: &ValidatedOrganismSnapshotV1) -> Self {
        let ledger = snapshot
            .ledger
            .iter()
            .map(|entry| (entry.partner_id(), entry.record()))
            .collect::<HashMap<_, _>>();

        Self {
            id: snapshot.id,
            agent: ActiveInferenceAgent::from_validated_snapshot_v1(&snapshot.agent),
            boundary: MarkovBoundaryOperator::from_validated_snapshot_v1(&snapshot.boundary),
            energy: snapshot.energy,
            cfg: snapshot.config,
            last_resource_observed: snapshot.last_resource_observed,
            ledger,
            lineage_id: snapshot.lineage_id,
            generation: snapshot.generation,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AgentId, AgentIdAllocator, InteractionRecord, OrganismConfig};

    fn allocated_ids() -> (AgentId, AgentId, AgentId) {
        let mut ids = AgentIdAllocator::new();
        (ids.allocate(), ids.allocate(), ids.allocate())
    }

    fn snapshot_json(organism: &Organism) -> String {
        serde_json::to_string(&organism.snapshot_v1()).expect("serialize organism snapshot")
    }

    fn restore_from_json(organism: &Organism) -> Organism {
        let encoded = snapshot_json(organism);
        let raw: OrganismSnapshotV1 =
            serde_json::from_str(&encoded).expect("deserialize organism snapshot");
        let validated = raw.validate().expect("validate organism snapshot");
        Organism::from_validated_snapshot_v1(&validated)
    }

    fn drive_asocial(organism: &mut Organism, start: usize, count: usize) {
        for i in start..start + count {
            let resource = 0.15 + (((i * 37) % 79) as f64 / 100.0);
            // Selection still runs and consumes the restored stochastic action stream; forcing the
            // external consequence simply guarantees both action paths are exercised repeatedly.
            let _ = organism.tick(resource.min(0.95), Some(i % 2));
        }
    }

    fn drive_social(
        organism: &mut Organism,
        partner_id: AgentId,
        start: usize,
        count: usize,
    ) {
        for i in start..start + count {
            let resource = 0.25 + (((i * 29) % 61) as f64 / 100.0);
            let before = organism
                .ledger
                .get(&partner_id)
                .copied()
                .unwrap_or_default();
            let (tick, pending) = organism.act_social(
                resource.min(0.9),
                Some(i % 3),
                Some((partner_id, before)),
            );

            // Deterministic factual partner consequence, mirroring Population's ordering: apply
            // the realized cross-organism result, write the factual ledger, then learn from it.
            let received = if i % 4 == 0 { 0.0125 } else { 0.0 };
            organism.energy = (organism.energy + received).min(1.0);
            let mut realized = before;
            realized.encounter_count = realized
                .encounter_count
                .checked_add(1)
                .expect("test encounter count");
            realized.given_to_partner += tick.transfer_given;
            realized.received_from_partner += received;
            organism.ledger.insert(partner_id, realized);
            organism.learn_from_realized_outcome(pending, Some((partner_id, realized)));
        }
    }

    #[test]
    fn serialized_validated_restore_preserves_exact_asocial_future() {
        let (id, _partner, _lineage) = allocated_ids();
        let cfg = OrganismConfig {
            action_temperature: 0.73,
            perceptual_grain: Some(0.125),
            spoilage_sigma: Some(0.22),
            ..OrganismConfig::default()
        };
        let mut uninterrupted = Organism::new(cfg, 0x0A11_FEED).with_id(id);
        drive_asocial(&mut uninterrupted, 0, 19);

        let mut restored = restore_from_json(&uninterrupted);
        assert_eq!(snapshot_json(&uninterrupted), snapshot_json(&restored));

        for i in 19..83 {
            drive_asocial(&mut uninterrupted, i, 1);
            drive_asocial(&mut restored, i, 1);
            assert_eq!(
                snapshot_json(&uninterrupted),
                snapshot_json(&restored),
                "asocial organism diverged after restored continuation step {i}"
            );
        }
    }

    #[test]
    fn serialized_validated_restore_preserves_social_ledger_and_future() {
        let (id, partner_id, lineage_id) = allocated_ids();
        let cfg = OrganismConfig {
            social_enabled: true,
            transfer_quantum: 0.035,
            action_temperature: 0.61,
            perceptual_grain: Some(0.1),
            ..OrganismConfig::default()
        };
        let mut uninterrupted = Organism::new(cfg, 0x50C1_A11F)
            .with_id(id)
            .with_lineage(lineage_id, 3);
        drive_social(&mut uninterrupted, partner_id, 0, 23);

        let mut restored = restore_from_json(&uninterrupted);
        assert_eq!(snapshot_json(&uninterrupted), snapshot_json(&restored));
        assert_eq!(uninterrupted.ledger, restored.ledger);

        for i in 23..91 {
            drive_social(&mut uninterrupted, partner_id, i, 1);
            drive_social(&mut restored, partner_id, i, 1);
            assert_eq!(
                snapshot_json(&uninterrupted),
                snapshot_json(&restored),
                "social organism diverged after restored continuation step {i}"
            );
            assert_eq!(uninterrupted.ledger, restored.ledger);
        }
    }

    #[test]
    fn persistence_canonicalizes_ledger_without_mutating_live_map() {
        let (id, partner_a, partner_b) = allocated_ids();
        let mut organism = Organism::new(
            OrganismConfig {
                social_enabled: true,
                ..OrganismConfig::default()
            },
            7,
        )
        .with_id(id);
        organism.ledger.insert(
            partner_b,
            InteractionRecord {
                given_to_partner: 0.2,
                received_from_partner: 0.4,
                encounter_count: 2,
            },
        );
        organism.ledger.insert(
            partner_a,
            InteractionRecord {
                given_to_partner: 0.1,
                received_from_partner: 0.3,
                encounter_count: 1,
            },
        );

        let before = organism.ledger.clone();
        let raw = organism.snapshot_v1();
        assert_eq!(organism.ledger, before, "snapshot must not reorder live HashMap state");
        let validated = raw.validate().expect("canonical ledger snapshot");
        let ids = validated
            .ledger()
            .iter()
            .map(OrganismLedgerEntrySnapshotV1::partner_id)
            .collect::<Vec<_>>();
        assert!(ids.windows(2).all(|pair| pair[0].raw() < pair[1].raw()));
    }
}
