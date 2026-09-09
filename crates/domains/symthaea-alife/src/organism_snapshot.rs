// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Versioned persistence contract for one population-owned ALife organism at a stable step boundary.
//!
//! Exact continuation needs more than genome/energy. A live [`crate::Organism`] carries learned
//! active-inference state, Markov-boundary dynamics, factual social history, identity/lineage,
//! physiology, the last true resource observation, and non-heritable configuration.
//!
//! This v1 profile is deliberately scoped to organisms owned by [`crate::Population`] **between**
//! complete `step` / `step_social` calls. It is not a mid-step checkpoint: transient
//! [`crate::PendingSocialLearning`] values exist only while `step_social` is executing and are
//! intentionally absent from this stable-boundary contract.
//!
//! Raw persisted bytes are not execution authority. Nested FEP snapshots must first validate into
//! their own non-serializable capabilities; this layer then cross-checks their read-only structural
//! shapes against the ALife configuration before producing [`ValidatedOrganismSnapshotV1`].
//!
//! `social_enabled == false` does **not** imply an empty ledger. `Population::step_social` records
//! factual encounters for every scheduled pair even when the organism has only the asocial action
//! space, so v1 preserves such history rather than rejecting a legitimate population state.

use serde::{Deserialize, Serialize};
use symthaea_fep::{
    ActiveInferenceAgentSnapshotErrorV1, ActiveInferenceAgentSnapshotShapeV1,
    ActiveInferenceAgentSnapshotV1, MarkovBoundarySnapshotErrorV1, MarkovBoundarySnapshotShapeV1,
    MarkovBoundarySnapshotV1, ValidatedActiveInferenceAgentSnapshotV1,
    ValidatedMarkovBoundarySnapshotV1, active_inference_snapshot_shape_v1,
    markov_boundary_snapshot_shape_v1,
};

use crate::{AgentId, InteractionRecord, OrganismConfig};

const ASOCIAL_DIM_V1: usize = 2;
const SOCIAL_DIM_V1: usize = 6;
const ASOCIAL_ACTIONS_V1: usize = 2;
const SOCIAL_ACTIONS_V1: usize = 3;
const BOUNDARY_SENSORY_DIM_V1: usize = 1;
const BOUNDARY_ACTIVE_DIM_V1: usize = 1;

/// Bit-exact persistence representation of [`OrganismConfig`].
///
/// Floating-point fields are stored as IEEE-754 bits so a round-trip cannot silently choose a
/// nearby decimal representation and thereby change future dynamics.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct OrganismConfigSnapshotV1 {
    set_point_bits: u64,
    metabolic_cost_bits: u64,
    forage_activity_cost_bits: u64,
    forage_efficiency_bits: u64,
    goal_precision_bits: u64,
    effective_temperature_bits: u64,
    dissipation_rate_bits: u64,
    death_energy_threshold_bits: u64,
    action_temperature_bits: u64,
    perceptual_grain_bits: Option<u64>,
    spoilage_sigma_bits: Option<u64>,
    resource_preference_bits: u64,
    resource_prior_bits: u64,
    social_enabled: bool,
    transfer_quantum_bits: u64,
}

impl OrganismConfigSnapshotV1 {
    pub(crate) fn from_config(config: OrganismConfig) -> Self {
        Self {
            set_point_bits: config.set_point.to_bits(),
            metabolic_cost_bits: config.metabolic_cost.to_bits(),
            forage_activity_cost_bits: config.forage_activity_cost.to_bits(),
            forage_efficiency_bits: config.forage_efficiency.to_bits(),
            goal_precision_bits: config.goal_precision.to_bits(),
            effective_temperature_bits: config.effective_temperature.to_bits(),
            dissipation_rate_bits: config.dissipation_rate.to_bits(),
            death_energy_threshold_bits: config.death_energy_threshold.to_bits(),
            action_temperature_bits: config.action_temperature.to_bits(),
            perceptual_grain_bits: config.perceptual_grain.map(f64::to_bits),
            spoilage_sigma_bits: config.spoilage_sigma.map(f64::to_bits),
            resource_preference_bits: config.resource_preference.to_bits(),
            resource_prior_bits: config.resource_prior.to_bits(),
            social_enabled: config.social_enabled,
            transfer_quantum_bits: config.transfer_quantum.to_bits(),
        }
    }

    fn validate(self) -> Result<OrganismConfig, OrganismSnapshotErrorV1> {
        let config = OrganismConfig {
            set_point: finite_from_bits("config.set_point", self.set_point_bits)?,
            metabolic_cost: finite_from_bits("config.metabolic_cost", self.metabolic_cost_bits)?,
            forage_activity_cost: finite_from_bits(
                "config.forage_activity_cost",
                self.forage_activity_cost_bits,
            )?,
            forage_efficiency: finite_from_bits(
                "config.forage_efficiency",
                self.forage_efficiency_bits,
            )?,
            goal_precision: finite_from_bits("config.goal_precision", self.goal_precision_bits)?,
            effective_temperature: finite_from_bits(
                "config.effective_temperature",
                self.effective_temperature_bits,
            )?,
            dissipation_rate: finite_from_bits(
                "config.dissipation_rate",
                self.dissipation_rate_bits,
            )?,
            death_energy_threshold: finite_from_bits(
                "config.death_energy_threshold",
                self.death_energy_threshold_bits,
            )?,
            action_temperature: finite_from_bits(
                "config.action_temperature",
                self.action_temperature_bits,
            )?,
            perceptual_grain: optional_finite_from_bits(
                "config.perceptual_grain",
                self.perceptual_grain_bits,
            )?,
            spoilage_sigma: optional_finite_from_bits(
                "config.spoilage_sigma",
                self.spoilage_sigma_bits,
            )?,
            resource_preference: finite_from_bits(
                "config.resource_preference",
                self.resource_preference_bits,
            )?,
            resource_prior: finite_from_bits("config.resource_prior", self.resource_prior_bits)?,
            social_enabled: self.social_enabled,
            transfer_quantum: finite_from_bits(
                "config.transfer_quantum",
                self.transfer_quantum_bits,
            )?,
        };

        // The embedded FEP contract requires a positive action temperature. Other ALife numeric
        // fields are intentionally not reinterpreted here: several experimental paths clamp or
        // otherwise define behavior for unusual finite values, and persistence must preserve those
        // semantics instead of inventing a second configuration policy.
        if config.action_temperature <= 0.0 {
            return Err(OrganismSnapshotErrorV1::NonPositive {
                field: "config.action_temperature",
            });
        }
        Ok(config)
    }
}

/// Deterministic persistence representation of one raw factual ledger entry.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct OrganismLedgerEntrySnapshotV1 {
    partner_id: AgentId,
    record: InteractionRecord,
}

impl OrganismLedgerEntrySnapshotV1 {
    pub(crate) fn new(partner_id: AgentId, record: InteractionRecord) -> Self {
        Self { partner_id, record }
    }

    pub fn partner_id(&self) -> AgentId {
        self.partner_id
    }

    pub fn record(&self) -> InteractionRecord {
        self.record
    }
}

/// Raw serializable persistence capsule for one canonical population-owned organism.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrganismSnapshotV1 {
    id: AgentId,
    agent: ActiveInferenceAgentSnapshotV1,
    boundary: MarkovBoundarySnapshotV1,
    energy_bits: u64,
    config: OrganismConfigSnapshotV1,
    last_resource_observed_bits: u64,
    /// Strictly sorted by `partner_id.raw()` so equal factual state has one canonical ordering.
    ledger: Vec<OrganismLedgerEntrySnapshotV1>,
    lineage_id: AgentId,
    generation: u32,
}

/// Non-serializable authority capability produced only after nested and cross-domain validation.
#[derive(Debug, Clone)]
pub struct ValidatedOrganismSnapshotV1 {
    id: AgentId,
    agent: ValidatedActiveInferenceAgentSnapshotV1,
    boundary: ValidatedMarkovBoundarySnapshotV1,
    energy: f64,
    config: OrganismConfig,
    last_resource_observed: f64,
    ledger: Vec<OrganismLedgerEntrySnapshotV1>,
    lineage_id: AgentId,
    generation: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OrganismSnapshotErrorV1 {
    NestedAgent(ActiveInferenceAgentSnapshotErrorV1),
    NestedBoundary(MarkovBoundarySnapshotErrorV1),
    ReservedIdentity { field: &'static str },
    FounderLineageMismatch,
    DescendantLineageSelfReference,
    NonFinite { field: &'static str },
    NonPositive { field: &'static str },
    AgentShapeMismatch {
        field: &'static str,
        expected: usize,
        observed: usize,
    },
    ActionTemperatureMismatch,
    BoundaryShapeMismatch {
        field: &'static str,
        expected: usize,
        observed: usize,
    },
    LedgerNotStrictlySorted,
    LedgerSelfPartner,
    LedgerReservedPartner,
    LedgerZeroEncounterCount,
    LedgerNegativeTransfer { field: &'static str },
}

impl From<ActiveInferenceAgentSnapshotErrorV1> for OrganismSnapshotErrorV1 {
    fn from(value: ActiveInferenceAgentSnapshotErrorV1) -> Self {
        Self::NestedAgent(value)
    }
}

impl From<MarkovBoundarySnapshotErrorV1> for OrganismSnapshotErrorV1 {
    fn from(value: MarkovBoundarySnapshotErrorV1) -> Self {
        Self::NestedBoundary(value)
    }
}

impl OrganismSnapshotV1 {
    /// Consume raw persistence and create organism restore authority only if nested snapshot
    /// validation and all canonical population-organism relationships succeed.
    pub fn validate(self) -> Result<ValidatedOrganismSnapshotV1, OrganismSnapshotErrorV1> {
        validate_population_identity(self.id, self.lineage_id, self.generation)?;
        let energy = finite_from_bits("energy", self.energy_bits)?;
        let last_resource_observed = finite_from_bits(
            "last_resource_observed",
            self.last_resource_observed_bits,
        )?;
        let config = self.config.validate()?;
        validate_ledger(self.id, &self.ledger)?;

        let agent = self.agent.validate()?;
        let boundary = self.boundary.validate()?;
        let agent_shape = active_inference_snapshot_shape_v1(&agent);
        let boundary_shape = markov_boundary_snapshot_shape_v1(&boundary);
        validate_cross_domain_shapes(config, agent_shape, boundary_shape)?;

        Ok(ValidatedOrganismSnapshotV1 {
            id: self.id,
            agent,
            boundary,
            energy,
            config,
            last_resource_observed,
            ledger: self.ledger,
            lineage_id: self.lineage_id,
            generation: self.generation,
        })
    }
}

impl ValidatedOrganismSnapshotV1 {
    pub fn id(&self) -> AgentId {
        self.id
    }

    pub fn lineage_id(&self) -> AgentId {
        self.lineage_id
    }

    pub fn generation(&self) -> u32 {
        self.generation
    }

    pub fn energy(&self) -> f64 {
        self.energy
    }

    pub fn config(&self) -> OrganismConfig {
        self.config
    }

    pub fn last_resource_observed(&self) -> f64 {
        self.last_resource_observed
    }

    pub fn agent(&self) -> &ValidatedActiveInferenceAgentSnapshotV1 {
        &self.agent
    }

    pub fn boundary(&self) -> &ValidatedMarkovBoundarySnapshotV1 {
        &self.boundary
    }

    pub fn ledger(&self) -> &[OrganismLedgerEntrySnapshotV1] {
        &self.ledger
    }
}

fn validate_population_identity(
    id: AgentId,
    lineage_id: AgentId,
    generation: u32,
) -> Result<(), OrganismSnapshotErrorV1> {
    if id == AgentId::UNALLOCATED {
        return Err(OrganismSnapshotErrorV1::ReservedIdentity { field: "id" });
    }
    if lineage_id == AgentId::UNALLOCATED {
        return Err(OrganismSnapshotErrorV1::ReservedIdentity {
            field: "lineage_id",
        });
    }
    if generation == 0 && lineage_id != id {
        return Err(OrganismSnapshotErrorV1::FounderLineageMismatch);
    }
    if generation > 0 && lineage_id == id {
        return Err(OrganismSnapshotErrorV1::DescendantLineageSelfReference);
    }
    Ok(())
}

fn validate_cross_domain_shapes(
    config: OrganismConfig,
    agent: ActiveInferenceAgentSnapshotShapeV1,
    boundary: MarkovBoundarySnapshotShapeV1,
) -> Result<(), OrganismSnapshotErrorV1> {
    let expected_dim = if config.social_enabled {
        SOCIAL_DIM_V1
    } else {
        ASOCIAL_DIM_V1
    };
    let expected_actions = if config.social_enabled {
        SOCIAL_ACTIONS_V1
    } else {
        ASOCIAL_ACTIONS_V1
    };

    require_agent_shape("agent.state_dim", expected_dim, agent.state_dim)?;
    require_agent_shape("agent.obs_dim", expected_dim, agent.obs_dim)?;
    require_agent_shape("agent.num_actions", expected_actions, agent.num_actions)?;
    if agent.action_temperature.to_bits() != config.action_temperature.to_bits() {
        return Err(OrganismSnapshotErrorV1::ActionTemperatureMismatch);
    }

    require_boundary_shape("boundary.internal_dim", expected_dim, boundary.internal_dim)?;
    require_boundary_shape(
        "boundary.sensory_dim",
        BOUNDARY_SENSORY_DIM_V1,
        boundary.sensory_dim,
    )?;
    require_boundary_shape(
        "boundary.active_dim",
        BOUNDARY_ACTIVE_DIM_V1,
        boundary.active_dim,
    )?;
    Ok(())
}

fn require_agent_shape(
    field: &'static str,
    expected: usize,
    observed: usize,
) -> Result<(), OrganismSnapshotErrorV1> {
    if observed != expected {
        return Err(OrganismSnapshotErrorV1::AgentShapeMismatch {
            field,
            expected,
            observed,
        });
    }
    Ok(())
}

fn require_boundary_shape(
    field: &'static str,
    expected: usize,
    observed: usize,
) -> Result<(), OrganismSnapshotErrorV1> {
    if observed != expected {
        return Err(OrganismSnapshotErrorV1::BoundaryShapeMismatch {
            field,
            expected,
            observed,
        });
    }
    Ok(())
}

fn validate_ledger(
    self_id: AgentId,
    ledger: &[OrganismLedgerEntrySnapshotV1],
) -> Result<(), OrganismSnapshotErrorV1> {
    let mut previous = None;
    for entry in ledger {
        if entry.partner_id == AgentId::UNALLOCATED {
            return Err(OrganismSnapshotErrorV1::LedgerReservedPartner);
        }
        if entry.partner_id == self_id {
            return Err(OrganismSnapshotErrorV1::LedgerSelfPartner);
        }

        let raw = entry.partner_id.raw();
        if previous.is_some_and(|prior| raw <= prior) {
            return Err(OrganismSnapshotErrorV1::LedgerNotStrictlySorted);
        }
        previous = Some(raw);

        if entry.record.encounter_count == 0 {
            return Err(OrganismSnapshotErrorV1::LedgerZeroEncounterCount);
        }
        validate_nonnegative_transfer(
            "ledger.given_to_partner",
            entry.record.given_to_partner,
        )?;
        validate_nonnegative_transfer(
            "ledger.received_from_partner",
            entry.record.received_from_partner,
        )?;
    }
    Ok(())
}

fn validate_nonnegative_transfer(
    field: &'static str,
    value: f64,
) -> Result<(), OrganismSnapshotErrorV1> {
    if !value.is_finite() {
        return Err(OrganismSnapshotErrorV1::NonFinite { field });
    }
    if value < 0.0 {
        return Err(OrganismSnapshotErrorV1::LedgerNegativeTransfer { field });
    }
    Ok(())
}

fn finite_from_bits(field: &'static str, bits: u64) -> Result<f64, OrganismSnapshotErrorV1> {
    let value = f64::from_bits(bits);
    if !value.is_finite() {
        return Err(OrganismSnapshotErrorV1::NonFinite { field });
    }
    Ok(value)
}

fn optional_finite_from_bits(
    field: &'static str,
    bits: Option<u64>,
) -> Result<Option<f64>, OrganismSnapshotErrorV1> {
    bits.map(|value| finite_from_bits(field, value)).transpose()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn agent_shape(
        dim: usize,
        actions: usize,
        temperature: f64,
    ) -> ActiveInferenceAgentSnapshotShapeV1 {
        ActiveInferenceAgentSnapshotShapeV1 {
            state_dim: dim,
            obs_dim: dim,
            num_actions: actions,
            action_temperature: temperature,
            td_learning_enabled: true,
            timestamp: 0,
            perception_cycles: 0,
        }
    }

    fn boundary_shape(dim: usize) -> MarkovBoundarySnapshotShapeV1 {
        MarkovBoundarySnapshotShapeV1 {
            internal_dim: dim,
            sensory_dim: BOUNDARY_SENSORY_DIM_V1,
            active_dim: BOUNDARY_ACTIVE_DIM_V1,
            alpha: 0.1,
            history_count: 0,
        }
    }

    fn allocated_id(raw: u64) -> AgentId {
        let mut ids = crate::AgentIdAllocator::new();
        for _ in 0..raw {
            ids.allocate();
        }
        ids.allocate()
    }

    #[test]
    fn config_snapshot_round_trip_preserves_every_float_bit() {
        let config = OrganismConfig {
            set_point: -0.0,
            metabolic_cost: 0.0,
            forage_activity_cost: 0.03125,
            forage_efficiency: 0.6,
            goal_precision: 1.75,
            effective_temperature: 2.0,
            dissipation_rate: 0.0007,
            death_energy_threshold: 0.02,
            action_temperature: 0.7,
            perceptual_grain: Some(2.0),
            spoilage_sigma: Some(-0.25),
            resource_preference: 0.9,
            resource_prior: 0.1,
            social_enabled: true,
            transfer_quantum: 0.03,
        };
        let restored = OrganismConfigSnapshotV1::from_config(config)
            .validate()
            .expect("valid bit-exact config");

        assert_eq!(restored.set_point.to_bits(), config.set_point.to_bits());
        assert_eq!(restored.metabolic_cost.to_bits(), config.metabolic_cost.to_bits());
        assert_eq!(
            restored.forage_activity_cost.to_bits(),
            config.forage_activity_cost.to_bits()
        );
        assert_eq!(restored.forage_efficiency.to_bits(), config.forage_efficiency.to_bits());
        assert_eq!(restored.goal_precision.to_bits(), config.goal_precision.to_bits());
        assert_eq!(
            restored.effective_temperature.to_bits(),
            config.effective_temperature.to_bits()
        );
        assert_eq!(restored.dissipation_rate.to_bits(), config.dissipation_rate.to_bits());
        assert_eq!(
            restored.death_energy_threshold.to_bits(),
            config.death_energy_threshold.to_bits()
        );
        assert_eq!(restored.action_temperature.to_bits(), config.action_temperature.to_bits());
        assert_eq!(
            restored.perceptual_grain.map(f64::to_bits),
            config.perceptual_grain.map(f64::to_bits)
        );
        assert_eq!(
            restored.spoilage_sigma.map(f64::to_bits),
            config.spoilage_sigma.map(f64::to_bits)
        );
        assert_eq!(restored.resource_preference.to_bits(), config.resource_preference.to_bits());
        assert_eq!(restored.resource_prior.to_bits(), config.resource_prior.to_bits());
        assert_eq!(restored.social_enabled, config.social_enabled);
        assert_eq!(restored.transfer_quantum.to_bits(), config.transfer_quantum.to_bits());
    }

    #[test]
    fn asocial_shape_matches_current_organism_constructor_contract() {
        let config = OrganismConfig::default();
        validate_cross_domain_shapes(
            config,
            agent_shape(ASOCIAL_DIM_V1, ASOCIAL_ACTIONS_V1, config.action_temperature),
            boundary_shape(ASOCIAL_DIM_V1),
        )
        .expect("asocial shape");
    }

    #[test]
    fn social_shape_matches_current_organism_constructor_contract() {
        let config = OrganismConfig {
            social_enabled: true,
            action_temperature: 0.4,
            ..OrganismConfig::default()
        };
        validate_cross_domain_shapes(
            config,
            agent_shape(SOCIAL_DIM_V1, SOCIAL_ACTIONS_V1, config.action_temperature),
            boundary_shape(SOCIAL_DIM_V1),
        )
        .expect("social shape");
    }

    #[test]
    fn social_config_rejects_asocial_agent_shape() {
        let config = OrganismConfig {
            social_enabled: true,
            ..OrganismConfig::default()
        };
        assert_eq!(
            validate_cross_domain_shapes(
                config,
                agent_shape(ASOCIAL_DIM_V1, ASOCIAL_ACTIONS_V1, config.action_temperature),
                boundary_shape(SOCIAL_DIM_V1),
            ),
            Err(OrganismSnapshotErrorV1::AgentShapeMismatch {
                field: "agent.state_dim",
                expected: SOCIAL_DIM_V1,
                observed: ASOCIAL_DIM_V1,
            })
        );
    }

    #[test]
    fn action_temperature_must_match_embedded_agent_exactly() {
        let config = OrganismConfig::default();
        assert_eq!(
            validate_cross_domain_shapes(
                config,
                agent_shape(ASOCIAL_DIM_V1, ASOCIAL_ACTIONS_V1, 0.9),
                boundary_shape(ASOCIAL_DIM_V1),
            ),
            Err(OrganismSnapshotErrorV1::ActionTemperatureMismatch)
        );
    }

    #[test]
    fn boundary_shape_must_match_organism_observation_geometry() {
        let config = OrganismConfig::default();
        assert_eq!(
            validate_cross_domain_shapes(
                config,
                agent_shape(ASOCIAL_DIM_V1, ASOCIAL_ACTIONS_V1, config.action_temperature),
                boundary_shape(SOCIAL_DIM_V1),
            ),
            Err(OrganismSnapshotErrorV1::BoundaryShapeMismatch {
                field: "boundary.internal_dim",
                expected: ASOCIAL_DIM_V1,
                observed: SOCIAL_DIM_V1,
            })
        );
    }

    #[test]
    fn population_identity_profile_distinguishes_founders_and_descendants() {
        let founder = allocated_id(3);
        assert_eq!(validate_population_identity(founder, founder, 0), Ok(()));

        let child = allocated_id(9);
        assert_eq!(validate_population_identity(child, founder, 1), Ok(()));
        assert_eq!(
            validate_population_identity(child, child, 1),
            Err(OrganismSnapshotErrorV1::DescendantLineageSelfReference)
        );
    }

    #[test]
    fn asocial_population_organism_may_legitimately_retain_encounter_history() {
        let self_id = allocated_id(1);
        let entries = [OrganismLedgerEntrySnapshotV1::new(
            allocated_id(2),
            InteractionRecord {
                given_to_partner: 0.0,
                received_from_partner: 0.0,
                encounter_count: 1,
            },
        )];

        // `Population::step_social` records scheduled encounters independently of whether the
        // organism's action space includes Transfer. The ledger therefore cannot be inferred from
        // `social_enabled`.
        validate_ledger(self_id, &entries).expect("asocial encounter history is valid factual state");
    }

    #[test]
    fn social_ledger_must_be_sorted_and_factual() {
        let self_id = allocated_id(1);
        let entries = vec![
            OrganismLedgerEntrySnapshotV1::new(
                allocated_id(2),
                InteractionRecord {
                    given_to_partner: 0.1,
                    received_from_partner: 0.2,
                    encounter_count: 1,
                },
            ),
            OrganismLedgerEntrySnapshotV1::new(
                allocated_id(4),
                InteractionRecord {
                    given_to_partner: 0.0,
                    received_from_partner: 0.3,
                    encounter_count: 5,
                },
            ),
        ];
        validate_ledger(self_id, &entries).expect("valid deterministic ledger");

        let reversed = [entries[1], entries[0]];
        assert_eq!(
            validate_ledger(self_id, &reversed),
            Err(OrganismSnapshotErrorV1::LedgerNotStrictlySorted)
        );
    }

    #[test]
    fn zero_encounter_ledger_entry_is_not_population_reachable() {
        let self_id = allocated_id(1);
        let entries = [OrganismLedgerEntrySnapshotV1::new(
            allocated_id(2),
            InteractionRecord::default(),
        )];
        assert_eq!(
            validate_ledger(self_id, &entries),
            Err(OrganismSnapshotErrorV1::LedgerZeroEncounterCount)
        );
    }
}

mod live_snapshot;
