// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact sham-fidelity validation for causal rescue replays.
//!
//! A mutation-reversion contrast is not interpretable unless the sham branch first reproduces the
//! observed natural endpoint exactly. This module therefore fingerprints the complete validated
//! Earth-forced Genesis endpoint — Population, scheduler, Earth forcing, and behavior+lifecycle
//! evidence — and returns a non-serializable fidelity capability only when the sham endpoint has
//! the same semantic fingerprint as the observed natural endpoint bound into the validated
//! experiment specification.
//!
//! The fingerprint is a SHA-256 content identity over versioned, length-delimited JSON encodings of
//! canonical snapshots regenerated from already-validated capabilities. It is **not** an origin
//! signature and does not authenticate who produced the bytes. The distinction is deliberate:
//! semantic endpoint equality and provenance authenticity are separate evidence claims.

use serde::{Serialize, Deserialize};
use sha2::{Digest, Sha256};

use crate::{
    CausalRescueEffectV1, CausalRescueEstimandErrorV1, EncounterScheduler,
    GenesisEarthExecutionCapsuleErrorV1, GenesisEarthExecutionCapsuleV1,
    GenesisEvidenceCheckpointV1, GenesisExecutionProtocolV1, Population,
    ValidatedCausalRescueExperimentSpecV1, ValidatedGenesisEarthExecutionCapsuleV1,
};
use crate::earth_forcing::EarthForcedEnvironment;
use crate::population::PopulationLiveSnapshotErrorV1;

const ENDPOINT_FINGERPRINT_SCHEMA_V1: &[u8] = b"genesis-earth-endpoint-fingerprint-v1";

/// Stable content identity for one fully validated Earth-forced Genesis endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenesisEarthEndpointFingerprintV1 {
    sha256: [u8; 32],
}

impl GenesisEarthEndpointFingerprintV1 {
    pub fn sha256(self) -> [u8; 32] {
        self.sha256
    }
}

/// Non-serializable authority that proves the supplied sham endpoint exactly matched the observed
/// natural endpoint for one already-validated experiment specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CausalRescueShamFidelityReceiptV1 {
    endpoint_fingerprint: GenesisEarthEndpointFingerprintV1,
    endpoint_behavior_next_tick: u64,
    lifecycle_next_sequence: u64,
    natural_offspring_id: crate::AgentId,
}

impl CausalRescueShamFidelityReceiptV1 {
    pub fn endpoint_fingerprint(self) -> GenesisEarthEndpointFingerprintV1 {
        self.endpoint_fingerprint
    }

    pub fn endpoint_behavior_next_tick(self) -> u64 {
        self.endpoint_behavior_next_tick
    }

    pub fn lifecycle_next_sequence(self) -> u64 {
        self.lifecycle_next_sequence
    }

    pub fn natural_offspring_id(self) -> crate::AgentId {
        self.natural_offspring_id
    }
}

/// Effect estimate whose construction required a sham-fidelity capability for the same validated
/// experiment. This remains one matched experimental effect estimate, not a population-level or
/// replicated causal conclusion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FidelityQualifiedCausalRescueEffectV1 {
    effect: CausalRescueEffectV1,
    endpoint_fingerprint: GenesisEarthEndpointFingerprintV1,
    natural_offspring_id: crate::AgentId,
}

impl FidelityQualifiedCausalRescueEffectV1 {
    pub fn effect(self) -> CausalRescueEffectV1 {
        self.effect
    }

    pub fn endpoint_fingerprint(self) -> GenesisEarthEndpointFingerprintV1 {
        self.endpoint_fingerprint
    }

    pub fn natural_offspring_id(self) -> crate::AgentId {
        self.natural_offspring_id
    }
}

#[derive(Debug)]
pub enum CausalRescueShamFidelityErrorV1 {
    ShamCapsule(GenesisEarthExecutionCapsuleErrorV1),
    PopulationLive(PopulationLiveSnapshotErrorV1),
    Serialization {
        component: &'static str,
        error: serde_json::Error,
    },
    ProtocolMismatch {
        expected: GenesisExecutionProtocolV1,
        observed: GenesisExecutionProtocolV1,
    },
    EndpointTickMismatch {
        expected: u64,
        observed: u64,
    },
    EndpointFingerprintMismatch {
        expected: GenesisEarthEndpointFingerprintV1,
        observed: GenesisEarthEndpointFingerprintV1,
    },
    FidelityReceiptSpecMismatch,
    Estimand(CausalRescueEstimandErrorV1),
}

/// Derive a complete semantic endpoint fingerprint from a validated execution capsule.
///
/// Each authority is regenerated through its own validated restore/snapshot path before hashing.
/// That avoids defining a second hand-maintained list of cognitive, scheduler, environmental, or
/// population fields here; additions to those snapshot contracts automatically become part of the
/// endpoint identity.
pub fn genesis_earth_endpoint_fingerprint_v1(
    endpoint: &ValidatedGenesisEarthExecutionCapsuleV1,
) -> Result<GenesisEarthEndpointFingerprintV1, CausalRescueShamFidelityErrorV1> {
    let population = Population::from_validated_snapshot_v1(endpoint.population(), endpoint.evidence())
        .map_err(CausalRescueShamFidelityErrorV1::PopulationLive)?;
    let population_snapshot = population
        .snapshot_v1_at_evidence_boundary(endpoint.evidence())
        .map_err(CausalRescueShamFidelityErrorV1::PopulationLive)?;

    let scheduler_snapshot =
        EncounterScheduler::from_validated_snapshot_v1(endpoint.scheduler()).snapshot_v1();
    let environment_snapshot =
        EarthForcedEnvironment::from_validated_snapshot_v1(endpoint.environment()).snapshot_v1();
    let evidence = endpoint.evidence();
    let evidence_snapshot = GenesisEvidenceCheckpointV1::new(
        evidence.behavior_start_tick(),
        evidence.behavior_next_tick(),
        evidence.behavior_batches().to_vec(),
        evidence.lifecycle().clone().into_checkpoint(),
    );

    let mut hasher = Sha256::new();
    hasher.update((ENDPOINT_FINGERPRINT_SCHEMA_V1.len() as u64).to_le_bytes());
    hasher.update(ENDPOINT_FINGERPRINT_SCHEMA_V1);
    let protocol_tag = match endpoint.protocol() {
        GenesisExecutionProtocolV1::EarthForcedSocialV1 => 1u64,
    };
    hasher.update(protocol_tag.to_le_bytes());
    hash_component(&mut hasher, "population", &population_snapshot)?;
    hash_component(&mut hasher, "scheduler", &scheduler_snapshot)?;
    hash_component(&mut hasher, "environment", &environment_snapshot)?;
    hash_component(&mut hasher, "evidence", &evidence_snapshot)?;

    let sha256: [u8; 32] = hasher.finalize().into();
    Ok(GenesisEarthEndpointFingerprintV1 { sha256 })
}

/// Validate a candidate sham endpoint against the exact observed natural endpoint bound into the
/// experiment specification. A receipt is returned only after full capsule validation and complete
/// semantic fingerprint equality.
pub fn verify_sham_fidelity_v1(
    experiment: &ValidatedCausalRescueExperimentSpecV1,
    sham_endpoint: GenesisEarthExecutionCapsuleV1,
) -> Result<CausalRescueShamFidelityReceiptV1, CausalRescueShamFidelityErrorV1> {
    let session = experiment.session();
    let fork_tick = session.fork().evidence().behavior_next_tick();
    let sham = sham_endpoint
        .validate_after(fork_tick)
        .map_err(CausalRescueShamFidelityErrorV1::ShamCapsule)?;
    let natural = session.natural_end();

    if sham.protocol() != natural.protocol() {
        return Err(CausalRescueShamFidelityErrorV1::ProtocolMismatch {
            expected: natural.protocol(),
            observed: sham.protocol(),
        });
    }
    if sham.evidence().behavior_next_tick() != natural.evidence().behavior_next_tick() {
        return Err(CausalRescueShamFidelityErrorV1::EndpointTickMismatch {
            expected: natural.evidence().behavior_next_tick(),
            observed: sham.evidence().behavior_next_tick(),
        });
    }

    let expected = genesis_earth_endpoint_fingerprint_v1(natural)?;
    let observed = genesis_earth_endpoint_fingerprint_v1(&sham)?;
    if observed != expected {
        return Err(CausalRescueShamFidelityErrorV1::EndpointFingerprintMismatch {
            expected,
            observed,
        });
    }

    Ok(CausalRescueShamFidelityReceiptV1 {
        endpoint_fingerprint: expected,
        endpoint_behavior_next_tick: natural.evidence().behavior_next_tick(),
        lifecycle_next_sequence: natural.evidence().lifecycle().next_sequence(),
        natural_offspring_id: session.natural_offspring_id(),
    })
}

/// Compute the primary mutation-rescue effect only after proving that `fidelity` belongs to this
/// same validated experiment specification.
pub fn compute_fidelity_qualified_primary_effect_v1(
    experiment: &ValidatedCausalRescueExperimentSpecV1,
    fidelity: CausalRescueShamFidelityReceiptV1,
    sham_normalized_deficit_area: f64,
    revert_normalized_deficit_area: f64,
) -> Result<FidelityQualifiedCausalRescueEffectV1, CausalRescueShamFidelityErrorV1> {
    let session = experiment.session();
    let expected_fingerprint = genesis_earth_endpoint_fingerprint_v1(session.natural_end())?;
    if fidelity.endpoint_fingerprint != expected_fingerprint
        || fidelity.endpoint_behavior_next_tick != session.natural_end().evidence().behavior_next_tick()
        || fidelity.lifecycle_next_sequence != session.natural_end().evidence().lifecycle().next_sequence()
        || fidelity.natural_offspring_id != session.natural_offspring_id()
    {
        return Err(CausalRescueShamFidelityErrorV1::FidelityReceiptSpecMismatch);
    }

    let effect = experiment
        .estimand()
        .compute_effect(sham_normalized_deficit_area, revert_normalized_deficit_area)
        .map_err(CausalRescueShamFidelityErrorV1::Estimand)?;
    Ok(FidelityQualifiedCausalRescueEffectV1 {
        effect,
        endpoint_fingerprint: expected_fingerprint,
        natural_offspring_id: session.natural_offspring_id(),
    })
}

fn hash_component<T: Serialize>(
    hasher: &mut Sha256,
    label: &'static str,
    value: &T,
) -> Result<(), CausalRescueShamFidelityErrorV1> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        CausalRescueShamFidelityErrorV1::Serialization {
            component: label,
            error,
        }
    })?;
    hasher.update((label.len() as u64).to_le_bytes());
    hasher.update(label.as_bytes());
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        BirthMutationInterventionV1, CausalRescueExperimentSpecV1,
        CausalRescueOutcomeProtocolV1, CausalRescueReplaySessionV1,
        CounterfactualRandomFieldV1, EarthForcedEnvironment, EvolutionBirthPlanV1,
        InheritanceMode, ObservedStochasticTapeSubjectV1, ObservedStochasticTapeV1,
        OrganismConfig, PairingMode, PopulationConfig, ResourcePerturbation,
        ResourcePerturbationSnapshotV1,
    };

    fn reproductive_cfg() -> PopulationConfig {
        PopulationConfig {
            death_energy_threshold: -1.0,
            reproduction_energy_threshold: 0.0,
            reproduction_energy_cost: 0.4,
            organism_cfg: OrganismConfig {
                social_enabled: true,
                ..OrganismConfig::default()
            },
            mutation_rate: 1.0,
            mutation_std: 0.08,
            inheritance: InheritanceMode::FromParent,
        }
    }

    fn validated_experiment_fixture() -> (
        ValidatedCausalRescueExperimentSpecV1,
        GenesisEarthExecutionCapsuleV1,
    ) {
        let mut execution = crate::GenesisEarthExecutionV1::new(
            reproductive_cfg(),
            2,
            0xF1DE_1001,
            PairingMode::Random,
            0xF1DE_1002,
            EarthForcedEnvironment::earth_like(79.0),
        )
        .expect("fresh execution");
        let fork_ids = execution
            .population()
            .organisms
            .iter()
            .map(|organism| organism.id)
            .collect::<Vec<_>>();
        let fork = execution.checkpoint_execution().expect("fork checkpoint");
        for _ in 0..3 {
            execution.step_social().expect("natural social tick");
        }
        let natural_end = execution
            .checkpoint_execution()
            .expect("natural endpoint checkpoint");

        let birth = natural_end
            .validated()
            .evidence()
            .lifecycle()
            .ledger()
            .records()
            .values()
            .find(|record| record.born_tick == 0 && record.reproductive_parent_id.is_some())
            .copied()
            .expect("tick-zero natural birth");
        let parent_id = birth.reproductive_parent_id.expect("parent");
        let source_id = birth.genome_source_id.expect("source");
        let parent_index = fork_ids.iter().position(|&id| id == parent_id).unwrap();
        let source_index = fork_ids.iter().position(|&id| id == source_id).unwrap();
        let fork_records = fork
            .validated()
            .evidence()
            .lifecycle()
            .ledger()
            .records();
        let plan = EvolutionBirthPlanV1 {
            reproductive_parent_index: parent_index,
            reproductive_parent_id: parent_id,
            reproductive_parent_genome: fork_records[&parent_id].genome.to_genome(),
            genome_source_index: source_index,
            genome_source_id: source_id,
            genome_source_genome: fork_records[&source_id].genome.to_genome(),
            offspring_genome: birth.genome.to_genome(),
        };
        let intervention = BirthMutationInterventionV1::revert_all_changed_for_plan(0, &plan)
            .expect("natural mutation");
        let tape = ObservedStochasticTapeV1 {
            subject: ObservedStochasticTapeSubjectV1 {
                start_behavior_next_tick: 0,
                start_lifecycle_next_sequence: fork
                    .validated()
                    .evidence()
                    .lifecycle()
                    .next_sequence(),
                end_behavior_next_tick: 3,
            },
            scalar_draws: vec![],
            scheduler_orders: vec![],
        };
        let perturbation = ResourcePerturbation::new(1, 1, 0.5, 0.0).unwrap();
        let session = CausalRescueReplaySessionV1::new(
            0,
            fork.persisted().clone(),
            natural_end.persisted().clone(),
            tape,
            CounterfactualRandomFieldV1::new(0xF1DE_2001),
            intervention,
            vec![ResourcePerturbationSnapshotV1::from_perturbation(perturbation)],
            CausalRescueOutcomeProtocolV1::new(0, 1, 0.9, 2),
        );
        let spec = CausalRescueExperimentSpecV1::new(session)
            .validate()
            .expect("validated experiment spec");
        (spec, natural_end.persisted().clone())
    }

    #[test]
    fn exact_natural_endpoint_produces_sham_fidelity_receipt() {
        let (spec, natural_endpoint) = validated_experiment_fixture();
        let expected = genesis_earth_endpoint_fingerprint_v1(spec.session().natural_end())
            .expect("fingerprint natural endpoint");
        let receipt = verify_sham_fidelity_v1(&spec, natural_endpoint)
            .expect("exact endpoint must pass sham fidelity");
        assert_eq!(receipt.endpoint_fingerprint(), expected);
        assert_eq!(
            receipt.endpoint_behavior_next_tick(),
            spec.session().natural_end().evidence().behavior_next_tick()
        );
    }

    #[test]
    fn one_bit_environment_drift_fails_exact_sham_fidelity() {
        let (spec, natural_endpoint) = validated_experiment_fixture();
        let mut value = serde_json::to_value(natural_endpoint).expect("serialize endpoint");
        let bits = value["environment"]["temperature_bits"]
            .as_u64()
            .expect("temperature bits");
        value["environment"]["temperature_bits"] = serde_json::json!(bits + 1);
        let drifted: GenesisEarthExecutionCapsuleV1 =
            serde_json::from_value(value).expect("deserialize drifted endpoint");
        assert!(matches!(
            verify_sham_fidelity_v1(&spec, drifted),
            Err(CausalRescueShamFidelityErrorV1::EndpointFingerprintMismatch { .. })
        ));
    }

    #[test]
    fn primary_effect_requires_fidelity_from_the_same_experiment() {
        let (spec, natural_endpoint) = validated_experiment_fixture();
        let receipt = verify_sham_fidelity_v1(&spec, natural_endpoint).expect("fidelity");
        let qualified = compute_fidelity_qualified_primary_effect_v1(&spec, receipt, 1.25, 2.0)
            .expect("fidelity-qualified effect");
        assert_eq!(qualified.effect().effect().to_bits(), 0.75f64.to_bits());
        assert_eq!(qualified.natural_offspring_id(), spec.session().natural_offspring_id());
    }

    #[test]
    fn endpoint_fingerprint_is_stable_across_repeated_regeneration() {
        let (spec, _natural_endpoint) = validated_experiment_fixture();
        let a = genesis_earth_endpoint_fingerprint_v1(spec.session().natural_end()).unwrap();
        let b = genesis_earth_endpoint_fingerprint_v1(spec.session().natural_end()).unwrap();
        assert_eq!(a, b);
    }
}
