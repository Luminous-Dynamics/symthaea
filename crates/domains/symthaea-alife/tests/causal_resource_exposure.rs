// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_alife::{
    BirthMutationInterventionV1, CausalRescueOutcomeProtocolV1, CausalRescueReplaySessionV1,
    CounterfactualRandomFieldV1, EarthForcedEnvironment, EncounterScheduler,
    EvolutionBirthPlanV1, GenesisEarthExecutionV1, GenesisResourceExposureErrorV1,
    GenesisResourceExposureRunnerV1, GenesisResourceExposureV1, InheritanceMode,
    ObservedStochasticTapeSubjectV1, ObservedStochasticTapeV1, OrganismConfig, PairingMode,
    PerturbationSchedule, PopulationConfig, ResourcePerturbation,
    ResourcePerturbationSnapshotV1, ValidatedCausalRescueReplaySessionV1,
    validate_resource_exposure_for_session_v1,
};

const POPULATION_SEED: u64 = 0xE770_1001;
const SCHEDULER_SEED: u64 = 0xE770_1002;
const FALLBACK_SEED: u64 = 0xE770_2001;

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

fn shock() -> ResourcePerturbation {
    ResourcePerturbation::new(1, 1, 0.5, 0.0).expect("valid focal shock")
}

fn validated_session() -> ValidatedCausalRescueReplaySessionV1 {
    let mut execution = GenesisEarthExecutionV1::new(
        reproductive_cfg(),
        2,
        POPULATION_SEED,
        PairingMode::Random,
        SCHEDULER_SEED,
        EarthForcedEnvironment::earth_like(79.0),
    )
    .expect("fresh natural execution");

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
    let parent_index = fork_ids
        .iter()
        .position(|&id| id == parent_id)
        .expect("parent in fork population");
    let source_index = fork_ids
        .iter()
        .position(|&id| id == source_id)
        .expect("source in fork population");
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
        .expect("mutation-enabled birth changes at least one trait");

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

    CausalRescueReplaySessionV1::new(
        0,
        fork.persisted().clone(),
        natural_end.persisted().clone(),
        tape,
        CounterfactualRandomFieldV1::new(FALLBACK_SEED),
        intervention,
        vec![ResourcePerturbationSnapshotV1::from_perturbation(shock())],
        CausalRescueOutcomeProtocolV1::new(0, 1, 0.9, 2),
    )
    .validate()
    .expect("validated causal replay session")
}

fn recorded_treatment_exposures() -> Vec<GenesisResourceExposureV1> {
    let mut runner = GenesisResourceExposureRunnerV1::new(
        reproductive_cfg(),
        2,
        POPULATION_SEED,
    );
    let mut scheduler = EncounterScheduler::new(PairingMode::Random, SCHEDULER_SEED);
    let mut environment = EarthForcedEnvironment::earth_like(79.0);
    let schedule = PerturbationSchedule::new(vec![shock()]);

    for tick in 0..3u64 {
        runner
            .step_social(
                |_| {
                    let baseline = environment.step();
                    schedule
                        .apply_resource(tick, baseline)
                        .expect("valid treatment resource")
                },
                &mut scheduler,
            )
            .expect("qualified exposure step");
    }
    runner.completed_exposures().to_vec()
}

#[test]
fn exact_consumed_treatment_sequence_validates_against_session() {
    let session = validated_session();
    let exposures = recorded_treatment_exposures();
    let validated = validate_resource_exposure_for_session_v1(&session, &exposures)
        .expect("exact Earth+shock exposure sequence must validate");
    assert_eq!(validated.start_tick(), 0);
    assert_eq!(validated.next_tick(), 3);
    assert_eq!(validated.exposures(), exposures.as_slice());
}

#[test]
fn one_bit_resource_drift_is_rejected() {
    let session = validated_session();
    let exposures = recorded_treatment_exposures();
    let mut value = serde_json::to_value(&exposures).expect("serialize exposure rows");
    let bits = value[1]["resource_bits"]
        .as_u64()
        .expect("resource bits");
    value[1]["resource_bits"] = serde_json::json!(bits + 1);
    let drifted: Vec<GenesisResourceExposureV1> =
        serde_json::from_value(value).expect("deserialize one-bit drift");
    assert!(matches!(
        validate_resource_exposure_for_session_v1(&session, &drifted),
        Err(GenesisResourceExposureErrorV1::ResourceMismatch { tick: 1, .. })
    ));
}

#[test]
fn exposure_population_count_must_match_natural_behavior_batch() {
    let session = validated_session();
    let exposures = recorded_treatment_exposures();
    let mut value = serde_json::to_value(&exposures).expect("serialize exposure rows");
    let population = value[2]["population_before"]
        .as_u64()
        .expect("population before");
    value[2]["population_before"] = serde_json::json!(population + 1);
    let drifted: Vec<GenesisResourceExposureV1> =
        serde_json::from_value(value).expect("deserialize population drift");
    assert!(matches!(
        validate_resource_exposure_for_session_v1(&session, &drifted),
        Err(GenesisResourceExposureErrorV1::BehaviorPopulationMismatch { tick: 2, .. })
    ));
}
