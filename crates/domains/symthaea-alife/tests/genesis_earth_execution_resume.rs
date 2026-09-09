// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! End-to-end exact-resume regression for evolutionary Genesis execution.
//!
//! The unit tests in `genesis_execution` prove the capsule composition for Random and FixedPartners
//! scheduling. This external test deliberately activates every population-owned continuation cursor
//! that an exact evolutionary replay depends on: RandomPeer source selection, mutation RNG,
//! construction-seed allocation, identity allocation, birth lifecycle evidence, and a cull-created
//! stale fixed-partner history.

use symthaea_alife::{
    EarthForcedEnvironment, GenesisEarthExecutionCapsuleV1, GenesisEarthExecutionV1,
    InheritanceMode, OrganismConfig, PairingMode, PopulationConfig,
};

fn evolutionary_cfg() -> PopulationConfig {
    PopulationConfig {
        death_energy_threshold: -1.0,
        // Every living organism reproduces at each completed tick. The test intentionally runs only
        // four social ticks total, keeping the deterministic doubling bounded while guaranteeing
        // that evolutionary RNG/identity/seed state is consumed on both sides of the checkpoint.
        reproduction_energy_threshold: 0.0,
        reproduction_energy_cost: 0.4,
        organism_cfg: OrganismConfig {
            social_enabled: true,
            transfer_quantum: 0.025,
            action_temperature: 0.63,
            perceptual_grain: Some(0.125),
            ..OrganismConfig::default()
        },
        mutation_rate: 0.61,
        mutation_std: 0.0375,
        inheritance: InheritanceMode::RandomPeer,
    }
}

fn new_execution() -> GenesisEarthExecutionV1 {
    GenesisEarthExecutionV1::new(
        evolutionary_cfg(),
        4,
        0xA11F_E701,
        PairingMode::FixedPartners,
        0x5CED_E702,
        EarthForcedEnvironment::earth_like(103.0).with_secular_drift(-0.0125),
    )
    .expect("fresh evolutionary execution")
}

#[test]
fn evolutionary_random_peer_world_is_bit_exact_across_cull_boundary_resume() {
    let mut uninterrupted = new_execution();
    let mut split = new_execution();

    // 4 founders -> 8 -> 16. Every birth consumes source RNG, mutation RNG, seed, ID, and lifecycle
    // sequence authority. Cull one organism at the stable boundary before the next scheduler call,
    // intentionally leaving a legitimate stale FixedPartners mapping in scheduler state.
    for _ in 0..2 {
        uninterrupted.step_social().expect("baseline prefix tick");
        split.step_social().expect("split prefix tick");
    }
    assert_eq!(uninterrupted.population().len(), 16);
    assert_eq!(split.population().len(), 16);
    assert_eq!(uninterrupted.cull_weakest(1).expect("baseline cull"), 1);
    assert_eq!(split.cull_weakest(1).expect("split cull"), 1);
    assert_eq!(uninterrupted.population().len(), 15);

    let baseline_prefix = uninterrupted
        .checkpoint_execution()
        .expect("baseline prefix checkpoint");
    let split_prefix = split
        .checkpoint_execution()
        .expect("split prefix checkpoint");
    assert_eq!(
        serde_json::to_string(baseline_prefix.persisted()).unwrap(),
        serde_json::to_string(split_prefix.persisted()).unwrap(),
        "identical evolutionary prefixes must produce identical complete capsules"
    );

    let encoded = serde_json::to_string(split_prefix.persisted()).expect("serialize capsule");
    let raw: GenesisEarthExecutionCapsuleV1 =
        serde_json::from_str(&encoded).expect("deserialize capsule");
    let validated = raw.validate_after(0).expect("revalidate capsule");
    let mut restored =
        GenesisEarthExecutionV1::from_validated_capsule(&validated).expect("restore execution");
    assert_eq!(restored.population().len(), 15);

    // 15 -> 30 -> 60. The first resumed fixed-partner scheduling call must rematch around the stale
    // dead-agent relationship while the population simultaneously consumes restored evolutionary
    // source/mutation RNG and allocator cursors.
    for _ in 0..2 {
        uninterrupted.step_social().expect("baseline suffix tick");
        restored.step_social().expect("restored suffix tick");
    }
    assert_eq!(uninterrupted.population().len(), 60);
    assert_eq!(restored.population().len(), 60);

    let baseline_final = uninterrupted
        .checkpoint_execution()
        .expect("baseline final checkpoint");
    let restored_final = restored
        .checkpoint_execution()
        .expect("restored final checkpoint");

    assert_eq!(
        serde_json::to_string(baseline_final.persisted()).unwrap(),
        serde_json::to_string(restored_final.persisted()).unwrap(),
        "evolutionary world diverged after serialized restore"
    );
}
