//! Standalone contract harness for the production scientific-decision modules.
//!
//! The modules under test are included directly from production examples/support paths.
//! This harness supplies only the small domain shapes required by the custody projection,
//! because Cargo cannot enumerate the parent workspace: root Cargo.toml includes crates/core/*,
//! while the checked-in tree has no crates/core/symthaea/Cargo.toml.
//!
//! The shim is deliberately shape-matched to the serialized fields of the real observation
//! adapter. It verifies replay/custody logic and schema behavior, but does NOT prove that the
//! production symthaea-futures-symtropy package integration compiles. That remains a separate
//! gate after the workspace member issue is repaired or a package build can bypass it.

extern crate self as symthaea_futures_symtropy;

pub mod evolutionary_rescue {
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
    pub struct EvolutionaryRescueSample {
        pub sampled_alive_count: usize,
        pub observed_mean_forage_efficiency: Option<f64>,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
    pub struct EvolutionaryRescueObservation {
        pub tick: u64,
        pub sample: Option<EvolutionaryRescueSample>,
    }
}

pub mod common {
    use crate::evolutionary_rescue::EvolutionaryRescueObservation;
    use serde::{Deserialize, Serialize};

    pub const WORLD_FORMAT_VERSION: &str = "evolutionary_rescue_world_v1";

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct SerializedWorld {
        pub format_version: String,
        pub seed: u64,
        pub census_obs: Vec<EvolutionaryRescueObservation>,
        pub noisy_obs: Vec<EvolutionaryRescueObservation>,
        pub shuffled_obs: Vec<EvolutionaryRescueObservation>,
        pub privileged_obs: Vec<EvolutionaryRescueObservation>,
        pub trajectory: Vec<bool>,
        pub max_population: usize,
        pub first_extinction_tick: Option<u64>,
        pub trait_level_just_before_extinction: Option<f64>,
    }
}

#[path = "../../../crates/domains/symthaea-futures-ensemble/examples/support/scientific_decision_replay.rs"]
pub mod scientific_decision_replay;

#[path = "../../../crates/domains/symthaea-futures-ensemble/examples/support/evolutionary_rescue_custody.rs"]
pub mod evolutionary_rescue_custody;
