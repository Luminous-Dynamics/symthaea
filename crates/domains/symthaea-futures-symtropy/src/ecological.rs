// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! The Futures Laboratory's first scenario family: extinction forecasting on `symthaea-alife`'s
//! ecological-collapse scenario (`ALIFE_PLAN_2026-07-08.md` Phases 5a/7), per
//! `SYMTHAEA_FUTURES_LABORATORY_PLAN_2026-07-25.md`'s "First experiment" section.
//!
//! [`EcologicalGroundTruth`] wraps the exact scenario `tests/phase5_earth_forcing.rs` in
//! `symthaea-alife` already proves out (a `Population` sharing a resource pool driven by
//! `EarthForcedEnvironment`'s real ice-albedo climate physics) — no new simulated world, no
//! changes to `symthaea-alife` itself.
//!
//! [`ExtinctionObservationPolicy`] is the firewall: it decides, per tick, what a forecaster is
//! allowed to see. Three deliberately simple design choices, each with a reason:
//! - **Fixed cohort by `AgentId`, not a live-recomputed fraction.** Recomputing `sample_size`
//!   from the *current* true population count each tick would leak that count back out
//!   arithmetically (`true_count ≈ sampled_alive_count / fraction`) even though no single field
//!   read looks like a violation. `sample_size` is computed once, from the experiment's known
//!   `initial_population_count` (a setup parameter, not a truth-read), and the cohort is
//!   "organisms whose `AgentId` is less than `sample_size`" — well-defined and permanent because
//!   `AgentIdAllocator` hands out `0, 1, 2, ...` monotonically to the initial population in
//!   construction order and never reuses an id. A tagged individual dying is a real, honest
//!   observation (field-biologist tagging), not a leak.
//! - **Climate-signal visibility is a plain bool**, not graded. `Some(temperature + noise)` or
//!   structurally `None` — the read of `truth.environment.temperature` never happens at all in
//!   the `false` branch. Graded partial visibility is a documented future extension, not built
//!   here.
//! - **Sensor noise draws from its own independent xorshift64 stream** (`noise_rng_state`),
//!   never derived from `Population`'s or `EarthForcedEnvironment`'s own state — the same
//!   xorshift64 step already used by `Population::next_unit` and
//!   `ActiveInferenceAgent::select_action`, kept independent so noise draws can't correlate with
//!   anything hidden.

use symthaea_alife::{EarthForcedEnvironment, Population, StepSummary};

/// Hidden ground truth for one ecological-collapse scenario instance. Lives only here — kept out
/// of every public signature in `symthaea-futures-core`, `-state`, and `-ensemble`. Evaluation
/// code (a future `BaselineRung::OracleUpperBound` generator) is the one deliberate, labeled
/// exception allowed to read the accessors below directly.
///
/// No `Clone`/`Debug`/`PartialEq` derive — `symthaea_alife::Population` itself has none of these
/// (it owns per-organism `ActiveInferenceAgent`s with no such derives either), so this wrapper
/// doesn't fight that.
pub struct EcologicalGroundTruth {
    pub environment: EarthForcedEnvironment,
    pub population: Population,
    /// Shared resource pool total the `[0, 1]` habitability proxy scales before dividing
    /// per-capita — same role as `tests/phase5_earth_forcing.rs`'s `PLANT_RESOURCE_TOTAL`.
    pub plant_resource_total: f64,
    tick: u64,
}

impl EcologicalGroundTruth {
    pub fn new(
        environment: EarthForcedEnvironment,
        population: Population,
        plant_resource_total: f64,
    ) -> Self {
        Self {
            environment,
            population,
            plant_resource_total,
            tick: 0,
        }
    }

    /// Advances one tick, reproducing `tests/phase5_earth_forcing.rs`'s exact coupling:
    /// `population.step(|n| environment.step() * plant_resource_total / n.max(1))`.
    ///
    /// Field-borrow-split deliberately: `let env = &mut self.environment;` taken *before*
    /// calling `self.population.step(closure)` gives the borrow checker two disjoint field
    /// paths through `self` rather than one closure trying to capture all of `self` (which
    /// would collide with `population.step`'s own `&mut self.population` borrow).
    pub fn step(&mut self) -> StepSummary {
        let env = &mut self.environment;
        let plant_resource_total = self.plant_resource_total;
        let summary = self
            .population
            .step(|n| env.step() * plant_resource_total / (n.max(1) as f64));
        self.tick += 1;
        summary
    }

    pub fn tick(&self) -> u64 {
        self.tick
    }

    pub fn is_extinct(&self) -> bool {
        self.population.organisms.is_empty()
    }

    pub fn true_population_count(&self) -> usize {
        self.population.organisms.len()
    }

    pub fn true_mean_energy(&self) -> Option<f64> {
        let n = self.population.organisms.len();
        if n == 0 {
            return None;
        }
        Some(
            self.population
                .organisms
                .iter()
                .map(|o| o.energy)
                .sum::<f64>()
                / n as f64,
        )
    }

    pub fn true_temperature(&self) -> f64 {
        self.environment.temperature
    }
}

/// One tick's worth of sampled cohort data — present only when
/// [`ExtinctionObservationPolicy`] actually takes a reading this tick.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EcologicalSample {
    /// How many of the tracked cohort (see module docs) are alive this tick — NOT the true
    /// total population, which may include untracked individuals.
    pub sampled_alive_count: usize,
    /// Sensor-noised mean energy over just the cohort's living members. `None` if the entire
    /// cohort happens to be dead this tick.
    pub observed_mean_energy: Option<f64>,
    /// `Some(noised temperature)` only when the policy's `reveal_climate_signal` is true;
    /// structurally `None` (never computed from ground truth at all) otherwise.
    pub observed_temperature: Option<f64>,
}

/// The observation the forecaster actually receives, once per tick.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EcologicalObservation {
    pub tick: u64,
    /// `None` on ticks outside the policy's observation frequency — regardless of ground truth.
    pub sample: Option<EcologicalSample>,
}

/// The observation firewall for the ecological-collapse scenario family. See module docs for
/// why each knob is designed the way it is.
pub struct ExtinctionObservationPolicy {
    observation_frequency_ticks: u64,
    sample_size: u64,
    sensor_noise_amplitude: f64,
    reveal_climate_signal: bool,
    noise_rng_state: u64,
}

impl ExtinctionObservationPolicy {
    /// `initial_population_count` must match the count the scenario's `Population::new` was
    /// actually constructed with — it's a legitimate experiment-setup parameter (known to
    /// whoever set up the run), not a read of live ground truth, which is exactly what makes
    /// `sample_size` safe to fix once here rather than recomputing it from the true population
    /// count on every tick.
    pub fn new(
        initial_population_count: usize,
        sample_fraction: f64,
        observation_frequency_ticks: u64,
        sensor_noise_amplitude: f64,
        reveal_climate_signal: bool,
        noise_seed: u64,
    ) -> Self {
        let sample_size =
            (sample_fraction.clamp(0.0, 1.0) * initial_population_count as f64).round() as u64;
        Self {
            observation_frequency_ticks: observation_frequency_ticks.max(1),
            sample_size,
            sensor_noise_amplitude,
            reveal_climate_signal,
            noise_rng_state: if noise_seed == 0 { 1 } else { noise_seed },
        }
    }

    /// xorshift64 step — same formula as `Population::next_unit` and
    /// `ActiveInferenceAgent::select_action`'s RNG, deliberately a separate stream (see module
    /// docs on why this must never share state with the ground truth's own RNGs).
    fn next_unit(&mut self) -> f64 {
        self.noise_rng_state ^= self.noise_rng_state << 13;
        self.noise_rng_state ^= self.noise_rng_state >> 7;
        self.noise_rng_state ^= self.noise_rng_state << 17;
        (self.noise_rng_state as f64) / (u64::MAX as f64)
    }

    fn sample_noise(&mut self) -> f64 {
        (self.next_unit() - 0.5) * 2.0 * self.sensor_noise_amplitude
    }
}

impl crate::ObservationPolicy for ExtinctionObservationPolicy {
    type GroundTruth = EcologicalGroundTruth;
    type Observation = EcologicalObservation;

    fn observe(&mut self, truth: &EcologicalGroundTruth, tick: u64) -> EcologicalObservation {
        if tick % self.observation_frequency_ticks != 0 {
            return EcologicalObservation { tick, sample: None };
        }

        let mut sampled_alive_count = 0usize;
        let mut energy_sum = 0.0;
        for organism in &truth.population.organisms {
            if organism.id.raw() < self.sample_size {
                sampled_alive_count += 1;
                let noise = self.sample_noise();
                energy_sum += organism.energy + noise;
            }
        }
        let observed_mean_energy =
            (sampled_alive_count > 0).then(|| energy_sum / sampled_alive_count as f64);

        let observed_temperature = if self.reveal_climate_signal {
            let noise = self.sample_noise();
            Some(truth.environment.temperature + noise)
        } else {
            None
        };

        EcologicalObservation {
            tick,
            sample: Some(EcologicalSample {
                sampled_alive_count,
                observed_mean_energy,
                observed_temperature,
            }),
        }
    }
}

/// A second observation policy for the *same* ground truth, fixing the cohort/population
/// mismatch documented in `SYMTHAEA_FUTURES_LABORATORY_PLAN_2026-07-25.md`'s rung-5 root-cause
/// diagnostic (2026-07-26): [`ExtinctionObservationPolicy`]'s fixed-original-cohort tracking
/// reports a signal (original cohort survival) that can decline to zero even while the whole
/// population thrives via untracked offspring — every rung built against it inherited that
/// mismatch, most visibly rung 5 (`FepDrivenGenerator`), which confidently predicted extinction
/// that never happened (68/100 predictions in `[0.6, 0.7)`, empirical frequency 0.0 everywhere,
/// ECE 0.6501).
///
/// **The fix is simpler than the original's fixed-cohort machinery, not more complex**: report
/// `min(true population count, sample_size)` — a capped census, not identity-based tracking of
/// specific individuals at all. `sample_size` is a fixed constant set at construction, never
/// derived from live population count, so this can never reveal "population is exactly N" for
/// N > sample_size — only "at least sample_size alive" (the common case, reported count ==
/// sample_size) or the exact remaining count once the population has genuinely dropped below
/// the threshold. That's an intentional, disclosed late-stage signal — a real ecological survey
/// methodology (stop counting once you've confirmed "at least K present"), not a leak: any
/// composition of individuals *above* the threshold, whoever they are, original cohort or
/// offspring, yields the identical output. Reusing [`EcologicalObservation`]/[`EcologicalSample`]
/// unchanged (not a new observation type) means every existing rung built against
/// `ExtinctionObservationPolicy` can be re-tested against this policy with zero code changes —
/// only the *policy* changed, not what a forecaster consumes.
///
/// **Disclosed scope**: no energy/temperature observation at all (unlike
/// `ExtinctionObservationPolicy`) — this policy is deliberately scoped to fixing the
/// population-count signal specifically; a version combining both is future work, not built
/// here.
pub struct PopulationCensusObservationPolicy {
    observation_frequency_ticks: u64,
    sample_size: usize,
}

impl PopulationCensusObservationPolicy {
    pub fn new(sample_size: usize, observation_frequency_ticks: u64) -> Self {
        Self {
            observation_frequency_ticks: observation_frequency_ticks.max(1),
            sample_size,
        }
    }
}

impl crate::ObservationPolicy for PopulationCensusObservationPolicy {
    type GroundTruth = EcologicalGroundTruth;
    type Observation = EcologicalObservation;

    fn observe(&mut self, truth: &EcologicalGroundTruth, tick: u64) -> EcologicalObservation {
        if tick % self.observation_frequency_ticks != 0 {
            return EcologicalObservation { tick, sample: None };
        }

        let sampled_alive_count = truth.population.len().min(self.sample_size);

        EcologicalObservation {
            tick,
            sample: Some(EcologicalSample {
                sampled_alive_count,
                observed_mean_energy: None,
                observed_temperature: None,
            }),
        }
    }
}

#[cfg(test)]
mod leakage_tests {
    use super::*;
    use crate::ObservationPolicy;
    use symthaea_alife::{OrganismConfig, PopulationConfig};

    /// Same fixture values `tests/phase5_earth_forcing.rs` already uses in `symthaea-alife` —
    /// not invented ones — so these leakage tests exercise the same real dynamics the rest of
    /// the suite trusts.
    fn population_config() -> PopulationConfig {
        PopulationConfig {
            death_energy_threshold: 0.05,
            reproduction_energy_threshold: 0.8,
            reproduction_energy_cost: 0.4,
            organism_cfg: OrganismConfig {
                forage_efficiency: 0.6,
                ..OrganismConfig::default()
            },
            ..Default::default()
        }
    }

    fn base_setup(seed: u64) -> (EarthForcedEnvironment, Population) {
        let env = EarthForcedEnvironment::earth_like(200.0);
        let pop = Population::new(population_config(), 5, seed); // ids 0..=4
        (env, pop)
    }

    #[test]
    fn unsampled_organism_energy_does_not_affect_observation() {
        // sample_fraction=0.4 over initial_population_count=5 -> sample_size=2 (ids 0,1 tracked;
        // id 4 is NOT tracked).
        let (env_a, pop_a) = base_setup(42);
        let (env_b, mut pop_b) = base_setup(42);
        pop_b.organisms[4].energy = pop_a.organisms[4].energy + 999.0;

        let truth_a = EcologicalGroundTruth::new(env_a, pop_a, 10.0);
        let truth_b = EcologicalGroundTruth::new(env_b, pop_b, 10.0);

        let mut policy_a = ExtinctionObservationPolicy::new(5, 0.4, 1, 0.05, false, 7);
        let mut policy_b = ExtinctionObservationPolicy::new(5, 0.4, 1, 0.05, false, 7);

        assert_eq!(policy_a.observe(&truth_a, 0), policy_b.observe(&truth_b, 0));
    }

    #[test]
    fn hidden_climate_signal_is_independent_of_true_temperature() {
        let (mut env_a, pop_a) = base_setup(42);
        let (mut env_b, pop_b) = base_setup(42);
        env_a.temperature = 250.0;
        env_b.temperature = 320.0; // wildly different true temperature

        let truth_a = EcologicalGroundTruth::new(env_a, pop_a, 10.0);
        let truth_b = EcologicalGroundTruth::new(env_b, pop_b, 10.0);

        let mut policy_a = ExtinctionObservationPolicy::new(5, 0.4, 1, 0.05, false, 7);
        let mut policy_b = ExtinctionObservationPolicy::new(5, 0.4, 1, 0.05, false, 7);

        let obs_a = policy_a.observe(&truth_a, 0);
        let obs_b = policy_b.observe(&truth_b, 0);
        assert_eq!(obs_a.sample.unwrap().observed_temperature, None);
        assert_eq!(obs_a, obs_b);
    }

    #[test]
    fn tick_outside_observation_frequency_is_none_regardless_of_ground_truth() {
        let (env_a, pop_a) = base_setup(42);
        let (env_b, mut pop_b) = base_setup(42);
        pop_b.organisms.clear(); // fully extinct in b, fully alive in a

        let truth_a = EcologicalGroundTruth::new(env_a, pop_a, 10.0);
        let truth_b = EcologicalGroundTruth::new(env_b, pop_b, 10.0);

        let mut policy_a = ExtinctionObservationPolicy::new(5, 0.4, 5, 0.05, true, 7);
        let mut policy_b = ExtinctionObservationPolicy::new(5, 0.4, 5, 0.05, true, 7);

        let obs_a = policy_a.observe(&truth_a, 3); // 3 is not a multiple of 5
        let obs_b = policy_b.observe(&truth_b, 3);
        assert_eq!(obs_a, obs_b);
        assert_eq!(obs_a.sample, None);
    }
}

#[cfg(test)]
mod population_census_tests {
    use super::*;
    use crate::ObservationPolicy;
    use symthaea_alife::{OrganismConfig, PopulationConfig};

    fn base_setup(seed: u64, initial_count: usize) -> (EarthForcedEnvironment, Population) {
        let env = EarthForcedEnvironment::earth_like(200.0);
        let cfg = PopulationConfig {
            death_energy_threshold: 0.05,
            reproduction_energy_threshold: 0.8,
            reproduction_energy_cost: 0.4,
            organism_cfg: OrganismConfig {
                forage_efficiency: 0.6,
                ..OrganismConfig::default()
            },
            ..Default::default()
        };
        let pop = Population::new(cfg, initial_count, seed);
        (env, pop)
    }

    #[test]
    fn population_above_threshold_reports_only_the_cap_regardless_of_true_count() {
        // 8 alive vs. 20 alive, sample_size=5 -- both above threshold, output must be identical
        // (this is the leakage-safety property this policy actually relies on: differing
        // exactly how far above the threshold the true count is must not be observable).
        let (env_a, pop_a) = base_setup(11, 8);
        let (env_b, pop_b) = base_setup(11, 20);
        let truth_a = EcologicalGroundTruth::new(env_a, pop_a, 10.0);
        let truth_b = EcologicalGroundTruth::new(env_b, pop_b, 10.0);

        let mut policy_a = PopulationCensusObservationPolicy::new(5, 1);
        let mut policy_b = PopulationCensusObservationPolicy::new(5, 1);

        let obs_a = policy_a.observe(&truth_a, 0);
        let obs_b = policy_b.observe(&truth_b, 0);
        assert_eq!(obs_a, obs_b);
        assert_eq!(obs_a.sample.unwrap().sampled_alive_count, 5);
    }

    #[test]
    fn organism_energy_never_affects_the_reported_count() {
        let (env_a, mut pop_a) = base_setup(11, 6);
        let (env_b, pop_b) = base_setup(11, 6);
        pop_a.organisms[0].energy = 0.9999; // differ an organism's energy, same total count
        let truth_a = EcologicalGroundTruth::new(env_a, pop_a, 10.0);
        let truth_b = EcologicalGroundTruth::new(env_b, pop_b, 10.0);

        let mut policy_a = PopulationCensusObservationPolicy::new(4, 1);
        let mut policy_b = PopulationCensusObservationPolicy::new(4, 1);

        assert_eq!(policy_a.observe(&truth_a, 0), policy_b.observe(&truth_b, 0));
    }

    #[test]
    fn below_threshold_reveals_the_true_remaining_count_intentionally() {
        let (env, pop) = base_setup(11, 3); // fewer than sample_size=5
        let truth = EcologicalGroundTruth::new(env, pop, 10.0);
        let mut policy = PopulationCensusObservationPolicy::new(5, 1);

        let obs = policy.observe(&truth, 0);
        assert_eq!(obs.sample.unwrap().sampled_alive_count, 3);
    }

    #[test]
    fn tick_outside_observation_frequency_is_none_regardless_of_ground_truth() {
        let (env_a, pop_a) = base_setup(11, 8);
        let (env_b, mut pop_b) = base_setup(11, 8);
        pop_b.organisms.clear(); // fully extinct in b, fully alive in a

        let truth_a = EcologicalGroundTruth::new(env_a, pop_a, 10.0);
        let truth_b = EcologicalGroundTruth::new(env_b, pop_b, 10.0);

        let mut policy_a = PopulationCensusObservationPolicy::new(5, 5);
        let mut policy_b = PopulationCensusObservationPolicy::new(5, 5);

        let obs_a = policy_a.observe(&truth_a, 3);
        let obs_b = policy_b.observe(&truth_b, 3);
        assert_eq!(obs_a, obs_b);
        assert_eq!(obs_a.sample, None);
    }
}
