// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! The Futures Laboratory's third scenario family, per the Phase 2 addendum's item 1 checklist
//! (`SYMTHAEA_FUTURES_LABORATORY_PLAN_2026-07-25.md`): hunt for the external-forcing lever
//! *before* building a rung hierarchy. Here the lever was already found and validated —
//! `symthaea-alife`'s own `tests/phase7_evolutionary_rescue.rs` ("evolving genome survives longer
//! under a real worsening climate") unifies `EarthForcedEnvironment::with_secular_drift` (a
//! permanent solar-constant decline layered on the seasonal cycle) with a heritable, mutable
//! `Genome` — a real bifurcation collapse is guaranteed eventually (an intrinsic property of the
//! ice-albedo model, not something any genome can prevent), but an evolving population survives
//! measurably longer than a frozen one (73-226 ticks longer, confirmed in 8/8 seeds).
//!
//! ## Why this is a genuinely new forecasting target, not `ecological` relabeled
//!
//! `ecological`/`predator_prey` both ask "does extinction happen within horizon" from a
//! population-count signal alone. Here the ground truth also carries a heritable trait
//! (`forage_efficiency`) whose trajectory is informative about *how much longer* the population
//! will survive — a forecaster that could observe the trait's rise would have a genuinely
//! different, richer signal than population count alone. **This first version deliberately does
//! not expose that signal**: `PopulationCensusObservationPolicy` here reveals only a capped
//! population count, exactly like `ecological`'s policy of the same name — the trait/genome state
//! is oracle/evaluation-only (`EvolutionaryRescueGroundTruth::true_mean_forage_efficiency`),
//! never crossing the observation firewall. A future policy that reveals a noised trait signal is
//! the natural next step for this family specifically, not built here.
//!
//! ## Scope note
//!
//! Matches Phase 1's own precedent: this is the ground-truth adapter + observation firewall only
//! (mirroring how `ecological`/`predator_prey` each started) — no `TrajectoryGenerator`/rung
//! implementations exist for this family yet.
//!
//! ## Phase 2.2B — the noised-trait / shuffled-trait observation infrastructure (2026-07-27)
//!
//! Per the plan's predeclared design, this scenario family's real question is not "can
//! evolutionary rescue be forecast" but "does partial, noisy information about the evolving
//! trait improve out-of-seed forecasts of collapse timing *beyond* population observations
//! alone" — evaluated across four information conditions. This increment builds the three new
//! observation policies (conditions 2-4); condition 1 is [`PopulationCensusObservationPolicy`]
//! above, unchanged.
//!
//! - [`NoisyTraitObservationPolicy`] (condition 2): capped census, identical to condition 1's
//!   formula, plus a noised cross-sectional mean `forage_efficiency` reading over a **fresh
//!   random subsample drawn every tick** — deliberately not a fixed tracked cohort (unlike
//!   `ExtinctionObservationPolicy`'s `AgentId`-based cohort in `ecological.rs`), because a
//!   persistent cohort here would let a forecaster infer individual lineage survival, a strictly
//!   richer (and unintended) signal than "the population's trait distribution shifted." Draws
//!   from its own independent xorshift64 stream, matching `ecological`'s established pattern for
//!   why noise must never share state with anything the ground truth or its own RNGs touch.
//! - [`PrivilegedTraitObservationPolicy`] (condition 4): capped census plus the *exact*
//!   [`EvolutionaryRescueGroundTruth::true_mean_forage_efficiency`], zero noise — an
//!   evaluation-only upper bound. **Never a real rung input**; any `TrajectoryGenerator` reading
//!   this policy's output is, by construction, cheating.
//! - [`shuffle_trait_readings`] (condition 3, the load-bearing control): **not** an
//!   `ObservationPolicy` at all, and deliberately so — condition 3 is a decorrelation ablation
//!   applied *after the fact* to a recorded trajectory of condition-2 readings (permute the
//!   `Some` values across their tick positions, holding `None` — unobserved — positions fixed),
//!   which is inherently non-causal (it needs the full sequence in hand) and therefore cannot be
//!   expressed as a per-tick `observe()` call. If a model trained on shuffled trait readings
//!   scores as well as one trained on the real (unshuffled) sequence, the improvement was never
//!   about evolutionary information — see the plan's acceptance gate.
//!
//! **Not built in this increment**: the four-condition experiment itself, any
//! `TrajectoryGenerator` rungs (the plan's own new rung hierarchy — historical baseline,
//! census-only predictor, trait-trend predictor, mechanistic adaptation-vs-forcing model,
//! FEP-on-census, FEP-on-census-plus-trait, privileged hindsight), and evidence-ledger wiring.
//! `NoisyTraitObservationPolicy::POLICY_VERSION` and its accessor methods exist so a future
//! ledger-recording step has concrete provenance fields to read, but nothing in this pass writes
//! to `symthaea-futures-ledger` — that remains real, disclosed, not-yet-done work.

use symthaea_alife::{EarthForcedEnvironment, Population};

/// Hidden ground truth for one evolutionary-rescue scenario instance. No `Clone`/`Debug`/
/// `PartialEq` derive, matching `ecological::EcologicalGroundTruth`'s reasoning: `Population`
/// itself has none of these (it owns per-organism `ActiveInferenceAgent`s with no such derives
/// either).
pub struct EvolutionaryRescueGroundTruth {
    pub environment: EarthForcedEnvironment,
    pub population: Population,
    pub plant_resource_total: f64,
    tick: u64,
}

impl EvolutionaryRescueGroundTruth {
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

    /// Advances one tick, reproducing `tests/phase7_evolutionary_rescue.rs`'s exact coupling:
    /// `population.step(|n| environment.step() * plant_resource_total / n.max(1))`. Field-borrow-
    /// split (same pattern `ecological::EcologicalGroundTruth::step` uses) because a closure
    /// capturing `self` directly would collide with the second mutable borrow.
    pub fn step(&mut self) {
        self.tick += 1;
        let environment = &mut self.environment;
        let plant_resource_total = self.plant_resource_total;
        self.population
            .step(|n| environment.step() * plant_resource_total / (n.max(1) as f64));
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

    /// Oracle/evaluation-only accessor — the whole point of this scenario family, but
    /// deliberately never exposed through [`ObservationPolicy`](crate::ObservationPolicy) in this
    /// first version (see module docs).
    pub fn true_mean_forage_efficiency(&self) -> f64 {
        let n = self.population.organisms.len();
        if n == 0 {
            return 0.0;
        }
        self.population
            .organisms
            .iter()
            .map(|o| o.cfg.forage_efficiency)
            .sum::<f64>()
            / n as f64
    }
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EvolutionaryRescueSample {
    pub sampled_alive_count: usize,
    /// `None` under [`PopulationCensusObservationPolicy`] (condition 1 — the trait signal never
    /// crosses the firewall). `Some(noised cross-sectional mean)` under
    /// [`NoisyTraitObservationPolicy`] (condition 2, and — after [`shuffle_trait_readings`] is
    /// applied to a recorded trajectory — condition 3). `Some(exact value)` under
    /// [`PrivilegedTraitObservationPolicy`] (condition 4, evaluation-only).
    pub observed_mean_forage_efficiency: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EvolutionaryRescueObservation {
    pub tick: u64,
    pub sample: Option<EvolutionaryRescueSample>,
}

/// Same design as `ecological::PopulationCensusObservationPolicy`: reports
/// `min(true population count, sample_size)` — a capped census, not identity-based tracking of
/// specific individuals. Leakage-safe because `sample_size` is a fixed construction-time
/// constant: any composition of individuals above the threshold yields an identical output, and
/// the exact count is only ever revealed once the population has genuinely dropped below
/// `sample_size`. Trait/genome state is never read here at all (see module docs).
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
    type GroundTruth = EvolutionaryRescueGroundTruth;
    type Observation = EvolutionaryRescueObservation;

    fn observe(
        &mut self,
        truth: &EvolutionaryRescueGroundTruth,
        tick: u64,
    ) -> EvolutionaryRescueObservation {
        if tick % self.observation_frequency_ticks != 0 {
            return EvolutionaryRescueObservation { tick, sample: None };
        }

        let count = truth.true_population_count().min(self.sample_size);
        EvolutionaryRescueObservation {
            tick,
            sample: Some(EvolutionaryRescueSample {
                sampled_alive_count: count,
                observed_mean_forage_efficiency: None,
            }),
        }
    }
}

/// xorshift64 step, identical formula to `ecological::ExtinctionObservationPolicy`'s — kept as a
/// free function here rather than shared across modules because each policy's stream must be
/// seeded and owned independently (see module docs on why noise must never share state with
/// anything else).
fn xorshift64_next_unit(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state as f64) / (u64::MAX as f64)
}

/// Condition 2 (and, via [`shuffle_trait_readings`], the raw material for condition 3): capped
/// census identical to [`PopulationCensusObservationPolicy`], plus a noised cross-sectional mean
/// `forage_efficiency` reading. See module docs for the design reasoning.
pub struct NoisyTraitObservationPolicy {
    observation_frequency_ticks: u64,
    sample_size: usize,
    trait_sample_size: usize,
    trait_noise_amplitude: f64,
    noise_rng_state: u64,
}

impl NoisyTraitObservationPolicy {
    /// A version tag for evidence-ledger provenance (not yet wired to
    /// `symthaea-futures-ledger` — see module docs). Bump this if the sampling or noise formula
    /// below ever changes, so recorded runs stay attributable to the policy version that
    /// produced them.
    pub const POLICY_VERSION: &'static str = "noisy_trait_v1";

    /// `trait_sample_size` is the cross-sectional cohort size drawn fresh every observation tick
    /// — a fixed, disclosed budget, not derived from the live population count. If the true
    /// population is smaller than this budget, the entire population is sampled instead (the
    /// disclosed small-population fallback the plan asks for) rather than erroring or
    /// over-sampling with replacement.
    pub fn new(
        sample_size: usize,
        trait_sample_size: usize,
        trait_noise_amplitude: f64,
        observation_frequency_ticks: u64,
        noise_seed: u64,
    ) -> Self {
        Self {
            observation_frequency_ticks: observation_frequency_ticks.max(1),
            sample_size,
            trait_sample_size,
            trait_noise_amplitude,
            noise_rng_state: if noise_seed == 0 { 1 } else { noise_seed },
        }
    }

    pub fn sample_size(&self) -> usize {
        self.sample_size
    }

    pub fn trait_sample_size(&self) -> usize {
        self.trait_sample_size
    }

    pub fn trait_noise_amplitude(&self) -> f64 {
        self.trait_noise_amplitude
    }

    fn next_unit(&mut self) -> f64 {
        xorshift64_next_unit(&mut self.noise_rng_state)
    }

    /// Draws a fresh random subsample (without replacement, capped at population size) of the
    /// currently-alive organisms' `forage_efficiency`, returns its noised mean. `None` only when
    /// the population is fully extinct. No organism identity persists across calls — a partial
    /// Fisher-Yates draw over the *current* index range each time, so which individuals
    /// contribute varies tick to tick even if the population itself is momentarily unchanged.
    fn sample_trait(&mut self, population: &symthaea_alife::Population) -> Option<f64> {
        let n = population.organisms.len();
        if n == 0 {
            return None;
        }
        let k = self.trait_sample_size.min(n).max(1);

        let mut indices: Vec<usize> = (0..n).collect();
        for i in 0..k {
            let remaining = n - i;
            let draw = (self.next_unit() * remaining as f64) as usize;
            let j = i + draw.min(remaining - 1);
            indices.swap(i, j);
        }

        let sum: f64 = indices[..k]
            .iter()
            .map(|&idx| population.organisms[idx].cfg.forage_efficiency)
            .sum();
        let mean = sum / k as f64;
        let noise = (self.next_unit() - 0.5) * 2.0 * self.trait_noise_amplitude;
        Some(mean + noise)
    }
}

impl crate::ObservationPolicy for NoisyTraitObservationPolicy {
    type GroundTruth = EvolutionaryRescueGroundTruth;
    type Observation = EvolutionaryRescueObservation;

    fn observe(
        &mut self,
        truth: &EvolutionaryRescueGroundTruth,
        tick: u64,
    ) -> EvolutionaryRescueObservation {
        if tick % self.observation_frequency_ticks != 0 {
            return EvolutionaryRescueObservation { tick, sample: None };
        }

        let sampled_alive_count = truth.true_population_count().min(self.sample_size);
        let observed_mean_forage_efficiency = self.sample_trait(&truth.population);

        EvolutionaryRescueObservation {
            tick,
            sample: Some(EvolutionaryRescueSample {
                sampled_alive_count,
                observed_mean_forage_efficiency,
            }),
        }
    }
}

/// Condition 4: capped census plus the *exact* true mean `forage_efficiency`, zero noise.
/// **Evaluation-only upper bound — never a legitimate rung input.** Exists so the plan's
/// four-condition comparison has a ceiling to measure the other three conditions against, the
/// same role `EcologicalGroundTruth`'s planned `OracleUpperBound` generator would play for
/// `ecological`.
pub struct PrivilegedTraitObservationPolicy {
    observation_frequency_ticks: u64,
    sample_size: usize,
}

impl PrivilegedTraitObservationPolicy {
    pub fn new(sample_size: usize, observation_frequency_ticks: u64) -> Self {
        Self {
            observation_frequency_ticks: observation_frequency_ticks.max(1),
            sample_size,
        }
    }
}

impl crate::ObservationPolicy for PrivilegedTraitObservationPolicy {
    type GroundTruth = EvolutionaryRescueGroundTruth;
    type Observation = EvolutionaryRescueObservation;

    fn observe(
        &mut self,
        truth: &EvolutionaryRescueGroundTruth,
        tick: u64,
    ) -> EvolutionaryRescueObservation {
        if tick % self.observation_frequency_ticks != 0 {
            return EvolutionaryRescueObservation { tick, sample: None };
        }

        let sampled_alive_count = truth.true_population_count().min(self.sample_size);
        let observed_mean_forage_efficiency = if truth.is_extinct() {
            None
        } else {
            Some(truth.true_mean_forage_efficiency())
        };

        EvolutionaryRescueObservation {
            tick,
            sample: Some(EvolutionaryRescueSample {
                sampled_alive_count,
                observed_mean_forage_efficiency,
            }),
        }
    }
}

/// Condition 3, the load-bearing control (see module docs for why this is a free function over
/// a recorded trajectory, not an `ObservationPolicy`). Permutes the `Some` trait readings across
/// their tick positions using an independent xorshift64 stream (`shuffle_seed`), leaving `None`
/// (unobserved-tick) positions exactly where they were — shuffling into or out of an
/// unobserved-tick slot would change *when* information arrives, confounding the ablation with a
/// timing artifact rather than isolating "is this specific trait-value sequence informative."
///
/// A Fisher-Yates shuffle restricted to the subsequence of `Some` indices.
pub fn shuffle_trait_readings(readings: &[Option<f64>], shuffle_seed: u64) -> Vec<Option<f64>> {
    let mut state = if shuffle_seed == 0 { 1 } else { shuffle_seed };
    let some_positions: Vec<usize> = readings
        .iter()
        .enumerate()
        .filter_map(|(i, r)| r.map(|_| i))
        .collect();
    let mut values: Vec<f64> = some_positions
        .iter()
        .map(|&i| readings[i].expect("index came from a Some position"))
        .collect();

    let n = values.len();
    for i in 0..n.saturating_sub(1) {
        let remaining = n - i;
        let draw = (xorshift64_next_unit(&mut state) * remaining as f64) as usize;
        let j = i + draw.min(remaining - 1);
        values.swap(i, j);
    }

    let mut out = readings.to_vec();
    for (&pos, &val) in some_positions.iter().zip(values.iter()) {
        out[pos] = Some(val);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ObservationPolicy;
    use symthaea_alife::{InheritanceMode, OrganismConfig, PopulationConfig};

    const SEASONAL_PERIOD_TICKS: f64 = 200.0;
    const SECULAR_DRIFT_PER_TICK: f64 = -0.01;
    const PLANT_RESOURCE_TOTAL: f64 = 3.0;

    fn population_config(mutation_rate: f64) -> PopulationConfig {
        PopulationConfig {
            death_energy_threshold: 0.05,
            reproduction_energy_threshold: 0.8,
            reproduction_energy_cost: 0.4,
            organism_cfg: OrganismConfig::default(),
            mutation_rate,
            mutation_std: 0.05,
            inheritance: InheritanceMode::FromParent,
        }
    }

    fn truth_with(
        mutation_rate: f64,
        initial_count: usize,
        seed: u64,
    ) -> EvolutionaryRescueGroundTruth {
        let environment = EarthForcedEnvironment::earth_like(SEASONAL_PERIOD_TICKS)
            .with_secular_drift(SECULAR_DRIFT_PER_TICK);
        let population = Population::new(population_config(mutation_rate), initial_count, seed);
        EvolutionaryRescueGroundTruth::new(environment, population, PLANT_RESOURCE_TOTAL)
    }

    #[test]
    fn population_census_never_leaks_trait_state() {
        // Same population count (12), same seed -- only mutation_rate (hence trait trajectory)
        // differs. If trait state ever leaked, these observations would diverge.
        let frozen = truth_with(0.0, 12, 7);
        let evolving = truth_with(0.1, 12, 7);

        let mut policy_a = PopulationCensusObservationPolicy::new(20, 1);
        let mut policy_b = PopulationCensusObservationPolicy::new(20, 1);

        assert_eq!(policy_a.observe(&frozen, 0), policy_b.observe(&evolving, 0));
    }

    #[test]
    fn population_census_caps_regardless_of_true_count_above_threshold() {
        let small = truth_with(0.1, 15, 3);
        let large = truth_with(0.1, 30, 3);

        let mut policy_small = PopulationCensusObservationPolicy::new(10, 1);
        let mut policy_large = PopulationCensusObservationPolicy::new(10, 1);

        let obs_small = policy_small.observe(&small, 0);
        let obs_large = policy_large.observe(&large, 0);
        assert_eq!(obs_small, obs_large);
        assert_eq!(
            obs_small.sample.unwrap().sampled_alive_count,
            10,
            "both should be capped at sample_size, not the true (differing) counts"
        );
    }

    #[test]
    fn off_frequency_tick_reveals_nothing_regardless_of_ground_truth() {
        let alive_truth = truth_with(0.1, 12, 1);
        let mut extinct_truth = truth_with(0.1, 12, 1);
        // Force one to extinction, leave the other alive -- the policy still must not reveal
        // anything different for an off-frequency tick.
        extinct_truth.population.organisms.clear();

        let mut policy = PopulationCensusObservationPolicy::new(5, 10);
        let obs_alive = policy.observe(&alive_truth, 3);
        let obs_extinct = policy.observe(&extinct_truth, 3);

        assert_eq!(
            obs_alive,
            EvolutionaryRescueObservation {
                tick: 3,
                sample: None
            }
        );
        assert_eq!(obs_alive, obs_extinct);
    }

    #[test]
    fn noisy_trait_census_component_matches_census_only_policy() {
        let truth = truth_with(0.1, 15, 5);
        let mut census_only = PopulationCensusObservationPolicy::new(10, 1);
        let mut noisy_trait = NoisyTraitObservationPolicy::new(10, 5, 0.01, 1, 99);

        let census_obs = census_only.observe(&truth, 0);
        let trait_obs = noisy_trait.observe(&truth, 0);
        assert_eq!(
            census_obs.sample.unwrap().sampled_alive_count,
            trait_obs.sample.unwrap().sampled_alive_count,
            "condition 2's census component must reproduce condition 1's formula exactly"
        );
    }

    #[test]
    fn noisy_trait_reading_stays_close_to_true_mean_within_noise_bounds() {
        let truth = truth_with(0.1, 20, 11);
        let true_mean = truth.true_mean_forage_efficiency();
        let noise_amplitude = 0.02;
        let mut policy = NoisyTraitObservationPolicy::new(20, 20, noise_amplitude, 1, 3);

        let obs = policy.observe(&truth, 0);
        let observed = obs.sample.unwrap().observed_mean_forage_efficiency.unwrap();
        assert!(
            (observed - true_mean).abs() <= noise_amplitude + 1e-9,
            "observed {observed} should stay within +-{noise_amplitude} of true mean {true_mean} \
             when the full population is sampled"
        );
    }

    #[test]
    fn noisy_trait_extinct_population_yields_no_trait_reading() {
        let mut truth = truth_with(0.1, 10, 2);
        truth.population.organisms.clear();
        let mut policy = NoisyTraitObservationPolicy::new(10, 10, 0.02, 1, 3);

        let obs = policy.observe(&truth, 0);
        let sample = obs.sample.unwrap();
        assert_eq!(sample.sampled_alive_count, 0);
        assert_eq!(sample.observed_mean_forage_efficiency, None);
    }

    #[test]
    fn noisy_trait_small_population_falls_back_to_sampling_everyone() {
        // trait_sample_size=50 requested but only 4 organisms exist -- must not panic or
        // over-sample with replacement, and must still produce a reading close to the true mean.
        let truth = truth_with(0.1, 4, 6);
        let true_mean = truth.true_mean_forage_efficiency();
        let mut policy = NoisyTraitObservationPolicy::new(10, 50, 0.01, 1, 3);

        let obs = policy.observe(&truth, 0);
        let observed = obs.sample.unwrap().observed_mean_forage_efficiency.unwrap();
        assert!((observed - true_mean).abs() <= 0.01 + 1e-9);
    }

    #[test]
    fn noisy_trait_resamples_fresh_each_tick_not_a_fixed_cohort() {
        // Same unchanging population observed on two different frequency-satisfying ticks --
        // the noise draws (hence the two readings) must differ, proving the RNG genuinely
        // advances rather than a fixed cohort/offset being reused.
        let truth = truth_with(0.1, 20, 9);
        let mut policy = NoisyTraitObservationPolicy::new(20, 5, 0.05, 1, 3);

        let first = policy
            .observe(&truth, 0)
            .sample
            .unwrap()
            .observed_mean_forage_efficiency
            .unwrap();
        let second = policy
            .observe(&truth, 1)
            .sample
            .unwrap()
            .observed_mean_forage_efficiency
            .unwrap();
        assert_ne!(
            first, second,
            "the RNG stream must advance between observations"
        );
    }

    #[test]
    fn noisy_trait_rng_stream_is_independent_of_population_rng_seed() {
        // Two ground truths differing only in the simulation-internal RNG seed used to
        // construct the population, but with the same noise_seed for the observation policy --
        // if the noise stream ever derived from the population's own RNG state, these would
        // diverge even when the trait distribution coincidentally matches.
        let truth_a = truth_with(0.0, 10, 1); // mutation_rate=0.0 -> uniform starting trait, seed 1
        let truth_b = truth_with(0.0, 10, 2); // same uniform starting trait, different seed
        let mut policy_a = NoisyTraitObservationPolicy::new(10, 10, 0.02, 1, 42);
        let mut policy_b = NoisyTraitObservationPolicy::new(10, 10, 0.02, 1, 42);

        // With mutation_rate 0.0 and OrganismConfig::default(), every organism starts with an
        // identical forage_efficiency regardless of the population's RNG seed, so the true mean
        // is identical -- isolating the noise stream as the only remaining source of difference.
        assert_eq!(
            truth_a.true_mean_forage_efficiency(),
            truth_b.true_mean_forage_efficiency()
        );
        assert_eq!(policy_a.observe(&truth_a, 0), policy_b.observe(&truth_b, 0));
    }

    #[test]
    fn privileged_trait_reveals_exact_true_mean_with_zero_noise() {
        let truth = truth_with(0.1, 15, 4);
        let true_mean = truth.true_mean_forage_efficiency();
        let mut policy = PrivilegedTraitObservationPolicy::new(15, 1);

        let obs = policy.observe(&truth, 0);
        assert_eq!(
            obs.sample.unwrap().observed_mean_forage_efficiency,
            Some(true_mean)
        );
    }

    #[test]
    fn privileged_trait_extinct_population_yields_no_reading() {
        let mut truth = truth_with(0.1, 15, 4);
        truth.population.organisms.clear();
        let mut policy = PrivilegedTraitObservationPolicy::new(15, 1);

        let obs = policy.observe(&truth, 0);
        let sample = obs.sample.unwrap();
        assert_eq!(sample.sampled_alive_count, 0);
        assert_eq!(sample.observed_mean_forage_efficiency, None);
    }

    #[test]
    fn shuffle_preserves_the_multiset_of_values_and_the_none_positions() {
        let readings = vec![
            Some(0.1),
            None,
            Some(0.2),
            Some(0.3),
            None,
            Some(0.4),
            Some(0.5),
        ];
        let shuffled = shuffle_trait_readings(&readings, 12345);

        assert_eq!(shuffled.len(), readings.len());
        assert_eq!(shuffled[1], None);
        assert_eq!(shuffled[4], None);

        let mut original_values: Vec<f64> = readings.iter().filter_map(|r| *r).collect();
        let mut shuffled_values: Vec<f64> = shuffled.iter().filter_map(|r| *r).collect();
        original_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        shuffled_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(
            original_values, shuffled_values,
            "shuffling must not add, drop, or alter any value -- only reorder them"
        );
    }

    #[test]
    fn shuffle_actually_reorders_and_is_deterministic_given_a_seed() {
        let readings: Vec<Option<f64>> = (0..30).map(|i| Some(i as f64)).collect();

        let shuffled_a = shuffle_trait_readings(&readings, 777);
        let shuffled_b = shuffle_trait_readings(&readings, 777);
        assert_eq!(
            shuffled_a, shuffled_b,
            "same seed must reproduce the same permutation"
        );
        assert_ne!(
            shuffled_a, readings,
            "a 30-element shuffle matching the identity permutation by chance is astronomically \
             unlikely -- this is a real permutation, not a no-op"
        );

        let shuffled_c = shuffle_trait_readings(&readings, 778);
        assert_ne!(
            shuffled_a, shuffled_c,
            "a different seed should (overwhelmingly likely) produce a different permutation"
        );
    }

    #[test]
    fn shuffle_of_all_none_is_a_no_op() {
        let readings: Vec<Option<f64>> = vec![None, None, None];
        assert_eq!(shuffle_trait_readings(&readings, 1), readings);
    }
}
