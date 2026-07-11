// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Belief spread: how a [`Meme`] propagates through a population of minds.
//!
//! Two layers, deliberately kept consistent:
//!
//! 1. **Closed form** — [`BeliefSpread::to_sir`] derives an [`Sir`] model (reusing
//!    `symthaea-epidemiology`) whose `β` comes from the *mean adoption
//!    probability* across the population and whose `γ` is a boredom/decay
//!    constant. `Sir::basic_reproduction_number()` is then a **measured R₀ for
//!    an idea**, not a disease.
//!
//! 2. **Agent-based** — [`BeliefSpread::measure_r0`] and [`BeliefSpread::outbreak`]
//!    actually spread the meme mind-to-mind. `measure_r0` independently confirms
//!    the discrete stochastic implementation realizes the closed-form R₀ (a real
//!    correctness check — such sims are easy to get subtly wrong). `outbreak`
//!    lets copies *mutate* as they spread, exposing the memetics insight that
//!    **low-fidelity transmission collapses spread** in a population aligned to
//!    the original idea — an emergent result, not a fitted one.
//!
//! ## Adoption model (the one modeling choice)
//!
//! Confirmation bias: a mind adopts an idea in proportion to how well it aligns
//! with what it already believes. [`resonance_gain`] sharply prefers
//! above-chance-similar ideas and ~never adopts below-chance ones. This is the
//! single assumption; everything else is mechanical.

use crate::meme::Meme;
use symthaea_core::hdc::binary_hv::BinaryHV;
use symthaea_epidemiology::Sir;

/// Deterministic seedable PRNG (splitmix64). Pure `std`, reproducible — tests
/// pin the seed so results are stable.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }
    /// Next raw 64-bit draw.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in `[0, n)`.
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
    /// Uniform in `[0, 1)`.
    pub fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    /// Bernoulli trial with success probability `p`.
    fn prob(&mut self, p: f32) -> bool {
        self.unit() < p
    }
}

/// Confirmation-bias response to resonance.
///
/// Maps Hamming similarity `r ∈ [0, 1]` (0.5 = chance) to an adoption gain in
/// `[0, 1]`: below chance ⇒ 0, and above chance rises *quadratically*, so a
/// drifted variant that has slipped toward chance is sharply less adoptable
/// than a faithful copy. This steepness is what makes transmission fidelity
/// matter for spread.
pub fn resonance_gain(r: f32) -> f32 {
    let x = ((r - 0.5) / 0.5).clamp(0.0, 1.0);
    x * x
}

/// Per-contact probability that a susceptible mind adopts `meme`.
///
/// `susceptibility * meme.fitness * resonance_gain(similarity(meme, belief))`,
/// clamped to `[0, 1]`. Fitness is the idea's intrinsic stickiness; resonance
/// is its fit to *this* mind.
pub fn adoption_probability(meme: &Meme, belief: &BinaryHV, susceptibility: f32) -> f32 {
    (susceptibility * meme.fitness * resonance_gain(meme.payload.similarity(belief)))
        .clamp(0.0, 1.0)
}

/// A population of minds, each a belief hypervector.
pub struct Population {
    pub beliefs: Vec<BinaryHV>,
}

impl Population {
    pub fn new(beliefs: Vec<BinaryHV>) -> Self {
        Self { beliefs }
    }

    pub fn len(&self) -> usize {
        self.beliefs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.beliefs.is_empty()
    }

    /// `n` minds whose beliefs cluster around `center` (each = `center` with
    /// `spread` fraction of bits flipped). Models a population that already
    /// broadly shares an idea.
    pub fn aligned_to(center: &BinaryHV, n: usize, spread: f32, seed: u64) -> Self {
        let beliefs = (0..n)
            .map(|i| center.add_noise(spread, seed.wrapping_add(i as u64)))
            .collect();
        Self::new(beliefs)
    }

    /// `n` minds with independent random beliefs (no shared prior).
    pub fn random(n: usize, seed: u64) -> Self {
        let beliefs = (0..n)
            .map(|i| BinaryHV::random(seed.wrapping_add(i as u64)))
            .collect();
        Self::new(beliefs)
    }

    /// Mean per-contact adoption probability of `meme` across the population.
    /// This is `p̄`, the bridge between the agent model and the closed form.
    pub fn mean_adoption(&self, meme: &Meme, susceptibility: f32) -> f64 {
        if self.beliefs.is_empty() {
            return 0.0;
        }
        let sum: f64 = self
            .beliefs
            .iter()
            .map(|b| adoption_probability(meme, b, susceptibility) as f64)
            .sum();
        sum / self.beliefs.len() as f64
    }
}

/// Parameters governing how ideas spread through a [`Population`].
#[derive(Debug, Clone, Copy)]
pub struct BeliefSpread {
    /// Baseline adoption scaling in `[0, 1]` (how open minds are, in general).
    pub susceptibility: f32,
    /// Contacts each carrier makes per step (the mixing / exposure rate `c`).
    pub contact_rate: f64,
    /// Per-step probability a carrier loses interest (`γ`); mean carrier
    /// lifetime is `1/γ` steps.
    pub recovery_rate: f64,
}

/// Outcome of a full agent-based [`BeliefSpread::outbreak`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OutbreakStats {
    /// Fraction of the population that ever adopted the idea (`R∞`).
    pub final_size: f64,
    /// Peak simultaneous fraction carrying the idea.
    pub peak: f64,
    /// Deepest transmission generation reached.
    pub max_generation: u32,
    /// Mean fidelity of adopted variants to the seed idea (1.0 = no drift).
    pub mean_fidelity_to_seed: f64,
    /// Steps simulated until the idea died out (or the cap was hit).
    pub steps: usize,
}

impl BeliefSpread {
    /// Number of contacts a carrier makes this step (`floor(c)` plus one more
    /// with probability `frac(c)`), so fractional `contact_rate` is unbiased.
    fn sample_contacts(&self, rng: &mut Rng) -> usize {
        let base = self.contact_rate.floor();
        let frac = (self.contact_rate - base) as f32;
        base as usize + if rng.prob(frac) { 1 } else { 0 }
    }

    /// Closed-form SIR model for `meme` in `population`.
    ///
    /// `β = contact_rate · p̄` (mass action) and `γ = recovery_rate`, so
    /// `Sir::basic_reproduction_number()` is the idea's R₀.
    pub fn to_sir(&self, meme: &Meme, population: &Population) -> Sir {
        let p_bar = population.mean_adoption(meme, self.susceptibility);
        Sir {
            beta: self.contact_rate * p_bar,
            gamma: self.recovery_rate,
        }
    }

    /// Empirically measure R₀: average number of secondary adoptions caused by
    /// a *single* carrier in an otherwise-fully-susceptible population, over
    /// `trials`. Secondaries do not themselves spread here — this isolates R₀.
    ///
    /// Should match `self.to_sir(meme, pop).basic_reproduction_number()` within
    /// Monte-Carlo error, confirming the discrete dynamics are unbiased.
    pub fn measure_r0(&self, meme: &Meme, pop: &Population, trials: usize, rng: &mut Rng) -> f64 {
        let n = pop.len();
        if n < 2 || trials == 0 {
            return 0.0;
        }
        let mut total: u64 = 0;
        for _ in 0..trials {
            let carrier = rng.below(n);
            let mut infected = vec![false; n];
            let mut count: u64 = 0;
            loop {
                for _ in 0..self.sample_contacts(rng) {
                    let t = rng.below(n);
                    if t == carrier || infected[t] {
                        continue; // wasted contact (susceptible pool ≈ all)
                    }
                    if rng.prob(adoption_probability(
                        meme,
                        &pop.beliefs[t],
                        self.susceptibility,
                    )) {
                        infected[t] = true;
                        count += 1;
                    }
                }
                if rng.prob(self.recovery_rate as f32) {
                    break; // carrier lost interest
                }
            }
            total += count;
        }
        total as f64 / trials as f64
    }

    /// Full agent-based outbreak with *mutating* transmission.
    ///
    /// One seed carrier starts with `seed_meme`; carriers transmit mutated
    /// copies (`mutation` = per-bit flip prob) to contacts, who adopt the
    /// *received variant* per [`adoption_probability`]. Runs until the idea dies
    /// out. In a population aligned to `seed_meme`, higher `mutation` drives
    /// variants toward chance-resonance and collapses spread.
    pub fn outbreak(
        &self,
        seed_meme: &Meme,
        pop: &Population,
        mutation: f32,
        rng: &mut Rng,
        max_steps: usize,
    ) -> OutbreakStats {
        let n = pop.len();
        if n == 0 {
            return OutbreakStats {
                final_size: 0.0,
                peak: 0.0,
                max_generation: 0,
                mean_fidelity_to_seed: 0.0,
                steps: 0,
            };
        }

        // Per-agent carried variant (None = susceptible or recovered).
        let mut carried: Vec<Option<Meme>> = vec![None; n];
        let mut infectious: Vec<bool> = vec![false; n];
        let mut ever_infected: Vec<bool> = vec![false; n];

        let seed_idx = rng.below(n);
        carried[seed_idx] = Some(seed_meme.clone());
        infectious[seed_idx] = true;
        ever_infected[seed_idx] = true;

        let mut next_id = seed_meme.id + 1;
        let mut max_generation = 0u32;
        let mut fidelity_sum = seed_meme.fidelity(seed_meme) as f64; // 1.0 for the seed
        let mut peak_count = 1usize;
        let mut steps = 0usize;

        loop {
            let active: Vec<usize> = (0..n).filter(|&i| infectious[i]).collect();
            if active.is_empty() || steps >= max_steps {
                break;
            }
            steps += 1;

            // Adoptions are staged so within-step ordering doesn't bias results.
            let mut pending: Vec<(usize, Meme)> = Vec::new();
            for &i in &active {
                let carrier_meme = carried[i].clone().expect("infectious ⇒ carries a meme");
                for _ in 0..self.sample_contacts(rng) {
                    let t = rng.below(n);
                    if t == i || ever_infected[t] {
                        continue;
                    }
                    let variant = carrier_meme.transmit(next_id, mutation, rng.next_u64());
                    next_id += 1;
                    if rng.prob(adoption_probability(
                        &variant,
                        &pop.beliefs[t],
                        self.susceptibility,
                    )) {
                        pending.push((t, variant));
                    }
                }
            }
            for (t, variant) in pending {
                if ever_infected[t] {
                    continue; // first adoption wins if contacted twice in a step
                }
                ever_infected[t] = true;
                infectious[t] = true;
                max_generation = max_generation.max(variant.generation);
                fidelity_sum += seed_meme.fidelity(&variant) as f64;
                carried[t] = Some(variant);
            }

            // Recoveries (after transmission, so mean lifetime is 1/γ).
            for &i in &active {
                if rng.prob(self.recovery_rate as f32) {
                    infectious[i] = false;
                    carried[i] = None;
                }
            }

            peak_count = peak_count.max(infectious.iter().filter(|&&x| x).count());
        }

        let total = ever_infected.iter().filter(|&&x| x).count();
        OutbreakStats {
            final_size: total as f64 / n as f64,
            peak: peak_count as f64 / n as f64,
            max_generation,
            mean_fidelity_to_seed: if total > 0 {
                fidelity_sum / total as f64
            } else {
                0.0
            },
            steps,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::meme::Meme;

    fn seed_meme(seed: u64, fitness: f32) -> Meme {
        Meme::seed(0, BinaryHV::random(seed), fitness)
    }

    #[test]
    fn resonance_gain_is_zero_below_chance_and_rises_above() {
        assert_eq!(resonance_gain(0.5), 0.0);
        assert_eq!(resonance_gain(0.3), 0.0);
        assert!((resonance_gain(1.0) - 1.0).abs() < 1e-6);
        assert!(resonance_gain(0.75) > resonance_gain(0.6));
    }

    #[test]
    fn r0_is_measured_not_hardcoded_and_matches_closed_form() {
        // A population aligned to the idea ⇒ high resonance ⇒ meaningful R₀.
        let center = BinaryHV::random(11);
        let pop = Population::aligned_to(&center, 3000, 0.08, 200);
        let meme = Meme::seed(0, center.clone(), 0.9);
        let spread = BeliefSpread {
            susceptibility: 1.0,
            contact_rate: 3.0,
            recovery_rate: 0.5,
        };

        let predicted = spread.to_sir(&meme, &pop).basic_reproduction_number();
        let mut rng = Rng::new(2024);
        let measured = spread.measure_r0(&meme, &pop, 4000, &mut rng);

        assert!(
            predicted > 1.0,
            "test needs a super-critical idea, R₀={predicted}"
        );
        let rel = (measured - predicted).abs() / predicted;
        assert!(
            rel < 0.10,
            "agent-based R₀ ({measured:.3}) must match closed form ({predicted:.3}), rel={rel:.3}"
        );
    }

    #[test]
    fn low_fidelity_transmission_collapses_spread() {
        // Population that already broadly shares the seed idea.
        let center = BinaryHV::random(5);
        let pop = Population::aligned_to(&center, 2000, 0.08, 700);
        let meme = Meme::seed(0, center.clone(), 0.9);
        let spread = BeliefSpread {
            susceptibility: 1.0,
            contact_rate: 3.0,
            recovery_rate: 0.5,
        };

        let faithful = spread.outbreak(&meme, &pop, 0.02, &mut Rng::new(1), 5000);
        let sloppy = spread.outbreak(&meme, &pop, 0.35, &mut Rng::new(1), 5000);

        // High-fidelity copies keep resonating with the aligned population and
        // spread widely; high-mutation copies drift to chance and fizzle.
        assert!(
            faithful.final_size > 0.5,
            "faithful idea should spread widely, got {}",
            faithful.final_size
        );
        assert!(
            sloppy.final_size < faithful.final_size * 0.5,
            "sloppy idea should collapse: sloppy={} vs faithful={}",
            sloppy.final_size,
            faithful.final_size
        );
        assert!(
            faithful.mean_fidelity_to_seed > sloppy.mean_fidelity_to_seed,
            "faithful lineage should stay closer to the seed"
        );
    }

    #[test]
    fn subcritical_idea_barely_spreads() {
        // A random (non-resonant) idea in a random population: near-chance
        // resonance ⇒ ~zero adoption gain ⇒ R₀ ≈ 0.
        let pop = Population::random(1000, 314);
        let meme = seed_meme(99, 0.9);
        let spread = BeliefSpread {
            susceptibility: 1.0,
            contact_rate: 3.0,
            recovery_rate: 0.5,
        };
        let r0 = spread.to_sir(&meme, &pop).basic_reproduction_number();
        assert!(r0 < 1.0, "unaligned idea should be sub-critical, R₀={r0}");
        let out = spread.outbreak(&meme, &pop, 0.02, &mut Rng::new(3), 2000);
        assert!(
            out.final_size < 0.1,
            "should not take off, got {}",
            out.final_size
        );
    }
}
