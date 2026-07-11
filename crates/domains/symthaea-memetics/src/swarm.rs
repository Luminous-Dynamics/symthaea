// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Multi-agent meme propagation — the *offline* substance of plan Phase 3.
//!
//! Phase 3's networked half (broadcasting to real peers over the mesh) is gated
//! on the mesh peer-authentication sign-off and is **not** here. But its *design*
//! — how a meme spreads across a population of independent immune agents, and how
//! **collective (herd) immunity** works when some agents are vaccinated — can be
//! simulated and validated with no network at all. That's this module.
//!
//! Each agent is an independent mind: its own belief and its own vaccinations.
//! A meme spreads agent-to-agent, mutating as it goes; a susceptible agent adopts
//! a received variant per [`adoption_probability`], unless it recognizes the
//! variant as a vaccinated pathogen (then it rejects — and, crucially, cannot
//! pass it on). This reproduces the SIR result that vaccinating more than the
//! **herd-immunity threshold** `1 − 1/R₀` collapses an outbreak — for *ideas*.

use crate::meme::Meme;
use crate::propagation::{BeliefSpread, Rng, adoption_probability};
use symthaea_core::hdc::binary_hv::BinaryHV;

/// Similarity to a vaccinated pathogen at/above which an agent rejects a variant.
const THREAT_MATCH_THRESHOLD: f32 = 0.7;

/// One mind in the swarm.
struct SwarmAgent {
    belief: BinaryHV,
    /// Vaccinated pathogen signatures — variants resonating with any of these
    /// are rejected (mutation-tolerant, like the single-agent immune system).
    vaccinated: Vec<BinaryHV>,
    /// Variant currently carried (spreading). `None` = susceptible or recovered.
    carrying: Option<Meme>,
    /// Whether this agent ever adopted the idea (for final-size counting).
    ever_adopted: bool,
}

impl SwarmAgent {
    fn recognizes_pathogen(&self, payload: &BinaryHV) -> bool {
        self.vaccinated
            .iter()
            .any(|s| payload.similarity(s).clamp(0.0, 1.0) >= THREAT_MATCH_THRESHOLD)
    }
}

/// Outcome of a swarm outbreak.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SwarmOutcome {
    /// Fraction of the population that ever adopted the idea (`R∞`).
    pub final_adoption: f64,
    /// Peak simultaneous carrying fraction.
    pub peak: f64,
    /// Deepest transmission generation reached.
    pub max_generation: u32,
    /// Steps until die-out (or cap).
    pub steps: usize,
    /// Fraction of the population vaccinated against the seeded idea.
    pub vaccinated_fraction: f64,
}

/// A fully-mixed population of independent immune agents.
pub struct MemeSwarm {
    agents: Vec<SwarmAgent>,
    spread: BeliefSpread,
    mutation: f32,
}

impl MemeSwarm {
    /// Build a swarm from per-agent belief states.
    pub fn new(beliefs: Vec<BinaryHV>, spread: BeliefSpread, mutation: f32) -> Self {
        let agents = beliefs
            .into_iter()
            .map(|belief| SwarmAgent {
                belief,
                vaccinated: Vec::new(),
                carrying: None,
                ever_adopted: false,
            })
            .collect();
        Self {
            agents,
            spread,
            mutation: mutation.clamp(0.0, 1.0),
        }
    }

    pub fn len(&self) -> usize {
        self.agents.len()
    }

    pub fn is_empty(&self) -> bool {
        self.agents.is_empty()
    }

    /// Vaccinate a `fraction` of the population against `pathogen` (random subset).
    /// Returns the actual fraction vaccinated.
    pub fn vaccinate_fraction(&mut self, pathogen: &BinaryHV, fraction: f64, rng: &mut Rng) -> f64 {
        let n = self.agents.len();
        if n == 0 {
            return 0.0;
        }
        let mut count = 0usize;
        for a in &mut self.agents {
            if rng.unit() < fraction as f32 {
                a.vaccinated.push(pathogen.clone());
                count += 1;
            }
        }
        count as f64 / n as f64
    }

    fn contacts(&self, rng: &mut Rng) -> usize {
        let base = self.spread.contact_rate.floor();
        let frac = (self.spread.contact_rate - base) as f32;
        base as usize + if rng.unit() < frac { 1 } else { 0 }
    }

    /// Run an outbreak seeded with `seed_meme` at a random agent, until die-out.
    pub fn run(&mut self, seed_meme: &Meme, rng: &mut Rng, max_steps: usize) -> SwarmOutcome {
        let n = self.agents.len();
        let vaccinated_fraction = if n > 0 {
            self.agents
                .iter()
                .filter(|a| !a.vaccinated.is_empty())
                .count() as f64
                / n as f64
        } else {
            0.0
        };
        if n == 0 {
            return SwarmOutcome {
                final_adoption: 0.0,
                peak: 0.0,
                max_generation: 0,
                steps: 0,
                vaccinated_fraction,
            };
        }

        // Seed at a random *susceptible-capable* agent. If it happens to be
        // vaccinated it simply won't adopt/spread (a realistic dud seeding).
        let seed_idx = rng.below(n);
        if !self.agents[seed_idx].recognizes_pathogen(&seed_meme.payload) {
            self.agents[seed_idx].carrying = Some(seed_meme.clone());
            self.agents[seed_idx].ever_adopted = true;
        }

        let mut next_id = seed_meme.id + 1;
        let mut max_generation = 0u32;
        let mut peak_carrying = self.agents.iter().filter(|a| a.carrying.is_some()).count();
        let mut steps = 0usize;

        loop {
            let carriers: Vec<usize> = (0..n)
                .filter(|&i| self.agents[i].carrying.is_some())
                .collect();
            if carriers.is_empty() || steps >= max_steps {
                break;
            }
            steps += 1;

            let mut pending: Vec<(usize, Meme)> = Vec::new();
            for &i in &carriers {
                let carrier_meme = self.agents[i].carrying.clone().expect("carrier has a meme");
                for _ in 0..self.contacts(rng) {
                    let t = rng.below(n);
                    if t == i || self.agents[t].ever_adopted {
                        continue;
                    }
                    let variant = carrier_meme.transmit(next_id, self.mutation, rng.next_u64());
                    next_id += 1;
                    // Immune agents reject a recognized pathogen variant outright.
                    if self.agents[t].recognizes_pathogen(&variant.payload) {
                        continue;
                    }
                    let p = adoption_probability(
                        &variant,
                        &self.agents[t].belief,
                        self.spread.susceptibility,
                    );
                    if rng.unit() < p {
                        pending.push((t, variant));
                    }
                }
            }
            for (t, variant) in pending {
                if self.agents[t].ever_adopted {
                    continue;
                }
                self.agents[t].ever_adopted = true;
                max_generation = max_generation.max(variant.generation);
                self.agents[t].carrying = Some(variant);
            }

            // Recoveries.
            for &i in &carriers {
                if rng.unit() < self.spread.recovery_rate as f32 {
                    self.agents[i].carrying = None;
                }
            }

            peak_carrying =
                peak_carrying.max(self.agents.iter().filter(|a| a.carrying.is_some()).count());
        }

        let adopted = self.agents.iter().filter(|a| a.ever_adopted).count();
        SwarmOutcome {
            final_adoption: adopted as f64 / n as f64,
            peak: peak_carrying as f64 / n as f64,
            max_generation,
            steps,
            vaccinated_fraction,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::propagation::Population;
    use symthaea_epidemiology::Sir;

    fn aligned_swarm(
        center: &BinaryHV,
        n: usize,
        spread: BeliefSpread,
        mutation: f32,
    ) -> MemeSwarm {
        let beliefs = Population::aligned_to(center, n, 0.08, 4242).beliefs;
        MemeSwarm::new(beliefs, spread, mutation)
    }

    /// A benign, resonant idea in an unvaccinated aligned population spreads to
    /// roughly the SIR final size — a genuine *population* R₀ (which the
    /// single-agent Phase 2 could not provide).
    #[test]
    fn population_final_size_near_sir() {
        let center = BinaryHV::random(21);
        let spread = BeliefSpread {
            susceptibility: 1.0,
            contact_rate: 3.0,
            recovery_rate: 0.5,
        };
        let meme = Meme::seed(0, center.clone(), 0.9);

        // Closed-form prediction from the same adoption model.
        let pop = Population::aligned_to(&center, 3000, 0.08, 4242);
        let sir: Sir = spread.to_sir(&meme, &pop);
        let predicted = sir.final_size();
        assert!(
            predicted > 0.5,
            "test needs a super-critical idea, R∞={predicted}"
        );

        let mut swarm = aligned_swarm(&center, 3000, spread, 0.02);
        let out = swarm.run(&meme, &mut Rng::new(7), 5000);

        // Low mutation keeps variants resonant, so the agent-based final size
        // should land near the closed-form R∞ (Monte-Carlo tolerance).
        assert!(
            (out.final_adoption - predicted).abs() < 0.12,
            "swarm final size {} should track SIR R∞ {}",
            out.final_adoption,
            predicted
        );
    }

    /// Herd immunity for ideas: vaccinating *above* the SIR threshold `1 − 1/R₀`
    /// collapses a pathogen meme's spread; *below* it, the meme still spreads.
    #[test]
    fn herd_immunity_collapses_pathogen_spread() {
        let pathogen_center = BinaryHV::random(88);
        let spread = BeliefSpread {
            susceptibility: 1.0,
            contact_rate: 3.0,
            recovery_rate: 0.5,
        };
        let pathogen = Meme::seed(0, pathogen_center.clone(), 0.9);

        let pop = Population::aligned_to(&pathogen_center, 2500, 0.08, 4242);
        let r0 = spread.to_sir(&pathogen, &pop).basic_reproduction_number();
        let herd = spread.to_sir(&pathogen, &pop).herd_immunity_threshold();
        assert!(r0 > 1.5, "need a contagious pathogen, R₀={r0}");

        // Below threshold: pathogen still reaches a substantial fraction.
        let mut low = aligned_swarm(&pathogen_center, 2500, spread, 0.02);
        let got_low =
            low.vaccinate_fraction(&pathogen_center, (herd * 0.4).min(0.99), &mut Rng::new(1));
        let out_low = low.run(&pathogen, &mut Rng::new(2), 5000);

        // Above threshold: pathogen fizzles.
        let mut high = aligned_swarm(&pathogen_center, 2500, spread, 0.02);
        let got_high =
            high.vaccinate_fraction(&pathogen_center, (herd + 0.15).min(0.99), &mut Rng::new(1));
        let out_high = high.run(&pathogen, &mut Rng::new(2), 5000);

        assert!(
            got_low < got_high,
            "sanity: vaccinated fractions ordered ({got_low} < {got_high})"
        );
        assert!(
            out_high.final_adoption < out_low.final_adoption * 0.5,
            "herd immunity should collapse spread: high-vax adoption {} vs low-vax {} (herd threshold {herd:.2}, R₀ {r0:.2})",
            out_high.final_adoption,
            out_low.final_adoption
        );
    }

    /// Collective immunity: vaccinating even a minority protects unvaccinated
    /// agents too — total adoption drops below the fully-susceptible baseline by
    /// more than just the vaccinated fraction (indirect protection).
    #[test]
    fn partial_vaccination_gives_indirect_protection() {
        let center = BinaryHV::random(55);
        let spread = BeliefSpread {
            susceptibility: 1.0,
            contact_rate: 3.0,
            recovery_rate: 0.5,
        };
        let pathogen = Meme::seed(0, center.clone(), 0.9);

        let mut none = aligned_swarm(&center, 2500, spread, 0.02);
        let baseline = none.run(&pathogen, &mut Rng::new(3), 5000).final_adoption;

        let mut some = aligned_swarm(&center, 2500, spread, 0.02);
        let vax = some.vaccinate_fraction(&center, 0.4, &mut Rng::new(1));
        let protected = some.run(&pathogen, &mut Rng::new(3), 5000).final_adoption;

        // If vaccination only protected the vaccinated, adoption would fall by at
        // most `vax`. Indirect protection means it falls by *more*.
        assert!(
            protected < baseline - vax,
            "expected indirect protection: baseline {baseline}, protected {protected}, vax {vax}"
        );
    }

    #[test]
    fn empty_swarm_is_inert() {
        let mut swarm = MemeSwarm::new(
            vec![],
            BeliefSpread {
                susceptibility: 1.0,
                contact_rate: 3.0,
                recovery_rate: 0.5,
            },
            0.02,
        );
        let out = swarm.run(
            &Meme::seed(0, BinaryHV::random(1), 0.9),
            &mut Rng::new(1),
            100,
        );
        assert_eq!(out.final_adoption, 0.0);
    }
}
