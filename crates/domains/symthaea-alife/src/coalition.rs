// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Blankets of blankets, per `ALIFE_PLAN_2026-07-08.md` Phase 2.
//!
//! Reuses `symthaea_fep::markov_blanket::{identify_coalitions, SwarmCoalition}` directly —
//! Friston's (2013) hierarchy (cells→organs→bodies→societies) is already implemented there as
//! permeability-driven agglomerative clustering; this module just feeds it real per-organism
//! data and adds the one thing missing for `symthaea-alife`'s purposes: an actual test of
//! whether coalescing *pays off*, not just whether permeability happens to be high.
//!
//! **On the "phi" field**: `identify_coalitions` takes `(peer_id, phi)` pairs because it was
//! built for the consciousness-swarm context. Per `ALIFE_PLAN_2026-07-08.md`'s Non-goals, Φ and
//! consciousness are explicitly out of scope through Phase 4 -- so the value fed into that slot
//! here is `1 / (1 + free_energy)`, a generic "how well is this organism doing" scalar, **not**
//! a claim about integrated information. It's reused because the struct's numeric slot is
//! generic, not because the number means the same thing it does elsewhere in the crate.
//!
//! **On how "pays off" is actually computed** (revised from a first, more elaborate draft):
//! the obvious approach -- pool member beliefs into a consensus `HiddenState` and run it through
//! `FreeEnergyCalculator::compute` like any other free-energy evaluation -- turned out to be
//! structurally mismatched with the question being asked, for two independent reasons found via
//! traced diagnostics:
//! 1. `compute_complexity`'s KL term measures divergence from the generative model's *fixed*
//!    prior, not "does this belief serve the group." Precision-pooling (the textbook way to
//!    combine independent Gaussian estimates) makes the pooled belief more confident as more
//!    members are added, which — regardless of whether the members actually agree — makes the
//!    KL-vs-fixed-prior term grow, so a same-species, clearly-should-pay-off coalition still
//!    scored *worse* than the sum of individuals.
//! 2. Evaluating per-member against each member's own real observation runs into the opposite
//!    problem: each member's own current belief was *just* moved by `perceive()` specifically to
//!    reduce error against that exact observation, giving it a permanent "freshly fit" advantage
//!    no shared/compromise belief can match — this is true regardless of whether pooling is
//!    actually beneficial, so it structurally favors "never pool," even for correlated members.
//!
//! A traced diagnostic also surfaced a deeper fact worth remembering on its own: even under a
//! *sustained, extreme, constant* resource signal (0.05 vs. 0.95, held for 800 ticks), belief
//! barely moved from its 0.5 default (`0.475` vs. `0.505`) while real physical energy diverged
//! completely (`0.0` vs. `~1.0`) — `ActiveInferenceAgent::update_belief`'s prior-anchoring
//! (already documented as the cause of Phase 0's modest ~14% tracking effect) is strong enough
//! that belief-based comparisons don't discriminate reliably here at all.
//!
//! Given that, `pays_off()` is decided directly and transparently by what actually does
//! discriminate cleanly: agreement among the members' *real* observations (empirical precision =
//! inverse variance of `last_resource_observed`/`energy`) — literally the FEP-relevant quantity
//! that governs whether combining independent measurements is informative in the first place.
//! `pooled_free_energy`/`sum_of_individual_free_energies` are still computed and exposed as
//! real, honest telemetry (not decorative), just not the sole gate.

use symthaea_fep::markov_blanket::{SwarmCoalition, identify_coalitions};
use symthaea_fep::{FreeEnergyCalculator, Observation};

use crate::organism::Organism;

/// Below this empirical observation precision, members disagree too much for pooling to be a
/// genuine information-theoretic win. Set from traced values: same-species members sharing one
/// resource signal show near-zero variance (precision >> 100); organisms with genuinely
/// divergent histories (sustained scarce vs. abundant resource) show variance around 0.45
/// (precision ≈ 2.2). This sits with real margin between the two, not fit to either afterward.
const MIN_OBSERVATION_PRECISION_TO_PAY_OFF: f64 = 10.0;

/// A structurally-identified coalition, with the real economic question already answered.
pub struct Coalition {
    pub swarm: SwarmCoalition,
    /// Indices into the `organisms` slice `detect_coalitions` was called with.
    pub member_indices: Vec<usize>,
    /// Sum of each member's own `current_free_energy()`, acting alone. Real, honest telemetry;
    /// see the module docs for why this isn't `pays_off()`'s sole gate.
    pub sum_of_individual_free_energies: f64,
    /// Free energy of the pooled belief evaluated as one macro-blanket against the members'
    /// averaged real observation. Real, honest telemetry; see the module docs.
    pub pooled_free_energy: f64,
    /// Empirical precision (inverse variance) of the members' real observations -- what
    /// `pays_off()` actually gates on.
    pub observation_precision: f64,
}

impl Coalition {
    /// Major-transitions-theory ground truth (Maynard Smith & Szathmáry 1995): coalescing only
    /// makes sense if the members' real observations agree enough that pooling is genuinely
    /// informative, not just averaging away real disagreement.
    pub fn pays_off(&self) -> bool {
        self.observation_precision > MIN_OBSERVATION_PRECISION_TO_PAY_OFF
    }
}

/// Structurally cluster `organisms` by mutual permeability, then compute the real pays-off
/// check for every candidate cluster found. Returns *all* candidates (paying and not) --
/// callers that only want ones that should actually form use [`detect_paying_coalitions`].
///
/// Mutual permeability between organisms `i` and `j` is the geometric mean of each organism's
/// own (already EMA-smoothed, i.e. already reflecting *stability* over recent ticks, not a
/// momentary spike) effective blanket permeability -- both parties must individually be "open"
/// for the pair to be considered mutually open.
pub fn detect_coalitions(organisms: &[Organism], permeability_threshold: f64) -> Vec<Coalition> {
    let n = organisms.len();
    if n < 2 {
        return Vec::new();
    }

    let peers: Vec<(String, f64)> = organisms
        .iter()
        .enumerate()
        .map(|(i, o)| {
            let wellbeing = 1.0 / (1.0 + o.agent.current_free_energy().max(0.0));
            (i.to_string(), wellbeing)
        })
        .collect();

    let mut pairwise = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            let pi = organisms[i].boundary.permeability().effective;
            let pj = organisms[j].boundary.permeability().effective;
            pairwise.push((i, j, (pi * pj).sqrt()));
        }
    }

    let raw = identify_coalitions(&peers, &pairwise, permeability_threshold);

    raw.into_iter()
        .filter_map(|swarm| build_coalition(organisms, swarm))
        .collect()
}

/// Only the coalitions that should actually form, per Phase 2's ground truth. Recomputed fresh
/// from current organism state every call: nothing here is "sticky," so a coalition that stops
/// paying off simply won't appear next time this is called.
pub fn detect_paying_coalitions(
    organisms: &[Organism],
    permeability_threshold: f64,
) -> Vec<Coalition> {
    detect_coalitions(organisms, permeability_threshold)
        .into_iter()
        .filter(|c| c.pays_off())
        .collect()
}

fn build_coalition(organisms: &[Organism], swarm: SwarmCoalition) -> Option<Coalition> {
    let member_indices: Vec<usize> = swarm
        .members
        .iter()
        .filter_map(|s| s.parse::<usize>().ok())
        .collect();
    if member_indices.len() < 2 {
        return None;
    }

    let sum_of_individual_free_energies: f64 = member_indices
        .iter()
        .map(|&i| organisms[i].agent.current_free_energy())
        .sum();

    let peer_beliefs: Vec<_> = member_indices
        .iter()
        .map(|&i| organisms[i].agent.belief.clone())
        .collect();
    let state_dim = peer_beliefs[0].mean.len();

    // Precision-weighted belief pooling (standard Bayesian sensor fusion of independent
    // Gaussian estimates), not `SwarmCoalition::transduce_gap_junction_alignment` -- that
    // method has its own hardcoded `mean_internal_permeability >= 0.6` gate, independent of
    // whatever `permeability_threshold` a caller passes to `detect_coalitions`, which silently
    // no-ops real, above-threshold-clustered coalitions whose organisms haven't happened to
    // reach that specific number (found via a traced diagnostic: three organisms sitting at a
    // consistent 0.549 effective permeability, well above a 0.3 clustering threshold, produced
    // zero coalitions). Doing the fusion directly keeps coalition formation gated by exactly
    // the one threshold this module exposes.
    let pooled_belief = pool_beliefs(&peer_beliefs, state_dim);
    let pooled_model = organisms[member_indices[0]].agent.model.clone();

    let observations: Vec<[f64; 2]> = member_indices
        .iter()
        .map(|&i| [organisms[i].last_resource_observed, organisms[i].energy])
        .collect();
    let mean_obs = [
        observations.iter().map(|o| o[0]).sum::<f64>() / observations.len() as f64,
        observations.iter().map(|o| o[1]).sum::<f64>() / observations.len() as f64,
    ];
    let variance: f64 = observations
        .iter()
        .map(|o| (o[0] - mean_obs[0]).powi(2) + (o[1] - mean_obs[1]).powi(2))
        .sum::<f64>()
        / observations.len() as f64;
    let observation_precision = 1.0 / (variance + 1e-3);

    let pooled_obs = Observation::new(vec![mean_obs[0], mean_obs[1]], 1.0, "pooled");
    let pooled_free_energy = FreeEnergyCalculator::new(1)
        .compute(&pooled_belief, &pooled_obs, &pooled_model)
        .total;

    Some(Coalition {
        swarm,
        member_indices,
        sum_of_individual_free_energies,
        pooled_free_energy,
        observation_precision,
    })
}

/// Precision-weighted mean per dimension (standard Bayesian fusion of independent Gaussian
/// estimates: `mean = Σ(precision_i * mean_i) / Σ(precision_i)`, pooled precision = `Σ(precision_i)`).
/// A more confident (higher-precision) member's belief counts for more in the consensus.
fn pool_beliefs(
    peer_beliefs: &[symthaea_fep::HiddenState],
    state_dim: usize,
) -> symthaea_fep::HiddenState {
    let mut pooled_mean = vec![0.0; state_dim];
    let mut pooled_precision = vec![0.0; state_dim];

    for belief in peer_beliefs {
        for d in 0..state_dim {
            let precision = belief.precision.get(d).copied().unwrap_or(1.0).max(1e-6);
            pooled_mean[d] += belief.mean.get(d).copied().unwrap_or(0.5) * precision;
            pooled_precision[d] += precision;
        }
    }
    for d in 0..state_dim {
        if pooled_precision[d] > 0.0 {
            pooled_mean[d] /= pooled_precision[d];
        }
    }

    symthaea_fep::HiddenState {
        mean: pooled_mean,
        precision: pooled_precision,
        mode_probs: vec![1.0],
        current_mode: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::environment::Environment;
    use crate::organism::OrganismConfig;

    #[test]
    fn no_coalitions_below_two_organisms() {
        let organisms = vec![Organism::new(OrganismConfig::default(), 1)];
        assert!(detect_coalitions(&organisms, 0.6).is_empty());
    }

    #[test]
    fn similar_organisms_can_form_a_structural_candidate() {
        // Same config, same environment, run in lockstep -- should converge to similar beliefs
        // and (once comfortable) similar high permeability, giving identify_coalitions a real
        // pair to cluster.
        let env = Environment::default();
        let mut organisms: Vec<_> = (0..3)
            .map(|i| Organism::new(OrganismConfig::default(), 100 + i))
            .collect();
        for t in 0..500u64 {
            let resource = env.resource_at(t);
            for o in organisms.iter_mut() {
                o.tick(resource, None);
            }
        }
        // Not asserting pays_off here (that's Phase 2c's dedicated test) -- just that the
        // clustering machinery finds *something* to group under a permissive threshold.
        let candidates = detect_coalitions(&organisms, 0.3);
        assert!(
            !candidates.is_empty(),
            "expected at least one structural candidate"
        );
    }
}
