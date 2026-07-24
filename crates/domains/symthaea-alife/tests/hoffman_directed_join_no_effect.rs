// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Tests a real, if simple, operationalization of Hoffman & Prakash (2014)'s Directed Join
//! composition theorem (`D = D1·A1·D2` -- one conscious agent's action `A1` becomes information
//! feeding another agent's decision process `D2`), per
//! `HOFFMAN_INTERFACE_THEORY_PLAN_2026-07-22.md`'s further-work suggestion. Deliberately distinct
//! from `coalition.rs`'s existing machinery, which does Bayesian precision-weighted belief
//! *pooling* -- a different operation than Hoffman's actual sequential action-to-decision
//! composition; conflating the two would misrepresent both.
//!
//! Mechanism: organism B's belief is nudged toward organism A's last chosen action (Forage -> nudge
//! toward 1.0, Rest -> nudge toward 0.0) before B's own tick, by directly writing
//! `agent.belief.mean` -- a public field, so this needed zero changes to `organism.rs`.
//!
//! Hypothesis: if B's own perceptual resolution is bad, a directed signal from a better-resolved
//! peer A should let B partially compensate, the same way real social/observational learning can
//! substitute for direct perception.
//!
//! **Result: no measurable effect, at any signal weight tested (0.0 through 0.8), even when B's
//! own perception is made maximally useless** (`perceptual_grain=Some(2.0)`, which
//! `quantize_to_grain` collapses every possible true resource value in `[0,1]` to a single
//! constant bucket, `0.0`). Root cause, confirmed by directly tracing belief:
//! `GenerativeModel::new`'s **fixed** `prior_mean = 0.5`, combined with `update_belief`'s
//! `prior_grad` pull toward it, keeps belief anchored around 0.47-0.52 regardless of what's
//! actually perceived -- even a constantly-zero observation never drags belief down toward the
//! crossover (~0.075-0.1, see `resource_preference`'s doc comment). Since 0.5 sits comfortably
//! above that threshold, the organism forages successfully anyway, with or without any peer
//! information. Perception quality (individual OR social/directed) is nearly irrelevant to this
//! specific decision for a *third*, independent reason found this session: Phase 1/2 found the
//! environment doesn't need resolution; the earlier investigation found the goal preference was
//! miscalibrated (now fixed); this finds a *second* hardcoded, uncalibrated constant
//! (`prior_mean`) accidentally makes the corrected system robust to bad perception by anchoring
//! belief near a value that's already on the right side of the decision threshold. Not
//! independently fixed here -- `prior_mean` is used throughout `GenerativeModel` for both state
//! dimensions and touching it is a larger, separately-scoped change.

use symthaea_alife::{Action, Environment, Organism, OrganismConfig};

fn nudge_belief_from_peer_action(organism: &mut Organism, peer_action: usize, weight: f64) {
    let target = if peer_action == Action::Forage.index() {
        1.0
    } else {
        0.0
    };
    let current = organism.agent.belief.mean[0];
    organism.agent.belief.mean[0] = current + weight * (target - current);
}

fn run_pair(peer_signal_weight: f64, seed: u64, ticks: u64) -> f64 {
    let cfg_a = OrganismConfig {
        forage_efficiency: 0.6,
        perceptual_grain: Some(0.02), // fine
        ..OrganismConfig::default()
    };
    let cfg_b = OrganismConfig {
        forage_efficiency: 0.6,
        // Extreme: quantize_to_grain(r, 2.0) rounds any r in [0,1] to a single constant bucket
        // (0.0) -- B's own perception is genuinely useless, not just lossy, giving a peer signal
        // maximum room to matter if it's going to matter at all.
        perceptual_grain: Some(2.0),
        ..OrganismConfig::default()
    };
    let mut a = Organism::new(cfg_a, seed);
    let mut b = Organism::new(cfg_b, seed.wrapping_add(1_000_000));
    let env = Environment::default();

    let mut sum_energy_b = 0.0;
    let mut count = 0u64;
    for t in 0..ticks {
        let r = env.resource_at(t);
        let a_tick = a.tick(r, None);
        if peer_signal_weight > 0.0 {
            nudge_belief_from_peer_action(&mut b, a_tick.action, peer_signal_weight);
        }
        let b_tick = b.tick(r, None);
        if t >= ticks / 4 {
            sum_energy_b += b_tick.energy;
            count += 1;
        }
    }
    sum_energy_b / count.max(1) as f64
}

#[test]
fn peer_signal_from_fine_grained_peer_makes_no_measurable_difference() {
    const SEEDS: u64 = 8;
    const TICKS: u64 = 3000;
    let mut baseline_sum = 0.0;
    let mut signaled_sum = 0.0;
    for s in 0..SEEDS {
        baseline_sum += run_pair(0.0, 1000 + s, TICKS);
        signaled_sum += run_pair(0.5, 1000 + s, TICKS);
    }
    let baseline_mean = baseline_sum / SEEDS as f64;
    let signaled_mean = signaled_sum / SEEDS as f64;
    assert!(
        (signaled_mean - baseline_mean).abs() < 0.01,
        "expected no measurable effect from the peer signal (both should already be near-\
         ceiling due to prior-anchoring): baseline={baseline_mean:.4}, signaled={signaled_mean:.4}"
    );
    // Both should be thriving despite B's maximally-broken perception -- the actual finding.
    assert!(
        baseline_mean > 0.9,
        "expected B to thrive even with useless perception (prior-anchoring keeps belief above \
         the decision threshold regardless): baseline={baseline_mean:.4}"
    );
}

#[test]
fn belief_stays_anchored_near_the_fixed_prior_regardless_of_perceived_garbage() {
    // Direct mechanism check: even under a perpetually-zero perceived observation
    // (perceptual_grain=2.0 collapses everything to bucket 0.0), belief never drifts down toward
    // that value -- it stays anchored near GenerativeModel::new's fixed prior_mean=0.5.
    let cfg = OrganismConfig {
        forage_efficiency: 0.6,
        perceptual_grain: Some(2.0),
        ..OrganismConfig::default()
    };
    let mut organism = Organism::new(cfg, 42);
    let env = Environment::default();
    for t in 0..500u64 {
        organism.tick(env.resource_at(t), None);
    }
    let belief_resource = organism.agent.belief.mean[0];
    assert!(
        (0.4..=0.6).contains(&belief_resource),
        "expected belief to stay anchored near the fixed prior 0.5 despite constantly \
         perceiving a quantized-to-zero observation, got {belief_resource}"
    );
}
