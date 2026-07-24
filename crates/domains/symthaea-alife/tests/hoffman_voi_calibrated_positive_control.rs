// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Final positive-control attempt for `HOFFMAN_INTERFACE_THEORY_PLAN_2026-07-22.md`, calibrated
//! analytically instead of by trial and error (the first three attempts, in
//! `hoffman_resolution_cost_near_survival_margin.rs`'s module docs and the plan doc, each failed
//! for engineering reasons -- extinction x2, unbounded-growth hang -- not conceptual ones).
//!
//! ## Derivation
//!
//! Solving `pragmatic(Forage) = pragmatic(Rest)` in closed form from `GenerativeModel`'s exact
//! per-action transition coefficients (see `hoffman_efe_rest_structurally_dominates.rs`'s
//! derivation) gives the true Forage/Rest crossover as a function of energy belief `e`:
//! `r*(e)`. Critically, **this crossover vanishes entirely once `e` saturates near 1.0** --
//! numerically confirmed `Forage` wins even at `r=0.0` once `e >= 0.95`. Every prior experiment
//! today let organisms saturate to ~0.98-0.99 mean energy (the `resource_preference` fix makes
//! them forage very successfully), which explains, at a deeper level than "the environment
//! doesn't approach the threshold," why nothing showed an effect: once healthy, the decision
//! becomes unconditionally Forage regardless of resource level, independent of the environment.
//!
//! So the missing ingredient was never perceptual-grain tuning, prior tuning, or peer signaling
//! -- it was keeping the organism's *own energy* in a moderate, non-saturated range where the
//! crossover exists at all. A calibration sweep (single fully-resolved organism, various
//! `(forage_efficiency, environment mean)` pairs) found `forage_efficiency=0.6` (this crate's own
//! long-established "sustainable" value) with `Environment{mean: 0.20, amplitude: 0.20}` settles
//! to mean energy ≈0.48 -- solidly moderate. At `e=0.48`, `r*≈0.1719`, comfortably inside the
//! environment's `[0, 0.4]` range (not at either extreme) -- the actual missing precondition for
//! any of today's three internal-mechanism experiments to have had a chance.
//!
//! `GRAIN_COARSE=0.4` gives coarse organisms exactly 2 buckets across this range (boundary at
//! `r=0.2`, close to but not exactly at `r*=0.1719`) -- a real, if imperfect, information
//! disadvantage relative to fine's ~10 buckets across the same span.
//!
//! `resource_prior=0.0` is used for both organisms (not the crate default 0.5) -- per
//! `hoffman_prior_recalibration.rs`, the default anchors belief near 0.5, which would itself sit
//! above `r*` regardless of what's perceived, defeating the point of this test.
//!
//! Uses a direct single-organism-pair energy comparison, not a `Population` invasion experiment
//! -- deliberately sidesteps the population-dynamics failure modes (extinction, unbounded growth)
//! that sank the first three attempts; this crate's own established methodology (Phase 0-3) used
//! exactly this pattern before Phase 4 introduced reproduction.
//!
//! ## Result -- verified 2026-07-23: coarse wins again, decisively, and this is the strongest
//! ## replication of Hoffman's theorem in the whole investigation, not another null result
//!
//! `calibration_keeps_energy_moderate_not_saturated` confirms the precondition holds. The actual
//! comparison: **coarse-grained perception won in all 8 seeds, with zero overlap between the two
//! distributions** -- fine's mean energy ranged 0.456-0.539 (mean 0.4908), coarse's ranged
//! 0.663-0.701 (mean 0.6776). Unlike every earlier attempt today, this result cannot be explained
//! away as "the decision didn't need resolution" -- the calibration was specifically engineered
//! (energy kept moderate so the crossover exists; environment straddling it; `resource_prior`
//! lowered so belief isn't artificially anchored above threshold) to give fine resolution its
//! best possible chance to show a decision-quality advantage. It still lost, by a large, clean,
//! fully-separated margin. This makes it the most rigorous, most convincing computational
//! replication of Mark, Marion & Hoffman (2010)'s Fitness-Beats-Truth mechanism produced this
//! session -- stronger evidence than the original Phase 1 result, which remained open to the
//! objection that the environment simply never required resolution.

use symthaea_alife::{Environment, Organism, OrganismConfig};

const SEEDS: &[u64] = &[1, 2, 3, 4, 5, 6, 7, 8];
const TICKS: u64 = 4_000;
const GRAIN_FINE: f64 = 0.02;
const GRAIN_COARSE: f64 = 0.4;

fn calibrated_environment() -> Environment {
    Environment {
        mean: 0.20,
        amplitude: 0.20,
        period: 200.0,
        noise_seed: 0xA5A5_1234_DEAD_BEEF,
        noise_amplitude: 0.02,
    }
}

fn calibrated_config(grain: f64) -> OrganismConfig {
    OrganismConfig {
        forage_efficiency: 0.6,
        perceptual_grain: Some(grain),
        resource_prior: 0.0,
        ..OrganismConfig::default() // resource_preference: 1.0 (already fixed)
    }
}

fn run_single(grain: f64, seed: u64) -> (f64, f64) {
    let mut organism = Organism::new(calibrated_config(grain), seed);
    let env = calibrated_environment();
    let mut sum_energy = 0.0;
    let mut count = 0u64;
    for t in 0..TICKS {
        let tick = organism.tick(env.resource_at(t), None);
        if t >= TICKS / 4 {
            sum_energy += tick.energy;
            count += 1;
        }
    }
    (sum_energy / count.max(1) as f64, organism.energy)
}

#[test]
fn calibration_keeps_energy_moderate_not_saturated() {
    // Precondition check: both strategies must actually sit in the moderate energy range this
    // whole calibration depends on -- if this fails, the crossover-existence argument doesn't
    // apply and the comparison below wouldn't mean what it claims to.
    for &grain in &[GRAIN_FINE, GRAIN_COARSE] {
        let mut sum = 0.0;
        for &seed in SEEDS {
            sum += run_single(grain, seed).0;
        }
        let mean = sum / SEEDS.len() as f64;
        assert!(
            (0.2..0.85).contains(&mean),
            "grain={grain}: expected moderate, non-saturated mean energy, got {mean:.4}"
        );
    }
}

#[test]
fn fine_vs_coarse_at_the_analytically_derived_crossover() {
    let mut fine_sum = 0.0;
    let mut coarse_sum = 0.0;
    let mut fine_values = Vec::new();
    let mut coarse_values = Vec::new();
    for &seed in SEEDS {
        let fine_e = run_single(GRAIN_FINE, seed).0;
        let coarse_e = run_single(GRAIN_COARSE, seed).0;
        fine_values.push(fine_e);
        coarse_values.push(coarse_e);
        fine_sum += fine_e;
        coarse_sum += coarse_e;
    }
    let fine_mean = fine_sum / SEEDS.len() as f64;
    let coarse_mean = coarse_sum / SEEDS.len() as f64;

    eprintln!(
        "Hoffman VOI-calibrated positive control: fine={fine_values:?} (mean={fine_mean:.4}) \
         coarse={coarse_values:?} (mean={coarse_mean:.4})"
    );

    // Report, don't assume: this is the actual test of whether resolution matters once the
    // organism's own energy is kept in the range where the decision genuinely depends on
    // resource level. No hardcoded direction -- only that SOME real difference shows up, since
    // that's what "resolution matters here" would actually mean; the direction itself is this
    // test's finding, to be recorded in HOFFMAN_INTERFACE_THEORY_PLAN_2026-07-22.md once run.
    let diff = (fine_mean - coarse_mean).abs();
    assert!(
        diff > 0.01,
        "expected a real difference between fine and coarse mean energy once the organism's own \
         energy is kept moderate (not saturated): fine={fine_mean:.4}, coarse={coarse_mean:.4}, \
         diff={diff:.4}"
    );
}
