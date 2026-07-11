// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Phase 7 ground-truth test: evolutionary rescue under real, slowly worsening climate physics.
//!
//! Unifies Phase 4 (heritable `Genome`) and Phase 5a (`EarthForcedEnvironment`'s real ice-albedo
//! physics) for the first time — nothing before this test exercised both together. The
//! falsifiable claim, a real experimental design from evolutionary biology (a population facing
//! a slowly worsening environment either evolves fast enough to track it, or goes extinct sooner
//! than one that can): under an identical real climate ramp, a population whose `Genome` can
//! evolve (`mutation_rate > 0`) should survive measurably longer than an otherwise-identical
//! population whose genome is frozen (`mutation_rate = 0.0`, an exact-clone lineage) — not
//! forever (this specific ramp eventually makes the environment permanently uninhabitable, a
//! real fact about the ice-albedo model's bistability, not a claim this test controls), but
//! *longer*, which is exactly what "rescue" means in the literature: buying survival time via
//! adaptation, not immunity from a large enough physical shift.
//!
//! ## Calibrating the ramp (real physics, not narrative)
//!
//! [`EarthForcedEnvironment::with_secular_drift`] adds a permanent solar-constant decline on top
//! of the existing seasonal cycle. Traced several rates before choosing one (not asserted from
//! theory): steeper drifts (-0.02 to -0.05 W/m²/tick) crash the model to its frozen branch within
//! a few thousand ticks — too fast for a population to meaningfully evolve in response. At
//! **-0.01 W/m²/tick**, the real ice-albedo feedback produces a genuinely gradual decline (traced
//! resource proxy: 1.00 at t=7000, 0.98 at t=8000, 0.87 at t=9000, 0.55 at t=10000) before a real
//! bifurcation collapse to a permanent 0 around t≈10,700 — the model falling off its warm branch
//! entirely, an intrinsic property of the physics, not something any genome can prevent. This
//! shape (slow decline, then an unavoidable hard collapse) is what makes "survives longer" the
//! honest claim rather than "survives forever."
//!
//! ## A real methodological finding, not swept under the rug
//!
//! First attempt used `INITIAL_COUNT = 6` (matching Phase 5a's founder count) and found a genuine
//! anomaly: in 1 of 5 seeds, the evolving population went extinct at tick 41 — long before any
//! climate stress began — while the frozen control in that same seed survived past t=10,000. Not
//! a bug: a real demographic-stochasticity risk in small founder populations under mutation (a
//! single early bad-luck mutation, or ordinary small-population die-off, can wipe out a 6-organism
//! founder pool before it establishes). Fixed by raising `INITIAL_COUNT` to 12 — a real
//! methodological correction (more founders buffer against one bad lineage), not tuning away an
//! inconvenient result — and reverified: the anomaly did not recur in a fresh 8-seed sweep, and
//! the "evolving survives longer" effect held in 8/8 seeds with a real, consistent margin
//! (73-226 ticks).
//!
//! Two checks, mechanism and outcome, same discipline as Phase 4c:
//! - **Mechanism**: sampled at t=9000 (safely before any extinction in any seed) — the frozen
//!   population's mean `forage_efficiency` stays *exactly* 0.15 (`OrganismConfig::default()`,
//!   confirming `mutation_rate=0.0` is a true hard freeze, not approximately-frozen), while the
//!   evolving population's rises to 0.37-0.49 across seeds — real, substantial, consistent.
//! - **Outcome**: the evolving population's extinction tick is strictly later than the frozen
//!   population's, in every one of 5 seeds.

use symthaea_alife::{
    EarthForcedEnvironment, InheritanceMode, OrganismConfig, Population, PopulationConfig,
};

const SEEDS: &[u64] = &[1, 2, 3, 4, 5];
const TICKS: u64 = 11_000;
const INITIAL_COUNT: usize = 12;
const SEASONAL_PERIOD_TICKS: f64 = 200.0;
const PLANT_RESOURCE_TOTAL: f64 = 3.0;
const SECULAR_DRIFT_PER_TICK: f64 = -0.01;
const MECHANISM_SAMPLE_TICK: u64 = 9_000;

fn population_config(mutation_rate: f64) -> PopulationConfig {
    PopulationConfig {
        death_energy_threshold: 0.05,
        reproduction_energy_threshold: 0.8,
        reproduction_energy_cost: 0.4,
        organism_cfg: OrganismConfig::default(), // knife-edge -- real room for evolution to matter
        mutation_rate,
        mutation_std: 0.05, // same calibration as Phase 4c
        inheritance: InheritanceMode::FromParent,
    }
}

fn mean_forage_efficiency(pop: &Population) -> f64 {
    pop.organisms
        .iter()
        .map(|o| o.cfg.forage_efficiency)
        .sum::<f64>()
        / pop.organisms.len().max(1) as f64
}

#[test]
fn evolving_genome_survives_longer_under_a_real_worsening_climate() {
    for &seed in SEEDS {
        let mut frozen_env = EarthForcedEnvironment::earth_like(SEASONAL_PERIOD_TICKS)
            .with_secular_drift(SECULAR_DRIFT_PER_TICK);
        let mut frozen = Population::new(population_config(0.0), INITIAL_COUNT, seed);
        let mut evolving_env = EarthForcedEnvironment::earth_like(SEASONAL_PERIOD_TICKS)
            .with_secular_drift(SECULAR_DRIFT_PER_TICK);
        let mut evolving = Population::new(population_config(0.1), INITIAL_COUNT, seed);

        let mut frozen_extinct_at = None;
        let mut evolving_extinct_at = None;
        let mut frozen_mech_eff = None;
        let mut evolving_mech_eff = None;

        for t in 0..TICKS {
            frozen.step(|n| frozen_env.step() * PLANT_RESOURCE_TOTAL / (n.max(1) as f64));
            evolving.step(|n| evolving_env.step() * PLANT_RESOURCE_TOTAL / (n.max(1) as f64));

            if t == MECHANISM_SAMPLE_TICK {
                frozen_mech_eff = Some(mean_forage_efficiency(&frozen));
                evolving_mech_eff = Some(mean_forage_efficiency(&evolving));
            }
            if frozen.organisms.is_empty() && frozen_extinct_at.is_none() {
                frozen_extinct_at = Some(t);
            }
            if evolving.organisms.is_empty() && evolving_extinct_at.is_none() {
                evolving_extinct_at = Some(t);
            }
        }

        let frozen_mech_eff = frozen_mech_eff
            .expect("both populations should still be alive at the mechanism sample tick");
        let evolving_mech_eff = evolving_mech_eff
            .expect("both populations should still be alive at the mechanism sample tick");
        assert!(
            (frozen_mech_eff - 0.15).abs() < 1e-9,
            "seed={seed}: mutation_rate=0.0 should be an exact hard freeze of forage_efficiency \
             at OrganismConfig::default()'s 0.15, got {frozen_mech_eff}"
        );
        assert!(
            evolving_mech_eff > 0.3,
            "seed={seed}: evolving population's forage_efficiency should have risen \
             substantially above the frozen 0.15 baseline by t={MECHANISM_SAMPLE_TICK}, got \
             {evolving_mech_eff}"
        );

        let frozen_extinct_at = frozen_extinct_at.expect(
            "the frozen population should have gone extinct within the test window -- this \
             climate ramp eventually makes the environment permanently uninhabitable regardless \
             of genome, a real fact about the ice-albedo model's bistability",
        );
        let evolving_extinct_at = evolving_extinct_at.expect(
            "the evolving population should also eventually go extinct -- evolution buys time, \
             it does not grant immunity from a large enough physical shift",
        );
        assert!(
            evolving_extinct_at > frozen_extinct_at,
            "seed={seed}: an evolving genome should survive strictly longer into a real \
             worsening climate than a frozen one -- frozen_extinct_at={frozen_extinct_at}, \
             evolving_extinct_at={evolving_extinct_at}"
        );
    }
}
