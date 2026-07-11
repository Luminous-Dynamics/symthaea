// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Phase 3 ground-truth tests, per `ALIFE_PLAN_2026-07-08.md` §3c and its "Done when".
//!
//! 1. `physical_cost_never_below_landauer_minimum` (§3c) — the real physical bound is actually
//!    respected by construction, not just "energy decreases somehow": every tick's charged
//!    `physical_cost` (Landauer + Prigogine) is at least the Landauer term alone, for the exact
//!    bit count and temperature that tick used. This isn't a tautology to game -- it's a
//!    regression guard against a future edit accidentally charging less than the physical floor
//!    (e.g. forgetting to add the Prigogine term, or computing bits with the wrong sign).
//! 2. `never_foraging_dies_while_always_foraging_survives` (Done when) — an organism that never
//!    engages in the only income-generating behavior available to it (forced to Rest every tick)
//!    is on a guaranteed, physical-cost-accelerated decline and dies, while one that does forage
//!    survives indefinitely under the same environment. Energy accounting -- including Phase 3's
//!    new physical_cost term -- must be doing real work for this to hold: if it were cosmetic
//!    telemetry (the `feedback_admittance_control` bug class), a permanently-resting organism
//!    wouldn't actually decline.
//!
//! **Two real findings shaped this file's final form, not tuning -- both kept here rather than
//! silently discarded:**
//! - A first draft compared "always forage" against the organism's own `select_action()` under
//!   `Environment::default()`. Result: the *opposite* of the hypothesis -- always-forage
//!   survived every seed to the tick cap, while `select_action()` died faster. Not a test bug;
//!   consistent with Phase 0's own honest finding that `select_action()` isn't reliably better
//!   than simpler baselines.
//! - A second draft compared "always forage" against a "forage only while below the homeostatic
//!   set-point, else rest" heuristic, at a resource share calibrated (bisection, same method as
//!   Phase 1a) to just below always-forage's own breakeven. Also the *opposite* of the
//!   hypothesis, and for a genuinely informative reason: at that exact calibrated point,
//!   always-forage's own zero-drift condition means each *forage* tick nets ~0 on average --
//!   while a *rest* tick is unconditionally negative (`metabolic_cost`/`physical_cost` apply
//!   regardless of action; only foraging can offset them). So near that specific point, resting
//!   at all can only make things worse, not better -- reducing forage duty-cycle is not
//!   generically "smarter," it depends delicately on where the resource level sits relative to
//!   breakeven. Rather than keep hand-tuning the scarcity level chasing that effect, this file
//!   uses the comparison that's true unconditionally: never engaging in the only
//!   income-generating action guarantees decline; sometimes doing so does not.

use symthaea_alife::{Action, Environment, Organism, OrganismConfig};

#[test]
fn physical_cost_never_below_landauer_minimum() {
    let mut organism = Organism::new(OrganismConfig::default(), 7);
    let env = Environment::default();

    for t in 0..1000u64 {
        let tick = organism.tick(env.resource_at(t), None);
        let landauer_only = symthaea_alife::landauer_minimum(
            tick.bits_processed,
            organism.cfg.effective_temperature,
        );
        assert!(
            tick.physical_cost >= landauer_only - 1e-12,
            "t={t}: physical_cost={:.6} was below the Landauer minimum={:.6} for \
             bits_processed={:.4} -- the physical floor was not actually respected",
            tick.physical_cost,
            landauer_only,
            tick.bits_processed
        );
    }
}

#[test]
fn never_foraging_dies_while_always_foraging_survives() {
    const SEEDS: &[u64] = &[1, 2, 3, 4, 5];
    const MAX_TICKS: u64 = 5_000;

    for &seed in SEEDS {
        let cfg = OrganismConfig::default();
        let env = Environment::default();

        let mut never_forages = Organism::new(cfg, seed);
        let mut always_forages = Organism::new(cfg, seed);

        let mut never_forages_died = false;

        for t in 0..MAX_TICKS {
            let resource = env.resource_at(t);
            let rest_tick = never_forages.tick(resource, Some(Action::Rest.index()));
            let forage_tick = always_forages.tick(resource, Some(Action::Forage.index()));

            if rest_tick.is_dead {
                never_forages_died = true;
            }
            assert!(
                !forage_tick.is_dead,
                "seed={seed}: an organism that forages every tick should survive under \
                 Environment::default(), but died at t={t}"
            );
        }

        assert!(
            never_forages_died,
            "seed={seed}: an organism that never forages (guaranteed zero income, but still \
             pays metabolic_cost and Phase 3's physical_cost every tick) should have died \
             within {MAX_TICKS} ticks -- if it didn't, energy accounting isn't doing real work"
        );
    }
}
