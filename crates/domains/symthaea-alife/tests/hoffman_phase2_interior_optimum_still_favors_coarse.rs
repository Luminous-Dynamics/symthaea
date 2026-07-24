// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! `HOFFMAN_INTERFACE_THEORY_PLAN_2026-07-22.md` Phase 2's actual result: a positive control
//! attempting to make fine-grained perception win by switching to a non-monotonic
//! (interior-optimum, `spoilage_sigma`) payoff -- too little OR too much true resource both
//! yield low foraging gain, peaked at 0.5. A calibration sweep over 4 `spoilage_sigma` values ×
//! 4 `forage_activity_cost` values (16 combinations, 8 seeds each) found coarse-grained
//! perception net *more* energy than fine-grained in every single combination -- no reversal.
//! This test locks in one representative combination (`sigma=0.15`, `cost=0.02`, both
//! organisms comfortably survive) as a regression check.
//!
//! Root cause isolated separately in
//! `tests/hoffman_action_selection_resource_insensitivity.rs`: `select_action`'s real, non-forced
//! choice doesn't depend on the resource observation at all at a constant true value, spanning
//! the full oscillation range. If the decision never uses the resolved detail regardless of
//! payoff shape, any real cost charged for resolving it is pure waste no matter how the payoff is
//! shaped -- which is exactly what both Phase 1 (monotonic) and this Phase 2 (non-monotonic)
//! found. A genuine positive control would need `symthaea-fep`'s `ActiveInferenceAgent` itself to
//! be made resource-sensitive first -- out of this plan's scope, left as an open question.

use symthaea_alife::{Environment, Organism, OrganismConfig};

fn run_single(grain: Option<f64>, seed: u64, ticks: u64) -> (f64, bool) {
    let cfg = OrganismConfig {
        forage_efficiency: 0.6,
        forage_activity_cost: 0.02,
        perceptual_grain: grain,
        spoilage_sigma: Some(0.15),
        ..OrganismConfig::default()
    };
    let mut organism = Organism::new(cfg, seed);
    let env = Environment::default();
    let mut sum_energy = 0.0;
    let mut count = 0u64;
    let mut died = false;
    for t in 0..ticks {
        let tick = organism.tick(env.resource_at(t), None);
        if t >= ticks / 4 {
            sum_energy += tick.energy;
            count += 1;
        }
        if tick.is_dead {
            died = true;
            break;
        }
    }
    (sum_energy / count.max(1) as f64, died)
}

#[test]
fn interior_optimum_payoff_still_favors_coarse_over_fine() {
    const TICKS: u64 = 3000;
    let (fine_energy, fine_died) = run_single(Some(0.02), 1000, TICKS);
    let (coarse_energy, coarse_died) = run_single(Some(0.4), 2000, TICKS);
    assert!(
        !fine_died && !coarse_died,
        "both should comfortably survive at this calibration"
    );
    assert!(
        coarse_energy > fine_energy,
        "even under a non-monotonic interior-optimum payoff, coarse should still net more \
         energy than fine here: fine={fine_energy}, coarse={coarse_energy}"
    );
}
