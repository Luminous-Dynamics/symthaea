// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Phase 2c ground-truth test, per `ALIFE_PLAN_2026-07-08.md` §2c.
//!
//! The claim (Maynard Smith & Szathmáry 1995, major transitions in evolution): coalescing
//! should only be favored when the collective genuinely outperforms the sum of individuals
//! acting alone -- not merely be triggered by high mutual permeability and asserted to be
//! beneficial.
//!
//! **Correction from the first draft**: the original negative control mixed prey and predator
//! organisms from `PredatorPreySim`, expecting the "species" mismatch to make pooling fail.
//! It didn't -- traced values showed both species converge to broadly overlapping belief ranges
//! (both use the same generic 2-D observation encoding and homeostatic set-point, just with
//! different real-world referents), so nothing in the *numbers* signaled incompatibility, and
//! `pays_off()` correctly (if surprisingly) said pooling helped. That's informative on its own:
//! the check operates on statistical compatibility of beliefs, not on a "species" label, which
//! is the mathematically honest thing for a Bayesian-fusion-based check to do. The negative
//! control below tests the actual claim instead: organisms with *genuinely numerically divergent*
//! observation histories (sustained low- vs. high-resource environments, not just a different
//! semantic label) should fail to pay off when pooled.

use symthaea_alife::coalition::detect_coalitions;
use symthaea_alife::{Organism, OrganismConfig, Population, PopulationConfig};

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

#[test]
fn same_species_coalition_pays_off() {
    let cfg = population_config();
    let mut population = Population::new(cfg, 4, 21);

    // Run long enough for beliefs to settle into a shared, correlated read on the common
    // resource signal.
    for _ in 0..1000u64 {
        population.step(|n| 3.0 / (n.max(1) as f64));
    }

    // A coalition is a small subgroup, not "the whole population merges into one blob" -- and
    // mechanically, `pool_beliefs` sums precision across members (correct for combining
    // independent estimates), which for a large group makes the KL-complexity term explode
    // roughly with ln(member count) regardless of whether pooling is genuinely beneficial. A
    // realistic small subset avoids that N-scaling artifact rather than papering over it.
    let subset_size = 3.min(population.organisms.len());
    let candidates = detect_coalitions(&population.organisms[..subset_size], 0.3);
    assert!(
        !candidates.is_empty(),
        "expected at least one structural candidate among same-species organisms"
    );

    let paying: Vec<_> = candidates.iter().filter(|c| c.pays_off()).collect();
    assert!(
        !paying.is_empty(),
        "same-species organisms sharing a correlated resource signal should have at least one \
         coalition where pooling genuinely lowers free energy relative to acting alone: {}",
        candidates
            .iter()
            .map(|c| format!(
                "(members={:?} pooled={:.4} sum_individual={:.4})",
                c.member_indices, c.pooled_free_energy, c.sum_of_individual_free_energies
            ))
            .collect::<Vec<_>>()
            .join(", ")
    );
}

#[test]
fn organisms_with_divergent_resource_histories_do_not_pay_off_when_pooled() {
    let cfg = population_config().organism_cfg;

    // One group lives under a sustained scarce resource, the other under a sustained abundant
    // one -- genuinely different observation histories, not just a different label. Both run
    // long enough for belief to actually settle toward their respective environments.
    const TICKS: u64 = 800;
    let scarce_group: Vec<Organism> = (0..2)
        .map(|i| {
            let mut o = Organism::new(cfg, 500 + i);
            for _ in 0..TICKS {
                o.tick(0.05, None);
            }
            o
        })
        .collect();
    let abundant_group: Vec<Organism> = (0..2)
        .map(|i| {
            let mut o = Organism::new(cfg, 900 + i);
            for _ in 0..TICKS {
                o.tick(0.95, None);
            }
            o
        })
        .collect();

    // Sanity check the premise on real, physical energy -- not belief. A prior run found belief
    // barely moves even under these sustained extremes (module docs in `coalition.rs` have the
    // full story); what genuinely diverges, and what `pays_off()` actually keys off, is the
    // real observation history (energy included).
    let scarce_energy: f64 =
        scarce_group.iter().map(|o| o.energy).sum::<f64>() / scarce_group.len() as f64;
    let abundant_energy: f64 =
        abundant_group.iter().map(|o| o.energy).sum::<f64>() / abundant_group.len() as f64;
    assert!(
        abundant_energy - scarce_energy > 0.5,
        "the two groups' real energy should have genuinely diverged, not just carry a different \
         label: scarce_energy={scarce_energy:.4}, abundant_energy={abundant_energy:.4}"
    );

    let mixed_count = scarce_group.len();
    let mut mixed = scarce_group;
    mixed.extend(abundant_group);

    let candidates = detect_coalitions(&mixed, 0.3);
    let paying_across_groups = candidates.iter().any(|c| {
        c.pays_off()
            && c.member_indices.iter().any(|&i| i < mixed_count)
            && c.member_indices.iter().any(|&i| i >= mixed_count)
    });

    assert!(
        !paying_across_groups,
        "pooling organisms whose real observation histories have genuinely diverged (scarce vs. \
         abundant resource) should not pass the real pays-off check, even if permeability \
         happened to cluster them: {}",
        candidates
            .iter()
            .map(|c| format!(
                "(members={:?} pooled={:.4} sum_individual={:.4} pays_off={})",
                c.member_indices,
                c.pooled_free_energy,
                c.sum_of_individual_free_energies,
                c.pays_off()
            ))
            .collect::<Vec<_>>()
            .join(", ")
    );
}
