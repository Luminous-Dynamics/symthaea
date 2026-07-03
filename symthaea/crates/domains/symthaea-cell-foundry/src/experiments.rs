// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Equifinality experiments: the faithful, reusable test of basal cognition.
//!
//! Levin's signature empirical signature of collective intelligence is
//! *equifinality* — a system reaching the same target state from different
//! starting perturbations, which is only explicable if something in the
//! system is actively computing toward a stored goal rather than just
//! executing fixed local rules. This module runs that exact test on the
//! bioelectric layer ([`crate::bioelectric`]): build an organoid with a
//! genuine spatial Vmem prepattern, capture it as a target, damage it in
//! several qualitatively different ways, and measure whether recovery
//! converges back toward the *same* target — and whether that convergence
//! specifically requires open gap junctions (the causal claim, not just
//! "damage heals over time").
//!
//! ## Why the template is built with `impose_vmem_pattern`, not organic
//! differentiation
//!
//! The Turing chemical layer in [`crate::morphogenetic_consciousness`] only
//! rarely drives cells past the neural/glial differentiation thresholds at
//! the cell densities used in this crate's tests (a pre-existing property of
//! that layer, unrelated to the bioelectric addition). Waiting on that to
//! produce spatial heterogeneity to regenerate would make these experiments
//! flaky and slow to validate. Instead, [`build_radial_bipolar_template`]
//! directly imposes a two-region Vmem prepattern (inner depolarized, outer
//! hyperpolarized) via [`crate::bioelectric::MorphogeneticField::impose_vmem_pattern`]
//! — this is not a shortcut so much as a faithful analog of how Levin's lab
//! actually studies this: bioelectric prepatterns are routinely imposed
//! experimentally (ionophores, optogenetics) independent of any
//! transcriptional change, and are established as *preceding and
//! instructing* subsequent differentiation, not the other way around.
//!
//! ## Why recovery works via neighbour-template propagation, and where that
//! breaks down
//!
//! In this model, a wound-boundary progenitor's Vmem drifts toward its own
//! (position-independent) fate-appropriate resting value regardless of gap
//! junction state — see `BioelectricState` in `crate::bioelectric`. What
//! *open* gap junctions add is fast diffusion from spatial neighbours
//! (`GAP_JUNCTION_DIFFUSION_RATE` = 0.35/day) that dominates that slow drift
//! (`HYPERPOLARIZATION_DRIFT` = 0.05/day) — so new tissue growing at a wound
//! edge adjacent to *surviving* target-consistent tissue gets pulled toward
//! the correct pattern by its neighbours, while blocked tissue just drifts
//! to its generic default. This means recovery in this model depends on
//! adjacent surviving tissue acting as a template — which is itself
//! faithful to Levin's actual mechanism (gap junctions let a cell "read" its
//! neighbours' accumulated state) — but it also means a perturbation that
//! destroys the *entire* pattern with no surviving template
//! ([`Perturbation::ScrambleVmem`]) is **not** expected to recover under
//! this model. That's an honest limitation, not a hidden failure: a full
//! positional-identity system (each cell independently "knowing" its target
//! Vmem from a French-flag-style morphogen readout) would be a legitimate
//! future extension. `run_equifinality_experiment` runs `ScrambleVmem`
//! anyway so this boundary is visible in results rather than swept away.
//!
//! References: see `crate::bioelectric` module docs.

use serde::{Deserialize, Serialize};

use crate::bioelectric::{VMEM_DEPOLARIZED, VMEM_HYPERPOLARIZED};
use crate::morphogenetic_consciousness::NeuralOrganoid;

/// A perturbation applied to a matured/patterned organoid to test recovery.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Perturbation {
    /// Cut away all cells with radius in `[min_r, max_r)`.
    Amputate { min_r: f32, max_r: f32 },
    /// Randomize Vmem across all surviving cells without removing tissue —
    /// see module docs for why this is expected to *not* recover under this
    /// model (no surviving template).
    ScrambleVmem { seed: u64 },
}

impl Perturbation {
    pub fn label(&self) -> String {
        match self {
            Perturbation::Amputate { min_r, max_r } => {
                format!("amputate[{min_r:.2},{max_r:.2})")
            }
            Perturbation::ScrambleVmem { seed } => format!("scramble_vmem(seed={seed})"),
        }
    }

    fn apply(&self, organoid: &mut NeuralOrganoid) {
        match *self {
            Perturbation::Amputate { min_r, max_r } => {
                organoid.amputate(min_r, max_r);
            }
            Perturbation::ScrambleVmem { seed } => {
                organoid.scramble_vmem(seed);
            }
        }
    }
}

/// One (perturbation, gap-junction-permeability) condition's trajectory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConditionResult {
    pub perturbation_label: String,
    pub gap_junction_permeability: f32,
    /// (day, discrepancy-from-target) pairs, recorded every recovery day.
    pub discrepancy_trajectory: Vec<(u32, f64)>,
    pub final_discrepancy: f64,
}

/// Full equifinality experiment result across all perturbations, each run
/// under both open (`permeability = 1.0`) and blocked (`permeability =
/// 0.0`) gap junctions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquifinalityResult {
    pub conditions: Vec<ConditionResult>,
}

impl EquifinalityResult {
    /// Mean final discrepancy across all open-permeability conditions and
    /// all blocked-permeability conditions, respectively.
    pub fn mean_final_by_permeability(&self) -> (f64, f64) {
        let mean = |v: &[f64]| {
            if v.is_empty() {
                0.0
            } else {
                v.iter().sum::<f64>() / v.len() as f64
            }
        };
        let open: Vec<f64> = self
            .conditions
            .iter()
            .filter(|c| c.gap_junction_permeability > 0.0)
            .map(|c| c.final_discrepancy)
            .collect();
        let blocked: Vec<f64> = self
            .conditions
            .iter()
            .filter(|c| c.gap_junction_permeability == 0.0)
            .map(|c| c.final_discrepancy)
            .collect();
        (mean(&open), mean(&blocked))
    }

    /// The core equifinality claim: across all perturbations, open gap
    /// junctions recover a lower mean final discrepancy than blocked ones —
    /// i.e. recovery toward the shared target specifically requires
    /// bioelectric coupling, not just "damage heals over time."
    pub fn open_beats_blocked(&self) -> bool {
        let (open_mean, blocked_mean) = self.mean_final_by_permeability();
        open_mean < blocked_mean
    }

    /// Spread (max - min) of final discrepancy among open-permeability
    /// conditions — small spread despite different starting perturbations
    /// is the equifinality signature proper (not just "open is better," but
    /// "open converges to the *same place* regardless of how it got hurt").
    pub fn open_run_spread(&self) -> f64 {
        Self::spread(
            self.conditions
                .iter()
                .filter(|c| c.gap_junction_permeability > 0.0)
                .map(|c| c.final_discrepancy),
        )
    }

    /// Same spread, for blocked-permeability conditions.
    pub fn blocked_run_spread(&self) -> f64 {
        Self::spread(
            self.conditions
                .iter()
                .filter(|c| c.gap_junction_permeability == 0.0)
                .map(|c| c.final_discrepancy),
        )
    }

    fn spread(values: impl Iterator<Item = f64>) -> f64 {
        let (min, max) = values.fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), v| {
            (min.min(v), max.max(v))
        });
        if min.is_finite() && max.is_finite() {
            max - min
        } else {
            0.0
        }
    }
}

/// Build a "mature" organoid with an imposed bipolar radial Vmem
/// prepattern: cells with radius >= `boundary_r` are hyperpolarized
/// (the "differentiated ring"), cells within stay depolarized (the "stem
/// core"). Captures this pattern as the organoid's target morphology. See
/// module docs for why the pattern is imposed rather than grown organically.
pub fn build_radial_bipolar_template(
    seed: u64,
    cells: usize,
    maturation_days: u32,
    boundary_r: f32,
) -> NeuralOrganoid {
    let mut organoid = NeuralOrganoid::new(cells, seed);
    for _ in 0..maturation_days {
        organoid.advance_day();
    }
    organoid.impose_vmem_pattern(|p| {
        let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        if r >= boundary_r {
            VMEM_HYPERPOLARIZED
        } else {
            VMEM_DEPOLARIZED
        }
    });
    organoid.capture_target_morphology();
    organoid
}

/// Run the equifinality experiment: for each perturbation in
/// `perturbations`, clone `template` (which must already have a captured
/// target morphology — see [`build_radial_bipolar_template`]), run it under
/// both open and fully-blocked gap-junction permeability for
/// `recovery_days`, and record discrepancy-from-target each day.
///
/// Reusable for other experiment designs: build any template with a
/// captured target (via `capture_target_morphology`, possibly after
/// `impose_vmem_pattern`), define any set of `Perturbation`s, and this
/// function runs the full open-vs-blocked x multi-perturbation matrix.
pub fn run_equifinality_experiment(
    template: &NeuralOrganoid,
    perturbations: &[Perturbation],
    recovery_days: u32,
) -> EquifinalityResult {
    let mut conditions = Vec::new();
    for perturbation in perturbations {
        for &permeability in &[1.0f32, 0.0f32] {
            let mut organoid = template.clone();
            organoid.set_gap_junction_permeability(permeability);
            perturbation.apply(&mut organoid);

            let mut trajectory = Vec::with_capacity(recovery_days as usize);
            for day in 1..=recovery_days {
                organoid.advance_day();
                let d = organoid.morphology_discrepancy().unwrap_or(0.0);
                trajectory.push((day, d));
            }
            let final_discrepancy = trajectory.last().map(|&(_, d)| d).unwrap_or(0.0);

            conditions.push(ConditionResult {
                perturbation_label: perturbation.label(),
                gap_junction_permeability: permeability,
                discrepancy_trajectory: trajectory,
                final_discrepancy,
            });
        }
    }
    EquifinalityResult { conditions }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_gap_junctions_recover_imposed_bipolar_pattern_better_than_blocked() {
        let template = build_radial_bipolar_template(7, 200, 20, 0.2);
        let perturbations = [Perturbation::Amputate {
            min_r: 0.8,
            max_r: 2.0,
        }];
        let result = run_equifinality_experiment(&template, &perturbations, 40);

        let (open_mean, blocked_mean) = result.mean_final_by_permeability();
        assert!(
            open_mean < blocked_mean,
            "open gap junctions (mean discrepancy={open_mean}) should recover the \
             imposed target pattern better than blocked ones (mean={blocked_mean})"
        );
    }

    #[test]
    fn open_gap_junctions_recover_across_multiple_amputation_sizes() {
        // The equifinality claim: regardless of how much tissue is removed,
        // open gap junctions recover closer to the *same shared target*
        // than blocked ones do, for each perturbation independently.
        let template = build_radial_bipolar_template(11, 200, 20, 0.2);
        let perturbations = [
            Perturbation::Amputate {
                min_r: 1.1,
                max_r: 2.0,
            },
            Perturbation::Amputate {
                min_r: 0.95,
                max_r: 2.0,
            },
            Perturbation::Amputate {
                min_r: 0.8,
                max_r: 2.0,
            },
        ];
        let result = run_equifinality_experiment(&template, &perturbations, 40);

        for perturbation in &perturbations {
            let label = perturbation.label();
            let open = result
                .conditions
                .iter()
                .find(|c| c.perturbation_label == label && c.gap_junction_permeability > 0.0)
                .unwrap()
                .final_discrepancy;
            let blocked = result
                .conditions
                .iter()
                .find(|c| c.perturbation_label == label && c.gap_junction_permeability == 0.0)
                .unwrap()
                .final_discrepancy;
            assert!(
                open < blocked,
                "perturbation {label}: open (discrepancy={open}) should recover better \
                 than blocked (discrepancy={blocked})"
            );
        }
    }

    #[test]
    fn perturbation_label_is_stable_and_distinct() {
        let a = Perturbation::Amputate {
            min_r: 0.5,
            max_r: 1.0,
        };
        let b = Perturbation::ScrambleVmem { seed: 3 };
        assert_ne!(a.label(), b.label());
        assert_eq!(a.label(), a.label());
    }

    #[test]
    fn equifinality_result_helpers_handle_empty_gracefully() {
        let result = EquifinalityResult { conditions: vec![] };
        assert_eq!(result.mean_final_by_permeability(), (0.0, 0.0));
        assert_eq!(result.open_run_spread(), 0.0);
        assert_eq!(result.blocked_run_spread(), 0.0);
        assert!(!result.open_beats_blocked());
    }
}
