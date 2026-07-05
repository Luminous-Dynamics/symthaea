// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evolutionary anatomical compiler: given a desired target morphology,
//! search over intervention parameters for a combination that drives the
//! organoid toward it -- Levin's "given a desired anatomy, find the
//! intervention" framing, distinct from [`crate::experiments`]'s existing
//! equifinality/dose-response experiments (which test recovery from a
//! hand-designed *perturbation* of a pattern the organoid already had).
//!
//! Uses [`argmin`]'s real, tested Particle Swarm solver
//! (`argmin::solver::particleswarm::ParticleSwarm`) over 4 continuous
//! intervention parameters. This is deliberately *not* CMA-ES: an earlier
//! sketch of this crate's roadmap proposed "genericizing the CMA-ES math"
//! from `symthaea/examples/cmaes_standing.rs`/`cmaes_walking.rs`, but those
//! examples turn out to contain a hand-rolled, non-standard, untested
//! diagonal evolution strategy (per-parameter self-adaptive step sizes, no
//! covariance matrix) -- not real CMA-ES. `argmin` was already a workspace
//! dependency (via `symthaea-fep`) but unused anywhere; it doesn't ship
//! CMA-ES either, but its Particle Swarm solver is real, documented,
//! maintained code, which is the more honest reuse target for the same
//! job: minimizing an expensive, gradient-free scalar cost function.

use argmin::core::{CostFunction, Error, Executor, State};
use argmin::solver::particleswarm::ParticleSwarm;

use crate::bioelectric::TargetMorphology;
use crate::morphogenetic_consciousness::NeuralOrganoid;

/// Lower/upper bounds for the 4 searched parameters, in the order
/// [`MorphologyFitness::cost`] reads them: gap-junction permeability,
/// positional-homing rate, gap-junction diffusion rate, K+-channel block.
/// Positional homing and the ion-channel model are fixed **on** (structural
/// toggles, not searched) -- see module docs.
const PARAM_LOWER_BOUNDS: [f64; 4] = [0.0, 0.0, 0.0, 0.0];
const PARAM_UPPER_BOUNDS: [f64; 4] = [1.0, 0.3, 1.0, 1.0];

/// Cost function: clone `template`, apply one candidate intervention
/// configuration, run `recovery_days` of development, and return the
/// resulting discrepancy against `target` (0.0 = exact match). Follows the
/// same "clone template, apply setters, loop `advance_day`, read
/// discrepancy" pattern as [`crate::experiments::run_equifinality_experiment`],
/// generalized to an arbitrary parameter vector instead of a hardcoded
/// permeability/homing-bool pair.
struct MorphologyFitness {
    template: NeuralOrganoid,
    target: TargetMorphology,
    recovery_days: u32,
}

impl CostFunction for MorphologyFitness {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, param: &Vec<f64>) -> Result<f64, Error> {
        let mut organoid = self.template.clone();
        organoid.set_gap_junction_permeability(param[0] as f32);
        organoid.set_positional_homing(true);
        organoid.set_positional_homing_rate(param[1] as f32);
        organoid.set_gap_junction_diffusion_rate(param[2] as f32);
        organoid.set_ion_channel_model_enabled(true);
        organoid.set_potassium_channel_block(param[3] as f32);
        organoid.target_morphology = Some(self.target.clone());

        for _ in 0..self.recovery_days {
            organoid.advance_day();
        }

        // No target captured is not reachable here (we just set one above),
        // but fall back to the worst-case discrepancy rather than panicking
        // if that assumption is ever violated by a future refactor.
        Ok(organoid.morphology_discrepancy().unwrap_or(1.0))
    }
}

/// Search for an intervention (gap-junction permeability, positional-homing
/// rate, gap-junction diffusion rate, K+-channel block) that drives
/// `template` toward `target` over `recovery_days` of development, using
/// `num_particles` particles for `max_iters` iterations of Particle Swarm
/// Optimization. Returns `(best_params, best_discrepancy)`.
///
/// `template` is not mutated -- every candidate clones it fresh.
pub fn search_intervention(
    template: &NeuralOrganoid,
    target: &TargetMorphology,
    recovery_days: u32,
    num_particles: usize,
    max_iters: u64,
) -> (Vec<f64>, f64) {
    let cost = MorphologyFitness {
        template: template.clone(),
        target: target.clone(),
        recovery_days,
    };
    let bounds = (PARAM_LOWER_BOUNDS.to_vec(), PARAM_UPPER_BOUNDS.to_vec());
    let solver = ParticleSwarm::new(bounds, num_particles);

    let result = Executor::new(cost, solver)
        .configure(|state| state.max_iters(max_iters))
        .run()
        .expect("particle swarm execution should not fail for this cost function");

    let best_params = result
        .state
        .get_best_param()
        .map(|particle| particle.position.clone())
        .unwrap_or_else(|| vec![0.0; PARAM_LOWER_BOUNDS.len()]);
    let best_cost = result.state.get_best_cost();

    (best_params, best_cost)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_target(seed: u64, cells: usize) -> TargetMorphology {
        let mut source = NeuralOrganoid::new(cells, seed);
        source.impose_vmem_pattern(|p| if p[0] >= 0.0 { -0.8 } else { -0.1 });
        source.capture_target_morphology();
        source.target_morphology.expect("just captured")
    }

    #[test]
    fn search_respects_parameter_bounds() {
        let template = NeuralOrganoid::new(30, 1);
        let target = build_target(2, 30);
        let (params, _cost) = search_intervention(&template, &target, 15, 6, 5);

        assert_eq!(params.len(), PARAM_LOWER_BOUNDS.len());
        for (i, &p) in params.iter().enumerate() {
            assert!(
                p >= PARAM_LOWER_BOUNDS[i] && p <= PARAM_UPPER_BOUNDS[i],
                "param {i}={p} outside bounds [{}, {}]",
                PARAM_LOWER_BOUNDS[i],
                PARAM_UPPER_BOUNDS[i]
            );
        }
    }

    #[test]
    fn search_reduces_discrepancy_below_naive_baseline() {
        let template = NeuralOrganoid::new(30, 3);
        let target = build_target(4, 30);

        // Naive baseline: gap junctions closed, no homing pull, ion-channel
        // model off -- the tissue has no mechanism to move toward the
        // target at all.
        let naive_cost = MorphologyFitness {
            template: template.clone(),
            target: target.clone(),
            recovery_days: 15,
        };
        let naive_discrepancy = naive_cost
            .cost(&vec![0.0, 0.0, 0.0, 0.0])
            .expect("naive baseline cost should not fail");

        let (_best_params, best_discrepancy) = search_intervention(&template, &target, 15, 6, 5);

        assert!(
            best_discrepancy < naive_discrepancy,
            "search should find a better intervention than the naive \
             no-mechanism baseline: naive={naive_discrepancy}, \
             best={best_discrepancy}"
        );
    }
}
