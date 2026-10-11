// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! RQ-006I development probe for context-optimizer objective plumbing.
//!
//! The probe holds primitive Phi/fitness constant while independently varying the source
//! harmonic-alignment and epistemic-coordinate fields. It then records the tradeoff coordinates
//! actually admitted by `ContextAwareOptimizer`. This freezes the current measurement path before
//! replacing historical 0.5 placeholders with canonical candidate signals.

use crate::consciousness::context_aware_evolution::{
    ContextAwareOptimizer, ReasoningContext,
};
use crate::consciousness::epistemic_tiers::EpistemicCoordinate;
use crate::consciousness::primitive_evolution::{CandidatePrimitive, EvolutionConfig};
use crate::hdc::BinaryHV;
use crate::hdc::primitive_system::PrimitiveTier;
use serde::{Deserialize, Serialize};

pub const OBJECTIVE_PLUMBING_PROBE_VERSION: &str = "rq-006i-objective-plumbing-v1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObjectivePlumbingObservation {
    pub primitive_name: String,
    pub source_fitness: f64,
    pub source_harmonic_alignment: f64,
    pub source_epistemic_quality: f64,
    pub observed_phi: f64,
    pub observed_harmonic: f64,
    pub observed_epistemic: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObjectivePlumbingProbeReport {
    pub probe_version: String,
    pub harmonic_case: Vec<ObjectivePlumbingObservation>,
    pub epistemic_case: Vec<ObjectivePlumbingObservation>,
    pub source_harmonic_spread: f64,
    pub observed_harmonic_spread: f64,
    pub source_epistemic_spread: f64,
    pub observed_epistemic_spread: f64,
}

pub fn run_objective_plumbing_probe() -> anyhow::Result<ObjectivePlumbingProbeReport> {
    let optimizer = ContextAwareOptimizer::new(EvolutionConfig::default())?;

    let harmonic_primitives = vec![
        primitive("harmonic-low", 0.7, 0.1, EpistemicCoordinate::axiom(), 101),
        primitive("harmonic-high", 0.7, 0.9, EpistemicCoordinate::axiom(), 102),
    ];
    let harmonic_result = optimizer.optimize_for_context(
        ReasoningContext::SocialInteraction,
        harmonic_primitives.clone(),
    )?;
    let harmonic_case = observations(&harmonic_primitives, &harmonic_result.frontier.all_points);

    let epistemic_primitives = vec![
        primitive("epistemic-low", 0.7, 0.5, EpistemicCoordinate::null(), 201),
        primitive("epistemic-high", 0.7, 0.5, EpistemicCoordinate::axiom(), 202),
    ];
    let epistemic_result = optimizer.optimize_for_context(
        ReasoningContext::TechnicalImplementation,
        epistemic_primitives.clone(),
    )?;
    let epistemic_case = observations(
        &epistemic_primitives,
        &epistemic_result.frontier.all_points,
    );

    Ok(ObjectivePlumbingProbeReport {
        probe_version: OBJECTIVE_PLUMBING_PROBE_VERSION.into(),
        source_harmonic_spread: spread(
            harmonic_case
                .iter()
                .map(|observation| observation.source_harmonic_alignment),
        ),
        observed_harmonic_spread: spread(
            harmonic_case
                .iter()
                .map(|observation| observation.observed_harmonic),
        ),
        source_epistemic_spread: spread(
            epistemic_case
                .iter()
                .map(|observation| observation.source_epistemic_quality),
        ),
        observed_epistemic_spread: spread(
            epistemic_case
                .iter()
                .map(|observation| observation.observed_epistemic),
        ),
        harmonic_case,
        epistemic_case,
    })
}

fn observations(
    source: &[CandidatePrimitive],
    admitted: &[(crate::consciousness::context_aware_evolution::TradeoffPoint, CandidatePrimitive)],
) -> Vec<ObjectivePlumbingObservation> {
    source
        .iter()
        .filter_map(|primitive| {
            admitted
                .iter()
                .find(|(_, admitted_primitive)| admitted_primitive.name == primitive.name)
                .map(|(point, _)| ObjectivePlumbingObservation {
                    primitive_name: primitive.name.clone(),
                    source_fitness: primitive.fitness,
                    source_harmonic_alignment: primitive.harmonic_alignment,
                    source_epistemic_quality: primitive.epistemic_coordinate.quality_score(),
                    observed_phi: point.phi,
                    observed_harmonic: point.harmonic,
                    observed_epistemic: point.epistemic,
                })
        })
        .collect()
}

fn primitive(
    name: &str,
    fitness: f64,
    harmonic_alignment: f64,
    epistemic_coordinate: EpistemicCoordinate,
    seed: u64,
) -> CandidatePrimitive {
    CandidatePrimitive {
        name: name.into(),
        tier: PrimitiveTier::Physical,
        definition: "fixed objective-plumbing development probe".into(),
        fitness,
        encoding: BinaryHV::random(seed),
        epistemic_coordinate,
        harmonic_alignment,
    }
}

fn spread(values: impl Iterator<Item = f64>) -> f64 {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for value in values {
        min = min.min(value);
        max = max.max(value);
    }
    if min.is_infinite() || max.is_infinite() {
        0.0
    } else {
        max - min
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_has_mechanically_distinct_source_axes() {
        let report = run_objective_plumbing_probe().expect("probe should execute");
        assert_eq!(report.harmonic_case.len(), 2);
        assert_eq!(report.epistemic_case.len(), 2);
        assert!(report.source_harmonic_spread >= 0.8 - f64::EPSILON);
        assert!(report.source_epistemic_spread >= 1.0 - f64::EPSILON);
    }

    #[test]
    fn current_subject_baseline_exposes_neutralized_harmonic_axis() {
        let report = run_objective_plumbing_probe().expect("probe should execute");
        assert_eq!(report.observed_harmonic_spread, 0.0);
        assert!(
            report
                .harmonic_case
                .iter()
                .all(|observation| (observation.observed_harmonic - 0.5).abs() <= f64::EPSILON)
        );
    }

    #[test]
    fn current_subject_baseline_exposes_neutralized_epistemic_axis() {
        let report = run_objective_plumbing_probe().expect("probe should execute");
        assert_eq!(report.observed_epistemic_spread, 0.0);
        assert!(
            report
                .epistemic_case
                .iter()
                .all(|observation| (observation.observed_epistemic - 0.5).abs() <= f64::EPSILON)
        );
    }

    #[test]
    fn phi_axis_still_preserves_source_fitness() {
        let report = run_objective_plumbing_probe().expect("probe should execute");
        for observation in report
            .harmonic_case
            .iter()
            .chain(report.epistemic_case.iter())
        {
            assert!((observation.observed_phi - observation.source_fitness).abs() <= f64::EPSILON);
        }
    }
}
