// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bayesian hypothesis comparison for the LCF rate anomaly.
//!
//! Maintains a probability distribution over the five candidate
//! explanations of the NASA rate gap ([`HypothesisType`]) and updates it
//! from experimental outcomes using the signature-level predictions in
//! [`crate::experimental_design::ExpectedOutcome`].
//!
//! ## Honesty notes
//! - Likelihoods are derived from *signature-level* predicted-rate ranges
//!   and neutron energies, not from full measurement models. Treat
//!   posteriors as decision-support weights, not calibrated probabilities.
//! - The default prior deliberately puts the largest mass on
//!   `MeasurementError` — a 40–50 order-of-magnitude gap against standard
//!   physics demands a skeptical prior.

use crate::experimental_design::ExperimentDesign;
use crate::hypothesis_models::HypothesisType;
use serde::{Deserialize, Serialize};

/// All candidate hypotheses, in canonical order.
pub const ALL_HYPOTHESES: [HypothesisType; 5] = [
    HypothesisType::HotSpots,
    HypothesisType::PhononCascade,
    HypothesisType::SuperScreening,
    HypothesisType::LatticeNuclear,
    HypothesisType::MeasurementError,
];

/// Neutron-energy resolution used to decide whether two predicted
/// outcomes are experimentally distinguishable (MeV).
pub const ENERGY_RESOLUTION_MEV: f64 = 0.5;

/// Probability assigned to the matching outcome class when updating from
/// an observation (the remainder is spread over non-matching classes).
pub const MATCH_LIKELIHOOD: f64 = 0.9;

/// An observed experimental outcome, in the same signature-level terms as
/// [`crate::experimental_design::ExpectedOutcome`].
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ObservedOutcome {
    /// Measured neutron rate (n/s), background-subtracted.
    pub neutron_rate_per_s: f64,
    /// Measured neutron energy peak (MeV), if spectroscopy was available.
    pub neutron_energy_mev: Option<f64>,
}

/// Hypotheses grouped into experimentally distinguishable outcome classes
/// for one experiment design.
#[derive(Debug, Clone)]
pub struct OutcomeClasses {
    /// Each class holds hypotheses whose predictions this experiment
    /// cannot tell apart (overlapping rate ranges, same energy within
    /// [`ENERGY_RESOLUTION_MEV`]).
    pub classes: Vec<Vec<HypothesisType>>,
    /// Hypotheses the design makes no prediction for (never updated
    /// relative to each other by this experiment).
    pub untested: Vec<HypothesisType>,
}

impl OutcomeClasses {
    /// Group a design's expected outcomes into distinguishable classes.
    pub fn from_design(design: &ExperimentDesign) -> Self {
        let outcomes = &design.expected_outcomes;
        let mut classes: Vec<Vec<usize>> = Vec::new();
        for (i, o) in outcomes.iter().enumerate() {
            let mut placed = false;
            for class in &mut classes {
                let rep = &outcomes[class[0]];
                let ranges_overlap = o.predicted_rate_range.0 <= rep.predicted_rate_range.1
                    && rep.predicted_rate_range.0 <= o.predicted_rate_range.1;
                let energy_close = (o.predicted_energy_mev - rep.predicted_energy_mev).abs()
                    < ENERGY_RESOLUTION_MEV;
                if ranges_overlap && energy_close {
                    class.push(i);
                    placed = true;
                    break;
                }
            }
            if !placed {
                classes.push(vec![i]);
            }
        }

        let tested: Vec<HypothesisType> = outcomes.iter().map(|o| o.hypothesis).collect();
        let untested = ALL_HYPOTHESES
            .iter()
            .copied()
            .filter(|h| !tested.contains(h))
            .collect();

        Self {
            classes: classes
                .into_iter()
                .map(|idxs| idxs.into_iter().map(|i| outcomes[i].hypothesis).collect())
                .collect(),
            untested,
        }
    }

    /// Number of distinguishable outcome classes.
    pub fn len(&self) -> usize {
        self.classes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.classes.is_empty()
    }

    /// Index of the class an observation matches, if any: the observed
    /// rate falls inside the class representative's predicted range and
    /// the energy (when both sides have one) agrees within resolution.
    pub fn matching_class(
        &self,
        design: &ExperimentDesign,
        observed: &ObservedOutcome,
    ) -> Option<usize> {
        for (k, class) in self.classes.iter().enumerate() {
            let rep_h = class[0];
            let rep = design
                .expected_outcomes
                .iter()
                .find(|o| o.hypothesis == rep_h)
                .expect("class member must exist in design outcomes");
            let rate_ok = observed.neutron_rate_per_s >= rep.predicted_rate_range.0
                && observed.neutron_rate_per_s <= rep.predicted_rate_range.1;
            let energy_ok = match observed.neutron_energy_mev {
                Some(e) if rep.predicted_energy_mev > 0.0 => {
                    (e - rep.predicted_energy_mev).abs() < ENERGY_RESOLUTION_MEV
                }
                _ => true,
            };
            if rate_ok && energy_ok {
                return Some(k);
            }
        }
        None
    }
}

/// Probability distribution over the anomaly hypotheses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HypothesisBelief {
    probs: Vec<(HypothesisType, f64)>,
}

impl HypothesisBelief {
    /// Skeptical default prior. `MeasurementError` carries the largest
    /// mass; among physical mechanisms, hot spots are weighted highest
    /// (see `docs/TECHNICAL_REPORT.md` §"most plausible explanation").
    pub fn default_priors() -> Self {
        Self::new(&[
            (HypothesisType::MeasurementError, 0.40),
            (HypothesisType::HotSpots, 0.25),
            (HypothesisType::SuperScreening, 0.15),
            (HypothesisType::PhononCascade, 0.10),
            (HypothesisType::LatticeNuclear, 0.10),
        ])
    }

    /// Uniform prior over all five hypotheses.
    pub fn uniform() -> Self {
        let p = 1.0 / ALL_HYPOTHESES.len() as f64;
        Self {
            probs: ALL_HYPOTHESES.iter().map(|&h| (h, p)).collect(),
        }
    }

    /// Build from explicit (hypothesis, weight) pairs; weights are
    /// normalized. Panics on non-positive or non-finite weights or on a
    /// missing hypothesis.
    pub fn new(pairs: &[(HypothesisType, f64)]) -> Self {
        assert_eq!(
            pairs.len(),
            ALL_HYPOTHESES.len(),
            "must specify all hypotheses"
        );
        for h in ALL_HYPOTHESES {
            assert!(
                pairs.iter().any(|(ph, _)| *ph == h),
                "missing hypothesis {h:?}"
            );
        }
        let total: f64 = pairs.iter().map(|(_, w)| w).sum();
        assert!(
            total.is_finite() && total > 0.0,
            "weights must be positive and finite"
        );
        for (_, w) in pairs {
            assert!(
                w.is_finite() && *w > 0.0,
                "each weight must be positive and finite"
            );
        }
        Self {
            probs: pairs.iter().map(|&(h, w)| (h, w / total)).collect(),
        }
    }

    pub fn prob(&self, h: HypothesisType) -> f64 {
        self.probs
            .iter()
            .find(|(ph, _)| *ph == h)
            .map(|(_, p)| *p)
            .unwrap_or(0.0)
    }

    pub fn iter(&self) -> impl Iterator<Item = (HypothesisType, f64)> + '_ {
        self.probs.iter().copied()
    }

    /// Maximum a-posteriori hypothesis.
    pub fn map_hypothesis(&self) -> HypothesisType {
        self.probs
            .iter()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(h, _)| *h)
            .expect("belief is never empty")
    }

    /// Shannon entropy in bits (max log2(5) ≈ 2.32 for uniform).
    pub fn entropy_bits(&self) -> f64 {
        -self
            .probs
            .iter()
            .map(|(_, p)| if *p > 0.0 { p * p.log2() } else { 0.0 })
            .sum::<f64>()
    }

    /// Likelihood of `observed` under each hypothesis for `design`.
    ///
    /// Tested hypotheses: [`MATCH_LIKELIHOOD`] if the observation lands in
    /// their outcome class, the remainder spread over the other classes.
    /// Untested hypotheses: uniform `1/K` over the K classes (they predict
    /// nothing, so every outcome is equally consistent with them).
    pub fn likelihoods(
        design: &ExperimentDesign,
        observed: &ObservedOutcome,
    ) -> Vec<(HypothesisType, f64)> {
        let classes = OutcomeClasses::from_design(design);
        if classes.is_empty() {
            // Design predicts nothing: totally uninformative.
            return ALL_HYPOTHESES.iter().map(|&h| (h, 1.0)).collect();
        }
        let k = classes.len() as f64;
        let matched = classes.matching_class(design, observed);
        let uniform = 1.0 / k;

        ALL_HYPOTHESES
            .iter()
            .map(|&h| {
                let lik =
                    if let Some(class_idx) = classes.classes.iter().position(|c| c.contains(&h)) {
                        match matched {
                            Some(m) if m == class_idx => MATCH_LIKELIHOOD,
                            // Observation landed elsewhere: spread the
                            // residual over non-matching classes.
                            Some(_) => (1.0 - MATCH_LIKELIHOOD) / (k - 1.0).max(1.0),
                            // No class matched at all — mildly disfavor every
                            // tested hypothesis relative to untested ones.
                            None => (1.0 - MATCH_LIKELIHOOD) / k,
                        }
                    } else {
                        uniform
                    };
                (h, lik)
            })
            .collect()
    }

    /// Bayes update from one experimental observation.
    pub fn update(&mut self, design: &ExperimentDesign, observed: &ObservedOutcome) {
        let liks = Self::likelihoods(design, observed);
        let mut total = 0.0;
        for (h, p) in &mut self.probs {
            let lik = liks
                .iter()
                .find(|(lh, _)| lh == h)
                .map(|(_, l)| *l)
                .unwrap_or(1.0);
            *p *= lik;
            total += *p;
        }
        assert!(
            total > 0.0,
            "posterior collapsed to zero — inconsistent likelihoods"
        );
        for (_, p) in &mut self.probs {
            *p /= total;
        }
    }
}

impl Default for HypothesisBelief {
    fn default() -> Self {
        Self::default_priors()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::experimental_design::{
        ExpectedOutcome, ExperimentDesign, ExperimentalSetup, SampleGeometry, TriggerSpec,
    };

    fn outcome(h: HypothesisType, range: (f64, f64), energy: f64) -> ExpectedOutcome {
        ExpectedOutcome {
            hypothesis: h,
            predicted_rate_range: range,
            predicted_energy_mev: energy,
            temperature_dependence: String::new(),
            trigger_dependence: String::new(),
            signatures: vec![],
        }
    }

    pub(crate) fn design(outcomes: Vec<ExpectedOutcome>) -> ExperimentDesign {
        ExperimentDesign {
            name: "synthetic".to_string(),
            research_question: String::new(),
            hypotheses_tested: outcomes.iter().map(|o| o.hypothesis).collect(),
            setup: ExperimentalSetup {
                host_material: "Pd".to_string(),
                loading_ratio: 0.7,
                sample_geometry: SampleGeometry {
                    shape: "cylinder".to_string(),
                    dimensions_cm: vec![1.0],
                    active_volume_cm3: 1.0,
                    surface_area_cm2: 1.0,
                },
                trigger: TriggerSpec {
                    trigger_type: "X-ray".to_string(),
                    intensity: 1.0,
                    intensity_units: "W".to_string(),
                    pulse_duration_s: None,
                    repetition_rate_hz: None,
                },
                temperature_range: (300.0, 300.0),
                controls: vec![],
            },
            expected_outcomes: outcomes,
            instrumentation: vec![],
            estimated_cost_usd: 100_000.0,
            duration_months: 6.0,
            priority: 0.5,
            success_criteria: vec![],
        }
    }

    #[test]
    fn test_priors_normalized() {
        for belief in [
            HypothesisBelief::default_priors(),
            HypothesisBelief::uniform(),
        ] {
            let total: f64 = belief.iter().map(|(_, p)| p).sum();
            assert!((total - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_uniform_entropy_is_log2_5() {
        let e = HypothesisBelief::uniform().entropy_bits();
        assert!((e - (5.0f64).log2()).abs() < 1e-9, "entropy {e}");
    }

    #[test]
    fn test_default_prior_is_skeptical() {
        let belief = HypothesisBelief::default_priors();
        assert_eq!(belief.map_hypothesis(), HypothesisType::MeasurementError);
    }

    #[test]
    fn test_outcome_classes_distinct_and_overlapping() {
        // Two clearly distinct predictions + one overlapping the first.
        let d = design(vec![
            outcome(HypothesisType::HotSpots, (100.0, 1000.0), 2.45),
            outcome(HypothesisType::MeasurementError, (0.0, 1.0), 0.0),
            outcome(HypothesisType::SuperScreening, (500.0, 2000.0), 2.45),
        ]);
        let classes = OutcomeClasses::from_design(&d);
        assert_eq!(classes.len(), 2, "{:?}", classes.classes);
        assert_eq!(classes.untested.len(), 2);
    }

    #[test]
    fn test_update_boosts_matching_hypothesis() {
        let d = design(vec![
            outcome(HypothesisType::HotSpots, (100.0, 1000.0), 2.45),
            outcome(HypothesisType::MeasurementError, (0.0, 1.0), 0.0),
        ]);
        let mut belief = HypothesisBelief::uniform();
        let before = belief.prob(HypothesisType::HotSpots);
        belief.update(
            &d,
            &ObservedOutcome {
                neutron_rate_per_s: 500.0,
                neutron_energy_mev: Some(2.45),
            },
        );
        assert!(belief.prob(HypothesisType::HotSpots) > before);
        assert!(belief.prob(HypothesisType::MeasurementError) < 0.2);
        let total: f64 = belief.iter().map(|(_, p)| p).sum();
        assert!((total - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_update_preserves_untested_ratios() {
        let d = design(vec![
            outcome(HypothesisType::HotSpots, (100.0, 1000.0), 2.45),
            outcome(HypothesisType::MeasurementError, (0.0, 1.0), 0.0),
        ]);
        let mut belief = HypothesisBelief::default_priors();
        let ratio_before = belief.prob(HypothesisType::PhononCascade)
            / belief.prob(HypothesisType::LatticeNuclear);
        belief.update(
            &d,
            &ObservedOutcome {
                neutron_rate_per_s: 0.5,
                neutron_energy_mev: None,
            },
        );
        let ratio_after = belief.prob(HypothesisType::PhononCascade)
            / belief.prob(HypothesisType::LatticeNuclear);
        assert!((ratio_before - ratio_after).abs() < 1e-9);
    }

    #[test]
    fn test_no_prediction_design_is_uninformative() {
        let d = design(vec![]);
        let mut belief = HypothesisBelief::default_priors();
        let before: Vec<f64> = belief.iter().map(|(_, p)| p).collect();
        belief.update(
            &d,
            &ObservedOutcome {
                neutron_rate_per_s: 42.0,
                neutron_energy_mev: None,
            },
        );
        let after: Vec<f64> = belief.iter().map(|(_, p)| p).collect();
        for (b, a) in before.iter().zip(after.iter()) {
            assert!((b - a).abs() < 1e-12);
        }
    }

    #[test]
    fn test_real_designs_produce_classes() {
        use crate::experimental_design::ExperimentDesigner;
        let program = ExperimentDesigner::design_program();
        for phase in &program.phases {
            for exp in &phase.experiments {
                let classes = OutcomeClasses::from_design(exp);
                if !exp.expected_outcomes.is_empty() {
                    assert!(!classes.is_empty(), "no classes for {}", exp.name);
                }
            }
        }
    }
}
