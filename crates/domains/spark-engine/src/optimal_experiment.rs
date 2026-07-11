// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bayesian optimal experiment design for the LCF anomaly program.
//!
//! Ranks candidate experiments by **expected information gain** (EIG, in
//! bits) about which anomaly hypothesis is true, given a
//! [`HypothesisBelief`], and derives a budget-constrained greedy sequence.
//! This turns the qualitative `prioritize_experiments()` heuristic into a
//! decision-theoretic ranking, and generates the "cheapest maximally
//! discriminating experiment" proposal as a markdown report.
//!
//! ## Model (documented approximation)
//! Outcomes are discretized into the distinguishable classes of
//! [`OutcomeClasses`]: if hypothesis `h` is true and the design predicts
//! an outcome for it, the observation lands in `h`'s class with
//! probability 1; hypotheses with no prediction produce a uniform outcome
//! over the K classes. EIG is then the mutual information between the
//! hypothesis and the outcome class. Signature-level, not a full
//! measurement model — see `bayesian.rs` honesty notes.

use crate::bayesian::{ALL_HYPOTHESES, HypothesisBelief, OutcomeClasses};
use crate::experimental_design::{ExperimentDesign, ExperimentDesigner};
use crate::hypothesis_models::HypothesisType;
use serde::{Deserialize, Serialize};

/// Minimum EIG (bits) for an experiment to be worth including in a
/// greedy sequence.
pub const MIN_USEFUL_EIG_BITS: f64 = 0.05;

/// Information-gain assessment of one candidate experiment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentInfoGain {
    pub name: String,
    pub hypotheses_tested: Vec<HypothesisType>,
    /// Expected information gain about the true hypothesis (bits).
    pub eig_bits: f64,
    pub cost_usd: f64,
    pub duration_months: f64,
    /// EIG per $100K — the "cheapest discriminating experiment" metric.
    pub eig_per_100k_usd: f64,
}

/// One step of a greedy budget-constrained experiment sequence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequenceStep {
    pub experiment_name: String,
    pub cost_usd: f64,
    pub cumulative_cost_usd: f64,
    /// EIG at the moment this experiment was selected (belief-dependent).
    pub eig_bits_at_selection: f64,
    /// Belief entropy after simulating this experiment under the
    /// current-MAP-hypothesis world (see [`greedy_sequence`] docs).
    pub entropy_after_bits: f64,
}

/// Expected information gain (bits) of `design` under `belief`.
///
/// EIG = H(hypothesis) − Σ_o P(o) · H(hypothesis | o), with the outcome
/// model described in the module docs. Always in `[0, H(belief)]`.
pub fn expected_information_gain(design: &ExperimentDesign, belief: &HypothesisBelief) -> f64 {
    let classes = OutcomeClasses::from_design(design);
    if classes.is_empty() {
        return 0.0;
    }
    let k = classes.len();
    let uniform = 1.0 / k as f64;

    // P(o_j | h): deterministic class for tested h, uniform for untested.
    let p_outcome_given = |h: HypothesisType, j: usize| -> f64 {
        match classes.classes.iter().position(|c| c.contains(&h)) {
            Some(class_idx) if class_idx == j => 1.0,
            Some(_) => 0.0,
            None => uniform,
        }
    };

    let prior_entropy = belief.entropy_bits();
    let mut expected_posterior_entropy = 0.0;

    for j in 0..k {
        // P(o_j) and unnormalized posterior over hypotheses.
        let joint: Vec<(HypothesisType, f64)> = ALL_HYPOTHESES
            .iter()
            .map(|&h| (h, belief.prob(h) * p_outcome_given(h, j)))
            .collect();
        let p_o: f64 = joint.iter().map(|(_, p)| p).sum();
        if p_o <= 0.0 {
            continue;
        }
        let posterior_entropy: f64 = -joint
            .iter()
            .map(|(_, p)| {
                let q = p / p_o;
                if q > 0.0 { q * q.log2() } else { 0.0 }
            })
            .sum::<f64>();
        expected_posterior_entropy += p_o * posterior_entropy;
    }

    (prior_entropy - expected_posterior_entropy).max(0.0)
}

/// Rank experiments by EIG per dollar (descending). Ties and zero-cost
/// degenerate entries are handled by falling back to raw EIG.
pub fn rank_experiments(
    designs: &[ExperimentDesign],
    belief: &HypothesisBelief,
) -> Vec<ExperimentInfoGain> {
    let mut ranked: Vec<ExperimentInfoGain> = designs
        .iter()
        .map(|d| {
            let eig = expected_information_gain(d, belief);
            let cost = d.estimated_cost_usd.max(1.0);
            ExperimentInfoGain {
                name: d.name.clone(),
                hypotheses_tested: d.hypotheses_tested.clone(),
                eig_bits: eig,
                cost_usd: d.estimated_cost_usd,
                duration_months: d.duration_months,
                eig_per_100k_usd: eig / (cost / 100_000.0),
            }
        })
        .collect();
    ranked.sort_by(|a, b| {
        b.eig_per_100k_usd
            .total_cmp(&a.eig_per_100k_usd)
            .then(b.eig_bits.total_cmp(&a.eig_bits))
    });
    ranked
}

/// Greedy budget-constrained sequence: repeatedly pick the affordable
/// experiment with the best EIG-per-dollar (recomputed as belief evolves),
/// stopping when nothing affordable clears [`MIN_USEFUL_EIG_BITS`].
///
/// Between steps the belief is advanced by simulating the outcome the
/// **current MAP hypothesis** would produce (a real Bayes update needs a
/// real observation; the expected posterior equals the prior, so planning
/// requires committing to a simulated world — we use the MAP one and say
/// so). The sequence is a *plan*, re-derive it as real data arrives.
pub fn greedy_sequence(
    designs: &[ExperimentDesign],
    belief: &HypothesisBelief,
    budget_usd: f64,
) -> Vec<SequenceStep> {
    let mut belief = belief.clone();
    let mut remaining: Vec<&ExperimentDesign> = designs.iter().collect();
    let mut spent = 0.0;
    let mut steps = Vec::new();

    loop {
        let mut best: Option<(usize, f64, f64)> = None; // (idx, eig, eig_per_$)
        for (i, d) in remaining.iter().enumerate() {
            if spent + d.estimated_cost_usd > budget_usd {
                continue;
            }
            let eig = expected_information_gain(d, &belief);
            if eig < MIN_USEFUL_EIG_BITS {
                continue;
            }
            let per_dollar = eig / d.estimated_cost_usd.max(1.0);
            if best.map(|(_, _, bpd)| per_dollar > bpd).unwrap_or(true) {
                best = Some((i, eig, per_dollar));
            }
        }
        let Some((idx, eig, _)) = best else { break };
        let design = remaining.remove(idx);
        spent += design.estimated_cost_usd;

        // Simulate the MAP-hypothesis world's outcome to advance belief.
        let map_h = belief.map_hypothesis();
        let observed = match design
            .expected_outcomes
            .iter()
            .find(|o| o.hypothesis == map_h)
        {
            Some(o) => crate::bayesian::ObservedOutcome {
                neutron_rate_per_s: 0.5 * (o.predicted_rate_range.0 + o.predicted_rate_range.1),
                neutron_energy_mev: if o.predicted_energy_mev > 0.0 {
                    Some(o.predicted_energy_mev)
                } else {
                    None
                },
            },
            // The MAP hypothesis predicts nothing for this design: in its
            // world the observation matches no predicted class. A negative
            // rate is outside every predicted range by construction.
            None => crate::bayesian::ObservedOutcome {
                neutron_rate_per_s: -1.0,
                neutron_energy_mev: None,
            },
        };
        belief.update(design, &observed);

        steps.push(SequenceStep {
            experiment_name: design.name.clone(),
            cost_usd: design.estimated_cost_usd,
            cumulative_cost_usd: spent,
            eig_bits_at_selection: eig,
            entropy_after_bits: belief.entropy_bits(),
        });
    }
    steps
}

/// Collect every experiment in the standard program into a flat list.
pub fn standard_candidate_experiments() -> Vec<ExperimentDesign> {
    ExperimentDesigner::design_program()
        .phases
        .into_iter()
        .flat_map(|p| p.experiments)
        .collect()
}

/// Generate the discriminating-experiment proposal as a markdown report.
pub fn generate_proposal_markdown(belief: &HypothesisBelief, budget_usd: f64) -> String {
    let designs = standard_candidate_experiments();
    let ranking = rank_experiments(&designs, belief);
    let sequence = greedy_sequence(&designs, belief, budget_usd);

    let mut md = String::new();
    md.push_str("# LCF Anomaly: Discriminating-Experiment Proposal\n\n");
    md.push_str(
        "Auto-generated by `spark-engine::optimal_experiment` from the \
         signature-level predictions in `experimental_design.rs`.\n\n",
    );

    md.push_str("## Priors over anomaly hypotheses\n\n");
    md.push_str("| Hypothesis | Prior |\n|---|---|\n");
    for (h, p) in belief.iter() {
        md.push_str(&format!("| {h:?} | {p:.2} |\n"));
    }
    md.push_str(&format!(
        "\nPrior entropy: **{:.3} bits** (uniform max: {:.3}).\n\n",
        belief.entropy_bits(),
        (ALL_HYPOTHESES.len() as f64).log2()
    ));

    md.push_str("## Ranking by expected information gain per dollar\n\n");
    md.push_str("| # | Experiment | EIG (bits) | Cost ($K) | Months | EIG / $100K |\n");
    md.push_str("|---|---|---|---|---|---|\n");
    for (i, r) in ranking.iter().enumerate() {
        md.push_str(&format!(
            "| {} | {} | {:.3} | {:.0} | {:.0} | {:.3} |\n",
            i + 1,
            r.name,
            r.eig_bits,
            r.cost_usd / 1000.0,
            r.duration_months,
            r.eig_per_100k_usd
        ));
    }

    // Flag designs whose machine-readable predictions are too sparse for
    // EIG to see (a single distinguishable outcome class): their zero
    // score means "prediction data not encoded", not "experiment useless".
    let sparse: Vec<&str> = designs
        .iter()
        .filter(|d| OutcomeClasses::from_design(d).len() < 2)
        .map(|d| d.name.as_str())
        .collect();
    if !sparse.is_empty() {
        md.push_str(&format!(
            "\n**Data-completeness caveat**: {} of {} designs encode fewer \
             than two distinguishable outcome classes in their machine-readable \
             `ExpectedOutcome` data ({}). Their EIG of 0.000 reflects missing \
             *encoded* predictions — the discrimination they describe lives in \
             free-text signatures. Enriching those predictions is tracked in \
             the improvement plan.\n",
            sparse.len(),
            designs.len(),
            sparse.join(", ")
        ));
    }

    md.push_str(&format!(
        "\n## Recommended sequence under ${:.0}K budget\n\n",
        budget_usd / 1000.0
    ));
    if sequence.is_empty() {
        md.push_str("_No affordable experiment clears the minimum-EIG bar._\n");
    } else {
        md.push_str("| Step | Experiment | Cost ($K) | Cumulative ($K) | EIG at selection | Entropy after |\n");
        md.push_str("|---|---|---|---|---|---|\n");
        for (i, s) in sequence.iter().enumerate() {
            md.push_str(&format!(
                "| {} | {} | {:.0} | {:.0} | {:.3} | {:.3} |\n",
                i + 1,
                s.experiment_name,
                s.cost_usd / 1000.0,
                s.cumulative_cost_usd / 1000.0,
                s.eig_bits_at_selection,
                s.entropy_after_bits
            ));
        }
    }

    md.push_str(
        "\n## Honesty notes\n\n\
         - EIG uses signature-level outcome classes (predicted rate ranges \
         + neutron energy), not full measurement models.\n\
         - The sequence simulates outcomes under the current MAP hypothesis; \
         re-derive after every real observation.\n\
         - None of this establishes net energy gain: under standard physics \
         Q ≈ 10⁻⁵⁷ at 300 K. The program discriminates *explanations of the \
         rate anomaly*; the credible application remains a compact neutron \
         source.\n",
    );
    md
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experimental_design::ExpectedOutcome;

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

    fn two_class_design() -> ExperimentDesign {
        crate::bayesian::tests::design(vec![
            outcome(HypothesisType::HotSpots, (100.0, 1000.0), 2.45),
            outcome(HypothesisType::MeasurementError, (0.0, 1.0), 0.0),
        ])
    }

    #[test]
    fn test_eig_bounded_by_prior_entropy() {
        let belief = HypothesisBelief::default_priors();
        let d = two_class_design();
        let eig = expected_information_gain(&d, &belief);
        assert!(eig > 0.0, "discriminating design must have positive EIG");
        assert!(eig <= belief.entropy_bits() + 1e-9);
    }

    #[test]
    fn test_eig_zero_for_no_predictions() {
        let d = crate::bayesian::tests::design(vec![]);
        let eig = expected_information_gain(&d, &HypothesisBelief::default_priors());
        assert_eq!(eig, 0.0);
    }

    #[test]
    fn test_eig_zero_when_outcomes_indistinguishable() {
        // Both hypotheses predict the same signature — one class, no info
        // about which of the two is true (untested ones also uniform).
        let d = crate::bayesian::tests::design(vec![
            outcome(HypothesisType::HotSpots, (100.0, 1000.0), 2.45),
            outcome(HypothesisType::SuperScreening, (200.0, 900.0), 2.45),
        ]);
        let eig = expected_information_gain(&d, &HypothesisBelief::uniform());
        assert!(eig.abs() < 1e-9, "single-class design leaked info: {eig}");
    }

    #[test]
    fn test_ranking_sorted_by_eig_per_dollar() {
        let designs = standard_candidate_experiments();
        let ranking = rank_experiments(&designs, &HypothesisBelief::default_priors());
        assert_eq!(ranking.len(), designs.len());
        for w in ranking.windows(2) {
            assert!(w[0].eig_per_100k_usd >= w[1].eig_per_100k_usd - 1e-12);
        }
    }

    #[test]
    fn test_standard_program_has_discriminating_experiments() {
        let designs = standard_candidate_experiments();
        let belief = HypothesisBelief::default_priors();
        let any_informative = designs
            .iter()
            .any(|d| expected_information_gain(d, &belief) > MIN_USEFUL_EIG_BITS);
        assert!(
            any_informative,
            "no experiment in the standard program is informative"
        );
    }

    #[test]
    fn test_greedy_sequence_respects_budget() {
        let designs = standard_candidate_experiments();
        let budget = 500_000.0;
        let seq = greedy_sequence(&designs, &HypothesisBelief::default_priors(), budget);
        assert!(
            !seq.is_empty(),
            "expected at least one affordable informative experiment"
        );
        assert!(seq.last().unwrap().cumulative_cost_usd <= budget);
        // Cumulative cost strictly increases.
        for w in seq.windows(2) {
            assert!(w[1].cumulative_cost_usd > w[0].cumulative_cost_usd);
        }
    }

    #[test]
    fn test_greedy_sequence_reduces_entropy() {
        let designs = standard_candidate_experiments();
        let belief = HypothesisBelief::default_priors();
        let seq = greedy_sequence(&designs, &belief, 1_000_000.0);
        assert!(!seq.is_empty());
        assert!(
            seq.last().unwrap().entropy_after_bits < belief.entropy_bits(),
            "sequence should reduce simulated entropy"
        );
    }

    #[test]
    fn test_proposal_markdown_sections() {
        let md = generate_proposal_markdown(&HypothesisBelief::default_priors(), 500_000.0);
        for section in [
            "# LCF Anomaly: Discriminating-Experiment Proposal",
            "## Priors over anomaly hypotheses",
            "## Ranking by expected information gain per dollar",
            "## Recommended sequence under $500K budget",
            "## Honesty notes",
        ] {
            assert!(md.contains(section), "missing section: {section}");
        }
    }

    #[test]
    fn test_proposal_deterministic() {
        let a = generate_proposal_markdown(&HypothesisBelief::default_priors(), 500_000.0);
        let b = generate_proposal_markdown(&HypothesisBelief::default_priors(), 500_000.0);
        assert_eq!(a, b);
    }
}
