use anyhow::{Result, bail};
use rand::{Rng, SeedableRng, rngs::StdRng};
use serde::Serialize;

#[derive(Debug, Clone, Copy, Serialize)]
pub struct BootstrapEstimate {
    pub mean_difference: f64,
    pub ci_95_lower: f64,
    pub ci_95_upper: f64,
    pub draws: usize,
    pub seed: u64,
    pub participant_clusters: usize,
}

/// Difference between a collaboration outcome and the stronger matched solo
/// condition. Positive values indicate synergy; negative values indicate a
/// collaboration gap. All scores must use the same bounded outcome measure.
pub fn synergy_delta(human_only: f64, ai_only: f64, collaborative: f64) -> Result<f64> {
    for (name, score) in [
        ("human_only", human_only),
        ("ai_only", ai_only),
        ("collaborative", collaborative),
    ] {
        if !score.is_finite() || !(0.0..=1.0).contains(&score) {
            bail!("{name} score must be finite and in [0, 1]");
        }
    }
    Ok(collaborative - human_only.max(ai_only))
}

/// Positive values indicate that collaboration underperformed the best solo
/// condition. Negative values indicate synergy.
pub fn collaboration_gap(best_solo: f64, collaborative: f64) -> Result<f64> {
    for (name, score) in [("best_solo", best_solo), ("collaborative", collaborative)] {
        if !score.is_finite() || !(0.0..=1.0).contains(&score) {
            bail!("{name} score must be finite and in [0, 1]");
        }
    }
    Ok(best_solo - collaborative)
}

/// Absolute performance change caused by reversing a pre-registered partner
/// order or handoff sequence. A value near zero means order-insensitive.
pub fn partner_order_sensitivity(forward: f64, reverse: f64) -> Result<f64> {
    for (name, score) in [("forward", forward), ("reverse", reverse)] {
        if !score.is_finite() || !(0.0..=1.0).contains(&score) {
            bail!("{name} score must be finite and in [0, 1]");
        }
    }
    Ok((forward - reverse).abs())
}

/// Recovery fraction over explicitly attempted and independently adjudicated
/// handoffs. Returns None when no handoff was attempted.
pub fn handoff_recovery_rate(recovered: u64, attempted: u64) -> Result<Option<f64>> {
    if recovered > attempted {
        bail!("recovered handoffs cannot exceed attempted handoffs");
    }
    if attempted == 0 {
        return Ok(None);
    }
    Ok(Some(recovered as f64 / attempted as f64))
}



/// A blinded, task-specific adjudication of one intervention decision.
/// These labels must come from a preregistered rubric and independent review,
/// not from the evaluated agent's own self-score.
#[derive(Debug, Clone, Copy)]
pub struct InterventionRecord {
    pub appropriate: bool,
    pub unnecessary: bool,
    pub critical_opportunity: bool,
    pub critical_missed: bool,
    /// Counterfactual outcome loss, only when an admissible replay is identifiable.
    pub counterfactual_regret: Option<f64>,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct InterventionSummary {
    pub decision_count: usize,
    pub critical_opportunity_count: usize,
    pub counterfactually_identifiable_count: usize,
    pub intervention_appropriateness: Option<f64>,
    pub unnecessary_intervention_rate: Option<f64>,
    pub missed_critical_intervention_rate: Option<f64>,
    pub mean_intervention_regret: Option<f64>,
}

/// Aggregate adaptive-intervention metrics over independently adjudicated
/// decision points. Missing denominators remain None; they are never converted
/// to zero. Missed-critical rate is divided only by critical opportunities.
pub fn summarize_interventions(records: &[InterventionRecord]) -> Result<InterventionSummary> {
    let mut appropriate_count = 0usize;
    let mut unnecessary_count = 0usize;
    let mut critical_opportunity_count = 0usize;
    let mut critical_missed_count = 0usize;
    let mut regret_sum = 0.0f64;
    let mut regret_count = 0usize;

    for (index, record) in records.iter().enumerate() {
        if record.unnecessary && record.appropriate {
            bail!("intervention record {index}: an unnecessary intervention cannot be adjudicated appropriate");
        }
        if record.critical_missed && !record.critical_opportunity {
            bail!("intervention record {index}: a missed critical intervention requires a critical opportunity");
        }
        if record.appropriate {
            appropriate_count += 1;
        }
        if record.unnecessary {
            unnecessary_count += 1;
        }
        if record.critical_opportunity {
            critical_opportunity_count += 1;
        }
        if record.critical_missed {
            critical_missed_count += 1;
        }
        if let Some(regret) = record.counterfactual_regret {
            if !regret.is_finite() || !(0.0..=1.0).contains(&regret) {
                bail!("intervention record {index}: counterfactual regret must be finite and in [0, 1]");
            }
            regret_sum += regret;
            regret_count += 1;
        }
    }

    let decision_count = records.len();
    Ok(InterventionSummary {
        decision_count,
        critical_opportunity_count,
        counterfactually_identifiable_count: regret_count,
        intervention_appropriateness: (decision_count > 0)
            .then_some(appropriate_count as f64 / decision_count as f64),
        unnecessary_intervention_rate: (decision_count > 0)
            .then_some(unnecessary_count as f64 / decision_count as f64),
        missed_critical_intervention_rate: (critical_opportunity_count > 0)
            .then_some(critical_missed_count as f64 / critical_opportunity_count as f64),
        mean_intervention_regret: (regret_count > 0)
            .then_some(regret_sum / regret_count as f64),
    })
}

/// Counterfactual marginal impact of a contribution on a bounded outcome.
/// Returns None when the replay is not causally identifiable (for example,
/// removing the event invalidates later dependent actions). This is not a legal
/// authorship or ownership measure.
pub fn marginal_contribution_impact(
    full_quality: f64,
    replay_quality: f64,
    replay_identifiable: bool,
) -> Result<Option<f64>> {
    for (name, score) in [("full_quality", full_quality), ("replay_quality", replay_quality)] {
        if !score.is_finite() || !(0.0..=1.0).contains(&score) {
            bail!("{name} must be finite and in [0, 1]");
        }
    }
    if !replay_identifiable {
        return Ok(None);
    }
    Ok(Some(full_quality - replay_quality))
}

/// Paired cluster bootstrap for participant-level means. Each array must contain
/// one pre-aggregated value per participant, in matched participant order.
/// This does not replace a multi-level participant/task model where task-level
/// records are available; it avoids treating turns as independent samples.
pub fn paired_cluster_bootstrap_difference(
    reference: &[f64],
    treatment: &[f64],
    draws: usize,
    seed: u64,
) -> Result<BootstrapEstimate> {
    if reference.len() != treatment.len() || reference.len() < 2 {
        bail!("paired bootstrap requires equal arrays with at least two participant clusters");
    }
    if draws < 1_000 {
        bail!("paired bootstrap requires at least 1,000 draws");
    }
    for (index, (a, b)) in reference.iter().zip(treatment).enumerate() {
        if !a.is_finite() || !b.is_finite() {
            bail!("non-finite value in participant cluster {index}");
        }
    }

    let differences: Vec<f64> = treatment
        .iter()
        .zip(reference)
        .map(|(treatment, reference)| treatment - reference)
        .collect();
    let observed_mean = mean(&differences);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut samples = Vec::with_capacity(draws);

    for _ in 0..draws {
        let mut sum = 0.0;
        for _ in 0..differences.len() {
            let index = rng.gen_range(0..differences.len());
            sum += differences[index];
        }
        samples.push(sum / differences.len() as f64);
    }
    samples.sort_by(f64::total_cmp);

    Ok(BootstrapEstimate {
        mean_difference: observed_mean,
        ci_95_lower: percentile(&samples, 0.025),
        ci_95_upper: percentile(&samples, 0.975),
        draws,
        seed,
        participant_clusters: differences.len(),
    })
}

pub fn synergy_bootstrap(
    human_only: &[f64],
    ai_only: &[f64],
    collaborative: &[f64],
    draws: usize,
    seed: u64,
) -> Result<BootstrapEstimate> {
    if human_only.len() != ai_only.len() || human_only.len() != collaborative.len() {
        bail!("synergy bootstrap requires matched human, AI, and collaborative observations");
    }
    if human_only.len() < 2 {
        bail!("synergy bootstrap requires at least two participant/task clusters");
    }
    let mut solo_best = Vec::with_capacity(human_only.len());
    let mut joint = Vec::with_capacity(human_only.len());
    for index in 0..human_only.len() {
        solo_best.push(human_only[index].max(ai_only[index]));
        joint.push(collaborative[index]);
    }
    paired_cluster_bootstrap_difference(&solo_best, &joint, draws, seed)
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn percentile(sorted: &[f64], quantile: f64) -> f64 {
    let index = (quantile * (sorted.len() - 1) as f64).round() as usize;
    sorted[index]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positive_synergy_means_joint_exceeds_best_solo() {
        assert!((synergy_delta(0.6, 0.7, 0.8).unwrap() - 0.1).abs() < 1e-12);
    }

    #[test]
    fn negative_synergy_is_a_collaboration_gap() {
        let delta = synergy_delta(0.7, 0.8, 0.6).unwrap();
        assert!((delta + 0.2).abs() < 1e-12);
        assert!((collaboration_gap(0.8, 0.6).unwrap() - 0.2).abs() < 1e-12);
    }

    #[test]
    fn rejects_unmatched_or_out_of_range_measurements() {
        assert!(synergy_delta(0.4, 1.1, 0.8).is_err());
        assert!(partner_order_sensitivity(0.6, f64::NAN).is_err());
    }

    #[test]
    fn handoff_rate_handles_no_attempts_without_inventing_zero() {
        assert_eq!(handoff_recovery_rate(0, 0).unwrap(), None);
        assert_eq!(handoff_recovery_rate(3, 4).unwrap(), Some(0.75));
        assert!(handoff_recovery_rate(5, 4).is_err());
    }

    #[test]
    fn intervention_metrics_use_distinct_denominators_and_blinded_labels() {
        let records = [
            InterventionRecord {
                appropriate: true,
                unnecessary: false,
                critical_opportunity: true,
                critical_missed: false,
                counterfactual_regret: Some(0.1),
            },
            InterventionRecord {
                appropriate: false,
                unnecessary: true,
                critical_opportunity: false,
                critical_missed: false,
                counterfactual_regret: Some(0.3),
            },
            InterventionRecord {
                appropriate: false,
                unnecessary: false,
                critical_opportunity: true,
                critical_missed: true,
                counterfactual_regret: None,
            },
        ];
        let summary = summarize_interventions(&records).unwrap();
        assert_eq!(summary.decision_count, 3);
        assert_eq!(summary.critical_opportunity_count, 2);
        assert_eq!(summary.counterfactually_identifiable_count, 2);
        assert!((summary.intervention_appropriateness.unwrap() - 1.0 / 3.0).abs() < 1e-12);
        assert!((summary.unnecessary_intervention_rate.unwrap() - 1.0 / 3.0).abs() < 1e-12);
        assert!((summary.missed_critical_intervention_rate.unwrap() - 0.5).abs() < 1e-12);
        assert!((summary.mean_intervention_regret.unwrap() - 0.2).abs() < 1e-12);
    }

    #[test]
    fn intervention_metrics_do_not_invent_zero_for_missing_denominators() {
        let summary = summarize_interventions(&[]).unwrap();
        assert_eq!(summary.intervention_appropriateness, None);
        assert_eq!(summary.unnecessary_intervention_rate, None);
        assert_eq!(summary.missed_critical_intervention_rate, None);
        assert_eq!(summary.mean_intervention_regret, None);
    }

    #[test]
    fn intervention_metric_rejects_inconsistent_and_unbounded_adjudications() {
        let unnecessary_and_appropriate = [InterventionRecord {
            appropriate: true,
            unnecessary: true,
            critical_opportunity: false,
            critical_missed: false,
            counterfactual_regret: None,
        }];
        assert!(summarize_interventions(&unnecessary_and_appropriate).is_err());

        let missed_without_opportunity = [InterventionRecord {
            appropriate: false,
            unnecessary: false,
            critical_opportunity: false,
            critical_missed: true,
            counterfactual_regret: None,
        }];
        assert!(summarize_interventions(&missed_without_opportunity).is_err());

        let invalid_regret = [InterventionRecord {
            appropriate: false,
            unnecessary: false,
            critical_opportunity: false,
            critical_missed: false,
            counterfactual_regret: Some(1.5),
        }];
        assert!(summarize_interventions(&invalid_regret).is_err());
    }

    #[test]
    fn non_identifiable_counterfactual_is_reported_as_unavailable() {
        assert_eq!(marginal_contribution_impact(0.8, 0.6, false).unwrap(), None);
        assert_eq!(marginal_contribution_impact(0.8, 0.6, true).unwrap(), Some(0.2));
        assert!(marginal_contribution_impact(1.1, 0.6, true).is_err());
    }

    #[test]
    fn paired_bootstrap_is_seed_reproducible() {
        let a = [0.3, 0.4, 0.5, 0.6];
        let b = [0.5, 0.6, 0.7, 0.8];
        let first = paired_cluster_bootstrap_difference(&a, &b, 2_000, 42).unwrap();
        let second = paired_cluster_bootstrap_difference(&a, &b, 2_000, 42).unwrap();
        assert!((first.mean_difference - 0.2).abs() < 1e-12);
        assert_eq!(first.ci_95_lower, second.ci_95_lower);
        assert_eq!(first.ci_95_upper, second.ci_95_upper);
        assert!(first.ci_95_lower <= first.mean_difference);
        assert!(first.ci_95_upper >= first.mean_difference);
    }
}
