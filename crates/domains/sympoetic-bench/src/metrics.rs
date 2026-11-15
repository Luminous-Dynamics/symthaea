use anyhow::{Context, Result, bail};
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
            bail!(
            "intervention record {index}: an unnecessary intervention cannot be adjudicated appropriate"
        );
        }
        if record.critical_missed && !record.critical_opportunity {
            bail!(
            "intervention record {index}: a missed critical intervention requires a critical opportunity"
        );
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
                bail!(
                    "intervention record {index}: counterfactual regret must be finite and in [0, 1]"
                );
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


/// Exact Shapley attribution over a complete, bounded coalition-value table.
///
/// Coalition index is a bit mask: index 0 is the empty coalition and index
/// (1 << n) - 1 is the full coalition. Every coalition must be identifiable.
/// Missing counterfactuals cause an error rather than being treated as zero.
/// This is a contribution-allocation measure, not a legal authorship measure.
pub fn exact_shapley_values(coalition_values: &[Option<f64>]) -> Result<Vec<f64>> {
    if coalition_values.len() < 2 || !coalition_values.len().is_power_of_two() {
        bail!("Shapley attribution requires a complete 2^n coalition table for n >= 1");
    }
    let player_count = coalition_values.len().trailing_zeros() as usize;
    if player_count > 12 {
        bail!(
            "exact Shapley attribution is limited to 12 players; use a sampled estimator above that"
        );
    }

    let mut values = Vec::with_capacity(coalition_values.len());
    for (coalition, value) in coalition_values.iter().enumerate() {
        let value = value.as_ref().with_context(|| {
            format!("coalition {coalition} is non-identifiable; do not impute its value")
        })?;
        if !value.is_finite() || !(0.0..=1.0).contains(value) {
            bail!("coalition {coalition} value must be finite and in [0, 1]");
        }
        values.push(*value);
    }

    let mut shapley = vec![0.0; player_count];
    for player in 0..player_count {
        let bit = 1usize << player;
        for coalition in 0..values.len() {
            if coalition & bit != 0 {
                continue;
            }
            let coalition_size = coalition.count_ones() as usize;
            let weight = 1.0
                / (player_count as f64
                    * binomial_coefficient(player_count - 1, coalition_size));
            shapley[player] +=
                weight * (values[coalition | bit] - values[coalition]);
        }
    }
    Ok(shapley)
}

fn binomial_coefficient(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    let k = k.min(n - k);
    (1..=k).fold(1.0, |value, index| {
        value * (n - k + index) as f64 / index as f64
    })
}

/// Difference-in-differences estimate of participant capability amplification.
///
/// Each arm contains one pre/post score per participant. The estimator first
/// computes within-participant gain, then bootstraps participants independently
/// within the collaborative and control arms. Positive values mean the
/// collaboration condition improved independent capability more than the
/// matched control. Scores must be measured on equivalent, non-identical test
/// forms with the same bounded rubric.
pub fn participant_capability_amplification(
    collaborative_pre: &[f64],
    collaborative_post: &[f64],
    control_pre: &[f64],
    control_post: &[f64],
    draws: usize,
    seed: u64,
) -> Result<CapabilityAmplificationEstimate> {
    if collaborative_pre.len() != collaborative_post.len()
        || control_pre.len() != control_post.len()
        || collaborative_pre.len() < 2
        || control_pre.len() < 2
    {
        bail!(
            "capability amplification requires matched pre/post scores and at least two participants per arm"
        );
    }
    if draws < 1_000 {
        bail!("capability amplification bootstrap requires at least 1,000 draws");
    }

    let mut collaborative_gains = Vec::with_capacity(collaborative_pre.len());
    let mut control_gains = Vec::with_capacity(control_pre.len());
    for (index, (&pre, &post)) in collaborative_pre.iter().zip(collaborative_post).enumerate() {
        validate_learning_score("collaborative_pre", index, pre)?;
        validate_learning_score("collaborative_post", index, post)?;
        collaborative_gains.push(post - pre);
    }
    for (index, (&pre, &post)) in control_pre.iter().zip(control_post).enumerate() {
        validate_learning_score("control_pre", index, pre)?;
        validate_learning_score("control_post", index, post)?;
        control_gains.push(post - pre);
    }

    let observed = mean(&collaborative_gains) - mean(&control_gains);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut samples = Vec::with_capacity(draws);
    for _ in 0..draws {
        let collaborative_sample = (0..collaborative_gains.len())
            .map(|_| collaborative_gains[rng.gen_range(0..collaborative_gains.len())])
            .sum::<f64>()
            / collaborative_gains.len() as f64;
        let control_sample = (0..control_gains.len())
            .map(|_| control_gains[rng.gen_range(0..control_gains.len())])
            .sum::<f64>()
            / control_gains.len() as f64;
        samples.push(collaborative_sample - control_sample);
    }
    samples.sort_by(f64::total_cmp);

    Ok(CapabilityAmplificationEstimate {
        mean_gain_difference: observed,
        ci_95_lower: percentile(&samples, 0.025),
        ci_95_upper: percentile(&samples, 0.975),
        draws,
        seed,
        collaborative_participants: collaborative_gains.len(),
        control_participants: control_gains.len(),
    })
}

fn validate_learning_score(label: &str, index: usize, score: f64) -> Result<()> {
    if !score.is_finite() || !(0.0..=1.0).contains(&score) {
        bail!("{label}[{index}] must be finite and in [0, 1]");
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct CapabilityAmplificationEstimate {
    pub mean_gain_difference: f64,
    pub ci_95_lower: f64,
    pub ci_95_upper: f64,
    pub draws: usize,
    pub seed: u64,
    pub collaborative_participants: usize,
    pub control_participants: usize,
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
    fn exact_shapley_recovers_additive_contributions() {
        let values = (0..8)
            .map(|mask| {
                Some(
                    0.1
                        + if mask & 1 != 0 { 0.1 } else { 0.0 }
                        + if mask & 2 != 0 { 0.2 } else { 0.0 }
                        + if mask & 4 != 0 { 0.3 } else { 0.0 },
                )
            })
            .collect::<Vec<_>>();
        let result = exact_shapley_values(&values).unwrap();
        assert!((result[0] - 0.1).abs() < 1e-12);
        assert!((result[1] - 0.2).abs() < 1e-12);
        assert!((result[2] - 0.3).abs() < 1e-12);
        let allocated = result.iter().sum::<f64>();
        let coalition_gain = values[7].unwrap() - values[0].unwrap();
        assert!((allocated - coalition_gain).abs() < 1e-12);
    }

    #[test]
    fn exact_shapley_splits_a_fully_complementary_team_contribution() {
        let values = vec![Some(0.0), Some(0.0), Some(0.0), Some(1.0)];
        let result = exact_shapley_values(&values).unwrap();
        assert!((result[0] - 0.5).abs() < 1e-12);
        assert!((result[1] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn exact_shapley_rejects_non_identifiable_or_malformed_coalitions() {
        assert!(exact_shapley_values(&[Some(0.0), None]).is_err());
        assert!(exact_shapley_values(&[Some(0.0), Some(1.1)]).is_err());
        assert!(exact_shapley_values(&[Some(0.0), Some(0.5), Some(0.7)]).is_err());
    }

    #[test]
    fn capability_amplification_uses_independent_participant_cluster_bootstrap() {
        let collaborative_pre = [0.3, 0.4, 0.5];
        let collaborative_post = [0.7, 0.8, 0.9];
        let control_pre = [0.3, 0.4, 0.5, 0.6];
        let control_post = [0.4, 0.5, 0.6, 0.7];
        let first = participant_capability_amplification(
            &collaborative_pre,
            &collaborative_post,
            &control_pre,
            &control_post,
            2_000,
            42,
        )
        .unwrap();
        let second = participant_capability_amplification(
            &collaborative_pre,
            &collaborative_post,
            &control_pre,
            &control_post,
            2_000,
            42,
        )
        .unwrap();
        assert!((first.mean_gain_difference - 0.3).abs() < 1e-12);
        assert_eq!(first.ci_95_lower, second.ci_95_lower);
        assert_eq!(first.ci_95_upper, second.ci_95_upper);
        assert_eq!(first.collaborative_participants, 3);
        assert_eq!(first.control_participants, 4);
    }

    #[test]
    fn capability_amplification_rejects_unmatched_or_invalid_learning_scores() {
        assert!(participant_capability_amplification(
            &[0.2],
            &[0.4],
            &[0.2, 0.3],
            &[0.3, 0.4],
            1_000,
            7,
        )
        .is_err());
        assert!(participant_capability_amplification(
            &[0.2, 0.3],
            &[0.4, f64::NAN],
            &[0.2, 0.3],
            &[0.3, 0.4],
            1_000,
            7,
        )
        .is_err());
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
