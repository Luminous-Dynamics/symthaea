use anyhow::{Context, Result, bail};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::collections::{BTreeMap, HashSet};
use serde::{Deserialize, Serialize};

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
            "record {index}: an unnecessary intervention cannot be appropriate"
        );
        }
        if record.critical_missed && !record.critical_opportunity {
            bail!(
            "record {index}: a missed critical intervention requires an opportunity"
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
                    "record {index}: counterfactual regret must be finite and in [0, 1]"
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
            "capability amplification needs matched scores and two participants per arm"
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


/// One participant's independent pre/post assessment within a preregistered
/// baseline-capability stratum. Strata must be derived from baseline task data
/// before condition outcomes are inspected; do not infer sensitive identity.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StratifiedCapabilityObservation {
    pub participant_id: String,
    pub baseline_stratum: String,
    pub condition: String,
    pub pre_score: f64,
    pub post_score: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct CapabilityStratumEstimate {
    pub baseline_stratum: String,
    pub collaborative_participants: usize,
    pub control_participants: usize,
    pub mean_gain_difference: f64,
    pub ci_95_lower: f64,
    pub ci_95_upper: f64,
    pub collaborative_negative_gain_rate: f64,
    pub control_negative_gain_rate: f64,
    /// Difference in the fraction of participants whose independent score fell.
    pub negative_gain_rate_difference: f64,
    pub negative_gain_rate_difference_ci_95_lower: f64,
    pub negative_gain_rate_difference_ci_95_upper: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct StratifiedCapabilitySummary {
    pub strata: Vec<CapabilityStratumEstimate>,
    pub seed: u64,
    pub draws_per_stratum: usize,
}

/// Report capability amplification and negative-gain rates separately within
/// preregistered baseline-capability strata. This exposes distributional
/// effects that a pooled average can conceal. It does not classify participants
/// by demographic or sensitive identity.
pub fn summarize_capability_amplification_by_stratum(
    observations: &[StratifiedCapabilityObservation],
    draws: usize,
    seed: u64,
) -> Result<StratifiedCapabilitySummary> {
    if draws < 1_000 {
        bail!("stratified capability bootstrap requires at least 1,000 draws");
    }
    if observations.is_empty() {
        bail!("stratified capability summary requires observations");
    }

    #[derive(Default)]
    struct Arm {
        pre: Vec<f64>,
        post: Vec<f64>,
        negative_gains: usize,
    }
    #[derive(Default)]
    struct Stratum {
        collaborative: Arm,
        control: Arm,
    }

    let mut participants = HashSet::new();
    let mut groups: BTreeMap<String, Stratum> = BTreeMap::new();

    for (index, observation) in observations.iter().enumerate() {
        require_nonempty_stratum("participant_id", &observation.participant_id)?;
        require_nonempty_stratum("baseline_stratum", &observation.baseline_stratum)?;
        if !participants.insert(observation.participant_id.as_str()) {
            bail!(
                "duplicate participant_id {:?} in stratified capability observations",
                observation.participant_id
            );
        }
        validate_learning_score("pre_score", index, observation.pre_score)?;
        validate_learning_score("post_score", index, observation.post_score)?;

        let arm = match observation.condition.as_str() {
            "collaborative" => &mut groups
                .entry(observation.baseline_stratum.clone())
                .or_default()
                .collaborative,
            "control" => &mut groups
                .entry(observation.baseline_stratum.clone())
                .or_default()
                .control,
            other => bail!("unsupported capability condition {other:?}"),
        };
        arm.pre.push(observation.pre_score);
        arm.post.push(observation.post_score);
        arm.negative_gains += (observation.post_score < observation.pre_score) as usize;
    }

    let mut estimates = Vec::with_capacity(groups.len());
    for (index, (stratum, group)) in groups.into_iter().enumerate() {
        if group.collaborative.pre.len() < 2 || group.control.pre.len() < 2 {
            bail!(
                "baseline stratum {:?} requires at least two participants in each arm",
                stratum
            );
        }
        let estimate = participant_capability_amplification(
            &group.collaborative.pre,
            &group.collaborative.post,
            &group.control.pre,
            &group.control.post,
            draws,
            seed.wrapping_add(index as u64),
        )?;
        let collaborative_n = group.collaborative.pre.len();
        let control_n = group.control.pre.len();
        let collaborative_negative_gain_rate =
            group.collaborative.negative_gains as f64 / collaborative_n as f64;
        let control_negative_gain_rate =
            group.control.negative_gains as f64 / control_n as f64;
        let negative_gain_rate_difference =
            collaborative_negative_gain_rate - control_negative_gain_rate;

        let collaborative_negative = group
            .collaborative
            .pre
            .iter()
            .zip(&group.collaborative.post)
            .map(|(pre, post)| (*post < *pre) as u8 as f64)
            .collect::<Vec<_>>();
        let control_negative = group
            .control
            .pre
            .iter()
            .zip(&group.control.post)
            .map(|(pre, post)| (*post < *pre) as u8 as f64)
            .collect::<Vec<_>>();
        let mut rng = StdRng::seed_from_u64(
            seed
                .wrapping_add(index as u64)
                .wrapping_add(0x9e37_79b9_7f4a_7c15),
        );
        let mut negative_rate_samples = Vec::with_capacity(draws);
        for _ in 0..draws {
            let collaborative_rate = (0..collaborative_negative.len())
                .map(|_| collaborative_negative[rng.gen_range(0..collaborative_negative.len())])
                .sum::<f64>()
                / collaborative_n as f64;
            let control_rate = (0..control_negative.len())
                .map(|_| control_negative[rng.gen_range(0..control_negative.len())])
                .sum::<f64>()
                / control_n as f64;
            negative_rate_samples.push(collaborative_rate - control_rate);
        }
        negative_rate_samples.sort_by(f64::total_cmp);

        estimates.push(CapabilityStratumEstimate {
            baseline_stratum: stratum,
            collaborative_participants: collaborative_n,
            control_participants: control_n,
            mean_gain_difference: estimate.mean_gain_difference,
            ci_95_lower: estimate.ci_95_lower,
            ci_95_upper: estimate.ci_95_upper,
            collaborative_negative_gain_rate,
            control_negative_gain_rate,
            negative_gain_rate_difference,
            negative_gain_rate_difference_ci_95_lower: percentile(
                &negative_rate_samples,
                0.025,
            ),
            negative_gain_rate_difference_ci_95_upper: percentile(
                &negative_rate_samples,
                0.975,
            ),
        });
    }

    Ok(StratifiedCapabilitySummary {
        strata: estimates,
        seed,
        draws_per_stratum: draws,
    })
}

fn require_nonempty_stratum(label: &str, value: &str) -> Result<()> {
    if value.trim().is_empty() {
        bail!("{label} must not be empty");
    }
    Ok(())
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


/// One independently adjudicated claim with a confidence estimate made before
/// feedback. Confidence is the probability assigned to the claim being correct.
#[derive(Debug, Clone, Copy)]
pub struct ConfidenceObservation {
    pub confidence: f64,
    pub correct: bool,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct ConfidenceSummary {
    pub observation_count: usize,
    pub correct_count: usize,
    pub incorrect_count: usize,
    /// Mean squared probability error.
    pub brier_score: f64,
    /// Fixed-width, count-weighted calibration error.
    pub expected_calibration_error: f64,
    /// AUC for confidence discriminating correct from incorrect judgments.
    /// None when the sample contains only correct or only incorrect judgments.
    pub metacognitive_sensitivity_auc: Option<f64>,
    /// Actual coverage after rounding the requested coverage up to whole cases.
    pub selective_coverage: f64,
    /// Accuracy among the highest-confidence cases at the requested coverage.
    pub selective_accuracy: f64,
}

/// Compute calibration, metacognitive sensitivity, and selective accuracy.
///
/// Calibration and sensitivity are distinct: a system may be calibrated on
/// average while failing to rank its own errors below its correct judgments.
/// Confidence observations must be elicited before ground-truth feedback.
pub fn summarize_confidence(
    observations: &[ConfidenceObservation],
    calibration_bins: usize,
    selective_coverage: f64,
) -> Result<ConfidenceSummary> {
    if observations.is_empty() {
        bail!("confidence summary requires at least one observation");
    }
    if !(2..=100).contains(&calibration_bins) {
        bail!("calibration_bins must be in 2..=100");
    }
    if !selective_coverage.is_finite() || !(0.0..=1.0).contains(&selective_coverage)
        || selective_coverage == 0.0
    {
        bail!("selective_coverage must be finite and in (0, 1]");
    }

    let mut brier_sum = 0.0;
    let mut bin_counts = vec![0usize; calibration_bins];
    let mut bin_confidence = vec![0.0; calibration_bins];
    let mut bin_correct = vec![0usize; calibration_bins];
    let mut correct_count = 0usize;

    for (index, observation) in observations.iter().enumerate() {
        if !observation.confidence.is_finite()
            || !(0.0..=1.0).contains(&observation.confidence)
        {
            bail!("confidence[{index}] must be finite and in [0, 1]");
        }
        let target = if observation.correct { 1.0 } else { 0.0 };
        let error = observation.confidence - target;
        brier_sum += error * error;

        let bin = ((observation.confidence * calibration_bins as f64).floor() as usize)
            .min(calibration_bins - 1);
        bin_counts[bin] += 1;
        bin_confidence[bin] += observation.confidence;
        if observation.correct {
            bin_correct[bin] += 1;
            correct_count += 1;
        }
    }

    let count = observations.len();
    let brier_score = brier_sum / count as f64;
    let mut expected_calibration_error = 0.0;
    for bin in 0..calibration_bins {
        if bin_counts[bin] == 0 {
            continue;
        }
        let bin_size = bin_counts[bin] as f64;
        let mean_confidence = bin_confidence[bin] / bin_size;
        let empirical_accuracy = bin_correct[bin] as f64 / bin_size;
        expected_calibration_error +=
            bin_size / count as f64 * (mean_confidence - empirical_accuracy).abs();
    }

    let incorrect_count = count - correct_count;
    let mut ranked = observations
        .iter()
        .map(|item| (item.confidence, item.correct))
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| left.0.total_cmp(&right.0));

    let metacognitive_sensitivity_auc = if correct_count == 0 || incorrect_count == 0 {
        None
    } else {
        let mut positive_rank_sum = 0.0;
        let mut start = 0usize;
        while start < ranked.len() {
            let mut end = start + 1;
            while end < ranked.len() && ranked[end].0 == ranked[start].0 {
                end += 1;
            }
            // Average one-based rank gives ties half credit in the AUC.
            let average_rank = (start + 1 + end) as f64 / 2.0;
            let correct_in_tie = ranked[start..end]
                .iter()
                .filter(|(_, correct)| *correct)
                .count();
            positive_rank_sum += average_rank * correct_in_tie as f64;
            start = end;
        }
        let positive = correct_count as f64;
        let negative = incorrect_count as f64;
        Some(
            (positive_rank_sum - positive * (positive + 1.0) / 2.0)
                / (positive * negative),
        )
    };

    let mut selective = observations.to_vec();
    selective.sort_by(|left, right| right.confidence.total_cmp(&left.confidence));
    let selected_count = ((count as f64 * selective_coverage).ceil() as usize)
        .max(1)
        .min(count);
    let cutoff_confidence = selective[selected_count - 1].confidence;
    let strictly_above = selective
        .iter()
        .filter(|item| item.confidence > cutoff_confidence)
        .collect::<Vec<_>>();
    let tied_at_cutoff = selective
        .iter()
        .filter(|item| item.confidence == cutoff_confidence)
        .collect::<Vec<_>>();
    let needed_from_tie = selected_count - strictly_above.len();
    let correct_above = strictly_above
        .iter()
        .filter(|item| item.correct)
        .count() as f64;
    let correct_at_tie = tied_at_cutoff
        .iter()
        .filter(|item| item.correct)
        .count() as f64;
    // Fractional tie handling makes the result invariant to input ordering.
    let selected_correct = correct_above
        + correct_at_tie * needed_from_tie as f64 / tied_at_cutoff.len() as f64;

    Ok(ConfidenceSummary {
        observation_count: count,
        correct_count,
        incorrect_count,
        brier_score,
        expected_calibration_error,
        metacognitive_sensitivity_auc,
        selective_coverage: selected_count as f64 / count as f64,
        selective_accuracy: selected_correct / selected_count as f64,
    })
}

/// An independently coded opportunity to observe human cognitive engagement.
/// These are process annotations, not personality or ability labels.
#[derive(Debug, Clone, Copy)]
pub struct CognitiveProcessObservation {
    pub human_planned: bool,
    pub human_attempted_before_ai: bool,
    pub human_monitored: bool,
    pub ai_supplied_core_step: bool,
    pub human_verified_ai_step: bool,
    pub human_reflected: bool,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct CognitiveProcessSummary {
    pub opportunity_count: usize,
    pub ai_core_step_count: usize,
    pub human_planning_rate: Option<f64>,
    pub human_first_attempt_rate: Option<f64>,
    pub human_monitoring_rate: Option<f64>,
    pub human_reflection_rate: Option<f64>,
    /// AI supplied a core step before a participant attempt / all opportunities.
    pub ai_core_step_substitution_rate: Option<f64>,
    /// Participant independently checked the AI's core steps / AI core steps.
    pub human_verification_rate: Option<f64>,
}

/// Aggregate observable learning-process indicators without treating tool use
/// or delegation as inherently harmful. Labels require blinded trace coding.
pub fn summarize_cognitive_process(
    observations: &[CognitiveProcessObservation],
) -> Result<CognitiveProcessSummary> {
    if observations.is_empty() {
        return Ok(CognitiveProcessSummary {
            opportunity_count: 0,
            ai_core_step_count: 0,
            human_planning_rate: None,
            human_first_attempt_rate: None,
            human_monitoring_rate: None,
            human_reflection_rate: None,
            ai_core_step_supply_rate: None,
            ai_core_step_substitution_rate: None,
            human_verification_rate: None,
        });
    }

    let mut planned = 0usize;
    let mut attempted_first = 0usize;
    let mut monitored = 0usize;
    let mut reflected = 0usize;
    let mut ai_core_steps = 0usize;
    let mut ai_substitution = 0usize;
    let mut verified_ai_steps = 0usize;

    for (index, observation) in observations.iter().enumerate() {
        if observation.human_verified_ai_step && !observation.ai_supplied_core_step {
            bail!(
                "process observation {index}: verification requires an AI-supplied step"
            );
        }
        planned += observation.human_planned as usize;
        attempted_first += observation.human_attempted_before_ai as usize;
        monitored += observation.human_monitored as usize;
        reflected += observation.human_reflected as usize;
        if observation.ai_supplied_core_step {
            ai_core_steps += 1;
            verified_ai_steps += observation.human_verified_ai_step as usize;
            ai_substitution += (!observation.human_attempted_before_ai) as usize;
        }
    }

    let rate = |numerator: usize, denominator: usize| {
        (denominator > 0).then_some(numerator as f64 / denominator as f64)
    };

    Ok(CognitiveProcessSummary {
        opportunity_count: observations.len(),
        ai_core_step_count: ai_core_steps,
        human_planning_rate: rate(planned, observations.len()),
        human_first_attempt_rate: rate(attempted_first, observations.len()),
        human_monitoring_rate: rate(monitored, observations.len()),
        human_reflection_rate: rate(reflected, observations.len()),
        ai_core_step_supply_rate: rate(ai_core_steps, observations.len()),
        ai_core_step_substitution_rate: rate(ai_substitution, ai_core_steps),
        human_verification_rate: rate(verified_ai_steps, ai_core_steps),
    })
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
    fn confidence_metrics_separate_calibration_from_metacognitive_sensitivity() {
        let observations = [
            ConfidenceObservation { confidence: 0.9, correct: true },
            ConfidenceObservation { confidence: 0.8, correct: true },
            ConfidenceObservation { confidence: 0.2, correct: false },
            ConfidenceObservation { confidence: 0.1, correct: false },
        ];
        let summary = summarize_confidence(&observations, 5, 0.5).unwrap();
        assert_eq!(summary.observation_count, 4);
        assert_eq!(summary.correct_count, 2);
        assert_eq!(summary.incorrect_count, 2);
        assert!((summary.metacognitive_sensitivity_auc.unwrap() - 1.0).abs() < 1e-12);
        assert!((summary.selective_accuracy - 1.0).abs() < 1e-12);
        assert!((summary.selective_coverage - 0.5).abs() < 1e-12);
        assert!(summary.brier_score < 0.05);
    }

    #[test]
    fn selective_accuracy_is_invariant_to_ordering_of_cutoff_ties() {
        let first = [
            ConfidenceObservation { confidence: 0.9, correct: true },
            ConfidenceObservation { confidence: 0.8, correct: false },
            ConfidenceObservation { confidence: 0.8, correct: true },
            ConfidenceObservation { confidence: 0.1, correct: false },
        ];
        let reversed_tie_order = [
            ConfidenceObservation { confidence: 0.9, correct: true },
            ConfidenceObservation { confidence: 0.8, correct: true },
            ConfidenceObservation { confidence: 0.8, correct: false },
            ConfidenceObservation { confidence: 0.1, correct: false },
        ];
        let a = summarize_confidence(&first, 5, 0.5).unwrap();
        let b = summarize_confidence(&reversed_tie_order, 5, 0.5).unwrap();
        assert!((a.selective_accuracy - 0.75).abs() < 1e-12);
        assert!((a.selective_accuracy - b.selective_accuracy).abs() < 1e-12);
    }

    #[test]
    fn confidence_auc_gives_half_credit_for_tied_confidence() {
        let observations = [
            ConfidenceObservation { confidence: 0.5, correct: true },
            ConfidenceObservation { confidence: 0.5, correct: false },
        ];
        let summary = summarize_confidence(&observations, 2, 1.0).unwrap();
        assert_eq!(summary.metacognitive_sensitivity_auc, Some(0.5));
        assert!((summary.expected_calibration_error).abs() < 1e-12);
    }

    #[test]
    fn confidence_sensitivity_is_unavailable_without_both_outcome_classes() {
        let observations = [
            ConfidenceObservation { confidence: 0.7, correct: true },
            ConfidenceObservation { confidence: 0.8, correct: true },
        ];
        let summary = summarize_confidence(&observations, 4, 0.8).unwrap();
        assert_eq!(summary.metacognitive_sensitivity_auc, None);
    }

    #[test]
    fn confidence_metrics_reject_invalid_bins_coverage_and_scores() {
        let valid = [ConfidenceObservation { confidence: 0.7, correct: true }];
        assert!(summarize_confidence(&valid, 1, 0.8).is_err());
        assert!(summarize_confidence(&valid, 4, 0.0).is_err());
        let invalid = [ConfidenceObservation { confidence: 1.1, correct: true }];
        assert!(summarize_confidence(&invalid, 4, 0.8).is_err());
    }

    #[test]
    fn cognitive_process_metrics_distinguish_scaffolding_from_substitution() {
        let observations = [
            CognitiveProcessObservation {
                human_planned: true,
                human_attempted_before_ai: true,
                human_monitored: true,
                ai_supplied_core_step: true,
                human_verified_ai_step: true,
                human_reflected: true,
            },
            CognitiveProcessObservation {
                human_planned: false,
                human_attempted_before_ai: false,
                human_monitored: false,
                ai_supplied_core_step: true,
                human_verified_ai_step: false,
                human_reflected: false,
            },
            CognitiveProcessObservation {
                human_planned: true,
                human_attempted_before_ai: true,
                human_monitored: true,
                ai_supplied_core_step: false,
                human_verified_ai_step: false,
                human_reflected: true,
            },
        ];
        let summary = summarize_cognitive_process(&observations).unwrap();
        assert_eq!(summary.opportunity_count, 3);
        assert_eq!(summary.ai_core_step_count, 2);
        assert!((summary.human_first_attempt_rate.unwrap() - 2.0 / 3.0).abs() < 1e-12);
        assert!((summary.ai_core_step_supply_rate.unwrap() - 2.0 / 3.0).abs() < 1e-12);
        assert!((summary.ai_core_step_substitution_rate.unwrap() - 0.5).abs() < 1e-12);
        assert_eq!(summary.human_verification_rate, Some(0.5));
    }

    #[test]
    fn cognitive_process_metrics_report_missing_denominators_as_unavailable() {
        let empty = summarize_cognitive_process(&[]).unwrap();
        assert_eq!(empty.human_planning_rate, None);
        assert_eq!(empty.human_verification_rate, None);
        let no_ai = [CognitiveProcessObservation {
            human_planned: true,
            human_attempted_before_ai: true,
            human_monitored: true,
            ai_supplied_core_step: false,
            human_verified_ai_step: false,
            human_reflected: true,
        }];
        assert_eq!(
            summarize_cognitive_process(&no_ai).unwrap().human_verification_rate,
            None
        );
    }

    #[test]
    fn cognitive_process_metrics_reject_impossible_verification_labels() {
        let invalid = [CognitiveProcessObservation {
            human_planned: false,
            human_attempted_before_ai: false,
            human_monitored: false,
            ai_supplied_core_step: false,
            human_verified_ai_step: true,
            human_reflected: false,
        }];
        assert!(summarize_cognitive_process(&invalid).is_err());
    }

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
    fn stratified_capability_amplification_exposes_unequal_benefit_and_harm() {
        let mut observations = Vec::new();
        for (i, (pre, post)) in [
            (0.1, 0.2),
            (0.2, 0.25),
            (0.3, 0.4),
        ]
        .into_iter()
        .enumerate()
        {
            observations.push(StratifiedCapabilityObservation {
                participant_id: format!("low-collab-{i}"),
                baseline_stratum: "low_baseline".to_owned(),
                condition: "collaborative".to_owned(),
                pre_score: pre,
                post_score: post,
            });
        }
        for (i, (pre, post)) in [
            (0.1, 0.15),
            (0.2, 0.3),
            (0.3, 0.35),
        ]
        .into_iter()
        .enumerate()
        {
            observations.push(StratifiedCapabilityObservation {
                participant_id: format!("low-control-{i}"),
                baseline_stratum: "low_baseline".to_owned(),
                condition: "control".to_owned(),
                pre_score: pre,
                post_score: post,
            });
        }
        for (i, (pre, post)) in [
            (0.7, 0.6),
            (0.75, 0.8),
            (0.8, 0.82),
        ]
        .into_iter()
        .enumerate()
        {
            observations.push(StratifiedCapabilityObservation {
                participant_id: format!("high-collab-{i}"),
                baseline_stratum: "high_baseline".to_owned(),
                condition: "collaborative".to_owned(),
                pre_score: pre,
                post_score: post,
            });
        }
        for (i, (pre, post)) in [
            (0.7, 0.75),
            (0.75, 0.8),
            (0.8, 0.82),
        ]
        .into_iter()
        .enumerate()
        {
            observations.push(StratifiedCapabilityObservation {
                participant_id: format!("high-control-{i}"),
                baseline_stratum: "high_baseline".to_owned(),
                condition: "control".to_owned(),
                pre_score: pre,
                post_score: post,
            });
        }

        let first = summarize_capability_amplification_by_stratum(&observations, 2_000, 42)
            .unwrap();
        let second = summarize_capability_amplification_by_stratum(&observations, 2_000, 42)
            .unwrap();
        assert_eq!(first.strata.len(), 2);
        assert_eq!(first.strata[0].baseline_stratum, "high_baseline");
        assert_eq!(first.strata[1].baseline_stratum, "low_baseline");
        assert!((first.strata[0].mean_gain_difference + 0.05).abs() < 1e-12);
        assert!((first.strata[0].collaborative_negative_gain_rate - 1.0 / 3.0).abs() < 1e-12);
        assert_eq!(first.strata[0].control_negative_gain_rate, 0.0);
        assert!(
            first.strata[0].negative_gain_rate_difference_ci_95_lower
                <= first.strata[0].negative_gain_rate_difference
        );
        assert!(
            first.strata[0].negative_gain_rate_difference_ci_95_upper
                >= first.strata[0].negative_gain_rate_difference
        );
        assert_eq!(
            first.strata[0].negative_gain_rate_difference_ci_95_lower,
            second.strata[0].negative_gain_rate_difference_ci_95_lower
        );
        assert_eq!(
            first.strata[0].ci_95_lower,
            second.strata[0].ci_95_lower
        );
        assert_eq!(
            first.strata[1].ci_95_upper,
            second.strata[1].ci_95_upper
        );
    }

    #[test]
    fn stratified_capability_summary_rejects_duplicate_or_underfilled_participants() {
        let duplicate = [
            StratifiedCapabilityObservation {
                participant_id: "p1".to_owned(),
                baseline_stratum: "low".to_owned(),
                condition: "collaborative".to_owned(),
                pre_score: 0.2,
                post_score: 0.3,
            },
            StratifiedCapabilityObservation {
                participant_id: "p1".to_owned(),
                baseline_stratum: "low".to_owned(),
                condition: "control".to_owned(),
                pre_score: 0.2,
                post_score: 0.25,
            },
        ];
        assert!(
            summarize_capability_amplification_by_stratum(&duplicate, 1_000, 5).is_err()
        );

        let underfilled = [
            StratifiedCapabilityObservation {
                participant_id: "c1".to_owned(),
                baseline_stratum: "low".to_owned(),
                condition: "collaborative".to_owned(),
                pre_score: 0.2,
                post_score: 0.3,
            },
            StratifiedCapabilityObservation {
                participant_id: "c2".to_owned(),
                baseline_stratum: "low".to_owned(),
                condition: "collaborative".to_owned(),
                pre_score: 0.3,
                post_score: 0.4,
            },
        ];
        assert!(
            summarize_capability_amplification_by_stratum(&underfilled, 1_000, 5).is_err()
        );
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
        let impact = marginal_contribution_impact(0.8, 0.6, true).unwrap().unwrap();
        assert!((impact - 0.2).abs() < 1e-12);
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
