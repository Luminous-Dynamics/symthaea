#![forbid(unsafe_code)]

//! Held-out forecast calibration for proof-oriented compute allocation.
//!
//! This deterministic synthetic experiment is not production telemetry or an
//! external benchmark result. Training and evaluation use disjoint seeds, and
//! the held-out evidence regime is absent from the fitting split.

use std::cmp::Ordering;

pub const FAMILY_COUNT: usize = 4;
pub const CONTEXT_COUNT: usize = 5;
pub const TRAIN_CONTEXT_COUNT: usize = 4;
pub const RELIABILITY_BIN_COUNT: usize = 10;
pub const DEFAULT_TRAIN_PER_CELL: usize = 128;
pub const DEFAULT_TEST_PER_CELL: usize = 64;
pub const DEFAULT_EPISODES_PER_CONTEXT: usize = 256;
pub const EPISODE_BUDGET: u32 = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(usize)]
pub enum OperationFamily {
    RefreshEvidence = 0,
    SeekIndependentEvidence = 1,
    VerifyAuthority = 2,
    RepairDerivation = 3,
}

impl OperationFamily {
    pub const ALL: [Self; FAMILY_COUNT] = [
        Self::RefreshEvidence,
        Self::SeekIndependentEvidence,
        Self::VerifyAuthority,
        Self::RepairDerivation,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RefreshEvidence => "refresh_evidence",
            Self::SeekIndependentEvidence => "seek_independent_evidence",
            Self::VerifyAuthority => "verify_authority",
            Self::RepairDerivation => "repair_derivation",
        }
    }

    pub const fn cost(self) -> u32 {
        match self {
            Self::RefreshEvidence => 1,
            Self::SeekIndependentEvidence => 2,
            Self::VerifyAuthority => 1,
            Self::RepairDerivation => 2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(usize)]
pub enum EvidenceRegime {
    Stable = 0,
    TemporalDrift = 1,
    ConflictingSources = 2,
    ThinEvidence = 3,
    HeldOutShift = 4,
}

impl EvidenceRegime {
    pub const ALL: [Self; CONTEXT_COUNT] = [
        Self::Stable,
        Self::TemporalDrift,
        Self::ConflictingSources,
        Self::ThinEvidence,
        Self::HeldOutShift,
    ];

    pub const TRAINING: [Self; TRAIN_CONTEXT_COUNT] = [
        Self::Stable,
        Self::TemporalDrift,
        Self::ConflictingSources,
        Self::ThinEvidence,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Stable => "stable",
            Self::TemporalDrift => "temporal_drift",
            Self::ConflictingSources => "conflicting_sources",
            Self::ThinEvidence => "thin_evidence",
            Self::HeldOutShift => "held_out_shift",
        }
    }

    pub const fn is_held_out(self) -> bool {
        matches!(self, Self::HeldOutShift)
    }
}

// Rows are evidence regimes; columns are operation families. The final row
// is not sampled during fitting and represents a deliberately shifted regime.
const SUCCESS_PROBABILITIES: [[f64; FAMILY_COUNT]; CONTEXT_COUNT] = [
    [0.82, 0.70, 0.79, 0.60],
    [0.38, 0.54, 0.72, 0.45],
    [0.56, 0.34, 0.77, 0.31],
    [0.63, 0.49, 0.46, 0.42],
    [0.22, 0.81, 0.37, 0.76],
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForecastMethod {
    ConstantHalf,
    GlobalBayes,
    FamilyBayes,
    ContextFamilyShrinkage,
}

impl ForecastMethod {
    pub const ALL: [Self; 4] = [
        Self::ConstantHalf,
        Self::GlobalBayes,
        Self::FamilyBayes,
        Self::ContextFamilyShrinkage,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ConstantHalf => "constant_half",
            Self::GlobalBayes => "global_bayes",
            Self::FamilyBayes => "family_bayes",
            Self::ContextFamilyShrinkage => "context_family_shrinkage",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForecastPartition {
    All,
    SeenContexts,
    HeldOutContext,
}

impl ForecastPartition {
    pub const ALL: [Self; 3] = [Self::All, Self::SeenContexts, Self::HeldOutContext];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::SeenContexts => "seen_contexts",
            Self::HeldOutContext => "held_out_context",
        }
    }

    fn includes(self, regime: EvidenceRegime) -> bool {
        match self {
            Self::All => true,
            Self::SeenContexts => !regime.is_held_out(),
            Self::HeldOutContext => regime.is_held_out(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Observation {
    pub family: OperationFamily,
    pub regime: EvidenceRegime,
    pub success: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct Counts {
    successes: u32,
    total: u32,
}

impl Counts {
    fn observe(&mut self, success: bool) {
        self.total = self.total.saturating_add(1);
        if success {
            self.successes = self.successes.saturating_add(1);
        }
    }

    fn rate(self, prior_mean: f64, prior_strength: f64) -> f64 {
        (f64::from(self.successes) + prior_mean * prior_strength)
            / (f64::from(self.total) + prior_strength)
    }
}

#[derive(Debug, Clone, PartialEq)]
struct ForecastModel {
    global: Counts,
    by_family: [Counts; FAMILY_COUNT],
    by_context_family: [[Counts; FAMILY_COUNT]; CONTEXT_COUNT],
}

impl ForecastModel {
    fn fit(training: &[Observation]) -> Self {
        let mut model = Self {
            global: Counts::default(),
            by_family: [Counts::default(); FAMILY_COUNT],
            by_context_family: [[Counts::default(); FAMILY_COUNT]; CONTEXT_COUNT],
        };

        for observation in training {
            model.global.observe(observation.success);
            model.by_family[observation.family as usize].observe(observation.success);
            model.by_context_family[observation.regime as usize][observation.family as usize]
                .observe(observation.success);
        }
        model
    }

    fn global_rate(&self) -> f64 {
        self.global.rate(0.5, 4.0)
    }

    fn family_rate(&self, family: OperationFamily) -> f64 {
        self.by_family[family as usize].rate(self.global_rate(), 8.0)
    }

    fn predict(
        &self,
        method: ForecastMethod,
        regime: EvidenceRegime,
        family: OperationFamily,
    ) -> f64 {
        match method {
            ForecastMethod::ConstantHalf => 0.5,
            ForecastMethod::GlobalBayes => self.global_rate(),
            ForecastMethod::FamilyBayes => self.family_rate(family),
            ForecastMethod::ContextFamilyShrinkage => {
                let family_prior = self.family_rate(family);
                self.by_context_family[regime as usize][family as usize].rate(family_prior, 12.0)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ForecastMetrics {
    pub method: ForecastMethod,
    pub partition: ForecastPartition,
    pub observations: usize,
    pub brier_score: f64,
    pub log_loss: f64,
    pub ece_10: f64,
    pub mean_predicted_probability: f64,
    pub observed_success_rate: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReliabilityBin {
    pub method: ForecastMethod,
    pub partition: ForecastPartition,
    pub bin: usize,
    pub observations: usize,
    pub mean_predicted_probability: Option<f64>,
    pub observed_success_rate: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionPolicy {
    FixedOrder,
    BudgetMatchedRandom,
    ConstantHalf,
    GlobalBayes,
    FamilyBayes,
    ContextFamilyShrinkage,
    OracleTrueProbabilityPerCost,
}

impl SelectionPolicy {
    pub const ALL: [Self; 7] = [
        Self::FixedOrder,
        Self::BudgetMatchedRandom,
        Self::ConstantHalf,
        Self::GlobalBayes,
        Self::FamilyBayes,
        Self::ContextFamilyShrinkage,
        Self::OracleTrueProbabilityPerCost,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::FixedOrder => "fixed_order",
            Self::BudgetMatchedRandom => "budget_matched_random",
            Self::ConstantHalf => "constant_half",
            Self::GlobalBayes => "global_bayes",
            Self::FamilyBayes => "family_bayes",
            Self::ContextFamilyShrinkage => "context_family_shrinkage",
            Self::OracleTrueProbabilityPerCost => "oracle_true_probability_per_cost",
        }
    }

    pub const fn promotion_candidate(self) -> bool {
        !matches!(self, Self::OracleTrueProbabilityPerCost)
    }

    fn forecast_method(self) -> Option<ForecastMethod> {
        match self {
            Self::ConstantHalf => Some(ForecastMethod::ConstantHalf),
            Self::GlobalBayes => Some(ForecastMethod::GlobalBayes),
            Self::FamilyBayes => Some(ForecastMethod::FamilyBayes),
            Self::ContextFamilyShrinkage => Some(ForecastMethod::ContextFamilyShrinkage),
            Self::FixedOrder | Self::BudgetMatchedRandom | Self::OracleTrueProbabilityPerCost => {
                None
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SelectionMetrics {
    pub regime: EvidenceRegime,
    pub policy: SelectionPolicy,
    pub episodes: usize,
    pub verified_proofs: usize,
    pub no_verified_proof: usize,
    pub total_spent: u64,
    pub total_budget: u64,
    pub budget_violations: usize,
    pub maximum_episode_spend: u32,
    pub total_operations: u64,
}

impl SelectionMetrics {
    pub fn success_rate(self) -> f64 {
        ratio(self.verified_proofs as u64, self.episodes as u64)
    }

    pub fn verified_proofs_per_work_unit(self) -> f64 {
        ratio(self.verified_proofs as u64, self.total_spent)
    }

    pub fn work_units_per_verified_proof(self) -> Option<f64> {
        (self.verified_proofs > 0).then(|| self.total_spent as f64 / self.verified_proofs as f64)
    }

    pub fn average_work_units_per_episode(self) -> f64 {
        ratio(self.total_spent, self.episodes as u64)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BenchmarkReport {
    pub seed: u64,
    pub training_observations: usize,
    pub evaluation_observations: usize,
    pub forecast_metrics: Vec<ForecastMetrics>,
    pub reliability_bins: Vec<ReliabilityBin>,
    pub selection_metrics: Vec<SelectionMetrics>,
}

#[derive(Debug, Clone, Copy)]
struct Episode {
    regime: EvidenceRegime,
    outcomes: [bool; FAMILY_COUNT],
    index: usize,
}

#[derive(Debug, Clone, Copy)]
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0xA0761D6478BD642F } else { seed },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }

    fn index(&mut self, upper: usize) -> usize {
        if upper == 0 {
            0
        } else {
            (self.next_u64() as usize) % upper
        }
    }
}

fn ratio(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn clamp_probability(probability: f64) -> f64 {
    probability.clamp(1.0e-9, 1.0 - 1.0e-9)
}

fn true_success_probability(regime: EvidenceRegime, family: OperationFamily) -> f64 {
    SUCCESS_PROBABILITIES[regime as usize][family as usize]
}

fn sample_observations(seed: u64, regimes: &[EvidenceRegime], per_cell: usize) -> Vec<Observation> {
    let mut rng = Rng::new(seed);
    let mut observations = Vec::with_capacity(regimes.len() * FAMILY_COUNT * per_cell);
    for regime in regimes {
        for family in OperationFamily::ALL {
            for _ in 0..per_cell {
                observations.push(Observation {
                    family,
                    regime: *regime,
                    success: rng.next_f64() < true_success_probability(*regime, family),
                });
            }
        }
    }
    observations
}

fn partition_observations(
    observations: &[Observation],
    partition: ForecastPartition,
) -> Vec<Observation> {
    observations
        .iter()
        .copied()
        .filter(|observation| partition.includes(observation.regime))
        .collect()
}

fn evaluate_forecasts(
    model: &ForecastModel,
    observations: &[Observation],
    method: ForecastMethod,
    partition: ForecastPartition,
) -> (ForecastMetrics, Vec<ReliabilityBin>) {
    let selected = partition_observations(observations, partition);
    let mut brier_sum = 0.0;
    let mut log_loss_sum = 0.0;
    let mut probability_sum = 0.0;
    let mut outcome_sum = 0u64;
    let mut bin_count = [0usize; RELIABILITY_BIN_COUNT];
    let mut bin_probability_sum = [0.0f64; RELIABILITY_BIN_COUNT];
    let mut bin_success_sum = [0u64; RELIABILITY_BIN_COUNT];

    for observation in &selected {
        let probability =
            clamp_probability(model.predict(method, observation.regime, observation.family));
        let outcome = if observation.success { 1.0 } else { 0.0 };
        let error = probability - outcome;
        brier_sum += error * error;
        log_loss_sum -= if observation.success {
            probability.ln()
        } else {
            (1.0 - probability).ln()
        };
        probability_sum += probability;
        outcome_sum += if observation.success { 1 } else { 0 };

        let bin = ((probability * RELIABILITY_BIN_COUNT as f64).floor() as usize)
            .min(RELIABILITY_BIN_COUNT - 1);
        bin_count[bin] += 1;
        bin_probability_sum[bin] += probability;
        bin_success_sum[bin] += if observation.success { 1 } else { 0 };
    }

    let count = selected.len();
    let mut ece = 0.0;
    let mut bins = Vec::with_capacity(RELIABILITY_BIN_COUNT);
    for (bin, &bin_n) in bin_count.iter().enumerate() {
        if bin_n > 0 {
            let mean_probability = bin_probability_sum[bin] / bin_n as f64;
            let observed_rate = bin_success_sum[bin] as f64 / bin_n as f64;
            ece += (bin_n as f64 / count.max(1) as f64) * (mean_probability - observed_rate).abs();
            bins.push(ReliabilityBin {
                method,
                partition,
                bin,
                observations: bin_n,
                mean_predicted_probability: Some(mean_probability),
                observed_success_rate: Some(observed_rate),
            });
        } else {
            bins.push(ReliabilityBin {
                method,
                partition,
                bin,
                observations: 0,
                mean_predicted_probability: None,
                observed_success_rate: None,
            });
        }
    }

    let denominator = count.max(1) as f64;
    (
        ForecastMetrics {
            method,
            partition,
            observations: count,
            brier_score: brier_sum / denominator,
            log_loss: log_loss_sum / denominator,
            ece_10: ece,
            mean_predicted_probability: probability_sum / denominator,
            observed_success_rate: outcome_sum as f64 / denominator,
        },
        bins,
    )
}

fn sample_episodes(seed: u64, per_regime: usize) -> Vec<Episode> {
    let mut rng = Rng::new(seed);
    let mut episodes = Vec::with_capacity(CONTEXT_COUNT * per_regime);
    let mut index = 0usize;
    for regime in EvidenceRegime::ALL {
        for _ in 0..per_regime {
            let outcomes = std::array::from_fn(|family_index| {
                let family = OperationFamily::ALL[family_index];
                rng.next_f64() < true_success_probability(regime, family)
            });
            episodes.push(Episode {
                regime,
                outcomes,
                index,
            });
            index += 1;
        }
    }
    episodes
}

fn policy_order(
    policy: SelectionPolicy,
    model: &ForecastModel,
    episode: Episode,
    seed: u64,
) -> [OperationFamily; FAMILY_COUNT] {
    let mut order = OperationFamily::ALL;
    match policy {
        SelectionPolicy::FixedOrder => {}
        SelectionPolicy::BudgetMatchedRandom => {
            let mut rng = Rng::new(seed ^ (episode.index as u64).wrapping_mul(0x9E3779B97F4A7C15));
            for i in (1..FAMILY_COUNT).rev() {
                let j = rng.index(i + 1);
                order.swap(i, j);
            }
        }
        SelectionPolicy::OracleTrueProbabilityPerCost => {
            order.sort_by(|a, b| {
                let score_a = true_success_probability(episode.regime, *a) / f64::from(a.cost());
                let score_b = true_success_probability(episode.regime, *b) / f64::from(b.cost());
                score_b
                    .partial_cmp(&score_a)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| (*a as usize).cmp(&(*b as usize)))
            });
        }
        _ => {
            let method = policy
                .forecast_method()
                .unwrap_or(ForecastMethod::ConstantHalf);
            order.sort_by(|a, b| {
                let score_a = model.predict(method, episode.regime, *a) / f64::from(a.cost());
                let score_b = model.predict(method, episode.regime, *b) / f64::from(b.cost());
                score_b
                    .partial_cmp(&score_a)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| (*a as usize).cmp(&(*b as usize)))
            });
        }
    }
    order
}

fn evaluate_selection(
    model: &ForecastModel,
    episodes: &[Episode],
    policy: SelectionPolicy,
    seed: u64,
) -> Vec<SelectionMetrics> {
    let mut results = Vec::with_capacity(CONTEXT_COUNT);
    for regime in EvidenceRegime::ALL {
        let mut verified_proofs = 0usize;
        let mut no_verified_proof = 0usize;
        let mut total_spent = 0u64;
        let mut total_budget = 0u64;
        let mut budget_violations = 0usize;
        let mut maximum_episode_spend = 0u32;
        let mut total_operations = 0u64;
        let mut count = 0usize;

        for episode in episodes
            .iter()
            .copied()
            .filter(|episode| episode.regime == regime)
        {
            let order = policy_order(policy, model, episode, seed);
            let mut spent = 0u32;
            let mut succeeded = false;
            for family in order {
                let cost = family.cost();
                if spent.saturating_add(cost) > EPISODE_BUDGET {
                    continue;
                }
                spent = spent.saturating_add(cost);
                total_operations += 1;
                if episode.outcomes[family as usize] {
                    succeeded = true;
                    break;
                }
            }

            if succeeded {
                verified_proofs += 1;
            } else {
                no_verified_proof += 1;
            }
            if spent > EPISODE_BUDGET {
                budget_violations += 1;
            }
            maximum_episode_spend = maximum_episode_spend.max(spent);
            total_spent += u64::from(spent);
            total_budget += u64::from(EPISODE_BUDGET);
            count += 1;
        }

        results.push(SelectionMetrics {
            regime,
            policy,
            episodes: count,
            verified_proofs,
            no_verified_proof,
            total_spent,
            total_budget,
            budget_violations,
            maximum_episode_spend,
            total_operations,
        });
    }
    results
}

/// Run the full held-out experiment. Fitting uses only the first four
/// evidence regimes and a training seed disjoint from evaluation.
pub fn run_benchmark(seed: u64) -> BenchmarkReport {
    let training_seed = seed ^ 0xA0761D6478BD642F;
    let evaluation_seed = seed ^ 0xE7037ED1A0B428DB;
    let selection_seed = seed ^ 0x8EBC6AF09C88C6E3;

    let training = sample_observations(
        training_seed,
        &EvidenceRegime::TRAINING,
        DEFAULT_TRAIN_PER_CELL,
    );
    let evaluation =
        sample_observations(evaluation_seed, &EvidenceRegime::ALL, DEFAULT_TEST_PER_CELL);
    let model = ForecastModel::fit(&training);

    let mut forecast_metrics = Vec::new();
    let mut reliability_bins = Vec::new();
    for method in ForecastMethod::ALL {
        for partition in ForecastPartition::ALL {
            let (metrics, bins) = evaluate_forecasts(&model, &evaluation, method, partition);
            forecast_metrics.push(metrics);
            reliability_bins.extend(bins);
        }
    }

    let episodes = sample_episodes(selection_seed, DEFAULT_EPISODES_PER_CONTEXT);
    let mut selection_metrics = Vec::new();
    for policy in SelectionPolicy::ALL {
        selection_metrics.extend(evaluate_selection(
            &model,
            &episodes,
            policy,
            selection_seed,
        ));
    }

    BenchmarkReport {
        seed,
        training_observations: training.len(),
        evaluation_observations: evaluation.len(),
        forecast_metrics,
        reliability_bins,
        selection_metrics,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn benchmark_is_deterministic_for_a_fixed_seed() {
        assert_eq!(run_benchmark(1729), run_benchmark(1729));
    }

    #[test]
    fn held_out_regime_is_absent_from_the_training_split() {
        assert!(!EvidenceRegime::TRAINING.contains(&EvidenceRegime::HeldOutShift));
        assert!(EvidenceRegime::ALL.contains(&EvidenceRegime::HeldOutShift));
        assert_eq!(
            EvidenceRegime::TRAINING.len() * FAMILY_COUNT * DEFAULT_TRAIN_PER_CELL,
            4 * FAMILY_COUNT * DEFAULT_TRAIN_PER_CELL
        );
    }

    #[test]
    fn fitting_uses_only_training_observations() {
        let training = sample_observations(11, &EvidenceRegime::TRAINING, 16);
        let evaluation = sample_observations(22, &EvidenceRegime::ALL, 16);
        let model = ForecastModel::fit(&training);
        let predictions_before: Vec<f64> = evaluation
            .iter()
            .map(|observation| {
                model.predict(
                    ForecastMethod::ContextFamilyShrinkage,
                    observation.regime,
                    observation.family,
                )
            })
            .collect();

        let mut flipped = evaluation;
        for observation in &mut flipped {
            observation.success = !observation.success;
        }

        let predictions_after: Vec<f64> = flipped
            .iter()
            .map(|observation| {
                model.predict(
                    ForecastMethod::ContextFamilyShrinkage,
                    observation.regime,
                    observation.family,
                )
            })
            .collect();
        assert_eq!(predictions_before, predictions_after);
    }

    #[test]
    fn all_predictors_emit_valid_probabilities_for_unseen_context() {
        let training = sample_observations(33, &EvidenceRegime::TRAINING, 32);
        let model = ForecastModel::fit(&training);
        for method in ForecastMethod::ALL {
            for family in OperationFamily::ALL {
                let probability = model.predict(method, EvidenceRegime::HeldOutShift, family);
                assert!((0.0..=1.0).contains(&probability));
            }
        }
    }

    #[test]
    fn proper_scores_are_zero_for_perfect_extreme_forecasts() {
        let probability = clamp_probability(1.0);
        let brier = (probability - 1.0f64).powi(2);
        let log_loss = -probability.ln();
        assert!(brier < 1.0e-12);
        assert!(log_loss < 1.0e-8);
    }

    #[test]
    fn report_contains_all_forecast_partitions_and_reliability_bins() {
        let report = run_benchmark(1729);
        assert_eq!(report.forecast_metrics.len(), ForecastMethod::ALL.len() * 3);
        assert_eq!(
            report.reliability_bins.len(),
            ForecastMethod::ALL.len() * 3 * RELIABILITY_BIN_COUNT
        );
        assert_eq!(
            report.training_observations,
            4 * FAMILY_COUNT * DEFAULT_TRAIN_PER_CELL
        );
        assert_eq!(
            report.evaluation_observations,
            CONTEXT_COUNT * FAMILY_COUNT * DEFAULT_TEST_PER_CELL
        );
        assert!(
            report
                .forecast_metrics
                .iter()
                .all(|metric| metric.observations > 0)
        );
    }

    #[test]
    fn every_selection_policy_respects_the_hard_episode_budget() {
        let report = run_benchmark(1729);
        assert_eq!(
            report.selection_metrics.len(),
            SelectionPolicy::ALL.len() * CONTEXT_COUNT
        );
        for metric in &report.selection_metrics {
            assert_eq!(metric.budget_violations, 0);
            assert!(metric.maximum_episode_spend <= EPISODE_BUDGET);
            assert_eq!(
                metric.verified_proofs + metric.no_verified_proof,
                metric.episodes
            );
        }
    }

    #[test]
    fn oracle_is_marked_diagnostic_only() {
        assert!(!SelectionPolicy::OracleTrueProbabilityPerCost.promotion_candidate());
        assert!(SelectionPolicy::ContextFamilyShrinkage.promotion_candidate());
    }

    #[test]
    fn selection_metrics_keep_success_and_cost_separate() {
        let report = run_benchmark(1729);
        for metric in report.selection_metrics {
            assert!(metric.success_rate() >= 0.0 && metric.success_rate() <= 1.0);
            assert!(metric.verified_proofs_per_work_unit() >= 0.0);
            assert!(metric.average_work_units_per_episode() <= f64::from(EPISODE_BUDGET));
            if metric.verified_proofs == 0 {
                assert_eq!(metric.work_units_per_verified_proof(), None);
            } else {
                assert!(metric.work_units_per_verified_proof().is_some());
            }
        }
    }

    #[test]
    fn held_out_shift_is_reported_separately_from_seen_contexts() {
        let report = run_benchmark(1729);
        for method in ForecastMethod::ALL {
            let seen = report
                .forecast_metrics
                .iter()
                .find(|metric| {
                    metric.method == method && metric.partition == ForecastPartition::SeenContexts
                })
                .unwrap();
            let shifted = report
                .forecast_metrics
                .iter()
                .find(|metric| {
                    metric.method == method && metric.partition == ForecastPartition::HeldOutContext
                })
                .unwrap();
            assert_eq!(seen.observations, 4 * FAMILY_COUNT * DEFAULT_TEST_PER_CELL);
            assert_eq!(shifted.observations, FAMILY_COUNT * DEFAULT_TEST_PER_CELL);
        }
    }
}
