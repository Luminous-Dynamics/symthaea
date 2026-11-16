//! Small, deterministic benchmark of material-response experiment selection for optical ISRU.
//!
//! This example reuses the currently exported Symthaea hypothesis-disagreement selector.
//! It does NOT implement a physical material model, Bayesian mutual information, or a
//! noise/cost-weighted acquisition function. All candidate experiments have equal declared
//! cost and measurement noise so the existing population-variance score is compared fairly.
//!
//! The hidden world is used only after a candidate has been selected, to generate a synthetic
//! observation and score the resulting decision. It is never passed to the selector.
//!
//! Every response curve is synthetic, dimensionless, and deliberately hidden-world-specific.
//! Nothing here is calibrated from Nectar A/B, the 2026 laser study, or any real asteroid material.
//! Do not interpret the curves as operating setpoints, predicted excavation rates, or safe limits.

use symthaea_core::hdc::conjecture_engine::select_most_informative_experiment;

const FLUX_LEVELS_W_CM2: [f64; 7] = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0];
const MORPHOLOGY_INDEXES: [f64; 3] = [0.1, 0.5, 0.9];
const MAX_EXPERIMENTS: usize = 6;
const COST_HOURS_PER_EXPERIMENT: f64 = 1.0;
const MEASUREMENT_SIGMA: f64 = 0.01;
const MEASUREMENT_TOLERANCE: f64 = 0.04;
const RANDOM_SEED: u64 = 0x4953_5255_444f_4531;

#[derive(Clone, Copy, Debug, PartialEq)]
struct Experiment {
    id: u8,
    irradiance_w_cm2: f64,
    morphology_index: f64,
    cost_hours: f64,
    measurement_sigma: f64,
    eligible: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Hypothesis {
    LowFluxPeak,
    HighFluxPeak,
    MorphologyShiftedPeak,
}

impl Hypothesis {
    const ALL: [Self; 3] = [
        Self::LowFluxPeak,
        Self::HighFluxPeak,
        Self::MorphologyShiftedPeak,
    ];

    fn label(self) -> &'static str {
        match self {
            Self::LowFluxPeak => "LOW_FLUX_PEAK",
            Self::HighFluxPeak => "HIGH_FLUX_PEAK",
            Self::MorphologyShiftedPeak => "MORPHOLOGY_SHIFTED_PEAK",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SyntheticWorld {
    LowFluxPeak,
    HighFluxPeak,
    MorphologyShiftedPeak,
    OutOfDomainShift,
}

impl SyntheticWorld {
    const ALL: [Self; 4] = [
        Self::LowFluxPeak,
        Self::HighFluxPeak,
        Self::MorphologyShiftedPeak,
        Self::OutOfDomainShift,
    ];

    fn label(self) -> &'static str {
        match self {
            Self::LowFluxPeak => "WORLD_LOW_FLUX",
            Self::HighFluxPeak => "WORLD_HIGH_FLUX",
            Self::MorphologyShiftedPeak => "WORLD_MORPHOLOGY_SHIFT",
            Self::OutOfDomainShift => "WORLD_OOD_SHIFT",
        }
    }

    fn expected_hypothesis(self) -> Option<Hypothesis> {
        match self {
            Self::LowFluxPeak => Some(Hypothesis::LowFluxPeak),
            Self::HighFluxPeak => Some(Hypothesis::HighFluxPeak),
            Self::MorphologyShiftedPeak => Some(Hypothesis::MorphologyShiftedPeak),
            Self::OutOfDomainShift => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Policy {
    FixedCanonical,
    OneFactorAtATime,
    SpaceFilling,
    SeededRandom,
    SymthaeaPredictionDisagreement,
}

impl Policy {
    const ALL: [Self; 5] = [
        Self::FixedCanonical,
        Self::OneFactorAtATime,
        Self::SpaceFilling,
        Self::SeededRandom,
        Self::SymthaeaPredictionDisagreement,
    ];

    fn label(self) -> &'static str {
        match self {
            Self::FixedCanonical => "FIXED_CANONICAL",
            Self::OneFactorAtATime => "ONE_FACTOR_AT_A_TIME",
            Self::SpaceFilling => "SPACE_FILLING",
            Self::SeededRandom => "SEEDED_RANDOM",
            Self::SymthaeaPredictionDisagreement => "SYMTHAEA_PREDICTION_DISAGREEMENT",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Decision {
    Identified(Hypothesis),
    OutOfDomain,
    Unresolved,
}

impl Decision {
    fn label(self) -> &'static str {
        match self {
            Self::Identified(hypothesis) => hypothesis.label(),
            Self::OutOfDomain => "ABSTAIN_OUT_OF_DOMAIN",
            Self::Unresolved => "UNRESOLVED",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct TrialResult {
    decision: Decision,
    selected_ids: Vec<u8>,
    total_cost_hours: f64,
    initial_disagreement_score: Option<f64>,
}

fn response_curve(
    peak_w_cm2: f64,
    width_w_cm2: f64,
    amplitude: f64,
    experiment: &Experiment,
) -> f64 {
    let z = (experiment.irradiance_w_cm2 - peak_w_cm2) / width_w_cm2;
    amplitude * (-0.5 * z * z).exp() * (0.9 + 0.1 * experiment.morphology_index)
}

/// Synthetic hypothesis prediction. This is a toy response surface, not material physics.
fn hypothesis_prediction(hypothesis: &Hypothesis, experiment: &Experiment) -> Option<f64> {
    let morphology = experiment.morphology_index;
    let prediction = match hypothesis {
        Hypothesis::LowFluxPeak => response_curve(
            260.0 + 130.0 * morphology,
            190.0,
            0.95 + 0.05 * morphology,
            experiment,
        ),
        Hypothesis::HighFluxPeak => response_curve(
            370.0 - 80.0 * morphology,
            210.0,
            1.0 - 0.1 * morphology,
            experiment,
        ),
        Hypothesis::MorphologyShiftedPeak => response_curve(
            310.0 + 240.0 * (morphology - 0.5),
            140.0,
            0.82 + 0.18 * morphology,
            experiment,
        ),
    };
    prediction.is_finite().then_some(prediction)
}

fn predict_hypothesis(hypothesis: &Hypothesis, experiment: &Experiment) -> Option<f64> {
    hypothesis_prediction(hypothesis, experiment)
}

/// Hidden synthetic world used only by the evaluator after an experiment is selected.
fn hidden_world_observation(world: SyntheticWorld, experiment: &Experiment) -> f64 {
    let noiseless = match world {
        SyntheticWorld::LowFluxPeak => hypothesis_prediction(
            &Hypothesis::LowFluxPeak,
            experiment,
        )
        .unwrap_or(0.0),
        SyntheticWorld::HighFluxPeak => hypothesis_prediction(
            &Hypothesis::HighFluxPeak,
            experiment,
        )
        .unwrap_or(0.0),
        SyntheticWorld::MorphologyShiftedPeak => hypothesis_prediction(
            &Hypothesis::MorphologyShiftedPeak,
            experiment,
        )
        .unwrap_or(0.0),
        SyntheticWorld::OutOfDomainShift => response_curve(
            250.0,
            1_000.0,
            0.93,
            experiment,
        ),
    };

    // Small deterministic observation perturbation. It is not a sampled physical noise model.
    let perturbation = match experiment.id % 3 {
        0 => -0.003,
        1 => 0.0,
        _ => 0.003,
    };
    noiseless + perturbation
}

fn full_catalogue() -> Vec<Experiment> {
    let mut experiments = Vec::with_capacity(FLUX_LEVELS_W_CM2.len() * MORPHOLOGY_INDEXES.len());
    let mut id = 0_u8;
    for flux in FLUX_LEVELS_W_CM2 {
        for morphology in MORPHOLOGY_INDEXES {
            // One deliberately ineligible negative control is present in the catalogue.
            // It must be filtered before any policy ranks or receives candidates.
            let eligible = !(flux == 700.0 && morphology == 0.9);
            experiments.push(Experiment {
                id,
                irradiance_w_cm2: flux,
                morphology_index: morphology,
                cost_hours: COST_HOURS_PER_EXPERIMENT,
                measurement_sigma: MEASUREMENT_SIGMA,
                eligible,
            });
            id = id.saturating_add(1);
        }
    }
    experiments
}

fn eligible_catalogue() -> Vec<Experiment> {
    full_catalogue()
        .into_iter()
        .filter(|experiment| experiment.eligible)
        .collect()
}

fn validate_matched_pool(pool: &[Experiment]) -> bool {
    if pool.is_empty() {
        return false;
    }
    let cost = pool[0].cost_hours;
    let sigma = pool[0].measurement_sigma;
    let mut identities = std::collections::BTreeSet::new();
    pool.iter().all(|experiment| {
        identities.insert(experiment.id)
            && experiment.eligible
            && experiment.cost_hours.is_finite()
            && experiment.cost_hours > 0.0
            && experiment.cost_hours == cost
            && experiment.measurement_sigma.is_finite()
            && experiment.measurement_sigma > 0.0
            && experiment.measurement_sigma == sigma
    })
}

fn by_id(pool: &[Experiment], id: u8) -> Option<Experiment> {
    pool.iter().copied().find(|experiment| experiment.id == id)
}

fn fixed_schedule(policy: Policy, pool: &[Experiment], seed: &mut u64) -> Vec<Experiment> {
    let ids: &[u8] = match policy {
        // Three flux levels crossed with two morphology strata.
        Policy::FixedCanonical => &[3, 5, 9, 11, 15, 17],
        // Irradiance changes while morphology is held at the central index.
        Policy::OneFactorAtATime => &[7, 4, 10, 13, 16, 19],
        // A deterministic spread over both axes.
        Policy::SpaceFilling => &[0, 5, 7, 9, 14, 17],
        Policy::SeededRandom | Policy::SymthaeaPredictionDisagreement => &[],
    };

    if policy == Policy::SeededRandom {
        let mut shuffled = pool.to_vec();
        for upper in (1..shuffled.len()).rev() {
            let index = (next_random(seed) as usize) % (upper + 1);
            shuffled.swap(upper, index);
        }
        return shuffled;
    }

    ids.iter().filter_map(|id| by_id(pool, *id)).collect()
}

fn next_random(seed: &mut u64) -> u64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed
}

/// Selector boundary: only remaining candidate experiments and surviving hypotheses are visible.
/// The hidden world, its response function, and evaluator outcomes are not parameters.
fn choose_active_next(
    remaining: &[Experiment],
    surviving_hypotheses: &[Hypothesis],
) -> Option<(Experiment, f64)> {
    let (chosen, disagreement) = select_most_informative_experiment(
        remaining,
        surviving_hypotheses,
        predict_hypothesis,
    )?;
    Some((*chosen, disagreement))
}

fn run_trial(policy: Policy, world: SyntheticWorld) -> TrialResult {
    let mut seed = RANDOM_SEED;
    let mut remaining = eligible_catalogue();
    let mut schedule = fixed_schedule(policy, &remaining, &mut seed);
    let mut surviving = Hypothesis::ALL.to_vec();
    let mut selected_ids = Vec::new();
    let mut total_cost_hours = 0.0;
    let mut initial_disagreement_score = None;
    let mut decision = Decision::Unresolved;

    for _ in 0..MAX_EXPERIMENTS {
        let selected = if policy == Policy::SymthaeaPredictionDisagreement {
            choose_active_next(&remaining, &surviving)
        } else if schedule.is_empty() {
            None
        } else {
            Some((schedule.remove(0), 0.0))
        };

        let Some((experiment, disagreement)) = selected else {
            break;
        };
        let Some(position) = remaining
            .iter()
            .position(|candidate| candidate.id == experiment.id)
        else {
            // A policy cannot replay or select a candidate outside the remaining pool.
            break;
        };
        remaining.remove(position);
        selected_ids.push(experiment.id);
        total_cost_hours += experiment.cost_hours;

        if initial_disagreement_score.is_none()
            && policy == Policy::SymthaeaPredictionDisagreement
        {
            initial_disagreement_score = Some(disagreement);
        }

        // Hidden world is queried only after the policy has committed to a candidate.
        let observed = hidden_world_observation(world, &experiment);
        surviving.retain(|hypothesis| {
            hypothesis_prediction(hypothesis, &experiment)
                .is_some_and(|prediction| (prediction - observed).abs() <= MEASUREMENT_TOLERANCE)
        });

        if surviving.is_empty() {
            decision = Decision::OutOfDomain;
            break;
        }
        if surviving.len() == 1 {
            decision = Decision::Identified(surviving[0]);
            break;
        }
    }

    TrialResult {
        decision,
        selected_ids,
        total_cost_hours,
        initial_disagreement_score,
    }
}

fn is_correct(world: SyntheticWorld, result: &TrialResult) -> bool {
    match (world.expected_hypothesis(), result.decision) {
        (Some(expected), Decision::Identified(actual)) => expected == actual,
        (None, Decision::OutOfDomain) => true,
        _ => false,
    }
}

fn policy_row(world: SyntheticWorld, policy: Policy, result: &TrialResult) -> String {
    let experiments_used = result.selected_ids.len();
    let cost_hours = result.total_cost_hours;
    let ids = result
        .selected_ids
        .iter()
        .map(u8::to_string)
        .collect::<Vec<_>>()
        .join(";");
    let score = result
        .initial_disagreement_score
        .map(|value| format!("{value:.6}"))
        .unwrap_or_else(|| "NA".to_string());

    format!(
        "{},{},{},{},{},{cost_hours:.2},{score},{ids}",
        world.label(),
        policy.label(),
        result.decision.label(),
        is_correct(world, result),
        experiments_used,
    )
}

fn main() {
    let pool = eligible_catalogue();
    assert!(
        validate_matched_pool(&pool),
        "benchmark requires a non-empty, unique, equally priced/noisy eligible candidate pool"
    );

    println!(
        "world,policy,decision,correct,experiments_used,cost_hours,initial_disagreement_score,selected_candidate_ids"
    );
    for world in SyntheticWorld::ALL {
        for policy in Policy::ALL {
            let result = run_trial(policy, world);
            println!("{}", policy_row(world, policy, &result));
        }
    }
    eprintln!(
        "synthetic benchmark only: candidate cost={COST_HOURS_PER_EXPERIMENT} h; common measurement sigma={MEASUREMENT_SIGMA}; prediction-disagreement selector is not Bayesian EIG"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::hdc::conjecture_engine::epistemic_value;

    #[test]
    fn catalogue_filters_ineligible_control_before_selection() {
        let complete = full_catalogue();
        let eligible = eligible_catalogue();
        assert_eq!(complete.len(), 21);
        assert_eq!(eligible.len(), 20);
        assert!(eligible.iter().all(|experiment| experiment.eligible));
        assert!(!eligible.iter().any(|experiment| experiment.irradiance_w_cm2 == 700.0
            && experiment.morphology_index == 0.9));
    }

    #[test]
    fn matched_budget_has_common_cost_and_sensor_noise() {
        let pool = eligible_catalogue();
        assert!(validate_matched_pool(&pool));
        assert!(pool.iter().all(|experiment| {
            experiment.cost_hours == COST_HOURS_PER_EXPERIMENT
                && experiment.measurement_sigma == MEASUREMENT_SIGMA
        }));
    }

    #[test]
    fn active_selector_uses_the_current_population_variance_score() {
        let pool = eligible_catalogue();
        let hypotheses = Hypothesis::ALL;
        let (chosen, score) = choose_active_next(&pool, &hypotheses).unwrap();
        let independent_score = epistemic_value(&chosen, &hypotheses, predict_hypothesis);
        assert!((score - independent_score).abs() < 1.0e-12);
    }

    #[test]
    fn active_policy_identifies_each_in_domain_synthetic_family() {
        for world in [
            SyntheticWorld::LowFluxPeak,
            SyntheticWorld::HighFluxPeak,
            SyntheticWorld::MorphologyShiftedPeak,
        ] {
            let result = run_trial(Policy::SymthaeaPredictionDisagreement, world);
            assert!(is_correct(world, &result), "{world:?}: {result:?}");
            assert_eq!(result.selected_ids.len(), 1);
        }
    }

    #[test]
    fn active_policy_abstains_on_the_held_out_ood_world() {
        let result = run_trial(
            Policy::SymthaeaPredictionDisagreement,
            SyntheticWorld::OutOfDomainShift,
        );
        assert_eq!(result.decision, Decision::OutOfDomain);
        assert_eq!(result.selected_ids.len(), 1);
    }

    #[test]
    fn active_policy_reduces_experiment_count_vs_ofat_for_close_peak_families() {
        for world in [
            SyntheticWorld::LowFluxPeak,
            SyntheticWorld::HighFluxPeak,
        ] {
            let active = run_trial(Policy::SymthaeaPredictionDisagreement, world);
            let ofat = run_trial(Policy::OneFactorAtATime, world);
            assert!(is_correct(world, &active));
            assert!(is_correct(world, &ofat));
            assert!(
                active.selected_ids.len() < ofat.selected_ids.len(),
                "active={active:?}, OFAT={ofat:?}"
            );
        }
    }

    #[test]
    fn all_policies_stay_within_budget_and_do_not_repeat_candidates() {
        for world in SyntheticWorld::ALL {
            for policy in Policy::ALL {
                let result = run_trial(policy, world);
                assert!(result.selected_ids.len() <= MAX_EXPERIMENTS);
                let unique: std::collections::BTreeSet<_> =
                    result.selected_ids.iter().copied().collect();
                assert_eq!(unique.len(), result.selected_ids.len());
            }
        }
    }

    #[test]
    fn factorial_fixed_design_can_false_confidently_misclassify_ood_world() {
        let result = run_trial(
            Policy::FixedCanonical,
            SyntheticWorld::OutOfDomainShift,
        );
        assert_eq!(
            result.decision,
            Decision::Identified(Hypothesis::LowFluxPeak)
        );
        assert!(!is_correct(SyntheticWorld::OutOfDomainShift, &result));
    }

    #[test]
    fn active_selector_correctly_abstains_where_fixed_design_false_commits() {
        let fixed = run_trial(
            Policy::FixedCanonical,
            SyntheticWorld::OutOfDomainShift,
        );
        let active = run_trial(
            Policy::SymthaeaPredictionDisagreement,
            SyntheticWorld::OutOfDomainShift,
        );
        assert!(!is_correct(SyntheticWorld::OutOfDomainShift, &fixed));
        assert!(is_correct(SyntheticWorld::OutOfDomainShift, &active));
        assert_eq!(active.decision, Decision::OutOfDomain);
    }

    #[test]
    fn seeded_random_policy_is_deterministic() {
        let first = run_trial(Policy::SeededRandom, SyntheticWorld::LowFluxPeak);
        let second = run_trial(Policy::SeededRandom, SyntheticWorld::LowFluxPeak);
        assert_eq!(first, second);
    }

    #[test]
    fn active_selector_is_deterministic_for_a_fixed_pool_and_hypothesis_set() {
        let pool = eligible_catalogue();
        let hypotheses = Hypothesis::ALL;
        assert_eq!(
            choose_active_next(&pool, &hypotheses),
            choose_active_next(&pool, &hypotheses)
        );
    }
}
