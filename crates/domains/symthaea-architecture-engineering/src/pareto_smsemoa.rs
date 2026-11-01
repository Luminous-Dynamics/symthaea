//! Deterministic SMS-EMOA reference optimizer for ZDT1 and DTLZ2.
//!
//! Uses binary tournament by nondomination rank/crowding distance, SBX and polynomial mutation.
//! Environmental selection removes one individual at a time from the worst front by minimum
//! exclusive hypervolume contribution against a frozen reference point. Objectives are minimized.
//! This is a compact reference variant, not a claim of state-of-the-art performance.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::pareto_anytime::{AnytimeError, AnytimeHistory, AnytimeRecorder, EvaluationOutcome};
use super::pareto_benchmarks::{dtlz2_three_objectives, zdt1_objectives};
use super::pareto_objectives::{
    analyze_pareto, exact_hypervolume, ObjectiveDirection, ParetoAnalysisPlan,
    ParetoAnalysisSummary, ParetoCandidate, ParetoError, ParetoObjectiveSpec,
};

const SCHEMA: &str = "symthaea.architecture-engineering.pareto-smsemoa";
const VERSION: &str = "0.1.0";
const ALGORITHM_ID: &str = "sms-emoa-reference-v0.1";
const MAX_POPULATION_SIZE: usize = 512;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SmsEmoaError {
    pub path: String,
    pub message: String,
}

impl SmsEmoaError {
    fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for SmsEmoaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}

impl Error for SmsEmoaError {}

/// Frozen SMS-EMOA configuration. All objectives are minimized by the benchmark wrappers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SmsEmoaConfig {
    pub population_size: usize,
    pub offspring_evaluations: usize,
    pub objective_specs: Vec<ParetoObjectiveSpec>,
    pub reference_point_utility: Vec<f64>,
    pub crossover_probability: f64,
    pub mutation_probability: f64,
    pub crossover_distribution_index: f64,
    pub mutation_distribution_index: f64,
    pub seed: u64,
}

impl SmsEmoaConfig {
    pub fn validate(&self) -> Result<(), SmsEmoaError> {
        if self.population_size < 4 || self.population_size > MAX_POPULATION_SIZE {
            return Err(SmsEmoaError::new(
                "config.population_size",
                format!("population size must be between four and {MAX_POPULATION_SIZE}"),
            ));
        }
        if self.offspring_evaluations == 0 {
            return Err(SmsEmoaError::new(
                "config.offspring_evaluations",
                "at least one offspring evaluation is required",
            ));
        }
        if !(2..=4).contains(&self.objective_specs.len()) {
            return Err(SmsEmoaError::new(
                "config.objective_specs",
                "SMS-EMOA reference implementation supports two through four objectives",
            ));
        }
        if self.reference_point_utility.len() != self.objective_specs.len()
            || self.reference_point_utility.iter().any(|value| !value.is_finite())
        {
            return Err(SmsEmoaError::new(
                "config.reference_point_utility",
                "reference point must have one finite coordinate per objective",
            ));
        }
        let mut ids = BTreeSet::new();
        for (index, objective) in self.objective_specs.iter().enumerate() {
            if objective.objective_id.trim().is_empty() || !ids.insert(objective.objective_id.as_str()) {
                return Err(SmsEmoaError::new(
                    format!("config.objective_specs[{index}].objective_id"),
                    "objective IDs must be non-empty and unique",
                ));
            }
            if objective.direction != ObjectiveDirection::Minimize {
                return Err(SmsEmoaError::new(
                    format!("config.objective_specs[{index}].direction"),
                    "ZDT1 and DTLZ2 SMS-EMOA wrappers require minimization objectives",
                ));
            }
            let range = objective.scale_upper - objective.scale_lower;
            if !objective.scale_lower.is_finite()
                || !objective.scale_upper.is_finite()
                || objective.scale_lower >= objective.scale_upper
                || !range.is_finite()
            {
                return Err(SmsEmoaError::new(
                    format!("config.objective_specs[{index}].scale"),
                    "scale anchors must be finite with a finite positive range",
                ));
            }
        }
        for (path, value) in [
            ("crossover_probability", self.crossover_probability),
            ("mutation_probability", self.mutation_probability),
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(SmsEmoaError::new(
                    format!("config.{path}"),
                    "probability must be finite and in [0, 1]",
                ));
            }
        }
        for (path, value) in [
            ("crossover_distribution_index", self.crossover_distribution_index),
            ("mutation_distribution_index", self.mutation_distribution_index),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(SmsEmoaError::new(
                    format!("config.{path}"),
                    "distribution index must be finite and positive",
                ));
            }
        }
        self.population_size.checked_add(self.offspring_evaluations)
            .ok_or_else(|| SmsEmoaError::new("config", "objective evaluation budget overflows usize"))?;
        Ok(())
    }

    pub fn config_sha256(&self) -> Result<String, SmsEmoaError> {
        self.validate()?;
        digest_serializable(self)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SmsEmoaCandidate {
    pub candidate_id: String,
    pub decision_values: Vec<f64>,
    pub objective_values: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SmsEmoaRunReport {
    pub schema: String,
    pub version: String,
    pub algorithm_id: String,
    pub problem_id: String,
    pub seed: u64,
    pub config_sha256: String,
    pub decision_dimensions: usize,
    pub objective_dimensions: usize,
    pub objective_evaluations: usize,
    pub reference_point_utility: Vec<f64>,
    pub final_population: Vec<SmsEmoaCandidate>,
    pub nondominated_front: Vec<SmsEmoaCandidate>,
    pub report_sha256: String,
}

#[derive(Serialize)]
struct SmsEmoaReportPayload<'a> {
    schema: &'a str,
    version: &'a str,
    algorithm_id: &'a str,
    problem_id: &'a str,
    seed: u64,
    config_sha256: &'a str,
    decision_dimensions: usize,
    objective_dimensions: usize,
    objective_evaluations: usize,
    reference_point_utility: &'a [f64],
    final_population: &'a [SmsEmoaCandidate],
    nondominated_front: &'a [SmsEmoaCandidate],
}

#[derive(Debug, Clone)]
struct Individual {
    decision: Vec<f64>,
    objectives: Vec<f64>,
    utility: Vec<f64>,
    rank: usize,
    crowding_distance: f64,
    ordinal: usize,
}

struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x9e37_79b9_7f4a_7c15 } else { seed },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut value = self.state;
        value ^= value >> 12;
        value ^= value << 25;
        value ^= value >> 27;
        self.state = value;
        value.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn unit_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    fn index(&mut self, upper_exclusive: usize) -> usize {
        (self.next_u64() % upper_exclusive as u64) as usize
    }
}

pub fn run_smsemoa_zdt1(
    config: &SmsEmoaConfig,
    decision_dimensions: usize,
) -> Result<SmsEmoaRunReport, SmsEmoaError> {
    if decision_dimensions < 2 || config.objective_specs.len() != 2 {
        return Err(SmsEmoaError::new(
            "dimensions",
            "ZDT1 requires at least two decision variables and two objective specifications",
        ));
    }
    run_smsemoa(
        "zdt1",
        config,
        decision_dimensions,
        2,
        |decision| {
            zdt1_objectives(decision)
                .map(|values| values.to_vec())
                .map_err(|error| SmsEmoaError::new("zdt1", error.to_string()))
        },
        None,
    )
}

pub fn run_smsemoa_zdt1_with_history(
    config: &SmsEmoaConfig,
    decision_dimensions: usize,
    history: &mut AnytimeHistory,
) -> Result<SmsEmoaRunReport, SmsEmoaError> {
    if decision_dimensions < 2 || config.objective_specs.len() != 2 {
        return Err(SmsEmoaError::new("dimensions", "ZDT1 requires at least two decision variables and two objective specifications"));
    }
    let budget = config.population_size.checked_add(config.offspring_evaluations)
        .ok_or_else(|| SmsEmoaError::new("config", "evaluation budget overflows usize"))?;
    let mut recorder = make_anytime_recorder(history, "zdt1", config, 2, budget)?;
    run_smsemoa("zdt1", config, decision_dimensions, 2, |decision| {
        zdt1_objectives(decision).map(|values| values.to_vec())
            .map_err(|error| SmsEmoaError::new("zdt1", error.to_string()))
    }, Some(&mut recorder))
}

pub fn run_smsemoa_dtlz2(
    config: &SmsEmoaConfig,
    decision_dimensions: usize,
) -> Result<SmsEmoaRunReport, SmsEmoaError> {
    if decision_dimensions < 3 || config.objective_specs.len() != 3 {
        return Err(SmsEmoaError::new(
            "dimensions",
            "three-objective DTLZ2 requires at least three decision variables and three objective specifications",
        ));
    }
    run_smsemoa(
        "dtlz2-3obj",
        config,
        decision_dimensions,
        3,
        |decision| {
            dtlz2_three_objectives(decision)
                .map(|values| values.to_vec())
                .map_err(|error| SmsEmoaError::new("dtlz2", error.to_string()))
        },
        None,
    )
}

pub fn run_smsemoa_dtlz2_with_history(
    config: &SmsEmoaConfig,
    decision_dimensions: usize,
    history: &mut AnytimeHistory,
) -> Result<SmsEmoaRunReport, SmsEmoaError> {
    if decision_dimensions < 3 || config.objective_specs.len() != 3 {
        return Err(SmsEmoaError::new("dimensions", "three-objective DTLZ2 requires at least three decision variables and three objective specifications"));
    }
    let budget = config.population_size.checked_add(config.offspring_evaluations)
        .ok_or_else(|| SmsEmoaError::new("config", "evaluation budget overflows usize"))?;
    let mut recorder = make_anytime_recorder(history, "dtlz2-3obj", config, 3, budget)?;
    run_smsemoa("dtlz2-3obj", config, decision_dimensions, 3, |decision| {
        dtlz2_three_objectives(decision).map(|values| values.to_vec())
            .map_err(|error| SmsEmoaError::new("dtlz2", error.to_string()))
    }, Some(&mut recorder))
}

fn run_smsemoa<F>(
    problem_id: &str,
    config: &SmsEmoaConfig,
    decision_dimensions: usize,
    objective_dimensions: usize,
    evaluator: F,
    mut recorder: Option<&mut AnytimeRecorder<'_>>,
) -> Result<SmsEmoaRunReport, SmsEmoaError>
where
    F: Fn(&[f64]) -> Result<Vec<f64>, SmsEmoaError>,
{
    config.validate()?;
    if config.objective_specs.len() != objective_dimensions {
        return Err(SmsEmoaError::new("objective_specs", "objective dimension mismatch"));
    }
    let config_sha256 = config.config_sha256()?;
    let mut rng = DeterministicRng::new(config.seed);
    let mut objective_evaluations = 0usize;
    let mut population = Vec::with_capacity(config.population_size);

    for ordinal in 0..config.population_size {
        let decision = (0..decision_dimensions).map(|_| rng.unit_f64()).collect::<Vec<_>>();
        let objectives = match evaluate_checked(&evaluator, &decision, objective_dimensions) {
            Ok(values) => values,
            Err(error) => {
                if let Some(recorder) = &mut recorder {
                    recorder.record_failure(&decision, EvaluationOutcome::Failed, error.to_string()).map_err(map_anytime_error)?;
                }
                return Err(error);
            }
        };
        if let Some(recorder) = &mut recorder {
            recorder.record_success(&decision, &objectives, true, Vec::new()).map_err(map_anytime_error)?;
        }
        let utility = transform_objectives(&objectives, &config.objective_specs)?;
        objective_evaluations = checked_add_usize(objective_evaluations, 1, "objective_evaluations")?;
        population.push(Individual {
            decision,
            objectives,
            utility,
            rank: usize::MAX,
            crowding_distance: 0.0,
            ordinal,
        });
    }

    for offspring_index in 0..config.offspring_evaluations {
        assign_rank_and_crowding(&mut population)?;
        let parent_left = binary_tournament(&population, &mut rng);
        let parent_right = binary_tournament(&population, &mut rng);
        let (child_left, child_right) = simulated_binary_crossover(
            &population[parent_left].decision,
            &population[parent_right].decision,
            config.crossover_probability,
            config.crossover_distribution_index,
            &mut rng,
        );
        let mut child = if rng.unit_f64() < 0.5 {
            child_left
        } else {
            child_right
        };
        polynomial_mutation(
            &mut child,
            config.mutation_probability,
            config.mutation_distribution_index,
            &mut rng,
        );
        let objectives = match evaluate_checked(&evaluator, &child, objective_dimensions) {
            Ok(values) => values,
            Err(error) => {
                if let Some(recorder) = &mut recorder {
                    recorder.record_failure(&child, EvaluationOutcome::Failed, error.to_string()).map_err(map_anytime_error)?;
                }
                return Err(error);
            }
        };
        if let Some(recorder) = &mut recorder {
            recorder.record_success(&child, &objectives, true, Vec::new()).map_err(map_anytime_error)?;
        }
        let utility = transform_objectives(&objectives, &config.objective_specs)?;
        objective_evaluations = checked_add_usize(objective_evaluations, 1, "objective_evaluations")?;

        let mut combined = population;
        combined.push(Individual {
            decision: child,
            objectives,
            utility,
            rank: usize::MAX,
            crowding_distance: 0.0,
            ordinal: config.population_size + offspring_index,
        });
        population = environmental_selection(
            combined,
            config.population_size,
            &config.reference_point_utility,
        )?;
    }

    let mut final_population = population.iter().map(to_candidate).collect::<Result<Vec<_>, _>>()?;
    let front_indices = fast_non_dominated_sort(&population)?
        .first().cloned().unwrap_or_default();
    let mut nondominated_front = front_indices.iter()
        .map(|index| to_candidate(&population[*index]))
        .collect::<Result<Vec<_>, _>>()?;
    final_population.sort_by(|left, right| left.candidate_id.cmp(&right.candidate_id));
    nondominated_front.sort_by(|left, right| left.candidate_id.cmp(&right.candidate_id));

    let report_sha256 = digest_serializable(&SmsEmoaReportPayload {
        schema: SCHEMA,
        version: VERSION,
        algorithm_id: ALGORITHM_ID,
        problem_id,
        seed: config.seed,
        config_sha256: &config_sha256,
        decision_dimensions,
        objective_dimensions,
        objective_evaluations,
        reference_point_utility: &config.reference_point_utility,
        final_population: &final_population,
        nondominated_front: &nondominated_front,
    })?;

    Ok(SmsEmoaRunReport {
        schema: SCHEMA.to_owned(),
        version: VERSION.to_owned(),
        algorithm_id: ALGORITHM_ID.to_owned(),
        problem_id: problem_id.to_owned(),
        seed: config.seed,
        config_sha256,
        decision_dimensions,
        objective_dimensions,
        objective_evaluations,
        reference_point_utility: config.reference_point_utility.clone(),
        final_population,
        nondominated_front,
        report_sha256,
    })
}

fn environmental_selection(
    mut combined: Vec<Individual>,
    population_size: usize,
    reference_point: &[f64],
) -> Result<Vec<Individual>, SmsEmoaError> {
    if combined.len() != population_size + 1 {
        return Err(SmsEmoaError::new(
            "environmental_selection",
            "SMS-EMOA environmental selection requires N+1 candidates",
        ));
    }
    let fronts = fast_non_dominated_sort(&combined)?;
    if fronts.is_empty() {
        return Err(SmsEmoaError::new("environmental_selection", "no nondominated front found"));
    }
    let last_front = fronts.last().cloned().unwrap_or_default();
    if last_front.is_empty() {
        return Err(SmsEmoaError::new("environmental_selection", "worst front is empty"));
    }
    let remove_index = if last_front.len() == 1 {
        last_front[0]
    } else {
        let points = last_front.iter().map(|index| combined[*index].utility.clone()).collect::<Vec<_>>();
        let total_hypervolume = exact_hypervolume(&points, reference_point)
            .map_err(map_pareto_error)?;
        let mut least_contribution = f64::INFINITY;
        let mut selected = last_front[0];
        for (position, index) in last_front.iter().enumerate() {
            let reduced = points.iter().enumerate()
                .filter_map(|(point_index, point)| (point_index != position).then_some(point.clone()))
                .collect::<Vec<_>>();
            let reduced_hypervolume = exact_hypervolume(&reduced, reference_point)
                .map_err(map_pareto_error)?;
            let contribution = (total_hypervolume - reduced_hypervolume).max(0.0);
            if contribution < least_contribution
                || (contribution.total_cmp(&least_contribution) == Ordering::Equal
                    && combined[*index].ordinal > combined[selected].ordinal)
            {
                least_contribution = contribution;
                selected = *index;
            }
        }
        selected
    };
    combined.remove(remove_index);
    if combined.len() != population_size {
        return Err(SmsEmoaError::new(
            "environmental_selection",
            "environmental selection did not retain the configured population",
        ));
    }
    Ok(combined)
}

fn assign_rank_and_crowding(population: &mut [Individual]) -> Result<(), SmsEmoaError> {
    let fronts = fast_non_dominated_sort(population)?;
    for (rank, front) in fronts.iter().enumerate() {
        assign_crowding_for_front(population, front)?;
        for index in front {
            population[*index].rank = rank;
        }
    }
    Ok(())
}

fn fast_non_dominated_sort(population: &[Individual]) -> Result<Vec<Vec<usize>>, SmsEmoaError> {
    let count = population.len();
    let mut dominates_set = vec![Vec::<usize>::new(); count];
    let mut dominated_by_count = vec![0usize; count];
    let mut first_front = Vec::new();

    for left in 0..count {
        for right in 0..count {
            if left == right {
                continue;
            }
            if dominates_minimization(&population[left].objectives, &population[right].objectives) {
                dominates_set[left].push(right);
            } else if dominates_minimization(&population[right].objectives, &population[left].objectives) {
                dominated_by_count[left] = dominated_by_count[left]
                    .checked_add(1)
                    .ok_or_else(|| SmsEmoaError::new("non_dominated_sort", "domination count overflow"))?;
            }
        }
        if dominated_by_count[left] == 0 {
            first_front.push(left);
        }
    }

    let mut fronts = Vec::new();
    if first_front.is_empty() && !population.is_empty() {
        return Err(SmsEmoaError::new("non_dominated_sort", "no first front was identified"));
    }
    let mut current = first_front;
    let mut assigned = 0usize;
    while !current.is_empty() {
        assigned = checked_add_usize(assigned, current.len(), "non_dominated_sort.assigned")?;
        let mut next = Vec::new();
        for index in &current {
            for dominated_index in &dominates_set[*index] {
                dominated_by_count[*dominated_index] = dominated_by_count[*dominated_index]
                    .checked_sub(1)
                    .ok_or_else(|| SmsEmoaError::new("non_dominated_sort", "domination count underflow"))?;
                if dominated_by_count[*dominated_index] == 0 {
                    next.push(*dominated_index);
                }
            }
        }
        next.sort_unstable();
        next.dedup();
        fronts.push(current);
        current = next;
    }
    if assigned != count {
        return Err(SmsEmoaError::new(
            "non_dominated_sort",
            "front decomposition did not assign every individual",
        ));
    }
    Ok(fronts)
}

fn assign_crowding_for_front(
    population: &mut [Individual],
    front: &[usize],
) -> Result<(), SmsEmoaError> {
    if front.is_empty() {
        return Ok(());
    }
    for index in front {
        population[*index].crowding_distance = 0.0;
    }
    if front.len() <= 2 {
        for index in front {
            population[*index].crowding_distance = f64::INFINITY;
        }
        return Ok(());
    }
    let objective_dimensions = population[front[0]].objectives.len();
    for objective in 0..objective_dimensions {
        let mut order = front.to_vec();
        order.sort_by(|left, right| {
            population[*left].objectives[objective]
                .total_cmp(&population[*right].objectives[objective])
                .then_with(|| population[*left].ordinal.cmp(&population[*right].ordinal))
        });
        let first = order[0];
        let last = order[order.len() - 1];
        let minimum = population[first].objectives[objective];
        let maximum = population[last].objectives[objective];
        let range = maximum - minimum;
        if range <= f64::EPSILON {
            continue;
        }
        population[first].crowding_distance = f64::INFINITY;
        population[last].crowding_distance = f64::INFINITY;
        for position in 1..(order.len() - 1) {
            let index = order[position];
            if population[index].crowding_distance.is_infinite() {
                continue;
            }
            let previous = population[order[position - 1]].objectives[objective];
            let next = population[order[position + 1]].objectives[objective];
            let contribution = (next - previous) / range;
            if !contribution.is_finite() {
                return Err(SmsEmoaError::new("crowding_distance", "contribution is not finite"));
            }
            population[index].crowding_distance += contribution;
        }
    }
    Ok(())
}

fn binary_tournament(population: &[Individual], rng: &mut DeterministicRng) -> usize {
    let left = rng.index(population.len());
    let right = rng.index(population.len());
    match population[left].rank.cmp(&population[right].rank) {
        Ordering::Less => left,
        Ordering::Greater => right,
        Ordering::Equal => match population[left].crowding_distance.total_cmp(&population[right].crowding_distance) {
            Ordering::Greater => left,
            Ordering::Less => right,
            Ordering::Equal => if rng.unit_f64() < 0.5 { left } else { right },
        },
    }
}

fn simulated_binary_crossover(
    parent_left: &[f64],
    parent_right: &[f64],
    crossover_probability: f64,
    distribution_index: f64,
    rng: &mut DeterministicRng,
) -> (Vec<f64>, Vec<f64>) {
    let mut child_left = parent_left.to_vec();
    let mut child_right = parent_right.to_vec();
    if rng.unit_f64() > crossover_probability {
        return (child_left, child_right);
    }
    let exponent = 1.0 / (distribution_index + 1.0);
    for index in 0..parent_left.len() {
        let first = parent_left[index];
        let second = parent_right[index];
        if rng.unit_f64() > 0.5 || (first - second).abs() <= 1e-14 {
            continue;
        }
        let lower = first.min(second);
        let upper = first.max(second);
        let random = rng.unit_f64();

        let beta_left = 1.0 + (2.0 * lower) / (upper - lower);
        let alpha_left = 2.0 - beta_left.powf(-(distribution_index + 1.0));
        let beta_q_left = if random <= 1.0 / alpha_left {
            (random * alpha_left).powf(exponent)
        } else {
            (1.0 / (2.0 - random * alpha_left)).powf(exponent)
        };
        let child_a = 0.5 * ((lower + upper) - beta_q_left * (upper - lower));

        let beta_right = 1.0 + (2.0 * (1.0 - upper)) / (upper - lower);
        let alpha_right = 2.0 - beta_right.powf(-(distribution_index + 1.0));
        let beta_q_right = if random <= 1.0 / alpha_right {
            (random * alpha_right).powf(exponent)
        } else {
            (1.0 / (2.0 - random * alpha_right)).powf(exponent)
        };
        let child_b = 0.5 * ((lower + upper) + beta_q_right * (upper - lower));
        let (first_child, second_child) = if rng.unit_f64() < 0.5 {
            (child_a, child_b)
        } else {
            (child_b, child_a)
        };
        child_left[index] = first_child.clamp(0.0, 1.0);
        child_right[index] = second_child.clamp(0.0, 1.0);
    }
    (child_left, child_right)
}

fn polynomial_mutation(
    decision: &mut [f64],
    mutation_probability: f64,
    distribution_index: f64,
    rng: &mut DeterministicRng,
) {
    let exponent = 1.0 / (distribution_index + 1.0);
    for value in decision {
        if rng.unit_f64() > mutation_probability {
            continue;
        }
        let random = rng.unit_f64();
        let delta_left = *value;
        let delta_right = 1.0 - *value;
        let delta = if random <= 0.5 {
            let xy = 1.0 - delta_left;
            let term = 2.0 * random + (1.0 - 2.0 * random) * xy.powf(distribution_index + 1.0);
            term.powf(exponent) - 1.0
        } else {
            let xy = 1.0 - delta_right;
            let term = 2.0 * (1.0 - random) + 2.0 * (random - 0.5) * xy.powf(distribution_index + 1.0);
            1.0 - term.powf(exponent)
        };
        *value = (*value + delta).clamp(0.0, 1.0);
    }
}

fn transform_objectives(
    objectives: &[f64],
    specs: &[ParetoObjectiveSpec],
) -> Result<Vec<f64>, SmsEmoaError> {
    if objectives.len() != specs.len() {
        return Err(SmsEmoaError::new("objectives", "objective/specification dimensions differ"));
    }
    objectives.iter().zip(specs).enumerate().map(|(index, (value, spec))| {
        let utility = (spec.scale_upper - *value) / (spec.scale_upper - spec.scale_lower);
        if !utility.is_finite() {
            return Err(SmsEmoaError::new(
                format!("objectives[{index}]"),
                "objective transform overflowed",
            ));
        }
        Ok(utility)
    }).collect()
}

fn evaluate_checked<F>(
    evaluator: &F,
    decision: &[f64],
    objective_dimensions: usize,
) -> Result<Vec<f64>, SmsEmoaError>
where
    F: Fn(&[f64]) -> Result<Vec<f64>, SmsEmoaError>,
{
    if decision.iter().any(|value| !value.is_finite() || !(0.0..=1.0).contains(value)) {
        return Err(SmsEmoaError::new("decision", "variation produced an invalid decision vector"));
    }
    let objectives = evaluator(decision)?;
    if objectives.len() != objective_dimensions || objectives.iter().any(|value| !value.is_finite()) {
        return Err(SmsEmoaError::new("objectives", "objective vector has invalid dimension or non-finite values"));
    }
    Ok(objectives)
}

fn dominates_minimization(left: &[f64], right: &[f64]) -> bool {
    left.iter().zip(right)
        .all(|(left_value, right_value)| left_value <= right_value)
        && left.iter().zip(right)
            .any(|(left_value, right_value)| left_value < right_value)
}

fn to_candidate(individual: &Individual) -> Result<SmsEmoaCandidate, SmsEmoaError> {
    Ok(SmsEmoaCandidate {
        candidate_id: digest_serializable(&individual.decision)?,
        decision_values: individual.decision.clone(),
        objective_values: individual.objectives.clone(),
    })
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SmsEmoaParetoAnalysis {
    pub plan: ParetoAnalysisPlan,
    pub analysis: ParetoAnalysisSummary,
}

pub fn verify_smsemoa_report(
    report: &SmsEmoaRunReport,
    config: &SmsEmoaConfig,
) -> Result<(), SmsEmoaError> {
    let recomputed = match report.problem_id.as_str() {
        "zdt1" => run_smsemoa_zdt1(config, report.decision_dimensions)?,
        "dtlz2-3obj" => run_smsemoa_dtlz2(config, report.decision_dimensions)?,
        other => return Err(SmsEmoaError::new("report.problem_id", format!("unsupported problem {other:?}"))),
    };
    if &recomputed != report {
        return Err(SmsEmoaError::new("report", "stored SMS-EMOA report differs from deterministic replay"));
    }
    Ok(())
}

pub fn analyze_smsemoa_report(
    report: &SmsEmoaRunReport,
    config: &SmsEmoaConfig,
) -> Result<SmsEmoaParetoAnalysis, SmsEmoaError> {
    verify_smsemoa_report(report, config)?;
    let mut unique_candidates = BTreeMap::<String, ParetoCandidate>::new();
    for candidate in &report.nondominated_front {
        let evidence_sha256 = digest_serializable(candidate)?;
        unique_candidates.entry(candidate.candidate_id.clone()).or_insert_with(|| ParetoCandidate {
            candidate_id: candidate.candidate_id.clone(),
            objective_values: candidate.objective_values.clone(),
            feasible: true,
            constraint_violations: Vec::new(),
            evidence_sha256,
        });
    }
    let plan = ParetoAnalysisPlan::new(
        format!("sms-emoa-{}-{}", report.problem_id, report.config_sha256),
        config.objective_specs.clone(),
        config.reference_point_utility.clone(),
        unique_candidates.into_values().collect(),
    ).map_err(map_pareto_error)?;
    let analysis = analyze_pareto(&plan).map_err(map_pareto_error)?;
    Ok(SmsEmoaParetoAnalysis { plan, analysis })
}

fn make_anytime_recorder<'a>(
    history: &'a mut AnytimeHistory,
    problem_id: &str,
    config: &SmsEmoaConfig,
    objective_dimensions: usize,
    expected_attempts: usize,
) -> Result<AnytimeRecorder<'a>, SmsEmoaError> {
    if history.header().algorithm_id != ALGORITHM_ID
        || history.header().problem_id != problem_id
        || history.header().seed != config.seed
    {
        return Err(SmsEmoaError::new("anytime.header", "history algorithm, problem, or seed differs from this SMS-EMOA run"));
    }
    if config.objective_specs.len() != objective_dimensions
        || history.header().objective_specs != config.objective_specs
        || config.objective_specs.iter().any(|spec| spec.direction != ObjectiveDirection::Minimize)
    {
        return Err(SmsEmoaError::new("anytime.objective_specs", "history objective definitions differ from SMS-EMOA configuration"));
    }
    if history.header().max_objective_evaluation_attempts < u64::try_from(expected_attempts).unwrap_or(u64::MAX) {
        return Err(SmsEmoaError::new("anytime.max_objective_evaluation_attempts", "history attempt budget is below the optimizer's exact budget"));
    }
    AnytimeRecorder::new(history).map_err(map_anytime_error)
}

fn map_anytime_error(error: AnytimeError) -> SmsEmoaError {
    SmsEmoaError::new(error.path, error.message)
}

fn map_pareto_error(error: ParetoError) -> SmsEmoaError {
    SmsEmoaError::new(error.path, error.message)
}

fn checked_add_usize(left: usize, right: usize, path: &str) -> Result<usize, SmsEmoaError> {
    left.checked_add(right)
        .ok_or_else(|| SmsEmoaError::new(path, "count overflowed usize"))
}

fn digest_serializable<T: Serialize>(value: &T) -> Result<String, SmsEmoaError> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        SmsEmoaError::new("serialization", format!("cannot serialize SMS-EMOA receipt: {error}"))
    })?;
    let digest = Sha256::digest(bytes);
    Ok(digest.iter().map(|byte| format!("{byte:02x}")).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zdt_config(seed: u64) -> SmsEmoaConfig {
        SmsEmoaConfig {
            population_size: 20,
            offspring_evaluations: 40,
            objective_specs: vec![
                ParetoObjectiveSpec {
                    objective_id: "f1".to_owned(),
                    direction: ObjectiveDirection::Minimize,
                    scale_lower: 0.0,
                    scale_upper: 1.0,
                },
                ParetoObjectiveSpec {
                    objective_id: "f2".to_owned(),
                    direction: ObjectiveDirection::Minimize,
                    scale_lower: 0.0,
                    scale_upper: 10.0,
                },
            ],
            reference_point_utility: vec![-0.1, -0.1],
            crossover_probability: 0.9,
            mutation_probability: 0.1,
            crossover_distribution_index: 15.0,
            mutation_distribution_index: 20.0,
            seed,
        }
    }

    fn dtlz_config(seed: u64) -> SmsEmoaConfig {
        SmsEmoaConfig {
            population_size: 20,
            offspring_evaluations: 40,
            objective_specs: vec![
                ParetoObjectiveSpec {
                    objective_id: "f1".to_owned(),
                    direction: ObjectiveDirection::Minimize,
                    scale_lower: 0.0,
                    scale_upper: 2.0,
                },
                ParetoObjectiveSpec {
                    objective_id: "f2".to_owned(),
                    direction: ObjectiveDirection::Minimize,
                    scale_lower: 0.0,
                    scale_upper: 2.0,
                },
                ParetoObjectiveSpec {
                    objective_id: "f3".to_owned(),
                    direction: ObjectiveDirection::Minimize,
                    scale_lower: 0.0,
                    scale_upper: 2.0,
                },
            ],
            reference_point_utility: vec![-0.1, -0.1, -0.1],
            crossover_probability: 0.9,
            mutation_probability: 0.1,
            crossover_distribution_index: 15.0,
            mutation_distribution_index: 20.0,
            seed,
        }
    }

    #[test]
    fn smsemoa_zdt1_is_reproducible_with_fixed_objective_budget() {
        let config = zdt_config(42);
        let left = run_smsemoa_zdt1(&config, 10).unwrap();
        let right = run_smsemoa_zdt1(&config, 10).unwrap();
        assert_eq!(left.report_sha256, right.report_sha256);
        assert_eq!(left.nondominated_front, right.nondominated_front);
        assert_eq!(left.objective_evaluations, 60);
        assert_eq!(left.final_population.len(), 20);
    }

    #[test]
    fn smsemoa_dtlz2_preserves_three_objective_bounds_and_archive() {
        let report = run_smsemoa_dtlz2(&dtlz_config(9), 7).unwrap();
        assert_eq!(report.objective_dimensions, 3);
        assert_eq!(report.objective_evaluations, 60);
        assert_eq!(report.final_population.len(), 20);
        assert!(report.final_population.iter().all(|candidate| {
            candidate.decision_values.iter().all(|value| value.is_finite() && (0.0..=1.0).contains(value))
                && candidate.objective_values.len() == 3
                && candidate.objective_values.iter().all(|value| value.is_finite())
        }));
    }

    #[test]
    fn report_replay_and_shared_pareto_indicators_are_bound_to_sms_archive() {
        let config = zdt_config(81);
        let report = run_smsemoa_zdt1(&config, 10).unwrap();
        verify_smsemoa_report(&report, &config).unwrap();
        let analysis = analyze_smsemoa_report(&report, &config).unwrap();
        assert!(!analysis.analysis.nondominated_candidate_ids.is_empty());

        let mut tampered = report.clone();
        tampered.final_population[0].objective_values[0] += 0.01;
        assert!(verify_smsemoa_report(&tampered, &config).is_err());
    }

    #[test]
    fn hypervolume_contribution_removes_the_least_contributing_point() {
        let population = vec![
            Individual { decision: vec![0.0], objectives: vec![0.2, 0.8], utility: vec![0.8, 0.2], rank: 0, crowding_distance: 0.0, ordinal: 0 },
            Individual { decision: vec![0.1], objectives: vec![0.8, 0.2], utility: vec![0.2, 0.8], rank: 0, crowding_distance: 0.0, ordinal: 1 },
            Individual { decision: vec![0.2], objectives: vec![0.6, 0.45], utility: vec![0.4, 0.55], rank: 0, crowding_distance: 0.0, ordinal: 2 },
            Individual { decision: vec![0.3], objectives: vec![0.45, 0.6], utility: vec![0.55, 0.4], rank: 0, crowding_distance: 0.0, ordinal: 3 },
        ];
        let selected = environmental_selection(population, 3, &[0.0, 0.0]).unwrap();
        assert_eq!(selected.len(), 3);
        assert!(!selected.iter().any(|individual| individual.ordinal == 3));
    }

    #[test]
    fn invalid_configurations_and_non_minimization_objectives_are_rejected() {
        let mut bad = zdt_config(1);
        bad.population_size = MAX_POPULATION_SIZE + 1;
        assert!(run_smsemoa_zdt1(&bad, 10).is_err());

        let mut bad_direction = zdt_config(1);
        bad_direction.objective_specs[0].direction = ObjectiveDirection::Maximize;
        assert!(run_smsemoa_zdt1(&bad_direction, 10).is_err());

        let mut bad_budget = zdt_config(1);
        bad_budget.offspring_evaluations = 0;
        assert!(run_smsemoa_zdt1(&bad_budget, 10).is_err());
    }
}
