//! Deterministic NSGA-II reference optimizer for the ZDT1 and DTLZ2 fixtures.
//!
//! This is a reference baseline, not a claim of state-of-the-art performance. Objectives are
//! minimized. The implementation uses binary tournament selection, simulated binary crossover,
//! polynomial mutation, fast non-dominated sorting and crowding-distance environmental selection.

use std::cmp::Ordering;
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::pareto_benchmarks::{dtlz2_three_objectives, zdt1_objectives};

const SCHEMA: &str = "symthaea.architecture-engineering.pareto-nsga2";
const VERSION: &str = "0.1.0";
const ALGORITHM_ID: &str = "nsga-ii-reference-v0.1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Nsga2Error {
    pub path: String,
    pub message: String,
}

impl Nsga2Error {
    fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for Nsga2Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}

impl Error for Nsga2Error {}

/// Frozen NSGA-II parameters. Mutation probability is per decision variable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Nsga2Config {
    pub population_size: usize,
    pub generations: usize,
    pub crossover_probability: f64,
    pub mutation_probability: f64,
    pub crossover_distribution_index: f64,
    pub mutation_distribution_index: f64,
    pub seed: u64,
}

impl Nsga2Config {
    pub fn validate(&self) -> Result<(), Nsga2Error> {
        if self.population_size < 4 {
            return Err(Nsga2Error::new(
                "config.population_size",
                "NSGA-II population size must be at least four",
            ));
        }
        if self.generations == 0 {
            return Err(Nsga2Error::new(
                "config.generations",
                "at least one evolutionary generation is required",
            ));
        }
        for (path, value) in [
            ("crossover_probability", self.crossover_probability),
            ("mutation_probability", self.mutation_probability),
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(Nsga2Error::new(
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
                return Err(Nsga2Error::new(
                    format!("config.{path}"),
                    "distribution index must be finite and positive",
                ));
            }
        }
        self.population_size.checked_mul(self.generations.saturating_add(1))
            .ok_or_else(|| Nsga2Error::new("config", "evaluation budget overflows usize"))?;
        Ok(())
    }

    pub fn config_sha256(&self) -> Result<String, Nsga2Error> {
        self.validate()?;
        digest_serializable(self)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Nsga2Candidate {
    pub candidate_id: String,
    pub decision_values: Vec<f64>,
    /// Objective values are in the problem's native minimization convention.
    pub objective_values: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Nsga2RunReport {
    pub schema: String,
    pub version: String,
    pub algorithm_id: String,
    pub problem_id: String,
    pub seed: u64,
    pub config_sha256: String,
    pub decision_dimensions: usize,
    pub objective_dimensions: usize,
    pub generations_completed: usize,
    pub objective_evaluations: usize,
    pub final_population: Vec<Nsga2Candidate>,
    pub nondominated_front: Vec<Nsga2Candidate>,
    pub report_sha256: String,
}

#[derive(Serialize)]
struct Nsga2ReportPayload<'a> {
    schema: &'a str,
    version: &'a str,
    algorithm_id: &'a str,
    problem_id: &'a str,
    seed: u64,
    config_sha256: &'a str,
    decision_dimensions: usize,
    objective_dimensions: usize,
    generations_completed: usize,
    objective_evaluations: usize,
    final_population: &'a [Nsga2Candidate],
    nondominated_front: &'a [Nsga2Candidate],
}

#[derive(Debug, Clone)]
struct Individual {
    decision: Vec<f64>,
    objectives: Vec<f64>,
    rank: usize,
    crowding_distance: f64,
    ordinal: usize,
}

struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    fn new(seed: u64) -> Self {
        let state = if seed == 0 { 0x9e37_79b9_7f4a_7c15 } else { seed };
        Self { state }
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

    fn shuffle<T>(&mut self, values: &mut [T]) {
        for index in (1..values.len()).rev() {
            let swap_index = self.index(index + 1);
            values.swap(index, swap_index);
        }
    }
}

pub fn run_nsga2_zdt1(
    config: &Nsga2Config,
    decision_dimensions: usize,
) -> Result<Nsga2RunReport, Nsga2Error> {
    if decision_dimensions < 2 {
        return Err(Nsga2Error::new(
            "decision_dimensions",
            "ZDT1 requires at least two decision variables",
        ));
    }
    run_nsga2(
        "zdt1",
        config,
        decision_dimensions,
        2,
        |decision| {
            let objectives = zdt1_objectives(decision)
                .map_err(|error| Nsga2Error::new("zdt1", error.to_string()))?;
            Ok(objectives.to_vec())
        },
    )
}

pub fn run_nsga2_dtlz2(
    config: &Nsga2Config,
    decision_dimensions: usize,
) -> Result<Nsga2RunReport, Nsga2Error> {
    if decision_dimensions < 3 {
        return Err(Nsga2Error::new(
            "decision_dimensions",
            "three-objective DTLZ2 requires at least three decision variables",
        ));
    }
    run_nsga2(
        "dtlz2-3obj",
        config,
        decision_dimensions,
        3,
        |decision| {
            let objectives = dtlz2_three_objectives(decision)
                .map_err(|error| Nsga2Error::new("dtlz2", error.to_string()))?;
            Ok(objectives.to_vec())
        },
    )
}

fn run_nsga2<F>(
    problem_id: &str,
    config: &Nsga2Config,
    decision_dimensions: usize,
    objective_dimensions: usize,
    evaluator: F,
) -> Result<Nsga2RunReport, Nsga2Error>
where
    F: Fn(&[f64]) -> Result<Vec<f64>, Nsga2Error>,
{
    config.validate()?;
    if decision_dimensions == 0 || objective_dimensions < 2 {
        return Err(Nsga2Error::new(
            "dimensions",
            "decision dimensions must be positive and objective dimensions at least two",
        ));
    }
    let config_sha256 = config.config_sha256()?;
    let mut rng = DeterministicRng::new(config.seed);
    let mut objective_evaluations = 0usize;
    let mut population = Vec::with_capacity(config.population_size);

    for ordinal in 0..config.population_size {
        let decision = (0..decision_dimensions)
            .map(|_| rng.unit_f64())
            .collect::<Vec<_>>();
        let objectives = evaluate_checked(&evaluator, &decision, objective_dimensions)?;
        objective_evaluations = checked_add_usize(objective_evaluations, 1, "objective_evaluations")?;
        population.push(Individual {
            decision,
            objectives,
            rank: usize::MAX,
            crowding_distance: 0.0,
            ordinal,
        });
    }

    for generation in 0..config.generations {
        assign_rank_and_crowding(&mut population)?;
        let mut offspring_decisions = Vec::with_capacity(config.population_size);
        while offspring_decisions.len() < config.population_size {
            let parent_left = binary_tournament(&population, &mut rng);
            let parent_right = binary_tournament(&population, &mut rng);
            let (mut child_left, mut child_right) = simulated_binary_crossover(
                &population[parent_left].decision,
                &population[parent_right].decision,
                config.crossover_probability,
                config.crossover_distribution_index,
                &mut rng,
            );
            polynomial_mutation(
                &mut child_left,
                config.mutation_probability,
                config.mutation_distribution_index,
                &mut rng,
            );
            polynomial_mutation(
                &mut child_right,
                config.mutation_probability,
                config.mutation_distribution_index,
                &mut rng,
            );
            offspring_decisions.push(child_left);
            if offspring_decisions.len() < config.population_size {
                offspring_decisions.push(child_right);
            }
        }

        let mut offspring = Vec::with_capacity(config.population_size);
        for (index, decision) in offspring_decisions.into_iter().enumerate() {
            let objectives = evaluate_checked(&evaluator, &decision, objective_dimensions)?;
            objective_evaluations = checked_add_usize(objective_evaluations, 1, "objective_evaluations")?;
            offspring.push(Individual {
                decision,
                objectives,
                rank: usize::MAX,
                crowding_distance: 0.0,
                ordinal: config.population_size + index,
            });
        }

        let mut combined = population;
        combined.extend(offspring);
        population = environmental_selection(combined, config.population_size)?;
        if generation + 1 == config.generations {
            break;
        }
    }

    let front_population = population.clone();
    let fronts = fast_non_dominated_sort(&front_population)?;
    let first_front = fronts.first().cloned().unwrap_or_default();
    let mut final_population = population.iter()
        .map(to_candidate)
        .collect::<Result<Vec<_>, _>>()?;
    let mut nondominated_front = first_front.iter()
        .map(|index| to_candidate(&front_population[*index]))
        .collect::<Result<Vec<_>, _>>()?;
    final_population.sort_by(|left, right| left.candidate_id.cmp(&right.candidate_id));
    nondominated_front.sort_by(|left, right| left.candidate_id.cmp(&right.candidate_id));

    let report_sha256 = digest_serializable(&Nsga2ReportPayload {
        schema: SCHEMA,
        version: VERSION,
        algorithm_id: ALGORITHM_ID,
        problem_id,
        seed: config.seed,
        config_sha256: &config_sha256,
        decision_dimensions,
        objective_dimensions,
        generations_completed: config.generations,
        objective_evaluations,
        final_population: &final_population,
        nondominated_front: &nondominated_front,
    })?;

    Ok(Nsga2RunReport {
        schema: SCHEMA.to_owned(),
        version: VERSION.to_owned(),
        algorithm_id: ALGORITHM_ID.to_owned(),
        problem_id: problem_id.to_owned(),
        seed: config.seed,
        config_sha256,
        decision_dimensions,
        objective_dimensions,
        generations_completed: config.generations,
        objective_evaluations,
        final_population,
        nondominated_front,
        report_sha256,
    })
}

fn evaluate_checked<F>(
    evaluator: &F,
    decision: &[f64],
    objective_dimensions: usize,
) -> Result<Vec<f64>, Nsga2Error>
where
    F: Fn(&[f64]) -> Result<Vec<f64>, Nsga2Error>,
{
    if decision.iter().any(|value| !value.is_finite() || !(0.0..=1.0).contains(value)) {
        return Err(Nsga2Error::new(
            "decision",
            "NSGA-II operator produced a non-finite or out-of-bounds decision variable",
        ));
    }
    let objectives = evaluator(decision)?;
    if objectives.len() != objective_dimensions
        || objectives.iter().any(|value| !value.is_finite())
    {
        return Err(Nsga2Error::new(
            "objectives",
            "objective evaluator returned a non-finite vector or incorrect dimension",
        ));
    }
    Ok(objectives)
}

fn binary_tournament(population: &[Individual], rng: &mut DeterministicRng) -> usize {
    let left = rng.index(population.len());
    let right = rng.index(population.len());
    match population[left].rank.cmp(&population[right].rank) {
        Ordering::Less => left,
        Ordering::Greater => right,
        Ordering::Equal => match population[left].crowding_distance
            .total_cmp(&population[right].crowding_distance)
        {
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

fn environmental_selection(
    mut combined: Vec<Individual>,
    population_size: usize,
) -> Result<Vec<Individual>, Nsga2Error> {
    let fronts = fast_non_dominated_sort(&combined)?;
    let mut selected = Vec::with_capacity(population_size);
    for front in fronts {
        assign_crowding_for_front(&mut combined, &front)?;
        if selected.len() + front.len() <= population_size {
            selected.extend(front.into_iter().map(|index| combined[index].clone()));
        } else {
            let mut partial = front;
            partial.sort_by(|left, right| {
                combined[*right].crowding_distance
                    .total_cmp(&combined[*left].crowding_distance)
                    .then_with(|| combined[*left].ordinal.cmp(&combined[*right].ordinal))
            });
            let remaining = population_size - selected.len();
            selected.extend(partial.into_iter().take(remaining).map(|index| combined[index].clone()));
            break;
        }
        if selected.len() == population_size {
            break;
        }
    }
    if selected.len() != population_size {
        return Err(Nsga2Error::new(
            "environmental_selection",
            "non-dominated sorting failed to fill the configured population",
        ));
    }
    Ok(selected)
}

fn assign_rank_and_crowding(population: &mut [Individual]) -> Result<(), Nsga2Error> {
    let fronts = fast_non_dominated_sort(population)?;
    for (rank, front) in fronts.iter().enumerate() {
        assign_crowding_for_front(population, front)?;
        for index in front {
            population[*index].rank = rank;
        }
    }
    Ok(())
}

fn fast_non_dominated_sort(population: &[Individual]) -> Result<Vec<Vec<usize>>, Nsga2Error> {
    let count = population.len();
    let mut dominates_set = vec![Vec::<usize>::new(); count];
    let mut dominated_by_count = vec![0usize; count];
    let mut first_front = Vec::new();

    for left in 0..count {
        for right in 0..count {
            if left == right {
                continue;
            }
            if dominates_minimization(
                &population[left].objectives,
                &population[right].objectives,
            ) {
                dominates_set[left].push(right);
            } else if dominates_minimization(
                &population[right].objectives,
                &population[left].objectives,
            ) {
                dominated_by_count[left] = dominated_by_count[left]
                    .checked_add(1)
                    .ok_or_else(|| Nsga2Error::new("non_dominated_sort", "domination count overflow"))?;
            }
        }
        if dominated_by_count[left] == 0 {
            first_front.push(left);
        }
    }

    let mut fronts = Vec::new();
    if first_front.is_empty() && !population.is_empty() {
        return Err(Nsga2Error::new("non_dominated_sort", "no first front was identified"));
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
                    .ok_or_else(|| Nsga2Error::new("non_dominated_sort", "domination counter underflow"))?;
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
        return Err(Nsga2Error::new(
            "non_dominated_sort",
            "front decomposition did not assign every candidate",
        ));
    }
    Ok(fronts)
}

fn assign_crowding_for_front(
    population: &mut [Individual],
    front: &[usize],
) -> Result<(), Nsga2Error> {
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
        population[first].crowding_distance = f64::INFINITY;
        population[last].crowding_distance = f64::INFINITY;
        let minimum = population[first].objectives[objective];
        let maximum = population[last].objectives[objective];
        let range = maximum - minimum;
        if range <= f64::EPSILON {
            continue;
        }
        for position in 1..(order.len() - 1) {
            let index = order[position];
            if population[index].crowding_distance.is_infinite() {
                continue;
            }
            let previous = population[order[position - 1]].objectives[objective];
            let next = population[order[position + 1]].objectives[objective];
            let contribution = (next - previous) / range;
            if !contribution.is_finite() {
                return Err(Nsga2Error::new(
                    "crowding_distance",
                    "crowding distance contribution is not finite",
                ));
            }
            population[index].crowding_distance += contribution;
        }
    }
    Ok(())
}

fn dominates_minimization(left: &[f64], right: &[f64]) -> bool {
    left.iter().zip(right)
        .all(|(left_value, right_value)| left_value <= right_value)
        && left.iter().zip(right)
            .any(|(left_value, right_value)| left_value < right_value)
}

fn to_candidate(individual: &Individual) -> Result<Nsga2Candidate, Nsga2Error> {
    let candidate_id = digest_serializable(&individual.decision)?;
    Ok(Nsga2Candidate {
        candidate_id,
        decision_values: individual.decision.clone(),
        objective_values: individual.objectives.clone(),
    })
}

fn checked_add_usize(left: usize, right: usize, path: &str) -> Result<usize, Nsga2Error> {
    left.checked_add(right)
        .ok_or_else(|| Nsga2Error::new(path, "count overflowed usize"))
}

fn digest_serializable<T: Serialize>(value: &T) -> Result<String, Nsga2Error> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        Nsga2Error::new("serialization", format!("cannot serialize NSGA-II receipt: {error}"))
    })?;
    let digest = Sha256::digest(bytes);
    Ok(digest.iter().map(|byte| format!("{byte:02x}")).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(seed: u64) -> Nsga2Config {
        Nsga2Config {
            population_size: 20,
            generations: 5,
            crossover_probability: 0.9,
            mutation_probability: 0.1,
            crossover_distribution_index: 15.0,
            mutation_distribution_index: 20.0,
            seed,
        }
    }

    #[test]
    fn nsga2_zdt1_is_reproducible_for_frozen_seed_and_config() {
        let cfg = config(42);
        let left = run_nsga2_zdt1(&cfg, 10).unwrap();
        let right = run_nsga2_zdt1(&cfg, 10).unwrap();
        assert_eq!(left.report_sha256, right.report_sha256);
        assert_eq!(left.nondominated_front, right.nondominated_front);
        assert_eq!(left.objective_evaluations, 120);
    }

    #[test]
    fn nsga2_dtlz2_respects_bounds_and_objective_dimensions() {
        let report = run_nsga2_dtlz2(&config(7), 7).unwrap();
        assert_eq!(report.objective_dimensions, 3);
        assert_eq!(report.final_population.len(), 20);
        assert_eq!(report.objective_evaluations, 120);
        assert!(report.final_population.iter().all(|candidate| {
            candidate.decision_values.iter().all(|value| value.is_finite() && (0.0..=1.0).contains(value))
                && candidate.objective_values.len() == 3
                && candidate.objective_values.iter().all(|value| value.is_finite())
        }));
    }

    #[test]
    fn optimizer_rejects_invalid_parameters_and_problem_dimensions() {
        let mut bad = config(1);
        bad.population_size = 3;
        assert!(run_nsga2_zdt1(&bad, 10).is_err());
        assert!(run_nsga2_zdt1(&config(1), 1).is_err());
        assert!(run_nsga2_dtlz2(&config(1), 2).is_err());
    }

    #[test]
    fn minimization_dominance_requires_all_objectives_no_worse() {
        assert!(dominates_minimization(&[0.2, 0.3], &[0.3, 0.4]));
        assert!(!dominates_minimization(&[0.2, 0.5], &[0.3, 0.4]));
        assert!(!dominates_minimization(&[0.2, 0.3], &[0.2, 0.3]));
    }
}
