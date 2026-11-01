//! Deterministic MOEA/D-DE reference optimizer for the ZDT1 and DTLZ2 fixtures.
//!
//! Implements decomposition into scalar subproblems with uniform 2D weights or a 3D
//! simplex-lattice design, neighbourhood mating, differential-evolution variation,
//! Tchebycheff scalarization and bounded neighbourhood replacement. Objectives are minimized.
//! This is a reference baseline, not a claim of state-of-the-art performance.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::pareto_anytime::{AnytimeError, AnytimeHistory, AnytimeRecorder, EvaluationOutcome};
use super::pareto_benchmarks::{dtlz2_three_objectives, zdt1_objectives};
use super::pareto_objectives::{
    analyze_pareto, ObjectiveDirection, ParetoAnalysisPlan, ParetoAnalysisSummary,
    ParetoCandidate, ParetoError, ParetoObjectiveSpec,
};

const SCHEMA: &str = "symthaea.architecture-engineering.pareto-moead";
const VERSION: &str = "0.1.0";
const ALGORITHM_ID: &str = "moead-de-reference-v0.1";
const ZERO_WEIGHT_FLOOR: f64 = 1e-6;
const MAX_POPULATION_SIZE: usize = 2_048;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoeadError {
    pub path: String,
    pub message: String,
}

impl MoeadError {
    fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for MoeadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}

impl Error for MoeadError {}

/// Weight construction mode fixes the objective decomposition and population size.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MoeadWeightMode {
    /// Uniform line weights for exactly two objectives.
    Uniform2d { population_size: usize },
    /// Integer simplex lattice for exactly three objectives; population is (H+1)(H+2)/2.
    Simplex3d { divisions: usize },
}

impl MoeadWeightMode {
    pub fn population_size(&self) -> Result<usize, MoeadError> {
        match self {
            Self::Uniform2d { population_size } => {
                if *population_size < 4 || *population_size > MAX_POPULATION_SIZE {
                    return Err(MoeadError::new(
                        "weight_mode.population_size",
                        format!("two-objective population must be between four and {MAX_POPULATION_SIZE}"),
                    ));
                }
                Ok(*population_size)
            }
            Self::Simplex3d { divisions } => {
                if *divisions < 2 {
                    return Err(MoeadError::new(
                        "weight_mode.divisions",
                        "three-objective simplex divisions must be at least two",
                    ));
                }
                let h = divisions.checked_add(1)
                    .ok_or_else(|| MoeadError::new("weight_mode.divisions", "division count overflow"))?;
                let count = h.checked_mul(divisions.checked_add(2).ok_or_else(|| {
                    MoeadError::new("weight_mode.divisions", "division count overflow")
                })?)
                    .and_then(|value| value.checked_div(2))
                    .ok_or_else(|| MoeadError::new("weight_mode", "simplex population count overflow"))?;
                if count > MAX_POPULATION_SIZE {
                    return Err(MoeadError::new(
                        "weight_mode.divisions",
                        format!("simplex population cannot exceed {MAX_POPULATION_SIZE}"),
                    ));
                }
                Ok(count)
            }
        }
    }

    fn objective_dimensions(&self) -> usize {
        match self {
            Self::Uniform2d { .. } => 2,
            Self::Simplex3d { .. } => 3,
        }
    }

    fn weights(&self) -> Result<Vec<Vec<f64>>, MoeadError> {
        let weights = match self {
            Self::Uniform2d { population_size } => {
                let denominator = (*population_size - 1) as f64;
                (0..*population_size).map(|index| {
                    let first = index as f64 / denominator;
                    vec![first, 1.0 - first]
                }).collect()
            }
            Self::Simplex3d { divisions } => {
                let denominator = *divisions as f64;
                let mut values = Vec::with_capacity(self.population_size()?);
                for first in 0..=*divisions {
                    for second in 0..=(*divisions - first) {
                        let third = *divisions - first - second;
                        values.push(vec![
                            first as f64 / denominator,
                            second as f64 / denominator,
                            third as f64 / denominator,
                        ]);
                    }
                }
                values
            }
        };
        if weights.len() != self.population_size()? {
            return Err(MoeadError::new(
                "weight_mode.weights",
                "weight-vector count differs from configured population",
            ));
        }
        Ok(weights)
    }
}

/// Frozen MOEA/D-DE parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MoeadConfig {
    pub weight_mode: MoeadWeightMode,
    pub generations: usize,
    pub neighbourhood_size: usize,
    /// Probability of selecting all three differential-evolution parents from the neighbourhood.
    pub neighbourhood_mating_probability: f64,
    /// Differential weight F in [0, 2].
    pub differential_weight: f64,
    /// Binomial crossover probability CR in [0, 1].
    pub crossover_rate: f64,
    /// Maximum number of neighbouring subproblems replaced by one offspring.
    pub max_replacements: usize,
    pub seed: u64,
}

impl MoeadConfig {
    pub fn validate(&self) -> Result<(), MoeadError> {
        let population_size = self.weight_mode.population_size()?;
        if self.generations == 0 {
            return Err(MoeadError::new("config.generations", "at least one generation is required"));
        }
        if self.neighbourhood_size < 4 || self.neighbourhood_size > population_size {
            return Err(MoeadError::new(
                "config.neighbourhood_size",
                "neighbourhood size must be between four and population size",
            ));
        }
        if !self.neighbourhood_mating_probability.is_finite()
            || !(0.0..=1.0).contains(&self.neighbourhood_mating_probability)
        {
            return Err(MoeadError::new(
                "config.neighbourhood_mating_probability",
                "neighbourhood mating probability must be finite and in [0, 1]",
            ));
        }
        if !self.differential_weight.is_finite()
            || self.differential_weight <= 0.0
            || self.differential_weight > 2.0
        {
            return Err(MoeadError::new(
                "config.differential_weight",
                "differential weight must be finite and in (0, 2]",
            ));
        }
        if !self.crossover_rate.is_finite() || !(0.0..=1.0).contains(&self.crossover_rate) {
            return Err(MoeadError::new(
                "config.crossover_rate",
                "crossover rate must be finite and in [0, 1]",
            ));
        }
        if self.max_replacements == 0 || self.max_replacements > self.neighbourhood_size {
            return Err(MoeadError::new(
                "config.max_replacements",
                "maximum replacements must be in [1, neighbourhood_size]",
            ));
        }
        population_size.checked_mul(self.generations.saturating_add(1))
            .ok_or_else(|| MoeadError::new("config", "objective evaluation budget overflows usize"))?;
        Ok(())
    }

    pub fn config_sha256(&self) -> Result<String, MoeadError> {
        self.validate()?;
        digest_serializable(self)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MoeadCandidate {
    pub candidate_id: String,
    pub decision_values: Vec<f64>,
    /// Native minimization objective vector.
    pub objective_values: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MoeadRunReport {
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
    pub weight_vectors: Vec<Vec<f64>>,
    pub final_population: Vec<MoeadCandidate>,
    pub nondominated_front: Vec<MoeadCandidate>,
    pub report_sha256: String,
}

#[derive(Serialize)]
struct MoeadReportPayload<'a> {
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
    weight_vectors: &'a [Vec<f64>],
    final_population: &'a [MoeadCandidate],
    nondominated_front: &'a [MoeadCandidate],
}

#[derive(Debug, Clone)]
struct Individual {
    decision: Vec<f64>,
    objectives: Vec<f64>,
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

    fn shuffle<T>(&mut self, values: &mut [T]) {
        for index in (1..values.len()).rev() {
            values.swap(index, self.index(index + 1));
        }
    }
}

pub fn run_moead_zdt1(
    config: &MoeadConfig,
    decision_dimensions: usize,
) -> Result<MoeadRunReport, MoeadError> {
    if decision_dimensions < 2 || config.weight_mode.objective_dimensions() != 2 {
        return Err(MoeadError::new(
            "dimensions",
            "ZDT1 requires at least two decision variables and Uniform2d weights",
        ));
    }
    run_moead(
        "zdt1",
        config,
        decision_dimensions,
        2,
        |decision| {
            zdt1_objectives(decision)
                .map(|values| values.to_vec())
                .map_err(|error| MoeadError::new("zdt1", error.to_string()))
        },
        None,
    )
}

pub fn run_moead_zdt1_with_history(
    config: &MoeadConfig,
    decision_dimensions: usize,
    history: &mut AnytimeHistory,
) -> Result<MoeadRunReport, MoeadError> {
    if decision_dimensions < 2 || config.weight_mode.objective_dimensions() != 2 {
        return Err(MoeadError::new("dimensions", "ZDT1 requires at least two decision variables and Uniform2d weights"));
    }
    let population_size = config.weight_mode.population_size()?;
    let budget = population_size.checked_mul(config.generations.saturating_add(1))
        .ok_or_else(|| MoeadError::new("config", "evaluation budget overflows usize"))?;
    let mut recorder = make_anytime_recorder(history, "zdt1", config.seed, 2, budget)?;
    run_moead("zdt1", config, decision_dimensions, 2, |decision| {
        zdt1_objectives(decision).map(|values| values.to_vec())
            .map_err(|error| MoeadError::new("zdt1", error.to_string()))
    }, Some(&mut recorder))
}

pub fn run_moead_dtlz2(
    config: &MoeadConfig,
    decision_dimensions: usize,
) -> Result<MoeadRunReport, MoeadError> {
    if decision_dimensions < 3 || config.weight_mode.objective_dimensions() != 3 {
        return Err(MoeadError::new(
            "dimensions",
            "three-objective DTLZ2 requires at least three decision variables and Simplex3d weights",
        ));
    }
    run_moead(
        "dtlz2-3obj",
        config,
        decision_dimensions,
        3,
        |decision| {
            dtlz2_three_objectives(decision)
                .map(|values| values.to_vec())
                .map_err(|error| MoeadError::new("dtlz2", error.to_string()))
        },
        None,
    )
}

pub fn run_moead_dtlz2_with_history(
    config: &MoeadConfig,
    decision_dimensions: usize,
    history: &mut AnytimeHistory,
) -> Result<MoeadRunReport, MoeadError> {
    if decision_dimensions < 3 || config.weight_mode.objective_dimensions() != 3 {
        return Err(MoeadError::new("dimensions", "three-objective DTLZ2 requires at least three decision variables and Simplex3d weights"));
    }
    let population_size = config.weight_mode.population_size()?;
    let budget = population_size.checked_mul(config.generations.saturating_add(1))
        .ok_or_else(|| MoeadError::new("config", "evaluation budget overflows usize"))?;
    let mut recorder = make_anytime_recorder(history, "dtlz2-3obj", config.seed, 3, budget)?;
    run_moead("dtlz2-3obj", config, decision_dimensions, 3, |decision| {
        dtlz2_three_objectives(decision).map(|values| values.to_vec())
            .map_err(|error| MoeadError::new("dtlz2", error.to_string()))
    }, Some(&mut recorder))
}

fn run_moead<F>(
    problem_id: &str,
    config: &MoeadConfig,
    decision_dimensions: usize,
    objective_dimensions: usize,
    evaluator: F,
    mut recorder: Option<&mut AnytimeRecorder<'_>>,
) -> Result<MoeadRunReport, MoeadError>
where
    F: Fn(&[f64]) -> Result<Vec<f64>, MoeadError>,
{
    config.validate()?;
    if config.weight_mode.objective_dimensions() != objective_dimensions {
        return Err(MoeadError::new("weight_mode", "weight mode objective dimension mismatch"));
    }
    let population_size = config.weight_mode.population_size()?;
    let weight_vectors = config.weight_mode.weights()?;
    let neighbourhoods = build_neighbourhoods(&weight_vectors, config.neighbourhood_size)?;
    let config_sha256 = config.config_sha256()?;
    let mut rng = DeterministicRng::new(config.seed);
    let mut objective_evaluations = 0usize;
    let mut population = Vec::with_capacity(population_size);
    let mut ideal = vec![f64::INFINITY; objective_dimensions];

    for _ in 0..population_size {
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
        objective_evaluations = checked_add_usize(objective_evaluations, 1, "objective_evaluations")?;
        update_ideal(&mut ideal, &objectives)?;
        population.push(Individual { decision, objectives });
    }

    let mut order = (0..population_size).collect::<Vec<_>>();
    for _generation in 0..config.generations {
        rng.shuffle(&mut order);
        for subproblem in &order {
            let pool = if rng.unit_f64() < config.neighbourhood_mating_probability {
                neighbourhoods[*subproblem].clone()
            } else {
                (0..population_size).collect()
            };
            let [parent_a, parent_b, parent_c] =
                select_three_distinct_parents(&pool, *subproblem, &mut rng)?;
            let child = differential_evolution_child(
                &population[*subproblem].decision,
                &population[parent_a].decision,
                &population[parent_b].decision,
                &population[parent_c].decision,
                config.differential_weight,
                config.crossover_rate,
                &mut rng,
            )?;
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
            objective_evaluations = checked_add_usize(objective_evaluations, 1, "objective_evaluations")?;
            update_ideal(&mut ideal, &objectives)?;

            let mut replacement_order = neighbourhoods[*subproblem].clone();
            rng.shuffle(&mut replacement_order);
            let mut replacements = 0usize;
            for neighbour in replacement_order {
                let current_value = tchebycheff(
                    &population[neighbour].objectives,
                    &weight_vectors[neighbour],
                    &ideal,
                )?;
                let child_value = tchebycheff(
                    &objectives,
                    &weight_vectors[neighbour],
                    &ideal,
                )?;
                if child_value < current_value {
                    population[neighbour] = Individual {
                        decision: child.clone(),
                        objectives: objectives.clone(),
                    };
                    replacements += 1;
                    if replacements >= config.max_replacements {
                        break;
                    }
                }
            }
        }
    }

    let final_population = population.iter()
        .map(to_candidate)
        .collect::<Result<Vec<_>, _>>()?;
    let front_indices = nondominated_indices(&population)?;
    let nondominated_front = front_indices.iter()
        .map(|index| to_candidate(&population[*index]))
        .collect::<Result<Vec<_>, _>>()?;
    let mut nondominated_front = nondominated_front;
    // Final population order is the subproblem/weight-vector order and must be preserved.
    nondominated_front.sort_by(|left, right| left.candidate_id.cmp(&right.candidate_id));

    let report_sha256 = digest_serializable(&MoeadReportPayload {
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
        weight_vectors: &weight_vectors,
        final_population: &final_population,
        nondominated_front: &nondominated_front,
    })?;

    Ok(MoeadRunReport {
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
        weight_vectors,
        final_population,
        nondominated_front,
        report_sha256,
    })
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MoeadParetoAnalysis {
    pub plan: ParetoAnalysisPlan,
    pub analysis: ParetoAnalysisSummary,
}

/// Deterministically replay a MOEA/D report from the exact problem, configuration and seed.
pub fn verify_moead_report(
    report: &MoeadRunReport,
    config: &MoeadConfig,
) -> Result<(), MoeadError> {
    let recomputed = match report.problem_id.as_str() {
        "zdt1" => run_moead_zdt1(config, report.decision_dimensions)?,
        "dtlz2-3obj" => run_moead_dtlz2(config, report.decision_dimensions)?,
        other => {
            return Err(MoeadError::new(
                "report.problem_id",
                format!("unsupported MOEA/D problem ID {other:?}"),
            ));
        }
    };
    if &recomputed != report {
        return Err(MoeadError::new(
            "report",
            "stored MOEA/D report differs from deterministic replay",
        ));
    }
    Ok(())
}

/// Convert a replay-verified nondominated archive to the shared Pareto indicator path.
/// Repeated decision vectors are deduplicated only here, not in the evaluation count or run report.
pub fn analyze_moead_report(
    report: &MoeadRunReport,
    config: &MoeadConfig,
    objective_specs: Vec<ParetoObjectiveSpec>,
    reference_point_utility: Vec<f64>,
) -> Result<MoeadParetoAnalysis, MoeadError> {
    verify_moead_report(report, config)?;
    if objective_specs.len() != report.objective_dimensions {
        return Err(MoeadError::new(
            "objective_specs",
            "objective specification count differs from the benchmark result",
        ));
    }
    if objective_specs.iter().any(|objective| objective.direction != ObjectiveDirection::Minimize) {
        return Err(MoeadError::new(
            "objective_specs.direction",
            "ZDT1 and DTLZ2 reference problems use native minimization objectives",
        ));
    }
    let mut unique_candidates = BTreeMap::<String, ParetoCandidate>::new();
    for candidate in &report.nondominated_front {
        let evidence_sha256 = digest_serializable(candidate)?;
        unique_candidates.entry(candidate.candidate_id.clone()).or_insert_with(|| {
            ParetoCandidate {
                candidate_id: candidate.candidate_id.clone(),
                objective_values: candidate.objective_values.clone(),
                feasible: true,
                constraint_violations: Vec::new(),
                evidence_sha256,
            }
        });
    }
    let candidates = unique_candidates.into_values().collect::<Vec<_>>();
    if candidates.is_empty() {
        return Err(MoeadError::new(
            "report.nondominated_front",
            "cannot compute Pareto indicators for an empty archive",
        ));
    }
    let plan = ParetoAnalysisPlan::new(
        format!("moead-{}-{}", report.problem_id, report.config_sha256),
        objective_specs,
        reference_point_utility,
        candidates,
    ).map_err(map_pareto_error)?;
    let analysis = analyze_pareto(&plan).map_err(map_pareto_error)?;
    Ok(MoeadParetoAnalysis { plan, analysis })
}

fn make_anytime_recorder<'a>(
    history: &'a mut AnytimeHistory,
    problem_id: &str,
    seed: u64,
    objective_dimensions: usize,
    expected_attempts: usize,
) -> Result<AnytimeRecorder<'a>, MoeadError> {
    let header = history.header();
    if header.algorithm_id != ALGORITHM_ID || header.problem_id != problem_id || header.seed != seed {
        return Err(MoeadError::new("anytime.header", "history algorithm, problem, or seed differs from this MOEA/D run"));
    }
    if header.objective_specs.len() != objective_dimensions
        || header.objective_specs.iter().any(|spec| spec.direction != ObjectiveDirection::Minimize)
    {
        return Err(MoeadError::new("anytime.objective_specs", "history objectives do not match the minimization problem"));
    }
    if header.max_objective_evaluation_attempts < u64::try_from(expected_attempts).unwrap_or(u64::MAX) {
        return Err(MoeadError::new("anytime.max_objective_evaluation_attempts", "history attempt budget is below the optimizer's exact budget"));
    }
    AnytimeRecorder::new(history).map_err(map_anytime_error)
}

fn map_anytime_error(error: AnytimeError) -> MoeadError {
    MoeadError::new(error.path, error.message)
}

fn map_pareto_error(error: ParetoError) -> MoeadError {
    MoeadError::new(error.path, error.message)
}

fn build_neighbourhoods(
    weights: &[Vec<f64>],
    neighbourhood_size: usize,
) -> Result<Vec<Vec<usize>>, MoeadError> {
    if weights.is_empty() || neighbourhood_size == 0 || neighbourhood_size > weights.len() {
        return Err(MoeadError::new(
            "neighbourhoods",
            "invalid weight-vector or neighbourhood count",
        ));
    }
    let mut neighbourhoods = Vec::with_capacity(weights.len());
    for (index, weight) in weights.iter().enumerate() {
        let mut distances = weights.iter().enumerate()
            .map(|(other_index, other)| {
                let distance = weight.iter().zip(other)
                    .map(|(left, right)| (left - right).powi(2))
                    .sum::<f64>()
                    .sqrt();
                (distance, other_index)
            })
            .collect::<Vec<_>>();
        distances.sort_by(|left, right| {
            left.0.total_cmp(&right.0).then_with(|| left.1.cmp(&right.1))
        });
        let neighbours = distances.into_iter()
            .take(neighbourhood_size)
            .map(|(_, other_index)| other_index)
            .collect::<Vec<_>>();
        if !neighbours.contains(&index) {
            return Err(MoeadError::new(
                "neighbourhoods",
                "weight vector is missing itself from its own neighbourhood",
            ));
        }
        neighbourhoods.push(neighbours);
    }
    Ok(neighbourhoods)
}

fn select_three_distinct_parents(
    pool: &[usize],
    excluded: usize,
    rng: &mut DeterministicRng,
) -> Result<[usize; 3], MoeadError> {
    let mut eligible = pool.iter().copied().filter(|index| *index != excluded).collect::<Vec<_>>();
    eligible.sort_unstable();
    eligible.dedup();
    if eligible.len() < 3 {
        return Err(MoeadError::new(
            "parent_selection",
            "at least three distinct parent indices excluding the target are required",
        ));
    }
    let mut selected = [0usize; 3];
    for slot in &mut selected {
        let position = rng.index(eligible.len());
        *slot = eligible.swap_remove(position);
    }
    Ok(selected)
}

fn differential_evolution_child(
    target: &[f64],
    base: &[f64],
    difference_left: &[f64],
    difference_right: &[f64],
    differential_weight: f64,
    crossover_rate: f64,
    rng: &mut DeterministicRng,
) -> Result<Vec<f64>, MoeadError> {
    if target.len() != base.len()
        || target.len() != difference_left.len()
        || target.len() != difference_right.len()
        || target.is_empty()
    {
        return Err(MoeadError::new(
            "differential_evolution",
            "parent decision vectors must have matching non-zero dimensions",
        ));
    }
    let forced_dimension = rng.index(target.len());
    let mut child = target.to_vec();
    for dimension in 0..target.len() {
        if rng.unit_f64() <= crossover_rate || dimension == forced_dimension {
            child[dimension] = (base[dimension]
                + differential_weight * (difference_left[dimension] - difference_right[dimension]))
                .clamp(0.0, 1.0);
        }
    }
    Ok(child)
}

fn tchebycheff(
    objective: &[f64],
    weight: &[f64],
    ideal: &[f64],
) -> Result<f64, MoeadError> {
    if objective.len() != weight.len() || objective.len() != ideal.len() || objective.is_empty() {
        return Err(MoeadError::new(
            "tchebycheff",
            "objective, weight and ideal vectors must have matching non-zero dimensions",
        ));
    }
    let mut maximum = f64::NEG_INFINITY;
    for index in 0..objective.len() {
        let effective_weight = if weight[index] <= 0.0 {
            ZERO_WEIGHT_FLOOR
        } else {
            weight[index]
        };
        let value = effective_weight * (objective[index] - ideal[index]).abs();
        if !value.is_finite() {
            return Err(MoeadError::new("tchebycheff", "scalarized objective overflowed"));
        }
        maximum = maximum.max(value);
    }
    Ok(maximum)
}

fn update_ideal(ideal: &mut [f64], objective: &[f64]) -> Result<(), MoeadError> {
    if ideal.len() != objective.len() {
        return Err(MoeadError::new("ideal", "ideal and objective dimensions differ"));
    }
    for index in 0..ideal.len() {
        if !objective[index].is_finite() {
            return Err(MoeadError::new("ideal", "objective value must be finite"));
        }
        ideal[index] = ideal[index].min(objective[index]);
    }
    Ok(())
}

fn evaluate_checked<F>(
    evaluator: &F,
    decision: &[f64],
    objective_dimensions: usize,
) -> Result<Vec<f64>, MoeadError>
where
    F: Fn(&[f64]) -> Result<Vec<f64>, MoeadError>,
{
    if decision.iter().any(|value| !value.is_finite() || !(0.0..=1.0).contains(value)) {
        return Err(MoeadError::new(
            "decision",
            "MOEA/D operator produced a non-finite or out-of-bounds decision variable",
        ));
    }
    let objectives = evaluator(decision)?;
    if objectives.len() != objective_dimensions
        || objectives.iter().any(|value| !value.is_finite())
    {
        return Err(MoeadError::new(
            "objectives",
            "objective evaluator returned a non-finite vector or incorrect dimension",
        ));
    }
    Ok(objectives)
}

fn nondominated_indices(population: &[Individual]) -> Result<Vec<usize>, MoeadError> {
    let mut result = Vec::new();
    for index in 0..population.len() {
        let dominated = (0..population.len()).any(|other| {
            other != index
                && dominates_minimization(
                    &population[other].objectives,
                    &population[index].objectives,
                )
        });
        if !dominated {
            result.push(index);
        }
    }
    if result.is_empty() && !population.is_empty() {
        return Err(MoeadError::new("nondominated_front", "population produced an empty front"));
    }
    Ok(result)
}

fn dominates_minimization(left: &[f64], right: &[f64]) -> bool {
    left.iter().zip(right)
        .all(|(left_value, right_value)| left_value <= right_value)
        && left.iter().zip(right)
            .any(|(left_value, right_value)| left_value < right_value)
}

fn to_candidate(individual: &Individual) -> Result<MoeadCandidate, MoeadError> {
    let candidate_id = digest_serializable(&individual.decision)?;
    Ok(MoeadCandidate {
        candidate_id,
        decision_values: individual.decision.clone(),
        objective_values: individual.objectives.clone(),
    })
}

fn checked_add_usize(left: usize, right: usize, path: &str) -> Result<usize, MoeadError> {
    left.checked_add(right)
        .ok_or_else(|| MoeadError::new(path, "count overflowed usize"))
}

fn digest_serializable<T: Serialize>(value: &T) -> Result<String, MoeadError> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        MoeadError::new("serialization", format!("cannot serialize MOEA/D receipt: {error}"))
    })?;
    let digest = Sha256::digest(bytes);
    Ok(digest.iter().map(|byte| format!("{byte:02x}")).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn zdt_config(seed: u64) -> MoeadConfig {
        MoeadConfig {
            weight_mode: MoeadWeightMode::Uniform2d { population_size: 20 },
            generations: 5,
            neighbourhood_size: 8,
            neighbourhood_mating_probability: 0.9,
            differential_weight: 0.5,
            crossover_rate: 1.0,
            max_replacements: 2,
            seed,
        }
    }

    fn dtlz_config(seed: u64) -> MoeadConfig {
        MoeadConfig {
            weight_mode: MoeadWeightMode::Simplex3d { divisions: 5 },
            generations: 3,
            neighbourhood_size: 8,
            neighbourhood_mating_probability: 0.9,
            differential_weight: 0.5,
            crossover_rate: 1.0,
            max_replacements: 2,
            seed,
        }
    }

    #[test]
    fn weight_vectors_are_normalized_and_population_counts_are_exact() {
        let two = MoeadWeightMode::Uniform2d { population_size: 20 }.weights().unwrap();
        assert_eq!(two.len(), 20);
        for weight in two {
            assert!((weight.iter().sum::<f64>() - 1.0).abs() < 1e-12);
            assert!(weight.iter().all(|value| *value >= 0.0));
        }
        let three_mode = MoeadWeightMode::Simplex3d { divisions: 5 };
        let three = three_mode.weights().unwrap();
        assert_eq!(three.len(), 21);
        for weight in three {
            assert!((weight.iter().sum::<f64>() - 1.0).abs() < 1e-12);
            assert!(weight.iter().all(|value| *value >= 0.0));
        }
    }

    #[test]
    fn moead_zdt1_is_reproducible_with_exact_evaluation_budget() {
        let config = zdt_config(42);
        let left = run_moead_zdt1(&config, 10).unwrap();
        let right = run_moead_zdt1(&config, 10).unwrap();
        assert_eq!(left.report_sha256, right.report_sha256);
        assert_eq!(left.nondominated_front, right.nondominated_front);
        assert_eq!(left.objective_evaluations, 20 * 6);
        assert_eq!(left.final_population.len(), 20);
    }

    #[test]
    fn moead_dtlz2_respects_three_objective_simplex_population_and_bounds() {
        let config = dtlz_config(7);
        let report = run_moead_dtlz2(&config, 7).unwrap();
        assert_eq!(report.objective_dimensions, 3);
        assert_eq!(report.final_population.len(), 21);
        assert_eq!(report.objective_evaluations, 21 * 4);
        assert!(report.final_population.iter().all(|candidate| {
            candidate.decision_values.iter().all(|value| value.is_finite() && (0.0..=1.0).contains(value))
                && candidate.objective_values.len() == 3
                && candidate.objective_values.iter().all(|value| value.is_finite())
        }));
    }

    #[test]
    fn report_replay_and_common_pareto_indicators_are_bound_to_moead_archive() {
        let config = zdt_config(87);
        let report = run_moead_zdt1(&config, 10).unwrap();
        verify_moead_report(&report, &config).unwrap();
        assert_eq!(report.weight_vectors.len(), report.final_population.len());

        let analysis = analyze_moead_report(
            &report,
            &config,
            vec![
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
                    scale_upper: 1.0,
                },
            ],
            vec![-0.1, -0.1],
        ).unwrap();
        assert!(!analysis.analysis.nondominated_candidate_ids.is_empty());

        let mut tampered = report.clone();
        tampered.weight_vectors[0][0] += 0.01;
        assert!(verify_moead_report(&tampered, &config).is_err());
    }

    #[test]
    fn final_population_keeps_weight_vector_subproblem_order() {
        let config = zdt_config(91);
        let report = run_moead_zdt1(&config, 10).unwrap();
        let weight_count = report.weight_vectors.len();
        assert_eq!(weight_count, report.final_population.len());
        for index in 0..weight_count {
            let candidate = &report.final_population[index];
            assert_eq!(candidate.decision_values.len(), 10);
            assert!(report.weight_vectors[index].iter().sum::<f64>() > 0.999999);
        }
    }

    #[test]
    fn tchebycheff_uses_floor_for_zero_weight_and_minimization_ideal() {
        let value = tchebycheff(&[0.5, 0.25], &[0.0, 1.0], &[0.1, 0.1]).unwrap();
        assert!((value - 0.15).abs() < 1e-12);
    }

    #[test]
    fn neighbourhoods_include_self_and_parent_selection_excludes_target() {
        let weights = MoeadWeightMode::Uniform2d { population_size: 8 }.weights().unwrap();
        let neighbourhoods = build_neighbourhoods(&weights, 4).unwrap();
        assert!(neighbourhoods.iter().enumerate().all(|(index, values)| values.contains(&index)));
        let mut rng = DeterministicRng::new(1);
        let parents = select_three_distinct_parents(&neighbourhoods[0], 0, &mut rng).unwrap();
        assert!(!parents.contains(&0));
        assert_eq!(parents.into_iter().collect::<BTreeSet<_>>().len(), 3);
    }

    #[test]
    fn moead_rejects_invalid_parameters_and_problem_weight_modes() {
        let mut bad = zdt_config(1);
        bad.neighbourhood_size = 3;
        assert!(run_moead_zdt1(&bad, 10).is_err());
        assert!(run_moead_dtlz2(&zdt_config(1), 3).is_err());
        assert!(run_moead_dtlz2(&dtlz_config(1), 2).is_err());
    }
}
