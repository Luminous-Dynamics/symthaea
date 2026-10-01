//! Matched direct-cleanup vs resonator cross-validation for HDC associative retrieval.
//!
//! The fixture and query construction are inherited from
//! `associative_cleanup_harness`. Each trial therefore evaluates two retrieval
//! mechanisms against the same key/value memory, query corruption, and seed.
//!
//! Results deliberately separate decoded accuracy from terminal dynamics:
//! correct convergence, spurious convergence, non-convergence, iterations, and
//! retrieval margin are reported independently. No composite score is defined.

use super::associative_cleanup_harness::{bipolar, fixture_seed, noisy_query};
use super::resonator::{Constraint, ResonatorNetwork};
use super::unified_hv::ContinuousHV;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const SCHEMA_VERSION: u32 = 1;
pub const TASK_NAME: &str = "hdc_associative_resonator_cross_validation";
pub const SCENARIO_SET: &str = "associative-cleanup-resonator-cross-validation-v1";
pub const SCENARIO_REVISION: &str = "sha256:associative-cleanup-fixture-v1";
pub const DEFAULT_SEED: u64 = 0x4352_4F53_5356_414C;
pub const DEFAULT_DIMENSIONS: &[usize] =
    &[1_024, 2_048, 4_096, 8_192, 16_384, 32_768];
pub const DEFAULT_MEMORY_LOADS: &[usize] = &[2, 4, 8, 16];
pub const DEFAULT_QUERY_NOISE_WEIGHTS: &[f32] = &[0.0, 0.10, 0.20, 0.35];
pub const DEFAULT_SOLVER_SEEDS: &[u64] = &[0x51, 0xA7, 0xD3];
pub const QUERIES_PER_LOAD: usize = 1;
pub const MAX_SOLVER_ITERATIONS: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CrossValidationSpec {
    pub schema_version: u32,
    pub dimensions: &'static [usize],
    pub memory_loads: &'static [usize],
    pub query_noise_weights: &'static [f32],
    pub solver_seeds: &'static [u64],
    pub seed: u64,
    pub queries_per_load: usize,
    pub max_solver_iterations: usize,
}

impl Default for CrossValidationSpec {
    fn default() -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            dimensions: DEFAULT_DIMENSIONS,
            memory_loads: DEFAULT_MEMORY_LOADS,
            query_noise_weights: DEFAULT_QUERY_NOISE_WEIGHTS,
            solver_seeds: DEFAULT_SOLVER_SEEDS,
            seed: DEFAULT_SEED,
            queries_per_load: QUERIES_PER_LOAD,
            max_solver_iterations: MAX_SOLVER_ITERATIONS,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CrossValidationCell {
    pub resolution: usize,
    pub memory_load: usize,
    pub query_noise_weight: f32,
    pub trials: u64,
    pub direct_correct: u64,
    pub direct_accuracy: f64,
    pub direct_mean_margin: f64,
    pub resonator_correct: u64,
    pub resonator_accuracy: f64,
    pub correct_convergence: u64,
    pub spurious_convergence: u64,
    pub non_convergence: u64,
    pub mean_iterations: f64,
    pub converged_mean_iterations: f64,
    pub resonator_mean_margin: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CrossValidationEvidence {
    pub schema_version: u32,
    pub task: String,
    pub scenario_set: String,
    pub scenario_revision: String,
    pub seed: u64,
    pub solver_seeds: Vec<u64>,
    pub queries_per_load: usize,
    pub max_solver_iterations: usize,
    pub spec_identity: String,
    pub cells: Vec<CrossValidationCell>,
    pub artifact_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CrossValidationError {
    UnsupportedSchema(u32),
    EmptyDimensions,
    EmptyMemoryLoads,
    EmptyNoiseWeights,
    EmptySolverSeeds,
    InvalidDimension(usize),
    UnsortedDimensions,
    InvalidMemoryLoad(usize),
    UnsortedMemoryLoads,
    InvalidNoiseWeight(u32),
    UnsortedNoiseWeights,
    ZeroQueries,
    TooManyQueries,
    ZeroIterationBudget,
}

impl std::fmt::Display for CrossValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSchema(v) => write!(f, "unsupported cross-validation schema: {v}"),
            Self::EmptyDimensions => f.write_str("dimension ladder is empty"),
            Self::EmptyMemoryLoads => f.write_str("memory-load ladder is empty"),
            Self::EmptyNoiseWeights => f.write_str("query-noise ladder is empty"),
            Self::EmptySolverSeeds => f.write_str("solver-seed ladder is empty"),
            Self::InvalidDimension(d) => write!(f, "invalid dimension: {d}"),
            Self::UnsortedDimensions => f.write_str("dimensions must be strictly ascending"),
            Self::InvalidMemoryLoad(n) => write!(f, "invalid memory load: {n}"),
            Self::UnsortedMemoryLoads => f.write_str("memory loads must be strictly ascending"),
            Self::InvalidNoiseWeight(bits) => write!(f, "invalid query-noise weight bits: {bits}"),
            Self::UnsortedNoiseWeights => write!(f, "query-noise weights must be strictly ascending"),
            Self::ZeroQueries => f.write_str("queries_per_load must be non-zero"),
            Self::TooManyQueries => f.write_str("queries_per_load exceeds the safety limit"),
            Self::ZeroIterationBudget => f.write_str("max_solver_iterations must be non-zero"),
        }
    }
}

impl std::error::Error for CrossValidationError {}

impl CrossValidationSpec {
    pub fn validate(&self) -> Result<(), CrossValidationError> {
        if self.schema_version != SCHEMA_VERSION {
            return Err(CrossValidationError::UnsupportedSchema(self.schema_version));
        }
        if self.dimensions.is_empty() {
            return Err(CrossValidationError::EmptyDimensions);
        }
        if self.memory_loads.is_empty() {
            return Err(CrossValidationError::EmptyMemoryLoads);
        }
        if self.query_noise_weights.is_empty() {
            return Err(CrossValidationError::EmptyNoiseWeights);
        }
        if self.solver_seeds.is_empty() {
            return Err(CrossValidationError::EmptySolverSeeds);
        }
        if self.queries_per_load == 0 {
            return Err(CrossValidationError::ZeroQueries);
        }
        if self.queries_per_load > 1_024 {
            return Err(CrossValidationError::TooManyQueries);
        }
        if self.max_solver_iterations == 0 {
            return Err(CrossValidationError::ZeroIterationBudget);
        }

        let mut previous = 0usize;
        for &dimension in self.dimensions {
            if dimension == 0 || !dimension.is_power_of_two() {
                return Err(CrossValidationError::InvalidDimension(dimension));
            }
            if dimension <= previous {
                return Err(CrossValidationError::UnsortedDimensions);
            }
            previous = dimension;
        }

        let mut previous = 0usize;
        for &load in self.memory_loads {
            if load < 2 {
                return Err(CrossValidationError::InvalidMemoryLoad(load));
            }
            if load <= previous {
                return Err(CrossValidationError::UnsortedMemoryLoads);
            }
            previous = load;
        }

        let mut previous = f32::NEG_INFINITY;
        for &weight in self.query_noise_weights {
            if !weight.is_finite() || !(0.0..=1.0).contains(&weight) {
                return Err(CrossValidationError::InvalidNoiseWeight(weight.to_bits()));
            }
            if weight <= previous {
                return Err(CrossValidationError::UnsortedNoiseWeights);
            }
            previous = weight;
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"symthaea:hdc-associative-resonator-cross-validation");
        bytes.extend_from_slice(&self.schema_version.to_be_bytes());
        bytes.extend_from_slice(&self.seed.to_be_bytes());
        bytes.extend_from_slice(&(self.queries_per_load as u64).to_be_bytes());
        bytes.extend_from_slice(&(self.max_solver_iterations as u64).to_be_bytes());
        bytes.extend_from_slice(&(self.dimensions.len() as u64).to_be_bytes());
        for &dimension in self.dimensions {
            bytes.extend_from_slice(&(dimension as u64).to_be_bytes());
        }
        bytes.extend_from_slice(&(self.memory_loads.len() as u64).to_be_bytes());
        for &load in self.memory_loads {
            bytes.extend_from_slice(&(load as u64).to_be_bytes());
        }
        bytes.extend_from_slice(&(self.query_noise_weights.len() as u64).to_be_bytes());
        for &weight in self.query_noise_weights {
            bytes.extend_from_slice(&weight.to_bits().to_be_bytes());
        }
        bytes.extend_from_slice(&(self.solver_seeds.len() as u64).to_be_bytes());
        for &seed in self.solver_seeds {
            bytes.extend_from_slice(&seed.to_be_bytes());
        }
        bytes
    }

    pub fn identity_digest(&self) -> String {
        digest(self.canonical_bytes())
    }
}

fn digest(bytes: impl AsRef<[u8]>) -> String {
    let mut out = String::from("sha256:");
    for byte in Sha256::digest(bytes) {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

fn nearest(values: &[ContinuousHV], query: &ContinuousHV) -> (usize, f32, f32) {
    let mut best_index = 0usize;
    let mut best = f32::NEG_INFINITY;
    let mut second = f32::NEG_INFINITY;
    for (index, value) in values.iter().enumerate() {
        let similarity = query.similarity(value);
        if similarity > best {
            second = best;
            best = similarity;
            best_index = index;
        } else if similarity > second {
            second = similarity;
        }
    }
    (best_index, best, second)
}

fn run_cell(
    resolution: usize,
    memory_load: usize,
    query_noise_weight: f32,
    seed: u64,
    solver_seeds: &[u64],
    queries_per_load: usize,
    max_solver_iterations: usize,
) -> CrossValidationCell {
    let keys: Vec<ContinuousHV> = (0..memory_load)
        .map(|i| bipolar(resolution, fixture_seed(seed, i, 0x4B4559)))
        .collect();
    let values: Vec<ContinuousHV> = (0..memory_load)
        .map(|i| bipolar(resolution, fixture_seed(seed, i, 0x56414C)))
        .collect();

    let bound_pairs: Vec<ContinuousHV> = keys
        .iter()
        .zip(values.iter())
        .map(|(key, value)| key.bind(value))
        .collect();
    let pair_refs: Vec<&ContinuousHV> = bound_pairs.iter().collect();
    let memory = ContinuousHV::bundle(&pair_refs);

    let mut direct_correct = 0u64;
    let mut direct_margin_sum = 0.0f64;
    let mut resonator_correct = 0u64;
    let mut resonator_margin_sum = 0.0f64;
    let mut correct_convergence = 0u64;
    let mut spurious_convergence = 0u64;
    let mut non_convergence = 0u64;
    let mut iteration_sum = 0u64;
    let mut converged_iteration_sum = 0u64;

    for target in 0..memory_load {
        for query_index in 0..queries_per_load {
            let query = noisy_query(
                &keys[target],
                query_noise_weight,
                fixture_seed(seed, target, query_index as u64 + 0x4E4F4953),
            );
            let released = query.bind(&memory);
            let (direct_index, direct_best, direct_second) = nearest(&values, &released);
            direct_correct += u64::from(direct_index == target);
            direct_margin_sum += f64::from(direct_best - direct_second);

            for &solver_seed in solver_seeds {
                let mut network = ResonatorNetwork::new(resolution)
                    .expect("dimension should be admissible");
                for (index, value) in values.iter().enumerate() {
                    network
                        .add_symbol(&format!("value-{index}"), value.values.clone())
                        .expect("value dimensions should match");
                }

                let constraint = Constraint::new(keys[target].values.clone(), memory.values.clone());
                let solutions = network
                    .solve_seeded(&[constraint], Some(max_solver_iterations), solver_seed)
                    .expect("single-constraint resonator solve should succeed");
                let solution = solutions.get("x").expect("single unknown is named x");
                let (_, resonator_best, resonator_second) = nearest(&values, &ContinuousHV::from_vec(solution.vector.clone()));
                let target_name = format!("value-{target}");
                let correct = solution.closest_symbol.as_deref() == Some(target_name.as_str());
                resonator_correct += u64::from(correct);
                resonator_margin_sum += f64::from(resonator_best - resonator_second);
                iteration_sum += solution.iterations as u64;
                if solution.converged {
                    converged_iteration_sum += solution.iterations as u64;
                    if correct {
                        correct_convergence += 1;
                    } else {
                        spurious_convergence += 1;
                    }
                } else {
                    non_convergence += 1;
                }
            }
        }
    }

    let direct_trials = (memory_load * queries_per_load) as u64;
    let trials = direct_trials * solver_seeds.len() as u64;
    CrossValidationCell {
        resolution,
        memory_load,
        query_noise_weight,
        trials,
        direct_correct,
        direct_accuracy: direct_correct as f64 / direct_trials as f64,
        direct_mean_margin: direct_margin_sum / direct_trials as f64,
        resonator_correct,
        resonator_accuracy: resonator_correct as f64 / trials as f64,
        correct_convergence,
        spurious_convergence,
        non_convergence,
        mean_iterations: iteration_sum as f64 / trials as f64,
        converged_mean_iterations: if correct_convergence + spurious_convergence > 0 {
            converged_iteration_sum as f64
                / (correct_convergence + spurious_convergence) as f64
        } else {
            0.0
        },
        resonator_mean_margin: resonator_margin_sum / trials as f64,
    }
}

fn solutions_for_single(
    solutions: &std::collections::HashMap<String, super::resonator::ResonatorSolution>,
    _target: usize,
) -> super::resonator::ResonatorSolution {
    solutions
        .get("x")
        .expect("single unknown is named x")
        .clone()
}

pub fn run_cross_validation(
    spec: CrossValidationSpec,
) -> Result<CrossValidationEvidence, CrossValidationError> {
    spec.validate()?;
    let mut cells = Vec::with_capacity(
        spec.dimensions.len()
            * spec.memory_loads.len()
            * spec.query_noise_weights.len(),
    );

    for &resolution in spec.dimensions {
        for &memory_load in spec.memory_loads {
            for &query_noise_weight in spec.query_noise_weights {
                cells.push(run_cell(
                    resolution,
                    memory_load,
                    query_noise_weight,
                    spec.seed,
                    spec.solver_seeds,
                    spec.queries_per_load,
                    spec.max_solver_iterations,
                ));
            }
        }
    }

    let mut evidence = CrossValidationEvidence {
        schema_version: SCHEMA_VERSION,
        task: TASK_NAME.to_owned(),
        scenario_set: SCENARIO_SET.to_owned(),
        scenario_revision: SCENARIO_REVISION.to_owned(),
        seed: spec.seed,
        solver_seeds: spec.solver_seeds.to_vec(),
        queries_per_load: spec.queries_per_load,
        max_solver_iterations: spec.max_solver_iterations,
        spec_identity: spec.identity_digest(),
        cells,
        artifact_digest: String::new(),
    };
    evidence.artifact_digest =
        digest(serde_json::to_vec(&evidence).expect("cross-validation evidence serializes"));
    Ok(evidence)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_spec_is_valid_and_identity_commits_to_solver_seeds() {
        let spec = CrossValidationSpec::default();
        spec.validate().unwrap();
        let changed = CrossValidationSpec {
            solver_seeds: &[1, 2],
            ..spec
        };
        assert_ne!(spec.identity_digest(), changed.identity_digest());
    }

    #[test]
    fn malformed_noise_ladder_fails_closed() {
        let spec = CrossValidationSpec {
            query_noise_weights: &[0.20, 0.10],
            ..Default::default()
        };
        assert!(matches!(
            spec.validate(),
            Err(CrossValidationError::UnsortedNoiseWeights)
        ));
    }

    #[test]
    fn paired_cross_validation_is_reproducible() {
        let spec = CrossValidationSpec {
            dimensions: &[1_024],
            memory_loads: &[2],
            query_noise_weights: &[0.0, 0.20],
            solver_seeds: &[7, 11],
            queries_per_load: 1,
            max_solver_iterations: 8,
            ..Default::default()
        };
        let a = run_cross_validation(spec).unwrap();
        let b = run_cross_validation(spec).unwrap();
        assert_eq!(a.cells, b.cells);
        assert_eq!(a.spec_identity, b.spec_identity);
    }

    #[test]
    fn terminal_outcomes_partition_all_resonator_trials() {
        let spec = CrossValidationSpec {
            dimensions: &[1_024],
            memory_loads: &[2],
            query_noise_weights: &[0.0],
            solver_seeds: &[7, 11],
            queries_per_load: 1,
            max_solver_iterations: 4,
            ..Default::default()
        };
        let evidence = run_cross_validation(spec).unwrap();
        let cell = &evidence.cells[0];
        assert_eq!(
            cell.correct_convergence + cell.spurious_convergence + cell.non_convergence,
            cell.trials
        );
    }
}
