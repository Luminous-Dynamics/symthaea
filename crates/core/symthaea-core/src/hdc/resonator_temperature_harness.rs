//! Deterministic cleanup-temperature sweep for coupled HDC resonator factorization.
//!
//! This harness isolates the softmax cleanup temperature already exposed by
//! ResonatorConfig. Representation geometry, dimensions, codebook sizes, query
//! fixtures, and solver seeds are held fixed within each cell. The sweep reports
//! terminal outcomes and convergence behavior rather than collapsing them into
//! a composite score.

use super::associative_cleanup_harness::{fixture_seed, noisy_query};
use super::resonator::{Factor, MultiConstraint, ResonatorConfig, ResonatorNetwork};
use super::unified_hv::ContinuousHV;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const SCHEMA_VERSION: u32 = 1;
pub const TASK_NAME: &str = "hdc_two_factor_resonator_temperature_sweep";
pub const SCENARIO_SET: &str = "two-factor-resonator-temperature-v1";
pub const SCENARIO_REVISION: &str = "sha256:two-factor-temperature-fixture-v1";
pub const DEFAULT_SEED: u64 = 0x5445_4d50_45524154;
pub const DEFAULT_DIMENSIONS: &[usize] = &[2_048, 8_192];
pub const DEFAULT_CODEBOOK_SIZES: &[usize] = &[4, 8];
pub const DEFAULT_TEMPERATURES: &[f32] = &[0.05, 0.10, 0.20];
pub const DEFAULT_QUERY_NOISE_WEIGHTS: &[f32] = &[0.0, 0.20];
pub const DEFAULT_SOLVER_SEEDS: &[u64] = &[0x51, 0xA7, 0xD3];
pub const QUERIES_PER_SIZE: usize = 1;
pub const MAX_SOLVER_ITERATIONS: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TemperatureSpec {
    pub schema_version: u32,
    pub dimensions: &'static [usize],
    pub codebook_sizes: &'static [usize],
    pub temperatures: &'static [f32],
    pub query_noise_weights: &'static [f32],
    pub solver_seeds: &'static [u64],
    pub seed: u64,
    pub queries_per_size: usize,
    pub max_solver_iterations: usize,
}

impl Default for TemperatureSpec {
    fn default() -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            dimensions: DEFAULT_DIMENSIONS,
            codebook_sizes: DEFAULT_CODEBOOK_SIZES,
            temperatures: DEFAULT_TEMPERATURES,
            query_noise_weights: DEFAULT_QUERY_NOISE_WEIGHTS,
            solver_seeds: DEFAULT_SOLVER_SEEDS,
            seed: DEFAULT_SEED,
            queries_per_size: QUERIES_PER_SIZE,
            max_solver_iterations: MAX_SOLVER_ITERATIONS,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemperatureCell {
    pub resolution: usize,
    pub codebook_size: usize,
    pub temperature: f32,
    pub query_noise_weight: f32,
    pub exhaustive_trials: u64,
    pub exhaustive_unordered_correct: u64,
    pub exhaustive_accuracy: f64,
    pub exhaustive_mean_margin: f64,
    pub resonator_trials: u64,
    pub factor_x_correct: u64,
    pub factor_y_correct: u64,
    pub unordered_joint_correct: u64,
    pub joint_accuracy: f64,
    pub correct_convergence: u64,
    pub spurious_convergence: u64,
    pub non_convergence: u64,
    pub mean_iterations: f64,
    pub converged_mean_iterations: f64,
    pub mean_factor_margin: f64,
    pub mean_terminal_energy: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemperatureEvidence {
    pub schema_version: u32,
    pub task: String,
    pub scenario_set: String,
    pub scenario_revision: String,
    pub seed: u64,
    pub solver_seeds: Vec<u64>,
    pub temperatures: Vec<f32>,
    pub query_noise_weights: Vec<f32>,
    pub queries_per_size: usize,
    pub max_solver_iterations: usize,
    pub exhaustive_search_space_formula: String,
    pub cleanup_rule: String,
    pub spec_identity: String,
    pub cells: Vec<TemperatureCell>,
    pub artifact_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TemperatureError {
    UnsupportedSchema(u32),
    EmptyDimensions,
    EmptyCodebookSizes,
    EmptyTemperatures,
    EmptyNoiseWeights,
    EmptySolverSeeds,
    InvalidDimension(usize),
    UnsortedDimensions,
    InvalidCodebookSize(usize),
    UnsortedCodebookSizes,
    InvalidTemperature(u32),
    UnsortedTemperatures,
    InvalidNoiseWeight(u32),
    UnsortedNoiseWeights,
    ZeroQueries,
    TooManyQueries,
    ZeroIterationBudget,
}

impl std::fmt::Display for TemperatureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSchema(v) => write!(f, "unsupported temperature schema: {v}"),
            Self::EmptyDimensions => f.write_str("dimension ladder is empty"),
            Self::EmptyCodebookSizes => f.write_str("codebook-size ladder is empty"),
            Self::EmptyTemperatures => f.write_str("temperature ladder is empty"),
            Self::EmptyNoiseWeights => f.write_str("query-noise ladder is empty"),
            Self::EmptySolverSeeds => f.write_str("solver-seed ladder is empty"),
            Self::InvalidDimension(d) => write!(f, "invalid dimension: {d}"),
            Self::UnsortedDimensions => f.write_str("dimensions must be strictly ascending"),
            Self::InvalidCodebookSize(n) => write!(f, "invalid codebook size: {n}"),
            Self::UnsortedCodebookSizes => f.write_str("codebook sizes must be strictly ascending"),
            Self::InvalidTemperature(bits) => write!(f, "invalid temperature bits: {bits}"),
            Self::UnsortedTemperatures => f.write_str("temperatures must be strictly ascending"),
            Self::InvalidNoiseWeight(bits) => write!(f, "invalid query-noise weight bits: {bits}"),
            Self::UnsortedNoiseWeights => f.write_str("query-noise weights must be strictly ascending"),
            Self::ZeroQueries => f.write_str("queries_per_size must be non-zero"),
            Self::TooManyQueries => f.write_str("queries_per_size exceeds the safety limit"),
            Self::ZeroIterationBudget => f.write_str("max_solver_iterations must be non-zero"),
        }
    }
}

impl std::error::Error for TemperatureError {}

impl TemperatureSpec {
    pub fn validate(&self) -> Result<(), TemperatureError> {
        if self.schema_version != SCHEMA_VERSION {
            return Err(TemperatureError::UnsupportedSchema(self.schema_version));
        }
        if self.dimensions.is_empty() {
            return Err(TemperatureError::EmptyDimensions);
        }
        if self.codebook_sizes.is_empty() {
            return Err(TemperatureError::EmptyCodebookSizes);
        }
        if self.temperatures.is_empty() {
            return Err(TemperatureError::EmptyTemperatures);
        }
        if self.query_noise_weights.is_empty() {
            return Err(TemperatureError::EmptyNoiseWeights);
        }
        if self.solver_seeds.is_empty() {
            return Err(TemperatureError::EmptySolverSeeds);
        }
        if self.queries_per_size == 0 {
            return Err(TemperatureError::ZeroQueries);
        }
        if self.queries_per_size > 1_024 {
            return Err(TemperatureError::TooManyQueries);
        }
        if self.max_solver_iterations == 0 {
            return Err(TemperatureError::ZeroIterationBudget);
        }

        let mut previous = 0usize;
        for &dimension in self.dimensions {
            if dimension == 0 || !dimension.is_power_of_two() {
                return Err(TemperatureError::InvalidDimension(dimension));
            }
            if dimension <= previous {
                return Err(TemperatureError::UnsortedDimensions);
            }
            previous = dimension;
        }

        let mut previous = 0usize;
        for &size in self.codebook_sizes {
            if size < 2 {
                return Err(TemperatureError::InvalidCodebookSize(size));
            }
            if size <= previous {
                return Err(TemperatureError::UnsortedCodebookSizes);
            }
            previous = size;
        }

        let mut previous = f32::NEG_INFINITY;
        for &temperature in self.temperatures {
            if !temperature.is_finite() || temperature <= 0.0 {
                return Err(TemperatureError::InvalidTemperature(temperature.to_bits()));
            }
            if temperature <= previous {
                return Err(TemperatureError::UnsortedTemperatures);
            }
            previous = temperature;
        }

        let mut previous = f32::NEG_INFINITY;
        for &weight in self.query_noise_weights {
            if !weight.is_finite() || !(0.0..=1.0).contains(&weight) {
                return Err(TemperatureError::InvalidNoiseWeight(weight.to_bits()));
            }
            if weight <= previous {
                return Err(TemperatureError::UnsortedNoiseWeights);
            }
            previous = weight;
        }

        Ok(())
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"symthaea:hdc-two-factor-resonator-temperature");
        bytes.extend_from_slice(&self.schema_version.to_be_bytes());
        bytes.extend_from_slice(&self.seed.to_be_bytes());
        bytes.extend_from_slice(&(self.queries_per_size as u64).to_be_bytes());
        bytes.extend_from_slice(&(self.max_solver_iterations as u64).to_be_bytes());

        bytes.extend_from_slice(&(self.dimensions.len() as u64).to_be_bytes());
        for &dimension in self.dimensions {
            bytes.extend_from_slice(&(dimension as u64).to_be_bytes());
        }
        bytes.extend_from_slice(&(self.codebook_sizes.len() as u64).to_be_bytes());
        for &size in self.codebook_sizes {
            bytes.extend_from_slice(&(size as u64).to_be_bytes());
        }
        bytes.extend_from_slice(&(self.temperatures.len() as u64).to_be_bytes());
        for &temperature in self.temperatures {
            bytes.extend_from_slice(&temperature.to_bits().to_be_bytes());
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

fn bipolar_codebook(resolution: usize, codebook_size: usize, seed: u64) -> Vec<ContinuousHV> {
    (0..codebook_size)
        .map(|index| {
            super::associative_cleanup_harness::bipolar(
                resolution,
                fixture_seed(seed, index, 0x54454d50),
            )
        })
        .collect()
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

fn exhaustive_pair(values: &[ContinuousHV], query: &ContinuousHV) -> ((usize, usize), f32, f32) {
    let mut best_pair = (0usize, 0usize);
    let mut best = f32::NEG_INFINITY;
    let mut second = f32::NEG_INFINITY;
    for (i, left) in values.iter().enumerate() {
        for (j, right) in values.iter().enumerate() {
            let similarity = query.similarity(&left.bind(right));
            if similarity > best {
                second = best;
                best = similarity;
                best_pair = (i, j);
            } else if similarity > second {
                second = similarity;
            }
        }
    }
    (best_pair, best, second)
}

fn run_cell(
    resolution: usize,
    codebook_size: usize,
    temperature: f32,
    query_noise_weight: f32,
    seed: u64,
    solver_seeds: &[u64],
    queries_per_size: usize,
    max_solver_iterations: usize,
) -> TemperatureCell {
    let values = bipolar_codebook(resolution, codebook_size, seed);
    let mut exhaustive_unordered_correct = 0u64;
    let mut exhaustive_margin_sum = 0.0f64;
    let mut factor_x_correct = 0u64;
    let mut factor_y_correct = 0u64;
    let mut unordered_joint_correct = 0u64;
    let mut correct_convergence = 0u64;
    let mut spurious_convergence = 0u64;
    let mut non_convergence = 0u64;
    let mut iteration_sum = 0u64;
    let mut converged_iteration_sum = 0u64;
    let mut factor_margin_sum = 0.0f64;
    let mut terminal_energy_sum = 0.0f64;

    for x_target in 0..codebook_size {
        for y_target in 0..codebook_size {
            let clean_query = values[x_target].bind(&values[y_target]);

            for query_index in 0..queries_per_size {
                let query = noisy_query(
                    &clean_query,
                    query_noise_weight,
                    fixture_seed(
                        seed,
                        x_target * codebook_size + y_target,
                        query_index as u64 + 0x54454d50,
                    ),
                );

                let (pair, best, second) = exhaustive_pair(&values, &query);
                exhaustive_unordered_correct += u64::from(
                    (pair == (x_target, y_target)) || (pair == (y_target, x_target)),
                );
                exhaustive_margin_sum += f64::from(best - second);

                for &solver_seed in solver_seeds {
                    let config = ResonatorConfig {
                        temperature,
                        max_iterations: max_solver_iterations,
                        ..Default::default()
                    };
                    let mut network = ResonatorNetwork::with_config(resolution, config)
                        .expect("dimension should be admissible");
                    for (index, value) in values.iter().enumerate() {
                        network
                            .add_symbol(&format!("value-{index}"), value.values.clone())
                            .expect("value dimensions should match");
                    }

                    let constraints = [
                        MultiConstraint::new(
                            Factor::Unknown("y".to_string()),
                            Factor::Unknown("x".to_string()),
                            Factor::Known(query.values.clone()),
                        ),
                        MultiConstraint::new(
                            Factor::Unknown("x".to_string()),
                            Factor::Unknown("y".to_string()),
                            Factor::Known(query.values.clone()),
                        ),
                    ];

                    let solutions = network
                        .solve_system_seeded(
                            &["x", "y"],
                            &constraints,
                            Some(max_solver_iterations),
                            solver_seed,
                        )
                        .expect("two-factor resonator solve should succeed");

                    let x_solution = solutions.get("x").expect("x solution is present");
                    let y_solution = solutions.get("y").expect("y solution is present");
                    let (x_index, x_best, x_second) =
                        nearest(&values, &ContinuousHV::from_vec(x_solution.vector.clone()));
                    let (y_index, y_best, y_second) =
                        nearest(&values, &ContinuousHV::from_vec(y_solution.vector.clone()));

                    factor_x_correct += u64::from(x_index == x_target);
                    factor_y_correct += u64::from(y_index == y_target);
                    let joint = (x_index == x_target && y_index == y_target)
                        || (x_index == y_target && y_index == x_target);
                    unordered_joint_correct += u64::from(joint);
                    factor_margin_sum += f64::from((x_best - x_second).min(y_best - y_second));

                    let iterations = x_solution.iterations.max(y_solution.iterations) as u64;
                    iteration_sum += iterations;
                    // Use the coupled solver's joint constraint energy rather than
                    // aggregating per-factor solution energies. The latter can describe
                    // each factor independently while missing the joint constraint state.
                    terminal_energy_sum += f64::from(network.current_energy());

                    let jointly_converged = x_solution.converged && y_solution.converged;
                    if jointly_converged {
                        converged_iteration_sum += iterations;
                        if joint {
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
    }

    let exhaustive_trials = (codebook_size * codebook_size * queries_per_size) as u64;
    let resonator_trials = exhaustive_trials * solver_seeds.len() as u64;
    TemperatureCell {
        resolution,
        codebook_size,
        temperature,
        query_noise_weight,
        exhaustive_trials,
        exhaustive_unordered_correct,
        exhaustive_accuracy: exhaustive_unordered_correct as f64 / exhaustive_trials as f64,
        exhaustive_mean_margin: exhaustive_margin_sum / exhaustive_trials as f64,
        resonator_trials,
        factor_x_correct,
        factor_y_correct,
        unordered_joint_correct,
        joint_accuracy: unordered_joint_correct as f64 / resonator_trials as f64,
        correct_convergence,
        spurious_convergence,
        non_convergence,
        mean_iterations: iteration_sum as f64 / resonator_trials as f64,
        converged_mean_iterations: if correct_convergence + spurious_convergence > 0 {
            converged_iteration_sum as f64
                / (correct_convergence + spurious_convergence) as f64
        } else {
            0.0
        },
        mean_factor_margin: factor_margin_sum / resonator_trials as f64,
        mean_terminal_energy: terminal_energy_sum / resonator_trials as f64,
    }
}

pub fn run_temperature_sweep(
    spec: TemperatureSpec,
) -> Result<TemperatureEvidence, TemperatureError> {
    spec.validate()?;
    let mut cells = Vec::with_capacity(
        spec.dimensions.len()
            * spec.codebook_sizes.len()
            * spec.temperatures.len()
            * spec.query_noise_weights.len(),
    );

    for &resolution in spec.dimensions {
        for &codebook_size in spec.codebook_sizes {
            for &temperature in spec.temperatures {
                for &query_noise_weight in spec.query_noise_weights {
                    cells.push(run_cell(
                        resolution,
                        codebook_size,
                        temperature,
                        query_noise_weight,
                        spec.seed,
                        spec.solver_seeds,
                        spec.queries_per_size,
                        spec.max_solver_iterations,
                    ));
                }
            }
        }
    }

    let mut evidence = TemperatureEvidence {
        schema_version: SCHEMA_VERSION,
        task: TASK_NAME.to_owned(),
        scenario_set: SCENARIO_SET.to_owned(),
        scenario_revision: SCENARIO_REVISION.to_owned(),
        seed: spec.seed,
        solver_seeds: spec.solver_seeds.to_vec(),
        temperatures: spec.temperatures.to_vec(),
        query_noise_weights: spec.query_noise_weights.to_vec(),
        queries_per_size: spec.queries_per_size,
        max_solver_iterations: spec.max_solver_iterations,
        exhaustive_search_space_formula: "n^2".to_owned(),
        cleanup_rule: "softmax-weighted codebook cleanup".to_owned(),
        spec_identity: spec.identity_digest(),
        cells,
        artifact_digest: String::new(),
    };
    evidence.artifact_digest =
        digest(serde_json::to_vec(&evidence).expect("temperature evidence serializes"));
    Ok(evidence)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_spec_is_valid_and_identity_commits_to_temperature_ladder() {
        let spec = TemperatureSpec::default();
        spec.validate().unwrap();
        let changed = TemperatureSpec {
            temperatures: &[0.05, 0.2],
            ..spec
        };
        assert_ne!(spec.identity_digest(), changed.identity_digest());
    }

    #[test]
    fn malformed_temperature_ladder_fails_closed() {
        let spec = TemperatureSpec {
            temperatures: &[0.2, 0.1],
            ..Default::default()
        };
        assert!(matches!(
            spec.validate(),
            Err(TemperatureError::UnsortedTemperatures)
        ));
    }

    #[test]
    fn temperature_sweep_is_reproducible() {
        let spec = TemperatureSpec {
            dimensions: &[1_024],
            codebook_sizes: &[2],
            temperatures: &[0.05, 0.1],
            query_noise_weights: &[0.0],
            solver_seeds: &[7, 11],
            queries_per_size: 1,
            max_solver_iterations: 8,
            ..Default::default()
        };
        let a = run_temperature_sweep(spec).unwrap();
        let b = run_temperature_sweep(spec).unwrap();
        assert_eq!(a.cells, b.cells);
        assert_eq!(a.spec_identity, b.spec_identity);
    }

    #[test]
    fn terminal_outcomes_partition_all_resonator_trials() {
        let spec = TemperatureSpec {
            dimensions: &[1_024],
            codebook_sizes: &[2],
            temperatures: &[0.1],
            query_noise_weights: &[0.0],
            solver_seeds: &[7, 11],
            queries_per_size: 1,
            max_solver_iterations: 4,
            ..Default::default()
        };
        let evidence = run_temperature_sweep(spec).unwrap();
        let cell = &evidence.cells[0];
        assert_eq!(
            cell.correct_convergence + cell.spurious_convergence + cell.non_convergence,
            cell.resonator_trials
        );
    }
}
