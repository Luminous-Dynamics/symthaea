//! Deterministic two-factor HDC resonator factorization cross-validation.
//!
//! This is the first genuinely coupled factorization layer after the single-unknown
//! associative cross-validation. The query is a noisy bound product X ⊛ Y, and the
//! resonator must recover both factors jointly. An exhaustive n² pair search is kept
//! as a transparent reference; no composite score or dimension ranking is defined.

use super::associative_cleanup_harness::{bipolar, fixture_seed, noisy_query};
use super::resonator::{Factor, MultiConstraint, ResonatorNetwork};
use super::unified_hv::ContinuousHV;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const SCHEMA_VERSION: u32 = 1;
pub const TASK_NAME: &str = "hdc_two_factor_resonator_factorization";
pub const SCENARIO_SET: &str = "two-factor-resonator-v1";
pub const SCENARIO_REVISION: &str = "sha256:two-factor-bipolar-fixture-v1";
pub const DEFAULT_SEED: u64 = 0x3246_4143_544F_5253;
pub const DEFAULT_DIMENSIONS: &[usize] = &[1_024, 2_048, 4_096, 8_192];
pub const DEFAULT_CODEBOOK_SIZES: &[usize] = &[4, 8];
pub const DEFAULT_QUERY_NOISE_WEIGHTS: &[f32] = &[0.0, 0.20, 0.35];
pub const DEFAULT_SOLVER_SEEDS: &[u64] = &[0x51, 0xA7, 0xD3];
pub const QUERIES_PER_SIZE: usize = 1;
pub const MAX_SOLVER_ITERATIONS: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TwoFactorSpec {
    pub schema_version: u32,
    pub dimensions: &'static [usize],
    pub codebook_sizes: &'static [usize],
    pub query_noise_weights: &'static [f32],
    pub solver_seeds: &'static [u64],
    pub seed: u64,
    pub queries_per_size: usize,
    pub max_solver_iterations: usize,
}

impl Default for TwoFactorSpec {
    fn default() -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            dimensions: DEFAULT_DIMENSIONS,
            codebook_sizes: DEFAULT_CODEBOOK_SIZES,
            query_noise_weights: DEFAULT_QUERY_NOISE_WEIGHTS,
            solver_seeds: DEFAULT_SOLVER_SEEDS,
            seed: DEFAULT_SEED,
            queries_per_size: QUERIES_PER_SIZE,
            max_solver_iterations: MAX_SOLVER_ITERATIONS,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TwoFactorCell {
    pub resolution: usize,
    pub codebook_size: usize,
    pub query_noise_weight: f32,
    pub exhaustive_trials: u64,
    pub exhaustive_correct: u64,
    pub exhaustive_accuracy: f64,
    pub exhaustive_mean_margin: f64,
    pub resonator_trials: u64,
    pub factor_x_correct: u64,
    pub factor_y_correct: u64,
    pub joint_correct: u64,
    pub joint_accuracy: f64,
    pub correct_convergence: u64,
    pub spurious_convergence: u64,
    pub non_convergence: u64,
    pub mean_iterations: f64,
    pub converged_mean_iterations: f64,
    pub mean_factor_margin: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TwoFactorEvidence {
    pub schema_version: u32,
    pub task: String,
    pub scenario_set: String,
    pub scenario_revision: String,
    pub seed: u64,
    pub solver_seeds: Vec<u64>,
    pub queries_per_size: usize,
    pub max_solver_iterations: usize,
    pub exhaustive_search_space_formula: String,
    pub spec_identity: String,
    pub cells: Vec<TwoFactorCell>,
    pub artifact_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TwoFactorError {
    UnsupportedSchema(u32),
    EmptyDimensions,
    EmptyCodebookSizes,
    EmptyNoiseWeights,
    EmptySolverSeeds,
    InvalidDimension(usize),
    UnsortedDimensions,
    InvalidCodebookSize(usize),
    UnsortedCodebookSizes,
    InvalidNoiseWeight(u32),
    UnsortedNoiseWeights,
    ZeroQueries,
    TooManyQueries,
    ZeroIterationBudget,
}

impl std::fmt::Display for TwoFactorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSchema(v) => write!(f, "unsupported two-factor schema: {v}"),
            Self::EmptyDimensions => f.write_str("dimension ladder is empty"),
            Self::EmptyCodebookSizes => f.write_str("codebook-size ladder is empty"),
            Self::EmptyNoiseWeights => f.write_str("query-noise ladder is empty"),
            Self::EmptySolverSeeds => f.write_str("solver-seed ladder is empty"),
            Self::InvalidDimension(d) => write!(f, "invalid dimension: {d}"),
            Self::UnsortedDimensions => f.write_str("dimensions must be strictly ascending"),
            Self::InvalidCodebookSize(n) => write!(f, "invalid codebook size: {n}"),
            Self::UnsortedCodebookSizes => f.write_str("codebook sizes must be strictly ascending"),
            Self::InvalidNoiseWeight(bits) => write!(f, "invalid query-noise weight bits: {bits}"),
            Self::UnsortedNoiseWeights => f.write_str("query-noise weights must be strictly ascending"),
            Self::ZeroQueries => f.write_str("queries_per_size must be non-zero"),
            Self::TooManyQueries => f.write_str("queries_per_size exceeds the safety limit"),
            Self::ZeroIterationBudget => f.write_str("max_solver_iterations must be non-zero"),
        }
    }
}

impl std::error::Error for TwoFactorError {}

impl TwoFactorSpec {
    pub fn validate(&self) -> Result<(), TwoFactorError> {
        if self.schema_version != SCHEMA_VERSION {
            return Err(TwoFactorError::UnsupportedSchema(self.schema_version));
        }
        if self.dimensions.is_empty() {
            return Err(TwoFactorError::EmptyDimensions);
        }
        if self.codebook_sizes.is_empty() {
            return Err(TwoFactorError::EmptyCodebookSizes);
        }
        if self.query_noise_weights.is_empty() {
            return Err(TwoFactorError::EmptyNoiseWeights);
        }
        if self.solver_seeds.is_empty() {
            return Err(TwoFactorError::EmptySolverSeeds);
        }
        if self.queries_per_size == 0 {
            return Err(TwoFactorError::ZeroQueries);
        }
        if self.queries_per_size > 1_024 {
            return Err(TwoFactorError::TooManyQueries);
        }
        if self.max_solver_iterations == 0 {
            return Err(TwoFactorError::ZeroIterationBudget);
        }

        let mut previous = 0usize;
        for &dimension in self.dimensions {
            if dimension == 0 || !dimension.is_power_of_two() {
                return Err(TwoFactorError::InvalidDimension(dimension));
            }
            if dimension <= previous {
                return Err(TwoFactorError::UnsortedDimensions);
            }
            previous = dimension;
        }

        let mut previous = 0usize;
        for &size in self.codebook_sizes {
            if size < 2 {
                return Err(TwoFactorError::InvalidCodebookSize(size));
            }
            if size <= previous {
                return Err(TwoFactorError::UnsortedCodebookSizes);
            }
            previous = size;
        }

        let mut previous = f32::NEG_INFINITY;
        for &weight in self.query_noise_weights {
            if !weight.is_finite() || !(0.0..=1.0).contains(&weight) {
                return Err(TwoFactorError::InvalidNoiseWeight(weight.to_bits()));
            }
            if weight <= previous {
                return Err(TwoFactorError::UnsortedNoiseWeights);
            }
            previous = weight;
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"symthaea:hdc-two-factor-resonator");
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

fn exhaustive_pair(
    values: &[ContinuousHV],
    query: &ContinuousHV,
) -> ((usize, usize), f32, f32) {
    let mut best_pair = (0usize, 0usize);
    let mut best = f32::NEG_INFINITY;
    let mut second = f32::NEG_INFINITY;

    for (i, left) in values.iter().enumerate() {
        for (j, right) in values.iter().enumerate() {
            let candidate = left.bind(right);
            let similarity = query.similarity(&candidate);
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
    query_noise_weight: f32,
    seed: u64,
    solver_seeds: &[u64],
    queries_per_size: usize,
    max_solver_iterations: usize,
) -> TwoFactorCell {
    let values: Vec<ContinuousHV> = (0..codebook_size)
        .map(|i| bipolar(resolution, fixture_seed(seed, i, 0x32464143)))
        .collect();

    let mut exhaustive_correct = 0u64;
    let mut exhaustive_margin_sum = 0.0f64;
    let mut factor_x_correct = 0u64;
    let mut factor_y_correct = 0u64;
    let mut joint_correct = 0u64;
    let mut correct_convergence = 0u64;
    let mut spurious_convergence = 0u64;
    let mut non_convergence = 0u64;
    let mut iteration_sum = 0u64;
    let mut converged_iteration_sum = 0u64;
    let mut factor_margin_sum = 0.0f64;

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
                        query_index as u64 + 0x32514E,
                    ),
                );

                let (pair, best, second) = exhaustive_pair(&values, &query);
                exhaustive_correct += u64::from(pair == (x_target, y_target));
                exhaustive_margin_sum += f64::from(best - second);

                for &solver_seed in solver_seeds {
                    let mut network =
                        ResonatorNetwork::new(resolution).expect("dimension should be admissible");
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
                    let joint = x_index == x_target && y_index == y_target;
                    joint_correct += u64::from(joint);
                    factor_margin_sum +=
                        f64::from((x_best - x_second).min(y_best - y_second));

                    iteration_sum += x_solution.iterations.max(y_solution.iterations) as u64;
                    let jointly_converged = x_solution.converged && y_solution.converged;
                    if jointly_converged {
                        converged_iteration_sum +=
                            x_solution.iterations.max(y_solution.iterations) as u64;
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

    TwoFactorCell {
        resolution,
        codebook_size,
        query_noise_weight,
        exhaustive_trials,
        exhaustive_correct,
        exhaustive_accuracy: exhaustive_correct as f64 / exhaustive_trials as f64,
        exhaustive_mean_margin: exhaustive_margin_sum / exhaustive_trials as f64,
        resonator_trials,
        factor_x_correct,
        factor_y_correct,
        joint_correct,
        joint_accuracy: joint_correct as f64 / resonator_trials as f64,
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
    }
}

pub fn run_two_factor(spec: TwoFactorSpec) -> Result<TwoFactorEvidence, TwoFactorError> {
    spec.validate()?;
    let mut cells = Vec::with_capacity(
        spec.dimensions.len() * spec.codebook_sizes.len() * spec.query_noise_weights.len(),
    );

    for &resolution in spec.dimensions {
        for &codebook_size in spec.codebook_sizes {
            for &query_noise_weight in spec.query_noise_weights {
                cells.push(run_cell(
                    resolution,
                    codebook_size,
                    query_noise_weight,
                    spec.seed,
                    spec.solver_seeds,
                    spec.queries_per_size,
                    spec.max_solver_iterations,
                ));
            }
        }
    }

    let mut evidence = TwoFactorEvidence {
        schema_version: SCHEMA_VERSION,
        task: TASK_NAME.to_owned(),
        scenario_set: SCENARIO_SET.to_owned(),
        scenario_revision: SCENARIO_REVISION.to_owned(),
        seed: spec.seed,
        solver_seeds: spec.solver_seeds.to_vec(),
        queries_per_size: spec.queries_per_size,
        max_solver_iterations: spec.max_solver_iterations,
        exhaustive_search_space_formula: "n^2".to_owned(),
        spec_identity: spec.identity_digest(),
        cells,
        artifact_digest: String::new(),
    };
    evidence.artifact_digest =
        digest(serde_json::to_vec(&evidence).expect("two-factor evidence serializes"));
    Ok(evidence)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_spec_is_valid_and_identity_commits_to_solver_seeds() {
        let spec = TwoFactorSpec::default();
        spec.validate().unwrap();
        let changed = TwoFactorSpec {
            solver_seeds: &[1, 2],
            ..spec
        };
        assert_ne!(spec.identity_digest(), changed.identity_digest());
    }

    #[test]
    fn malformed_noise_ladder_fails_closed() {
        let spec = TwoFactorSpec {
            query_noise_weights: &[0.20, 0.10],
            ..Default::default()
        };
        assert!(matches!(
            spec.validate(),
            Err(TwoFactorError::UnsortedNoiseWeights)
        ));
    }

    #[test]
    fn two_factor_cross_validation_is_reproducible() {
        let spec = TwoFactorSpec {
            dimensions: &[1_024],
            codebook_sizes: &[2],
            query_noise_weights: &[0.0, 0.20],
            solver_seeds: &[7, 11],
            queries_per_size: 1,
            max_solver_iterations: 8,
            ..Default::default()
        };
        let a = run_two_factor(spec).unwrap();
        let b = run_two_factor(spec).unwrap();
        assert_eq!(a.cells, b.cells);
        assert_eq!(a.spec_identity, b.spec_identity);
    }

    #[test]
    fn terminal_outcomes_partition_all_resonator_trials() {
        let spec = TwoFactorSpec {
            dimensions: &[1_024],
            codebook_sizes: &[2],
            query_noise_weights: &[0.0],
            solver_seeds: &[7, 11],
            queries_per_size: 1,
            max_solver_iterations: 4,
            ..Default::default()
        };
        let evidence = run_two_factor(spec).unwrap();
        let cell = &evidence.cells[0];
        assert_eq!(
            cell.correct_convergence + cell.spurious_convergence + cell.non_convergence,
            cell.resonator_trials
        );
    }
}
