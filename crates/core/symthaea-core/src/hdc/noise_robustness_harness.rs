//! Deterministic noise-robustness task family for HDC dimension experiments.
//!
//! Unlike the baseline prototype-retrieval fixture, this harness sweeps
//! perturbation severity at every dimension. The task identity therefore binds
//! dimension, noise regime, seed, and sample count rather than collapsing them
//! into one scalar frontier.

use super::unified_hv::ContinuousHV;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const SCHEMA_VERSION: u32 = 1;
pub const TASK_NAME: &str = "hdc_noise_robustness_prototype_retrieval";
pub const SCENARIO_SET: &str = "noise-robustness-prototype-retrieval-v1";
pub const SCENARIO_REVISION: &str = "sha256:noise-robustness-fixture-v1";
pub const DEFAULT_SEED: u64 = 0x4E4F_4953_4552_4F42;
pub const DEFAULT_DIMENSIONS: &[usize] =
    &[1_024, 2_048, 4_096, 8_192, 16_384, 32_768, 65_536, 131_072, 262_144];
pub const DEFAULT_NOISE_WEIGHTS: &[f32] = &[0.10, 0.20, 0.35, 0.50, 0.65, 0.80];
pub const CLASS_COUNT: usize = 4;
pub const QUERIES_PER_CLASS: usize = 32;
pub const PROTOTYPE_WEIGHT: f32 = 1.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NoiseRobustnessSpec {
    pub schema_version: u32,
    pub dimensions: &'static [usize],
    pub noise_weights: &'static [f32],
    pub seed: u64,
    pub queries_per_class: usize,
}

impl Default for NoiseRobustnessSpec {
    fn default() -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            dimensions: DEFAULT_DIMENSIONS,
            noise_weights: DEFAULT_NOISE_WEIGHTS,
            seed: DEFAULT_SEED,
            queries_per_class: QUERIES_PER_CLASS,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NoiseRobustnessCell {
    pub resolution: usize,
    pub noise_weight: f32,
    pub correct: u64,
    pub total: u64,
    pub accuracy: f64,
    pub mean_margin: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NoiseRobustnessEvidence {
    pub schema_version: u32,
    pub task: String,
    pub scenario_set: String,
    pub scenario_revision: String,
    pub seed: u64,
    pub queries_per_class: usize,
    pub spec_identity: String,
    pub cells: Vec<NoiseRobustnessCell>,
    pub artifact_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NoiseRobustnessError {
    UnsupportedSchema(u32),
    EmptyDimensions,
    EmptyNoiseWeights,
    InvalidDimension(usize),
    UnsortedDimensions,
    InvalidNoiseWeight(u32),
    UnsortedNoiseWeights,
    ZeroQueries,
    TooManyQueries,
}

impl std::fmt::Display for NoiseRobustnessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSchema(v) => write!(f, "unsupported noise robustness schema: {v}"),
            Self::EmptyDimensions => f.write_str("dimension ladder is empty"),
            Self::EmptyNoiseWeights => f.write_str("noise ladder is empty"),
            Self::InvalidDimension(d) => write!(f, "invalid dimension: {d}"),
            Self::UnsortedDimensions => f.write_str("dimensions must be strictly ascending"),
            Self::InvalidNoiseWeight(bits) => write!(f, "invalid noise weight bits: {bits}"),
            Self::UnsortedNoiseWeights => f.write_str("noise weights must be strictly ascending"),
            Self::ZeroQueries => f.write_str("queries_per_class must be non-zero"),
            Self::TooManyQueries => f.write_str("queries_per_class exceeds the safety limit"),
        }
    }
}

impl std::error::Error for NoiseRobustnessError {}

impl NoiseRobustnessSpec {
    pub fn validate(&self) -> Result<(), NoiseRobustnessError> {
        if self.schema_version != SCHEMA_VERSION {
            return Err(NoiseRobustnessError::UnsupportedSchema(self.schema_version));
        }
        if self.dimensions.is_empty() {
            return Err(NoiseRobustnessError::EmptyDimensions);
        }
        if self.noise_weights.is_empty() {
            return Err(NoiseRobustnessError::EmptyNoiseWeights);
        }
        if self.queries_per_class == 0 {
            return Err(NoiseRobustnessError::ZeroQueries);
        }
        if self.queries_per_class > 4_096 {
            return Err(NoiseRobustnessError::TooManyQueries);
        }

        let mut previous = 0usize;
        for &dimension in self.dimensions {
            if dimension == 0 || !dimension.is_power_of_two() {
                return Err(NoiseRobustnessError::InvalidDimension(dimension));
            }
            if dimension <= previous {
                return Err(NoiseRobustnessError::UnsortedDimensions);
            }
            previous = dimension;
        }

        let mut previous = f32::NEG_INFINITY;
        for &weight in self.noise_weights {
            if !weight.is_finite() || !(0.0..=1.0).contains(&weight) {
                return Err(NoiseRobustnessError::InvalidNoiseWeight(weight.to_bits()));
            }
            if weight <= previous {
                return Err(NoiseRobustnessError::UnsortedNoiseWeights);
            }
            previous = weight;
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"symthaea:hdc-noise-robustness");
        bytes.extend_from_slice(&self.schema_version.to_be_bytes());
        bytes.extend_from_slice(&self.seed.to_be_bytes());
        bytes.extend_from_slice(&(self.queries_per_class as u64).to_be_bytes());
        bytes.extend_from_slice(&(self.dimensions.len() as u64).to_be_bytes());
        for &dimension in self.dimensions {
            bytes.extend_from_slice(&(dimension as u64).to_be_bytes());
        }
        bytes.extend_from_slice(&(self.noise_weights.len() as u64).to_be_bytes());
        for &weight in self.noise_weights {
            bytes.extend_from_slice(&weight.to_bits().to_be_bytes());
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

fn fixture_seed(seed: u64, class: usize, query: usize) -> u64 {
    let mut x = seed
        ^ (class as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (query as u64).wrapping_mul(0xD1B5_4A32_D192_ED03);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn query(prototype: &ContinuousHV, noise_weight: f32, seed: u64) -> ContinuousHV {
    let noise = ContinuousHV::random(prototype.dim(), seed);
    let prototype_weight = PROTOTYPE_WEIGHT - noise_weight;
    let values = prototype
        .values
        .iter()
        .zip(noise.values.iter())
        .map(|(&p, &n)| prototype_weight * p + noise_weight * n)
        .collect();
    ContinuousHV::from_vec(values)
}

fn run_cell(
    resolution: usize,
    noise_weight: f32,
    seed: u64,
    queries_per_class: usize,
) -> NoiseRobustnessCell {
    let prototypes: Vec<ContinuousHV> = (0..CLASS_COUNT)
        .map(|class| ContinuousHV::random(resolution, seed.wrapping_add(class as u64 + 1)))
        .collect();

    let mut correct = 0u64;
    let mut margin_sum = 0.0f64;
    for class in 0..CLASS_COUNT {
        for query_index in 0..queries_per_class {
            let q = query(
                &prototypes[class],
                noise_weight,
                fixture_seed(seed, class, query_index),
            );
            let mut best_class = 0usize;
            let mut best = f32::NEG_INFINITY;
            let mut second = f32::NEG_INFINITY;
            for (candidate, prototype) in prototypes.iter().enumerate() {
                let similarity = q.similarity(prototype);
                if similarity > best {
                    second = best;
                    best = similarity;
                    best_class = candidate;
                } else if similarity > second {
                    second = similarity;
                }
            }
            if best_class == class {
                correct += 1;
            }
            margin_sum += f64::from(best - second);
        }
    }

    let total = (CLASS_COUNT * queries_per_class) as u64;
    NoiseRobustnessCell {
        resolution,
        noise_weight,
        correct,
        total,
        accuracy: correct as f64 / total as f64,
        mean_margin: margin_sum / total as f64,
    }
}

pub fn run_noise_robustness(
    spec: NoiseRobustnessSpec,
) -> Result<NoiseRobustnessEvidence, NoiseRobustnessError> {
    spec.validate()?;
    let mut cells = Vec::with_capacity(spec.dimensions.len() * spec.noise_weights.len());
    for &resolution in spec.dimensions {
        for &noise_weight in spec.noise_weights {
            cells.push(run_cell(
                resolution,
                noise_weight,
                spec.seed,
                spec.queries_per_class,
            ));
        }
    }

    let mut evidence = NoiseRobustnessEvidence {
        schema_version: SCHEMA_VERSION,
        task: TASK_NAME.to_owned(),
        scenario_set: SCENARIO_SET.to_owned(),
        scenario_revision: SCENARIO_REVISION.to_owned(),
        seed: spec.seed,
        queries_per_class: spec.queries_per_class,
        spec_identity: spec.identity_digest(),
        cells,
        artifact_digest: String::new(),
    };
    evidence.artifact_digest = digest(
        serde_json::to_vec(&evidence).expect("noise robustness evidence serializes"),
    );
    Ok(evidence)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_spec_is_valid_and_identity_commits_to_experiment_shape() {
        let spec = NoiseRobustnessSpec::default();
        spec.validate().unwrap();
        let changed = NoiseRobustnessSpec {
            noise_weights: &[0.10, 0.20],
            ..spec
        };
        assert_ne!(spec.identity_digest(), changed.identity_digest());
    }

    #[test]
    fn malformed_noise_ladder_fails_closed() {
        let spec = NoiseRobustnessSpec {
            noise_weights: &[0.20, 0.10],
            ..Default::default()
        };
        assert!(matches!(
            spec.validate(),
            Err(NoiseRobustnessError::UnsortedNoiseWeights)
        ));
    }

    #[test]
    fn paired_cells_are_reproducible() {
        let spec = NoiseRobustnessSpec {
            dimensions: &[1_024, 2_048],
            noise_weights: &[0.20, 0.50],
            queries_per_class: 2,
            ..Default::default()
        };
        let a = run_noise_robustness(spec).unwrap();
        let b = run_noise_robustness(spec).unwrap();
        assert_eq!(a.cells, b.cells);
        assert_eq!(a.spec_identity, b.spec_identity);
    }

    #[test]
    fn cell_count_is_dimension_times_noise_regimes() {
        let spec = NoiseRobustnessSpec {
            dimensions: &[1_024, 2_048, 4_096],
            noise_weights: &[0.10, 0.20],
            queries_per_class: 1,
            ..Default::default()
        };
        let evidence = run_noise_robustness(spec).unwrap();
        assert_eq!(evidence.cells.len(), 6);
        assert!(evidence.cells.iter().all(|c| c.total == 4));
    }
}
