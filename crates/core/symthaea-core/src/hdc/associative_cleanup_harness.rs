//! Deterministic associative-memory cleanup task family for HDC dimension research.
//!
//! This family is structurally distinct from direct prototype classification:
//! key/value pairs are bound, superposed into one memory, then a query key is
//! used to release a noisy value estimate and clean it against the value
//! codebook. The experiment varies memory load and query corruption at each
//! dimension.

use super::unified_hv::ContinuousHV;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const SCHEMA_VERSION: u32 = 1;
pub const TASK_NAME: &str = "hdc_associative_cleanup_key_value";
pub const SCENARIO_SET: &str = "associative-cleanup-key-value-v1";
pub const SCENARIO_REVISION: &str = "sha256:associative-cleanup-fixture-v1";
pub const DEFAULT_SEED: u64 = 0x4153_534F_4349_4154;
pub const DEFAULT_DIMENSIONS: &[usize] =
    &[1_024, 2_048, 4_096, 8_192, 16_384, 32_768, 65_536, 131_072, 262_144];
pub const DEFAULT_MEMORY_LOADS: &[usize] = &[2, 4, 8, 16];
pub const DEFAULT_QUERY_NOISE_WEIGHTS: &[f32] = &[0.0, 0.10, 0.20, 0.35];
pub const QUERIES_PER_LOAD: usize = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AssociativeCleanupSpec {
    pub schema_version: u32,
    pub dimensions: &'static [usize],
    pub memory_loads: &'static [usize],
    pub query_noise_weights: &'static [f32],
    pub seed: u64,
    pub queries_per_load: usize,
}

impl Default for AssociativeCleanupSpec {
    fn default() -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            dimensions: DEFAULT_DIMENSIONS,
            memory_loads: DEFAULT_MEMORY_LOADS,
            query_noise_weights: DEFAULT_QUERY_NOISE_WEIGHTS,
            seed: DEFAULT_SEED,
            queries_per_load: QUERIES_PER_LOAD,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssociativeCleanupCell {
    pub resolution: usize,
    pub memory_load: usize,
    pub query_noise_weight: f32,
    pub correct: u64,
    pub total: u64,
    pub accuracy: f64,
    pub mean_margin: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssociativeCleanupEvidence {
    pub schema_version: u32,
    pub task: String,
    pub scenario_set: String,
    pub scenario_revision: String,
    pub seed: u64,
    pub queries_per_load: usize,
    pub spec_identity: String,
    pub cells: Vec<AssociativeCleanupCell>,
    pub artifact_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AssociativeCleanupError {
    UnsupportedSchema(u32),
    EmptyDimensions,
    EmptyMemoryLoads,
    EmptyNoiseWeights,
    InvalidDimension(usize),
    UnsortedDimensions,
    InvalidMemoryLoad(usize),
    UnsortedMemoryLoads,
    InvalidNoiseWeight(u32),
    UnsortedNoiseWeights,
    ZeroQueries,
    TooManyQueries,
}

impl std::fmt::Display for AssociativeCleanupError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSchema(v) => write!(f, "unsupported associative cleanup schema: {v}"),
            Self::EmptyDimensions => f.write_str("dimension ladder is empty"),
            Self::EmptyMemoryLoads => f.write_str("memory-load ladder is empty"),
            Self::EmptyNoiseWeights => f.write_str("query-noise ladder is empty"),
            Self::InvalidDimension(d) => write!(f, "invalid dimension: {d}"),
            Self::UnsortedDimensions => f.write_str("dimensions must be strictly ascending"),
            Self::InvalidMemoryLoad(n) => write!(f, "invalid memory load: {n}"),
            Self::UnsortedMemoryLoads => f.write_str("memory loads must be strictly ascending"),
            Self::InvalidNoiseWeight(bits) => write!(f, "invalid query-noise weight bits: {bits}"),
            Self::UnsortedNoiseWeights => f.write_str("query-noise weights must be strictly ascending"),
            Self::ZeroQueries => f.write_str("queries_per_load must be non-zero"),
            Self::TooManyQueries => f.write_str("queries_per_load exceeds the safety limit"),
        }
    }
}

impl std::error::Error for AssociativeCleanupError {}

impl AssociativeCleanupSpec {
    pub fn validate(&self) -> Result<(), AssociativeCleanupError> {
        if self.schema_version != SCHEMA_VERSION {
            return Err(AssociativeCleanupError::UnsupportedSchema(self.schema_version));
        }
        if self.dimensions.is_empty() {
            return Err(AssociativeCleanupError::EmptyDimensions);
        }
        if self.memory_loads.is_empty() {
            return Err(AssociativeCleanupError::EmptyMemoryLoads);
        }
        if self.query_noise_weights.is_empty() {
            return Err(AssociativeCleanupError::EmptyNoiseWeights);
        }
        if self.queries_per_load == 0 {
            return Err(AssociativeCleanupError::ZeroQueries);
        }
        if self.queries_per_load > 1_024 {
            return Err(AssociativeCleanupError::TooManyQueries);
        }

        let mut previous = 0usize;
        for &dimension in self.dimensions {
            if dimension == 0 || !dimension.is_power_of_two() {
                return Err(AssociativeCleanupError::InvalidDimension(dimension));
            }
            if dimension <= previous {
                return Err(AssociativeCleanupError::UnsortedDimensions);
            }
            previous = dimension;
        }

        let mut previous = 0usize;
        for &load in self.memory_loads {
            if load < 2 {
                return Err(AssociativeCleanupError::InvalidMemoryLoad(load));
            }
            if load <= previous {
                return Err(AssociativeCleanupError::UnsortedMemoryLoads);
            }
            previous = load;
        }

        let mut previous = f32::NEG_INFINITY;
        for &weight in self.query_noise_weights {
            if !weight.is_finite() || !(0.0..=1.0).contains(&weight) {
                return Err(AssociativeCleanupError::InvalidNoiseWeight(weight.to_bits()));
            }
            if weight <= previous {
                return Err(AssociativeCleanupError::UnsortedNoiseWeights);
            }
            previous = weight;
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"symthaea:hdc-associative-cleanup");
        bytes.extend_from_slice(&self.schema_version.to_be_bytes());
        bytes.extend_from_slice(&self.seed.to_be_bytes());
        bytes.extend_from_slice(&(self.queries_per_load as u64).to_be_bytes());
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

pub(crate) fn fixture_seed(seed: u64, index: usize, salt: u64) -> u64 {
    let mut x = seed
        ^ (index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ salt.wrapping_mul(0xD1B5_4A32_D192_ED03);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Convert a deterministic continuous random vector into an exact bipolar HV.
/// This deliberately uses ±1 components so the cleanup experiment has an
/// explicit, exact self-inverse binding algebra rather than relying on
/// ContinuousHV's non-self-inverse real-valued binding.
pub(crate) fn bipolar(dim: usize, seed: u64) -> ContinuousHV {
    let random = ContinuousHV::random(dim, seed);
    ContinuousHV::from_vec(
        random
            .values
            .into_iter()
            .map(|v| if v >= 0.0 { 1.0 } else { -1.0 })
            .collect(),
    )
}

pub(crate) fn noisy_query(value: &ContinuousHV, noise_weight: f32, seed: u64) -> ContinuousHV {
    if noise_weight == 0.0 {
        return value.clone();
    }
    let noise = bipolar(value.dim(), seed);
    let signal_weight = 1.0 - noise_weight;
    let values = value
        .values
        .iter()
        .zip(noise.values.iter())
        .map(|(&v, &n)| signal_weight * v + noise_weight * n)
        .collect();
    ContinuousHV::from_vec(values)
}

fn run_cell(
    resolution: usize,
    memory_load: usize,
    query_noise_weight: f32,
    seed: u64,
    queries_per_load: usize,
) -> AssociativeCleanupCell {
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

    let mut correct = 0u64;
    let mut margin_sum = 0.0f64;

    for target in 0..memory_load {
        for query_index in 0..queries_per_load {
            let key_noise = noisy_query(
                &keys[target],
                query_noise_weight,
                fixture_seed(seed, target, query_index as u64 + 0x4E4F4953),
            );
            let released = key_noise.bind(&memory);

            let mut best_index = 0usize;
            let mut best = f32::NEG_INFINITY;
            let mut second = f32::NEG_INFINITY;
            for (candidate, value) in values.iter().enumerate() {
                let similarity = released.similarity(value);
                if similarity > best {
                    second = best;
                    best = similarity;
                    best_index = candidate;
                } else if similarity > second {
                    second = similarity;
                }
            }

            if best_index == target {
                correct += 1;
            }
            margin_sum += f64::from(best - second);
        }
    }

    let total = (memory_load * queries_per_load) as u64;
    AssociativeCleanupCell {
        resolution,
        memory_load,
        query_noise_weight,
        correct,
        total,
        accuracy: correct as f64 / total as f64,
        mean_margin: margin_sum / total as f64,
    }
}

pub fn run_associative_cleanup(
    spec: AssociativeCleanupSpec,
) -> Result<AssociativeCleanupEvidence, AssociativeCleanupError> {
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
                    spec.queries_per_load,
                ));
            }
        }
    }

    let mut evidence = AssociativeCleanupEvidence {
        schema_version: SCHEMA_VERSION,
        task: TASK_NAME.to_owned(),
        scenario_set: SCENARIO_SET.to_owned(),
        scenario_revision: SCENARIO_REVISION.to_owned(),
        seed: spec.seed,
        queries_per_load: spec.queries_per_load,
        spec_identity: spec.identity_digest(),
        cells,
        artifact_digest: String::new(),
    };
    evidence.artifact_digest =
        digest(serde_json::to_vec(&evidence).expect("associative cleanup evidence serializes"));
    Ok(evidence)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_spec_is_valid_and_identity_commits_to_shape() {
        let spec = AssociativeCleanupSpec::default();
        spec.validate().unwrap();
        let changed = AssociativeCleanupSpec {
            memory_loads: &[2, 4],
            ..spec
        };
        assert_ne!(spec.identity_digest(), changed.identity_digest());
    }

    #[test]
    fn malformed_noise_ladder_fails_closed() {
        let spec = AssociativeCleanupSpec {
            query_noise_weights: &[0.20, 0.10],
            ..Default::default()
        };
        assert!(matches!(
            spec.validate(),
            Err(AssociativeCleanupError::UnsortedNoiseWeights)
        ));
    }

    #[test]
    fn bipolar_binding_is_exactly_self_inverse() {
        let a = bipolar(1_024, 7);
        let b = bipolar(1_024, 11);
        let recovered = a.bind(&a.bind(&b));
        assert_eq!(recovered, b);
    }

    #[test]
    fn paired_cells_are_reproducible() {
        let spec = AssociativeCleanupSpec {
            dimensions: &[1_024, 2_048],
            memory_loads: &[2, 4],
            query_noise_weights: &[0.0, 0.20],
            queries_per_load: 1,
            ..Default::default()
        };
        let a = run_associative_cleanup(spec).unwrap();
        let b = run_associative_cleanup(spec).unwrap();
        assert_eq!(a.cells, b.cells);
        assert_eq!(a.spec_identity, b.spec_identity);
    }

    #[test]
    fn cell_count_is_dimension_times_load_times_noise() {
        let spec = AssociativeCleanupSpec {
            dimensions: &[1_024, 2_048, 4_096],
            memory_loads: &[2, 4],
            query_noise_weights: &[0.0, 0.20],
            queries_per_load: 1,
            ..Default::default()
        };
        let evidence = run_associative_cleanup(spec).unwrap();
        assert_eq!(evidence.cells.len(), 12);
        assert!(evidence.cells.iter().all(|c| c.total >= 2));
    }
}
