//! Deterministic sequence/order retrieval task family for HDC dimension research.
//!
//! This family tests position encoding separately from associative key/value
//! cleanup. Each sequence is encoded as a permutation-marked superposition.
//! Queries probe a known position and recover the item by nearest-codebook
//! similarity. Sequence length and symbol corruption are independent axes.

use super::unified_hv::ContinuousHV;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const SCHEMA_VERSION: u32 = 1;
pub const TASK_NAME: &str = "hdc_sequence_order_retrieval";
pub const SCENARIO_SET: &str = "sequence-order-retrieval-v1";
pub const SCENARIO_REVISION: &str = "sha256:sequence-order-fixture-v1";
pub const DEFAULT_SEED: u64 = 0x5345_5155_454E_4345;
pub const DEFAULT_DIMENSIONS: &[usize] =
    &[1_024, 2_048, 4_096, 8_192, 16_384, 32_768, 65_536, 131_072, 262_144];
pub const DEFAULT_SEQUENCE_LENGTHS: &[usize] = &[2, 4, 8, 16, 32];
pub const DEFAULT_QUERY_NOISE_WEIGHTS: &[f32] = &[0.0, 0.10, 0.20, 0.35];
pub const QUERIES_PER_POSITION: usize = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SequenceOrderSpec {
    pub schema_version: u32,
    pub dimensions: &'static [usize],
    pub sequence_lengths: &'static [usize],
    pub query_noise_weights: &'static [f32],
    pub seed: u64,
    pub queries_per_position: usize,
}

impl Default for SequenceOrderSpec {
    fn default() -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            dimensions: DEFAULT_DIMENSIONS,
            sequence_lengths: DEFAULT_SEQUENCE_LENGTHS,
            query_noise_weights: DEFAULT_QUERY_NOISE_WEIGHTS,
            seed: DEFAULT_SEED,
            queries_per_position: QUERIES_PER_POSITION,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SequenceOrderCell {
    pub resolution: usize,
    pub sequence_length: usize,
    pub query_noise_weight: f32,
    pub correct: u64,
    pub total: u64,
    pub accuracy: f64,
    pub mean_margin: f64,
    pub order_discrimination: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SequenceOrderEvidence {
    pub schema_version: u32,
    pub task: String,
    pub scenario_set: String,
    pub scenario_revision: String,
    pub seed: u64,
    pub queries_per_position: usize,
    pub spec_identity: String,
    pub cells: Vec<SequenceOrderCell>,
    pub artifact_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SequenceOrderError {
    UnsupportedSchema(u32),
    EmptyDimensions,
    EmptyLengths,
    EmptyNoiseWeights,
    InvalidDimension(usize),
    UnsortedDimensions,
    InvalidSequenceLength(usize),
    UnsortedSequenceLengths,
    InvalidNoiseWeight(u32),
    UnsortedNoiseWeights,
    ZeroQueries,
    TooManyQueries,
}

impl std::fmt::Display for SequenceOrderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSchema(v) => write!(f, "unsupported sequence-order schema: {v}"),
            Self::EmptyDimensions => f.write_str("dimension ladder is empty"),
            Self::EmptyLengths => f.write_str("sequence-length ladder is empty"),
            Self::EmptyNoiseWeights => f.write_str("query-noise ladder is empty"),
            Self::InvalidDimension(d) => write!(f, "invalid dimension: {d}"),
            Self::UnsortedDimensions => f.write_str("dimensions must be strictly ascending"),
            Self::InvalidSequenceLength(n) => write!(f, "invalid sequence length: {n}"),
            Self::UnsortedSequenceLengths => f.write_str("sequence lengths must be strictly ascending"),
            Self::InvalidNoiseWeight(bits) => write!(f, "invalid query-noise weight bits: {bits}"),
            Self::UnsortedNoiseWeights => write!(f, "query-noise weights must be strictly ascending"),
            Self::ZeroQueries => f.write_str("queries_per_position must be non-zero"),
            Self::TooManyQueries => f.write_str("queries_per_position exceeds the safety limit"),
        }
    }
}

impl std::error::Error for SequenceOrderError {}

impl SequenceOrderSpec {
    pub fn validate(&self) -> Result<(), SequenceOrderError> {
        if self.schema_version != SCHEMA_VERSION {
            return Err(SequenceOrderError::UnsupportedSchema(self.schema_version));
        }
        if self.dimensions.is_empty() {
            return Err(SequenceOrderError::EmptyDimensions);
        }
        if self.sequence_lengths.is_empty() {
            return Err(SequenceOrderError::EmptyLengths);
        }
        if self.query_noise_weights.is_empty() {
            return Err(SequenceOrderError::EmptyNoiseWeights);
        }
        if self.queries_per_position == 0 {
            return Err(SequenceOrderError::ZeroQueries);
        }
        if self.queries_per_position > 1_024 {
            return Err(SequenceOrderError::TooManyQueries);
        }

        let mut previous = 0usize;
        for &dimension in self.dimensions {
            if dimension == 0 || !dimension.is_power_of_two() {
                return Err(SequenceOrderError::InvalidDimension(dimension));
            }
            if dimension <= previous {
                return Err(SequenceOrderError::UnsortedDimensions);
            }
            previous = dimension;
        }

        let mut previous = 0usize;
        for &length in self.sequence_lengths {
            if length == 0 || length > 256 {
                return Err(SequenceOrderError::InvalidSequenceLength(length));
            }
            if length <= previous {
                return Err(SequenceOrderError::UnsortedSequenceLengths);
            }
            previous = length;
        }

        let mut previous = f32::NEG_INFINITY;
        for &weight in self.query_noise_weights {
            if !weight.is_finite() || !(0.0..=1.0).contains(&weight) {
                return Err(SequenceOrderError::InvalidNoiseWeight(weight.to_bits()));
            }
            if weight <= previous {
                return Err(SequenceOrderError::UnsortedNoiseWeights);
            }
            previous = weight;
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"symthaea:hdc-sequence-order");
        bytes.extend_from_slice(&self.schema_version.to_be_bytes());
        bytes.extend_from_slice(&self.seed.to_be_bytes());
        bytes.extend_from_slice(&(self.queries_per_position as u64).to_be_bytes());
        bytes.extend_from_slice(&(self.dimensions.len() as u64).to_be_bytes());
        for &dimension in self.dimensions {
            bytes.extend_from_slice(&(dimension as u64).to_be_bytes());
        }
        bytes.extend_from_slice(&(self.sequence_lengths.len() as u64).to_be_bytes());
        for &length in self.sequence_lengths {
            bytes.extend_from_slice(&(length as u64).to_be_bytes());
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

fn fixture_seed(seed: u64, index: usize, salt: u64) -> u64 {
    let mut x = seed
        ^ (index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ salt.wrapping_mul(0xD1B5_4A32_D192_ED03);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn bipolar(dim: usize, seed: u64) -> ContinuousHV {
    let random = ContinuousHV::random(dim, seed);
    ContinuousHV::from_vec(
        random.values
            .into_iter()
            .map(|v| if v >= 0.0 { 1.0 } else { -1.0 })
            .collect(),
    )
}

fn permute(input: &ContinuousHV, shift: usize) -> ContinuousHV {
    if input.values.is_empty() {
        return input.clone();
    }
    let n = input.values.len();
    let shift = shift % n;
    if shift == 0 {
        return input.clone();
    }
    let mut out = vec![0.0; n];
    for (index, value) in input.values.iter().enumerate() {
        out[(index + shift) % n] = *value;
    }
    ContinuousHV::from_vec(out)
}

fn bundle(vectors: &[ContinuousHV]) -> ContinuousHV {
    let dim = vectors[0].dim();
    let mut sums = vec![0.0f32; dim];
    for vector in vectors {
        for (sum, value) in sums.iter_mut().zip(vector.values.iter()) {
            *sum += *value;
        }
    }
    ContinuousHV::from_vec(
        sums.into_iter()
            .map(|sum| if sum >= 0.0 { 1.0 } else { -1.0 })
            .collect(),
    )
}

fn noisy_query(value: &ContinuousHV, noise_weight: f32, seed: u64) -> ContinuousHV {
    if noise_weight == 0.0 {
        return value.clone();
    }
    let noise = bipolar(value.dim(), seed);
    let signal_weight = 1.0 - noise_weight;
    ContinuousHV::from_vec(
        value.values.iter()
            .zip(noise.values.iter())
            .map(|(&value, &noise)| signal_weight * value + noise_weight * noise)
            .collect(),
    )
}

fn run_cell(
    resolution: usize,
    sequence_length: usize,
    query_noise_weight: f32,
    seed: u64,
    queries_per_position: usize,
) -> SequenceOrderCell {
    let items: Vec<ContinuousHV> = (0..sequence_length)
        .map(|position| bipolar(resolution, fixture_seed(seed, position, 0x4954454D)))
        .collect();

    let permuted: Vec<ContinuousHV> = items
        .iter()
        .enumerate()
        .map(|(position, item)| permute(item, position))
        .collect();
    let sequence = bundle(&permuted);

    let mut correct = 0u64;
    let mut margin_sum = 0.0f64;
    let mut order_discrimination_sum = 0.0f64;

    for position in 0..sequence_length {
        let probe = permute(&sequence, resolution - (position % resolution));
        for query_index in 0..queries_per_position {
            let query = noisy_query(
                &probe,
                query_noise_weight,
                fixture_seed(
                    seed,
                    position,
                    query_index as u64 + 0x51554552,
                ),
            );

            let mut best_index = 0usize;
            let mut best = f32::NEG_INFINITY;
            let mut second = f32::NEG_INFINITY;
            for (candidate, item) in items.iter().enumerate() {
                let similarity = query.similarity(item);
                if similarity > best {
                    second = best;
                    best = similarity;
                    best_index = candidate;
                } else if similarity > second {
                    second = similarity;
                }
            }

            if best_index == position {
                correct += 1;
            }
            margin_sum += f64::from(best - second);

            let wrong_order_probe = permute(
                &sequence,
                resolution - ((sequence_length - 1 - position) % resolution),
            );
            let wrong_order_similarity = query.similarity(&wrong_order_probe);
            order_discrimination_sum += f64::from(query.similarity(&probe) - wrong_order_similarity);
        }
    }

    let total = (sequence_length * queries_per_position) as u64;
    SequenceOrderCell {
        resolution,
        sequence_length,
        query_noise_weight,
        correct,
        total,
        accuracy: correct as f64 / total as f64,
        mean_margin: margin_sum / total as f64,
        order_discrimination: order_discrimination_sum / total as f64,
    }
}

pub fn run_sequence_order(
    spec: SequenceOrderSpec,
) -> Result<SequenceOrderEvidence, SequenceOrderError> {
    spec.validate()?;
    let mut cells = Vec::with_capacity(
        spec.dimensions.len()
            * spec.sequence_lengths.len()
            * spec.query_noise_weights.len(),
    );

    for &resolution in spec.dimensions {
        for &sequence_length in spec.sequence_lengths {
            for &query_noise_weight in spec.query_noise_weights {
                cells.push(run_cell(
                    resolution,
                    sequence_length,
                    query_noise_weight,
                    spec.seed,
                    spec.queries_per_position,
                ));
            }
        }
    }

    let mut evidence = SequenceOrderEvidence {
        schema_version: SCHEMA_VERSION,
        task: TASK_NAME.to_owned(),
        scenario_set: SCENARIO_SET.to_owned(),
        scenario_revision: SCENARIO_REVISION.to_owned(),
        seed: spec.seed,
        queries_per_position: spec.queries_per_position,
        spec_identity: spec.identity_digest(),
        cells,
        artifact_digest: String::new(),
    };
    evidence.artifact_digest =
        digest(serde_json::to_vec(&evidence).expect("sequence-order evidence serializes"));
    Ok(evidence)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_spec_is_valid_and_identity_commits_to_shape() {
        let spec = SequenceOrderSpec::default();
        spec.validate().unwrap();
        let changed = SequenceOrderSpec {
            sequence_lengths: &[2, 4],
            ..spec
        };
        assert_ne!(spec.identity_digest(), changed.identity_digest());
    }

    #[test]
    fn malformed_noise_ladder_fails_closed() {
        let spec = SequenceOrderSpec {
            query_noise_weights: &[0.20, 0.10],
            ..Default::default()
        };
        assert!(matches!(
            spec.validate(),
            Err(SequenceOrderError::UnsortedNoiseWeights)
        ));
    }

    #[test]
    fn permutation_is_exactly_invertible() {
        let value = bipolar(1_024, 7);
        let restored = permute(&permute(&value, 17), value.dim() - 17);
        assert_eq!(restored, value);
    }

    #[test]
    fn paired_cells_are_reproducible() {
        let spec = SequenceOrderSpec {
            dimensions: &[1_024, 2_048],
            sequence_lengths: &[2, 4],
            query_noise_weights: &[0.0, 0.20],
            queries_per_position: 1,
            ..Default::default()
        };
        let a = run_sequence_order(spec).unwrap();
        let b = run_sequence_order(spec).unwrap();
        assert_eq!(a.cells, b.cells);
        assert_eq!(a.spec_identity, b.spec_identity);
    }

    #[test]
    fn cell_count_is_dimension_times_length_times_noise() {
        let spec = SequenceOrderSpec {
            dimensions: &[1_024, 2_048, 4_096],
            sequence_lengths: &[2, 4],
            query_noise_weights: &[0.0, 0.20],
            queries_per_position: 1,
            ..Default::default()
        };
        let evidence = run_sequence_order(spec).unwrap();
        assert_eq!(evidence.cells.len(), 12);
        assert!(evidence.cells.iter().all(|c| c.total >= 2));
    }
}
