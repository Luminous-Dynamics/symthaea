//! Deterministic empirical dimension-sweep contracts for HDC research.
//!
//! The sweep intentionally separates three things:
//! 1. the semantic experiment specification and identity,
//! 2. deterministic empirical measurements,
//! 3. execution provenance (CI runner, wall time, toolchain, etc.).
//!
//! This module does not declare a winning dimension. It provides reproducible
//! evidence so task-specific quality can be compared against representation
//! cost and concentration behavior.

use sha2::{Digest, Sha256};
use serde::{Deserialize, Serialize};

use super::dimension_observatory::OBSERVATORY_DIMENSIONS;

pub const DIMENSION_SWEEP_SCHEMA_VERSION: u32 = 1;
pub const DIMENSION_SWEEP_DOMAIN: &[u8] = b"symthaea:hdc-dimension-sweep";
pub const DEFAULT_SWEEP_SEED: u64 = 0xD1_M3_2026_u64;
pub const DEFAULT_SAMPLES_PER_DIMENSION: u32 = 32;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DimensionSweepSpec {
    pub schema_version: u32,
    pub dimensions: Vec<u32>,
    pub seed: u64,
    pub samples_per_dimension: u32,
}

impl DimensionSweepSpec {
    pub fn default_observatory() -> Self {
        Self {
            schema_version: DIMENSION_SWEEP_SCHEMA_VERSION,
            dimensions: OBSERVATORY_DIMENSIONS.iter().map(|&d| d as u32).collect(),
            seed: DEFAULT_SWEEP_SEED,
            samples_per_dimension: DEFAULT_SAMPLES_PER_DIMENSION,
        }
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(64 + self.dimensions.len() * 4);
        bytes.extend_from_slice(DIMENSION_SWEEP_DOMAIN);
        push_u32(&mut bytes, self.schema_version);
        push_u64(&mut bytes, self.seed);
        push_u32(&mut bytes, self.samples_per_dimension);
        push_u32(&mut bytes, self.dimensions.len() as u32);
        for &dimension in &self.dimensions {
            push_u32(&mut bytes, dimension);
        }
        bytes
    }

    pub fn identity_digest(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(self.canonical_bytes());
        hasher.finalize().into()
    }

    pub fn identity(&self) -> String {
        format!("sha256:{}", hex_digest(&self.identity_digest()))
    }

    pub fn validate(&self) -> Result<(), DimensionSweepError> {
        if self.schema_version != DIMENSION_SWEEP_SCHEMA_VERSION {
            return Err(DimensionSweepError::UnsupportedSchema(self.schema_version));
        }
        if self.dimensions.is_empty() {
            return Err(DimensionSweepError::EmptyDimensions);
        }
        if self.samples_per_dimension == 0 {
            return Err(DimensionSweepError::ZeroSamples);
        }
        for &dimension in &self.dimensions {
            let d = dimension as usize;
            if d == 0 || !d.is_power_of_two() {
                return Err(DimensionSweepError::InvalidDimension(d));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DimensionSweepRecord {
    pub dimension: u32,
    pub samples: u32,
    pub mean_abs_cosine: f64,
    pub max_abs_cosine: f64,
    pub mean_bind_abs_cosine: f64,
    pub mean_bundle_member_cosine: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DimensionSweepEvidence {
    pub schema_version: u32,
    pub sweep_identity: String,
    pub records: Vec<DimensionSweepRecord>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DimensionSweepError {
    UnsupportedSchema(u32),
    EmptyDimensions,
    ZeroSamples,
    InvalidDimension(usize),
}

impl std::fmt::Display for DimensionSweepError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSchema(v) => write!(f, "unsupported sweep schema: {v}"),
            Self::EmptyDimensions => f.write_str("dimension sweep has no dimensions"),
            Self::ZeroSamples => f.write_str("dimension sweep requires at least one sample"),
            Self::InvalidDimension(d) => write!(f, "dimension is not a positive power of two: {d}"),
        }
    }
}

impl std::error::Error for DimensionSweepError {}

/// Run a deterministic empirical sweep.
///
/// The generated vectors use a fixed SplitMix64 stream and are deliberately
/// independent of wall-clock state, process identity, or external RNG state.
pub fn run_dimension_sweep(
    spec: &DimensionSweepSpec,
) -> Result<DimensionSweepEvidence, DimensionSweepError> {
    spec.validate()?;

    let mut records = Vec::with_capacity(spec.dimensions.len());
    for &dimension in &spec.dimensions {
        let d = dimension as usize;
        let mut abs_cosine_sum = 0.0;
        let mut max_abs_cosine: f64 = 0.0;
        let mut bind_abs_cosine_sum = 0.0;
        let mut bundle_member_cosine_sum = 0.0;

        for sample in 0..spec.samples_per_dimension {
            let a = deterministic_vector(d, spec.seed, dimension as u64, sample as u64, 0);
            let b = deterministic_vector(d, spec.seed, dimension as u64, sample as u64, 1);
            let c = deterministic_vector(d, spec.seed, dimension as u64, sample as u64, 2);

            let cosine = cosine(&a, &b).abs();
            abs_cosine_sum += cosine;
            max_abs_cosine = max_abs_cosine.max(cosine);

            let bound: Vec<f32> = a.iter().zip(&b).map(|(&x, &y)| x * y).collect();
            bind_abs_cosine_sum += cosine(&bound, &a).abs();

            let bundle: Vec<f32> = a
                .iter()
                .zip(&b)
                .zip(&c)
                .map(|((&x, &y), &z)| (x + y + z) / 3.0)
                .collect();
            bundle_member_cosine_sum += cosine(&bundle, &a);
        }

        let n = f64::from(spec.samples_per_dimension);
        records.push(DimensionSweepRecord {
            dimension,
            samples: spec.samples_per_dimension,
            mean_abs_cosine: abs_cosine_sum / n,
            max_abs_cosine,
            mean_bind_abs_cosine: bind_abs_cosine_sum / n,
            mean_bundle_member_cosine: bundle_member_cosine_sum / n,
        });
    }

    Ok(DimensionSweepEvidence {
        schema_version: DIMENSION_SWEEP_SCHEMA_VERSION,
        sweep_identity: spec.identity(),
        records,
    })
}

fn deterministic_vector(
    dimension: usize,
    seed: u64,
    dimension_tag: u64,
    sample: u64,
    stream: u64,
) -> Vec<f32> {
    let mut state = seed
        ^ dimension_tag.rotate_left(17)
        ^ sample.rotate_left(31)
        ^ stream.rotate_left(47);
    let mut values = Vec::with_capacity(dimension);

    for _ in 0..dimension {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        let unit = ((z >> 40) as u32) as f32 / ((1u32 << 24) - 1) as f32;
        values.push(unit * 2.0 - 1.0);
    }

    values
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0f64;
    let mut aa = 0.0f64;
    let mut bb = 0.0f64;

    for (&x, &y) in a.iter().zip(b) {
        let x = f64::from(x);
        let y = f64::from(y);
        dot += x * y;
        aa += x * x;
        bb += y * y;
    }

    dot / (aa * bb).sqrt()
}

fn push_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_be_bytes());
}

fn push_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_be_bytes());
}

fn hex_digest(digest: &[u8; 32]) -> String {
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_spec_covers_observatory() {
        let spec = DimensionSweepSpec::default_observatory();
        assert_eq!(spec.schema_version, DIMENSION_SWEEP_SCHEMA_VERSION);
        assert_eq!(spec.dimensions.len(), OBSERVATORY_DIMENSIONS.len());
        assert_eq!(spec.samples_per_dimension, DEFAULT_SAMPLES_PER_DIMENSION);
        spec.validate().unwrap();
    }

    #[test]
    fn identity_is_stable_and_sensitive_to_semantic_fields() {
        let spec = DimensionSweepSpec::default_observatory();
        let identity = spec.identity();
        assert!(identity.starts_with("sha256:"));
        assert_eq!(identity.len(), 71);
        assert_eq!(identity, spec.identity());

        let mut changed = spec.clone();
        changed.samples_per_dimension += 1;
        assert_ne!(identity, changed.identity());
    }

    #[test]
    fn canonical_encoding_is_explicit_and_order_sensitive() {
        let mut a = DimensionSweepSpec::default_observatory();
        let mut b = a.clone();
        b.dimensions.reverse();
        assert_ne!(a.canonical_bytes(), b.canonical_bytes());

        a.seed = a.seed.wrapping_add(1);
        assert_ne!(a.canonical_bytes(), spec_bytes_without_seed(&b));
    }

    fn spec_bytes_without_seed(spec: &DimensionSweepSpec) -> Vec<u8> {
        let mut clone = spec.clone();
        clone.seed = DEFAULT_SWEEP_SEED;
        clone.canonical_bytes()
    }

    #[test]
    fn empirical_sweep_is_deterministic() {
        let spec = DimensionSweepSpec {
            dimensions: vec![1_024, 2_048],
            seed: DEFAULT_SWEEP_SEED,
            samples_per_dimension: 2,
            schema_version: DIMENSION_SWEEP_SCHEMA_VERSION,
        };
        let a = run_dimension_sweep(&spec).unwrap();
        let b = run_dimension_sweep(&spec).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.records.len(), 2);
    }

    #[test]
    fn empirical_concentration_improves_on_smoke_ladder() {
        let spec = DimensionSweepSpec {
            dimensions: vec![1_024, 16_384, 65_536],
            seed: DEFAULT_SWEEP_SEED,
            samples_per_dimension: 8,
            schema_version: DIMENSION_SWEEP_SCHEMA_VERSION,
        };
        let evidence = run_dimension_sweep(&spec).unwrap();

        assert!(evidence.records[1].mean_abs_cosine < evidence.records[0].mean_abs_cosine);
        assert!(evidence.records[2].mean_abs_cosine < evidence.records[1].mean_abs_cosine);
        assert!(evidence.records[2].max_abs_cosine < evidence.records[0].max_abs_cosine);
    }

    #[test]
    fn validation_fails_closed() {
        let mut spec = DimensionSweepSpec::default_observatory();
        spec.samples_per_dimension = 0;
        assert_eq!(
            spec.validate(),
            Err(DimensionSweepError::ZeroSamples)
        );

        spec.samples_per_dimension = 1;
        spec.dimensions = vec![3];
        assert_eq!(
            spec.validate(),
            Err(DimensionSweepError::InvalidDimension(3))
        );
    }
}
