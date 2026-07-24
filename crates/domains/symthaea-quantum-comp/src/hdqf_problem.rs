//! Deterministic controlled instances for the HDQF factorization pilot.
//!
//! This module models factorization of XOR-bound binary hypervectors. Under the
//! fixed bipolar mapping used by HDC, XOR is equivalent to component-wise
//! multiplication. These instances are synthetic research fixtures; they are
//! not integer-factorization or deployed-cryptography targets.

use crate::classical_hdc::BinaryHypervector;
use crate::errors::{QuantumCompError, Result};
use crate::provenance::fnv1a64;
use crate::rng::XorShift64;

/// Stable generator implementation label recorded in manifests.
pub const HDQF_GENERATOR_VERSION: &str = "hdqf-generator-v1";

const UNIQUE_GENERATION_CANDIDATE_LIMIT: usize = 1_000_000;
const UNIQUE_GENERATION_ATTEMPTS: usize = 1_024;

/// Preregistered synthetic instance families.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HdqfInstanceFamily {
    /// Random codebooks conditioned on the planted product having one factorization.
    PlantedUnique,
    /// Independent random codebooks without uniqueness conditioning.
    Random,
    /// Codebooks with a deliberately duplicated product.
    CollisionRich,
    /// Codewords derived from shared prototypes.
    Correlated,
    /// A planted solution with a deliberately constructed near-collision.
    Adversarial,
}

impl HdqfInstanceFamily {
    fn label(self) -> &'static str {
        match self {
            Self::PlantedUnique => "planted_unique",
            Self::Random => "random",
            Self::CollisionRich => "collision_rich",
            Self::Correlated => "correlated",
            Self::Adversarial => "adversarial",
        }
    }
}

/// Configuration for one deterministic HDQF instance.
#[derive(Debug, Clone, PartialEq)]
pub struct HdqfProblemConfig {
    /// Hypervector bit dimension `D`.
    pub dimension: usize,
    /// Number of factor codebooks `F`.
    pub factor_count: usize,
    /// Uniform entries per codebook `N`.
    pub codebook_size: usize,
    /// Independent target-bit flip probability.
    pub epsilon: f32,
    /// Synthetic instance family.
    pub family: HdqfInstanceFamily,
    /// Probability that correlated-family bits retain the shared prototype bit.
    pub correlation_rate: f32,
    /// Hamming distance of the injected adversarial alternative product.
    pub adversarial_margin_bits: usize,
    /// Deterministic instance seed.
    pub seed: u64,
}

impl HdqfProblemConfig {
    /// Validates dimensions, probabilities, and family-specific parameters.
    pub fn validate(&self) -> Result<()> {
        if self.dimension == 0 {
            return Err(QuantumCompError::InvalidDimension);
        }
        if self.factor_count < 2 {
            return Err(QuantumCompError::InvalidConfig(
                "HDQF factor_count must be at least two",
            ));
        }
        if self.codebook_size < 2 {
            return Err(QuantumCompError::InvalidConfig(
                "HDQF codebook_size must be at least two",
            ));
        }
        if !(0.0..=0.5).contains(&self.epsilon) {
            return Err(QuantumCompError::InvalidProbability);
        }
        if !(0.0..=1.0).contains(&self.correlation_rate) {
            return Err(QuantumCompError::InvalidProbability);
        }
        if self.adversarial_margin_bits == 0 || self.adversarial_margin_bits > self.dimension {
            return Err(QuantumCompError::InvalidConfig(
                "adversarial margin must be in 1..=dimension",
            ));
        }
        self.candidate_count()?;
        Ok(())
    }

    /// Returns `N^F`, rejecting sizes that overflow `usize`.
    pub fn candidate_count(&self) -> Result<usize> {
        if self.factor_count > u32::MAX as usize {
            return Err(QuantumCompError::InvalidConfig(
                "HDQF factor count exceeds exponent representation",
            ));
        }
        self.codebook_size
            .checked_pow(self.factor_count as u32)
            .ok_or(QuantumCompError::InvalidConfig(
                "HDQF candidate count overflows usize",
            ))
    }
}

/// A generated bound-hypervector factorization problem.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HdqfProblemInstance {
    /// Codebooks in factor order.
    pub codebooks: Vec<Vec<BinaryHypervector>>,
    /// Product before target noise is applied.
    pub clean_target: BinaryHypervector,
    /// Observed exact or noisy target.
    pub target: BinaryHypervector,
    /// Tuple selected before collision or noise analysis.
    pub planted_indices: Vec<usize>,
    /// Realized target bit flips.
    pub realized_noise_bits: usize,
    /// Requested noise probability stored as IEEE-754 bits for exact replay.
    pub epsilon_bits: u32,
    /// Generator seed.
    pub seed: u64,
    /// Instance family.
    pub family: HdqfInstanceFamily,
}

impl HdqfProblemInstance {
    /// Returns `D`.
    pub fn dimension(&self) -> usize {
        self.target.dimension()
    }

    /// Returns `F`.
    pub fn factor_count(&self) -> usize {
        self.codebooks.len()
    }

    /// Returns uniform `N`.
    pub fn codebook_size(&self) -> usize {
        self.codebooks.first().map_or(0, Vec::len)
    }

    /// Returns whether the observed target contains injected bit-flip noise.
    pub fn is_noisy(&self) -> bool {
        self.epsilon_bits != 0
    }

    /// Returns the requested target bit-flip probability.
    pub fn epsilon(&self) -> f32 {
        f32::from_bits(self.epsilon_bits)
    }

    /// Computes the bound product for one complete factor tuple.
    pub fn product_for_indices(&self, indices: &[usize]) -> Result<BinaryHypervector> {
        if indices.len() != self.factor_count() {
            return Err(QuantumCompError::DimensionMismatch {
                expected: self.factor_count(),
                actual: indices.len(),
            });
        }
        let mut product = BinaryHypervector::zeros(self.dimension())?;
        for (codebook, &index) in self.codebooks.iter().zip(indices) {
            let factor = codebook.get(index).ok_or(QuantumCompError::InvalidConfig(
                "HDQF factor index is outside its codebook",
            ))?;
            product = product.bind_xor(factor)?;
        }
        Ok(product)
    }

    /// Stable dependency-free canonical serialization used for replay and hashing.
    pub fn canonical_text(&self) -> String {
        let mut out = String::new();
        out.push_str("HDQF_INSTANCE_V1\n");
        out.push_str(&format!("generator={HDQF_GENERATOR_VERSION}\n"));
        out.push_str(&format!("family={}\n", self.family.label()));
        out.push_str(&format!("seed={}\n", self.seed));
        out.push_str(&format!("dimension={}\n", self.dimension()));
        out.push_str(&format!("factor_count={}\n", self.factor_count()));
        out.push_str(&format!("codebook_size={}\n", self.codebook_size()));
        out.push_str(&format!("epsilon_bits={:08x}\n", self.epsilon_bits));
        out.push_str(&format!(
            "realized_noise_bits={}\n",
            self.realized_noise_bits
        ));
        out.push_str("planted_indices=");
        append_indices(&mut out, &self.planted_indices);
        out.push('\n');
        out.push_str("clean_target=");
        append_hypervector(&mut out, &self.clean_target);
        out.push('\n');
        out.push_str("target=");
        append_hypervector(&mut out, &self.target);
        out.push('\n');
        for (factor, codebook) in self.codebooks.iter().enumerate() {
            out.push_str(&format!("codebook_{factor}="));
            for (index, vector) in codebook.iter().enumerate() {
                if index > 0 {
                    out.push(';');
                }
                append_hypervector(&mut out, vector);
            }
            out.push('\n');
        }
        out
    }

    /// Stable non-cryptographic fingerprint of [`Self::canonical_text`].
    pub fn reproducibility_fingerprint(&self) -> u64 {
        fnv1a64(self.canonical_text().as_bytes())
    }
}

/// Generates a deterministic instance from a validated configuration.
pub fn generate_hdqf_instance(config: &HdqfProblemConfig) -> Result<HdqfProblemInstance> {
    config.validate()?;
    if config.family == HdqfInstanceFamily::PlantedUnique
        && config.candidate_count()? > UNIQUE_GENERATION_CANDIDATE_LIMIT
    {
        return Err(QuantumCompError::InvalidConfig(
            "planted-unique generation exceeds exact conditioning limit",
        ));
    }

    let attempts = if config.family == HdqfInstanceFamily::PlantedUnique {
        UNIQUE_GENERATION_ATTEMPTS
    } else {
        1
    };

    for attempt in 0..attempts {
        let attempt_seed = config
            .seed
            .wrapping_add((attempt as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let mut instance = generate_unconditioned(config, attempt_seed)?;
        if config.family != HdqfInstanceFamily::PlantedUnique
            || exact_factorization_count(&instance, 2)? == 1
        {
            apply_target_noise(
                &mut instance,
                config.epsilon,
                attempt_seed ^ 0xD1B5_4A32_D192_ED03,
            )?;
            return Ok(instance);
        }
    }

    Err(QuantumCompError::InvalidConfig(
        "could not generate planted-unique HDQF instance within attempt limit",
    ))
}

fn generate_unconditioned(config: &HdqfProblemConfig, seed: u64) -> Result<HdqfProblemInstance> {
    let mut rng = XorShift64::new(seed);
    let mut codebooks = Vec::with_capacity(config.factor_count);

    let shared_prototypes = if config.family == HdqfInstanceFamily::Correlated {
        let mut prototypes = Vec::with_capacity(config.factor_count);
        for _ in 0..config.factor_count {
            prototypes.push(BinaryHypervector::random(config.dimension, rng.next_u64())?);
        }
        Some(prototypes)
    } else {
        None
    };

    for factor in 0..config.factor_count {
        let mut codebook = Vec::with_capacity(config.codebook_size);
        for _ in 0..config.codebook_size {
            let vector = if let Some(prototypes) = &shared_prototypes {
                prototypes[factor].with_bitflip_noise(1.0 - config.correlation_rate, rng.next_u64())
            } else {
                BinaryHypervector::random(config.dimension, rng.next_u64())?
            };
            codebook.push(vector);
        }
        codebooks.push(codebook);
    }

    let mut planted_indices = (0..config.factor_count)
        .map(|_| rng.next_usize(config.codebook_size).unwrap_or(0))
        .collect::<Vec<_>>();

    match config.family {
        HdqfInstanceFamily::CollisionRich => {
            planted_indices.fill(0);
            codebooks[0][1] = codebooks[0][0].clone();
        }
        HdqfInstanceFamily::Adversarial => {
            planted_indices.fill(0);
            let mut delta = BinaryHypervector::zeros(config.dimension)?;
            for bit in 0..config.adversarial_margin_bits {
                delta.set_bit(bit, true)?;
            }
            codebooks[0][1] = codebooks[0][0].bind_xor(&delta)?;
        }
        _ => {}
    }

    let mut instance = HdqfProblemInstance {
        clean_target: BinaryHypervector::zeros(config.dimension)?,
        target: BinaryHypervector::zeros(config.dimension)?,
        codebooks,
        planted_indices,
        realized_noise_bits: 0,
        epsilon_bits: if config.epsilon == 0.0 {
            0.0f32.to_bits()
        } else {
            config.epsilon.to_bits()
        },
        seed,
        family: config.family,
    };
    instance.clean_target = instance.product_for_indices(&instance.planted_indices)?;
    instance.target = instance.clean_target.clone();
    Ok(instance)
}

fn apply_target_noise(instance: &mut HdqfProblemInstance, epsilon: f32, seed: u64) -> Result<()> {
    instance.target = instance.clean_target.with_bitflip_noise(epsilon, seed);
    instance.realized_noise_bits = instance.clean_target.hamming_distance(&instance.target)?;
    Ok(())
}

fn exact_factorization_count(instance: &HdqfProblemInstance, stop_after: usize) -> Result<usize> {
    let mut indices = vec![0; instance.factor_count()];
    let mut count = 0usize;
    enumerate_indices(
        instance.codebook_size(),
        0,
        &mut indices,
        &mut |candidate| {
            if instance.product_for_indices(candidate)? == instance.clean_target {
                count += 1;
            }
            Ok(count < stop_after)
        },
    )?;
    Ok(count)
}

pub(crate) fn enumerate_indices(
    codebook_size: usize,
    depth: usize,
    indices: &mut [usize],
    visit: &mut impl FnMut(&[usize]) -> Result<bool>,
) -> Result<bool> {
    if depth == indices.len() {
        return visit(indices);
    }
    for index in 0..codebook_size {
        indices[depth] = index;
        if !enumerate_indices(codebook_size, depth + 1, indices, visit)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn append_indices(out: &mut String, indices: &[usize]) {
    for (position, index) in indices.iter().enumerate() {
        if position > 0 {
            out.push(',');
        }
        out.push_str(&index.to_string());
    }
}

fn append_hypervector(out: &mut String, vector: &BinaryHypervector) {
    for (position, word) in vector.words().iter().enumerate() {
        if position > 0 {
            out.push(',');
        }
        out.push_str(&format!("{word:016x}"));
    }
}
