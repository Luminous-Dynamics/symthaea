// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Encrypted Vector Operations for Federated Learning
//!
//! High-level API for encrypting and aggregating gradient vectors
//! using homomorphic encryption. Designed for secure FL workflows.

use crate::{
    encoding::FixedPointEncoder,
    paillier::{PaillierCiphertext, PaillierKeyPair, PaillierPublicKey},
    HeError, HeResult,
};
use serde::{Deserialize, Serialize};
use tracing::{debug, instrument};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// An encrypted vector of values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedVector {
    /// Encrypted elements
    pub elements: Vec<PaillierCiphertext>,
    /// Number of elements
    pub len: usize,
}

impl EncryptedVector {
    /// Create an encrypted vector from encrypted elements
    pub fn new(elements: Vec<PaillierCiphertext>) -> Self {
        let len = elements.len();
        Self { elements, len }
    }

    /// Get the length of the vector
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if vector is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get an element by index
    pub fn get(&self, index: usize) -> Option<&PaillierCiphertext> {
        self.elements.get(index)
    }

    /// Homomorphic element-wise addition
    #[instrument(skip_all, fields(len = self.len))]
    pub fn add(&self, other: &Self) -> HeResult<Self> {
        if self.len != other.len {
            return Err(HeError::DimensionMismatch {
                expected: self.len,
                actual: other.len,
            });
        }

        let elements: Vec<_> = self
            .elements
            .iter()
            .zip(other.elements.iter())
            .map(|(a, b)| a.add(b))
            .collect();

        Ok(Self::new(elements))
    }

    /// Homomorphic scalar multiplication (all elements by same scalar)
    pub fn scalar_mul(&self, scalar: i64) -> Self {
        let elements: Vec<_> = self.elements.iter().map(|e| e.scalar_mul(scalar)).collect();
        Self::new(elements)
    }

    /// Compute weighted sum: sum(elements * weights)
    pub fn weighted_sum(&self, weights: &[i64]) -> HeResult<PaillierCiphertext> {
        if self.len != weights.len() {
            return Err(HeError::DimensionMismatch {
                expected: self.len,
                actual: weights.len(),
            });
        }

        if self.elements.is_empty() {
            return Err(HeError::ParameterError("Empty vector".to_string()));
        }

        let weighted: Vec<_> = self
            .elements
            .iter()
            .zip(weights.iter())
            .map(|(e, &w)| e.scalar_mul(w))
            .collect();

        let mut sum = weighted[0].clone();
        for elem in weighted.iter().skip(1) {
            sum = sum.add(elem);
        }

        Ok(sum)
    }

    /// Re-randomize all elements for privacy
    pub fn rerandomize(&self, public_key: &PaillierPublicKey) -> Self {
        let elements: Vec<_> = self
            .elements
            .iter()
            .map(|e| e.rerandomize(public_key))
            .collect();
        Self::new(elements)
    }

    #[cfg(feature = "parallel")]
    /// Parallel homomorphic addition (requires 'parallel' feature)
    pub fn add_parallel(&self, other: &Self) -> HeResult<Self> {
        if self.len != other.len {
            return Err(HeError::DimensionMismatch {
                expected: self.len,
                actual: other.len,
            });
        }

        let elements: Vec<_> = self
            .elements
            .par_iter()
            .zip(other.elements.par_iter())
            .map(|(a, b)| a.add(b))
            .collect();

        Ok(Self::new(elements))
    }
}

/// Encrypted gradient for federated learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedGradient {
    /// The encrypted gradient vector
    pub vector: EncryptedVector,
    /// Metadata
    pub metadata: GradientMetadata,
}

/// Metadata about an encrypted gradient
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientMetadata {
    /// Participant ID who contributed this gradient
    pub participant_id: String,
    /// Training round number
    pub round: u64,
    /// Number of samples used in this gradient
    pub sample_count: u64,
    /// Original dimension before encryption
    pub dimension: usize,
    /// Encoding precision used
    pub precision: u32,
    /// Timestamp of creation
    pub timestamp: u64,
}

impl EncryptedGradient {
    /// Create a new encrypted gradient
    pub fn new(vector: EncryptedVector, metadata: GradientMetadata) -> Self {
        Self { vector, metadata }
    }

    /// Create from a float gradient vector
    #[instrument(skip_all, fields(len = gradient.len(), participant = %metadata.participant_id))]
    pub fn from_gradient(
        gradient: &[f64],
        public_key: &PaillierPublicKey,
        encoder: &FixedPointEncoder,
        metadata: GradientMetadata,
    ) -> HeResult<Self> {
        debug!(len = gradient.len(), "Encrypting gradient");

        let encoded = encoder.encode_vec(gradient)?;
        let elements: Vec<_> = encoded.iter().map(|&v| public_key.encrypt(v)).collect();

        Ok(Self::new(EncryptedVector::new(elements), metadata))
    }

    /// Decrypt the gradient
    #[instrument(skip_all)]
    pub fn decrypt(
        &self,
        keypair: &PaillierKeyPair,
        encoder: &FixedPointEncoder,
    ) -> HeResult<Vec<f64>> {
        debug!(len = self.vector.len(), "Decrypting gradient");

        let decrypted: Result<Vec<_>, _> = self
            .vector
            .elements
            .iter()
            .map(|c| keypair.decrypt_i64(c))
            .collect();

        let values = decrypted?;
        Ok(encoder.decode_vec(&values))
    }

    /// Aggregate multiple gradients (encrypted addition)
    #[instrument(skip_all, fields(count = gradients.len()))]
    pub fn aggregate(gradients: &[Self]) -> HeResult<EncryptedVector> {
        if gradients.is_empty() {
            return Err(HeError::ParameterError("No gradients to aggregate".to_string()));
        }

        let first = &gradients[0].vector;
        let mut result = first.clone();

        for grad in gradients.iter().skip(1) {
            result = result.add(&grad.vector)?;
        }

        debug!(
            "Aggregated {} gradients",
            gradients.len()
        );

        Ok(result)
    }

    /// Aggregate with weights (for weighted FedAvg)
    #[instrument(skip_all)]
    pub fn aggregate_weighted(gradients: &[Self], weights: &[i64]) -> HeResult<EncryptedVector> {
        if gradients.len() != weights.len() {
            return Err(HeError::DimensionMismatch {
                expected: gradients.len(),
                actual: weights.len(),
            });
        }

        if gradients.is_empty() {
            return Err(HeError::ParameterError("No gradients to aggregate".to_string()));
        }

        let weighted: Vec<_> = gradients
            .iter()
            .zip(weights.iter())
            .map(|(g, &w)| g.vector.scalar_mul(w))
            .collect();

        let mut result = weighted[0].clone();
        for vec in weighted.iter().skip(1) {
            result = result.add(vec)?;
        }

        Ok(result)
    }
}

/// Builder for encrypted gradients
pub struct EncryptedGradientBuilder<'a> {
    public_key: &'a PaillierPublicKey,
    encoder: FixedPointEncoder,
    participant_id: String,
    round: u64,
}

impl<'a> EncryptedGradientBuilder<'a> {
    /// Create a new builder
    pub fn new(public_key: &'a PaillierPublicKey) -> Self {
        Self {
            public_key,
            encoder: FixedPointEncoder::for_gradients(),
            participant_id: String::new(),
            round: 0,
        }
    }

    /// Set encoding precision
    pub fn with_precision(mut self, precision: u32) -> Self {
        self.encoder = FixedPointEncoder::new(crate::encoding::EncodingParams::new(precision));
        self
    }

    /// Set participant ID
    pub fn with_participant(mut self, id: impl Into<String>) -> Self {
        self.participant_id = id.into();
        self
    }

    /// Set training round
    pub fn with_round(mut self, round: u64) -> Self {
        self.round = round;
        self
    }

    /// Build the encrypted gradient
    pub fn encrypt(self, gradient: &[f64], sample_count: u64) -> HeResult<EncryptedGradient> {
        let metadata = GradientMetadata {
            participant_id: self.participant_id,
            round: self.round,
            sample_count,
            dimension: gradient.len(),
            precision: self.encoder.params().precision,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        };

        EncryptedGradient::from_gradient(gradient, self.public_key, &self.encoder, metadata)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoding::EncodingParams;

    fn setup() -> (PaillierKeyPair, FixedPointEncoder) {
        let keypair = PaillierKeyPair::generate(1024).unwrap();
        let encoder = FixedPointEncoder::new(EncodingParams::new(6));
        (keypair, encoder)
    }

    #[test]
    fn test_encrypted_vector_add() {
        let (keypair, _) = setup();
        let pk = keypair.public_key();

        let vec1 = EncryptedVector::new(vec![pk.encrypt(10), pk.encrypt(20), pk.encrypt(30)]);

        let vec2 = EncryptedVector::new(vec![pk.encrypt(1), pk.encrypt(2), pk.encrypt(3)]);

        let sum = vec1.add(&vec2).unwrap();
        assert_eq!(sum.len(), 3);

        assert_eq!(keypair.decrypt_i64(&sum.elements[0]).unwrap(), 11);
        assert_eq!(keypair.decrypt_i64(&sum.elements[1]).unwrap(), 22);
        assert_eq!(keypair.decrypt_i64(&sum.elements[2]).unwrap(), 33);
    }

    #[test]
    fn test_encrypted_vector_scalar_mul() {
        let (keypair, _) = setup();
        let pk = keypair.public_key();

        let vec = EncryptedVector::new(vec![pk.encrypt(5), pk.encrypt(10)]);
        let scaled = vec.scalar_mul(3);

        assert_eq!(keypair.decrypt_i64(&scaled.elements[0]).unwrap(), 15);
        assert_eq!(keypair.decrypt_i64(&scaled.elements[1]).unwrap(), 30);
    }

    #[test]
    fn test_encrypted_gradient_roundtrip() {
        let (keypair, encoder) = setup();
        let pk = keypair.public_key();

        let gradient = vec![0.1, -0.2, 0.3, -0.4, 0.5];
        let metadata = GradientMetadata {
            participant_id: "test".to_string(),
            round: 1,
            sample_count: 100,
            dimension: gradient.len(),
            precision: 6,
            timestamp: 0,
        };

        let encrypted =
            EncryptedGradient::from_gradient(&gradient, pk, &encoder, metadata).unwrap();
        let decrypted = encrypted.decrypt(&keypair, &encoder).unwrap();

        for (orig, dec) in gradient.iter().zip(decrypted.iter()) {
            assert!((orig - dec).abs() < 1e-5);
        }
    }

    #[test]
    fn test_gradient_aggregation() {
        let (keypair, encoder) = setup();
        let pk = keypair.public_key();

        let make_gradient = |values: Vec<f64>, id: &str| {
            let metadata = GradientMetadata {
                participant_id: id.to_string(),
                round: 1,
                sample_count: 100,
                dimension: values.len(),
                precision: 6,
                timestamp: 0,
            };
            EncryptedGradient::from_gradient(&values, pk, &encoder, metadata).unwrap()
        };

        let g1 = make_gradient(vec![1.0, 2.0, 3.0], "p1");
        let g2 = make_gradient(vec![4.0, 5.0, 6.0], "p2");
        let g3 = make_gradient(vec![7.0, 8.0, 9.0], "p3");

        let aggregated = EncryptedGradient::aggregate(&[g1, g2, g3]).unwrap();

        // Decrypt and verify sum
        let result: Vec<f64> = aggregated
            .elements
            .iter()
            .map(|c| encoder.decode(keypair.decrypt_i64(c).unwrap()))
            .collect();

        // Expected: [12.0, 15.0, 18.0]
        assert!((result[0] - 12.0).abs() < 1e-5);
        assert!((result[1] - 15.0).abs() < 1e-5);
        assert!((result[2] - 18.0).abs() < 1e-5);
    }

    #[test]
    fn test_builder_pattern() {
        let (keypair, _) = setup();
        let pk = keypair.public_key();

        let gradient = vec![0.1, 0.2, 0.3];
        let encrypted = EncryptedGradientBuilder::new(pk)
            .with_participant("agent-1")
            .with_round(5)
            .with_precision(6)
            .encrypt(&gradient, 50)
            .unwrap();

        assert_eq!(encrypted.metadata.participant_id, "agent-1");
        assert_eq!(encrypted.metadata.round, 5);
        assert_eq!(encrypted.metadata.sample_count, 50);
    }

    #[test]
    fn test_dimension_mismatch() {
        let (keypair, _) = setup();
        let pk = keypair.public_key();

        let vec1 = EncryptedVector::new(vec![pk.encrypt(1), pk.encrypt(2)]);
        let vec2 = EncryptedVector::new(vec![pk.encrypt(1)]);

        let result = vec1.add(&vec2);
        assert!(result.is_err());
    }
}
