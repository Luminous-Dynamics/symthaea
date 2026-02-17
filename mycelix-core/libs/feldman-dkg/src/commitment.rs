//! Feldman commitments for verifiable secret sharing
//!
//! Commitments allow verification of shares without revealing the secret.

use k256::{ProjectivePoint, AffinePoint};
use k256::elliptic_curve::group::GroupEncoding;
use serde::{Deserialize, Serialize};

use crate::polynomial::Polynomial;
use crate::scalar::Scalar;
use crate::error::{DkgError, DkgResult};

/// A Feldman commitment to a polynomial coefficient
///
/// C_i = g^{a_i} where a_i is the i-th coefficient
#[derive(Clone, Debug)]
pub struct Commitment(pub(crate) ProjectivePoint);

impl Commitment {
    /// Create a commitment to a scalar value
    pub fn new(scalar: &Scalar) -> Self {
        Self(scalar.base_mul())
    }

    /// Create commitment from a point
    pub fn from_point(point: ProjectivePoint) -> Self {
        Self(point)
    }

    /// Get the underlying point
    pub fn point(&self) -> &ProjectivePoint {
        &self.0
    }

    /// Convert to affine point
    pub fn to_affine(&self) -> AffinePoint {
        self.0.to_affine()
    }

    /// Serialize to compressed bytes (33 bytes)
    pub fn to_bytes(&self) -> Vec<u8> {
        self.0.to_affine().to_bytes().to_vec()
    }

    /// Deserialize from compressed bytes
    pub fn from_bytes(bytes: &[u8]) -> DkgResult<Self> {
        if bytes.len() != 33 {
            return Err(DkgError::SerializationError(
                format!("Invalid commitment length: expected 33, got {}", bytes.len()),
            ));
        }

        let mut arr = [0u8; 33];
        arr.copy_from_slice(bytes);

        let point = AffinePoint::from_bytes(&arr.into());
        if point.is_some().into() {
            Ok(Self(point.unwrap().into()))
        } else {
            Err(DkgError::SerializationError("Invalid curve point".into()))
        }
    }
}

impl std::ops::Add for &Commitment {
    type Output = Commitment;

    fn add(self, rhs: Self) -> Self::Output {
        Commitment(self.0 + rhs.0)
    }
}

impl std::ops::Mul<&Scalar> for &Commitment {
    type Output = Commitment;

    fn mul(self, rhs: &Scalar) -> Self::Output {
        Commitment(self.0 * rhs.inner())
    }
}

impl PartialEq for Commitment {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for Commitment {}

impl Serialize for Commitment {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let bytes = self.to_bytes();
        serializer.serialize_bytes(&bytes)
    }
}

impl<'de> Deserialize<'de> for Commitment {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let bytes: Vec<u8> = Deserialize::deserialize(deserializer)?;
        Commitment::from_bytes(&bytes).map_err(serde::de::Error::custom)
    }
}

/// A set of commitments to polynomial coefficients
///
/// For a polynomial f(x) = a_0 + a_1*x + ... + a_{t-1}*x^{t-1},
/// the commitment set is [C_0, C_1, ..., C_{t-1}] where C_i = g^{a_i}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CommitmentSet {
    /// Commitments to each coefficient
    commitments: Vec<Commitment>,
}

impl CommitmentSet {
    /// Create commitments from a polynomial
    pub fn from_polynomial(poly: &Polynomial) -> Self {
        let commitments = poly
            .coefficients()
            .iter()
            .map(Commitment::new)
            .collect();
        Self { commitments }
    }

    /// Create from raw commitments
    pub fn new(commitments: Vec<Commitment>) -> Self {
        Self { commitments }
    }

    /// Get the number of commitments (= threshold)
    pub fn len(&self) -> usize {
        self.commitments.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.commitments.is_empty()
    }

    /// Get commitment to the secret (C_0 = g^{a_0})
    pub fn secret_commitment(&self) -> &Commitment {
        &self.commitments[0]
    }

    /// Get all commitments
    pub fn commitments(&self) -> &[Commitment] {
        &self.commitments
    }

    /// Verify that a share is consistent with these commitments
    ///
    /// For participant i with share s_i = f(i), verify:
    /// g^{s_i} = Π_{j=0}^{t-1} C_j^{i^j}
    ///
    /// This uses the property that g^{f(i)} = g^{Σ a_j * i^j} = Π g^{a_j * i^j}
    pub fn verify_share(&self, participant_index: u32, share: &Scalar) -> bool {
        // Left side: g^share
        let left = share.base_mul();

        // Right side: Π C_j^{i^j}
        let i = Scalar::from_u64(participant_index as u64);
        let mut right = ProjectivePoint::IDENTITY;
        let mut i_power = Scalar::one(); // i^0 = 1

        for commitment in &self.commitments {
            // Add C_j * i^j
            right += commitment.0 * i_power.inner();
            i_power *= i.clone();
        }

        left == right
    }

    /// Serialize to bytes for cross-boundary use (Rust ↔ Holochain DHT)
    ///
    /// Format: 4-byte BE count + n × 33-byte compressed SEC1 points
    pub fn to_bytes(&self) -> Vec<u8> {
        let count = self.commitments.len() as u32;
        let mut bytes = Vec::with_capacity(4 + count as usize * 33);
        bytes.extend_from_slice(&count.to_be_bytes());
        for commitment in &self.commitments {
            bytes.extend_from_slice(&commitment.to_bytes());
        }
        bytes
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> DkgResult<Self> {
        if bytes.len() < 4 {
            return Err(DkgError::SerializationError(
                "CommitmentSet too short: need at least 4 bytes for count".into(),
            ));
        }

        let count = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;

        let expected_len = 4 + count * 33;
        if bytes.len() != expected_len {
            return Err(DkgError::SerializationError(
                format!(
                    "CommitmentSet length mismatch: expected {} bytes for {} commitments, got {}",
                    expected_len, count, bytes.len()
                ),
            ));
        }

        let mut commitments = Vec::with_capacity(count);
        for i in 0..count {
            let start = 4 + i * 33;
            let commitment = Commitment::from_bytes(&bytes[start..start + 33])?;
            commitments.push(commitment);
        }

        Ok(Self { commitments })
    }

    /// Combine multiple commitment sets (for DKG)
    ///
    /// When multiple dealers contribute, the combined public key commitment
    /// is the sum of all C_0 commitments.
    pub fn combine(sets: &[&CommitmentSet]) -> DkgResult<Commitment> {
        if sets.is_empty() {
            return Err(DkgError::CryptoError("No commitment sets to combine".into()));
        }

        let mut combined = ProjectivePoint::IDENTITY;
        for set in sets {
            combined += set.secret_commitment().0;
        }

        Ok(Commitment(combined))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::polynomial::Polynomial;
    use rand::rngs::OsRng;

    #[test]
    fn test_commitment_creation() {
        let scalar = Scalar::from_u64(42);
        let commitment = Commitment::new(&scalar);

        // Verify it's on the curve (non-identity)
        assert_ne!(commitment.0, ProjectivePoint::IDENTITY);
    }

    #[test]
    fn test_commitment_serialization() {
        let scalar = Scalar::from_u64(12345);
        let commitment = Commitment::new(&scalar);

        let bytes = commitment.to_bytes();
        assert_eq!(bytes.len(), 33);

        let recovered = Commitment::from_bytes(&bytes).unwrap();
        assert_eq!(commitment, recovered);
    }

    #[test]
    fn test_share_verification() {
        // Create a polynomial with known coefficients
        let secret = Scalar::from_u64(100);
        let poly = Polynomial::random_with_secret(secret, 3, &mut OsRng).unwrap();

        // Create commitments
        let commitments = CommitmentSet::from_polynomial(&poly);

        // Generate and verify shares for participants 1, 2, 3
        for i in 1..=3u32 {
            let share = poly.evaluate_at_index(i);
            assert!(
                commitments.verify_share(i, &share),
                "Share verification failed for participant {}",
                i
            );
        }
    }

    #[test]
    fn test_invalid_share_verification() {
        let secret = Scalar::from_u64(100);
        let poly = Polynomial::random_with_secret(secret, 3, &mut OsRng).unwrap();
        let commitments = CommitmentSet::from_polynomial(&poly);

        // Create a wrong share
        let wrong_share = Scalar::from_u64(999999);

        // Should fail verification
        assert!(!commitments.verify_share(1, &wrong_share));
    }

    #[test]
    fn test_commitment_set_serialization() {
        let poly = Polynomial::random(3, &mut OsRng).unwrap();
        let commitments = CommitmentSet::from_polynomial(&poly);

        let bytes = commitments.to_bytes();
        assert_eq!(bytes.len(), 4 + 3 * 33); // 4-byte count + 3 commitments

        let recovered = CommitmentSet::from_bytes(&bytes).unwrap();
        assert_eq!(recovered.len(), commitments.len());

        // Verify each commitment matches
        for (original, recovered) in commitments.commitments().iter().zip(recovered.commitments().iter()) {
            assert_eq!(original, recovered);
        }
    }

    #[test]
    fn test_commitment_set_from_bytes_errors() {
        // Too short
        assert!(CommitmentSet::from_bytes(&[0]).is_err());

        // Wrong length
        let mut bytes = vec![0, 0, 0, 2]; // claims 2 commitments
        bytes.extend_from_slice(&[0; 33]); // only 1 commitment worth of data
        assert!(CommitmentSet::from_bytes(&bytes).is_err());
    }

    #[test]
    fn test_commitment_set_combination() {
        // Create two polynomials
        let poly1 = Polynomial::random(3, &mut OsRng).unwrap();
        let poly2 = Polynomial::random(3, &mut OsRng).unwrap();

        let commit1 = CommitmentSet::from_polynomial(&poly1);
        let commit2 = CommitmentSet::from_polynomial(&poly2);

        // Combined commitment should be sum of secret commitments
        let combined = CommitmentSet::combine(&[&commit1, &commit2]).unwrap();

        // Verify it equals g^(secret1 + secret2)
        let expected_secret = poly1.secret().clone() + poly2.secret().clone();
        let expected_commitment = Commitment::new(&expected_secret);

        assert_eq!(combined, expected_commitment);
    }
}
