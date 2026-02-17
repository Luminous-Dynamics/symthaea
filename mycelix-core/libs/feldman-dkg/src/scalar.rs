//! Scalar field operations for secp256k1
//!
//! Wraps k256::Scalar with additional operations needed for DKG.

use k256::Scalar as K256Scalar;
use k256::elliptic_curve::ops::Reduce;
use k256::elliptic_curve::scalar::FromUintUnchecked;
use k256::elliptic_curve::subtle::ConstantTimeEq;
use k256::{ProjectivePoint, U256};
use rand_core::CryptoRngCore;
use serde::{Deserialize, Serialize};
use zeroize::Zeroize;

use crate::error::{DkgError, DkgResult};

/// A scalar value in the secp256k1 field
#[derive(Clone, Debug, Zeroize)]
pub struct Scalar(pub(crate) K256Scalar);

impl Scalar {
    /// Create a scalar from a u64 value
    pub fn from_u64(value: u64) -> Self {
        Self(K256Scalar::from(value))
    }

    /// Create a scalar from bytes (big-endian)
    pub fn from_bytes(bytes: &[u8; 32]) -> DkgResult<Self> {
        let uint = U256::from_be_slice(bytes);
        Ok(Self(K256Scalar::from_uint_unchecked(uint)))
    }

    /// Convert to bytes (big-endian)
    pub fn to_bytes(&self) -> [u8; 32] {
        self.0.to_bytes().into()
    }

    /// Generate a random scalar
    pub fn random<R: CryptoRngCore>(rng: &mut R) -> Self {
        let mut bytes = [0u8; 32];
        rng.fill_bytes(&mut bytes);
        // Reduce to ensure it's in the field
        let uint = U256::from_be_slice(&bytes);
        Self(<K256Scalar as Reduce<U256>>::reduce(uint))
    }

    /// Create zero scalar
    pub fn zero() -> Self {
        Self(K256Scalar::ZERO)
    }

    /// Create one scalar
    pub fn one() -> Self {
        Self(K256Scalar::ONE)
    }

    /// Check if scalar is zero
    pub fn is_zero(&self) -> bool {
        self.0.is_zero().into()
    }

    /// Compute modular inverse
    pub fn invert(&self) -> DkgResult<Self> {
        let inv = self.0.invert();
        if inv.is_some().into() {
            Ok(Self(inv.unwrap()))
        } else {
            Err(DkgError::CryptoError("Cannot invert zero".into()))
        }
    }

    /// Negate the scalar
    pub fn negate(&self) -> Self {
        Self(-self.0)
    }

    /// Multiply scalar by generator point
    pub fn base_mul(&self) -> ProjectivePoint {
        ProjectivePoint::GENERATOR * self.0
    }

    /// Get the inner k256 scalar
    pub fn inner(&self) -> &K256Scalar {
        &self.0
    }
}

impl std::ops::Add for Scalar {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl std::ops::Add for &Scalar {
    type Output = Scalar;

    fn add(self, rhs: Self) -> Self::Output {
        Scalar(self.0 + rhs.0)
    }
}

impl std::ops::Sub for Scalar {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl std::ops::Sub for &Scalar {
    type Output = Scalar;

    fn sub(self, rhs: Self) -> Self::Output {
        Scalar(self.0 - rhs.0)
    }
}

impl std::ops::Mul for Scalar {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0 * rhs.0)
    }
}

impl std::ops::Mul for &Scalar {
    type Output = Scalar;

    fn mul(self, rhs: Self) -> Self::Output {
        Scalar(self.0 * rhs.0)
    }
}

impl std::ops::Mul<&Scalar> for Scalar {
    type Output = Self;

    fn mul(self, rhs: &Scalar) -> Self::Output {
        Self(self.0 * rhs.0)
    }
}

impl std::ops::AddAssign for Scalar {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl std::ops::MulAssign for Scalar {
    fn mul_assign(&mut self, rhs: Self) {
        self.0 *= rhs.0;
    }
}

impl PartialEq for Scalar {
    fn eq(&self, other: &Self) -> bool {
        self.0.ct_eq(&other.0).into()
    }
}

impl Eq for Scalar {}

impl Serialize for Scalar {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let bytes = self.to_bytes();
        serializer.serialize_bytes(&bytes)
    }
}

impl<'de> Deserialize<'de> for Scalar {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let bytes: Vec<u8> = Deserialize::deserialize(deserializer)?;
        if bytes.len() != 32 {
            return Err(serde::de::Error::custom("Invalid scalar length"));
        }
        let mut arr = [0u8; 32];
        arr.copy_from_slice(&bytes);
        Scalar::from_bytes(&arr).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::OsRng;

    #[test]
    fn test_scalar_arithmetic() {
        let a = Scalar::from_u64(5);
        let b = Scalar::from_u64(3);

        let sum = &a + &b;
        assert_eq!(sum, Scalar::from_u64(8));

        let diff = &a - &b;
        assert_eq!(diff, Scalar::from_u64(2));

        let prod = &a * &b;
        assert_eq!(prod, Scalar::from_u64(15));
    }

    #[test]
    fn test_scalar_inverse() {
        let a = Scalar::from_u64(7);
        let a_inv = a.invert().unwrap();
        let product = &a * &a_inv;
        assert_eq!(product, Scalar::one());
    }

    #[test]
    fn test_random_scalar() {
        let s1 = Scalar::random(&mut OsRng);
        let s2 = Scalar::random(&mut OsRng);
        assert_ne!(s1, s2);
    }

    #[test]
    fn test_scalar_serialization() {
        let s = Scalar::from_u64(12345);
        let bytes = s.to_bytes();
        let recovered = Scalar::from_bytes(&bytes).unwrap();
        assert_eq!(s, recovered);
    }
}
