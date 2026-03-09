//! Polynomial operations for secret sharing
//!
//! Implements polynomials over the scalar field for Shamir's secret sharing.

use rand_core::CryptoRngCore;
use serde::{Deserialize, Serialize};
use zeroize::{Zeroize, ZeroizeOnDrop};

use crate::error::{DkgError, DkgResult};
use crate::scalar::Scalar;

/// A polynomial over the scalar field
///
/// f(x) = a_0 + a_1*x + a_2*x^2 + ... + a_{t-1}*x^{t-1}
///
/// where a_0 is the secret and t is the threshold.
#[derive(Clone, Debug, Zeroize, ZeroizeOnDrop, Serialize, Deserialize)]
pub struct Polynomial {
    /// Coefficients [a_0, a_1, ..., a_{t-1}]
    /// a_0 is the constant term (secret)
    coefficients: Vec<Scalar>,
}

impl Polynomial {
    /// Create a new polynomial with the given coefficients
    pub fn new(coefficients: Vec<Scalar>) -> DkgResult<Self> {
        if coefficients.is_empty() {
            return Err(DkgError::CryptoError(
                "Polynomial must have at least one coefficient".into(),
            ));
        }
        Ok(Self { coefficients })
    }

    /// Generate a random polynomial of degree (threshold - 1) with given secret
    ///
    /// The secret becomes the constant term a_0.
    pub fn random_with_secret<R: CryptoRngCore>(
        secret: Scalar,
        threshold: usize,
        rng: &mut R,
    ) -> DkgResult<Self> {
        if threshold == 0 {
            return Err(DkgError::InvalidThreshold {
                threshold,
                participants: 0,
            });
        }

        let mut coefficients = Vec::with_capacity(threshold);
        coefficients.push(secret);

        // Generate random coefficients for higher degree terms
        for _ in 1..threshold {
            coefficients.push(Scalar::random(rng));
        }

        Ok(Self { coefficients })
    }

    /// Generate a completely random polynomial of given degree
    pub fn random<R: CryptoRngCore>(threshold: usize, rng: &mut R) -> DkgResult<Self> {
        let secret = Scalar::random(rng);
        Self::random_with_secret(secret, threshold, rng)
    }

    /// Get the degree of the polynomial
    pub fn degree(&self) -> usize {
        self.coefficients.len() - 1
    }

    /// Get the threshold (degree + 1)
    pub fn threshold(&self) -> usize {
        self.coefficients.len()
    }

    /// Get the secret (constant term a_0)
    pub fn secret(&self) -> &Scalar {
        &self.coefficients[0]
    }

    /// Get all coefficients
    pub fn coefficients(&self) -> &[Scalar] {
        &self.coefficients
    }

    /// Evaluate the polynomial at point x
    ///
    /// Uses Horner's method for efficiency:
    /// f(x) = a_0 + x*(a_1 + x*(a_2 + ... + x*a_{t-1}))
    pub fn evaluate(&self, x: &Scalar) -> Scalar {
        let mut result = Scalar::zero();

        // Horner's method: evaluate from highest degree down
        for coeff in self.coefficients.iter().rev() {
            result *= x.clone();
            result += coeff.clone();
        }

        result
    }

    /// Evaluate at a participant index (converts u32 to scalar)
    pub fn evaluate_at_index(&self, index: u32) -> Scalar {
        let x = Scalar::from_u64(index as u64);
        self.evaluate(&x)
    }
}

/// Lagrange interpolation to recover the secret from shares
///
/// Given t points (x_i, y_i), recovers the constant term of the
/// unique polynomial of degree < t passing through all points.
pub fn lagrange_interpolate_at_zero(points: &[(Scalar, Scalar)]) -> DkgResult<Scalar> {
    if points.is_empty() {
        return Err(DkgError::InterpolationError("No points provided".into()));
    }

    let mut result = Scalar::zero();

    for (i, (x_i, y_i)) in points.iter().enumerate() {
        // Compute Lagrange basis polynomial L_i(0)
        // L_i(0) = Π_{j≠i} (0 - x_j) / (x_i - x_j)
        //        = Π_{j≠i} (-x_j) / (x_i - x_j)
        //        = Π_{j≠i} x_j / (x_j - x_i)

        let mut numerator = Scalar::one();
        let mut denominator = Scalar::one();

        for (j, (x_j, _)) in points.iter().enumerate() {
            if i != j {
                numerator *= x_j.clone();
                denominator *= x_j.clone() - x_i.clone();
            }
        }

        if denominator.is_zero() {
            return Err(DkgError::InterpolationError(
                "Duplicate x values in interpolation".into(),
            ));
        }

        let basis = numerator * denominator.invert()?;
        result += y_i.clone() * basis;
    }

    Ok(result)
}

/// Lagrange coefficient for participant i when reconstructing at x=0
pub fn lagrange_coefficient(i: usize, participant_indices: &[u32]) -> DkgResult<Scalar> {
    let x_i = Scalar::from_u64(participant_indices[i] as u64);

    let mut numerator = Scalar::one();
    let mut denominator = Scalar::one();

    for (j, &idx) in participant_indices.iter().enumerate() {
        if i != j {
            let x_j = Scalar::from_u64(idx as u64);
            numerator *= x_j.clone();
            denominator *= x_j - x_i.clone();
        }
    }

    if denominator.is_zero() {
        return Err(DkgError::InterpolationError(
            "Duplicate indices in coefficient calculation".into(),
        ));
    }

    Ok(numerator * denominator.invert()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::OsRng;

    #[test]
    fn test_polynomial_evaluation() {
        // f(x) = 5 + 3x + 2x^2
        let coeffs = vec![
            Scalar::from_u64(5),
            Scalar::from_u64(3),
            Scalar::from_u64(2),
        ];
        let poly = Polynomial::new(coeffs).unwrap();

        // f(0) = 5
        let y0 = poly.evaluate(&Scalar::zero());
        assert_eq!(y0, Scalar::from_u64(5));

        // f(1) = 5 + 3 + 2 = 10
        let y1 = poly.evaluate(&Scalar::one());
        assert_eq!(y1, Scalar::from_u64(10));

        // f(2) = 5 + 6 + 8 = 19
        let y2 = poly.evaluate(&Scalar::from_u64(2));
        assert_eq!(y2, Scalar::from_u64(19));
    }

    #[test]
    fn test_lagrange_interpolation() {
        // Create a polynomial with secret = 42
        let secret = Scalar::from_u64(42);
        let poly = Polynomial::random_with_secret(secret.clone(), 3, &mut OsRng).unwrap();

        // Generate 3 shares at indices 1, 2, 3
        let shares: Vec<(Scalar, Scalar)> = (1..=3)
            .map(|i| {
                let x = Scalar::from_u64(i);
                let y = poly.evaluate(&x);
                (x, y)
            })
            .collect();

        // Recover secret via interpolation
        let recovered = lagrange_interpolate_at_zero(&shares).unwrap();
        assert_eq!(recovered, secret);
    }

    #[test]
    fn test_insufficient_shares() {
        // Create a 3-threshold polynomial
        let secret = Scalar::from_u64(100);
        let poly = Polynomial::random_with_secret(secret.clone(), 3, &mut OsRng).unwrap();

        // Only 2 shares (insufficient)
        let shares: Vec<(Scalar, Scalar)> = (1..=2)
            .map(|i| {
                let x = Scalar::from_u64(i);
                let y = poly.evaluate(&x);
                (x, y)
            })
            .collect();

        // Interpolation will produce wrong result with insufficient shares
        let recovered = lagrange_interpolate_at_zero(&shares).unwrap();
        // This should NOT equal the secret (though it will produce some value)
        // We just verify it computes without error
        assert_ne!(recovered, secret);
    }

    #[test]
    fn test_lagrange_coefficient() {
        let indices = vec![1, 2, 3];

        // Compute coefficients
        let l0 = lagrange_coefficient(0, &indices).unwrap();
        let l1 = lagrange_coefficient(1, &indices).unwrap();
        let l2 = lagrange_coefficient(2, &indices).unwrap();

        // Sum of Lagrange coefficients at x=0 should equal 1
        let sum = l0 + l1 + l2;
        assert_eq!(sum, Scalar::one());
    }
}
