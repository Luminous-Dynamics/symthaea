// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! The prime field GF(p) — integers mod a prime, a genuine field (every nonzero
//! element has a multiplicative inverse).

/// The finite field `GF(p)` for a prime modulus `p`.
#[derive(Debug, Clone, Copy)]
pub struct PrimeField {
    p: u64,
}

impl PrimeField {
    /// Construct `GF(p)`. `Err` if `p < 2` or `p` is composite.
    pub fn new(p: u64) -> Result<PrimeField, String> {
        if p < 2 || !is_prime(p) {
            return Err(format!("{p} is not prime"));
        }
        Ok(PrimeField { p })
    }

    /// The modulus.
    pub fn modulus(&self) -> u64 {
        self.p
    }

    /// Reduce an integer into `0..p`.
    pub fn reduce(&self, a: i64) -> u64 {
        a.rem_euclid(self.p as i64) as u64
    }

    /// `a + b`.
    pub fn add(&self, a: u64, b: u64) -> u64 {
        (a + b) % self.p
    }

    /// `a − b`.
    pub fn sub(&self, a: u64, b: u64) -> u64 {
        (a + self.p - b % self.p) % self.p
    }

    /// `a · b`.
    pub fn mul(&self, a: u64, b: u64) -> u64 {
        ((a as u128 * b as u128) % self.p as u128) as u64
    }

    /// `aᵉ` by fast exponentiation.
    pub fn pow(&self, mut a: u64, mut e: u64) -> u64 {
        let mut result = 1u64;
        a %= self.p;
        while e > 0 {
            if e & 1 == 1 {
                result = self.mul(result, a);
            }
            a = self.mul(a, a);
            e >>= 1;
        }
        result
    }

    /// The multiplicative inverse `a⁻¹` via Fermat's little theorem
    /// (`a^{p−2}`). `None` for `a ≡ 0`.
    pub fn inverse(&self, a: u64) -> Option<u64> {
        if a % self.p == 0 {
            return None;
        }
        Some(self.pow(a, self.p - 2))
    }
}

/// Trial-division primality test (adequate for the modest primes used here).
fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n % 2 == 0 {
        return n == 2;
    }
    let mut d = 3;
    while d * d <= n {
        if n % d == 0 {
            return false;
        }
        d += 2;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_composite_modulus() {
        assert!(PrimeField::new(6).is_err());
        assert!(PrimeField::new(1).is_err());
        assert!(PrimeField::new(7).is_ok());
    }

    #[test]
    fn field_arithmetic_gf7() {
        let f = PrimeField::new(7).unwrap();
        assert_eq!(f.add(3, 5), 1); // 8 mod 7
        assert_eq!(f.sub(2, 5), 4); // -3 mod 7
        assert_eq!(f.mul(3, 5), 1); // 15 mod 7
        // Inverse of 3 is 5 (3·5 = 1); Fermat: 3^6 = 1.
        assert_eq!(f.inverse(3), Some(5));
        assert_eq!(f.pow(3, 6), 1);
        assert_eq!(f.inverse(0), None);
    }

    #[test]
    fn every_nonzero_has_an_inverse() {
        let f = PrimeField::new(13).unwrap();
        for a in 1..13 {
            let inv = f.inverse(a).unwrap();
            assert_eq!(f.mul(a, inv), 1, "a={a}");
        }
    }
}
